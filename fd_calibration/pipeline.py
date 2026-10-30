"""
End-to-end pipeline pro experiment:
1. Načti / vygeneruj data
2. Rozděl na train / val (kalibrace) / test
3. Natrénuj klasifikátor
4. Nafituj kalibraci (FD vs. baseline metody) na val
5. Evaluuj na test (klasifikace + kalibrační metriky)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax

from .multiclass import FDMulticlassCalibrator
from .metrics import (
    expected_calibration_error,
    brier_score_multiclass,
    ovr_auc,
    accuracy,
    balanced_accuracy,
)


# ---------------------------------------------------------------------------
# Baseline kalibrátory
# ---------------------------------------------------------------------------

class TemperatureScalingCalibrator:
    """Temperature scaling pro vícetřídní klasifikaci (vstup: logity)."""

    def __init__(self) -> None:
        self.T: float = 1.0
        self.classes_: np.ndarray | None = None

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScalingCalibrator":
        from scipy.optimize import minimize_scalar
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        def nll(T: float) -> float:
            if T <= 0:
                return float("inf")
            p = softmax(logits / T, axis=1)
            p = np.clip(p, 1e-12, 1.0)
            return float(-np.mean(np.log(p[np.arange(len(y)), y])))

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.T = float(res.x)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.T, axis=1)


class PlattOvRCalibrator:
    """Platt scaling v one-vs-rest schématu."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.models: list[LogisticRegression] = []

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattOvRCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models = []
        for k_idx, k in enumerate(self.classes_):
            y_bin = (y == k).astype(int)
            lr = LogisticRegression()
            lr.fit(scores[:, k_idx].reshape(-1, 1), y_bin)
            self.models.append(lr)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        K = len(self.classes_)
        P = np.zeros((scores.shape[0], K))
        for k_idx in range(K):
            P[:, k_idx] = self.models[k_idx].predict_proba(
                scores[:, k_idx].reshape(-1, 1)
            )[:, 1]
        s = P.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        return P / s


class IsotonicOvRCalibrator:
    """Isotonic regression v one-vs-rest schématu."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.models: list[IsotonicRegression] = []

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicOvRCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models = []
        for k_idx, k in enumerate(self.classes_):
            y_bin = (y == k).astype(int)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(scores[:, k_idx], y_bin)
            self.models.append(ir)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        K = len(self.classes_)
        P = np.zeros((scores.shape[0], K))
        for k_idx in range(K):
            P[:, k_idx] = self.models[k_idx].predict(scores[:, k_idx])
        s = P.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        return P / s


# ---------------------------------------------------------------------------
# Klasifikátory
# ---------------------------------------------------------------------------

def build_classifier(name: str, random_state: int = 42) -> Any:
    """Postav klasifikátor podle jména."""
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=200, random_state=random_state
        )
    if name == "svm":
        return SVC(probability=True, random_state=random_state)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError("pip install xgboost") from e
        return XGBClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1,
            eval_metric="mlogloss", verbosity=0,
        )
    raise ValueError(f"Neznámý klasifikátor: {name}")


def get_scores(clf: Any, X: np.ndarray) -> np.ndarray:
    """Získej skóre/pravděpodobnosti z klasifikátoru, vždy tvar (N, K)."""
    if hasattr(clf, "decision_function"):
        try:
            s = clf.decision_function(X)
            if s.ndim == 1:
                s = np.column_stack([-s, s])
            return s
        except Exception:
            pass
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    raise RuntimeError(f"Klasifikátor {type(clf)} neumí poskytnout skóre.")


# ---------------------------------------------------------------------------
# Hlavní experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    classifier_name: str
    method_name: str
    accuracy: float
    balanced_accuracy: float
    ece: float
    brier: float
    ovr_auc: dict[int, float] = field(default_factory=dict)
    fd_summary: str = ""


def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    classifier_name: str = "rf",
    test_size: float = 0.3,
    val_size: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Train klasifikátor + porovnej kalibrační metody."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=test_size + val_size,
        stratify=y, random_state=random_state,
    )
    val_frac = val_size / (test_size + val_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, test_size=1 - val_frac,
        stratify=y_rest, random_state=random_state,
    )

    if verbose:
        print(f"Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_te)}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    clf = build_classifier(classifier_name, random_state=random_state)
    clf.fit(X_tr, y_tr)

    s_val = get_scores(clf, X_val)
    s_te = get_scores(clf, X_te)

    if hasattr(clf, "predict_proba"):
        p_val = np.clip(clf.predict_proba(X_val), 1e-12, 1.0)
        p_te = np.clip(clf.predict_proba(X_te), 1e-12, 1.0)
        logits_val = np.log(p_val)
        logits_te = np.log(p_te)
    else:
        logits_val = s_val
        logits_te = s_te

    classes = np.unique(y)
    results: list[ExperimentResult] = []

    def make(method: str, probs: np.ndarray, preds: np.ndarray, fd_summary: str = "") -> ExperimentResult:
        return ExperimentResult(
            classifier_name=classifier_name,
            method_name=method,
            accuracy=accuracy(preds, y_te),
            balanced_accuracy=balanced_accuracy(preds, y_te, classes),
            ece=expected_calibration_error(probs, y_te),
            brier=brier_score_multiclass(probs, y_te, len(classes)),
            ovr_auc=ovr_auc(s_te, y_te, classes),
            fd_summary=fd_summary,
        )

    # 1. Bez kalibrace
    if hasattr(clf, "predict_proba"):
        p_raw = clf.predict_proba(X_te)
        results.append(make("No calibration", p_raw, clf.predict(X_te)))

    # 2. FD
    fd = FDMulticlassCalibrator().fit(s_val, y_val, classes=classes)
    p_fd = fd.predict_proba(s_te)
    results.append(make("FD (degenerated levels)", p_fd,
                        classes[np.argmax(p_fd, axis=1)], fd.summary()))

    # 3. Platt OvR
    platt = PlattOvRCalibrator().fit(s_val, y_val)
    p_platt = platt.predict_proba(s_te)
    results.append(make("Platt OvR", p_platt, classes[np.argmax(p_platt, axis=1)]))

    # 4. Isotonic OvR
    iso = IsotonicOvRCalibrator().fit(s_val, y_val)
    p_iso = iso.predict_proba(s_te)
    results.append(make("Isotonic OvR", p_iso, classes[np.argmax(p_iso, axis=1)]))

    # 5. Temperature scaling
    ts = TemperatureScalingCalibrator().fit(logits_val, y_val)
    p_ts = ts.predict_proba(logits_te)
    results.append(make(f"Temperature scaling (T={ts.T:.3f})", p_ts,
                        classes[np.argmax(p_ts, axis=1)]))

    if verbose:
        print_results(results, class_names)

    return results


def print_results(
    results: list[ExperimentResult], class_names: list[str] | None = None
) -> None:
    """Tabulkový výpis výsledků."""
    print()
    print(f"{'Method':<35s} {'Acc':>7s} {'BalAcc':>7s} {'ECE':>7s} {'Brier':>7s}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.method_name:<35s} {r.accuracy:>7.4f} {r.balanced_accuracy:>7.4f} "
            f"{r.ece:>7.4f} {r.brier:>7.4f}"
        )
    for r in results:
        if r.fd_summary:
            print()
            print("FD parameters:")
            print(r.fd_summary)