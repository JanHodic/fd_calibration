"""
Experimentální grid runner pro FD kalibraci.

Prochází systematicky parametrický prostor (K, N, sigma, prevalenční profil,
klasifikátor) × seedy a výsledky ukládá do parquet. Výstup je čtený skriptem
analyze_results.py.

Spuštění:
    python run_experiment_grid.py                      # plný grid
    python run_experiment_grid.py --quick              # rychlý smoke test
    python run_experiment_grid.py --out results.parquet
    python run_experiment_grid.py --jobs 8             # paralelně
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from fd_calibration import (
    FDMulticlassCalibrator,
    TemperatureScalingCalibrator,
    PlattOvRCalibrator,
    IsotonicOvRCalibrator,
    expected_calibration_error,
    brier_score_multiclass,
    accuracy,
    balanced_accuracy,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Prevalenční profily
# ---------------------------------------------------------------------------

def prevalence_uniform(K: int) -> np.ndarray:
    return np.ones(K) / K


def prevalence_mild(K: int) -> np.ndarray:
    raw = np.array([2.0 ** (-k) for k in range(K)])
    return raw / raw.sum()


def prevalence_extreme(K: int) -> np.ndarray:
    full = np.array([0.50, 0.18, 0.10, 0.08, 0.07, 0.07])
    if K <= 6:
        p = full[:K]
    else:
        p = np.concatenate([full, np.full(K - 6, 0.02)])
    return p / p.sum()


PREVALENCE_PROFILES = {
    "uniform": prevalence_uniform,
    "mild": prevalence_mild,
    "extreme": prevalence_extreme,
}


# ---------------------------------------------------------------------------
# Konfigurace gridu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    K_values: tuple[int, ...]
    N_values: tuple[int, ...]
    sigma_values: tuple[float, ...]
    prevalence_profiles: tuple[str, ...]
    classifier_names: tuple[str, ...]
    seeds: tuple[int, ...]
    n_features: int = 32


def full_grid() -> GridConfig:
    return GridConfig(
        K_values=(2, 3, 4, 5, 6, 8, 10),
        N_values=(500, 1000, 2000, 5000, 10000),
        sigma_values=(1.0, 1.5, 2.0, 2.5, 3.0),
        prevalence_profiles=("uniform", "mild", "extreme"),
        classifier_names=("rf", "gb", "svm", "logreg") + (("xgb",) if HAS_XGB else ()),
        seeds=tuple(range(30)),
    )


def quick_grid() -> GridConfig:
    return GridConfig(
        K_values=(3, 6),
        N_values=(1000,),
        sigma_values=(2.0,),
        prevalence_profiles=("uniform", "extreme"),
        classifier_names=("rf", "gb"),
        seeds=tuple(range(5)),
    )


# ---------------------------------------------------------------------------
# Generátor syntetických dat
# ---------------------------------------------------------------------------

def generate_data(
    K: int, N: int, sigma: float, prevalence: np.ndarray,
    n_features: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gausovské shluky s danou separabilitou a prevalencí.

    Separabilita per class lineárně klesá od sigma k sigma/2,
    takže pozdější třídy jsou obtížnější (modeluje jemnější nálezy).
    """
    rng = np.random.default_rng(seed)

    sep_per_class = np.linspace(sigma, sigma * 0.5, K)

    centers = rng.normal(size=(K, n_features))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * sep_per_class[:, None]

    counts = rng.multinomial(N, prevalence)

    X_list, y_list = [], []
    for k in range(K):
        if counts[k] == 0:
            continue
        X_k = rng.normal(loc=centers[k], scale=1.0, size=(counts[k], n_features))
        X_list.append(X_k)
        y_list.append(np.full(counts[k], k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Klasifikátory
# ---------------------------------------------------------------------------

def build_classifier(name: str, seed: int) -> Any:
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=200, random_state=seed)
    if name == "svm":
        return SVC(probability=True, random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=200, random_state=seed, n_jobs=1,
            eval_metric="mlogloss", verbosity=0,
        )
    raise ValueError(f"Neznámý klasifikátor: {name}")


def get_proba(clf: Any, X: np.ndarray, K: int) -> np.ndarray:
    """Pravděpodobnosti tvaru (N, K); chybějící třídy doplní nulami."""
    p = clf.predict_proba(X)
    if p.shape[1] < K:
        full = np.zeros((p.shape[0], K))
        for i, c in enumerate(clf.classes_):
            full[:, int(c)] = p[:, i]
        return full
    return p


# ---------------------------------------------------------------------------
# Jeden experimentální běh
# ---------------------------------------------------------------------------

def run_single(
    K: int, N: int, sigma: float, profile: str,
    clf_name: str, seed: int, n_features: int,
) -> list[dict[str, Any]]:
    t_start = time.time()

    prevalence = PREVALENCE_PROFILES[profile](K)
    X, y = generate_data(K, N, sigma, prevalence, n_features, seed)
    classes = np.arange(K)

    try:
        X_tr, X_rest, y_tr, y_rest = train_test_split(
            X, y, test_size=0.6, stratify=y, random_state=seed,
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed,
        )
    except ValueError:
        return []

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    base_row = {
        "K": K, "N": N, "sigma": sigma, "profile": profile,
        "classifier": clf_name, "seed": seed,
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
    }

    try:
        clf = build_classifier(clf_name, seed)
        clf.fit(X_tr, y_tr)
    except Exception as e:
        return [{**base_row, "method": "ERROR", "error": str(e)}]

    s_val = get_proba(clf, X_val, K)
    s_te = get_proba(clf, X_te, K)

    logits_val = np.log(np.clip(s_val, 1e-12, None))
    logits_te = np.log(np.clip(s_te, 1e-12, None))

    rows: list[dict[str, Any]] = []

    def evaluate(method: str, probs: np.ndarray) -> dict:
        preds = classes[np.argmax(probs, axis=1)]
        row = dict(base_row)
        row["method"] = method
        row["accuracy"] = accuracy(preds, y_te)
        row["bal_accuracy"] = balanced_accuracy(preds, y_te, classes)
        row["ece"] = expected_calibration_error(probs, y_te)
        row["brier"] = brier_score_multiclass(probs, y_te, K)
        return row

    # 1. Bez kalibrace
    rows.append(evaluate("none", s_te))

    # 2. FD
    try:
        fd = FDMulticlassCalibrator().fit(s_val, y_val, classes=classes)
        rows.append(evaluate("fd", fd.predict_proba(s_te)))
    except Exception as e:
        rows.append({**base_row, "method": "fd", "error": str(e)})

    # 3. Platt OvR
    try:
        platt = PlattOvRCalibrator().fit(s_val, y_val)
        rows.append(evaluate("platt", platt.predict_proba(s_te)))
    except Exception as e:
        rows.append({**base_row, "method": "platt", "error": str(e)})

    # 4. Isotonic OvR
    try:
        iso = IsotonicOvRCalibrator().fit(s_val, y_val)
        rows.append(evaluate("isotonic", iso.predict_proba(s_te)))
    except Exception as e:
        rows.append({**base_row, "method": "isotonic", "error": str(e)})

    # 5. Temperature scaling
    try:
        ts = TemperatureScalingCalibrator().fit(logits_val, y_val)
        rows.append(evaluate("temperature", ts.predict_proba(logits_te)))
    except Exception as e:
        rows.append({**base_row, "method": "temperature", "error": str(e)})

    elapsed = time.time() - t_start
    for r in rows:
        r["elapsed_sec"] = elapsed
    return rows


# ---------------------------------------------------------------------------
# Grid orchestrator
# ---------------------------------------------------------------------------

def iter_grid(cfg: GridConfig) -> Iterator[tuple]:
    return itertools.product(
        cfg.K_values, cfg.N_values, cfg.sigma_values,
        cfg.prevalence_profiles, cfg.classifier_names, cfg.seeds,
    )


def total_jobs(cfg: GridConfig) -> int:
    return (
        len(cfg.K_values) * len(cfg.N_values) * len(cfg.sigma_values)
        * len(cfg.prevalence_profiles) * len(cfg.classifier_names)
        * len(cfg.seeds)
    )


def run_grid(cfg: GridConfig, n_jobs: int, verbose: bool = True) -> pd.DataFrame:
    jobs = list(iter_grid(cfg))
    if verbose:
        print(f"Celkem {len(jobs)} jobů × ~5 kalibračních metod")
        print(f"Paralelně {n_jobs} jobů")

    t0 = time.time()

    def worker(args):
        K, N, sigma, profile, clf, seed = args
        return run_single(K, N, sigma, profile, clf, seed, cfg.n_features)

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(worker)(args) for args in jobs
    )

    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)

    if verbose:
        print(f"\nHotovo za {(time.time() - t0) / 60:.1f} min, {len(df)} řádků")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results.parquet")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--features", type=int, default=32)
    args = parser.parse_args()

    cfg = quick_grid() if args.quick else full_grid()
    if args.features != 32:
        cfg = GridConfig(**{**asdict(cfg), "n_features": args.features})

    print(f"Grid: K={cfg.K_values}, N={cfg.N_values}, sigma={cfg.sigma_values}")
    print(f"Profily={cfg.prevalence_profiles}, clf={cfg.classifier_names}")
    print(f"Seedů={len(cfg.seeds)}, celkem jobů={total_jobs(cfg)}")

    df = run_grid(cfg, n_jobs=args.jobs)

    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    print(f"Výsledky uloženy: {out_path} ({len(df)} řádků)")

    print("\nŘádků podle metody:")
    print(df["method"].value_counts())
    print("\nŘádků podle klasifikátoru:")
    print(df["classifier"].value_counts())


if __name__ == "__main__":
    main()