"""
Metriky pro hodnocení klasifikace a kalibrace.

- Expected Calibration Error (ECE)
- Brierovo skóre (binární i vícetřídní)
- Reliability diagram
- One-vs-rest AUC
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error pro vícetřídní klasifikaci.

    Používá confidence (max pravděpodobnost) a accuracy v každém binu.

    Parameters
    ----------
    probs : (N, K)
        Predikované pravděpodobnosti.
    y_true : (N,)
        Skutečné štítky (indexy tříd 0..K-1).
    n_bins : int
        Počet binů.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == 0.0)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_in_bin / N) * abs(avg_conf - avg_acc)
    return float(ece)


def brier_score_multiclass(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_classes: int | None = None,
) -> float:
    """Vícetřídní Brierovo skóre (suma kvadrátů odchylek od one-hot)."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    if n_classes is None:
        n_classes = probs.shape[1]
    one_hot = np.zeros((len(y_true), n_classes))
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_data(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Data pro reliability diagram (confidence vs. accuracy v binech).

    Returns
    -------
    dict s klíči: 'bin_centers', 'accuracies', 'confidences', 'counts'.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    acc_in_bin = np.full(n_bins, np.nan)
    conf_in_bin = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == 0.0)
        n = mask.sum()
        counts[i] = n
        if n > 0:
            acc_in_bin[i] = accuracies[mask].mean()
            conf_in_bin[i] = confidences[mask].mean()

    return {
        "bin_centers": centers,
        "accuracies": acc_in_bin,
        "confidences": conf_in_bin,
        "counts": counts,
    }


def ovr_auc(
    scores: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray | None = None,
) -> dict[int, float]:
    """One-vs-rest AUC pro každou třídu zvlášť.

    Vrací dict {třída: AUC}.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    result: dict[int, float] = {}
    for k_idx, k in enumerate(classes):
        y_bin = (y_true == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            result[int(k)] = float("nan")
            continue
        try:
            result[int(k)] = float(roc_auc_score(y_bin, scores[:, k_idx]))
        except ValueError:
            result[int(k)] = float("nan")
    return result


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Klasifikační přesnost."""
    return float(np.mean(np.asarray(y_pred) == np.asarray(y_true)))


def balanced_accuracy(
    y_pred: np.ndarray, y_true: np.ndarray, classes: np.ndarray | None = None
) -> float:
    """Vyvážená přesnost přes třídy."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    recalls = []
    for k in classes:
        mask = y_true == k
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == k).mean()))
    return float(np.mean(recalls))
