"""
Binární Fermiho-Diracova kalibrace (Kim et al., 2021).

Implementuje:
- numericky stabilní výpočet μ/N jako funkce β (Eq. 6)
- výpočet AUC z FD parametrů (přes průměrné ranky)
- inverzní úlohu: fit (β, μ) z (AUC, ρ, N)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


# ---------------------------------------------------------------------------
# Numericky stabilní log(sinh(x))
# ---------------------------------------------------------------------------

def _log_sinh(x: np.ndarray) -> np.ndarray:
    """Numericky stabilní log(sinh(x)) pro x > 0.

    Pro malé x: log(sinh(x)) ≈ log(x).
    Pro velké x: log(sinh(x)) ≈ x - log(2).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x) if x.ndim else np.array(0.0)

    if x.ndim == 0:
        if x < 1e-6:
            return np.log(max(x, 1e-300))
        if x > 50:
            return x - np.log(2)
        return float(np.log(np.sinh(x)))

    small = x < 1e-6
    large = x > 50
    middle = ~(small | large)
    out[small] = np.log(np.clip(x[small], 1e-300, None))
    out[large] = x[large] - np.log(2)
    out[middle] = np.log(np.sinh(x[middle]))
    return out


# ---------------------------------------------------------------------------
# FD distribuce a parametry
# ---------------------------------------------------------------------------

def fd_prob(r: np.ndarray, beta: float, mu: float) -> np.ndarray:
    """Rank-podmíněná pravděpodobnost pozitivní třídy.

    P(y=1 | r) = 1 / (1 + exp(β(r - μ)))
    """
    return expit(-beta * (np.asarray(r, dtype=float) - mu))


def mu_over_N(beta: float, rho: float, N: int) -> float:
    """μ/N jako funkce β, ρ, N (Eq. 6 z Kim et al., 2021).

    μ/N = 1/2 - (1/βN) * ln[sinh(βN(1-ρ)/2) / sinh(βNρ/2)]
    """
    bN = beta * N
    if bN < 1e-10:
        # Limit β → 0: použijeme Taylor sinh(x) ≈ x
        # Výsledek je nedefinovaný (μ jde do nekonečna), vrátíme rozumný odhad
        return 0.5
    a1 = bN * (1 - rho) / 2
    a2 = bN * rho / 2
    log_ratio = _log_sinh(np.array(a1)) - _log_sinh(np.array(a2))
    return 0.5 - float(log_ratio) / bN


def auc_from_params(beta: float, mu: float, N: int, N1: int) -> float:
    """Výpočet AUC z FD parametrů přes vztah AUC = (<r|0> - <r|1>)/N + 1/2.

    <r|1> = (1/N1) * Σ r * P(1|r)
    <r|0> = (1/N0) * Σ r * P(0|r)
    """
    r = np.arange(1, N + 1, dtype=float)
    p1 = fd_prob(r, beta, mu)
    p0 = 1.0 - p1
    N0 = N - N1
    if N1 == 0 or N0 == 0:
        return 0.5
    avg_r_1 = np.sum(r * p1) / N1
    avg_r_0 = np.sum(r * p0) / N0
    return float((avg_r_0 - avg_r_1) / N + 0.5)


# ---------------------------------------------------------------------------
# Inverzní úloha: fit (β, μ) z (AUC, ρ, N)
# ---------------------------------------------------------------------------

def fit_fd_binary(
    auc: float,
    N1: int,
    N: int,
    beta_bounds: tuple[float, float] = (1e-6, 50.0),
) -> tuple[float, float]:
    """Najdi (β, μ) tak, aby FD distribuce reprodukovala zadané AUC.

    Postup:
    1. μ je funkcí β přes Eq. 6.
    2. β najdeme řešením rovnice auc_from_params(β, μ(β)) = auc.

    Returns
    -------
    (beta, mu) : tuple[float, float]
    """
    if not (0.5 <= auc <= 1.0):
        raise ValueError(f"AUC musí být v [0.5, 1.0], bylo {auc}.")
    if N1 <= 0 or N1 >= N:
        raise ValueError(f"N1 musí být v (0, N), bylo N1={N1}, N={N}.")

    rho = N1 / N

    # Triviální případ: náhodný klasifikátor
    if auc <= 0.5 + 1e-6:
        return 1e-6, mu_over_N(1e-6, rho, N) * N

    def objective(beta: float) -> float:
        mu_n = mu_over_N(beta, rho, N)
        return auc_from_params(beta, mu_n * N, N, N1) - auc

    # Najdi interval, kde objective mění znaménko
    lo, hi = beta_bounds
    # Pokud objective(hi) < 0, klasifikátor je tak dobrý, že potřebujeme větší β
    # ale 50 by mělo stačit pro AUC až 0.9999
    try:
        beta = brentq(objective, lo, hi, xtol=1e-8)
    except ValueError as e:
        # Fallback: zkusíme širší interval
        beta = brentq(objective, 1e-8, 200.0, xtol=1e-8)

    mu = mu_over_N(beta, rho, N) * N
    return beta, mu


# ---------------------------------------------------------------------------
# Optimální práh
# ---------------------------------------------------------------------------

def optimal_threshold(beta: float, mu: float, rho: float, N: int) -> float:
    """Optimální rank-práh maximalizující balanced accuracy (Eq. 9).

    r* / N = μ/N + (1/βN) * ln((1-ρ)/ρ)
    """
    return mu + np.log((1 - rho) / rho) / beta
