"""
Binární Fermiho-Diracova kalibrace (Kim et al., 2021) -- přesný diskrétní model.

Rank-podmíněná pravděpodobnost pozitivní třídy je parametrizována jako
    q_r = sigma(alpha - beta * r),   r = 1..N,   alpha = beta * mu,
a parametry (beta, alpha) jsou určeny tak, aby PŘESNĚ platily obě diskrétní
podmínky zachování:
    (C1)  sum_r q_r       = N1                    (počet pozitivních)
    (C2)  sum_r r * q_r   = N1 (N+1)/2 + N1 N0 (1/2 - AUC)   (průměrný rank)

Poznámka: uzavřený vzorec pro mu/N z Kim et al. (jejich Eq. 6) je spojitá
aproximace a podmínku (C1) pro konečné N nesplňuje (např. N=4, N1=2 dává
mu=2 místo symetrického mu=2.5). Zde se proto alpha hledá numericky.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logit


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fd_prob(r: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """q_r = sigma(alpha - beta r). Pro beta=0 je q_r = sigma(alpha) konstantní."""
    return expit(alpha - beta * np.asarray(r, dtype=float))


def fd_logprob(r: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """log q_r, numericky stabilní i pro velmi malé q_r."""
    return log_expit(alpha - beta * np.asarray(r, dtype=float))


def mu_from_alpha(beta: float, alpha: float) -> float:
    """Chemický potenciál mu = alpha / beta (rank, kde q_r = 1/2). Pro beta=0 nedefinováno."""
    return float(alpha / beta) if beta > 0 else float("nan")


# ---------------------------------------------------------------------------
# Podmínky zachování
# ---------------------------------------------------------------------------

def solve_alpha(beta: float, N: int, N1: int) -> float:
    """Najdi alpha tak, aby sum_r q_r = N1 (podmínka C1).

    Levá strana je v alpha ostře rostoucí, takže kořen je jednoznačný.
    Pro beta=0 má uzavřené řešení alpha = logit(N1/N).
    """
    if N1 <= 0 or N1 >= N:
        raise ValueError(f"N1 musí být v (0, N), bylo N1={N1}, N={N}.")
    if beta == 0.0:
        return float(logit(N1 / N))
    r = np.arange(1, N + 1, dtype=float)

    def f(a: float) -> float:
        return float(expit(a - beta * r).sum()) - N1

    lo = beta * 1.0 - 60.0   # sigma(<= -60) ~ 0  -> sum ~ 0 < N1
    hi = beta * N + 60.0     # sigma(>= +60) ~ 1  -> sum ~ N > N1
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500))


def expected_auc(beta: float, alpha: float, N: int, N1: int) -> float:
    """Očekávané AUC z podmínky (C2):  AUC = 1/2 + (N1(N+1)/2 - sum_r r q_r) / (N1 N0).

    Platí přesně, pokud je splněna (C1); proto se volá až po solve_alpha.
    """
    N0 = N - N1
    r = np.arange(1, N + 1, dtype=float)
    S = float((r * fd_prob(r, beta, alpha)).sum())
    return 0.5 + (N1 * (N + 1) / 2.0 - S) / (N1 * N0)


# ---------------------------------------------------------------------------
# Inverzní úloha: (AUC, N1, N) -> (beta, alpha)
# ---------------------------------------------------------------------------

def fit_fd_binary(
    auc: float,
    N1: int,
    N: int,
    tol: float = 1e-9,
    check: bool = True,
) -> tuple[float, float]:
    """Najdi (beta, alpha) splňující přesně obě diskrétní podmínky (C1) a (C2).

    Postup: pro dané beta vyřešíme alpha z (C1); vnější kořenové hledání
    v beta pak vynutí (C2). Očekávané AUC je v beta rostoucí od 1/2 (beta=0).

    Returns
    -------
    (beta, alpha); mu = alpha / beta.
    """
    if not (0.5 <= auc <= 1.0):
        raise ValueError(f"AUC musí být v [0.5, 1.0], bylo {auc}.")
    if N1 <= 0 or N1 >= N:
        raise ValueError(f"N1 musí být v (0, N), bylo N1={N1}, N={N}.")

    # Náhodný klasifikátor: beta=0, q_r = rho
    if auc <= 0.5 + 1e-9:
        return 0.0, float(logit(N1 / N))

    def g(beta: float) -> float:
        return expected_auc(beta, solve_alpha(beta, N, N1), N, N1) - auc

    lo = 1e-8
    hi = 1.0
    while g(hi) < 0.0:
        hi *= 4.0
        if hi > 1e6:  # AUC prakticky 1 -> skoková funkce
            break
    beta = float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500))
    alpha = solve_alpha(beta, N, N1)

    if check:
        r = np.arange(1, N + 1, dtype=float)
        q = fd_prob(r, beta, alpha)
        c1 = abs(q.sum() - N1)
        c2 = abs(expected_auc(beta, alpha, N, N1) - auc)
        if c1 > 1e-6 * N or c2 > 1e-6:
            raise RuntimeError(
                f"Podmínky zachování nesplněny: |sum q - N1|={c1:.2e}, |AUC diff|={c2:.2e}"
            )
    return beta, alpha


# ---------------------------------------------------------------------------
# Optimální práh
# ---------------------------------------------------------------------------

def optimal_threshold(beta: float, alpha: float, rho: float) -> float:
    """Rank-práh r* maximalizující balanced accuracy (Eq. 9 Kim et al.):
    q_{r*} = rho  <=>  r* = (alpha - logit(rho)) / beta.
    """
    if beta <= 0:
        return float("nan")
    return float((alpha - logit(rho)) / beta)
