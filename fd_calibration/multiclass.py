"""
Vícetřídní FD kalibrace pomocí degenerovaných hladin.

Pro každou třídu k (one-vs-rest):
- AUC_k a prevalence rho_k na validační sadě,
- je-li AUC_k < 1/2, OTOČÍ se znaménko skóre (stejně při fitu i predikci),
- fit (beta_k, alpha_k) přesným diskrétním modelem (core.fit_fd_binary),
- při selhání / chybějící třídě fallback na konstantní prevalenci
  (beta=0, alpha=logit(rho_k)) -- zaznamenáno v params[k]["fallback"].

Predikce: rank nového vzorku v rámci validační sady, log q_k, normalizace
přes třídy v logaritmickém prostoru (log-sum-exp), takže řádky sčítají
na 1 i pro velmi malé pravděpodobnosti.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import logit, logsumexp
from sklearn.metrics import roc_auc_score

from .core import fit_fd_binary, fd_logprob, mu_from_alpha


class FDMulticlassCalibrator:

    def __init__(self) -> None:
        self.params: dict[int, dict] = {}
        self.classes_: np.ndarray | None = None
        self._sorted_val: list[np.ndarray] = []   # setříděná (již otočená) validační skóre
        self.N_: int | None = None

    # ------------------------------------------------------------------
    def fit(self, scores, y, classes=None) -> "FDMulticlassCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)
        if classes is None:
            classes = np.unique(y)
        self.classes_ = np.asarray(classes)
        K = len(self.classes_)
        self.N_ = scores.shape[0]
        if scores.shape[1] != K:
            raise ValueError(f"scores má {scores.shape[1]} sloupců, ale tříd je {K}.")

        self.params.clear()
        self._sorted_val = []

        for k_idx, k in enumerate(self.classes_):
            y_bin = (y == k).astype(int)
            N_k = int(y_bin.sum())
            rho = N_k / self.N_
            s_k = scores[:, k_idx].copy()
            sign = 1.0
            par = {"N_k": N_k, "rho": rho, "sign": sign, "fallback": None}

            if N_k == 0 or N_k == self.N_:
                par.update(beta=0.0, alpha=float(logit(np.clip(rho, 1e-6, 1 - 1e-6))),
                           auc=np.nan, fallback="missing_class")
                warnings.warn(f"Třída {k}: N_k={N_k} z {self.N_}; fallback na konstantní prevalenci.")
            else:
                auc = float(roc_auc_score(y_bin, s_k))
                if auc < 0.5:                       # bod 2: skutečně otočit pořadí
                    sign = -1.0
                    s_k = -s_k
                    auc = 1.0 - auc
                par["sign"] = sign
                par["auc"] = auc
                try:
                    beta, alpha = fit_fd_binary(auc, N_k, self.N_)
                    par.update(beta=beta, alpha=alpha)
                except Exception as e:              # bod 3: definovaný a zaznamenaný fallback
                    par.update(beta=0.0, alpha=float(logit(rho)), fallback=f"fit_failed: {e}")
                    warnings.warn(f"Třída {k}: fit selhal ({e}); fallback na konstantní prevalenci.")

            par["mu"] = mu_from_alpha(par["beta"], par["alpha"])
            self.params[int(k)] = par
            self._sorted_val.append(np.sort(s_k))
        return self

    # ------------------------------------------------------------------
    def _ranks(self, new_scores: np.ndarray) -> np.ndarray:
        """Rank (1 = nejvyšší) každého nového vzorku v rámci validační sady, per třída."""
        n_new = new_scores.shape[0]
        R = np.empty((n_new, len(self.classes_)))
        for k_idx, k in enumerate(self.classes_):
            s = self.params[int(k)]["sign"] * new_scores[:, k_idx]      # stejné otočení jako při fitu
            n_higher = self.N_ - np.searchsorted(self._sorted_val[k_idx], s, side="right")
            R[:, k_idx] = n_higher + 1
        return R

    def predict_log_proba(self, scores) -> np.ndarray:
        if not self.params:
            raise RuntimeError("Kalibrátor není fitován.")
        scores = np.asarray(scores, dtype=float)
        R = self._ranks(scores)
        L = np.empty_like(R)
        for k_idx, k in enumerate(self.classes_):
            p = self.params[int(k)]
            L[:, k_idx] = fd_logprob(R[:, k_idx], p["beta"], p["alpha"])
        return L - logsumexp(L, axis=1, keepdims=True)                  # bod 4: normalizace v log-prostoru

    def predict_proba(self, scores) -> np.ndarray:
        return np.exp(self.predict_log_proba(scores))

    def predict(self, scores) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(scores), axis=1)]

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if not self.params:
            return "Kalibrátor není fitován."
        lines = ["Třída | N_k    | ρ_k    | AUC_k  | β_k        | μ_k       | sign | fallback"]
        lines.append("-" * 85)
        for k in self.classes_:
            p = self.params[int(k)]
            lines.append(
                f"{str(k):6s}| {p['N_k']:6d} | {p['rho']:.4f} | {p['auc']:.4f} | "
                f"{p['beta']:.4e} | {p['mu']:9.2f} | {p['sign']:+.0f}   | {p['fallback'] or '-'}"
            )
        return "\n".join(lines)
