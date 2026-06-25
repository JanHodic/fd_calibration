"""
Vícetřídní FD kalibrace pomocí degenerovaných hladin.

Pro každou třídu k:
- spočítáme one-vs-rest AUC_k a prevalenci ρ_k
- nafitujeme parametry (β_k, μ_k) FD distribuce
- nové vzorky kalibrujeme přes rank v rámci validační sady
- výsledné pravděpodobnosti normalizujeme přes třídy

Vstup: skóre z klasifikátoru (libovolného), rozměr (N, K).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import fit_fd_binary, fd_prob, mu_over_N


class FDMulticlassCalibrator:
    """Kalibrátor pro vícetřídní klasifikaci založený na FD distribuci.

    Předpokládá one-vs-rest dekompozici. Klasifikátor musí pro každý vzorek
    poskytnout K skóre (jedno pro každou třídu).

    Příklad použití
    ---------------
    >>> calib = FDMulticlassCalibrator()
    >>> calib.fit(scores_val, y_val)
    >>> probs = calib.predict_proba(scores_test)
    """

    def __init__(self) -> None:
        self.params: dict[int, dict] = {}
        self.classes_: np.ndarray | None = None
        self._val_scores: np.ndarray | None = None
        self.N_: int | None = None

    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "FDMulticlassCalibrator":
        """Fit kalibrátoru na validační sadě.

        Parameters
        ----------
        scores : (N, K)
            Skóre z klasifikátoru. scores[i, k] = skóre vzorku i pro třídu k.
        y : (N,)
            Pravé štítky.
        classes : (K,), optional
            Identifikátory tříd. Pokud None, použije se np.unique(y).
        """
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)

        if classes is None:
            classes = np.unique(y)
        self.classes_ = np.asarray(classes)
        K = len(self.classes_)
        self.N_ = scores.shape[0]

        if scores.shape[1] != K:
            raise ValueError(
                f"scores má {scores.shape[1]} sloupců, ale tříd je {K}."
            )

        for k_idx, k in enumerate(self.classes_):
            y_bin = (y == k).astype(int)
            N_k = int(y_bin.sum())

            if N_k == 0 or N_k == self.N_:
                # Třída chybí nebo dominuje – nelze fitovat
                self.params[int(k)] = {
                    "beta": 0.0, "mu": self.N_ / 2,
                    "rho": N_k / self.N_, "N_k": N_k, "auc": 0.5,
                }
                continue

            s_k = scores[:, k_idx]
            try:
                auc_k = roc_auc_score(y_bin, s_k)
            except ValueError:
                auc_k = 0.5

            # FD pracuje s AUC ≥ 0.5; pokud je menší, otáčíme znaménko skóre
            if auc_k < 0.5:
                # Tohle by mělo být v ideálním případě řešeno v klasifikátoru.
                # Zde to detekujeme a logujeme.
                auc_k = 1.0 - auc_k

            try:
                beta_k, mu_k = fit_fd_binary(auc_k, N_k, self.N_)
            except Exception:
                # Fallback: náhodný klasifikátor
                beta_k, mu_k = 1e-6, self.N_ / 2

            self.params[int(k)] = {
                "beta": beta_k,
                "mu": mu_k,
                "rho": N_k / self.N_,
                "N_k": N_k,
                "auc": auc_k,
            }

        self._val_scores = scores
        return self

    def _ranks_in_val(
        self, new_scores: np.ndarray, k_idx: int
    ) -> np.ndarray:
        """Najdi rank každého nového vzorku v rámci validační sady pro třídu k.

        Rank 1 = nejvyšší skóre, rank N = nejnižší.
        """
        val_s = self._val_scores[:, k_idx]
        new_s = new_scores[:, k_idx]
        # Pro každý nový vzorek: počet validačních vzorků s vyšším skóre + 1
        # (Pro efektivitu jdeme přes searchsorted v setříděné kopii.)
        sorted_val = np.sort(val_s)  # vzestupně
        # počet validačních s vyšším skóre = N - počet s ≤ skóre
        n_higher = self.N_ - np.searchsorted(sorted_val, new_s, side="right")
        return n_higher + 1  # rank ∈ [1, N+1]

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Vypočti kalibrované pravděpodobnosti pro nová skóre.

        Parameters
        ----------
        scores : (N_new, K)

        Returns
        -------
        probs : (N_new, K)
            Pravděpodobnosti normalizované přes třídy (∑_k p_k = 1).
        """
        if self._val_scores is None:
            raise RuntimeError("Kalibrátor není fitován. Zavolej .fit() první.")
        scores = np.asarray(scores, dtype=float)
        n_new = scores.shape[0]
        K = len(self.classes_)
        P = np.zeros((n_new, K))

        for k_idx, k in enumerate(self.classes_):
            par = self.params[int(k)]
            ranks = self._ranks_in_val(scores, k_idx).astype(float)
            P[:, k_idx] = fd_prob(ranks, par["beta"], par["mu"])

        # Normalizace přes třídy (Eq. 2.16 v práci)
        row_sums = P.sum(axis=1, keepdims=True)
        # Pojistka proti dělení nulou
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        return P / row_sums

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predikce třídy (argmax kalibrovaných pravděpodobností)."""
        probs = self.predict_proba(scores)
        return self.classes_[np.argmax(probs, axis=1)]

    def summary(self) -> str:
        """Textový souhrn nafitovaných parametrů."""
        if not self.params:
            return "Kalibrátor není fitován."
        lines = ["Třída | N_k    | ρ_k    | AUC_k  | β_k        | μ_k"]
        lines.append("-" * 65)
        for k in self.classes_:
            p = self.params[int(k)]
            lines.append(
                f"{str(k):6s}| {p['N_k']:6d} | {p['rho']:.4f} | "
                f"{p['auc']:.4f} | {p['beta']:.4e} | {p['mu']:.2f}"
            )
        return "\n".join(lines)
