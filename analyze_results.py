"""
Agregace výsledků z run_experiment_grid.py.

Vstup: results.parquet
Výstup: tabulky win rate, Wilcoxon testy, grafy závislosti na parametrech,
        LaTeX tabulky.

Spuštění:
    python analyze_results.py --input results.parquet --out-dir analysis/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


METRICS = ["accuracy", "bal_accuracy", "ece", "brier"]
LOWER_IS_BETTER = {"ece", "brier"}
GROUP_COLS = ("K", "N", "sigma", "profile", "classifier", "seed")


# ---------------------------------------------------------------------------
# Načtení
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    df = df[df["method"] != "ERROR"]
    return df


# ---------------------------------------------------------------------------
# Párové srovnání
# ---------------------------------------------------------------------------

def paired_differences(
    df: pd.DataFrame, method_a: str, method_b: str, metric: str,
) -> np.ndarray:
    """Vektor rozdílů (A − B) na stejné konfiguraci + seedu.
    Znaménko přetočeno tak, aby kladné vždy znamenalo 'A je lepší'."""
    key = list(GROUP_COLS)
    a = df[df["method"] == method_a][key + [metric]]
    b = df[df["method"] == method_b][key + [metric]]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    diffs = m[f"{metric}_a"].values - m[f"{metric}_b"].values
    if metric in LOWER_IS_BETTER:
        diffs = -diffs
    return diffs


def wilcoxon_summary(diffs: np.ndarray, n_boot: int = 10000) -> dict:
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 5:
        return {"n": len(diffs), "p": np.nan, "median": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "win_rate": np.nan}
    try:
        _, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="pratt")
    except ValueError:
        p = np.nan
    rng = np.random.default_rng(42)
    boot = np.array([
        np.median(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(diffs), "p": p, "median": float(np.median(diffs)),
        "ci_low": float(lo), "ci_high": float(hi),
        "win_rate": float((diffs > 0).mean()),
    }


def pairwise_summary(df: pd.DataFrame, target: str = "fd") -> pd.DataFrame:
    rows = []
    for baseline in [m for m in df["method"].unique() if m != target]:
        for metric in METRICS:
            d = paired_differences(df, target, baseline, metric)
            rows.append({"target": target, "baseline": baseline,
                         "metric": metric, **wilcoxon_summary(d)})
    return pd.DataFrame(rows)


def win_rate_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    classifiers = sorted(df["classifier"].unique())
    methods = [m for m in sorted(df["method"].unique()) if m != "fd"]
    mat = np.full((len(classifiers), len(methods)), np.nan)
    for i, clf in enumerate(classifiers):
        sub = df[df["classifier"] == clf]
        for j, bl in enumerate(methods):
            d = paired_differences(sub, "fd", bl, metric)
            if len(d):
                mat[i, j] = float((d > 0).mean())
    return pd.DataFrame(mat, index=classifiers, columns=methods)


def dependency_curve(
    df: pd.DataFrame, param: str, metric: str,
    target: str = "fd", baseline: str = "temperature",
) -> pd.DataFrame:
    rows = []
    for val in sorted(df[param].unique()):
        d = paired_differences(df[df[param] == val], target, baseline, metric)
        rows.append({param: val, **wilcoxon_summary(d, n_boot=2000)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grafy
# ---------------------------------------------------------------------------

def plot_dependency(df, param, metric, out_path, target="fd", baseline="temperature"):
    c = dependency_curve(df, param, metric, target, baseline)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = c[param].values
    ax.plot(x, c["median"].values, "o-", label=f"medián Δ ({target} − {baseline})")
    ax.fill_between(x, c["ci_low"].values, c["ci_high"].values,
                    alpha=0.25, label="95 % bootstrap CI")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(param)
    ax.set_ylabel(f"Δ {metric} (kladné = {target} lepší)")
    ax.set_title(f"Závislost zisku {target} vs {baseline} na {param}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_rate_heatmap(df, metric, out_path):
    mat = win_rate_matrix(df, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.3 < v < 0.7 else "white")
    fig.colorbar(im, ax=ax, label="Win rate FD")
    ax.set_title(f"Win rate FD vs baseline — {metric}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "---"
    if p < 1e-10:
        return r"$< 10^{-10}$"
    if p < 1e-3:
        return f"${p:.1e}$"
    return f"${p:.3f}$"


def latex_pairwise_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Baseline & Metrika & $n$ & Medián $\Delta$ & 95\,\% CI & Win rate & $p$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['baseline']} & {r['metric']} & {int(r['n'])} & "
            f"{r['median']:+.4f} & [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & "
            f"{r['win_rate']:.3f} & {_fmt_p(r['p'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.parquet")
    parser.add_argument("--out-dir", default="analysis_out")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    df = load_results(args.input)
    print(f"Načteno {len(df)} řádků, metody: {sorted(df['method'].unique())}")

    # 1. Celkový pairwise summary
    summary = pairwise_summary(df)
    print("\n=== FD vs baseline (všechny konfigurace) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out / "pairwise_summary.csv", index=False)
    (out / "pairwise_summary.tex").write_text(latex_pairwise_table(summary), encoding="utf-8")

    # 2. Win rate heatmapy
    for metric in METRICS:
        plot_win_rate_heatmap(df, metric, out / f"winrate_{metric}.png")

    # 3. Závislosti na parametrech
    for param in ["K", "N", "sigma"]:
        for metric in ["bal_accuracy", "ece"]:
            plot_dependency(df, param, metric, out / f"dep_{param}_{metric}.png")

    # 4. Breakdown podle profilu a klasifikátoru
    for col in ["profile", "classifier"]:
        for val in df[col].unique():
            s = pairwise_summary(df[df[col] == val])
            s.to_csv(out / f"summary_{col}_{val}.csv", index=False)
            print(f"\n=== {col} = {val} ===")
            print(s[["baseline", "metric", "median", "win_rate", "p"]].to_string(index=False))

    print(f"\nHotovo. Výstupy v {out}/")


if __name__ == "__main__":
    main()