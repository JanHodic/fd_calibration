"""
Načítání dat pro experimenty.

- ODIR-5K loader (stub – nutno doplnit cestu k datům)
- Generátor syntetických dat pro odladění pipeline
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# Standardní třídy v ODIR-5K
ODIR_CLASSES = [
    "N",   # Normal (healthy)
    "D",   # Diabetic retinopathy
    "G",   # Glaucoma
    "C",   # Cataract
    "A",   # Age-related macular degeneration (AMD)
    "H",   # Hypertension
    "M",   # Pathological myopia
    "O",   # Other diseases/abnormalities
]


def load_odir5k(
    data_dir: str | Path,
    features_csv: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Načti ODIR-5K dataset.

    Předpoklad: features už extrahovány a uloženy v CSV.

    Struktura features_csv:
        - sloupce f0, f1, ..., fD-1: příznaky
        - sloupec 'label': index třídy (0..7) nebo identifikátor z ODIR_CLASSES

    Parameters
    ----------
    data_dir : path
        Kořenový adresář ODIR-5K.
    features_csv : str
        Cesta k CSV s extrahovanými příznaky (relativně k data_dir).
        Pokud None, hledá se data_dir/features.csv.

    Returns
    -------
    X : (N, D) np.ndarray
    y : (N,) np.ndarray (integer class indices)
    class_names : list[str]
    """
    data_dir = Path(data_dir)
    if features_csv is None:
        features_csv = data_dir / "features.csv"
    else:
        features_csv = data_dir / features_csv

    if not features_csv.exists():
        raise FileNotFoundError(
            f"Soubor s příznaky {features_csv} nenalezen. "
            f"Nejprve extrahuj příznaky z fundus fotografií."
        )

    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(float)

    labels = df["label"].values
    if labels.dtype == object:
        # Stringové štítky – mapuj přes ODIR_CLASSES
        mapping = {name: i for i, name in enumerate(ODIR_CLASSES)}
        y = np.array([mapping[lbl] for lbl in labels], dtype=int)
    else:
        y = labels.astype(int)

    return X, y, ODIR_CLASSES


def generate_synthetic_retina(
    n_samples: int = 2000,
    n_classes: int = 6,
    n_features: int = 32,
    class_prevalence: list[float] | None = None,
    class_separability: list[float] | None = None,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vygeneruj syntetická data připomínající strukturu ODIR-5K.

    Třídy mají různou prevalenci a různou separabilitu od ostatních
    (modeluje fakt, že např. zdravé oko je výrazně rozlišitelné, zatímco
    hypertenzní retinopatie je obtížnější).

    Parameters
    ----------
    n_samples : int
    n_classes : int
        Počet diagnostických tříd.
    n_features : int
        Dimenze příznakového prostoru.
    class_prevalence : list[float], optional
        Relativní prevalence tříd (suma = 1). Pokud None, použije se
        realistická distribuce s dominantní třídou 'zdravý'.
    class_separability : list[float], optional
        Vzdálenost středu třídy od počátku (vyšší = lépe oddělitelná).
        Pokud None, použijí se hodnoty modelující reálné nemoci sítnice.
    random_state : int

    Returns
    -------
    X, y, class_names
    """
    rng = np.random.default_rng(random_state)

    if class_prevalence is None:
        # Realistická distribuce: zdravé oko převažuje
        raw = [0.50, 0.18, 0.10, 0.08, 0.07, 0.07][:n_classes]
        s = sum(raw)
        class_prevalence = [x / s for x in raw]
    if class_separability is None:
        # Zdravé oko a glaukom dobře oddělitelné, hypertenzní hůře
        class_separability = [3.0, 2.5, 2.2, 2.0, 1.5, 1.2][:n_classes]

    class_names = ["Healthy", "Glaucoma", "DR", "AMD", "Hypertensive", "Other"][:n_classes]

    # Náhodné středy tříd na hyperkouli
    centers = rng.normal(size=(n_classes, n_features))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * np.array(class_separability)[:, None]

    # Počty vzorků v každé třídě
    counts = rng.multinomial(n_samples, class_prevalence)

    X_list = []
    y_list = []
    for k in range(n_classes):
        X_k = rng.normal(loc=centers[k], scale=1.0, size=(counts[k], n_features))
        X_list.append(X_k)
        y_list.append(np.full(counts[k], k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Promíchání
    perm = rng.permutation(len(y))
    return X[perm], y[perm], class_names
