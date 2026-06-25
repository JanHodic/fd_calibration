"""
Demo: spustit experiment na syntetických datech imitujících strukturu ODIR-5K.

Použití:
    python demo.py                  # syntetická data, Random Forest
    python demo.py --clf gb         # Gradient Boosting
    python demo.py --clf svm        # SVM
"""

from __future__ import annotations

import argparse

from fd_calibration import (
    generate_synthetic_retina,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf", default="rf", choices=["rf", "gb", "svm"],
                        help="Klasifikátor: rf|gb|svm")
    parser.add_argument("--n", type=int, default=3000, help="Počet vzorků")
    parser.add_argument("--k", type=int, default=6, help="Počet tříd")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generuji {args.n} syntetických vzorků pro {args.k} tříd...")
    X, y, class_names = generate_synthetic_retina(
        n_samples=args.n,
        n_classes=args.k,
        random_state=args.seed,
    )
    print(f"Tvar dat: X={X.shape}, y={y.shape}")
    print(f"Třídy: {class_names}")
    import numpy as np
    unique, counts = np.unique(y, return_counts=True)
    print(f"Rozložení tříd: {dict(zip([class_names[i] for i in unique], counts))}")

    print(f"\nKlasifikátor: {args.clf}")
    run_experiment(
        X, y, class_names,
        classifier_name=args.clf,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
