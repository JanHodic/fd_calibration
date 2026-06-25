# FD Calibration

Vícetřídní kalibrace klasifikátorů přes Fermiho–Diracovo rozdělení.
Aplikace: diagnostika onemocnění sítnice (ODIR-5K).

## Instalace

```bash
pip install -r requirements.txt
```

## Rychlý start

```bash
python demo.py                    # Random Forest, syntetická data
python demo.py --clf gb           # Gradient Boosting
python demo.py --clf svm --n 2000 # SVM, menší dataset
```

## Použití v kódu

### Samostatný FD kalibrátor

```python
from fd_calibration import FDMulticlassCalibrator

# scores_val: (N_val, K) skóre z klasifikátoru
# y_val: (N_val,) štítky
calib = FDMulticlassCalibrator()
calib.fit(scores_val, y_val)

# Kalibrované pravděpodobnosti
probs = calib.predict_proba(scores_test)

# Souhrn nafitovaných parametrů
print(calib.summary())
```

### End-to-end experiment

```python
from fd_calibration import generate_synthetic_retina, run_experiment

X, y, class_names = generate_synthetic_retina(n_samples=3000, n_classes=6)
results = run_experiment(X, y, class_names, classifier_name="rf")
```

### Reálná data (ODIR-5K)

```python
from fd_calibration import load_odir5k, run_experiment

# Připrav features.csv (sloupce f0...f<D-1>, label)
X, y, class_names = load_odir5k("/path/to/odir5k", features_csv="features.csv")
results = run_experiment(X, y, class_names, classifier_name="rf")
```

## Struktura balíčku

```
fd_calibration/
├── core.py            # Binární FD: fit (β, μ), AUC, optimální práh
├── multiclass.py      # FDMulticlassCalibrator (degenerované hladiny)
├── metrics.py         # ECE, Brier, reliability, OvR AUC
├── data.py            # ODIR-5K loader + syntetický generátor
└── pipeline.py        # End-to-end experiment + baseline kalibrátory
demo.py                # Spustitelné demo
```

## Co každá komponenta dělá

- **`core.fit_fd_binary(auc, N1, N)`** → najde `(β, μ)` tak, aby FD distribuce
  reprodukovala zadané AUC při dané prevalenci. Numericky stabilní (log-sinh).

- **`FDMulticlassCalibrator`** → fituje per-class `(β_k, μ_k)` v one-vs-rest schématu,
  pro nové vzorky vypočítá rank v rámci validační sady a aplikuje FD,
  výsledné pravděpodobnosti normalizuje přes třídy.

- **`run_experiment`** → porovná FD kalibraci s baseline (Platt OvR, Isotonic OvR,
  Temperature scaling, žádná kalibrace) na metrikách accuracy, balanced accuracy,
  ECE, Brier.

## Další kroky

Až budou připraveny příznaky z fundus obrázků (CNN embeddingy, HOG, DINOv2),
stačí je uložit do CSV ve formátu pro `load_odir5k` a spustit `run_experiment`.
