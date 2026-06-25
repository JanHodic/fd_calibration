"""FD Calibration: vícetřídní kalibrace klasifikátorů přes Fermi-Diracovo rozdělení."""

from .core import (
    fd_prob,
    mu_over_N,
    auc_from_params,
    fit_fd_binary,
    optimal_threshold,
)
from .multiclass import FDMulticlassCalibrator
from .metrics import (
    expected_calibration_error,
    brier_score_multiclass,
    reliability_data,
    ovr_auc,
    accuracy,
    balanced_accuracy,
)
from .pipeline import (
    run_experiment,
    print_results,
    ExperimentResult,
    build_classifier,
    TemperatureScalingCalibrator,
    PlattOvRCalibrator,
    IsotonicOvRCalibrator,
)
from .data import (
    load_odir5k,
    generate_synthetic_retina,
    ODIR_CLASSES,
)

__all__ = [
    "fd_prob",
    "mu_over_N",
    "auc_from_params",
    "fit_fd_binary",
    "optimal_threshold",
    "FDMulticlassCalibrator",
    "expected_calibration_error",
    "brier_score_multiclass",
    "reliability_data",
    "ovr_auc",
    "accuracy",
    "balanced_accuracy",
    "run_experiment",
    "print_results",
    "ExperimentResult",
    "build_classifier",
    "TemperatureScalingCalibrator",
    "PlattOvRCalibrator",
    "IsotonicOvRCalibrator",
    "load_odir5k",
    "generate_synthetic_retina",
    "ODIR_CLASSES",
]
