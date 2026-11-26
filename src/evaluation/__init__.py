"""Evaluation metrics: pixel-level classification metrics and biometric HC metrics."""

from src.evaluation.biometric_metrics import (
    assign_trimester_group,
    compute_sample_biometric_metrics,
    dice_coefficient_pct,
    hausdorff_distance_mm,
    summarize_biometric_results,
    summarize_by_trimester,
)
from src.evaluation.pixel_metrics import (
    PixelEvalResult,
    evaluate_arm_pixel_metrics,
    plot_confusion_matrices,
    plot_literature_comparison,
    plot_roc_curves,
    results_to_dataframe,
)

__all__ = [
    "dice_coefficient_pct",
    "hausdorff_distance_mm",
    "compute_sample_biometric_metrics",
    "assign_trimester_group",
    "summarize_biometric_results",
    "summarize_by_trimester",
    "PixelEvalResult",
    "evaluate_arm_pixel_metrics",
    "results_to_dataframe",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "plot_literature_comparison",
]
