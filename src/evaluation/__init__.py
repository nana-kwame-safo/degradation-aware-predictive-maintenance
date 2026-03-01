"""
Evaluation package exports for reliability and policy-oriented diagnostics.

The package combines:
- numeric metrics and stratified diagnostics
- decision-focused reliability summaries
- policy trigger evaluation helpers
- artifact plotting utilities
"""

from .metrics import (
    error_asymmetry_metrics,
    regression_metrics,
    rul_metrics,
    stratified_metrics_by_rul_bands,
    stratified_metrics_by_rul_bins,
    summarize_unit_metrics,
    unit_level_metrics,
    unit_level_error_summary,
)
from .plots import (
    plot_error_vs_rul,
    plot_policy_timelines,
    plot_pred_vs_true,
    plot_unit_mae_hist,
)
from .policy import first_trigger, policy_metrics
from .reliability_analysis import (
    alert_threshold_metrics,
    alert_threshold_sweep,
    weighted_error_cost,
)

__all__ = [
    "error_asymmetry_metrics",
    "regression_metrics",
    "rul_metrics",
    "stratified_metrics_by_rul_bands",
    "stratified_metrics_by_rul_bins",
    "unit_level_metrics",
    "summarize_unit_metrics",
    "unit_level_error_summary",
    "alert_threshold_metrics",
    "alert_threshold_sweep",
    "weighted_error_cost",
    "first_trigger",
    "policy_metrics",
    "plot_pred_vs_true",
    "plot_error_vs_rul",
    "plot_unit_mae_hist",
    "plot_policy_timelines",
]
