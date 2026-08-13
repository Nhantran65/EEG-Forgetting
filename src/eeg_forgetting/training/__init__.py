"""Training and evaluation primitives for reproducible EEG pilots."""

from .metrics import balanced_accuracy, multiclass_metrics, subject_balanced_accuracy

__all__ = ["balanced_accuracy", "multiclass_metrics", "subject_balanced_accuracy"]
