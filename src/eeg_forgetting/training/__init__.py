"""Training and evaluation primitives for reproducible EEG pilots."""

from .metrics import balanced_accuracy, subject_balanced_accuracy

__all__ = ["balanced_accuracy", "subject_balanced_accuracy"]
