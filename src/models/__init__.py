"""Model definitions used by the ECG generalization benchmark."""

from .ecg_fm import ECGFMClassifier, describe_parameter_policy

__all__ = ["ECGFMClassifier", "describe_parameter_policy"]
