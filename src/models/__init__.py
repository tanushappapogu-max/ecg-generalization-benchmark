"""Model definitions used by the ECG generalization benchmark."""

from .ecg_fm import ECGFMClassifier, describe_parameter_policy
from .inception_time import InceptionTime1D
from .resnet1d import ResNet1D
from .transformer1d import ECGTransformer1D

__all__ = [
    "ECGFMClassifier",
    "ECGTransformer1D",
    "InceptionTime1D",
    "ResNet1D",
    "describe_parameter_policy",
]
