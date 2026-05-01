"""Compatibility adapter namespace reserved for TorchLens 2.0."""

from . import lovely, torchshow
from .torchextractor import Extractor

__all__ = ["Extractor", "lovely", "torchshow"]
