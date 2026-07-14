"""Data-integrity core for the TorchLens model-menagerie crawler."""

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
)

__version__ = "0.1.0"

__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "__version__",
]
