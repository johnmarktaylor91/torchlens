"""Reporting helpers for TorchLens observer metadata."""

from __future__ import annotations

from typing import Any

from .. import _state
from ._explain import explain


def log_value(name: str, value: Any) -> None:
    """Record an arbitrary scalar-like value on the active ``ModelLog``.

    Parameters
    ----------
    name:
        Value name.
    value:
        Scalar or JSON-like value to record.

    Raises
    ------
    RuntimeError
        If no TorchLens capture is active.
    """

    model_log = _state._active_model_log
    if model_log is None:
        raise RuntimeError("torchlens.report.log_value() must be called during log_forward_pass.")
    model_log.report_values[str(name)] = value


__all__ = ["explain", "log_value"]
