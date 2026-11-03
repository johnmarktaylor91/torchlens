"""Environment and model compatibility diagnostics, without executing user models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .._doctor_types import (
    _SHOW_CELL_WIDTH_CAP as _SHOW_CELL_WIDTH_CAP,
    DoctorCheck as DoctorCheck,
    DoctorReport as DoctorReport,
    Severity,
    Status,
    register_capability_snapshot_provider,
)
from ..compat._report import _model_checks, _qualified_type_name
from ._environment import _environment_checks, capability_snapshot

# The report types live in the basis module so the backend scorecard and ``compat`` can
# import them without reaching this package; ``torchlens.report.DoctorCheck`` stays the
# public path through these re-exports. The full capability snapshot (torch plus the tf
# backend's flags) is this package's knowledge, so it is handed to the types here.
register_capability_snapshot_provider(capability_snapshot)


def doctor(model: nn.Module | None = None, inputs: Any = None) -> DoctorReport:
    """Inspect the environment and, optionally, a model and its input tree.

    Parameters
    ----------
    model:
        Optional eager model to inspect; its forward is never executed.
    inputs:
        Optional input tree inspected only when a model is supplied.

    Returns
    -------
    DoctorReport
        Environment checks, the ``backend_scorecard:<name>`` rows, then model
        compatibility findings, if a model was supplied.

    Examples
    --------
    >>> report = tl.report.doctor(model, x)
    >>> type(report).__name__
    'DoctorReport'
    >>> report.torch_version == torch.__version__
    True
    >>> len(report.checks) > 0
    True
    """

    from ..backends._scorecard_rows import backend_scorecard_rows

    checks = tuple(_environment_row(row) for row in _environment_checks())
    checks += backend_scorecard_rows()
    model_type = None
    if model is not None:
        checks += _model_checks(model, inputs)
        model_type = _qualified_type_name(model)
    return DoctorReport(model_type, str(torch.__version__), checks)


_ENVIRONMENT_STATUS: dict[str, tuple[Status, Severity]] = {
    "PASS": ("pass", "ok"),
    "WARN": ("scope", "warning"),
    "FAIL": ("known_broken", "error"),
    "SKIP": ("not_tested", "info"),
}
_ENVIRONMENT_REMEDIES = {
    "pytorch": "Install a supported PyTorch build.",
    "runtime capabilities": "Inspect capability_snapshot() for unavailable runtime integrations.",
    "torch wrapper bindings": "Reinstall wrappers with wrap_torch() and refresh stale references.",
    "mechanical belt": "Run a capture first; avoid pre-wrap references to failed belt probes.",
    "cuda": "Use CPU capture or install a CUDA-enabled PyTorch build on a compatible device.",
    "graphviz": "Install torchlens[viz] and the Graphviz system executable.",
    "safetensors": "Install safetensors to enable portable tensor artifacts.",
    "extras": "Reinstall TorchLens with the optional extras needed for your workflow.",
    "model fingerprint": "Check the fingerprint diagnostic and report the failing environment.",
}


def _environment_row(row: Any) -> DoctorCheck:
    """Adapt an existing environment probe to the unified status vocabulary."""

    status, severity = _ENVIRONMENT_STATUS[row.status]
    key = "torch_capabilities" if row.name == "runtime capabilities" else row.name
    return DoctorCheck(
        key,
        row.name.replace("_", " ").capitalize(),
        status,
        severity,
        status != "pass",
        row.detail,
        _ENVIRONMENT_REMEDIES[row.name] if status != "pass" else "",
    )
