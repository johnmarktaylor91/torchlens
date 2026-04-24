"""Robustness sprint PR 1 — regression coverage for correctness guards.

Covers:
    - Nested ``log_forward_pass`` / ``active_logging`` must raise rather than
      silently corrupt the outer ModelLog.
    - functorch / vmap / grad transforms emit a one-shot UserWarning so the
      user knows their log is incomplete (rather than silently empty).
    - pyproject pins ``torch>=2.4`` (matching the autocast API already in use)
      and advertises Python 3.13 support.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state


class _Tiny(nn.Module):
    """Two-layer model small enough to log cheaply in many test variations."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


# ---------------------------------------------------------------------------
# Nested log_forward_pass guard
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_nested_active_logging_raises_runtime_error() -> None:
    """Entering ``active_logging`` while one is already active must raise.

    Previously this silently overwrote ``_active_model_log`` and cleared it on
    inner exit, corrupting the outer log mid-pass. The guard turns silent
    data corruption into a loud error.
    """
    outer = object()  # active_logging only reads the reference; None sentinel
    inner = object()

    with pytest.raises(RuntimeError, match="not re-entrant"):
        with _state.active_logging(outer):  # type: ignore[arg-type]
            with _state.active_logging(inner):  # type: ignore[arg-type]
                pass


@pytest.mark.smoke
def test_nested_log_forward_pass_via_forward_hook_raises() -> None:
    """A user forward hook that calls log_forward_pass must fail loudly.

    This is the realistic nested-logging trigger: the hook runs during the
    outer's ``active_logging`` context (during the forward pass, inside the
    decorated wrapper), so the inner ``log_forward_pass`` hits the guard.
    """
    outer = _Tiny()
    inner_model = _Tiny()
    x = torch.randn(2, 4)

    def evil_hook(_module, _inputs, _output):
        tl.log_forward_pass(inner_model, x)

    outer.a.register_forward_hook(evil_hook)

    with pytest.raises(RuntimeError, match="not re-entrant"):
        tl.log_forward_pass(outer, x, layers_to_save="none")


def test_logging_state_cleared_after_guard_fires() -> None:
    """After the nested-call guard raises, the outer session must still exit cleanly."""
    model = _Tiny()
    x = torch.randn(2, 4)

    with pytest.raises(RuntimeError, match="not re-entrant"):
        with _state.active_logging(object()):  # type: ignore[arg-type]
            with _state.active_logging(object()):  # type: ignore[arg-type]
                pass

    assert _state._logging_enabled is False
    assert _state._active_model_log is None

    # Follow-up forward pass should succeed — the outer `with` cleaned up.
    log = tl.log_forward_pass(model, x, layers_to_save="none")
    assert len(log.layer_logs) > 0


# ---------------------------------------------------------------------------
# Functorch / vmap skip-warning
# ---------------------------------------------------------------------------


_HAS_FUNCTORCH_VMAP = hasattr(torch, "func") and hasattr(torch.func, "vmap")


@pytest.mark.skipif(not _HAS_FUNCTORCH_VMAP, reason="torch.func.vmap not available")
def test_vmap_emits_userwarning_once_per_session() -> None:
    """When a model's forward uses torch.func.vmap, TorchLens skips the inner ops
    but must warn once so the user knows the log is incomplete.
    """

    class VmappedSum(nn.Module):
        """Uses vmap internally but returns a tensor from a normal (loggable) op
        so the output path still exercises tl_* tagging.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = torch.func.vmap(lambda row: row.sum())(x)
            return v + 1  # normal op outside vmap, tagged by TorchLens

    model = VmappedSum()
    x = torch.randn(4, 8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.log_forward_pass(model, x, layers_to_save="none")

    vmap_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "functorch" in str(w.message).lower()
    ]
    assert len(vmap_warnings) == 1, (
        f"Expected exactly one functorch warning, got {len(vmap_warnings)}: "
        f"{[str(w.message) for w in vmap_warnings]}"
    )


@pytest.mark.skipif(not _HAS_FUNCTORCH_VMAP, reason="torch.func.vmap not available")
def test_vmap_warning_flag_resets_between_sessions() -> None:
    """Each call to log_forward_pass starts with a fresh warning flag."""

    class VmappedSum(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = torch.func.vmap(lambda row: row.sum())(x)
            return v + 1

    model = VmappedSum()
    x = torch.randn(4, 8)

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        tl.log_forward_pass(model, x, layers_to_save="none")
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        tl.log_forward_pass(model, x, layers_to_save="none")

    first_count = sum(1 for w in first if "functorch" in str(w.message).lower())
    second_count = sum(1 for w in second if "functorch" in str(w.message).lower())
    assert first_count == 1
    assert second_count == 1, "warning flag should reset between sessions"


def test_non_vmap_forward_pass_emits_no_functorch_warning() -> None:
    """A normal forward pass must not produce spurious functorch warnings."""
    model = _Tiny()
    x = torch.randn(2, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.log_forward_pass(model, x, layers_to_save="none")

    functorch_warnings = [w for w in caught if "functorch" in str(w.message).lower()]
    assert functorch_warnings == []


# ---------------------------------------------------------------------------
# Version floor + classifiers
# ---------------------------------------------------------------------------


def test_pyproject_pins_torch_floor() -> None:
    """The torch dependency must pin >=2.4 — earlier versions lack the autocast
    API signatures that TorchLens already uses (torch/amp refactor, 2.4).
    """
    from pathlib import Path

    try:
        import tomllib
    except ImportError:  # python 3.10 fallback
        import tomli as tomllib  # type: ignore[import-not-found]

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    deps = data["project"]["dependencies"]
    torch_pins = [d for d in deps if d.startswith("torch") and "=" in d]
    assert torch_pins, f"Expected a pinned torch dependency; got deps={deps}"
    # The pin should be torch>=2.4 (or higher) — not a bare 'torch'.
    pin = torch_pins[0]
    assert ">=2.4" in pin or ">=2.5" in pin or ">=2.6" in pin, f"torch floor too loose: {pin!r}"


def test_pyproject_advertises_python_313_classifier() -> None:
    """Python 3.13 is stable and torch 2.5+ supports it; classifier should reflect that."""
    from pathlib import Path

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    classifiers = data["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.13" in classifiers
