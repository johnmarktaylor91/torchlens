"""Phase 10 tests for intervention spec persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import (
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
)
from torchlens.intervention.save import _write_tlspec_tensor_blob
from torchlens.intervention.save import resolve_function_registry_key, save_intervention
from torchlens.intervention.types import FunctionRegistryKey


class _ReluModel(nn.Module):
    """Small model with a relu site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return torch.relu(x) + 1


class _TanhModel(nn.Module):
    """Small model without a relu site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return torch.tanh(x) + 1


def _log(model: nn.Module | None = None, x: torch.Tensor | None = None) -> tl.ModelLog:
    """Capture an intervention-ready model log.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    model = model or _ReluModel()
    x = x if x is not None else torch.randn(2, 3)
    return tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)


@pytest.mark.smoke
def test_audit_save_load_and_compat(tmp_path: Path) -> None:
    """Audit save writes the tlspec directory and loaded specs compare cleanly."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.set(tl.func("relu"), tl.zero_ablate())
    path = tmp_path / "mylog.tlspec"

    log.save_intervention(path, level="audit")

    assert path.is_dir()
    assert (path / "spec.json").exists()
    assert (path / "manifest.json").exists()
    assert (path / "README.md").exists()
    assert (path / "tensors").is_dir()
    spec = tl.load_intervention_spec(path)
    assert tl.load(path) == spec
    compat = tl.check_spec_compat(spec, _log(_ReluModel(), x))
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.targets_resolve_identically is True


@pytest.mark.smoke
def test_portable_rejects_opaque_hook(tmp_path: Path) -> None:
    """Portable saves fail closed for opaque local callables."""

    log = _log()

    def opaque_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Return activation unchanged.

        Parameters
        ----------
        activation:
            Activation tensor.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Input activation.
        """

        del hook
        return activation

    log.attach_hooks({tl.func("relu"): opaque_hook})
    with pytest.raises(OpaqueCallableInExecutableSaveError):
        log.save_intervention(tmp_path / "opaque.tlspec", level="portable")


@pytest.mark.smoke
def test_function_resolution_failure_raises() -> None:
    """Unresolvable function registry keys raise replay precondition errors."""

    key = FunctionRegistryKey(
        namespace="torch",
        qualname="definitely_missing_torch_function",
        dispatch_kind="function",
    )
    with pytest.raises(ReplayPreconditionError):
        resolve_function_registry_key(key)


@pytest.mark.smoke
def test_target_manifest_mismatch_returns_fail(tmp_path: Path) -> None:
    """Selectors resolving to nothing on a new log produce FAIL compatibility."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.set(tl.func("relu"), tl.zero_ablate())
    path = tmp_path / "target.tlspec"
    log.save_intervention(path, level="audit")

    spec = tl.load_intervention_spec(path)
    compat = tl.check_spec_compat(spec, _log(_TanhModel(), x))

    assert compat.outcome == "FAIL"
    assert compat.diff.missing_labels


@pytest.mark.smoke
def test_atomic_save_cleans_up_after_tensor_write_failure(tmp_path: Path) -> None:
    """A tensor-sidecar write exception leaves no final partial tlspec dir."""

    log = _log()
    log.set(tl.func("relu"), torch.zeros(2, 3))
    target = tmp_path / "crash.tlspec"

    def crashing_writer(**kwargs: Any) -> Any:
        """Write one sidecar and then simulate a crash.

        Parameters
        ----------
        **kwargs:
            Tensor writer keyword arguments.

        Returns
        -------
        Any
            Never returned.
        """

        _write_tlspec_tensor_blob(**kwargs)
        raise RuntimeError("simulated tensor crash")

    with pytest.raises(RuntimeError, match="simulated tensor crash"):
        save_intervention(
            log,
            target,
            level="audit",
            _write_tensor_blob_fn=crashing_writer,
        )

    assert not target.exists()
    assert not list(tmp_path.glob("tmp.*"))
