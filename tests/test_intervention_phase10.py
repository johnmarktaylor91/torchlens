"""Phase 10 tests for intervention spec persistence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
from torchlens.intervention.save import _sync_spec_records_from_log
from torchlens.intervention.save import resolve_function_registry_key, save_intervention
from torchlens.intervention.types import FireRecord, FunctionRegistryKey, InterventionSpec


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


def _log(model: nn.Module | None = None, x: torch.Tensor | None = None) -> tl.Trace:
    """Capture an intervention-ready model log.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tl.Trace
        Captured model log.
    """

    model = model or _ReluModel()
    x = x if x is not None else torch.randn(2, 3)
    return tl.trace(model, x, intervention_ready=True)


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
def test_live_forward_records_persist_in_saved_intervention_spec(tmp_path: Path) -> None:
    """Live forward hook records survive intervention spec save/load."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.func("relu"): tl.zero_ablate()},
    )
    path = tmp_path / "forward_records.tlspec"

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    assert spec.records
    assert all(isinstance(record, FireRecord) for record in spec.records)
    assert any(record.direction == "forward" for record in spec.records)


@pytest.mark.smoke
def test_live_backward_records_persist_in_saved_intervention_spec(tmp_path: Path) -> None:
    """Live backward hook records survive intervention spec save/load."""

    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(_ReluModel(), x, save_grads="all", backward_ready=True)
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_clamp(0, 0), confirm_mutation=True)
    log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    path = tmp_path / "backward_records.tlspec"

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    backward_records = [record for record in spec.records if record.direction == "backward"]
    assert backward_records
    assert backward_records[0].backward_pass_index == 1
    assert backward_records[0].call_index == 1


def test_intervention_tlspec_v2_writes_and_v1_still_loads(tmp_path: Path) -> None:
    """New intervention specs write v2 while the loader still accepts v1 specs."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.func("relu"): tl.zero_ablate()},
    )
    path = tmp_path / "versioned_records.tlspec"

    log.save_intervention(path, level="portable")
    spec_path = path / "spec.json"
    manifest_path = path / "manifest.json"
    spec_json = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert spec_json["format_version"] == "2"
    assert manifest_json["format_version"] == "2"

    spec_json["format_version"] = "1"
    manifest_json["format_version"] = "1"
    spec_path.write_text(json.dumps(spec_json), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    loaded = tl.load_intervention_spec(path)
    assert loaded.metadata["format_version"] == "1"

    spec_json["format_version"] = "3"
    spec_path.write_text(json.dumps(spec_json), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported intervention .tlspec format_version"):
        tl.load_intervention_spec(path)


def test_fire_record_ledger_deduplicates_by_structure_not_timestamp() -> None:
    """Save-time ledger merge ignores timestamp for duplicate fire records."""

    first = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        helper=tl.grad_clamp(0, 0),
        site_label="relu_back_1_1",
        timing="post",
        direction="backward",
        helper_name="grad_clamp",
        timestamp=1.0,
        backward_pass_index=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=0,
        replaced=True,
    )
    duplicate = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        helper=tl.grad_clamp(0, 0),
        site_label="relu_back_1_1",
        timing="post",
        direction="backward",
        helper_name="grad_clamp",
        timestamp=2.0,
        backward_pass_index=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=0,
        replaced=True,
    )
    distinct_tuple = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        helper=tl.grad_clamp(0, 0),
        site_label="relu_back_1_1",
        timing="post",
        direction="backward",
        helper_name="grad_clamp",
        timestamp=1.0,
        backward_pass_index=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=1,
        replaced=True,
    )
    spec = InterventionSpec(records=[first, duplicate, distinct_tuple])

    _sync_spec_records_from_log(spec, SimpleNamespace(layer_list=[], grad_fn_logs={}))

    assert spec.records == [first, distinct_tuple]


@pytest.mark.smoke
def test_portable_rejects_opaque_hook(tmp_path: Path) -> None:
    """Portable saves fail closed for opaque local callables."""

    log = _log()

    def opaque_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Return out unchanged.

        Parameters
        ----------
        out:
            Activation tensor.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Input out.
        """

        del hook
        return out

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
