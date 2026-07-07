"""Phase 10 tests for intervention spec persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import TupleIndex
from torchlens.intervention.errors import (
    MultiMatchWarning,
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
)
from torchlens.intervention.resolver import _selector_from_spec
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


class _ChunkModel(nn.Module):
    """Small model with a tuple-output operation."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return two tensor chunks.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Two chunks of the relu output.
        """

        return torch.chunk(torch.relu(x), 2, dim=1)


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

    assert log._intervention_spec.records

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    assert spec.records
    assert all(isinstance(record, FireRecord) for record in spec.records)
    assert any(record.direction == "forward" for record in spec.records)


@pytest.mark.smoke
def test_predicate_intervention_spec_round_trip_preserves_targets_and_hooks(
    tmp_path: Path,
) -> None:
    """Predicate interventions save executable targets and hook specs."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "predicate_intervention.tlspec"

    assert log._intervention_spec.records

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    assert spec.targets
    assert spec.hook_specs
    assert spec.hook_specs[0].site_target.selector_kind == "label"
    assert spec.hook_specs[0].helper is not None


def test_container_path_fire_records_save_at_default_level(tmp_path: Path) -> None:
    """Tuple-output FireRecord container paths save and load at the default level."""

    log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        intervention_ready=True,
        hooks={tl.func("chunk"): tl.zero_ablate()},
    )
    path = tmp_path / "chunk_intervention.tlspec"

    log.save_intervention(path)
    spec = tl.load_intervention_spec(path)

    assert {record.container_path for record in spec.records} == {
        (TupleIndex(0),),
        (TupleIndex(1),),
    }


def test_where_hook_save_manifest_tolerates_nonportable_predicate(tmp_path: Path) -> None:
    """Live ``tl.where`` hook specs save at every supported intervention level."""

    for level in ("audit", "executable_with_callables", "portable"):
        log = _log(_ReluModel(), torch.randn(2, 3))
        log.attach_hooks(
            tl.where(lambda ctx: ctx.func_name == "relu", name_hint="relu predicate"),
            tl.zero_ablate(),
            strict=False,
            confirm_mutation=True,
        )
        path = tmp_path / f"where_{level}.tlspec"

        log.save_intervention(path, level=level)

        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["spec_compat_info"]["target_manifest"][0]["resolved_status"] == (
            "unresolved_nonportable"
        )


def test_target_spec_selector_rehydration_covers_path_and_pattern_selectors() -> None:
    """TargetSpec rehydration supports output_at, input_at, and regex selectors."""

    log = tl.trace(_ChunkModel(), torch.randn(2, 4), intervention_ready=True)

    with pytest.warns(MultiMatchWarning):
        output_labels = log.resolve_sites(tl.output_at(0).to_target_spec()).labels()
    assert "chunk_1_2" in output_labels
    with pytest.warns(MultiMatchWarning):
        regex_labels = log.resolve_sites(tl.regex("chunk_[12]").to_target_spec()).labels()
    assert regex_labels == ("chunk_1_2", "chunk_2_3")
    assert repr(_selector_from_spec("input_at", (0,), {})) == "tl.input_at((0,))"


def test_loaded_forward_hook_spec_executes_on_fresh_trace(tmp_path: Path) -> None:
    """A loaded forward hook spec re-executes the intended hook."""

    x = torch.ones(1, 3)
    log = _log(_ReluModel(), x)
    log.attach_hooks(tl.func("relu"), tl.scale(0.0), confirm_mutation=True)
    path = tmp_path / "forward_execute.tlspec"

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)
    fresh = _log(_ReluModel(), x)
    fresh._intervention_spec = spec
    rerun = fresh.run(_ReluModel(), x)

    assert torch.equal(rerun[rerun.output_layers[0]].out, torch.ones_like(x))


def test_loaded_import_ref_has_no_load_side_effect_until_execution(tmp_path: Path) -> None:
    """Executable import refs import lazily after load and only when executed."""

    module_path = tmp_path / "side_effect_mod.py"
    sentinel = tmp_path / "sentinel"
    module_path.write_text(
        "from pathlib import Path\n"
        "SENTINEL = Path(__file__).with_name('sentinel')\n"
        "SENTINEL.write_text('imported')\n"
        "def hook(out, *, hook):\n"
        "    return out * 0\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import side_effect_mod

        log = _log(_ReluModel(), torch.ones(1, 3))
        log.attach_hooks(tl.func("relu"), side_effect_mod.hook, confirm_mutation=True)
        path = tmp_path / "lazy_import.tlspec"
        log.save_intervention(path, level="executable_with_callables")
        sentinel.unlink()
        sys.modules.pop("side_effect_mod", None)

        spec = tl.load_intervention_spec(path)

        assert not sentinel.exists()
        fresh = _log(_ReluModel(), torch.ones(1, 3))
        fresh._intervention_spec = spec
        rerun = fresh.run(_ReluModel(), torch.ones(1, 3))
        assert sentinel.exists()
        assert torch.equal(rerun[rerun.output_layers[0]].out, torch.ones(1, 3))
    finally:
        sys.modules.pop("side_effect_mod", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


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


def test_loaded_backward_grad_fn_spec_executes_after_round_trip(tmp_path: Path) -> None:
    """Structured grad_fn selector payloads execute after save/load."""

    x = torch.ones(1, 3, requires_grad=True)
    log = tl.trace(_ReluModel(), x, save_grads="all", backward_ready=True)
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_scale(2.0), confirm_mutation=True)
    log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    path = tmp_path / "backward_execute.tlspec"

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    assert isinstance(spec.hook_specs[0].site_target.selector_value, dict)
    x_fresh = torch.ones(1, 3, requires_grad=True)
    fresh = tl.trace(_ReluModel(), x_fresh, save_grads="all", backward_ready=True)
    fresh._intervention_spec = spec
    fresh.log_backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

    assert torch.equal(fresh[fresh.input_layers[0]].grad, torch.full_like(x_fresh, 2.0))


def test_backward_hook_spec_saves_before_first_backward_and_executes(
    tmp_path: Path,
) -> None:
    """Sticky backward recipes save unresolved and resolve at execution time."""

    x = torch.ones(1, 3, requires_grad=True)
    log = tl.trace(_ReluModel(), x, save_grads="all", backward_ready=True)
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_scale(2.0), confirm_mutation=True)
    path = tmp_path / "backward_before_pass.tlspec"

    log.save_intervention(path, level="portable")
    spec = tl.load_intervention_spec(path)

    assert spec.metadata["target_manifest"][0]["resolved_status"] == "unresolved_backward"
    x_fresh = torch.ones(1, 3, requires_grad=True)
    fresh = tl.trace(_ReluModel(), x_fresh, save_grads="all", backward_ready=True)
    fresh._intervention_spec = spec
    fresh.log_backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

    assert torch.equal(fresh[fresh.input_layers[0]].grad, torch.full_like(x_fresh, 2.0))


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
