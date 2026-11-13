"""Phase 10 tests for intervention spec persistence."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _format3_envelope import redigest
from example_models import TinyReluAdd as _ReluModel
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens._ir.container import TupleIndex
from torchlens._ir.selector_eval import selector_from_spec
from torchlens._selection.selectors import output_at
from torchlens.intervention import check_spec_compat
from torchlens.intervention._resolver_registry import (
    function_registry_key_from_callable,
)
from torchlens.intervention._save import (
    _sync_spec_records_from_log,
    resolve_function_registry_key,
    save_intervention,
)
from torchlens.intervention._types import (
    Edit,
    FireRecord,
    FunctionRegistryKey,
    InterventionSpec,
)
from torchlens.intervention.errors import (
    MultiMatchWarning,
    OpaqueCallableError,
    UntrustedCallableError,
)
from torchlens.io import load_intervention_spec

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword

_INTERVENTION_RESOLVER = importlib.import_module("torchlens.intervention._resolver_registry")
#: Stage sticky hooks on the trace without propagating them.
_STAGE = tl.options.InterventionOptions(engine="set_only")


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


class _DeepReluStack(nn.Module):
    """Model exposing more than eight matching ReLU sites."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nine sequential ReLUs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final tensor after nine ReLUs.
        """

        out = x
        for _index in range(9):
            out = torch.relu(out)
        return out


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
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def test_audit_save_load_and_compat(tmp_path: Path) -> None:
    """Audit save writes the tlspec directory and loaded specs compare cleanly."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.do(
        tl.func("relu"),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(engine="set_only"),
    )
    path = tmp_path / "mylog.tlspec"

    log.save_intervention(path, level="audit")

    assert path.is_dir()
    # The 3.0 layout: manifest.json plus intervention.json; a zero_ablate has no
    # tensors, so there is no blobs/ directory (and no 2.x spec.json or tensors/).
    assert sorted(child.name for child in path.iterdir()) == ["intervention.json", "manifest.json"]
    spec = load_intervention_spec(path, trust_custom_callables=True)
    assert tl.load(path) == spec
    compat = check_spec_compat(spec, _log(_ReluModel(), x))
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.has_identical_target_resolution is True


def _saved_fire_records(trace: Any) -> list[Any]:
    """Read forward op edits and backward call fires from their persisted record fields."""

    records = [record for op in trace.ops for record in (op.edits or ())]
    for call in trace.autograd.grad_fn_calls:
        fire = call.fire_record
        if fire is None:
            continue
        values = fire if isinstance(fire, tuple) else (fire,)
        assert all(isinstance(record, FireRecord) for record in values)
        records.extend(values)
    return records


def test_live_forward_records_persist_in_saved_trace(tmp_path: Path) -> None:
    """Live forward hook records survive trace save/load on the ops that fired.

    The 3.0 recipe holds rules, not fires (design section 8); a saved trace
    carries its FireRecords per op, so the round trip goes through the trace.
    """

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(
            replayable=True,
            hooks={tl.func("relu"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "forward_records.tlspec"

    assert _saved_fire_records(log)

    log.save(path, level="portable")
    records = _saved_fire_records(tl.load(path))

    assert records
    assert all(isinstance(record, FireRecord) for record in records)
    assert any(record.signal == "forward" for record in records)
    assert any(record.replaced is True for record in records)


def test_predicate_intervention_spec_round_trip_preserves_targets_and_hooks(
    tmp_path: Path,
) -> None:
    """Predicate interventions save executable targets and hook specs."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "predicate_intervention.tlspec"

    assert log._intervention_spec.records

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path, trust_custom_callables=True)

    assert spec.targets
    assert spec.hook_specs
    assert spec.hook_specs[0].site_target.selector_kind == "label"
    assert spec.hook_specs[0].helper is not None


def test_container_path_fire_records_survive_trace_save(tmp_path: Path) -> None:
    """Tuple-output FireRecord container paths survive a default-level trace save."""

    log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(
            replayable=True,
            hooks={tl.func("chunk"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "chunk_intervention.tlspec"

    log.save(path)

    assert {record.container_path for record in _saved_fire_records(tl.load(path))} == {
        (TupleIndex(0),),
        (TupleIndex(1),),
    }


def test_where_hook_save_manifest_tolerates_nonportable_predicate(tmp_path: Path) -> None:
    """Live ``tl.where`` hook specs are audit-only when the predicate is opaque."""

    log = _log(_ReluModel(), torch.randn(2, 3))
    log._attach_hooks(
        tl.where(lambda ctx: ctx.func_name == "relu", name_hint="relu predicate"),
        tl.zero_ablate(),
        strict=False,
    )
    audit_path = tmp_path / "where_audit.tlspec"

    log.save_intervention(audit_path, level="audit")

    manifest = json.loads((audit_path / "manifest.json").read_text())
    selector = manifest["spec_compat_info"]["target_manifest"][0]["selector"]
    assert manifest["spec_compat_info"]["target_manifest"][0]["resolved_status"] == (
        "unresolved_nonportable"
    )
    assert "__opaque_audit__" in selector["selector_value"]

    for level in ("executable_with_callables", "portable"):
        path = tmp_path / f"where_{level}.tlspec"
        with pytest.raises(OpaqueCallableError, match="Callable selector"):
            log.save_intervention(path, level=level)


def test_target_spec_selector_rehydration_covers_path_and_pattern_selectors() -> None:
    """TargetSpec rehydration supports output_at, input_at, and regex selectors."""

    log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )

    output_spec = output_at(0).to_target_spec()
    output_labels = log[
        selector_from_spec(output_spec.selector_kind, output_spec.selector_value, {})
    ].labels
    assert "chunk_1_2:1" in output_labels
    regex_spec = tl.label_matches("chunk_[12]").to_target_spec()
    regex_labels = log[
        selector_from_spec(regex_spec.selector_kind, regex_spec.selector_value, {})
    ].labels
    assert regex_labels == ("chunk_1_2:1", "chunk_2_3:1")
    assert repr(selector_from_spec("input_at", (0,), {})) == "tl.model_input((0,))"


def test_loaded_output_at_and_regex_hook_specs_execute_on_fresh_trace(
    tmp_path: Path,
) -> None:
    """Loaded path and pattern selector hook specs execute after save/load."""

    x = torch.ones(1, 4)
    output_log = _log(_ChunkModel(), x)
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        output_log._attach_hooks(output_at(1), tl.zero_ablate())
    output_path = tmp_path / "loaded_output_at.tlspec"
    output_log.save_intervention(output_path, level="portable")
    output_spec = load_intervention_spec(output_path)
    output_fresh = _log(_ChunkModel(), x)
    output_fresh._intervention_spec = output_spec
    output_rerun = output_fresh.run(_ChunkModel(), x)

    output_tensor = output_rerun[output_rerun.output_layers[1]].out
    assert torch.count_nonzero(output_tensor) == 0

    regex_log = _log(_ReluModel(), torch.ones(1, 3))
    regex_log._attach_hooks(tl.label_matches("relu"), tl.scale(0.0))
    regex_path = tmp_path / "loaded_regex.tlspec"
    regex_log.save_intervention(regex_path, level="portable")
    regex_spec = load_intervention_spec(regex_path)
    regex_fresh = _log(_ReluModel(), torch.ones(1, 3))
    regex_fresh._intervention_spec = regex_spec
    regex_rerun = regex_fresh.run(_ReluModel(), torch.ones(1, 3))

    assert torch.equal(regex_rerun[regex_rerun.output_layers[0]].out, torch.ones(1, 3))


def test_stored_helper_keyword_outside_the_constructor_refuses_typed(tmp_path: Path) -> None:
    """A saved builtin helper carrying a keyword its constructor does not take refuses typed.

    This is the refusal a spec saved under different helper keywords meets (the 2.16
    fixtures used to reach it through ``force_shape_change`` before the format stamp
    refused them first); it must stay typed, never a raw ``TypeError``.
    """

    from torchlens._errors import InvalidArgumentError

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(
            replayable=True,
            hooks={tl.func("relu"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "foreign_keyword.tlspec"
    log.save_intervention(path, level="portable")
    spec_path = path / "intervention.json"
    spec_json = json.loads(spec_path.read_text(encoding="utf-8"))

    def _plant(value: Any) -> int:
        planted = 0
        if isinstance(value, dict):
            if value.get("name") == "zero_ablate" and isinstance(value.get("kwargs"), dict):
                value["kwargs"]["force_shape_change"] = False
                planted += 1
            planted += sum(_plant(item) for item in value.values())
        elif isinstance(value, list):
            planted += sum(_plant(item) for item in value)
        return planted

    assert _plant(spec_json) >= 1
    spec_path.write_text(json.dumps(spec_json), encoding="utf-8")
    redigest(path, "intervention.json")

    with pytest.raises(InvalidArgumentError) as excinfo:
        load_intervention_spec(path)
    assert excinfo.value.fields["code"] == "intervention_helper_arguments_invalid"
    assert "force_shape_change" in str(excinfo.value)


def test_loaded_import_ref_has_no_load_side_effect_until_execution(tmp_path: Path) -> None:
    """A saved hook callable is a placeholder: loading never imports its module."""

    module_path = tmp_path / "side_effect_mod.py"
    sentinel = tmp_path / "sentinel"
    module_path.write_text(
        "from pathlib import Path\n"
        "SENTINEL = Path(__file__).with_name('sentinel')\n"
        "SENTINEL.write_text('imported')\n"
        "def hook(out, *, ctx):\n"
        "    return out * 0\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import side_effect_mod

        log = _log(_ReluModel(), torch.ones(1, 3))
        log._attach_hooks(tl.func("relu"), side_effect_mod.hook)
        path = tmp_path / "lazy_import.tlspec"
        log.save_intervention(path, level="executable_with_callables")
        sentinel.unlink()
        sys.modules.pop("side_effect_mod", None)

        # 3.0 never saves a user callable (io.1): the rule is a placeholder that
        # names it, and no load, default or trusted, imports or executes the module.
        for trusted in (False, True):
            loaded = load_intervention_spec(path, trust_custom_callables=trusted)
            assert not sentinel.exists()
            assert "side_effect_mod" not in sys.modules
            assert loaded.metadata["executable"] is False
            [placeholder] = loaded.metadata["recipe_placeholders"]
            assert placeholder["qualname"] == "side_effect_mod.hook"
            edit = loaded.hook_specs[0].helper
            assert edit is not None
            assert edit.portability == "opaque_audit"
            assert edit.factory is None
            assert dict(edit.metadata)["qualname"] == "side_effect_mod.hook"
    finally:
        sys.modules.pop("side_effect_mod", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_loaded_helper_import_ref_preserves_identity_and_executes_lazily(
    tmp_path: Path,
) -> None:
    """A saved import-ref helper is a placeholder: loading never imports its module."""

    module_path = tmp_path / "side_effect_helper_mod.py"
    sentinel = tmp_path / "helper_sentinel"
    module_path.write_text(
        "from pathlib import Path\n"
        "SENTINEL = Path(__file__).with_name('helper_sentinel')\n"
        "SENTINEL.write_text('imported')\n"
        "def make_hook():\n"
        "    def hook(out, *, ctx):\n"
        "        return out * 0\n"
        "    return hook\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import side_effect_helper_mod

        helper = Edit(
            name="side_effect_helper",
            portability="import_ref",
            factory=side_effect_helper_mod.make_hook,
        )
        log = _log(_ReluModel(), torch.ones(1, 3))
        log._attach_hooks(tl.func("relu"), helper)
        path = tmp_path / "lazy_helper_import.tlspec"
        log.save_intervention(path, level="executable_with_callables")
        sentinel.unlink()
        sys.modules.pop("side_effect_helper_mod", None)

        # 3.0 never saves a user callable (io.1): the rule is a placeholder that
        # names it, and no load, default or trusted, imports or executes the module.
        for trusted in (False, True):
            loaded = load_intervention_spec(path, trust_custom_callables=trusted)
            assert not sentinel.exists()
            assert "side_effect_helper_mod" not in sys.modules
            assert loaded.metadata["executable"] is False
            [placeholder] = loaded.metadata["recipe_placeholders"]
            assert placeholder["qualname"] == "side_effect_helper_mod.make_hook"
            edit = loaded.hook_specs[0].helper
            assert edit is not None
            assert edit.portability == "opaque_audit"
            assert edit.factory is None
            assert dict(edit.metadata)["qualname"] == "side_effect_helper_mod.make_hook"
    finally:
        sys.modules.pop("side_effect_helper_mod", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_live_backward_records_persist_in_saved_trace(tmp_path: Path) -> None:
    """Backward call fires survive with the exact live call and transaction metadata."""

    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    log._attach_hooks(tl.grad_fn(name="relu"), tl.clamp(low=0, high=0))
    log.backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    path = tmp_path / "backward_records.tlspec"
    before = [record for record in _saved_fire_records(log) if record.signal == "backward"]
    assert before

    log.save(path, level="portable")
    loaded = tl.load(path)

    backward_records = [
        record for record in _saved_fire_records(loaded) if record.signal == "backward"
    ]
    assert backward_records
    assert backward_records[0].backward_pass_number == 1
    assert backward_records[0].call_index == 1
    fields = (
        "signal",
        "target_label",
        "call_label",
        "backward_pass_number",
        "call_index",
        "grad_kind",
        "tuple_index",
        "edit_name",
        "timestamp",
    )
    assert [tuple(getattr(record, name) for name in fields) for record in backward_records] == [
        tuple(getattr(record, name) for name in fields) for record in before
    ]


def test_loaded_backward_grad_fn_spec_executes_after_round_trip(tmp_path: Path) -> None:
    """Structured grad_fn selector payloads execute after save/load."""

    x = torch.ones(1, 3, requires_grad=True)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    log._attach_hooks(tl.grad_fn(name="relu"), tl.scale(2.0))
    log.backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    path = tmp_path / "backward_execute.tlspec"

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path, trust_custom_callables=True)

    assert isinstance(spec.hook_specs[0].site_target.selector_value, dict)
    x_fresh = torch.ones(1, 3, requires_grad=True)
    fresh = tl.trace(
        _ReluModel(),
        x_fresh,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    fresh._intervention_spec = spec
    fresh.backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

    assert torch.equal(fresh[fresh.input_layers[0]].grad, torch.full_like(x_fresh, 2.0))


def test_backward_hook_spec_saves_before_first_backward_and_executes(
    tmp_path: Path,
) -> None:
    """Sticky backward recipes save unresolved and resolve at execution time."""

    x = torch.ones(1, 3, requires_grad=True)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    log._attach_hooks(tl.grad_fn(name="relu"), tl.scale(2.0))
    path = tmp_path / "backward_before_pass.tlspec"

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path, trust_custom_callables=True)

    assert spec.metadata["target_manifest"][0]["resolved_status"] == "unresolved_backward"
    x_fresh = torch.ones(1, 3, requires_grad=True)
    fresh = tl.trace(
        _ReluModel(),
        x_fresh,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    fresh._intervention_spec = spec
    fresh.backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

    assert torch.equal(fresh[fresh.input_layers[0]].grad, torch.full_like(x_fresh, 2.0))


def test_intervention_tlspec_v2_writes_and_v1_still_loads(tmp_path: Path) -> None:
    """New intervention specs write the 3.0 recipe; an unknown recipe format refuses typed."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(
            replayable=True,
            hooks={tl.func("relu"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "versioned_records.tlspec"

    log.save_intervention(path, level="portable")
    recipe_path = path / "intervention.json"
    manifest_path = path / "manifest.json"
    recipe_json = json.loads(recipe_path.read_text(encoding="utf-8"))
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert recipe_json["format_version"] == 3
    assert manifest_json["format_version"] == 3
    assert not (path / "spec.json").exists()

    # A recipe stamped with a format this TorchLens does not read refuses typed.
    recipe_json["format_version"] = 5
    recipe_path.write_text(json.dumps(recipe_json), encoding="utf-8")
    redigest(path, "intervention.json")
    with pytest.raises(ArtifactError) as excinfo:
        load_intervention_spec(path)
    assert excinfo.value.fields["code"] == "bundle_load_failed"
    assert "recipe format_version 5" in str(excinfo.value)


def test_fire_record_ledger_deduplicates_by_structure_not_timestamp() -> None:
    """Save-time ledger merge ignores timestamp for duplicate fire records."""

    first = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        edit=tl.clamp(low=0, high=0),
        site_label="relu_back_1_1",
        timing="post",
        signal="backward",
        edit_name="clamp",
        timestamp=1.0,
        backward_pass_number=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=0,
        replaced=True,
    )
    duplicate = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        edit=tl.clamp(low=0, high=0),
        site_label="relu_back_1_1",
        timing="post",
        signal="backward",
        edit_name="clamp",
        timestamp=2.0,
        backward_pass_number=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=0,
        replaced=True,
    )
    distinct_tuple = FireRecord(
        target_label="relu_back_1_1",
        call_label="relu_back_1_1:1",
        engine="live",
        edit=tl.clamp(low=0, high=0),
        site_label="relu_back_1_1",
        timing="post",
        signal="backward",
        edit_name="clamp",
        timestamp=1.0,
        backward_pass_number=1,
        call_index=1,
        grad_kind="grad_input",
        tuple_index=1,
        replaced=True,
    )
    spec = InterventionSpec(records=[first, duplicate, distinct_tuple])

    _sync_spec_records_from_log(spec, SimpleNamespace(_layer_list=[], _grad_fn_logs={}))

    assert spec.records == [first, distinct_tuple]


def test_portable_rejects_opaque_hook(tmp_path: Path) -> None:
    """Portable saves fail closed for opaque local callables."""

    log = _log()

    def opaque_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
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

        del ctx
        return out

    log._attach_hooks({tl.func("relu"): opaque_hook})
    with pytest.raises(OpaqueCallableError) as excinfo:
        log.save_intervention(tmp_path / "opaque.tlspec", level="portable")
    assert excinfo.value.fields["code"] == "intervention_spec_not_portable"
    assert not (tmp_path / "opaque.tlspec").exists()


def test_loaded_spec_tolerates_custom_key_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis-mode loading tolerates custom keys without importing them."""

    path = tmp_path / "custom_key.tlspec"
    _log().save_intervention(path, level="audit")
    spec_path = path / "intervention.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["function_registry_keys"] = [
        {
            "layer_label": "malicious_1_1",
            "key": {
                "namespace": "custom",
                "qualname": "neg",
                "dispatch_kind": "function",
                "import_path": "operator:neg",
            },
        }
    ]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    redigest(path, "intervention.json")
    # The model-free tlspec loader imports its own torchlens loader module on
    # demand; only a module outside the package is an untrusted import.
    real_import_module = importlib.import_module

    def fail_import(module_name: str) -> object:
        """Fail if loading attempts to import an untrusted custom module.

        Parameters
        ----------
        module_name:
            Module name passed to the import system.

        Returns
        -------
        object
            The torchlens module itself; any other module name raises.
        """

        if module_name.startswith("torchlens."):
            return real_import_module(module_name)
        raise AssertionError(f"unexpected import of {module_name}")

    with monkeypatch.context() as import_patch:
        import_patch.setattr(_INTERVENTION_RESOLVER.importlib, "import_module", fail_import)
        assert load_intervention_spec(path)

    assert load_intervention_spec(path, allowed_custom_callable_modules={"operator"})


def test_loaded_custom_key_is_denied_at_execution_until_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loaded foreign callables remain gated until execution explicitly trusts them."""

    module_name = "torchlens_loaded_execution_gate"
    marker = tmp_path / "imported_at_execution"
    (tmp_path / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "def payload(value):\n    return -value\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # The temp module has a fixed name and an import-time side effect (writes the
    # marker). Ensure it is NOT already cached so this run imports it fresh, and let
    # monkeypatch drop it at teardown (restoring the absent state) so the test is
    # order-independent -- otherwise a cached module makes the trusted import a no-op.
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    path = tmp_path / "execution_gate.tlspec"
    _log().save_intervention(path, level="audit")
    spec_path = path / "intervention.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["function_registry_keys"] = [
        {
            "layer_label": "foreign_1_1",
            "key": {
                "namespace": "custom",
                "qualname": "payload",
                "dispatch_kind": "function",
                "import_path": f"{module_name}:payload",
            },
        }
    ]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    redigest(path, "intervention.json")

    loaded = load_intervention_spec(path)
    loaded_key = FunctionRegistryKey(**loaded.metadata["function_registry_keys"][0]["key"])
    assert not marker.exists()

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(loaded_key)
    assert not marker.exists()

    resolved = resolve_function_registry_key(loaded_key, trust_custom_callables=True)
    assert marker.read_text(encoding="utf-8") == "executed"
    assert resolved(3) == -3


def test_custom_function_key_trust_gate_and_allowlist() -> None:
    """Trusted custom keys resolve, while a supplied allowlist stays restrictive."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname="neg",
        dispatch_kind="function",
        import_path="operator:neg",
    )

    assert (
        resolve_function_registry_key(key, trust_custom_callables=True)
        is __import__("operator").neg
    )
    assert (
        resolve_function_registry_key(key, allowed_custom_callable_modules={"operator"})
        is __import__("operator").neg
    )
    with pytest.raises(UntrustedCallableError, match="not in allowed_custom_callable_modules"):
        resolve_function_registry_key(
            key,
            trust_custom_callables=True,
            allowed_custom_callable_modules={"torch"},
        )


@pytest.mark.parametrize(
    ("namespace", "qualname"),
    [
        ("torch", "relu"),
        ("torch.Tensor", "relu"),
        ("torch.nn.functional", "relu"),
        ("operator", "neg"),
    ],
)
def test_trusted_function_namespaces_resolve_unchanged(namespace: str, qualname: str) -> None:
    """Fixed trusted registry roots remain available without a trust opt-in."""

    key = FunctionRegistryKey(namespace=namespace, qualname=qualname, dispatch_kind="function")

    assert callable(resolve_function_registry_key(key))


@pytest.mark.smoke
def test_internal_torch_builtin_key_preserves_replay_identity() -> None:
    """Internal torch builtins resolve without collapsing to public wrappers."""

    builtin = torch._C._VariableFunctionsClass.tensordot
    key = function_registry_key_from_callable(builtin)

    assert key.namespace == "custom"
    assert key.import_path == "torch._C._VariableFunctionsClass:tensordot"
    assert resolve_function_registry_key(key) is builtin


# --- Round-2 RCE regression: torchlens.* auto-trust bypass -------------------
# A custom key whose import PATH starts with "torchlens" but whose resolved
# callable really lives in os/importlib/builtins/subprocess (reached by walking
# attributes off a torchlens module that did ``import os`` etc. at top level)
# must be DENIED. Trust is decided by the resolved callable's real ``__module__``,
# never by the import-path string prefix.

_R2_TORCHLENS_PATH_TO_FOREIGN_CALLABLE = [
    ("torchlens._io.tlspec:os.system", "system"),
    ("torchlens.report._environment:importlib.import_module", "import_module"),
    ("torchlens._selection.selectors:builtins.eval", "eval"),
    ("torchlens.report._environment:subprocess.getoutput", "getoutput"),
]


@pytest.mark.parametrize(("import_path", "qualname"), _R2_TORCHLENS_PATH_TO_FOREIGN_CALLABLE)
def test_r2_torchlens_prefixed_foreign_callable_denied_by_default(
    import_path: str, qualname: str
) -> None:
    """os/importlib/builtins/subprocess reached via a torchlens path stay denied."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname=qualname,
        dispatch_kind="function",
        import_path=import_path,
    )

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


def test_r2_genuine_torchlens_custom_callable_still_resolves() -> None:
    """A real torchlens-owned custom callable stays auto-trusted after the fix."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname="clamp",
        dispatch_kind="function",
        import_path="torchlens.intervention._helpers:clamp",
    )

    resolved = resolve_function_registry_key(key)
    assert callable(resolved)
    assert str(getattr(resolved, "__module__", "")).startswith("torchlens")


def test_r2_load_tolerates_torchlens_prefixed_foreign_key_no_foreign_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis-only load tolerates the malicious key, importing no foreign module.

    The foreign-deny split is preserved: load tolerates the well-formed key (does
    not fail closed), never imports a NON-torchlens module, and never executes
    os.system. Inspecting the resolved identity only touches the already-installed
    torchlens module the path names.
    """

    import importlib as _importlib
    import os

    path = tmp_path / "r2_foreign.tlspec"
    _log().save_intervention(path, level="audit")
    spec_path = path / "intervention.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["function_registry_keys"] = [
        {
            "layer_label": "malicious_1_1",
            "key": {
                "namespace": "custom",
                "qualname": "system",
                "dispatch_kind": "function",
                "import_path": "torchlens._io.tlspec:os.system",
            },
        }
    ]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    redigest(path, "intervention.json")

    calls: list[str] = []
    monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd) or 0)

    real_import_module = _importlib.import_module

    def guarded_import(module_name: str) -> object:
        """Allow torchlens imports; fail on any foreign (non-torchlens) import.

        Parameters
        ----------
        module_name:
            Module name passed to the import system.

        Returns
        -------
        object
            The imported torchlens module.
        """

        assert module_name == "torchlens" or module_name.startswith("torchlens."), (
            f"unexpected foreign import of {module_name}"
        )
        return real_import_module(module_name)

    monkeypatch.setattr(_INTERVENTION_RESOLVER.importlib, "import_module", guarded_import)
    assert load_intervention_spec(path)
    assert calls == []


def test_target_manifest_mismatch_returns_fail(tmp_path: Path) -> None:
    """Selectors resolving to nothing on a new log produce FAIL compatibility."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.do(
        tl.func("relu"),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(engine="set_only"),
    )
    path = tmp_path / "target.tlspec"
    log.save_intervention(path, level="audit")

    spec = load_intervention_spec(path)
    compat = check_spec_compat(spec, _log(_TanhModel(), x))

    assert compat.outcome == "FAIL"
    assert compat.diff.missing_labels


def test_save_intervention_allows_more_than_eight_matching_sites(tmp_path: Path) -> None:
    """Persistence reuses the validated trace fanout bound for dense selectors."""

    x = torch.randn(2, 3)
    log = _log(_DeepReluStack(), x)
    with pytest.warns(MultiMatchWarning, match="fan out"):
        log._attach_hooks(tl.func("relu"), tl.zero_ablate())
    path = tmp_path / "deep_relu.tlspec"

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path)

    assert spec.metadata["target_manifest"][0]["resolved_status"] == "resolved"
    assert len(spec.metadata["target_manifest"][0]["resolved_labels"]) == 9


def test_check_spec_compat_allows_saved_selectors_with_more_than_eight_sites(
    tmp_path: Path,
) -> None:
    """Compatibility checks do not reinstate the resolver default fanout ceiling."""

    x = torch.randn(2, 3)
    log = _log(_DeepReluStack(), x)
    with pytest.warns(MultiMatchWarning, match="fan out"):
        log._attach_hooks(tl.func("relu"), tl.zero_ablate())
    path = tmp_path / "deep_relu_compat.tlspec"

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path)
    with pytest.warns(MultiMatchWarning, match="fan out"):
        compat = check_spec_compat(spec, _log(_DeepReluStack(), x))

    assert compat.outcome == "EXACT"


def test_atomic_save_cleans_up_after_tensor_write_failure(tmp_path: Path) -> None:
    """A tensor-sidecar write exception leaves no final partial tlspec dir."""

    log = _log()
    log.do(
        tl.func("relu"),
        torch.zeros(2, 3),
        intervention=tl.options.InterventionOptions(engine="set_only"),
    )
    target = tmp_path / "crash.tlspec"

    def crashing_writer(**kwargs: Any) -> Any:
        """Simulate a crash after the recipe's tensor chunks are written.

        Parameters
        ----------
        **kwargs:
            Chunk hook keyword arguments (``tmp_path`` and ``chunks``).

        Returns
        -------
        Any
            Never returned.
        """

        assert kwargs["chunks"]
        raise RuntimeError("simulated tensor crash")

    with pytest.raises(RuntimeError, match="simulated tensor crash"):
        save_intervention(
            log,
            target,
            level="audit",
            _write_tensor_blob_fn=crashing_writer,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(f"{target.name}.tmp.*"))
