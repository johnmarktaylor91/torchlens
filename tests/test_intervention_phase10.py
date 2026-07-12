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
from torchlens.io import load_intervention_spec
from torchlens.ir.container import TupleIndex
from torchlens.intervention.errors import (
    MultiMatchWarning,
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
    UntrustedCallableError,
)
from torchlens.intervention.resolver import _selector_from_spec
from torchlens.intervention.save import _write_tlspec_tensor_blob
from torchlens.intervention.save import _sync_spec_records_from_log
from torchlens.intervention.save import resolve_function_registry_key, save_intervention
from torchlens.intervention.types import (
    FireRecord,
    FunctionRegistryKey,
    HelperSpec,
    InterventionSpec,
)
from torchlens.validation import check_spec_compat


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
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


@pytest.mark.smoke
def test_audit_save_load_and_compat(tmp_path: Path) -> None:
    """Audit save writes the tlspec directory and loaded specs compare cleanly."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    path = tmp_path / "mylog.tlspec"

    log.save_intervention(path, level="audit")

    assert path.is_dir()
    assert (path / "spec.json").exists()
    assert (path / "manifest.json").exists()
    assert (path / "README.md").exists()
    assert (path / "tensors").is_dir()
    spec = load_intervention_spec(path, trust_custom_callables=True)
    assert tl.load(path) == spec
    compat = check_spec_compat(spec, _log(_ReluModel(), x))
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.targets_resolve_identically is True


@pytest.mark.smoke
def test_live_forward_records_persist_in_saved_intervention_spec(tmp_path: Path) -> None:
    """Live forward hook records survive intervention spec save/load."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            hooks={tl.func("relu"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "forward_records.tlspec"

    assert log._intervention_spec.records

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path, trust_custom_callables=True)

    assert spec.records
    assert all(isinstance(record, FireRecord) for record in spec.records)
    assert any(record.direction == "forward" for record in spec.records)
    assert any(record.replaced is True for record in spec.records)


@pytest.mark.smoke
def test_predicate_intervention_spec_round_trip_preserves_targets_and_hooks(
    tmp_path: Path,
) -> None:
    """Predicate interventions save executable targets and hook specs."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
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


def test_container_path_fire_records_save_at_default_level(tmp_path: Path) -> None:
    """Tuple-output FireRecord container paths save and load at the default level."""

    log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            hooks={tl.func("chunk"): tl.zero_ablate()},
        ),
    )
    path = tmp_path / "chunk_intervention.tlspec"

    log.save_intervention(path)
    spec = load_intervention_spec(path)

    assert {record.container_path for record in spec.records} == {
        (TupleIndex(0),),
        (TupleIndex(1),),
    }


def test_where_hook_save_manifest_tolerates_nonportable_predicate(tmp_path: Path) -> None:
    """Live ``tl.where`` hook specs are audit-only when the predicate is opaque."""

    log = _log(_ReluModel(), torch.randn(2, 3))
    log.attach_hooks(
        tl.where(lambda ctx: ctx.func_name == "relu", name_hint="relu predicate"),
        tl.zero_ablate(),
        strict=False,
        confirm_mutation=True,
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
        with pytest.raises(OpaqueCallableInExecutableSaveError, match="Callable selector"):
            log.save_intervention(path, level=level)


def test_target_spec_selector_rehydration_covers_path_and_pattern_selectors() -> None:
    """TargetSpec rehydration supports output_at, input_at, and regex selectors."""

    log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )

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
    spec = load_intervention_spec(path)
    fresh = _log(_ReluModel(), x)
    fresh._intervention_spec = spec
    rerun = fresh.run(_ReluModel(), x)

    assert torch.equal(rerun[rerun.output_layers[0]].out, torch.ones_like(x))


def test_loaded_output_at_and_regex_hook_specs_execute_on_fresh_trace(
    tmp_path: Path,
) -> None:
    """Loaded path and pattern selector hook specs execute after save/load."""

    x = torch.ones(1, 4)
    output_log = _log(_ChunkModel(), x)
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        output_log.attach_hooks(tl.output_at(1), tl.zero_ablate(), confirm_mutation=True)
    output_path = tmp_path / "loaded_output_at.tlspec"
    output_log.save_intervention(output_path, level="portable")
    output_spec = load_intervention_spec(output_path)
    output_fresh = _log(_ChunkModel(), x)
    output_fresh._intervention_spec = output_spec
    output_rerun = output_fresh.run(_ChunkModel(), x)

    output_tensor = output_rerun[output_rerun.output_layers[1]].out
    assert torch.count_nonzero(output_tensor) == 0

    regex_log = _log(_ReluModel(), torch.ones(1, 3))
    regex_log.attach_hooks(tl.regex("relu"), tl.scale(0.0), confirm_mutation=True)
    regex_path = tmp_path / "loaded_regex.tlspec"
    regex_log.save_intervention(regex_path, level="portable")
    regex_spec = load_intervention_spec(regex_path)
    regex_fresh = _log(_ReluModel(), torch.ones(1, 3))
    regex_fresh._intervention_spec = regex_spec
    regex_rerun = regex_fresh.run(_ReluModel(), torch.ones(1, 3))

    assert torch.equal(regex_rerun[regex_rerun.output_layers[0]].out, torch.ones(1, 3))


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

        spec = load_intervention_spec(path)

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


def test_loaded_helper_import_ref_preserves_identity_and_executes_lazily(
    tmp_path: Path,
) -> None:
    """Helper import refs load lazily and remain executable helper specs."""

    module_path = tmp_path / "side_effect_helper_mod.py"
    sentinel = tmp_path / "helper_sentinel"
    module_path.write_text(
        "from pathlib import Path\n"
        "SENTINEL = Path(__file__).with_name('helper_sentinel')\n"
        "SENTINEL.write_text('imported')\n"
        "def make_hook():\n"
        "    def hook(out, *, hook):\n"
        "        return out * 0\n"
        "    return hook\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import side_effect_helper_mod

        helper = HelperSpec(
            helper_name="side_effect_helper",
            portability="import_ref",
            factory=side_effect_helper_mod.make_hook,
        )
        log = _log(_ReluModel(), torch.ones(1, 3))
        log.attach_hooks(tl.func("relu"), helper, confirm_mutation=True)
        path = tmp_path / "lazy_helper_import.tlspec"
        log.save_intervention(path, level="executable_with_callables")
        sentinel.unlink()
        sys.modules.pop("side_effect_helper_mod", None)

        spec = load_intervention_spec(path)

        assert not sentinel.exists()
        assert spec.hook_specs[0].helper is not None
        assert spec.hook_specs[0].helper.name == "side_effect_helper"
        fresh = _log(_ReluModel(), torch.ones(1, 3))
        fresh._intervention_spec = spec
        rerun = fresh.run(_ReluModel(), torch.ones(1, 3))
        assert sentinel.exists()
        assert torch.equal(rerun[rerun.output_layers[0]].out, torch.ones(1, 3))
    finally:
        sys.modules.pop("side_effect_helper_mod", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


@pytest.mark.smoke
def test_live_backward_records_persist_in_saved_intervention_spec(tmp_path: Path) -> None:
    """Live backward hook records survive intervention spec save/load."""

    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_clamp(0, 0), confirm_mutation=True)
    log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    path = tmp_path / "backward_records.tlspec"

    log.save_intervention(path, level="portable")
    spec = load_intervention_spec(path, trust_custom_callables=True)

    backward_records = [record for record in spec.records if record.direction == "backward"]
    assert backward_records
    assert backward_records[0].backward_pass_index == 1
    assert backward_records[0].call_index == 1


def test_loaded_backward_grad_fn_spec_executes_after_round_trip(tmp_path: Path) -> None:
    """Structured grad_fn selector payloads execute after save/load."""

    x = torch.ones(1, 3, requires_grad=True)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_scale(2.0), confirm_mutation=True)
    log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
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
    fresh.log_backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

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
    log.attach_hooks(tl.grad_fn(type="relu"), tl.grad_scale(2.0), confirm_mutation=True)
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
    fresh.log_backward(fresh[fresh.output_layers[0]].out.sum(), retain_graph=True)

    assert torch.equal(fresh[fresh.input_layers[0]].grad, torch.full_like(x_fresh, 2.0))


def test_intervention_tlspec_v2_writes_and_v1_still_loads(tmp_path: Path) -> None:
    """New intervention specs write v2 while the loader still accepts v1 specs."""

    log = tl.trace(
        _ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            hooks={tl.func("relu"): tl.zero_ablate()},
        ),
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

    loaded = load_intervention_spec(path)
    assert loaded.metadata["format_version"] == "1"

    spec_json["format_version"] = "3"
    spec_path.write_text(json.dumps(spec_json), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported intervention .tlspec format_version"):
        load_intervention_spec(path)


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

    log.attach_hooks({tl.func("relu"): opaque_hook}, confirm_mutation=True)
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
def test_custom_function_key_is_refused_without_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default resolution refuses bundle custom keys before importing modules."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname="neg",
        dispatch_kind="function",
        import_path="operator:neg",
    )

    def fail_import(module_name: str) -> object:
        """Fail if an untrusted custom module import is attempted.

        Parameters
        ----------
        module_name:
            Module name passed to the import system.

        Returns
        -------
        object
            This function always raises.
        """

        raise AssertionError(f"unexpected import of {module_name}")

    monkeypatch.setattr("torchlens.intervention.resolver.importlib.import_module", fail_import)

    with pytest.raises(UntrustedCallableError, match="arbitrary code"):
        resolve_function_registry_key(key)


@pytest.mark.smoke
def test_loaded_spec_tolerates_custom_key_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis-mode loading tolerates custom keys without importing them."""

    path = tmp_path / "custom_key.tlspec"
    _log().save_intervention(path, level="audit")
    spec_path = path / "spec.json"
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

    def fail_import(module_name: str) -> object:
        """Fail if loading attempts to import an untrusted custom module.

        Parameters
        ----------
        module_name:
            Module name passed to the import system.

        Returns
        -------
        object
            This function always raises.
        """

        raise AssertionError(f"unexpected import of {module_name}")

    with monkeypatch.context() as import_patch:
        import_patch.setattr("torchlens.intervention.resolver.importlib.import_module", fail_import)
        assert load_intervention_spec(path)

    assert load_intervention_spec(path, allowed_custom_callable_modules={"operator"})


@pytest.mark.smoke
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
    spec_path = path / "spec.json"
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

    loaded = load_intervention_spec(path)
    loaded_key = FunctionRegistryKey(**loaded.metadata["function_registry_keys"][0]["key"])
    assert not marker.exists()

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(loaded_key)
    assert not marker.exists()

    resolved = resolve_function_registry_key(loaded_key, trust_custom_callables=True)
    assert marker.read_text(encoding="utf-8") == "executed"
    assert resolved(3) == -3


@pytest.mark.smoke
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


@pytest.mark.smoke
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
def test_red_team_custom_module_side_effect_is_not_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malicious custom import path cannot execute its module side effect by default."""

    module_name = "torchlens_custom_import_side_effect"
    marker = tmp_path / "imported"
    (tmp_path / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "def payload():\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    key = FunctionRegistryKey(
        namespace="custom",
        qualname="payload",
        dispatch_kind="function",
        import_path=f"{module_name}:payload",
    )

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)

    assert not marker.exists()


@pytest.mark.smoke
def test_target_manifest_mismatch_returns_fail(tmp_path: Path) -> None:
    """Selectors resolving to nothing on a new log produce FAIL compatibility."""

    x = torch.randn(2, 3)
    log = _log(_ReluModel(), x)
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    path = tmp_path / "target.tlspec"
    log.save_intervention(path, level="audit")

    spec = load_intervention_spec(path)
    compat = check_spec_compat(spec, _log(_TanhModel(), x))

    assert compat.outcome == "FAIL"
    assert compat.diff.missing_labels


@pytest.mark.smoke
def test_atomic_save_cleans_up_after_tensor_write_failure(tmp_path: Path) -> None:
    """A tensor-sidecar write exception leaves no final partial tlspec dir."""

    log = _log()
    log.set(tl.func("relu"), torch.zeros(2, 3), confirm_mutation=True)
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
