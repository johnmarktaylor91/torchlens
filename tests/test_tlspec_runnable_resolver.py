"""Stage 3 safe-load and sparse callable reattachment tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._io import runnable_load
from torchlens._io.runnable import (
    build_sparse_run_descriptor,
    preflight_sparse_run_descriptor,
)
from torchlens.errors import ReattachError
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    ReadinessStatus,
    ResolverStatus,
    RunnableErrorCode,
    SparseRunDescriptor,
)
from torchlens.utils._torch_compat import resolve_runnable_torch_alias


class ResolverModel(nn.Module):
    """Small graph covering exact, private-to-public, and decorated resolution."""

    def __init__(self) -> None:
        """Initialize one stateful linear call."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run stock exact and private-backed torch callables."""

        return torch.relu(self.linear(value))


def _descriptor() -> SparseRunDescriptor:
    """Capture and return one passed sparse descriptor.

    Returns
    -------
    SparseRunDescriptor
        Runnable descriptor with linear and relu registry entries.
    """

    trace = tl.trace(
        ResolverModel(),
        torch.ones(1, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.preflight.passed
    return descriptor


def _replace_key(
    descriptor: SparseRunDescriptor,
    registry_id: str,
    key: FunctionRegistryKey,
) -> SparseRunDescriptor:
    """Return a descriptor with one crafted registry key.

    Parameters
    ----------
    descriptor:
        Source descriptor.
    registry_id:
        Registry entry to replace.
    key:
        Crafted replacement key.

    Returns
    -------
    SparseRunDescriptor
        Descriptor retaining all call-to-registry references.
    """

    return replace(
        descriptor,
        callable_registry=tuple(
            replace(entry, key=key) if entry.registry_id == registry_id else entry
            for entry in descriptor.callable_registry
        ),
    )


def _entry_for_name(descriptor: SparseRunDescriptor, name: str) -> str:
    """Return the registry ID whose key has the requested terminal name."""

    return next(
        entry.registry_id
        for entry in descriptor.callable_registry
        if entry.key.qualname.rsplit(".", maxsplit=1)[-1] == name
    )


def _codes(descriptor: SparseRunDescriptor) -> set[RunnableErrorCode]:
    """Return readiness diagnostic codes for one crafted descriptor."""

    report, _ = preflight_sparse_run_descriptor(descriptor)
    return {diagnostic.code for diagnostic in report.diagnostics}


@pytest.mark.smoke
def test_locked_ladder_exact_alias_reverse_and_decorated_rungs() -> None:
    """Resolve representative keys through every locked ladder rung."""

    descriptor = _descriptor()
    report, attachments = preflight_sparse_run_descriptor(descriptor)
    assert report.status is ReadinessStatus.READY
    assert attachments is not None
    by_name = {record.recorded_key.qualname: record for record in report.resolver_records}
    assert by_name["relu"].status is ResolverStatus.RESOLVED_EXACT
    linear = next(
        record for record in report.resolver_records if record.recorded_key.qualname == "linear"
    )
    assert linear.status is ResolverStatus.RESOLVED_EXACT
    assert linear.resolved_qualname == "torch.nn.functional.linear"
    assert all(id(func) not in _state._decorated_to_orig for func in attachments.values())

    relu_id = _entry_for_name(descriptor, "relu")
    reverse_descriptor = _replace_key(
        descriptor,
        relu_id,
        FunctionRegistryKey(
            namespace="custom",
            qualname="relu",
            dispatch_kind="function",
            import_path="torch.legacy:relu",
        ),
    )
    reverse_report, reverse_attachments = preflight_sparse_run_descriptor(reverse_descriptor)
    reverse_record = next(
        record for record in reverse_report.resolver_records if record.registry_id == relu_id
    )
    assert reverse_report.status is ReadinessStatus.READY
    assert reverse_attachments is not None
    assert reverse_record.provenance.startswith("reverse_index:")


def test_cross_version_alias_fixture_and_bounds() -> None:
    """Resolve every cross-version golden alias only inside its recorded bounds."""

    fixture_path = Path(__file__).parent / "fixtures" / "runnable_resolver_aliases.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixtures
    for fixture in fixtures:
        resolved = resolve_runnable_torch_alias(fixture["source"], fixture["recorded_version"])
        assert resolved is not None, fixture
        namespace, qualname, provenance = resolved
        assert namespace == fixture["target_namespace"]
        assert qualname == fixture["target_qualname"]
        assert provenance
        assert runnable_load._getattr_allowlisted(namespace, qualname) is not None

    assert resolve_runnable_torch_alias("Tensor.add", "2.0.1") is None
    assert resolve_runnable_torch_alias("Tensor.add", "2.13.0") is None


def test_resolver_namespace_table_and_exact_binding_monotonicity() -> None:
    """Cover every stock namespace and prove aliases never reinterpret exact keys."""

    exact_cases = (
        (FunctionRegistryKey("torch", "add", "function"), "torch.add"),
        (FunctionRegistryKey("torch.Tensor", "add", "method"), "torch.Tensor.add"),
        (
            FunctionRegistryKey("torch.nn.functional", "relu", "function"),
            "torch.nn.functional.relu",
        ),
        (FunctionRegistryKey("operator", "add", "function"), "operator.add"),
        (
            FunctionRegistryKey("custom", "linear", "function", import_path="torch._C._nn:linear"),
            "torch._C._nn.linear",
        ),
        (
            FunctionRegistryKey(
                "custom", "special_erf", "function", import_path="torch._C._special:special_erf"
            ),
            "torch._C._special.special_erf",
        ),
    )
    for key, expected_path in exact_cases:
        resolved = runnable_load._resolve_exact_key(key, runnable_load._stock_path_from_key(key))
        assert resolved is not None, key
        assert resolved[1] == expected_path

    descriptor = _descriptor()
    linear_id = _entry_for_name(descriptor, "linear")
    exact_private = _replace_key(
        descriptor,
        linear_id,
        FunctionRegistryKey("custom", "linear", "function", import_path="torch._C._nn:linear"),
    )
    report, _ = preflight_sparse_run_descriptor(exact_private)
    record = next(item for item in report.resolver_records if item.registry_id == linear_id)
    assert record.status is ResolverStatus.RESOLVED_EXACT
    assert record.provenance.startswith("exact_getattr:")


@pytest.mark.smoke
def test_attachment_is_all_or_none_and_resolution_runs_under_pause_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never retain partial callables and enclose resolution in pause_logging."""

    descriptor = _descriptor()
    registry_id = descriptor.callable_registry[-1].registry_id
    broken = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey("torch", "definitely_absent", "function"),
    )
    entered = 0

    @contextmanager
    def observed_pause() -> Iterator[None]:
        """Observe the resolver's logging pause context."""

        nonlocal entered
        entered += 1
        with _state.pause_logging():
            yield

    original_pause = _state.pause_logging

    @contextmanager
    def nonrecursive_pause() -> Iterator[None]:
        """Delegate to the original pause without recursing through the patch."""

        nonlocal entered
        entered += 1
        with original_pause():
            yield

    del observed_pause
    monkeypatch.setattr(_state, "pause_logging", nonrecursive_pause)
    report, attachments = preflight_sparse_run_descriptor(broken)
    assert entered == 1
    assert report.status is ReadinessStatus.UNAVAILABLE
    assert attachments is None


@pytest.mark.smoke
def test_safe_load_survives_unresolved_key_and_run_fails_once_with_full_report(
    tmp_path: Path,
) -> None:
    """Keep analysis usable while premature execution raises one aggregate error."""

    source = tl.trace(
        ResolverModel(),
        torch.ones(1, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "unresolved.tlspec"
    source.save(path, level="runnable")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["callable_registry"][0]["key"] = {
        "namespace": "torch",
        "qualname": "definitely_absent",
        "dispatch_kind": "function",
        "version": 1,
        "import_path": None,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = tl.load(path)
    assert len(loaded.layer_list) == len(source.layer_list)
    assert loaded.readiness.status is ReadinessStatus.UNAVAILABLE
    assert "_runnable_callables_by_call_id" not in loaded.__dict__
    with pytest.raises(ReattachError) as captured:
        loaded.run(torch.ones(1, 3))
    assert captured.value.fields["readiness"] is loaded.readiness
    assert captured.value.fields["diagnostics"] == loaded.readiness.diagnostics


def test_load_defers_unsupported_callable_schema_to_structured_readiness(
    tmp_path: Path,
) -> None:
    """Keep validation strict publicly while analysis load reports a version ceiling."""

    source = tl.trace(
        ResolverModel(),
        torch.ones(1, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "future_ref.tlspec"
    source.save(path, level="runnable")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["callable_ref_schema"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="callable_ref_schema"):
        tl.validation.validate_tlspec(path)
    loaded = tl.load(path)
    assert len(loaded.layer_list) == len(source.layer_list)
    assert RunnableErrorCode.UNSUPPORTED_REF_SCHEMA in {
        diagnostic.code for diagnostic in loaded.readiness.diagnostics
    }


def test_untrusted_custom_default_denies_without_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map a foreign custom key to security readiness without importing it."""

    descriptor = _descriptor()
    registry_id = descriptor.callable_registry[0].registry_id
    crafted = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey(
            "custom",
            "payload.callable",
            "function",
            import_path="attacker_stage3_payload:callable",
        ),
    )
    imported = False
    importlib_called = False
    original_import = __import__

    def guarded_import(*args: Any, **kwargs: Any) -> Any:
        """Fail if the artifact-selected module reaches Python import."""

        nonlocal imported
        if args and args[0] == "attacker_stage3_payload":
            imported = True
            raise AssertionError("artifact-selected module import attempted")
        return original_import(*args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None) -> Any:
        """Fail if readiness calls importlib for an artifact-selected module."""

        nonlocal importlib_called
        if name == "attacker_stage3_payload":
            importlib_called = True
            raise AssertionError("artifact-selected importlib call attempted")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    assert RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT in _codes(crafted)
    assert not imported
    assert not importlib_called


def test_non_torch_backend_returns_structured_unsupported_readiness() -> None:
    """Keep non-torch analysis available while marking replay unavailable."""

    report, attachments = preflight_sparse_run_descriptor(replace(_descriptor(), backend="tf"))
    assert report.status is ReadinessStatus.UNAVAILABLE
    assert attachments is None
    assert {diagnostic.code for diagnostic in report.diagnostics} >= {
        RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY
    }


def test_load_time_failure_taxonomy_and_execution_skeletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every load-time code and retain Stage 5-only taxonomy members."""

    descriptor = _descriptor()
    registry_id = descriptor.callable_registry[0].registry_id
    call = next(item for item in descriptor.calls if item.registry_id == registry_id)

    missing = replace(descriptor, calls=(replace(call, registry_id="missing:id"),))
    assert RunnableErrorCode.MISSING_CALLABLE_REF in _codes(missing)

    unsupported_schema = replace(descriptor, callable_ref_schema=cast(Any, 2))
    assert RunnableErrorCode.UNSUPPORTED_REF_SCHEMA in _codes(unsupported_schema)

    unresolved = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey("torch", "definitely_absent", "function"),
    )
    assert RunnableErrorCode.UNRESOLVED_QUALNAME in _codes(unresolved)

    private = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey("custom", "gone", "function", import_path="torch._C._gone:gone"),
    )
    assert RunnableErrorCode.PRIVATE_API_UNAVAILABLE in _codes(private)

    removed = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey("torch", "gesv", "function"),
    )
    assert RunnableErrorCode.CALLABLE_REMOVED in _codes(removed)

    drifted_calls = tuple(
        replace(item, num_positional_args=10_000) if item.registry_id == registry_id else item
        for item in descriptor.calls
    )
    drifted = _replace_key(
        replace(descriptor, calls=drifted_calls),
        registry_id,
        FunctionRegistryKey("operator", "add", "function"),
    )
    assert RunnableErrorCode.SIGNATURE_DRIFT in _codes(drifted)

    def candidate_func(value: Any) -> Any:
        """Return one value for an artificial ambiguity candidate."""

        return value

    def other_candidate_func(value: Any) -> Any:
        """Return one value through a distinct artificial candidate."""

        return value

    candidates = (
        runnable_load._ReverseCandidate("torch.same", "legacy", candidate_func),
        runnable_load._ReverseCandidate("torch.same", "legacy", other_candidate_func),
    )
    ambiguous = _replace_key(
        descriptor,
        registry_id,
        FunctionRegistryKey("custom", "legacy", "function", import_path="torch.same:legacy"),
    )
    monkeypatch.setattr(runnable_load, "_reverse_candidates", lambda *args: candidates)
    ambiguous_report, _ = preflight_sparse_run_descriptor(ambiguous)
    ambiguous_diagnostic = next(
        item
        for item in ambiguous_report.diagnostics
        if item.code is RunnableErrorCode.AMBIGUOUS_QUALNAME
    )
    assert len([item for item in ambiguous_diagnostic.details if item[0] == "candidate"]) == 2

    relu_id = _entry_for_name(descriptor, "relu")
    moved = _replace_key(
        descriptor,
        relu_id,
        FunctionRegistryKey(
            "custom",
            "relu",
            "function",
            import_path="torch._VariableFunctionsClass:relu",
        ),
    )
    assert RunnableErrorCode.CALLABLE_MOVED_OR_RENAMED in _codes(moved)
    monkeypatch.setattr(runnable_load, "_unwrap_decorated", lambda func: func)
    assert RunnableErrorCode.WRAPPER_SHADOWED in _codes(descriptor)
    assert RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value == "runtime_signature_drift"
    assert RunnableErrorCode.SEMANTIC_DRIFT.value == "semantic_drift"
