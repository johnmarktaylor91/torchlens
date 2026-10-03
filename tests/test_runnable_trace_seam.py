"""Static ownership contracts for the narrow runnable-to-Trace seam."""

from __future__ import annotations

import ast
import pickle
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import pytest
from _source_corpus import package_ast, package_files

from torchlens._runnable_seam import (
    LEGACY_RUNNABLE_TRACE_FIELD_MAP,
    RUNNABLE_TRACE_PUBLIC_MEMBERS,
    RunnableCoordinator,
    RunnableTraceState,
    normalize_runnable_trace_state,
)
from torchlens.constants import MODEL_LOG_FIELD_ORDER
from torchlens.data_classes.trace import Trace

_PACKAGE_ROOT = Path(__file__).parents[1] / "torchlens"
_MIGRATION_DIRECT_READER_ALLOWLIST = frozenset()


def _module_ast(relative_path: str) -> ast.Module:
    """Parse one package module for static ownership checks.

    Parameters
    ----------
    relative_path:
        Path relative to the ``torchlens`` package root.

    Returns
    -------
    ast.Module
        Parsed module syntax tree.
    """

    return ast.parse((_PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


def _direct_runnable_reader_modules() -> frozenset[str]:
    """Return modules that directly name a legacy runnable Trace attribute.

    Returns
    -------
    frozenset[str]
        Package-relative Python module paths containing attribute syntax or a
        string-key lookup for ``_runnable_*`` state.
    """

    field_names = {f"_runnable_{item.name}" for item in fields(RunnableTraceState)}
    readers: set[str] = set()
    for path in package_files():
        relative = path.relative_to(_PACKAGE_ROOT).as_posix()
        if relative == "_state.py":
            continue
        tree = package_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in field_names:
                readers.add(relative)
                break
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in field_names
            ):
                readers.add(relative)
                break
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if function_name not in {
                "get",
                "setdefault",
                "pop",
                "getattr",
                "setattr",
                "hasattr",
                "delattr",
            }:
                continue
            if any(
                isinstance(argument, ast.Constant) and argument.value in field_names
                for argument in node.args
            ):
                readers.add(relative)
                break
    return frozenset(readers)


def test_runnable_trace_surface_and_state_are_declared() -> None:
    """Keep the complete public surface and private state schema explicit."""

    assert (
        frozenset(
            {
                "readiness",
                "runnable_descriptor",
                "archived_activations",
                "load_state_dict",
                "run",
            }
        )
        == RUNNABLE_TRACE_PUBLIC_MEMBERS
    )
    assert all(hasattr(Trace, member) for member in RUNNABLE_TRACE_PUBLIC_MEMBERS)
    assert {field.name for field in fields(RunnableTraceState)} == {
        "descriptor",
        "readiness",
        "staged_user_state",
        "embedded_state",
        "capture_state",
        "embedded_nonpersistent_buffers",
        "archived_activations",
        "path_faithfulness",
        "first_mismatch",
        "poisoned",
        "callables_by_call_id",
        "host_rng_consumed",
        "capture_ambient",
        "capture_fp32_unrepresentable",
        "state_alias_topology",
        "capture_state_signatures",
        "persistent_buffer_universe",
        "host_rng_unreplayable",
        "host_rng_channels",
        "host_rng_replayable_reads",
        "rng_monitor_uncertain",
        "rng_monitor_uncertain_detail",
        "output_losslessness",
        "input_nontensor_leaves",
        "input_structure",
        "input_tensor_sites",
        "input_metadata_reads",
        "input_label_layouts",
        "module_training_modes",
        "state_compromised",
    }


def test_runnable_coordinator_has_exactly_four_verbs() -> None:
    """Keep the ownership coordinator limited to the four declared operations."""

    verbs = {
        name
        for name, value in RunnableCoordinator.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert verbs == {"produce", "decode", "prepare", "execute"}


@pytest.mark.heavy
def test_trace_runnable_fields_are_collapsed_behind_the_seam() -> None:
    """Keep the collapsed state owner singular and legacy readers absent."""

    declared = frozenset(name for name in Trace.__annotations__ if name.startswith("_runnable"))
    ordered = frozenset(name for name in MODEL_LOG_FIELD_ORDER if name.startswith("_runnable"))
    assert declared == {"_runnable"}
    assert ordered == {"_runnable"}
    assert frozenset(LEGACY_RUNNABLE_TRACE_FIELD_MAP) == {
        f"_runnable_{item.name}" for item in fields(RunnableTraceState)
    }
    assert _direct_runnable_reader_modules() == _MIGRATION_DIRECT_READER_ALLOWLIST


def test_legacy_plain_pickle_fields_normalize_into_collapsed_state() -> None:
    """Keep plain-pickle compatibility while removing legacy live attributes."""

    state: dict[str, object] = {
        "_runnable_descriptor": "descriptor",
        "_runnable_poisoned": True,
    }
    runnable_state = normalize_runnable_trace_state(state)
    assert runnable_state.descriptor == "descriptor"
    assert runnable_state.poisoned is True
    assert state == {"_runnable": runnable_state}


def test_trace_pickled_at_pre_seam_commit_loads_with_collapsed_state() -> None:
    """Load a real Trace pickled at ``85c8c60f`` through the current state seam."""

    fixture = Path(__file__).parent / "fixtures" / "legacy_trace_85c8c60f.pkl"
    trace = pickle.loads(fixture.read_bytes())  # noqa: S301 - trusted checked-in fixture

    assert isinstance(trace, Trace)
    assert len(trace) == 2
    assert isinstance(trace.__dict__.get("_runnable"), RunnableTraceState)
    assert not any(legacy_name in trace.__dict__ for legacy_name in LEGACY_RUNNABLE_TRACE_FIELD_MAP)
    assert "_build_state" not in trace.__dict__
    assert "_raw_graph_ws" not in trace.__dict__
    assert "_module_capture_ws" not in trace.__dict__
    assert "_wrapper_runtime_ws" not in trace.__dict__


def test_runnable_fork_keeps_mutable_witness_ledgers_independent() -> None:
    """Fork mutable witness state while retaining immutable state bindings by identity."""

    parent = Trace.__new__(Trace)
    immutable_state = MappingProxyType({})
    parent_state = RunnableTraceState(
        staged_user_state=immutable_state,
        embedded_state=immutable_state,
        capture_state=immutable_state,
        input_metadata_reads={("args", 0): {"shape": (2, 3)}},
    )

    from torchlens.data_classes._trace_fork import _fork_model_field

    forked_state = _fork_model_field(parent, "_runnable", parent_state, {})

    assert forked_state is not parent_state
    assert forked_state.staged_user_state is immutable_state
    assert forked_state.embedded_state is immutable_state
    assert forked_state.capture_state is immutable_state
    assert forked_state.input_metadata_reads == parent_state.input_metadata_reads
    assert forked_state.input_metadata_reads is not parent_state.input_metadata_reads
    assert (
        forked_state.input_metadata_reads[("args", 0)]
        is not parent_state.input_metadata_reads[("args", 0)]
    )


def test_public_runnable_schema_has_no_runtime_imports() -> None:
    """Keep the public schema module behavior-free and transport-independent."""

    forbidden = {"_io", "capture", "_runnable_execution", "_runnable_state"}
    imported: set[str] = set()
    for node in ast.walk(_module_ast("runnable.py")):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.lstrip(".").split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert imported.isdisjoint(forbidden)


def test_bundle_does_not_import_witness_builder_internals() -> None:
    """Keep bundle transport independent of private witness construction."""

    imported_names: set[str] = set()
    for node in ast.walk(_module_ast("_io/bundle.py")):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
    assert not {name for name in imported_names if "witness" in name.lower()}
