"""Family grammar refuses forged cells before record construction, with cell paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _format3_envelope import copy_artifact, save_tiny, set_cell
from torch import nn

import torchlens as tl
from torchlens._io.format3.decode import Decoder
from torchlens._io.format3.schema import FAMILIES
from torchlens._io.format3.values import ValueEncoder
from torchlens._io.format_errors import ArtifactError
from torchlens.intervention.types import Edit, FireRecord


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real small trace for cell surgery.

    Parameters
    ----------
    tmp_path_factory:
        Artifact directory factory.
    """

    root = tmp_path_factory.mktemp("family-schemas") / "trace"
    trace = save_tiny(root)
    trace.cleanup()
    return root


def _forged_cell(saved: Path, target: Path, column: str, value: Any, record: str = "trace") -> Path:
    """Copy an artifact, replace one cell and restamp its manifest digests.

    Parameters
    ----------
    saved, target:
        Original artifact and mutation destination.
    column, value:
        Column and decoded replacement value.
    record:
        Record table whose first row is edited.
    """

    root = copy_artifact(saved, target)
    cell = ValueEncoder().encode(value, at=f"{record}[0].{column}")
    set_cell(root, record, column, 0, cell)
    return root


def _telemetry(status: str, *, available: bool = True) -> dict[str, Any]:
    """Build one complete launch envelope for grammar boundary cases.

    Parameters
    ----------
    status, available:
        Launch attribution and envelope availability.
    """

    launch = dict.fromkeys(("device", "launch_name", "duration", "stream", "runtime_correlation"))
    return {
        "_available": available,
        "_launches": [{**launch, "attribution_status": status}],
        "_relations": [],
    }


@pytest.mark.parametrize(
    "column,value,code,path",
    [
        ("grad_fn_timing_provenance", "invented_clock", "grad_fn_timing_provenance", ""),
        ("_provenance_distributed_scope", "global", "distributed_scope", ""),
        ("checkpoint_invocation_witness", {}, "checkpoint_witness", ""),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": True,
                "tokens": {},
                "degrade_flags": [],
                "verdict": "no_checkpoint_invocation_observed",
            },
            "checkpoint_witness",
            ".token_count",
        ),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 0,
                "tokens": {},
                "degrade_flags": [],
                "verdict": "evidence_incomplete",
            },
            "checkpoint_witness",
            ".verdict",
        ),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 1,
                "tokens": {
                    1: {
                        "pack_count": -1,
                        "unpack_evidence_count": 0,
                        "site_key_candidates": [],
                        "window_evidence": [],
                    }
                },
                "degrade_flags": [],
                "verdict": "checkpoint_invocations_observed",
            },
            "checkpoint_witness",
            ".tokens.1.pack_count",
        ),
        (
            "annotations",
            {
                "capture_advisories": [
                    {
                        "kind": "escape",
                        "count": True,
                        "first_location": None,
                        "message": "m",
                    }
                ]
            },
            "capture_advisories",
            ".capture_advisories[0].count",
        ),
        (
            "annotations",
            {
                "sidecar": {
                    "example.facts": {
                        "schema_id": "facts_v1",
                        "version": False,
                        "owner": "example",
                        "payload": {},
                    }
                }
            },
            "sidecar",
            ".sidecar.example.facts.version",
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": _telemetry("guessed"),
            },
            "kernel_telemetry",
            "._kernel_telemetry._launches[0].attribution_status",
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": {
                    "_available": False,
                    "_launches": [],
                    "_relations": [],
                }
            },
            "kernel_telemetry",
            "._kernel_telemetry._launches",
        ),
        (
            "_source_snapshots",
            [{"path": "model.py", "digest": "bad", "text": None}],
            "source_snapshots",
            "[0].digest",
        ),
    ],
)
def test_forged_family_cell_refuses_with_code_and_path(
    saved: Path,
    tmp_path: Path,
    column: str,
    value: Any,
    code: str,
    path: str,
) -> None:
    """A digest-valid forgery reaches its family's grammar refusal.

    Parameters
    ----------
    saved, tmp_path:
        Real artifact and isolated mutation directory.
    column, value, code, path:
        Cell mutation, family code stem and expected nested diagnostic path.
    """

    loaded = tl.load(saved)
    loaded.cleanup()
    root = _forged_cell(saved, tmp_path / "forged", column, value)
    with pytest.raises(ArtifactError) as caught:
        tl.load(root)
    assert caught.value.fields["code"] == f"artifact_{code}_invalid"
    assert caught.value.fields["at"] == f"trace[0].{column}{path}"
    assert f"trace[0].{column}{path}" in str(caught.value)


@pytest.mark.parametrize(
    "column,value",
    [
        ("grad_fn_timing_provenance", "unmeasured"),
        ("_provenance_distributed_scope", "rank_local_shard"),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 0,
                "tokens": {},
                "degrade_flags": (),
                "verdict": "no_checkpoint_invocation_observed",
            },
        ),
        (
            "annotations",
            {
                "capture_advisories": (
                    {
                        "kind": "escape",
                        "count": 1,
                        "first_location": None,
                        "message": "",
                    },
                )
            },
        ),
        (
            "annotations",
            {
                "sidecar": {
                    "example.facts": {
                        "schema_id": "facts_v1",
                        "version": 1,
                        "owner": "example",
                        "payload": {"any": 3},
                    }
                }
            },
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": _telemetry("unavailable", available=False),
            },
        ),
        ("_source_snapshots", [{"path": "model.py", "digest": "a" * 64, "text": None}]),
    ],
)
def test_valid_family_cells_remain_loadable(
    saved: Path,
    tmp_path: Path,
    column: str,
    value: Any,
) -> None:
    """Admitted tuple containers, optional values and open payloads still load.

    Parameters
    ----------
    saved, tmp_path:
        Baseline artifact and isolated mutation directory.
    column, value:
        Valid replacement cell.
    """

    root = _forged_cell(saved, tmp_path / "valid", column, value)
    loaded = tl.load(root)
    loaded.cleanup()


@pytest.fixture(scope="module")
def saved_backward(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a trace with a real backward-pass record.

    Parameters
    ----------
    tmp_path_factory:
        Artifact directory factory.
    """

    root = tmp_path_factory.mktemp("family-backward") / "trace"
    trace = tl.trace(
        nn.Linear(3, 2),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    try:
        trace.log_backward(trace[trace.output_layers[0]].out.sum())
        trace.save(root)
    finally:
        trace.cleanup()
    return root


def _audit(kind: str, **changes: Any) -> dict[str, Any]:
    """Return one minimal audit row with optional single-field mutations.

    Parameters
    ----------
    kind, changes:
        Row variant and overwritten fields.
    """

    if kind == "EVENT":
        return {
            "kind": kind,
            "schema": "intervention_event_v2",
            "event_id": "event",
            "transaction_id": "transaction",
            "door": "selection",
            "selection_repr": "selected",
            "parent_event_id": None,
            "lane": "capture",
            "status": "armed",
            "fire_count": 0,
            "edit_names": [],
            "site_keys": [],
            "zero_fire_rule_ids": [],
            "rules": [],
            "error": None,
            "event_digest": "a" * 64,
            **changes,
        }
    target = {
        "ACT": {"sites": []},
        "EDGE": {"edges": []},
        "PARAM": {"params": [], "disclosure": ""},
    }[kind]
    return {
        "kind": kind,
        "edit": "edit",
        "resolve_digest": "a" * 64,
        "selection_repr": "",
        **target,
        **changes,
    }


def _flops(**changes: Any) -> dict[str, Any]:
    """Return a coherent counted backward record with optional mutations.

    Parameters
    ----------
    changes:
        Overwritten fields.
    """

    return {
        "convention": "",
        "executed_flops": 3,
        "rows": [("node", 1, None, 2, None)],
        "uncounted_ops": [],
        "unattributed": [("other", 1)],
        **changes,
    }


@pytest.mark.parametrize(
    "record,column,value,code,path",
    [
        ("trace", "annotations", {"logged_values": {1: 2}}, "annotations", ".logged_values<key 1>"),
        (
            "trace",
            "annotations",
            {"distributed": {"boundaries": [{}]}},
            "annotations",
            ".distributed.boundaries[0]",
        ),
        (
            "trace",
            "_graph_structure_evidence",
            {"claims": {}, "discharge": "absent"},
            "structure_evidence",
            ".claims",
        ),
        ("trace", "_cost_forward_peak_memory_reason", 5, "forward_peak_memory", ""),
        ("trace", "backward_peak_memory_reason", "", "backward_peak_memory", ""),
        (
            "trace",
            "intervention_audit",
            [_audit("ACT", patch_source={})],
            "intervention_audit",
            "[0].patch_source",
        ),
        (
            "trace",
            "intervention_audit",
            [_audit("EDGE", edit=None)],
            "intervention_audit",
            "[0].edit",
        ),
        (
            "trace",
            "intervention_audit",
            [_audit("PARAM", disclosure=False)],
            "intervention_audit",
            "[0].disclosure",
        ),
        (
            "trace",
            "intervention_audit",
            [_audit("EVENT", fire_count=True)],
            "intervention_audit",
            "[0].fire_count",
        ),
        (
            "trace",
            "intervention_audit",
            [_audit("EVENT", event_digest="a" * 64 + "\n")],
            "intervention_audit",
            "[0].event_digest",
        ),
        (
            "trace",
            "intervention_audit",
            [_audit("EVENT", rules=[None])],
            "intervention_audit",
            "[0].rules[0]",
        ),
        ("op", "annotations", {"collective": {}}, "annotations", ".collective"),
        ("op", "annotations", {"save_mode": "borrow"}, "annotations", ".save_mode"),
        ("op", "annotations", {"saved_out_version": True}, "annotations", ".saved_out_version"),
        (
            "op",
            "annotations",
            {"varying_across_passes": {1: 2}},
            "annotations",
            ".varying_across_passes<key 1>",
        ),
        ("op", "annotations", {"dedup_source_id": True}, "annotations", ".dedup_source_id"),
        (
            "op",
            "annotations",
            {"dedup_reference_label": 2},
            "annotations",
            ".dedup_reference_label",
        ),
        ("op", "episode_step", True, "episode_step", ""),
        (
            "backward_pass",
            "_executed_flops_record",
            _flops(executed_flops=True),
            "executed_flops_record",
            ".executed_flops",
        ),
        (
            "backward_pass",
            "_executed_flops_record",
            _flops(rows=[("node", 0, None, 2, None)]),
            "executed_flops_record",
            ".rows[0][1]",
        ),
        (
            "backward_pass",
            "_executed_flops_record",
            _flops(executed_flops=10),
            "executed_flops_record",
            "",
        ),
        (
            "backward_pass",
            "_executed_flops_record",
            _flops(rows=[("node", 1, None, None, None)]),
            "executed_flops_record",
            ".rows[0][3]",
        ),
    ],
)
def test_remaining_family_cells_refuse_with_paths(
    saved_backward: Path,
    tmp_path: Path,
    record: str,
    column: str,
    value: Any,
    code: str,
    path: str,
) -> None:
    """Record-cell forgeries retain the family code and precise nested path.

    Parameters
    ----------
    saved_backward, tmp_path:
        Baseline artifact and isolated mutation directory.
    record, column, value, code, path:
        Mutated cell, expected refusal family and nested diagnostic path.
    """

    loaded = tl.load(saved_backward)
    loaded.cleanup()
    root = _forged_cell(saved_backward, tmp_path / "forged", column, value, record)
    with pytest.raises(ArtifactError) as caught:
        tl.load(root)
    assert caught.value.fields["code"] == f"artifact_{code}_invalid"
    assert caught.value.fields["at"] == f"{record}[0].{column}{path}"
    assert caught.value.fields["field"] in str(caught.value)
    assert "Remedy:" in str(caught.value)


@pytest.mark.parametrize(
    "record,column,value",
    [
        ("trace", "annotations", {"logged_values": {"user": object.__name__}, "user": 1}),
        ("trace", "intervention_audit", [_audit("ACT")]),
        ("trace", "intervention_audit", [_audit("EDGE")]),
        ("trace", "intervention_audit", [_audit("PARAM")]),
        ("trace", "intervention_audit", [_audit("EVENT")]),
        (
            "op",
            "annotations",
            {
                "save_mode": "copy",
                "saved_out_version": -1,
                "dedup_source_id": -1,
                "dedup_source_version": -2,
                "dedup_reference_label": "",
                "varying_across_passes": {},
            },
        ),
        (
            "op",
            "annotations",
            {
                "collective": dict.fromkeys(
                    ("kind", "func", "correlation", "group", "events", "roles", "witness"), None
                )
                | {"schema": "collective_boundary_v1", "user": True}
            },
        ),
        ("backward_pass", "_executed_flops_record", _flops()),
        (
            "backward_pass",
            "_executed_flops_record",
            _flops(executed_flops=None, rows=[("", 1, "", None, 0)]),
        ),
    ],
)
def test_remaining_valid_cells_load(
    saved_backward: Path,
    tmp_path: Path,
    record: str,
    column: str,
    value: Any,
) -> None:
    """Open annotations and grammar-valid audit and counted rows remain accepted.

    Parameters
    ----------
    saved_backward, tmp_path:
        Baseline artifact and isolated mutation directory.
    record, column, value:
        Table and replacement cell.
    """

    root = _forged_cell(saved_backward, tmp_path / "valid", column, value, record)
    loaded = tl.load(root)
    loaded.cleanup()


def _migration_samples() -> dict[str, list[Any]]:
    """Supply accepted boundaries and malformed neighbors for the migration oracle.

    Notes
    -----
    Candidates include scalar, container and required-key boundaries.
    """

    claims = {
        "graph_structure": "observed_canonical",
        "shapes_dtypes": "hypothesis",
        "flops_geometry_bytes": "hypothesis_estimate",
        "declared_buffer_mutations": "hypothesis",
        "measured_values": "unavailable",
        "timing_allocator_memory": "unavailable",
        "declared_branch_assumptions": "none",
    }
    digest = "a" * 64
    occurrence = {"consumer": "", "edge_address": "()", "value_digest": digest}
    recipe = {"edge_address": "()", "resolve_digest": digest}
    snapshots = [{"path": "a.py", "digest": digest, "text": None}]
    samples = {
        "distributed_scope": ["rank_local_shard", "global", 0, {}, []],
        "timing_provenance": [None, "unmeasured", "perf_counter", "clock", True, {}],
        "checkpoint_witness": [
            None,
            {},
            {
                "token_count": 0,
                "tokens": {},
                "degrade_flags": (),
                "verdict": "no_checkpoint_invocation_observed",
            },
        ],
        "capture_advisories": [
            [],
            (),
            None,
            {},
            [{"kind": "k", "count": 1, "first_location": None, "message": ""}],
        ],
        "sidecar_namespace": [
            {},
            None,
            {"a.b\n": {"schema_id": "id", "owner": "a", "version": 1, "payload": None}},
        ],
        "source_snapshots": [
            None,
            [],
            snapshots,
            snapshots * 2,
            [{**snapshots[0], "digest": digest + "\n"}],
        ],
        "kernel_telemetry": [{"_available": True, "_launches": [], "_relations": ()}, {}, None],
        "annotation_families": [
            None,
            {},
            [],
            {"logged_values": {1: None}},
            {"logged_values": {"x": None}, "distributed": None},
        ],
        "op_annotations": [
            None,
            {},
            [],
            {"save_mode": "borrow"},
            {"saved_out_version": True},
        ],
        "intervention_audit": [
            None,
            [],
            (),
            {},
            *[[_audit(k)] for k in ("ACT", "EDGE", "PARAM", "EVENT")],
            [_audit("EVENT", event_digest=digest + "\n")],
        ],
        "param_occurrence": [
            occurrence,
            {**occurrence, "edge_address": "[]"},
            {**occurrence, "value_digest": digest + "\n"},
            {},
            None,
        ],
        "selection_recipe": [
            recipe,
            {**recipe, "edge_address": "[]"},
            {**recipe, "note": "", "param_address": "p"},
            {**recipe, "resolve_digest": digest + "\n"},
            {},
        ],
        "evidence_claims": [
            {"claims": claims, "discharge": "absent"},
            {"claims": {}, "discharge": "absent"},
            {"claims": claims, "discharge": None},
        ],
        "forward_peak_memory_reason": [None, "", "unavailable", 1, False],
        "backward_peak_memory_reason": [None, "", "unavailable", 1, False],
        "episode_step_stamps": [None, True, -1, "0"],
        "executed_flops_records": [
            None,
            {},
            _flops(),
            _flops(executed_flops=10),
            _flops(executed_flops=True),
        ],
    }
    for family in ("ACT", "EDGE", "PARAM", "EVENT"):
        valid = _audit(family)
        for key in sorted(set(valid) - {"sites", "edges", "params"}):
            samples["intervention_audit"].append([{**valid, key: None}])
            samples["intervention_audit"].append([{k: v for k, v in valid.items() if k != key}])
    return samples


def _legacy_accepts(family: str, value: Any) -> bool:
    """Run the original procedural grammar as an independent migration oracle.

    Parameters
    ----------
    family, value:
        Family name and candidate value.
    """

    from torchlens._io import (
        _forgery_identity_facts as identity,
        _load_validators as load,
        forgery_validation as legacy,
    )
    from torchlens.report._backward_attribution import executed_flops_record_problem

    if family == "executed_flops_records":
        return value is None or executed_flops_record_problem(value) is None
    trace = SimpleNamespace(
        _layer_list=[], _cost_forward_peak_memory=None, _cost_backward_peak_memory=None
    )
    node = FAMILIES[family]
    if "column" in node:
        setattr(trace, node["column"], value)
    callbacks = {
        "capture_advisories": lambda: legacy._validate_capture_advisories(
            SimpleNamespace(annotations={"capture_advisories": value})
        ),
        "sidecar_namespace": lambda: legacy._validate_sidecar_namespace(
            SimpleNamespace(annotations={"sidecar": value})
        ),
        "kernel_telemetry": lambda: legacy._validate_kernel_telemetry(
            SimpleNamespace(annotations={"_kernel_telemetry": value})
        ),
        "annotation_families": lambda: load._validate_annotation_families(trace),
        "op_annotations": lambda: load._validate_op_annotations(SimpleNamespace(annotations=value)),
        "param_occurrence": lambda: legacy._validate_param_occurrence(0, value),
        "selection_recipe": lambda: legacy._validate_selection_recipe(0, value),
        "evidence_claims": lambda: legacy._validate_evidence_claims(value),
        "forward_peak_memory_reason": lambda: identity._validate_peak_memory_reason(
            trace, "forward", code="artifact_forward_peak_memory_invalid"
        ),
        "backward_peak_memory_reason": lambda: identity._validate_peak_memory_reason(
            trace, "backward", code="artifact_backward_peak_memory_invalid"
        ),
        "episode_step_stamps": lambda: identity._validate_episode_step_stamps(
            SimpleNamespace(_layer_list=[SimpleNamespace(episode_step=value, label="op")])
        ),
    }
    callback = callbacks.get(family)
    try:
        if callback is not None:
            callback()
        else:
            getattr(legacy, f"_validate_{family}")(trace)
    except (ArtifactError, TypeError, ValueError):
        return False
    return True


@pytest.mark.parametrize(
    "family,value",
    [(family, value) for family, samples in _migration_samples().items() for value in samples],
)
def test_family_migration_matches_original_grammar(family: str, value: Any) -> None:
    """The schema migration preserves the original validator's acceptance boundary.

    Parameters
    ----------
    family, value:
        Independent family and accepted or refused candidate value.
    """

    from torchlens._io.format3._families import _apply_family

    node = FAMILIES[family]
    if family in ("param_occurrence", "selection_recipe"):
        node = {**node, "code": "artifact_intervention_audit_invalid"}
    accepted = True
    try:
        _apply_family(node, value, f"family.{family}")
    except ArtifactError:
        accepted = False
    assert accepted == _legacy_accepts(family, value)


@pytest.mark.parametrize("guard_first", [False, True])
@pytest.mark.parametrize("audit", [None, []])
def test_recipe_binding_preserves_audit_presence_and_decode_order(
    guard_first: bool, audit: Any
) -> None:
    """Conditional recipes retain the original absent-audit acceptance boundary.

    Parameters
    ----------
    guard_first, audit:
        Cell decode order and absent or present audit guard.
    """

    decoder = Decoder(defer_records=True)
    edit = FireRecord(edit=Edit(name="helper", selection_recipe={}))
    cells = [
        (ValueEncoder().encode([edit], at="op[0].edits"), "op[0].edits"),
        (audit, "trace[0].intervention_audit"),
    ]
    if guard_first:
        cells.reverse()
    if audit is None:
        for cell, at in cells:
            decoder.decode(cell, at=at)
    else:
        with pytest.raises(ArtifactError) as caught:
            for cell, at in cells:
                decoder.decode(cell, at=at)
        assert caught.value.fields["code"] == "artifact_intervention_audit_invalid"
        assert caught.value.fields["at"] == "op[0].edits[0].edit.selection_recipe"
