"""W051-IO: load-time integrity anchors the loader never checked (AUD-CODE 3.0).

(a) forged ``Op.edge_substitutions`` carriers, (b) manifest <-> metadata
anchors and op-graph coherence, (d) TorchLens-owned annotation families,
(e) an absent ``root_entry_point`` on a tlspec v9+ artifact. Tampering is
done on the format 3 record cells this test itself wrote (the op table and the
trace record, decoded, edited, re-encoded and re-digested so the forgery
reaches the load-time check instead of the file digest), then ``tl.load``
must refuse typed -- never load a wrong graph.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from _format3_envelope import read_manifest, read_record, write_manifest, write_record

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens._io.format3 import contract
from torchlens._io.format3.decode import decode_value
from torchlens._io.format3.values import ValueEncoder

#: The op-table columns the forgeries below edit (the 2.x pickled ``Op`` attributes).
_OP_FIELDS = (
    "label",
    "parents",
    "pass_number",
    "num_passes",
    "annotations",
    "edge_substitutions",
    "edge_replacement_stamps",
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    torch.manual_seed(0)
    trace = tl.trace(_Model(), torch.randn(1, 1, 12, 12))
    try:
        path = tmp_path_factory.mktemp("w051_anchor") / "source.tlspec"
        trace.save(path)
        yield path
    finally:
        trace.cleanup()


def _tampered(
    source: Path,
    target: Path,
    mutate_state: Callable[[dict[str, Any]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Copy ``source`` and forge it through a decoded view of its records.

    ``mutate_state`` sees ``{"_layer_list": [op rows], "annotations": {...}}``: one
    namespace per op-table row carrying the ``_OP_FIELDS`` cells, and the trace
    record's annotations. Every edited cell is re-encoded with the writer's value
    encoder and both record files are re-digested, so the forged values reach the
    loader's structural checks.
    """

    shutil.copytree(source, target)
    if mutate_state is not None:
        op_table = read_record(target, "op")
        trace_record = read_record(target, "trace")
        columns = op_table["columns"]
        rows = [
            SimpleNamespace(**{name: decode_value(columns[name][i]) for name in _OP_FIELDS})
            for i in range(op_table["rows"])
        ]
        state = {
            "_layer_list": rows,
            "annotations": decode_value(trace_record["fields"]["annotations"]),
        }
        mutate_state(state)
        encoder = ValueEncoder()
        for i, row in enumerate(rows):
            for name in _OP_FIELDS:
                columns[name][i] = encoder.encode(getattr(row, name), f"op[{i}].{name}")
        trace_record["fields"]["annotations"] = encoder.encode(
            state["annotations"], "trace.annotations"
        )
        write_record(target, "op", op_table)
        write_record(target, "trace", trace_record)
    if mutate_manifest is not None:
        manifest = read_manifest(target)
        mutate_manifest(manifest)
        write_manifest(target, manifest)
    return target


def _assert_refuses(path: Path, code: str, **fields: Any) -> ArtifactError:
    with pytest.raises(ArtifactError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == code, excinfo.value.fields
    for key, expected in fields.items():
        assert excinfo.value.fields.get(key) == expected, excinfo.value.fields
    return excinfo.value


# --- (b) manifest <-> metadata anchors ---------------------------------------


def test_manifest_row_count_disagreeing_with_metadata_refuses(
    source_bundle: Path, tmp_path: Path
) -> None:
    """A dropped or grafted op row: the declared op count disagrees with the op records.

    Format 3 declares the retained-op count in ``records.op.rows`` (the carried
    ``n_layers`` is for blind readers; the loader rebuilds nothing from it, per
    docs/maintainers/saved_format_3.md), so the forgery edits that declaration.
    """

    def declare_999(manifest: dict[str, Any]) -> None:
        manifest["records"]["op"]["rows"] = 999

    path = _tampered(source_bundle, tmp_path / "rows.tlspec", mutate_manifest=declare_999)
    error = _assert_refuses(path, contract.REFUSAL_CODES["record_file_corrupt"])
    assert "999" in error.fields["detail"], error.fields


# --- (b) op-graph coherence ---------------------------------------------------


def test_duplicate_label_refuses(source_bundle: Path, tmp_path: Path) -> None:
    """A repeated op label refuses at the format 3 record identity check.

    The format 3 reader resolves op rows by label before the graph validator
    runs, so the duplicate is caught there with the cell position named.
    """

    def duplicate(state: dict[str, Any]) -> None:
        ops = state["_layer_list"]
        ops[2].label = ops[1].label

    path = _tampered(source_bundle, tmp_path / "dup.tlspec", duplicate)
    _assert_refuses(path, contract.REFUSAL_CODES["record_schema_invalid"], at="op[2].label")


def test_dangling_parent_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def dangle(state: dict[str, Any]) -> None:
        state["_layer_list"][2].parents = ["nonexistent_op_99"]

    path = _tampered(source_bundle, tmp_path / "dangle.tlspec", dangle)
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="dangling_relation")


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("attr", "value"),
    [("pass_number", 0), ("num_passes", 0), ("pass_number", "1")],
)
def test_pass_stamp_out_of_range_refuses(
    source_bundle: Path, tmp_path: Path, attr: str, value: Any
) -> None:
    def stamp(state: dict[str, Any]) -> None:
        setattr(state["_layer_list"][2], attr, value)

    path = _tampered(source_bundle, tmp_path / f"{attr}_{value}.tlspec", stamp)
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="pass_stamp_range")


# --- (a) tier-(ii) edge carriers ---------------------------------------------


def test_forged_edge_substitutions_refuse(source_bundle: Path, tmp_path: Path) -> None:
    def forge(state: dict[str, Any]) -> None:
        op = state["_layer_list"][2]
        op.edge_substitutions = {("positional", (0,)): {"forged": True}}
        op.edge_replacement_stamps = {("positional", (0,)): {"forged": True}}

    path = _tampered(source_bundle, tmp_path / "edge.tlspec", forge)
    _assert_refuses(path, "artifact_edge_substitutions_invalid", field="Op.edge_substitutions")


def test_edge_carrier_without_matching_stamps_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def forge(state: dict[str, Any]) -> None:
        state["_layer_list"][2].edge_substitutions = {("positional", (0,)): {}}

    path = _tampered(source_bundle, tmp_path / "edge2.tlspec", forge)
    _assert_refuses(path, "artifact_edge_substitutions_invalid")


def test_genuine_edge_intervention_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Model(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(replayable=True, save_args=True),
    )
    edge = next(e for e in trace.graph.edges if e.parent_label == "relu_1_2")
    fork = trace.fork()
    fork.do(edge.__selection__(), trace["relu_1_2"].out.clone() * 0.5)
    path = tmp_path / "genuine_edge.tlspec"
    fork.save(path)
    loaded = tl.load(path)
    carriers = [op for op in loaded._layer_list if getattr(op, "edge_substitutions", None)]
    assert carriers, "the genuine tier-(ii) carrier must persist and load"


# --- (d) TorchLens-owned annotation families -----------------------------------


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (
            lambda st: st["annotations"].__setitem__("logged_values", 5),
            'Trace.annotations["logged_values"]',
        ),
        (
            lambda st: st["annotations"].__setitem__("distributed", "junk"),
            'Trace.annotations["distributed"]',
        ),
        (
            lambda st: st["_layer_list"][2].annotations.__setitem__("collective", "junk"),
            'Op.annotations["collective"]',
        ),
        (
            lambda st: st["_layer_list"][2].annotations.__setitem__("save_mode", "bogus"),
            'Op.annotations["save_mode"]',
        ),
        (
            lambda st: st["_layer_list"][2].annotations.__setitem__("saved_out_version", "x"),
            'Op.annotations["saved_out_version"]',
        ),
        (
            lambda st: st["_layer_list"][2].annotations.__setitem__("varying_across_passes", 12),
            'Op.annotations["varying_across_passes"]',
        ),
        (
            lambda st: st["_layer_list"][2].annotations.__setitem__("dedup_source_id", "abc"),
            'Op.annotations["dedup_source_id"]',
        ),
    ],
    ids=[
        "logged_values",
        "distributed",
        "collective",
        "save_mode",
        "saved_out_version",
        "varying",
        "dedup",
    ],
)
def test_owned_annotation_families_refuse_off_shape(
    source_bundle: Path, tmp_path: Path, mutate: Callable[[dict[str, Any]], None], field: str
) -> None:
    def apply(state: dict[str, Any]) -> None:
        state.setdefault("annotations", {})
        ops = state["_layer_list"]
        if getattr(ops[2], "annotations", None) is None:
            ops[2].annotations = {}
        mutate(state)

    path = _tampered(source_bundle, tmp_path / "ann.tlspec", apply)
    _assert_refuses(path, "artifact_annotations_invalid", field=field)


def _boundary_payload(schema: str) -> dict[str, Any]:
    """A shape-complete boundary payload of the named family (plain data)."""

    payload: dict[str, Any] = {
        "schema": schema,
        "kind": "all_reduce",
        "func": "torch.distributed.all_reduce",
        "correlation": {"membership_digest": "d", "lifetime_ordinal": 0, "channel": "coll"},
        "group": {"global_ranks": [0, 1], "size": 2},
        "events": {"async_op": False, "completion_binding": "synchronous"},
        "roles": [],
        "witness": {"policy_resolved": "none"},
        "disclosures": [],
    }
    if schema == "functional_collective_boundary_v0":
        payload["func"] = "torch.distributed._functional_collectives.all_reduce"
        payload["events"] = {
            "async_op": True,
            "event_model": "funcol_issue_is_launch",
            "completion_binding": "observed_wait",
        }
        payload["disclosures"] = ["read_of_inflight_destination"]
    return payload


def _with_boundary(payload: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    def apply(state: dict[str, Any]) -> None:
        state.setdefault("annotations", {})["distributed"] = {"boundaries": [dict(payload)]}
        op = state["_layer_list"][2]
        if getattr(op, "annotations", None) is None:
            op.annotations = {}
        op.annotations["collective"] = dict(payload)

    return apply


def _drop_event_model(payload: dict[str, Any]) -> None:
    payload["events"] = {"async_op": True, "completion_binding": "observed_wait"}


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("schema", "corrupt"),
    [
        ("functional_collective_boundary_v0", _drop_event_model),
        ("functional_collective_boundary_v0", lambda p: p.__setitem__("disclosures", "x")),
        ("functional_collective_boundary_v0", lambda p: p.pop("witness")),
    ],
    ids=[
        "v0_without_event_model",
        "v0_string_disclosures",
        "v0_without_witness",
    ],
)
@pytest.mark.parametrize("where", ["journal", "op"])
def test_malformed_boundary_payload_refuses(
    source_bundle: Path,
    tmp_path: Path,
    schema: str,
    corrupt: Callable[[dict[str, Any]], Any] | None,
    where: str,
) -> None:
    """Off-shape rows of either family (or an unknown family) still refuse."""

    good_schema = (
        "functional_collective_boundary_v0" if "v0" in schema else "collective_boundary_v1"
    )
    bad = _boundary_payload(good_schema)
    bad["schema"] = schema
    if corrupt is not None:
        corrupt(bad)

    def apply(state: dict[str, Any]) -> None:
        _with_boundary(_boundary_payload(good_schema))(state)
        if where == "journal":
            state["annotations"]["distributed"]["boundaries"] = [bad]
        else:
            state["_layer_list"][2].annotations["collective"] = bad

    path = _tampered(source_bundle, tmp_path / "bad_boundary.tlspec", apply)
    field = (
        'Trace.annotations["distributed"]' if where == "journal" else 'Op.annotations["collective"]'
    )
    _assert_refuses(path, "artifact_annotations_invalid", field=field)


# --- (e) root_entry_point on tlspec v9+ ---------------------------------------


def test_absent_root_entry_point_refuses_from_the_armed_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The armed rule fires inside the governed window once the stamp reaches the floor."""

    from torchlens._io import _forgery_identity_facts as facts
    from torchlens._io.state_contract import governed_artifact_load

    monkeypatch.setattr(facts, "_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC", tl._io.TLSPEC_VERSION)
    torch.manual_seed(0)
    trace = tl.trace(_Model(), torch.randn(1, 1, 12, 12))
    trace._provenance_root_entry_point = None
    with governed_artifact_load(), pytest.raises(ArtifactError) as excinfo:
        facts._validate_root_entry_point(trace)
    assert excinfo.value.fields["code"] == "artifact_root_entry_point_invalid"
    assert excinfo.value.fields["reason"] == "absent_on_current_schema"
    # Outside the governed window (plain session state) the same trace passes.
    facts._validate_root_entry_point(trace)
