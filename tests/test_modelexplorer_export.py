"""Schema v3 exporter core tests: ids, namespaces, edges, attrs, group rows.

Covers modelexplorer memo D1-D7 and D19 on deliberate toys: the universal
always-appended ordinal id rule, dot-split pass-qualified namespaces with
escaping and malformed-stack disclosure, occurrence-preserving ported edges
with the typed unresolvable-parent failure, the curated attr set, boundary
policy, and the ``.tlspec`` round trip (composition row 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._model_explorer._ids import escape_component, mint_node_ids
from torchlens.export._model_explorer._namespace import (
    address_call_counts,
    namespace_for_entry,
)


class _PortNet(nn.Module):
    """Duplicate operands + kwargs: the occurrence-preserving port case."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed ``y + y`` (two edges) and a kwarg-named consumer."""

        y = self.fc(x)
        doubled = y + y
        return torch.where(doubled > 0, doubled, torch.zeros_like(doubled))


class _EscapeNet(nn.Module):
    """ModuleDict keys carrying %, /, and | (the D1 escape set)."""

    def __init__(self) -> None:
        """Register one weird-keyed child."""

        super().__init__()
        self.layers = nn.ModuleDict({"we%ird/na|me": nn.Linear(3, 3)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through the weird-keyed child."""

        return self.layers["we%ird/na|me"](x)


@pytest.fixture(scope="module")
def port_log() -> Any:
    """Trace the port toy once per module."""

    log = tl.trace(_PortNet().eval(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def test_ids_append_ordinal_always(port_log: Any) -> None:
    """Every node id ends with its 1-based graph-local ordinal (memo D2)."""

    payload = tl.export.to_model_explorer_dict(port_log)
    for node in payload["graphs"][0]["nodes"]:
        base, _, ordinal = node["id"].rpartition("|")
        assert base
        assert ordinal.isdigit() and int(ordinal) >= 1


def test_ids_are_stable_across_captures() -> None:
    """Same architecture, fresh weights, different batch -> identical id
    SEQUENCES (composition row 5)."""

    first = tl.trace(_PortNet().eval(), torch.randn(2, 4))
    second = tl.trace(_PortNet().eval(), torch.randn(5, 4))
    ids_first = [n["id"] for n in tl.export.to_model_explorer_dict(first)["graphs"][0]["nodes"]]
    ids_second = [n["id"] for n in tl.export.to_model_explorer_dict(second)["graphs"][0]["nodes"]]
    assert ids_first == ids_second


def test_ids_survive_tlspec_round_trip(port_log: Any, tmp_path: Path) -> None:
    """Site keys and derived ids are byte-identical after save/load
    (composition row 6)."""

    before = tl.export.to_model_explorer_dict(port_log)
    tl.save(port_log, tmp_path / "trace.tlspec")
    loaded = tl.load(tmp_path / "trace.tlspec")
    after = tl.export.to_model_explorer_dict(loaded)
    assert [n["id"] for n in before["graphs"][0]["nodes"]] == [
        n["id"] for n in after["graphs"][0]["nodes"]
    ]


def test_duplicate_operands_emit_two_edges(port_log: Any) -> None:
    """``y + y`` records two ported edges, never a deduplicated one (D4)."""

    payload = tl.export.to_model_explorer_dict(port_log)
    add_node = next(n for n in payload["graphs"][0]["nodes"] if n["label"].startswith("__add__"))
    assert len(add_node["incomingEdges"]) == 2
    slots = sorted(edge["targetNodeInputId"] for edge in add_node["incomingEdges"])
    assert slots == ["0", "1"]
    tags = [item["attrs"][0]["value"] for item in add_node["inputsMetadata"]]
    assert len(tags) == 2


def test_unresolvable_parent_refuses_typed(port_log: Any) -> None:
    """A parent ref outside both label spellings is a typed failure (D4)."""

    from torchlens.export._model_explorer._edges import incoming_edges

    class _Ghost:
        label = "child_1_1:1"
        layer_label = "child_1_1"
        num_passes = 1
        parents = ("ghost_1_9",)
        parent_arg_positions = {"args": {0: "ghost_1_9"}, "kwargs": {}}

    with pytest.raises(Exception) as excinfo:
        incoming_edges(_Ghost(), {})
    assert excinfo.value.fields["code"] == "model_explorer_parent_unresolvable"


def test_namespace_components_are_escaped() -> None:
    """%, /, | in module names percent-encode reversibly (D1)."""

    log = tl.trace(_EscapeNet().eval(), torch.randn(2, 3))
    payload = tl.export.to_model_explorer_dict(log)
    namespaces = {n["namespace"] for n in payload["graphs"][0]["nodes"]}
    inner = next(ns for ns in namespaces if ns.startswith("layers/"))
    assert inner == "layers/we%25ird%2Fna%7Cme"
    assert escape_component("we%ird/na|me") == "we%25ird%2Fna%7Cme"


def test_namespace_pass_qualifies_every_call() -> None:
    """A reused address qualifies EVERY call including call 1 (D1)."""

    class _Reuse(nn.Module):
        """Same block called twice."""

        def __init__(self) -> None:
            """Build the shared block."""

            super().__init__()
            self.blk = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call blk twice."""

            return self.blk(self.blk(x))

    log = tl.trace(_Reuse().eval(), torch.randn(2, 4))
    payload = tl.export.to_model_explorer_dict(log)
    namespaces = sorted(
        n["namespace"] for n in payload["graphs"][0]["nodes"] if n["namespace"].startswith("blk")
    )
    assert namespaces == ["blk:1", "blk:2"]


def test_malformed_stack_degrades_with_disclosure_and_strict_refuses() -> None:
    """Broken prefix continuity keeps the verified prefix; strict refuses."""

    class _Fake:
        label = "op_1_1:1"
        module_call_stack = ("encoder:1", "decoder.sub:1")

    namespace, verified, levels = namespace_for_entry(_Fake(), {})
    assert namespace == "encoder"
    assert not verified
    assert len(levels) == 1
    with pytest.raises(Exception) as excinfo:
        namespace_for_entry(_Fake(), {}, strict=True)
    assert excinfo.value.fields["code"] == "model_explorer_namespace_malformed"


def test_curated_attrs_order_and_omission(port_log: Any) -> None:
    """Attr keys follow the D6 display order; absent facts are omitted."""

    payload = tl.export.to_model_explorer_dict(port_log)
    linear = next(n for n in payload["graphs"][0]["nodes"] if n["label"] == "linear")
    keys = [attr["key"] for attr in linear["attrs"]]
    expected_order = [
        "kind",
        "op",
        "shape",
        "dtype",
        "device",
        "module",
        "pass",
        "act_bytes",
        "params",
        "flops",
        "time",
        "saved",
        "nonfinite",
        "torchlens_label",
        "site_key",
    ]
    assert keys == [key for key in expected_order if key in keys]
    assert "pass" not in keys  # single-pass op omits the pass row
    attr_map = {attr["key"]: attr["value"] for attr in linear["attrs"]}
    assert attr_map["kind"] == "parameterized"
    assert attr_map["shape"] == "2x4"
    assert attr_map["dtype"] == "float32"
    assert attr_map["module"] == "fc:1"
    assert attr_map["params"].startswith("20")


@pytest.mark.smoke
def test_source_attr_is_opt_in_and_public_drops_it(port_log: Any) -> None:
    """``source`` appears only with include_source=True, never public (D14)."""

    default_payload = tl.export.to_model_explorer_dict(port_log)
    with_source = tl.export.to_model_explorer_dict(port_log, include_source=True)
    public = tl.export.to_model_explorer_dict(
        port_log, include_source=True, privacy_profile="public"
    )

    def keys(payload: dict[str, Any]) -> set[str]:
        return {attr["key"] for node in payload["graphs"][0]["nodes"] for attr in node["attrs"]}

    assert "source" not in keys(default_payload)
    assert "source" in keys(with_source)
    assert "source" not in keys(public)
    assert "nonfinite" not in keys(public)


def test_boundary_groups_and_pin_to_top(port_log: Any) -> None:
    """Inputs/Outputs get synthetic groups; inputs pin to group top (D19)."""

    payload = tl.export.to_model_explorer_dict(port_log)
    nodes = payload["graphs"][0]["nodes"]
    input_node = next(n for n in nodes if n["namespace"] == "Inputs")
    output_node = next(n for n in nodes if n["namespace"] == "Outputs")
    assert input_node["config"] == {"pinToGroupTop": True}
    assert "config" not in output_node
    rows = payload["graphs"][0]["groupNodeAttributes"]
    assert "Inputs" in rows and "Outputs" in rows


def test_group_rows_cover_every_namespace_with_module_facts(port_log: Any) -> None:
    """Per-namespace rows carry address/class/params/op counts (D7)."""

    payload = tl.export.to_model_explorer_dict(port_log)
    rows = payload["graphs"][0]["groupNodeAttributes"]
    fc_row = rows["fc"]
    assert fc_row["address"] == "fc"
    assert fc_row["class"] == "Linear"
    assert fc_row["params"] == "20"
    assert fc_row["ops"] == "1"
    assert fc_row["calls"] == "1"
    root_row = rows[""]
    assert root_row["id_fidelity"] == "site_key"
    assert root_row["namespace_fidelity"] == "full"
    assert root_row["view"] == "execution"


def test_privacy_profile_closed_vocabulary(port_log: Any) -> None:
    """An unknown privacy profile refuses typed."""

    with pytest.raises(Exception) as excinfo:
        tl.export.to_model_explorer_dict(port_log, privacy_profile="internal")
    assert excinfo.value.fields["code"] == "model_explorer_privacy_profile_invalid"


def test_legacy_entries_get_prefixed_ids() -> None:
    """Entries without site keys mint visibly prefixed legacy ids (D2)."""

    class _Legacy:
        label = "op_1_1:1"
        layer_label = "op_1_1"
        num_passes = 1
        site_key = None

    ids, legacy_count = mint_node_ids([_Legacy(), _Legacy()])
    assert legacy_count == 2
    assert ids[0].startswith("legacy|op_1_1|")
    assert ids == ["legacy|op_1_1|1", "legacy|op_1_1|2"]


def test_call_counts_drive_qualification() -> None:
    """address_call_counts sees distinct call indices per address."""

    class _Entry:
        def __init__(self, stack: tuple[str, ...]) -> None:
            self.module_call_stack = stack

    counts = address_call_counts(
        [_Entry(("a:1", "a.b:1")), _Entry(("a:1", "a.b:2")), _Entry(("a:1",))]
    )
    assert counts["a"] == {1}
    assert counts["a.b"] == {1, 2}


def test_exported_file_is_strict_top_level(port_log: Any, tmp_path: Path) -> None:
    """The written file carries ONLY the vendor GraphCollection keys (D7)."""

    path = tl.export.model_explorer(port_log, tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"label", "graphs", "graphSorting"}
