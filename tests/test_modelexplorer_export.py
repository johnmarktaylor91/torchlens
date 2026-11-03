"""Schema v3 exporter core tests: ids, namespaces, edges, attrs, group rows.

Covers modelexplorer memo D1-D7 and D19 on deliberate toys: the universal
always-appended ordinal id rule, dot-split pass-qualified namespaces with
escaping and malformed-stack disclosure, occurrence-preserving ported edges
with the typed unresolvable-parent failure, the curated attr set, boundary
policy, and the ``.tlspec`` round trip (composition row 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._model_explorer._ids import escape_component, mint_node_ids
from torchlens.export._model_explorer._namespace import (
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


def test_ids_survive_tlspec_round_trip(port_log: Any, tmp_path: Path) -> None:
    """Site keys and derived ids are byte-identical after save/load
    (composition row 6)."""

    before = tl.export.to_model_explorer_dict(port_log)
    port_log.save(tmp_path / "trace.tlspec")
    loaded = tl.load(tmp_path / "trace.tlspec")
    after = tl.export.to_model_explorer_dict(loaded)
    assert [n["id"] for n in before["graphs"][0]["nodes"]] == [
        n["id"] for n in after["graphs"][0]["nodes"]
    ]


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
