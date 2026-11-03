"""Bundle value-diff tests (memo D15, B9).

Aligned D2 ids make Model Explorer's own sync machinery do the matching;
``bundle_delta`` node data upgrades presence-diff to numeric deltas; mapping
entries exist ONLY for legacy-id nodes (with the fallback disabled so
accidental id equality cannot lie); missing values are coverage, never zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._model_explorer._diff import _sync_navigation


class _Net(nn.Module):
    """Small perturbable chain."""

    def __init__(self) -> None:
        """Two layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain chain."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def diff_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export one deterministic perturbed pair per module."""

    torch.manual_seed(0)
    subject = _Net().eval()
    reference = _Net().eval()
    reference.load_state_dict(subject.state_dict())
    with torch.no_grad():
        reference.fc1.weight += 0.05
    inputs = torch.randn(2, 8)
    subject_log = tl.trace(subject, inputs)
    reference_log = tl.trace(reference, inputs)
    try:
        yield tl.export.model_explorer_diff(
            (subject_log, reference_log), tmp_path_factory.mktemp("diff")
        )
    finally:
        subject_log.cleanup()
        reference_log.cleanup()


def test_diff_writes_both_targets_and_manifest(diff_dir: Path) -> None:
    """The product carries paired collections, per-pane data, and config."""

    names = {path.name for path in diff_dir.iterdir()}
    assert {
        "left.json",
        "right.json",
        "nodedata.left.bundle_delta.json",
        "nodedata.right.bundle_delta.json",
        "embed-config.json",
        "manifest.json",
        "README.txt",
    } <= names
    manifest = json.loads((diff_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["coverage"]["aligned"] > 0
    assert manifest["coverage"]["one_sided"] == 0


def test_pair_arity_refuses_typed() -> None:
    """Anything but exactly two captures refuses typed."""

    with pytest.raises(Exception) as excinfo:
        tl.export.model_explorer_diff((), "/tmp/unused")
    assert excinfo.value.fields["code"] == "model_explorer_diff_pair_invalid"


def test_legacy_ids_get_guarded_mapping_entries() -> None:
    """Legacy-id nodes are the ONLY ones that ride mappingEntries (D15)."""

    def payload_with(node_id: str) -> dict[str, Any]:
        return {
            "graphs": [
                {
                    "id": "g",
                    "nodes": [
                        {
                            "id": node_id,
                            "attrs": [{"key": "torchlens_label", "value": "x:1"}],
                        }
                    ],
                }
            ]
        }

    aligned = _sync_navigation(payload_with("s1|a|op||1|1"), payload_with("s1|a|op||1|1"))
    assert "mappingEntries" not in aligned
    legacy = _sync_navigation(payload_with("legacy|op_1_1|1"), payload_with("legacy|op_1_1|1"))
    assert legacy["mappingEntries"] == [
        {"leftNodeIds": ["legacy|op_1_1|1"], "rightNodeIds": ["legacy|op_1_1|1"]}
    ]
    assert legacy["disableMappingFallback"] is True
