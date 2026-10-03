"""Selector-valued ``layers=``: resolve on batch zero, freeze, attest (D12).

The frozen plan is ordered structural site keys + pass-qualified labels;
every later batch re-resolves and must attest the exact plan. Zero-site and
plan-drift refusals fire typed BEFORE any wrong shard commits; mapping
selectors namespace children; the frozen plan and request record ride the
manifest and the resume signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import extract_dataset, open_extraction


class _TwoBlock(nn.Module):
    """Two named blocks so module selectors have territory."""

    def __init__(self) -> None:
        """Build deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.enc = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4, 2), nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run both blocks."""

        return self.head(self.enc(inputs))


def test_selector_run_freezes_plan_and_matches_string_run(tmp_path: Path) -> None:
    """A selector harvest equals the same sites harvested by label."""

    model = _TwoBlock().eval()
    stimuli = torch.randn(6, 3)
    by_selector = extract_dataset(model, stimuli, tl.func("relu"), batch_size=3, progress=False)
    assert sorted(by_selector) == ["relu_1_2:1", "relu_2_4:1"]
    by_label = extract_dataset(
        model, stimuli, ["relu_1_2", "relu_2_4"], batch_size=3, progress=False
    )
    for key, tensor in by_selector.items():
        assert torch.equal(tensor, by_label[key.split(":")[0]])


@pytest.mark.smoke
def test_selector_disk_run_records_plan_and_resumes(tmp_path: Path) -> None:
    """The frozen plan rides the manifest; a compatible resume adopts it."""

    model = _TwoBlock().eval()
    stimuli = torch.randn(6, 3)
    out = tmp_path / "artifact"
    extract_dataset(model, stimuli, tl.func("relu"), batch_size=3, output_dir=out, progress=False)
    manifest = json.loads((out / "manifest.json").read_text())
    plan = manifest["run"]["selector_plan"]
    assert plan["request"]["kind"] == "selector"
    assert [entry["label"] for entry in plan["entries"]] == ["relu_1_2:1", "relu_2_4:1"]
    assert all(entry["site_key"] for entry in plan["entries"])
    assert manifest["signature"]["layers_kind"] == "selector"
    # Completed compatible resume: a true no-op.
    paths = extract_dataset(
        model,
        stimuli,
        tl.func("relu"),
        batch_size=3,
        output_dir=out,
        progress=False,
        resume=True,
    )
    assert len(paths) == 2
    reader = open_extraction(out)
    assert sorted(reader.keys) == ["relu_1_2:1", "relu_2_4:1"]


@pytest.mark.smoke
def test_selector_mapping_namespaces_children(tmp_path: Path) -> None:
    """A mapping selector matching several sites namespaces its children."""

    model = _TwoBlock().eval()
    out = extract_dataset(
        model,
        torch.randn(4, 3),
        {"act": tl.func("relu")},
        batch_size=2,
        progress=False,
    )
    assert sorted(out) == ["act/relu_1_2:1", "act/relu_2_4:1"]
    single = extract_dataset(
        model,
        torch.randn(4, 3),
        {"first": tl.in_module("enc") & tl.func("relu")},
        batch_size=2,
        progress=False,
    )
    assert sorted(single) == ["first"], "single-site mapping keys stay unnamespaced"


def test_selector_zero_sites_refuses(tmp_path: Path) -> None:
    """A selector matching nothing refuses typed (D12)."""

    model = _TwoBlock().eval()
    # The capture itself discloses the zero-match selector (a warning); the
    # extraction refusal is the typed gate right behind it.
    with (
        pytest.raises(InvalidArgumentError) as excinfo,
        pytest.warns(UserWarning, match="matched zero sites"),
    ):
        extract_dataset(model, torch.randn(4, 3), tl.func("conv2d"), progress=False)
    assert excinfo.value.fields["code"] == "extraction_selector_no_sites"


def test_selector_excess_fanout_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An under-constrained selector refuses above the site ceiling."""

    monkeypatch.setattr("torchlens._extraction.selector_plan._MAX_SELECTOR_SITES", 1)
    model = _TwoBlock().eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(model, torch.randn(4, 3), tl.func("relu"), progress=False)
    assert excinfo.value.fields["code"] == "extraction_selector_excess_fanout"


class _WidthBranching(nn.Module):
    """A model whose op graph depends on the batch's token width."""

    def __init__(self) -> None:
        """Build deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one extra relu for wide batches (batch-dependent control flow)."""

        hidden = torch.relu(self.fc(inputs))
        if inputs.shape[0] > 2:
            hidden = torch.relu(hidden)
        return hidden


def test_selector_plan_drift_refuses_before_commit(tmp_path: Path) -> None:
    """D12 attestation: a batch resolving a different plan refuses typed."""

    model = _WidthBranching().eval()
    stimuli = torch.randn(5, 4)  # batches of 3 then 2: extra relu on batch 0 only
    out = tmp_path / "artifact"
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            stimuli,
            tl.func("relu"),
            batch_size=3,
            output_dir=out,
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_selector_plan_violated"
    ledger = (out / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 1, "only the attested prefix committed"
