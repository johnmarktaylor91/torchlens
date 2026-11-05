"""tvscope B6: the two-rung site inventory + the round-trip invariant.

The invariant (red before this lane on the audience's first question):
every emitted selector selects the advertised site with the returned dict
keyed by the REQUESTED string, or resolution raises a typed, useful
ambiguity whose remedy shows a WORKING spelling.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.errors
import torchlens.inventory as inv


class _Loopy(nn.Module):
    """A reused block: the multi-pass (recurrent) inventory case."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three passes through one module."""

        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


class _Branchy(nn.Module):
    """Two same-class leaves under distinct addresses (ambiguity case)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Sequential(nn.Linear(8, 8))
        self.b = nn.Sequential(nn.Linear(8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of both branches."""

        return self.a(x) + self.b(x)


def test_no_input_rung_lists_modules_without_forward() -> None:
    """Rung one: free listing, shapes honestly unknown, no forward runs."""

    result = inv.list_sites(_Branchy())
    assert result.rung == "modules"
    assert len(result) > 0
    assert all(row.shape is None for row in result)
    assert any("shapes" in d for d in result.disclosures)


def test_ambiguous_spelling_teaches_working_selectors() -> None:
    """Ambiguity is typed and its remedy lists working spellings."""

    model = _Branchy().eval()
    result = inv.list_sites(model, torch.randn(2, 8))
    with pytest.raises(torchlens.errors.SiteInventoryError) as excinfo:
        result.resolve("0")
    assert excinfo.value.fields["code"] == "site_selector_ambiguous"
    candidates = excinfo.value.fields["candidates"]
    assert candidates
    # every taught spelling WORKS
    out = tl.extract(model, torch.randn(2, 8), list(candidates))
    assert all(c in out for c in candidates)


def test_unknown_spelling_refuses_typed_with_candidates() -> None:
    """Unknown needles refuse typed, nearest candidates disclosed."""

    result = inv.list_sites(_Branchy().eval(), torch.randn(2, 8))
    with pytest.raises(torchlens.errors.SiteInventoryError) as excinfo:
        result.resolve("definitely_not_a_site")
    assert excinfo.value.fields["code"] == "site_selector_unknown"


def test_structure_only_shapes_are_labeled_hypotheses() -> None:
    """A structure_only forward rung labels every shape a hypothesis."""

    model = _Branchy().eval()
    result = inv.list_sites(
        model,
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    assert result.shapes_are_hypotheses is True
    assert any("HYPOTHESIS" in d for d in result.disclosures)
    payload = result.to_json()
    assert payload["shapes_are_hypotheses"] is True


def test_inventory_serializes_to_json() -> None:
    """Structured rows are JSON-portable (the deferred-CLI plumbing)."""

    import json

    result = inv.list_sites(_Loopy().eval(), torch.randn(2, 8))
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["schema"] == "tl_site_inventory_v1"
    assert payload["rows"][0]["selector"]
