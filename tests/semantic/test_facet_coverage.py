"""Tests for the facet-coverage report feeding the maintenance pipeline."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import facet_coverage


class Block(nn.Module):
    """Tiny transformer block with attention and MLP children."""

    def __init__(self, d: int = 8) -> None:
        """Initialize attention and MLP children."""

        super().__init__()
        self.attn = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual attention and MLP updates."""

        x = x + self.attn(x)
        return x + self.mlp(x)


class Exotic(nn.Module):
    """Custom module with no matching semantic recipe."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an elementwise transform."""

        return x.sin() * 2.0


class Model(nn.Module):
    """Model mixing a recipe-classified block and an unclassified module."""

    def __init__(self) -> None:
        """Initialize the block, the exotic module, and a norm."""

        super().__init__()
        self.block = Block()
        self.exotic = Exotic()
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run block, exotic transform, and norm."""

        return self.norm(self.exotic(self.block(x)))


def test_facet_coverage_classifies_and_inventories_unclassified() -> None:
    """Coverage rows split recipe-classified modules from structural-only ones."""

    torch.manual_seed(0)
    model = Model().eval()
    log = tl.trace(
        model, torch.randn(2, 3, 8), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    report = facet_coverage(log)

    by_address = {row.address: row for row in report.rows}
    assert "transformer_residuals" in by_address["block"].recipes
    assert "resid_post" in by_address["block"].available
    assert "layer_norm" in by_address["norm"].recipes
    assert by_address["exotic"].recipes == ()
    assert any(row.address == "exotic" for row in report.unclassified)
    assert any(row.address == "block" for row in report.classified)
    assert report.recipe_counts()["transformer_residuals"] >= 1


def test_facet_coverage_reports_typed_absences() -> None:
    """Declared-but-missing facets appear with their typed absence status."""

    torch.manual_seed(0)

    class NoPreBlock(nn.Module):
        """Block with no residual-stream input: its output shape matches no input.

        ``resid_pre`` is anchored by dataflow + shape (the residual stream is
        shape-preserved through a block), so a block that narrows its hidden
        dimension on output has NO input op that can earn the anchor -- a
        genuine structural absence, not a capture gap: no ``save=`` setting
        could ever produce the facet here.
        """

        def __init__(self, d: int = 8) -> None:
            """Initialize attention and MLP children."""

            super().__init__()
            self.attn = nn.Linear(d, d)
            self.mlp = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run attention then a residual MLP update, narrowing the output."""

            y = self.attn(x)
            return (y + self.mlp(y))[:, :, : y.shape[-1] // 2]

    class TwoBlocks(nn.Module):
        """Chain a full block into the pre-less block."""

        def __init__(self) -> None:
            """Initialize both blocks."""

            super().__init__()
            self.block = Block()
            self.nopre = NoPreBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run both blocks."""

            return self.nopre(self.block(x))

    model = TwoBlocks().eval()
    log = tl.trace(
        model, torch.randn(2, 3, 8), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    report = facet_coverage(log)

    nopre_row = next(row for row in report.rows if row.address == "nopre")
    missing = {facet: status for facet, status, _detail in nopre_row.missing}
    assert missing.get("resid_pre") == "structurally_absent"
    statuses = {status for _facet, status, _detail in nopre_row.missing}
    assert statuses <= {"needs_capture", "structurally_absent", "declared_not_produced"}
    assert ("resid_pre", "structurally_absent") in report.missing_counts()


@pytest.mark.smoke
def test_facet_coverage_discloses_multi_call_modules() -> None:
    """A reused module's typed facets refusal becomes a disclosed row, not a crash."""

    torch.manual_seed(0)

    class Reuser(nn.Module):
        """Model calling one norm module twice."""

        def __init__(self) -> None:
            """Initialize the shared norm."""

            super().__init__()
            self.norm = nn.LayerNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the same norm twice."""

            return self.norm(self.norm(x))

    model = Reuser().eval()
    log = tl.trace(
        model, torch.randn(2, 3, 8), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    report = facet_coverage(log)

    norm_row = next(row for row in report.rows if row.address == "norm")
    assert norm_row.note != ""
    assert norm_row in report.unresolved
    assert norm_row not in report.unclassified
    assert "Unresolved facet views" in report.to_markdown()


def test_facet_coverage_markdown_lists_candidates() -> None:
    """The markdown rendering includes classified tables and candidate classes."""

    torch.manual_seed(0)
    model = Model().eval()
    log = tl.trace(
        model, torch.randn(2, 3, 8), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    text = facet_coverage(log).to_markdown()

    assert "classified:" in text
    assert "| block | Block | transformer_residuals |" in text
    assert "Unclassified module classes" in text
    assert "Exotic x1" in text
