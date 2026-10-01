"""F39 unhide gates: promoted surface pins + the unhide teaching ratchet.

Lane F39 (WALKTHROUGH B-IV) promotes the shipped-but-hidden surfaces into the
declared story: ``tl.stats``/``tl.aggregate`` enter ``__all__`` (conflict-ledger
row 8 -- the promotion five memos claimed), and README/docs surface
``trace.profile()``, ``emit_nvtx=True``, ``register_op_rule``,
``fastlog.Recorder``, ``utils.list_modules``/``list_ops``, the
verification-backed receptive-field story, and the already-interactive HTML
export. These tests pin the promotion both directions (a rename or a quiet
re-hide fails here, not in a launch review) and EXECUTE the new stats doc's
fences so the teaching surface cannot drift from the code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_stats_and_aggregate_are_declared_surface() -> None:
    """The five-memo promotion: both names in ``__all__`` and resolving."""

    assert "stats" in tl.__all__
    assert "aggregate" in tl.__all__
    import torchlens.stats as stats_module

    assert tl.stats is stats_module
    assert tl.aggregate is stats_module.aggregate


@pytest.mark.smoke
def test_stats_namespace_serves_the_advertised_accumulators() -> None:
    """Every accumulator the stats doc tables advertises resolves."""

    for name in (
        "StreamingStat",
        "Mean",
        "Norm",
        "Quantile",
        "TopK",
        "Covariance",
        "CrossCovariance",
        "CKA",
        "cka",
        "PCA",
        "Histogram",
        "Spine",
        "Aggregator",
        "save_fitted",
        "load_fitted",
    ):
        assert hasattr(tl.stats, name), name


@pytest.mark.smoke
def test_readme_keeps_the_unhidden_surfaces_visible() -> None:
    """The unhide ratchet: README must keep naming each surfaced capability.

    Word-presence only (prose stays free to improve); dropping a claim
    re-hides a shipped surface and needs a deliberate edit here.
    """

    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    for claim in (
        "tl.aggregate",
        "tl.stats",
        ".profile(",
        "honesty()",
        "emit_nvtx=True",
        "register_op_rule",
        "Recorder",
        "list_modules",
        "list_ops",
        "docs/reference/stats.md",
        "tl.export.html",
        "interactive",
        'scope="receptive_field"',
    ):
        assert claim in readme, f"README lost the unhidden-surface claim {claim!r}"


class _UnhideEncoder(nn.Module):
    """Conv+ReLU encoder mirroring the canonical docs demo model."""

    def __init__(self) -> None:
        """Initialize one padded convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the ReLU-activated convolution of ``value``.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return torch.relu(self.conv(value))


class _UnhideModel(nn.Module):
    """conv -> relu -> conv demo satisfying the stats doc's ambient contract."""

    def __init__(self) -> None:
        """Initialize the encoder and a convolutional head."""

        super().__init__()
        self.encoder = _UnhideEncoder()
        self.head = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the head applied to the encoded input.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.head(self.encoder(value))


@pytest.mark.heavy
def test_stats_doc_python_fences_execute() -> None:
    """Execute every python fence of docs/reference/stats.md top to bottom.

    Same ambient contract as the canonical-page harness: ``model`` (a small
    conv -> relu -> conv model) is the only injected name; everything else the
    doc code must define itself. A fence referencing an undefined name is a
    documentation bug.
    """

    text = (_REPO / "docs" / "reference" / "stats.md").read_text(encoding="utf-8")
    blocks = re.compile(r"```python\n(.*?)\n```", re.DOTALL).findall(text)
    assert blocks, "docs/reference/stats.md has no python fences"
    torch.manual_seed(0)
    namespace: dict[str, object] = {"model": _UnhideModel().eval()}
    for index, block in enumerate(blocks):
        code = compile(block, f"docs/reference/stats.md:python-block-{index}", "exec")
        exec(code, namespace)  # noqa: S102 -- executing our own documentation


@pytest.mark.smoke
def test_stats_doc_is_linked_from_readme_and_glossary() -> None:
    """The doc of record is reachable from both teaching entry points."""

    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    glossary = (_REPO / "docs" / "reference" / "glossary.md").read_text(encoding="utf-8")
    assert "docs/reference/stats.md" in readme
    assert "stats.md" in glossary
