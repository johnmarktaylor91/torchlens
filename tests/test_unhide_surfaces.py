"""F39 unhide gates: promoted surface pins + the unhide teaching ratchet.

Lane F39 (WALKTHROUGH B-IV) promotes the shipped-but-hidden surfaces into the
declared story: ``tl.stats``/``tl.stats.aggregate`` enter ``__all__`` (conflict-ledger
row 8 -- the promotion five memos claimed), and README/docs surface
``trace.profile()``, ``emit_nvtx=True``, ``register_op_rule``,
``fastlog.Recorder``, ``torchlens.inventory.list_sites``, the
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


def test_stats_and_aggregate_are_declared_surface() -> None:
    """The five-memo promotion: ``stats`` in ``__all__``; ``tl.stats.aggregate`` resolves."""

    assert "stats" in tl.__all__
    import torchlens.stats as stats_module

    assert tl.stats is stats_module
    assert tl.stats.aggregate is stats_module.aggregate


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
        "RunningStats",
        "Aggregator",
        "save_fitted",
        "load_fitted",
    ):
        assert hasattr(tl.stats, name), name


#: Each unhidden surface's doc of record: (spelling the old README carried, the page
#: that now carries it, the spelling there, the most README links a reader follows
#: to reach that page, or None when no README path of two links or fewer exists).
#: README.md is JMT's text (4 Nov 2026 rewrite); a surface the README no longer
#: names is pinned on the page that now teaches it.
_UNHIDDEN_SURFACE_HOMES: tuple[tuple[str, str, str, int | None], ...] = (
    ("tl.stats.aggregate", "docs/also_in_torchlens.md", "tl.stats.aggregate", 1),
    ("tl.stats", "docs/also_in_torchlens.md", "tl.stats", 1),
    (".profile(", "docs/also_in_torchlens.md", ".profile(", 1),
    ("evidence_labels()", "docs/reference/glossary.md", "ProfileReport.evidence_labels", 1),
    ("emit_nvtx=True", "docs/reference/glossary.md", "emit_nvtx", 1),
    ("register_op_rule", "docs/reference/device_attribution.md", "register_op_rule", None),
    ("Recorder", "docs/reference/glossary.md", "Recorder", 1),
    ("torchlens.inventory", "docs/neuroai/journey.md", "torchlens.inventory", 1),
    ("list_sites", "docs/neuroai/journey.md", "list_sites", 1),
    ("docs/reference/stats.md", "docs/also_in_torchlens.md", "reference/stats.md", 1),
    ("tl.export.html", "docs/also_in_torchlens.md", "tl.export.html", 1),
    ("interactive", "README.md", "interactive", 0),
    (
        'scope="receptive_field"',
        "docs/receptive_projective_fields.md",
        'scope="receptive_field"',
        2,
    ),
)
_MD_LINK = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)\s]*)?\)")


def _linked_pages(page: str) -> set[str]:
    """Repo-relative markdown pages one relative link away from ``page``."""

    source = _REPO / page
    found = set()
    for target in _MD_LINK.findall(source.read_text(encoding="utf-8")):
        if "://" in target:
            continue
        resolved = (source.parent / target).resolve()
        if resolved.is_file() and resolved.is_relative_to(_REPO):
            found.add(resolved.relative_to(_REPO).as_posix())
    return found


def _readme_link_distance(page: str, limit: int) -> int | None:
    """README links a reader follows to reach ``page`` (0 for the README), or None."""

    frontier, seen = {"README.md"}, {"README.md"}
    for hops in range(limit + 1):
        if page in frontier:
            return hops
        frontier = {nxt for cur in frontier for nxt in _linked_pages(cur)} - seen
        seen |= frontier
    return None


def test_readme_keeps_the_unhidden_surfaces_visible() -> None:
    """The unhide ratchet: each surfaced capability stays named on its doc of record.

    Word-presence only (prose stays free to improve); dropping a claim
    re-hides a shipped surface and needs a deliberate edit here. Each row also
    pins how many README links a reader follows to reach that page.
    """

    for claim, home, spelling, hops in _UNHIDDEN_SURFACE_HOMES:
        text = (_REPO / home).read_text(encoding="utf-8")
        assert spelling in text, f"{home} lost the unhidden-surface claim {claim!r} ({spelling!r})"
        if hops is not None:
            assert _readme_link_distance(home, hops) is not None, (
                f"{home} (home of {claim!r}) is no longer within {hops} link(s) of README.md"
            )


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


def test_stats_doc_is_linked_from_readme_and_glossary() -> None:
    """The doc of record is reachable from both teaching entry points."""

    glossary = (_REPO / "docs" / "reference" / "glossary.md").read_text(encoding="utf-8")
    # The README reaches the stats page through "Also in TorchLens" (one page between).
    assert _readme_link_distance("docs/reference/stats.md", 2) is not None
    assert "stats.md" in glossary
