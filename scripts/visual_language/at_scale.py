"""The deck's "At scale and edge cases" section: real architectures, drawn as a user sees them.

The audit pack's real-model pages (its sections E and J) and VISUALS.md's at-scale rows, moved
into the deck so one generator draws everything. Real architectures only for now; a row whose
model needs a download the deck does not make yet is listed as pending with its reason. These
renders are slow next to the teaching slides; they run in every full build.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from scripts.visual_language.sentinels import OUTPUT
from scripts.visual_language.slides import Panel, Slide


def _panel(name: str, fixture: str, collapse: str) -> Panel:
    return Panel(
        name,
        fixture,
        kwargs={**OUTPUT, "collapse": collapse},
        label=f'collapse="{collapse}"',
        conventions=False,
    )


@dataclass(frozen=True)
class ScaleRow:
    """One real-architecture page: what it shows a reader and the renders that show it."""

    id: str
    title: str
    why: str
    panels: tuple[Panel, ...] = ()
    status: str = "active"
    reason: str = ""
    extras: Mapping[str, str] = field(default_factory=dict)


SCALE_ROWS: tuple[ScaleRow, ...] = (
    ScaleRow(
        "scale-resnet18",
        "At scale: ResNet-18",
        "The overview a user gets on a real model: stem, four stages, head; collapse none, "
        "auto and max side by side.",
        (
            _panel("a", "ResNet18", "none"),
            _panel("b", "ResNet18", "auto"),
            _panel("c", "ResNet18", "max"),
        ),
    ),
    ScaleRow(
        "scale-resnet50",
        "At scale: ResNet-50",
        "Deep stages under collapse max: segment boxes name an address range and never claim "
        "to be a module.",
        (_panel("a", "ResNet50", "max"),),
    ),
    ScaleRow(
        "scale-densenet121",
        "At scale: DenseNet-121",
        "High fan-in: every dense layer concatenates all earlier ones (the head-label pileup "
        "row of the 2026-08-31 eyeball).",
        (_panel("a", "DenseNet121", "auto"),),
    ),
    ScaleRow(
        "scale-gpt2",
        "At scale: GPT-2 small",
        "Rolled transformer, fold monotonicity, auxiliary outputs.",
        status="pending",
        reason="needs a pinned Hugging Face download; the deck builds offline today",
    ),
    ScaleRow(
        "scale-generation",
        "At scale: a generation episode",
        "The above-ceiling path: about 2,800 ops of distilgpt2 generation.",
        status="pending",
        reason="needs a pinned Hugging Face download; the deck builds offline today",
    ),
)


def scale_slides() -> tuple[Slide, ...]:
    """The section as deck slides, one per active row."""

    return tuple(
        Slide(row.id, row.title, row.why, panels=row.panels, layout="grid")
        for row in SCALE_ROWS
        if row.status == "active"
    )
