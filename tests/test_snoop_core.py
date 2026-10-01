"""Core echo narration invariants (lane F28; snoop memo tests 1-3, 15-16).

Pinned here: the exactly-once contract (T-C), tier parity (T-B), scope
independence from retention, held-ancestor structure lines, followed_by
releases, and multi-pass Recorder resets -- each a live tripwire for the
mount classes the panel measured into the ground.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import EchoOptions

pytestmark = pytest.mark.smoke


class TwoBlock(nn.Module):
    """Tiny two-block MLP with real module structure."""

    def __init__(self) -> None:
        """Build two linear blocks around a relu."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1 -> relu -> fc2."""

        return self.fc2(torch.relu(self.fc1(x)))


def _transcript(sink: io.StringIO) -> list[str]:
    """Return the sink's lines without the trailing empty split."""

    return [line for line in sink.getvalue().split("\n") if line]


def test_record_echo_true_narrates_and_keeps_nothing() -> None:
    """ "Narrate everything, keep nothing" is legal and is the fast path."""

    sink = io.StringIO()
    recording = tl.record(TwoBlock(), torch.randn(2, 4), echo=EchoOptions(select=True, sink=sink))
    lines = _transcript(sink)
    assert recording.records == []
    op_lines = [line for line in lines if line.lstrip().startswith("#")]
    # input source + 3 ops = 4 event lines; enter/exit pairs for fc1/fc2.
    assert len(op_lines) == 4
    assert sum(1 for line in lines if line.lstrip().startswith(">")) == 2
    assert sum(1 for line in lines if line.lstrip().startswith("<")) == 2
    assert lines[-1].startswith("-- echo:")


def test_exactly_once_per_event_with_saves_nothing_narrator() -> None:
    """T-C: line count equals distinct event count (the 554-vs-277 case)."""

    seen: list[str] = []
    tl.record(
        TwoBlock(),
        torch.randn(2, 4),
        echo=EchoOptions(select=True, sink=seen.append),
    )
    op_lines = [line for line in seen if line.lstrip().startswith("#")]
    labels = [line.split()[1] for line in op_lines]
    assert len(labels) == len(set(labels)) == 4


def test_tier_parity_transcripts_are_byte_identical() -> None:
    """T-B: the same model narrated on record and trace renders identically.

    The regression tripwire that keeps the disqualified hooks-mount asymmetry
    (weaker lines on the expensive tier) from returning.
    """

    model = TwoBlock()
    x = torch.randn(2, 4)
    trace_sink, record_sink = io.StringIO(), io.StringIO()
    tl.trace(model, x, echo=EchoOptions(select=True, sink=trace_sink))
    tl.record(model, x, echo=EchoOptions(select=True, sink=record_sink))
    assert trace_sink.getvalue() == record_sink.getvalue()


def test_selector_scope_with_held_ancestors() -> None:
    """Scoped narration prints no unselected module noise (Sol's refinement)."""

    sink = io.StringIO()
    tl.trace(
        TwoBlock(),
        torch.randn(2, 4),
        echo=EchoOptions(select=tl.in_module("fc2"), sink=sink),
    )
    lines = _transcript(sink)
    body = [line for line in lines if not line.startswith("-- echo:")]
    assert any("fc2" in line and line.lstrip().startswith(">") for line in body)
    assert not any("fc1" in line for line in body)
    assert any("linear_2" in line for line in body)


def test_save_scope_and_echo_scope_are_independent() -> None:
    """A narrated-but-unsaved op prints; a saved-but-unnarrated op stays silent."""

    sink = io.StringIO()
    recording = tl.record(
        TwoBlock(),
        torch.randn(2, 4),
        save=tl.func("relu"),
        echo=EchoOptions(select=tl.in_module("fc2"), sink=sink),
    )
    lines = _transcript(sink)
    assert any("linear_2" in line for line in lines)
    assert not any("relu" in line for line in lines)
    assert any("relu" in record.ctx.label for record in recording.records)


def test_echo_modules_is_structure_only() -> None:
    """echo='modules' is the cheap flight recorder: structure lines only."""

    sink = io.StringIO()
    tl.trace(TwoBlock(), torch.randn(2, 4), echo=EchoOptions(select="modules", sink=sink))
    body = [line for line in _transcript(sink) if not line.startswith("-- echo:")]
    assert body
    assert all(line.lstrip().startswith((">", "<")) for line in body)


def test_multi_output_op_narrates_one_line_per_output() -> None:
    """Memo test 1: one line per output on multi-output ops."""

    class Splitter(nn.Module):
        """Emits two outputs from one op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Split and re-add so both outputs are consumed."""

            a, b = torch.split(x, 2, dim=1)
            return a + b

    sink = io.StringIO()
    tl.record(Splitter(), torch.randn(2, 4), echo=EchoOptions(select=True, sink=sink))
    split_lines = [line for line in _transcript(sink) if "split" in line]
    assert len(split_lines) == 2
    assert any("out=1" in line for line in split_lines)


def test_followed_by_releases_are_marked_late() -> None:
    """Bounded followed_by releases buffered matches marked late=N."""

    sink = io.StringIO()
    tl.record(
        TwoBlock(),
        torch.randn(2, 4),
        echo=EchoOptions(select=tl.func("linear") & tl.followed_by(tl.func("relu")), sink=sink),
        lookback=4,
    )
    lines = _transcript(sink)
    released = [line for line in lines if "linear_1" in line]
    assert len(released) == 1
    assert "late=" in released[0]


def test_recorder_resets_per_pass_and_marks_passes() -> None:
    """Recorder resets ordinals/tail per forward; pass=N labels later passes."""

    sink = io.StringIO()
    model = TwoBlock()
    with tl.fastlog.Recorder(model, echo=EchoOptions(select=True, sink=sink)) as recorder:
        recorder.log(torch.randn(2, 4))
        recorder.log(torch.randn(2, 4))
    lines = _transcript(sink)
    relu_lines = [line for line in lines if "relu" in line]
    assert len(relu_lines) == 2
    assert "pass=" not in relu_lines[0]
    assert "pass=2" in relu_lines[1]
    # Ordinals restart per pass: both relu lines carry the same ordinal.
    assert relu_lines[0].split()[0] == relu_lines[1].split()[0]


def test_record_context_requires_grad_is_populated() -> None:
    """Snoop build row 2: ``tensor_requires_grad`` is live data, not None.

    The panel measured 0/554 populated at its baseline commit; the field is
    populated today and this pins it (reading the dataclass as runtime
    reality was a bug two labs committed and one caught).
    """

    seen: list[object] = []

    def probe(ctx: object) -> bool:
        """Record the requires_grad field for every op context."""

        if getattr(ctx, "kind", None) == "op":
            seen.append(ctx.tensor_requires_grad)
        return False

    model = TwoBlock()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    tl.record(model, torch.randn(2, 4), save=probe)
    assert seen
    assert all(isinstance(value, bool) for value in seen)
    assert any(value is True for value in seen)
