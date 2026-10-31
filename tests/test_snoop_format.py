"""Echo format contract: byte stability, CI cleanliness, volume honesty.

Pins snoop memo test 12 (byte-stable transcripts, zero escape bytes off-tty
-- the escape-leak regression class behind the ``terminal_file_line_link``
defect), the max_lines suppression accounting, the zero-match disclosure,
and the hard grammar gate: a stats line renders with the finiteness family
ABSENT.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._snoop._format import render_line, render_stats_segment
from torchlens._source_links import terminal_file_line_link
from torchlens.options import NarrationOptions
from torchlens.stats import FamilyEvidence, tensor_stats
from torchlens.types import NarrationEvent

_ESCAPE_BYTES = re.compile("[\x1b\x9d\x07]")


class SmallNet(nn.Module):
    """Tiny deterministic module."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc -> relu."""

        return torch.relu(self.fc(x))


def _run_transcript(seed: int) -> str:
    """Capture one seeded transcript through a StringIO sink."""

    torch.manual_seed(seed)
    model = SmallNet()
    torch.manual_seed(seed)
    x = torch.randn(2, 4)
    sink = io.StringIO()
    tl.record(model, x, narrate=NarrationOptions(select=True, sink=sink, stats="sampled"))
    return sink.getvalue()


def test_transcripts_are_byte_stable_across_seeded_runs() -> None:
    """Two seeded runs render byte-identical transcripts (memo test 12)."""

    assert _run_transcript(7) == _run_transcript(7)


@pytest.mark.smoke
def test_non_tty_transcript_is_escape_clean() -> None:
    """Zero ANSI/OSC-8 bytes on a non-tty sink (the escape-leak regression)."""

    transcript = _run_transcript(3)
    assert not _ESCAPE_BYTES.search(transcript)


@pytest.mark.smoke
def test_terminal_file_line_link_gates_on_tty() -> None:
    """The shipped OSC-8 defect: escapes only when links are enabled."""

    plain = terminal_file_line_link("/tmp/model.py", 12, enable_links=False)
    assert plain == "/tmp/model.py:12"
    assert not _ESCAPE_BYTES.search(plain)
    linked = terminal_file_line_link("/tmp/model.py", 12, enable_links=True)
    assert "\x1b]8;;" in linked
    # Default resolves from sys.stdout.isatty(); under pytest capture that is
    # not a tty, so the default path must be byte-clean too.
    assert not _ESCAPE_BYTES.search(terminal_file_line_link("/tmp/model.py", 12))


def test_sampled_stats_segment_never_claims_finiteness() -> None:
    """HARD GATE: the grammar renders with the finiteness family absent."""

    segment = render_stats_segment(
        replace(
            tensor_stats(torch.ones(6912)),
            mean=0.01,
            std=0.71,
            finite_min=None,
            finite_max=None,
            mean_evidence=FamilyEvidence("sampled", 6912, 4096),
            std_evidence=FamilyEvidence("sampled", 6912, 4096),
            nonfinite_evidence=FamilyEvidence("unavailable", 6912, reason="sampled narration"),
        )
    )
    assert "sampled=4096/6912" in segment
    assert "~=" in segment
    for forbidden in ("nan", "inf", "finite"):
        assert forbidden not in segment


@pytest.mark.smoke
def test_exact_stats_segment_prints_census_or_clearance() -> None:
    """Exact rungs print nonzero census tokens, or the exact clearance."""

    dirty = render_stats_segment(
        replace(
            tensor_stats(torch.ones(10)),
            mean=1.0,
            std=None,
            finite_min=None,
            finite_max=None,
            num_nan=2,
            num_posinf=0,
            num_neginf=1,
        )
    )
    assert "nan=2" in dirty and "-inf=1" in dirty and "+inf=" not in dirty
    clean = render_stats_segment(
        replace(tensor_stats(torch.ones(10)), mean=None, std=None, finite_min=None, finite_max=None)
    )
    assert clean == "finite"


def test_missing_facts_are_omitted_never_zeroed() -> None:
    """A metadata line omits absent columns instead of fabricating them."""

    line = render_line(NarrationEvent(kind="op", ordinal=3, label="relu_1_2_raw"))
    assert "relu_1_2_raw" in line
    for forbidden in ("None", "@", "pass=", "out="):
        assert forbidden not in line


@pytest.mark.smoke
def test_max_lines_suppression_is_announced_and_counted() -> None:
    """One suppression marker, exact counts, footer discloses the total."""

    sink = io.StringIO()
    tl.record(
        SmallNet(),
        torch.randn(2, 4),
        narrate=NarrationOptions(select=True, sink=sink, max_lines=1),
    )
    lines = [line for line in sink.getvalue().split("\n") if line]
    markers = [line for line in lines if "max_lines=1 reached" in line]
    assert len(markers) == 1
    footer = lines[-1]
    assert "1 lines narrated" in footer
    assert "suppressed" in footer


@pytest.mark.smoke
def test_zero_match_scoped_capture_warns_once_with_code() -> None:
    """A complete scoped run with zero matches warns echo_zero_match."""

    sink = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.record(
            SmallNet(),
            torch.randn(2, 4),
            narrate=NarrationOptions(select=tl.func("softmax"), sink=sink),
        )
    codes = [getattr(item.message, "fields", {}).get("code") for item in caught]
    assert codes.count("echo_zero_match") == 1


def test_matched_scoped_capture_does_not_warn() -> None:
    """The zero-match disclosure stays silent when the scope matched."""

    sink = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.record(
            SmallNet(),
            torch.randn(2, 4),
            narrate=NarrationOptions(select=tl.func("relu"), sink=sink),
        )
    codes = [getattr(item.message, "fields", {}).get("code") for item in caught]
    assert "echo_zero_match" not in codes


@pytest.mark.smoke
def test_file_sink_flushes_per_line(tmp_path) -> None:
    """File sinks flush per line: bytes are on disk before the run ends."""

    target = tmp_path / "run.log"
    lines_at_first_call: list[int] = []

    class SpyModule(nn.Module):
        """Module that inspects the sink file mid-forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op, then check the file, then run another."""

            y = torch.relu(x)
            lines_at_first_call.append(len(target.read_text().splitlines()))
            return y + 1

    tl.record(
        SpyModule(), torch.randn(2, 4), narrate=NarrationOptions(select=True, sink=str(target))
    )
    assert lines_at_first_call and lines_at_first_call[0] >= 1
    assert len(target.read_text().splitlines()) > lines_at_first_call[0]


@pytest.mark.parametrize("verdict", [True, False])
def test_reuse_verdict_grammar_has_no_invented_count(verdict: bool) -> None:
    """Positive verdicts say presence; negative verdicts prove exact finiteness."""

    from types import SimpleNamespace

    from torchlens._snoop._stats import reuse_stats

    trace = SimpleNamespace(_nonfinite_capture={"events": {"raw": verdict}})
    record = reuse_stats(trace, "raw", 10, shape=(10,), dtype="f32", device="cpu")
    assert record is not None
    assert render_stats_segment(record) == ("nonfinite present" if verdict else "finite")


def test_sampled_nonfinite_evidence_cannot_clear_even_with_zero_counts() -> None:
    """A planted sampled zero census must never become a finiteness clearance."""

    record = replace(
        tensor_stats(torch.ones(10)),
        nonfinite_evidence=FamilyEvidence("sampled", 10, 4),
    )
    line = render_stats_segment(record)
    assert "finite" not in line and "nan=" not in line and "inf=" not in line


def test_mixed_evidence_marks_only_sampled_families() -> None:
    """An exact census stays exact when a different family was sampled."""

    record = replace(
        tensor_stats(torch.arange(32.0)),
        std_evidence=FamilyEvidence("sampled", 32, 8),
    )
    line = render_stats_segment(record)
    assert "mean=" in line and "mean~=" not in line
    assert "sd~=" in line and "sampled=8/32" in line
    assert line.endswith("finite")
