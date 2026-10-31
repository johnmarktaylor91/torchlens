"""Render-path counters for the U07.5 profiling pass.

The profile of a default ``draw()`` (counters, then wall clock as a report)
put two rows on top. By wall time: Graphviz layouts -- the sibling-order
verifier lays out the plain and the ordered DOT with ``-Tplain``, and the
final render then laid out the ordered source again, three layouts where two
carry all the information. By Python self time: the DOT line scanner behind
the verifier stepped through every character of every DOT line. These tests
pin the fixed behavior with deterministic counts, never a wall-clock
threshold.
"""

from __future__ import annotations

import random
import re
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import _render_ordering, _render_utils


def _reference_dot_line_is_complete(line: str) -> bool:
    """The pre-U07.5 character-by-character scanner, kept as the oracle."""

    in_quote = False
    html_depth = 0
    previous = ""
    escaped = False
    for char in line:
        if in_quote:
            in_quote = escaped or char != '"'
            escaped = not escaped and char == "\\"
        elif html_depth:
            html_depth += {"<": 1, ">": -1}.get(char, 0)
        elif char == '"':
            in_quote = True
        elif char == "<":
            html_depth = 1
        elif char == ">" and previous != "-":
            return False
        previous = char
    return not in_quote and html_depth == 0


class _CountingPattern:
    """Proxy for the scanner's regex that counts the positions it visits."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self.pattern = pattern
        self.visited = 0
        self.scanned_chars = 0

    def finditer(self, line: str) -> Any:
        """Yield the real matches, counting them and the line length."""

        self.scanned_chars += len(line)
        for match in self.pattern.finditer(line):
            self.visited += 1
            yield match


class _Branchy(nn.Module):
    """Fan-out model so the sibling-order post-pass has chains to verify."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.heads = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        hidden = torch.relu(self.stem(x))
        return torch.stack([head(hidden) for head in self.heads]).sum(0)


def test_dot_line_scanner_matches_reference_scanner() -> None:
    """The jump scanner agrees with the character scanner on adversarial lines."""
    rng = random.Random(0)
    alphabet = 'ab-"\\<> '
    lines = ["a -> b [label=<<B>x</B>>]", 'a [label="q\\"uote"]', "a -> b", "x>", "-"]
    lines += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24))) for _ in range(5000)]
    for line in lines:
        assert _render_ordering._dot_line_is_complete(line) == _reference_dot_line_is_complete(
            line
        ), line


def test_dot_line_scanner_visits_only_state_characters(monkeypatch: Any, tmp_path: Any) -> None:
    """On a real draw the scanner visits far fewer positions than DOT characters."""
    counter = _CountingPattern(_render_ordering._DOT_LINE_SPECIAL)
    monkeypatch.setattr(_render_ordering, "_DOT_LINE_SPECIAL", counter)
    trace = tl.trace(_Branchy(), torch.randn(2, 4))
    try:
        trace.draw(path=str(tmp_path / "branchy"), vis_save_only=True, file_format="svg")
    finally:
        trace.cleanup()
    assert counter.scanned_chars > 0, "the sibling-order verifier did not scan the DOT"
    # Before U07.5 the loop stepped once per character (visited == scanned).
    assert counter.visited * 2 < counter.scanned_chars


def _count_graphviz_runs(monkeypatch: Any) -> list[list[str]]:
    """Spy on the Graphviz spawn seam; return the list the calls land in."""

    calls: list[list[str]] = []
    real = _render_utils.run_bounded_subprocess

    def spy(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(list(cmd))
        return real(cmd, **kwargs)

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", spy)
    return calls


def test_default_draw_reuses_the_verified_layout(monkeypatch: Any, tmp_path: Any) -> None:
    """A default draw runs two Graphviz layouts, not three: the final render reuses one."""
    calls = _count_graphviz_runs(monkeypatch)
    trace = tl.trace(_Branchy(), torch.randn(2, 4))
    try:
        dot = trace.draw(path=str(tmp_path / "reuse"), vis_save_only=True, file_format="svg")
        decision = trace._last_sibling_ordering_decision
    finally:
        trace.cleanup()
    assert decision.survivor_count > 0, "the fixture must exercise the ordering post-pass"
    # Before U07.5: baseline -Tplain, ordered -Tplain, final -Tsvg = 3 layouts.
    assert len(calls) == 2, calls
    assert any("-Tsvg" in cmd for cmd in calls)
    svg = (tmp_path / "reuse.svg").read_text(encoding="utf-8")
    assert "<svg" in svg and "linear" in svg
    assert "tl:sibling-order:start" in dot


def test_reused_artifact_equals_a_fresh_render(monkeypatch: Any, tmp_path: Any) -> None:
    """The reused artifact is byte-identical to rendering the final DOT afresh."""
    trace = tl.trace(_Branchy(), torch.randn(2, 4))
    try:
        dot = trace.draw(path=str(tmp_path / "reuse"), vis_save_only=True, file_format="svg")
    finally:
        trace.cleanup()
    (tmp_path / "final.dot").write_text(dot, encoding="utf-8")
    fresh = _render_utils.run_bounded_subprocess(
        ["dot", "-Tsvg", str(tmp_path / "final.dot")], timeout=120
    ).stdout
    assert (tmp_path / "reuse.svg").read_bytes() == fresh
