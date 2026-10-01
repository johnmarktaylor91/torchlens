"""F08 charset contract: canonical ASCII, degrade map, detection ladder.

Summary memo 3.9: ``str(result)`` is byte-stable ASCII with no ANSI/OSC-8
bytes; unicode appears only at explicit display boundaries; the two
renders differ ONLY through the declared 1:1 glyph table (CI-checked
byte-for-byte here); detection fails only toward ASCII. Includes the
dual-charset goldens (volatile Memory line masked; everything else
byte-exact).
"""

from __future__ import annotations

import io
import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report._summary_charset import GLYPH_TABLE, degrade, detect_style

pytestmark = pytest.mark.smoke


class _GoldenToy(nn.Module):
    """Deterministic toy for the byte-exact dual-charset goldens."""

    def __init__(self) -> None:
        """One Conv stack with a functional flatten."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv -> relu -> flatten -> fc."""

        return self.fc(torch.flatten(self.relu(self.conv(x)), 1))


@pytest.fixture(scope="module")
def golden_report():
    """One rebuilt report over the golden toy (trace torn down after)."""

    trace = tl.trace(_GoldenToy().eval(), torch.zeros(1, 3, 8, 8))
    try:
        yield trace.summary()
    finally:
        trace.cleanup()


def _mask_volatile(text: str) -> str:
    """Mask the measured Memory line (host RSS legitimately varies)."""

    return re.sub(r"(?m)^Memory   .*$", "Memory   <masked>", text)


#: The dual-charset goldens (memo item 17: goldens pin EXPLICIT charsets
#: only, never auto). The Memory line is masked; every other byte is exact.
_GOLDEN_ASCII = """\
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params        (%)  fwd flops
---------------------------------------------------------
conv (Conv2d)  (1, 4, 8, 8)     108   ... 7.8%     13.82K
relu (ReLU)    (1, 4, 8, 8)       -          -        256
flatten_1_3    (1, 256)           -          -          0
fc (Linear)    (1, 5)         1.28K  ### 92.2%      2.56K
---------------------------------------------------------
Params   1,388 declared | trainable 1,388 (100%)
Compute  16.64K FLOPs fwd (fma=2) | 8.19K MACs | 4/4 ops known (formula-exact)
Memory   <masked>
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | health CHECKED-AND-CLEAN
More:    result.details() | view='compute' for per-row MACs | docs/reference/summary.md"""

_GOLDEN_UNICODE = """\
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params        (%)  fwd flops
─────────────────────────────────────────────────────────
conv (Conv2d)  (1, 4, 8, 8)     108   ░░░ 7.8%     13.82K
relu (ReLU)    (1, 4, 8, 8)       -          -        256
flatten_1_3    (1, 256)           -          -          0
fc (Linear)    (1, 5)         1.28K  ███ 92.2%      2.56K
─────────────────────────────────────────────────────────
Params   1,388 declared | trainable 1,388 (100%)
Compute  16.64K FLOPs fwd (fma=2) | 8.19K MACs | 4/4 ops known (formula-exact)
Memory   <masked>
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | health CHECKED-AND-CLEAN
More:    result.details() | view='compute' for per-row MACs | docs/reference/summary.md"""


def test_dual_charset_goldens(golden_report) -> None:
    """Byte-exact goldens in BOTH explicit charsets; degrade ties them."""

    assert _mask_volatile(str(golden_report)) == _GOLDEN_ASCII
    assert _mask_volatile(golden_report.render("unicode")) == _GOLDEN_UNICODE
    assert degrade(_GOLDEN_UNICODE) == _GOLDEN_ASCII


def test_degrade_map_holds_byte_for_byte(golden_report) -> None:
    """CI check: ascii_render == degrade(unicode_render, glyph table)."""

    assert str(golden_report) == degrade(golden_report.render("unicode"))


def test_glyph_table_is_one_to_one(golden_report) -> None:
    """Every glyph maps to exactly one ASCII char (widths preserved)."""

    for unicode_char, ascii_char in GLYPH_TABLE.items():
        assert len(unicode_char) == 1
        assert len(ascii_char) == 1
        assert ascii_char.isascii()
    ascii_lines = str(golden_report).split("\n")
    unicode_lines = golden_report.render("unicode").split("\n")
    assert [len(line) for line in ascii_lines] == [len(line) for line in unicode_lines]


def test_no_escape_bytes_ever(golden_report) -> None:
    """A11: no ESC byte in any returned string, any charset."""

    assert "\x1b" not in str(golden_report)
    assert "\x1b" not in golden_report.render("unicode")
    assert "\x1b" not in golden_report.render("html")


def test_str_is_pure_ascii(golden_report) -> None:
    """The canonical payload encodes as ASCII."""

    str(golden_report).encode("ascii")


def test_detection_env_override(monkeypatch) -> None:
    """Rung 2: the environment override wins over everything below."""

    monkeypatch.setenv("TORCHLENS_SUMMARY_STYLE", "unicode")
    assert detect_style(io.StringIO()) == "unicode"
    monkeypatch.setenv("TORCHLENS_SUMMARY_STYLE", "ascii")
    assert detect_style(io.StringIO()) == "ascii"


def test_detection_ci_rung_beats_a_tty(monkeypatch) -> None:
    """Rung 3: CI/GITHUB_ACTIONS resolve ASCII even on a capable pty."""

    monkeypatch.delenv("TORCHLENS_SUMMARY_STYLE", raising=False)
    monkeypatch.setenv("CI", "true")

    class _Tty(io.StringIO):
        """A capable interactive stream."""

        encoding = "utf-8"

        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    assert detect_style(_Tty()) == "ascii"


def test_detection_falls_toward_ascii(monkeypatch) -> None:
    """Non-tty, dumb TERM, and non-round-tripping encodings all -> ASCII."""

    monkeypatch.delenv("TORCHLENS_SUMMARY_STYLE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert detect_style(io.StringIO()) == "ascii"

    class _AsciiTty(io.StringIO):
        """Interactive but with an encoding that cannot carry the glyphs."""

        encoding = "ascii"

        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("TERM", "xterm")
    assert detect_style(_AsciiTty()) == "ascii"

    class _DumbTty(_AsciiTty):
        """TERM=dumb interactive stream."""

        encoding = "utf-8"

    monkeypatch.setenv("TERM", "dumb")
    assert detect_style(_DumbTty()) == "ascii"


def test_detection_affirms_unicode_only_when_verified(monkeypatch) -> None:
    """The ONLY path to unicode: interactive + TERM + verified encoding."""

    monkeypatch.delenv("TORCHLENS_SUMMARY_STYLE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    class _Utf8Tty(io.StringIO):
        """A verified capable interactive stream."""

        encoding = "utf-8"

        def isatty(self) -> bool:
            return True

    assert detect_style(_Utf8Tty()) == "unicode"


def test_print_resolves_at_the_boundary(golden_report, monkeypatch) -> None:
    """print(style='auto') detects the actual sink, not global stdout."""

    monkeypatch.delenv("TORCHLENS_SUMMARY_STYLE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    sink = io.StringIO()
    golden_report.print(file=sink)
    assert sink.getvalue().rstrip("\n") == str(golden_report)  # non-tty -> ASCII
    unicode_sink = io.StringIO()
    golden_report.print(style="unicode", file=unicode_sink)
    assert "─" in unicode_sink.getvalue()
