"""summary D1: ``str`` of a summary is always ASCII; the structured data keeps names."""

from __future__ import annotations

from torchlens.report._summary_charset import degrade

_NAME = "café_προβολή"


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_summary_charset.py.
def test_degrade_keeps_width_and_alignment() -> None:
    line = f"| {_NAME} | 15 |"
    ascii_line = degrade(line)
    assert ascii_line.isascii()
    assert len(ascii_line) == len(line)
    assert ascii_line == "| cafe_??????? | 15 |"
    assert degrade("plain ascii ─ × █ ░") == "plain ascii - x # ."
