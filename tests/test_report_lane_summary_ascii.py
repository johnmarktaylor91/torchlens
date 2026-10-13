"""summary D1: ``str`` of a summary is always ASCII; the structured data keeps names."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.report._summary_charset import degrade

_NAME = "café_προβολή"


def _named_model() -> nn.Module:
    model = nn.Sequential()
    model.add_module(_NAME, nn.Linear(4, 3))
    model.add_module("act", nn.ReLU())
    return model


def test_summary_str_is_ascii_for_a_non_ascii_module_name() -> None:
    report = tl.summary(_named_model(), torch.randn(2, 4))

    text = str(report)
    assert text.isascii()
    assert "cafe_" in text
    # The structured data keeps the real name: the unicode render and the table.
    assert _NAME in report.render("unicode")
    frame = report.to_pandas()
    assert any(_NAME in cell for cell in frame.astype(str).to_numpy().ravel())


def test_degrade_keeps_width_and_alignment() -> None:
    line = f"| {_NAME} | 15 |"
    ascii_line = degrade(line)
    assert ascii_line.isascii()
    assert len(ascii_line) == len(line)
    assert ascii_line == "| cafe_??????? | 15 |"
    assert degrade("plain ascii ─ × █ ░") == "plain ascii - x # ."
