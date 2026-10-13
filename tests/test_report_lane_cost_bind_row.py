"""costreport: the bind row reads "planned / not shipped / not measured", no number."""

from __future__ import annotations

import re

import torch
from torch import nn

from torchlens.report import cost_report
from torchlens.report._cost_perf import COST_REPORT_TIERS


def test_bind_row_is_words_only() -> None:
    result = cost_report(nn.Linear(4, 2), torch.randn(2, 4), repeats=1)
    lines = str(result).splitlines()

    bind_rows = [line for line in lines if line.strip().startswith("bind ")]
    assert len(bind_rows) == 1
    assert bind_rows[0].split(None, 1)[1] == "planned / not shipped / not measured"
    assert not re.search(r"\d", bind_rows[0])
    assert "bind" not in COST_REPORT_TIERS
    assert "bind" not in {tier.tier for tier in result.tiers}
