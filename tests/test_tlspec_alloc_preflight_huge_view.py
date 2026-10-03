"""r57 C3 over-trigger guard for a genuinely huge PURE VIEW, in the ``serial`` tier.

Split out of ``test_tlspec_alloc_preflight_fake.py`` (smoke) because it is
memory-heavy by construction, not slow: the default capture retains a contiguous
copy of the 4e8-element ``expand`` (1.49 GiB) and the run materializes another,
so one process peaks near 3.5 GB RSS. Under four xdist workers on a 16 GB CI
runner the host's ``MemAvailable`` fell to 1.38 GB, and the default
``save_budget="auto"`` (50% of available) correctly refused the 1.49 GiB copy
before allocating it. The test is unchanged; it runs away from parallel worker
load (the ``serial`` CI steps run without xdist).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl

pytestmark = pytest.mark.serial

_CAPTURE = {"capture": tl.options.CaptureOptions(intervention_ready=True)}


class _HugeView(nn.Module):
    """A genuinely huge PURE VIEW (``expand`` of a size-1 dim) that allocates nothing.

    Its logical numel is enormous, but storage aliases the input, so the alloc
    preflight must charge ZERO new bytes and never over-refuse it (r51 anti-pattern).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**8, 4)  # 4e8-element view, no allocation in the forward
        return wide[0].sum() + x.sum()


def test_genuinely_huge_view_model_runs_verified(tmp_path: Path) -> None:
    """A model whose real forward produces a HUGE view (4e8-element ``expand``) runs
    VERIFIED -- the alloc preflight charges 0 new bytes for it (r51 over-catch avoided)."""

    x = torch.randn(1, 4)
    trace = tl.trace(_HugeView().eval(), x, **_CAPTURE)
    bundle = tmp_path / "huge_view.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
