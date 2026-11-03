"""External counter pins (A07 row gate): closed forms + gpt2@16 + oracles.

Lane A07 (2026-08-27). External oracles rank ABOVE internal
consistency (summary memo 3.1): summary and profile once agreed with each
other while both +31% wrong. The gpt2 pin is the identity-partition figure
3,974,725,648 (costreport item 24; supersedes the label-sweep 3,974,578,192).
Config-built at the REAL architecture dims -- compute counts are shape-derived,
so the pin runs at zero network.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl

GPT2_TRUE_FORWARD_FLOPS_T16 = 3_974_725_648
GPT2_DECLARED_UNIQUE_PARAMS = 124_439_808
GPT2_PER_PATH_PARAMS = 163_037_184  # wte/lm_head tie counted per path


@pytest.mark.heavy
def test_resnet18_flops_against_fvcore_when_available() -> None:
    """Supplementary cross-tool oracle (fvcore counts MACs as 'flops')."""

    fvcore_analysis = pytest.importorskip("fvcore.nn")
    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    fv_total = int(fvcore_analysis.FlopCountAnalysis(model, x).total())
    log = tl.trace(model, x, save=None)
    try:
        tl_flops = int(log.cost.total_flops_forward)
        # fvcore reports MAC-basis counts and skips most elementwise ops; the
        # FMA-family share dominates, so TL's fma=2 total lands within 10% of
        # 2x fvcore's figure. Divergences beyond that band are real bugs.
        assert abs(tl_flops - 2 * fv_total) / (2 * fv_total) < 0.10
    finally:
        log.cleanup()
