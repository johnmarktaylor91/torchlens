"""External counter pins (A07 row gate): closed forms + gpt2@16 + oracles.

Lane A07 (megasprint 2026-08-27). External oracles rank ABOVE internal
consistency (summary memo 3.1): summary and profile once agreed with each
other while both +31% wrong. The gpt2 pin is the identity-partition figure
3,974,725,648 (costreport item 24; supersedes the label-sweep 3,974,578,192).
Config-built at the REAL architecture dims -- compute counts are shape-derived,
so the pin runs at zero network.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

GPT2_TRUE_FORWARD_FLOPS_T16 = 3_974_725_648
GPT2_DECLARED_UNIQUE_PARAMS = 124_439_808
GPT2_PER_PATH_PARAMS = 163_037_184  # wte/lm_head tie counted per path


@pytest.mark.smoke
def test_one_linear_closed_form_pins() -> None:
    """one-Linear(8,16,bias) batch 2: 544 FLOPs / 192 B tracked; alias owns 0."""

    model = nn.Linear(8, 16, bias=True)
    log = tl.trace(model, torch.randn(2, 8), capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        assert int(log.total_flops_forward) == 544  # 2*2*8*16 + 2*16 bias adds
        assert int(log.total_activation_memory) == 192  # 64 input + 128 output
        output_rows = [op for op in log.layer_list if op.is_output]
        assert output_rows and all(op.flops_forward is None for op in output_rows)
    finally:
        log.cleanup()


@pytest.mark.heavy
def test_gpt2_at_16_tokens_external_pins() -> None:
    """gpt2@16: 124,439,808 declared-unique params; 3,974,725,648 forward FLOPs."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config())
    model.eval()
    token_ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        log = tl.trace(model, token_ids, capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        assert log.num_params == GPT2_DECLARED_UNIQUE_PARAMS
        assert log.num_params == sum(p.numel() for p in model.parameters())
        assert log.num_params_by_path == GPT2_PER_PATH_PARAMS
        assert log.tied_param_groups, "the wte/lm_head tie must be detected and named"
        assert int(log.total_flops_forward) == GPT2_TRUE_FORWARD_FLOPS_T16
        # The alias overstatement is dead: the old printed figure is unreachable.
        assert int(log.total_flops_forward) != 5_209_841_680
    finally:
        log.cleanup()


@pytest.mark.heavy
def test_resnet18_flops_against_fvcore_when_available() -> None:
    """Supplementary cross-tool oracle (fvcore counts MACs as 'flops')."""

    fvcore_analysis = pytest.importorskip("fvcore.nn")
    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    fv_total = int(fvcore_analysis.FlopCountAnalysis(model, x).total())
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        tl_flops = int(log.total_flops_forward)
        # fvcore reports MAC-basis counts and skips most elementwise ops; the
        # FMA-family share dominates, so TL's fma=2 total lands within 10% of
        # 2x fvcore's figure. Divergences beyond that band are real bugs.
        assert abs(tl_flops - 2 * fv_total) / (2 * fv_total) < 0.10
    finally:
        log.cleanup()
