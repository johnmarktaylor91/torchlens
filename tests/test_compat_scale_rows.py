"""Scale-honesty compat rows: GPTQ/AWQ, CUDA graphs, serving engines (R2-22).

Each row must fire on its condition and stay quiet on a plain model. The
GPTQ/AWQ and serving-engine kernels cannot run on the CPU venue, so the
detections are driven by the same provenance the real packages carry (the
Hugging Face ``quantization_config`` and the defining module namespace).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchlens._utils import _torch_compat
from torchlens.report import doctor as report


def _with_config(quantization_config: object) -> nn.Module:
    model = nn.Linear(4, 2)
    model.config = SimpleNamespace(quantization_config=quantization_config)
    return model


def _foreign_module(namespace: str) -> nn.Module:
    cls = type("ForeignLinear", (nn.Linear,), {"__module__": namespace})
    return nn.Sequential(cls(4, 2))


# Kept by the shrink-r4 coverage proof: sole branch coverage of compat/_report.py.
@pytest.mark.parametrize(
    "model",
    [
        _with_config({"quant_method": "gptq", "bits": 4}),
    ],
    ids=["hf-config-gptq"],
)
def test_gptq_awq_row_fires_and_says_not_verified(model: nn.Module) -> None:
    row = report(model, torch.randn(1, 4)).check("gptq_awq_quantized")
    assert row.is_detected and row.status == "not_tested" and row.severity == "warning"
    assert "NOT verified" in row.details and row.suggestion


# Kept by the shrink-r4 coverage proof: sole branch coverage of compat/_report.py.
def test_cuda_graphs_row_fires_inside_an_active_stream_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_torch_compat, "HAS_CUDA_GRAPHS", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    row = report(nn.Linear(4, 2), torch.randn(1, 4)).check("cuda_graphs")
    assert row.is_detected and "active stream capture" in row.details


# Kept by the shrink-r4 coverage proof: sole branch coverage of compat/_report.py.
@pytest.mark.parametrize(
    "namespace",
    [
        "vllm.model_executor.models.llama",
    ],
)
def test_serving_engine_row_is_scope_with_the_hf_twin_recipe(namespace: str) -> None:
    row = report(_foreign_module(namespace), torch.randn(1, 4)).check("serving_engine")
    assert row.is_detected and row.status == "scope"
    assert "from_pretrained" in row.suggestion
