"""Scale-honesty compat rows: GPTQ/AWQ, CUDA graphs, serving engines (R2-22).

Each row must fire on its condition and stay quiet on a plain model. The
GPTQ/AWQ and serving-engine kernels cannot run on the CPU venue, so the
detections are driven by the same provenance the real packages carry (the
Hugging Face ``quantization_config`` and the defining module namespace).
"""

from __future__ import annotations

import enum
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchlens.compat import report
from torchlens.utils import _torch_compat


class _QuantMethod(str, enum.Enum):
    """Stand-in for transformers' QuantizationMethod enum (a str Enum)."""

    AWQ = "awq"


def _with_config(quantization_config: object) -> nn.Module:
    model = nn.Linear(4, 2)
    model.config = SimpleNamespace(quantization_config=quantization_config)
    return model


def _foreign_module(namespace: str) -> nn.Module:
    cls = type("ForeignLinear", (nn.Linear,), {"__module__": namespace})
    return nn.Sequential(cls(4, 2))


@pytest.mark.parametrize(
    "model",
    [
        _with_config({"quant_method": "gptq", "bits": 4}),
        _with_config(SimpleNamespace(quant_method=_QuantMethod.AWQ)),
        _foreign_module("awq.modules.linear.gemm"),
        _foreign_module("gptqmodel.nn_modules.qlinear.torch"),
    ],
    ids=["hf-config-gptq", "hf-config-awq-enum", "awq-kernel-module", "gptqmodel-module"],
)
def test_gptq_awq_row_fires_and_says_not_verified(model: nn.Module) -> None:
    row = report(model, torch.randn(1, 4)).check("gptq_awq_quantized")
    assert row.is_detected and row.status == "not_tested" and row.severity == "warning"
    assert "NOT verified" in row.details and row.suggestion


def _graphed_like(model: nn.Module) -> nn.Module:
    """Install a forward with make_graphed_callables' exact qualified name."""

    original = model.forward

    def new_fwd(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    new_fwd.__qualname__ = "make_graphed_callables.<locals>.make_graphed_forward.<locals>.new_fwd"
    model.forward = new_fwd
    return model


def test_cuda_graphs_row_fires_on_a_graphed_forward() -> None:
    model = nn.Sequential(_graphed_like(nn.Linear(4, 2)))
    row = report(model, torch.randn(1, 4)).check("cuda_graphs")
    assert row.is_detected and row.status == "not_tested"
    assert "make_graphed_callables" in row.details and "NOT verified" in row.details


def test_cuda_graphs_row_fires_inside_an_active_stream_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_torch_compat, "HAS_CUDA_GRAPHS", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    row = report(nn.Linear(4, 2), torch.randn(1, 4)).check("cuda_graphs")
    assert row.is_detected and "active stream capture" in row.details


def test_cuda_graphs_probe_matches_the_build() -> None:
    assert _torch_compat.HAS_CUDA_GRAPHS is (
        hasattr(torch._C, "_CUDAGraph") and hasattr(torch._C, "_CudaStreamBase")
    )
    assert "HAS_CUDA_GRAPHS" in _torch_compat.get_torch_capability_snapshot()


@pytest.mark.parametrize(
    "namespace", ["vllm.model_executor.models.llama", "sglang.srt.models.llama"]
)
def test_serving_engine_row_is_scope_with_the_hf_twin_recipe(namespace: str) -> None:
    row = report(_foreign_module(namespace), torch.randn(1, 4)).check("serving_engine")
    assert row.is_detected and row.status == "scope"
    assert "from_pretrained" in row.suggestion


def test_scale_rows_stay_quiet_on_a_plain_model() -> None:
    compat = report(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))
    for key in ("gptq_awq_quantized", "cuda_graphs", "serving_engine"):
        row = compat.check(key)
        assert not row.is_detected and row.status == "pass" and row.severity == "ok", key


def test_a_namesake_class_in_user_code_is_not_a_serving_engine() -> None:
    """Provenance, not naming: a user class called VllmLlama does not fire."""

    class VllmLlama(nn.Linear):
        pass

    row = report(nn.Sequential(VllmLlama(4, 2)), torch.randn(1, 4)).check("serving_engine")
    assert not row.is_detected
