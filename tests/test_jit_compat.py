"""Regression tests for TorchScript compatibility and model patch idempotence."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
import torchlens.backends.torch.wrappers as torch_wrappers


class _AttentionStyleModule(nn.Module):
    """Small attention-like module that exercises decorated JIT builtin ops."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run a compact scaled dot-product attention block."""

        scale = float(q.size(-1)) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        return torch.matmul(weights, v)


class _StoredFuncModel(nn.Module):
    """Module with a stale torch function stored on the model instance."""

    def __init__(self, stale_relu: Any) -> None:
        """Initialize the module with a detached function reference."""

        super().__init__()
        self.act = stale_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stored callable."""

        return self.act(x)


@pytest.fixture(autouse=True)
def _ensure_wrapped() -> Iterator[None]:
    """Ensure torch wrappers are installed for each test."""

    torch_wrappers.wrap_torch()
    yield


def _require_torch_jit() -> None:
    """Skip when TorchScript APIs are unavailable."""

    if not hasattr(torch, "jit") or not hasattr(torch.jit, "script"):
        pytest.skip("torch.jit.script is unavailable")


def test_attention_style_module_scripts_with_decorated_builtins() -> None:
    """TorchScript should parse an attention-style module after wrapping."""

    _require_torch_jit()

    model = _AttentionStyleModule()
    scripted = torch.jit.script(model)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)

    expected = model(q, k, v)
    observed = scripted(q, k, v)

    assert torch.allclose(observed, expected)


def test_jit_builtin_registration_sanitizes_dtype_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JIT builtin registration should survive wrapper annotation sanitizing."""

    _require_torch_jit()
    import torch.jit._builtins as _jit_builtins

    original_cos = _state._decorated_to_orig[id(torch.cos)]
    builtin_name = _jit_builtins._builtin_table[id(original_cos)]
    monkeypatch.setattr(torch.cos, "__annotations__", {"dtype": "DType", "return": Any})

    torch_wrappers._register_jit_builtin_wrappers()

    assert _jit_builtins._builtin_table[id(torch.cos)] == builtin_name
    assert torch.cos.__annotations__["dtype"] is int
    assert not hasattr(torch.cos, "__wrapped__")


def test_patch_model_instance_guard_skips_second_trace_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracing the same model twice should only do one instance patch walk."""

    torch_wrappers._PATCHED_MODEL_INSTANCES.clear()
    original_relu = _state._decorated_to_orig[id(torch.relu)]
    model = _StoredFuncModel(original_relu)
    walk_count = 0
    original_patch_model_instance = torch_wrappers.patch_model_instance

    def counted_patch_model_instance(model_arg: Any) -> None:
        """Count only calls that will perform the guarded module walk."""

        nonlocal walk_count
        if model_arg is model and model_arg not in torch_wrappers._PATCHED_MODEL_INSTANCES:
            walk_count += 1
        original_patch_model_instance(model_arg)

    monkeypatch.setattr(torch_wrappers, "patch_model_instance", counted_patch_model_instance)

    first_trace = tl.trace(model, torch.randn(4), layers_to_save="all")
    second_trace = tl.trace(model, torch.randn(4), layers_to_save="all")

    assert walk_count == 1
    assert model.act is _state._orig_to_decorated[id(original_relu)]
    assert any("relu" in label.lower() for label in first_trace.layer_labels)
    assert any("relu" in label.lower() for label in second_trace.layer_labels)


def _calls_softsign(x: torch.Tensor) -> torch.Tensor:
    """Call a wrapped pure-Python torch.nn.functional op (not an ATen builtin)."""

    return torch.nn.functional.softsign(x)


def test_jit_script_wrapped_functional_python_op() -> None:
    """TorchScript must compile a function calling a wrapped non-builtin functional op.

    Regression: ``F.softsign`` is a pure-Python functional op absent from torch's jit
    ``_builtin_table``, so torchlens's wrapper is not registered as a builtin. jit then
    pulls softsign's original source but resolves names against the wrapper module's
    globals, which previously lacked the ``torch.overrides`` boilerplate, failing with
    ``undefined value has_torch_function_unary`` (surfaced by spikingjelly
    ``@torch.jit.script`` surrogates in the model menagerie).
    """

    _require_torch_jit()

    scripted = torch.jit.script(_calls_softsign)
    x = torch.randn(8)
    assert torch.allclose(scripted(x), torch.nn.functional.softsign(x))
