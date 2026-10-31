"""Deferred-selector escrow stays bounded for ``tl.module`` saves, and the bound holds everywhere.

Reported case (AGT-2408): ``tl.trace(model, ids, save=tl.module("model.layers.21"))`` on a
9B decoder refused with ``escrow_budget_exceeded`` because a module selector escrowed every
op's output of the whole forward. These tests pin:

1. on a tiny random-init HF decoder, a single-module selector escrows only the ops inside
   that decoder layer, saves a value equal to a forward hook's, and succeeds under a spill
   bound that the whole-pass escrow crosses;
2. the spill bound refuses the same way on every available device (no CPU exemption);
3. the refusal names the components that forced the escrow and the live spelling that
   avoids it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture.session as session_mod
from torchlens._capture.plan import CapturePlan
from torchlens._capture.preflight import _escrow_remedy
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.errors import TorchLensWarning

_LAYER = "model.layers.2"

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tiny_qwen2() -> tuple[nn.Module, torch.Tensor]:
    """Build a seeded 4-layer random-init Qwen2 decoder and a 24-token prompt."""

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen2ForCausalLM"):
        pytest.skip("this transformers has no Qwen2")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.Qwen2ForCausalLM(config).eval(), torch.randint(0, 128, (1, 24))


@contextmanager
def _escrow_copies() -> Iterator[dict[str, int]]:
    """Count escrow copies and their bytes through the one copy helper."""

    counts = {"copies": 0, "bytes": 0}
    real_retain = session_mod.CaptureSession._retain_activation

    def counting_retain(self: Any, raw_index: int, tensor: Any) -> None:
        real_retain(self, raw_index, tensor)
        counts["copies"] += 1
        counts["bytes"] += self.activation_escrow[raw_index].nbytes

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(session_mod.CaptureSession, "_retain_activation", counting_retain)
        yield counts


@contextmanager
def _budgets(*, ram_bytes: int, spill_bytes: int) -> Iterator[None]:
    """Compile every capture plan with the given escrow RAM budget and spill bound."""

    real_compile = CapturePlan.compile.__func__

    def compile_with_budgets(cls: Any, **kwargs: Any) -> Any:
        plan = real_compile(cls, **kwargs)
        profile = dataclasses.replace(
            plan.retention_profile,
            activation_ram_budget_bytes=ram_bytes,
            escrow_spill_budget_bytes=spill_bytes,
        )
        return dataclasses.replace(plan, retention_profile=profile)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CapturePlan, "compile", classmethod(compile_with_budgets))
        yield


def _hook_output(model: nn.Module, ids: torch.Tensor, address: str) -> torch.Tensor:
    """Return a plain forward hook's capture of one module's hidden-state output."""

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: Any, out: Any) -> None:
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    handle = model.get_submodule(address).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return captured["out"]


@pytest.mark.heavy
def test_single_module_selector_escrows_one_decoder_layer():
    """The reported spelling escrows one layer's ops, not the forward, and saves the hook value."""

    model, ids = _tiny_qwen2()
    reference = _hook_output(model, ids, _LAYER)
    full = tl.trace(model, ids)
    with _escrow_copies() as counts, torch.no_grad():
        trace = tl.trace(model, ids, save=tl.module(_LAYER))

    assert torch.equal(trace.modules[_LAYER].out, reference)
    saved = [op.layer_label for op in trace.layer_list if op.has_saved_activation]
    assert len(saved) == len(full.modules[_LAYER]._single_call_or_error().output_ops) == 1
    # One of four decoder layers plus embedding and head: far below a third of all ops.
    assert 0 < counts["copies"] * 3 < len(full.layer_list), counts


@pytest.mark.heavy
def test_module_selector_fits_a_bound_the_whole_pass_escrow_crosses():
    """Under a spill bound sized to one layer, tl.module fits and a whole-pass escrow refuses."""

    model, ids = _tiny_qwen2()
    reference = _hook_output(model, ids, _LAYER)
    with _escrow_copies() as counts, torch.no_grad():
        tl.trace(model, ids, save=tl.module(_LAYER))
    layer_bytes = counts["bytes"]
    full = tl.trace(model, ids)
    whole_pass_bytes = sum(
        op.out.nelement() * op.out.element_size()
        for op in full.layer_list
        if op.has_saved_activation and isinstance(op.out, torch.Tensor)
    )
    spill_bound = 2 * layer_bytes
    assert spill_bound < whole_pass_bytes

    with _budgets(ram_bytes=1, spill_bytes=spill_bound), torch.no_grad():
        with pytest.warns(TorchLensWarning):
            trace = tl.trace(model, ids, save=tl.module(_LAYER))
        assert torch.equal(trace.modules[_LAYER].out, reference)
        with pytest.raises(SaveBudgetExceededError) as excinfo:
            tl.trace(model, ids, save=tl.module(_LAYER) | tl.func("linear"))
    assert excinfo.value.fields["code"] == "escrow_budget_exceeded"
    assert "tl.module terms joined by |" in excinfo.value.fields["remedy"]


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x))) + x


@pytest.mark.parametrize("device", _DEVICES)
def test_escrow_bound_refuses_on_every_device(device: str):
    """The spill bound is byte arithmetic: it refuses alike on CPU and on an accelerator."""

    torch.manual_seed(0)
    model = _Net().to(device).eval()
    x = torch.randn(4, 16, device=device)
    with (
        _budgets(ram_bytes=1, spill_bytes=256),
        torch.no_grad(),
        pytest.raises(SaveBudgetExceededError) as excinfo,
    ):
        tl.trace(model, x, save=tl.module("b") | tl.func("relu"))
    fields = excinfo.value.fields
    assert fields["code"] == "escrow_budget_exceeded"
    assert fields["budget_bytes"] == 256
    assert fields["spilled_bytes"] + fields["pending_bytes"] > 256


@pytest.mark.parametrize("device", _DEVICES)
def test_bounded_module_escrow_fits_the_same_bound_on_every_device(device: str):
    """The same bound admits a tl.module save whose one-pass escrow fits it, on every device."""

    torch.manual_seed(0)
    model = _Net().to(device).eval()
    x = torch.randn(4, 16, device=device)
    full = tl.trace(model, x)
    with (
        _budgets(ram_bytes=1, spill_bytes=256),
        torch.no_grad(),
        pytest.warns(TorchLensWarning),
    ):
        trace = tl.trace(model, x, save=tl.module("b"))
    assert torch.equal(trace.modules["b"].out, full.modules["b"].out)


@pytest.mark.parametrize(
    ("selector", "exit_addresses", "expected"),
    [
        (
            ["add_1_2", 3, "conv1", -1],
            None,
            ["['add_1_2', 3] need FINAL graph numbering", "address_preflight"],
        ),
        (
            tl.module("block") | tl.func("relu"),
            None,
            ["joins tl.module with a non-module term", "tl.module terms joined by |"],
        ),
        (
            tl.module("model.layers.21"),
            ("model.layers.21",),
            [
                "matching passes of model.layers.21",
                "ctx.address == 'model.layers.21'",
                "trace.modules['model.layers.21'].out",
            ],
        ),
        (None, None, ["Remedy: address the same sites with a live spelling"]),
    ],
)
def test_escrow_remedy_names_the_forcing_components(selector, exit_addresses, expected):
    """The refusal text names what forced the escrow and ends with a live spelling."""

    remedy = _escrow_remedy(selector, exit_addresses)
    for fragment in expected:
        assert fragment in remedy
    assert "Remedy: " in remedy
