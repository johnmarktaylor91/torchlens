"""Soundness of the descriptive family split: resolvers on recorded inputs give call-time values.

The descriptive fields, the function config (salient args) and the forward FLOPs are
each a per-function constant part plus a resolver. These tests spy on the resolvers
during real captures, keep the recorded inputs (tensors replaced by their shape and
dtype, containers copied), free the model and mutate the caller-owned inputs, and
then run each resolver again on the recorded inputs: every value must equal the one
the call stored.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens._capture import flops as flops_mod, salient_args as salient_mod
from torchlens._capture._recorded_tensor import (
    RecordedTensor,
    UnrecordableArgument,
    record_call_values,
)
from torchlens.backends.torch import _ops_descriptive_fields as desc_mod


class _Corpus(nn.Module):
    """One forward that reaches every salient extractor family and every FLOPs tier."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.deconv = nn.ConvTranspose2d(4, 4, 3, stride=2, output_padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.gn = nn.GroupNorm(2, 4)
        self.inorm = nn.InstanceNorm2d(4)
        self.ln = nn.LayerNorm(8)
        self.lin = nn.Linear(8, 8)
        self.emb = nn.Embedding(10, 8, padding_idx=0)
        self.lstm = nn.LSTM(8, 8, batch_first=True)
        self.gru_cell = nn.GRUCell(8, 8)
        self.size = [6, 6]
        self.dims = [0, 2, 1]

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Run the corpus."""

        h = self.conv(x)
        h = self.deconv(h)
        h = self.inorm(self.gn(self.bn(h)))
        h = F.max_pool2d(h, 2, stride=1)
        h = F.avg_pool2d(h, 2)
        h = F.adaptive_avg_pool2d(h, (2, 2)) + F.adaptive_max_pool2d(h, 2)
        h = F.interpolate(h, size=self.size, mode="nearest")
        h = F.pad(h, (1, 1), mode="constant", value=0.5)
        h = F.leaky_relu(h, 0.2) + F.elu(h) + F.hardtanh(h, -2.0, 2.0) + F.threshold(h, 0.1, 0.0)
        h = torch.clamp(h, min=-1.0, max=1.0)
        flat = h.flatten(1)[:, :8]
        e = self.emb(tokens)
        seq = self.ln(e + flat.unsqueeze(1))
        seq = F.dropout(seq, 0.0, training=False)
        att = F.scaled_dot_product_attention(seq, seq, seq, is_causal=True)
        out, _ = self.lstm(att)
        cell = self.gru_cell(out[:, -1], out[:, 0])
        mixed = torch.einsum("bij,bkj->bik", out, out)
        mixed = torch.bmm(mixed, out) + torch.matmul(out, self.lin.weight)
        mixed = torch.addmm(cell, cell, self.lin.weight.t()).unsqueeze(1) + mixed
        mixed = torch.cat([mixed, mixed], dim=1)
        mixed = torch.stack([mixed, mixed], dim=0).sum(dim=0, keepdim=True)[0]
        mixed = mixed.permute(*self.dims).transpose(1, 2)
        red = mixed.mean(dim=1) + mixed.amax(dim=1) + mixed.max(dim=1).values
        red = F.softmax(red, dim=-1) + F.log_softmax(red, dim=-1)
        return self.lin(F.linear(red, self.lin.weight, bias=self.lin.bias))


def _patch_everywhere(original: Callable[..., Any], spy: Callable[..., Any]) -> list[tuple]:
    """Point every torchlens module attribute holding ``original`` at ``spy``."""

    import sys

    patched = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "") or ""
        if not name.startswith("torchlens"):
            continue
        for attr, value in list(vars(module).items()):
            if value is original:
                setattr(module, attr, spy)
                patched.append((module, attr, original))
    return patched


@pytest.fixture
def spied() -> Iterator[dict[str, list[tuple]]]:
    """Record each resolver's recorded inputs and call-time result during captures."""

    seen: dict[str, list[tuple]] = {"salient": [], "flops": [], "descriptive": []}
    orig_salient = salient_mod.extract_salient_args
    orig_flops = flops_mod.compute_forward_flops
    orig_set = desc_mod._set_descriptive_fields

    def salient_spy(layer_type, func_name, args, kwargs, param_shapes=None):
        result = orig_salient(layer_type, func_name, args, kwargs, param_shapes)
        if salient_mod.salient_reads_args(layer_type):
            rec = record_call_values(args, kwargs)
            shapes = None if param_shapes is None else list(param_shapes)
            seen["salient"].append(((layer_type, func_name, *rec, shapes), result))
        return result

    def flops_spy(func_name, output_shape, param_shapes, args, kwargs):
        result = orig_flops(func_name, output_shape, param_shapes, args, kwargs)
        rec = record_call_values(args, kwargs)
        shapes = list(param_shapes or [])
        seen["flops"].append(((func_name, output_shape, shapes, *rec), result))
        return result

    def set_spy(fields_dict, func, func_name):
        orig_set(fields_dict, func, func_name)
        family = desc_mod.observe_descriptive(func, func_name)
        written = tuple(fields_dict[key] for key in desc_mod.DESCRIPTIVE_FIELDS)
        seen["descriptive"].append(((family, func_name), written))

    patched = (
        _patch_everywhere(orig_salient, salient_spy)
        + _patch_everywhere(orig_flops, flops_spy)
        + _patch_everywhere(orig_set, set_spy)
    )
    try:
        yield seen
    finally:
        for module, attr, original in patched:
            setattr(module, attr, original)


def _holds_live_tensor(value: Any) -> bool:
    """Whether a recorded input still reaches a live tensor."""

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_holds_live_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_live_tensor(item) for item in value)
    return False


def test_resolvers_on_recorded_inputs_give_the_call_time_values(spied) -> None:
    """After the forward, model freed and inputs mutated, every resolver repeats its value."""

    torch.manual_seed(0)
    model = _Corpus().eval()
    x = torch.randn(2, 3, 8, 8)
    tokens = torch.tensor([[1, 2, 3]])
    trace = tl.trace(model, (x, tokens))
    stored_flops = sorted(
        (op.func_name, op.cost.flops_forward)
        for op in trace._layer_list
        if op.cost.flops_forward is not None
    )
    # Mutate and free everything the call saw.
    model.size[0] = 99
    model.dims.reverse()
    x.resize_(0)
    tokens.zero_()
    del model, trace
    gc.collect()

    assert spied["flops"] and spied["salient"] and spied["descriptive"]
    for inputs, value in spied["flops"]:
        assert not _holds_live_tensor(inputs)
        assert flops_mod.compute_forward_flops(*inputs) == value, inputs[0]
    for inputs, value in spied["salient"]:
        assert not _holds_live_tensor(inputs)
        assert salient_mod.extract_salient_args(*inputs) == value, inputs[0]
    for (family, func_name), value in spied["descriptive"]:
        again = desc_mod.resolve_descriptive(family, func_name)
        assert again == value, func_name

    called = sorted((inputs[0], value) for inputs, value in spied["flops"] if value is not None)
    assert set(stored_flops) <= set(called)
    tiers = {flops_mod.forward_flops_plan(inputs[0])[0] for inputs, _ in spied["flops"]}
    assert {flops_mod.TIER_ZERO, flops_mod.TIER_ELEMENTWISE, flops_mod.TIER_SPECIALTY} <= tiers
    layer_types = {inputs[0] for inputs, _ in spied["salient"]}
    assert len(layer_types) >= 12, sorted(layer_types)
    nonempty = [value for _, value in spied["salient"] if value]
    assert len(nonempty) >= 10


def test_record_call_values_holds_no_tensor_and_copies_containers() -> None:
    """Tensors become shape and dtype; caller-owned containers are copied."""

    weight = torch.zeros(4, 3, dtype=torch.float16)
    size = [2, 3]
    args, kwargs = record_call_values(
        (weight, size, (weight, 1)), {"w": [weight], "opts": {"k": size}, "s": "x"}
    )
    assert args[0] == RecordedTensor(weight) and args[0].shape == (4, 3)
    assert args[0].dtype is torch.float16
    assert isinstance(args[2], tuple) and args[2][1] == 1
    size[0] = 7
    assert args[1] == [2, 3] and kwargs["opts"] == {"k": [2, 3]}
    assert not _holds_live_tensor((args, kwargs))
    with pytest.raises(UnrecordableArgument):
        record_call_values(({weight},), {})


@pytest.mark.parametrize(
    ("func_name", "args", "kwargs"),
    [
        ("linear", (torch.zeros(2, 8), torch.zeros(4, 8), torch.zeros(4)), {}),
        ("linear", (torch.zeros(2, 8), torch.zeros(4, 8)), {"bias": torch.zeros(4)}),
        ("conv2d", (torch.zeros(1, 3, 8, 8), torch.zeros(4, 3, 3, 3)), {"stride": 2}),
        ("matmul", (torch.zeros(5), torch.zeros(5, 2)), {}),
        ("einsum", ("ij,jk->ik", torch.zeros(2, 3), torch.zeros(3, 4)), {}),
        ("multi_dot", ([torch.zeros(2, 3), torch.zeros(3, 4), torch.zeros(4, 1)],), {}),
    ],
)
def test_forward_flops_specialty_rules_agree_on_recorded_arguments(func_name, args, kwargs):
    """Each shape rule gives the same count on live and recorded arguments."""

    live = flops_mod.compute_forward_flops(func_name, None, [], args, kwargs)
    recorded = flops_mod.compute_forward_flops(
        func_name, None, [], *record_call_values(args, kwargs)
    )
    assert recorded == live
    assert flops_mod.forward_flops_plan(func_name)[0] == flops_mod.TIER_SPECIALTY


def test_forward_flops_plan_reads_custom_rules_live() -> None:
    """A rule registered after a first call applies to the next one (no stale constant)."""

    name = "__fd_probe_op__"
    assert flops_mod.forward_flops_plan(name)[0] == flops_mod.TIER_UNKNOWN
    flops_mod._CUSTOM_OP_RULES[name] = (lambda *_a: 11, None)
    try:
        assert flops_mod.forward_flops_plan(name)[0] == flops_mod.TIER_CUSTOM
        assert flops_mod.compute_forward_flops(name, (2,), [], (), {}) == 11
    finally:
        del flops_mod._CUSTOM_OP_RULES[name]


def test_forward_flops_plan_matches_the_inline_dispatch_for_every_table_entry() -> None:
    """The plan names the tier the call's dispatch takes, for every name in every table."""

    names = (
        set(flops_mod.ZERO_FLOPS_OPS)
        | set(flops_mod.ZERO_ARITHMETIC_CONSTRUCTIONS)
        | set(flops_mod.ELEMENTWISE_FLOPS)
        | set(flops_mod.SPECIALTY_HANDLERS)
    )
    for name in sorted(names):
        tier, payload = flops_mod.forward_flops_plan(name)
        assert tier in (flops_mod.TIER_ZERO, flops_mod.TIER_ELEMENTWISE, flops_mod.TIER_SPECIALTY)
        if tier == flops_mod.TIER_ZERO:
            assert flops_mod.compute_forward_flops(name, (2, 3), [], (), {}) == 0, name
        elif tier == flops_mod.TIER_ELEMENTWISE:
            assert flops_mod.compute_forward_flops(name, (2, 3), [], (), {}) == payload * 6, name
            assert flops_mod.compute_forward_flops(name, None, [], (), {}) is None, name
        else:
            assert payload is flops_mod.SPECIALTY_HANDLERS[name], name
    assert flops_mod.forward_flops_plan(None) == (flops_mod.TIER_UNKNOWN, None)
    assert flops_mod.compute_forward_flops(None, (2,), [], (), {}) is None
