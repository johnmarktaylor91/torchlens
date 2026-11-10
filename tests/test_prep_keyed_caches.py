"""Preparation facts kept between traces are reused only while their keys hold.

Two caches survive from one trace of a model to the next: the parameter plan
(each parameter's address, owner, shape, dtype, size and alias addresses) and
the stateful-module scope facts. Each is checked against the live model before
use. These tests change exactly one key between two traces of one model and
show the second trace sees the change; show that parameter and buffer values
changed between traces are read fresh (values are never keys); show that a
replayed plan builds the records a full walk builds; and show that neither
cache keeps a dropped model or its tensors alive.
"""

from __future__ import annotations

import gc
import types
import weakref
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_label import get_param_meta
from torchlens.backends.torch import _param_prep_plan as plan_mod, _scope_facts_cache as facts_mod
from torchlens.backends.torch._module_state import stateful_modules

_FIELDS = (
    "address",
    "module_address",
    "name",
    "shape",
    "dtype",
    "num_params",
    "param_memory",
    "is_trainable",
    "all_addresses",
    "all_module_addresses",
    "has_optimizer",
)


class _Block(nn.Module):
    """A linear layer, a scale, a buffer and a stateless activation."""

    offset: torch.Tensor

    def __init__(self) -> None:
        """Build the block."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.ones(4))
        self.register_buffer("offset", torch.zeros(4))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale, shift and activate."""

        return self.act(self.fc(x) * self.scale + self.offset)


class _Model(nn.Module):
    """Two blocks, the second sharing the first's linear layer under a second name."""

    def __init__(self) -> None:
        """Build the model."""

        super().__init__()
        self.a = _Block()
        self.b = _Block()
        self.b.tied = self.a.fc  # a shared module: its parameters gain alias addresses
        self.b.gain = self.a.scale  # a tied parameter: one record, two addresses
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks and the head."""

        return self.head(self.b(self.a(x)))


def _fields(trace: Any) -> dict[str, tuple[Any, ...]]:
    """Return every parameter record's planned fields, by address."""

    return {
        param.address: tuple(
            list(getattr(param, name)) if name.startswith("all_") else getattr(param, name)
            for name in _FIELDS
        )
        for param in trace.params
    }


def _addresses(trace: Any) -> set[str]:
    """Return the parameter addresses a trace recorded."""

    return {param.address for param in trace.params}


def _model() -> tuple[_Model, torch.Tensor]:
    """Return a seeded model in eval mode and an input."""

    torch.manual_seed(0)
    return _Model().eval(), torch.randn(3, 4)


# ---------------------------------------------------------------------------
# The parameter plan: reuse and equality with a full walk
# ---------------------------------------------------------------------------


def test_second_trace_reuses_the_plan_and_builds_the_same_records() -> None:
    """A replayed plan gives the records a full walk gives, aliases included."""

    model, x = _model()
    first = tl.trace(model, x)
    plan = plan_mod._PLANS.get(model)
    assert plan is not None
    second = tl.trace(model, x)
    assert plan_mod._PLANS.get(model) is plan  # reused, not rebuilt
    assert _fields(second) == _fields(first)
    aliased = second.params["a.fc.weight"]
    assert "b.tied.weight" in aliased.all_addresses
    assert "b.tied" in aliased.all_module_addresses
    # Records and their lists are each trace's own.
    assert second.params["a.fc.weight"] is not first.params["a.fc.weight"]
    assert (
        second.params["a.fc.weight"].all_addresses is not first.params["a.fc.weight"].all_addresses
    )
    assert "b.gain" in second.params["a.scale"].all_addresses
    # Session end left no stamp behind.
    assert all(get_param_meta(param) is None for param in model.parameters())

    plan_mod._PLANS.pop(model, None)
    full = tl.trace(model, x)
    assert _fields(full) == _fields(second)


def test_replayed_plan_draws_the_barcodes_a_full_walk_draws() -> None:
    """With the same seed, a replay and a full walk stamp the same barcodes."""

    model, x = _model()
    seeded = tl.options.CaptureOptions(seed=5)
    tl.trace(model, x)
    assert plan_mod._PLANS.get(model) is not None
    replay_trace = tl.trace(model, x, capture=seeded)
    replayed = {p.address: p.barcode for p in replay_trace.params}
    plan_mod._PLANS.pop(model, None)
    full_trace = tl.trace(model, x, capture=seeded)
    full = {p.address: p.barcode for p in full_trace.params}
    assert replayed == full


# ---------------------------------------------------------------------------
# The parameter plan: one key changed between two traces
# ---------------------------------------------------------------------------


def test_swapped_parameter_object_is_seen() -> None:
    """Key: parameter identity."""

    model, x = _model()
    tl.trace(model, x)
    new = nn.Parameter(torch.full((4,), 2.0))
    model.a.scale = new
    second = tl.trace(model, x)
    assert torch.equal(second.params["a.scale"].value, new.detach())
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))


def test_changed_dtype_is_seen() -> None:
    """Key: dtype (the same parameter objects, converted in place)."""

    model, x = _model()
    tl.trace(model, x)
    model.double()
    second = tl.trace(model, x.double())
    assert second.params["a.fc.weight"].dtype is torch.float64
    assert second.params["a.fc.weight"].param_memory == 16 * 8


def test_changed_shape_is_seen() -> None:
    """Key: shape (the same parameter object, new data)."""

    model, x = _model()
    tl.trace(model, x)
    param = model.a.scale
    param.data = torch.ones(1)  # still broadcasts in the forward
    second = tl.trace(model, x)
    assert model.a.scale is param
    assert tuple(second.params["a.scale"].shape) == (1,)
    assert second.params["a.scale"].num_params == 1
    assert second.params["a.scale"].param_memory == 4


def test_flipped_requires_grad_is_seen() -> None:
    """Key: requires_grad."""

    model, x = _model()
    first = tl.trace(model, x)
    assert first.params["head.bias"].is_trainable is True
    model.head.bias.requires_grad_(False)
    second = tl.trace(model, x)
    assert second.params["head.bias"].is_trainable is False
    assert model.head.bias.requires_grad is False
    model.head.bias.requires_grad_(True)
    third = tl.trace(model, x)
    assert third.params["head.bias"].is_trainable is True


def test_parameter_slot_set_to_none_added_or_removed_is_seen() -> None:
    """Key: the parameter slots of each module (None, added, removed)."""

    model, x = _model()
    tl.trace(model, x)
    model.head.bias = None  # type: ignore[assignment]
    second = tl.trace(model, x)
    assert "head.bias" not in _addresses(second)
    model.a.act.register_parameter("gain", nn.Parameter(torch.ones(1)))
    third = tl.trace(model, x)
    assert "a.act.gain" in _addresses(third)
    del model.a.act._parameters["gain"]
    fourth = tl.trace(model, x)
    assert "a.act.gain" not in _addresses(fourth)


def test_submodule_added_removed_or_replaced_is_seen() -> None:
    """Key: the module tree (add, remove and replace a submodule)."""

    model, x = _model()
    tl.trace(model, x)
    model.extra = nn.Linear(1, 1)
    second = tl.trace(model, x)
    assert {"extra.weight", "extra.bias"} <= _addresses(second)
    del model.extra
    third = tl.trace(model, x)
    assert "extra.weight" not in _addresses(third)
    old = model.head
    model.head = nn.Linear(4, 2)
    fourth = tl.trace(model, x)
    assert fourth.params["head.weight"].value is not None
    assert torch.equal(fourth.params["head.weight"].value, model.head.weight.detach())
    assert not torch.equal(model.head.weight, old.weight)


def test_rebound_forward_is_seen() -> None:
    """A forward rebound on an instance between traces runs in the next trace."""

    model, x = _model()
    first = tl.trace(model, x)

    def doubled(self: _Block, inp: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(inp) * self.scale + self.offset) * 2

    model.b.forward = types.MethodType(doubled, model.b)  # type: ignore[method-assign]
    second = tl.trace(model, x)
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))
    assert len(second) > len(first)
    del model.b.forward
    third = tl.trace(model, x)
    assert len(third) == len(first)


# ---------------------------------------------------------------------------
# Values are never keys
# ---------------------------------------------------------------------------


def test_parameter_and_buffer_values_changed_between_traces_are_read_fresh() -> None:
    """In-place changes to parameter and buffer data reach the next trace."""

    model, x = _model()
    tl.trace(model, x)
    with torch.no_grad():
        model.a.fc.weight.fill_(0.5)
        model.b.offset.fill_(3.0)
    second = tl.trace(model, x)
    assert torch.equal(second.params["a.fc.weight"].value, torch.full((4, 4), 0.5))
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))
    with torch.no_grad():
        model.b.offset.fill_(-1.0)
    third = tl.trace(model, x)
    (output_op,) = third.output_ops
    assert torch.equal(output_op.out, model(x))
    assert not torch.equal(third.output_ops[0].out, second.output_ops[0].out)


# ---------------------------------------------------------------------------
# Scope facts: tree and own-state bits
# ---------------------------------------------------------------------------


def _reference_facts(model: nn.Module) -> tuple[frozenset[int], frozenset[str]]:
    """Decide the scope facts without the cache."""

    return facts_mod._decide(model)[4], facts_mod._decide(model)[5]


def test_scope_facts_follow_own_state_and_tree_changes() -> None:
    """Keys: the tree and each module's own-state bit (parameter or buffer, None or not)."""

    model, x = _model()
    tl.trace(model, x)
    ids, addresses = facts_mod.cached_scope_facts(model)
    assert (ids, addresses) == _reference_facts(model)
    assert "a.act" not in addresses
    assert facts_mod._FACTS.get(model) is not None
    cached = facts_mod._FACTS[model]
    assert facts_mod.cached_scope_facts(model) == (ids, addresses)
    assert facts_mod._FACTS[model] is cached  # reused

    model.a.act.register_buffer("count", torch.zeros(1))
    tl.trace(model, x)
    after = facts_mod.cached_scope_facts(model)
    assert "a.act" in after[1] and after == _reference_facts(model)

    model.a.act.count = None  # type: ignore[assignment]  # a None buffer owns no state
    assert "a.act" not in facts_mod.cached_scope_facts(model)[1]
    model.a.act.register_parameter("gain", nn.Parameter(torch.ones(1)))
    assert "a.act" in facts_mod.cached_scope_facts(model)[1]

    model.c = nn.Sequential(nn.ReLU(), nn.Linear(4, 4))
    tl.trace(model, x)
    facts = facts_mod.cached_scope_facts(model)
    assert facts == _reference_facts(model)
    assert set(stateful_modules(model)) == set(facts[0])


# ---------------------------------------------------------------------------
# Nothing kept alive
# ---------------------------------------------------------------------------


def test_caches_keep_no_dropped_model_or_tensor_alive() -> None:
    """A model, its parameters and its buffers die once the user drops them and the trace."""

    model, x = _model()
    tl.trace(model, x)
    tl.trace(model, x)
    assert plan_mod._PLANS.get(model) is not None
    assert facts_mod._FACTS.get(model) is not None
    swapped = weakref.ref(model.head.bias)
    model.head.bias = nn.Parameter(torch.ones(2))  # dropped while the plan still names it
    gc.collect()
    assert swapped() is None
    trace = tl.trace(model, x)
    model_ref = weakref.ref(model)
    param_refs = [weakref.ref(param) for param in model.parameters()]
    buffer_refs = [weakref.ref(buffer) for buffer in model.buffers()]
    del trace, model
    gc.collect()
    assert model_ref() is None
    assert all(ref() is None for ref in param_refs + buffer_refs)
    assert all(key is not None for key in plan_mod._PLANS.keys())
    assert all(key is not None for key in facts_mod._FACTS.keys())
