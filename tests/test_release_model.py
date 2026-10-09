"""Regression tests for releasing persistent model preparation."""

from __future__ import annotations

import io
import pickle
import typing

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import get_module_meta
from torchlens.backends.torch.model_prep import is_forward_call_decorated


class _ReleaseModel(nn.Module):
    """Small nested model whose child receives a persistent forward wrapper."""

    def __init__(self) -> None:
        """Initialize a deterministic linear child."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the child and a functional activation."""
        return torch.relu(self.linear(inputs))


def _new_model() -> _ReleaseModel:
    """Return a deterministically initialized test model."""
    torch.manual_seed(17)
    return _ReleaseModel().eval()


def test_whole_model_pickle_and_torch_save_work_after_trace_and_release() -> None:
    """A capture leaves no child wrapper, so serialization needs no release."""
    model = _new_model()
    inputs = torch.randn(2, 3)
    tl.trace(model, inputs)

    assert "forward" not in model.linear.__dict__
    assert pickle.loads(pickle.dumps(model))(inputs).shape == (2, 2)
    torch.save(model, io.BytesIO())

    tl.release_model(model)

    assert pickle.loads(pickle.dumps(model))(inputs).shape == (2, 2)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    assert buffer.tell() > 0


def test_release_is_idempotent_and_never_traced_model_is_a_noop() -> None:
    """Repeated release and release before tracing leave forward behavior intact."""
    model = _new_model()
    inputs = torch.randn(2, 3)
    expected = model(inputs).detach().clone()

    tl.release_model(model)
    assert torch.equal(model(inputs), expected)

    tl.trace(model, inputs)
    assert not is_forward_call_decorated(model.linear.forward)
    assert get_module_meta(model.linear) is not None
    tl.release_model(model)
    tl.release_model(model)

    assert not is_forward_call_decorated(model.linear.forward)
    assert get_module_meta(model) is None
    assert get_module_meta(model.linear) is None
    assert torch.equal(model(inputs), expected)
    assert all(not name.startswith("tl_") for module in model.modules() for name in vars(module))


def test_retrace_after_release_matches_fresh_model() -> None:
    """A released model is fully re-prepared and captures like a fresh twin."""
    model = _new_model()
    fresh_model = _new_model()
    inputs = torch.randn(2, 3)

    tl.trace(model, inputs)
    tl.release_model(model)
    retraced = tl.trace(model, inputs)
    fresh = tl.trace(fresh_model, inputs)

    assert len(retraced.layer_list) == len(fresh.layer_list)
    assert [op.layer_label for op in retraced.layer_list] == [
        op.layer_label for op in fresh.layer_list
    ]


def test_releasing_one_model_preserves_an_independent_prepared_model() -> None:
    """Release does not disturb the preparation metadata of another model tree."""
    first = _new_model()
    second = _new_model()
    inputs = torch.randn(2, 3)
    tl.trace(first, inputs)
    initial_second = tl.trace(second, inputs)

    tl.release_model(first)
    repeated_second = tl.trace(second, inputs)

    assert get_module_meta(second.linear) is not None
    assert not is_forward_call_decorated(second.linear.forward)
    assert [op.layer_label for op in repeated_second.layer_list] == [
        op.layer_label for op in initial_second.layer_list
    ]


class _HeldActivationModel(nn.Module):
    """Model whose plain attributes hold torch function references."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.act = torch.nn.functional.relu  # whichever epoch is live NOW
        self.extra_acts = [torch.sigmoid, torch.nn.functional.gelu]
        self.act_table = {"tanh": torch.tanh}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.lin(x))
        for act in self.extra_acts:
            y = act(y)
        return self.act_table["tanh"](y)


def test_release_normalizes_prewrap_function_attrs_for_torch_save() -> None:
    """grind-r4 b8 R56 direction 1: a model built BEFORE wrapping holds
    pristine originals; pickled WHILE wrapped, every held ref fails pickle's
    by-reference identity check. release_model must normalize them."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    try:
        model = _HeldActivationModel()  # holds pristine originals
    finally:
        wrap_torch()

    tl.trace(model, torch.randn(1, 4))  # wrappers now live at the public names
    tl.release_model(model)
    torch.save(model, io.BytesIO())
    pickle.dumps(model)
    # The normalized refs still compute the same functions.
    out = model(torch.randn(1, 4))
    assert out.shape == (1, 4)


def test_release_normalizes_wrapper_attrs_after_unwrap() -> None:
    """R56 direction 2: a model built WHILE wrapped holds epoch wrappers;
    after unwrap_torch() those refs fail pickle. release_model normalizes
    them back to the pristine originals."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    wrap_torch()
    model = _HeldActivationModel()  # holds wrap-epoch wrappers
    tl.trace(model, torch.randn(1, 4))
    unwrap_torch()
    try:
        tl.release_model(model)
        torch.save(model, io.BytesIO())
        pickle.dumps(model)
        assert model.act is torch.nn.functional.relu, (
            "the held wrapper was not normalized to the pristine original"
        )
    finally:
        wrap_torch()


def test_release_leaves_foreign_and_user_callables_alone() -> None:
    """The normalization is ledger-fenced: user callables never swap."""

    def user_act(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    model = _HeldActivationModel()
    model.custom = user_act
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)
    assert model.custom is user_act


class _ActPair(typing.NamedTuple):
    """Namedtuple holding torch activation functions (a common config idiom)."""

    act: object
    gate: object


class _TaggedTuple(tuple):
    """Non-namedtuple tuple subclass carrying instance state."""

    def __new__(cls, values: tuple, tag: str = "") -> _TaggedTuple:
        instance = super().__new__(cls, values)
        instance.tag = tag
        return instance


def _prewrap_model(factory) -> nn.Module:
    """Build a model while torch is UNWRAPPED so attrs hold pristine originals."""
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    try:
        return factory()
    finally:
        wrap_torch()


class _NamedTupleModel(nn.Module):
    """Model holding activation functions in a namedtuple attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.acts = _ActPair(act=torch.nn.functional.relu, gate=torch.sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.acts.gate(self.acts.act(self.lin(x)))


class _KeyedTableModel(nn.Module):
    """Model holding a dict keyed by torch functions."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        # Per-activation config table keyed by function identity: pure data,
        # exactly the pickle-failing shape (keys are held pre-wrap originals).
        self.table = {torch.nn.functional.relu: 0.5, torch.nn.functional.gelu: 1.0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _FrozensetModel(nn.Module):
    """Model holding torch functions in a frozenset attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.allowed = frozenset({torch.nn.functional.relu})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def test_release_preserves_namedtuple_attribute_type() -> None:
    """grind-r5 P1 (b3/b8 x3 labs): release_model must not rebuild a namedtuple
    attribute as a plain tuple -- that breaks attribute access and the model's
    own next forward."""

    model = _prewrap_model(_NamedTupleModel)
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)

    assert type(model.acts) is _ActPair, "namedtuple degraded to plain tuple"
    assert model.acts.act is not None  # attribute access must survive
    assert model(torch.randn(1, 4)).shape == (1, 4)  # forward must survive
    torch.save(model, io.BytesIO())  # and the stated goal still holds


def test_release_skips_non_namedtuple_tuple_subclasses() -> None:
    """A tuple subclass with instance state is left untouched (disclosed
    residual) rather than corrupted by a blind rebuild."""

    model = _prewrap_model(_HeldActivationModel)
    tagged = _TaggedTuple((torch.nn.functional.relu,), tag="keep-me")
    model.tagged = tagged
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)

    assert model.tagged is tagged
    assert model.tagged.tag == "keep-me"


def test_release_normalizes_dict_keys_for_torch_save() -> None:
    """grind-r5 P1 (b3 sol / b8 sol): function-KEYED dict attrs are inside the
    documented one-level coverage; keys must normalize or torch.save still
    raises PicklingError after the documented remedy."""

    model = _prewrap_model(_KeyedTableModel)
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)

    torch.save(model, io.BytesIO())
    pickle.dumps(model)
    assert set(model.table.values()) == {0.5, 1.0}


def test_release_normalizes_frozenset_members() -> None:
    """frozenset attrs are one level of builtin nesting too; members must
    normalize so pickle's identity check passes."""

    model = _prewrap_model(_FrozensetModel)
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)

    torch.save(model, io.BytesIO())
    pickle.dumps(model)


@pytest.mark.smoke
def test_release_set_holding_both_epochs_never_loses_a_member() -> None:
    """A set holding BOTH a pristine original and its wrap-epoch counterpart
    must keep its cardinality: the swap is skipped, never silently merged."""
    from torchlens import _state

    model = _prewrap_model(_HeldActivationModel)
    original = model.act  # pristine original captured pre-wrap
    tl.trace(model, torch.randn(1, 4))
    wrapper = _state._orig_to_decorated.get(id(original))
    assert wrapper is not None, "test setup: original must be ledgered"
    model.both = {original, wrapper}
    tl.release_model(model)

    assert len(model.both) == 2, "release_model silently dropped a set member"


def test_release_then_unwrap_keeps_model_serializable() -> None:
    """grind-r5 P1 (b8 opus F1 MED-HIGH): the documented remedy must not be a
    one-way trip. release_model normalizes to the wrapped epoch's values; a
    later unwrap_torch() must re-normalize, not leave the model permanently
    holding a dead wrapper."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    model = _prewrap_model(_HeldActivationModel)
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)
    torch.save(model, io.BytesIO())  # serializable while wrapped

    unwrap_torch()
    try:
        torch.save(model, io.BytesIO())  # and STILL serializable after unwrap
        pickle.dumps(model)
        assert model.act is torch.nn.functional.relu, (
            "held wrapper was not re-normalized to the pristine original at unwrap"
        )
    finally:
        wrap_torch()
    # ... and re-normalized again on the re-wrap flip.
    torch.save(model, io.BytesIO())


def test_faulted_release_evicts_preparation_bookkeeping(monkeypatch) -> None:
    """grind-r5 P1 (b3 opus R07-2): a fault mid-release must not leave a
    half-stripped tree the registry still certifies as prepared -- the next
    capture must re-prepare from scratch and emit full module containment."""

    from torchlens.backends.torch import model_prep

    model = _new_model()
    inputs = torch.randn(2, 3)
    tl.trace(model, inputs)

    real_clear_meta = model_prep.clear_meta
    calls = {"n": 0}

    def _faulting_clear_meta(module):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected fault mid-release")
        real_clear_meta(module)

    monkeypatch.setattr(model_prep, "clear_meta", _faulting_clear_meta)
    with pytest.raises(RuntimeError, match="injected fault mid-release"):
        tl.release_model(model)
    monkeypatch.undo()

    # The half-released tree must NOT be certified prepared any more ...
    from torchlens import _state

    assert model not in _state._prepared_models

    # ... so the next capture re-prepares from scratch with full containment.
    control = tl.trace(_new_model(), inputs)
    retraced = tl.trace(model, inputs)
    assert [op.layer_label for op in retraced.layer_list] == [
        op.layer_label for op in control.layer_list
    ]
    control_modules = [tuple(control[label].modules) for label in ("linear_1_1", "relu_1_2")]
    retraced_modules = [tuple(retraced[label].modules) for label in ("linear_1_1", "relu_1_2")]
    assert retraced_modules == control_modules


def test_release_leaves_no_instance_forward_on_plain_modules() -> None:
    """grind-r5 P1 (b4 fable): restoring forward as an INSTANCE attribute
    churns the implementation fingerprint; when the original is the plain
    class method the instance override must be dropped entirely."""

    model = _new_model()
    tl.trace(model, torch.randn(2, 3))
    tl.release_model(model)

    for module in model.modules():
        assert "forward" not in module.__dict__, (
            f"{type(module).__name__} kept an instance-level forward after release"
        )
