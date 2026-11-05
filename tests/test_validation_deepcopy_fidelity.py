"""Validation deepcopy fidelity and plain-attribute fallback coverage.

Rung-1 menagerie findings (2026-10-04): FreeV's ``AttrDict`` (``self.__dict__ =
self``) deep-copies into an object with an empty attribute namespace, so the
ground-truth forward on the copy raised ``AttributeError``; WavTokenizer's legacy
``weight_norm`` makes deepcopy raise, and the fallback refused a NumPy ``int64``
``hop_length`` and ``nn.LSTM._flat_weights`` (a list aliasing registered
parameters). These tests pin the fixes and their narrowness.
"""

from __future__ import annotations

import copy
import io
import threading
import types
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens._user_public_impls as user_public_impls
from torchlens._capture_state_helpers import (
    _ModuleTreePlainAttrSnapshot,
)
from torchlens._plain_attr_fidelity import (
    assert_copy_kept_instance_attrs,
    deepcopy_for_validation,
)
from torchlens.validation import validate_forward_pass


class AttrDict(dict):
    """Config dict whose items double as attributes (HiFi-GAN-family idiom)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Alias the instance namespace to the dict items."""

        super().__init__(*args, **kwargs)
        self.__dict__ = self


class AttrDictConfigModel(nn.Module):
    """Model that reads a config value through an AttrDict attribute."""

    def __init__(self) -> None:
        """Build the config and one linear layer."""

        super().__init__()
        self.h = AttrDict({"n_fft": 8, "upsample_rates": [2, 2]})
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the linear output by a config-derived constant."""

        return self.lin(x) * float(self.h.n_fft // 2 + 1)


class Config:
    """Plain config object that deep-copies faithfully."""

    def __init__(self) -> None:
        """Set one attribute."""

        self.scale = 2.0


class UncopyableSeanetLikeModel(nn.Module):
    """Un-deepcopyable model with a NumPy scalar attribute and an LSTM."""

    def __init__(self) -> None:
        """Build a conv, an LSTM and a NumPy-scalar hop length."""

        super().__init__()
        self.ratios = [2, 2]
        self.hop_length = np.prod(self.ratios)
        self.conv = nn.Conv1d(64, 64, 3, padding=1)
        self.lstm = nn.LSTM(64, 64, 1)

    def __deepcopy__(self, memo: dict[int, Any]) -> UncopyableSeanetLikeModel:
        """Refuse deepcopy as a legacy ``weight_norm`` module does."""

        raise RuntimeError("Only Tensors created explicitly by the user support deepcopy")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv then LSTM with a residual, scaled by the hop length."""

        y = self.conv(x)
        recurrent, _ = self.lstm(y.permute(2, 0, 1))
        return (recurrent.permute(1, 2, 0) + y) * float(self.hop_length)


def _module_with_structured_numpy_void() -> tuple[nn.Module, set[str]]:
    """A structured ``np.void`` is a writable view into its array, not an immutable scalar.

    Snapshotting it by reference would let an in-place field write during the
    forward pass go unseen; the sibling numpy scalar stays supported.
    """

    records = np.zeros(2, dtype=[("hop", "i8")])
    record = records[0]
    record["hop"] = 5
    assert records[0]["hop"] == 5  # the scalar writes through to its array
    module = nn.Module()
    module.record = record
    module.hop_length = np.prod([2, 2])
    return module, {"Module[0].record"}


class SlotsState:
    """Buffer-like state object whose values live in ``__slots__``."""

    __slots__ = ("size", "items", "lock", "spare")

    def __init__(self) -> None:
        """Fill every slot but ``spare``."""

        self.size = 0
        self.items: list[torch.Tensor] = []
        self.lock = threading.RLock()


class Group:
    """Group whose elements point back at it (an equivariant-library idiom)."""

    def __init__(self) -> None:
        """Make two elements that reference the group."""

        self.order = 2
        self.elements = [Element(self, index) for index in range(2)]


class Element:
    """Group element holding a back-reference to its group."""

    def __init__(self, group: Group, index: int) -> None:
        """Store the group and the index."""

        self.group = group
        self.index = index


def _module_with_opaque_state() -> tuple[nn.Module, set[str]]:
    """C-level values, object arrays and objects used as dict keys stay unsupported."""

    module = nn.Module()
    module.stream = io.BytesIO(b"state")
    module.table = np.array([object()], dtype=object)
    module.by_config = {PlainConfig(): 1}
    module.ok_slots = SlotsState()
    return module, {"Module[0].by_config", "Module[0].stream", "Module[0].table"}


@pytest.mark.parametrize(
    "build",
    [_module_with_structured_numpy_void, _module_with_opaque_state],
    ids=["structured-numpy-void", "opaque-state"],
)
def test_unsupported_plain_attribute_state_is_still_refused(
    build: Any,
) -> None:
    """The deepcopy-fallback snapshot names what it cannot restore instead of copying it.

    A structured ``np.void``, C-level objects, object arrays and objects used
    as dict keys are reported as unsupported paths; a silent by-reference copy
    would hide in-place writes the forward makes to them.
    """

    module, expected = build()
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert set(snapshot.unsupported_attr_paths) == expected
    assert snapshot.is_complete is False


def test_attrdict_model_validates_through_live_model_fallback() -> None:
    """A deepcopy that drops AttrDict attributes falls back instead of raising."""

    torch.manual_seed(0)
    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert (
            user_public_impls._validate_forward_pass_torch(
                AttrDictConfigModel(), [torch.randn(2, 8)], {}, validate_metadata=True
            )
            is True
        )
    # The ground-truth fallback warns once per model class; the replay one every time.
    with pytest.warns(RuntimeWarning, match="model could not be copied"):
        assert validate_forward_pass(AttrDictConfigModel(), torch.randn(2, 8)) is True


def test_copy_fidelity_ignores_types_with_custom_copy_protocol() -> None:
    """A type that customizes copying may drop attributes on purpose."""

    class DropsCache:
        """Object whose deepcopy intentionally drops a cache attribute."""

        def __init__(self) -> None:
            """Set a value and a cache."""

            self.value = 1
            self.cache = [1, 2]

        def __deepcopy__(self, memo: dict[int, Any]) -> DropsCache:
            """Copy without the cache."""

            copied = DropsCache.__new__(DropsCache)
            copied.value = self.value
            return copied

    source = nn.Module()
    source.thing = DropsCache()
    copied = nn.Module()
    copied.thing = source.thing.__deepcopy__({})
    assert_copy_kept_instance_attrs(source, copied)


def test_uncopyable_model_with_numpy_scalar_and_lstm_validates() -> None:
    """NumPy scalars and LSTM weight aliases no longer refuse the fallback."""

    torch.manual_seed(0)
    model = UncopyableSeanetLikeModel()
    snapshot = _ModuleTreePlainAttrSnapshot(model)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert (
            user_public_impls._validate_forward_pass_torch(
                model, [torch.randn(1, 64, 16)], {}, validate_metadata=True
            )
            is True
        )


def test_numpy_scalar_attribute_reassignment_is_restored() -> None:
    """A NumPy scalar reassigned during a forward is restored by value and type."""

    module = nn.Module()
    module.hop_length = np.prod([2, 2])
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert snapshot.is_complete
    module.hop_length = 9
    snapshot.restore_changed_attrs()
    assert isinstance(module.hop_length, np.int64)
    assert module.hop_length == 4
    module.hop_length = np.int32(4)  # same value, different dtype: still a change
    snapshot.restore_changed_attrs()
    assert module.hop_length.dtype == np.int64


def test_lstm_flat_weights_alias_snapshots_and_restores_by_identity() -> None:
    """``_flat_weights`` is compared and restored by identity to registered params."""

    lstm = nn.LSTM(64, 64, 1)
    registered = list(lstm._flat_weights)
    snapshot = _ModuleTreePlainAttrSnapshot(lstm)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    lstm._flat_weights = [weight.detach().clone() for weight in registered]
    snapshot.restore_changed_attrs()
    assert all(
        restored is original
        for restored, original in zip(lstm._flat_weights, registered, strict=True)
    )


def test_registered_tensor_alias_attribute_is_identity_tracked() -> None:
    """A plain attribute aliasing a registered parameter is tracked by identity."""

    linear = nn.Linear(128, 128)
    # ``linear.weight_alias = linear.weight`` would REGISTER a second parameter;
    # writing the instance namespace keeps it a plain attribute.
    linear.__dict__["weight_alias"] = linear.weight
    assert "weight_alias" not in linear._parameters
    snapshot = _ModuleTreePlainAttrSnapshot(linear)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    linear.__dict__["weight_alias"] = linear.weight.detach().clone()
    snapshot.restore_changed_attrs()
    assert linear.__dict__["weight_alias"] is linear.weight
    assert set(linear._parameters) == {"weight", "bias"}  # restore registered nothing new


class PlainConfig:
    """Config object kept as a plain attribute (an ``hp`` / ``cfg`` idiom)."""

    def __init__(self) -> None:
        """Set a scale and a call counter."""

        self.scale = 0.5
        self.calls = 0


class RichPlainStateModel(nn.Module):
    """Model whose plain attributes used to refuse the deepcopy fallback.

    It keeps a Python module, a large positional table, dtype and device
    descriptors, a config object and a NumPy generator as plain attributes, and
    its forward mutates the table and the config in place.
    """

    def __init__(self, uncopyable: bool) -> None:
        """Build the layer and the plain state."""

        super().__init__()
        self.uncopyable = uncopyable
        self.F = torch.nn.functional
        self.lin = nn.Linear(8, 8)
        self.pe = torch.randn(1024, 8)
        self._dtype = torch.float32
        self._device = torch.device("cpu")
        self.cfg = PlainConfig()
        self.rng = np.random.RandomState(0)

    def __deepcopy__(self, memo: dict[int, Any]) -> RichPlainStateModel:
        """Refuse deepcopy when asked to, as a model with an opaque handle does."""

        if self.uncopyable:
            raise RuntimeError("this model cannot be deep-copied")
        copied = RichPlainStateModel.__new__(RichPlainStateModel)
        memo[id(self)] = copied
        for name, value in self.__dict__.items():
            copied.__dict__[name] = copy.deepcopy(value, memo)
        return copied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the table rows, then bump the table and the config in place."""

        out = self.F.relu(self.lin(x.to(self._dtype)) + self.pe[: x.shape[0]] * self.cfg.scale)
        self.pe.mul_(1.0 + 1e-3)
        self.cfg.calls += 1
        self.cfg.scale = self.cfg.scale + 0.25
        return out


def test_module_object_attribute_no_longer_breaks_the_validation_deepcopy() -> None:
    """A Python module held as an attribute is shared, so the deepcopy succeeds."""

    model = RichPlainStateModel(uncopyable=False)
    copied = deepcopy_for_validation(model)
    assert copied.F is torch.nn.functional
    assert copied.pe is not model.pe and copied.cfg is not model.cfg
    with pytest.raises(TypeError, match="module"):
        copy.deepcopy(model)


def test_rich_plain_state_validates_through_live_model_fallback() -> None:
    """Large tensors, descriptors, modules, configs and generators are restored and proven."""

    torch.manual_seed(0)
    model = RichPlainStateModel(uncopyable=True)
    snapshot = _ModuleTreePlainAttrSnapshot(model)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    pe, cfg = model.pe, model.cfg
    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert (
            user_public_impls._validate_forward_pass_torch(
                model, [torch.randn(4, 8)], {}, validate_metadata=True
            )
            is True
        )
    assert model.pe is pe and model.cfg is cfg


def test_in_place_writes_to_rich_plain_state_are_restored_in_place() -> None:
    """Restores write back into the original objects, so aliases see the old values."""

    model = RichPlainStateModel(uncopyable=True)
    pe, cfg, rng = model.pe, model.cfg, model.rng
    pe_before = pe.clone()
    expected_draw = np.random.RandomState(0).rand()
    snapshot = _ModuleTreePlainAttrSnapshot(model)
    pe.add_(1.0)
    cfg.calls = 7
    cfg.extra = "added"
    rng.rand()
    model._dtype = torch.float64
    snapshot.restore_changed_attrs()
    assert model.pe is pe and torch.equal(pe, pe_before)
    assert model.cfg is cfg and vars(cfg) == {"scale": 0.5, "calls": 0}
    assert model.rng is rng and rng.rand() == expected_draw
    assert model._dtype is torch.float32
    model.cfg = PlainConfig()  # a reassignment is a change even with equal values
    snapshot.restore_changed_attrs()
    assert model.cfg is cfg


def test_nan_tensor_attribute_compares_unchanged_by_bytes() -> None:
    """A NaN-holding plain tensor is unchanged when its bytes are, not flagged forever."""

    module = nn.Module()
    module.table = torch.full((5000,), float("nan"))
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert snapshot.is_complete
    snapshot.restore_changed_attrs()
    module.table[0] = 1.0
    snapshot.restore_changed_attrs()
    assert torch.isnan(module.table).all()


def test_large_non_alias_tensor_state_is_captured_whole() -> None:
    """Large plain tensors, alone, mixed with a registered alias and copied, are cloned whole."""

    module = nn.Linear(128, 128)
    module.cache = [torch.zeros(5000)]
    module.mixed = [module.weight, torch.zeros(5000)]
    module.copied_weight = module.weight.detach().clone()
    copied_before = module.copied_weight.clone()
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    module.cache[0].add_(1.0)
    module.copied_weight.mul_(2.0)
    snapshot.restore_changed_attrs()
    assert torch.equal(module.cache[0], torch.zeros(5000))
    assert torch.equal(module.copied_weight, copied_before)


def test_slots_state_is_restored_through_its_slots() -> None:
    """Slot values are compared and restored in place; an unset slot is cleared again."""

    module = nn.Module()
    state = module.state = SlotsState()
    lock = state.lock
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    state.size = 3
    state.items.append(torch.ones(2))
    state.spare = "filled"
    snapshot.restore_changed_attrs()
    assert module.state is state and state.size == 0 and state.items == []
    assert state.lock is lock and not hasattr(state, "spare")


def test_cyclic_objects_and_simple_namespaces_are_restored() -> None:
    """Back-references inside an object graph and ``SimpleNamespace`` configs restore."""

    module = nn.Module()
    group = module.group = Group()
    config = module.config = types.SimpleNamespace(depth=2, names=["a"])
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert snapshot.is_complete, snapshot.unsupported_attr_paths
    group.elements[1].index = 9
    group.elements[0].group = Group()
    config.depth = 5
    config.names.append("b")
    snapshot.restore_changed_attrs()
    assert module.group is group and [e.index for e in group.elements] == [0, 1]
    assert all(element.group is group for element in group.elements)
    assert module.config is config and vars(config) == {"depth": 2, "names": ["a"]}
