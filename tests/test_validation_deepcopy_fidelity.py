"""Validation deepcopy fidelity and plain-attribute fallback coverage.

Rung-1 menagerie findings (2026-10-04): FreeV's ``AttrDict`` (``self.__dict__ =
self``) deep-copies into an object with an empty attribute namespace, so the
ground-truth forward on the copy raised ``AttributeError``; WavTokenizer's legacy
``weight_norm`` makes deepcopy raise, and the fallback refused a NumPy ``int64``
``hop_length`` and ``nn.LSTM._flat_weights`` (a list aliasing registered
parameters). These tests pin the fixes and their narrowness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens._user_public_impls as user_public_impls
from torchlens._capture_state_helpers import (
    _model_for_ground_truth_validation,
    _ModuleTreePlainAttrSnapshot,
)
from torchlens._plain_attr_fidelity import assert_copy_kept_instance_attrs
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


class FaithfulConfigModel(nn.Module):
    """Model whose plain config object survives deepcopy."""

    def __init__(self) -> None:
        """Build the config and one linear layer."""

        super().__init__()
        self.cfg = Config()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the linear output by the config value."""

        return self.lin(x) * self.cfg.scale


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


def test_copy_fidelity_flags_attrdict_and_accepts_faithful_copies() -> None:
    """The fidelity check names the dropped names and passes faithful copies."""

    import copy

    model = AttrDictConfigModel()
    with pytest.raises(ValueError, match=r"dropped instance attributes \['n_fft'"):
        assert_copy_kept_instance_attrs(model, copy.deepcopy(model))
    faithful = FaithfulConfigModel()
    assert_copy_kept_instance_attrs(faithful, copy.deepcopy(faithful))


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


def test_faithful_deepcopy_keeps_the_copy_path() -> None:
    """A model that deep-copies faithfully is still validated on its copy."""

    model = FaithfulConfigModel()
    ground_truth_model, snapshot = _model_for_ground_truth_validation(model)
    assert ground_truth_model is not model
    assert snapshot is None


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


def test_non_alias_large_tensor_state_is_still_refused() -> None:
    """Large tensors that are not registered aliases keep the honest refusal."""

    module = nn.Linear(128, 128)
    module.cache = [torch.zeros(5000)]
    module.mixed = [module.weight, torch.zeros(5000)]
    module.copied_weight = module.weight.detach().clone()
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert set(snapshot.unsupported_attr_paths) == {
        "Linear[0].cache",
        "Linear[0].copied_weight",
        "Linear[0].mixed",
    }


def test_structured_numpy_void_attribute_is_still_refused() -> None:
    """A structured ``np.void`` is a writable view into its array, not an immutable scalar.

    Snapshotting it by reference would let an in-place field write during the
    forward pass go unseen, so the fallback must keep refusing it.
    """

    records = np.zeros(2, dtype=[("hop", "i8")])
    record = records[0]
    record["hop"] = 5
    assert records[0]["hop"] == 5  # the scalar writes through to its array
    module = nn.Module()
    module.record = record
    module.hop_length = np.prod([2, 2])
    snapshot = _ModuleTreePlainAttrSnapshot(module)
    assert set(snapshot.unsupported_attr_paths) == {"Module[0].record"}
