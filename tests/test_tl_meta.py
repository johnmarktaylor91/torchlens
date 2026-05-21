"""Tests for TorchLens private ``._tl`` metadata helpers."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from torchlens.backends.torch._tl import (
    DecorationTag,
    ModuleMeta,
    ParamMeta,
    TensorMeta,
    TorchLensTLCollisionError,
    clear_meta,
    clear_tensor_label,
    copy_replacement_meta,
    get,
    get_buffer_address,
    get_label_list,
    get_module_meta,
    get_param_meta,
    get_tensor_label,
    get_tensor_meta,
    increment_param_call_index,
    is_decorated_function,
    is_forward_call_decorated,
    is_tensor_replacement_wrapped,
    is_tracked,
    mark_decorated_function,
    mark_forward_call_decorated,
    mark_tensor_replacement_wrapped,
    promote_label_to_buffer_parent_and_clear_label,
    restore_param_requires_grad,
    set_buffer_address,
    set_module_meta,
    set_param_meta,
    set_tensor_label,
)


def _plain_function() -> None:
    """Provide a callable that accepts custom attributes."""


@pytest.mark.smoke
def test_tensor_helpers_lazy_init_and_field_operations() -> None:
    """Tensor helpers should lazy-init and mutate only requested fields."""
    t = torch.ones(2)

    assert get_tensor_meta(t) is None
    assert get_tensor_label(t) is None

    set_tensor_label(t, "layer_1_raw")
    assert isinstance(t._tl, TensorMeta)
    assert get_tensor_label(t) == "layer_1_raw"
    assert is_tracked(t)

    set_buffer_address(t, "block.buf")
    assert get_buffer_address(t) == "block.buf"

    clear_tensor_label(t)
    assert get_tensor_label(t) is None
    assert get_buffer_address(t) == "block.buf"

    set_tensor_label(t, "buffer_parent_raw")
    promote_label_to_buffer_parent_and_clear_label(t)
    assert t._tl.label_raw is None
    assert t._tl.buffer_parent == "buffer_parent_raw"
    assert t._tl.address == "block.buf"


@pytest.mark.smoke
def test_clear_meta_is_idempotent_and_preserves_foreign_tl() -> None:
    """clear_meta should remove TorchLens metadata but leave foreign _tl alone."""
    t = torch.ones(1)
    set_tensor_label(t, "x_raw")
    clear_meta(t)
    assert not hasattr(t, "_tl")
    clear_meta(t)

    t._tl = {"foreign": True}
    clear_meta(t)
    assert t._tl == {"foreign": True}


@pytest.mark.smoke
def test_get_label_list_is_sparse_and_checks_foreign_tl() -> None:
    """get_label_list should return only real labels and reject foreign _tl."""
    a = torch.ones(1)
    b = torch.ones(1)
    c = torch.ones(1)
    d = torch.ones(1)
    set_tensor_label(a, "a_raw")
    set_buffer_address(b, "buf")
    d._tl = ModuleMeta(address="module", module_type="Linear")

    assert get_label_list([a, b, c, d]) == ["a_raw"]

    c._tl = object()
    with pytest.raises(TorchLensTLCollisionError):
        get_label_list([c])


@pytest.mark.smoke
def test_param_helpers_set_increment_and_restore() -> None:
    """Parameter helpers should preserve pre-capture requires_grad and call counts."""
    p = nn.Parameter(torch.ones(1), requires_grad=False)
    set_param_meta(p, barcode="abc", address="linear.weight", requires_grad_before=False)

    meta = get_param_meta(p)
    assert isinstance(meta, ParamMeta)
    assert meta.param_barcode == "abc"
    assert meta.param_address == "linear.weight"
    assert meta.call_index == 0
    assert increment_param_call_index(p) == 1

    p.requires_grad = True
    restore_param_requires_grad(p)
    assert p.requires_grad is False


@pytest.mark.smoke
def test_module_helpers_set_and_read_meta() -> None:
    """Module helpers should attach permanent module metadata."""
    module = nn.Linear(2, 2)
    set_module_meta(module, address="block.0", module_type="Linear")

    meta = get_module_meta(module)
    assert isinstance(meta, ModuleMeta)
    assert meta.address == "block.0"
    assert meta.module_type == "Linear"
    assert is_tracked(module)


@pytest.mark.smoke
def test_decoration_helpers_are_independent_flags() -> None:
    """Decoration helpers should set and read independent sentinel flags."""
    fn = _plain_function

    assert not is_decorated_function(fn)
    assert not is_forward_call_decorated(fn)
    assert not is_tensor_replacement_wrapped(fn)

    mark_decorated_function(fn)
    assert is_decorated_function(fn)
    assert not is_forward_call_decorated(fn)

    mark_forward_call_decorated(fn)
    mark_tensor_replacement_wrapped(fn)
    assert is_forward_call_decorated(fn)
    assert is_tensor_replacement_wrapped(fn)
    assert isinstance(get(fn), DecorationTag)

    clear_meta(fn)


@pytest.mark.smoke
def test_copy_replacement_meta_preserves_subclass_and_is_shallow_copy() -> None:
    """copy_replacement_meta should copy the dataclass while preserving subclass."""
    src = torch.ones(1)
    dst = torch.zeros(1)
    set_tensor_label(src, "src_raw")

    copy_replacement_meta(src, dst)

    assert isinstance(dst._tl, TensorMeta)
    assert dst._tl is not src._tl
    assert dst._tl == src._tl


@pytest.mark.smoke
@pytest.mark.parametrize(
    "meta",
    [
        TensorMeta(label_raw="x", address="buf", buffer_parent="parent"),
        ParamMeta(
            param_barcode="abc",
            param_address="linear.weight",
            call_index=2,
            requires_grad_before_capture=True,
        ),
        ModuleMeta(address="block", module_type="Sequential"),
        DecorationTag(
            is_decorated_function=True,
            forward_call_is_decorated=True,
            tensor_replacement_wrapped=True,
        ),
    ],
)
def test_meta_dataclasses_round_trip(meta: Any) -> None:
    """Metadata dataclasses should deepcopy, pickle, and replace cleanly."""
    assert copy.deepcopy(meta) == meta
    assert pickle.loads(pickle.dumps(meta)) == meta
    replaced = replace(meta)
    assert replaced == meta
    assert type(replaced) is type(meta)


@pytest.mark.smoke
def test_foreign_tl_raises_for_read_helpers_and_is_tracked() -> None:
    """Read helpers should raise when a foreign _tl object is present."""
    t = torch.ones(1)
    t._tl = object()

    for helper in (get, is_tracked, get_tensor_meta, get_tensor_label, get_buffer_address):
        with pytest.raises(TorchLensTLCollisionError):
            helper(t)


@pytest.mark.smoke
def test_wrong_kind_meta_raises_on_kind_specific_helpers() -> None:
    """Kind-specific helpers should reject another TorchLens metadata subclass."""
    t = torch.ones(1)
    t._tl = ParamMeta()
    with pytest.raises(TorchLensTLCollisionError):
        set_tensor_label(t, "x")
    with pytest.raises(TorchLensTLCollisionError):
        set_buffer_address(t, "buf")

    p = nn.Parameter(torch.ones(1))
    p._tl = TensorMeta()
    with pytest.raises(TorchLensTLCollisionError):
        set_param_meta(p, barcode="abc", address="w", requires_grad_before=True)
    with pytest.raises(TorchLensTLCollisionError):
        increment_param_call_index(p)
    with pytest.raises(TorchLensTLCollisionError):
        restore_param_requires_grad(p)

    module = nn.Linear(1, 1)
    module._tl = TensorMeta()
    with pytest.raises(TorchLensTLCollisionError):
        set_module_meta(module, address="", module_type="Linear")

    _plain_function._tl = TensorMeta()
    with pytest.raises(TorchLensTLCollisionError):
        mark_decorated_function(_plain_function)
    with pytest.raises(TorchLensTLCollisionError):
        is_decorated_function(_plain_function)
    clear_meta(_plain_function)
