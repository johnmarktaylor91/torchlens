"""Opaque module-forward argument objects are released at module entry.

A module argument that is not a tensor or a plain list, tuple or dict (a model's
key-value cache object, say) used to stay in the module-entry record for the
whole forward, pinning the tensors it holds. When the capture's
``module_identity_mode`` nulls ``ModuleCall.forward_args`` before the trace is
returned, such an object is now replaced by a payload-free stub that renders
the same ``<TypeName>`` summary.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._summary import format_call_arg
from torchlens._ir.workspaces import ReleasedObjectStub, ReleasedTensorStub
from torchlens.backends.torch._module_arg_stubs import stub_module_call_args


class _Holder:
    """An opaque argument object holding a tensor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def test_default_mode_releases_opaque_argument_objects() -> None:
    """Under a mode that nulls forward args, opaque objects become stubs; literals stay."""

    holder = _Holder(torch.zeros(3))
    trace = SimpleNamespace(_options_module_identity_mode="torch_module")
    args, kwargs = stub_module_call_args((holder, 2, "s"), {"cache": holder, "flag": None}, trace)

    assert args[0] == ReleasedObjectStub(type_name="_Holder")
    assert args[1:] == (2, "s")
    assert kwargs == {"cache": ReleasedObjectStub(type_name="_Holder"), "flag": None}


def test_modes_that_keep_forward_args_keep_the_live_object() -> None:
    """``pytree_module`` and ``object_module`` keep the object their ModuleCall exposes."""

    holder = _Holder(torch.zeros(3))
    for mode in ("pytree_module", "object_module"):
        trace = SimpleNamespace(_options_module_identity_mode=mode)
        args, kwargs = stub_module_call_args((torch.ones(2), holder), {"cache": holder}, trace)
        assert isinstance(args[0], ReleasedTensorStub)
        assert args[1] is holder
        assert kwargs["cache"] is holder


def test_released_object_summary_matches_the_live_object() -> None:
    """The summary of a released object is the summary of the object it replaced."""

    holder = _Holder(torch.zeros(3))
    stub = ReleasedObjectStub(type_name=type(holder).__name__)
    assert format_call_arg(stub) == format_call_arg(holder) == "<_Holder>"
    assert format_call_arg({"cache": stub}) == "{cache: <_Holder>}"


class _Inner(nn.Module):
    """Reads its tensor through an opaque keyword argument."""

    def forward(self, x: torch.Tensor, cache: _Holder) -> torch.Tensor:
        return x + cache.tensor


class _Outer(nn.Module):
    """Builds a fresh opaque argument per forward and remembers it weakly."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()
        self.refs: list[weakref.ref[_Holder]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        holder = _Holder(x * 2)
        self.refs.append(weakref.ref(holder))
        return self.inner(x, cache=holder)


def test_trace_does_not_keep_the_opaque_argument_alive() -> None:
    """A default trace keeps the summary but not the opaque object or its tensor."""

    model = _Outer()
    trace = tl.trace(model, torch.ones(4))
    gc.collect()

    assert model.refs, "the forward ran"
    assert all(ref() is None for ref in model.refs)
    inner = next(module for module in trace.modules if str(module.address) == "inner")
    summaries = [inner.calls.get(key).forward_kwargs_summary for key in inner.calls]
    assert summaries and all(summary == "{cache: <_Holder>}" for summary in summaries)
