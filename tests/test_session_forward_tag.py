"""A capture never marks the user's own ``forward`` callable as a TorchLens wrapper.

A module whose instance ``forward`` is a torch function (``act.forward = torch.relu``) gets a
session wrapper for each capture. The wrapper copies the original's attributes
(``functools.update_wrapper``); it must not share the original's TorchLens tag, or marking
the wrapper as a decorated forward marks ``torch.relu`` itself, and the next capture strips
every module forward that is ``torch.relu`` as if TorchLens had left it behind.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens._utils._tensor_label import is_forward_call_decorated
from torchlens.backends.torch.wrappers import wrap_torch


class _Holder(nn.Module):
    """Calls a submodule whose ``forward`` is a plain torch function."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Module()
        self.act.forward = torch.relu  # type: ignore[method-assign]

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x) * 2.0


def _trace_fresh() -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(_Holder(), torch.ones(2, 3))


def test_capture_leaves_the_torch_function_unmarked() -> None:
    """After a capture, ``torch.relu`` is not tagged as a decorated module forward."""

    wrap_torch()  # the instance forward must be the wrapped torch function
    _trace_fresh()
    assert not is_forward_call_decorated(torch.relu)


def test_second_capture_keeps_the_instance_forward() -> None:
    """A second model whose forward is ``torch.relu`` still captures."""

    wrap_torch()
    _trace_fresh()
    trace = _trace_fresh()
    assert any(op.func_name == "relu" for op in trace._layer_list)  # type: ignore[attr-defined]
