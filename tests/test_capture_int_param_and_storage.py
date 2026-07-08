"""Regression tests for two capture-robustness fixes surfaced by real models.

1. Integer/bool-dtype ``nn.Parameter`` (e.g. a fixed lookup buffer declared as
   ``nn.Parameter(torch.arange(n), requires_grad=False)``) is legal PyTorch and
   never gradient-capable. Capture prep used to force ``requires_grad = True`` on
   every parameter, which *raises* for non-floating/complex dtypes. Real model:
   D-SCRIPT (``samsledje/D-SCRIPT``).

2. Passing a ``torch.Storage`` as an op argument used to be hashed via ``str()``,
   which walks every element through the wrapped ``__getitem__`` and re-enters
   logging -> runaway recursion. Storages must be summarized by size/dtype only,
   mirroring the ``torch.Tensor`` fast path.
"""

import pytest
import torch
import torch.nn as nn

import torchlens as tl


@pytest.mark.smoke
def test_int_dtype_parameter_traces_without_crash():
    """A model with a non-float fixed Parameter must trace, and its
    requires_grad must be preserved (never force-flipped to True)."""

    class HasIntParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            # int64 Parameter, legal PyTorch, never gradient-capable
            self.index_buffer = nn.Parameter(torch.arange(4), requires_grad=False)

        def forward(self, x):
            return self.lin(x) + self.index_buffer.float()

    model = HasIntParam()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    assert trace is not None
    # The force-set must never have touched a non-float param.
    assert model.index_buffer.requires_grad is False
    assert model.index_buffer.dtype == torch.int64


def test_storage_arg_hash_is_summarized_not_walked():
    """Hashing a torch Storage argument must summarize by size/dtype, never
    iterate its elements (which would re-enter logging)."""
    from torchlens.backends.torch.tensor_tracking import _append_arg_hash

    storage = torch.arange(5, dtype=torch.float32).untyped_storage()
    out: list = []
    _append_arg_hash(storage, "p", out)
    assert len(out) == 1
    assert "storage" in out[0]
