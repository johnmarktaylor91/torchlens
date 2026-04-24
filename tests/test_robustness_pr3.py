"""Robustness sprint PR 3 — parallel/framework robustness.

Covers the parallel-wrapper and framework-integration subset of the catalog:

    - ``DistributedDataParallel`` unwraps via ``.module`` just like ``DataParallel``
      (previously only the latter was handled, so DDP users got silently-wrong
      layer addressing).
    - ``FullyShardedDataParallel`` raises with a clear message (parameters are
      sharded; there is no single unsharded module to log).
    - Dict-subclass inputs (e.g. HuggingFace ``BatchEncoding``-alikes) continue
      to work — the existing ``_move_tensors_to_device`` handles them, and we
      add a regression guard here so future refactors don't break it.
    - Dataclass-style outputs (HuggingFace ``ModelOutput``-alikes) don't crash
      the output-extraction path.
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.user_funcs import _unwrap_data_parallel


class _Tiny(nn.Module):
    """Two-layer model small enough to log cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


# ---------------------------------------------------------------------------
# DataParallel (pre-existing behavior — regression guard)
# ---------------------------------------------------------------------------


def test_data_parallel_unwrap_still_works() -> None:
    """The original DataParallel unwrap must keep working."""
    model = nn.DataParallel(_Tiny())
    unwrapped = _unwrap_data_parallel(model)
    assert isinstance(unwrapped, _Tiny)


# ---------------------------------------------------------------------------
# DistributedDataParallel (new)
# ---------------------------------------------------------------------------


def test_distributed_data_parallel_unwraps_like_data_parallel() -> None:
    """DDP has the same ``.module`` attribute and should unwrap identically.

    We can't initialize a real ``DistributedDataParallel`` without a
    process group, so we subclass it and bypass ``__init__`` to validate
    the isinstance dispatch in ``_unwrap_data_parallel``.
    """
    from torch.nn.parallel import DistributedDataParallel

    inner = _Tiny()

    # DistributedDataParallel requires a full torch.distributed init which is
    # expensive in tests. Construct a plausible wrapper by hand that shares
    # the DDP class identity so our isinstance check fires.
    wrapped = DistributedDataParallel.__new__(DistributedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = inner  # type: ignore[attr-defined]

    unwrapped = _unwrap_data_parallel(wrapped)
    assert unwrapped is inner


# ---------------------------------------------------------------------------
# FSDP (new)
# ---------------------------------------------------------------------------


def _fsdp_available() -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _fsdp_available(), reason="FSDP not available")
def test_fsdp_raises_with_clear_message() -> None:
    """FSDP cannot be unwrapped — its parameters are sharded."""
    from torch.distributed.fsdp import FullyShardedDataParallel

    inner = _Tiny()
    wrapped = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = inner  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="FullyShardedDataParallel"):
        _unwrap_data_parallel(wrapped)


@pytest.mark.skipif(not _fsdp_available(), reason="FSDP not available")
def test_fsdp_error_from_log_forward_pass_entry() -> None:
    """The entry-point (log_forward_pass) must surface the FSDP error."""
    from torch.distributed.fsdp import FullyShardedDataParallel

    inner = _Tiny()
    wrapped = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = inner  # type: ignore[attr-defined]
    x = torch.randn(2, 4)

    with pytest.raises(RuntimeError, match="FullyShardedDataParallel"):
        tl.log_forward_pass(wrapped, x, layers_to_save="none")


# ---------------------------------------------------------------------------
# HuggingFace-style dict-subclass inputs
# ---------------------------------------------------------------------------


class _BatchEncodingLike(UserDict):
    """Minimal stand-in for HuggingFace's ``BatchEncoding`` (UserDict subclass).

    BatchEncoding wraps a dict of tensor fields and adds attribute access
    plus ``to()`` methods. We only need the dict interface for TorchLens'
    input-extraction path.
    """


class _BatchEncodingAcceptingModel(nn.Module):
    """A model that takes a BatchEncoding-like input and returns a tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, batch):  # type: ignore[no-untyped-def]
        return self.lin(batch["input_ids"])


@pytest.mark.smoke
def test_huggingface_batch_encoding_like_input_logs() -> None:
    """A UserDict-based input (BatchEncoding shape) must be loggable."""
    model = _BatchEncodingAcceptingModel()
    batch = _BatchEncodingLike({"input_ids": torch.randn(2, 4)})

    log = tl.log_forward_pass(model, [batch], layers_to_save="all")
    assert len(log.layer_logs) > 0


# ---------------------------------------------------------------------------
# HuggingFace-style dataclass outputs
# ---------------------------------------------------------------------------


@dataclass
class _ModelOutputLike:
    """Stand-in for transformers' ``ModelOutput`` — dataclass of tensors."""

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class _HFOutputModel(nn.Module):
    """Returns a ``ModelOutput``-shaped dataclass with two tensor fields."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> _ModelOutputLike:
        hidden = self.proj(x)
        return _ModelOutputLike(
            last_hidden_state=hidden,
            pooler_output=hidden.mean(dim=0, keepdim=True),
        )


def test_dataclass_output_does_not_crash() -> None:
    """Dataclass outputs are extracted via attribute crawl; logging runs.

    This is a 'don't crash' guard. The catalog's concern was that dataclass
    outputs might silently lose their tensors during output extraction.
    Here we verify the forward pass completes and captures at least some
    tensor ops (the inner Linear).
    """
    model = _HFOutputModel()
    x = torch.randn(2, 4)

    log = tl.log_forward_pass(model, x, layers_to_save="none")
    # At minimum the Linear projection should be logged.
    layer_types = [ll.layer_type for ll in log.layer_logs.values()]
    assert any("linear" in t.lower() or "addmm" in t.lower() for t in layer_types), (
        f"Expected a linear/addmm op in the log; got: {layer_types}"
    )


# ---------------------------------------------------------------------------
# Positive path: standard models still work
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_standard_model_still_logs_cleanly() -> None:
    """Nothing in PR 3 should regress the golden path."""
    model = _Tiny()
    x = torch.randn(2, 4)
    log = tl.log_forward_pass(model, x, layers_to_save="all")
    assert len(log.layer_logs) > 0
