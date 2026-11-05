"""A ``save_args=True`` capture-cache entry loads back with equal arguments and outputs.

Two shapes put objects in the pickled trace that torch's own unpickler cannot
rebuild: tensor metadata carrying the label storage pin (view and in-place
ops), and a bare ``torch.UntypedStorage`` argument (``Tensor.set_``). Both
used to write an entry whose load raised ``AttributeError: type object
'torch.storage.UntypedStorage' has no attribute 'dtype'``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _ViewsAndInPlace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sliced = x[:, 1:3] * 2
        bumped = x.clone()
        bumped.add_(1)
        bumped[0].mul_(3)
        return torch.cat([sliced.flatten(), bumped.flatten(), x.t().contiguous().view(-1)])


class _SetStorage(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rebound = torch.empty(0)
        rebound.set_(x.untyped_storage(), 0, (2, 4), (4, 1))
        return rebound * 2


def _storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """Return a ``uint8`` tensor over every byte of ``storage``."""

    return torch.empty(0, dtype=torch.uint8).set_(storage)


def _assert_same(fresh: Any, loaded: Any, where: str) -> None:
    """Assert two saved values are exactly equal, recursing into containers."""

    assert type(fresh) is type(loaded), where
    if isinstance(fresh, torch.Tensor):
        assert (fresh.dtype, fresh.shape, fresh.stride()) == (
            loaded.dtype,
            loaded.shape,
            loaded.stride(),
        ), where
        assert torch.equal(fresh, loaded), where
    elif isinstance(fresh, torch.UntypedStorage):
        assert torch.equal(_storage_bytes(fresh), _storage_bytes(loaded)), where
    elif isinstance(fresh, (list, tuple)):
        assert len(fresh) == len(loaded), where
        for index, (left, right) in enumerate(zip(fresh, loaded, strict=True)):
            _assert_same(left, right, f"{where}[{index}]")
    elif isinstance(fresh, dict):
        assert fresh.keys() == loaded.keys(), where
        for key in fresh:
            _assert_same(fresh[key], loaded[key], f"{where}[{key!r}]")
    else:
        assert fresh == loaded, where


def _capture(cache_dir: Path) -> Any:
    return tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, save_args=True)


# ``Tensor.set_`` makes the capture report its in-place target as an argument
# with no known source on every supported torch; that advisory is not what this
# file checks, so the set-storage case ignores exactly that message.
_SET_TARGET_ADVISORY = "TorchLens found tensor arguments with no graph/source provenance"


@pytest.mark.parametrize(
    ("model_cls", "shape", "saves_storage"),
    [(_ViewsAndInPlace, (3, 4), False), (_SetStorage, (2, 4), True)],
    ids=["views-inplace", "set-storage"],
)
def test_save_args_cache_round_trip_loads_equal_args_and_outputs(
    tmp_path: Path, model_cls: type[nn.Module], shape: tuple[int, int], saves_storage: bool
) -> None:
    """The cache hit carries the same saved arguments and outputs as the fresh capture."""

    torch.manual_seed(0)
    model, x = model_cls(), torch.randn(*shape)
    with warnings.catch_warnings():
        if saves_storage:
            warnings.filterwarnings("ignore", message=_SET_TARGET_ADVISORY, category=UserWarning)
        fresh = tl.trace(model, x, capture=_capture(tmp_path / "cache"))
        loaded = tl.trace(model, x, capture=_capture(tmp_path / "cache"))
    assert loaded.provenance.cache_hit is True
    for fresh_op, loaded_op in zip(fresh._layer_list, loaded._layer_list, strict=True):
        label = str(fresh_op.label)
        assert str(loaded_op.label) == label
        _assert_same(fresh_op.call.saved_args, loaded_op.call.saved_args, f"{label}.saved_args")
        _assert_same(
            fresh_op.call.saved_kwargs, loaded_op.call.saved_kwargs, f"{label}.saved_kwargs"
        )
        _assert_same(fresh_op.out, loaded_op.out, f"{label}.out")
    stored_storages = [
        arg
        for op in loaded._layer_list
        for arg in (op.call.saved_args or ())
        if isinstance(arg, torch.UntypedStorage)
    ]
    assert bool(stored_storages) is saves_storage
