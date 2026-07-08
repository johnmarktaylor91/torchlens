"""Regression tests for the Lightning ``LayerProfilerCallback`` teardown."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

pytest.importorskip("lightning")

from torchlens.callbacks.lightning import LayerProfilerCallback
from torchlens.data_classes.trace import Trace


class _TinyModel(nn.Module):
    """Minimal model with a single saved-activation layer."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x)


class _FakeTrainer:
    """Minimal stand-in for a Lightning ``Trainer``."""

    global_step = 3


def _make_callback(tmp_path: Path) -> LayerProfilerCallback:
    """Return a callback writing to a scratch container path."""

    return LayerProfilerCallback(tmp_path / "profile.jsonl")


def test_maybe_profile_calls_trace_cleanup(tmp_path: Path) -> None:
    """``_maybe_profile`` must call ``Trace.cleanup()`` on every profiled batch.

    Regression test for the ref-cycle leak: ``_maybe_profile`` used to call
    ``tl.trace()`` every profiled batch without ever calling ``log.cleanup()``,
    so the CUDA cache was never emptied and the trace was never purged from
    the backward registry.
    """

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    with patch.object(Trace, "cleanup", autospec=True) as mock_cleanup:
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert mock_cleanup.call_count == 1


def test_maybe_profile_calls_cleanup_even_if_write_fails(tmp_path: Path) -> None:
    """``log.cleanup()`` must run even when writing the JSONL record fails.

    The fix wraps record-building/writing in ``try/finally: log.cleanup()``.
    An exception mid-write must not skip cleanup and leak the trace.
    """

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    with (
        patch.object(Trace, "cleanup", autospec=True) as mock_cleanup,
        patch.object(Path, "open", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert mock_cleanup.call_count == 1


def test_maybe_profile_releases_activation_tensor_deterministically(tmp_path: Path) -> None:
    """The profiled trace's activation payload is freed without ``gc.collect()``.

    Before the fix, ``cleanup()`` was never called, so each profiled batch's
    saved activation tensors (and the trace holding them) stayed alive via the
    Trace<->Op reference cycle until an incidental cyclic-GC pass -- an
    unbounded-until-GC leak in a tight training loop. With the fix,
    ``cleanup()`` unconditionally clears every ``Op`` attribute, so the
    activation tensor's refcount drops to zero and it is freed immediately by
    plain reference counting -- no ``gc.collect()`` required.
    """

    import gc

    import torchlens

    # Clear any pending garbage left over from earlier tests in this process
    # (unrelated cyclic references, e.g. frames from prior torch/trace calls)
    # so this test's own refcount-only assertion below isn't polluted by
    # unrelated prior state.
    gc.collect()

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    tensor_ref: weakref.ReferenceType[torch.Tensor] | None = None
    orig_trace = torchlens.trace

    def _spying_trace(*args: Any, **kwargs: Any) -> Any:
        nonlocal tensor_ref
        log = orig_trace(*args, **kwargs)
        for op in log.layer_list:
            if getattr(op, "is_input", False):
                # The input node's ``out`` is the caller's own input tensor
                # (not a freshly computed activation) and may be kept alive
                # by the caller's own local scope; skip it.
                continue
            candidate = getattr(op, "out", None)
            if isinstance(candidate, torch.Tensor):
                tensor_ref = weakref.ref(candidate)
                break
        return log

    with patch("torchlens.trace", _spying_trace):
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert tensor_ref is not None, "no saved activation tensor was observed"
    assert tensor_ref() is None, (
        "activation tensor was still alive immediately after _maybe_profile "
        "returned -- cleanup() did not run (or did not run deterministically)"
    )
