"""Anti-recurrence tests for cert10 buffer-surfacing blockers.

Two independent bugs, same theme (registered buffers not surfaced correctly):

1. ``BlobRef`` leak on load (torchlens/_io/rehydrate.py): after any default
   eager ``tl.load()`` of a bundle with registered buffers, ``Buffer.
   initial_value`` returned a raw internal ``BlobRef`` NamedTuple instead of a
   ``torch.Tensor``. Root cause was ordering -- ``rebuild_trace_accessors()``
   ran BEFORE ``_rehydrate_object()`` resolved ``_buffer_initial_values`` from
   BlobRef placeholders into real tensors, so ``accessor_rebuild.py`` baked
   the still-unresolved BlobRef into each ``Buffer`` at construction. Fixed by
   reordering ``rehydrate_trace`` so accessors are rebuilt AFTER blob
   resolution.

2. ``tl.record()`` crash on buffer-only output (torchlens/capture/trace.py,
   torchlens/fastlog/_recorder.py): a model returning a registered buffer as
   (part of) its output crashed fastlog's one-shot ``tl.record()`` with a
   ``RuntimeError``, even though ``tl.trace()`` handled the identical model
   fine. Two compounding causes: (a) the postprocess=False branch (always
   true for the fastlog Recorder) skipped output extraction entirely and ran
   session cleanup -- which wipes buffer tensor labels -- before the Recorder
   manually extracted outputs from the now-cleaned-up model; (b) the
   Recorder's per-pass trace never set ``_source_model_ref``, which the
   direct-registered-buffer-output fallback (backend.py
   ``_is_direct_registered_buffer_output``) requires to identify an unlabeled
   output tensor as one of the model's own buffers. Fixed by extracting/
   marking outputs BEFORE cleanup (mirroring the postprocess=True branch) and
   setting ``_source_model_ref`` unconditionally for every capture path.

Both tests below FAIL on the pre-fix code and PASS after the fix (verified by
stashing the fix commit and re-running).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

import torchlens as tl  # noqa: E402
from torchlens._io import BlobRef  # noqa: E402


class _BatchNormBufferModel(nn.Module):
    """Small model with real BatchNorm running-stat buffers."""

    def __init__(self) -> None:
        """Register a BatchNorm layer with running_mean/running_var buffers."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through BatchNorm.

        Parameters
        ----------
        x:
            Input tensor of shape ``(N, 4)``.

        Returns
        -------
        torch.Tensor
            BatchNorm output.
        """

        return self.bn(x)


class _BufferOnlyOutputModel(nn.Module):
    """Model whose sole output is a static registered buffer, untouched by any op."""

    def __init__(self) -> None:
        """Register the static buffer."""

        super().__init__()
        self.register_buffer("counter", torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the untouched buffer, ignoring the input.

        Parameters
        ----------
        x:
            Ignored input tensor.

        Returns
        -------
        torch.Tensor
            The registered buffer itself.
        """

        return self.counter


def test_loaded_buffer_initial_value_is_real_tensor_not_blobref() -> None:
    """Default eager ``tl.load()`` must resolve ``Buffer.initial_value`` to a Tensor.

    Regression coverage for the ordering bug in ``rehydrate_trace``:
    ``rebuild_trace_accessors()`` used to run before ``_rehydrate_object()``
    resolved ``_buffer_initial_values``, permanently baking a raw ``BlobRef``
    into every ``Buffer.initial_value`` after a plain ``tl.load()``.
    """

    torch.manual_seed(0)
    model = _BatchNormBufferModel()
    model.eval()
    x = torch.randn(5, 4)
    log = tl.trace(model, x)

    original_running_mean = log.buffers["bn.running_mean"].initial_value.clone()
    original_running_var = log.buffers["bn.running_var"].initial_value.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "bundle"
        tl.save(log, str(bundle_path))
        loaded = tl.load(str(bundle_path))

        loaded_mean_buffer = loaded.buffers["bn.running_mean"]
        loaded_var_buffer = loaded.buffers["bn.running_var"]

        # The core regression: initial_value must be a real torch.Tensor, not
        # the internal BlobRef placeholder that scrub/rehydrate use on disk.
        assert type(loaded_mean_buffer.initial_value) is torch.Tensor, (
            f"expected torch.Tensor, got {type(loaded_mean_buffer.initial_value)!r} "
            "-- rebuild_trace_accessors() must run AFTER _rehydrate_object() "
            "resolves _buffer_initial_values"
        )
        assert type(loaded_var_buffer.initial_value) is torch.Tensor
        assert not isinstance(loaded_mean_buffer.initial_value, BlobRef)

        # Bit-exact against the originally traced initial values.
        assert torch.equal(loaded_mean_buffer.initial_value, original_running_mean)
        assert torch.equal(loaded_var_buffer.initial_value, original_running_var)

        # to_pandas() must not leak raw BlobRef tuples into the initial_value column.
        buffer_df = loaded.buffers.to_pandas()
        assert "initial_value" in buffer_df.columns
        for value in buffer_df["initial_value"]:
            assert not isinstance(value, BlobRef)
            if value is not None:
                assert isinstance(value, torch.Tensor)

        # Re-save + reload must round-trip clean (no dangling cross-generation
        # blob_id baked in from the unresolved first load).
        bundle_path_2 = Path(tmpdir) / "bundle2"
        tl.save(loaded, str(bundle_path_2))
        reloaded = tl.load(str(bundle_path_2))
        reloaded_mean_buffer = reloaded.buffers["bn.running_mean"]
        assert type(reloaded_mean_buffer.initial_value) is torch.Tensor
        assert torch.equal(reloaded_mean_buffer.initial_value, original_running_mean)

    log.cleanup()


def test_record_does_not_crash_on_registered_buffer_output() -> None:
    """``tl.record()`` must handle a model returning a registered buffer as output.

    Regression coverage: fastlog's one-shot ``tl.record()`` used to raise
    ``RuntimeError: TorchLens could not attribute a model output tensor to
    any traced op`` for any model whose forward pass returns one of its own
    registered buffers untouched, even though ``tl.trace()`` handled the
    identical model without issue.
    """

    model = _BufferOnlyOutputModel()
    x = torch.zeros(3)

    # Must not raise. default_op=True saves every op (including the
    # synthesized output node) so the buffer's out value is retained for the
    # assertions below; the crash this guards against triggers during output
    # extraction itself, regardless of what the predicate saves.
    recording = tl.record(model, x, default_op=True)
    assert recording is not None
    assert recording.failed is False

    trace = recording.to_trace()
    try:
        # The buffer output must be captured as a real output layer bound to
        # a genuine buffer source node, exactly like tl.trace() would --
        # not silently dropped/misattributed and not a crash. (Retaining the
        # saved activation value on this late-synthesized node is a separate,
        # pre-existing predicate-mode limitation -- structural attribution is
        # what this fix restores.)
        assert trace.output_layers == ["output_1"]
        output_layer = trace["output_1"]

        assert len(output_layer.parents) == 1
        parent = trace[output_layer.parents[0]]
        assert parent.is_buffer
        assert parent.address == "counter"
    finally:
        trace.cleanup()
