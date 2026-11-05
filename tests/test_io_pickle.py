"""Tests for portable pickle scrubbing and rehydration."""

from __future__ import annotations

import pickle

import torch
from safetensors.torch import save_file
from torch import nn

import torchlens as tl
from torchlens import Trace, trace as trace_fn
from torchlens._io import TLSPEC_VERSION
from torchlens._io.rehydrate import rehydrate_trace
from torchlens._io.scrub import scrub_for_save
from torchlens._quantities import Bytes


class _TinyIOModel(nn.Module):
    """Small model covering modules, params, buffers, and captured args."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


class _PlainPickleModel(nn.Module):
    """Model using only plain-picklable ops for compatibility checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain-pickle compatibility model."""
        return torch.sin(x)


def _build_live_log() -> Trace:
    """Create a canonical live log for portable I/O tests."""

    torch.manual_seed(0)
    model = _TinyIOModel()
    x = torch.randn(2, 4)
    return trace_fn(
        model,
        x,
        capture=tl.options.CaptureOptions(
            save_args=True,
            save_rng=True,
            save_source_context=True,
            seed=0,
        ),
        save="all",
    )


def _write_manifest(tmp_path, blob_specs) -> dict[str, object]:
    """Persist blob specs into a minimal temporary manifest layout."""

    bundle_path = tmp_path
    blob_dir = bundle_path / "blobs"
    blob_dir.mkdir()
    tensors = []
    for blob_id, tensor, kind, label in blob_specs:
        relative_path = f"blobs/{blob_id}.safetensors"
        save_file({"tensor": tensor.detach().cpu()}, bundle_path / relative_path)
        tensors.append(
            {
                "blob_id": blob_id,
                "kind": kind,
                "label": label,
                "relative_path": relative_path,
            }
        )
    return {"tlspec_version": TLSPEC_VERSION, "tensors": tensors}


def test_scrubbed_pickle_roundtrip_rehydrates_accessors(tmp_path) -> None:
    """Scrubbed portable state should rehydrate modules, buffers, and blobs."""

    live_log = _build_live_log()
    scrubbed_state, blob_specs, unsupported_tensor_records = scrub_for_save(
        live_log,
        include_outs=True,
        include_grads=False,
        include_saved_args=True,
        include_rng_states=True,
    )
    assert unsupported_tensor_records == []
    manifest = _write_manifest(tmp_path, blob_specs)

    roundtripped_state = pickle.loads(pickle.dumps(scrubbed_state))
    restored = rehydrate_trace(
        roundtripped_state,
        manifest,
        tmp_path,
        lazy=False,
        map_location="cpu",
        materialize_nested=True,
    )

    assert restored.modules["self"].address == "self"
    assert restored.modules["bn"].address == "bn"
    assert restored.modules["bn"]._source_trace is restored
    assert restored.buffers["bn.running_mean"].address == "bn.running_mean"

    first_saved_layer = next(layer for layer in restored._layer_list if layer.has_saved_activation)
    assert isinstance(first_saved_layer.out, torch.Tensor)
    assert (
        restored._layer_logs[first_saved_layer.layer_label].ops[first_saved_layer.pass_number - 1]
        is first_saved_layer
    )

    first_arg_layer = next(
        layer
        for layer in restored._layer_list
        if layer.call.saved_args is not None and len(layer.call.saved_args) > 0
    )
    assert isinstance(first_arg_layer.call.saved_args[0], torch.Tensor)


def test_trace_setstate_coerces_legacy_backward_gradfn_refs_dict() -> None:
    """A genuinely-on-disk legacy ``_backward_gradfn_refs`` dict must not crash.

    Regression gate for cert9 BLOCKER-1 (round-8 over-narrow guard on
    f7065122): ``__getstate__`` was changed to always emit
    ``state["_backward_gradfn_refs"] = []``, but that only fixes the
    *in-memory* getstate/setstate roundtrip. Every ``.tlspec`` bundle or
    plain pickle actually serialized by code strictly older than that commit
    still has ``_backward_gradfn_refs: {}`` (a ``dict``) baked into its saved
    state -- and because the key is *present*, the old
    ``default_fill_state``-only restore path left it untouched as a ``dict``.
    The runtime backward walk does
    ``trace.__dict__.setdefault("_backward_gradfn_refs", []).append(...)``
    (``backends/torch/backward.py:153,1905``), and ``dict.setdefault`` is a
    no-op when the key already exists regardless of type, so ``.append()``
    crashed with ``AttributeError: 'dict' object has no attribute 'append'``.

    This reproduces the actual on-disk legacy artifact shape (not the
    already-fixed in-memory case covered by
    ``test_trace_setstate_default_fills_grad_fn_param_refs``, which
    deliberately pops this field to isolate its own fix).
    """

    live_log = _build_live_log()
    old_state = live_log.__getstate__()
    old_state["_backward_gradfn_refs"] = {}  # pre-f7065122 on-disk shape

    restored = Trace.__new__(Trace)
    restored.__setstate__(old_state)

    assert isinstance(restored._backward_gradfn_refs, list)
    assert restored._backward_gradfn_refs == []

    out = restored[restored.output_layers[0]].out
    restored.backward(out.sum())

    assert isinstance(restored._backward_gradfn_refs, list)


def test_trace_setstate_coerces_wrong_typed_total_backward_memory() -> None:
    """``total_backward_memory`` must restore as ``Bytes``, even if missing.

    Regression gate for cert9 MAJOR-2: the explicit ``Bytes``-coercion pass
    in ``__setstate__`` upgrades default-filled ints back to ``Bytes`` for a
    fixed tuple of memory fields, but ``total_backward_memory`` was missing
    from that tuple. A ``Trace`` restored from a state dict that omits (or
    predates) this field ended up with a plain ``int(0)`` instead of
    ``Bytes(0)``, breaking ``isinstance`` checks and ``Bytes``-style format
    specs (``f"{x:.2f MB}"`` raised ``ValueError``).
    """

    live_log = _build_live_log()
    state = live_log.__getstate__()
    state.pop("_cost_total_backward_memory")

    restored = Trace.__new__(Trace)
    restored.__setstate__(state)

    assert isinstance(restored.cost.total_backward_memory, Bytes)
    # Must format cleanly like every sibling memory field, not just compare
    # equal-by-value (Bytes subclasses int, so a bare `int` would pass a
    # naive `== 0` check while still failing `isinstance`/format checks).
    f"{restored.cost.total_backward_memory:.2f MB}"
