"""Tests for portable pickle scrubbing and rehydration."""

from __future__ import annotations

import pickle

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from torchlens import Trace, trace as trace_fn
from torchlens._io import TLSPEC_VERSION, TorchLensIOError
from torchlens._io.rehydrate import rehydrate_trace
from torchlens._io.scrub import scrub_for_save


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
        layers_to_save="all",
        save_arg_values=True,
        save_rng_states=True,
        save_code_context=True,
        random_seed=0,
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

    first_saved_layer = next(layer for layer in restored.layer_list if layer.has_saved_activation)
    assert isinstance(first_saved_layer.out, torch.Tensor)
    assert (
        restored.layer_logs[first_saved_layer.layer_label].ops[first_saved_layer.pass_index - 1]
        is first_saved_layer
    )

    first_arg_layer = next(
        layer
        for layer in restored.layer_list
        if layer.saved_args is not None and len(layer.saved_args) > 0
    )
    assert isinstance(first_arg_layer.saved_args[0], torch.Tensor)


def test_trace_setstate_default_fills_pre_sprint_state() -> None:
    """Missing version tags should warn and default-fill new S1 fields."""

    live_log = _build_live_log()
    old_state = live_log.__getstate__()
    old_state.pop("tlspec_version", None)
    old_state.pop("_activation_transform_repr", None)

    restored = Trace.__new__(Trace)
    with pytest.warns(DeprecationWarning):
        restored.__setstate__(old_state)

    assert restored.tlspec_version == TLSPEC_VERSION
    assert restored._activation_transform_repr is None


def test_trace_setstate_default_fills_grad_fn_param_refs() -> None:
    """A state dict predating ``_grad_fn_param_refs`` must not crash ``log_backward()``.

    Regression gate for cert8 BLOCKER-1: ``_grad_fn_param_refs`` is a real,
    populated ``FieldPolicy.KEEP`` field written unconditionally (no
    ``getattr``/``setdefault`` guard) by
    ``backends/torch/backward.py::_walk_and_hook_backward_graph``, but it was
    absent from ``MODEL_LOG_FIELD_ORDER`` and therefore from
    ``_MODEL_LOG_DEFAULT_FILL``. Loading a state dict that predates the field
    (an explicitly supported path -- ``read_tlspec_version`` only rejects
    *newer* versions, so pre-versioning/legacy states are accepted with a
    ``DeprecationWarning``) left the restored ``Trace`` without the attribute
    at all, so the first backward pass crashed with
    ``AttributeError: 'Trace' object has no attribute '_grad_fn_param_refs'``
    mid-way through backward-graph hook installation.

    ``_backward_gradfn_refs`` (a separate, pre-existing, out-of-scope
    ``__getstate__``/``__setstate__`` type-mismatch bug -- confirmed present
    on main independent of this fix) is also stripped here so this repro
    isolates the one field this fix addresses.
    """

    live_log = _build_live_log()
    old_state = live_log.__getstate__()
    assert "_grad_fn_param_refs" in old_state
    old_state.pop("_grad_fn_param_refs")
    old_state.pop("_backward_gradfn_refs", None)
    old_state.pop("_phase_timings", None)
    old_state.pop("_replay_arg_version_data_complete", None)

    restored = Trace.__new__(Trace)
    restored.__setstate__(old_state)

    assert restored._grad_fn_param_refs == {}
    assert restored._phase_timings == {}
    assert restored._replay_arg_version_data_complete is True

    out = restored[restored.output_layers[0]].out
    restored.log_backward(out.sum())

    # The field is genuinely populated by the backward walk, not just
    # present-but-empty -- confirms the fill value is functionally correct,
    # not merely attribute-error-silencing.
    assert restored._grad_fn_param_refs != {}
    assert all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in restored._grad_fn_param_refs.items()
    )


def test_trace_setstate_rejects_newer_io_versions() -> None:
    """Future portable versions must fail fast."""

    live_log = _build_live_log()
    future_state = live_log.__getstate__()
    future_state["tlspec_version"] = TLSPEC_VERSION + 1

    restored = Trace.__new__(Trace)
    with pytest.raises(TorchLensIOError):
        restored.__setstate__(future_state)


def test_plain_pickle_roundtrip_still_works() -> None:
    """Existing plain ``pickle.dump`` behavior should remain available."""

    live_log = trace_fn(
        _PlainPickleModel(),
        torch.randn(2, 4),
        layers_to_save="all",
        random_seed=0,
    )
    restored = pickle.loads(pickle.dumps(live_log))

    assert isinstance(restored, Trace)
    assert restored.model_class_name == live_log.model_class_name
    assert len(restored.layer_list) == len(live_log.layer_list)
    assert restored.layer_list[0].source_trace is restored
