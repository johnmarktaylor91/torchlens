"""Regression tests for plain ``pickle.dump`` compatibility."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from torchlens import Trace, trace as trace_fn


class _PlainPickleModel(nn.Module):
    """Simple model used for plain pickle regression checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain pickle test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return torch.sin(x) + torch.cos(x)


def _build_trace(seed: int = 0) -> Trace:
    """Build a deterministic ``Trace`` for plain pickle tests.

    Parameters
    ----------
    seed:
        Random seed used for model initialization and input generation.

    Returns
    -------
    Trace
        Logged forward pass with saved outs.
    """

    torch.manual_seed(seed)
    model = _PlainPickleModel()
    inputs = torch.randn(2, 4)
    return trace_fn(model, inputs, layers_to_save="all", random_seed=seed)


def _first_saved_layer(trace: Trace) -> Any:
    """Return the first saved layer from one model log.

    Parameters
    ----------
    trace:
        Model log under test.

    Returns
    -------
    Any
        First saved layer-pass entry.
    """

    return next(layer for layer in trace.layer_list if layer.has_saved_outs)


def test_plain_pickle_dump_and_load_still_work(tmp_path: Path) -> None:
    """Fresh ``Trace`` objects should still survive plain pickle round-trips."""

    trace = _build_trace()
    pickle_path = tmp_path / "trace.pkl"

    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    assert isinstance(restored, Trace)
    assert restored.model_class_name == trace.model_class_name
    assert len(restored.layer_list) == len(trace.layer_list)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_trace is restored
    assert isinstance(_first_saved_layer(restored).out, torch.Tensor)


def test_old_style_pickle_without_io_format_version_warns_and_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forged pre-sprint pickles should warn on load and keep accessors working."""

    trace = _build_trace()
    pickle_path = tmp_path / "old_style_trace.pkl"
    original_getstate = Trace.__getstate__

    def _legacy_getstate(self: Trace) -> dict[str, Any]:
        """Return a forged pre-sprint pickle state for one ``Trace``.

        Parameters
        ----------
        self:
            Model log being pickled.

        Returns
        -------
        dict[str, Any]
            State missing the portable format version tag.
        """

        state = original_getstate(self)
        state.pop("tlspec_version", None)
        return state

    monkeypatch.setattr(Trace, "__getstate__", _legacy_getstate)
    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)

    with pytest.warns(DeprecationWarning):
        with pickle_path.open("rb") as handle:
            restored = pickle.load(handle)

    assert isinstance(restored, Trace)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_trace is restored
    assert isinstance(_first_saved_layer(restored).out, torch.Tensor)


def test_plain_pickle_preserves_in_memory_outs(tmp_path: Path) -> None:
    """Plain pickles should keep outs resident rather than turning them into refs."""

    trace = _build_trace()
    source_layer = _first_saved_layer(trace)
    assert isinstance(source_layer.out, torch.Tensor)

    pickle_path = tmp_path / "in_memory_trace.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    restored_layer = _first_saved_layer(restored)
    assert isinstance(restored_layer.out, torch.Tensor)
    assert restored_layer.out_ref is None
    assert torch.equal(restored_layer.out, source_layer.out)
