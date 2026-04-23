"""Regression tests for plain ``pickle.dump`` compatibility."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from torchlens import ModelLog, log_forward_pass


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


def _build_model_log(seed: int = 0) -> ModelLog:
    """Build a deterministic ``ModelLog`` for plain pickle tests.

    Parameters
    ----------
    seed:
        Random seed used for model initialization and input generation.

    Returns
    -------
    ModelLog
        Logged forward pass with saved activations.
    """

    torch.manual_seed(seed)
    model = _PlainPickleModel()
    inputs = torch.randn(2, 4)
    return log_forward_pass(model, inputs, layers_to_save="all", random_seed=seed)


def _first_saved_layer(model_log: ModelLog) -> Any:
    """Return the first saved layer from one model log.

    Parameters
    ----------
    model_log:
        Model log under test.

    Returns
    -------
    Any
        First saved layer-pass entry.
    """

    return next(layer for layer in model_log.layer_list if layer.has_saved_activations)


def test_plain_pickle_dump_and_load_still_work(tmp_path: Path) -> None:
    """Fresh ``ModelLog`` objects should still survive plain pickle round-trips."""

    model_log = _build_model_log()
    pickle_path = tmp_path / "model_log.pkl"

    with pickle_path.open("wb") as handle:
        pickle.dump(model_log, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    assert isinstance(restored, ModelLog)
    assert restored.model_name == model_log.model_name
    assert len(restored.layer_list) == len(model_log.layer_list)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_model_log is restored
    assert isinstance(_first_saved_layer(restored).activation, torch.Tensor)


def test_old_style_pickle_without_io_format_version_warns_and_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forged pre-sprint pickles should warn on load and keep accessors working."""

    model_log = _build_model_log()
    pickle_path = tmp_path / "old_style_model_log.pkl"
    original_getstate = ModelLog.__getstate__

    def _legacy_getstate(self: ModelLog) -> dict[str, Any]:
        """Return a forged pre-sprint pickle state for one ``ModelLog``.

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
        state.pop("io_format_version", None)
        return state

    monkeypatch.setattr(ModelLog, "__getstate__", _legacy_getstate)
    with pickle_path.open("wb") as handle:
        pickle.dump(model_log, handle)

    with pytest.warns(DeprecationWarning):
        with pickle_path.open("rb") as handle:
            restored = pickle.load(handle)

    assert isinstance(restored, ModelLog)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_model_log is restored
    assert isinstance(_first_saved_layer(restored).activation, torch.Tensor)


def test_plain_pickle_preserves_in_memory_activations(tmp_path: Path) -> None:
    """Plain pickles should keep activations resident rather than turning them into refs."""

    model_log = _build_model_log()
    source_layer = _first_saved_layer(model_log)
    assert isinstance(source_layer.activation, torch.Tensor)

    pickle_path = tmp_path / "in_memory_model_log.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(model_log, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    restored_layer = _first_saved_layer(restored)
    assert isinstance(restored_layer.activation, torch.Tensor)
    assert restored_layer.activation_ref is None
    assert torch.equal(restored_layer.activation, source_layer.activation)
