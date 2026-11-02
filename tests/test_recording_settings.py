"""``Recording.options``: the read-only record of the settings a recording was made with."""

from __future__ import annotations

import dataclasses
import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.recording.types import Recording


class _Tiny(nn.Module):
    """Two-op model for settings readback."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer and a relu."""

        return torch.relu(self.fc(x))


def _double(t: torch.Tensor) -> torch.Tensor:
    """Activation transform used for readback identity."""

    return t * 2


def test_options_reads_back_every_door_as_live_values() -> None:
    """Capture knobs, recorder knobs and door keywords read back from one record."""

    keep = tl.func("relu")
    recording = tl.record(
        _Tiny(),
        torch.ones(1, 3),
        save=keep,
        capture=tl.options.CaptureOptions(
            activation_transform=_double, save_grads=True, lookback=2
        ),
        recording=tl.options.RecordingOptions(history_size=5, on_predicate_error="accumulate"),
    )
    settings = recording.options
    assert isinstance(settings, tl.types.RecordingSettings)
    assert settings.activation_transform is _double
    assert settings.save_grads is True
    assert settings.lookback == 2
    assert settings.history_size == 5
    assert settings.on_predicate_error == "accumulate"
    assert settings.keep_op is not None
    assert settings.stop_after is None
    assert settings.storage is None


def test_options_record_is_frozen_and_has_no_public_constructor() -> None:
    """The record cannot be mutated or built by keyword; neither can a Recording."""

    recording = tl.record(_Tiny(), torch.ones(1, 3), save=tl.func("relu"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        recording.options.history_size = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        tl.types.RecordingSettings(history_size=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Recording(records=[])  # type: ignore[call-arg]


def test_removed_repr_accessors_are_gone() -> None:
    """The string-only ``*_repr`` accessors left with no alias."""

    recording = tl.record(_Tiny(), torch.ones(1, 3), save=tl.func("relu"))
    for name in (
        "keep_op_repr",
        "save_grads_repr",
        "activation_transform_repr",
        "grad_transform_repr",
    ):
        assert not hasattr(recording, name)


def test_private_builder_defaults_and_replace() -> None:
    """``_build`` fills defaults; ``_replace`` copies with changes; pickling round-trips."""

    recording = Recording._build(
        records=[],
        by_pass={},
        by_label={},
        by_address={},
        path=None,
        num_ops=1,
        start_times=[],
        end_times=[],
        predicate_failures=[],
        num_predicate_failure_overflows=0,
        history_size=4,
    )
    assert recording.status == "complete"
    assert recording.grad_records == []
    assert recording.options.history_size is None
    copy = recording._replace(num_ops=2)
    assert copy.num_ops == 2 and recording.num_ops == 1
    assert pickle.loads(pickle.dumps(copy)).num_ops == 2
    with pytest.raises(KeyError):
        Recording._build(records=[])
