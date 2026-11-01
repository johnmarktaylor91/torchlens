"""``Recording.output``: the model's return value, held in memory only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _DictOut(nn.Module):
    """Model returning a dict holding a tuple, to prove the object is returned as-is."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        """Return a nested container."""

        hidden = torch.relu(self.fc(x))
        return {"logits": hidden, "extra": (hidden.sum(), "tag")}


def _code(exc_info: pytest.ExceptionInfo[PayloadUnavailableError]) -> str:
    return str(exc_info.value.fields["code"])


def test_output_is_the_exact_returned_object() -> None:
    """The output is the model's own container, not a reconstruction."""

    model = _DictOut()
    x = torch.ones(1, 3)
    recording = tl.record(model, x, save=tl.func("relu"))
    output = recording.output
    assert isinstance(output, dict)
    assert output["extra"][1] == "tag"
    assert torch.equal(output["logits"], model(x)["logits"])


def test_record_returns_one_object() -> None:
    """``tl.record`` returns the Recording alone; ``return_output=`` is gone."""

    recording = tl.record(_DictOut(), torch.ones(1, 3), save=tl.func("relu"))
    assert isinstance(recording, tl.Recording)
    with pytest.raises(TypeError):
        tl.record(_DictOut(), torch.ones(1, 3), save=tl.func("relu"), return_output=True)  # type: ignore[call-arg]


def test_output_refuses_after_stop_after() -> None:
    """A capture that stop_after= ended has no output: typed refusal, never None."""

    recording = tl.record(
        _DictOut(), torch.ones(1, 3), save=lambda ctx: True, stop_after=tl.func("linear")
    )
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = recording.output
    assert _code(exc_info) == "output_capture_stopped"


def test_output_is_not_stored_with_saved_recordings(tmp_path: Path) -> None:
    """A recording loaded from disk, or unpickled, refuses with ``output_not_stored``."""

    bundle = tmp_path / "out.tlfast"
    live = tl.record(
        _DictOut(),
        torch.ones(1, 3),
        save=lambda ctx: True,
        storage=tl.options.StorageOptions(path=bundle, retain_in_memory=True),
    )
    assert isinstance(live.output, dict)
    loaded = tl.load(bundle)
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = loaded.output
    assert _code(exc_info) == "output_not_stored"
    # A live recording's capture events hold autograd nodes, so the pickling
    # contract is checked on the state pair itself: the output slot is dropped.
    restored = object.__new__(tl.Recording)
    restored.__setstate__(live.__getstate__())
    assert restored.num_ops == live.num_ops
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = restored.output
    assert _code(exc_info) == "output_not_stored"
