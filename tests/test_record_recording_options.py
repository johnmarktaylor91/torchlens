"""``tl.record`` takes its recorder knobs as ``recording=RecordingOptions(...)``.

The nine former flat keywords (``default_op``, ``default_module``, ``history_size``,
``include_source_events``, ``max_predicate_failures``, ``on_predicate_error``,
``on_forward_error``, ``default_grad``, ``postprocess``) ride the grouped option,
matching ``tl.trace``'s ``capture=CaptureOptions(...)``. The flat spellings are gone:
passing one fails like any unknown keyword.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArgumentTypeError

_MOVED = (
    ("default_op", True),
    ("default_module", True),
    ("history_size", 3),
    ("include_source_events", True),
    ("max_predicate_failures", 4),
    ("on_predicate_error", "accumulate"),
    ("on_forward_error", "return_partial"),
    ("default_grad", True),
    ("postprocess", "none"),
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU())


def test_recording_options_history_size_reaches_the_recorder() -> None:
    """``recording=RecordingOptions(history_size=...)`` is the value the recorder used."""

    recording = tl.record(
        _model(),
        torch.randn(1, 2),
        save=tl.func("relu"),
        recording=tl.options.RecordingOptions(history_size=3),
    )
    assert recording.options.history_size == 3
    assert recording.history_size == 3


def test_omitted_recording_options_keep_the_door_defaults() -> None:
    """Unset fields resolve exactly as an omitted ``recording=`` does."""

    x = torch.randn(1, 2)
    capture = tl.options.CaptureOptions(backward_ready=True)
    bare = tl.record(_model(), x, save=tl.func("relu"), capture=capture)
    grouped = tl.record(
        _model(),
        x,
        save=tl.func("relu"),
        capture=capture,
        recording=tl.options.RecordingOptions(history_size=8),
    )
    assert grouped.options.default_op == bare.options.default_op
    assert grouped.options.default_module == bare.options.default_module
    assert grouped.options.default_grad == bare.options.default_grad
    assert len(grouped) == len(bare)


@pytest.mark.parametrize(("keyword", "value"), _MOVED, ids=[name for name, _ in _MOVED])
def test_former_flat_keyword_is_refused(keyword: str, value: object) -> None:
    """Each moved keyword fails the way any unknown keyword fails."""

    with pytest.raises(TypeError, match=keyword):
        tl.record(_model(), torch.randn(1, 2), save=tl.func("relu"), **{keyword: value})


@pytest.mark.parametrize("field_name", ["keep_op", "intervene"])
def test_door_owned_fields_on_recording_options_refuse(field_name: str) -> None:
    """``keep_op`` and ``intervene`` are the door's ``save=``/``intervene=``, never a no-op."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.record(
            _model(),
            torch.randn(1, 2),
            save=tl.func("relu"),
            recording=tl.options.RecordingOptions(**{field_name: None}),
        )
    assert info.value.fields["code"] == "record_capture_option_unsupported"
    assert field_name in str(info.value)


def test_recording_must_be_recording_options() -> None:
    """A different options object as ``recording=`` refuses typed."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.record(
            _model(),
            torch.randn(1, 2),
            save=tl.func("relu"),
            recording=tl.options.CaptureOptions(),  # type: ignore[arg-type]
        )
    assert info.value.fields["code"] == "option_group_type_invalid"
