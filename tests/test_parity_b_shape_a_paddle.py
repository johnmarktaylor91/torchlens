"""Input-shape guessing on Paddle: every roster row, layer priors and state restoration.

Each guessed shape must run and validate on every config-built Paddle roster
model; an any-shape function must report ambiguity; a weight-pinned model
must report a uniquely determined shape; ``data_format``, ``Embedding`` and
recurrent priors must produce the layer's own layout; training flags and the
Paddle RNG must come back unchanged.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.debug import guess_input_shape

paddle = pytest.importorskip("paddle")

from tests.conformance._rosters import paddle as paddle_roster  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

_ROWS = {row.family: row for row in paddle_roster.roster(include_ecosystem=False)}
_OTHER_FRAMEWORKS = ("tinygrad", "keras", "mlx", "jax")


def _framework_free_is_ambiguous(model: Any) -> bool:
    """Whether a callable with no Paddle reference meets another imported framework."""

    from torchlens.debug._shape_priors import framework_of, walk_references

    if any(framework_of(obj) == "paddle" for _path, obj in walk_references(model)[1:]):
        return False
    return any(name in sys.modules for name in _OTHER_FRAMEWORKS)


@pytest.mark.parametrize("family", sorted(_ROWS))
def test_guessed_shape_runs_and_validates_on_every_roster_row(family: str) -> None:
    """The guessed example runs the model and passes forward-pass validation."""

    model, _example = _ROWS[family].builder()
    report = guess_input_shape(model)
    if _framework_free_is_ambiguous(model):
        assert report.reason == "unknown_entry" and "framework" in report.message
        return
    assert report.found, report.message
    assert report.example_input is not None and report.shape is not None
    assert tuple(report.example_input.shape) == report.shape
    with paddle.no_grad():
        model(report.example_input)
    assert tl.validation.validate_forward_pass(model, [report.example_input]) is True


def test_any_shape_function_reports_ambiguity() -> None:
    """An elementwise function runs on several candidates and says so."""

    model, _example = _ROWS["fn_chain_branch"].builder()
    report = guess_input_shape(model)
    assert report.found and "ambiguous" in report.message


def test_weight_pinned_mlp_is_uniquely_determined() -> None:
    """A Linear-first layer pins its input width."""

    model, example = _ROWS["mod_mlp"].builder()
    report = guess_input_shape(model)
    assert report.found and report.shape == (1, int(example.shape[-1]))
    assert "uniquely determined" in report.message
    assert report.strategy == "paddle_layer_prior"


def test_cnn_side_is_searched_and_called_a_working_candidate() -> None:
    """The CNN head limits the side; the search finds one that fits and does not claim uniqueness."""

    model, example = _ROWS["mod_cnn"].builder()
    report = guess_input_shape(model)
    assert report.found and report.shape is not None
    # Sides 6 and 7 both pool to the 3x3 map the head expects; either is a valid answer.
    assert report.shape[:2] == tuple(example.shape)[:2] and report.shape[2] in (6, 7)
    assert "uniquely determined" not in report.message


def test_channel_last_conv_uses_its_data_format() -> None:
    """An NHWC Conv2D gets a channel-last candidate with flexible spatial axes."""

    model = paddle.nn.Sequential(paddle.nn.Conv2D(3, 4, 3, data_format="NHWC"))
    report = guess_input_shape(model)
    assert report.found and report.shape == (1, 32, 32, 3)
    assert report.flexible_dims == (1, 2)
    assert "NHWC layout" in report.message


def test_embedding_gets_integer_indices_within_the_vocabulary() -> None:
    """An Embedding-first layer gets int64 indices below its vocabulary size."""

    report = guess_input_shape(paddle.nn.Embedding(10, 4))
    assert report.found and report.shape == (1, 8)
    assert report.dtype == paddle.int64
    values = report.example_input.numpy()
    assert values.min() >= 0 and values.max() < 10


def test_lstm_gets_a_batch_major_sequence() -> None:
    """A batch-major LSTM gets (batch, seq, input_size)."""

    report = guess_input_shape(paddle.nn.LSTM(4, 6))
    assert report.found and report.shape == (1, 8, 4)


def test_training_flags_and_rng_are_restored() -> None:
    """Guessing leaves training mode and the Paddle RNG stream as they were."""

    model, _example = _ROWS["mod_nested_residual"].builder()
    model.train()
    paddle.seed(7)
    report = guess_input_shape(model)
    after = paddle.rand([4]).numpy()
    paddle.seed(7)
    expected = paddle.rand([4]).numpy()
    assert report.found
    assert all(layer.training for layer in model.sublayers(include_self=True))
    np.testing.assert_array_equal(after, expected)


def test_device_argument_is_refused_for_paddle() -> None:
    """``device=`` stays a torch-only argument."""

    from torchlens._errors import ShapeInferenceError

    model, _example = _ROWS["mod_mlp"].builder()
    with pytest.raises(ShapeInferenceError, match="device"):
        guess_input_shape(model, device="cpu")
