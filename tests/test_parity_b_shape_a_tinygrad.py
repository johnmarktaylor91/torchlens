"""Input-shape guessing on tinygrad: every roster row, layer priors and state restoration.

Each guessed shape must run and validate on every config-built and random-init
tinygrad roster model; an any-shape function must report ambiguity; a
weight-pinned model must report a uniquely determined shape; ``Embedding``
gets integer indices; ``Tensor.training`` and the tinygrad RNG come back
unchanged.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.debug import guess_input_shape

tinygrad = pytest.importorskip("tinygrad")

from tests.conformance._rosters import tinygrad as tinygrad_roster  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

_ROWS = {
    row.family: row
    for row in (*tinygrad_roster.config_built_rows(), *tinygrad_roster.random_init_rows())
}
_OTHER_FRAMEWORKS = ("paddle", "keras", "mlx", "jax")


def _framework_free_is_ambiguous(model: Any) -> bool:
    """Whether a callable with no tinygrad reference meets another imported framework."""

    from torchlens.debug._shape_priors import framework_of, walk_references

    if any(framework_of(obj) == "tinygrad" for _path, obj in walk_references(model)[1:]):
        return False
    return any(name in sys.modules for name in _OTHER_FRAMEWORKS)


def _run(model: Any, example: Any) -> None:
    """One forward with every output realized."""

    output = model(example)
    for leaf in output if isinstance(output, (tuple, list)) else (output,):
        leaf.realize()


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
    _run(model, report.example_input)
    assert tl.validation.validate_forward_pass(model, [report.example_input]) is True


def test_any_shape_function_reports_ambiguity() -> None:
    """An elementwise function runs on several candidates and says so."""

    model, _example = _ROWS["fn_chain_branch"].builder()
    if _framework_free_is_ambiguous(model):
        pytest.skip("another prior framework is imported in this process")
    report = guess_input_shape(model)
    assert report.found and "ambiguous" in report.message


def test_weight_pinned_mlp_is_uniquely_determined() -> None:
    """A Linear-first function pins its input width."""

    model, example = _ROWS["fn_mlp_linear"].builder()
    report = guess_input_shape(model)
    assert report.found and report.shape == (1, int(example.shape[-1]))
    assert "uniquely determined" in report.message
    assert report.strategy == "tinygrad_layer_prior"


def test_cnn_side_is_searched_and_called_a_working_candidate() -> None:
    """The CNN head limits the side; the search finds one that fits and does not claim uniqueness."""

    model, example = _ROWS["mod_cnn"].builder()
    report = guess_input_shape(model)
    assert report.found and report.shape is not None
    # Sides 6 and 7 both pool to the 3x3 map the head expects; either is a valid answer.
    assert report.shape[:2] == tuple(example.shape)[:2] and report.shape[2] in (6, 7)
    assert "uniquely determined" not in report.message


def test_embedding_gets_integer_indices_within_the_vocabulary() -> None:
    """An Embedding-first model gets integer indices below its vocabulary size."""

    from tinygrad import dtypes, nn

    layer = nn.Embedding(10, 4)
    report = guess_input_shape(lambda x: layer(x))
    assert report.found and report.shape == (1, 8)
    assert report.dtype == dtypes.int32
    values = report.example_input.numpy()
    assert values.min() >= 0 and values.max() < 10


def test_training_flag_and_rng_are_restored() -> None:
    """Guessing leaves ``Tensor.training`` and the tinygrad RNG stream as they were."""

    from tinygrad import Tensor

    model, _example = _ROWS["mod_nested_residual"].builder()
    previous = Tensor.training
    Tensor.training = True
    try:
        Tensor.manual_seed(7)
        report = guess_input_shape(model)
        assert Tensor.training is True
        after = Tensor.rand(4).numpy()
        Tensor.manual_seed(7)
        expected = Tensor.rand(4).numpy()
    finally:
        Tensor.training = previous
    assert report.found
    np.testing.assert_array_equal(after, expected)
