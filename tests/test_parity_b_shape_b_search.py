"""End-to-end input-shape guessing for TensorFlow/Keras, MLX and JAX roster rows.

``tl.debug.guess_input_shape`` runs the neutral search over the Keras, MLX and
JAX priors with no example input. For every config-built roster row the
report must be found, its example input must reach the same validation state
as the roster's own example, and a function root must never be reported as
uniquely determined. A ``functools.partial`` of a raw ``fn(params, x)``
function must report its ambiguity.
"""

from __future__ import annotations

import functools
import types
from typing import Any

import pytest

from tests.test_parity_b_shape_b_priors import _rows, _validation_state


def _check_search(row: Any, backend: str) -> None:
    """One roster row through the public guess: found, validated, honest about determination."""

    import torchlens as tl

    model, roster_example = row.builder()
    report = tl.debug.guess_input_shape(model)
    assert report.found, (row.family, report.reason, report.message)
    guessed = _validation_state(model, report.example_input, backend)
    assert guessed == _validation_state(model, roster_example, backend), row.family
    assert guessed in {"passed", "unverified"}, (row.family, guessed)
    if isinstance(model, (types.FunctionType, functools.partial)):
        assert "; uniquely determined by" not in report.message, (row.family, report.message)


@pytest.mark.backend_jax
@pytest.mark.parametrize("index", range(9))
def test_jax_roster_row_guessed_end_to_end(index: int) -> None:
    """Each JAX roster row (Equinox, NNX, raw functions) is guessed and validates."""

    pytest.importorskip("jax")
    pytest.importorskip("equinox")
    pytest.importorskip("flax.nnx")
    _check_search(_rows("jax")[index], "jax")


@pytest.mark.backend_jax
def test_jax_partial_params_guess_reports_ambiguity() -> None:
    """A partial of ``fn(params, x)`` gets a working shape the report calls ambiguous."""

    pytest.importorskip("jax")
    import jax.numpy as jnp

    import torchlens as tl

    def apply(params: dict[str, Any], x: Any) -> Any:
        """A dense layer over explicit parameters."""

        return jnp.tanh(x @ params["w"] + params["b"])

    params = {"w": jnp.ones((3, 5)), "b": jnp.zeros((5,))}
    report = tl.debug.guess_input_shape(functools.partial(apply, params))
    assert report.found, report.message
    assert report.shape is not None and report.shape[-1] == 3
    assert "; uniquely determined by" not in report.message
    assert "ambiguous" in report.message


@pytest.mark.tf_backend
@pytest.mark.parametrize("index", range(9))
def test_tf_roster_row_guessed_end_to_end(index: int) -> None:
    """Each TensorFlow/Keras roster row is guessed and validates."""

    pytest.importorskip("tensorflow")
    pytest.importorskip("keras")
    _check_search(_rows("tf")[index], "tf")


@pytest.mark.tf_backend
def test_keras_unbuilt_model_guess_fails_lazy_and_stays_unbuilt() -> None:
    """The public guess refuses an unbuilt Keras model and does not build it."""

    pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    import torchlens as tl

    model = keras.Sequential([keras.layers.Dense(4)])
    report = tl.debug.guess_input_shape(model)
    assert not report.found
    assert report.reason == "lazy_uninitialized"
    assert report.message.count("left untouched") == 1
    assert not model.built


@pytest.mark.backend_mlx
@pytest.mark.parametrize("index", range(10))
def test_mlx_roster_row_guessed_end_to_end(index: int) -> None:
    """Each MLX roster row is guessed and validates."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    _check_search(_rows("mlx")[index], "mlx")
