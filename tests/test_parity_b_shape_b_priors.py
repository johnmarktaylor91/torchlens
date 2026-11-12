"""Input-shape priors for TensorFlow/Keras, MLX and JAX (Equinox, Flax NNX, raw functions).

For every config-built roster row of the three backends, the row's model gets
a prior candidate with no example input. The candidate (after a bounded
square-side search for convolution candidates, the step the neutral search
controller runs) must run, trace through ``tl.trace`` and reach the same
replay-validation state as the roster's own example input. Function roots
must report their candidates as undetermined (ambiguous), and a function that
names no framework must be claimed by no prior. The priors must restore model
and RNG state and never build an unbuilt model.
"""

from __future__ import annotations

import functools
import types
import warnings
from typing import Any

import pytest

from torchlens.debug._shape_priors import SearchRequest, ShapeCandidate, _fn_roots, resolve_priors

_PREFERRED = (224, 256, 384, 299, 128, 96, 64, 32, 28)
_MAX_SIDE = 64


def _request(batch_size: int = 2) -> SearchRequest:
    """The search's default arguments with the roster's batch size."""

    return SearchRequest(
        batch_size=batch_size,
        input_dtype=None,
        channels=None,
        spatial_rank="auto",
        seq_len=None,
        min_size=1,
        max_size=512,
        preferred_sizes=_PREFERRED,
    )


def _resized(candidate: ShapeCandidate, side: int) -> tuple[int, ...]:
    """The candidate's shape with every spatial axis set to ``side``."""

    return tuple(
        side if axis in candidate.spatial_axes else dim for axis, dim in enumerate(candidate.shape)
    )


def _working_example(priors: Any, model: Any, candidate: ShapeCandidate) -> Any:
    """The candidate's input, side-searched for convolution candidates (as the search does)."""

    shapes = [candidate.shape]
    if candidate.spatial_axes:
        shapes += [_resized(candidate, side) for side in range(candidate.min_side, _MAX_SIDE + 1)]
    last: Exception | None = None
    for shape in shapes:
        example = priors.make_input(shape, candidate.dtype, candidate.value_range, 0)
        try:
            priors.forward(model, example)
        except Exception as exc:  # noqa: BLE001 - try the next side
            last = exc
            assert priors.classify_error(exc).kind != "blocker", str(exc)
            continue
        return example
    raise AssertionError(f"no working input for {candidate}: {last!r}")


def _validation_state(model: Any, example: Any, backend: str) -> str:
    """Replay-validation state of a fresh capture of ``model`` on ``example``."""

    import torchlens as tl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, example, backend=backend)
        return str(trace.validation_replay_status.state)


def _check_row(row: Any, backend: str) -> None:
    """One roster row: claim, candidate, working input and validation parity."""

    model, roster_example = row.builder()
    resolved = resolve_priors(model)
    if resolved is None:
        assert isinstance(model, types.FunctionType)
        assert _fn_roots.referenced_values(model) == [], row.family
        return
    name, priors = resolved
    assert name == backend, row.family
    with priors.state_scope(model):
        found = list(priors.candidates(model, _request()))
        assert found, row.family
        example = _working_example(priors, model, found[0])
    if isinstance(model, types.FunctionType):
        assert not any(candidate.determined for candidate in found), row.family
    guessed = _validation_state(model, example, backend)
    assert guessed == _validation_state(model, roster_example, backend), row.family
    assert guessed in {"passed", "unverified"}, (row.family, guessed)


def _rows(module_name: str) -> list[Any]:
    """Config-built rows of one backend roster."""

    from tests.conformance._rosters import jax as jax_roster, mlx as mlx_roster, tf as tf_roster

    if module_name == "mlx":
        return list(mlx_roster.roster(include_ecosystem=False))
    return list({"tf": tf_roster, "jax": jax_roster}[module_name].config_built_roster())


# ---------------------------------------------------------------- neutral
def test_hypotheses_are_bounded_and_start_from_closed_over_widths() -> None:
    """Function hypotheses are capped and try the function's own widths first."""

    shapes = _fn_roots.hypotheses((7, 5), _request(batch_size=3))
    assert len(shapes) <= _fn_roots.MAX_HYPOTHESES
    assert shapes[:2] == [(3, 7), (3, 5)]
    assert all(dim <= 512 for shape in shapes for dim in shape)


def test_function_naming_no_framework_is_not_attributed() -> None:
    """A bare function that reads no framework name is claimed by no prior."""

    def plain(v: Any) -> Any:
        """Elementwise doubling with no framework name."""

        return v * 2.0

    assert _fn_roots.referenced_values(plain) == []
    assert resolve_priors(plain) is None


def test_framework_attribution_reads_globals_closure_and_partial_args() -> None:
    """Names, closure cells and bound partial arguments attribute a function."""

    fake = types.ModuleType("jax.numpy")

    def make(module: Any) -> Any:
        """A function that closes over ``module``."""

        def fn(v: Any) -> Any:
            """Reads the closed-over module."""

            return module.tanh(v)

        return fn

    assert _fn_roots.names_framework(make(fake), ("jax",))
    assert not _fn_roots.names_framework(make(fake), ("mlx",))
    bound = functools.partial(lambda params, v: v, {"w": fake})
    assert _fn_roots.names_framework(bound, ("jax",))


# ---------------------------------------------------------------- JAX
@pytest.mark.backend_jax
@pytest.mark.parametrize("index", range(9))
def test_jax_roster_row_guess_runs_and_validates(index: int) -> None:
    """Each JAX config-built row: the guessed input runs and validates as the roster's does."""

    pytest.importorskip("jax")
    pytest.importorskip("equinox")
    pytest.importorskip("flax.nnx")
    rows = _rows("jax")
    assert len(rows) == 9
    _check_row(rows[index], "jax")


@pytest.mark.backend_jax
def test_jax_partial_params_root_uses_parameter_shapes_and_stays_ambiguous() -> None:
    """``partial(fn, params)`` is probed by ``eval_shape`` from the params' widths."""

    pytest.importorskip("jax")
    import jax.numpy as jnp

    def fn(params: Any, x: Any) -> Any:
        """The JAX docs page's raw function."""

        return jnp.tanh(x @ params["w"] + params["b"])

    params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}
    name, priors = resolve_priors(fn) or ("", None)
    assert name == "jax" and priors is not None
    assert list(priors.candidates(fn, _request())) == []  # params unbound: one input cannot probe
    bound = functools.partial(fn, params)
    found = list(priors.candidates(bound, _request()))
    assert found and found[0].shape == (2, 3)
    assert not found[0].determined
    example = _working_example(priors, bound, found[0])
    assert _validation_state(fn, (params, example), "jax") == "passed"


@pytest.mark.backend_jax
def test_jax_nnx_state_and_modes_restored() -> None:
    """NNX RNG state, statistics and mode flags come back after the scope."""

    pytest.importorskip("flax.nnx")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    class Dropper(nnx.Module):
        """A linear layer followed by dropout that owns an RNG stream."""

        def __init__(self, rngs: Any) -> None:
            self.lin = nnx.Linear(4, 3, rngs=rngs)
            self.drop = nnx.Dropout(0.5, rngs=rngs)

        def __call__(self, v: Any) -> Any:
            """Apply the layer, then dropout."""

            return self.drop(self.lin(v))

    model = Dropper(nnx.Rngs(0))
    before = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(nnx.state(model))]
    deterministic = model.drop.deterministic
    _, priors = resolve_priors(model) or ("", None)
    assert priors is not None
    with priors.state_scope(model):
        model(jnp.ones((2, 4)))  # a real call advances the dropout stream
    after = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(nnx.state(model))]
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert model.drop.deterministic == deterministic


# ---------------------------------------------------------------- TF / Keras
@pytest.mark.tf_backend
@pytest.mark.parametrize("index", range(9))
def test_tf_roster_row_guess_runs_and_validates(index: int) -> None:
    """Each TF config-built row: the guessed input runs and validates as the roster's does."""

    pytest.importorskip("tensorflow")
    pytest.importorskip("keras")
    rows = _rows("tf")
    assert len(rows) == 9
    _check_row(rows[index], "tf")


@pytest.mark.tf_backend
def test_keras_unbuilt_model_is_refused_and_left_unbuilt() -> None:
    """An unbuilt Keras model gets a lazy reason and is never built by the prior."""

    pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    model = keras.Sequential([keras.layers.Dense(3), keras.layers.Dense(2)])
    _, priors = resolve_priors(model) or ("", None)
    assert priors is not None
    assert priors.lazy_reason(model) is not None
    assert list(priors.candidates(model, _request())) == []
    assert not model.built


@pytest.mark.tf_backend
def test_keras_declared_input_is_determined_and_state_restored() -> None:
    """``keras.Input`` pins the shape; non-trainable state returns after the scope."""

    pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    model = keras.Sequential(
        [keras.Input((5,)), keras.layers.BatchNormalization(), keras.layers.Dense(2)]
    )
    _, priors = resolve_priors(model) or ("", None)
    assert priors is not None
    (candidate,) = priors.candidates(model, _request())
    assert candidate.shape == (2, 5) and candidate.determined
    variable = model.non_trainable_variables[0]
    before = variable.numpy().copy()
    with priors.state_scope(model):
        variable.assign(before + 1.0)
    assert (variable.numpy() == before).all()


# ---------------------------------------------------------------- MLX
@pytest.mark.backend_mlx
@pytest.mark.parametrize("index", range(10))
def test_mlx_roster_row_guess_runs_and_validates(index: int) -> None:
    """Each MLX config-built row: the guessed input runs and validates as the roster's does."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    rows = _rows("mlx")
    assert len(rows) == 10
    _check_row(rows[index], "mlx")


@pytest.mark.backend_mlx
def test_mlx_training_flags_and_rng_key_restored() -> None:
    """MLX training flags and the global PRNG key come back after the scope."""

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    import numpy as np

    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5), nn.Linear(3, 2))
    model.train()
    key = np.array(mx.random.state[0])
    _, priors = resolve_priors(model) or ("", None)
    assert priors is not None
    with priors.state_scope(model):
        assert not model.training
        mx.random.uniform(shape=(3,))  # a draw advances the global key
    assert model.training and all(m.training for _, m in model.named_modules())
    assert np.array_equal(np.array(mx.random.state[0]), key)


@pytest.mark.backend_mlx
def test_mlx_conv_prior_is_channel_last() -> None:
    """``Conv2d`` weights ``(out, kh, kw, in)`` give an NHWC candidate with ``in`` channels."""

    pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=5), nn.ReLU())
    _, priors = resolve_priors(model) or ("", None)
    assert priors is not None
    (candidate,) = priors.candidates(model, _request())
    assert candidate.layout == "NHWC" and candidate.shape[-1] == 3
    assert candidate.spatial_axes == (1, 2) and candidate.min_side == 5
    assert not candidate.determined
