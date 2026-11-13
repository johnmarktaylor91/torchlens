"""TensorFlow edits: the six built-in helpers live and on a finished trace.

Every built-in helper (``zero_ablate``, ``scale``, ``add``, ``clamp``,
``mean_ablate``, ``replace_with``) applied live with ``intervene=`` must give
the value worked out by hand with NumPy, and the edited capture must pass the
TensorFlow validator. The same helper applied to a finished trace with
``fork().do`` must equal the live capture and pass the backend replay check.
A value from another framework and a helper TensorFlow has no application for
are refused typed, and a finished-trace edit whose downstream cone reaches an
op the replay adapter cannot replay is refused naming that op.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends.tf.backend import TFBackend  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402
from torchlens.intervention._backend_replay import verify_backend_replay  # noqa: E402
from torchlens.intervention.errors import ReplayPreconditionError  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_X = np.asarray([[-1.5, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]], dtype=np.float32)
_REPLACEMENT = np.full((2, 4), 0.5, dtype=np.float32)


def _chain(v: Any) -> Any:
    """``relu(a * a - 2) + a`` with ``a = v + 1``: a fan-out and a repeated parent."""

    a = v + 1.0
    return tf.nn.relu(a * a - 2.0) + a


def _site_value() -> np.ndarray:
    """The rectifier's output for ``_X``, by hand."""

    a = _X + 1.0
    return np.maximum(a * a - 2.0, 0.0)


def _expected(edit: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The function's output with ``edit`` applied to the rectifier, by hand."""

    return edit(_site_value()) + (_X + 1.0)


_HELPERS: dict[str, tuple[Callable[[], Any], Callable[[np.ndarray], np.ndarray]]] = {
    "zero_ablate": (tl.zero_ablate, np.zeros_like),
    "scale": (lambda: tl.scale(3.0), lambda r: r * 3.0),
    "add": (lambda: tl.add(0.25), lambda r: r + 0.25),
    "clamp": (lambda: tl.clamp(low=0.5, high=1.0), lambda r: np.clip(r, 0.5, 1.0)),
    "mean_ablate_all": (
        lambda: tl.mean_ablate(dim=None),
        lambda r: np.zeros_like(r) + r.mean(),
    ),
    "mean_ablate_axis": (
        lambda: tl.mean_ablate(dim=1),
        lambda r: np.zeros_like(r) + r.mean(axis=1, keepdims=True),
    ),
    "mean_ablate_source": (
        lambda: tl.mean_ablate(tf.constant(_REPLACEMENT * 4.0), dim=None),
        lambda r: np.zeros_like(r) + 2.0,
    ),
    "replace_with": (
        lambda: tl.replace_with(tf.constant(_REPLACEMENT)),
        lambda r: _REPLACEMENT.copy(),
    ),
    "replace_with_callable": (
        lambda: tl.replace_with(lambda: tf.constant(_REPLACEMENT)),
        lambda r: _REPLACEMENT.copy(),
    ),
}


def _output(trace: Any) -> np.ndarray:
    """The trace's final output payload as a NumPy array."""

    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    return np.asarray((outputs or list(trace._layer_list))[-1].out)


def _relu(trace: Any) -> Any:
    """The rectifier op of a ``_chain`` trace."""

    return next(op for op in trace._layer_list if str(op.func_name).lower() == "relu")


@pytest.mark.parametrize("name", sorted(_HELPERS))
def test_live_helper_matches_hand_value_and_validates(name: str) -> None:
    helper, by_hand = _HELPERS[name]
    trace = tl.trace(
        _chain, tf.constant(_X), backend="tf", intervene=tl.when(tl.func("relu"), helper())
    )
    np.testing.assert_allclose(_output(trace), _expected(by_hand), rtol=1e-6, atol=1e-6)
    assert TFBackend().validate_trace(trace) is True


@pytest.mark.parametrize("name", sorted(_HELPERS))
def test_fork_helper_equals_live_capture_and_replays(name: str) -> None:
    helper, _by_hand = _HELPERS[name]
    clean = tl.trace(_chain, tf.constant(_X), backend="tf")
    live = tl.trace(
        _chain, tf.constant(_X), backend="tf", intervene=tl.when(tl.func("relu"), helper())
    )
    edited = clean.fork().do(_relu(clean).label, helper())
    np.testing.assert_array_equal(_output(edited), _output(live))
    assert verify_backend_replay(edited) == []
    assert edited.is_pristine is False


def test_fork_with_a_selector_equals_live_capture() -> None:
    clean = tl.trace(_chain, tf.constant(_X), backend="tf")
    live = tl.trace(
        _chain,
        tf.constant(_X),
        backend="tf",
        intervene=tl.when(tl.func("relu"), tl.scale(0.5)),
    )
    edited = clean.fork().do(tl.func("relu"), tl.scale(0.5))
    np.testing.assert_array_equal(_output(edited), _output(live))


def test_noop_fork_reproduces_every_payload() -> None:
    clean = tl.trace(_chain, tf.constant(_X), backend="tf")
    before = {op.label: np.asarray(op.out) for op in clean._layer_list}
    edited = clean.fork().do(_relu(clean).label, lambda out, *, ctx: out)
    for op in edited._layer_list:
        np.testing.assert_array_equal(np.asarray(op.out), before[op.label])


def test_live_helper_refuses_a_value_from_another_framework() -> None:
    import torch

    with pytest.raises(BackendUnsupportedError, match="TensorFlow tensor"):
        tl.trace(
            _chain,
            tf.constant(_X),
            backend="tf",
            intervene=tl.when(tl.func("relu"), tl.add(torch.ones(2, 4))),
        )


def test_live_mean_ablate_refuses_a_population_source() -> None:
    import torch

    from torchlens.intervention import population as make_population

    population = make_population(torch.ones(3, 2, 4), origin="test: three equal members")
    with pytest.raises(BackendUnsupportedError, match="Population"):
        tl.trace(
            _chain,
            tf.constant(_X),
            backend="tf",
            intervene=tl.when(tl.func("relu"), tl.mean_ablate(population, dim=None)),
        )


def test_fork_refuses_a_helper_without_a_tf_application() -> None:
    clean = tl.trace(_chain, tf.constant(_X), backend="tf")
    with pytest.raises(BackendUnsupportedError, match="no application for intervention helper"):
        clean.fork().do(_relu(clean).label, tl.add_noise(0.1, seed=0))


def _random_after_relu(v: Any) -> Any:
    """A random draw sized from the rectifier's output: the draw is in its cone."""

    r = tf.nn.relu(v)
    return r + tf.random.uniform(tf.shape(r), seed=3)


def test_fork_refuses_an_unreplayable_op_by_name() -> None:
    clean = tl.trace(_random_after_relu, tf.constant(_X), backend="tf")
    with pytest.raises(ReplayPreconditionError) as caught:
        clean.fork().do(_relu(clean).label, tl.scale(2.0))
    assert caught.value.fields["code"] == "backend_replay_op_unreplayable"
    assert "RandomUniform" in str(caught.value)
