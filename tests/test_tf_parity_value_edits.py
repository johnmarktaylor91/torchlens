"""TensorFlow value-dependent intervention conditions match torch on twin models.

A value-dependent condition is a callable (bare, or the condition of ``tl.when``)
evaluated per recorded op against the live ``RecordContext``. Each test builds the
same computation in TensorFlow and torch with shared weights, applies one condition
to both backends, and compares the downstream output. TensorFlow op callbacks are
read-only, so the edited TF op keeps its original recorded value (marked
``is_edited``) and the replacement is recorded as the next op; torch records the
edited value on the site op itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests.backends.conftest import tensorflow_backend_modules
from torchlens.backends.tf.backend import TFBackend
from torchlens.errors import BackendUnsupportedError
from torchlens.recording._exceptions import PredicateError

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]

_RNG = np.random.default_rng(7)
_W1 = _RNG.standard_normal((4, 3)).astype("float32")
_B1 = _RNG.standard_normal((3,)).astype("float32")
_W2 = _RNG.standard_normal((3, 2)).astype("float32")
_B2 = _RNG.standard_normal((2,)).astype("float32")
_X = _RNG.standard_normal((2, 4)).astype("float32")


def _keras_mlp() -> Any:
    """Keras ``Dense(3, relu) -> Dense(2)`` carrying the shared weights."""

    inputs = keras.Input(shape=(4,))
    hidden = keras.layers.Dense(3, activation="relu", name="hidden")(inputs)
    outputs = keras.layers.Dense(2, name="head")(hidden)
    model = keras.Model(inputs, outputs)
    model.get_layer("hidden").set_weights([_W1, _B1])
    model.get_layer("head").set_weights([_W2, _B2])
    return model


def _torch_mlp() -> Any:
    """Torch ``Linear(4, 3) -> relu -> Linear(3, 2)`` twin with the same weights."""

    import torch

    class _MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(4, 3)
            self.head = torch.nn.Linear(3, 2)
            with torch.no_grad():
                self.hidden.weight.copy_(torch.from_numpy(_W1.T.copy()))
                self.hidden.bias.copy_(torch.from_numpy(_B1))
                self.head.weight.copy_(torch.from_numpy(_W2.T.copy()))
                self.head.bias.copy_(torch.from_numpy(_B2))

        def forward(self, x: Any) -> Any:
            return self.head(torch.relu(self.hidden(x)))

    return _MLP()


def _tf_relu_and_final(trace: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the TF trace's relu output and final output."""

    relu = next(layer for layer in trace._layer_list if str(layer.func_name) == "Relu")
    return np.asarray(relu.out), np.asarray(trace._layer_list[-1].out)


def _torch_relu_and_final(trace: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the torch trace's relu output and final output."""

    relu = next(layer for layer in trace._layer_list if str(layer.func_name) == "relu")
    final = trace._layer_list[-1].out
    return relu.out.detach().numpy(), final.detach().numpy()


def _validation_state(trace: Any) -> str:
    """Run replay validation and return its state (``passed``, ``unverified``, ...)."""

    TFBackend().validate_trace(trace)
    return str(trace.validation_replay_status.state)


def _plain_keras_state() -> str:
    """Validation state of the unedited Keras capture (parameter reads are unverified)."""

    return _validation_state(tl.trace(_keras_mlp(), tf.constant(_X), backend="tf"))


def _is_hidden_relu(ctx: Any) -> bool:
    """Portable condition: the relu op whose output has three features."""

    return str(ctx.func_name).lower() == "relu" and tuple(ctx.shape or ())[-1:] == (3,)


def _trace_both(intervene: Any) -> tuple[Any, Any]:
    """Trace the Keras model and its torch twin under one ``intervene=`` value."""

    import torch

    tf_trace = tl.trace(_keras_mlp(), tf.constant(_X), backend="tf", intervene=intervene)
    torch_trace = tl.trace(_torch_mlp(), torch.from_numpy(_X), intervene=intervene)
    return tf_trace, torch_trace


def test_tf_bare_callable_condition_matches_torch_twin() -> None:
    """A bare callable returning an action edits the same op as torch, same values."""

    def predicate(ctx: Any) -> Any:
        return tl.scale(0.5) if _is_hidden_relu(ctx) else None

    tf_trace, torch_trace = _trace_both(predicate)
    tf_relu, tf_final = _tf_relu_and_final(tf_trace)
    torch_relu, torch_final = _torch_relu_and_final(torch_trace)
    np.testing.assert_allclose(torch_relu, 0.5 * tf_relu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(tf_final, torch_final, rtol=1e-5, atol=1e-6)
    plain_final = np.asarray(
        tl.trace(_keras_mlp(), tf.constant(_X), backend="tf")._layer_list[-1].out
    )
    assert not np.allclose(tf_final, plain_final)
    assert _validation_state(tf_trace) == _plain_keras_state()


def test_tf_when_with_callable_condition_matches_torch_twin() -> None:
    """``tl.when(callable, action)`` behaves like torch's value-dependent ``when``."""

    tf_trace, torch_trace = _trace_both(tl.when(_is_hidden_relu, tl.scale(-2.0)))
    tf_relu, tf_final = _tf_relu_and_final(tf_trace)
    torch_relu, torch_final = _torch_relu_and_final(torch_trace)
    assert np.any(tf_relu != 0.0)
    np.testing.assert_allclose(torch_relu, -2.0 * tf_relu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(tf_final, torch_final, rtol=1e-5, atol=1e-6)
    relu = next(layer for layer in tf_trace._layer_list if str(layer.func_name) == "Relu")
    assert relu.is_edited is True
    assert _validation_state(tf_trace) == _plain_keras_state()


def _failed_decisions(trace: Any) -> list[tuple[str, str]]:
    """Run validation and return ``(op type, reason)`` for every failed op decision."""

    TFBackend().validate_trace(trace)
    decisions = trace.validation_replay_status.decisions
    return sorted((d["func_name"], d["reason"]) for d in decisions if d["decision"] == "failed")


def _tf_flag_forward(x: Any) -> Any:
    """Branch on a scalar-bool tensor, the value-dependent control-flow case."""

    hidden = tf.nn.relu(x)
    flag = tf.reduce_any(hidden > 0.5)
    if flag:
        return hidden * 2.0
    return hidden - 1.0


def _torch_flag_module() -> Any:
    """Torch twin of :func:`_tf_flag_forward`."""

    import torch

    class _Flag(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            hidden = torch.relu(x)
            flag = (hidden > 0.5).any()
            if flag:
                return hidden * 2.0
            return hidden - 1.0

    return _Flag()


def _ablate_true_flag(ctx: Any) -> Any:
    """Zero a scalar-bool op output that is True (reads the live value)."""

    return tl.zero_ablate() if ctx.is_scalar_bool and ctx.bool_value else None


def test_tf_scalar_bool_value_condition_flips_branch_like_torch() -> None:
    """Editing a True branch flag to False takes the other branch, as on torch.

    The TF condition reads the live ``is_scalar_bool``/``bool_value`` fields. Torch's
    live predicate context leaves both fields unset, so the torch twin addresses the
    same op (``any``) with a selector; both runs must take the edited branch.
    """

    import torch

    values = np.array([0.2, 1.0, -3.0], dtype="float32")
    plain = tl.trace(_tf_flag_forward, tf.constant(values), backend="tf")
    np.testing.assert_allclose(np.asarray(plain._layer_list[-1].out), np.maximum(values, 0) * 2.0)

    tf_trace = tl.trace(
        _tf_flag_forward, tf.constant(values), backend="tf", intervene=_ablate_true_flag
    )
    torch_trace = tl.trace(
        _torch_flag_module(),
        torch.from_numpy(values),
        intervene=tl.when(tl.func("any"), tl.zero_ablate()),
    )
    expected = np.maximum(values, 0) - 1.0
    np.testing.assert_allclose(np.asarray(tf_trace._layer_list[-1].out), expected)
    np.testing.assert_allclose(torch_trace._layer_list[-1].out.detach().numpy(), expected)
    any_op = next(layer for layer in tf_trace._layer_list if str(layer.func_name) == "Any")
    assert any_op.is_edited is True
    # ``Any`` and the ``ZerosLike`` replacement have no replay classification yet
    # (docs/backends/tf.md, Known gaps): the edited trace fails validation on exactly
    # those ops and nothing else.
    zeros_gap = [("ZerosLike", "op_unclassified")]
    assert _failed_decisions(tf_trace) == sorted(_failed_decisions(plain) + zeros_gap)


def test_tf_value_condition_invalid_return_refuses_typed() -> None:
    """A bare callable returning a non-action refuses with torch's predicate code."""

    with pytest.raises(PredicateError) as excinfo:
        tl.trace(_keras_mlp(), tf.constant(_X), backend="tf", intervene=lambda ctx: 3)
    assert excinfo.value.fields["code"] == "predicate_return_invalid"


def test_tf_value_condition_mixed_with_module_clause_refuses_typed() -> None:
    """A multi-rule spec naming a module boundary has no op-site form; it refuses typed."""

    spec = tl.when(tl.module("hidden"), tl.zero_ablate()) & tl.when(_is_hidden_relu, tl.scale(2.0))
    with pytest.raises(BackendUnsupportedError) as excinfo:
        tl.trace(_keras_mlp(), tf.constant(_X), backend="tf", intervene=spec)
    assert excinfo.value.code == "trace_option_unsupported"
