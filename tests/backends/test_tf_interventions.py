"""TensorFlow static-label intervention preview tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl
from torchlens.backends.tf.backend import TFBackend
from torchlens.backends.tf.interventions import TFInterventionSiteUnreachableError
from torchlens.errors import BackendUnsupportedError

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()


pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]


def _relu_plus_ten(x: Any) -> Any:
    """Run ``relu -> +10`` through wrapped and unwrapped entry points.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    Any
        Model output.
    """

    return tf.nn.relu(x) + 10.0


class DenseModel(keras.Model):
    """Deterministic Keras fixture with a post-module op."""

    def __init__(self) -> None:
        """Initialize deterministic parameters."""

        super().__init__(name="m")
        self.dense = keras.layers.Dense(
            2,
            name="dense",
            kernel_initializer=keras.initializers.Ones(),
            bias_initializer="zeros",
        )

    def call(self, x: Any) -> Any:
        """Run ``dense -> +1``.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Any
            Model output.
        """

        return self.dense(x) + 1.0


def test_tf_op_level_zero_ablation_changes_downstream_and_marks_site() -> None:
    """A fired op-level site substitutes downstream values and marks the site op."""

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(
        _relu_plus_ten,
        x,
        backend="tf",
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )

    relu_op = next(op for op in trace._layer_list if op.func_name == "Relu")
    add_op = next(op for op in trace._layer_list if op.func_name == "AddV2")
    assert relu_op.is_edited is True
    assert relu_op.edits and relu_op.edits[0]["backend"] == "tf"
    assert np.allclose(np.asarray(relu_op.out), [1.0, 0.0, 2.0])
    assert np.allclose(np.asarray(add_op.out), [10.0, 10.0, 10.0])


def test_tf_intervention_graph_stays_honest_by_construction() -> None:
    """Downstream records consume the recorded replacement op, not the site op."""

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(
        _relu_plus_ten,
        x,
        backend="tf",
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )

    add_op = next(op for op in trace._layer_list if op.func_name == "AddV2")
    zeros_op = next(op for op in trace._layer_list if op.func_name == "ZerosLike")
    # Torch parity: parents resolves through the CONDITIONAL label map -- a
    # single-pass referenced op (zeros_op, here) by its bare layer_label, a
    # multi-pass one by its pass-qualified label.
    assert tuple(add_op.parents) == (zeros_op.layer_label,)
    assert np.allclose(np.asarray(zeros_op.out), [0.0, 0.0, 0.0])


def test_tf_intervened_capture_still_passes_validation() -> None:
    """The intervened stream is a real executed stream, so replay stays green."""

    x = tf.constant([[1.0, -2.0], [0.5, 0.3]])
    trace = tl.trace(
        lambda t: tf.reduce_sum(tf.nn.relu(t * 2.0)),
        x,
        backend="tf",
        intervene=tl.when(tl.func("relu"), tl.scale(0.5)),
    )

    result = TFBackend().validate_trace(trace, validate_metadata=False)
    state = getattr(result, "state", None)
    assert result is True or state in {"passed", "unverified"}
    if state == "unverified":
        assert "0 failures" in getattr(result, "message", "")


def test_tf_module_boundary_intervention_substitutes_module_output() -> None:
    """A tl.module condition substitutes the Keras module-call output."""

    model = DenseModel()
    x = tf.ones((1, 3))
    model(x)
    baseline = tl.trace(model, x, backend="tf")
    trace = tl.trace(
        model,
        x,
        backend="tf",
        intervene=tl.when(tl.module("dense"), tl.scale(0.5)),
    )

    baseline_out = np.asarray(list(baseline._layer_list)[-1].out)
    intervened_out = np.asarray(list(trace._layer_list)[-1].out)
    assert np.allclose(baseline_out, [[4.0, 4.0]])
    assert np.allclose(intervened_out, [[2.5, 2.5]])


def test_tf_in_module_boundary_intervention_fires_on_containment() -> None:
    """A tl.in_module condition fires at contained module exits."""

    model = DenseModel()
    x = tf.ones((1, 3))
    model(x)
    trace = tl.trace(
        model,
        x,
        backend="tf",
        intervene=tl.when(tl.in_module("dense"), tl.zero_ablate()),
    )

    final_out = np.asarray(list(trace._layer_list)[-1].out)
    assert np.allclose(final_out, [[1.0, 1.0]])


def test_tf_unreachable_op_site_refuses_typed() -> None:
    """A selector matching only unwrapped captured ops refuses fail-closed."""

    x = tf.constant([1.0, -1.0, 2.0])

    def raw_op_add(value: Any) -> Any:
        # tf.raw_ops entries are generated copies of the op wrappers, not the
        # wrapped python functions, so their outputs never reach the wrap layer.
        return tf.raw_ops.AddV2(x=tf.nn.relu(value), y=tf.constant(10.0))

    with pytest.raises(TFInterventionSiteUnreachableError, match="never\\s+passed through"):
        tl.trace(
            raw_op_add,
            x,
            backend="tf",
            intervene=tl.when(tl.func("addv2"), tl.zero_ablate()),
        )


def test_tf_unmatched_site_is_a_no_op() -> None:
    """A selector matching nothing anywhere intervenes nowhere and passes."""

    x = tf.constant([1.0, -1.0, 2.0])
    # A selector matching nothing anywhere is a disclosed no-op (see
    # tests/test_zero_match_disclosures.py), not a silent one: it warns.
    with pytest.warns(UserWarning, match="fired at zero sites"):
        trace = tl.trace(
            _relu_plus_ten,
            x,
            backend="tf",
            intervene=tl.when(tl.func("nonexistent_op"), tl.zero_ablate()),
        )

    add_op = next(op for op in trace._layer_list if op.func_name == "AddV2")
    assert np.allclose(np.asarray(add_op.out), [11.0, 10.0, 12.0])


def test_tf_callable_action_substitutes() -> None:
    """A plain callable action runs as the replacement hook."""

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(
        _relu_plus_ten,
        x,
        backend="tf",
        intervene=tl.when(tl.func("relu"), lambda out: out + tf.ones_like(out)),
    )

    add_ops = [op for op in trace._layer_list if op.func_name == "AddV2"]
    assert np.allclose(np.asarray(add_ops[0].out), [2.0, 1.0, 3.0])
    assert np.allclose(np.asarray(add_ops[-1].out), [12.0, 11.0, 13.0])


def test_tf_value_dependent_condition_applies() -> None:
    """A callable condition runs per op on the live context and edits matched ops."""

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(
        _relu_plus_ten,
        x,
        backend="tf",
        intervene=tl.when(lambda ctx: str(ctx.func_name) == "Relu", tl.zero_ablate()),
    )
    add_op = next(op for op in trace._layer_list if op.func_name == "AddV2")
    assert np.allclose(np.asarray(add_op.out), [10.0, 10.0, 10.0])
    assert TFBackend().validate_trace(trace) is True


def test_tf_unsupported_helper_refuses_typed() -> None:
    """Helpers without a TensorFlow implementation refuse typed."""

    x = tf.constant([1.0, -1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="no TensorFlow implementation"):
        tl.trace(
            _relu_plus_ten,
            x,
            backend="tf",
            intervene=tl.when(tl.func("relu"), tl.mean_ablate(dim=None)),
        )


def test_tf_shape_changing_replacement_refuses_typed() -> None:
    """Shape-changing replacements refuse instead of corrupting the graph."""

    x = tf.constant([1.0, -1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="shape-preserving"):
        tl.trace(
            _relu_plus_ten,
            x,
            backend="tf",
            intervene=tl.when(tl.func("relu"), lambda out: tf.reshape(out, (1, 3))),
        )


def test_tf_halt_stops_inside_the_capture_path() -> None:
    """stop_after= halts the eager capture at the frontier op."""

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(_relu_plus_ten, x, backend="tf", stop_after=tl.func("relu"))
    assert trace.is_stopped is True
    assert not any(op.func_name == "AddV2" for op in trace._layer_list)


def test_tf_recipes_snapshot_dispatches() -> None:
    """facets= attaches a per-trace facet registry snapshot on both tf capture paths."""

    def recipe(record: Any) -> dict[str, Any]:
        return {"is_relu": "relu" in str(getattr(record, "label", ""))}

    x = tf.constant([1.0, -1.0, 2.0])
    trace = tl.trace(_relu_plus_ten, x, backend="tf", facets=[recipe])
    assert trace.facet_functions is not None
    assert any("recipe" in entry.public.name for entry in trace.facet_functions.functions)

    compiled = tf.function(lambda value: tf.reduce_sum(tf.nn.relu(value)))
    static = tl.trace(compiled, tf.ones((2, 2)), backend="tf", facets=recipe)
    assert static.facet_functions is not None

    with pytest.raises(BackendUnsupportedError, match="recipe callables") as info:
        tl.trace(_relu_plus_ten, x, backend="tf", facets=object())
    assert info.value.code == "trace_option_unsupported"


def test_tf_graph_only_capture_refuses_intervene() -> None:
    """Static FuncGraph capture cannot substitute into a frozen graph."""

    compiled = tf.function(lambda x: tf.reduce_sum(tf.nn.relu(x)))
    with pytest.raises(BackendUnsupportedError, match="eager live capture"):
        tl.trace(
            compiled,
            tf.ones((2, 2)),
            backend="tf",
            intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        )


def test_tf_module_sites_without_module_tree_refuse_typed() -> None:
    """Module-boundary conditions on a bare callable refuse typed."""

    with pytest.raises(BackendUnsupportedError, match="object-module attribution"):
        tl.trace(
            lambda x: tf.nn.relu(x),
            tf.ones((2,)),
            backend="tf",
            intervene=tl.when(tl.module("dense"), tl.zero_ablate()),
        )


def test_tf_intervene_with_grad_options_refuses_typed() -> None:
    """intervene= and grad_options= are mutually exclusive on tf."""

    from torchlens.backends.tf import GradOptions

    with pytest.raises(BackendUnsupportedError, match="does not combine"):
        tl.trace(
            lambda x: tf.reduce_sum(tf.nn.relu(x)),
            tf.ones((2,)),
            backend="tf",
            intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
            capture=tl.options.CaptureOptions(grad_options=GradOptions()),
        )


def test_tf_backward_direction_refuses_typed() -> None:
    """Backward-direction decisions refuse on the forward-only preview."""

    x = tf.constant([1.0, -1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="forward-only"):
        tl.trace(
            _relu_plus_ten,
            x,
            backend="tf",
            intervene=tl.when(tl.grad(tl.func("relu")), tl.zero_ablate()),
        )
