"""TensorFlow conformance models with independent NumPy value oracles.

Inventories are derived from the model definition in execution order, using
``Op.func_name``. A missing op is a capture defect, never a list edit.
Framework imports and checkpoint downloads occur only inside builders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._adapters import RosterModel
from . import ecosystem_enabled

__tl_layer__ = "L9"

_RNG = np.random.default_rng(0)
_X24 = _RNG.standard_normal((2, 4)).astype(np.float32)
_W44 = (_RNG.standard_normal((4, 4)) * 0.5).astype(np.float32)
_B4 = _RNG.standard_normal((4,)).astype(np.float32)
_IMG = _RNG.standard_normal((1, 6, 6, 1)).astype(np.float32)


#: The boolean row traits ``_config`` accepts as keywords, all ``False`` by default.
_CONFIG_FLAGS = ("has_constants", "random_init")


def _config(
    family: str,
    builder: Callable[[], tuple[Any, Any]],
    ops: tuple[str, ...],
    intermediates: dict[str, np.ndarray | list[np.ndarray]],
    **flags: bool,
) -> dict[str, Any]:
    """Describe one config-built model without importing its framework.

    ``flags`` are the row traits in :data:`_CONFIG_FLAGS` (``has_constants``,
    ``random_init``); an unknown trait is a programming error.
    """
    unknown = sorted(set(flags) - set(_CONFIG_FLAGS))
    if unknown:
        raise ValueError(f"unknown roster row traits: {unknown}")
    return {
        "family": family,
        "realism": "config_built",
        "builder": builder,
        "checkpoint_evidence": "",
        "expected_op_kinds": ops,
        "expected_intermediates": intermediates,
        **{flag: flags.get(flag, False) for flag in _CONFIG_FLAGS},
        "group": "config_built",
    }


def _build_chain() -> tuple[Any, Any]:
    """Build the branching function root and its example input."""
    import tensorflow as tf

    def chain(v: Any) -> Any:
        """Add, branch into a product, and apply ReLU."""
        a = v + 1.0
        m = a * v
        return tf.nn.relu(m)

    return chain, tf.constant(_X24)


def _build_multi() -> tuple[Any, Any]:
    """Build the two-output function root and its example input."""
    import tensorflow as tf

    def multi(v: Any) -> Any:
        """Return a scalar reduction and a scaled array."""
        return tf.reduce_sum(v), v * 2.0

    return multi, tf.constant(_X24)


def _build_loop() -> tuple[Any, Any]:
    """Build three repeated affine/ReLU steps with closure constants."""
    import tensorflow as tf

    weights = tf.constant(_W44)
    bias = tf.constant(_B4)

    def loop(v: Any) -> Any:
        """Apply the same fixed affine transform three times."""
        for _ in range(3):
            v = tf.nn.relu(tf.matmul(v, weights) + bias)
        return v

    return loop, tf.constant(_X24)


def _build_norm() -> tuple[Any, Any]:
    """Build the moments/softmax function root and its example input."""
    import tensorflow as tf

    def norm_softmax(v: Any) -> Any:
        """Normalize each row, then compute its softmax."""
        mean, variance = tf.nn.moments(v, axes=[-1], keepdims=True)
        normalized = (v - mean) / tf.sqrt(variance + 1e-5)
        return tf.nn.softmax(normalized, axis=-1)

    return norm_softmax, tf.constant(_X24)


def _build_reshape() -> tuple[Any, Any]:
    """Build reshape/transpose/matmul with an in-forward ones tensor."""
    import tensorflow as tf

    def reshape_transpose(v: Any) -> Any:
        """Multiply the transposed reshape by literal ones."""
        return tf.matmul(tf.transpose(tf.reshape(v, [4, 2])), tf.ones([4, 1]))

    return reshape_transpose, tf.constant(_X24)


def _build_mlp() -> tuple[Any, Any]:
    """Build the seeded two-layer Keras MLP and its example input."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [keras.Input((4,)), keras.layers.Dense(3, activation="relu"), keras.layers.Dense(2)]
    )
    return model, tf.constant(_X24)


def _build_shared() -> tuple[Any, Any]:
    """Build and warm a Keras model that calls one Dense layer twice."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)

    class Shared(keras.Model):
        """A shared affine layer with an intervening ReLU."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = keras.layers.Dense(4)

        def call(self, v: Any) -> Any:
            """Reuse the same layer on both sides of ReLU."""
            return self.lin(tf.nn.relu(self.lin(v)))

    model = Shared()
    x = tf.constant(_X24)
    model(x)
    return model, x


def _build_cnn() -> tuple[Any, Any]:
    """Build the seeded convolution/pool/flatten/Dense Keras model."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.Input((6, 6, 1)),
            keras.layers.Conv2D(2, 3, padding="same", activation="relu"),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(3),
        ]
    )
    return model, tf.constant(_IMG)


def _build_nested() -> tuple[Any, Any]:
    """Build and warm a seeded nested residual Keras model."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)

    class Nested(keras.Model):
        """A nested Dense/GELU block with residual layer normalization."""

        def __init__(self) -> None:
            super().__init__()
            self.block = keras.Sequential([keras.layers.Dense(4), keras.layers.Activation("gelu")])
            self.ln = keras.layers.LayerNormalization()

        def call(self, v: Any) -> Any:
            """Normalize the sum of the input and the nested block."""
            return self.ln(v + self.block(v))

    model = Nested()
    x = tf.constant(_X24)
    model(x)
    return model, x


def _build_mobilenet() -> tuple[Any, Any]:
    """Load Keras ImageNet MobileNetV2 weights only when called."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)
    model = keras.applications.MobileNetV2(weights="imagenet", input_shape=(96, 96, 3))
    image = np.random.default_rng(0).standard_normal((1, 96, 96, 3)).astype(np.float32)
    return model, tf.constant(image)


def _build_efficientnet() -> tuple[Any, Any]:
    """Load Keras ImageNet EfficientNetB0 weights only when called."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(0)
    # The pretrained classifier requires 224x224; the feature model accepts 96x96 (minimum 32).
    model = keras.applications.EfficientNetB0(
        weights="imagenet", input_shape=(96, 96, 3), include_top=False
    )
    image = np.random.default_rng(0).standard_normal((1, 96, 96, 3)).astype(np.float32)
    return model, tf.constant(image)


def _loop_intermediates() -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute every repeated affine/ReLU value with NumPy."""
    h = _X24
    matmuls, additions, relus = [], [], []
    for _ in range(3):
        matmuls.append(h @ _W44)
        additions.append(matmuls[-1] + _B4)
        h = np.maximum(additions[-1], 0)
        relus.append(h)
    return {"input": _X24, "MatMul": matmuls, "AddV2": additions, "Relu": relus}


def _norm_intermediates() -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute the captured moments decomposition and stable softmax."""
    mean = _X24.mean(-1, keepdims=True)
    centered = _X24 - mean
    squared = centered * centered
    variance = squared.mean(-1, keepdims=True)
    adjusted = variance + 1e-5
    denominator = np.sqrt(adjusted)
    normalized = centered / denominator
    exponential = np.exp(normalized - normalized.max(-1, keepdims=True))
    return {
        "input": _X24,
        "Mean": [mean, variance],
        "StopGradient": mean,
        "SquaredDifference": squared,
        "Sub": centered,
        "AddV2": adjusted,
        "Sqrt": denominator,
        "RealDiv": normalized,
        "Softmax": exponential / exponential.sum(-1, keepdims=True),
    }


ROW_SPECS: tuple[dict[str, Any], ...] = (
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "chain_branch_function",
        _build_chain,
        ("input", "AddV2", "Mul", "Relu"),
        {
            "input": _X24,
            "AddV2": _X24 + 1,
            "Mul": (_X24 + 1) * _X24,
            "Relu": np.maximum((_X24 + 1) * _X24, 0),
        },
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "multi_output_function",
        _build_multi,
        ("input", "Sum", "Mul"),
        {"input": _X24, "Sum": np.asarray(_X24.sum()), "Mul": _X24 * 2},
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "loop3_function",
        _build_loop,
        # The closure weight and bias are ``constant`` source ops, each emitted
        # when the first op that reads it is captured (shared constants rule).
        ("input", "constant", "MatMul", "constant", "AddV2", "Relu")
        + ("MatMul", "AddV2", "Relu") * 2,
        _loop_intermediates(),
        has_constants=True,
    ),
    # Confirmed by a capture: fn_norm_softmax.
    _config(
        "norm_softmax_function",
        _build_norm,
        (
            "input",
            "Mean",
            "StopGradient",
            "SquaredDifference",
            "Mean",
            "Sub",
            "AddV2",
            "Sqrt",
            "RealDiv",
            "Softmax",
        ),
        _norm_intermediates(),
    ),
    # Confirmed by a capture: fn_reshape_transpose.
    _config(
        "reshape_transpose_function",
        _build_reshape,
        ("input", "Reshape", "Transpose", "Fill", "MatMul"),
        {
            "input": _X24,
            "Reshape": _X24.reshape(4, 2),
            "Transpose": _X24.reshape(4, 2).T,
            "Fill": np.ones((4, 1), np.float32),
            "MatMul": _X24.reshape(4, 2).T @ np.ones((4, 1), np.float32),
        },
        has_constants=True,
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "mlp_module",
        _build_mlp,
        (
            "input",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
            "Relu",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
        ),
        {},
        random_init=True,
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "shared_weight_module",
        _build_shared,
        (
            "input",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
            "Relu",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
        ),
        {},
        random_init=True,
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "cnn_module",
        _build_cnn,
        (
            "input",
            "ReadVariableOp",
            "Conv2D",
            "ReadVariableOp",
            "Reshape",
            "Squeeze",
            "BiasAdd",
            "Relu",
            "MaxPool",
            "Shape",
            "Reshape",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
        ),
        {},
        random_init=True,
    ),
    # Confirmed by a capture: mod_nested_residual, including repeated op kinds.
    _config(
        "nested_residual_module",
        _build_nested,
        (
            "input",
            "ReadVariableOp",
            "MatMul",
            "ReadVariableOp",
            "BiasAdd",
            "Mul",
            "Neg",
            "Mul",
            "Erfc",
            "Mul",
            "AddV2",
            "Mean",
            "StopGradient",
            "SquaredDifference",
            "Mean",
            "ReadVariableOp",
            "Reshape",
            "ReadVariableOp",
            "Reshape",
            "AddV2",
            "Rsqrt",
            "Mul",
            "Neg",
            "Mul",
            "AddV2",
            "Mul",
            "AddV2",
        ),
        {},
        random_init=True,
    ),
    {
        "family": "mobilenet_v2_imagenet",
        "realism": "pretrained",
        "builder": _build_mobilenet,
        "checkpoint_evidence": (
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96.h5; sha256:<unverified>"
        ),
        "expected_op_kinds": None,
        "expected_intermediates": {},
        "has_constants": False,
        "random_init": True,
        "group": "ecosystem",
    },
    {
        "family": "efficientnet_b0_imagenet",
        "realism": "pretrained",
        "builder": _build_efficientnet,
        # Keras pins an MD5, not a SHA256; do not mislabel it as stronger evidence.
        "checkpoint_evidence": (
            "efficientnetb0_notop.h5; md5:50bc09e76180e00e4465e1a485ddc09d; sha256:<unverified>"
        ),
        "expected_op_kinds": None,
        "expected_intermediates": {},
        "has_constants": True,
        "random_init": True,
        "group": "ecosystem",
    },
)

CONFIG_BUILT_FAMILIES: tuple[str, ...] = tuple(
    spec["family"] for spec in ROW_SPECS if spec["group"] == "config_built"
)
ECOSYSTEM_FAMILIES: tuple[str, ...] = tuple(
    spec["family"] for spec in ROW_SPECS if spec["group"] == "ecosystem"
)


def config_built_roster() -> tuple[RosterModel, ...]:
    """The per-change rows: no network, no claim."""
    return tuple(RosterModel(**spec) for spec in ROW_SPECS if spec["group"] == "config_built")


def ecosystem_roster() -> tuple[RosterModel, ...]:
    """The pinned pretrained rows that can run (parked families are declared, not run)."""
    parked: tuple[str, ...] = ()
    return tuple(
        RosterModel(**spec)
        for spec in ROW_SPECS
        if spec["group"] == "ecosystem" and spec["family"] not in parked
    )


def roster() -> tuple[RosterModel, ...]:
    """The scored roster: config-built rows, plus ecosystem rows when switched on.

    Constructs no model and downloads nothing; builders do that when called.
    """
    if ecosystem_enabled():
        return config_built_roster() + ecosystem_roster()
    return config_built_roster()
