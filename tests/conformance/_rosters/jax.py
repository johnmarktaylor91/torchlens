"""JAX conformance models with independent NumPy value oracles.

Inventories are derived from the model definition in execution order, using
``Op.func_name``. A missing op is a capture defect, never a list edit.
Framework imports occur only inside builders. The ecosystem rows load pinned
pretrained Flax checkpoints into the NNX implementations in
``tests/conformance/_jax_checkpoints.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .. import _jax_checkpoints as checkpoints
from .._adapters import RosterModel
from . import ecosystem_enabled

__tl_layer__ = "L9"

_RNG = np.random.default_rng(0)
_X24 = _RNG.standard_normal((2, 4)).astype(np.float32)
_W44 = (_RNG.standard_normal((4, 4)) * 0.5).astype(np.float32)
_B4 = _RNG.standard_normal((4,)).astype(np.float32)
_IMG = _RNG.standard_normal((1, 6, 6, 1)).astype(np.float32)
# loop3_function's input, weight and bias are multiples of 1/8 or coarser: every product and
# partial sum of its matmul and bias add is exact in float32, so any summation order (the
# framework's kernel or NumPy's BLAS, which differ by CPU) gives the oracle's bits. Random
# values left a 7e-4 entry built from terms near 1, where one rounding exceeds the band.
_LOOP_X = np.array([[0.5, -1.0, 0.25, 0.75], [-0.5, 0.25, 1.0, -0.25]], dtype=np.float32)
_LOOP_W = np.array(
    [
        [0.5, -0.25, 0.25, 0.0],
        [0.25, 0.5, -0.5, 0.25],
        [-0.25, 0.25, 0.5, 0.5],
        [0.5, 0.0, -0.25, 0.25],
    ],
    dtype=np.float32,
)
_LOOP_B = np.array([0.25, -0.25, 0.125, 0.5], dtype=np.float32)


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
    """Build the branching raw function and its direct array input."""
    import jax
    import jax.numpy as jnp

    def chain(v: Any) -> Any:
        """Add, branch into a product, and apply ReLU."""
        a = v + 1.0
        m = a * v
        return jax.nn.relu(m)

    return chain, jnp.asarray(_X24)


def _build_multi() -> tuple[Any, Any]:
    """Build the two-output raw function and its direct array input."""
    import jax.numpy as jnp

    def multi(v: Any) -> Any:
        """Return a scalar reduction and a scaled array."""
        return v.sum(), v * 2.0

    return multi, jnp.asarray(_X24)


def _build_loop() -> tuple[Any, Any]:
    """Build the loop with closure constants to exercise the constants rule."""
    import jax
    import jax.numpy as jnp

    weights = jnp.asarray(_LOOP_W)
    bias = jnp.asarray(_LOOP_B)

    def loop(v: Any) -> Any:
        """Apply the same fixed affine transform three times."""
        for _ in range(3):
            v = jax.nn.relu(jnp.matmul(v, weights) + bias)
        return v

    return loop, jnp.asarray(_LOOP_X)


def _build_norm() -> tuple[Any, Any]:
    """Build the mean/variance/softmax raw function and its input."""
    import jax
    import jax.numpy as jnp

    def norm_softmax(v: Any) -> Any:
        """Normalize each row, then compute its softmax."""
        mean = jnp.mean(v, axis=-1, keepdims=True)
        variance = jnp.var(v, axis=-1, keepdims=True)
        normalized = (v - mean) / jnp.sqrt(variance + 1e-5)
        return jax.nn.softmax(normalized, axis=-1)

    return norm_softmax, jnp.asarray(_X24)


def _build_reshape() -> tuple[Any, Any]:
    """Build reshape/transpose/matmul with an in-forward ones tensor."""
    import jax.numpy as jnp

    def reshape_transpose(v: Any) -> Any:
        """Multiply the transposed reshape by literal ones."""
        return jnp.transpose(v.reshape(4, 2)) @ jnp.ones((4, 1))

    return reshape_transpose, jnp.asarray(_X24)


def _build_eqx_mlp() -> tuple[Any, Any]:
    """Build the seeded Equinox module root and a single example."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    key1, key2, _ = jax.random.split(jax.random.PRNGKey(0), 3)
    model = eqx.nn.Sequential(
        [
            eqx.nn.Linear(4, 3, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(3, 2, key=key2),
        ]
    )
    return model, jnp.asarray(_X24[0])


def _build_nnx_mlp() -> tuple[Any, Any]:
    """Build the seeded Flax NNX MLP root and its batched input."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    class MLP(nnx.Module):
        """Two affine layers with an intervening ReLU."""

        def __init__(self, rngs: Any) -> None:
            self.l1 = nnx.Linear(4, 3, rngs=rngs)
            self.l2 = nnx.Linear(3, 2, rngs=rngs)

        def __call__(self, v: Any) -> Any:
            """Apply the two affine layers."""
            return self.l2(jax.nn.relu(self.l1(v)))

    return MLP(nnx.Rngs(0)), jnp.asarray(_X24)


def _build_nnx_nested() -> tuple[Any, Any]:
    """Build the seeded nested residual Flax NNX module root."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    class Nested(nnx.Module):
        """An affine/GELU residual followed by layer normalization."""

        def __init__(self, rngs: Any) -> None:
            self.lin = nnx.Linear(4, 4, rngs=rngs)
            self.ln = nnx.LayerNorm(4, rngs=rngs)

        def __call__(self, v: Any) -> Any:
            """Normalize the residual block output."""
            return self.ln(v + jax.nn.gelu(self.lin(v)))

    return Nested(nnx.Rngs(0)), jnp.asarray(_X24)


def _build_conv_pool() -> tuple[Any, Any]:
    """Build the convolution/pool raw function and its NHWC image."""
    import jax
    import jax.numpy as jnp

    def conv_pool(v: Any) -> Any:
        """Convolve with literal weights, pool, flatten, and reduce."""
        kernel = jnp.ones((3, 3, 1, 2), jnp.float32) * 0.1
        h = jax.lax.conv_general_dilated(
            v, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        h = jax.nn.relu(h)
        h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
        return h.reshape(h.shape[0], -1).sum(-1)

    return conv_pool, jnp.asarray(_IMG)


def _loop_intermediates() -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute every repeated affine/ReLU value with NumPy."""
    h = _LOOP_X
    matmuls, additions, relus = [], [], []
    for _ in range(3):
        matmuls.append(h @ _LOOP_W)
        additions.append(matmuls[-1] + _LOOP_B)
        h = np.maximum(additions[-1], 0)
        relus.append(h)
    return {"input": _LOOP_X, "dot_general": matmuls, "add": additions, "max": relus}


def _norm_intermediates() -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute finite float intermediates in the captured variance/softmax decomposition."""
    total = _X24.sum(-1)
    mean = total[:, None] / np.float32(4)
    centered = _X24 - mean
    squared = centered * centered
    square_total = squared.sum(-1)
    variance = square_total[:, None] / np.float32(4)
    adjusted = variance + 1e-5
    denominator = np.sqrt(adjusted)
    normalized = centered / denominator
    maximum = normalized.max(-1)
    shifted = normalized - maximum[:, None]
    exponential = np.exp(shifted)
    exp_total = exponential.sum(-1)
    softmax = exponential / exp_total[:, None]
    # The guarded variance branch broadcasts NaN; omit that entire op kind and
    # convert_element_type, rather than publishing an incomplete occurrence list.
    # The predicate gt is boolean, outside these finite-float value oracles.
    return {
        "input": _X24,
        "reduce_sum": [total, total, square_total, exp_total],
        "div": [mean, mean, variance, normalized, softmax],
        "sub": [centered, np.asarray(4, dtype=np.float32), centered, shifted],
        "square": squared,
        "select_n": variance,
        "add": adjusted,
        "sqrt": denominator,
        "reduce_max": maximum,
        "max": maximum,
        "stop_gradient": maximum[:, None],
        "exp": exponential,
    }


def _conv_intermediates() -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute SAME convolution and VALID max pooling with NumPy only."""
    padded = np.pad(_IMG, ((0, 0), (1, 1), (1, 1), (0, 0)))
    height, width = _IMG.shape[1], _IMG.shape[2]
    taps = np.stack(
        [padded[:, i : i + height, j : j + width, :] for i in range(3) for j in range(3)]
    )
    convolved = np.repeat(taps.sum(axis=0) * np.float32(0.1), 2, axis=-1)
    relu = np.maximum(convolved, 0)
    pooled = relu.reshape(1, 3, 2, 3, 2, 2).max(axis=(2, 4))
    flattened = pooled.reshape(1, -1)
    return {
        "input": _IMG,
        "broadcast_in_dim": np.ones((3, 3, 1, 2), np.float32),
        "mul": np.full((3, 3, 1, 2), 0.1, np.float32),
        "conv_general_dilated": convolved,
        "max": relu,
        "reduce_window_max": pooled,
        "reshape": flattened,
        "reduce_sum": flattened.sum(-1),
    }


# mlp_eqx_vmap_function and shared_weight_eqx_vmap_function remain pending the
# constants rule: jax.vmap(mlp) closes over arrays and refuses today.
ROW_SPECS: tuple[dict[str, Any], ...] = (
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "chain_branch_function",
        _build_chain,
        ("input", "add", "mul", "max"),
        {
            "input": _X24,
            "add": _X24 + 1,
            "mul": (_X24 + 1) * _X24,
            "max": np.maximum((_X24 + 1) * _X24, 0),
        },
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "multi_output_function",
        _build_multi,
        ("input", "reduce_sum", "mul"),
        {"input": _X24, "reduce_sum": np.asarray(_X24.sum()), "mul": _X24 * 2},
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "loop3_function",
        _build_loop,
        # The closure weight and bias are ``constant`` source ops emitted before
        # the equations (shared constants rule).
        ("input", "constant", "constant") + ("dot_general", "broadcast_in_dim", "add", "max") * 3,
        _loop_intermediates(),
        has_constants=True,
    ),
    # Confirmed by a capture: fn_norm_softmax.
    _config(
        "norm_softmax_function",
        _build_norm,
        (
            "input",
            "reduce_sum",
            "broadcast_in_dim",
            "div",
            "reduce_sum",
            "broadcast_in_dim",
            "div",
            "sub",
            "square",
            "convert_element_type",
            "sub",
            "reduce_sum",
            "broadcast_in_dim",
            "div",
            "gt",
            "convert_element_type",
            "broadcast_in_dim",
            "select_n",
            "sub",
            "add",
            "sqrt",
            "div",
            "reduce_max",
            "max",
            "broadcast_in_dim",
            "stop_gradient",
            "sub",
            "exp",
            "reduce_sum",
            "broadcast_in_dim",
            "div",
        ),
        _norm_intermediates(),
        has_constants=True,
    ),
    # Confirmed by a capture: fn_reshape_transpose.
    _config(
        "reshape_transpose_function",
        _build_reshape,
        ("input", "reshape", "transpose", "broadcast_in_dim", "dot_general"),
        {
            "input": _X24,
            "reshape": _X24.reshape(4, 2),
            "transpose": _X24.reshape(4, 2).T,
            "broadcast_in_dim": np.ones((4, 1), np.float32),
            "dot_general": _X24.reshape(4, 2).T @ np.ones((4, 1), np.float32),
        },
        has_constants=True,
    ),
    # Confirmed by a capture: mod_mlp_eqx_root.
    _config(
        "mlp_eqx_module",
        _build_eqx_mlp,
        ("input", "dot_general", "add", "max", "dot_general", "add"),
        {},
        random_init=True,
    ),
    # Confirmed by a capture: mod_mlp_nnx.
    _config(
        "mlp_nnx_module",
        _build_nnx_mlp,
        ("input", "dot_general", "reshape", "add", "max", "dot_general", "reshape", "add"),
        {},
        random_init=True,
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "nested_residual_nnx_module",
        _build_nnx_nested,
        (
            "input",
            "dot_general",
            "reshape",
            "add",
            "integer_pow",
            "mul",
            "add",
            "mul",
            "tanh",
            "add",
            "mul",
            "mul",
            "add",
            "square",
            "reduce_sum",
            "div",
            "reduce_sum",
            "div",
            "square",
            "sub",
            "max",
            "reshape",
            "reshape",
            "sub",
            "add",
            "rsqrt",
            "reshape",
            "mul",
            "mul",
            "reshape",
            "add",
        ),
        {},
        random_init=True,
    ),
    # inventory derived from the definition, not yet confirmed by a capture
    _config(
        "conv_pool_function",
        _build_conv_pool,
        (
            "input",
            "broadcast_in_dim",
            "mul",
            "conv_general_dilated",
            "max",
            "reduce_window_max",
            "reshape",
            "reduce_sum",
        ),
        _conv_intermediates(),
        has_constants=True,
    ),
    {
        "family": "bert_uncased_l2_h256_flax",
        "realism": "pretrained",
        "builder": checkpoints.build_bert,
        "checkpoint_evidence": checkpoints.BERT_L2_H256.evidence,
        "expected_op_kinds": None,
        "expected_intermediates": {},
        # Position ids (iota), the all-zero token types and the attention
        # scale are values no recorded op derives from an input or parameter.
        "has_constants": True,
        "random_init": False,
        "group": "ecosystem",
    },
    {
        "family": "vit_base_patch16_in21k_flax",
        "realism": "pretrained",
        "builder": checkpoints.build_vit,
        "checkpoint_evidence": checkpoints.VIT_B16_IN21K.evidence,
        "expected_op_kinds": None,
        "expected_intermediates": {},
        # The attention scale is a Python scalar no recorded op produces.
        "has_constants": True,
        "random_init": False,
        "group": "ecosystem",
    },
)

CONFIG_BUILT_FAMILIES: tuple[str, ...] = tuple(
    spec["family"] for spec in ROW_SPECS if spec["group"] == "config_built"
)
ECOSYSTEM_FAMILIES: tuple[str, ...] = tuple(
    spec["family"] for spec in ROW_SPECS if spec["group"] == "ecosystem"
)
#: Declared ecosystem families that cannot run yet (none: both checkpoints are pinned).
PARKED_ECOSYSTEM_FAMILIES: tuple[str, ...] = ()


def config_built_roster() -> tuple[RosterModel, ...]:
    """The per-change rows: no network, no claim."""
    return tuple(RosterModel(**spec) for spec in ROW_SPECS if spec["group"] == "config_built")


def ecosystem_roster() -> tuple[RosterModel, ...]:
    """The pinned pretrained rows that can run (parked families are declared, not run)."""
    parked: tuple[str, ...] = PARKED_ECOSYSTEM_FAMILIES
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
