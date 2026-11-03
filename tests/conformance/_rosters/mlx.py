"""MLX conformance roster (SPEC 4.4): config_built rows and ecosystem rows.

The config_built rows are the probe models the backends-correct lane ran on
every preview, in the backend's own spelling. Each row declares its exact op
inventory in capture order (``Op.func_name`` per op, inputs and ``constant``
source ops included) and, where the model is deterministic, its expected
intermediate values computed with NumPy from the row's own definition. The
flags ``has_constants`` (a closure constant or a literal producer is in the
forward) and ``random_init`` (parameters are random-initialized lazily from a
seed, the realize-once check) select the c1 cells that apply.

Rows are described by :class:`RosterRow` and folded into the runner's
``RosterModel`` by :func:`roster`. Ecosystem rows join the roster only when
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1`` (the same switch the torch reference
roster uses for its pretrained rows); a row whose dependency is missing then
fails its cells rather than reading as passed.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .._adapters import RosterModel
from . import ECOSYSTEM_ENV

_RNG = np.random.default_rng(0)
X24 = _RNG.standard_normal((2, 4)).astype(np.float32)
W44 = (_RNG.standard_normal((4, 4)) * 0.5).astype(np.float32)
B4 = _RNG.standard_normal((4,)).astype(np.float32)
IMG = _RNG.standard_normal((1, 6, 6, 1)).astype(np.float32)  # NHWC, MLX's layout


@dataclasses.dataclass(frozen=True)
class RosterRow:
    """One backend roster row, backend-neutral.

    Parameters
    ----------
    family:
        Row name (the probe's model name).
    group:
        ``"config_built"`` or ``"ecosystem"``.
    builder:
        Returns ``(model, example_input)``; imports the framework lazily.
    expected_op_kinds:
        Exact ``Op.func_name`` sequence in capture order, or ``None`` for no
        inventory.
    expected_intermediates:
        Returns ``{op_kind: array or [arrays]}`` computed with NumPy, or
        ``None`` when the row has none to declare.
    has_constants:
        The forward uses a closure constant or a literal producer.
    random_init:
        Parameters are random-initialized lazily from a seed.
    checkpoint_evidence:
        Pinned checkpoint identity for ecosystem rows.
    not_run_reason:
        Why an ecosystem row cannot run here, or ``None`` when it can.
    """

    family: str
    group: str
    builder: Callable[[], tuple[Any, Any]]
    expected_op_kinds: tuple[str, ...] | None = None
    expected_intermediates: Callable[[], dict[str, Any]] | None = None
    has_constants: bool = False
    random_init: bool = False
    checkpoint_evidence: str = ""
    not_run_reason: str | None = None


def _relu(a: np.ndarray) -> np.ndarray:
    """NumPy reference for the rectified linear unit."""

    return np.maximum(a, 0)


def _softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy reference for a numerically stable softmax along ``axis``."""

    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _x() -> Any:
    """The shared 2x4 input as an MLX array."""

    import mlx.core as mx

    return mx.array(X24)


# ----------------------------------------------------------------- functions
def _build_chain_branch() -> tuple[Any, Any]:
    """The chain_branch row: add, multiply and maximum as a plain function."""

    import mlx.core as mx

    def chain(v: Any) -> Any:
        """Add one, multiply by the input and rectify."""

        a = v + 1.0
        m = a * v
        return mx.maximum(m, 0.0)

    return chain, _x()


def _chain_branch_intermediates() -> dict[str, Any]:
    """Expected intermediates of the chain_branch row."""

    a = X24 + 1
    return {"add": a, "multiply": a * X24, "maximum": _relu(a * X24)}


def _build_multi_output() -> tuple[Any, Any]:
    """The multi_output row: a function returning a sum and a scaled copy."""

    def multi(v: Any) -> Any:
        """Return the input's sum and twice the input."""

        return v.sum(), v * 2.0

    return multi, _x()


def _build_loop3() -> tuple[Any, Any]:
    """The loop3 row: three matmul, add, maximum steps with closure weights."""

    import mlx.core as mx

    weight = mx.array(W44)
    bias = mx.array(B4)

    def loop(v: Any) -> Any:
        """Apply the shared affine-relu step three times."""

        for _ in range(3):
            v = mx.maximum(mx.matmul(v, weight) + bias, 0.0)
        return v

    return loop, _x()


def _loop3_intermediates() -> dict[str, Any]:
    """Expected intermediates of the loop3 row, one list per op kind."""

    h = X24
    matmuls, adds, relus = [], [], []
    for _ in range(3):
        matmuls.append(h @ W44)
        adds.append(matmuls[-1] + B4)
        h = _relu(adds[-1])
        relus.append(h)
    return {"matmul": matmuls, "add": adds, "maximum": relus}


def _build_norm_softmax() -> tuple[Any, Any]:
    """The norm_softmax row: mean-variance normalisation then softmax."""

    import mlx.core as mx

    def norm_softmax(v: Any) -> Any:
        """Normalise over the last axis and apply softmax."""

        mu = mx.mean(v, axis=-1, keepdims=True)
        var = mx.var(v, axis=-1, keepdims=True)
        n = (v - mu) / mx.sqrt(var + 1e-5)
        return mx.softmax(n, axis=-1)

    return norm_softmax, _x()


def _norm_softmax_intermediates() -> dict[str, Any]:
    """Expected intermediates of the norm_softmax row."""

    mu = X24.mean(-1, keepdims=True)
    var = X24.var(-1, keepdims=True)
    n = (X24 - mu) / np.sqrt(var + 1e-5)
    return {"mean": mu, "var": var, "softmax": _softmax(n)}


def _build_reshape_transpose() -> tuple[Any, Any]:
    """The reshape_transpose row: reshape, transpose and a matmul with ones."""

    import mlx.core as mx

    def reshape_t(v: Any) -> Any:
        """Reshape to 4x2, transpose and multiply by a column of ones."""

        return mx.transpose(v.reshape(4, 2)) @ mx.ones((4, 1))

    return reshape_t, _x()


def _reshape_transpose_intermediates() -> dict[str, Any]:
    """Expected intermediates of the reshape_transpose row."""

    return {
        "reshape": X24.reshape(4, 2),
        "transpose": X24.reshape(4, 2).T,
        "ones": np.ones((4, 1), np.float32),
        "matmul": X24.reshape(4, 2).T @ np.ones((4, 1), np.float32),
    }


def _build_where_clip() -> tuple[Any, Any]:
    """The where_clip row: a leaky relu by ``where`` followed by ``clip``."""

    import mlx.core as mx

    def where_clip(v: Any) -> Any:
        """Leaky-rectify with ``where`` and clip to [-1, 1]."""

        leaky = mx.where(v > 0, v, 0.1 * v)
        return mx.clip(leaky, -1.0, 1.0)

    return where_clip, _x()


def _where_clip_intermediates() -> dict[str, Any]:
    """Expected intermediates of the where_clip row."""

    leaky = np.where(X24 > 0, X24, np.float32(0.1) * X24)
    return {"greater": X24 > 0, "where": leaky, "clip": np.clip(leaky, -1.0, 1.0)}


# ------------------------------------------------------------------- modules
def _build_mlp() -> tuple[Any, Any]:
    """The mlp row: a seeded two-layer ``nn.Sequential`` perceptron."""

    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)), _x()


def _build_shared_twice() -> tuple[Any, Any]:
    """The shared_twice row: one linear layer applied twice with a relu between."""

    import mlx.core as mx
    import mlx.nn as nn

    class Shared(nn.Module):
        """One linear layer applied twice."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def __call__(self, v: Any) -> Any:
            return self.lin(nn.relu(self.lin(v)))

    mx.random.seed(1)
    return Shared(), _x()


def _build_cnn() -> tuple[Any, Any]:
    """The cnn row: conv, relu, max-pool, flatten and a linear head."""

    import mlx.core as mx
    import mlx.nn as nn

    class CNN(nn.Module):
        """Conv, relu, max-pool, flatten, linear."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(2 * 3 * 3, 3)

        def __call__(self, v: Any) -> Any:
            h = self.pool(nn.relu(self.conv(v)))
            return self.fc(h.reshape(h.shape[0], -1))

    mx.random.seed(2)
    return CNN(), mx.array(IMG)


def _build_nested_residual() -> tuple[Any, Any]:
    """The nested_residual row: a residual gelu block followed by layer norm."""

    import mlx.core as mx
    import mlx.nn as nn

    class Nested(nn.Module):
        """Residual gelu block followed by a layer norm."""

        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.GELU())
            self.ln = nn.LayerNorm(4)

        def __call__(self, v: Any) -> Any:
            return self.ln(v + self.block(v))

    mx.random.seed(3)
    return Nested(), _x()


CONFIG_BUILT: tuple[RosterRow, ...] = (
    RosterRow(
        "fn_chain_branch",
        "config_built",
        _build_chain_branch,
        ("input", "add", "multiply", "maximum"),
        _chain_branch_intermediates,
    ),
    RosterRow(
        "fn_multi_output",
        "config_built",
        _build_multi_output,
        ("input", "sum", "multiply"),
    ),
    RosterRow(
        "fn_loop3",
        "config_built",
        _build_loop3,
        (
            "input",
            "constant",
            "matmul",
            "constant",
            "add",
            "maximum",
            "matmul",
            "add",
            "maximum",
            "matmul",
            "add",
            "maximum",
        ),
        _loop3_intermediates,
        has_constants=True,
    ),
    RosterRow(
        "fn_norm_softmax",
        "config_built",
        _build_norm_softmax,
        ("input", "mean", "var", "subtract", "add", "sqrt", "divide", "softmax"),
        _norm_softmax_intermediates,
    ),
    RosterRow(
        "fn_reshape_transpose",
        "config_built",
        _build_reshape_transpose,
        ("input", "reshape", "transpose", "ones", "matmul"),
        _reshape_transpose_intermediates,
        has_constants=True,
    ),
    RosterRow(
        "fn_where_clip",
        "config_built",
        _build_where_clip,
        ("input", "greater", "multiply", "where", "clip"),
        _where_clip_intermediates,
    ),
    RosterRow(
        "mod_mlp",
        "config_built",
        _build_mlp,
        ("input", "linear", "relu", "linear"),
        random_init=True,
    ),
    RosterRow(
        "mod_shared_twice",
        "config_built",
        _build_shared_twice,
        ("input", "linear", "relu", "linear"),
        random_init=True,
    ),
    RosterRow(
        "mod_cnn",
        "config_built",
        _build_cnn,
        ("input", "conv2d", "relu", "max_pool2d", "reshape", "linear"),
        random_init=True,
    ),
    RosterRow(
        "mod_nested_residual",
        "config_built",
        _build_nested_residual,
        ("input", "linear", "gelu", "add", "layernorm"),
        random_init=True,
    ),
)


# ----------------------------------------------------------------- ecosystem
#: The MLX ecosystem families (FORKS F-I default). Both load through
#: ``mlx_lm`` (the ``mlx-lm`` distribution), a test dependency JMT has not yet
#: approved: until then the rows report ``not_run`` with that reason.
ECOSYSTEM_DEPENDENCY = "mlx-lm"
ECOSYSTEM_FAMILIES: tuple[tuple[str, str, str], ...] = (
    (
        "llama_class_mlx_lm",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-4bit (Hugging Face, mlx-lm format, 4-bit)",
    ),
    (
        "qwen_class_mlx_lm",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit (Hugging Face, mlx-lm format, 4-bit)",
    ),
)


def ecosystem_not_run_reason() -> str | None:
    """Why the ecosystem rows cannot run here, or ``None`` when they can."""

    if os.environ.get(ECOSYSTEM_ENV) != "1":
        return f"{ECOSYSTEM_ENV}!=1: weights are not fetched in this run"
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return f"parked_dependency:{ECOSYSTEM_DEPENDENCY} (not installed; JMT's call)"
    return None


def _build_mlx_lm(repo: str) -> Callable[[], tuple[Any, Any]]:
    """A builder loading the mlx-lm checkpoint ``repo`` with a short token prompt."""

    def build() -> tuple[Any, Any]:
        """Load the model and tokenizer, then encode the fixed prompt."""

        import mlx.core as mx
        from mlx_lm import load

        model, tokenizer = load(repo)
        tokens = mx.array(tokenizer.encode("TorchLens traces MLX."))[None, :]
        return model, tokens

    return build


ECOSYSTEM: tuple[RosterRow, ...] = tuple(
    RosterRow(
        family,
        "ecosystem",
        _build_mlx_lm(repo),
        None,
        None,
        checkpoint_evidence=evidence,
        not_run_reason=ecosystem_not_run_reason(),
    )
    for family, repo, evidence in ECOSYSTEM_FAMILIES
)


def roster(include_ecosystem: bool | None = None) -> tuple[RosterModel, ...]:
    """The MLX rows as the runner's ``RosterModel`` records.

    Parameters
    ----------
    include_ecosystem:
        Whether the ecosystem rows are included; ``None`` (the default)
        includes them when ``TORCHLENS_CONFORMANCE_ECOSYSTEM=1``.

    Returns
    -------
    tuple[RosterModel, ...]
        The config_built rows, then the ecosystem rows when included.
    """

    if include_ecosystem is None:
        include_ecosystem = os.environ.get(ECOSYSTEM_ENV) == "1"
    rows = CONFIG_BUILT + (ECOSYSTEM if include_ecosystem else ())
    return tuple(
        RosterModel(
            family=row.family,
            realism="config_built" if row.group == "config_built" else "pretrained",
            builder=row.builder,
            checkpoint_evidence=row.checkpoint_evidence,
            expected_op_kinds=row.expected_op_kinds,
            expected_intermediates=(
                row.expected_intermediates() if row.expected_intermediates else None
            ),
            has_constants=row.has_constants,
            random_init=row.random_init,
            group=row.group,
        )
        for row in rows
    )


__all__ = [
    "CONFIG_BUILT",
    "ECOSYSTEM",
    "ECOSYSTEM_DEPENDENCY",
    "ECOSYSTEM_ENV",
    "ECOSYSTEM_FAMILIES",
    "RosterRow",
    "ecosystem_not_run_reason",
    "roster",
]
