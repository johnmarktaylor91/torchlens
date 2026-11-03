"""tinygrad conformance roster: the probe's small models plus the rule plants.

Two groups (SPEC 4.4):

- ``config_built``: ten fixed-weight models (function chain with branches,
  multi-output, 3-step loop, norm plus softmax, reshape and transpose, MLP,
  shared-weight module, CNN, nested residual, constants plant). Every weight
  is a literal so no row consumes RNG; the ``random_init`` group below holds
  the random-init layers separately. ``expected_op_kinds`` is each
  model's independent op inventory as the PYTHON device records it, checked
  against the definition; ``expected_intermediates`` is computed here with
  NumPy only (FORKS F-N) and keyed by the recorded op kind.
- ``random_init``: ``tinygrad.nn`` Linear, Conv2d and LayerNorm with default
  init. Until realize-once capture lands the backend refuses them typed
  (``rng_state_in_graph``, FORKS F-K) and the scorecard shows the cells as
  unsupported.
- ``ecosystem``: two pinned pretrained families vendored from tinygrad's own
  ``examples/`` definitions (FORKS F-I, no new dependency). They run only
  under ``TORCHLENS_CONFORMANCE_ECOSYSTEM=1``: :func:`runner_rows` leaves
  them out otherwise, and their builders raise naming
  :data:`ECOSYSTEM_NOT_RUN_REASON` if called anyway.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .._adapters import RosterModel
from . import ecosystem_enabled
from ._tinygrad_inventories import _PYTHON_INVENTORIES

__tl_layer__ = "L9"

#: The named reason the ecosystem cells carry when the flag is off.
ECOSYSTEM_NOT_RUN_REASON = (
    "ecosystem rows need pretrained weights; set TORCHLENS_CONFORMANCE_ECOSYSTEM=1 "
    "with network access or a weights cache"
)

_INPUT = np.asarray([[-1.0, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]], dtype=np.float32)
_W44 = np.asarray(
    [
        [0.1, -0.2, 0.3, 0.05],
        [0.4, 0.5, -0.6, 0.15],
        [-0.7, 0.8, 0.9, -0.25],
        [1.0, -1.1, 1.2, 0.35],
    ],
    dtype=np.float32,
)
_B4 = np.asarray([0.0, 0.1, -0.1, 0.2], dtype=np.float32)
_W43 = np.asarray(
    [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [1.0, -1.1, 1.2]], dtype=np.float32
)
_B3 = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
_W32 = np.asarray([[1.0, 0.5], [-1.0, 0.5], [0.25, 2.0]], dtype=np.float32)
_B2 = np.asarray([0.05, -0.05], dtype=np.float32)
_CONV_W = (np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) - 8.5) / 10.0
_CONV_B = np.asarray([0.1, -0.1], dtype=np.float32)
_FC_W = (np.arange(54, dtype=np.float32).reshape(3, 18) - 26.5) / 100.0
_FC_B = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
_IMG = (np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6) - 17.5) / 17.5


#: Defaults of the optional :class:`RosterModel` fields ``_row`` accepts as keywords.
_ROW_DEFAULTS: dict[str, Any] = {
    "expected_op_kinds": None,
    "expected_intermediates": None,
    "has_constants": False,
    "random_init": False,
    "checkpoint_evidence": "",
}


def _row(
    family: str,
    builder: Callable[[], tuple[Any, Any]],
    *,
    group: str = "config_built",
    **fields: Any,
) -> RosterModel:
    """Build one roster row; ecosystem rows are the pretrained group.

    ``fields`` are the optional :class:`RosterModel` fields in
    :data:`_ROW_DEFAULTS` (``expected_op_kinds``, ``expected_intermediates``,
    ``has_constants``, ``random_init``, ``checkpoint_evidence``).
    """

    unknown = sorted(set(fields) - set(_ROW_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown roster row fields: {unknown}")
    values = {**_ROW_DEFAULTS, **fields}
    intermediates: Mapping[str, Any] | None = values.pop("expected_intermediates")
    return RosterModel(
        family=family,
        realism="pretrained" if group == "ecosystem" else "config_built",
        builder=builder,
        expected_intermediates=dict(intermediates or {}),
        group=group,
        **values,
    )


def _np_relu(value: np.ndarray) -> np.ndarray:
    """NumPy reference for the rectified linear unit."""

    return np.maximum(value, 0.0)


def _np_softmax(value: np.ndarray) -> np.ndarray:
    """NumPy reference for a numerically stable softmax over the last axis."""

    shifted = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- builders


def _build_chain_branch() -> tuple[Any, Any]:
    """The chain_branch row: add, multiply and relu as a plain function."""

    from tinygrad import Tensor

    def chain(value: Any) -> Any:
        """Add one, multiply by the input and rectify."""

        added = value + 1.0
        product = added * value
        return product.relu()

    return chain, Tensor(_INPUT)


def _build_multi_output() -> tuple[Any, Any]:
    """The multi_output row: a function returning a sum and a scaled copy."""

    from tinygrad import Tensor

    def multi(value: Any) -> Any:
        """Return the input's sum and twice the input."""

        return value.sum(), value * 2.0

    return multi, Tensor(_INPUT)


def _build_loop3() -> tuple[Any, Any]:
    """The loop3 row: three matmul, add, relu steps with closure weights."""

    from tinygrad import Tensor

    weight = Tensor(_W44.tolist())
    bias = Tensor(_B4.tolist())

    def loop(value: Any) -> Any:
        """Apply the shared affine-relu step three times."""

        for _ in range(3):
            value = (value.matmul(weight) + bias).relu()
        return value

    return loop, Tensor(_INPUT)


def _build_norm_softmax() -> tuple[Any, Any]:
    """The norm_softmax row: mean-variance normalisation then softmax."""

    from tinygrad import Tensor

    def norm_softmax(value: Any) -> Any:
        """Normalise over the last axis and apply softmax."""

        mean = value.mean(axis=-1, keepdim=True)
        var = ((value - mean) ** 2).mean(axis=-1, keepdim=True)
        normed = (value - mean) / (var + 1e-5).sqrt()
        return normed.softmax(axis=-1)

    return norm_softmax, Tensor(_INPUT)


def _build_reshape_transpose() -> tuple[Any, Any]:
    """The reshape_transpose row: reshape, transpose and a matmul with ones."""

    from tinygrad import Tensor

    def reshape_transpose(value: Any) -> Any:
        """Reshape to 4x2, transpose and multiply by a column of ones."""

        return value.reshape(4, 2).transpose(0, 1).matmul(Tensor.ones(4, 1))

    return reshape_transpose, Tensor(_INPUT)


def _linear(weight: np.ndarray, bias: np.ndarray) -> Any:
    """A ``tinygrad.nn.Linear`` with literal weights (no RNG consumed)."""

    from tinygrad import Tensor, nn

    layer = nn.Linear(weight.shape[0], weight.shape[1])
    layer.weight = Tensor(weight.T.tolist())
    layer.bias = Tensor(bias.tolist())
    return layer


def _build_mlp_linear() -> tuple[Any, Any]:
    """The mlp_linear row: two closure linear layers with a relu between."""

    from tinygrad import Tensor

    first = _linear(_W43, _B3)
    second = _linear(_W32, _B2)

    def mlp(value: Any) -> Any:
        """Apply the first layer, relu, then the second layer."""

        return second(first(value).relu())

    return mlp, Tensor(_INPUT)


def _build_shared_twice() -> tuple[Any, Any]:
    """The shared_twice row: one linear layer applied twice with a relu between."""

    from tinygrad import Tensor

    class Shared:
        """One linear layer applied twice."""

        def __init__(self) -> None:
            self.lin = _linear(_W44, _B4)

        def __call__(self, value: Any) -> Any:
            return self.lin(self.lin(value).relu())

    return Shared(), Tensor(_INPUT)


def _build_cnn() -> tuple[Any, Any]:
    """The cnn row: conv, relu, max-pool, flatten and a linear head."""

    from tinygrad import Tensor, nn

    class Cnn:
        """Conv, relu, max-pool, flatten, linear with literal weights."""

        def __init__(self) -> None:
            self.conv = nn.Conv2d(1, 2, 3, padding=1)
            self.conv.weight = Tensor(_CONV_W.tolist())
            self.conv.bias = Tensor(_CONV_B.tolist())
            self.fc = _linear(_FC_W.T, _FC_B)

        def __call__(self, value: Any) -> Any:
            hidden = self.conv(value).relu().max_pool2d(2)
            return self.fc(hidden.reshape(hidden.shape[0], -1))

    return Cnn(), Tensor(_IMG)


def _build_nested_residual() -> tuple[Any, Any]:
    """The nested_residual row: a residual gelu block followed by layer norm."""

    from tinygrad import Tensor, nn

    class Nested:
        """Residual gelu block followed by a layer norm (affine ones and zeros)."""

        def __init__(self) -> None:
            self.lin = _linear(_W44, _B4)
            self.ln = nn.LayerNorm(4)

        def __call__(self, value: Any) -> Any:
            return self.ln(value + self.lin(value).gelu())

    return Nested(), Tensor(_INPUT)


def _build_constants_plant() -> tuple[Any, Any]:
    """The constants_plant row: a closure weight plus a literal ``ones`` constant."""

    from tinygrad import Tensor

    weight = Tensor(_W44.tolist())

    def constants(value: Any) -> Any:
        """Multiply by the closure weight and add a literal ones row."""

        return value.matmul(weight) + Tensor.ones(1, 4)

    return constants, Tensor(_INPUT)


def _build_random_linear() -> tuple[Any, Any]:
    """The random_linear row: a seeded ``nn.Linear`` with random parameters."""

    from tinygrad import Tensor, nn

    Tensor.manual_seed(0)
    layer = nn.Linear(4, 3)
    return (lambda value: layer(value).relu()), Tensor(_INPUT)


def _build_random_conv() -> tuple[Any, Any]:
    """The random_conv row: a seeded ``nn.Conv2d`` with random parameters."""

    from tinygrad import Tensor, nn

    Tensor.manual_seed(0)
    layer = nn.Conv2d(1, 2, 3, padding=1)
    return (lambda value: layer(value).relu()), Tensor(_IMG)


def _build_random_layernorm() -> tuple[Any, Any]:
    """The random_layernorm row: a seeded ``nn.LayerNorm`` over the input."""

    from tinygrad import Tensor, nn

    Tensor.manual_seed(0)
    lin = nn.Linear(4, 4)
    norm = nn.LayerNorm(4)
    return (lambda value: norm(lin(value))), Tensor(_INPUT)


# ------------------------------------------------------- NumPy intermediates


def _intermediates_chain() -> dict[str, Any]:
    """Expected output of the chain_branch row."""

    added = _INPUT + 1.0
    return {"output": _np_relu(added * _INPUT)}


def _intermediates_multi() -> dict[str, Any]:
    """Expected outputs of the multi_output row."""

    return {"output": [np.asarray(_INPUT.sum(), dtype=np.float32), _INPUT * 2.0]}


def _intermediates_loop3() -> dict[str, Any]:
    """Expected output of the loop3 row."""

    value = _INPUT
    for _ in range(3):
        value = _np_relu(value @ _W44 + _B4)
    return {"output": value}


def _intermediates_norm_softmax() -> dict[str, Any]:
    """Expected output of the norm_softmax row."""

    mean = _INPUT.mean(axis=-1, keepdims=True)
    var = ((_INPUT - mean) ** 2).mean(axis=-1, keepdims=True)
    return {"output": _np_softmax((_INPUT - mean) / np.sqrt(var + 1e-5))}


def _intermediates_reshape_transpose() -> dict[str, Any]:
    """Expected output of the reshape_transpose row."""

    return {"output": _INPUT.reshape(4, 2).T @ np.ones((4, 1), dtype=np.float32)}


def _intermediates_mlp() -> dict[str, Any]:
    """Expected output of the mlp_linear row."""

    return {"output": _np_relu(_INPUT @ _W43 + _B3) @ _W32 + _B2}


def _intermediates_shared() -> dict[str, Any]:
    """Expected output of the shared_twice row."""

    return {"output": _np_relu(_INPUT @ _W44 + _B4) @ _W44 + _B4}


def _intermediates_cnn() -> dict[str, Any]:
    """Expected output of the cnn row, convolved by hand in NumPy."""

    padded = np.pad(_IMG, ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros((1, 2, 6, 6), dtype=np.float32)
    for out_channel in range(2):
        for row in range(6):
            for col in range(6):
                patch = padded[0, 0, row : row + 3, col : col + 3]
                conv[0, out_channel, row, col] = (patch * _CONV_W[out_channel, 0]).sum()
        conv[0, out_channel] += _CONV_B[out_channel]
    relu = _np_relu(conv)
    pooled = relu.reshape(1, 2, 3, 2, 3, 2).max(axis=(3, 5))
    return {"output": pooled.reshape(1, -1) @ _FC_W.T + _FC_B}


def _intermediates_nested() -> dict[str, Any]:
    """Expected output of the nested_residual row (tanh-approximate gelu)."""

    hidden = _INPUT @ _W44 + _B4
    gelu = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
    residual = _INPUT + gelu
    mean = residual.mean(axis=-1, keepdims=True)
    var = residual.var(axis=-1, keepdims=True)
    return {"output": (residual - mean) / np.sqrt(var + 1e-5)}


def _intermediates_constants() -> dict[str, Any]:
    """Expected output of the constants_plant row."""

    return {"output": _INPUT @ _W44 + np.ones((1, 4), dtype=np.float32)}


def _default_device() -> str | None:
    """tinygrad's default device name, or ``None`` when tinygrad is absent."""

    try:
        from tinygrad import Device
    except ImportError:
        return None
    return str(Device.DEFAULT).split(":", 1)[0].upper()


def _inventory(family: str) -> tuple[str, ...] | None:
    """The pinned inventory for ``family`` on the current device, else ``None``."""

    if _default_device() != "PYTHON":
        return None
    return _PYTHON_INVENTORIES.get(family)


_CONFIG_BUILT: tuple[
    tuple[str, Callable[[], tuple[Any, Any]], Callable[[], dict[str, Any]], bool], ...
] = (
    ("fn_chain_branch", _build_chain_branch, _intermediates_chain, False),
    ("fn_multi_output", _build_multi_output, _intermediates_multi, False),
    ("fn_loop3", _build_loop3, _intermediates_loop3, True),
    ("fn_norm_softmax", _build_norm_softmax, _intermediates_norm_softmax, False),
    ("fn_reshape_transpose", _build_reshape_transpose, _intermediates_reshape_transpose, True),
    ("fn_mlp_linear", _build_mlp_linear, _intermediates_mlp, True),
    ("mod_shared_twice", _build_shared_twice, _intermediates_shared, True),
    ("mod_cnn", _build_cnn, _intermediates_cnn, True),
    ("mod_nested_residual", _build_nested_residual, _intermediates_nested, True),
    ("fn_constants_plant", _build_constants_plant, _intermediates_constants, True),
)


def config_built_rows() -> tuple[RosterModel, ...]:
    """The per-change rows: ten fixed-weight models, none consuming RNG."""

    return tuple(
        _row(
            family,
            builder,
            expected_op_kinds=_inventory(family),
            expected_intermediates=intermediates(),
            has_constants=has_constants,
        )
        for family, builder, intermediates, has_constants in _CONFIG_BUILT
    )


def random_init_rows() -> tuple[RosterModel, ...]:
    """The random-init rows: default-init Linear, Conv2d and LayerNorm (refused until realize-once)."""

    return (
        _row("nn_linear_random_init", _build_random_linear, random_init=True),
        _row("nn_conv2d_random_init", _build_random_conv, random_init=True),
        _row("nn_layernorm_random_init", _build_random_layernorm, random_init=True),
    )


def _ecosystem_builder(family: str) -> Callable[[], tuple[Any, Any]]:
    """A builder that names the not-run reason until the ecosystem flag is on."""

    def build() -> tuple[Any, Any]:
        """Raise the not-run reason; the ecosystem row has no runnable definition yet."""

        raise RuntimeError(
            f"{family}: {ECOSYSTEM_NOT_RUN_REASON}"
            if not ecosystem_enabled()
            else f"{family}: vendored definition and weight loading not implemented yet"
        )

    return build


def ecosystem_rows() -> tuple[RosterModel, ...]:
    """Two pinned pretrained families from tinygrad's ``examples/`` (definitions vendored).

    The rows are declared with their checkpoint evidence; their builders
    run only under the ecosystem flag and are filled in when the weights
    path is wired (recorded in the packet report).
    """

    return (
        _row(
            "resnet18_tinygrad_examples",
            _ecosystem_builder("resnet18_tinygrad_examples"),
            group="ecosystem",
            checkpoint_evidence="torchvision resnet18-f37072fd.pth (sha256 f37072fd...)",
        ),
        _row(
            "gpt2_small_tinygrad_examples",
            _ecosystem_builder("gpt2_small_tinygrad_examples"),
            group="ecosystem",
            checkpoint_evidence="huggingface gpt2 (revision pinned in the ecosystem run)",
        ),
    )


def runner_rows() -> tuple[RosterModel, ...]:
    """The rows the conformance runner drives: ecosystem rows only under the flag."""

    rows = config_built_rows() + random_init_rows()
    return rows + ecosystem_rows() if ecosystem_enabled() else rows


def all_rows() -> tuple[RosterModel, ...]:
    """Every tinygrad roster row across the three groups."""

    return config_built_rows() + random_init_rows() + ecosystem_rows()
