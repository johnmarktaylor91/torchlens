"""Torch's conformance roster: the reference rows every preview is held to.

The ``config_built`` rows mirror the cross-backend probe's small models
(function chain with a branch, multi-output, three-step loop, norm plus
softmax, reshape and transpose, MLP, shared-weight module, CNN, nested
residual) plus one row with a closure constant and a literal producer and
one row that draws randomness in its forward. Weights are fixed NumPy arrays
copied into the modules, so every expected intermediate is computed with
NumPy from the row's own definition, independent of torch and of TorchLens.

``expected_op_kinds`` is each row's independent op inventory in capture
order and torch naming (``Op.func_name``), input ops included. Each list was
derived from the model definition and checked against a capture; a missing
or extra op is a capture defect, never a list edit.

The ``ecosystem`` rows are pinned pretrained checkpoints; they run only when
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1`` (weights cached or network allowed).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .._adapters import RosterModel
from . import ecosystem_enabled

__tl_layer__ = "L9"

_RNG = np.random.default_rng(0)
X24 = _RNG.standard_normal((2, 4)).astype(np.float32)
W44 = (_RNG.standard_normal((4, 4)) * 0.5).astype(np.float32)
B4 = _RNG.standard_normal((4,)).astype(np.float32)
#: NCHW image for the CNN row.
IMG = _RNG.standard_normal((1, 1, 6, 6)).astype(np.float32)
W34 = (_RNG.standard_normal((3, 4)) * 0.5).astype(np.float32)
B3 = _RNG.standard_normal((3,)).astype(np.float32)
W23 = (_RNG.standard_normal((2, 3)) * 0.5).astype(np.float32)
B2 = _RNG.standard_normal((2,)).astype(np.float32)
CONV_W = (_RNG.standard_normal((2, 1, 3, 3)) * 0.5).astype(np.float32)
CONV_B = _RNG.standard_normal((2,)).astype(np.float32)
FC_W = (_RNG.standard_normal((3, 18)) * 0.3).astype(np.float32)
FC_B = _RNG.standard_normal((3,)).astype(np.float32)
LN_EPS = 1e-5


# ------------------------------------------------------------------ NumPy oracle
def _relu(a: np.ndarray) -> np.ndarray:
    """NumPy reference for the rectified linear unit."""

    return np.maximum(a, 0).astype(np.float32)


def _softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy reference for a numerically stable softmax along ``axis``."""

    shifted = np.exp(a - a.max(axis=axis, keepdims=True))
    return (shifted / shifted.sum(axis=axis, keepdims=True)).astype(np.float32)


def _gelu(a: np.ndarray) -> np.ndarray:
    """NumPy reference for the exact (erf) GELU."""

    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * a * (1.0 + erf(a / math.sqrt(2.0)))).astype(np.float32)


def _linear(a: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """NumPy reference for ``torch.nn.Linear`` (weight is [out, in])."""

    return (a @ weight.T + bias).astype(np.float32)


def _layer_norm(a: np.ndarray) -> np.ndarray:
    """NumPy reference for layer normalisation without affine parameters."""

    mean = a.mean(-1, keepdims=True)
    var = a.var(-1, keepdims=True)
    return ((a - mean) / np.sqrt(var + LN_EPS)).astype(np.float32)


def _conv2d_same3(image: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """3x3 convolution, stride 1, zero padding 1 (cross-correlation, as torch)."""

    batch, _channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((batch, weight.shape[0], height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            patch = padded[:, :, row : row + 3, col : col + 3]
            out[:, :, row, col] = np.tensordot(patch, weight, axes=([1, 2, 3], [1, 2, 3]))
    return (out + bias[None, :, None, None]).astype(np.float32)


def _max_pool2(a: np.ndarray) -> np.ndarray:
    """NumPy reference for a non-overlapping 2x2 max pool."""

    batch, channels, height, width = a.shape
    blocks = a.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return blocks.max(axis=(3, 5)).astype(np.float32)


# ------------------------------------------------------------------ models
def _tensor(array: np.ndarray) -> Any:
    """Copy a NumPy array into a torch tensor."""

    import torch

    return torch.from_numpy(array.copy())


def _set_linear(layer: Any, weight: np.ndarray, bias: np.ndarray) -> None:
    """Load fixed NumPy weights into a ``torch.nn.Linear`` layer."""

    import torch

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(bias))


def _build_chain_branch() -> tuple[Any, Any]:
    """The chain_branch row: add, multiply and relu in a module forward."""

    from torch import nn

    class ChainBranch(nn.Module):
        """Add one, multiply by the input and rectify."""

        def forward(self, v: Any) -> Any:
            """Add one, multiply by the input and rectify."""

            a = v + 1.0
            m = a * v
            return nn.functional.relu(m)

    return ChainBranch().eval(), _tensor(X24)


def _build_multi_output() -> tuple[Any, Any]:
    """The multi_output row: a module returning a sum and a scaled copy."""

    from torch import nn

    class MultiOutput(nn.Module):
        """Return the input's sum and twice the input."""

        def forward(self, v: Any) -> Any:
            """Return the input's sum and twice the input."""

            return v.sum(), v * 2.0

    return MultiOutput().eval(), _tensor(X24)


def _build_loop3() -> tuple[Any, Any]:
    """The loop3 row: three matmul, add, relu steps over buffer weights."""

    from torch import nn

    class Loop3(nn.Module):
        """Three affine-relu steps sharing one weight and bias buffer."""

        weight: Any
        bias: Any

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", _tensor(W44))
            self.register_buffer("bias", _tensor(B4))

        def forward(self, v: Any) -> Any:
            """Apply the shared affine-relu step three times."""

            import torch

            for _ in range(3):
                v = nn.functional.relu(torch.matmul(v, self.weight) + self.bias)
            return v

    return Loop3().eval(), _tensor(X24)


def _build_norm_softmax() -> tuple[Any, Any]:
    """The norm_softmax row: mean-variance normalisation then softmax."""

    from torch import nn

    class NormSoftmax(nn.Module):
        """Normalise over the last dimension and apply softmax."""

        def forward(self, v: Any) -> Any:
            """Normalise over the last dimension and apply softmax."""

            import torch

            mu = torch.mean(v, dim=-1, keepdim=True)
            var = torch.var(v, dim=-1, unbiased=False, keepdim=True)
            n = (v - mu) / torch.sqrt(var + 1e-5)
            return nn.functional.softmax(n, dim=-1)

    return NormSoftmax().eval(), _tensor(X24)


def _build_reshape_transpose() -> tuple[Any, Any]:
    """The reshape_transpose row: reshape, transpose and a matmul with ones."""

    from torch import nn

    class ReshapeTranspose(nn.Module):
        """Reshape to 4x2, transpose and multiply by a column of ones."""

        def forward(self, v: Any) -> Any:
            """Reshape to 4x2, transpose and multiply by a column of ones."""

            import torch

            return torch.matmul(torch.reshape(v, (4, 2)).transpose(0, 1), torch.ones(4, 1))

    return ReshapeTranspose().eval(), _tensor(X24)


def _build_mlp() -> tuple[Any, Any]:
    """The mlp row: a two-layer ``Sequential`` perceptron with fixed weights."""

    from torch import nn

    mlp = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    _set_linear(mlp[0], W34, B3)
    _set_linear(mlp[2], W23, B2)
    return mlp.eval(), _tensor(X24)


def _build_shared_twice() -> tuple[Any, Any]:
    """The shared_twice row: one linear layer applied twice with a relu between."""

    from torch import nn

    class SharedTwice(nn.Module):
        """One linear layer applied twice."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: Any) -> Any:
            """Apply the linear layer, relu, then the same layer again."""

            return self.lin(nn.functional.relu(self.lin(v)))

    model = SharedTwice()
    _set_linear(model.lin, W44, B4)
    return model.eval(), _tensor(X24)


def _build_cnn() -> tuple[Any, Any]:
    """The cnn row: conv, relu, max-pool, flatten and a linear head."""

    import torch
    from torch import nn

    class SmallCnn(nn.Module):
        """Conv, relu, max-pool, flatten, linear."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(18, 3)

        def forward(self, v: Any) -> Any:
            """Convolve, rectify, pool, flatten and project."""

            h = self.pool(nn.functional.relu(self.conv(v)))
            return self.fc(torch.flatten(h, 1))

    model = SmallCnn()
    _set_linear(model.conv, CONV_W, CONV_B)
    _set_linear(model.fc, FC_W, FC_B)
    return model.eval(), _tensor(IMG)


def _build_nested_residual() -> tuple[Any, Any]:
    """The nested_residual row: a residual gelu block followed by layer norm."""

    from torch import nn

    class NestedResidual(nn.Module):
        """Residual gelu block followed by a layer norm."""

        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.GELU())
            self.ln = nn.LayerNorm(4, eps=LN_EPS, elementwise_affine=False)

        def forward(self, v: Any) -> Any:
            """Add the block's output to the input and normalise."""

            return self.ln(v + self.block(v))

    model = NestedResidual()
    _set_linear(model.block[0], W44, B4)
    return model.eval(), _tensor(X24)


def _build_constants() -> tuple[Any, Any]:
    """A closure tensor (no parameter, no buffer) and a literal producer."""

    import torch
    from torch import nn

    closure_weight = _tensor(W44)

    class ClosureConstant(nn.Module):
        """Matmul with a closure weight plus a literal ``full`` constant."""

        def forward(self, v: Any) -> Any:
            """Multiply by the closure weight and add the literal shift."""

            shift = torch.full((1, 4), 0.5)
            return torch.matmul(v, closure_weight) + shift

    return ClosureConstant().eval(), _tensor(X24)


def _build_random_forward() -> tuple[Any, Any]:
    """A forward that draws from the global RNG (dropout in training mode)."""

    from torch import nn

    class RandomForward(nn.Module):
        """Scale the input and apply training-mode dropout."""

        def forward(self, v: Any) -> Any:
            """Scale by two and apply dropout with ``training=True``."""

            return nn.functional.dropout(v * 2.0, p=0.5, training=True)

    return RandomForward(), _tensor(X24)


# ------------------------------------------------------------------ expectations
def _expected_intermediates() -> dict[str, dict[str, Any]]:
    """NumPy expectations per family, keyed by op kind (capture order per kind)."""

    chain_a = X24 + 1.0
    loop_matmuls, loop_relus = [], []
    hidden = X24
    for _ in range(3):
        loop_matmuls.append((hidden @ W44).astype(np.float32))
        hidden = _relu(loop_matmuls[-1] + B4)
        loop_relus.append(hidden)
    mu = X24.mean(-1, keepdims=True)
    var = X24.var(-1, keepdims=True)
    normed = ((X24 - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
    mlp_hidden = _linear(X24, W34, B3)
    shared_first = _linear(X24, W44, B4)
    shared_second = _linear(_relu(shared_first), W44, B4)
    conv = _conv2d_same3(IMG, CONV_W, CONV_B)
    pooled = _max_pool2(_relu(conv))
    block = _gelu(_linear(X24, W44, B4))
    return {
        "chain_branch": {"relu": _relu(chain_a * X24)},
        "multi_output": {"sum": np.asarray(X24.sum(), dtype=np.float32)},
        "loop3": {"matmul": loop_matmuls, "relu": loop_relus},
        "norm_softmax": {"softmax": _softmax(normed)},
        "reshape_transpose": {
            "matmul": (X24.reshape(4, 2).T @ np.ones((4, 1), np.float32)).astype(np.float32)
        },
        "mlp": {"linear": [mlp_hidden, _linear(_relu(mlp_hidden), W23, B2)]},
        "shared_twice": {"linear": [shared_first, shared_second]},
        "cnn": {
            "conv2d": conv,
            "max_pool2d": pooled,
            "linear": _linear(pooled.reshape(1, 18), FC_W, FC_B),
        },
        "nested_residual": {"gelu": block, "layer_norm": _layer_norm(X24 + block)},
        "closure_constant": {"matmul": (X24 @ W44).astype(np.float32)},
    }


#: Input, weight buffer, then per step: matmul, (bias buffer once), add, relu.
_LOOP3_OPS: tuple[str, ...] = (
    "none",
    "none",
    "matmul",
    "none",
    "__add__",
    "relu",
    "matmul",
    "__add__",
    "relu",
    "matmul",
    "__add__",
    "relu",
    "none",
)

#: Rows: family, builder, op inventory, has_constants, random_init. Input,
#: output and buffer ops carry ``func_name`` ``"none"``. The closure-constant
#: row's ``"constant"`` op is the shared constants rule's closure source op
#: (SPEC section 2, rule 1).
_CONFIG_BUILT: tuple[
    tuple[str, Callable[[], tuple[Any, Any]], tuple[str, ...] | None, bool, bool], ...
] = (
    (
        "chain_branch",
        _build_chain_branch,
        ("none", "__add__", "__mul__", "relu", "none"),
        False,
        False,
    ),
    ("multi_output", _build_multi_output, ("none", "sum", "__mul__", "none", "none"), False, False),
    ("loop3", _build_loop3, _LOOP3_OPS, False, False),
    (
        "norm_softmax",
        _build_norm_softmax,
        ("none", "mean", "var", "__sub__", "__add__", "sqrt", "__truediv__", "softmax", "none"),
        False,
        False,
    ),
    (
        "reshape_transpose",
        _build_reshape_transpose,
        ("none", "reshape", "transpose", "ones", "matmul", "none"),
        True,
        False,
    ),
    ("mlp", _build_mlp, ("none", "linear", "relu", "linear", "none"), False, False),
    (
        "shared_twice",
        _build_shared_twice,
        ("none", "linear", "relu", "linear", "none"),
        False,
        False,
    ),
    (
        "cnn",
        _build_cnn,
        ("none", "conv2d", "relu", "max_pool2d", "flatten", "linear", "none"),
        False,
        False,
    ),
    (
        "nested_residual",
        _build_nested_residual,
        ("none", "linear", "gelu", "__add__", "layer_norm", "none"),
        False,
        False,
    ),
    (
        "closure_constant",
        _build_constants,
        ("none", "full", "constant", "matmul", "__add__", "none"),
        True,
        False,
    ),
    ("random_forward", _build_random_forward, ("none", "__mul__", "dropout", "none"), False, True),
)


# ------------------------------------------------------------------ ecosystem
def _build_resnet18_imagenet() -> tuple[Any, Any]:
    """The resnet18_imagenet ecosystem row: torchvision weights and a fixed image."""

    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(0)
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(), torch.randn(1, 3, 64, 64)


def _build_mobilenet_v3_small_imagenet() -> tuple[Any, Any]:
    """The mobilenet_v3_small_imagenet ecosystem row: torchvision weights and a fixed image."""

    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    torch.manual_seed(0)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    return mobilenet_v3_small(weights=weights).eval(), torch.randn(1, 3, 64, 64)


#: Pinned ecosystem checkpoints. torchvision names each weight file with the
#: SHA-256 prefix it verifies on download.
RESNET18_EVIDENCE = "torchvision ResNet18_Weights.IMAGENET1K_V1 resnet18-f37072fd.pth"
MOBILENET_V3_SMALL_EVIDENCE = (
    "torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1 mobilenet_v3_small-047dcff4.pth"
)

_ECOSYSTEM: tuple[tuple[str, Callable[[], tuple[Any, Any]], str], ...] = (
    ("resnet18_imagenet", _build_resnet18_imagenet, RESNET18_EVIDENCE),
    (
        "mobilenet_v3_small_imagenet",
        _build_mobilenet_v3_small_imagenet,
        MOBILENET_V3_SMALL_EVIDENCE,
    ),
)


# ------------------------------------------------------------------ assembly
def config_built_roster() -> tuple[RosterModel, ...]:
    """Torch's per-change rows (no network, no claim)."""

    expected = _expected_intermediates()
    return tuple(
        RosterModel(
            family=family,
            realism="config_built",
            builder=builder,
            expected_op_kinds=op_kinds,
            expected_intermediates=expected.get(family, {}),
            has_constants=has_constants,
            random_init=random_init,
            group="config_built",
        )
        for family, builder, op_kinds, has_constants, random_init in _CONFIG_BUILT
    )


def ecosystem_roster() -> tuple[RosterModel, ...]:
    """Torch's pinned pretrained rows (need weights; see :data:`ECOSYSTEM_ENV`)."""

    return tuple(
        RosterModel(
            family=family,
            realism="pretrained",
            builder=builder,
            checkpoint_evidence=evidence,
            group="ecosystem",
        )
        for family, builder, evidence in _ECOSYSTEM
    )


def roster() -> tuple[RosterModel, ...]:
    """Every torch row: config-built always, ecosystem when switched on."""

    rows = config_built_roster()
    if ecosystem_enabled():
        rows = rows + ecosystem_roster()
    return rows
