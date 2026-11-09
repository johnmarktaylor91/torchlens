"""Canonical op spellings across backends, for cross-backend labelling only.

Every backend logs ops under its own function spelling: MLX ``subtract`` and
``layer_norm``, Paddle ``functional.linear`` and ``tensor.__add__``, JAX
primitives (``dot_general``), TensorFlow op types (``MatMul``), tinygrad UOp
names (``cmplt``). The canonical spelling is the one TorchLens already uses for
torch: ``_normalize_func_name`` (lowercase, underscores stripped, commutative
reflected dunders folded), so ``layer_norm`` and ``layernorm`` meet.

Each backend keeps a short list beside its wrapper inventory, in
``torchlens/backends/<backend>/_spellings.py``:

- ``SPELLINGS``: the backend's spelling to the canonical torch spelling;
- ``UNMAPPED``: spellings with no canonical torch spelling (the "marked other"
  set), each with a coarse family; such an op keeps its own name as its label.

:data:`CANONICAL_FAMILY` gives each canonical spelling one coarse family.
:func:`op_spelling` answers both for any backend and name.

Contract: a spelling is used only to prefer one structural candidate over
another and to label the other side's ops in drawings. It never pairs two ops
on its own and never encodes a decomposition (a linear is not "matmul plus
add" here); pairing comes from module boundaries and graph structure.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import MappingProxyType
from typing import NamedTuple

#: Coarse op families. ``other`` means no family is claimed.
FAMILIES: frozenset[str] = frozenset(
    {
        "source",
        "create",
        "arith",
        "unary",
        "activation",
        "softmax",
        "matmul",
        "conv",
        "pool",
        "norm",
        "reduce",
        "compare",
        "select",
        "shape",
        "index",
        "attention",
        "embedding",
        "dropout",
        "recurrent",
        "cast",
        "identity",
        "region",
        "other",
    }
)

_FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "source": ("input", "constant", "buffer"),
    "create": ("zeros", "ones", "full", "arange", "linspace", "eye", "tensor")
    + ("zeroslike", "oneslike", "fulllike"),
    "arith": ("add", "sub", "rsub", "mul", "div", "rtruediv", "floordivide", "remainder")
    + ("pow", "rpow", "neg", "maximum", "minimum", "and", "or", "xor"),
    "unary": ("abs", "square", "sqrt", "rsqrt", "exp", "exp2", "expm1", "log", "log2")
    + ("log10", "log1p", "erf", "erfc", "erfinv", "sin", "cos", "tan", "floor", "ceil")
    + ("round", "trunc", "sign", "reciprocal", "clamp"),
    "activation": ("relu", "relu6", "gelu", "sigmoid", "tanh", "silu", "leakyrelu", "elu")
    + ("selu", "celu", "softplus", "softsign", "mish", "logsigmoid", "hardswish")
    + ("hardsigmoid", "hardtanh", "hardshrink", "softshrink", "glu", "prelu"),
    "softmax": ("softmax", "logsoftmax", "softmin"),
    "matmul": ("matmul", "rmatmul", "mm", "bmm", "addmm", "linear", "bilinear", "einsum")
    + ("inner", "outer", "tensordot"),
    "conv": ("conv1d", "conv2d", "conv3d", "convtranspose1d", "convtranspose2d")
    + ("convtranspose3d", "convolution"),
    "pool": ("maxpool1d", "maxpool2d", "maxpool3d", "avgpool1d", "avgpool2d", "avgpool3d")
    + ("adaptiveavgpool1d", "adaptiveavgpool2d", "adaptiveavgpool3d", "interpolate"),
    "norm": ("layernorm", "rmsnorm", "batchnorm", "groupnorm", "instancenorm"),
    "reduce": ("sum", "mean", "prod", "var", "std", "max", "min", "amax", "amin", "argmax")
    + ("argmin", "cumsum", "cumprod", "logsumexp", "any", "all"),
    "compare": ("eq", "ne", "gt", "ge", "lt", "le", "logicaland", "logicalor", "logicalnot"),
    "select": ("where", "tril", "triu"),
    "shape": ("reshape", "view", "flatten", "unflatten", "squeeze", "unsqueeze", "transpose")
    + ("permute", "movedim", "t", "expand", "broadcastto", "cat", "stack", "split", "tile")
    + ("repeat", "repeatinterleave", "pad", "flip", "diag", "diagonal"),
    "index": ("getitem", "setitem", "take", "takealongdim", "gather", "indexselect")
    + ("scatter", "scatteradd", "sort", "argsort", "topk"),
    "attention": ("scaleddotproductattention", "multiheadattentionforward"),
    "embedding": ("embedding",),
    "dropout": ("dropout", "dropout2d", "dropout3d"),
    "recurrent": ("lstm", "gru", "rnntanh"),
    "cast": ("to",),
    "identity": ("identity", "clone", "contiguous", "detach"),
}

#: Canonical spelling to its coarse family. Keys are torch's normalized
#: spellings, except :data:`SYNTHETIC_CANONICAL`.
CANONICAL_FAMILY: MappingProxyType[str, str] = MappingProxyType(
    {name: family for family, names in _FAMILY_MEMBERS.items() for name in names}
)

#: Canonical spellings that are TorchLens source labels or an identity no torch
#: function spells, so they are not torch function names.
SYNTHETIC_CANONICAL: frozenset[str] = frozenset({"input", "constant", "buffer", "identity"})

#: Backends that keep a spelling list (``torchlens/backends/<name>/_spellings.py``).
SPELLING_BACKENDS: tuple[str, ...] = ("torch", "mlx", "paddle", "jax", "tf", "tinygrad")

#: Source labels every backend records under the same name.
_SHARED_SOURCE_NAMES: frozenset[str] = frozenset({"input", "constant", "buffer"})

#: Prefixes of region ops (a jitted or control-flow region recorded as one op).
_REGION_PREFIXES: tuple[str, ...] = ("region:", "region_out:")


class Spelling(NamedTuple):
    """The canonical spelling and family of one backend op name.

    Attributes
    ----------
    canonical:
        Canonical torch spelling, or the backend's own normalized name when
        the op has none.
    family:
        Coarse family from :data:`FAMILIES`.
    listed:
        Whether the backend's list names the op (in ``SPELLINGS`` or
        ``UNMAPPED``); ``False`` means the fallback answered.
    """

    canonical: str
    family: str
    listed: bool


def normalize_spelling(func_name: str) -> str:
    """Return ``func_name`` lowercased with underscores stripped (torch's rule)."""

    return func_name.lower().replace("_", "")


def op_spelling(backend: str, func_name: str) -> Spelling:
    """Return the canonical spelling and family of ``func_name`` on ``backend``.

    Parameters
    ----------
    backend:
        Provenance backend name (``"torch"``, ``"mlx"``, ...).
    func_name:
        The op's recorded ``func_name``.

    Returns
    -------
    Spelling
        A listed answer from the backend's list, else the fallback: the
        normalized name with family ``other`` and ``listed=False``.
    """

    if func_name in _SHARED_SOURCE_NAMES:
        return Spelling(func_name, "source", True)
    if func_name.startswith(_REGION_PREFIXES):
        return Spelling(func_name, "region", True)
    table = _backend_table(backend)
    if table is None:
        return Spelling(normalize_spelling(func_name), "other", False)
    return table.lookup(func_name)


class _BackendTable(NamedTuple):
    """One backend's loaded spelling list.

    ``key`` is the backend's ``spelling_key`` (torch only): it normalizes a raw
    name first, and a name it maps onto a canonical spelling is listed as
    itself, since torch's spelling is the canonical one by definition.
    """

    spellings: MappingProxyType[str, str]
    unmapped: MappingProxyType[str, str]
    key: Callable[[str], str] | None

    def lookup(self, func_name: str) -> Spelling:
        """Resolve ``func_name`` through this list, falling back to its normalized name."""

        name = func_name if self.key is None else self.key(func_name)
        canonical = self.spellings.get(name)
        if canonical is None and self.key is not None and name in CANONICAL_FAMILY:
            canonical = name
        if canonical is not None:
            return Spelling(canonical, CANONICAL_FAMILY[canonical], True)
        family = self.unmapped.get(name)
        if family is not None:
            return Spelling(normalize_spelling(name), family, True)
        return Spelling(normalize_spelling(name), "other", False)


_TABLES: dict[str, _BackendTable] = {}


def _backend_table(backend: str) -> _BackendTable | None:
    """Load (once) the spelling list of ``backend``; ``None`` for an unknown backend.

    The module is chosen by the backend name at run time, the same name-driven
    dispatch ``_capture.prepare`` uses, so this module imports no backend
    package and a lookup loads only the backend it names.
    """

    if backend not in SPELLING_BACKENDS:
        return None
    table = _TABLES.get(backend)
    if table is None:
        module = importlib.import_module(f"torchlens.backends.{backend}._spellings")
        table = _BackendTable(
            spellings=MappingProxyType(dict(module.SPELLINGS)),
            unmapped=MappingProxyType(dict(module.UNMAPPED)),
            key=getattr(module, "spelling_key", None),
        )
        _TABLES[backend] = table
    return table
