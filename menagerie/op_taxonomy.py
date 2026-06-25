"""Deterministic operation taxonomy for menagerie trace summaries."""

from __future__ import annotations

from collections.abc import Iterable

OP_TAXONOMY_VERSION = "1.0.0"

OP_CATEGORIES = (
    "conv",
    "linear",
    "attention",
    "norm",
    "elementwise",
    "reduction",
    "reshape",
    "embedding",
    "pooling",
    "activation",
    "other",
)

_MODULE_CATEGORY_TOKENS: tuple[tuple[str, str], ...] = (
    ("multiheadattention", "attention"),
    ("attention", "attention"),
    ("conv", "conv"),
    ("linear", "linear"),
    ("bilinear", "linear"),
    ("embedding", "embedding"),
    ("batchnorm", "norm"),
    ("layernorm", "norm"),
    ("groupnorm", "norm"),
    ("instancenorm", "norm"),
    ("rmsnorm", "norm"),
    ("localresponsenorm", "norm"),
    ("maxpool", "pooling"),
    ("avgpool", "pooling"),
    ("adaptiveavgpool", "pooling"),
    ("adaptivemaxpool", "pooling"),
    ("relu", "activation"),
    ("gelu", "activation"),
    ("silu", "activation"),
    ("swish", "activation"),
    ("elu", "activation"),
    ("leakyrelu", "activation"),
    ("tanh", "activation"),
    ("sigmoid", "activation"),
    ("mish", "activation"),
    ("softmax", "activation"),
)

_FUNC_EXACT: dict[str, str] = {
    "add": "elementwise",
    "sub": "elementwise",
    "subtract": "elementwise",
    "mul": "elementwise",
    "multiply": "elementwise",
    "div": "elementwise",
    "divide": "elementwise",
    "true_divide": "elementwise",
    "floor_divide": "elementwise",
    "pow": "elementwise",
    "exp": "elementwise",
    "expm1": "elementwise",
    "log": "elementwise",
    "log1p": "elementwise",
    "sqrt": "elementwise",
    "rsqrt": "elementwise",
    "reciprocal": "elementwise",
    "neg": "elementwise",
    "negative": "elementwise",
    "abs": "elementwise",
    "absolute": "elementwise",
    "clamp": "elementwise",
    "clip": "elementwise",
    "minimum": "elementwise",
    "maximum": "elementwise",
    "where": "elementwise",
    "eq": "elementwise",
    "ne": "elementwise",
    "lt": "elementwise",
    "le": "elementwise",
    "gt": "elementwise",
    "ge": "elementwise",
    "logical_and": "elementwise",
    "logical_or": "elementwise",
    "logical_not": "elementwise",
    "dropout": "elementwise",
    "sum": "reduction",
    "mean": "reduction",
    "amax": "reduction",
    "amin": "reduction",
    "max": "reduction",
    "min": "reduction",
    "argmax": "reduction",
    "argmin": "reduction",
    "prod": "reduction",
    "all": "reduction",
    "any": "reduction",
    "count_nonzero": "reduction",
    "cumsum": "reduction",
    "cumprod": "reduction",
    "topk": "reduction",
    "sort": "reduction",
    "kthvalue": "reduction",
    "median": "reduction",
    "view": "reshape",
    "reshape": "reshape",
    "flatten": "reshape",
    "squeeze": "reshape",
    "unsqueeze": "reshape",
    "permute": "reshape",
    "transpose": "reshape",
    "t": "reshape",
    "movedim": "reshape",
    "moveaxis": "reshape",
    "contiguous": "reshape",
    "clone": "reshape",
    "detach": "reshape",
    "expand": "reshape",
    "expand_as": "reshape",
    "repeat": "reshape",
    "repeat_interleave": "reshape",
    "cat": "reshape",
    "concat": "reshape",
    "concatenate": "reshape",
    "stack": "reshape",
    "split": "reshape",
    "chunk": "reshape",
    "unbind": "reshape",
    "select": "reshape",
    "slice": "reshape",
    "narrow": "reshape",
    "index_select": "reshape",
    "gather": "reshape",
    "scatter": "reshape",
    "pad": "reshape",
    "linear": "linear",
    "matmul": "linear",
    "mm": "linear",
    "bmm": "linear",
    "mv": "linear",
    "addmm": "linear",
    "addmv": "linear",
    "addr": "linear",
    "addbmm": "linear",
    "baddbmm": "linear",
    "einsum": "linear",
    "embedding": "embedding",
    "embedding_bag": "embedding",
    "one_hot": "embedding",
    "scaled_dot_product_attention": "attention",
    "multi_head_attention_forward": "attention",
    "relu": "activation",
    "relu_": "activation",
    "gelu": "activation",
    "silu": "activation",
    "swish": "activation",
    "elu": "activation",
    "celu": "activation",
    "selu": "activation",
    "leaky_relu": "activation",
    "tanh": "activation",
    "sigmoid": "activation",
    "mish": "activation",
    "softmax": "activation",
    "log_softmax": "activation",
    "hardtanh": "activation",
    "hardswish": "activation",
    "hardsigmoid": "activation",
    "softplus": "activation",
    "softsign": "activation",
}

_FUNC_PREFIXES: tuple[tuple[str, str], ...] = (
    ("conv_transpose", "conv"),
    ("conv", "conv"),
    ("convolution", "conv"),
    ("batch_norm", "norm"),
    ("layer_norm", "norm"),
    ("group_norm", "norm"),
    ("instance_norm", "norm"),
    ("rms_norm", "norm"),
    ("max_pool", "pooling"),
    ("avg_pool", "pooling"),
    ("adaptive_avg_pool", "pooling"),
    ("adaptive_max_pool", "pooling"),
    ("fractional_max_pool", "pooling"),
    ("lp_pool", "pooling"),
)

_NORM_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("batch", "batch_norm"),
    ("layer", "layer_norm"),
    ("group", "group_norm"),
    ("instance", "instance_norm"),
    ("rms", "rms_norm"),
)

_ACTIVATION_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("leaky_relu", "leaky_relu"),
    ("relu", "relu"),
    ("gelu", "gelu"),
    ("silu", "silu"),
    ("swish", "silu"),
    ("elu", "elu"),
    ("tanh", "tanh"),
    ("sigmoid", "sigmoid"),
    ("mish", "mish"),
)


def _normalize_name(value: str | None) -> str:
    """Normalize a function or module name for deterministic matching.

    Parameters
    ----------
    value:
        Raw function or module class name.

    Returns
    -------
    str
        Lowercase name with common namespace and punctuation noise removed.
    """

    if value is None:
        return ""
    normalized = str(value).strip().lower()
    for prefix in ("torch.nn.functional.", "torch.", "nn.", "aten::"):
        normalized = normalized.removeprefix(prefix)
    return normalized.replace("-", "_").replace(" ", "_")


def _compact_name(value: str | None) -> str:
    """Return a punctuation-free normalized name for module-token matching.

    Parameters
    ----------
    value:
        Raw function or module class name.

    Returns
    -------
    str
        Lowercase alphanumeric token.
    """

    return "".join(character for character in _normalize_name(value) if character.isalnum())


def _first_matching_subtype(
    func_name: str | None,
    module_class_name: str | None,
    subtype_tokens: Iterable[tuple[str, str]],
) -> str:
    """Return the first subtype whose token appears in the op or module name.

    Parameters
    ----------
    func_name:
        Raw Torch operation name.
    module_class_name:
        Owning module class name, when available.
    subtype_tokens:
        Ordered ``(token, subtype)`` pairs.

    Returns
    -------
    str
        Matched subtype, or ``"none"`` when no subtype matches.
    """

    names = (_normalize_name(func_name), _normalize_name(module_class_name))
    compact_names = (_compact_name(func_name), _compact_name(module_class_name))
    for token, subtype in subtype_tokens:
        compact_token = token.replace("_", "")
        if any(token in name for name in names) or any(
            compact_token in name for name in compact_names
        ):
            return subtype
    return "none"


def classify_norm_type(func_name: str | None, module_class_name: str | None = None) -> str:
    """Classify a normalization op into a canonical norm subtype.

    Parameters
    ----------
    func_name:
        Raw Torch operation name.
    module_class_name:
        Owning module class name, when available.

    Returns
    -------
    str
        One of ``batch_norm``, ``layer_norm``, ``group_norm``, ``instance_norm``,
        ``rms_norm``, or ``none``.
    """

    if classify_op(func_name, module_class_name) != "norm":
        return "none"
    return _first_matching_subtype(func_name, module_class_name, _NORM_SUBTYPES)


def classify_activation_type(func_name: str | None, module_class_name: str | None = None) -> str:
    """Classify an activation op into a canonical activation subtype.

    Parameters
    ----------
    func_name:
        Raw Torch operation name.
    module_class_name:
        Owning module class name, when available.

    Returns
    -------
    str
        One of the supported activation subtype names, or ``none``.
    """

    if classify_op(func_name, module_class_name) != "activation":
        return "none"
    return _first_matching_subtype(func_name, module_class_name, _ACTIVATION_SUBTYPES)


def classify_op(func_name: str | None, module_class_name: str | None = None) -> str:
    """Classify a Torch operation into the closed menagerie op taxonomy.

    Module-class tokens are intentionally checked before function names. A semantic wrapper
    such as an attention or convolution block can therefore classify internal primitive calls
    by the owning module category when both signals are present.

    Parameters
    ----------
    func_name:
        Raw ``Op.func_name`` value from TorchLens.
    module_class_name:
        Owning module class name, when available.

    Returns
    -------
    str
        One of ``OP_CATEGORIES``. Unknown inputs deterministically return ``"other"``.
    """

    compact_module_name = _compact_name(module_class_name)
    for token, category in _MODULE_CATEGORY_TOKENS:
        if token in compact_module_name:
            return category

    normalized_func_name = _normalize_name(func_name)
    if normalized_func_name in _FUNC_EXACT:
        return _FUNC_EXACT[normalized_func_name]

    for prefix, category in _FUNC_PREFIXES:
        if normalized_func_name.startswith(prefix):
            return category

    return "other"
