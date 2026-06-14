"""O(1) tensor/parameter extraction from torch function arguments via lookup table.

Replaces the generic 3-level BFS crawl (get_vars_of_type_from_obj) with direct
position-based extraction for known torch function signatures.

Three-tier strategy:
    Tier 1: Static table FUNC_ARG_SPECS for built-in torch functions.
    Tier 2: Dynamic cache _state._dynamic_arg_specs for first-seen functions.
    Tier 3: BFS fallback (one crawl per unique normalized func_name, then cached).

Keys are *normalized* function names: ``func_name.lower().replace("_", "")``.
This collapses variants like ``add``, ``add_``, ``__add__``, ``__iadd__`` into
a single entry ``"add"``, since they all share the same arg-position layout.
"""

from dataclasses import dataclass

import torch

from .. import _state


def _normalize_func_name(func_name: str) -> str:
    """Normalize a raw function name for lookup table keying."""
    return func_name.lower().replace("_", "")


def _get_tensor_kwarg(kwargs: dict[str, object], name: str) -> object:
    """Return a tensor-bearing kwarg by exact or normalized name.

    Parameters
    ----------
    kwargs:
        Keyword arguments passed to the torch function.
    name:
        Expected keyword name, either raw (``"attn_mask"``) or normalized
        (``"attnmask"``).

    Returns
    -------
    object
        Matching kwarg value, or ``None`` when absent.
    """

    if name in kwargs:
        return kwargs[name]
    normalized_name = _normalize_func_name(name)
    for key, value in kwargs.items():
        if _normalize_func_name(str(key)) == normalized_name:
            return value
    return None


@dataclass(frozen=True)
class ArgSpec:
    """Which argument positions can hold tensors or parameters.

    Attributes:
        positions: Positional arg indices that can hold a single tensor/parameter.
        sequence_positions: Indices holding sequences (list/tuple) of tensors.
        tensor_kwargs: Keyword argument names that can hold tensors/parameters.
    """

    positions: tuple[int, ...] = ()
    sequence_positions: tuple[int, ...] = ()
    tensor_kwargs: tuple[str, ...] = ()


def extract_tensors_and_params(
    spec: ArgSpec,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter]]:
    """Extract tensors (excluding Parameters) and Parameters from known arg positions.

    Returns:
        (arg_tensors, arg_parameters) — matches get_vars_of_type_from_obj output.
    """
    tensors: list[torch.Tensor] = []
    params: list[torch.nn.Parameter] = []

    def _append_tensor_or_param(value: object) -> None:
        """Append tensor-like values from a known argument slot.

        Parameters
        ----------
        value:
            Candidate tensor, parameter, or shallow sequence of tensors.
        """

        if isinstance(value, torch.nn.Parameter):
            params.append(value)
        elif isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.nn.Parameter):
                    params.append(item)
                elif isinstance(item, torch.Tensor):
                    tensors.append(item)

    for pos in spec.positions:
        if pos < len(args):
            _append_tensor_or_param(args[pos])

    for pos in spec.sequence_positions:
        if pos < len(args):
            seq = args[pos]
            if isinstance(seq, (list, tuple)):
                for item in seq:
                    if isinstance(item, torch.nn.Parameter):
                        params.append(item)
                    elif isinstance(item, torch.Tensor):
                        tensors.append(item)

    for name in spec.tensor_kwargs:
        val = _get_tensor_kwarg(kwargs, name)
        if val is not None:
            _append_tensor_or_param(val)

    return tensors, params


def _cache_dynamic_spec(
    normalized_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    found_tensors: list[torch.Tensor],
    found_params: list[torch.nn.Parameter],
) -> None:
    """Construct and cache an ArgSpec from BFS crawl results (Tier 3)."""
    all_found_ids = {id(t) for t in found_tensors} | {id(p) for p in found_params}

    positions = []
    sequence_positions = []
    tensor_kwargs_found = []

    for i, arg in enumerate(args):
        if id(arg) in all_found_ids:
            positions.append(i)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if id(item) in all_found_ids:
                    sequence_positions.append(i)
                    break

    for key, val in kwargs.items():
        if val is not None and id(val) in all_found_ids:
            tensor_kwargs_found.append(key)

    spec = ArgSpec(
        positions=tuple(positions),
        sequence_positions=tuple(sequence_positions),
        tensor_kwargs=tuple(tensor_kwargs_found),
    )
    _state._dynamic_arg_specs[normalized_name] = spec


# ============================================================================
# Shared ArgSpec instances (reduce object count)
# ============================================================================

_P0 = ArgSpec(positions=(0,))
_P0_INPUT = ArgSpec(positions=(0,), tensor_kwargs=("input", "self", "tensor"))
_P01 = ArgSpec(positions=(0, 1))
_P01_INPUT_TARGET = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "target"))
_P01_BINARY = ArgSpec(
    positions=(0, 1),
    tensor_kwargs=(
        "input",
        "other",
        "self",
        "tensor",
        "target",
        "mat2",
        "vec",
        "vec1",
        "vec2",
        "batch1",
        "batch2",
        "tensor1",
        "tensor2",
        "end",
        "weight",
        "mask",
        "index",
        "indices",
        "source",
        "src",
        "values",
        "sorted_sequence",
        "boundaries",
    ),
)
_P012 = ArgSpec(positions=(0, 1, 2))
_P0123 = ArgSpec(positions=(0, 1, 2, 3))
_S0 = ArgSpec(positions=tuple(range(10)), tensor_kwargs=("tensors",))
_NONE = ArgSpec()

# ============================================================================
# FUNC_ARG_SPECS — keyed by normalized func_name
# ============================================================================

FUNC_ARG_SPECS: dict[str, ArgSpec] = {}

# ---------------------------------------------------------------------------
# Unary: only position 0 is a tensor (self/input)
# ---------------------------------------------------------------------------

_UNARY_FUNCS = [
    # View/reshape
    "view",
    "viewas",
    "reshape",
    "reshapeas",
    "flatten",
    "unflatten",
    "ravel",
    "contiguous",
    "narrow",
    "narrowcopy",
    # Transpose/permute
    "t",
    "transpose",
    "permute",
    "adjoint",
    "swapaxes",
    "swapdims",
    "moveaxis",
    "movedim",
    # Squeeze/unsqueeze/expand
    "squeeze",
    "unsqueeze",
    "expand",
    "expandas",
    "expandcopy",
    # Copy/clone
    "clone",
    "detach",
    # Type/device conversion
    "to",
    "cpu",
    "cuda",
    "float",
    "double",
    "half",
    "bfloat16",
    "int",
    "long",
    "short",
    "byte",
    "char",
    "bool",
    "type",
    "typeas",
    "cfloat",
    "cdouble",
    "chalf",
    # Shape queries
    "size",
    "dim",
    "numel",
    "nelement",
    "ndimension",
    "elementsize",
    "iscontiguous",
    "iscomplex",
    "isfloatingpoint",
    "issigned",
    # Memory / storage
    "pinmemory",
    "sharememory",
    "recordstream",
    "storage",
    "storageoffset",
    "dataptr",
    "untypedstorage",
    "storagetype",
    # In-place init
    "set",
    "fill",
    "filldiagonal",
    "zero",
    "requiresgrad",
    "retaingrad",
    "detachcopy",
    # Strided
    "asstrided",
    # Repeat/tile
    "repeat",
    "repeatinterleave",
    "tile",
    # Flip/roll/rotate
    "roll",
    "rot90",
    "flip",
    "fliplr",
    "flipud",
    # Diagonal/triangular
    "diagonal",
    "diagonalcopy",
    "diag",
    "diagembed",
    "diagflat",
    "tril",
    "triu",
    # Select/scatter
    "select",
    "selectscatter",
    "slicescatter",
    "asstridedscatter",
    "diagonalscatter",
    # Split/chunk/unbind
    "chunk",
    "split",
    "splitwithsizes",
    "tensorsplit",
    "hsplit",
    "vsplit",
    "dsplit",
    "unbind",
    # Complex
    "viewasreal",
    "viewascomplex",
    "viewasrealcopy",
    "viewascomplexcopy",
    "real",
    "imag",
    "resolveconj",
    "resolveneg",
    "conj",
    "conjphysical",
    "angle",
    # Unfold/fold/pixel
    "unfold",
    "fold",
    "pixelshuffle",
    "pixelunshuffle",
    "channelshuffle",
    "nativechannelshuffle",
    # Broadcast
    "broadcastto",
    "atleast1d",
    "atleast2d",
    "atleast3d",
    # Naming
    "rename",
    "refinenames",
    "hasnames",
    # Sparse / quantization
    "tosparse",
    "todense",
    "tomkldnn",
    "coalesce",
    "sparsemask",
    "iscoalesced",
    "sparsedim",
    "densedim",
    "indices",
    "values",
    "crowindices",
    "colindices",
    "rowindices",
    "ccolindices",
    # Conversion
    "item",
    "tolist",
    "numpy",
    # --- Activations (unary) ---
    "relu",
    "relu6",
    "threshold",
    "hardtanh",
    "sigmoid",
    "hardsigmoid",
    "hardswish",
    "silu",
    "mish",
    "gelu",
    "celu",
    "elu",
    "selu",
    "softplus",
    "softshrink",
    "hardshrink",
    "tanhshrink",
    "softsign",
    "logsigmoid",
    "logit",
    "expit",
    "rrelu",
    "leakyrelu",
    # --- Unary math ---
    "neg",
    "negative",
    "pos",
    "positive",
    "abs",
    "absolute",
    "sign",
    "sgn",
    "signbit",
    "ceil",
    "floor",
    "round",
    "trunc",
    "frac",
    "fix",
    # Symbolic-shape unary (exercised by meta-device / symbolic-shape paths)
    "symfloat",
    "symint",
    "symnot",
    "reciprocal",
    "square",
    "nantonum",
    # Checks
    "isnan",
    "isinf",
    "isfinite",
    "isneginf",
    "isposinf",
    "isreal",
    # Exponential / log
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    # Trig
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sinc",
    # Special functions
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
    "digamma",
    "polygamma",
    "mvlgamma",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "deg2rad",
    "rad2deg",
    # Power / root
    "sqrt",
    "rsqrt",
    # Logical / bitwise unary
    "logicalnot",
    "bitwisenot",
    "invert",
    "not",
    # --- Reductions ---
    "sum",
    "nansum",
    "mean",
    "nanmean",
    "std",
    "var",
    "prod",
    "norm",
    "frobnorm",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "logsumexp",
    "logcumsumexp",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "countnonzero",
    "all",
    "any",
    "nonzero",
    # Softmax
    "softmax",
    "logsoftmax",
    # Sort/search
    "sort",
    "argsort",
    "msort",
    "topk",
    "kthvalue",
    "median",
    "nanmedian",
    "mode",
    "unique",
    "uniqueconsecutive",
    "searchsorted",
    # Trace / linalg (unary)
    "trace",
    "det",
    "logdet",
    "slogdet",
    "cholesky",
    "qr",
    "svd",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "lu",
    "luunpack",
    "pinverse",
    "inverse",
    "matrixrank",
    "matrixnorm",
    "vectornorm",
    "matrixexp",
    "matrixpower",
    "householderproduct",
    # Histogram
    "histc",
    "bincount",
    "histogram",
    "histogramdd",
    # --- Pooling ---
    "avgpool1d",
    "avgpool2d",
    "avgpool3d",
    "maxpool1d",
    "maxpool2d",
    "maxpool3d",
    "maxpool1dwithindices",
    "maxpool2dwithindices",
    "maxpool3dwithindices",
    "adaptiveavgpool1d",
    "adaptiveavgpool2d",
    "adaptiveavgpool3d",
    "adaptivemaxpool1d",
    "adaptivemaxpool2d",
    "adaptivemaxpool3d",
    "lppool1d",
    "lppool2d",
    "fractionalmaxpool2d",
    "fractionalmaxpool3d",
    "maxunpool1d",
    "maxunpool2d",
    "maxunpool3d",
    # --- Dropout ---
    "dropout",
    "dropout2d",
    "dropout3d",
    "alphadropout",
    "featurealphadropout",
    # --- Upsampling / interpolation ---
    "interpolate",
    "upsample",
    "upsamplebilinear",
    "upsamplenearest",
    # Padding
    "pad",
    # In-place random (tensor.xxx_())
    "bernoulli",
    "uniform",
    "random",
    "geometric",
    "exponential",
    "cauchy",
    "logistic",
    "lognormal",
    # FFT
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
    # Other
    "identity",
    "copy",
    "data",
    "len",
    "format",
    "contains",
    "get",
    "assubclass",
    # nn.init functions (in-place on tensor)
    "calculategain",
    "nograduniform",  # torch.nn.init._no_grad_uniform_ (unary in-place init helper)
    "kaiminguniform",
    "kaimingnormal",
    "xavieruniform",
    "xaviernormal",
    "constant",
    "dirac",
    "orthogonal",
    "sparse",
    # Misc
    "str",
    "repr",
    "hash",
    "copymemory",
    "stft",
    "istft",
    "bartlettwindow",
    "blackmanwindow",
    "hammingwindow",
    "hannwindow",
    "kaiserwindow",
    # CUDNN / MKLDNN
    "cudnnisacceptable",
    "mkldnnadaptiveavgpool2d",
    "mkldnnconvolution",
    "mkldnnmaxpool2d",
    "mkldnnmaxpool3d",
    "mkldnnlinearbackwardweights",
    # Internal PyTorch helpers (no tensor args, but decorated)
    "mhashapecheck",
    "single",
    "verifybatchsize",
    "verifyspatialsize",
    "listwithdefault",
    "calculatefaninandfanout",
    "calculatecorrectfan",
    "nopermutation",
    "pair",
    "triple",
    "quadruple",
    "ntuple",
    "checkcatinputs",
    # Hook/utility custom_methods
    "registerhook",
    "isinteger",
    # Dunder / internal
    "index",
    "symint",
    "checkkeypaddingmask",
]

for _name in _UNARY_FUNCS:
    FUNC_ARG_SPECS[_name] = _P0_INPUT

# ---------------------------------------------------------------------------
# Binary: positions 0 and 1 can both hold tensors
# ---------------------------------------------------------------------------

_BINARY_FUNCS = [
    # Arithmetic
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "div",
    "divide",
    "truedivide",
    "floordivide",
    "remainder",
    "fmod",
    "rsub",
    "pow",
    "floatpower",
    # Clamp (min/max can be tensors)
    "clamp",
    "clampmin",
    "clampmax",
    "clip",
    # Reversed ops (tensor.__r*__)
    "radd",
    "rsub",
    "rmul",
    "rdiv",
    "rtruediv",
    "rfloordiv",
    "rmod",
    "rpow",
    # In-place ops (tensor.__i*__)
    "iadd",
    "isub",
    "imul",
    "idiv",
    "itruediv",
    "ifloordiv",
    "imod",
    "ipow",
    # Comparison
    "eq",
    "ne",
    "gt",
    "lt",
    "ge",
    "le",
    "greater",
    "greaterequal",
    "less",
    "lessequal",
    "equal",
    "notequal",
    "isclose",
    "allclose",
    # Min / max (binary form: torch.max(a, b))
    "max",
    "min",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    # Symbolic-shape binary (exercised by meta-device / symbolic-shape paths)
    "symmax",
    "symmin",
    # Logical / bitwise binary
    "logicaland",
    "logicalor",
    "logicalxor",
    "bitwiseand",
    "bitwiseor",
    "bitwisexor",
    "bitwiseleftshift",
    "bitwiserightshift",
    "and",
    "or",
    "xor",
    "iand",
    "ior",
    "ixor",
    "rand",
    "ror",
    "rxor",
    "lshift",
    "rshift",
    "ilshift",
    "irshift",
    "rlshift",
    "rrshift",
    # Math binary
    "atan2",
    "arctan2",
    "copysign",
    "nextafter",
    "hypot",
    "xlogy",
    "xlog1py",
    "cross",
    "dist",
    "dot",
    "vdot",
    "outer",
    "inner",
    "kron",
    # Matrix multiply
    "matmul",
    "mm",
    "mv",
    "bmm",
    "rmatmul",
    "imatmul",
    "multidot",
    # Masked ops
    "maskedfill",
    "maskedscatter",
    "maskedselect",
    # Activations with weight tensor
    "prelu",
    "heaviside",
    # Index ops
    "take",
    "takealongdim",
    # Scatter/index arithmetic
    "scatteradd",
    "scatterreduce",
    "indexadd",
    "indexreduce",
    # Complex construction
    "complex",
    "polar",
    # linalg binary
    "choleskysolve",
    "lusolve",
    "solve",
    "triangularsolve",
    "lstsq",
    # Embedding
    "embedding",
    # Normal (torch.normal(mean_tensor, std_tensor))
    "normal",
]

for _name in _BINARY_FUNCS:
    FUNC_ARG_SPECS[_name] = _P01_BINARY

# ---------------------------------------------------------------------------
# Ternary: positions 0, 1, 2
# ---------------------------------------------------------------------------

_TERNARY_FUNCS = [
    "addcmul",
    "addcdiv",
    "lerp",
    "where",
    "addmm",
    "addbmm",
    "baddbmm",
    "addmv",
]

for _name in _TERNARY_FUNCS:
    FUNC_ARG_SPECS[_name] = _P012

# ---------------------------------------------------------------------------
# Sequence: position 0 is a list/tuple of tensors
# ---------------------------------------------------------------------------

_SEQUENCE_FUNCS = [
    "cat",
    "concat",
    "concatenate",
    "stack",
    "hstack",
    "vstack",
    "dstack",
    "rowstack",
    "columnstack",
    "blockdiag",
    "broadcasttensors",
    "aligntensors",
    "meshgrid",
    "cartesianprod",
    "combinations",
]

for _name in _SEQUENCE_FUNCS:
    FUNC_ARG_SPECS[_name] = _S0

# ---------------------------------------------------------------------------
# Factory: no tensor inputs
# ---------------------------------------------------------------------------

_FACTORY_FUNCS = [
    "zeros",
    "ones",
    "rand",
    "randn",
    "randint",
    "randperm",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "full",
    "empty",
    "tensor",
    "astensor",
    "fromnumpy",
    "fromfile",
    "scalartensor",
    "sparsecoottensor",
    "sparsecsr tensor",
    "vander",
    "trilindices",
    "triuindices",
    "load",
]

for _name in _FACTORY_FUNCS:
    FUNC_ARG_SPECS[_name] = _NONE

# Factory-from-source functions inherit shape/dtype/device from a tensor source.
# Record that source as a topology parent, matching view/reshape-style dependencies
# even when the output values are freshly allocated.
_FACTORY_SOURCE_SPEC = ArgSpec(positions=(0,), tensor_kwargs=("input", "self"))
for _name in [
    "zeroslike",
    "oneslike",
    "randlike",
    "randnlike",
    "emptylike",
    "newempty",
    "newemptystrided",
    "newzeros",
    "newones",
]:
    FUNC_ARG_SPECS[_name] = _FACTORY_SOURCE_SPEC

FUNC_ARG_SPECS["fulllike"] = ArgSpec(
    positions=(0, 1), tensor_kwargs=("input", "self", "fill_value")
)
FUNC_ARG_SPECS["newfull"] = ArgSpec(positions=(0, 2), tensor_kwargs=("self", "fill_value"))
FUNC_ARG_SPECS["newtensor"] = ArgSpec(positions=(0, 1), tensor_kwargs=("self", "data"))

# ---------------------------------------------------------------------------
# Special patterns (custom ArgSpec per function or group)
# ---------------------------------------------------------------------------

# __getitem__: self + index (can be tensor or tuple of tensors)
FUNC_ARG_SPECS["getitem"] = ArgSpec(
    positions=(0, 1), sequence_positions=(1,), tensor_kwargs=("self", "index")
)

# __setitem__: self + index + value
FUNC_ARG_SPECS["setitem"] = ArgSpec(
    positions=(0, 1, 2), sequence_positions=(1,), tensor_kwargs=("self", "index", "value")
)

# __delitem__: just self
FUNC_ARG_SPECS["delitem"] = _P0

# scatter/index_copy/index_fill: (self, dim, index, src/value)
_SCATTER_SPEC = ArgSpec(
    positions=(0, 2, 3), tensor_kwargs=("input", "self", "index", "src", "source", "value")
)
for _name in [
    "scatter",
    "scatteradd",
    "scatterreduce",
    "indexadd",
    "indexreduce",
    "indexcopy",
    "indexfill",
]:
    FUNC_ARG_SPECS[_name] = _SCATTER_SPEC

# gather/index_select: (self, dim, index)
_GATHER_SPEC = ArgSpec(positions=(0, 2), tensor_kwargs=("input", "self", "index"))
for _name in ["gather", "indexselect"]:
    FUNC_ARG_SPECS[_name] = _GATHER_SPEC

# index_put: (self, indices_tuple, values)
FUNC_ARG_SPECS["indexput"] = ArgSpec(
    positions=(0, 2),
    sequence_positions=(1,),
    tensor_kwargs=("input", "self", "indices", "values"),
)

# linear: F.linear(input, weight, bias) — weight/bias can be keyword args
FUNC_ARG_SPECS["linear"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias"))

# conv: F.conv2d(input, weight, bias, stride, ...) — bias at position 2
_CONV_SPEC = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias"))
for _name in [
    "conv1d",
    "conv2d",
    "conv3d",
    "convtranspose1d",
    "convtranspose2d",
    "convtranspose3d",
]:
    FUNC_ARG_SPECS[_name] = _CONV_SPEC

# CUDNN conv variants
for _name in [
    "cudnnconvolution",
    "cudnnconvolutiontranspose",
    "cudnnconvolutionrelu",
    "cudnnconvolutionaddrely",
]:
    FUNC_ARG_SPECS[_name] = _CONV_SPEC

# batch_norm: F.batch_norm(input, running_mean, running_var, weight, bias, ...)
# All 5 args commonly passed positionally by nn.BatchNorm*.forward()
_NORM_WITH_RUNNING_STATS_SPEC = ArgSpec(
    positions=(0, 1, 2, 3, 4),
    tensor_kwargs=("input", "running_mean", "running_var", "weight", "bias"),
)
FUNC_ARG_SPECS["batchnorm"] = _NORM_WITH_RUNNING_STATS_SPEC
FUNC_ARG_SPECS["cudnnbatchnorm"] = _NORM_WITH_RUNNING_STATS_SPEC

# instance_norm: similar to batch_norm
FUNC_ARG_SPECS["instancenorm"] = _NORM_WITH_RUNNING_STATS_SPEC

# layer_norm: F.layer_norm(input, normalized_shape, weight, bias, ...)
# weight at pos 2, bias at pos 3
FUNC_ARG_SPECS["layernorm"] = ArgSpec(
    positions=(0, 2, 3), tensor_kwargs=("input", "weight", "bias")
)

# group_norm: F.group_norm(input, num_groups, weight, bias, ...)
# weight at pos 2, bias at pos 3
FUNC_ARG_SPECS["groupnorm"] = ArgSpec(
    positions=(0, 2, 3), tensor_kwargs=("input", "weight", "bias")
)

# Loss functions: (input, target, weight=None, ...)
# weight can be positional (pos 2) or kwarg
_LOSS_WITH_WEIGHT = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "target", "weight"))
for _name in ["nllloss", "crossentropy", "nllloss2d"]:
    FUNC_ARG_SPECS[_name] = _LOSS_WITH_WEIGHT

# Loss functions: (input, target)
for _name in [
    "mseloss",
    "l1loss",
    "smoothl1loss",
    "huberloss",
    "bceloss",
    "bcewithlogitsloss",
    "cosinesimilarity",
    "cosineembeddingloss",
    "hingeembeddingloss",
    "marginrankingloss",
    "softmarginloss",
    "multilabelsoftmarginloss",
    "multimarginloss",
    "multilabelmarginloss",
    "poissonnllloss",
    "gaussiannllloss",
    "kldiv",
    "ctcloss",
    "tripletmarginloss",
]:
    FUNC_ARG_SPECS[_name] = _P01_INPUT_TARGET

# bilinear: (input1, input2, weight, bias) — bias can be positional
FUNC_ARG_SPECS["bilinear"] = ArgSpec(
    positions=(0, 1, 2, 3), tensor_kwargs=("input1", "input2", "weight", "bias")
)

# scaled_dot_product_attention: (query, key, value, attn_mask=None, ...)
# attn_mask can be positional (pos 3) or kwarg
FUNC_ARG_SPECS["scaleddotproductattention"] = ArgSpec(
    positions=(0, 1, 2, 3), tensor_kwargs=("query", "key", "value", "attn_mask")
)

# multi_head_attention_forward: (query, key, value, ...)
FUNC_ARG_SPECS["multiheadattentionforward"] = ArgSpec(
    positions=(0, 1, 2, 5, 6, 7, 8, 11, 12, 14, 16, 18, 19, 20, 21),
    tensor_kwargs=(
        "query",
        "key",
        "value",
        "in_proj_weight",
        "in_proj_bias",
        "bias_k",
        "bias_v",
        "out_proj_weight",
        "out_proj_bias",
        "key_padding_mask",
        "attn_mask",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "static_k",
        "static_v",
    ),
)

# grid_sample: (input, grid)
FUNC_ARG_SPECS["gridsample"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid"))
FUNC_ARG_SPECS["cudnngridsampler"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid"))
FUNC_ARG_SPECS["cudnnaffinegridgenerator"] = _P0

# affine_grid: (theta, size) — theta is a tensor
FUNC_ARG_SPECS["affinegrid"] = _P0

# one_hot: (tensor, num_classes)
FUNC_ARG_SPECS["onehot"] = _P0

FUNC_ARG_SPECS["cat"] = ArgSpec(sequence_positions=(0,), tensor_kwargs=("tensors",))
FUNC_ARG_SPECS["concat"] = FUNC_ARG_SPECS["cat"]
FUNC_ARG_SPECS["concatenate"] = FUNC_ARG_SPECS["cat"]
FUNC_ARG_SPECS["stack"] = ArgSpec(sequence_positions=(0,), tensor_kwargs=("tensors",))
FUNC_ARG_SPECS["where"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("condition", "input", "other")
)

# Vararg equation/operand forms and uncommon tensor-valued optional kwargs.
FUNC_ARG_SPECS["einsum"] = ArgSpec(
    positions=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    tensor_kwargs=("operands",),
)
FUNC_ARG_SPECS["tensordot"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "other", "a", "b"))
FUNC_ARG_SPECS["searchsorted"] = ArgSpec(
    positions=(0, 1), tensor_kwargs=("sorted_sequence", "input", "values")
)
FUNC_ARG_SPECS["bincount"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "weights"))
FUNC_ARG_SPECS["histogram"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "weight"))
FUNC_ARG_SPECS["histogramdd"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "weight"))
FUNC_ARG_SPECS["stft"] = ArgSpec(positions=(0, 4), tensor_kwargs=("input", "window"))
FUNC_ARG_SPECS["istft"] = ArgSpec(positions=(0, 4), tensor_kwargs=("input", "window"))
FUNC_ARG_SPECS["maskedfill"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "self", "mask", "value")
)
FUNC_ARG_SPECS["maskedscatter"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "self", "mask", "source")
)
FUNC_ARG_SPECS["maskedselect"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "self", "mask"))
FUNC_ARG_SPECS["multidot"] = _S0

# Matrix-multiply family whose second operand kwarg is NOT named "other":
# torch.mm(input, mat2), torch.bmm(input, mat2), torch.mv(input, vec).
# The generic _P01_BINARY entry only knows ("input", "other"), so the kwarg
# form (e.g. torch.bmm(a, mat2=b)) would drop the second tensor.
_MM_SPEC = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "mat2"))
FUNC_ARG_SPECS["mm"] = _MM_SPEC
FUNC_ARG_SPECS["bmm"] = _MM_SPEC
FUNC_ARG_SPECS["mv"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "vec"))

# torch.normal(mean, std) — kwargs are "mean"/"std", not "input"/"other".
# (Also covers tensor.normal_(mean=..., std=...), whose kwargs are floats and
# are simply ignored by the tensor-type check.)
FUNC_ARG_SPECS["normal"] = ArgSpec(positions=(0, 1), tensor_kwargs=("mean", "std"))

# Ternary functions with their real schema kwarg names. The shared _P012
# entry has no tensor_kwargs, so keyword-passed operands were dropped.
FUNC_ARG_SPECS["addmm"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "mat1", "mat2"))
_ADDBMM_SPEC = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "batch1", "batch2"))
FUNC_ARG_SPECS["addbmm"] = _ADDBMM_SPEC
FUNC_ARG_SPECS["baddbmm"] = _ADDBMM_SPEC
FUNC_ARG_SPECS["addmv"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "mat", "vec"))
_ADDC_SPEC = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "tensor1", "tensor2"))
FUNC_ARG_SPECS["addcmul"] = _ADDC_SPEC
FUNC_ARG_SPECS["addcdiv"] = _ADDC_SPEC
FUNC_ARG_SPECS["lerp"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "end", "weight"))

# High-confidence public schemas missing from the generic groups above. These
# entries are schema-derived to keep keyword-passed tensor operands from being
# hidden by a first-call dynamic fallback that only saw positional operands.
for _name in [
    "adaptivemaxpool1dwithindices",
    "adaptivemaxpool2dwithindices",
    "adaptivemaxpool3dwithindices",
    "airyai",
    "aminmax",
    "argwhere",
    "besselj0",
    "besselj1",
    "bessely0",
    "bessely1",
    "chebyshevpolynomialt",
    "chebyshevpolynomialu",
    "chebyshevpolynomialv",
    "chebyshevpolynomialw",
    "choleskyex",
    "cond",
    "corrcoef",
    "dequantize",
    "dropout1d",
    "entr",
    "erfcx",
    "frexp",
    "gammaln",
    "geqrf",
    "glu",
    "gradient",
    "hermitepolynomialh",
    "hermitepolynomialhe",
    "hfftn",
    "ihfftn",
    "inv",
    "invex",
    "isconj",
    "isinference",
    "isnonzero",
    "laguerrepolynomiall",
    "ldlfactor",
    "ldlfactorex",
    "legendrepolynomialp",
    "localresponsenorm",
    "logndtr",
    "lppool3d",
    "lufactor",
    "lufactorex",
    "modifiedbesseli0",
    "modifiedbesseli1",
    "modifiedbesselk0",
    "modifiedbesselk1",
    "multigammaln",
    "multinomial",
    "nanquantile",
    "ndtr",
    "ndtri",
    "normalize",
    "pcalowrank",
    "pdist",
    "pinv",
    "poisson",
    "psi",
    "quantile",
    "randintlike",
    "renorm",
    "scaledmodifiedbesselk0",
    "scaledmodifiedbesselk1",
    "shiftedchebyshevpolynomialt",
    "shiftedchebyshevpolynomialu",
    "shiftedchebyshevpolynomialv",
    "shiftedchebyshevpolynomialw",
    "softmin",
    "stdmean",
    "svdvals",
    "tensorinv",
    "unsafechunk",
    "varmean",
]:
    FUNC_ARG_SPECS[_name] = _P0_INPUT

_P0_A_SPEC = ArgSpec(positions=(0,), tensor_kwargs=("A",))
for _name in [
    "choleskyex",
    "cond",
    "inv",
    "invex",
    "ldlfactor",
    "ldlfactorex",
    "lufactor",
    "lufactorex",
    "pcalowrank",
    "pinv",
    "svdvals",
    "tensorinv",
]:
    FUNC_ARG_SPECS[_name] = _P0_A_SPEC

for _name in [
    "gammainc",
    "gammaincc",
    "gcd",
    "igamma",
    "igammac",
    "lcm",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "zeta",
]:
    FUNC_ARG_SPECS[_name] = _P01_BINARY

FUNC_ARG_SPECS["addr"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "vec1", "vec2"))
FUNC_ARG_SPECS["alignas"] = ArgSpec(positions=(0, 1), tensor_kwargs=("self", "other"))
FUNC_ARG_SPECS["binarycrossentropy"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "target", "weight")
)
FUNC_ARG_SPECS["bucketize"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "boundaries"))
FUNC_ARG_SPECS["cdist"] = ArgSpec(positions=(0, 1), tensor_kwargs=("x1", "x2"))
FUNC_ARG_SPECS["chainmatmul"] = _S0
FUNC_ARG_SPECS["choleskyinverse"] = ArgSpec(positions=(0,), tensor_kwargs=("L",))
FUNC_ARG_SPECS["cov"] = ArgSpec(
    positions=(0, 2, 3), tensor_kwargs=("input", "fweights", "aweights")
)
FUNC_ARG_SPECS["cumulativetrapezoid"] = ArgSpec(positions=(0, 1), tensor_kwargs=("y", "x"))
FUNC_ARG_SPECS["diff"] = ArgSpec(positions=(0, 3, 4), tensor_kwargs=("input", "prepend", "append"))
FUNC_ARG_SPECS["fakequantizeperchannelaffine"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "scale", "zero_point")
)
FUNC_ARG_SPECS["fakequantizepertensoraffine"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "scale", "zero_point")
)
FUNC_ARG_SPECS["ger"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "vec2"))
FUNC_ARG_SPECS["gumbelsoftmax"] = ArgSpec(positions=(0,), tensor_kwargs=("logits",))
FUNC_ARG_SPECS["hspmm"] = ArgSpec(positions=(0, 1), tensor_kwargs=("mat1", "mat2"))
FUNC_ARG_SPECS["isin"] = ArgSpec(positions=(0, 1), tensor_kwargs=("elements", "test_elements"))
FUNC_ARG_SPECS["issetto"] = ArgSpec(positions=(0, 1), tensor_kwargs=("self", "tensor"))
FUNC_ARG_SPECS["ldlsolve"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("LD", "pivots", "B"))
FUNC_ARG_SPECS["lobpcg"] = ArgSpec(positions=(0, 2, 3, 5), tensor_kwargs=("A", "B", "X", "iK"))
FUNC_ARG_SPECS["map"] = ArgSpec(positions=(0, 1), tensor_kwargs=("self", "tensor"))
FUNC_ARG_SPECS["moduleload"] = ArgSpec(positions=(0, 1), tensor_kwargs=("self", "other"))
FUNC_ARG_SPECS["orgqr"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "tau"))
FUNC_ARG_SPECS["ormqr"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "tau", "other"))
FUNC_ARG_SPECS["pairwisedistance"] = ArgSpec(positions=(0, 1), tensor_kwargs=("x1", "x2"))
FUNC_ARG_SPECS["quantizeperchannel"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "scales", "zero_points")
)
FUNC_ARG_SPECS["quantizepertensor"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input", "scale", "zero_point")
)
FUNC_ARG_SPECS["quantizepertensordynamic"] = ArgSpec(positions=(0,), tensor_kwargs=("input",))
FUNC_ARG_SPECS["resize"] = _P0
FUNC_ARG_SPECS["smm"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "mat"))
FUNC_ARG_SPECS["solveex"] = ArgSpec(positions=(0, 1), tensor_kwargs=("A", "B"))
FUNC_ARG_SPECS["solvetriangular"] = ArgSpec(positions=(0, 1), tensor_kwargs=("A", "B"))
FUNC_ARG_SPECS["sspaddmm"] = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "mat1", "mat2"))
FUNC_ARG_SPECS["svdlowrank"] = ArgSpec(positions=(0, 3), tensor_kwargs=("A", "M"))
FUNC_ARG_SPECS["tensorsolve"] = ArgSpec(positions=(0, 1), tensor_kwargs=("A", "B"))
FUNC_ARG_SPECS["trapezoid"] = ArgSpec(positions=(0, 1), tensor_kwargs=("y", "x"))
FUNC_ARG_SPECS["trapz"] = ArgSpec(positions=(0, 1), tensor_kwargs=("y", "x"))
FUNC_ARG_SPECS["tripletmarginwithdistanceloss"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("anchor", "positive", "negative")
)
FUNC_ARG_SPECS["truncnormal"] = ArgSpec(positions=(0,), tensor_kwargs=("tensor",))
FUNC_ARG_SPECS["unravelindex"] = ArgSpec(positions=(0,), tensor_kwargs=("indices",))
FUNC_ARG_SPECS["unsafesplit"] = ArgSpec(positions=(0,), tensor_kwargs=("tensor",))
FUNC_ARG_SPECS["vecdot"] = ArgSpec(positions=(0, 1), tensor_kwargs=("x", "y"))

# ---------------------------------------------------------------------------
# Phase 5b validated audit-fragment fills
# ---------------------------------------------------------------------------

_PHASE5B_VALIDATED_ARG_SPECS = {
    "affinegridgenerator": ArgSpec(positions=(0,), tensor_kwargs=("theta",)),
    "aliascopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "alignto": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "ampforeachnonfinitecheckandunscale": ArgSpec(
        positions=(1, 2), sequence_positions=(0,), tensor_kwargs=("self", "found_inf", "inv_scale")
    ),
    "ampupdatescale": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "growth_tracker", "found_inf")
    ),
    "asstridedcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "autocast": ArgSpec(),
    "autocasttofullprecision": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "autocasttoreducedprecision": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "batchnormelemt": ArgSpec(
        positions=(0, 1, 2, 3, 4), tensor_kwargs=("input", "weight", "bias", "mean", "invstd")
    ),
    "batchnormgatherstats": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "mean", "invstd", "running_mean", "running_var"),
    ),
    "batchnormgatherstatswithcounts": ArgSpec(
        positions=(0, 1, 2, 3, 4, 7),
        tensor_kwargs=("input", "mean", "invstd", "running_mean", "running_var", "counts"),
    ),
    "batchnormimplindex": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "weight", "bias", "running_mean", "running_var"),
    ),
    "batchnormstats": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "batchnormupdatestats": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("input", "running_mean", "running_var")
    ),
    "binarycrossentropywithlogits": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("self", "target", "weight", "pos_weight")
    ),
    "binomial": ArgSpec(positions=(0, 1), tensor_kwargs=("count", "prob")),
    "castbyte": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castchar": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castdouble": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castfloat": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "casthalf": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castint": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castlong": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "castshort": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "ccolindicescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "chooseqparamsoptimized": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "chooseqparamspertensor": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "chunkcat": ArgSpec(sequence_positions=(0,), tensor_kwargs=("tensors",)),
    "coalesced": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "colindicescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "computelinearcombination": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "coefficients")),
    "conjcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "constantpadnd": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "convertindicesfromcootocsr": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "convertindicesfromcsrtocoo": ArgSpec(
        positions=(0, 1), tensor_kwargs=("crow_indices", "col_indices")
    ),
    "convertweighttoint4pack": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "convertweighttoint4packforcpu": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "convolution": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias")),
    "convolutionmode": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias")),
    "convtbc": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")),
    "copyfrom": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "dst")),
    "copyfromandresize": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "dst")),
    "crowindicescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "csltcompress": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "csltsparsemm": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("compressed_A", "dense_B", "bias", "alpha")
    ),
    "csltsparsemmsearch": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("compressed_A", "dense_B", "bias", "alpha")
    ),
    "cudnnconvolutionaddrelu": ArgSpec(
        positions=(0, 1, 2, 4), tensor_kwargs=("self", "weight", "z", "bias")
    ),
    "cudnnctcloss": ArgSpec(
        positions=(0, 1, 2, 3),
        tensor_kwargs=("log_probs", "targets", "input_lengths", "target_lengths"),
    ),
    "cudnninitdropoutstate": ArgSpec(),
    "cudnnrnnflattenweight": ArgSpec(sequence_positions=(0,), tensor_kwargs=("weight_arr",)),
    "cufftclearplancache": ArgSpec(),
    "cufftgetplancachemaxsize": ArgSpec(),
    "cufftgetplancachesize": ArgSpec(),
    "cufftsetplancachemaxsize": ArgSpec(),
    "cummaxhelper": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "values", "indices")),
    "cumminhelper": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "values", "indices")),
    "dirichletgrad": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("x", "alpha", "total")),
    "dlpack": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "dlpackdevice": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "dynquantmatmul4bit": ArgSpec(positions=(0, 1), tensor_kwargs=("inp", "packed_weights")),
    "dynquantpack4bitweight": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("weights", "scales_zeros", "bias")
    ),
    "efficientzerotensor": ArgSpec(),
    "embeddingbag": ArgSpec(
        positions=(0, 1, 2, 6), tensor_kwargs=("weight", "indices", "offsets", "per_sample_weights")
    ),
    "embeddingbagforwardonly": ArgSpec(
        positions=(0, 1, 2, 6), tensor_kwargs=("weight", "indices", "offsets", "per_sample_weights")
    ),
    "embeddingrenorm": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "indices")),
    "emptyaffinequantized": ArgSpec(),
    "emptyperchannelaffinequantized": ArgSpec(tensor_kwargs=("scales", "zero_points")),
    "euclideandist": ArgSpec(positions=(0, 1), tensor_kwargs=("x1", "x2")),
    "fakequantizelearnableperchannelaffine": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "scale", "zero_point")
    ),
    "fakequantizelearnablepertensoraffine": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "scale", "zero_point")
    ),
    "fakequantizepertensoraffinecachemasktensorqparams": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("self", "scale", "zero_point", "fake_quant_enabled")
    ),
    "fbgemmlinearfp16weight": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("input", "packed_weight", "bias")
    ),
    "fbgemmlinearfp16weightfp32activation": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("input", "packed_weight", "bias")
    ),
    "fbgemmlinearint8weight": ArgSpec(
        positions=(0, 1, 2, 3, 6),
        tensor_kwargs=("input", "weight", "packed", "col_offsets", "bias"),
    ),
    "fbgemmlinearint8weightfp32activation": ArgSpec(
        positions=(0, 1, 2, 3, 6),
        tensor_kwargs=("input", "weight", "packed", "col_offsets", "bias"),
    ),
    "fbgemmlinearquantizeweight": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "fbgemmpackgemmmatrixfp16": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "fbgemmpackquantizedmatrix": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "featuredropout": ArgSpec(positions=(0,), tensor_kwargs=("input", "self")),
    "fftc2c": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "fftc2r": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "fftr2c": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "fillmemeffdropoutmask": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "floordiv": ArgSpec(positions=(0, 1)),
    "foobar": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "foreachabs": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachacos": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachadd": ArgSpec(
        positions=(1,), sequence_positions=(0, 1), tensor_kwargs=("self", "other")
    ),
    "foreachaddcdiv": ArgSpec(
        positions=(3,),
        sequence_positions=(0, 1, 2),
        tensor_kwargs=("self", "tensor1", "tensor2", "scalars"),
    ),
    "foreachaddcmul": ArgSpec(
        positions=(3,),
        sequence_positions=(0, 1, 2),
        tensor_kwargs=("self", "tensor1", "tensor2", "scalars"),
    ),
    "foreachasin": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachatan": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachceil": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachclampmax": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "other")),
    "foreachclampmin": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "other")),
    "foreachcopy": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "src")),
    "foreachcos": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachcosh": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachdiv": ArgSpec(
        positions=(1,), sequence_positions=(0, 1), tensor_kwargs=("self", "other")
    ),
    "foreacherf": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreacherfc": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachexp": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachexpm1": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachfloor": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachfrac": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachlerp": ArgSpec(
        sequence_positions=(0, 1, 2), tensor_kwargs=("self", "tensors1", "weights")
    ),
    "foreachlgamma": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachlog": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachlog10": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachlog1p": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachlog2": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachmax": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachmaximum": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "other")),
    "foreachminimum": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "other")),
    "foreachmul": ArgSpec(
        positions=(1,), sequence_positions=(0, 1), tensor_kwargs=("self", "other")
    ),
    "foreachneg": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachnorm": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachpow": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "exponent")),
    "foreachreciprocal": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachround": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachrsqrt": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsigmoid": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsign": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsin": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsinh": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsqrt": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachsub": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "other")),
    "foreachtan": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachtanh": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachtrunc": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachzero": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "fractionalmaxpool2dwithindices": ArgSpec(
        positions=(0, 5), tensor_kwargs=("input", "_random_samples")
    ),
    "fractionalmaxpool3dwithindices": ArgSpec(
        positions=(0, 5), tensor_kwargs=("input", "_random_samples")
    ),
    "frobeniusnorm": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "fusedadagrad": ArgSpec(
        sequence_positions=(0, 1, 2, 3),
        tensor_kwargs=(
            "self",
            "grads",
            "state_sums",
            "state_steps",
            "lr",
            "grad_scale",
            "found_inf",
        ),
    ),
    "fusedadam": ArgSpec(
        sequence_positions=(0, 1, 2, 3, 4, 5),
        tensor_kwargs=(
            "self",
            "grads",
            "exp_avgs",
            "exp_avg_sqs",
            "max_exp_avg_sqs",
            "state_steps",
            "lr",
            "grad_scale",
            "found_inf",
        ),
    ),
    "fusedadamw": ArgSpec(
        sequence_positions=(0, 1, 2, 3, 4, 5),
        tensor_kwargs=(
            "self",
            "grads",
            "exp_avgs",
            "exp_avg_sqs",
            "max_exp_avg_sqs",
            "state_steps",
            "lr",
            "grad_scale",
            "found_inf",
        ),
    ),
    "fuseddropout": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "fusedmovingavgobsfakequant": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6),
        tensor_kwargs=(
            "self",
            "observer_on",
            "fake_quant_on",
            "running_min",
            "running_max",
            "scale",
            "zero_point",
        ),
    ),
    "fusedmovingavgobsfqhelper": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6),
        tensor_kwargs=(
            "self",
            "observer_on",
            "fake_quant_on",
            "running_min",
            "running_max",
            "scale",
            "zero_point",
        ),
    ),
    "fusedrmsnorm": ArgSpec(positions=(0, 2), tensor_kwargs=("input", "weight")),
    "fusedsdpchoice": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("query", "key", "value", "attn_mask")
    ),
    "fusedsgd": ArgSpec(
        sequence_positions=(0, 1, 2),
        tensor_kwargs=("self", "grads", "momentum_buffer_list", "lr", "grad_scale", "found_inf"),
    ),
    "fwprimalcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "gridsampler": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid")),
    "gridsampler2d": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid")),
    "gridsampler2dcpufallback": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid")),
    "gridsampler3d": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "grid")),
    "groupedmm": ArgSpec(positions=(0, 1, 2, 3), tensor_kwargs=("self", "mat2", "offs", "bias")),
    "gru": ArgSpec(
        positions=(0, 1, 2),
        sequence_positions=(3,),
        tensor_kwargs=("input", "data", "batch_sizes", "hx", "params"),
    ),
    "grucell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5), tensor_kwargs=("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh")
    ),
    "hascompatibleshallowcopytype": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "from")),
    "histogramddbinedges": ArgSpec(positions=(0,), tensor_kwargs=("self", "weight")),
    "histogramddfrombincts": ArgSpec(positions=(0,), tensor_kwargs=("self", "weight")),
    "histogramddfrombintensors": ArgSpec(
        positions=(0,), sequence_positions=(1,), tensor_kwargs=("self", "bins", "weight")
    ),
    "indexputimpl": ArgSpec(
        positions=(0, 2), sequence_positions=(1,), tensor_kwargs=("self", "indices", "values")
    ),
    "indicescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "infersize": ArgSpec(),
    "inprojection": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        tensor_kwargs=("q", "k", "v", "w_q", "w_k", "w_v", "b_q", "b_k", "b_v"),
    ),
    "inprojectionpacked": ArgSpec(
        positions=(0, 1, 2, 3, 4), tensor_kwargs=("q", "k", "v", "w", "b")
    ),
    "intmm": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "mat2")),
    "intrepr": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "isalltrue": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "isanytrue": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "isdistributed": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "isneg": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "ispinned": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "lstm": ArgSpec(
        positions=(0, 1),
        sequence_positions=(2, 3),
        tensor_kwargs=("input", "data", "batch_sizes", "hx", "params"),
    ),
    "lstmcell": ArgSpec(
        positions=(0, 2, 3, 4, 5),
        sequence_positions=(1,),
        tensor_kwargs=("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh"),
    ),
    "lstmmps": ArgSpec(
        positions=(0,), sequence_positions=(1, 2), tensor_kwargs=("input", "hx", "params")
    ),
    "makedualcopy": ArgSpec(positions=(0, 1), tensor_kwargs=("primal", "tangent")),
    "makeperchannelquantizedtensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "scale", "zero_point")
    ),
    "makepertensorquantizedtensor": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "miopenbatchnorm": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "weight", "bias", "running_mean", "running_var"),
    ),
    "miopenconvolution": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")),
    "miopenconvolutionaddrelu": ArgSpec(
        positions=(0, 1, 2, 4), tensor_kwargs=("self", "weight", "z", "bias")
    ),
    "miopenconvolutionrelu": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")),
    "miopenconvolutiontranspose": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")
    ),
    "miopendepthwiseconvolution": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")
    ),
    "miopenrnn": ArgSpec(
        positions=(0, 3, 4, 13),
        sequence_positions=(1,),
        tensor_kwargs=("input", "weight", "hx", "cx", "dropout_state"),
    ),
    "mpsconvolution": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "weight", "bias")),
    "mpsconvolutiontranspose": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "weight")),
    "nativebatchnorm": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "weight", "bias", "running_mean", "running_var"),
    ),
    "nativebatchnormlegit": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "weight", "bias", "running_mean", "running_var"),
    ),
    "nativebatchnormlegitnotraining": ArgSpec(
        positions=(0, 1, 2, 3, 4),
        tensor_kwargs=("input", "weight", "bias", "running_mean", "running_var"),
    ),
    "nativedropout": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "nativegroupnorm": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias")),
    "nativelayernorm": ArgSpec(positions=(0, 2, 3), tensor_kwargs=("input", "weight", "bias")),
    "nativemultiheadattention": ArgSpec(
        positions=(0, 1, 2, 5, 6, 7, 8, 9),
        tensor_kwargs=(
            "query",
            "key",
            "value",
            "qkv_weight",
            "qkv_bias",
            "proj_weight",
            "proj_bias",
            "mask",
        ),
    ),
    "nativenorm": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "negview": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "negviewcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedcomputecontiguousstridesoffsets": ArgSpec(
        positions=(0,), tensor_kwargs=("nested_size",)
    ),
    "nestedfrompadded": ArgSpec(
        positions=(0, 1), tensor_kwargs=("padded", "cpu_nested_shape_example")
    ),
    "nestedfrompaddedandnestedexample": ArgSpec(
        positions=(0, 1), tensor_kwargs=("padded", "nt_example")
    ),
    "nestedfrompaddedtensor": ArgSpec(
        positions=(0, 1, 2, 4, 5),
        tensor_kwargs=("padded", "offsets", "dummy", "min_seqlen", "max_seqlen"),
    ),
    "nestedgetjaggeddummy": ArgSpec(positions=(0,), tensor_kwargs=("any",)),
    "nestedgetlengths": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetmaxseqlen": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetminseqlen": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetoffsets": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetraggedidx": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetvalues": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedgetvaluescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedtensorfrommask": ArgSpec(positions=(0, 1), tensor_kwargs=("t", "mask")),
    "nestedtensorfrommaskleftaligned": ArgSpec(positions=(0, 1), tensor_kwargs=("t", "mask")),
    "nestedtensorfromtensorlist": ArgSpec(sequence_positions=(0,), tensor_kwargs=("list",)),
    "nestedtensorsize": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedtensorsoftmaxwithshape": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "query")),
    "nestedtensorstorageoffsets": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedtensorstrides": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nestedviewfrombuffer": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("self", "nested_size", "nested_strides", "offsets")
    ),
    "nestedviewfrombuffercopy": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("self", "nested_size", "nested_strides", "offsets")
    ),
    "nestedviewfromjagged": ArgSpec(
        positions=(0, 1, 2, 3, 5, 6),
        tensor_kwargs=("self", "offsets", "dummy", "lengths", "min_seqlen", "max_seqlen"),
    ),
    "nestedviewfromjaggedcopy": ArgSpec(
        positions=(0, 1, 2, 3, 5, 6),
        tensor_kwargs=("self", "offsets", "dummy", "lengths", "min_seqlen", "max_seqlen"),
    ),
    "nnpackavailable": ArgSpec(),
    "nnpackspatialconvolution": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("input", "weight", "bias")
    ),
    "nnz": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "nogradembeddingrenorm": ArgSpec(positions=(0, 1), tensor_kwargs=("weight", "input")),
    "nogradfill": ArgSpec(positions=(0,), tensor_kwargs=("tensor",)),
    "nogradnormal": ArgSpec(positions=(0,), tensor_kwargs=("tensor",)),
    "nogradtruncnormal": ArgSpec(positions=(0,), tensor_kwargs=("tensor",)),
    "nogradzero": ArgSpec(positions=(0,), tensor_kwargs=("tensor",)),
    "nonzerostatic": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "normexceptdim": ArgSpec(positions=(0,), tensor_kwargs=("v",)),
    "nuclearnorm": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "permutecopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "prelukernel": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "weight")),
    "put": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "input", "index", "source")),
    "qperchannelaxis": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "qperchannelscales": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "qperchannelzeropoints": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "qscale": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "qscheme": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "quantizedbatchnorm": ArgSpec(
        positions=(0, 1, 2, 3, 4), tensor_kwargs=("input", "weight", "bias", "mean", "var")
    ),
    "quantizedgrucell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        tensor_kwargs=(
            "input",
            "hx",
            "w_ih",
            "w_hh",
            "b_ih",
            "b_hh",
            "packed_ih",
            "packed_hh",
            "col_offsets_ih",
            "col_offsets_hh",
        ),
    ),
    "quantizedlstmcell": ArgSpec(
        positions=(0, 2, 3, 4, 5, 6, 7, 8, 9),
        sequence_positions=(1,),
        tensor_kwargs=(
            "input",
            "hx",
            "w_ih",
            "w_hh",
            "b_ih",
            "b_hh",
            "packed_ih",
            "packed_hh",
            "col_offsets_ih",
            "col_offsets_hh",
        ),
    ),
    "quantizedmaxpool1d": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "quantizedmaxpool2d": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "quantizedmaxpool3d": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "quantizedrnnrelucell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        tensor_kwargs=(
            "input",
            "hx",
            "w_ih",
            "w_hh",
            "b_ih",
            "b_hh",
            "packed_ih",
            "packed_hh",
            "col_offsets_ih",
            "col_offsets_hh",
        ),
    ),
    "quantizedrnntanhcell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        tensor_kwargs=(
            "input",
            "hx",
            "w_ih",
            "w_hh",
            "b_ih",
            "b_hh",
            "packed_ih",
            "packed_hh",
            "col_offsets_ih",
            "col_offsets_hh",
        ),
    ),
    "qzeropoint": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "range": ArgSpec(),
    "reduceex": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "registerpostaccumulategradhook": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "reshapealiascopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "reshapefromtensor": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "shape")),
    "resizeas": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "the_template")),
    "resizeassparse": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "the_template")),
    "resizeoutput": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "rmsnorm": ArgSpec(positions=(0, 2), tensor_kwargs=("input", "weight")),
    "rnnrelu": ArgSpec(
        positions=(0, 1, 2),
        sequence_positions=(3,),
        tensor_kwargs=("input", "data", "batch_sizes", "hx", "params"),
    ),
    "rnnrelucell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5), tensor_kwargs=("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh")
    ),
    "rnntanh": ArgSpec(
        positions=(0, 1, 2),
        sequence_positions=(3,),
        tensor_kwargs=("input", "data", "batch_sizes", "hx", "params"),
    ),
    "rnntanhcell": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5), tensor_kwargs=("input", "hx", "w_ih", "w_hh", "b_ih", "b_hh")
    ),
    "rowindicescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "rowwiseprune": ArgSpec(positions=(0, 1), tensor_kwargs=("weight", "mask")),
    "safesoftmax": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sampledirichlet": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "saturateweighttofp16": ArgSpec(positions=(0,), tensor_kwargs=("weight",)),
    "scaleddotproductattentionmath": ArgSpec(
        positions=(0, 1, 2, 3, 6),
        tensor_kwargs=("query", "key", "value", "attn_mask", "dropout_mask"),
    ),
    "scaleddotproductattentionmathformps": ArgSpec(
        positions=(0, 1, 2, 3, 6),
        tensor_kwargs=("query", "key", "value", "attn_mask", "dropout_mask"),
    ),
    "scaleddotproductcudnnattention": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("query", "key", "value", "attn_bias")
    ),
    "scaleddotproductefficientattention": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("query", "key", "value", "attn_bias")
    ),
    "scaleddotproductflashattention": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("query", "key", "value")
    ),
    "scaleddotproductflashattentionforcpu": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("query", "key", "value", "attn_mask")
    ),
    "scaledgroupedmm": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5, 6),
        tensor_kwargs=("self", "mat2", "scale_a", "scale_b", "offs", "bias", "scale_result"),
    ),
    "scaledmm": ArgSpec(
        positions=(0, 1, 2, 3, 4, 5),
        tensor_kwargs=("self", "mat2", "scale_a", "scale_b", "bias", "scale_result"),
    ),
    "segmentreduce": ArgSpec(
        positions=(0,), tensor_kwargs=("data", "lengths", "indices", "offsets")
    ),
    "selectcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "shapeastensor": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "slicecopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sliceinverse": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "src")),
    "sobolenginedraw": ArgSpec(positions=(0, 2), tensor_kwargs=("quasi", "sobolstate")),
    "sobolengineff": ArgSpec(positions=(0, 2), tensor_kwargs=("self", "sobolstate")),
    "sobolengineinitializestate": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sobolenginescramble": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "ltm")),
    "sparsebroadcastto": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparsebroadcasttocopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparsecootensor": ArgSpec(positions=(0, 1), tensor_kwargs=("indices", "values")),
    "sparsecsrprod": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparsecsrsum": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparsecsrtensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("crow_indices", "col_indices", "values")
    ),
    "sparsemaskprojection": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "mask")),
    "sparseresize": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparseresizeandclear": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sparsesemistructuredaddmm": ArgSpec(
        positions=(0, 1, 2, 3), tensor_kwargs=("input", "mat1", "mat1_meta", "mat2")
    ),
    "sparsesemistructuredapply": ArgSpec(positions=(0, 1), tensor_kwargs=("input", "thread_masks")),
    "sparsesemistructuredapplydense": ArgSpec(
        positions=(0, 1), tensor_kwargs=("input", "thread_masks")
    ),
    "sparsesemistructuredlinear": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("input", "weight", "meta", "bias")
    ),
    "sparsesemistructuredmm": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("mat1", "mat1_meta", "mat2")
    ),
    "sparsesemistructuredtile": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "sparsesparsematmul": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "other")),
    "sparsesum": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sphericalbesselj0": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "splitcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "splitwithsizescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "squeezecopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "standardgamma": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "sumtosize": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "tcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "tocpu": ArgSpec(sequence_positions=(0,), tensor_kwargs=("tensors",)),
    "tosparsesemistructured": ArgSpec(positions=(0,), tensor_kwargs=("dense",)),
    "transformbiasrescaleqkv": ArgSpec(positions=(0, 1), tensor_kwargs=("qkv", "qkv_bias")),
    "transformerencoderlayerfwd": ArgSpec(
        positions=(0, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        tensor_kwargs=(
            "src",
            "qkv_weight",
            "qkv_bias",
            "proj_weight",
            "proj_bias",
            "norm_weight_1",
            "norm_bias_1",
            "norm_weight_2",
            "norm_bias_2",
            "ffn_weight_1",
            "ffn_bias_1",
            "ffn_weight_2",
            "ffn_bias_2",
            "mask",
        ),
    ),
    "transposecopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "trilinear": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("i1", "i2", "i3")),
    "tritonmultiheadattention": ArgSpec(
        positions=(0, 1, 2, 5, 6, 7, 8, 9),
        tensor_kwargs=(
            "query",
            "key",
            "value",
            "qkv_weight",
            "qkv_bias",
            "proj_weight",
            "proj_bias",
            "mask",
        ),
    ),
    "tritonscaleddotattention": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("q", "k", "v")),
    "truediv": ArgSpec(positions=(0, 1), tensor_kwargs=("self", "input", "other")),
    "unbindcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "unfoldcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "unique2": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "unpooloutputsize": ArgSpec(positions=(0,), tensor_kwargs=("input",)),
    "unsafeindex": ArgSpec(
        positions=(0,), sequence_positions=(1,), tensor_kwargs=("self", "indices")
    ),
    "unsafeindexput": ArgSpec(
        positions=(0, 2), sequence_positions=(1,), tensor_kwargs=("self", "indices", "values")
    ),
    "unsafemaskedindex": ArgSpec(
        positions=(0, 1), sequence_positions=(2,), tensor_kwargs=("self", "mask", "indices")
    ),
    "unsafemaskedindexputaccumulate": ArgSpec(
        positions=(0, 1, 3),
        sequence_positions=(2,),
        tensor_kwargs=("self", "mask", "indices", "values"),
    ),
    "unsafesplitwithsizes": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "unsqueezecopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "updatenames": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "usecudnnctcloss": ArgSpec(
        positions=(0, 1, 2, 3),
        tensor_kwargs=("log_probs", "targets", "input_lengths", "target_lengths"),
    ),
    "usecudnnrnnflattenweight": ArgSpec(),
    "validatecompressedsparseindices": ArgSpec(
        positions=(1, 2), tensor_kwargs=("compressed_idx", "plain_idx")
    ),
    "validatesparsebsctensorargs": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("ccol_indices", "row_indices", "values")
    ),
    "validatesparsebsrtensorargs": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("crow_indices", "col_indices", "values")
    ),
    "validatesparsecompressedtensorargs": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("compressed_indices", "plain_indices", "values")
    ),
    "validatesparsecootensorargs": ArgSpec(positions=(0, 1), tensor_kwargs=("indices", "values")),
    "validatesparsecsctensorargs": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("ccol_indices", "row_indices", "values")
    ),
    "validatesparsecsrtensorargs": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("crow_indices", "col_indices", "values")
    ),
    "valuescopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "viewcopy": ArgSpec(positions=(0,), tensor_kwargs=("self",)),
    "weightint4packmm": ArgSpec(
        positions=(0, 1, 3), tensor_kwargs=("self", "mat2", "qScaleAndZeros")
    ),
    "weightint4packmmforcpu": ArgSpec(
        positions=(0, 1, 3), tensor_kwargs=("self", "mat2", "qScaleAndZeros")
    ),
    "weightint4packmmwithscalesandzeros": ArgSpec(
        positions=(0, 1, 3, 4), tensor_kwargs=("self", "mat2", "qScale", "qZeros")
    ),
    "weightint8packmm": ArgSpec(positions=(0, 1, 2), tensor_kwargs=("self", "mat2", "scales")),
    "weightnorm": ArgSpec(positions=(0, 1), tensor_kwargs=("v", "g")),
    "weightnorminterface": ArgSpec(positions=(0, 1), tensor_kwargs=("v", "g")),
}

for _name, _spec in _PHASE5B_VALIDATED_ARG_SPECS.items():
    FUNC_ARG_SPECS[_name] = _spec

# Tensor iterator and subclass custom_methods
for _name in [
    "iter",
    "initsubclass",
    "torchfunction",
    "new",
    "subclasshook",
    "makesubclass",
    "reinforce",
]:
    FUNC_ARG_SPECS[_name] = _P0

# Cleanup loop variable leakage
del _name, _spec
