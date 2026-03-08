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
from typing import Dict, List, Tuple

import torch

from .. import _state


def _normalize_func_name(func_name: str) -> str:
    """Normalize a raw function name for lookup table keying."""
    return func_name.lower().replace("_", "")


@dataclass(frozen=True)
class ArgSpec:
    """Which argument positions can hold tensors or parameters.

    Attributes:
        positions: Positional arg indices that can hold a single tensor/parameter.
        sequence_positions: Indices holding sequences (list/tuple) of tensors.
        tensor_kwargs: Keyword argument names that can hold tensors/parameters.
    """

    positions: Tuple[int, ...] = ()
    sequence_positions: Tuple[int, ...] = ()
    tensor_kwargs: Tuple[str, ...] = ()


def extract_tensors_and_params(
    spec: ArgSpec,
    args: tuple,
    kwargs: dict,
) -> Tuple[List[torch.Tensor], List[torch.nn.Parameter]]:
    """Extract tensors (excluding Parameters) and Parameters from known arg positions.

    Returns:
        (arg_tensors, arg_parameters) — matches get_vars_of_type_from_obj output.
    """
    tensors: List[torch.Tensor] = []
    params: List[torch.nn.Parameter] = []

    for pos in spec.positions:
        if pos < len(args):
            arg = args[pos]
            if isinstance(arg, torch.nn.Parameter):
                params.append(arg)
            elif isinstance(arg, torch.Tensor):
                tensors.append(arg)

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
        val = kwargs.get(name)
        if val is not None:
            if isinstance(val, torch.nn.Parameter):
                params.append(val)
            elif isinstance(val, torch.Tensor):
                tensors.append(val)

    return tensors, params


def _cache_dynamic_spec(
    normalized_name: str,
    args: tuple,
    kwargs: dict,
    found_tensors: list,
    found_params: list,
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
_P01 = ArgSpec(positions=(0, 1))
_P012 = ArgSpec(positions=(0, 1, 2))
_P0123 = ArgSpec(positions=(0, 1, 2, 3))
_S0 = ArgSpec(sequence_positions=(0,))
_NONE = ArgSpec()

# ============================================================================
# FUNC_ARG_SPECS — keyed by normalized func_name
# ============================================================================

FUNC_ARG_SPECS: Dict[str, ArgSpec] = {}

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
    # Hook/utility methods
    "registerhook",
    "isinteger",
    # Dunder / internal
    "index",
    "symint",
    "checkkeypaddingmask",
]

for _name in _UNARY_FUNCS:
    FUNC_ARG_SPECS[_name] = _P0

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
    FUNC_ARG_SPECS[_name] = _P01

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
    "zeroslike",
    "oneslike",
    "randlike",
    "randnlike",
    "emptylike",
    "fulllike",
    "newtensor",
    "newempty",
    "newemptystrided",
    "newzeros",
    "newones",
    "newfull",
    "load",
]

for _name in _FACTORY_FUNCS:
    FUNC_ARG_SPECS[_name] = _NONE

# ---------------------------------------------------------------------------
# Special patterns (custom ArgSpec per function or group)
# ---------------------------------------------------------------------------

# __getitem__: self + index (can be tensor or tuple of tensors)
FUNC_ARG_SPECS["getitem"] = ArgSpec(positions=(0, 1), sequence_positions=(1,))

# __setitem__: self + index + value
FUNC_ARG_SPECS["setitem"] = ArgSpec(positions=(0, 1, 2), sequence_positions=(1,))

# __delitem__: just self
FUNC_ARG_SPECS["delitem"] = _P0

# scatter/index_copy/index_fill: (self, dim, index, src/value)
_SCATTER_SPEC = ArgSpec(positions=(0, 2, 3))
for _name in ["scatter", "indexcopy", "indexfill"]:
    FUNC_ARG_SPECS[_name] = _SCATTER_SPEC

# gather/index_select: (self, dim, index)
_GATHER_SPEC = ArgSpec(positions=(0, 2))
for _name in ["gather", "indexselect"]:
    FUNC_ARG_SPECS[_name] = _GATHER_SPEC

# index_put: (self, indices_tuple, values)
FUNC_ARG_SPECS["indexput"] = ArgSpec(positions=(0, 2), sequence_positions=(1,))

# linear: F.linear(input, weight, bias) — bias often passed positionally
FUNC_ARG_SPECS["linear"] = _P012

# conv: F.conv2d(input, weight, bias, stride, ...) — bias at position 2
for _name in [
    "conv1d",
    "conv2d",
    "conv3d",
    "convtranspose1d",
    "convtranspose2d",
    "convtranspose3d",
]:
    FUNC_ARG_SPECS[_name] = _P012

# CUDNN conv variants
for _name in [
    "cudnnconvolution",
    "cudnnconvolutiontranspose",
    "cudnnconvolutionrelu",
    "cudnnconvolutionaddrely",
]:
    FUNC_ARG_SPECS[_name] = _P012

# batch_norm: F.batch_norm(input, running_mean, running_var, weight, bias, ...)
# All 5 args commonly passed positionally by nn.BatchNorm*.forward()
FUNC_ARG_SPECS["batchnorm"] = ArgSpec(positions=(0, 1, 2, 3, 4))
FUNC_ARG_SPECS["cudnnbatchnorm"] = ArgSpec(positions=(0, 1, 2, 3, 4))

# instance_norm: similar to batch_norm
FUNC_ARG_SPECS["instancenorm"] = ArgSpec(positions=(0, 1, 2, 3, 4))

# layer_norm: F.layer_norm(input, normalized_shape, weight, bias, ...)
# weight at pos 2, bias at pos 3
FUNC_ARG_SPECS["layernorm"] = ArgSpec(positions=(0, 2, 3))

# group_norm: F.group_norm(input, num_groups, weight, bias, ...)
# weight at pos 2, bias at pos 3
FUNC_ARG_SPECS["groupnorm"] = ArgSpec(positions=(0, 2, 3))

# Loss functions: (input, target, weight=None, ...)
# weight can be positional (pos 2) or kwarg
_LOSS_WITH_WEIGHT = ArgSpec(positions=(0, 1, 2), tensor_kwargs=("weight",))
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
    FUNC_ARG_SPECS[_name] = _P01

# bilinear: (input1, input2, weight, bias) — bias can be positional
FUNC_ARG_SPECS["bilinear"] = ArgSpec(positions=(0, 1, 2, 3))

# scaled_dot_product_attention: (query, key, value, attn_mask=None, ...)
# attn_mask can be positional (pos 3) or kwarg
FUNC_ARG_SPECS["scaleddotproductattention"] = ArgSpec(
    positions=(0, 1, 2, 3), tensor_kwargs=("attnmask",)
)

# multi_head_attention_forward: (query, key, value, ...)
FUNC_ARG_SPECS["multiheadattentionforward"] = _P012

# grid_sample: (input, grid)
FUNC_ARG_SPECS["gridsample"] = _P01
FUNC_ARG_SPECS["cudnngridsampler"] = _P01
FUNC_ARG_SPECS["cudnnaffinegridgenerator"] = _P0

# affine_grid: (theta, size) — theta is a tensor
FUNC_ARG_SPECS["affinegrid"] = _P0

# one_hot: (tensor, num_classes)
FUNC_ARG_SPECS["onehot"] = _P0

# Tensor iterator and subclass methods
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
del _name
