"""
FLOPs (Floating Point Operations) computation module for TorchLens.

Computes per-layer forward and backward FLOPs based on operation type and
tensor shapes. Based on whisperLiang's contribution in PR #53.

Conventions:
- Multiply-accumulate (MAC) = 2 FLOPs (1 mul + 1 add)
- Transcendental functions counted at empirical cost (exp=8, sigmoid=4, etc.)
- Memory-only operations (view, reshape, transpose) = 0 FLOPs
- Unknown operations return None (not counted, not guessed)

Maintenance — checking coverage against the full op list:
    from torchlens.constants import ORIG_TORCH_FUNCS
    from torchlens.flops import ZERO_FLOPS_OPS, ELEMENTWISE_FLOPS, SPECIALTY_HANDLERS
    all_names = {name for _, name in ORIG_TORCH_FUNCS}
    covered = ZERO_FLOPS_OPS | set(ELEMENTWISE_FLOPS) | set(SPECIALTY_HANDLERS)
    uncovered = all_names - covered
    print(f"Covered: {len(covered)}, Uncovered: {len(uncovered)}")
    # Most uncovered ops are internal/private (_*) or non-compute (type casts, etc.)
    # Run scripts/check_flops_coverage.py for a detailed breakdown.
"""

from typing import Dict, Optional, Tuple
import math


# ============================================================================
# Helpers
# ============================================================================


def _prod(shape) -> int:
    """Product of a shape tuple."""
    result = 1
    for v in shape:
        result *= v
    return result


def _numel(shape) -> int:
    """Number of elements from a shape tuple. Scalar shape () = 1 element."""
    if shape is None:
        return 0
    return _prod(shape)


def _safe_shape(t) -> Optional[Tuple[int, ...]]:
    """Safely extract shape tuple from a tensor or shape tuple."""
    if t is None:
        return None
    if isinstance(t, tuple):
        return t
    try:
        return tuple(t.shape)
    except Exception:
        return None


# ============================================================================
# Zero-cost operations (memory layout only, no compute)
# ============================================================================

ZERO_FLOPS_OPS = {
    # Reshape / view
    "view",
    "view_as",
    "reshape",
    "reshape_as",
    "flatten",
    "unflatten",
    "ravel",
    "contiguous",
    "narrow",
    "narrow_copy",
    # Transpose / permute
    "t",
    "t_",
    "transpose",
    "transpose_",
    "permute",
    "adjoint",
    "swapaxes",
    "swapaxes_",
    "swapdims",
    "swapdims_",
    "moveaxis",
    "movedim",
    # Squeeze / unsqueeze / expand
    "squeeze",
    "squeeze_",
    "unsqueeze",
    "unsqueeze_",
    "expand",
    "expand_as",
    "expand_copy",
    # Split / chunk / cat / stack (memory ops, not compute)
    "cat",
    "concat",
    "concatenate",
    "stack",
    "hstack",
    "vstack",
    "dstack",
    "row_stack",
    "column_stack",
    "block_diag",
    "chunk",
    "split",
    "split_with_sizes",
    "tensor_split",
    "hsplit",
    "vsplit",
    "dsplit",
    "unbind",
    # Copy / clone
    "clone",
    "detach",
    "detach_",
    "contiguous",
    "copy_",
    # Indexing / gathering (memory access, not arithmetic)
    "__getitem__",
    "__setitem__",
    "index_select",
    "gather",
    "masked_select",
    "take",
    "take_along_dim",
    "select",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "index_put",
    "index_put_",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_reduce",
    "index_reduce_",
    # Type / device conversion
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
    "type_as",
    "cfloat",
    "cdouble",
    "chalf",
    # Shape queries (no compute)
    "size",
    "dim",
    "numel",
    "nelement",
    "ndimension",
    "element_size",
    "is_contiguous",
    "is_complex",
    "is_floating_point",
    "is_signed",
    # Misc non-compute
    "pin_memory",
    "share_memory_",
    "record_stream",
    "storage",
    "storage_offset",
    "data_ptr",
    "untyped_storage",
    "storage_type",
    "set_",
    "fill_",
    "fill_diagonal_",
    "zero_",
    "requires_grad_",
    "retain_grad",
    "detach_copy",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_subclass",
    "repeat",
    "repeat_interleave",
    "tile",
    "roll",
    "rot90",
    "flip",
    "fliplr",
    "flipud",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "diag",
    "diag_embed",
    "diagflat",
    "tril",
    "triu",
    "tril_",
    "triu_",
    "select_scatter",
    "slice_scatter",
    "as_strided_scatter",
    "view_as_real",
    "view_as_complex",
    "view_as_real_copy",
    "view_as_complex_copy",
    "real",
    "imag",
    "resolve_conj",
    "resolve_neg",
    "conj",
    "conj_physical",
    "conj_physical_",
    "T",
    "mT",
    "H",
    "unfold",
    "fold",
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "native_channel_shuffle",
    "meshgrid",
    "cartesian_prod",
    "combinations",
    "broadcast_to",
    "broadcast_tensors",
    "align_tensors",
    "align_as",
    "align_to",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "rename",
    "rename_",
    "refine_names",
    "has_names",
    # Quantization / sparse (layout, not compute)
    "to_sparse",
    "to_dense",
    "to_mkldnn",
    "coalesce",
    "sparse_mask",
    "is_coalesced",
    "sparse_dim",
    "dense_dim",
    "indices",
    "values",
    "crow_indices",
    "col_indices",
    "row_indices",
    "ccol_indices",
    # Conversion
    "item",
    "tolist",
    "numpy",
}


# ============================================================================
# Element-wise operations: FLOPs per output element
# ============================================================================

ELEMENTWISE_FLOPS: Dict[str, int] = {
    # Basic arithmetic (1 FLOP per element)
    "add": 1,
    "add_": 1,
    "__add__": 1,
    "__iadd__": 1,
    "__radd__": 1,
    "sub": 1,
    "sub_": 1,
    "subtract": 1,
    "subtract_": 1,
    "__sub__": 1,
    "__isub__": 1,
    "__rsub__": 1,
    "rsub": 1,
    "mul": 1,
    "mul_": 1,
    "multiply": 1,
    "multiply_": 1,
    "__mul__": 1,
    "__imul__": 1,
    "__rmul__": 1,
    "div": 1,
    "div_": 1,
    "divide": 1,
    "divide_": 1,
    "true_divide": 1,
    "true_divide_": 1,
    "__div__": 1,
    "__idiv__": 1,
    "__rdiv__": 1,
    "__truediv__": 1,
    "__itruediv__": 1,
    "__rtruediv__": 1,
    "floor_divide": 1,
    "floor_divide_": 1,
    "__floordiv__": 1,
    "__ifloordiv__": 1,
    "__rfloordiv__": 1,
    "remainder": 1,
    "remainder_": 1,
    "fmod": 1,
    "fmod_": 1,
    "__mod__": 1,
    "__imod__": 1,
    "__rmod__": 1,
    "neg": 1,
    "neg_": 1,
    "negative": 1,
    "negative_": 1,
    "__neg__": 1,
    "pos": 1,
    "__pos__": 1,
    "positive": 1,
    "abs": 1,
    "abs_": 1,
    "absolute": 1,
    "absolute_": 1,
    "__abs__": 1,
    "sign": 1,
    "sign_": 1,
    "sgn": 1,
    "sgn_": 1,
    "signbit": 1,
    "ceil": 1,
    "ceil_": 1,
    "floor": 1,
    "floor_": 1,
    "round": 1,
    "round_": 1,
    "trunc": 1,
    "trunc_": 1,
    "frac": 1,
    "frac_": 1,
    "fix": 1,
    "fix_": 1,
    "reciprocal": 1,
    "reciprocal_": 1,
    "square": 1,
    "square_": 1,
    "clamp": 1,
    "clamp_": 1,
    "clamp_min": 1,
    "clamp_min_": 1,
    "clamp_max": 1,
    "clamp_max_": 1,
    "clip": 1,
    "clip_": 1,
    "nan_to_num": 1,
    "nan_to_num_": 1,
    # Comparison / logical (1 FLOP per element)
    "eq": 1,
    "eq_": 1,
    "__eq__": 1,
    "ne": 1,
    "ne_": 1,
    "__ne__": 1,
    "gt": 1,
    "gt_": 1,
    "__gt__": 1,
    "lt": 1,
    "lt_": 1,
    "__lt__": 1,
    "ge": 1,
    "ge_": 1,
    "__ge__": 1,
    "le": 1,
    "le_": 1,
    "__le__": 1,
    "greater": 1,
    "greater_": 1,
    "greater_equal": 1,
    "greater_equal_": 1,
    "less": 1,
    "less_": 1,
    "less_equal": 1,
    "less_equal_": 1,
    "equal": 1,
    "not_equal": 1,
    "not_equal_": 1,
    "isclose": 1,
    "allclose": 1,
    "maximum": 1,
    "minimum": 1,
    "fmax": 1,
    "fmin": 1,
    "where": 1,
    "masked_fill": 1,
    "masked_fill_": 1,
    "masked_scatter": 1,
    "masked_scatter_": 1,
    "logical_and": 1,
    "logical_and_": 1,
    "logical_or": 1,
    "logical_or_": 1,
    "logical_not": 1,
    "logical_not_": 1,
    "logical_xor": 1,
    "logical_xor_": 1,
    "bitwise_and": 1,
    "bitwise_and_": 1,
    "bitwise_or": 1,
    "bitwise_or_": 1,
    "bitwise_not": 1,
    "bitwise_not_": 1,
    "bitwise_xor": 1,
    "bitwise_xor_": 1,
    "bitwise_left_shift": 1,
    "bitwise_left_shift_": 1,
    "bitwise_right_shift": 1,
    "bitwise_right_shift_": 1,
    "__and__": 1,
    "__iand__": 1,
    "__rand__": 1,
    "__or__": 1,
    "__ior__": 1,
    "__ror__": 1,
    "__xor__": 1,
    "__ixor__": 1,
    "__rxor__": 1,
    "__invert__": 1,
    "__lshift__": 1,
    "__ilshift__": 1,
    "__rlshift__": 1,
    "__rshift__": 1,
    "__irshift__": 1,
    "__rrshift__": 1,
    "isnan": 1,
    "isinf": 1,
    "isfinite": 1,
    "isneginf": 1,
    "isposinf": 1,
    "isreal": 1,
    # Simple activations (1 FLOP per element)
    "relu": 1,
    "relu_": 1,
    "relu6": 1,
    "threshold": 1,
    "threshold_": 1,
    "hardtanh": 1,
    "hardtanh_": 1,
    "leaky_relu": 1,
    "leaky_relu_": 1,
    "rrelu": 1,
    "rrelu_": 1,
    "prelu": 1,
    "heaviside": 1,
    "heaviside_": 1,
    # Power (2 FLOPs: compute + result)
    "pow": 2,
    "pow_": 2,
    "__pow__": 2,
    "__ipow__": 2,
    "__rpow__": 2,
    "float_power": 2,
    "float_power_": 2,
    "sqrt": 2,
    "sqrt_": 2,
    "rsqrt": 2,
    "rsqrt_": 2,
    # Compound element-wise (multiple FLOPs per element)
    "addcdiv": 3,
    "addcdiv_": 3,  # div + mul + add
    "addcmul": 3,
    "addcmul_": 3,  # mul + mul + add
    "lerp": 3,
    "lerp_": 3,  # (1-w)*a + w*b
    "copysign": 2,
    "copysign_": 2,
    "nextafter": 2,
    "nextafter_": 2,
    "hypot": 3,
    "hypot_": 3,  # sqrt(a^2 + b^2)
    "xlogy": 10,
    "xlogy_": 10,  # x * log(y)
    "xlog1py": 10,  # x * log1p(y)
    # Transcendental functions
    "exp": 8,
    "exp_": 8,
    "exp2": 8,
    "exp2_": 8,
    "expm1": 8,
    "expm1_": 8,
    "log": 8,
    "log_": 8,
    "log2": 8,
    "log2_": 8,
    "log10": 8,
    "log10_": 8,
    "log1p": 8,
    "log1p_": 8,
    "sin": 8,
    "sin_": 8,
    "cos": 8,
    "cos_": 8,
    "tan": 8,
    "tan_": 8,
    "asin": 8,
    "asin_": 8,
    "acos": 8,
    "acos_": 8,
    "atan": 8,
    "atan_": 8,
    "arcsin": 8,
    "arcsin_": 8,
    "arccos": 8,
    "arccos_": 8,
    "arctan": 8,
    "arctan_": 8,
    "atan2": 8,
    "atan2_": 8,
    "arctan2": 8,
    "arctan2_": 8,
    "sinh": 8,
    "sinh_": 8,
    "cosh": 8,
    "cosh_": 8,
    "tanh": 6,
    "tanh_": 6,
    "asinh": 8,
    "asinh_": 8,
    "acosh": 8,
    "acosh_": 8,
    "atanh": 8,
    "atanh_": 8,
    "arcsinh": 8,
    "arcsinh_": 8,
    "arccosh": 8,
    "arccosh_": 8,
    "arctanh": 8,
    "arctanh_": 8,
    "sinc": 10,
    "sinc_": 10,  # sin(πx)/(πx)
    "erf": 8,
    "erf_": 8,
    "erfc": 8,
    "erfc_": 8,
    "erfinv": 8,
    "erfinv_": 8,
    "lgamma": 12,
    "lgamma_": 12,
    "digamma": 12,
    "digamma_": 12,
    "polygamma": 12,
    "polygamma_": 12,
    "mvlgamma": 12,
    "mvlgamma_": 12,
    "i0": 8,
    "i0_": 8,
    "i0e": 8,
    "i1": 8,
    "i1e": 8,
    "deg2rad": 1,
    "deg2rad_": 1,
    "rad2deg": 1,
    "rad2deg_": 1,
    # Activation functions (compound)
    "sigmoid": 4,
    "sigmoid_": 4,  # neg + exp + add + div
    "hardsigmoid": 3,
    "hardswish": 4,
    "silu": 5,  # x * sigmoid(x)
    "mish": 8,  # x * tanh(softplus(x))
    "gelu": 14,  # 0.5 * x * (1 + erf(x/sqrt(2)))
    "celu": 9,
    "celu_": 9,  # max(0,x) + min(0,alpha*(exp(x/alpha)-1))
    "elu": 9,
    "elu_": 9,  # same as celu
    "selu": 10,
    "selu_": 10,  # scale * elu(x, alpha)
    "softplus": 9,  # log(1 + exp(x))
    "softshrink": 2,
    "hardshrink": 2,
    "tanhshrink": 7,  # x - tanh(x)
    "softsign": 3,  # x / (1 + |x|)
    "logsigmoid": 5,  # log(sigmoid(x))
    "logit": 12,
    "logit_": 12,  # log(x / (1 - x))
    "expit": 4,  # same as sigmoid
}


# ============================================================================
# Specialty handlers: operations needing shape-specific formulas
# ============================================================================


def _matmul_flops(output_shape, parent_param_shapes, creation_args):
    """MatMul family: mm, matmul, __matmul__, bmm, addmm, etc."""
    # Try to get input shapes from creation_args
    if not creation_args:
        return None
    shapes = []
    for arg in creation_args:
        s = _safe_shape(arg)
        if s is not None and len(s) >= 1:
            shapes.append(s)
    if len(shapes) < 2:
        return None
    a_shape, b_shape = shapes[0], shapes[1]
    if len(a_shape) == 0 or len(b_shape) == 0:
        return None
    # For 1D x 1D (dot product)
    if len(a_shape) == 1 and len(b_shape) == 1:
        return 2 * a_shape[0]
    # For 2D x 2D: M x K @ K x N = 2*M*K*N
    m = a_shape[-2] if len(a_shape) >= 2 else 1
    k = a_shape[-1]
    n = b_shape[-1]
    batch = _prod(output_shape[:-2]) if len(output_shape) > 2 else 1
    return 2 * batch * m * k * n


def _addmm_flops(output_shape, parent_param_shapes, creation_args):
    """addmm: beta * M + alpha * (A @ B). FLOPs = 2*m*k*n + output_numel."""
    if not creation_args or len(creation_args) < 3:
        return None
    shapes = []
    for arg in creation_args[1:3]:
        s = _safe_shape(arg)
        if s is not None:
            shapes.append(s)
    if len(shapes) < 2:
        return None
    a_shape, b_shape = shapes[0], shapes[1]
    if len(a_shape) < 2 or len(b_shape) < 2:
        return None
    m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
    return 2 * m * k * n + _numel(output_shape)


def _linear_flops(output_shape, parent_param_shapes, creation_args):
    """nn.Linear: 2 * batch * in_features * out_features (+ bias)."""
    if parent_param_shapes:
        # Weight shape is [out_features, in_features]
        weight_shape = parent_param_shapes[0]
        if len(weight_shape) >= 2:
            out_features, in_features = weight_shape[0], weight_shape[1]
            batch = _numel(output_shape) // out_features if out_features > 0 else 0
            flops = 2 * batch * in_features * out_features
            if len(parent_param_shapes) > 1:  # bias
                flops += _numel(output_shape)
            return flops
    return None


def _conv_flops(output_shape, parent_param_shapes, creation_args):
    """Convolution: 2 * output_numel * in_channels_per_group * kernel_size."""
    if not parent_param_shapes:
        return None
    weight_shape = parent_param_shapes[0]
    if len(weight_shape) < 3:
        return None
    out_numel = _numel(output_shape)
    channels_per_group = weight_shape[1]
    kernel_size = _prod(weight_shape[2:])
    flops = 2 * out_numel * channels_per_group * kernel_size
    if len(parent_param_shapes) > 1:  # bias
        flops += _numel(output_shape)
    return flops


def _batchnorm_flops(output_shape, parent_param_shapes, creation_args):
    """BatchNorm: ~5 FLOPs per element (mean, var, normalize, scale, shift)."""
    return 5 * _numel(output_shape)


def _layernorm_flops(output_shape, parent_param_shapes, creation_args):
    """LayerNorm: ~5 FLOPs per element."""
    return 5 * _numel(output_shape)


def _groupnorm_flops(output_shape, parent_param_shapes, creation_args):
    """GroupNorm: ~5 FLOPs per element."""
    return 5 * _numel(output_shape)


def _instancenorm_flops(output_shape, parent_param_shapes, creation_args):
    """InstanceNorm: ~5 FLOPs per element."""
    return 5 * _numel(output_shape)


def _reduction_flops(output_shape, parent_param_shapes, creation_args):
    """Reductions (sum, mean, max, min, prod, etc.): ~input_numel FLOPs."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return _numel(input_shape)
    return None


def _var_std_flops(output_shape, parent_param_shapes, creation_args):
    """Variance/std: ~3 * input_numel (mean, squared diff, sum)."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return 3 * _numel(input_shape)
    return None


def _norm_flops(output_shape, parent_param_shapes, creation_args):
    """Vector/matrix norm: ~2 * input_numel (square + sum, then sqrt)."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return 2 * _numel(input_shape)
    return None


def _softmax_flops(output_shape, parent_param_shapes, creation_args):
    """Softmax: ~5 FLOPs per element (max, sub, exp, sum, div)."""
    return 5 * _numel(output_shape)


def _log_softmax_flops(output_shape, parent_param_shapes, creation_args):
    """LogSoftmax: ~6 FLOPs per element."""
    return 6 * _numel(output_shape)


def _pool_flops(output_shape, parent_param_shapes, creation_args):
    """Pooling: ~kernel_size comparisons/additions per output element.
    Approximate as output_numel since kernel_size is typically small."""
    return _numel(output_shape)


def _adaptive_pool_flops(output_shape, parent_param_shapes, creation_args):
    """Adaptive pooling: each output element averages over a region of input."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return _numel(input_shape)
    return _numel(output_shape)


def _embedding_flops(output_shape, parent_param_shapes, creation_args):
    """Embedding: lookup, no arithmetic."""
    return 0


def _dropout_flops(output_shape, parent_param_shapes, creation_args):
    """Dropout: 1 comparison + 1 multiply per element."""
    return 2 * _numel(output_shape)


def _interpolate_flops(output_shape, parent_param_shapes, creation_args):
    """Interpolation/upsample: ~4 FLOPs per output element (bilinear)."""
    return 4 * _numel(output_shape)


def _sort_flops(output_shape, parent_param_shapes, creation_args):
    """Sort: O(n log n) comparisons."""
    n = _numel(output_shape)
    if n > 0:
        return int(n * math.log2(max(n, 2)))
    return 0


def _cumulative_flops(output_shape, parent_param_shapes, creation_args):
    """Cumulative ops (cumsum, cumprod): input_numel."""
    return _numel(output_shape)


def _cross_entropy_flops(output_shape, parent_param_shapes, creation_args):
    """Cross-entropy: softmax + nll_loss ≈ 6 * input_numel."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return 6 * _numel(input_shape)
    return None


def _mse_loss_flops(output_shape, parent_param_shapes, creation_args):
    """MSE loss: sub + square + mean ≈ 3 * input_numel."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return 3 * _numel(input_shape)
    return None


def _binary_cross_entropy_flops(output_shape, parent_param_shapes, creation_args):
    """Binary cross entropy: ~10 FLOPs per element."""
    if creation_args:
        input_shape = _safe_shape(creation_args[0])
        if input_shape:
            return 10 * _numel(input_shape)
    return None


# Registry mapping func_applied_name to handler
SPECIALTY_HANDLERS = {
    # MatMul family
    "mm": _matmul_flops,
    "matmul": _matmul_flops,
    "__matmul__": _matmul_flops,
    "__rmatmul__": _matmul_flops,
    "bmm": _matmul_flops,
    "addmm": _addmm_flops,
    "addmm_": _addmm_flops,
    "addbmm": _matmul_flops,
    "addbmm_": _matmul_flops,
    "baddbmm": _matmul_flops,
    "baddbmm_": _matmul_flops,
    "multi_dot": _matmul_flops,
    "chain_matmul": _matmul_flops,
    "linear": _linear_flops,
    # Convolution
    "conv1d": _conv_flops,
    "conv2d": _conv_flops,
    "conv3d": _conv_flops,
    "conv_transpose1d": _conv_flops,
    "conv_transpose2d": _conv_flops,
    "conv_transpose3d": _conv_flops,
    "convolution": _conv_flops,
    "_convolution": _conv_flops,
    "_convolution_mode": _conv_flops,
    # Normalization
    "batch_norm": _batchnorm_flops,
    "native_batch_norm": _batchnorm_flops,
    "layer_norm": _layernorm_flops,
    "native_layer_norm": _layernorm_flops,
    "rms_norm": _layernorm_flops,
    "group_norm": _groupnorm_flops,
    "native_group_norm": _groupnorm_flops,
    "instance_norm": _instancenorm_flops,
    # Reductions
    "sum": _reduction_flops,
    "mean": _reduction_flops,
    "nanmean": _reduction_flops,
    "nansum": _reduction_flops,
    "prod": _reduction_flops,
    "max": _reduction_flops,
    "min": _reduction_flops,
    "amax": _reduction_flops,
    "amin": _reduction_flops,
    "aminmax": _reduction_flops,
    "argmax": _reduction_flops,
    "argmin": _reduction_flops,
    "any": _reduction_flops,
    "all": _reduction_flops,
    "count_nonzero": _reduction_flops,
    "logsumexp": _reduction_flops,
    "logcumsumexp": _reduction_flops,
    # Variance / std
    "var": _var_std_flops,
    "std": _var_std_flops,
    "var_mean": _var_std_flops,
    "std_mean": _var_std_flops,
    # Norms
    "norm": _norm_flops,
    "frobenius_norm": _norm_flops,
    "nuclear_norm": _norm_flops,
    "vector_norm": _norm_flops,
    "matrix_norm": _norm_flops,
    # Softmax family
    "softmax": _softmax_flops,
    "_softmax": _softmax_flops,
    "softmin": _softmax_flops,
    "log_softmax": _log_softmax_flops,
    "_log_softmax": _log_softmax_flops,
    "gumbel_softmax": _softmax_flops,
    # Pooling
    "max_pool1d": _pool_flops,
    "max_pool2d": _pool_flops,
    "max_pool3d": _pool_flops,
    "max_pool1d_with_indices": _pool_flops,
    "max_pool2d_with_indices": _pool_flops,
    "max_pool3d_with_indices": _pool_flops,
    "avg_pool1d": _pool_flops,
    "avg_pool2d": _pool_flops,
    "avg_pool3d": _pool_flops,
    "lp_pool1d": _pool_flops,
    "lp_pool2d": _pool_flops,
    "lp_pool3d": _pool_flops,
    "adaptive_avg_pool1d": _adaptive_pool_flops,
    "adaptive_avg_pool2d": _adaptive_pool_flops,
    "adaptive_avg_pool3d": _adaptive_pool_flops,
    "adaptive_max_pool1d": _adaptive_pool_flops,
    "adaptive_max_pool2d": _adaptive_pool_flops,
    "adaptive_max_pool3d": _adaptive_pool_flops,
    "adaptive_max_pool1d_with_indices": _adaptive_pool_flops,
    "adaptive_max_pool2d_with_indices": _adaptive_pool_flops,
    "adaptive_max_pool3d_with_indices": _adaptive_pool_flops,
    "fractional_max_pool2d": _pool_flops,
    "fractional_max_pool2d_with_indices": _pool_flops,
    "fractional_max_pool3d": _pool_flops,
    "fractional_max_pool3d_with_indices": _pool_flops,
    "max_unpool1d": _pool_flops,
    "max_unpool2d": _pool_flops,
    "max_unpool3d": _pool_flops,
    # Embedding
    "embedding": _embedding_flops,
    "embedding_bag": _embedding_flops,
    "one_hot": _embedding_flops,
    # Dropout
    "dropout": _dropout_flops,
    "dropout_": _dropout_flops,
    "dropout1d": _dropout_flops,
    "dropout2d": _dropout_flops,
    "dropout3d": _dropout_flops,
    "alpha_dropout": _dropout_flops,
    "alpha_dropout_": _dropout_flops,
    "feature_dropout": _dropout_flops,
    "feature_dropout_": _dropout_flops,
    "feature_alpha_dropout": _dropout_flops,
    "feature_alpha_dropout_": _dropout_flops,
    "native_dropout": _dropout_flops,
    # Interpolation
    "interpolate": _interpolate_flops,
    "grid_sample": _interpolate_flops,
    "grid_sampler": _interpolate_flops,
    "grid_sampler_2d": _interpolate_flops,
    "grid_sampler_3d": _interpolate_flops,
    "affine_grid": _interpolate_flops,
    "affine_grid_generator": _interpolate_flops,
    # Sort / topk
    "sort": _sort_flops,
    "argsort": _sort_flops,
    "msort": _sort_flops,
    "topk": _sort_flops,
    "kthvalue": _sort_flops,
    "median": _sort_flops,
    "nanmedian": _sort_flops,
    "mode": _sort_flops,
    "searchsorted": _sort_flops,
    # Cumulative
    "cumsum": _cumulative_flops,
    "cumsum_": _cumulative_flops,
    "cumprod": _cumulative_flops,
    "cumprod_": _cumulative_flops,
    "cummax": _cumulative_flops,
    "cummin": _cumulative_flops,
    "cumulative_trapezoid": _cumulative_flops,
    # Loss functions
    "cross_entropy": _cross_entropy_flops,
    "nll_loss": _reduction_flops,
    "mse_loss": _mse_loss_flops,
    "l1_loss": _mse_loss_flops,
    "smooth_l1_loss": _mse_loss_flops,
    "huber_loss": _mse_loss_flops,
    "binary_cross_entropy": _binary_cross_entropy_flops,
    "binary_cross_entropy_with_logits": _binary_cross_entropy_flops,
    "hinge_embedding_loss": _reduction_flops,
    "margin_ranking_loss": _reduction_flops,
    "cosine_embedding_loss": _reduction_flops,
    "triplet_margin_loss": _reduction_flops,
    "triplet_margin_with_distance_loss": _reduction_flops,
    "multi_margin_loss": _reduction_flops,
    "multilabel_margin_loss": _reduction_flops,
    "multilabel_soft_margin_loss": _reduction_flops,
    "soft_margin_loss": _reduction_flops,
    "gaussian_nll_loss": _reduction_flops,
    "poisson_nll_loss": _reduction_flops,
    "kl_div": _reduction_flops,
    "ctc_loss": _reduction_flops,
    # Einsum
    "einsum": _matmul_flops,
    # Cosine similarity / distance
    "cosine_similarity": _norm_flops,
    "pairwise_distance": _norm_flops,
    "pdist": _norm_flops,
    "cdist": _norm_flops,
    # Attention
    "scaled_dot_product_attention": _matmul_flops,
    "multi_head_attention_forward": _matmul_flops,
}


# ============================================================================
# Backward FLOPs multipliers
# ============================================================================

BACKWARD_MULTIPLIERS: Dict[str, float] = {
    # Convolution / linear: dL/dX + dL/dW ≈ 2x forward
    "conv1d": 2.0,
    "conv2d": 2.0,
    "conv3d": 2.0,
    "conv_transpose1d": 2.0,
    "conv_transpose2d": 2.0,
    "conv_transpose3d": 2.0,
    "convolution": 2.0,
    "_convolution": 2.0,
    "_convolution_mode": 2.0,
    "linear": 2.0,
    "mm": 2.0,
    "matmul": 2.0,
    "__matmul__": 2.0,
    "__rmatmul__": 2.0,
    "bmm": 2.0,
    "addmm": 2.0,
    "addmm_": 2.0,
    "addbmm": 2.0,
    "addbmm_": 2.0,
    "baddbmm": 2.0,
    "baddbmm_": 2.0,
    "multi_dot": 2.0,
    "chain_matmul": 2.0,
    "einsum": 2.0,
    # Normalization: gradients through mean/variance ≈ 2.5x
    "batch_norm": 2.5,
    "native_batch_norm": 2.5,
    "layer_norm": 2.5,
    "native_layer_norm": 2.5,
    "rms_norm": 2.5,
    "group_norm": 2.5,
    "native_group_norm": 2.5,
    "instance_norm": 2.5,
    # Simple activations: element-wise derivative ≈ 1x
    "relu": 1.0,
    "relu_": 1.0,
    "relu6": 1.0,
    "threshold": 1.0,
    "threshold_": 1.0,
    "hardtanh": 1.0,
    "hardtanh_": 1.0,
    "leaky_relu": 1.0,
    "leaky_relu_": 1.0,
    "prelu": 1.0,
    # Complex activations: derivative requires more work ≈ 1.5x
    "sigmoid": 1.5,
    "sigmoid_": 1.5,
    "tanh": 1.5,
    "tanh_": 1.5,
    "gelu": 1.5,
    "silu": 1.5,
    "mish": 1.5,
    "elu": 1.5,
    "elu_": 1.5,
    "celu": 1.5,
    "celu_": 1.5,
    "selu": 1.5,
    "selu_": 1.5,
    "softplus": 1.5,
    "logsigmoid": 1.5,
    # Softmax: gradient is O(n) per element ≈ 2x
    "softmax": 2.0,
    "_softmax": 2.0,
    "log_softmax": 2.0,
    "_log_softmax": 2.0,
    # Attention ≈ 2.5x
    "scaled_dot_product_attention": 2.5,
    "multi_head_attention_forward": 2.5,
    # Pooling ≈ 1.0x (just routing gradients)
    "max_pool1d": 1.0,
    "max_pool2d": 1.0,
    "max_pool3d": 1.0,
    "avg_pool1d": 1.0,
    "avg_pool2d": 1.0,
    "avg_pool3d": 1.0,
    # Reductions ≈ 1.0x (broadcast gradient back)
    "sum": 1.0,
    "mean": 1.0,
    "prod": 1.5,
    # Dropout: same as forward
    "dropout": 1.0,
    "dropout_": 1.0,
}

# Default backward multiplier for ops not in the dict
_DEFAULT_BACKWARD_MULTIPLIER = 1.0


# ============================================================================
# Public API
# ============================================================================


def compute_forward_flops(
    func_applied_name: str,
    output_shape: Optional[Tuple[int, ...]],
    parent_param_shapes: list,
    creation_args: tuple,
    creation_kwargs: dict,
) -> Optional[int]:
    """Compute forward FLOPs for a single operation.

    Args:
        func_applied_name: Name of the function that was applied.
        output_shape: Shape of the output tensor.
        parent_param_shapes: Shapes of parameter tensors involved.
        creation_args: Positional args to the function.
        creation_kwargs: Keyword args to the function.

    Returns:
        FLOPs count, 0 for zero-cost ops, or None if unknown.
    """
    if func_applied_name is None:
        return None

    # Check zero-cost ops first
    if func_applied_name in ZERO_FLOPS_OPS:
        return 0

    # Check element-wise ops
    if func_applied_name in ELEMENTWISE_FLOPS:
        if output_shape is None:
            return None
        return ELEMENTWISE_FLOPS[func_applied_name] * _numel(output_shape)

    # Check specialty handlers
    if func_applied_name in SPECIALTY_HANDLERS:
        handler = SPECIALTY_HANDLERS[func_applied_name]
        return handler(output_shape, parent_param_shapes, creation_args)

    # Unknown operation
    return None


def compute_backward_flops(
    func_applied_name: str,
    forward_flops: Optional[int],
) -> Optional[int]:
    """Estimate backward FLOPs based on forward FLOPs and operation type.

    Args:
        func_applied_name: Name of the function.
        forward_flops: Forward FLOPs count (from compute_forward_flops).

    Returns:
        Estimated backward FLOPs, or None if forward_flops is None.
    """
    if forward_flops is None or func_applied_name is None:
        return None
    multiplier = BACKWARD_MULTIPLIERS.get(func_applied_name, _DEFAULT_BACKWARD_MULTIPLIER)
    return int(forward_flops * multiplier)
