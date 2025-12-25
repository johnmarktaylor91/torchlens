"""
FLOPs (Floating Point Operations) computation module for TorchLens.

This module provides accurate FLOPs counting for forward and backward passes
following standard conventions:
- Multiply-accumulate (MAC) is counted as 2 FLOPs (1 mul + 1 add)
- Transcendental functions (exp, log, sin, cos, etc.) are counted as multiple FLOPs
- Memory operations (reshape, view, etc.) are counted as 0 FLOPs

References:
- PyTorch FLOPs counting: https://pytorch.org/docs/stable/profiler.html
- DeepSpeed FLOPs profiler: https://www.deepspeed.ai/tutorials/flops-profiler/
- fvcore FLOPs counter: https://github.com/facebookresearch/fvcore
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import math


def _prod(x) -> int:
    """Compute product of iterable."""
    result = 1
    for v in x:
        result *= v
    return result


def _numel(shape) -> int:
    """Compute number of elements from shape."""
    return _prod(shape) if shape else 0


def _get_tensor_shape(t) -> Optional[Tuple[int, ...]]:
    """Safely get tensor shape."""
    if t is None:
        return None
    try:
        return tuple(t.shape)
    except:
        return None


# ============================================================================
# Forward FLOPs Computation
# ============================================================================

def compute_conv_flops(
    output_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    bias_shape: Optional[Tuple[int, ...]] = None,
    groups: int = 1,
    transposed: bool = False
) -> int:
    """
    Compute FLOPs for convolution operations.
    
    For Conv: FLOPs = 2 * output_elements * in_channels_per_group * kernel_size
    For ConvTranspose: FLOPs = 2 * output_elements * out_channels_per_group * kernel_size
    
    Args:
        output_shape: Shape of output tensor [N, C_out, *spatial]
        weight_shape: Shape of weight tensor [C_out, C_in/groups, *kernel] or 
                      [C_in, C_out/groups, *kernel] for transposed
        bias_shape: Shape of bias tensor (optional)
        groups: Number of groups for grouped convolution
        transposed: Whether this is a transposed convolution
    """
    out_numel = _numel(output_shape)
    kernel_size = _prod(weight_shape[2:]) if len(weight_shape) > 2 else 1
    
    if transposed:
        # For transposed conv, weight shape is [in_channels, out_channels/groups, *kernel]
        channels_per_group = weight_shape[1]
    else:
        # For regular conv, weight shape is [out_channels, in_channels/groups, *kernel]
        channels_per_group = weight_shape[1]
    
    # MAC = 2 FLOPs (1 mul + 1 add)
    flops = 2 * out_numel * channels_per_group * kernel_size
    
    # Add bias (1 add per output element)
    if bias_shape is not None:
        flops += out_numel
    
    return flops


def compute_linear_flops(
    output_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    bias_shape: Optional[Tuple[int, ...]] = None
) -> int:
    """
    Compute FLOPs for linear/fully-connected layer.
    
    FLOPs = 2 * batch * in_features * out_features
    
    Args:
        output_shape: Shape of output tensor [..., out_features]
        weight_shape: Shape of weight tensor [out_features, in_features]
        bias_shape: Shape of bias tensor (optional)
    """
    out_features, in_features = weight_shape[0], weight_shape[1]
    batch = _prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    
    # MAC = 2 FLOPs
    flops = 2 * batch * in_features * out_features
    
    # Add bias
    if bias_shape is not None:
        flops += batch * out_features
    
    return flops


def compute_matmul_flops(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
    output_shape: Tuple[int, ...]
) -> int:
    """
    Compute FLOPs for matrix multiplication.
    
    For matmul(A, B) where A is [..., m, k] and B is [..., k, n]:
    FLOPs = 2 * batch * m * k * n
    
    Args:
        shape_a: Shape of first input tensor
        shape_b: Shape of second input tensor  
        output_shape: Shape of output tensor [..., m, n]
    """
    if len(shape_a) < 2 or len(shape_b) < 2:
        # Vector-matrix or matrix-vector multiplication
        return 2 * _numel(output_shape)
    
    # Get dimensions
    m = shape_a[-2] if len(shape_a) >= 2 else 1
    k = shape_a[-1]  # Contraction dimension
    n = shape_b[-1] if len(shape_b) >= 2 else 1
    
    # Batch dimensions
    batch = _prod(output_shape[:-2]) if len(output_shape) > 2 else 1
    
    return 2 * batch * m * k * n


def compute_bmm_flops(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
) -> int:
    """
    Compute FLOPs for batch matrix multiplication.
    
    For bmm(A, B) where A is [B, m, k] and B is [B, k, n]:
    FLOPs = 2 * B * m * k * n
    """
    if len(shape_a) != 3 or len(shape_b) != 3:
        return 2 * _numel(shape_a)  # Fallback
    
    batch, m, k = shape_a
    _, _, n = shape_b
    
    return 2 * batch * m * k * n


def compute_addmm_flops(
    mat1_shape: Tuple[int, ...],
    mat2_shape: Tuple[int, ...],
    bias_shape: Optional[Tuple[int, ...]] = None
) -> int:
    """
    Compute FLOPs for addmm: beta * input + alpha * (mat1 @ mat2)
    
    FLOPs = 2 * m * k * n + m * n (for addition)
    """
    if len(mat1_shape) < 2 or len(mat2_shape) < 2:
        return 0
    
    m, k = mat1_shape[-2], mat1_shape[-1]
    n = mat2_shape[-1]
    
    flops = 2 * m * k * n  # Matrix multiplication
    flops += m * n  # Addition with bias
    flops += 2 * m * n  # Scaling by alpha and beta
    
    return flops


def compute_attention_flops(
    query_shape: Tuple[int, ...],
    key_shape: Tuple[int, ...],
    value_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...]
) -> int:
    """
    Compute FLOPs for scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Components:
    1. Q @ K^T: 2 * batch * heads * seq_q * seq_k * head_dim
    2. Scale: batch * heads * seq_q * seq_k
    3. Softmax: 5 * batch * heads * seq_q * seq_k
    4. Attn @ V: 2 * batch * heads * seq_q * head_dim * seq_k
    
    Args:
        query_shape: [batch, heads, seq_q, head_dim] or [batch, seq_q, embed_dim]
        key_shape: [batch, heads, seq_k, head_dim] or [batch, seq_k, embed_dim]
        value_shape: [batch, heads, seq_k, head_dim] or [batch, seq_k, embed_dim]
        output_shape: Output tensor shape
    """
    # Handle different input formats
    if len(query_shape) == 4:
        # [batch, heads, seq, head_dim]
        batch, heads, seq_q, head_dim = query_shape
        seq_k = key_shape[2]
    elif len(query_shape) == 3:
        # [batch, seq, embed_dim] - single head or fused
        batch, seq_q, embed_dim = query_shape
        seq_k = key_shape[1]
        heads = 1
        head_dim = embed_dim
    else:
        # Fallback
        return 4 * _numel(output_shape)
    
    # Q @ K^T
    flops = 2 * batch * heads * seq_q * seq_k * head_dim
    
    # Scale by 1/sqrt(d_k)
    flops += batch * heads * seq_q * seq_k
    
    # Softmax: exp + sum + div ≈ 5 ops per element
    flops += 5 * batch * heads * seq_q * seq_k
    
    # Attn @ V
    flops += 2 * batch * heads * seq_q * head_dim * seq_k
    
    return flops


def compute_multihead_attention_flops(
    query_shape: Tuple[int, ...],
    embed_dim: int,
    num_heads: int,
    kdim: Optional[int] = None,
    vdim: Optional[int] = None,
    has_bias: bool = True
) -> int:
    """
    Compute FLOPs for MultiheadAttention module.
    
    Components:
    1. Q, K, V projections: 3 * 2 * batch * seq * embed_dim * embed_dim
    2. Attention computation
    3. Output projection: 2 * batch * seq * embed_dim * embed_dim
    """
    if len(query_shape) < 2:
        return 0
    
    kdim = kdim or embed_dim
    vdim = vdim or embed_dim
    
    batch = _prod(query_shape[:-2]) if len(query_shape) > 2 else 1
    seq_len = query_shape[-2] if len(query_shape) >= 2 else 1
    
    head_dim = embed_dim // num_heads
    
    # Q projection
    flops = 2 * batch * seq_len * embed_dim * embed_dim
    # K projection
    flops += 2 * batch * seq_len * kdim * embed_dim
    # V projection
    flops += 2 * batch * seq_len * vdim * embed_dim
    
    # Bias for projections
    if has_bias:
        flops += 3 * batch * seq_len * embed_dim
    
    # Attention: Q @ K^T, softmax, attn @ V
    flops += 2 * batch * num_heads * seq_len * seq_len * head_dim  # Q @ K^T
    flops += 5 * batch * num_heads * seq_len * seq_len  # Softmax
    flops += 2 * batch * num_heads * seq_len * head_dim * seq_len  # Attn @ V
    
    # Output projection
    flops += 2 * batch * seq_len * embed_dim * embed_dim
    if has_bias:
        flops += batch * seq_len * embed_dim
    
    return flops


def compute_rnn_flops(
    input_shape: Tuple[int, ...],
    hidden_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
    rnn_type: str = "rnn"
) -> int:
    """
    Compute FLOPs for RNN/LSTM/GRU.
    
    RNN cell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    - FLOPs per timestep = 2 * (input_size * hidden_size + hidden_size * hidden_size) + hidden_size * 5 (tanh)
    
    LSTM has 4 gates, GRU has 3 gates.
    """
    if len(input_shape) < 2:
        return 0
    
    # input_shape: [seq_len, batch, input_size] or [batch, seq_len, input_size]
    if len(input_shape) == 3:
        seq_len = input_shape[0]
        batch = input_shape[1]
        input_size = input_shape[2]
    else:
        seq_len = 1
        batch = input_shape[0]
        input_size = input_shape[-1]
    
    num_directions = 2 if bidirectional else 1
    
    # Gate multiplier: RNN=1, LSTM=4, GRU=3
    gate_mult = {"rnn": 1, "lstm": 4, "gru": 3}.get(rnn_type.lower(), 1)
    
    flops_per_cell = 0
    
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        
        # Input-hidden transformation: W_ih @ x
        flops_per_cell += 2 * layer_input_size * hidden_size * gate_mult
        
        # Hidden-hidden transformation: W_hh @ h
        flops_per_cell += 2 * hidden_size * hidden_size * gate_mult
        
        # Bias additions
        flops_per_cell += 2 * hidden_size * gate_mult
        
        # Activation functions
        if rnn_type.lower() == "lstm":
            # 4 gates: 3 sigmoid + 1 tanh + element-wise ops
            flops_per_cell += hidden_size * (3 * 3 + 5 + 4)  # sigmoid=3, tanh=5, element-wise=4
        elif rnn_type.lower() == "gru":
            # 3 gates: 2 sigmoid + 1 tanh + element-wise ops
            flops_per_cell += hidden_size * (2 * 3 + 5 + 3)
        else:
            # Simple RNN: 1 tanh
            flops_per_cell += hidden_size * 5
    
    total_flops = flops_per_cell * seq_len * batch * num_directions
    
    return total_flops


def compute_norm_flops(
    input_shape: Tuple[int, ...],
    norm_type: str = "batchnorm",
    affine: bool = True
) -> int:
    """
    Compute FLOPs for normalization layers.
    
    BatchNorm/LayerNorm/etc:
    1. Mean computation: N ops
    2. Variance computation: 2N ops (subtract mean, square)
    3. Normalize: 2N ops (subtract mean, divide by std)
    4. Scale and shift (if affine): 2N ops
    
    Total: ~7N ops (or 5N without affine)
    """
    numel = _numel(input_shape)
    
    # Mean: sum + divide
    flops = numel + 1
    
    # Variance: (x - mean)^2, sum, divide
    flops += 2 * numel + 1
    
    # Normalize: (x - mean) / sqrt(var + eps)
    flops += 2 * numel + 1  # subtract, sqrt, divide
    
    # Scale and shift
    if affine:
        flops += 2 * numel
    
    return flops


def compute_embedding_flops(
    num_indices: int,
    embedding_dim: int
) -> int:
    """
    Compute FLOPs for embedding lookup.
    
    Embedding is essentially a table lookup, which is O(1) per index.
    However, we count the memory access as 1 op per element retrieved.
    """
    return num_indices * embedding_dim


def compute_einsum_flops(
    equation: str,
    operand_shapes: List[Tuple[int, ...]]
) -> int:
    """
    Compute FLOPs for einsum operation.
    
    This is a simplified estimation based on output size and contraction dimensions.
    """
    if not equation or not operand_shapes:
        return 0
    
    try:
        # Parse einsum equation
        if '->' in equation:
            inputs_str, output_str = equation.split('->')
        else:
            inputs_str = equation
            output_str = ''
        
        input_strs = inputs_str.split(',')
        
        if len(input_strs) != len(operand_shapes):
            return 0
        
        # Build dimension mapping
        dim_sizes = {}
        for inp_str, shape in zip(input_strs, operand_shapes):
            inp_str = inp_str.strip()
            for i, char in enumerate(inp_str):
                if i < len(shape):
                    if char in dim_sizes:
                        dim_sizes[char] = max(dim_sizes[char], shape[i])
                    else:
                        dim_sizes[char] = shape[i]
        
        # Find contraction dimensions (in inputs but not in output)
        output_dims = set(output_str.strip())
        input_dims = set(''.join(input_strs).replace(',', '').replace(' ', ''))
        contraction_dims = input_dims - output_dims
        
        # Compute total operations
        total_size = 1
        for dim, size in dim_sizes.items():
            total_size *= size
        
        # Each element requires 1 mul + 1 add for each contraction
        flops = 2 * total_size
        
        return flops
    except:
        return 0


# ============================================================================
# Activation Function FLOPs
# ============================================================================

# FLOPs per element for various activation functions
ACTIVATION_FLOPS = {
    # Simple comparisons/selections (1 op)
    "relu": 1,
    "relu6": 2,  # max(0, min(6, x))
    "leakyrelu": 2,  # comparison + mul
    "threshold": 2,
    "hardshrink": 2,
    "softshrink": 3,
    
    # Sigmoid family
    "sigmoid": 4,  # 1 / (1 + exp(-x)) = neg + exp + add + div
    "hardsigmoid": 4,  # clamp((x + 3) / 6, 0, 1)
    "logsigmoid": 5,  # log(sigmoid(x))
    
    # Tanh family
    "tanh": 6,  # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    "hardtanh": 2,  # clamp
    "tanhshrink": 7,  # x - tanh(x)
    
    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # erf approximation uses ~10 ops
    "gelu": 14,
    
    # SiLU/Swish: x * sigmoid(x)
    "silu": 5,
    "swish": 5,
    
    # HardSwish: x * relu6(x + 3) / 6
    "hardswish": 5,
    
    # Softplus: log(1 + exp(x))
    "softplus": 3,
    
    # Mish: x * tanh(softplus(x))
    "mish": 10,
    
    # ELU family
    "elu": 4,  # x if x > 0 else alpha * (exp(x) - 1)
    "selu": 5,  # scale * elu(x, alpha)
    "celu": 4,
    
    # PReLU: max(0, x) + alpha * min(0, x)
    "prelu": 4,
    
    # Softsign: x / (1 + |x|)
    "softsign": 3,
    
    # Softmax: exp(x_i) / sum(exp(x_j))
    # Per element: exp + (sum contribution) + div
    "softmax": 5,
    "logsoftmax": 6,
    
    # GLU: x[:half] * sigmoid(x[half:])
    "glu": 5,
}


# ============================================================================
# Pooling FLOPs
# ============================================================================

def compute_pool_flops(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: Union[int, Tuple[int, ...]],
    pool_type: str = "max"
) -> int:
    """
    Compute FLOPs for pooling operations.
    
    MaxPool: kernel_size - 1 comparisons per output element
    AvgPool: kernel_size - 1 additions + 1 division per output element
    """
    if isinstance(kernel_size, int):
        ks = kernel_size
    else:
        ks = _prod(kernel_size)
    
    out_numel = _numel(output_shape)
    
    if "max" in pool_type.lower():
        # Max pooling: ks - 1 comparisons per output
        return out_numel * (ks - 1)
    elif "avg" in pool_type.lower():
        # Avg pooling: ks - 1 additions + 1 division per output
        return out_numel * ks
    else:
        return out_numel * ks


def compute_adaptive_pool_flops(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    pool_type: str = "max"
) -> int:
    """
    Compute FLOPs for adaptive pooling.
    
    Estimates kernel size from input/output ratio.
    """
    in_numel = _numel(input_shape)
    out_numel = _numel(output_shape)
    
    if out_numel == 0:
        return 0
    
    # Estimate effective kernel size
    kernel_size = max(1, in_numel // out_numel)
    
    if "max" in pool_type.lower():
        return out_numel * (kernel_size - 1)
    else:
        return out_numel * kernel_size


# ============================================================================
# Element-wise Operation FLOPs
# ============================================================================

ELEMENTWISE_FLOPS = {
    # Basic arithmetic (1 op per element)
    "add": 1, "iadd": 1, "__add__": 1, "__radd__": 1,
    "sub": 1, "isub": 1, "__sub__": 1, "__rsub__": 1,
    "mul": 1, "imul": 1, "__mul__": 1, "__rmul__": 1,
    "div": 1, "idiv": 1, "__div__": 1, "__truediv__": 1, "truediv": 1,
    "floordiv": 1, "__floordiv__": 1,
    "mod": 1, "__mod__": 1,
    "fmod": 1,
    "remainder": 1,
    
    # Power operations
    "pow": 10, "__pow__": 10,  # Expensive
    "sqrt": 4,
    "rsqrt": 5,  # 1/sqrt
    "square": 1,
    
    # Exponential and logarithmic
    "exp": 8,
    "exp2": 8,
    "expm1": 9,
    "log": 8,
    "log2": 8,
    "log10": 9,
    "log1p": 9,
    
    # Trigonometric
    "sin": 8,
    "cos": 8,
    "tan": 10,
    "asin": 10,
    "acos": 10,
    "atan": 8,
    "atan2": 10,
    "sinh": 10,
    "cosh": 10,
    "tanh": 6,
    "asinh": 12,
    "acosh": 12,
    "atanh": 10,
    
    # Rounding and sign
    "abs": 1,
    "neg": 1, "__neg__": 1,
    "sign": 1,
    "floor": 1,
    "ceil": 1,
    "round": 1,
    "trunc": 1,
    "frac": 2,
    
    # Clamping
    "clamp": 2,
    "clamp_min": 1,
    "clamp_max": 1,
    "clip": 2,
    
    # Special functions
    "erf": 10,
    "erfc": 11,
    "erfinv": 15,
    "lgamma": 15,
    "digamma": 15,
    "polygamma": 20,
    "mvlgamma": 20,
    
    # Comparison (1 op)
    "eq": 1, "__eq__": 1,
    "ne": 1, "__ne__": 1,
    "lt": 1, "__lt__": 1,
    "le": 1, "__le__": 1,
    "gt": 1, "__gt__": 1,
    "ge": 1, "__ge__": 1,
    
    # Logical
    "logical_and": 1,
    "logical_or": 1,
    "logical_not": 1,
    "logical_xor": 1,
    
    # Bitwise (typically 1 op)
    "bitwise_and": 1,
    "bitwise_or": 1,
    "bitwise_xor": 1,
    "bitwise_not": 1,
    
    # Other element-wise
    "lerp": 3,  # a + weight * (b - a)
    "addcmul": 3,  # input + value * tensor1 * tensor2
    "addcdiv": 3,  # input + value * tensor1 / tensor2
    "reciprocal": 1,
    "negative": 1,
    "positive": 0,
    
    # Type checks (essentially free)
    "isnan": 1,
    "isinf": 1,
    "isfinite": 1,
    "isneginf": 1,
    "isposinf": 1,
    "isreal": 1,
}


# ============================================================================
# Reduction Operation FLOPs
# ============================================================================

def compute_reduction_flops(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    reduction_type: str = "sum"
) -> int:
    """
    Compute FLOPs for reduction operations.
    
    Sum/Mean: N-1 additions (+ 1 division for mean)
    Prod: N-1 multiplications
    Max/Min: N-1 comparisons
    Norm: N squares + N-1 additions + 1 sqrt
    """
    in_numel = _numel(input_shape)
    out_numel = _numel(output_shape)
    
    # Number of elements reduced per output
    reduction_size = max(1, in_numel // out_numel) if out_numel > 0 else in_numel
    
    reduction_type = reduction_type.lower()
    
    if reduction_type in {"sum", "nansum"}:
        return in_numel - out_numel  # N-1 additions per output
    
    elif reduction_type == "mean":
        return in_numel  # N-1 additions + 1 division per output
    
    elif reduction_type == "prod":
        return in_numel - out_numel  # N-1 multiplications
    
    elif reduction_type in {"max", "min", "amax", "amin"}:
        return in_numel - out_numel  # N-1 comparisons
    
    elif reduction_type in {"argmax", "argmin"}:
        return in_numel - out_numel  # N-1 comparisons
    
    elif reduction_type in {"norm", "frobenius_norm"}:
        # square each element + sum + sqrt
        return in_numel * 2 + out_numel
    
    elif reduction_type == "var":
        # mean + (x - mean)^2 + mean of squares
        return in_numel * 3
    
    elif reduction_type == "std":
        # var + sqrt
        return in_numel * 3 + out_numel
    
    elif reduction_type == "logsumexp":
        # exp each + sum + log
        return in_numel * 9 + out_numel * 8
    
    elif reduction_type in {"all", "any"}:
        return in_numel - out_numel  # N-1 logical ops
    
    elif reduction_type == "count_nonzero":
        return in_numel  # N comparisons
    
    else:
        return in_numel


# ============================================================================
# Zero-FLOPs Operations (Memory/Shape only)
# ============================================================================

ZERO_FLOPS_OPS = {
    # Reshape/View operations
    "reshape", "view", "view_as",
    "flatten", "unflatten", "ravel",
    "squeeze", "unsqueeze",
    "permute", "transpose", "t", "mT", "mH",
    "contiguous", "to",
    "expand", "expand_as",
    "repeat", "repeat_interleave",
    "tile",
    
    # Indexing/Slicing
    "getitem", "__getitem__",
    "setitem", "__setitem__",
    "select", "narrow", "slice",
    "index_select", "masked_select",
    "gather", "scatter", "scatter_add",
    "take", "take_along_dim",
    "index_copy", "index_add", "index_fill",
    
    # Concatenation/Splitting (memory ops)
    "cat", "concat", "concatenate",
    "stack", "vstack", "hstack", "dstack", "column_stack", "row_stack",
    "chunk", "split", "tensor_split", "hsplit", "vsplit", "dsplit",
    "unbind",
    
    # Memory operations
    "clone", "detach",
    "contiguous",
    "pin_memory",
    "share_memory_",
    
    # Tensor creation (no computation)
    "zeros", "ones", "full", "empty",
    "zeros_like", "ones_like", "full_like", "empty_like",
    "new_zeros", "new_ones", "new_full", "new_empty",
    "arange", "linspace", "logspace",
    "eye", "diag", "diagflat",
    "tril", "triu",
    
    # Padding (memory copy)
    "pad",
    "constant_pad_nd",
    "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
    "replication_pad1d", "replication_pad2d", "replication_pad3d",
    "circular_pad1d", "circular_pad2d", "circular_pad3d",
    
    # Pixel operations (rearrangement)
    "pixel_shuffle", "pixel_unshuffle",
    
    # Dropout (in inference mode)
    "dropout", "dropout2d", "dropout3d",
    "alpha_dropout", "feature_alpha_dropout",
    
    # Type conversion
    "type", "type_as",
    "float", "double", "half", "bfloat16",
    "int", "long", "short", "byte", "bool",
    "to",
    
    # Device operations
    "cpu", "cuda", "to",
    
    # Misc
    "as_strided",
    "movedim", "moveaxis",
    "swapaxes", "swapdims",
    "flip", "fliplr", "flipud",
    "rot90", "roll",
}



# ============================================================================
# Main FLOPs Computation Function
# ============================================================================

def compute_flops_for_layer(layer_type: str, t, fields_dict: Dict[str, Any]) -> Optional[int]:
    """
    Compute FLOPs for a single layer based on layer type and tensor information.
    
    Args:
        layer_type: Type of the layer (e.g., 'conv2d', 'linear')
        t: Output tensor
        fields_dict: Dictionary containing layer information including:
            - parent_param_shapes: List of parameter shapes
            - creation_kwargs: Keyword arguments used in layer creation
            - creation_args: Positional arguments used in layer creation
            - tensor_shape: Input tensor shape
            - parent_tensor_shapes: Shapes of parent tensors
            
    Returns:
        Number of FLOPs or None if cannot be computed
    """
    if t is None:
        return None
    
    layer_type = str(layer_type).lower()
    output_shape = _get_tensor_shape(t)
    
    if output_shape is None:
        return None
    
    param_shapes = fields_dict.get("parent_param_shapes", []) or []
    creation_kwargs = fields_dict.get("creation_kwargs", {}) or {}
    creation_args = fields_dict.get("creation_args", []) or []
    input_shape = fields_dict.get("tensor_shape", None)
    parent_shapes = fields_dict.get("parent_tensor_shapes", []) or []
    
    out_numel = _numel(output_shape)
    
    # ==================== Zero-FLOPs Operations ====================
    if layer_type in ZERO_FLOPS_OPS:
        return 0
    
    # ==================== Convolution Layers ====================
    if layer_type in {"conv1d", "conv2d", "conv3d"}:
        if param_shapes and param_shapes[0] is not None:
            weight_shape = param_shapes[0]
            bias_shape = param_shapes[1] if len(param_shapes) > 1 else None
            groups = creation_kwargs.get("groups", 1)
            return compute_conv_flops(output_shape, weight_shape, bias_shape, groups, transposed=False)
    
    if layer_type in {"convtranspose1d", "convtranspose2d", "convtranspose3d"}:
        if param_shapes and param_shapes[0] is not None:
            weight_shape = param_shapes[0]
            bias_shape = param_shapes[1] if len(param_shapes) > 1 else None
            groups = creation_kwargs.get("groups", 1)
            return compute_conv_flops(output_shape, weight_shape, bias_shape, groups, transposed=True)
    
    # ==================== Linear Layer ====================
    if layer_type == "linear":
        if param_shapes and param_shapes[0] is not None:
            weight_shape = param_shapes[0]
            bias_shape = param_shapes[1] if len(param_shapes) > 1 else None
            return compute_linear_flops(output_shape, weight_shape, bias_shape)
    
    # ==================== Matrix Operations ====================
    if layer_type == "matmul":
        if parent_shapes and len(parent_shapes) >= 2:
            shape_a = parent_shapes[0] if parent_shapes[0] else output_shape
            shape_b = parent_shapes[1] if parent_shapes[1] else output_shape
            return compute_matmul_flops(shape_a, shape_b, output_shape)
        elif input_shape:
            # Estimate from input and output
            return compute_matmul_flops(input_shape, output_shape, output_shape)
        return 2 * out_numel  # Fallback
    
    if layer_type == "bmm":
        if parent_shapes and len(parent_shapes) >= 2:
            shape_a = parent_shapes[0] if parent_shapes[0] else output_shape
            shape_b = parent_shapes[1] if parent_shapes[1] else output_shape
            return compute_bmm_flops(shape_a, shape_b)
        return 2 * out_numel
    
    if layer_type in {"mm", "mv"}:
        if parent_shapes and len(parent_shapes) >= 2:
            shape_a = parent_shapes[0] if parent_shapes[0] else output_shape
            shape_b = parent_shapes[1] if parent_shapes[1] else output_shape
            if len(shape_a) >= 2 and len(shape_b) >= 1:
                m, k = shape_a[-2], shape_a[-1]
                n = shape_b[-1] if len(shape_b) >= 2 else 1
                return 2 * m * k * n
        return 2 * out_numel
    
    if layer_type == "addmm":
        if parent_shapes and len(parent_shapes) >= 3:
            mat1_shape = parent_shapes[1] if len(parent_shapes) > 1 else output_shape
            mat2_shape = parent_shapes[2] if len(parent_shapes) > 2 else output_shape
            return compute_addmm_flops(mat1_shape, mat2_shape)
        return 2 * out_numel
    
    if layer_type == "baddbmm":
        # batch1 @ batch2 + input
        if parent_shapes and len(parent_shapes) >= 3:
            batch1 = parent_shapes[1] if len(parent_shapes) > 1 else output_shape
            batch2 = parent_shapes[2] if len(parent_shapes) > 2 else output_shape
            if len(batch1) == 3 and len(batch2) == 3:
                b, m, k = batch1
                _, _, n = batch2
                return 2 * b * m * k * n + out_numel
        return 2 * out_numel
    
    if layer_type == "einsum":
        equation = creation_kwargs.get("equation", "") or (creation_args[0] if creation_args else "")
        if equation and parent_shapes:
            return compute_einsum_flops(equation, parent_shapes)
        return 2 * out_numel
    
    if layer_type == "tensordot":
        if parent_shapes and len(parent_shapes) >= 2:
            # tensordot is generalized matrix multiplication
            return 2 * out_numel * max(1, _numel(parent_shapes[0]) // out_numel)
        return 2 * out_numel
    
    # ==================== Attention ====================
    if layer_type == "scaled_dot_product_attention" or layer_type == "scaleddotproductattention":
        if parent_shapes and len(parent_shapes) >= 3:
            q_shape = parent_shapes[0] or output_shape
            k_shape = parent_shapes[1] or output_shape
            v_shape = parent_shapes[2] or output_shape
            return compute_attention_flops(q_shape, k_shape, v_shape, output_shape)
        return 4 * out_numel
    
    if layer_type == "multiheadattention" or "multihead" in layer_type:
        embed_dim = creation_kwargs.get("embed_dim", output_shape[-1] if output_shape else 512)
        num_heads = creation_kwargs.get("num_heads", 8)
        if parent_shapes and parent_shapes[0]:
            return compute_multihead_attention_flops(
                parent_shapes[0], embed_dim, num_heads,
                creation_kwargs.get("kdim"), creation_kwargs.get("vdim"),
                creation_kwargs.get("bias", True)
            )
        return 6 * out_numel
    
    # ==================== Normalization Layers ====================
    if "batchnorm" in layer_type:
        affine = creation_kwargs.get("affine", True)
        return compute_norm_flops(output_shape, "batchnorm", affine)
    
    if layer_type in {"layernorm", "layer_norm"}:
        elementwise_affine = creation_kwargs.get("elementwise_affine", True)
        return compute_norm_flops(output_shape, "layernorm", elementwise_affine)
    
    if layer_type in {"groupnorm", "group_norm"}:
        affine = creation_kwargs.get("affine", True)
        return compute_norm_flops(output_shape, "groupnorm", affine)
    
    if layer_type in {"instancenorm", "instance_norm", "instancenorm1d", "instancenorm2d", "instancenorm3d"}:
        affine = creation_kwargs.get("affine", False)
        return compute_norm_flops(output_shape, "instancenorm", affine)
    
    if layer_type in {"rmsnorm", "rms_norm"}:
        # RMSNorm: sqrt(mean(x^2)) + normalize + scale
        return out_numel * 4
    
    # ==================== Activation Functions ====================
    if layer_type in ACTIVATION_FLOPS:
        return out_numel * ACTIVATION_FLOPS[layer_type]
    
    # ==================== Pooling Layers ====================
    if "maxpool" in layer_type or "avgpool" in layer_type:
        kernel_size = creation_kwargs.get("kernel_size", 2)
        if isinstance(kernel_size, (list, tuple)):
            ks = _prod(kernel_size)
        else:
            ks = kernel_size
        pool_type = "max" if "max" in layer_type else "avg"
        return compute_pool_flops(input_shape or output_shape, output_shape, ks, pool_type)
    
    if "adaptiveavgpool" in layer_type or "adaptivemaxpool" in layer_type:
        pool_type = "max" if "max" in layer_type else "avg"
        return compute_adaptive_pool_flops(input_shape or output_shape, output_shape, pool_type)
    
    # ==================== Element-wise Operations ====================
    if layer_type in ELEMENTWISE_FLOPS:
        return out_numel * ELEMENTWISE_FLOPS[layer_type]
    
    # ==================== Reduction Operations ====================
    if layer_type in {"sum", "mean", "prod", "max", "min", "amax", "amin", 
                      "argmax", "argmin", "norm", "var", "std", "logsumexp",
                      "all", "any", "count_nonzero", "nansum", "nanmean"}:
        return compute_reduction_flops(input_shape or output_shape, output_shape, layer_type)
    
    # ==================== RNN Layers ====================
    if layer_type in {"rnn", "rnnbase"}:
        hidden_size = creation_kwargs.get("hidden_size", output_shape[-1] if output_shape else 256)
        num_layers = creation_kwargs.get("num_layers", 1)
        bidirectional = creation_kwargs.get("bidirectional", False)
        if input_shape:
            return compute_rnn_flops(input_shape, hidden_size, num_layers, bidirectional, "rnn")
        return out_numel * 4
    
    if layer_type == "lstm":
        hidden_size = creation_kwargs.get("hidden_size", output_shape[-1] if output_shape else 256)
        num_layers = creation_kwargs.get("num_layers", 1)
        bidirectional = creation_kwargs.get("bidirectional", False)
        if input_shape:
            return compute_rnn_flops(input_shape, hidden_size, num_layers, bidirectional, "lstm")
        return out_numel * 8
    
    if layer_type == "gru":
        hidden_size = creation_kwargs.get("hidden_size", output_shape[-1] if output_shape else 256)
        num_layers = creation_kwargs.get("num_layers", 1)
        bidirectional = creation_kwargs.get("bidirectional", False)
        if input_shape:
            return compute_rnn_flops(input_shape, hidden_size, num_layers, bidirectional, "gru")
        return out_numel * 6
    
    if layer_type in {"rnncell", "lstmcell", "grucell"}:
        # Single cell computation
        mult = {"rnncell": 4, "lstmcell": 8, "grucell": 6}.get(layer_type, 4)
        return out_numel * mult
    
    # ==================== Embedding ====================
    if layer_type == "embedding":
        embedding_dim = output_shape[-1] if output_shape else 1
        num_indices = _prod(output_shape[:-1]) if len(output_shape) > 1 else 1
        return compute_embedding_flops(num_indices, embedding_dim)
    
    if layer_type == "embeddingbag":
        # EmbeddingBag includes reduction
        embedding_dim = output_shape[-1] if output_shape else 1
        num_indices = _prod(output_shape[:-1]) if len(output_shape) > 1 else 1
        return num_indices * embedding_dim * 2  # lookup + reduction
    
    # ==================== Upsampling/Interpolation ====================
    if layer_type in {"upsample", "interpolate"}:
        mode = creation_kwargs.get("mode", "nearest")
        if mode == "nearest":
            return 0
        elif mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            # Bilinear: 4 muls + 3 adds per output
            # Bicubic: 16 muls + 15 adds per output
            mult = {"linear": 3, "bilinear": 7, "bicubic": 31, "trilinear": 15}.get(mode, 7)
            return out_numel * mult
        return out_numel * 4
    
    if layer_type in {"upsamplingnearest2d", "nearest"}:
        return 0
    
    if layer_type in {"upsamplingbilinear2d", "bilinear"}:
        return out_numel * 7
    
    # ==================== Loss Functions ====================
    if layer_type in {"mseloss", "mse_loss"}:
        # (pred - target)^2, mean
        return out_numel * 2 + (out_numel if "mean" in str(creation_kwargs.get("reduction", "mean")) else 0)
    
    if layer_type in {"l1loss", "l1_loss", "maeloss"}:
        return out_numel * 2
    
    if layer_type in {"crossentropyloss", "cross_entropy", "nll_loss", "nllloss"}:
        # log_softmax + nll
        return out_numel * 7
    
    if layer_type in {"bceloss", "bce_loss", "binary_cross_entropy"}:
        # -y*log(p) - (1-y)*log(1-p)
        return out_numel * 12
    
    if layer_type in {"bcewithlogitsloss", "binary_cross_entropy_with_logits"}:
        # sigmoid + bce
        return out_numel * 16
    
    if layer_type in {"kldivloss", "kl_div"}:
        return out_numel * 4
    
    if layer_type in {"huberloss", "huber_loss", "smoothl1loss", "smooth_l1_loss"}:
        return out_numel * 4
    
    # ==================== Misc Operations ====================
    if layer_type == "where":
        return out_numel * 2  # condition check + selection
    
    if layer_type in {"masked_fill", "masked_fill_"}:
        return out_numel
    
    if layer_type == "topk":
        # Partial sort: O(n * log(k))
        k = creation_kwargs.get("k", 1)
        if input_shape:
            n = _numel(input_shape)
            return int(n * math.log2(max(k, 2)))
        return out_numel * 5
    
    if layer_type == "sort":
        # O(n * log(n))
        if input_shape:
            n = _numel(input_shape)
            return int(n * math.log2(max(n, 2)))
        return out_numel * int(math.log2(max(out_numel, 2)))
    
    if layer_type == "argsort":
        if input_shape:
            n = _numel(input_shape)
            return int(n * math.log2(max(n, 2)))
        return out_numel * int(math.log2(max(out_numel, 2)))
    
    if layer_type == "unique":
        if input_shape:
            n = _numel(input_shape)
            return int(n * math.log2(max(n, 2)))  # Sort-based unique
        return out_numel * 5
    
    if layer_type in {"cumsum", "cumprod"}:
        return out_numel
    
    if layer_type == "diff":
        return out_numel
    
    if layer_type in {"trace", "diag", "diagonal"}:
        return out_numel
    
    if layer_type in {"det", "logdet", "slogdet"}:
        # O(n^3) for determinant via LU decomposition
        if output_shape:
            n = output_shape[-1] if len(output_shape) >= 1 else 1
            return n ** 3
        return out_numel
    
    if layer_type in {"inverse", "inv"}:
        # O(n^3)
        if output_shape and len(output_shape) >= 2:
            n = output_shape[-1]
            batch = _prod(output_shape[:-2]) if len(output_shape) > 2 else 1
            return batch * n ** 3
        return out_numel
    
    if layer_type in {"svd", "linalg_svd"}:
        # O(min(m,n) * m * n)
        if output_shape and len(output_shape) >= 2:
            m, n = output_shape[-2], output_shape[-1]
            return min(m, n) * m * n
        return out_numel
    
    if layer_type in {"qr", "linalg_qr"}:
        if output_shape and len(output_shape) >= 2:
            m, n = output_shape[-2], output_shape[-1]
            return 2 * m * n ** 2 - 2 * n ** 3 // 3
        return out_numel
    
    if layer_type in {"cholesky", "linalg_cholesky"}:
        if output_shape and len(output_shape) >= 2:
            n = output_shape[-1]
            return n ** 3 // 3
        return out_numel
    
    if layer_type in {"eig", "linalg_eig", "eigvals", "linalg_eigvals"}:
        if output_shape and len(output_shape) >= 1:
            n = output_shape[-1]
            return 10 * n ** 3  # Iterative algorithm
        return out_numel
    
    if layer_type in {"solve", "linalg_solve", "lstsq", "linalg_lstsq"}:
        if output_shape and len(output_shape) >= 2:
            m, n = output_shape[-2], output_shape[-1]
            return 2 * m * n ** 2
        return out_numel
    
    # ==================== FFT Operations ====================
    if layer_type in {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn"}:
        # FFT: O(n * log(n))
        if input_shape:
            n = _numel(input_shape)
            return int(5 * n * math.log2(max(n, 2)))  # 5 ops per butterfly
        return int(5 * out_numel * math.log2(max(out_numel, 2)))
    
    # ==================== Convolution-like Operations ====================
    if layer_type in {"conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}:
        if param_shapes and param_shapes[0] is not None:
            return compute_conv_flops(output_shape, param_shapes[0], 
                                     param_shapes[1] if len(param_shapes) > 1 else None,
                                     creation_kwargs.get("groups", 1), transposed=True)
    
    if layer_type in {"unfold", "fold"}:
        return 0  # Memory rearrangement
    
    # ==================== Unknown Operation ====================
    # Return None for unknown operations
    return None



# ============================================================================
# Backward FLOPs Computation
# ============================================================================

# Backward FLOPs multipliers for different operation types
# These are based on the computational complexity of computing gradients
BACKWARD_MULTIPLIERS = {
    # Convolution: compute dL/dX (same as forward) + dL/dW (same as forward)
    "conv1d": 2.0,
    "conv2d": 2.0,
    "conv3d": 2.0,
    "convtranspose1d": 2.0,
    "convtranspose2d": 2.0,
    "convtranspose3d": 2.0,
    
    # Linear: dL/dX = dL/dY @ W^T, dL/dW = X^T @ dL/dY
    "linear": 2.0,
    
    # Matrix operations: similar to forward for each gradient
    "matmul": 2.0,
    "bmm": 2.0,
    "mm": 2.0,
    "mv": 2.0,
    "addmm": 2.0,
    "baddbmm": 2.0,
    
    # Normalization: backward involves computing gradients through mean/var
    "batchnorm1d": 2.5,
    "batchnorm2d": 2.5,
    "batchnorm3d": 2.5,
    "layernorm": 2.5,
    "groupnorm": 2.5,
    "instancenorm": 2.5,
    
    # Attention: backward computes gradients for Q, K, V
    "scaled_dot_product_attention": 2.5,
    "scaleddotproductattention": 2.5,
    "multiheadattention": 2.5,
    
    # RNN: backward through time
    "rnn": 2.0,
    "lstm": 2.0,
    "gru": 2.0,
    "rnncell": 2.0,
    "lstmcell": 2.0,
    "grucell": 2.0,
    
    # Softmax: backward is more complex due to Jacobian
    "softmax": 2.0,
    "logsoftmax": 2.0,
    
    # Simple activations: backward is similar complexity
    "relu": 1.0,
    "relu6": 1.0,
    "leakyrelu": 1.0,
    "sigmoid": 1.5,  # sigmoid' = sigmoid * (1 - sigmoid)
    "tanh": 1.5,  # tanh' = 1 - tanh^2
    "gelu": 1.5,
    "silu": 1.5,
    "swish": 1.5,
    
    # Element-wise: backward is typically same as forward
    "add": 1.0,
    "sub": 1.0,
    "mul": 1.5,  # Need to multiply by other operand
    "div": 1.5,
    "pow": 2.0,  # More complex gradient
    
    # Reductions: backward broadcasts gradients
    "sum": 1.0,
    "mean": 1.0,
    "max": 1.0,
    "min": 1.0,
    
    # Embedding: backward updates embedding table
    "embedding": 1.0,
    
    # Loss functions: backward is typically simpler
    "mseloss": 1.0,
    "crossentropyloss": 1.0,
    "nllloss": 1.0,
}


def compute_flops_for_layer_backward(
    layer_type: str, 
    t, 
    fields_dict: Dict[str, Any]
) -> Optional[int]:
    """
    Estimate backward FLOPs for a single layer.
    
    The backward pass typically involves:
    1. Computing gradient w.r.t. input (dL/dX)
    2. Computing gradient w.r.t. parameters (dL/dW, dL/db)
    
    For most operations:
    - Convolution/Linear: backward ≈ 2x forward (input grad + weight grad)
    - Activations: backward ≈ 1x forward (element-wise)
    - Normalization: backward ≈ 2-3x forward (complex gradient through mean/var)
    - Attention: backward ≈ 2.5x forward
    
    Args:
        layer_type: Type of the layer
        t: Output tensor
        fields_dict: Dictionary containing layer information
        
    Returns:
        Number of backward FLOPs or None if cannot be computed
    """
    layer_type_lower = str(layer_type).lower()
    
    # Get forward FLOPs
    forward_flops = fields_dict.get("flops", None)
    
    if forward_flops is None:
        forward_flops = compute_flops_for_layer(layer_type, t, fields_dict)
    
    if forward_flops is None:
        return None
    
    # Zero-cost operations in forward are also zero-cost in backward
    if forward_flops == 0:
        return 0
    
    # Check for specific multiplier
    for op_name, multiplier in BACKWARD_MULTIPLIERS.items():
        if op_name in layer_type_lower:
            return int(forward_flops * multiplier)
    
    # ==================== Specific Backward Computations ====================
    
    # Convolution backward
    if any(conv in layer_type_lower for conv in ["conv1d", "conv2d", "conv3d"]):
        # Backward computes dL/dX and dL/dW, each similar to forward
        return 2 * forward_flops
    
    # Linear backward
    if layer_type_lower == "linear":
        return 2 * forward_flops
    
    # Normalization backward (more complex due to mean/var gradients)
    if any(norm in layer_type_lower for norm in ["batchnorm", "layernorm", "groupnorm", "instancenorm"]):
        # Backward involves computing gradients through mean and variance
        # which requires additional reductions
        return int(2.5 * forward_flops)
    
    # Attention backward
    if "attention" in layer_type_lower:
        # Backward computes gradients for Q, K, V projections and attention weights
        return int(2.5 * forward_flops)
    
    # RNN backward (backprop through time)
    if layer_type_lower in {"rnn", "lstm", "gru", "rnnbase"}:
        return 2 * forward_flops
    
    # Softmax backward
    if layer_type_lower in {"softmax", "logsoftmax"}:
        # Softmax backward: dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
        # This involves a matrix-vector product
        return 2 * forward_flops
    
    # Simple activations (element-wise backward)
    if layer_type_lower in {"relu", "relu6", "leakyrelu", "threshold", "hardshrink", "softshrink"}:
        return forward_flops
    
    # Activations with more complex gradients
    if layer_type_lower in {"sigmoid", "tanh", "gelu", "silu", "swish", "hardswish", "mish"}:
        # These require computing the derivative which involves the forward output
        return int(1.5 * forward_flops)
    
    if layer_type_lower in {"elu", "selu", "celu", "softplus"}:
        return int(1.5 * forward_flops)
    
    # Element-wise operations
    if layer_type_lower in {"add", "iadd", "sub", "isub"}:
        return forward_flops
    
    if layer_type_lower in {"mul", "imul", "div", "idiv", "truediv"}:
        # Multiplication backward: dL/da = dL/dc * b, dL/db = dL/dc * a
        return int(1.5 * forward_flops)
    
    if layer_type_lower in {"pow", "sqrt", "rsqrt"}:
        return 2 * forward_flops
    
    if layer_type_lower in {"exp", "log", "log2", "log10"}:
        return int(1.5 * forward_flops)
    
    # Trigonometric functions
    if layer_type_lower in {"sin", "cos", "tan", "asin", "acos", "atan"}:
        return int(1.5 * forward_flops)
    
    # Reduction operations (backward broadcasts gradients)
    if layer_type_lower in {"sum", "mean", "prod", "max", "min", "norm"}:
        # Backward broadcasts the gradient to input shape
        input_shape = fields_dict.get("tensor_shape", None)
        output_shape = _get_tensor_shape(t)
        if input_shape and output_shape:
            # Broadcasting cost
            return _numel(input_shape)
        return forward_flops
    
    # Pooling backward
    if "pool" in layer_type_lower:
        # MaxPool: backward routes gradient to max element
        # AvgPool: backward distributes gradient evenly
        return forward_flops
    
    # Embedding backward
    if layer_type_lower == "embedding":
        # Backward updates embedding table entries
        return forward_flops
    
    # Matrix operations
    if layer_type_lower in {"matmul", "bmm", "mm", "mv", "addmm", "baddbmm"}:
        return 2 * forward_flops
    
    # Loss functions (backward is typically simpler)
    if "loss" in layer_type_lower:
        return forward_flops
    
    # FFT backward
    if "fft" in layer_type_lower:
        return forward_flops  # IFFT has same complexity as FFT
    
    # Linear algebra operations
    if layer_type_lower in {"inverse", "inv", "det", "svd", "qr", "cholesky", "eig", "solve"}:
        return 2 * forward_flops
    
    # Default: assume backward is similar to forward
    return forward_flops


# ============================================================================
# Utility Functions
# ============================================================================

def format_flops(flops: Optional[int], precision: int = 2) -> str:
    """
    Format FLOPs number to human readable string.
    
    Args:
        flops: Number of FLOPs
        precision: Number of decimal places
        
    Returns:
        Formatted string (e.g., "1.23G", "456.78M")
    """
    if flops is None:
        return "N/A"
    
    if flops >= 1e15:
        return f"{flops/1e15:.{precision}f}P"
    elif flops >= 1e12:
        return f"{flops/1e12:.{precision}f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.{precision}f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.{precision}f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.{precision}f}K"
    else:
        return str(int(flops))


def get_total_flops(layer_list, include_backward: bool = True) -> Dict[str, int]:
    """
    Calculate total FLOPs from a list of layers.
    
    Args:
        layer_list: List of TensorLogEntry objects
        include_backward: Whether to include backward FLOPs
        
    Returns:
        Dictionary with 'forward', 'backward', and 'total' FLOPs
    """
    total_forward = 0
    total_backward = 0
    
    for layer in layer_list:
        fwd = getattr(layer, 'flops', None)
        if fwd is not None:
            total_forward += fwd
        
        if include_backward:
            bwd = getattr(layer, 'backward_flops', None)
            if bwd is not None:
                total_backward += bwd
    
    return {
        'forward': total_forward,
        'backward': total_backward,
        'total': total_forward + total_backward
    }


def get_flops_by_type(layer_list) -> Dict[str, Dict[str, int]]:
    """
    Get FLOPs breakdown by layer type.
    
    Args:
        layer_list: List of TensorLogEntry objects
        
    Returns:
        Dictionary mapping layer types to their FLOPs counts
    """
    flops_by_type = {}
    
    for layer in layer_list:
        layer_type = getattr(layer, 'layer_type', 'unknown')
        fwd = getattr(layer, 'flops', None) or 0
        bwd = getattr(layer, 'backward_flops', None) or 0
        
        if layer_type not in flops_by_type:
            flops_by_type[layer_type] = {'forward': 0, 'backward': 0, 'count': 0}
        
        flops_by_type[layer_type]['forward'] += fwd
        flops_by_type[layer_type]['backward'] += bwd
        flops_by_type[layer_type]['count'] += 1
    
    return flops_by_type
