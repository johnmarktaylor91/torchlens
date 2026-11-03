"""Shape helpers and the shape-derived FLOP rules for weighted, recurrent and chained calls.

Split from ``_capture/flops.py`` (size ratchet): the rule tables stay there and
import these. Conventions are that module's: one multiply-add counts 2 FLOPs,
and a rule returns ``None`` rather than guess.
"""

import string
from collections.abc import Sequence
from typing import cast

Shape = tuple[int, ...]
ParamShapes = Sequence[Shape]
CapturedArgs = tuple[object, ...]


def _prod(shape: Sequence[int]) -> int:
    """Product of a shape tuple."""
    result = 1
    for v in shape:
        result *= v
    return result


def _numel(shape: Sequence[int] | None) -> int:
    """Number of elements from a shape tuple. Scalar shape () = 1 element."""
    if shape is None:
        return 0
    return _prod(shape)


def _safe_shape(t: object) -> Shape | None:
    """Safely extract shape tuple from a tensor or shape tuple."""
    if t is None:
        return None
    if isinstance(t, tuple):
        if not all(isinstance(dimension, int) for dimension in t):
            return None
        return cast(Shape, t)
    try:
        return cast(Shape, tuple(getattr(t, "shape")))
    except Exception:
        return None


def _operand_weight(
    param_shapes: ParamShapes, saved_args: CapturedArgs, min_rank: int
) -> tuple[Shape, bool] | None:
    """Return ``(weight_shape, has_bias)`` for a linear or convolution call.

    The weight is read from the parameter shapes when the first one is a
    weight (rank ``min_rank`` or more); otherwise from the call's second
    positional argument (``F.linear(x, w, b)``, ``F.conv2d(x, w, b)``), so a
    weight computed in ``forward`` (weight norm, spectral norm, a weight passed
    in as an input) is counted too. The bias is present when a second
    parameter shape follows a parameter weight, or when the third positional
    argument is a tensor.
    """

    if param_shapes and len(param_shapes[0]) >= min_rank:
        return tuple(param_shapes[0]), len(param_shapes) > 1
    if len(saved_args) < 2:
        return None
    weight = _safe_shape(saved_args[1]) if not isinstance(saved_args[1], tuple) else None
    if weight is None or len(weight) < min_rank:
        return None
    has_bias = len(saved_args) > 2 and _safe_shape(saved_args[2]) is not None
    if has_bias and isinstance(saved_args[2], tuple):
        has_bias = False
    return weight, has_bias


#: Calls whose weight and bias may arrive as keywords (``F.linear(x, w, bias=b)``).
_WEIGHTED_FUNCS: frozenset[str] = frozenset(
    {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    }
)


def _with_operand_kwargs(
    func_name: str, saved_args: CapturedArgs, saved_kwargs: dict[str, object] | None
) -> CapturedArgs:
    """Append keyword ``input``, ``weight`` and ``bias`` to a weighted call's positional args."""

    if func_name not in _WEIGHTED_FUNCS or not saved_kwargs:
        return saved_args
    args = list(saved_args)
    for position, name in enumerate(("input", "weight", "bias")):
        if len(args) == position and name in saved_kwargs:
            args.append(saved_kwargs[name])
    return tuple(args)


def _expand_einsum_ellipsis(equation: str, shapes: Sequence[Shape]) -> str | None:
    """Spell an einsum equation's ``...`` as explicit labels (right-aligned).

    Each operand's ellipsis covers its leading dimensions; they broadcast
    right-aligned, so the widest ellipsis takes fresh labels and narrower
    ones take that run's trailing labels. An implicit output keeps the
    ellipsis dimensions first, then the labels used once, sorted. Returns
    the equation unchanged without an ellipsis, or ``None`` when it cannot
    be spelled out.
    """

    if "..." not in equation:
        return equation
    inputs, separator, output = equation.partition("->")
    terms = inputs.split(",")
    if len(terms) != len(shapes) or any(term.count("...") > 1 for term in [*terms, output]):
        return None
    widths = [
        len(shape) - (len(term) - 3) if "..." in term else 0
        for term, shape in zip(terms, shapes, strict=True)
    ]
    width = max(widths)
    pool = [label for label in string.ascii_letters if label not in equation]
    if min(widths) < 0 or len(pool) < width:
        return None
    run = "".join(pool[:width])
    spelled = [
        term.replace("...", run[width - used :]) for term, used in zip(terms, widths, strict=True)
    ]
    if separator:
        spelled_output = output.replace("...", run)
    else:
        named = "".join(term.replace("...", "") for term in terms)
        spelled_output = run + "".join(
            sorted(label for label in set(named) if named.count(label) == 1)
        )
    return ",".join(spelled) + "->" + spelled_output


# Recurrent kernels (fused ``torch._VF`` RNN, GRU and LSTM calls and their cells).
#: Fused sequence kernels: ``(input, hx, weights, ...)`` or, for packed
#: sequences, ``(data, batch_sizes, hx, weights, ...)``.
_RNN_SEQUENCE_FUNCS: frozenset[str] = frozenset({"rnn_tanh", "rnn_relu", "gru", "lstm"})
#: Single-step cells: ``(input, hx, w_ih, w_hh, b_ih, b_hh)``.
_RNN_CELL_FUNCS: frozenset[str] = frozenset({"rnn_tanh_cell", "rnn_relu_cell", "gru_cell"})


def _rnn_flops_from_shapes(output_shape: Shape | None, param_shapes: ParamShapes) -> int | None:
    """FLOPs of one recurrent call from its sequence output and weight shapes.

    Every row of the output (one time step of one sequence, or one packed
    row) runs each layer and direction once, and each weight matrix
    (``w_ih``, ``w_hh`` and an LSTM projection ``w_hr``) multiplies one
    vector per row, so the multiply-accumulates are ``rows * sum(numel of
    every 2-D weight)``. Each bias vector is added once per row. The gate
    nonlinearities and gate products inside the kernel are not counted.
    """

    matrices = [shape for shape in param_shapes if len(shape) == 2]
    if not matrices or not output_shape or output_shape[-1] <= 0:
        return None
    rows = _numel(output_shape) // output_shape[-1]
    macs = rows * sum(_prod(shape) for shape in matrices)
    bias_adds = rows * sum(_prod(shape) for shape in param_shapes if len(shape) == 1)
    return 2 * macs + bias_adds


def _is_index_vector(arg: object) -> bool:
    """Whether ``arg`` is a 1-D integer tensor (a packed sequence's batch sizes)."""

    shape = _safe_shape(arg)
    return shape is not None and len(shape) == 1 and "int" in str(getattr(arg, "dtype", ""))


def _rnn_output_is_sequence(output_shape: Shape | None, saved_args: CapturedArgs) -> bool | None:
    """True for a fused RNN's sequence output, False for a final-state output.

    ``None`` when the shapes cannot tell the two apart (a one-step, one-layer
    call whose sequence output has the hidden state's shape).
    """

    if output_shape is None or len(saved_args) < 2:
        return None
    input_shape = _safe_shape(saved_args[0])
    packed = _is_index_vector(saved_args[1])
    hx = saved_args[2] if packed and len(saved_args) > 2 else saved_args[1]
    states = hx if isinstance(hx, (list, tuple)) else (hx,)
    state_shapes = {
        shape for shape in (_safe_shape(state) for state in states) if shape is not None
    }
    if input_shape is None:
        return None
    is_sequence = tuple(output_shape[:-1]) == tuple(input_shape[:-1])
    is_state = tuple(output_shape) in state_shapes
    if is_sequence == is_state:
        return None
    return is_sequence


def _rnn_sequence_flops(
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    saved_args: CapturedArgs,
) -> int | None:
    """Fused RNN, GRU and LSTM kernels: the whole call on its sequence output.

    The call's work is attributed to its sequence output; its final hidden
    and cell states count 0 (they come out of the same call).
    """

    is_sequence = _rnn_output_is_sequence(output_shape, saved_args)
    if is_sequence is None:
        return None
    if not is_sequence:
        return 0
    return _rnn_flops_from_shapes(output_shape, param_shapes)


def _rnn_cell_flops(
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    saved_args: CapturedArgs,
) -> int | None:
    """Single-output recurrent cells (``rnn_tanh_cell``, ``rnn_relu_cell``, ``gru_cell``)."""

    del saved_args
    return _rnn_flops_from_shapes(output_shape, param_shapes)


def _lstm_cell_flops(
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    saved_args: CapturedArgs,
) -> int | None:
    """``lstm_cell``: half the call's work on each of its two outputs.

    The hidden and cell states it returns have one shape and cannot be told
    apart here, so the call's work is split evenly over the two: the call's
    total is exact, the even split is a convention.
    """

    del saved_args
    flops = _rnn_flops_from_shapes(output_shape, param_shapes)
    return None if flops is None or flops % 2 else flops // 2


def _multi_dot_macs(shapes: Sequence[Shape]) -> int | None:
    """Multiply-accumulates of ``torch.linalg.multi_dot`` in its optimal order.

    ``multi_dot`` multiplies the chain in the order with the fewest
    multiply-accumulates (a 1-D first or last operand is a row or column
    vector), so that minimum is the work it runs.
    """

    if len(shapes) < 2:
        return None
    matrices: list[Shape] = []
    for index, shape in enumerate(shapes):
        if len(shape) == 2:
            matrices.append(shape)
        elif len(shape) == 1 and index == 0:
            matrices.append((1, shape[0]))
        elif len(shape) == 1 and index == len(shapes) - 1:
            matrices.append((shape[0], 1))
        else:
            return None
    if any(left[1] != right[0] for left, right in zip(matrices, matrices[1:], strict=False)):
        return None
    dims = [matrices[0][0], *(matrix[1] for matrix in matrices)]
    count = len(shapes)
    cost = [[0] * count for _ in range(count)]
    for span in range(1, count):
        for left in range(count - span):
            right = left + span
            cost[left][right] = min(
                cost[left][split]
                + cost[split + 1][right]
                + dims[left] * dims[split + 1] * dims[right + 1]
                for split in range(left, right)
            )
    return cost[0][count - 1]


def _multi_dot_flops(
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    saved_args: CapturedArgs,
) -> int | None:
    """``torch.linalg.multi_dot([A, B, ...])``: 2 x the optimal chain's MACs."""

    del output_shape, param_shapes
    if not saved_args or not isinstance(saved_args[0], (list, tuple)):
        return None
    shapes = [_safe_shape(operand) for operand in saved_args[0]]
    if any(shape is None for shape in shapes):
        return None
    macs = _multi_dot_macs(cast(list[Shape], shapes))
    return None if macs is None else 2 * macs
