"""Per-output accumulation depth of a replayed op (band-C eligibility input).

Band-C (deep-numeric) replay tolerance is earned by the FAITHFUL reduction
depth of an op, the number of multiply-adds summed into each output element,
never by graph position or a func allowlist. The helpers here read that depth
from the op's saved operands for every reduction category (convolution,
matmul-family, scatter-reduce fan-in, dimension reductions, scaled dot-product
attention); depth 0 means it could not be determined and the op stays
ineligible (fail-toward-strict). ``core`` consumes ``_op_reduction_depth`` and
re-exports it under its original name (split out of ``validation/core.py``
under the file-size ratchet).
"""

from collections.abc import Sequence
from typing import Any

import torch

from .._data_classes.op import Op

# Reduction-category func-name groups, used only to compute the per-output
# accumulation depth (NOT to gate eligibility -- a shallow op in any category
# below depth 64 stays strict). "Reduction" here means any op that sums many
# inputs into a single output element; everything else (elementwise, view,
# structural, copy) has depth 1 and never reaches the threshold.
_CONV_FUNC_PREFIX = "conv"  # conv1d/2d/3d + conv_transpose1d/2d/3d
_CONV_TRANSPOSE_FUNC_PREFIX = "conv_transpose"  # conv_transpose1d/2d/3d
_MATMUL_LINEAR_FUNCS = frozenset({"addmm", "baddbmm", "bmm", "linear", "matmul", "mm"})
# Fused attention: captured as one atomic op (see _op_reduction_depth's sdpa
# branch for why its depth is the key/value sequence length, not 1).
_SDPA_FUNC_NAME = "scaled_dot_product_attention"
# Scatter/segment/index aggregations that are band-C-eligible ONLY when they are
# ADDITIVE/mean accumulators -- i.e. they sum many FP32 source elements into one
# destination slot, so summation-order round-off genuinely accrues. Plain
# overwrite ``scatter``/``scatter_`` (no FP accumulation -- the last write wins)
# and ``max``/``min``/``amax``/``amin`` reductions (no summation-order drift)
# are EXCLUDED here and stay on the strict global tolerance. ``scatter_reduce*``
# is conditionally additive: only its ``reduce="sum"``/``"mean"`` modes accumulate
# (its ``"prod"``/``"amax"``/``"amin"`` modes do not), so it is gated dynamically
# inside the depth predicate rather than by membership here.
_SCATTER_REDUCE_FUNCS = frozenset(
    {
        "scatter_add",
        "scatter_add_",
        "scatter_reduce",
        "scatter_reduce_",
        "segment_reduce",
        "index_add",
        "index_add_",
    }
)
# scatter_reduce reduce-modes that ADD (FP reorder drift) vs that select/overwrite.
_ADDITIVE_SCATTER_REDUCE_MODES = frozenset({"sum", "mean"})
_DIM_REDUCE_FUNCS = frozenset(
    {
        "sum",
        "mean",
        "prod",
        "norm",
        "var",
        "std",
        "nansum",
        "nanmean",
        "logsumexp",
    }
)


def _reduced_numel_over_dims(tensor: torch.Tensor, dim: Any, default_all: bool) -> int:
    """Return how many input elements are summed into each output element.

    For a dimension-reducing op (``sum``/``mean``/...), the per-output
    accumulation depth is the product of the reduced dimension sizes. ``dim``
    follows torch's reduce conventions: ``None`` reduces over every dimension
    (when ``default_all``), an int or sequence of ints names the reduced dims.

    Runtime-verified torch behavior for an EXPLICIT empty ``dim=()`` (torch 2.x):
    for the reduce family this predicate gates (``sum``/``mean``/``nansum``/
    ``nanmean``/``var``/``std``/``norm``/``amax``/``amin``), ``dim=()`` is NOT an
    identity/no-op -- it reduces over EVERY dimension and returns a scalar (e.g.
    ``torch.sum(x, dim=()).shape == ()``). So ``dim=()`` is treated as reduce-all
    (depth = ``numel``), the same as ``dim=None`` with ``default_all`` -- not depth
    1. (``prod``/``logsumexp`` reject a tuple ``dim`` outright at runtime, so a
    captured op there will not present ``dim=()`` to this path.)
    """

    if dim is None:
        return int(tensor.numel()) if default_all else 1
    if isinstance(dim, int):
        dims: tuple[int, ...] = (dim,)
    else:
        try:
            dims = tuple(int(d) for d in dim)
        except TypeError:
            return 0
    if not dims:
        # Empty dim. For default_all reducers an explicit dim=() reduces over all
        # dims (runtime-verified scalar output), matching dim=None's reduce-all.
        return int(tensor.numel()) if default_all else 1
    ndim = tensor.dim()
    depth = 1
    for d in dims:
        depth *= int(tensor.shape[d % ndim]) if ndim else 1
    return depth


def _conv_transpose_reduction_depth(weight: Any, groups: Any) -> int:
    """Return ``in_channels/groups * prod(kernel)`` for a transposed-conv weight, else 0."""

    if isinstance(weight, torch.Tensor) and weight.dim() >= 2 and weight.shape[0] > 0:
        kernel_numel = 1
        for kdim in weight.shape[2:]:
            kernel_numel *= int(kdim)
        if groups is None:
            groups = 1
        if not isinstance(groups, int) or groups <= 0 or int(weight.shape[0]) % groups != 0:
            return 0
        return (int(weight.shape[0]) // groups) * kernel_numel
    return 0


def _op_reduction_depth(layer: Op) -> int:
    """Return the per-output FP32 accumulation depth for a replay op.

    The reduction depth is the number of multiply-adds summed into each output
    element, which is what drives accumulation-order round-off between an original
    op and its faithful replay. It is read from the op's saved operand shapes /
    indices, consulting BOTH ``saved_args`` and ``saved_kwargs`` so that operands
    passed by keyword (``conv2d(x, weight=w)``, ``matmul(a, other=b)``,
    ``scatter_add(out, dim=d, index=idx, src=s)``) are still measured correctly --
    reading positionally only would drop the operand to depth 0 and wrongly
    withhold band C, a fresh false-negative.

    Depth by category:

    - forward ``conv*``: ``in_channels/groups * prod(kernel_size)`` =
      ``weight.numel() / weight.shape[0]`` products per output element (forward
      conv weight is ``[out_channels, in_channels/groups, *kernel]``, so the
      out-channel axis is ``shape[0]``).
    - ``conv_transpose*``: transposed-conv weight is
      ``[in_channels, out_channels/groups, *kernel]`` -- the IN-channel axis is
      ``shape[0]``. The true per-output accumulation depth is
      ``in_channels/groups * prod(kernel)`` = ``weight.shape[0] // groups *
      prod(weight.shape[2:])``, with ``groups`` read from positional arg 6 / kwarg
      ``groups`` (default 1; an unreadable or non-dividing ``groups`` returns 0,
      fail-toward-strict). The forward formula ``weight.numel() // weight.shape[0]``
      would read ``out_channels/groups * prod(kernel)``, and omitting the
      ``// groups`` divisor over-reports a grouped transpose (e.g.
      ``ConvTranspose2d(128, 128, 1, groups=128)`` is depth 1, not 128).
    - ``linear`` / ``addmm`` / ``mm`` / ``matmul`` / ``bmm`` / ``baddbmm``:
      the contracted dimension ``K`` of the first matrix operand.
    - ``scatter_add*`` / additive ``scatter_reduce*`` (``reduce="sum"``/``"mean"``)
      / ``segment_reduce`` / ``index_add``: the max number of source elements
      summed into a single destination COORDINATE (the graph fan-in = max
      duplicate destination-tuple count, counted per actual destination position,
      NOT per raw index value). Plain overwrite ``scatter``/``scatter_`` and
      ``max``/``min``/``amax``/``amin``/``prod`` reduce-modes do not accumulate and
      are not in the eligible set, so they stay strict. A depth-2 scatter (two
      sources per destination) stays well below threshold.
    - ``sum`` / ``mean`` / ``prod`` / ``norm`` / ``var`` / ``std`` and other
      dimension reductions: the numel of the reduced dimension(s).
    - ``scaled_dot_product_attention``: TorchLens captures fused attention as
      ONE atomic op, never decomposed into its constituent
      matmul(Q,K) -> softmax -> matmul(.,V), so it is a reduction, not
      elementwise. Depth is the key/value sequence length (``key.shape[-2]``
      for ``[..., seq_len_kv, head_dim]``-layout operands): the fused op's
      final stage, ``attn_weights @ value``, sums over that many terms for
      every output element -- the same contracted-dimension role the matmul
      family above uses for its depth, and the stage whose round-off reaches
      the returned tensor most directly (the first stage's head_dim-deep
      QK^T dot product is renormalized by softmax before it gets here, so its
      own, usually-smaller, head_dim depth does not drive the final error the
      same way).
    - elementwise / copy / view / structural ops: 1.

    Returns
    -------
    int
        Per-output accumulation depth, or ``0`` when it cannot be determined.
        A return of ``0`` conservatively withholds band C (fail-toward-strict).
    """

    saved_args: Sequence[Any] = getattr(layer, "saved_args", None) or ()
    saved_kwargs: dict[str, Any] = getattr(layer, "saved_kwargs", None) or {}
    func_name = layer.func_name

    def _operand(index: int, *kwarg_names: str) -> Any:
        """Read an operand positionally, falling back to its keyword names."""
        if index < len(saved_args):
            return saved_args[index]
        for name in kwarg_names:
            if name in saved_kwargs:
                return saved_kwargs[name]
        return None

    # conv_transpose* (checked FIRST -- "conv_transpose" also startswith "conv"):
    # transposed weight is [in_channels, out_channels/groups, *kernel], so the true
    # per-output accumulation depth is in_channels/groups * prod(kernel) =
    # shape[0] // groups * prod(shape[2:]). groups is read from positional arg 6 /
    # kwarg "groups" (default 1). Omitting the // groups divisor over-reports a
    # grouped transpose (ConvTranspose2d(128,128,1,groups=128) is depth 1, not 128);
    # an unreadable / non-dividing groups returns 0 (fail-toward-strict).
    if func_name.startswith(_CONV_TRANSPOSE_FUNC_PREFIX):
        return _conv_transpose_reduction_depth(_operand(1, "weight"), _operand(6, "groups"))

    # forward conv*: weight is [out_channels, in_channels/groups, *kernel], so depth
    # = in_channels/groups * prod(kernel) = weight.numel() // weight.shape[0].
    if func_name.startswith(_CONV_FUNC_PREFIX):
        weight = _operand(1, "weight")
        if isinstance(weight, torch.Tensor) and weight.dim() >= 2 and weight.shape[0] > 0:
            return int(weight.numel() // weight.shape[0])
        return 0

    # matmul / linear family: depth = contracted dimension K.
    if func_name in _MATMUL_LINEAR_FUNCS:
        if func_name == "linear":
            weight = _operand(1, "weight")
            if isinstance(weight, torch.Tensor) and weight.dim() >= 1:
                return int(weight.shape[-1])
            first = _operand(0, "input")
            if isinstance(first, torch.Tensor) and first.dim() >= 1:
                return int(first.shape[-1])
            return 0
        if func_name == "addmm":
            mat1 = _operand(1, "mat1")
            if isinstance(mat1, torch.Tensor) and mat1.dim() >= 1:
                return int(mat1.shape[-1])
            return 0
        if func_name == "baddbmm":
            first_matrix = _operand(1, "batch1")
        elif func_name in {"mm", "bmm"}:
            first_matrix = _operand(0, "input", "mat1")
        else:  # matmul
            first_matrix = _operand(0, "input")
        if isinstance(first_matrix, torch.Tensor) and first_matrix.dim() >= 1:
            return int(first_matrix.shape[-1])
        return 0

    # Additive scatter / segment / index_add: depth = max per-coordinate fan-in.
    # scatter_reduce* is conditionally additive -- only its sum/mean modes accumulate;
    # a prod/amax/amin reduce-mode does not drift and is withheld (depth 0).
    if func_name in _SCATTER_REDUCE_FUNCS:
        if func_name.startswith("scatter_reduce"):
            # scatter_reduce(input, dim, index, src, reduce, *, include_self): the
            # reduce mode arrives as kwarg "reduce" (TorchLens-captured form) or as
            # positional arg 4 (free-function torch.scatter_reduce(..., reduce)).
            reduce_mode = saved_kwargs.get("reduce")
            if reduce_mode is None and len(saved_args) > 4:
                reduce_mode = saved_args[4]
            if reduce_mode not in _ADDITIVE_SCATTER_REDUCE_MODES:
                return 0
        return _scatter_fan_in_depth(func_name, saved_args, saved_kwargs)

    # dimension reductions (sum/mean/...): depth = numel of reduced dims.
    if func_name in _DIM_REDUCE_FUNCS:
        tensor = _operand(0, "input")
        if not isinstance(tensor, torch.Tensor):
            return 0
        dim = saved_kwargs.get("dim")
        if dim is None:
            dim = _operand(1, "dim")
        # sum/mean/prod with no dim reduce over all elements; var/std/norm too.
        return _reduced_numel_over_dims(tensor, dim, default_all=True)

    # scaled_dot_product_attention: fused and atomic (never decomposed into
    # matmul/softmax/matmul), so it is NOT elementwise depth-1 -- its real
    # per-output accumulation depth is the key/value sequence length, read
    # from the key operand's second-to-last dim ([..., seq_len_kv, head_dim]).
    # An unreadable key returns 0 (fail-toward-strict), same as every other
    # category above.
    if func_name == _SDPA_FUNC_NAME:
        key = _operand(1, "key")
        if isinstance(key, torch.Tensor) and key.dim() >= 2:
            return int(key.shape[-2])
        return 0

    # Everything else (elementwise, view, structural, copy) has depth 1: a single
    # value flows through per output element, so no FP32 reorder drift accrues.
    return 1


def _scatter_fan_in_depth(
    func_name: str,
    saved_args: Any,
    saved_kwargs: dict[str, Any],
) -> int:
    """Return the max number of source elements aggregated per destination slot.

    For ``scatter_add``-family ops the per-output accumulation depth is the graph
    fan-in: the maximum number of source elements that sum into a single
    destination COORDINATE. This MUST be counted per actual destination position,
    not per raw index value: a row-/feature-wise scatter writes many independent
    destinations at index ``0``, and counting the duplicate raw index value ``0``
    globally would conflate those independent destinations into one huge fan-in and
    wrongly grant band C to a genuinely shallow (e.g. depth-2) scatter.

    For an n-D ``scatter_add``/``scatter_reduce`` (``self``, ``index`` and ``src``
    all share rank), the destination coordinate of source element at position
    ``(i_0, ..., i_{n-1})`` is ``index[...]`` along the scatter ``dim`` and the
    SOURCE position ``i_d`` along every other dim. We rebuild the full destination
    tuple per source element and count the max number of identical tuples. A
    1-D ``index_add`` (index value IS the whole destination coordinate) and
    ``segment_reduce`` (max segment length) are already per-coordinate.
    """

    def _arg(index: int, *kwarg_names: str) -> Any:
        """Return a positional or keyword argument captured for replay."""

        if index < len(saved_args):
            return saved_args[index]
        for name in kwarg_names:
            if name in saved_kwargs:
                return saved_kwargs[name]
        return None

    if func_name.startswith("index_add"):
        # index_add(input, dim, index, source): 1-D index value IS the full
        # destination coordinate along the indexed dim, so each raw index value
        # already names a distinct destination -- count raw duplicates directly.
        index = _arg(2, "index")
        return _max_raw_index_multiplicity(index)
    if func_name == "segment_reduce":
        # segment_reduce(data, reduce, lengths=...): fan-in = max segment length.
        lengths = saved_kwargs.get("lengths")
        if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
            return int(lengths.max().item())
        return 0

    # scatter_add*/scatter_reduce*: index is the 3rd operand, dim the 2nd.
    index = _arg(2, "index")
    if not isinstance(index, torch.Tensor) or index.numel() == 0:
        return 0
    dim = _arg(1, "dim")
    if not isinstance(dim, int):
        return 0
    return _max_destination_coordinate_fan_in(index, dim)


def _max_raw_index_multiplicity(index: Any) -> int:
    """Return the max count of any repeated raw index value (1-D index_add fan-in)."""

    if not isinstance(index, torch.Tensor) or index.numel() == 0:
        return 0
    try:
        flat = index.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat - flat.min())
        return int(counts.max().item())
    except (RuntimeError, ValueError):
        return 0


def _max_destination_coordinate_fan_in(index: torch.Tensor, dim: int) -> int:
    """Return the max number of n-D scatter source elements per destination tuple.

    Builds, for each source element, the full destination coordinate -- ``index``
    along the (normalized) scatter ``dim`` and the source position along every other
    dim -- then counts the largest group of identical destination coordinates. Two
    sources landing on the same destination report depth 2 no matter how many other
    destinations also happen to be indexed at the same raw value.
    """

    try:
        ndim = index.dim()
        if ndim == 0:
            return int(index.numel())
        scatter_dim = dim % ndim
        idx64 = index.to(torch.int64)
        # Per-axis destination coordinate: the index value along the scatter dim,
        # the source position (arange, broadcast) along every other dim.
        coord_axes = []
        for axis in range(ndim):
            if axis == scatter_dim:
                coord_axes.append(idx64.reshape(-1))
            else:
                shape = [1] * ndim
                shape[axis] = index.shape[axis]
                axis_positions = (
                    torch.arange(index.shape[axis], dtype=torch.int64)
                    .reshape(shape)
                    .expand_as(index)
                    .reshape(-1)
                )
                coord_axes.append(axis_positions)
        dest_tuples = torch.stack(coord_axes, dim=1)
        _, counts = torch.unique(dest_tuples, dim=0, return_counts=True)
        return int(counts.max().item())
    except (RuntimeError, ValueError):
        return 0
