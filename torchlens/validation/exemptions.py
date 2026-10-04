"""Validation exemption registries for perturbation and forward-replay checks.

Four registries control which operations are exempt from validation, and why:

1. ``SKIP_VALIDATION_ENTIRELY`` -- ops whose output is nondeterministic even
   with identical inputs and RNG state (e.g., ``empty_like`` returns
   uninitialized memory).  Both forward replay AND perturbation are skipped.

2. ``SKIP_PERTURBATION_ENTIRELY`` -- ops where no perturbable parent VALUE can
   change the output (shape/type templates, RNG-determined outputs, native
   ops unsafe to perturb).  Forward replay still runs to verify correctness.

3. ``STRUCTURAL_ARG_POSITIONS`` -- ops where SPECIFIC arg positions are
   structural (e.g., the index tensor in ``embedding``).  If the perturbed
   layer's tensor matches one of these positions, perturbation is skipped
   for that parent only.

4. ``CUSTOM_EXEMPTION_CHECKS`` -- ops requiring per-case logic that doesn't
   fit a simple position mapping (e.g., ``__getitem__`` tensor indexing,
   ``lstm`` hidden/cell states).

Additionally, ``posthoc_perturb_check`` handles dynamic exemptions that can
only be determined AFTER executing the function -- cases where perturbation
genuinely doesn't change the output for valid reasons (bool output, type
casting, special-value args like all-zeros making perturbation irrelevant).
"""

# Exemptions are tripwire-sensitive. Add a new exemption only when the predicate
# is proved from runtime evidence and a negative test shows it cannot mask the
# unintended value-sensitive case.

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any

import torch

from ..data_classes.op import Op
from ..utils.tensor_utils import tensor_all_nan

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


# ---------------------------------------------------------------------------
# Registry 1: Skip ALL validation (forward replay + perturbation).
# These funcs produce nondeterministic output (e.g. uninitialized memory),
# so even forward replay would fail.
# ---------------------------------------------------------------------------
SKIP_VALIDATION_ENTIRELY: dict[str, str] = {
    "empty_like": "returns uninitialized memory by construction; saved bytes are not replayable",
    # Membership for "new" is NECESSARY but not SUFFICIENT: Tensor.new is
    # overloaded, and only the argless / integer-sizes / torch.Size forms
    # return uninitialized memory. uninitialized_by_design_applies() proves
    # the size-only form per call; the value-bearing new(tensor)/new(data)
    # overloads fall through to real replay (b1-sol R08-1).
    "new": (
        "torch.Tensor.new() returns uninitialized memory by construction for the "
        "argless/size-only overloads ONLY, proved per call by "
        "uninitialized_by_design_applies"
    ),
    "new_empty": "torch.Tensor.new_empty() returns uninitialized memory by construction",
    "new_empty_strided": (
        "torch.Tensor.new_empty_strided() returns uninitialized memory by construction"
    ),
    "newempty": "canonical torch.Tensor.new_empty() spelling returns uninitialized memory",
    "newemptystrided": (
        "canonical torch.Tensor.new_empty_strided() spelling returns uninitialized memory"
    ),
}

# ---------------------------------------------------------------------------
# Registry 2: Skip perturbation only (forward replay still runs).
# Ops whose output values do not depend on any perturbable parent VALUE.
# Each entry maps func name -> per-entry justification (r3 fix of finding
# R19-F1: this was the only registry shaped as a bare set; every membership
# now carries its own proof, matching ``SKIP_VALIDATION_ENTIRELY``). Contract
# clause: C2 perturbation sensitivity (``validation/CLAUDE.md`` replay step 5)
# -- an entry is admissible only when the parent VALUE provably cannot affect
# the output, proved from the saved call, never assumed from the op's name.
# Round-31 registry narrowing: ``fill_`` (tensor fill VALUE at arg 1) and
# ``expand_as`` (arg 0 values flow into the output) moved to
# ``STRUCTURAL_ARG_POSITIONS`` so their genuine value edges stay
# perturbation-tested.
# ---------------------------------------------------------------------------
SKIP_PERTURBATION_ENTIRELY: dict[str, str] = {
    "new_zeros": "output is all zeros by construction; no parent value reaches it",
    "new_ones": "output is all ones by construction; no parent value reaches it",
    "zero_": "in-place zero fill; the destination's prior values are discarded",
    "zeros_like": "shape/dtype/device template only; the output value is constant zero",
    "ones_like": "shape/dtype/device template only; the output value is constant one",
    "rand_like": "values are RNG-drawn; the parent supplies shape/dtype/device only",
    "randn_like": "values are RNG-drawn; the parent supplies shape/dtype/device only",
    # meshgrid/broadcast_tensors moved to CUSTOM_EXEMPTION_CHECKS
    # (_check_zipped_sibling_exempt): per-output parent projection landed
    # (R08-2), so only genuine CROSS-member perturbations are exempt and each
    # output's OWN value edge is perturbation-tested again.
    # The six torchvision PyCapsule ops moved to ``STRUCTURAL_ARG_POSITIONS``
    # keyed on their coordinate/offset arg only (b1p2 D2 adjudication, all
    # three labs converged): the whole-op skip was wider than its segfault
    # justification, so feature/score value edges return under perturbation
    # with zero segfault exposure.
    "exponential_": "in-place RNG draw; the output is determined by RNG state, not inputs",
}

# ---------------------------------------------------------------------------
# Registry 3: Specific arg positions that are structural (not value-sensitive).
# When the perturbed layer's tensor matches saved_args[pos], skip perturbation.
# ---------------------------------------------------------------------------
STRUCTURAL_ARG_POSITIONS: dict[str, set[int]] = {
    # Value-bearing Tensor.new(tensor)/new(data): the SELF tensor (arg 0)
    # supplies dtype/device only -- its values never reach the output -- while
    # the data source (arg 1) stays strictly perturbation-tested. The
    # size-only overloads never get here (uninitialized_by_design_applies
    # exempts them before replay). R08-1 narrowing companion.
    "new": {0},
    "copy_": {0},  # destination values are overwritten; source values determine output
    # Zipped foreach spelling of ``copy_`` (r29 F5): each destination member is
    # TOTALLY overwritten by its zipped source member, so the destination list
    # (arg 0, matched per zipped slot ``(0, j)``) is value-irrelevant by
    # construction; source members (arg 1) stay strictly tested.
    "_foreach_copy_": {0},
    "foreachcopy": {0},  # canonicalized spelling
    "fill_": {0},  # destination values are overwritten; the fill VALUE (arg 1) stays tested
    "expand_as": {1},  # shape template only; arg 0 values flow into the output
    "expandas": {1},  # canonicalized spelling
    # F2 tightening: the legacy OOB-justified index/target/mask blankets
    # (``cross_entropy`` target, ``embedding`` indices, ``gather``/
    # ``index_select``/``scatter*`` index tensors, ``masked_fill`` masks) were
    # REMOVED from this registry. Those args are genuine VALUE dependencies --
    # the index/mask values select which data flows to the output -- so a
    # blanket skip could excuse a genuinely-missed dependency. They are now
    # perturbed IN-DOMAIN (``index_domain_rotation_values`` rotates valid
    # indices; boolean masks flip), with narrow exemptions only for provably
    # degenerate domains (``_check_index_domain_degenerate``) and provable
    # value-irrelevance (``_index_domain_value_irrelevance_decision``).
    "_pack_padded_sequence": {1},  # lengths tensor
    "_pad_packed_sequence": {1},  # lengths tensor
    "type_as": {1},  # type template tensor (value irrelevant)
    "new_tensor": {0},  # source tensor is a dtype/device/layout factory
    "newtensor": {0},  # canonicalized torch.Tensor.new_tensor spelling
    # Tensor.new_full/new_zeros/new_ones self args are pure dtype/device/layout
    # factory templates: the output is determined entirely by size/fill args,
    # never by the self tensor's VALUES (e.g. PyG SAGPooling's
    # ``num_nodes.new_full((n,), -1)``). Same class as new_tensor arg 0.
    "new_full": {0},
    "newfull": {0},
    "new_zeros": {0},
    "newzeros": {0},
    "new_ones": {0},
    "newones": {0},
    # torchvision C++ ops (PyCapsule): NARROWED from whole-op perturbation
    # skips (b1p2 D2 adjudication). Only the coordinate/offset arg is skipped
    # -- perturbed box coordinates / sampling offsets can index out of bounds
    # inside the native kernel and segfault past Python exception handling --
    # while feature and score args are genuine value edges that stay
    # perturbation-tested. This is a SAFETY skip, not a value-irrelevance
    # proof; the value edge through the coordinates stays guarded by replay.
    "nms": {0},  # boxes; scores (arg 1) stay tested
    "deform_conv2d": {1},  # sampling offsets; input/weight/bias/mask stay tested
    "roi_align": {1},  # boxes; the feature map (arg 0) stays tested
    "roi_pool": {1},  # boxes
    "ps_roi_align": {1},  # boxes
    "ps_roi_pool": {1},  # boxes
}


STRUCTURAL_ARG_KWARG_ALIASES: dict[str, dict[int, set[str]]] = {
    "_pack_padded_sequence": {1: {"lengths"}},
    "_pad_packed_sequence": {1: {"lengths"}},
    "type_as": {1: {"tensor", "other"}},
    "nms": {0: {"boxes"}},
    "deform_conv2d": {1: {"offset"}},
    "roi_align": {1: {"boxes", "rois"}},
    "roi_pool": {1: {"boxes", "rois"}},
    "ps_roi_align": {1: {"boxes", "rois"}},
    "ps_roi_pool": {1: {"boxes", "rois"}},
}


@dataclass(frozen=True)
class PosthocPerturbDecision:
    """Structured decision returned by posthoc perturbation predicates.

    Attributes
    ----------
    exempt:
        Whether the unchanged perturbation output is justified.
    reason:
        Stable reason code for diagnostics and golden snapshots.
    justification:
        Optional human-readable proof for design exemptions.
    """

    exempt: bool
    reason: str
    justification: str | None = None


# ---------------------------------------------------------------------------
# Index-domain perturbation standard (F2 tightening).
#
# The ops below consume an integer index/target arg whose VALUES genuinely
# determine the output, but whose valid domain is bounded by a sibling arg's
# shape (random draws can go out of bounds and crash the kernel). The legacy
# treatment blanket-skipped perturbing those args, which could excuse a
# genuinely-missed dependency. The tightened standard perturbs them IN-DOMAIN:
# every in-range entry is rotated by one position (``(v + 1) % n``, a bijection
# on ``[0, n)``), out-of-range sentinels (e.g. ``cross_entropy`` ignore_index)
# are preserved, and only the provably degenerate domain (``n <= 1`` or no
# in-range entries) is exempted pre-execution.
# ---------------------------------------------------------------------------

_INDEX_DOMAIN_INT_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

# func_name -> (positional index-arg slot, kwarg spellings of the index arg).
_INDEX_DOMAIN_ARG_SPECS: dict[str, tuple[int, frozenset[str]]] = {
    # aten spelling: embedding(weight, indices, ...) -- domain = weight rows.
    "embedding": (1, frozenset({"indices", "input"})),
    # gather/index_select/scatter*(input, dim, index, ...) -- domain =
    # input.shape[dim].
    "gather": (2, frozenset({"index"})),
    "index_select": (2, frozenset({"index"})),
    "scatter": (2, frozenset({"index"})),
    "scatter_": (2, frozenset({"index"})),
    "scatter_add": (2, frozenset({"index"})),
    "scatter_add_": (2, frozenset({"index"})),
    "scatteradd": (2, frozenset({"index"})),
    # cross_entropy(input, target, ...) -- domain = the class dimension.
    "cross_entropy": (1, frozenset({"target"})),
}


def _parent_is_index_domain_arg(layer: Op, parent_label: str) -> bool:
    """Return whether ``parent_label`` occupies the op's index/target arg slot.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is inspected.
    parent_label:
        Perturbed parent label.

    Returns
    -------
    bool
        True when the parent is registered at the index-arg position or one of
        its kwarg spellings. Position identity only -- never tensor equality.
    """

    spec = _INDEX_DOMAIN_ARG_SPECS.get(getattr(layer, "func_name", None) or "")
    if spec is None:
        return False
    index_pos, index_kwargs = spec
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    if (parent_arg_positions.get("args", {}) or {}).get(index_pos) == parent_label:
        return True
    kwarg_map = parent_arg_positions.get("kwargs", {}) or {}
    return any(kwarg_map.get(name) == parent_label for name in index_kwargs)


def _index_domain_size(layer: Op) -> int | None:
    """Return the exclusive upper bound of the op's valid index domain.

    Parameters
    ----------
    layer:
        Captured index-consuming op.

    Returns
    -------
    int or None
        Number of valid index values (``weight`` rows for ``embedding``,
        ``input.shape[dim]`` for the gather/scatter family, the class-dim size
        for ``cross_entropy``), or ``None`` when the saved call shape cannot
        prove a bound (callers then stay strict).
    """

    func_name = getattr(layer, "func_name", None)
    args: tuple[Any, ...] = getattr(layer, "saved_args", None) or ()
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    if func_name == "embedding":
        weight = kwargs.get("weight", args[0] if args else None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
        return None
    if func_name == "cross_entropy":
        logits = kwargs.get("input", args[0] if args else None)
        if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
            return None
        return int(logits.shape[1]) if logits.ndim >= 2 else int(logits.shape[0])
    source = kwargs.get("input", args[0] if args else None)
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    if not isinstance(source, torch.Tensor) or not isinstance(dim, int):
        return None
    if dim < 0:
        dim = source.ndim + dim
    if dim < 0 or dim >= source.ndim:
        return None
    return int(source.shape[dim])


def _saved_index_domain_arg_value(layer: Op) -> Any:
    """Return the saved index/target argument value for an index-domain op.

    Parameters
    ----------
    layer:
        Captured index-consuming op.

    Returns
    -------
    Any
        The saved argument at the index slot (positional or kwarg spelling),
        or ``None`` when it cannot be located.
    """

    spec = _INDEX_DOMAIN_ARG_SPECS.get(getattr(layer, "func_name", None) or "")
    if spec is None:
        return None
    index_pos, index_kwargs = spec
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    for name in index_kwargs:
        if name in kwargs:
            return kwargs[name]
    args: tuple[Any, ...] = getattr(layer, "saved_args", None) or ()
    if len(args) > index_pos:
        return args[index_pos]
    return None


def index_domain_rotation_values(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return a domain-safe rotated index perturbation for ``parent_values``.

    Every in-domain entry is rotated by one valid position
    (``(v + 1) % n``, guaranteed distinct from ``v`` when ``n >= 2``);
    out-of-domain entries (e.g. ``ignore_index`` sentinels) are preserved so
    the perturbed call stays executable. Mirrors the ``one_hot`` precedent.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor or None
        Rotated in-domain indices, or ``None`` when this parent is not an
        integer index arg of an index-domain op or no in-domain rotation
        exists (callers fall through to the generic strategies).
    """

    if not _parent_is_index_domain_arg(layer, parent_label):
        return None
    if not isinstance(parent_values, torch.Tensor):
        return None
    if parent_values.dtype not in _INDEX_DOMAIN_INT_DTYPES:
        return None
    domain_size = _index_domain_size(layer)
    if domain_size is None or domain_size < 2:
        return None
    in_domain = (parent_values >= 0) & (parent_values < domain_size)
    if not bool(in_domain.any()):
        return None
    rotated = (parent_values + 1).remainder(domain_size)
    return torch.where(in_domain, rotated, parent_values)


def _check_index_domain_degenerate(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt an index parent ONLY when no in-domain perturbation exists.

    A domain of ``n <= 1`` valid values, or a saved index tensor with zero
    in-domain entries (e.g. an all-``ignore_index`` target), admits no valid
    alternate index at all -- value-irrelevance is forced by the domain
    constraint, not assumed. Any perturbable domain returns False so the
    strict in-domain rotation check runs.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured index-consuming op.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is a provably unperturbable index arg.
    """

    del self
    if len(layers_to_perturb) != 1:
        return False
    if not _parent_is_index_domain_arg(layer, layers_to_perturb[0]):
        return False
    saved_index = _saved_index_domain_arg_value(layer)
    if not isinstance(saved_index, torch.Tensor):
        return False
    if saved_index.dtype not in _INDEX_DOMAIN_INT_DTYPES:
        return False
    domain_size = _index_domain_size(layer)
    if domain_size is None:
        return False
    if domain_size < 2:
        return True
    in_domain = (saved_index >= 0) & (saved_index < domain_size)
    return not bool(in_domain.any())


# ---------------------------------------------------------------------------
# Custom exemption check functions
# Signature: callable(self, layer, layers_to_perturb) -> bool
#   self = Trace instance
#   layer = Op being validated
#   layers_to_perturb = list of layer labels being perturbed
# ---------------------------------------------------------------------------


def _check_getitem_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt ``__getitem__`` only when the perturbed parent is an index arg."""

    del self
    positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    return bool(positions) and positions.isdisjoint({0})


# In-place ops whose FIRST positional arg (``args[0]``) is the destination
# tensor written into. The RoPE/normalizing-flow/GNN "partition-write into an
# uninitialized buffer" idiom chains these into ``empty``/``empty_like``/
# ``new_empty`` allocations. Every entry writes its destination at args[0]; the
# canonicalized TorchLens spellings (no underscore) are included alongside.
INPLACE_DESTINATION_WRITE_FUNCS: set[str] = {
    "__setitem__",
    "index_copy_",
    "indexcopy_",
    "index_copy",
    "indexcopy",
    "index_fill_",
    "indexfill_",
    "index_fill",
    "indexfill",
    "index_add_",
    "indexadd_",
    "index_add",
    "indexadd",
    "masked_scatter_",
    "maskedscatter_",
    "masked_scatter",
    "maskedscatter",
    "scatter_",
    "scatter",
}


def uninitialized_by_design_applies(op: Any) -> bool:
    """Return whether the registry-1 uninitialized-memory exemption holds.

    Registry membership alone is not proof for ``Tensor.new``: the func is
    OVERLOADED, and only the argless / integer-sizes / ``torch.Size`` forms
    return uninitialized memory. ``new(tensor)`` and ``new(sequence_data)``
    are initialized, value-bearing, deterministic calls -- classifying them
    uninitialized skipped replay entirely and blessed an actually WRONG
    replay ``exempted`` without execution (b1-sol R08-1, reproduced). The
    proof is fail-closed: a call not provably size-only is NOT exempt and
    falls through to real replay, where a wrong value fails loud.

    Parameters
    ----------
    op:
        Candidate operation record.

    Returns
    -------
    bool
        True when the op is registry-listed AND (for ``new``) the saved call
        is provably the uninitialized size-only/argless overload: at most the
        self tensor as parent, no keyword arguments, and every non-tensor
        positional argument a plain ``int`` or a ``torch.Size``.
    """

    if op is None or getattr(op, "func_name", None) not in SKIP_VALIDATION_ENTIRELY:
        return False
    if getattr(op, "func_name", None) != "new":
        return True
    parents = getattr(op, "parents", ()) or ()
    if len(parents) > 1:
        # A second tensor parent is the value-bearing new(tensor) overload.
        return False
    if getattr(op, "non_tensor_kwargs", None):
        return False
    for arg in getattr(op, "non_tensor_pos_args", None) or ():
        if isinstance(arg, bool):
            return False
        if isinstance(arg, (int, torch.Size)):
            continue
        # Sequences are DATA (legacy constructor semantics), not sizes.
        return False
    return True


def _uninitialized_value_origin(op: Any, source_trace: Any, depth: int = 0) -> bool:
    """Return whether an op's VALUE is itself uninitialized memory.

    True iff ``op`` is a DIRECT uninitialized-memory source
    (``empty``/``empty_like``/``new_empty``/...). The exemption it gates skips
    the perturbation-insensitivity check ONLY when the perturbed destination
    parent has no meaningful value to perturb -- i.e. it is literally
    uninitialized allocation memory.

    The walk does NOT chain through in-place destination writes. An in-place
    write (``index_copy_``/``__setitem__``/...) produces a tensor that contains
    REAL written data -- it is never "uninitialized memory" as a whole, even
    when its destination buffer was allocated by ``empty_like``. Following the
    chain back to the allocation was a tripwire hole (B1 review,
    ``TwoIndexCopyDim2``): the second write of an
    ``out = empty_like(x); out.index_copy_(...); out.index_copy_(...)`` idiom
    consumes the FIRST write's live data through its destination parent, so a
    wrong replay that drops that destination dependency must still fail the
    perturbation check. Only a parent that is itself an allocation op is exempt;
    the ``depth``/``source_trace`` parameters are retained for signature
    compatibility but are no longer used to recurse.
    """

    if op is None:
        return False
    # Same per-call proof as the replay-skip consumer: a value-bearing
    # new(tensor)/new(data) result is REAL data, never uninitialized
    # allocation memory (R08-1 sibling site).
    return uninitialized_by_design_applies(op)


def _perturbed_parent_is_uninitialized_setitem_dest(
    layer: Op,
    layers_to_perturb: list[str],
) -> bool:
    """Return whether a perturbed in-place-write parent is an uninitialized dest.

    Returns True iff EVERY perturbed parent of this in-place destination-write op
    occupies its DESTINATION slot (``args[0]``) AND that perturbed parent is
    ITSELF a direct uninitialized-memory allocation
    (``empty``/``empty_like``/``new_empty``). Perturbing literally-uninitialized
    allocation memory has no semantic meaning -- the unwritten positions are
    garbage overwritten by a sibling/subsequent write and never meaningfully
    consumed -- so a "no output change" result is not a capture bug.

    The check is deliberately NARROW so it cannot mask a real dropped-dependency
    bug: if the perturbed parent is a genuine data tensor -- INCLUDING a prior
    in-place write whose buffer was allocated by ``empty_like`` but which now
    holds real written data -- or is NOT the destination, this returns False and
    the strict perturbation check stands. The exemption does NOT chain through
    intermediate writes (B1 review hole: a chained ``index_copy_`` destination
    holds live data and must stay strict).

    Parameters
    ----------
    layer:
        The in-place destination-write op being validated.
    layers_to_perturb:
        Parent layer labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbation is an exempt uninitialized-destination case.
    """

    if not layers_to_perturb:
        return False

    # Resolve the source op of each perturbed parent from this op's saved source
    # trace so we can read its producing func_name without a Trace handle.
    source_trace = getattr(layer, "_source_trace", None)
    if source_trace is None:
        return False

    # The destination tensor of an in-place write is the first positional arg.
    # The perturbed parent must occupy that args[0] slot to be the destination.
    dest_labels: set[str] = set()
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    for key, parent_label in parent_arg_positions.get("args", {}).items():
        # args[0] is the destination; a nested key like (0, ...) is NOT the
        # whole-destination tensor (it would index INTO the value/index args).
        if key == 0:
            dest_labels.add(parent_label)

    for perturbed_label in layers_to_perturb:
        if perturbed_label not in dest_labels:
            return False
        try:
            perturbed_op = source_trace[perturbed_label]
        except Exception:
            return False
        if not _uninitialized_value_origin(perturbed_op, source_trace):
            return False
    return True


def _check_setitem_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt ``__setitem__`` for structural masks or proven full overwrites."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args

    # Case 1: saved_args[1] is a bool tensor and perturbed layer matches it (mask arg)
    if (
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 1)
        and len(args) > 1
        and isinstance(args[1], torch.Tensor)
        and args[1].dtype == torch.bool
    ):
        return True

    # Case 2: saved_args[1] is a tuple whose first element is a bool tensor
    if (
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 1)
        and len(args) > 1
        and isinstance(args[1], tuple)
        and isinstance(args[1][0], torch.Tensor)
        and args[1][0].dtype == torch.bool
    ):
        return True

    # Case 3: perturbed layer is the destination, but the indexed destination
    # slice is fully overwritten by the replacement value.
    return bool(
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0)
        and _setitem_destination_slice_is_fully_overwritten(perturbed_tensor, args)
    )


def _setitem_destination_slice_is_fully_overwritten(
    perturbed_tensor: torch.Tensor | None,
    args: tuple[Any, ...],
) -> bool:
    """Return whether a ``__setitem__`` call overwrites the perturbed destination slice.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    args:
        Saved ``__setitem__`` positional arguments.

    Returns
    -------
    bool
        True when the replacement is a tensor, the perturbed tensor is the
        destination, the selected region covers the whole destination without
        duplicate targets, and the replacement exactly matches the selected
        shape. A non-tensor replacement is NEVER exempted here: this pre-exec
        contract is deliberately unchanged, so this path can only ever grow
        stricter, never wider.
    """

    if len(args) < 3 or not isinstance(args[2], torch.Tensor):
        return False
    return _setitem_destination_coverage_is_total(perturbed_tensor, args)


def _setitem_destination_coverage_is_total(
    perturbed_tensor: torch.Tensor | None,
    args: tuple[Any, ...],
) -> bool:
    """Return whether a ``__setitem__`` write provably covers its whole destination.

    Shared geometry proof behind BOTH the pre-execution
    ``_setitem_destination_slice_is_fully_overwritten`` gate and the posthoc
    ``full_destination_overwrite`` / ``scalar_destination_overwrite`` decisions,
    so a scalar right-hand side is held to the SAME index-coverage and
    duplicate-target rigor a tensor right-hand side already was.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    args:
        Saved ``__setitem__`` positional arguments.

    Returns
    -------
    bool
        True when the perturbed tensor is the destination, ``destination[index]``
        selects every destination element exactly once, and a tensor replacement
        (when present) exactly matches the selected shape.
    """

    if len(args) < 3:
        return False
    destination, index, replacement = args[:3]
    if not isinstance(perturbed_tensor, torch.Tensor):
        return False
    if not isinstance(destination, torch.Tensor):
        return False
    if not torch.equal(perturbed_tensor, destination):
        return False
    try:
        selected = destination[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    if not _setitem_index_targets_are_unique(index):
        return False
    if selected.numel() != destination.numel():
        return False
    if not _index_positions_cover_destination_exactly(destination, index):
        return False
    if isinstance(replacement, torch.Tensor):
        return tuple(selected.shape) == tuple(replacement.shape)
    return True


def _setitem_index_targets_are_unique(index: Any) -> bool:
    """Return whether a ``__setitem__`` index cannot duplicate write targets.

    Parameters
    ----------
    index:
        Index argument supplied to ``Tensor.__setitem__``.

    Returns
    -------
    bool
        True for basic indexing and for verified unique tensor/list advanced
        indices. Duplicate advanced indices can make ``selected.numel()`` equal
        the destination size while leaving some destination elements live.
    """

    components = index if isinstance(index, tuple) else (index,)
    for component in components:
        if isinstance(component, list):
            try:
                component = torch.as_tensor(component)
            except (TypeError, ValueError):
                return False
        if isinstance(component, torch.Tensor):
            if component.dtype == torch.bool:
                continue
            if not _tensor_is_integer_index(component):
                return False
            flattened = component.reshape(-1)
            if int(flattened.numel()) != int(torch.unique(flattened).numel()):
                return False
            continue
        if component is None or component is Ellipsis or isinstance(component, (slice, int)):
            continue
        return False
    return True


def _index_positions_cover_destination_exactly(
    destination: torch.Tensor,
    index: Any,
) -> bool:
    """Return whether ``destination[index]`` addresses every element exactly once.

    Value-based uniqueness (``torch.unique`` on raw index values) is blind to
    negative-index aliasing: ``0`` and ``-2`` are distinct VALUES that address
    the SAME position on a length-2 dim, so a "fully overwritten" proof counted
    a full overwrite while an element survived with its prior value. Indexing
    an identity-POSITION tensor with the saved index makes torch's own indexing
    semantics normalize negatives, slices, ellipsis, and boolean masks exactly;
    requiring the selected positions to be unique and to number the whole
    destination is the exact single-coverage proof.

    Parameters
    ----------
    destination:
        Destination tensor of the write.
    index:
        Saved index argument (``__setitem__`` index or ``index_put`` indices
        tuple).

    Returns
    -------
    bool
        True only when the index selects each destination position exactly
        once and selects all of them. Any indexing failure returns False so
        callers fail closed.
    """

    try:
        positions = torch.arange(destination.numel(), device=destination.device).reshape(
            destination.shape
        )
        covered = positions[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    flattened = covered.reshape(-1)
    if int(flattened.numel()) != int(destination.numel()):
        return False
    return int(torch.unique(flattened).numel()) == int(destination.numel())


def _tensor_is_integer_index(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` has an integer dtype accepted for indexing.

    Parameters
    ----------
    tensor:
        Tensor index component.

    Returns
    -------
    bool
        True when the tensor dtype is an integer indexing dtype.
    """

    return not tensor.dtype.is_floating_point and not tensor.dtype.is_complex


def _check_index_put_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt ``index_put``/``index_put_`` when the destination is fully overwritten.

    The exact analogue of :func:`_check_setitem_exempt` Case 4 for the
    ``index_put`` family. ``index_put(input, indices, values, accumulate=False)``
    overwrites ``input[indices]`` with ``values`` when ``accumulate`` is False, so
    the destination's prior value at those positions is provably irrelevant. This
    exemption fires ONLY when the perturbed parent IS the destination (``args[0]``)
    and the written positions are fully overwritten; it must NOT exempt a perturbed
    VALUE or INDEX parent (those genuinely influence the output), and it must NOT
    exempt the accumulating case (where the prior destination value IS added in).

    The destination is identified by ARG POSITION (``parent_arg_positions["args"][0]
    == layers_to_perturb[0]``), not solely by tensor-content equality: a value-parent
    whose contents happen to equal the destination would otherwise be falsely
    exempted by the downstream ``torch.equal`` check. Requiring the perturbed parent
    to occupy arg slot 0 mirrors how the ``__mod__`` divisor exemption keys off
    ``parent_arg_positions``.
    """
    if not _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0):
        return False
    perturbed_tensor = self[layers_to_perturb[0]].out
    return _index_put_destination_is_fully_overwritten(perturbed_tensor, layer)


def _perturbed_parent_is_arg_position(
    layer: Op,
    layers_to_perturb: list[str],
    position: int,
) -> bool:
    """Return whether the perturbed parent occupies positional arg ``position``.

    Reads ``layer.parent_arg_positions["args"]`` (arg index -> parent layer label)
    and checks the perturbed label is the parent registered at ``position``. Used to
    confirm a perturbed parent is the DESTINATION (slot 0) by structure rather than
    by tensor-content equality, which can collide when a value-parent's contents
    match the destination.
    """

    arg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
    return arg_positions.get(position) == layers_to_perturb[0]


def _perturbed_parent_arg_positions(layer: Op, layers_to_perturb: list[str]) -> set[int]:
    """Return positional arg slots occupied by the perturbed parent.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is being inspected.
    layers_to_perturb:
        Single perturbed parent label supplied by validation.

    Returns
    -------
    set[int]
        Positional arg indices whose parent label is the perturbed layer. Missing
        or ambiguous metadata returns an empty set so callers fail closed.
    """

    if len(layers_to_perturb) != 1:
        return set()
    arg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
    return {
        position
        for position, parent_label in arg_positions.items()
        if parent_label == layers_to_perturb[0]
    }


def _index_put_destination_is_fully_overwritten(
    perturbed_tensor: torch.Tensor | None,
    layer: Op,
) -> bool:
    """Return whether an ``index_put`` call overwrites the perturbed destination.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    layer:
        Captured ``index_put``/``index_put_`` op.

    Returns
    -------
    bool
        True only when the perturbed tensor is the destination (``args[0]``), the
        call is non-accumulating, and the indexed positions cover the ENTIRE
        destination and are exactly written by the broadcast ``values`` (so the
        destination's prior value is wholly irrelevant). Returns False for any
        perturbed VALUE/INDEX parent, for the accumulating case, and for a
        partial overwrite (where un-indexed destination elements still flow
        through).
    """

    args = layer.saved_args
    if not isinstance(perturbed_tensor, torch.Tensor):
        return False
    if args is None or len(args) < 3:
        return False
    destination, indices, values = args[0], args[1], args[2]
    if not isinstance(destination, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    # Narrow: the perturbed parent must be the DESTINATION, never the values/index.
    if not torch.equal(perturbed_tensor, destination):
        return False
    # accumulate=True adds the value to the prior destination, so the prior value
    # is NOT irrelevant -- never exempt that case. accumulate may arrive as a
    # positional arg (index 3) or as a keyword.
    accumulate = False
    if len(args) > 3:
        accumulate = bool(args[3])
    elif "accumulate" in (layer.saved_kwargs or {}):
        accumulate = bool(layer.saved_kwargs["accumulate"])
    if accumulate:
        return False
    # index_put indices are an advanced-indexing tuple/list of LongTensors.
    if isinstance(indices, list):
        index = tuple(indices)
    elif isinstance(indices, tuple):
        index = indices
    else:
        index = (indices,)
    try:
        selected = destination[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    # The written slice must broadcast the replacement exactly (every selected
    # element is overwritten, none left at its prior value).
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(selected.shape), tuple(values.shape))
    except RuntimeError:
        return False
    if tuple(broadcast_shape) != tuple(selected.shape):
        return False
    # The exemption is only sound when the WHOLE destination is overwritten: any
    # un-indexed element keeps its prior value and so still influences the output
    # (the partial-overwrite false-exemption guard). Require the indexed region to
    # cover every destination element, with no duplicate indices inflating the
    # count -- duplicates would match the numel without covering everything.
    if not _index_put_indices_are_unique(index):
        return False
    if int(selected.numel()) != int(destination.numel()):
        return False
    return _index_positions_cover_destination_exactly(destination, index)


def _index_put_indices_are_unique(index: tuple[Any, ...]) -> bool:
    """Return whether advanced ``index_put`` indices address distinct positions.

    Duplicate indices would let ``selected.numel()`` reach ``destination.numel()``
    without actually covering every destination element, so the full-overwrite
    coverage check would be fooled. This conservatively requires each integer
    index tensor to hold unique values; a non-integer (e.g. boolean mask) or any
    structure it cannot verify returns False so the exemption is withheld.
    """

    for component in index:
        if not isinstance(component, torch.Tensor):
            return False
        if component.dtype == torch.bool:
            # A bool mask selects each True position once -> inherently unique.
            continue
        flattened = component.reshape(-1)
        if int(flattened.numel()) != int(torch.unique(flattened).numel()):
            return False
    return True


def _perturbed_parent_occupies_arg_slot(
    layer: Op,
    layers_to_perturb: list[str],
    slot: int,
) -> bool:
    """Return whether the perturbed parent sits anywhere inside positional ``slot``.

    ``parent_arg_positions["args"]`` keys a parent that was nested inside a
    container argument by a TUPLE path -- a real ``lstm(input, (h0, c0))``
    capture registers ``(1, 0)`` and ``(1, 1)``, never a bare ``1``. Matching
    only bare integer keys therefore misses every genuinely nested structural
    argument. This resolves the OWNING positional slot for both flat and nested
    keys, so an exemption keyed on argument POSITION keeps the reach its
    content-equality predecessor had without ever consulting tensor VALUES.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is inspected.
    layers_to_perturb:
        Single perturbed parent label supplied by validation.
    slot:
        Positional argument index the exemption is scoped to.

    Returns
    -------
    bool
        True when the perturbed parent occupies ``slot`` or any position nested
        inside it. Missing or ambiguous metadata returns False (fail closed).
    """

    if len(layers_to_perturb) != 1:
        return False
    arg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
    for position, parent_label in arg_positions.items():
        if parent_label != layers_to_perturb[0]:
            continue
        if position == slot:
            return True
        if isinstance(position, tuple) and position and position[0] == slot:
            return True
    return False


def _check_lstm_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt lstm when the perturbed layer is a hidden/cell state arg.

    Keyed on ARGUMENT POSITION, never on ``torch.equal`` against the hidden
    argument: a data input that merely happens to equal a zero-initialized
    ``h0`` must NOT be excused as structural.
    """
    del self
    return _perturbed_parent_occupies_arg_slot(layer, layers_to_perturb, 1)


def _check_interpolate_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt interpolate when the perturbed layer is the scale_factor arg.

    Keyed on ARGUMENT POSITION, never on ``torch.equal`` against the saved
    scale factor.
    """
    del self
    if len(layers_to_perturb) != 1:
        return False
    args = layer.saved_args
    kwargs = layer.saved_kwargs
    if (
        _perturbed_parent_occupies_arg_slot(layer, layers_to_perturb, 2)
        and len(args) >= 3
        and args[2] is not None
    ):
        return True
    kwarg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("kwargs", {})
    return (
        kwargs.get("scale_factor") is not None
        and kwarg_positions.get("scale_factor") == layers_to_perturb[0]
    )


def _get_scatter_destination_dim_index(
    layer: Op,
) -> tuple[torch.Tensor, int, torch.Tensor] | None:
    """Return scatter destination, dim, and index tensors when they are replayable.

    Parameters
    ----------
    layer:
        Scatter operation being validated.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor] | None
        Destination tensor, scatter dimension, and index tensor, or ``None``
        when the call shape is unsupported or uses reduce semantics.
    """

    args = layer.saved_args
    kwargs = layer.saved_kwargs
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return None
    if kwargs.get("reduce") is not None:
        return None
    if len(args) > 4 and args[4] is not None:
        return None

    dest = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    index = kwargs.get("index", args[2] if len(args) > 2 else None)
    if not isinstance(dim, int) or not isinstance(index, torch.Tensor):
        return None
    if dim < 0:
        dim = dest.ndim + dim
    if dim < 0 or dim >= dest.ndim:
        return None
    return dest, dim, index


def _scatter_index_fully_overwrites_dim(dest: torch.Tensor, dim: int, index: torch.Tensor) -> bool:
    """Return whether scatter index covers every destination slot along ``dim``.

    Parameters
    ----------
    dest:
        Scatter destination tensor.
    dim:
        Normalized scatter dimension.
    index:
        Scatter index tensor.

    Returns
    -------
    bool
        True when every slice orthogonal to ``dim`` contains each valid
        destination index, making the destination's prior values irrelevant.
    """

    if index.ndim != dest.ndim or index.shape[dim] < dest.shape[dim]:
        return False
    if any(index.shape[axis] < dest.shape[axis] for axis in range(dest.ndim) if axis != dim):
        return False
    n_positions = dest.shape[dim]
    if n_positions == 0:
        return False
    moved = index.detach().cpu().movedim(dim, -1).reshape(-1, index.shape[dim])
    required = set(range(n_positions))
    for row in moved:
        row_values = {int(value) for value in row.tolist() if 0 <= int(value) < n_positions}
        if not required.issubset(row_values):
            return False
    return True


def _check_scatter_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt scatter destination perturbation when scatter fully overwrites it."""

    if not _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0):
        return False
    perturbed_tensor = self[layers_to_perturb[0]].out
    scatter_components = _get_scatter_destination_dim_index(layer)
    if scatter_components is None:
        return False
    dest, dim, index = scatter_components
    if not torch.equal(perturbed_tensor, dest):
        return False
    return _scatter_index_fully_overwrites_dim(dest, dim, index)


def _check_one_arg_where_index_exempt(layer: Op) -> bool:
    """Return whether ``layer`` is the one-arg ``torch.where(condition)`` index form.

    One-arg ``torch.where(condition)`` is torch's alias for
    ``nonzero(condition, as_tuple=True)``: its output is a DISCRETE integer INDEX set, not a
    value select, so small value-perturbation legitimately cannot change it -- the same
    category as the topk/sort/max/min index exemption.

    NARROW + tripwire-safe: the 3-arg VALUE-select ``where`` must NOT match. In particular the
    mixed form ``torch.where(cond, input=a, other=b)`` is a genuine value select that TorchLens
    records as ``len(saved_args) == 1`` with ``input``/``other`` in ``saved_kwargs`` (and an
    integer output dtype when the branches are integer), so arg-count and dtype alone are NOT
    sufficient. This asserts NO branch appears in args OR kwargs, matching only the true one-arg
    forms: positional ``where(cond)`` -> ``saved_args == (cond,)``, ``saved_kwargs == {}``; and
    keyword ``where(condition=cond)`` -> ``saved_args == ()``, ``saved_kwargs == {'condition'}``.

    DUAL-LAB reviewed 2026-06-30 (Claude + Codex independently converged on this exact
    predicate; the kwarg guard is load-bearing, the int-dtype gate mirrors the topk/sort
    precedent). The 3-arg ``_check_where_exempt`` (``len(saved_args) >= 3``) path is untouched
    and stays fully armed.

    Parameters
    ----------
    layer:
        Operation whose perturbation-insensitivity is being excused.

    Returns
    -------
    bool
        Whether the op is a genuine one-arg ``where`` index form.
    """

    if getattr(layer, "func_name", None) != "where":
        return False
    if layer.dtype not in (torch.int, torch.long, torch.int32, torch.int64):
        return False
    saved_args = layer.saved_args or ()
    saved_kwargs = getattr(layer, "saved_kwargs", None) or {}
    if len(saved_args) == 1 and saved_kwargs == {}:
        return True
    return bool(len(saved_args) == 0 and set(saved_kwargs) == {"condition"})


def _check_where_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt ``where`` parents only when saved value semantics prove irrelevance.

    The condition parent is irrelevant when the saved true/false branches are
    equal after broadcasting. A branch parent is irrelevant only when the SAVED
    condition selects the opposite branch at every element in the full broadcast
    output shape. Branch identity is taken from ``parent_arg_positions`` and the
    saved condition is the only selectedness source; any missing metadata,
    non-tensor saved arg, duplicate branch parent, or broadcast failure returns
    False.
    """

    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if not perturbed_positions:
        return False
    args = layer.saved_args
    if args is None:
        return False
    if len(args) < 3:
        return False
    condition, true_branch, false_branch = args[:3]
    if not (
        isinstance(condition, torch.Tensor)
        and isinstance(true_branch, torch.Tensor)
        and isinstance(false_branch, torch.Tensor)
    ):
        return False
    if 0 in perturbed_positions:
        if perturbed_positions != {0}:
            return False
        try:
            true_values, false_values = torch.broadcast_tensors(true_branch, false_branch)
        except RuntimeError:
            return False
        return bool(torch.equal(true_values, false_values))
    if not perturbed_positions.issubset({1, 2}):
        return False
    if len(perturbed_positions) != 1:
        return False
    selectedness = _where_saved_condition_selectedness(condition, true_branch, false_branch)
    if selectedness is None:
        return False
    selected_true, total_elements = selectedness
    if 1 in perturbed_positions:
        return selected_true == 0
    if 2 in perturbed_positions:
        return selected_true == total_elements
    return False


def _where_saved_condition_selectedness(
    condition: torch.Tensor,
    true_branch: torch.Tensor,
    false_branch: torch.Tensor,
) -> tuple[int, int] | None:
    """Return selected true-branch count over the full saved ``where`` shape.

    Parameters
    ----------
    condition:
        Saved ``where`` condition tensor.
    true_branch:
        Saved true-branch tensor.
    false_branch:
        Saved false-branch tensor.

    Returns
    -------
    tuple[int, int] | None
        ``(true_selected_count, total_elements)`` after broadcasting the saved
        condition to ``broadcast_shapes(condition, true_branch, false_branch)``.
        Returns ``None`` on empty output or any broadcast/count failure.
    """

    try:
        condition_bool = condition.to(dtype=torch.bool)
        output_shape = torch.broadcast_shapes(
            tuple(condition_bool.shape),
            tuple(true_branch.shape),
            tuple(false_branch.shape),
        )
        broadcast_condition = torch.broadcast_to(condition_bool, output_shape)
        total_elements = int(broadcast_condition.numel())
        if total_elements == 0:
            return None
        true_selected = int(torch.count_nonzero(broadcast_condition).item())
    except (TypeError, RuntimeError):
        return None
    return true_selected, total_elements


def _masked_fill_saved_mask_selectedness(
    mask: torch.Tensor,
    input_tensor: torch.Tensor,
    value: torch.Tensor | Number,
) -> tuple[int, int] | None:
    """Return selected fill-value count over the full saved ``masked_fill`` shape.

    Parameters
    ----------
    mask:
        Saved boolean mask argument.
    input_tensor:
        Saved input/destination tensor.
    value:
        Saved scalar tensor or Python scalar fill value.

    Returns
    -------
    tuple[int, int] | None
        ``(fill_selected_count, total_elements)`` after broadcasting the saved
        mask over the full input/value output shape. Returns ``None`` if the
        saved data cannot prove exact selectedness.
    """

    try:
        mask_bool = mask.to(dtype=torch.bool)
        value_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else ()
        output_shape = torch.broadcast_shapes(
            tuple(mask_bool.shape),
            tuple(input_tensor.shape),
            value_shape,
        )
        broadcast_mask = torch.broadcast_to(mask_bool, output_shape)
        total_elements = int(broadcast_mask.numel())
        if total_elements == 0:
            return None
        fill_selected = int(torch.count_nonzero(broadcast_mask).item())
    except (TypeError, RuntimeError):
        return None
    return fill_selected, total_elements


def _masked_fill_input_equals_value_everywhere(
    input_tensor: torch.Tensor,
    mask: torch.Tensor,
    value: torch.Tensor | Number,
) -> bool:
    """Return whether ``input == value`` at every broadcast output position.

    When the saved input already equals the fill value everywhere, ANY mask
    (including every possible perturbation) produces the identical output, so
    the mask's value-irrelevance is proved, not assumed.

    Parameters
    ----------
    input_tensor:
        Saved input/destination tensor.
    mask:
        Saved boolean mask argument (bounds the broadcast output shape).
    value:
        Saved scalar tensor or Python scalar fill value.

    Returns
    -------
    bool
        True only when the equality is provable over the full output shape.
    """

    try:
        if isinstance(value, torch.Tensor):
            value_tensor = value
        else:
            value_tensor = torch.as_tensor(
                value, dtype=input_tensor.dtype, device=input_tensor.device
            )
        equal = torch.eq(input_tensor, value_tensor)
        output_shape = torch.broadcast_shapes(
            tuple(equal.shape),
            tuple(mask.shape),
        )
        broadcast_equal = torch.broadcast_to(equal, output_shape)
        if broadcast_equal.numel() == 0:
            return False
        return bool(broadcast_equal.all().item())
    except (TypeError, ValueError, RuntimeError):
        return False


def _check_masked_fill_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt ``masked_fill`` parents only when saved values prove irrelevance.

    ``masked_fill(input, mask, value)`` is equivalent to
    ``where(mask, value, input)``. The input parent is irrelevant only when the
    saved mask is true at every output element; a tensor fill-value parent is
    irrelevant only when the saved mask is false at every output element. The
    mask parent (F2 tightening: no longer a structural-position blanket) is
    irrelevant only when the saved input already equals the fill value at
    every broadcast position -- then every possible mask yields the same
    output. Any other mask perturbation must run and register sensitivity.
    """

    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if not perturbed_positions:
        return False
    args = layer.saved_args
    if args is None or len(args) < 3:
        return False
    input_tensor, mask, value = args[:3]
    if not (
        isinstance(input_tensor, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        and isinstance(value, (torch.Tensor, Number))
    ):
        return False
    if perturbed_positions == {1}:
        return _masked_fill_input_equals_value_everywhere(input_tensor, mask, value)
    if not perturbed_positions.issubset({0, 2}):
        return False
    if len(perturbed_positions) != 1:
        return False
    selectedness = _masked_fill_saved_mask_selectedness(mask, input_tensor, value)
    if selectedness is None:
        return False
    fill_selected, total_elements = selectedness
    if 0 in perturbed_positions:
        return fill_selected == total_elements
    if 2 in perturbed_positions:
        return fill_selected == 0
    return False


def _batch_norm_weight_operand(layer: Op) -> Any:
    """Return the saved ``weight`` (gamma) operand of a batch_norm/instance_norm op.

    Positional layout matches torch's real ATen call signature:
    ``(input, weight, bias, running_mean, running_var, training, momentum,
    eps, cudnn_enabled)`` -- NOT ``F.batch_norm``'s Python-level
    ``(input, running_mean, running_var, weight, bias, ...)`` ordering, which
    is why ``_check_norm_running_stat_exempt`` reads running_mean/running_var
    from positions 3/4, not 1/2.
    """

    args = layer.saved_args
    if args is None or len(args) <= 1:
        return None
    return args[1]


def _check_norm_zero_weight_annihilates(layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt input/running_mean/running_var when ``weight`` is provably all-zero.

    ``batch_norm``'s output is
    ``((input - running_mean) / sqrt(running_var + eps)) * weight + bias``
    (the training-mode form substitutes batch statistics for the running
    buffers, but keeps the same ``* weight + bias`` outer shape): an exactly
    zero ``weight`` annihilates the WHOLE normalized term in either mode,
    leaving the output identically equal to ``bias`` regardless of input,
    running_mean, or running_var. This is the same zero-annihilator proof
    armed for ``mul``/``multiply``/``addcmul``, applied to batch_norm's own
    ``weight`` operand. timm's "zero_init_last" convention zero-initializes
    the LAST BatchNorm's weight in every residual block of many ResNet-family
    architectures, so an untrained instance hits this at validation time.

    Narrow by construction: only positions {0, 3, 4} (input, running_mean,
    running_var) are ever exempted here. Perturbing ``weight`` or ``bias``
    themselves (positions 1/2) is NEVER exempted by this check -- weight
    moving off zero, or bias directly, both genuinely change the output.
    """

    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if not perturbed_positions or not perturbed_positions.issubset({0, 3, 4}):
        return False
    weight = _batch_norm_weight_operand(layer)
    return isinstance(weight, torch.Tensor) and weight.numel() > 0 and bool(torch.all(weight == 0))


def _check_norm_running_stat_exempt(self: "Trace", layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt training-mode running-stat parents and zero-weight-annihilated parents.

    Parameters
    ----------
    self:
        Trace containing saved parent payloads.
    layer:
        ``batch_norm`` or ``instance_norm`` op being validated.
    layers_to_perturb:
        Parent layer labels currently being perturbed.

    Returns
    -------
    bool
        True when EITHER: (a) the perturbed parent is ``running_mean`` or
        ``running_var`` (args 3/4) for a training-mode BatchNorm/InstanceNorm
        call -- those buffers are update targets in training mode; batch/input
        statistics determine the output value; or (b) every perturbed parent
        is input/running_mean/running_var (args 0/3/4) and the op's saved
        ``weight`` operand is provably all-zero -- see
        ``_check_norm_zero_weight_annihilates``.
    """

    del self
    if _check_norm_zero_weight_annihilates(layer, layers_to_perturb):
        return True

    args = layer.saved_args
    if args is None or len(args) <= 5 or args[5] is not True:
        return False
    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    return bool(perturbed_positions) and perturbed_positions.issubset({3, 4})


def _check_scatter_or_index_domain_exempt(
    self: "Trace",
    layer: Op,
    layers_to_perturb: list[str],
) -> bool:
    """Exempt scatter for a fully-overwritten destination or a degenerate index domain."""

    return _check_scatter_exempt(self, layer, layers_to_perturb) or (
        _check_index_domain_degenerate(self, layer, layers_to_perturb)
    )


# ---------------------------------------------------------------------------
# Registry 4: Custom exemption checks keyed by func name.
# ---------------------------------------------------------------------------
def _perturbed_parents_are_zipped_siblings(layer: Op, layers_to_perturb: list[str]) -> bool:
    """Return whether every perturbed parent is a CROSS-member zipped input.

    ``meshgrid``/``broadcast_tensors`` zip N inputs to N outputs: output ``j``
    carries EXACTLY input ``j``'s values, so perturbing input ``k != j`` is
    legitimately insensitive for output ``j``, while perturbing input ``j``
    must change it. The former whole-op skip (R08-2, b1-sol) exempted BOTH
    directions, so a dropped or misattributed value edge on these
    multi-output ops was never perturbation-proved.

    The projection is fail-closed: a missing ``multi_output_index``, an
    unrecognized arg-position shape, or a perturbed parent not found in the
    positional map keeps the perturbation STRICT (returns False).

    Parameters
    ----------
    layer:
        Zipped multi-output operation record (one output's op).
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    bool
        True when every perturbed parent sits at a zipped index other than
        this output's own ``multi_output_index``.
    """

    own_index = getattr(layer, "multi_output_index", None)
    if not isinstance(own_index, int):
        return False
    parent_arg_positions = getattr(layer, "parent_arg_positions", None) or {}
    args_map = parent_arg_positions.get("args") if isinstance(parent_arg_positions, dict) else None
    if not isinstance(args_map, dict) or not args_map:
        return False
    for perturbed_label in layers_to_perturb:
        positions = [position for position, label in args_map.items() if label == perturbed_label]
        if not positions:
            return False
        for position in positions:
            zipped_index = position[-1] if isinstance(position, tuple) and position else position
            if not isinstance(zipped_index, int) or zipped_index == own_index:
                return False
    return True


def _check_zipped_sibling_exempt(
    source_trace: "Trace", layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Custom check: exempt only cross-member zipped-sibling perturbations.

    Parameters
    ----------
    source_trace:
        Trace being validated (unused; custom-check signature).
    layer:
        Zipped multi-output operation record.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    bool
        True when the perturbation targets only zipped siblings of this
        output (provably value-irrelevant); False keeps it strict.
    """

    return _perturbed_parents_are_zipped_siblings(layer, layers_to_perturb)


CUSTOM_EXEMPTION_CHECKS: dict[str, Callable[["Trace", Op, list[str]], bool]] = {
    # R08-2 per-output parent projection: only CROSS-member zipped-sibling
    # perturbations are exempt; each output's own value edge stays tested.
    # Both spellings of broadcast_tensors are registered -- capture
    # canonicalizes to "broadcasttensors", which the old whole-op skip never
    # matched (a silently dead registry row).
    "meshgrid": _check_zipped_sibling_exempt,
    "broadcast_tensors": _check_zipped_sibling_exempt,
    "broadcasttensors": _check_zipped_sibling_exempt,
    "__getitem__": _check_getitem_exempt,
    "__setitem__": _check_setitem_exempt,
    "index_put": _check_index_put_exempt,
    "index_put_": _check_index_put_exempt,
    "lstm": _check_lstm_exempt,
    "interpolate": _check_interpolate_exempt,
    "scatter": _check_scatter_or_index_domain_exempt,
    "scatter_": _check_scatter_or_index_domain_exempt,
    "scatter_add": _check_index_domain_degenerate,
    "scatter_add_": _check_index_domain_degenerate,
    "scatteradd": _check_index_domain_degenerate,
    "embedding": _check_index_domain_degenerate,
    "gather": _check_index_domain_degenerate,
    "index_select": _check_index_domain_degenerate,
    "cross_entropy": _check_index_domain_degenerate,
    "where": _check_where_exempt,
    "maskedfill": _check_masked_fill_exempt,
    "masked_fill": _check_masked_fill_exempt,
    "masked_fill_": _check_masked_fill_exempt,
    "batch_norm": _check_norm_running_stat_exempt,
    "instance_norm": _check_norm_running_stat_exempt,
}


# ---------------------------------------------------------------------------
# Structural position helper (used by core.py)
# ---------------------------------------------------------------------------


def perturbed_layer_at_structural_position(
    self: "Trace",
    layer: Op,
    layers_to_perturb: list[str],
    exempt_positions: set[int],
) -> bool:
    """Check if the perturbed layer occupies a structural arg position.

    The decision is based on ``parent_arg_positions`` identity, not tensor value
    equality. Identical tensor values from another parent must not be enough to
    classify the perturbed parent as structural.
    """
    del self
    if len(layers_to_perturb) != 1:
        return False
    perturbed_label = layers_to_perturb[0]
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    func_name = getattr(layer, "func_name", None)
    alias_by_position = (
        STRUCTURAL_ARG_KWARG_ALIASES.get(func_name, {}) if isinstance(func_name, str) else {}
    )
    recorded_args = parent_arg_positions.get("args", {}) or {}
    for pos in exempt_positions:
        if recorded_args.get(pos) == perturbed_label:
            return True
        # A CONTAINER at a structural position: its members carry zipped/nested
        # tuple keys ``(pos, j)`` (e.g. the ``_foreach_copy_`` destination
        # list). Value-irrelevance of the position covers its members; slots at
        # other positions are unaffected.
        for key, label in recorded_args.items():
            if isinstance(key, tuple) and key and key[0] == pos and label == perturbed_label:
                return True
        aliases = alias_by_position.get(pos, set())
        for alias in aliases:
            if parent_arg_positions.get("kwargs", {}).get(alias) == perturbed_label:
                return True
    return False


def _binary_extrema_nonperturbed_arg_dominates(
    func_name: str,
    args: tuple[Any, ...],
    layer: Op,
    layers_to_perturb: list[str],
) -> bool:
    """Return whether a binary extrema output ignores the perturbed operand.

    Parameters
    ----------
    func_name:
        Captured extrema function name.
    args:
        Saved positional arguments.
    layer:
        Captured op being checked.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        True when the non-perturbed tensor dominates the perturbed tensor at
        every output element, proving the perturbed operand cannot affect the
        selected extrema values.
    """

    if len(layers_to_perturb) != 1 or len(args) < 2:
        return False
    if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
        return False
    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if perturbed_positions == {0}:
        perturbed, other = args[0], args[1]
    elif perturbed_positions == {1}:
        perturbed, other = args[1], args[0]
    else:
        return False
    try:
        if func_name in ("max", "maximum"):
            return bool(torch.all(other >= perturbed).item())
        if func_name in ("min", "minimum"):
            return bool(torch.all(other <= perturbed).item())
    except RuntimeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Posthoc perturbation check — excuses failures after execution.
# These handle genuinely dynamic/value-dependent cases that can't be
# determined before running the function.
# ---------------------------------------------------------------------------


def posthoc_perturb_check(
    self: "Trace",
    layer_to_validate_parents_for: Op,
    layers_to_perturb: list[str],
    verbose: bool = False,
) -> PosthocPerturbDecision:
    """Post-hoc exemption check: called when perturbation did NOT change the output.

    This function runs AFTER execution, handling dynamic cases that cannot be
    determined pre-execution.  It checks a cascade of valid excuses:

    1. **Bool output** -- discrete output, perturbation may not flip it.
    2. **Discrete index outputs** (topk, sort) -- indices are order-dependent,
       not value-dependent.
    3. **Type casting** (``to()``) -- value irrelevant when casting type.
    4. **Full overwrite** (__setitem__ with same-shape replacement).
    5. **Structural output templates** for *_like/meshgrid/broadcast_tensors.
    6. **Narrow value proofs** such as a non-perturbed multiplicative zero
       operand (plain ``mul``/``multiply`` or the ``addcmul`` fused form) or a
       dominated binary extrema operand.

    Returns a structured decision. ``exempt=False`` means replay-level probes
    may still provide diagnostic evidence before validation fails.
    """
    args = layer_to_validate_parents_for.saved_args or ()

    decision = _posthoc_discrete_output_decision(layer_to_validate_parents_for, layers_to_perturb)
    if decision.exempt:
        return decision
    decision = _posthoc_structural_output_decision(
        layer_to_validate_parents_for, args, layers_to_perturb
    )
    if decision.exempt:
        return decision
    decision = _posthoc_overwrite_decision(layer_to_validate_parents_for, layers_to_perturb, args)
    if decision.exempt:
        return decision
    decision = _posthoc_value_proof_decision(layer_to_validate_parents_for, layers_to_perturb, args)
    if decision.exempt:
        return decision

    del self, layers_to_perturb, verbose
    return PosthocPerturbDecision(False, "no_posthoc_exemption")


#: Elementwise comparison spellings whose bool outputs admit the
#: threshold-straddle probe (R08): substituting the perturbed operand with
#: the comparand itself and its two adjacent values MUST flip some element
#: of any genuine two-operand comparison, so an output pinned to the saved
#: value under all three substitutions proves the recorded parent has no
#: value influence.
_ELEMENTWISE_COMPARISON_FUNCS = frozenset(
    {
        "gt",
        "greater",
        "lt",
        "less",
        "ge",
        "greater_equal",
        "le",
        "less_equal",
        "eq",
        "ne",
        "not_equal",
        "__gt__",
        "__lt__",
        "__ge__",
        "__le__",
        "__eq__",
        "__ne__",
    }
)


def _bool_comparison_straddle_probe(layer: Op, layers_to_perturb: list[str]) -> bool | None:
    """Probe a bool comparison by straddling its comparand (R08).

    Re-executes the comparison with the perturbed operand replaced by the
    comparand itself and its two adjacent representable values. For any
    genuine elementwise comparison these three substitutions produce at
    least two distinct outputs, so:

    * an output that DIFFERS from the saved output under any substitution
      proves the recorded edge transmits value (the original perturbation
      magnitude simply never crossed the threshold);
    * an output pinned exactly to the saved value under ALL THREE proves the
      recorded parent has no value influence on this op — the spurious-edge
      class the blanket bool exemption used to bless.

    Parameters
    ----------
    layer:
        Bool-output comparison op whose unchanged perturbation is being
        classified.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool | None
        ``True`` (edge transmits value), ``False`` (provably no influence),
        or ``None`` when the probe cannot run (non-comparison func, kwargs
        or multi-slot/self-comparison operands, non-finite comparand,
        execution failure) — the caller then falls back to the disclosed
        heuristic exemption.
    """

    func = getattr(layer, "func", None)
    if func is None or layer.func_name not in _ELEMENTWISE_COMPARISON_FUNCS:
        return None
    args: tuple[Any, ...] = tuple(layer.saved_args or ())
    kwargs = dict(getattr(layer, "saved_kwargs", None) or {})
    if kwargs or len(args) != 2:
        return None
    positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if positions != {0} and positions != {1}:
        # Multi-slot (self-comparison) or unrecoverable positions: the
        # straddle would move both operands together and prove nothing.
        return None
    parent_index = next(iter(positions))
    parent_saved = args[parent_index]
    other = args[1 - parent_index]
    saved_output = layer.out
    if not isinstance(parent_saved, torch.Tensor) or not isinstance(saved_output, torch.Tensor):
        return None
    try:
        with torch.no_grad():
            if isinstance(other, torch.Tensor):
                base = other.detach().to(dtype=parent_saved.dtype).broadcast_to(parent_saved.shape)
            elif isinstance(other, (bool, int, float)):
                base = torch.full_like(parent_saved, other)
            else:
                return None
            if base.dtype == torch.bool or base.is_complex():
                return None
            if base.is_floating_point():
                if not bool(torch.isfinite(base).all()):
                    return None
                below = torch.nextafter(base, torch.full_like(base, float("-inf")))
                above = torch.nextafter(base, torch.full_like(base, float("inf")))
            else:
                one = torch.ones_like(base)
                below = base - one
                above = base + one
            from ..utils.tensor_utils import tensor_nanequal

            for substitute in (base, below, above):
                probe_args = list(args)
                probe_args[parent_index] = substitute
                probe_output = func(*probe_args)
                if not isinstance(probe_output, torch.Tensor):
                    return None
                if not tensor_nanequal(probe_output, saved_output, allow_tolerance=False):
                    return True
    except Exception:
        return None
    return False


def _bool_probe_battery(parent_saved: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return dtype-directed substitutes chosen to flip a value-transmitting predicate.

    Sign / zero / extreme / non-finite / inversion coverage: ``isnan``/``isfinite``
    flip on the non-finite rows, ``logical_*`` and ``any``/``all`` on the
    all-False/all-True rows, ``signbit``-style predicates on the sign rows, and
    magnitude thresholds on the extreme rows.
    """

    if parent_saved.dtype == torch.bool:
        return (
            torch.zeros_like(parent_saved),
            torch.ones_like(parent_saved),
            ~parent_saved,
        )
    if parent_saved.is_complex():
        return (
            torch.zeros_like(parent_saved),
            torch.ones_like(parent_saved),
            torch.full_like(parent_saved, complex(0.0, 1.0)),
            torch.full_like(parent_saved, complex(float("nan"), 0.0)),
        )
    if parent_saved.is_floating_point():
        return (
            torch.zeros_like(parent_saved),
            torch.ones_like(parent_saved),
            -torch.ones_like(parent_saved),
            -parent_saved,
            torch.full_like(parent_saved, float("nan")),
            torch.full_like(parent_saved, float("inf")),
            torch.full_like(parent_saved, float("-inf")),
            torch.full_like(parent_saved, torch.finfo(parent_saved.dtype).max / 2),
        )
    info = torch.iinfo(parent_saved.dtype)
    substitutes = [
        torch.zeros_like(parent_saved),
        torch.ones_like(parent_saved),
        torch.full_like(parent_saved, info.max),
        torch.full_like(parent_saved, info.min),
    ]
    if info.min < 0:
        substitutes.append(-torch.ones_like(parent_saved))
    return tuple(substitutes)


def _bool_predicate_influence_probe(layer: Op, layers_to_perturb: list[str]) -> bool | None:
    """Probe a NON-comparison bool-output op with a substitution battery (R08-2).

    The comparison family gets the threshold-straddle probe; the rest of the
    bool universe (``logical_*``, ``isnan``/``isfinite``, ``any``/``all``,
    ``bitwise_*`` masks) used to fall back to the blanket
    ``discrete_bool_output`` pass with no evidence, so a capture bug that
    drops or invents a parent edge on a predicate op was unfalsifiable by
    perturbation (round-6 armed proof: a frozen replay callable settled
    ``exempted`` while its comparison sibling failed). Re-execute the op with
    the perturbed slot substituted by :func:`_bool_probe_battery`; any flip
    proves the recorded edge transmits value, and a battery-wide pin proves
    the recorded parent has no value influence -- the caller then falls
    through to the ``perturbation_insensitive`` failure exactly like the
    comparison family.

    Returns
    -------
    bool | None
        ``True`` (edge transmits value), ``False`` (provably no influence),
        or ``None`` when the probe cannot run (comparison func -- the
        straddle owns those -- kwargs, multi-slot operands, non-tensor
        parent/output, execution failure); the caller then falls back to the
        disclosed heuristic exemption.
    """

    func = getattr(layer, "func", None)
    if func is None or layer.func_name in _ELEMENTWISE_COMPARISON_FUNCS:
        return None
    args: tuple[Any, ...] = tuple(layer.saved_args or ())
    kwargs = dict(getattr(layer, "saved_kwargs", None) or {})
    if kwargs or not args:
        return None
    positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if len(positions) != 1:
        return None
    parent_index = next(iter(positions))
    if not (0 <= parent_index < len(args)):
        return None
    parent_saved = args[parent_index]
    saved_output = layer.out
    if not isinstance(parent_saved, torch.Tensor) or not isinstance(saved_output, torch.Tensor):
        return None
    try:
        with torch.no_grad():
            from ..utils.tensor_utils import tensor_nanequal

            for substitute in _bool_probe_battery(parent_saved):
                # CLONE every retained operand per battery run (r8 R08, fable
                # MH): the bool universe includes IN-PLACE ops
                # (``logical_and_``, ``bitwise_*_`` masks), and executing one
                # against the record's retained ``saved_args`` MUTATED the
                # capture evidence itself -- the probe then returned a verdict
                # computed on the corrupted operand, and every LATER
                # replay/comparison read poisoned payloads. The substitute is
                # already a fresh tensor; each raw slot clones fresh per run
                # so an in-place probe can only ever write scratch.
                probe_args = [
                    item.clone() if isinstance(item, torch.Tensor) else item for item in args
                ]
                probe_args[parent_index] = substitute
                probe_output = func(*probe_args)
                if not isinstance(probe_output, torch.Tensor):
                    return None
                if not tensor_nanequal(probe_output, saved_output, allow_tolerance=False):
                    return True
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        # Typed catch (r8 R08, sol fault-injection): these are the classes a
        # legitimate op raises when it rejects a battery substitute
        # (dtype/shape/value refusals, unimplemented substitute dtypes) --
        # "probe cannot run", which reverts to the DISCLOSED heuristic
        # exemption (status quo ante). The old blanket ``except Exception``
        # also swallowed genuine capture-bug crashes (a poisoned replay
        # callable, a torchlens internal error) into that same evidence-free
        # pass, disarming exactly the tripwire this probe strengthens; those
        # now propagate.
        return None
    return False


def _posthoc_discrete_output_decision(
    layer: Op, layers_to_perturb: list[str]
) -> PosthocPerturbDecision:
    """Return the explicit posthoc decision for discrete output tensors.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for bool/index outputs, otherwise a non-exempt result.
    """

    if layer.dtype == torch.bool:
        # R08: the bool exemption is no longer a blanket pass. For the
        # elementwise-comparison family the threshold-straddle probe settles
        # it with evidence; a probe-proven no-influence edge falls through
        # to the perturbation_insensitive failure (the remaining posthoc
        # excuses still get their chance).
        probe = _bool_comparison_straddle_probe(layer, layers_to_perturb)
        if probe is False:
            return PosthocPerturbDecision(False, "bool_comparison_no_value_influence")
        if probe is True:
            return PosthocPerturbDecision(
                True,
                "discrete_bool_output",
                "threshold-straddle probe flipped the output: the recorded edge "
                "transmits value; the original perturbation magnitude did not "
                "cross the comparison threshold",
            )
        # R08-2: the straddle covers comparisons only; the rest of the bool
        # universe gets the substitution-battery probe so a provably spurious
        # edge on a logical/predicate op falls through to the failure instead
        # of the old evidence-free blanket pass.
        predicate_probe = _bool_predicate_influence_probe(layer, layers_to_perturb)
        if predicate_probe is False:
            return PosthocPerturbDecision(False, "bool_predicate_no_value_influence")
        if predicate_probe is True:
            return PosthocPerturbDecision(
                True,
                "discrete_bool_output",
                "substitution battery flipped the output: the recorded edge "
                "transmits value; the original perturbation magnitude did not "
                "cross the predicate's decision boundary",
            )
        return PosthocPerturbDecision(True, "discrete_bool_output")
    if layer.func_name in ("topk", "sort", "max", "min", "multinomial") and layer.dtype in (
        torch.int,
        torch.long,
        torch.int32,
        torch.int64,
    ):
        # multinomial draws ONE discrete category index from a normalized
        # distribution via a single random number against the CDF; whether a
        # perturbed distribution moves the draw across a bucket boundary
        # depends on the specific draw, not on whether the recorded parent
        # edge transmits value. By the time this decision runs, the unit-step
        # and geometric-magnitude perturbation retries (same ladder the
        # bool-comparison straddle probe rides) have already tried a wide
        # range of magnitudes against the saved output and none flipped the
        # sampled index -- same discrete-order-dependent shape as the
        # topk/sort/max/min index family above, not a continuous value this
        # validation pass can promise sensitivity for.
        return PosthocPerturbDecision(True, "discrete_index_output")
    if _check_one_arg_where_index_exempt(layer):
        return PosthocPerturbDecision(True, "discrete_index_output")
    return PosthocPerturbDecision(False, "not_discrete_output")


def _perturbed_parents_only_occupy_template_slot(
    layer: Op,
    layers_to_perturb: list[str],
    template_arg_roots: tuple[int, ...] = (0,),
    template_kwarg_names: tuple[str, ...] = ("input",),
) -> bool:
    """Return whether EVERY perturbed parent occupies only the template slot.

    A structural-template exemption is only sound for the TEMPLATE argument:
    its values never flow into the output, only its shape/dtype/device do.
    For the ``*_like`` family that is ``args[0]`` / ``input=`` (the F2
    tightening); for ``to(other)`` it is ``args[1]`` / ``other=`` (the H3
    tightening -- the perturbed data SOURCE of a cast is a genuine value
    dependency, and an unchanged output there is a dropped substitution, not
    structure). Any other parent slot must NOT be excused as structural.
    Missing position metadata fails closed.

    Parameters
    ----------
    layer:
        Captured op carrying a structural template argument.
    layers_to_perturb:
        Parent labels currently being perturbed.
    template_arg_roots:
        Positional root indices of the template slot.
    template_kwarg_names:
        Keyword spellings of the template slot.

    Returns
    -------
    bool
        True only when every perturbed parent's every registered position is
        the template slot.
    """

    if not layers_to_perturb:
        return False
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    args_map = parent_arg_positions.get("args", {}) or {}
    kwargs_map = parent_arg_positions.get("kwargs", {}) or {}
    for perturbed_label in layers_to_perturb:
        arg_keys = [key for key, label in args_map.items() if label == perturbed_label]
        kwarg_names = [name for name, label in kwargs_map.items() if label == perturbed_label]
        if not arg_keys and not kwarg_names:
            return False
        for key in arg_keys:
            root = key[0] if isinstance(key, tuple) and key else key
            if root not in template_arg_roots:
                return False
        for name in kwarg_names:
            if name not in template_kwarg_names:
                return False
    return True


def _posthoc_structural_output_decision(
    layer: Op,
    args: tuple[Any, ...],
    layers_to_perturb: list[str],
) -> PosthocPerturbDecision:
    """Return posthoc decisions for structural output-template operations.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.
    args:
        Saved positional arguments for ``layer``.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for structural cases, otherwise a non-exempt result.
    """

    if (
        layer.func_name == "to"
        and len(args) > 1
        and isinstance(args[1], torch.Tensor)
        and _perturbed_parents_only_occupy_template_slot(
            layer,
            layers_to_perturb,
            template_arg_roots=(1,),
            template_kwarg_names=("other",),
        )
    ):
        # H3 tightening: only the TEMPLATE tensor (args[1] / other=) is
        # structural -- solely its dtype/device flow into the output. A
        # perturbed data SOURCE (args[0]) whose replay output stays unchanged
        # is a dropped substitution and must fall through to the failure path.
        return PosthocPerturbDecision(True, "type_template_output")
    if _integer_cast_quantization_applies(layer, args):
        return PosthocPerturbDecision(
            True,
            "integer_cast_quantization",
            "float->integer cast output changes only when the perturbation crosses an "
            "integer boundary; a same-bucket perturbation is quantization, not a "
            "dropped dependency (arg-identity logging still guards the mapping)",
        )
    if layer.func_name in [
        "meshgrid",
        "broadcast_tensors",
        "broadcasttensors",
    ] and _perturbed_parents_are_zipped_siblings(layer, layers_to_perturb):
        # R08-2 narrowing: only a CROSS-member zipped-sibling perturbation is
        # structural here. An output's OWN input staying insensitive is a
        # dropped/misattributed value edge and must fall through to the
        # failure path.
        return PosthocPerturbDecision(True, "structural_output_template")
    if layer.func_name in [
        "full_like",
        "zeros_like",
        "ones_like",
        "empty_like",
        "rand_like",
        "randn_like",
    ] and _perturbed_parents_only_occupy_template_slot(layer, layers_to_perturb):
        # F2 tightening: only the TEMPLATE parent (args[0] / input=) is
        # structural. A perturbed runtime value parent -- e.g. a ``full_like``
        # fill_value tensor -- whose replay output stays unchanged is a missed
        # or broken dependency and must fall through to the failure path.
        return PosthocPerturbDecision(True, "structural_output_template")
    if layer.func_name == "bernoulli" and "p" in layer.saved_kwargs:
        return PosthocPerturbDecision(True, "rng_probability_template")
    if layer.func_name == "bernoulli_" and _perturbed_parents_only_occupy_template_slot(
        layer, layers_to_perturb
    ):
        # bernoulli_ overwrites EVERY destination element with fresh draws --
        # Bernoulli(0.5) for the bare form (self's values are IGNORED;
        # zeros.bernoulli_() produces ones) and Bernoulli(p) for the explicit
        # form -- so only the destination's shape/dtype/device flow into the
        # output: a template, exactly like args[0] of the *_like family. A
        # probability edge (out-of-place bernoulli slot 0, or bernoulli_'s
        # slot 1 / p=) is NOT exempted here: it is a genuine value dependency
        # validated by the complement-probability perturbation (deephunt L17).
        return PosthocPerturbDecision(
            True,
            "rng_probability_template",
            "bernoulli_ overwrites every destination element with fresh draws; "
            "only the destination's shape/dtype/device flow into the output",
        )
    if _unique_disabled_auxiliary_output(layer):
        return PosthocPerturbDecision(
            True,
            "structural_output_template",
            "disabled unique auxiliary output is empty by construction",
        )
    if _pad_packed_sequence_lengths_output(layer):
        return PosthocPerturbDecision(
            True,
            "structural_output_template",
            "pad_packed_sequence lengths output is structural metadata",
        )
    if _pack_padded_sequence_metadata_output(layer):
        return PosthocPerturbDecision(
            True,
            "structural_output_template",
            "pack_padded_sequence batch metadata output is structural",
        )
    if _empty_getitem_output(layer):
        return PosthocPerturbDecision(
            True,
            "structural_output_template",
            "getitem output is empty, so no selected data values can change",
        )
    return PosthocPerturbDecision(False, "not_structural_output")


def _unique_disabled_auxiliary_output(layer: Op) -> bool:
    """Return whether a unique op output is a disabled auxiliary tensor.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.

    Returns
    -------
    bool
        True when this is an empty ``unique``/``_unique2`` auxiliary output
        produced because inverse/count outputs were not requested.
    """

    output = getattr(layer, "out", None)
    return (
        layer.func_name in {"unique", "_unique2"}
        and isinstance(output, torch.Tensor)
        and output.numel() == 0
        and not bool((getattr(layer, "saved_kwargs", {}) or {}).get("return_inverse", False))
        and not bool((getattr(layer, "saved_kwargs", {}) or {}).get("return_counts", False))
    )


def _pad_packed_sequence_lengths_output(layer: Op) -> bool:
    """Return whether a pad_packed_sequence output is structural lengths metadata.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.

    Returns
    -------
    bool
        True when the output is the integer lengths tensor from
        ``pad_packed_sequence``.
    """

    output = getattr(layer, "out", None)
    return (
        layer.func_name == "_pad_packed_sequence"
        and isinstance(output, torch.Tensor)
        and output.dtype in {torch.int, torch.long, torch.int32, torch.int64}
    )


def _pack_padded_sequence_metadata_output(layer: Op) -> bool:
    """Return whether a pack_padded_sequence output is structural metadata.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.

    Returns
    -------
    bool
        True when the output is an integer metadata tensor from
        ``pack_padded_sequence``.
    """

    output = getattr(layer, "out", None)
    return (
        layer.func_name == "_pack_padded_sequence"
        and isinstance(output, torch.Tensor)
        and output.dtype in {torch.int, torch.long, torch.int32, torch.int64}
    )


def _empty_getitem_output(layer: Op) -> bool:
    """Return whether ``__getitem__`` produced an empty tensor output.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.

    Returns
    -------
    bool
        True when the output is an empty tensor selected by ``__getitem__``.
    """

    output = getattr(layer, "out", None)
    return (
        layer.func_name == "__getitem__"
        and isinstance(output, torch.Tensor)
        and output.numel() == 0
    )


_INTEGER_CAST_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _integer_cast_quantization_applies(layer: Op, args: tuple[Any, ...]) -> bool:
    """Return whether an unchanged perturbation is float->integer quantization.

    NARROW predicate for ``Tensor.to(integer_dtype)``: the perturbation ran the
    REAL func with the perturbed floating parent substituted (that is how the
    insensitivity was observed), so an unchanged integer output proves the
    perturbation stayed inside the same integer buckets -- a quantization
    effect, never a dropped-dependency mask. Float->float and other casts stay
    strict, where value insensitivity would indicate a real capture bug.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    bool
        Whether the integer-cast quantization exemption applies.
    """

    if layer.func_name != "to" or len(args) < 2:
        return False
    if args[1] not in _INTEGER_CAST_DTYPES:
        return False
    source = args[0]
    return isinstance(source, torch.Tensor) and torch.is_floating_point(source)


def _posthoc_overwrite_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return posthoc decisions for overwrite-style destination parents.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for proved overwrite cases, otherwise non-exempt.
    """

    if (
        layer.func_name == "__setitem__"
        and _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0)
        and len(args) > 2
        and isinstance(args[0], torch.Tensor)
        and _setitem_destination_coverage_is_total(args[0], args)
    ):
        reason = (
            "full_destination_overwrite"
            if isinstance(args[2], torch.Tensor)
            else "scalar_destination_overwrite"
        )
        return PosthocPerturbDecision(True, reason)
    if layer.func_name in INPLACE_DESTINATION_WRITE_FUNCS and (
        _perturbed_parent_is_uninitialized_setitem_dest(layer, layers_to_perturb)
    ):
        return PosthocPerturbDecision(True, "uninitialized_destination_overwrite")
    return PosthocPerturbDecision(False, "not_overwrite")


def _posthoc_value_proof_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return explicit value-proof decisions that are not generic probes.

    Parameters
    ----------
    layer:
        Operation whose unchanged perturbation output is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for narrow proved cases, otherwise non-exempt.
    """

    if layer.func_name in _INDEX_DOMAIN_ARG_SPECS:
        decision = _index_domain_value_irrelevance_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in _MULTIPLICATIVE_ANNIHILATOR_FUNC_NAMES and len(args) > 1:
        decision = _multiplicative_zero_annihilator_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
        decision = _locally_constant_nan_multiplication_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in _ADDCMUL_FUNC_NAMES and len(args) > 2:
        decision = _addcmul_zero_annihilator_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in {"__matmul__", "matmul", "mm", "bmm"} and len(args) > 1:
        decision = _matmul_zero_annihilator_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name == "scaled_dot_product_attention" and len(args) > 2:
        decision = _sdpa_zero_query_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in {"softmax", "_softmax"} and len(args) > 0:
        decision = _softmax_singleton_dim_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name == "linear" and len(args) > 1:
        decision = _linear_zero_weight_input_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in {"conv1d", "conv2d", "conv3d"} and len(args) > 1:
        decision = _conv_zero_weight_input_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in {"__add__", "add", "__radd__", "__sub__", "sub", "__rsub__"}:
        decision = _locally_constant_nonfinite_addition_decision(layer, layers_to_perturb, args)
        if decision.exempt:
            return decision
    if layer.func_name in ("max", "min", "maximum", "minimum") and len(args) > 1:
        if _binary_extrema_nonperturbed_arg_dominates(
            layer.func_name, args, layer, layers_to_perturb
        ):
            return PosthocPerturbDecision(
                True,
                "binary_extrema_dominated",
                "non-perturbed extrema operand dominates every output element",
            )
    if layer.func_name in ("remainder", "fmod", "__mod__") and len(args) > 1:
        dividend, divisor = args[:2]
        if isinstance(dividend, torch.Tensor) and isinstance(divisor, torch.Tensor):
            arg_positions = layer.parent_arg_positions.get("args", {})
            perturbed_label = layers_to_perturb[0]
            if arg_positions.get(1) == perturbed_label and torch.equal(layer.out, dividend):
                return PosthocPerturbDecision(
                    True,
                    "mod_divisor_irrelevant",
                    "saved dividend is already the output, so divisor value is irrelevant",
                )
    if layer.func_name == "max" and len(args) > 0 and not torch.is_floating_point(args[0]):
        return PosthocPerturbDecision(
            True,
            "discrete_value_output",
            "non-floating max output is a discrete value result",
        )
    return PosthocPerturbDecision(False, "not_value_proved")


def _tensor_or_number_is_constant(value: Any) -> bool:
    """Return whether ``value`` is a scalar or a tensor with one constant value.

    Parameters
    ----------
    value:
        Saved scatter source argument (tensor or Python scalar).

    Returns
    -------
    bool
        True when every element provably equals one constant.
    """

    if isinstance(value, Number):
        return True
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return False
    first = value.reshape(-1)[0]
    return bool(torch.eq(value, first).all().item())


def _tensor_constant_along_dim(tensor: torch.Tensor, dim: int) -> bool:
    """Return whether ``tensor`` holds identical values at every index of ``dim``.

    Parameters
    ----------
    tensor:
        Source tensor being indexed.
    dim:
        Normalized dimension the index selects along.

    Returns
    -------
    bool
        True when swapping any two positions along ``dim`` provably leaves the
        tensor unchanged.
    """

    if tensor.numel() == 0 or tensor.shape[dim] == 0:
        return False
    reference = tensor.select(dim, 0).unsqueeze(dim)
    return bool(torch.eq(tensor, reference).all().item())


def _index_domain_value_irrelevance_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for an index parent that provably cannot matter.

    These are the ONLY excuses for an in-domain index rotation leaving the
    output unchanged (F2 tightening); each is a mathematical irrelevance proof
    read from the saved co-arguments, so a genuinely missed/broken index
    dependency (where the co-arguments DO vary) still falls through to
    ``perturbation_insensitive``:

    - ``embedding``: every weight row is identical, so any index selects the
      same values.
    - ``gather``/``index_select``: the source is constant along the indexed
      dim, so any in-range index reads the same values.
    - ``cross_entropy``: the logits are constant along the class dim and no
      per-class ``weight`` reweights the reduction, so every target picks an
      equal-probability class.
    - ``scatter``/``scatter_``: the source value is one constant and the saved
      index fully covers the destination dim, so every slot ends at that
      constant under any valid index permutation.
    - ``scatter_add`` family: additionally requires exactly-once coverage
      (``index.shape[dim] == dest.shape[dim]``), because duplicate writes make
      per-slot sums depend on index multiplicities.

    Parameters
    ----------
    layer:
        Index-consuming op whose unchanged perturbed replay is being
        classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when irrelevance is proved, otherwise non-exempt.
    """

    not_proved = PosthocPerturbDecision(False, "not_index_domain_value_irrelevant")
    if len(layers_to_perturb) != 1:
        return not_proved
    if not _parent_is_index_domain_arg(layer, layers_to_perturb[0]):
        return not_proved
    func_name = layer.func_name
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    if func_name == "embedding":
        weight = kwargs.get("weight", args[0] if args else None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 2 and weight.shape[0] > 0:
            if bool(torch.eq(weight, weight[0:1]).all().item()):
                return PosthocPerturbDecision(
                    True,
                    "index_domain_value_irrelevant",
                    "every embedding weight row is identical, so any valid index "
                    "selects the same values",
                )
        return not_proved
    if func_name == "cross_entropy":
        logits = kwargs.get("input", args[0] if args else None)
        if kwargs.get("weight") is not None or (len(args) > 2 and args[2] is not None):
            return not_proved
        if isinstance(logits, torch.Tensor) and logits.ndim >= 1:
            class_dim = 1 if logits.ndim >= 2 else 0
            if _tensor_constant_along_dim(logits, class_dim):
                return PosthocPerturbDecision(
                    True,
                    "index_domain_value_irrelevant",
                    "logits are constant along the class dim with no per-class "
                    "weight, so every target class yields the same loss",
                )
        return not_proved
    if func_name in ("gather", "index_select"):
        source = kwargs.get("input", args[0] if args else None)
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if isinstance(source, torch.Tensor) and isinstance(dim, int):
            if dim < 0:
                dim = source.ndim + dim
            if 0 <= dim < source.ndim and _tensor_constant_along_dim(source, dim):
                return PosthocPerturbDecision(
                    True,
                    "index_domain_value_irrelevant",
                    "the indexed source is constant along the gather dim, so any "
                    "in-range index reads the same values",
                )
        return not_proved
    scatter_components = _get_scatter_destination_dim_index(layer)
    if scatter_components is None:
        return not_proved
    dest, dim, index = scatter_components
    source = kwargs.get("src", kwargs.get("value", args[3] if len(args) > 3 else None))
    if source is None or not _tensor_or_number_is_constant(source):
        return not_proved
    if not _scatter_index_fully_overwrites_dim(dest, dim, index):
        return not_proved
    if func_name in ("scatter_add", "scatter_add_", "scatteradd") and (
        index.shape[dim] != dest.shape[dim]
    ):
        return not_proved
    return PosthocPerturbDecision(
        True,
        "index_domain_value_irrelevant",
        "a constant scatter source with full destination-dim coverage writes "
        "the same value to every slot under any valid index",
    )


# Every elementwise-multiplication spelling whose output is provably zero when
# either operand is zero: dunder, functional, reverse-dunder, and the in-place
# family. The annihilator proof itself is spelling-independent (``x * 0 == 0``
# holds for all of them, including in-place ``mul_``/``__imul__`` where the
# saved receiver snapshot is the pre-call value), so listing a spelling here
# NEVER weakens the perturbation net: the exemption still requires the
# NON-perturbed co-operand to be provably all-zero. Omitting a spelling was
# the round-26 W3-5/F2 false-positive bug (``x + 0.0 * x.sum()`` via
# ``__rmul__`` and ``view.mul_(0.0)`` wrongly failed ``perturbation_insensitive``
# while the zero-TENSOR twin passed).
_MULTIPLICATIVE_ANNIHILATOR_FUNC_NAMES = frozenset(
    {
        "__mul__",
        "mul",
        "__rmul__",
        "__imul__",
        "mul_",
        "multiply",
        "multiply_",
    }
)


#: ``addcmul``/``addcmul_`` compute ``input + value * tensor1 * tensor2``: a
#: fused multiply-add whose multiplied PAIR sits at positions {1, 2} (not the
#: plain-``mul`` {0, 1} the annihilator proof above assumes), with position 0
#: the additive ``input`` that is never multiplied. Kept as its own registry
#: and decision function rather than folded into
#: ``_MULTIPLICATIVE_ANNIHILATOR_FUNC_NAMES`` so the shared decision's
#: position assumption stays correct for every spelling it already covers.
_ADDCMUL_FUNC_NAMES = frozenset({"addcmul", "addcmul_"})


def _addcmul_zero_annihilator_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for ``addcmul``'s multiplied-pair zero annihilator.

    ``addcmul(input, tensor1, tensor2, value=1)`` computes
    ``input + value * tensor1 * tensor2``. A perturbed parent at the
    ``tensor1``/``tensor2`` slot (args[1]/args[2]) can never reach the output
    when the SIBLING multiplied operand is provably all-zero, or when
    ``value`` itself is exactly zero -- the same proof
    ``_multiplicative_zero_annihilator_decision`` already makes for plain
    ``mul``/``multiply``, extended to addcmul's three-operand fused form. A
    perturbed ``input`` (args[0]) is the additive term, never multiplied, so
    it is out of scope here and stays strict (returns non-exempt).

    timm's ConvNeXtV2 Global Response Norm layer initializes its gain to
    ``nn.Parameter(torch.zeros(...))`` and feeds it to ``addcmul`` as
    ``tensor1``, so this is the random-init shape of a real architecture, not
    a synthetic edge case.

    Parameters
    ----------
    layer:
        ``addcmul``/``addcmul_`` op whose unchanged perturbed replay is being
        classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the sibling multiplied operand (or ``value``) is
        provably zero, otherwise non-exempt.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_addcmul_zero_annihilator")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(1) == perturbed_label:
        other_position = 2
    elif arg_positions.get(2) == perturbed_label:
        other_position = 1
    else:
        return PosthocPerturbDecision(False, "not_addcmul_zero_annihilator")

    saved_kwargs = getattr(layer, "saved_kwargs", None) or {}
    value: Any = saved_kwargs.get("value", 1)
    if "value" not in saved_kwargs and len(args) > 3:
        value = args[3]
    if isinstance(value, Number) and value == 0:
        return PosthocPerturbDecision(
            True,
            "addcmul_zero_annihilator",
            "addcmul value=0 annihilates both multiplied operands",
        )

    if len(args) <= other_position:
        return PosthocPerturbDecision(False, "not_addcmul_zero_annihilator")
    if _is_all_zero_value(args[other_position]):
        return PosthocPerturbDecision(
            True,
            "addcmul_zero_annihilator",
            f"non-perturbed addcmul operand at args[{other_position}] is all zero",
        )
    return PosthocPerturbDecision(False, "not_addcmul_zero_annihilator")


def _multiplicative_zero_annihilator_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for multiplication by a saved zero operand.

    Parameters
    ----------
    layer:
        Multiplication op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the non-perturbed operand is provably zero,
        otherwise non-exempt.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_multiplicative_zero_annihilator")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) == perturbed_label:
        other_position = 1
    elif arg_positions.get(1) == perturbed_label:
        other_position = 0
    else:
        return PosthocPerturbDecision(False, "not_multiplicative_zero_annihilator")
    if len(args) <= other_position:
        return PosthocPerturbDecision(False, "not_multiplicative_zero_annihilator")
    if _is_all_zero_value(args[other_position]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            f"non-perturbed multiplication operand at args[{other_position}] is all zero",
        )
    return PosthocPerturbDecision(False, "not_multiplicative_zero_annihilator")


def _sdpa_zero_query_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for sdpa's key operand under a zero query.

    ``scaled_dot_product_attention(query, key, value)`` computes
    ``softmax(query @ key.transpose(-2, -1) / sqrt(head_dim)) @ value``. When
    ``query`` is provably all-zero, every row of ``query @ key.transpose(-2, -1)``
    is exactly zero regardless of ``key``, so the softmax is exactly uniform
    regardless of ``key`` and the output reduces to an unweighted average of
    ``value`` -- the perturbed ``key`` operand provably cannot influence the
    output, the same zero-annihilator shape already proved for
    ``mul``/``addcmul``/batch_norm's weight, applied here to sdpa's query
    operand instead.

    Several ViT/CaiT-style class-attention implementations (including
    menagerie's compact CaiT reimplementation) zero-initialize the class
    token and rely on ``nn.MultiheadAttention``'s zero-initialized
    ``in_proj_bias``, so the FIRST class-attention block's query is exactly
    zero at random init for any input, not a synthetic edge case.

    A perturbed ``query`` (args[0]) or ``value`` (args[2]) is out of scope
    here and stays strict: both genuinely influence sdpa's output even when
    query is zero (perturbing query away from zero changes the attention
    weights; perturbing value always changes the weighted/unweighted
    average).

    Parameters
    ----------
    layer:
        ``scaled_dot_product_attention`` op whose unchanged perturbed replay
        is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer`` (``query, key, value, ...``).

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the perturbed parent is the key operand and the
        saved query is provably all-zero, otherwise non-exempt.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_sdpa_zero_query")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(1) != perturbed_label:
        return PosthocPerturbDecision(False, "not_sdpa_zero_query")
    query = args[0] if len(args) > 0 else None
    if not _is_all_zero_value(query):
        return PosthocPerturbDecision(False, "not_sdpa_zero_query")
    return PosthocPerturbDecision(
        True,
        "sdpa_zero_query_uniform_attention",
        "saved query is all zero: softmax(query @ key^T) is exactly uniform regardless "
        "of key, so the key operand cannot influence sdpa's output",
    )


def _softmax_singleton_dim_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for softmax reduced over a size-1 dimension.

    ``softmax(x)_i = exp(x_i) / sum_j exp(x_j)``: when the reduction dimension
    has exactly one element, this collapses to ``exp(x_0) / exp(x_0) == 1``
    for ANY finite ``x_0`` -- a shape-based identity, not a magnitude
    heuristic, and true regardless of the logits' actual values or training
    state (unlike the zero-annihilator exemptions, it never depends on a
    weight staying exactly zero).

    timm's ``csatv2`` family computes channel self-attention with a trivial
    single-key dimension; every ``softmax`` op in a freshly constructed
    ``csatv2``/``csatv2_21m`` reduces over a size-1 dimension, confirmed by
    direct introspection of the real model.

    Parameters
    ----------
    layer:
        ``softmax`` op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the perturbed parent is softmax's (sole) input
        and the reduced dimension has size 1, otherwise non-exempt.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_softmax_singleton_dim")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) != perturbed_label:
        return PosthocPerturbDecision(False, "not_softmax_singleton_dim")
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.ndim == 0:
        return PosthocPerturbDecision(False, "not_softmax_singleton_dim")
    saved_kwargs = getattr(layer, "saved_kwargs", None) or {}
    dim = saved_kwargs.get("dim")
    if dim is None and len(args) > 1:
        dim = args[1]
    # Only an explicit integer dim proves which axis is reduced: F.softmax's
    # implicit-dim rule (dim=None) picks dim 0 or 1 by rank, not the last dim,
    # so a missing dim stays strict rather than guessing.
    if isinstance(dim, bool) or not isinstance(dim, int):
        return PosthocPerturbDecision(False, "not_softmax_singleton_dim")
    if dim < -out.ndim or dim >= out.ndim:
        return PosthocPerturbDecision(False, "not_softmax_singleton_dim")
    if out.shape[dim] == 1:
        return PosthocPerturbDecision(
            True,
            "softmax_singleton_reduction_dim",
            f"softmax's reduction dim={dim} has size 1: softmax of a single element is "
            "identically 1 regardless of its value",
        )
    return PosthocPerturbDecision(False, "not_softmax_singleton_dim")


def _locally_constant_nan_multiplication_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for multiplication by a saved NaN operand.

    Parameters
    ----------
    layer:
        Multiplication op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the non-perturbed operand is all NaN and the
        output is therefore locally constant under finite perturbations of the
        other operand.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_locally_constant_nan_multiplication")
    if not _is_all_nan_value(getattr(layer, "out", None)):
        return PosthocPerturbDecision(False, "not_locally_constant_nan_multiplication")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) == perturbed_label:
        other_position = 1
    elif arg_positions.get(1) == perturbed_label:
        other_position = 0
    else:
        return PosthocPerturbDecision(False, "not_locally_constant_nan_multiplication")
    if _is_all_nan_value(args[other_position]):
        return PosthocPerturbDecision(
            True,
            "locally_constant_by_construction",
            (
                f"non-perturbed multiplication operand at args[{other_position}] is all-NaN; "
                "any finite perturbation of the other operand remains all-NaN"
            ),
        )
    return PosthocPerturbDecision(False, "not_locally_constant_nan_multiplication")


def _matmul_zero_annihilator_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for matmul by a saved zero operand.

    Parameters
    ----------
    layer:
        Matrix multiplication op whose unchanged perturbed replay is being
        classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the non-perturbed matrix operand is all zero and
        the output is all zero.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_matmul_zero_annihilator")
    if not _is_all_zero_value(getattr(layer, "out", None)):
        return PosthocPerturbDecision(False, "not_matmul_zero_annihilator")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) == perturbed_label:
        other_position = 1
    elif arg_positions.get(1) == perturbed_label:
        other_position = 0
    else:
        return PosthocPerturbDecision(False, "not_matmul_zero_annihilator")
    if _is_all_zero_value(args[other_position]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            f"non-perturbed matmul operand at args[{other_position}] is all zero",
        )
    return PosthocPerturbDecision(False, "not_matmul_zero_annihilator")


def _linear_zero_weight_input_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for a linear input under all-zero weights.

    Parameters
    ----------
    layer:
        Linear op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the perturbed parent is the input and the saved
        weight matrix is all zero.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_linear_zero_weight_input")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) != perturbed_label:
        return PosthocPerturbDecision(False, "not_linear_zero_weight_input")
    if _is_all_zero_value(args[1]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            "linear input is annihilated by an all-zero saved weight matrix",
        )
    return PosthocPerturbDecision(False, "not_linear_zero_weight_input")


def _conv_zero_weight_input_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for convolution input under all-zero kernels.

    Parameters
    ----------
    layer:
        Convolution op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the perturbed parent is the input and the saved
        convolution kernel is all zero.
    """

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_conv_zero_weight_input")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) != perturbed_label:
        return PosthocPerturbDecision(False, "not_conv_zero_weight_input")
    if _is_all_zero_value(args[1]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            "convolution input is annihilated by an all-zero saved kernel",
        )
    return PosthocPerturbDecision(False, "not_conv_zero_weight_input")


def _locally_constant_nonfinite_addition_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for finite addends swamped by saved non-finites.

    Parameters
    ----------
    layer:
        Additive op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when a non-perturbed non-finite operand proves the
        perturbed finite operand cannot affect the representable output.
    """

    if len(layers_to_perturb) != 1 or len(args) < 2:
        return PosthocPerturbDecision(False, "not_locally_constant_nonfinite_addition")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) == perturbed_label:
        other_position = 1
    elif arg_positions.get(1) == perturbed_label:
        other_position = 0
    else:
        return PosthocPerturbDecision(False, "not_locally_constant_nonfinite_addition")
    if _is_all_inf_value(args[other_position]):
        if not _is_all_inf_value(getattr(layer, "out", None)):
            return PosthocPerturbDecision(False, "not_locally_constant_nonfinite_addition")
        return PosthocPerturbDecision(
            True,
            "locally_constant_by_construction",
            (
                f"non-perturbed additive operand at args[{other_position}] is all-inf; "
                "any finite perturbation of the other operand remains all-inf in this dtype"
            ),
        )
    if _is_all_nan_value(args[other_position]):
        if not _is_all_nan_value(getattr(layer, "out", None)):
            return PosthocPerturbDecision(False, "not_locally_constant_nonfinite_addition")
        return PosthocPerturbDecision(
            True,
            "locally_constant_by_construction",
            (
                f"non-perturbed additive operand at args[{other_position}] is all-NaN; "
                "any finite perturbation of the other operand remains all-NaN"
            ),
        )
    return PosthocPerturbDecision(False, "not_locally_constant_nonfinite_addition")


def _is_all_zero_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-zero tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is zero.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return bool(torch.all(torch.eq(value, 0)).item())


def _is_all_inf_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-infinite tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is infinite.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return bool(torch.all(torch.isinf(value)).item())


def _is_all_nan_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-NaN tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is NaN.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return tensor_all_nan(value)
