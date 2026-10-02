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
from ..constants import get_orig_torch_funcs

# COMMUTATIVE reflected operator dunders (invoked when a non-tensor is on the LEFT, e.g.
# ``int & tensor`` routes to ``tensor.__rand__(int)``). For a COMMUTATIVE op the swapped operand
# order is irrelevant (``a & b == b & a``), so the reflected form is the SAME operation as its
# forward form -- map it to the forward op's normalized name so it (a) labels as the real op
# (``and``, not ``rand``), (b) picks up the forward binary arg-spec, (c) groups into the forward
# op's equivalence class, and (d) does NOT collide with a same-spelled factory. Without this,
# ``"__rand__".lower().replace("_","")`` yields ``"rand"`` -- mislabeling a bitwise-AND and
# (fatally) colliding with the ``torch.rand`` factory arg-spec, silently dropping tensor parents.
#
# NON-commutative reflected dunders (``__rsub__``, ``__rtruediv__``, ``__rmatmul__``, ...) are
# deliberately NOT mapped: they compute ``other <op> self`` (operands SWAPPED), so their
# ``r``-prefixed normalized name honestly signals the reversed order -- and e.g. ``torch.rsub`` is
# a real reverse-subtract function, so ``rsub`` is a correct, informative label. Collapsing them
# to ``sub`` would erase the reversed-operand distinction. Their parents/arg-spec were already
# correct (the binary (0,1) spec via their own ``r`` key); only ``__rand__`` was broken, by the
# ``torch.rand`` factory collision.
_COMMUTATIVE_REFLECTED_DUNDERS = {
    "__radd__": "add",
    "__rmul__": "mul",
    "__rand__": "and",
    "__ror__": "or",
    "__rxor__": "xor",
}


def _normalize_func_name(func_name: str) -> str:
    """Normalize a raw function name for lookup table keying.

    Commutative reflected operator dunders are mapped to their forward operation (see
    ``_COMMUTATIVE_REFLECTED_DUNDERS``); everything else is lowercased with underscores stripped.
    """
    forward = _COMMUTATIVE_REFLECTED_DUNDERS.get(func_name)
    if forward is not None:
        return forward
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

    # An index present in BOTH ``positions`` and ``sequence_positions`` (e.g.
    # ``_foreach_add`` arg 1: a single Tensor in the ``.Tensor`` overload, a
    # Tensor list in ``.List``) was already fully extracted above --
    # ``_append_tensor_or_param`` walks shallow sequences -- so re-walking it
    # here would duplicate every member's parent edge (round-31 M3).
    position_set = set(spec.positions)
    for pos in spec.sequence_positions:
        if pos < len(args) and pos not in position_set:
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


_ATEN_PACKET_NAMESPACE_PREFIXES: dict[str, tuple[str, ...]] = {
    "torch.linalg": ("linalg_",),
    "torch.special": ("special_",),
}


def _iter_aten_packet_names(namespace_name: str, func_name: str) -> tuple[str, ...]:
    """Return candidate ATen packet names for a wrapped torch target.

    Parameters
    ----------
    namespace_name:
        Namespace string from ``get_orig_torch_funcs()``.
    func_name:
        Wrapped callable name in that namespace.

    Returns
    -------
    tuple[str, ...]
        Candidate ``torch.ops.aten`` packet names to inspect for schemas.
    """

    canonical_name = func_name.strip("_")
    candidates = [canonical_name]
    for prefix in _ATEN_PACKET_NAMESPACE_PREFIXES.get(namespace_name, ()):
        candidates.append(f"{prefix}{canonical_name}")
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _schema_tensor_arg_kind(schema_arg: object) -> str | None:
    """Classify whether a schema argument carries tensor provenance.

    Parameters
    ----------
    schema_arg:
        One ``torch.FunctionSchema`` argument entry.

    Returns
    -------
    str | None
        ``"single"`` for a tensor/optional tensor, ``"sequence"`` for a tensor
        list, else ``None``.
    """

    arg_type = str(getattr(schema_arg, "type", ""))
    if "Tensor" not in arg_type:
        return None
    if arg_type.startswith("List[") or arg_type.endswith("[]"):
        return "sequence"
    return "single"


def _schema_arg_is_parent_candidate(schema_arg: object) -> bool:
    """Return whether a schema argument should become a graph parent.

    Parameters
    ----------
    schema_arg:
        One ``torch.FunctionSchema`` argument entry.

    Returns
    -------
    bool
        ``True`` when the argument is an input operand rather than an ``out=``
        destination slot.
    """

    if getattr(schema_arg, "name", None) == "out" and bool(
        getattr(schema_arg, "kwarg_only", False)
    ):
        return False
    alias_info = getattr(schema_arg, "alias_info", None)
    return not (
        alias_info is not None
        and bool(getattr(alias_info, "is_write", False))
        and not bool(getattr(alias_info, "is_read", False))
    )


def _merge_schema_tensor_slots(spec: ArgSpec, schemas: tuple[object, ...]) -> ArgSpec:
    """Return ``spec`` widened by tensor operands present in authoritative schemas.

    Parameters
    ----------
    spec:
        Existing static argument spec.
    schemas:
        Function schemas associated with wrapped variants of the same operator.

    Returns
    -------
    ArgSpec
        Merged argument spec. Existing positions stay intact; schema-backed
        tensor positions and kwarg names are appended when missing.
    """

    positions = list(spec.positions)
    position_set = set(spec.positions)
    sequence_positions = list(spec.sequence_positions)
    sequence_position_set = set(spec.sequence_positions)
    tensor_kwargs = list(spec.tensor_kwargs)
    tensor_kwarg_set = set(spec.tensor_kwargs)

    for schema in schemas:
        for index, schema_arg in enumerate(getattr(schema, "arguments", ()) or ()):
            if not _schema_arg_is_parent_candidate(schema_arg):
                continue
            tensor_kind = _schema_tensor_arg_kind(schema_arg)
            if tensor_kind is None:
                continue
            if tensor_kind == "single" and index not in position_set:
                positions.append(index)
                position_set.add(index)
            if tensor_kind == "sequence" and index not in sequence_position_set:
                sequence_positions.append(index)
                sequence_position_set.add(index)
            arg_name = getattr(schema_arg, "name", None)
            if isinstance(arg_name, str) and arg_name not in tensor_kwarg_set:
                tensor_kwargs.append(arg_name)
                tensor_kwarg_set.add(arg_name)

    return ArgSpec(
        positions=tuple(positions),
        sequence_positions=tuple(sequence_positions),
        tensor_kwargs=tuple(tensor_kwargs),
    )


def _apply_schema_tensor_position_corrections() -> None:
    """Upgrade under-specified static specs from ATen schemas.

    Every static spec is widened by the union of its ATen schemas' input-tensor
    slots (round-22 F1/F2/F4/F5 class fix). The pass previously refused to touch
    non-unary-style specs (any nonzero position or sequence position), which left
    hand-grouped multi-operand entries permanently exempt from schema correction:
    ``lu_solve`` keyed to the generic binary (0, 1) spec dropped its third tensor
    operand, ``cosine_similarity`` grouped with the input/target losses dropped
    both kwarg-passed operands, ``ctc_loss`` dropped tensor lengths, and
    ``searchsorted`` dropped ``sorter``. The merge is append-only (existing
    positions/kwargs stay intact) and parent-candidate-filtered (write-only
    ``out=`` destinations never join), so widening a correct spec is a no-op and
    widening an under-specified spec can only restore dropped parent edges.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Mutates ``FUNC_ARG_SPECS`` in place.
    """

    corrected_specs: dict[str, ArgSpec] = {}
    for namespace_name, func_name in get_orig_torch_funcs(include_torchvision=False):
        normalized_name = _normalize_func_name(func_name.strip("_"))
        current_spec = corrected_specs.get(normalized_name, FUNC_ARG_SPECS.get(normalized_name))
        if current_spec is None:
            continue

        schemas: list[object] = []
        for packet_name in _iter_aten_packet_names(namespace_name, func_name):
            packet = getattr(torch.ops.aten, packet_name, None)
            if packet is None:
                continue
            for overload_name in packet.overloads():
                overload = getattr(packet, overload_name, None)
                schema = getattr(overload, "_schema", None)
                if schema is not None:
                    schemas.append(schema)
        if not schemas:
            continue

        widened_spec = _merge_schema_tensor_slots(current_spec, tuple(schemas))
        if widened_spec != current_spec:
            corrected_specs[normalized_name] = widened_spec

    FUNC_ARG_SPECS.update(corrected_specs)


_schema_corrections_applied = False


def _ensure_schema_tensor_position_corrections() -> None:
    """Apply the ATen schema correction pass exactly once, on demand.

    The sweep over ``torch.ops.aten`` schemas is torch-capture setup, but this
    module is imported on every backend's first capture dispatch (via selector
    helpers and postprocess). Running it eagerly at import time made non-torch
    first captures pay the full torch schema sweep (~24% of a small Paddle
    first capture). ``wrap_torch()`` calls this before wrappers can log any
    op, so every torch capture still reads the fully corrected table; callers
    that audit ``FUNC_ARG_SPECS`` outside a capture must call it explicitly.
    """

    global _schema_corrections_applied
    if _schema_corrections_applied:
        return
    _apply_schema_tensor_position_corrections()
    _schema_corrections_applied = True


DYNAMIC_SPEC_UNCACHEABLE = object()
"""Sentinel cached for Tier-2 names whose BFS-found tensors cannot be represented
by an ``ArgSpec`` (tensors nested deeper than top-level args, shallow sequences,
or top-level kwargs). Such names must re-crawl every call: caching a lossy spec
would silently drop the unrepresentable operands from every later call."""


def _shallow_holds_tensor(value: object) -> bool:
    """Return whether ``value`` is a tensor or a shallow sequence holding one.

    Parameters
    ----------
    value:
        Candidate argument value.

    Returns
    -------
    bool
        ``True`` for a tensor/Parameter or a list/tuple containing one.
    """

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, torch.Tensor) for item in value)
    return False


def dynamic_spec_covers_call(
    spec: ArgSpec,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Return whether a Tier-2 cached spec covers every shallow tensor in this call.

    The dynamic cache derives an ``ArgSpec`` from a previously OBSERVED call shape.
    A later call may put a tensor at a slot an earlier call filled with a scalar
    (``x % 2.0`` then ``a % b``); extracting through the frozen spec would silently
    drop that operand's parent edge, making capture correctness depend on call
    order within and ACROSS traces (round-22 F3b: the cache is process-global).
    This coverage check is the guard: when the live call carries a tensor at any
    position or kwarg the cached spec does not extract, the caller must fall back
    to a fresh BFS crawl (and union-merge the result) instead of trusting the
    cache. With it, every top-level / shallow-sequence tensor operand is extracted
    identically regardless of what any earlier call looked like.

    Parameters
    ----------
    spec:
        Cached dynamic spec for the normalized function name.
    args:
        Live positional arguments.
    kwargs:
        Live keyword arguments.

    Returns
    -------
    bool
        ``True`` when extraction through ``spec`` finds every shallow tensor in
        the live call; ``False`` when a fresh crawl is required.
    """

    position_set = set(spec.positions)
    sequence_position_set = set(spec.sequence_positions)
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if index not in position_set:
                return False
        elif isinstance(arg, (list, tuple)) and any(isinstance(item, torch.Tensor) for item in arg):
            if index not in position_set and index not in sequence_position_set:
                return False
    if not kwargs:
        return True
    covered_kwargs = {_normalize_func_name(str(name)) for name in spec.tensor_kwargs}
    for key, value in kwargs.items():
        if _shallow_holds_tensor(value) and _normalize_func_name(str(key)) not in covered_kwargs:
            return False
    return True


def _cache_dynamic_spec(
    normalized_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    found_tensors: list[torch.Tensor],
    found_params: list[torch.nn.Parameter],
) -> None:
    """Construct, union-merge, and cache an ArgSpec from BFS crawl results (Tier 3).

    Round-22 F3 hardening. The cache used to freeze the FIRST observed call shape
    for the lifetime of the process, so one ``x % scalar`` observation dropped the
    tensor RHS parent of every later ``a % b`` -- in the same trace AND in every
    later ``tl.trace`` (``_state._dynamic_arg_specs`` is never cleared). Now:

    * the derived spec is UNION-merged (append-only) with any existing cached spec,
      so a new observation can only widen coverage, never narrow it; and
    * when the merged spec cannot re-extract everything the BFS found (tensors
      nested beyond ArgSpec's representable shapes), the name is marked
      ``DYNAMIC_SPEC_UNCACHEABLE`` so every later call re-crawls instead of
      silently dropping the unrepresentable operands on calls after the first.

    Together with the ``dynamic_spec_covers_call`` guard at the lookup site, this
    restores order-independence: extraction results for a call no longer depend on
    which call shapes were observed earlier in the process.

    Parameters
    ----------
    normalized_name:
        Normalized function name key.
    args:
        Positional arguments of the crawled call.
    kwargs:
        Keyword arguments of the crawled call.
    found_tensors:
        Tensors the BFS crawl located anywhere in the call.
    found_params:
        Parameters the BFS crawl located anywhere in the call.

    Returns
    -------
    None
        Mutates ``_state._dynamic_arg_specs`` in place.
    """

    existing = _state._dynamic_arg_specs.get(normalized_name)
    if existing is DYNAMIC_SPEC_UNCACHEABLE:
        return

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
        if val is None:
            continue
        if (
            id(val) in all_found_ids
            or isinstance(val, (list, tuple))
            and any(id(item) in all_found_ids for item in val)
        ):
            tensor_kwargs_found.append(key)

    if isinstance(existing, ArgSpec):
        existing_positions = set(existing.positions)
        existing_sequences = set(existing.sequence_positions)
        existing_kwargs = set(existing.tensor_kwargs)
        positions = list(existing.positions) + [p for p in positions if p not in existing_positions]
        sequence_positions = list(existing.sequence_positions) + [
            p for p in sequence_positions if p not in existing_sequences
        ]
        tensor_kwargs_found = list(existing.tensor_kwargs) + [
            k for k in tensor_kwargs_found if k not in existing_kwargs
        ]

    spec = ArgSpec(
        positions=tuple(positions),
        sequence_positions=tuple(sequence_positions),
        tensor_kwargs=tuple(tensor_kwargs_found),
    )
    re_tensors, re_params = extract_tensors_and_params(spec, args, kwargs)
    re_found_ids = {id(t) for t in re_tensors} | {id(p) for p in re_params}
    if not all_found_ids <= re_found_ids:
        _state._dynamic_arg_specs[normalized_name] = DYNAMIC_SPEC_UNCACHEABLE
        return
    _state._dynamic_arg_specs[normalized_name] = spec


# ============================================================================
# Variadic-arity transform ops -- never name-cache a positional ArgSpec
# ============================================================================

# These functions take a VARIADIC number of tensor operands, either directly
# (``block_diag(*tensors)``) or via a transformed callable (``vmap``/``grad``).
# The Tier-3 dynamic cache keys an ``ArgSpec`` by normalized func name from the
# FIRST observed call. Reusing that narrow spec silently drops later operands'
# parent edges, orphaning and dropping real ops from the trace. These names must
# ALWAYS take the fresh Tier-3 crawl and must never populate or read the
# name-keyed ArgSpec cache. Container APIs such as ``cat(tensors)`` and
# ``stack(tensors)`` are deliberately excluded: their sequence-position specs
# already inspect every element of their single container argument.
VARIADIC_TENSOR_ARG_FUNCS: frozenset[str] = frozenset(
    {
        "aligntensors",
        "atleast1d",
        "atleast2d",
        "atleast3d",
        "blockdiag",
        "broadcasttensors",
        "cartesianprod",
        "chainmatmul",
        "einsum",
        "vmap",
        "grad",
        "gradandvalue",
        "autogradjacobian",
        "autogradhessian",
        "autogradvjp",
        "autogradjvp",
        "autogradhvp",
        "autogradvhp",
        "meshgrid",
        # Collective boundary nodes pass their contribution tensors as
        # positional call_args; the list-taking collectives (and root-only
        # scatter, whose contribution arity differs by rank role) vary per
        # call, so they must never lock in a first-observed ArgSpec arity.
        "reducescatter",
        "alltoall",
        "scatter",
    }
)


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

# ``min``/``max`` are optional tensor bounds for clamp-family APIs.  Keep the
# generic binary keyword aliases as well because the Tensor method and torch
# function forms use different spellings for the primary operand.
_CLAMP_SPEC = ArgSpec(
    positions=_P012.positions,
    tensor_kwargs=(*_P01_BINARY.tensor_kwargs, "min", "max"),
)

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
    "isshared",
    "isview",
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
    "array",
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
    # torch.sym_sum(xs): N-ary add over a list/tuple of ints or SymInts (never
    # Tensors -- "only does something special for integers" per its own
    # docstring). Position 0 is the single sequence argument; _P0_INPUT's
    # generic list-walk in _append_tensor_or_param extracts nothing from it
    # (ints, not tensors), the same honest no-op as its symint/symfloat/symnot
    # neighbors above. Exercised by nested/jagged-tensor shape arithmetic.
    "symsum",
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
    # The MODERN torch.signal.windows namespace (B3 R02 inventory fix). Same shape as
    # the legacy top-level ``*_window`` twins above: the first positional argument is a
    # length, never a tensor, so nothing beyond position 0 can be a tensor arg.
    "bartlett",
    "blackman",
    "cosine",
    "exponential",
    "gaussian",
    "generalcosine",
    "generalhamming",
    "hamming",
    "hann",
    "kaiser",
    "nuttall",
    # torch.from_dlpack builds a tensor from a FOREIGN capsule/producer object, so it
    # has no tensor argument at all -- the same source-factory shape as from_numpy.
    "fromdlpack",
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
    "copy",
    "div",
    "divide",
    "truedivide",
    "floordivide",
    "remainder",
    "fmod",
    # "mod" is Tensor.__mod__, the PUBLIC ``%`` operator (the only decorated callable
    # normalizing to this key). It was missing here, so it fell to the Tier-2 dynamic
    # cache and a first ``x % scalar`` observation froze positions=(0,), dropping the
    # tensor RHS parent of every later ``a % b`` (round-22 F3a).
    "mod",
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
    # NOTE: no "rand" here -- reflected __rand__ normalizes to "and" (see
    # _COMMUTATIVE_REFLECTED_DUNDERS); "rand" belongs to the torch.rand factory only.
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

for _name in ["clamp", "clampmin", "clampmax", "clip"]:
    FUNC_ARG_SPECS[_name] = _CLAMP_SPEC

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
    # NOTE: "tensor" (torch.tensor) is NOT a pure factory -- ``torch.tensor(data)``
    # accepts an existing tensor as ``data`` (torch warns but executes), a real
    # data-lineage edge. It gets an explicit spec below (round-22 F6).
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

# ``fill_value`` is schema-typed as ``Scalar``, but PyTorch also accepts a scalar
# tensor there.  When it is a tensor its runtime value is a real data dependency;
# extraction's type checks leave ordinary Python scalar calls parentless.
FUNC_ARG_SPECS["full"] = ArgSpec(positions=(1,), tensor_kwargs=("fill_value",))

# Guardrail: a normalized name must not be claimed by BOTH the binary-op and factory-func
# tables -- they assign conflicting arg-specs (tensor parents at positions 0,1 vs. NO tensor
# parents), and dict last-writer-wins would silently corrupt whichever loses. A collision means
# two different ops normalize to the same key (e.g. the historic __rand__/torch.rand "rand"
# collision that dropped __rand__'s parents, now resolved by mapping reflected dunders to their
# forward op in _COMMUTATIVE_REFLECTED_DUNDERS). Fail LOUDLY at import so this class of silent
# dataflow corruption can never recur unnoticed.
_BINARY_FACTORY_KEY_COLLISIONS = set(_BINARY_FUNCS) & set(_FACTORY_FUNCS)
if _BINARY_FACTORY_KEY_COLLISIONS:
    # A real ``raise``, never ``assert`` (R24-1): ``python -O`` strips
    # asserts, and this guard's whole promise is that the corruption "can
    # never recur unnoticed" -- under ``-O`` a future table collision would
    # import clean and silently mis-assign tensor parents (wrong dataflow
    # edges in every trace).
    raise RuntimeError(
        "arg-spec key collision between binary-op and factory-func tables: "
        f"{sorted(_BINARY_FACTORY_KEY_COLLISIONS)} -- these ops disagree on tensor-parent "
        "positions; give them distinct normalized keys (see _COMMUTATIVE_REFLECTED_DUNDERS)."
    )

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

# torch.tensor(data): a value-COPY factory whose ``data`` may be an EXISTING tensor
# (legal; torch emits a UserWarning recommending clone().detach() but executes).
# That is a data-lineage edge exactly like ``as_tensor``/``clone``/``detach``, so the
# source must become a graph parent; dropping it disconnected the op from its input
# ancestry with NO unattributed marker (round-22 F6 -- the only fully silent drop
# found, because the narrower scalar-only ``aten::tensor`` packet disarmed the
# witness; see _arg_position_is_tensor_operand in backends/torch/ops.py for the
# witness-side fix). Scalar/list ``data`` holds no tensor, so extraction's
# isinstance checks keep plain factory calls parentless.
FUNC_ARG_SPECS["tensor"] = ArgSpec(positions=(0,), tensor_kwargs=("data",))

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

# linear_cross_entropy: F.linear_cross_entropy(input, linear_weight, target, *, linear_bias=None,
# weight=None, ...) — three positional tensors, optional tensor kwargs linear_bias/weight
FUNC_ARG_SPECS["linearcrossentropy"] = ArgSpec(
    positions=(0, 1, 2),
    tensor_kwargs=("input", "linear_weight", "target", "linear_bias", "weight"),
)

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
    "hingeembeddingloss",
    "softmarginloss",
    "multilabelsoftmarginloss",
    "multimarginloss",
    "multilabelmarginloss",
    "poissonnllloss",
    "gaussiannllloss",
    "kldiv",
    "ctcloss",
]:
    FUNC_ARG_SPECS[_name] = _P01_INPUT_TARGET

# Three-input loss functions must retain every value-affecting tensor operand.
_THREE_INPUT_TARGET_LOSS_SPEC = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("input1", "input2", "target")
)
FUNC_ARG_SPECS["marginrankingloss"] = _THREE_INPUT_TARGET_LOSS_SPEC
FUNC_ARG_SPECS["cosineembeddingloss"] = _THREE_INPUT_TARGET_LOSS_SPEC
FUNC_ARG_SPECS["tripletmarginloss"] = ArgSpec(
    positions=(0, 1, 2), tensor_kwargs=("anchor", "positive", "negative")
)

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
FUNC_ARG_SPECS["histogram"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "bins", "weight"))
FUNC_ARG_SPECS["histogramdd"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "bins", "weight"))
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

# quantile: ``q`` can be a tensor that changes the selected value.
_QUANTILE_SPEC = ArgSpec(positions=(0, 1), tensor_kwargs=(*_P0_INPUT.tensor_kwargs, "q"))
FUNC_ARG_SPECS["quantile"] = _QUANTILE_SPEC
FUNC_ARG_SPECS["nanquantile"] = _QUANTILE_SPEC

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
    "ndtr",
    "ndtri",
    "normalize",
    "pcalowrank",
    "pdist",
    "pinv",
    "poisson",
    "psi",
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

# ``spacing`` accepts a tensor or a sequence of tensors.  It is an input to
# the numerical gradient computation, not display-only metadata, so it must
# be included in parent extraction.
FUNC_ARG_SPECS["gradient"] = ArgSpec(
    positions=_P0_INPUT.positions,
    tensor_kwargs=(*_P0_INPUT.tensor_kwargs, "spacing"),
)

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
# ``align_as`` (named-tensor Tensor method) was REMOVED from torch in 2.13, so on this runtime
# it is undecorated and this static spec is unused. It is RETAINED (not pruned) because the
# pinned torch<=2.12 CI legs still decorate ``Tensor.align_as``; deleting the spec there would
# make it a decorated-without-static-spec op absent from the arg-spec ledger, hard-failing
# tests/test_arg_spec_coverage.py::test_every_decorated. It is version-varying, so it is dropped
# from _HIGH_CONFIDENCE_STATIC_NAMES (mirrors F5's handling of torch-version-varying fills).
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
    "foreachclone": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
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
    "foreachmm": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "mat2")),
    "foreachmul": ArgSpec(
        positions=(1,), sequence_positions=(0, 1), tensor_kwargs=("self", "other")
    ),
    "foreachneg": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachnorm": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
    "foreachpow": ArgSpec(sequence_positions=(0, 1), tensor_kwargs=("self", "exponent")),
    "foreachpowsum": ArgSpec(sequence_positions=(0,), tensor_kwargs=("self",)),
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
    "hashtensor": ArgSpec(positions=(0,), tensor_kwargs=("self", "input")),
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
    # F.max_unpool*'s internal size check: (output_size: list[int], dim: int),
    # no tensor-bearing arguments.
    "checkunpooloutputsize": ArgSpec(),
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
    # The rest of the PUBLIC sparse-compressed family (B3 R02 inventory fix): each takes
    # the two index tensors plus the value tensor in positions 0-2, spelled per layout.
    "sparsecsctensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("ccol_indices", "row_indices", "values")
    ),
    "sparsebsrtensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("crow_indices", "col_indices", "values")
    ),
    "sparsebsctensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("ccol_indices", "row_indices", "values")
    ),
    "sparsecompressedtensor": ArgSpec(
        positions=(0, 1, 2), tensor_kwargs=("compressed_indices", "plain_indices", "values")
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

# Schema correction is deliberately NOT applied at import time: this module is
# imported on every backend's first capture dispatch (selector helpers,
# postprocess), and the ATen schema sweep is a torch-only cost that dominated
# non-torch first-capture profiles. ``wrap_torch()`` arms it before any torch
# op record can be built, so torch capture always reads the corrected table.

# Cleanup loop variable leakage
del _name, _spec
