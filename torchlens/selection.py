"""Selection / intervention algebra: the dual selection types (L6 stage 1).

``Selection`` is the composable QUERY — a frozen AST over leaf terms and
boolean combinators, model/trace-independent; resolving it is explicit.
``ResolvedSelection`` is returned by ``selection.resolve(trace)`` — frozen,
concrete, trace-bound: an ordered tuple of :class:`SiteEntry` rows.

Every public spelling here ships DOCUMENTED-UNSTABLE pending its
naming-session ratification (megasprint provisional-name protocol).

TWO-LEVEL DENOTATION (normative): a selection denotes the pair
(TOUCHED-SITE FAMILY, SELECTED-ELEMENT SET). Zero-mask entries are retained,
first-class, by every operator and by ``resolve()`` — an element-empty site
stays touched (this is what makes ``~`` an involution). Equality and the
resolve digest compare BOTH levels; ``.empty`` / ``__bool__`` report the
ELEMENT level only.

Operator set: ``|  &  -  ~`` plus reflected ``__ror__/__rand__/__rsub__``.
There is deliberately NO ``__xor__``: ``(a - b) | (b - a)`` spells it.
Family rules: ``fam(A|B) = fam(A) ∪ fam(B)``; ``fam(A&B) = fam(A) ∩ fam(B)``
(shared sites stay touched even element-empty); ``fam(A-B) = fam(A)``
(subtraction never un-touches); ``fam(~A) = fam(A)`` (complement never
changes reach). ``~`` is ALWAYS the touched-site mask complement — never
predicate negation (users who mean "all sites not matching s" spell ``~s``
BEFORE lifting; ``~lift(s) != lift(~s)`` is a pinned non-law).

Selection masks are always EXACT AS SETS — producer inexactness (receptive-
field hulls) is carried as provenance disclosure (``provenance.relation``,
one of the closed lattice ``exact | upper_bound | lower_bound | unknown``),
never as a fuzzy mask.

The kind vocabulary ``ACT | PARAM | EDGE`` is a CLOSED public vocabulary
(S2-routed). Mixed-kind composition refuses typed
(``selection_kind_incompatible``); the matrix cells change only via explicit
D-ruling landing as an S2 amendment, never a lane-local edit.

``ResolvedSelection`` is SESSION-TIME ONLY and is never persisted.
"""

from __future__ import annotations

import hashlib
import random as _random_module
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from .errors._base import ConfigurationError, TorchLensError

if TYPE_CHECKING:
    from .data_classes.trace import Trace

__all__ = [
    "ResolvedSelection",
    "Selection",
    "SelectionError",
    "SiteEntry",
    "params",
    "random_selection",
    "units",
]

SelectionKindName = Literal["ACT", "PARAM", "EDGE"]
_SELECTION_KINDS: tuple[str, ...] = ("ACT", "PARAM", "EDGE")

RelationName = Literal["exact", "upper_bound", "lower_bound", "unknown"]
_RELATIONS: tuple[str, ...] = ("exact", "upper_bound", "lower_bound", "unknown")

_OPERATOR_SYMBOLS = {"or": "|", "and": "&", "sub": "-"}


class SelectionError(ConfigurationError, ValueError):
    """Carrier class for the closed ``selection_*`` refusal codes.

    Branch on ``exc.fields["code"]`` (``selection_trace_mismatch`` /
    ``selection_bool_ambiguous`` / ``selection_kind_incompatible`` /
    ``selection_unresolvable`` / ``selection_apply_invalid`` /
    ``selection_alignment_invalid``), never on message text.
    """


# ---------------------------------------------------------------------------
# Provenance relation lattice (closed, S2-routed) + composition tables.
# ---------------------------------------------------------------------------

_FLIP: dict[str, str] = {
    "exact": "exact",
    "upper_bound": "lower_bound",
    "lower_bound": "upper_bound",
    "unknown": "unknown",
}


def _join_relation(a: str, b: str) -> str:
    """JOIN transfer for both ``|`` and ``&`` (symmetric; exact is identity)."""

    if a == "exact":
        return b
    if b == "exact":
        return a
    if a == b:
        return a
    return "unknown"


def _difference_relation(a: str, b: str) -> str:
    """DIFFERENCE transfer (ordered): ``rel(A - B) = join(rel(A), FLIP(rel(B)))``."""

    return _join_relation(a, _FLIP[b])


def _meet_relation(a: str, b: str) -> str:
    """Multi-site MEET (exact on top, unknown on bottom, upper/lower incomparable)."""

    return _join_relation(a, b)


@dataclass(frozen=True)
class SelectionProvenance:
    """Disclosure attached to one resolved site entry.

    ``relation`` states how the mask relates to true semantic support:
    ``exact`` (the mask IS the support as a set), ``upper_bound`` (superset),
    ``lower_bound`` (subset), or ``unknown``. Producer inexactness (interval
    hulls) rides here — the mask itself is always exact as a set.
    """

    relation: str
    source: str = ""

    def __post_init__(self) -> None:
        """Validate the closed relation vocabulary."""

        if self.relation not in _RELATIONS:
            raise ValueError(
                f"provenance relation must be one of {_RELATIONS}; got {self.relation!r}."
            )


# ---------------------------------------------------------------------------
# Canonical internal mask representation. NEVER handed out: any user-facing
# tensor access returns a FRESH materialization (mask immutability pin).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Mask:
    """Internal canonical mask over one site's output index space."""

    shape: tuple[int, ...]
    form: str  # "whole" | "empty" | "slices" | "dense"
    slice_bounds: tuple[tuple[int, int, int], ...] | None = None
    dense: torch.Tensor | None = field(default=None, compare=False)

    def numel(self) -> int:
        """Return the site's index-space size."""

        total = 1
        for extent in self.shape:
            total *= extent
        return total

    def to_dense(self) -> torch.Tensor:
        """Return a FRESH dense bool tensor (safe to hand out)."""

        if self.form == "whole":
            return torch.ones(self.shape, dtype=torch.bool)
        if self.form == "empty":
            return torch.zeros(self.shape, dtype=torch.bool)
        if self.form == "slices":
            if self.slice_bounds is None:
                raise RuntimeError("slices-form mask lost its slice_bounds")
            mask = torch.zeros(self.shape, dtype=torch.bool)
            mask[tuple(slice(*bounds) for bounds in self.slice_bounds)] = True
            return mask
        if self.dense is None:
            raise RuntimeError("dense-form mask lost its dense tensor")
        return self.dense.clone()

    def _dense_ro(self) -> torch.Tensor:
        """Return the dense form for internal composition (never handed out)."""

        if self.form == "dense":
            if self.dense is None:
                raise RuntimeError("dense-form mask lost its dense tensor")
            return self.dense
        return self.to_dense()

    def count(self) -> int:
        """Return the number of selected elements."""

        if self.form == "whole":
            return self.numel()
        if self.form == "empty":
            return 0
        if self.form == "slices":
            if self.slice_bounds is None:
                raise RuntimeError("slices-form mask lost its slice_bounds")
            total = 1
            for start, stop, step in self.slice_bounds:
                total *= max(0, (stop - start + step - 1) // step)
            return total
        if self.dense is None:
            raise RuntimeError("dense-form mask lost its dense tensor")
        return int(self.dense.sum().item())

    def canonical_bytes(self) -> bytes:
        """Return canonical identity bytes (dense packing, form-independent)."""

        return self._dense_ro().to(torch.uint8).numpy().tobytes()


def _mask_whole(shape: tuple[int, ...]) -> _Mask:
    """Build the whole-form mask (every element of ``shape`` selected)."""

    return _Mask(shape=shape, form="whole")


def _mask_empty(shape: tuple[int, ...]) -> _Mask:
    """Build the empty-form mask (no element of ``shape`` selected)."""

    return _Mask(shape=shape, form="empty")


def _mask_from_dense(shape: tuple[int, ...], dense: torch.Tensor) -> _Mask:
    """Build a canonical mask from a dense bool tensor (normalizing extremes)."""

    selected = int(dense.sum().item())
    if selected == 0:
        return _mask_empty(shape)
    if selected == dense.numel():
        return _mask_whole(shape)
    return _Mask(shape=shape, form="dense", dense=dense.clone())


def _mask_from_slices(shape: tuple[int, ...], slices: tuple[slice, ...]) -> _Mask:
    """Build a canonical mask from a slice tuple over ``shape``."""

    bounds: list[tuple[int, int, int]] = []
    for one_slice, extent in zip(slices, shape, strict=True):
        start, stop, step = one_slice.indices(extent)
        if stop <= start:
            return _mask_empty(shape)
        bounds.append((start, stop, step))
    if all(
        start == 0 and stop == extent and step == 1
        for (start, stop, step), extent in zip(bounds, shape, strict=True)
    ):
        return _mask_whole(shape)
    return _Mask(shape=shape, form="slices", slice_bounds=tuple(bounds))


def _mask_union(a: _Mask, b: _Mask) -> _Mask:
    """Union two same-shape masks (whole/empty fast paths, else dense OR)."""

    if a.form == "whole" or b.form == "empty":
        return a if a.form == "whole" else (a if b.form == "empty" else b)
    if b.form == "whole" or a.form == "empty":
        return b if b.form == "whole" else (b if a.form == "empty" else a)
    return _mask_from_dense(a.shape, a._dense_ro() | b._dense_ro())


def _mask_intersect(a: _Mask, b: _Mask) -> _Mask:
    """Intersect two same-shape masks (whole/empty fast paths, else dense AND)."""

    if a.form == "empty" or b.form == "whole":
        return a
    if b.form == "empty" or a.form == "whole":
        return b
    return _mask_from_dense(a.shape, a._dense_ro() & b._dense_ro())


def _mask_difference(a: _Mask, b: _Mask) -> _Mask:
    """Subtract ``b`` from ``a`` (set difference on same-shape masks)."""

    if a.form == "empty" or b.form == "empty":
        return a
    if b.form == "whole":
        return _mask_empty(a.shape)
    return _mask_from_dense(a.shape, a._dense_ro() & ~b._dense_ro())


def _mask_complement(a: _Mask) -> _Mask:
    """Complement a mask within its own index space."""

    if a.form == "whole":
        return _mask_empty(a.shape)
    if a.form == "empty":
        return _mask_whole(a.shape)
    return _mask_from_dense(a.shape, ~a._dense_ro())


def _mask_equal(a: _Mask, b: _Mask) -> bool:
    """Compare two masks as element sets (form-independent equality)."""

    if a.shape != b.shape:
        return False
    if a.form == b.form == "whole" or a.form == b.form == "empty":
        return True
    return bool(torch.equal(a._dense_ro(), b._dense_ro()))


# ---------------------------------------------------------------------------
# SiteEntry + ResolvedSelection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteEntry:
    """One (site, mask, provenance) row of a resolved selection.

    ``site_key`` is ``(layer_label, pass_index)`` for ACT sites and
    ``(parameter_address,)`` for PARAM sites. ``structural_site_key`` carries
    the L1 structural site key when present (bridging relation; ``None``
    otherwise). ``mask`` returns a FRESH dense bool materialization on every
    access — mutating the returned tensor cannot alter the selection.
    """

    kind: str
    site_key: tuple[Any, ...]
    provenance: SelectionProvenance
    _mask: _Mask
    structural_site_key: Any = None

    @property
    def mask(self) -> torch.Tensor:
        """Return a fresh dense bool mask over the site's output index space."""

        return self._mask.to_dense()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the site's output index-space shape."""

        return self._mask.shape

    @property
    def selected_count(self) -> int:
        """Return the number of selected elements at this site."""

        return self._mask.count()

    def __repr__(self) -> str:
        """Return a compact disclosure row."""

        return (
            f"SiteEntry({self.site_key!r}, {self.selected_count}/{self._mask.numel()} "
            f"elements, relation={self.provenance.relation})"
        )


def _canonical_entry_order(entries: Iterable[SiteEntry]) -> tuple[SiteEntry, ...]:
    """Return entries in canonical deterministic order (sorted by site key)."""

    return tuple(sorted(entries, key=lambda entry: (entry.kind, tuple(map(str, entry.site_key)))))


def _graph_entry_order(trace: Any, entries: Iterable[SiteEntry]) -> tuple[SiteEntry, ...]:
    """Return entries in GRAPH ORDER (F10; lovely matrix row, bug 14).

    ACT entries order by the trace's execution order of their
    ``(layer_label, pass_index)`` address — a resolved selection reads in
    the order the network ran, not in lexicographic site-key order
    (``relu_1_10`` no longer sorts before ``relu_1_2``). PARAM/EDGE
    entries (and any address the trace cannot place) keep the canonical
    site-key sort, appended after placed entries deterministically.
    """

    from .utils.fail_open import fail_open

    placed = _canonical_entry_order(entries)

    def _execution_order() -> dict[tuple[Any, ...], int]:
        """Map each ``(layer_label, pass_index)`` address to its graph step."""

        return {
            (op.layer_label, op.pass_index): step
            for step, op in enumerate(getattr(trace, "layer_list", ()) or ())
        }

    order = fail_open(_execution_order, lambda _error: {})
    if not order:
        return placed

    def sort_key(indexed: tuple[int, SiteEntry]) -> tuple[int, int, int]:
        """Graph step first; unplaced entries keep canonical order after."""

        position, entry = indexed
        step = order.get(tuple(entry.site_key)) if entry.kind == "ACT" else None
        if step is None:
            return (1, position, 0)
        return (0, step, position)

    return tuple(entry for _, entry in sorted(enumerate(placed), key=sort_key))


class ResolvedSelection:
    """Frozen, concrete, trace-bound selection: an ordered tuple of SiteEntry.

    Zero-mask (element-empty) entries are retained first-class — the family
    level of the two-level denotation. ``.empty`` and ``__bool__`` report the
    ELEMENT level only; read the family via iteration / ``len()``.
    Session-time only; never persisted.
    """

    __slots__ = ("_digest", "_entries", "_kind", "_trace")

    _digest: str
    _entries: tuple[SiteEntry, ...]
    _kind: str
    _trace: Any

    def __init__(self, trace: Any, kind: str, entries: Iterable[SiteEntry]) -> None:
        """Freeze a resolved selection (internal constructor).

        Parameters
        ----------
        trace:
            The trace the selection resolved against (identity-bound).
        kind:
            Selection kind, one of the closed ``ACT | PARAM | EDGE``.
        entries:
            Site entries; canonicalized to deterministic order.
        """

        if kind not in _SELECTION_KINDS:
            raise ValueError(f"selection kind must be one of {_SELECTION_KINDS}; got {kind!r}.")
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_entries", _graph_entry_order(trace, entries))
        digest = hashlib.sha256()
        digest.update(kind.encode())
        for entry in self._entries:
            digest.update(repr(entry.site_key).encode())
            digest.update(repr(entry.shape).encode())
            digest.update(entry._mask.canonical_bytes())
        object.__setattr__(self, "_digest", digest.hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("ResolvedSelection is frozen; construct a new selection instead.")

    @property
    def kind(self) -> str:
        """Return the selection kind (``ACT`` / ``PARAM`` / ``EDGE``)."""

        return self._kind

    @property
    def empty(self) -> bool:
        """Return whether NO element is selected (element level).

        A selection can be element-empty while still touching sites; the
        touched-site family stays readable via iteration / ``len()``.
        """

        return all(entry.selected_count == 0 for entry in self._entries)

    @property
    def resolve_digest(self) -> str:
        """Return the freeze-time digest of the canonical (family, masks) pair."""

        return self._digest

    def align_to(self, target: Trace) -> ResolvedSelection:
        """Re-bind this resolved selection onto another trace (cross-run).

        The stage-4a cross-run door: alignment keys on the L1 structural
        site keys each entry records (position proof) AND the same-policy
        ``(layer_label, pass_index)`` address, under the L1 cross-stamp rule
        (same-policy captures only — healthy, agreeing grouping stamps on
        both sides; anything else refuses typed). Masks travel unchanged
        onto an identical index space; per-entry provenance discloses the
        cross-run origin. ``do()`` still refuses foreign resolved selections
        (``selection_trace_mismatch``) — this explicit spelling is the only
        rebind. DOCUMENTED-UNSTABLE pending naming-session ratification.

        Raises
        ------
        SelectionError
            ``selection_alignment_invalid`` with a closed reason set
            (``kind_unsupported`` / ``grouping_stamp_degraded`` /
            ``grouping_stamp_mismatch`` / ``site_key_unavailable`` /
            ``site_not_in_target`` / ``index_space_mismatch``).
        """

        from ._selection_align import align_resolved_selection

        return align_resolved_selection(self, target)

    def __bool__(self) -> bool:
        """Return whether ANY element is selected (element level)."""

        return not self.empty

    def __len__(self) -> int:
        """Return the touched-site family size (zero-mask entries included)."""

        return len(self._entries)

    def __iter__(self) -> Iterator[SiteEntry]:
        """Iterate site entries in canonical order."""

        return iter(self._entries)

    def __getitem__(self, index: int) -> SiteEntry:
        """Return one site entry by position."""

        return self._entries[index]

    def __eq__(self, other: object) -> bool:
        """Compare BOTH denotation levels: family AND per-site masks."""

        if not isinstance(other, ResolvedSelection):
            return NotImplemented
        if self._kind != other._kind or len(self._entries) != len(other._entries):
            return False
        for mine, theirs in zip(self._entries, other._entries, strict=True):
            if mine.kind != theirs.kind or mine.site_key != theirs.site_key:
                return False
            if not _mask_equal(mine._mask, theirs._mask):
                return False
        return True

    def __hash__(self) -> int:
        """Hash the freeze-time canonical digest."""

        return hash(self._digest)

    def __repr__(self) -> str:
        """Return a compact two-level summary."""

        elements = sum(entry.selected_count for entry in self._entries)
        return f"ResolvedSelection[{self._kind}]({len(self._entries)} sites, {elements} elements)"

    def __selection__(self) -> ResolvedSelection:
        """Pass anywhere a selection is expected."""

        return self

    # Boolean composition ---------------------------------------------------

    def __or__(self, other: Any) -> Any:
        return _compose_any("or", self, other)

    def __ror__(self, other: Any) -> Any:
        return _compose_any("or", other, self)

    def __and__(self, other: Any) -> Any:
        return _compose_any("and", self, other)

    def __rand__(self, other: Any) -> Any:
        return _compose_any("and", other, self)

    def __sub__(self, other: Any) -> Any:
        return _compose_any("sub", self, other)

    def __rsub__(self, other: Any) -> Any:
        return _compose_any("sub", other, self)

    def __invert__(self) -> ResolvedSelection:
        """Complement per TOUCHED site (never model-universe).

        The complement of a no-touched-sites selection is the no-touched-sites
        selection; the complement of an element-empty-but-touched selection is
        full masks over the same family (the involution's other half). EDGE
        selections complement within the trace's dataflow edge family (the
        one well-defined universe).
        """

        if self._kind == "EDGE":
            return _edge_family_complement(self)
        entries = tuple(
            SiteEntry(
                kind=entry.kind,
                site_key=entry.site_key,
                provenance=SelectionProvenance(
                    relation=_FLIP[entry.provenance.relation],
                    source=entry.provenance.source,
                ),
                _mask=_mask_complement(entry._mask),
                structural_site_key=entry.structural_site_key,
            )
            for entry in self._entries
        )
        return ResolvedSelection(self._trace, self._kind, entries)


# ---------------------------------------------------------------------------
# Query AST terms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelectorTerm:
    """AST leaf lifting a ``BaseSelector`` predicate into a Selection."""

    selector: Any

    def __repr__(self) -> str:
        return f"selector({self.selector!r})"


@dataclass(frozen=True)
class _BoxTerm:
    """AST leaf lifting a ``ReceptiveFieldBox`` region into a Selection."""

    box: Any

    def __repr__(self) -> str:
        return f"rf_box({self.box.op_label!r} -> {self.box.io_role!r})"


@dataclass(frozen=True)
class _GradientTerm:
    """AST leaf lifting a ``GradientReceptiveField`` region into a Selection."""

    gradient_rf: Any

    def __repr__(self) -> str:
        return f"rf_gradient({self.gradient_rf.op_label!r} -> {self.gradient_rf.io_role!r})"


@dataclass(frozen=True)
class _FacetTerm:
    """AST leaf lifting a ``FacetSpec`` region into a Selection."""

    spec: Any

    def __repr__(self) -> str:
        home = self.spec.home_label or getattr(self.spec, "home_address", None)
        return f"facet({home!r})"


@dataclass(frozen=True)
class _ParamTerm:
    """AST leaf for ``tl.params(name, mask=...)`` (PARAM-kind producer)."""

    name: str
    mask: torch.Tensor | None

    def __repr__(self) -> str:
        suffix = "" if self.mask is None else ", mask=<bool tensor>"
        return f"params({self.name!r}{suffix})"


@dataclass(frozen=True)
class _UnitTerm:
    """AST leaf for ``tl.units(site, indices|mask)`` (explicit ACT elements)."""

    site: str
    indices: tuple[tuple[int, ...], ...] | None
    index_mask: torch.Tensor | None

    def __repr__(self) -> str:
        if self.index_mask is not None:
            return f"units({self.site!r}, mask=<bool tensor>)"
        if self.indices is None:  # constructor guarantees one of the two forms
            return f"units({self.site!r})"
        return f"units({self.site!r}, n={len(self.indices)})"


@dataclass(frozen=True)
class _WholeSiteTerm:
    """AST leaf selecting every element of one site (Op/Layer lift)."""

    site_label: str
    pass_index: int | None

    def __repr__(self) -> str:
        if self.pass_index is None:
            return f"site({self.site_label!r})"
        return f"site({self.site_label!r}:{self.pass_index})"


@dataclass(frozen=True)
class _RandomTerm:
    """AST leaf for ``tl.random_selection`` (seeded size-matched control)."""

    like: Any
    within: Any
    seed: int

    def __repr__(self) -> str:
        return f"random_selection(like={self.like!r}, within={self.within!r}, seed={self.seed})"


@dataclass(frozen=True)
class _Combinator:
    """AST interior node combining operand terms with one set operator."""

    op: str  # "or" | "and" | "sub" | "invert"
    operands: tuple[Any, ...]

    def __repr__(self) -> str:
        if self.op == "invert":
            return f"~{self.operands[0]!r}"
        symbol = _OPERATOR_SYMBOLS[self.op]
        return "(" + f" {symbol} ".join(repr(operand) for operand in self.operands) + ")"


class Selection:
    """Composable, trace-independent selection query (frozen AST).

    Compose with ``|  &  -  ~``; resolve explicitly with
    ``selection.resolve(trace)``. Truthiness is deliberately AMBIGUOUS on the
    query type: ``bool(selection)`` refuses typed
    (``selection_bool_ambiguous``) because ``a and b`` on a lazily-composable
    AST silently discards the left operand. Resolve first;
    ``ResolvedSelection`` is truthy on "has any selected element".
    """

    __slots__ = ("_direction", "_kind", "_node")

    _direction: str
    _kind: str
    _node: Any

    def __init__(self, node: Any, *, kind: str, direction: str = "forward") -> None:
        """Freeze one query AST node (internal constructor).

        Users build selections through producers (``__selection__`` lifts,
        ``tl.units`` / ``tl.params`` / ``tl.random_selection``) and operators,
        never by constructing AST nodes directly.
        """

        if kind not in _SELECTION_KINDS:
            raise ValueError(f"selection kind must be one of {_SELECTION_KINDS}; got {kind!r}.")
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_direction", direction)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("Selection is frozen; compose a new selection instead.")

    @property
    def kind(self) -> str:
        """Return the selection kind (``ACT`` / ``PARAM`` / ``EDGE``)."""

        return self._kind

    def __selection__(self) -> Selection:
        """Pass anywhere a selection is expected."""

        return self

    def __bool__(self) -> bool:
        """Refuse ambiguous truthiness on the lazily-composable query type."""

        raise SelectionError(
            "bool() on a Selection query is ambiguous: `a and b` / `a or b` would "
            "silently discard an operand instead of composing. Use `a & b` / `a | b`, "
            "or resolve first (`selection.resolve(trace)`) and test the resolved "
            "selection.",
            code="selection_bool_ambiguous",
        )

    def __repr__(self) -> str:
        """Return the stable, readable AST spelling (pinned by test)."""

        return f"Selection[{self._kind}]({self._node!r})"

    # Boolean composition ---------------------------------------------------

    def __or__(self, other: Any) -> Any:
        return _compose_any("or", self, other)

    def __ror__(self, other: Any) -> Any:
        return _compose_any("or", other, self)

    def __and__(self, other: Any) -> Any:
        return _compose_any("and", self, other)

    def __rand__(self, other: Any) -> Any:
        return _compose_any("and", other, self)

    def __sub__(self, other: Any) -> Any:
        return _compose_any("sub", self, other)

    def __rsub__(self, other: Any) -> Any:
        return _compose_any("sub", other, self)

    def __invert__(self) -> Selection:
        """Mask complement per touched site (uniform; never predicate negation)."""

        return Selection(
            _Combinator("invert", (self._node,)), kind=self._kind, direction=self._direction
        )

    # Resolution ------------------------------------------------------------

    def resolve(self, trace: Trace) -> ResolvedSelection:
        """Resolve this query against one trace's site lifecycle.

        Returns
        -------
        ResolvedSelection
            Frozen (family, masks) pair. Empty results are LEGAL and
            first-class — emptiness is disclosure, never an error.

        Raises
        ------
        SelectionError
            ``code="selection_unresolvable"`` with a closed ``reason`` field
            (``site_not_in_trace`` / ``value_not_saved`` / ``non_tensor_site``
            / ``no_index_space`` / ``mask_shape_mismatch`` /
            ``facet_write_mask_unavailable`` / ``population_too_small`` /
            ``multipass_bare_label`` / ``value_criterion_invalid`` /
            ``basis_dim_mismatch``).
        """

        return _resolve_node(self._node, trace, self._kind)


# ---------------------------------------------------------------------------
# The producer operator mixin + lifting.
# ---------------------------------------------------------------------------


class _SelectionOperand:
    """Operator mixin for region-shaped producers (``__slots__``-safe).

    Supplies ``|  &  -  ~`` plus reflected forms that lift ``self`` via
    ``__selection__()`` and compose — so ``box | box``, ``box | facet``,
    ``facet - unit-term`` all yield a :class:`Selection` with no explicit
    conversion call. Implementers define ``__selection__``.
    """

    __slots__ = ()

    def __selection__(self) -> Any:
        """Lift this producer to a Selection."""

        raise NotImplementedError

    def __or__(self, other: Any) -> Any:
        return _compose_any("or", self, other)

    def __ror__(self, other: Any) -> Any:
        return _compose_any("or", other, self)

    def __and__(self, other: Any) -> Any:
        return _compose_any("and", self, other)

    def __rand__(self, other: Any) -> Any:
        return _compose_any("and", other, self)

    def __sub__(self, other: Any) -> Any:
        return _compose_any("sub", self, other)

    def __rsub__(self, other: Any) -> Any:
        return _compose_any("sub", other, self)

    def __invert__(self) -> Selection:
        return ~self.__selection__()


def _lift(value: Any) -> Selection | ResolvedSelection | None:
    """Lift a composable operand, or return ``None`` when not selection-shaped."""

    if isinstance(value, (Selection, ResolvedSelection)):
        return value
    lift = getattr(value, "__selection__", None)
    if lift is None:
        return None
    lifted = lift()
    if not isinstance(lifted, (Selection, ResolvedSelection)):
        raise TypeError(
            f"__selection__ on {type(value).__name__} must return a Selection; "
            f"got {type(lifted).__name__}."
        )
    return lifted


def _check_kinds(op: str, left_kind: str, right_kind: str) -> None:
    """Refuse incoherent mixed-kind composition (closed matrix, S2-homed)."""

    if left_kind != right_kind:
        raise SelectionError(
            f"cannot compose a {left_kind} selection with a {right_kind} selection "
            f"under {_OPERATOR_SYMBOLS[op]!r}: the kind-combination matrix refuses "
            "mixed kinds (future endpoint adapters are named possibilities, not "
            "promises).",
            code="selection_kind_incompatible",
            left_kind=left_kind,
            right_kind=right_kind,
            operator=_OPERATOR_SYMBOLS[op],
        )


def _check_directions(left_direction: str, right_direction: str) -> None:
    """Refuse forward x backward composition (inherited refusal)."""

    if left_direction != right_direction:
        from .intervention.errors import SelectorCompositionError

        raise SelectorCompositionError(
            "Cross-graph composition not supported: a forward selection and a "
            "backward selection cannot be combined."
        )


def _compose_any(op: str, left: Any, right: Any) -> Any:
    """Compose two operands under one boolean operator (the ONE dispatch point).

    query OP query -> query (AST node). resolved OP resolved, SAME trace ->
    resolved (mask compose now); DIFFERENT traces -> typed refusal. resolved
    OP query -> the query resolves against the resolved side's trace, then
    composes.
    """

    lifted_left = _lift(left)
    lifted_right = _lift(right)
    if lifted_left is None or lifted_right is None:
        return NotImplemented

    if isinstance(lifted_left, ResolvedSelection) or isinstance(lifted_right, ResolvedSelection):
        resolved_left, resolved_right = _align_resolved_operands(lifted_left, lifted_right)
        _check_kinds(op, resolved_left._kind, resolved_right._kind)
        return _compose_resolved(op, resolved_left, resolved_right)

    if not isinstance(lifted_left, Selection) or not isinstance(lifted_right, Selection):
        raise RuntimeError("query composition operands must both be Selection here")
    return _compose_queries(op, lifted_left, lifted_right)


def _align_resolved_operands(
    lifted_left: Any, lifted_right: Any
) -> tuple[ResolvedSelection, ResolvedSelection]:
    """Return both operands resolved on one shared trace (mixed sides resolve now)."""

    if isinstance(lifted_left, ResolvedSelection) and isinstance(lifted_right, ResolvedSelection):
        if lifted_left._trace is not lifted_right._trace:
            raise SelectionError(
                "cannot compose resolved selections bound to different traces. "
                "Re-resolve one side against the other's trace first.",
                code="selection_trace_mismatch",
            )
        return lifted_left, lifted_right
    if isinstance(lifted_left, ResolvedSelection):
        if not isinstance(lifted_right, Selection):
            raise RuntimeError("mixed composition lost its Selection operand")
        return lifted_left, lifted_right.resolve(lifted_left._trace)
    if not isinstance(lifted_left, Selection) or not isinstance(lifted_right, ResolvedSelection):
        raise RuntimeError("mixed composition lost its Selection operand")
    return lifted_left.resolve(lifted_right._trace), lifted_right


def _compose_queries(op: str, lifted_left: Selection, lifted_right: Selection) -> Selection:
    """Compose two query selections into one AST node (n-ary flatten for or)."""

    _check_kinds(op, lifted_left._kind, lifted_right._kind)
    _check_directions(lifted_left._direction, lifted_right._direction)
    if op == "or":
        # n-ary flatten for constant depth under chained composition.
        operands: list[Any] = []
        for side in (lifted_left, lifted_right):
            node = side._node
            if isinstance(node, _Combinator) and node.op == "or":
                operands.extend(node.operands)
            else:
                operands.append(node)
        node_out: Any = _Combinator("or", tuple(operands))
    else:
        node_out = _Combinator(op, (lifted_left._node, lifted_right._node))
    return Selection(node_out, kind=lifted_left._kind, direction=lifted_left._direction)


def _compose_resolved(
    op: str, left: ResolvedSelection, right: ResolvedSelection
) -> ResolvedSelection:
    """Compose two same-trace resolved selections (family + element rules)."""

    left_by_key = {entry.site_key: entry for entry in left._entries}
    right_by_key = {entry.site_key: entry for entry in right._entries}
    if op == "or":
        entries = _compose_entries_union(left_by_key, right_by_key)
    elif op == "and":
        entries = _compose_entries_intersection(left_by_key, right_by_key)
    elif op == "sub":
        entries = _compose_entries_difference(left_by_key, right_by_key)
    else:
        raise ValueError(f"unknown operator {op!r}")
    return ResolvedSelection(left._trace, left._kind, entries)


def _compose_entries_union(
    left_by_key: dict[Any, SiteEntry], right_by_key: dict[Any, SiteEntry]
) -> list[SiteEntry]:
    """Union entries: every touched site survives; shared sites join masks."""

    entries: list[SiteEntry] = []
    for key, entry in left_by_key.items():
        other = right_by_key.get(key)
        if other is None:
            entries.append(entry)
        else:
            entries.append(_entry_compose(entry, other, _mask_union, _join_relation))
    for key, entry in right_by_key.items():
        if key not in left_by_key:
            entries.append(entry)
    return entries


def _compose_entries_intersection(
    left_by_key: dict[Any, SiteEntry], right_by_key: dict[Any, SiteEntry]
) -> list[SiteEntry]:
    """Intersect entries; shared sites stay touched even when element-empty."""

    entries: list[SiteEntry] = []
    for key, entry in left_by_key.items():
        other = right_by_key.get(key)
        if other is not None:
            entries.append(_entry_compose(entry, other, _mask_intersect, _join_relation))
    return entries


def _compose_entries_difference(
    left_by_key: dict[Any, SiteEntry], right_by_key: dict[Any, SiteEntry]
) -> list[SiteEntry]:
    """Difference entries; subtraction never un-touches: fam(A - B) = fam(A)."""

    entries: list[SiteEntry] = []
    for key, entry in left_by_key.items():
        other = right_by_key.get(key)
        if other is None:
            entries.append(entry)
        else:
            entries.append(_entry_compose(entry, other, _mask_difference, _difference_relation))
    return entries


def _entry_compose(
    left: SiteEntry,
    right: SiteEntry,
    mask_op: Any,
    relation_op: Any,
) -> SiteEntry:
    """Compose two entries at one shared site."""

    if left.shape != right.shape:
        raise SelectionError(
            f"site {left.site_key!r} resolved with two different index spaces "
            f"({left.shape} vs {right.shape}); the selections cannot compose.",
            code="selection_unresolvable",
            reason="mask_shape_mismatch",
            site_key=left.site_key,
        )
    source = left.provenance.source or right.provenance.source
    return SiteEntry(
        kind=left.kind,
        site_key=left.site_key,
        provenance=SelectionProvenance(
            relation=relation_op(left.provenance.relation, right.provenance.relation),
            source=source,
        ),
        _mask=mask_op(left._mask, right._mask),
        structural_site_key=left.structural_site_key or right.structural_site_key,
    )


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------


def _unresolvable(
    reason: str,
    message: str,
    *,
    code: str = "selection_unresolvable",
    **fields: Any,
) -> SelectionError:
    """Build the closed-reason resolver refusal.

    ``code`` is always the contracted ``selection_unresolvable``; the explicit
    keyword exists so a raise site can spell the code visibly (the S-17
    census counts site-visible codes, the ``_encoding_error`` pattern).
    """

    return SelectionError(
        message,
        code=code,
        reason=reason,
        **fields,
    )


def _site_shape(op: Any) -> tuple[int, ...]:
    """Return one ACT site's output index space, refusing typed when absent."""

    shape = getattr(op, "shape", None)
    if shape is None:
        out = getattr(op, "out", None)
        if out is not None and not isinstance(out, torch.Tensor):
            raise _unresolvable(
                "non_tensor_site",
                f"site {op.label!r} has a non-tensor output; element-masked "
                "selections address single-tensor outputs only.",
                site=op.label,
            )
        raise _unresolvable(
            "no_index_space",
            f"site {op.label!r} has no known output index space (shape unknown).",
            site=op.label,
        )
    if not isinstance(shape, tuple) or not all(isinstance(extent, int) for extent in shape):
        raise _unresolvable(
            "no_index_space",
            f"site {op.label!r} has no usable output index space (shape {shape!r}).",
            site=op.label,
        )
    return shape


def _act_entry(op: Any, mask: _Mask, relation: str, source: str) -> SiteEntry:
    """Build one ACT site entry for a resolved op."""

    layer_label = getattr(op, "layer_label", None) or op.label
    pass_index = getattr(op, "pass_index", 1) or 1
    return SiteEntry(
        kind="ACT",
        site_key=(layer_label, pass_index),
        provenance=SelectionProvenance(relation=relation, source=source),
        _mask=mask,
        # The L1 structural-position identity lives on Op as ``site_key``
        # (site_key_v1 strings); SiteEntry's field keeps the qualified name
        # because SiteEntry.site_key is already the (label, pass) address.
        structural_site_key=getattr(op, "site_key", None),
    )


def _forward_ops(trace: Any) -> tuple[Any, ...]:
    """Return the trace's forward ops in execution order."""

    from .intervention.resolver import _iter_sites

    return tuple(_iter_sites(trace, "forward"))


def _find_act_ops(trace: Any, site: str) -> tuple[Any, ...]:
    """Return the ops matching one site spelling (label / layer label / io role)."""

    matches = []
    for op in _forward_ops(trace):
        if (
            op.label == site
            or getattr(op, "layer_label", None) == site
            or getattr(op, "io_role", None) == site
        ):
            matches.append(op)
    return tuple(matches)


def _resolve_node(node: Any, trace: Any, kind: str) -> ResolvedSelection:
    """Resolve one AST node against a trace."""

    if isinstance(node, _Combinator):
        return _resolve_combinator(node, trace, kind)
    for term_type, resolver in _TERM_RESOLVERS:
        if isinstance(node, term_type):
            return resolver(node, trace)
    raise TypeError(f"unknown selection AST node {type(node).__name__}")


def _resolve_combinator(node: _Combinator, trace: Any, kind: str) -> ResolvedSelection:
    """Resolve an operator node by folding its resolved operands."""

    if node.op == "invert":
        return ~_resolve_node(node.operands[0], trace, kind)
    resolved = [_resolve_node(operand, trace, kind) for operand in node.operands]
    result = resolved[0]
    for operand in resolved[1:]:
        result = _compose_resolved(node.op, result, operand)
    return result


def _resolve_selector_term(node: _SelectorTerm, trace: Any) -> ResolvedSelection:
    """Resolve a lifted selector to whole-site masks over its matched sites."""

    from .intervention.resolver import (
        _iter_sites,
        _resolve_unchecked,
        _selector_resolution_direction,
    )

    direction = _selector_resolution_direction(node.selector)
    sites = tuple(_iter_sites(trace, direction))
    matched = _resolve_unchecked(sites, node.selector, strict=False)
    entries = [
        _act_entry(op, _mask_whole(_site_shape(op)), "exact", f"selector {node.selector!r}")
        for op in matched
    ]
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_box_term(node: _BoxTerm, trace: Any) -> ResolvedSelection:
    """Resolve a receptive-field interval hull to its exact-as-a-set mask."""

    box = node.box
    matches = _find_act_ops(trace, box.io_role)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"receptive-field box addresses io role {box.io_role!r}, which is not a "
            "site of this trace.",
            site=box.io_role,
        )
    op = matches[0]
    shape = _site_shape(op)
    if shape != tuple(box.input_shape):
        raise _unresolvable(
            "mask_shape_mismatch",
            f"receptive-field box space {tuple(box.input_shape)!r} does not match "
            f"site {op.label!r} output space {shape!r}.",
            site=op.label,
        )
    relation = "exact" if (box.exact and not box.sparse_possible) else "upper_bound"
    mask = _mask_from_slices(shape, box.slices())
    source = f"rf hull of {box.op_label!r} unit {box.unit!r}"
    return ResolvedSelection(trace, "ACT", [_act_entry(op, mask, relation, source)])


def _resolve_gradient_term(node: _GradientTerm, trace: Any) -> ResolvedSelection:
    """Resolve an empirical gradient receptive field (exact-set term)."""

    gradient = node.gradient_rf
    matches = _find_act_ops(trace, gradient.io_role)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"gradient receptive field addresses io role {gradient.io_role!r}, which "
            "is not a site of this trace.",
            site=gradient.io_role,
        )
    op = matches[0]
    shape = _site_shape(op)
    support = gradient.support_mask
    if tuple(support.shape) != shape:
        raise _unresolvable(
            "mask_shape_mismatch",
            f"gradient support mask shape {tuple(support.shape)!r} does not match "
            f"site {op.label!r} output space {shape!r}.",
            site=op.label,
        )
    mask = _mask_from_dense(shape, support.bool())
    source = f"gradient rf of {gradient.op_label!r} unit {gradient.unit!r}"
    return ResolvedSelection(trace, "ACT", [_act_entry(op, mask, "exact", source)])


def _resolve_facet_term(node: _FacetTerm, trace: Any) -> ResolvedSelection:
    """Resolve a facet's write region over its home site."""

    spec = node.spec
    home_label = spec.home_label
    if home_label is None:
        raise _unresolvable(
            "facet_write_mask_unavailable",
            "facet spec has no stable home label to resolve against.",
        )
    matches = _find_act_ops(trace, home_label)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"facet home {home_label!r} is not a site of this trace.",
            site=home_label,
        )
    pass_index = getattr(spec, "pass_index", None)
    if pass_index is not None:
        matches = tuple(op for op in matches if getattr(op, "pass_index", 1) == pass_index)
        if not matches:
            raise _unresolvable(
                "site_not_in_trace",
                f"facet home {home_label!r} pass {pass_index} is not a site of this trace.",
                site=home_label,
            )
    op = matches[0]
    shape = _site_shape(op)
    try:
        write_mask = spec.write_mask()
    except TypeError as exc:
        raise _unresolvable(
            "non_tensor_site",
            f"facet home {home_label!r} has a non-tensor output: {exc}",
            site=home_label,
        ) from exc
    except Exception as exc:
        try:
            home_out = getattr(op, "out", None)
        except TorchLensError:  # unsaved payload reads raise typed
            home_out = None
        if home_out is None:
            raise _unresolvable(
                "value_not_saved",
                f"facet {getattr(spec, 'recipe_id', '<unknown>')!r} needs the saved "
                f"home value at {home_label!r}, which this capture did not retain.",
                site=home_label,
            ) from exc
        raise _unresolvable(
            "facet_write_mask_unavailable",
            f"facet {getattr(spec, 'recipe_id', '<unknown>')!r} cannot produce its "
            f"write mask: {exc}",
            site=home_label,
        ) from exc
    if tuple(write_mask.shape) != shape:
        raise _unresolvable(
            "mask_shape_mismatch",
            f"facet write mask shape {tuple(write_mask.shape)!r} does not match "
            f"site {op.label!r} output space {shape!r}.",
            site=op.label,
        )
    mask = _mask_from_dense(shape, write_mask.bool().cpu())
    source = f"facet {getattr(spec, 'recipe_id', '<unknown>')!r} write region"
    return ResolvedSelection(trace, "ACT", [_act_entry(op, mask, "exact", source)])


def _resolve_param_term(node: _ParamTerm, trace: Any) -> ResolvedSelection:
    """Resolve a named-parameter element region."""

    target = None
    for param in getattr(trace, "params", ()):
        if getattr(param, "address", None) == node.name or param.name == node.name:
            target = param
            break
    if target is None:
        raise _unresolvable(
            "site_not_in_trace",
            f"parameter {node.name!r} is not recorded on this trace.",
            site=node.name,
        )
    shape = getattr(target, "shape", None)
    if not isinstance(shape, tuple):
        raise _unresolvable(
            "no_index_space",
            f"parameter {node.name!r} has no usable index space (shape {shape!r}).",
            site=node.name,
        )
    if node.mask is None:
        mask = _mask_whole(shape)
    else:
        if tuple(node.mask.shape) != shape:
            raise _unresolvable(
                "mask_shape_mismatch",
                f"supplied mask shape {tuple(node.mask.shape)!r} does not match "
                f"parameter {node.name!r} shape {shape!r}.",
                site=node.name,
            )
        mask = _mask_from_dense(shape, node.mask.bool())
    address = getattr(target, "address", None) or target.name
    entry = SiteEntry(
        kind="PARAM",
        site_key=(address,),
        provenance=SelectionProvenance(relation="exact", source=f"params({node.name!r})"),
        _mask=mask,
    )
    return ResolvedSelection(trace, "PARAM", [entry])


def _resolve_unit_term(node: _UnitTerm, trace: Any) -> ResolvedSelection:
    """Resolve an explicit site + index set."""

    matches = _find_act_ops(trace, node.site)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"site {node.site!r} is not a site of this trace.",
            site=node.site,
        )
    if len(matches) > 1 and len({getattr(op, "layer_label", None) for op in matches}) == 1:
        # A layer-wide spelling of a multi-pass layer names N distinct ops;
        # units() addresses ONE op's index space, so never guess a pass.
        from .intervention.resolver import multipass_bare_label_message

        layer_label = getattr(matches[0], "layer_label", None) or node.site
        pass_indices = sorted(int(getattr(op, "pass_index", 1) or 1) for op in matches)
        raise _unresolvable(
            "multipass_bare_label",
            multipass_bare_label_message(layer_label, pass_indices),
            site=node.site,
            pass_indices=tuple(pass_indices),
        )
    entries = []
    for op in matches:
        shape = _site_shape(op)
        if node.index_mask is not None:
            if tuple(node.index_mask.shape) != shape:
                raise _unresolvable(
                    "mask_shape_mismatch",
                    f"units mask shape {tuple(node.index_mask.shape)!r} does not "
                    f"match site {op.label!r} output space {shape!r}.",
                    site=op.label,
                )
            mask = _mask_from_dense(shape, node.index_mask.bool())
        else:
            if node.indices is None:
                raise RuntimeError("units node has neither index_mask nor indices")
            dense = torch.zeros(shape, dtype=torch.bool)
            for coordinates in node.indices:
                if len(coordinates) != len(shape):
                    raise _unresolvable(
                        "mask_shape_mismatch",
                        f"unit index {coordinates!r} has {len(coordinates)} coordinate(s) but "
                        f"site {op.label!r} output space {shape!r} has rank {len(shape)}: "
                        "units() takes one integer per output axis, position-for-position. "
                        "Partial coordinates never broadcast across the remaining axes; to "
                        "select every position along an axis, pass a bool mask over the full "
                        "output space instead.",
                        code="selection_unresolvable",
                        site=op.label,
                        given_rank=len(coordinates),
                        expected_rank=len(shape),
                    )
                if any(
                    coordinate < 0 or coordinate >= extent
                    for coordinate, extent in zip(coordinates, shape, strict=True)
                ):
                    raise _unresolvable(
                        "mask_shape_mismatch",
                        f"unit index {coordinates!r} is outside site {op.label!r} "
                        f"output space {shape!r}.",
                        site=op.label,
                    )
                dense[coordinates] = True
            mask = _mask_from_dense(shape, dense)
        entries.append(_act_entry(op, mask, "exact", f"units({node.site!r})"))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_whole_site_term(node: _WholeSiteTerm, trace: Any) -> ResolvedSelection:
    """Resolve an Op/Layer whole-output lift."""

    matches = _find_act_ops(trace, node.site_label)
    if node.pass_index is not None:
        matches = tuple(op for op in matches if getattr(op, "pass_index", 1) == node.pass_index)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"site {node.site_label!r} is not a site of this trace.",
            site=node.site_label,
        )
    entries = [
        _act_entry(op, _mask_whole(_site_shape(op)), "exact", f"site({node.site_label!r})")
        for op in matches
    ]
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_random_term(node: _RandomTerm, trace: Any) -> ResolvedSelection:
    """Resolve a seeded, size-matched random control selection."""

    like = _resolve_operand_for_random(node.like, trace)
    within = _resolve_operand_for_random(node.within, trace)
    requested = sum(entry.selected_count for entry in like._entries)
    population: list[tuple[SiteEntry, int]] = []
    for entry in within._entries:
        flat = entry._mask._dense_ro().reshape(-1)
        for index in torch.nonzero(flat, as_tuple=False).reshape(-1).tolist():
            population.append((entry, index))
    if len(population) < requested:
        raise _unresolvable(
            "population_too_small",
            f"random_selection needs {requested} elements but the `within` "
            f"population has only {len(population)}.",
            requested=requested,
            available=len(population),
        )
    rng = _random_module.Random(node.seed)
    sampled = rng.sample(range(len(population)), requested)
    dense_by_key: dict[tuple[Any, ...], tuple[SiteEntry, torch.Tensor]] = {}
    for position in sampled:
        entry, flat_index = population[position]
        if entry.site_key not in dense_by_key:
            dense_by_key[entry.site_key] = (
                entry,
                torch.zeros(entry.shape, dtype=torch.bool),
            )
        dense_by_key[entry.site_key][1].reshape(-1)[flat_index] = True
    entries = [
        SiteEntry(
            kind=entry.kind,
            site_key=entry.site_key,
            provenance=SelectionProvenance(
                relation="exact", source=f"random_selection(seed={node.seed})"
            ),
            _mask=_mask_from_dense(entry.shape, dense),
            structural_site_key=entry.structural_site_key,
        )
        for entry, dense in dense_by_key.values()
    ]
    return ResolvedSelection(trace, within._kind, entries)


def _resolve_operand_for_random(operand: Any, trace: Any) -> ResolvedSelection:
    """Resolve one random_selection operand, honoring trace binding."""

    if isinstance(operand, ResolvedSelection):
        if operand._trace is not trace:
            raise SelectionError(
                "random_selection operand is bound to a different trace.",
                code="selection_trace_mismatch",
            )
        return operand
    if not isinstance(operand, Selection):
        raise RuntimeError("random_selection operand must lift to a Selection")
    return operand.resolve(trace)


# ---------------------------------------------------------------------------
# Stage-1 producer constructors.
# ---------------------------------------------------------------------------


def units(
    site: str,
    indices: Iterable[tuple[int, ...] | int] | torch.Tensor,
) -> Selection:
    """Select an explicit site + index set (unit-term producer).

    Parameters
    ----------
    site:
        Site spelling: a pass-qualified op label (``'relu_1_2:1'``), a bare
        layer label (single-pass layers only — a bare label naming a
        multi-pass layer refuses ``selection_unresolvable`` /
        ``multipass_bare_label``), or an input io role.
    indices:
        Either a bool mask over the site's output index space, or an iterable
        of integer coordinate tuples (ints accepted for 1-d sites).
    """

    if not isinstance(site, str) or not site:
        raise ValueError("units(site, ...) requires a non-empty site label string.")
    if isinstance(indices, torch.Tensor):
        if indices.dtype != torch.bool:
            raise ValueError(
                "units(site, indices) tensor form must be a bool mask over the "
                f"site's output index space; got dtype {indices.dtype}."
            )
        return Selection(_UnitTerm(site=site, indices=None, index_mask=indices.clone()), kind="ACT")
    normalized: list[tuple[int, ...]] = []
    for coordinates in indices:
        if isinstance(coordinates, int):
            coordinates = (coordinates,)
        coordinates = tuple(int(coordinate) for coordinate in coordinates)
        if any(coordinate < 0 for coordinate in coordinates):
            raise ValueError(f"units indices must be non-negative; got {coordinates!r}.")
        normalized.append(coordinates)
    return Selection(_UnitTerm(site=site, indices=tuple(normalized), index_mask=None), kind="ACT")


def params(name: str, mask: torch.Tensor | None = None) -> Selection:
    """Select a named-parameter element region (param-term producer).

    ``mask=None`` selects the whole parameter. Fenced from
    ``module.parameters()``: this addresses the TRACE's recorded parameter
    geometry by name/address.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("params(name, ...) requires a non-empty parameter name string.")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("params mask must be a bool tensor over the parameter shape.")
        mask = mask.clone()
    return Selection(_ParamTerm(name=name, mask=mask), kind="PARAM")


def random_selection(
    *,
    like: Any,
    within: Any,
    seed: int,
) -> Selection:
    """Seeded, size-matched random control selection.

    Both operands resolve at ``resolve(trace)`` time; the sampling population
    is ``within``'s selected-element set; the result kind is ``within``'s
    kind; ``|result| == |like|`` sampled without replacement. A population
    smaller than ``|like|`` (including an empty complement) refuses typed
    (``selection_unresolvable``, ``reason="population_too_small"``).
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("random_selection seed must be a non-negative int.")
    like_lifted = _lift(like)
    within_lifted = _lift(within)
    if like_lifted is None or within_lifted is None:
        raise ValueError(
            "random_selection operands must be selections or region-shaped "
            "producers implementing __selection__."
        )
    kind = within_lifted._kind if isinstance(within_lifted, Selection) else within_lifted._kind
    return Selection(_RandomTerm(like=like_lifted, within=within_lifted, seed=seed), kind=kind)


# ---------------------------------------------------------------------------
# Lift helpers used by producer classes (lazy to keep this module light).
# ---------------------------------------------------------------------------


def _selection_from_selector(selector: Any) -> Selection:
    """Lift a BaseSelector as a selector-term (predicate meaning intact)."""

    from .intervention.resolver import _selector_resolution_direction

    direction = _selector_resolution_direction(selector)
    return Selection(_SelectorTerm(selector=selector), kind="ACT", direction=direction)


def _selection_from_box(box: Any) -> Selection:
    """Lift a ReceptiveFieldBox as an ACT hull term."""

    return Selection(_BoxTerm(box=box), kind="ACT")


def _selection_from_gradient(gradient_rf: Any) -> Selection:
    """Lift a GradientReceptiveField as an exact empirical ACT term."""

    return Selection(_GradientTerm(gradient_rf=gradient_rf), kind="ACT")


def _selection_from_facet(spec: Any) -> Selection:
    """Lift a FacetSpec's write region as an ACT term."""

    return Selection(_FacetTerm(spec=spec), kind="ACT")


def _selection_from_op(op: Any) -> Selection:
    """Lift one Op (whole-output, one pass)."""

    layer_label = getattr(op, "layer_label", None) or op.label
    pass_index = getattr(op, "pass_index", 1) or 1
    return Selection(_WholeSiteTerm(site_label=layer_label, pass_index=pass_index), kind="ACT")


def _selection_from_layer(layer: Any) -> Selection:
    """Lift one Layer (whole-output, ALL passes)."""

    layer_label = getattr(layer, "layer_label", None) or layer.label
    return Selection(_WholeSiteTerm(site_label=layer_label, pass_index=None), kind="ACT")


# ---------------------------------------------------------------------------
# Stage 2: Selection-targeted edits — THE NORMATIVE MASK-APPLICATION CONTRACT.
# Helpers stay mask-oblivious; the ENGINE owns masking (edit-then-scatter):
# the edit hook computes its full replacement tensor exactly as today, then
# the engine applies ``torch.where(mask, edited, original)`` on a FRESH
# tensor — never in-place on, and never a view aliasing, the stored capture
# value. Whole-site masks short-circuit the scatter (exactly today's
# behavior). No broadcasting in v1.
# ---------------------------------------------------------------------------


def _apply_invalid(
    reason: str,
    message: str,
    *,
    code: str = "selection_apply_invalid",
    **fields: Any,
) -> SelectionError:
    """Build the closed-reason mask-application refusal.

    ``code`` is always the contracted ``selection_apply_invalid``; the
    explicit keyword exists so a raise site can spell the code visibly (the
    S-17 census counts site-visible codes, the ``_encoding_error`` pattern).
    """

    return SelectionError(message, code=code, reason=reason, **fields)


# _validate_edited / _masked_factory / _derive_masked_edit moved to
# torchlens/intervention/masked_edit.py (R43 file-size ratchet); the
# do-plan below imports the derivation lazily.


def _warn_if_dense_subspace_edit(resolved: ResolvedSelection) -> None:
    """Point-of-use disclosure: editing a DENSE direction's support set is
    full-axis ablation (set semantics), never a projection along the
    direction (list-A row 8; the projection-valued edit is a named fork,
    not a shipped capability). The marker rides ``provenance.source``, so
    the audit record discloses it either way; this warning surfaces it at
    the ``do()`` call itself."""

    import warnings as _warnings

    from .selection_subspace import DENSE_SUPPORT_NOTE

    if any(DENSE_SUPPORT_NOTE in entry.provenance.source for entry in resolved):
        from .errors import TorchLensWarning

        _warnings.warn(
            TorchLensWarning(
                "do() on a dense-direction subspace selection edits EVERY "
                "element of the bound axis (the documented support-set "
                "semantics), not the component along the direction; "
                "projection-valued edits are not a shipped capability. "
                "Remedy: use a sparse direction (or tol=) to target a "
                "subset, or accept the full-axis edit knowingly",
                code="dense_subspace_full_axis_edit",
            ),
            stacklevel=4,
        )


def build_selection_do_plan(
    trace: Any, selection_like: Any, edit: Any
) -> tuple[ResolvedSelection, list[dict[str, Any]], dict[str, Any]]:
    """Resolve a selection target and derive the per-site edit plan.

    Returns ``(resolved, [{"op", "entry", "is_leaf", "edit"}, ...], audit_record)``.
    ACT selections only: PARAM edits route through parameter substitution
    (``intervention/param_substitution.py`` — the value each consumer sees is
    substituted on the replay engine, the live parameter is never written);
    EDGE selections are stage-3 edge-substitution territory.
    """

    if edit is None:
        raise ValueError(
            "do(selection, edit) requires an edit: pass an Edit/HelperSpec "
            "(e.g. tl.zero_ablate()), a hook callable, or a replacement tensor."
        )
    resolved = _resolve_do_target(trace, selection_like)
    _warn_if_dense_subspace_edit(resolved)

    from .intervention.masked_edit import _derive_masked_edit
    from .intervention.types import HelperSpec

    if not isinstance(edit, HelperSpec) and not callable(edit):
        # Raw replacement TENSORS route through the shipped replace_with helper and
        # obey the scatter contract; anything else refuses with the string-label
        # path's validation (a scalar lifted through the tensor-only helper used to
        # crash bare at push time).
        if not isinstance(edit, torch.Tensor):
            raise ValueError(
                f"do(selection, edit) got a {type(edit).__name__} replacement value; "
                "replacement values must be tensors matching the site output. For a "
                "constant fill over the selected elements pass a full-shape tensor "
                "(e.g. torch.full_like(site_out, c)), or use an edit helper such as "
                "tl.zero_ablate() or tl.scale()."
            )
        from .intervention.predicates import replace_with

        edit = replace_with(edit)

    plan: list[dict[str, Any]] = []
    for entry in resolved:
        layer_label, pass_index = entry.site_key
        ops = [
            op
            for op in _find_act_ops(trace, layer_label)
            if getattr(op, "pass_index", 1) == pass_index
        ]
        if not ops:
            raise _unresolvable(
                "site_not_in_trace",
                f"resolved site {entry.site_key!r} is not present on this trace.",
                site=layer_label,
            )
        op = ops[0]
        plan.append(
            {
                "op": op,
                "entry": entry,
                "is_leaf": getattr(op, "func", None) is None,
                "edit": _derive_masked_edit(edit, entry, resolved.resolve_digest, op.label),
            }
        )

    # The per-site "source" disclosure (the A04 dense-subspace point-of-use
    # stamp) is admitted as an OPTIONAL string field in the closed
    # load-validator site schema as of tlspec v9 (C07 amendment; the pre-v9
    # validator refused it, so a saved selection-intervened artifact refused
    # its own load -- C03 measured defect, field_intent.tsv row).
    audit: dict[str, Any] = {
        "kind": resolved.kind,
        "selection_repr": repr(selection_like),
        "resolve_digest": resolved.resolve_digest,
        "sites": [
            {
                "site_key": repr(entry.site_key),
                "relation": entry.provenance.relation,
                "selected": entry.selected_count,
                "source": entry.provenance.source,
            }
            for entry in resolved
        ],
        "edit": getattr(edit, "helper_name", getattr(edit, "__name__", repr(type(edit)))),
    }
    source_identity = getattr(edit, "kwargs", None)
    if getattr(edit, "helper_name", None) == "patch_from" and source_identity:
        audit["patch_source"] = dict(source_identity)
    return resolved, plan, audit


def _resolve_do_target(trace: Any, selection_like: Any) -> ResolvedSelection:
    """Resolve the ``do()`` target on this trace, refusing PARAM/EDGE kinds."""

    lifted = _lift(selection_like)
    if lifted is None:
        raise RuntimeError("build_selection_do_plan requires a selection-shaped input")
    if isinstance(lifted, ResolvedSelection):
        if lifted._trace is not trace:
            raise SelectionError(
                "the resolved selection is bound to a different trace; resolve "
                "against this trace first, or bridge explicitly with "
                "resolved.align_to(trace) (cross-run alignment on L1 site "
                "keys, same-policy captures only).",
                code="selection_trace_mismatch",
            )
        resolved = lifted
    else:
        resolved = lifted.resolve(trace)
    if resolved.kind == "PARAM":
        raise _apply_invalid(
            "not_maskable",
            "learned-parameter edits do not ride the node-edit plan: PARAM "
            "selections apply through parameter SUBSTITUTION on the replay "
            "engine (do(tl.params(...), edit) substitutes the value each "
            "consuming op sees; the live parameter is never written).",
        )
    if resolved.kind == "EDGE":
        raise _apply_invalid(
            "not_maskable",
            "edge selections address edge substitution (stage 3), not node edits.",
        )
    return resolved


# ---------------------------------------------------------------------------
# Stage 3: EDGE selections (whole-edge granularity at introduction).
# The canonical occurrence address is (child_func_call_id, arg_kind,
# arg_path) — exactly EdgeUseRecord's discriminating fields. EdgeTable
# edge ids stay in-session accelerators, never the address.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EdgeTerm:
    """Explicit edge-occurrence set (whole-edge granularity)."""

    addresses: tuple[tuple[Any, ...], ...]
    display: tuple[str, ...] = ()

    def __repr__(self) -> str:
        if self.display:
            return f"edges({', '.join(self.display)})"
        return f"edges(n={len(self.addresses)})"


def edge_address_of(record: Any) -> tuple[Any, ...]:
    """Return one EdgeUseRecord's canonical occurrence address."""

    return (record.child_func_call_id, record.arg_kind, tuple(record.arg_path))


def _require_edge_provenance(trace: Any) -> None:
    """Refuse typed when edge provenance was not captured."""

    if not bool(getattr(trace, "intervention_ready", False)):
        raise SelectionError(
            "edge provenance requires an intervention_ready capture "
            "(EdgeUseRecords exist only under it). Re-capture with "
            "tl.trace(model, x, capture=tl.options.CaptureOptions("
            "intervention_ready=True)).",
            code="edge_provenance_unavailable",
        )


def _trace_edge_records(trace: Any) -> tuple[Any, ...]:
    """Return the trace's dataflow edge family (EdgeUseRecords, child-owned)."""

    _require_edge_provenance(trace)
    records: list[Any] = []
    for op in _forward_ops(trace):
        records.extend(getattr(op, "edge_uses", ()) or ())
    return tuple(records)


def _edge_entry(record: Any) -> SiteEntry:
    """Build one EDGE site entry (whole-edge mask over a 1-element space)."""

    return SiteEntry(
        kind="EDGE",
        site_key=edge_address_of(record),
        provenance=SelectionProvenance(
            relation="exact",
            source=f"edge {record.parent_label!r} -> {record.child_label!r}",
        ),
        _mask=_mask_whole((1,)),
    )


def _selection_from_edge(record: Any) -> Selection:
    """Lift one EdgeUseRecord as an EDGE selection term."""

    return Selection(
        _EdgeTerm(
            addresses=(edge_address_of(record),),
            display=(f"{record.parent_label}->{record.child_label}",),
        ),
        kind="EDGE",
    )


def _resolve_edge_term(node: _EdgeTerm, trace: Any) -> ResolvedSelection:
    """Resolve an explicit edge-occurrence set against one trace."""

    family = {edge_address_of(record): record for record in _trace_edge_records(trace)}
    entries = []
    for address in node.addresses:
        record = family.get(address)
        if record is None:
            raise _unresolvable(
                "site_not_in_trace",
                f"edge occurrence {address!r} is not part of this trace's dataflow edge family.",
                site=repr(address),
            )
        entries.append(_edge_entry(record))
    return ResolvedSelection(trace, "EDGE", entries)


def _edge_family_complement(resolved: ResolvedSelection) -> ResolvedSelection:
    """Complement an EDGE selection within the trace's dataflow edge family.

    The edge universe is well-defined (the trace's EdgeTable occurrences), so
    ``~`` on a RESOLVED edge selection complements against it: edges outside
    the selected set enter with whole masks; selected edges leave. (On a
    QUERY, ``~`` defers to resolve time.)
    """

    selected = {entry.site_key for entry in resolved._entries if entry.selected_count > 0}
    entries = [
        _edge_entry(record)
        for record in _trace_edge_records(resolved._trace)
        if edge_address_of(record) not in selected
    ]
    return ResolvedSelection(resolved._trace, "EDGE", entries)


#: Ordered term-type dispatch for ``_resolve_node`` (combinators handled first).
_TERM_RESOLVERS: list[tuple[type, Any]] = [
    (_SelectorTerm, _resolve_selector_term),
    (_BoxTerm, _resolve_box_term),
    (_GradientTerm, _resolve_gradient_term),
    (_FacetTerm, _resolve_facet_term),
    (_ParamTerm, _resolve_param_term),
    (_UnitTerm, _resolve_unit_term),
    (_WholeSiteTerm, _resolve_whole_site_term),
    (_RandomTerm, _resolve_random_term),
    (_EdgeTerm, _resolve_edge_term),
]


def register_term_resolver(term_type: type, resolver: Any) -> None:
    """Register one selection AST term resolver (producer-extension seam).

    Internal seam: sibling producer modules (``selection_values``) register
    their frozen term types at import time. A term can only enter an AST
    through its constructor, and the constructor lives in the registering
    module, so registration always precedes the first resolve of that term.
    """

    _TERM_RESOLVERS.append((term_type, resolver))
