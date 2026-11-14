"""Selection / intervention algebra: the dual selection types.

``Selection`` is the composable QUERY — a frozen AST over leaf terms and
boolean combinators, model/trace-independent; resolving it is explicit.
``Selection`` is returned by ``trace[selector]`` — frozen,
concrete, trace-bound: an ordered tuple of :class:`Site` rows.

Every public spelling here ships DOCUMENTED-UNSTABLE pending its
naming-session ratification (provisional-name protocol).

TWO-LEVEL DENOTATION (normative): a selection denotes the pair
(TOUCHED-SITE FAMILY, SELECTED-ELEMENT SET). Zero-mask entries are retained,
first-class, by every operator and by ``resolve()`` — an element-empty site
stays touched (this is what makes ``~`` an involution). Equality and the
resolve digest compare BOTH levels; ``.is_empty`` / ``__bool__`` report the
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

The kind vocabulary ``ACT | PARAM | EDGE`` is a CLOSED public vocabulary.
Mixed-kind composition refuses typed (``selection_kind_incompatible``); the matrix
cells change only via an explicit, documented change, never a local edit.

``Selection`` is SESSION-TIME ONLY and is never persisted.
"""

from __future__ import annotations

import hashlib
import random as _random_module
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import torch

from .._utils._record_group_table import record_member
from ..errors._base import TorchLensError
from ._display import selection_card, selection_repr, site_card, site_repr
from ._entry_order import _graph_entry_order
from ._fork_binding import bind_to_target
from ._site_record import _require_edge_provenance, bind_entries, edge_address_of, site_record
from ._site_refusals import multipass_bare_label_message
from .align import align_resolved_selection
from .errors import SelectionError, SelectorCompositionError
from .mask import (
    _Mask,
    _mask_complement,
    _mask_difference,
    _mask_equal,
    _mask_from_dense,
    _mask_from_slices,
    _mask_intersect,
    _mask_union,
    _mask_whole,
)
from .provenance import (
    SelectionProvenance,
    _compose_provenance,
    _difference_relation,
    _flip_provenance,
    _join_relation,
)
from .selectors import _grad_selection_refusal

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = [
    "Selection",
    "Selector",
    "Site",
    "param",
    "random_like",
    "units",
]

SelectionKindName = Literal["ACT", "PARAM", "EDGE"]
_SELECTION_KINDS: tuple[str, ...] = ("ACT", "PARAM", "EDGE")

_OPERATOR_SYMBOLS = {"or": "|", "and": "&", "sub": "-"}


# ---------------------------------------------------------------------------
# Site + Selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One (site, mask, provenance) row of a resolved selection.

    ``key`` is ``(layer_label, pass_index)`` for ACT sites and
    ``(parameter_address,)`` for PARAM sites. ``structural_site_key`` carries
    the L1 structural site key when present (bridging relation; ``None``
    otherwise). ``mask`` returns a FRESH dense bool materialization on every
    access — mutating the returned tensor cannot alter the selection.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> site = trace[tl.units("relu_1_2", [(0, 0, 1, 1)])][0]
    >>> site.key, site.shape, site.num_selected
    (('relu_1_2', 1), (1, 4, 8, 8), 1)
    """

    kind: str
    key: tuple[Any, ...]
    provenance: SelectionProvenance
    _mask: _Mask
    structural_site_key: Any = None
    #: Weak trace link filled by ``Selection``; outside equality and hashing.
    _trace_ref: Any = field(default=None, compare=False, repr=False)

    @property
    def record(self) -> Any:
        """The ``Op`` (ACT), ``Param`` (PARAM) or ``EdgeUseRecord`` (EDGE) this row names.

        Weak trace link: ``selection_trace_released`` once the trace is gone,
        ``selection_unresolvable`` (``site_not_in_trace``) when it has no such record.
        """

        return site_record(self, _forward_ops, _trace_edge_records)

    @property
    def mask(self) -> torch.Tensor:
        """Return a fresh dense bool mask over the site's output index space."""

        return self._mask.to_dense()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the site's output index-space shape."""

        return self._mask.shape

    @property
    def num_selected(self) -> int:
        """Return the number of selected elements at this site."""

        return self._mask.count()

    __repr__ = site_repr
    __str__ = site_card


def _entry_label(trace: Any, entry: Site) -> str:
    """Return the pass-qualified label of one row (the record's op label for ACT)."""

    if entry.kind != "ACT" or len(entry.key) != 2:
        return ":".join(str(part) for part in entry.key)
    layer_label, pass_index = entry.key
    for op in _forward_ops(trace):
        if (getattr(op, "layer_label", None) or op.label) == layer_label and (
            getattr(op, "pass_number", 1) or 1
        ) == pass_index:
            return str(op.label)
    return f"{layer_label}:{pass_index}"


class Selection:
    """Frozen, concrete, trace-bound selection: an ordered tuple of Site.

    Zero-mask (element-empty) entries are retained first-class — the family
    level of the two-level denotation. ``.is_empty`` and ``__bool__`` report the
    ELEMENT level only; read the family via iteration / ``len()``.
    Session-time only; never persisted.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> selection = trace[tl.func("conv2d")]
    >>> isinstance(selection, tl.Selection)
    True
    >>> selection.labels, selection.kind, selection.is_empty
    (('conv2d_1_1:1', 'conv2d_2_3:1'), 'ACT', False)
    """

    __slots__ = ("_digest", "_entries", "_kind", "_selector", "_trace")

    _digest: str
    _entries: tuple[Site, ...]
    _kind: str
    _selector: Any
    _trace: Any

    def __init__(
        self, trace: Any, kind: str, entries: Iterable[Site], *, selector: Any = None
    ) -> None:
        """Freeze a resolved selection (internal constructor).

        Parameters
        ----------
        trace:
            The trace the selection resolved against (identity-bound).
        kind:
            Selection kind, one of the closed ``ACT | PARAM | EDGE``.
        entries:
            Site entries; canonicalized to deterministic order.
        selector:
            The selector that produced the rows, when one did (``None`` for
            selections built by composition or complement).
        """

        if kind not in _SELECTION_KINDS:
            raise ValueError(f"selection kind must be one of {_SELECTION_KINDS}; got {kind!r}.")
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_selector", selector)
        object.__setattr__(self, "_entries", _graph_entry_order(trace, entries))
        bind_entries(trace, self._entries)
        digest = hashlib.sha256()
        digest.update(kind.encode())
        for entry in self._entries:
            digest.update(repr(entry.key).encode())
            digest.update(repr(entry.shape).encode())
            digest.update(entry._mask.canonical_bytes())
        object.__setattr__(self, "_digest", digest.hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("Selection is frozen; construct a new selection instead.")

    @property
    def kind(self) -> str:
        """Return the selection kind (``ACT`` / ``PARAM`` / ``EDGE``)."""

        return self._kind

    @property
    def selector(self) -> Any:
        """Return the selector that produced these rows, or ``None``.

        Set when the selection came from ``trace[selector]`` or a lifted
        selector; ``None`` for selections built by composition or complement.
        """

        return self._selector

    def _resolve(self, trace: Trace) -> Selection:
        """Return this selection on ``trace``: itself, or rebound onto a fork of its trace."""

        return bind_to_target(self, trace)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return one pass-qualified label per row, in row order.

        ACT rows read the op label of the trace record at their
        ``(layer_label, pass_index)`` key; PARAM and EDGE rows read the
        string form of their key.
        """

        return tuple(_entry_label(self._trace, entry) for entry in self._entries)

    @property
    def is_empty(self) -> bool:
        """Return whether NO element is selected (element level).

        A selection can be element-empty while still touching sites; the
        touched-site family stays readable via iteration / ``len()``.
        """

        return all(entry.num_selected == 0 for entry in self._entries)

    @property
    def digest(self) -> str:
        """Return the freeze-time digest of the canonical (family, masks) pair."""

        return self._digest

    def align_to(self, target: Trace) -> Selection:
        """Re-bind this resolved selection onto another trace (cross-run).

        The stage-4a cross-run door: alignment keys on the L1 structural
        site keys each entry records (position proof) AND the same-policy
        ``(layer_label, pass_index)`` address, under the L1 cross-stamp rule
        (same-policy captures only — healthy, agreeing grouping stamps on
        both sides; anything else refuses typed). Masks travel unchanged
        onto an identical index space; per-entry provenance discloses the
        cross-run origin. ``do()`` still refuses foreign resolved selections
        (``selection_trace_mismatch``) unless the fork descends from the selection's
        trace (it then rebinds itself). DOCUMENTED-UNSTABLE pending ratification.

        Raises
        ------
        SelectionError
            ``selection_alignment_invalid`` with a closed reason set
            (``kind_unsupported`` / ``grouping_stamp_degraded`` /
            ``grouping_stamp_mismatch`` / ``site_key_unavailable`` /
            ``site_not_in_target`` / ``index_space_mismatch``).
        """

        return align_resolved_selection(self, target)

    def __bool__(self) -> bool:
        """Return whether ANY element is selected (element level)."""

        return not self.is_empty

    def __len__(self) -> int:
        """Return the touched-site family size (zero-mask entries included)."""

        return len(self._entries)

    def __iter__(self) -> Iterator[Site]:
        """Iterate site entries in canonical order."""

        return iter(self._entries)

    @overload
    def __getitem__(self, index: int) -> Site: ...

    @overload
    def __getitem__(self, index: slice) -> Selection: ...

    def __getitem__(self, index: int | slice) -> Site | Selection:
        """Return one site entry by position, or a sliced Selection."""

        if isinstance(index, slice):
            return Selection(self._trace, self._kind, self._entries[index], selector=self._selector)
        return self._entries[index]

    def to_pandas(self) -> Any:
        """Return a pandas table with one row per site.

        Columns: ``label``, ``kind``, ``key``, ``num_selected``, ``numel``,
        ``shape``, ``relation``, ``source``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [
            {
                "label": label,
                "kind": entry.kind,
                "key": entry.key,
                "num_selected": entry.num_selected,
                "numel": entry._mask.numel(),
                "shape": entry.shape,
                "relation": entry.provenance.relation,
                "source": entry.provenance.source,
            }
            for label, entry in zip(self.labels, self._entries, strict=True)
        ]
        return pd.DataFrame(rows)

    def __eq__(self, other: object) -> bool:
        """Compare BOTH denotation levels: family AND per-site masks."""

        if not isinstance(other, Selection):
            return NotImplemented
        if self._kind != other._kind or len(self._entries) != len(other._entries):
            return False
        for mine, theirs in zip(self._entries, other._entries, strict=True):
            if mine.kind != theirs.kind or mine.key != theirs.key:
                return False
            if not _mask_equal(mine._mask, theirs._mask):
                return False
        return True

    def __hash__(self) -> int:
        """Hash the freeze-time canonical digest."""

        return hash(self._digest)

    __repr__ = selection_repr
    __str__ = selection_card

    def __selection__(self) -> Selection:
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

    def __invert__(self) -> Selection:
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
            Site(
                kind=entry.kind,
                key=entry.key,
                provenance=_flip_provenance(entry.provenance),
                _mask=_mask_complement(entry._mask),
                structural_site_key=entry.structural_site_key,
            )
            for entry in self._entries
        )
        return Selection(self._trace, self._kind, entries)


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
    """AST leaf lifting a ``GradReceptiveField`` region into a Selection."""

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
    """AST leaf for ``tl.param(name, mask=...)`` (PARAM-kind producer)."""

    name: str
    mask: torch.Tensor | None

    def __repr__(self) -> str:
        suffix = "" if self.mask is None else ", mask=<bool tensor>"
        return f"param({self.name!r}{suffix})"


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
        return f"site({f'{self.site_label}:{self.pass_index}'!r})"


@dataclass(frozen=True)
class _RandomTerm:
    """AST leaf for ``tl.random_like`` (seeded size-matched control)."""

    like: Any
    within: Any
    seed: int

    def __repr__(self) -> str:
        return f"random_like(like={self.like!r}, within={self.within!r}, seed={self.seed})"


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


class Selector:
    """Composable, trace-independent selection query (frozen AST).

    Compose with ``|  &  -  ~``; resolve against a trace with
    ``trace[selector]``, the one resolution door. Truthiness is deliberately
    AMBIGUOUS on the query type: ``bool(selector)`` refuses typed
    (``selection_bool_ambiguous``) because ``a and b`` on a lazily-composable
    AST silently discards the left operand. Resolve first; the
    ``Selection`` is truthy on "has any selected element".

    Examples
    --------
    >>> query = tl.units("relu_1_2", [(0, 0, 1, 1)]) | tl.units("relu_1_2", [(0, 1, 2, 2)])
    >>> isinstance(query, tl.Selector), query.kind
    (True, 'ACT')
    >>> trace = tl.trace(model, x)
    >>> trace[query][0].num_selected
    2
    """

    __slots__ = ("_kind", "_node", "_signal")

    _kind: str
    _node: Any
    _signal: str

    def __init__(self, node: Any, *, kind: str, signal: str = "forward") -> None:
        """Freeze one query AST node (internal constructor).

        Users build selections through producers (``__selection__`` lifts,
        ``tl.units`` / ``tl.param`` / ``tl.random_like``) and operators,
        never by constructing AST nodes directly.
        """

        if kind not in _SELECTION_KINDS:
            raise ValueError(f"selection kind must be one of {_SELECTION_KINDS}; got {kind!r}.")
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_signal", signal)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("Selection is frozen; compose a new selection instead.")

    @property
    def kind(self) -> str:
        """Return the selection kind (``ACT`` / ``PARAM`` / ``EDGE``)."""

        return self._kind

    def __selection__(self) -> Selector:
        """Pass anywhere a selection is expected."""

        return self

    def __bool__(self) -> bool:
        """Refuse ambiguous truthiness on the lazily-composable query type."""

        raise SelectionError(
            "bool() on a Selection query is ambiguous: `a and b` / `a or b` would "
            "silently discard an operand instead of composing. Use `a & b` / `a | b`, "
            "or resolve first (`trace[selector]`) and test the resolved "
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

    def __invert__(self) -> Selector:
        """Mask complement per touched site (uniform; never predicate negation)."""

        return Selector(_Combinator("invert", (self._node,)), kind=self._kind, signal=self._signal)

    # Resolution ------------------------------------------------------------

    def _resolve(self, trace: Trace) -> Selection:
        """Resolve this query against one trace's site lifecycle.

        The public door is ``trace[selector]``, which calls this.

        Returns
        -------
        Selection
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

        resolved = _resolve_node(self._node, trace, self._kind)
        if resolved._selector is None:
            object.__setattr__(resolved, "_selector", self)
        return resolved


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

    def __invert__(self) -> Selector:
        return ~self.__selection__()


def _lift(value: Any) -> Selector | Selection | None:
    """Lift a composable operand, or return ``None`` when not selection-shaped."""

    if isinstance(value, (Selector, Selection)):
        return value
    lift = getattr(value, "__selection__", None)
    if lift is None:
        return None
    lifted = lift()
    if not isinstance(lifted, (Selector, Selection)):
        raise TypeError(
            f"__selection__ on {type(value).__name__} must return a Selection; "
            f"got {type(lifted).__name__}."
        )
    return lifted


def _check_kinds(op: str, left_kind: str, right_kind: str) -> None:
    """Refuse incoherent mixed-kind composition (closed matrix)."""

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


def _check_signals(left_signal: str, right_signal: str) -> None:
    """Refuse forward x backward composition (inherited refusal)."""

    if left_signal != right_signal:
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

    if isinstance(lifted_left, Selection) or isinstance(lifted_right, Selection):
        resolved_left, resolved_right = _align_resolved_operands(lifted_left, lifted_right)
        _check_kinds(op, resolved_left._kind, resolved_right._kind)
        return _compose_resolved(op, resolved_left, resolved_right)

    if not isinstance(lifted_left, Selector) or not isinstance(lifted_right, Selector):
        raise RuntimeError("query composition operands must both be Selection here")
    return _compose_queries(op, lifted_left, lifted_right)


def _align_resolved_operands(lifted_left: Any, lifted_right: Any) -> tuple[Selection, Selection]:
    """Return both operands resolved on one shared trace (mixed sides resolve now)."""

    if isinstance(lifted_left, Selection) and isinstance(lifted_right, Selection):
        if lifted_left._trace is not lifted_right._trace:
            raise SelectionError(
                "cannot compose resolved selections bound to different traces. "
                "Re-resolve one side against the other's trace first.",
                code="selection_trace_mismatch",
            )
        return lifted_left, lifted_right
    if isinstance(lifted_left, Selection):
        if not isinstance(lifted_right, Selector):
            raise RuntimeError("mixed composition lost its Selection operand")
        return lifted_left, lifted_right._resolve(lifted_left._trace)
    if not isinstance(lifted_left, Selector) or not isinstance(lifted_right, Selection):
        raise RuntimeError("mixed composition lost its Selection operand")
    return lifted_left._resolve(lifted_right._trace), lifted_right


def _compose_queries(op: str, lifted_left: Selector, lifted_right: Selector) -> Selector:
    """Compose two query selections into one AST node (n-ary flatten for or)."""

    _check_kinds(op, lifted_left._kind, lifted_right._kind)
    _check_signals(lifted_left._signal, lifted_right._signal)
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
    return Selector(node_out, kind=lifted_left._kind, signal=lifted_left._signal)


def _compose_resolved(op: str, left: Selection, right: Selection) -> Selection:
    """Compose two same-trace resolved selections (family + element rules)."""

    left_by_key = {entry.key: entry for entry in left._entries}
    right_by_key = {entry.key: entry for entry in right._entries}
    if op == "or":
        entries = _compose_entries_union(left_by_key, right_by_key)
    elif op == "and":
        entries = _compose_entries_intersection(left_by_key, right_by_key)
    elif op == "sub":
        entries = _compose_entries_difference(left_by_key, right_by_key)
    else:
        raise ValueError(f"unknown operator {op!r}")
    return Selection(left._trace, left._kind, entries)


def _compose_entries_union(
    left_by_key: dict[Any, Site], right_by_key: dict[Any, Site]
) -> list[Site]:
    """Union entries: every touched site survives; shared sites join masks."""

    entries: list[Site] = []
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
    left_by_key: dict[Any, Site], right_by_key: dict[Any, Site]
) -> list[Site]:
    """Intersect entries; shared sites stay touched even when element-empty."""

    entries: list[Site] = []
    for key, entry in left_by_key.items():
        other = right_by_key.get(key)
        if other is not None:
            entries.append(_entry_compose(entry, other, _mask_intersect, _join_relation))
    return entries


def _compose_entries_difference(
    left_by_key: dict[Any, Site], right_by_key: dict[Any, Site]
) -> list[Site]:
    """Difference entries; subtraction never un-touches: fam(A - B) = fam(A)."""

    entries: list[Site] = []
    for key, entry in left_by_key.items():
        other = right_by_key.get(key)
        if other is None:
            entries.append(entry)
        else:
            entries.append(_entry_compose(entry, other, _mask_difference, _difference_relation))
    return entries


def _entry_compose(
    left: Site,
    right: Site,
    mask_op: Any,
    relation_op: Any,
) -> Site:
    """Compose two entries at one shared site."""

    if left.shape != right.shape:
        raise SelectionError(
            f"site {left.key!r} resolved with two different index spaces "
            f"({left.shape} vs {right.shape}); the selections cannot compose.",
            code="selection_unresolvable",
            reason="mask_shape_mismatch",
            site_key=left.key,
        )
    relation = relation_op(left.provenance.relation, right.provenance.relation)
    return Site(
        kind=left.kind,
        key=left.key,
        provenance=_compose_provenance(left.provenance, right.provenance, relation),
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

    ``code`` defaults to the contracted ``selection_unresolvable``; the
    keyword lets a raise site spell its code visibly (the
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


def _act_entry(op: Any, mask: _Mask, relation: str, source: str) -> Site:
    """Build one ACT site entry for a resolved op."""

    layer_label = getattr(op, "layer_label", None) or op.label
    pass_index = getattr(op, "pass_number", 1) or 1
    return Site(
        kind="ACT",
        key=(layer_label, pass_index),
        provenance=SelectionProvenance(relation=relation, source=source),
        _mask=mask,
        # The L1 structural-position identity lives on Op as ``site_key``
        # (site_key_v1 strings); Site's field keeps the qualified name
        # because Site.key is already the (label, pass) address.
        structural_site_key=record_member(op, "site_key", None),
    )


def _forward_ops(trace: Any) -> tuple[Any, ...]:
    """Return the trace's forward ops in execution order."""
    from .resolver import _iter_sites

    return tuple(_iter_sites(trace, "forward"))


def _find_act_ops(trace: Any, site: str) -> tuple[Any, ...]:
    """Return the ops matching one site spelling (label / layer label / io role)."""

    matches = []
    for op in _forward_ops(trace):
        if (
            op.label == site
            or getattr(op, "layer_label", None) == site
            or record_member(op, "io_role", None) == site
        ):
            matches.append(op)
    return tuple(matches)


def _resolve_node(node: Any, trace: Any, kind: str) -> Selection:
    """Resolve one AST node against a trace."""

    if isinstance(node, _Combinator):
        return _resolve_combinator(node, trace, kind)
    for term_type, resolver in _TERM_RESOLVERS:
        if isinstance(node, term_type):
            return resolver(node, trace)
    raise TypeError(f"unknown selection AST node {type(node).__name__}")


def _resolve_combinator(node: _Combinator, trace: Any, kind: str) -> Selection:
    """Resolve an operator node by folding its resolved operands."""

    if node.op == "invert":
        return ~_resolve_node(node.operands[0], trace, kind)
    resolved = [_resolve_node(operand, trace, kind) for operand in node.operands]
    result = resolved[0]
    for operand in resolved[1:]:
        result = _compose_resolved(node.op, result, operand)
    return result


def _resolve_selector_term(node: _SelectorTerm, trace: Any) -> Selection:
    """Resolve a lifted selector to whole-site masks over its matched sites."""
    from .resolver import _normalize_query, _resolve_branch_sites, split_site_branches

    matched: list[Any] = []
    for branch in split_site_branches(node.selector):
        if branch.universe == "gradient":
            raise _grad_selection_refusal(node.selector, code="grad_selection_unsupported")
        branch_selector = _normalize_query(branch.selector)
        matched.extend(_resolve_branch_sites(trace, branch_selector, branch.universe, strict=False))
    entries = [
        _act_entry(op, _mask_whole(_site_shape(op)), "exact", f"selector {node.selector!r}")
        for op in matched
    ]
    return Selection(trace, "ACT", entries, selector=node.selector)


def _resolve_box_term(node: _BoxTerm, trace: Any) -> Selection:
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
    return Selection(trace, "ACT", [_act_entry(op, mask, relation, source)])


def _resolve_gradient_term(node: _GradientTerm, trace: Any) -> Selection:
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
    return Selection(trace, "ACT", [_act_entry(op, mask, "exact", source)])


def _resolve_facet_term(node: _FacetTerm, trace: Any) -> Selection:
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
        matches = tuple(op for op in matches if getattr(op, "pass_number", 1) == pass_index)
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
                f"facet {getattr(spec, 'function_id', '<unknown>')!r} needs the saved "
                f"home value at {home_label!r}, which this capture did not retain.",
                site=home_label,
            ) from exc
        raise _unresolvable(
            "facet_write_mask_unavailable",
            f"facet {getattr(spec, 'function_id', '<unknown>')!r} cannot produce its "
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
    source = f"facet {getattr(spec, 'function_id', '<unknown>')!r} write region"
    return Selection(trace, "ACT", [_act_entry(op, mask, "exact", source)])


def _resolve_param_term(node: _ParamTerm, trace: Any) -> Selection:
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
    entry = Site(
        kind="PARAM",
        key=(address,),
        provenance=SelectionProvenance(relation="exact", source=f"param({node.name!r})"),
        _mask=mask,
    )
    return Selection(trace, "PARAM", [entry])


def _multipass_bare_label_refusal(site: str, matches: Sequence[Any]) -> SelectionError:
    """Refuse a multi-pass layer's bare spelling: units() addresses ONE op's index space."""
    layer_label = getattr(matches[0], "short_layer_label", None) or site
    pass_indices = sorted(int(getattr(op, "pass_number", 1) or 1) for op in matches)
    return _unresolvable(
        "multipass_bare_label",
        multipass_bare_label_message(layer_label, pass_indices),
        site=site,
        pass_indices=tuple(pass_indices),
    )


def _resolve_unit_term(node: _UnitTerm, trace: Any) -> Selection:
    """Resolve an explicit site + index set."""
    matches = _find_act_ops(trace, node.site)
    if not matches:
        raise _unresolvable(
            "site_not_in_trace",
            f"site {node.site!r} is not a site of this trace.",
            site=node.site,
        )
    if len(matches) > 1 and len({getattr(op, "layer_label", None) for op in matches}) == 1:
        raise _multipass_bare_label_refusal(node.site, matches)
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
    return Selection(trace, "ACT", entries)


def _resolve_whole_site_term(node: _WholeSiteTerm, trace: Any) -> Selection:
    """Resolve an Op/Layer whole-output lift."""

    matches = _find_act_ops(trace, node.site_label)
    if node.pass_index is not None:
        matches = tuple(op for op in matches if getattr(op, "pass_number", 1) == node.pass_index)
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
    return Selection(trace, "ACT", entries)


def _resolve_random_term(node: _RandomTerm, trace: Any) -> Selection:
    """Resolve a seeded, size-matched random control selection."""

    like = _resolve_operand_for_random(node.like, trace)
    within = _resolve_operand_for_random(node.within, trace)
    requested = sum(entry.num_selected for entry in like._entries)
    population: list[tuple[Site, int]] = []
    for entry in within._entries:
        flat = entry._mask._dense_ro().reshape(-1)
        for index in torch.nonzero(flat, as_tuple=False).reshape(-1).tolist():
            population.append((entry, index))
    if len(population) < requested:
        raise _unresolvable(
            "population_too_small",
            f"random_like needs {requested} elements but the `within` "
            f"population has only {len(population)}.",
            requested=requested,
            available=len(population),
        )
    rng = _random_module.Random(node.seed)
    sampled = rng.sample(range(len(population)), requested)
    dense_by_key: dict[tuple[Any, ...], tuple[Site, torch.Tensor]] = {}
    for position in sampled:
        entry, flat_index = population[position]
        if entry.key not in dense_by_key:
            dense_by_key[entry.key] = (
                entry,
                torch.zeros(entry.shape, dtype=torch.bool),
            )
        dense_by_key[entry.key][1].reshape(-1)[flat_index] = True
    entries = [
        Site(
            kind=entry.kind,
            key=entry.key,
            provenance=SelectionProvenance(
                relation="exact", source=f"random_like(seed={node.seed})"
            ),
            _mask=_mask_from_dense(entry.shape, dense),
            structural_site_key=entry.structural_site_key,
        )
        for entry, dense in dense_by_key.values()
    ]
    return Selection(trace, within._kind, entries)


def _resolve_operand_for_random(operand: Any, trace: Any) -> Selection:
    """Resolve one random_like operand (a fork ancestor's selection carries over)."""

    if not isinstance(operand, (Selection, Selector)):
        raise RuntimeError("random_like operand must lift to a Selection")
    return operand._resolve(trace)


# ---------------------------------------------------------------------------
# Stage-1 producer constructors.
# ---------------------------------------------------------------------------


def units(
    site: str,
    indices: Iterable[tuple[int, ...] | int] | torch.Tensor,
) -> Selector:
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> selection = trace[tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)])]
    >>> selection.labels, selection[0].num_selected
    (('relu_1_2:1',), 2)
    """

    if not isinstance(site, str) or not site:
        raise ValueError("units(site, ...) requires a non-empty site label string.")
    if isinstance(indices, torch.Tensor):
        if indices.dtype != torch.bool:
            raise ValueError(
                "units(site, indices) tensor form must be a bool mask over the "
                f"site's output index space; got dtype {indices.dtype}."
            )
        return Selector(_UnitTerm(site=site, indices=None, index_mask=indices.clone()), kind="ACT")
    normalized: list[tuple[int, ...]] = []
    for coordinates in indices:
        if isinstance(coordinates, int):
            coordinates = (coordinates,)
        coordinates = tuple(int(coordinate) for coordinate in coordinates)
        if any(coordinate < 0 for coordinate in coordinates):
            raise ValueError(f"units indices must be non-negative; got {coordinates!r}.")
        normalized.append(coordinates)
    return Selector(_UnitTerm(site=site, indices=tuple(normalized), index_mask=None), kind="ACT")


def param(name: str, mask: torch.Tensor | None = None) -> Selector:
    """Select a named-parameter element region (param-term producer).

    ``mask=None`` selects the whole parameter. Fenced from
    ``module.parameters()``: this addresses the TRACE's recorded parameter
    geometry by name/address.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> selection = trace[tl.param("0.weight")]
    >>> selection.kind, selection[0].num_selected
    ('PARAM', 108)
    """

    if not isinstance(name, str) or not name:
        raise ValueError("param(name, ...) requires a non-empty parameter name string.")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("param mask must be a bool tensor over the parameter shape.")
        mask = mask.clone()
    return Selector(_ParamTerm(name=name, mask=mask), kind="PARAM")


def random_like(
    *,
    like: Any,
    within: Any,
    seed: int,
) -> Selector:
    """Seeded, size-matched random control selection.

    Both operands resolve at ``trace[selector]`` time; the sampling population
    is ``within``'s selected-element set; the result kind is ``within``'s
    kind; ``|result| == |like|`` sampled without replacement. A population
    smaller than ``|like|`` (including an empty complement) refuses typed
    (``selection_unresolvable``, ``reason="population_too_small"``).

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> probe = tl.units("relu_1_2", [(0, 0, 0, 0), (0, 1, 1, 1)])
    >>> control = tl.random_like(like=probe, within=tl.func("conv2d"), seed=0)
    >>> sum(site.num_selected for site in trace[control])
    2
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("random_like seed must be a non-negative int.")
    like_lifted = _lift(like)
    within_lifted = _lift(within)
    if like_lifted is None or within_lifted is None:
        raise ValueError(
            "random_like operands must be selections or region-shaped "
            "producers implementing __selection__."
        )
    kind = within_lifted._kind if isinstance(within_lifted, Selector) else within_lifted._kind
    return Selector(_RandomTerm(like=like_lifted, within=within_lifted, seed=seed), kind=kind)


# ---------------------------------------------------------------------------
# Lift helpers used by producer classes (lazy to keep this module light).
# ---------------------------------------------------------------------------


def _selection_from_selector(selector: Any) -> Selector:
    """Lift a BaseSelector as a selector-term (predicate meaning intact)."""
    from .resolver import split_site_branches

    branches = split_site_branches(selector)
    if any(branch.universe == "gradient" for branch in branches):
        # Gradient selections are not addressable this release (SPEC 1.3).
        raise _grad_selection_refusal(selector, code="grad_selection_unsupported")
    signal = "backward" if any(branch.universe == "node" for branch in branches) else "forward"
    return Selector(_SelectorTerm(selector=selector), kind="ACT", signal=signal)


def _selection_from_box(box: Any) -> Selector:
    """Lift a ReceptiveFieldBox as an ACT hull term."""

    return Selector(_BoxTerm(box=box), kind="ACT")


def _selection_from_gradient(gradient_rf: Any) -> Selector:
    """Lift a GradReceptiveField as an exact empirical ACT term."""

    return Selector(_GradientTerm(gradient_rf=gradient_rf), kind="ACT")


def _selection_from_facet(spec: Any) -> Selector:
    """Lift a FacetSpec's write region as an ACT term."""

    return Selector(_FacetTerm(spec=spec), kind="ACT")


def _selection_from_op(op: Any) -> Selector:
    """Lift one Op (whole-output, one pass)."""

    layer_label = getattr(op, "layer_label", None) or op.label
    pass_index = getattr(op, "pass_number", 1) or 1
    return Selector(_WholeSiteTerm(site_label=layer_label, pass_index=pass_index), kind="ACT")


def _selection_from_layer(layer: Any) -> Selector:
    """Lift one Layer (whole-output, ALL passes)."""

    layer_label = getattr(layer, "layer_label", None) or layer.label
    return Selector(_WholeSiteTerm(site_label=layer_label, pass_index=None), kind="ACT")


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
    census counts site-visible codes, the ``_encoding_error`` pattern).
    """

    return SelectionError(message, code=code, reason=reason, **fields)


# _validate_edited / _masked_factory / _derive_masked_edit live in
# torchlens/intervention/_masked_edit.py; the plan builder itself
# (_build_selection_do_plan, _resolve_do_target) lives in
# torchlens/_selection/do_plan.py and is imported lazily below.


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


def _trace_edge_records(trace: Any) -> tuple[Any, ...]:
    """Return the trace's dataflow edge family (EdgeUseRecords, child-owned)."""

    _require_edge_provenance(trace)
    records: list[Any] = []
    for op in _forward_ops(trace):
        records.extend(getattr(op, "edge_uses", ()) or ())
    return tuple(records)


def _edge_entry(record: Any) -> Site:
    """Build one EDGE site entry (whole-edge mask over a 1-element space)."""

    return Site(
        kind="EDGE",
        key=edge_address_of(record),
        provenance=SelectionProvenance(
            relation="exact",
            source=f"edge {record.parent_label!r} -> {record.child_label!r}",
        ),
        _mask=_mask_whole((1,)),
    )


def _selection_from_edge(record: Any) -> Selector:
    """Lift one EdgeUseRecord as an EDGE selection term."""

    return Selector(
        _EdgeTerm(
            addresses=(edge_address_of(record),),
            display=(f"{record.parent_label}->{record.child_label}",),
        ),
        kind="EDGE",
    )


def _resolve_edge_term(node: _EdgeTerm, trace: Any) -> Selection:
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
    return Selection(trace, "EDGE", entries)


def _edge_family_complement(resolved: Selection) -> Selection:
    """Complement an EDGE selection within the trace's dataflow edge family.

    The edge universe is well-defined (the trace's EdgeTable occurrences), so
    ``~`` on a RESOLVED edge selection complements against it: edges outside
    the selected set enter with whole masks; selected edges leave. (On a
    QUERY, ``~`` defers to resolve time.)
    """

    selected = {entry.key for entry in resolved._entries if entry.num_selected > 0}
    entries = [
        _edge_entry(record)
        for record in _trace_edge_records(resolved._trace)
        if edge_address_of(record) not in selected
    ]
    return Selection(resolved._trace, "EDGE", entries)


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
