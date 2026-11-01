"""The selector base class, its composites and the address-universe classification.

Split from :mod:`torchlens.intervention.selectors`, which re-exports every name here that callers
import from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .._errors import ArgumentTypeError
from . import selectors as _selectors
from .errors import SelectorCompositionError, UnclassifiedSelectorError
from .types import FrozenTargetSpec, TargetSpec

if TYPE_CHECKING:
    from .selectors import SelectorKind, SelectorLike


#: Public ``tl.*`` spelling of each selector kind whose stored value predates its rename
#: (the stored ``selector_kind`` stays, so persisted specs keep loading).
_KIND_SPELLINGS: dict[str, str] = {
    "contains": "label_contains",
    "input_at": "model_input",
    "output": "model_output",
    "regex": "label_matches",
    "grad_fn_label": "intervention.grad_fn_label",
    "output_at": "intervention.output_at",
    "backward_pass": "in_backward_pass",
}


@dataclass(frozen=True)
class BaseSelector:
    """Base class for typed TorchLens site selectors."""

    selector_kind: SelectorKind
    selector_value: Any

    def __and__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector that matches the intersection of two selectors.

        Parameters
        ----------
        other:
            Selector to intersect with this selector.

        Returns
        -------
        CompositeSelector
            Intersection selector. A non-selector operand that implements
            ``__selection__`` defers to that object's reflected operator, so
            ``selector & rf_box`` composes as a Selection; junk operands keep
            their shipped refusals.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        return _compose("and", self, other)

    def __rand__(self, other: str) -> CompositeSelector:
        """Return ``other & self`` for a bare label string on the left.

        Parameters
        ----------
        other:
            Bare label string; it keeps the meaning a bare string has at the
            door that evaluates the composite.

        Returns
        -------
        CompositeSelector
            Intersection selector.
        """

        if not isinstance(other, str):
            return NotImplemented
        return _compose("and", other, self)

    def __or__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector that matches the union of two selectors.

        Parameters
        ----------
        other:
            Selector to union with this selector.

        Returns
        -------
        CompositeSelector
            Union selector. A non-selector operand that implements
            ``__selection__`` defers to that object's reflected operator.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        if _any_followed_by(self, other):
            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "OR-composed followed_by selectors cannot be evaluated safely."
            )
        return _compose("or", self, other)

    def __ror__(self, other: str) -> CompositeSelector:
        """Return ``other | self`` for a bare label string on the left.

        Parameters
        ----------
        other:
            Bare label string; it keeps the meaning a bare string has at the
            door that evaluates the composite.

        Returns
        -------
        CompositeSelector
            Union selector.
        """

        if not isinstance(other, str):
            return NotImplemented
        if _any_followed_by(self):
            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "OR-composed followed_by selectors cannot be evaluated safely."
            )
        return _compose("or", other, self)

    def __invert__(self) -> BaseSelector:
        """Return a selector that matches the complement of this selector.

        Returns
        -------
        BaseSelector
            Negated selector (``tl.grad`` keeps the complement inside its
            universe and returns a gradient address).
        """

        if _any_followed_by(self):
            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "negated followed_by selectors cannot be evaluated safely."
            )
        if classify_selector_universe(self) == "mixed":
            raise SelectorCompositionError(
                f"~{self!r} negates a union of addresses in different universes "
                "(activation, gradient, node); its complement is not one address set.",
                code="selector_cross_universe_not",
                remedy="negate inside each branch before joining them, e.g. tl.grad(~a) | ~a",
            )
        return NotSelector(self)

    def __sub__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector matching this selector minus ``other`` (DESUGAR).

        ``a - b`` desugars to ``CompositeSelector("and", (a, NotSelector(b)))``
        — no new selector kind, no spec round-trip change. Selector evaluation
        is a per-subject characteristic function, so difference of match-sets
        IS ``a(x) and not b(x)``; this coincides with Selection-level ``-`` on
        whole-site lifted terms (law-tested).

        Parameters
        ----------
        other:
            Selector (or predicate callable) to subtract. A non-selector
            operand implementing ``__selection__`` defers to that object's
            reflected operator.

        Returns
        -------
        CompositeSelector
            Difference selector.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        if isinstance(other, BaseSelector):
            negated: SelectorLike = ~other  # shipped guards (followed_by refusal) apply
        elif callable(other):
            negated = NotSelector(other)
        else:
            raise ArgumentTypeError(
                f"cannot subtract {type(other).__name__} from a selector; pass a "
                "selector or predicate callable.",
                code="selector_subtraction_operand_invalid",
                remedy="subtract a selector, predicate callable, or region producer",
            )
        return _compose("and", self, negated)

    def __selection__(self) -> Any:
        """Lift this selector as a Selection selector-term.

        The lifted term keeps predicate meaning intact: it resolves to
        whole-site masks over matched sites. Selection-level ``~`` is mask
        complement — users who mean "all sites not matching s" spell ``~s``
        BEFORE lifting (``~lift(s) != lift(~s)``, pinned non-law).
        """

        from .._selection import _selection_from_selector

        return _selection_from_selector(self)

    def to_target_spec(self) -> TargetSpec:
        """Convert the selector to a mutable target spec.

        Returns
        -------
        TargetSpec
            Target spec carrying this selector's kind and payload.
        """

        return TargetSpec(
            selector_kind=self.selector_kind,
            selector_value=self.selector_value,
        )

    def __dir__(self) -> list[str]:
        """Return selector attributes for tab completion.

        Returns
        -------
        list[str]
            Standard selector attributes. Selectors are not bound to a
            ``Trace``, so layer-name completion lives on log accessors.
        """

        return sorted(set(super().__dir__()) | {"selector_kind", "selector_value"})

    def _ipython_key_completions_(self) -> list[str]:
        """Return key completions for IPython.

        Returns
        -------
        list[str]
            Empty list because selector instances have no bound layer universe.
        """

        return []

    def __repr__(self) -> str:
        """Return a concise constructor-style representation.

        Returns
        -------
        str
            Repr containing the selector kind and payload.
        """

        spelling = _KIND_SPELLINGS.get(self.selector_kind, self.selector_kind)
        return f"tl.{spelling}({self.selector_value!r})"

    def __call__(self, ctx: Any) -> bool:
        """Return whether this selector matches a predicate record context.

        Parameters
        ----------
        ctx:
            Capture-time ``RecordContext`` or a layer-like object.

        Returns
        -------
        bool
            Whether ``ctx`` matches this selector.
        """

        from .._ir.selector_eval import evaluate

        return evaluate(self, ctx, lifecycle="capture")


@dataclass(frozen=True, repr=False)
class CompositeSelector(BaseSelector):
    """Selector composed with ``&`` or ``|``.

    ``&`` / ``|`` build nested binary composites; deserialized target specs
    may carry a flat n-ary child tuple. Both shapes evaluate identically.
    Degenerate arities follow the standard identity semantics in evaluation
    and spec round-trips alike: an empty ``and`` matches everything, an empty
    ``or`` matches nothing, and a unary composite matches like its child.

    Parameters
    ----------
    operator:
        ``"and"`` for intersection or ``"or"`` for union.
    selectors:
        Selectors to combine.
    """

    operator: Literal["and", "or"]
    selectors: tuple[SelectorLike, ...]

    def __init__(self, operator: Literal["and", "or"], selectors: tuple[SelectorLike, ...]) -> None:
        """Create a composite selector.

        Parameters
        ----------
        operator:
            ``"and"`` for intersection or ``"or"`` for union.
        selectors:
            Selectors to combine.
        """

        object.__setattr__(self, "selector_kind", operator)
        object.__setattr__(self, "selector_value", selectors)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "selectors", selectors)

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state without the transient composition memo."""

        state = dict(self.__dict__)
        state.pop(_FACTS_KEY, None)
        return state

    def to_target_spec(self) -> TargetSpec:
        """Convert the composite selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with nested selector payloads.
        """

        nested = tuple(
            item.to_target_spec() if isinstance(item, BaseSelector) else item
            for item in self.selectors
        )
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

    def __repr__(self) -> str:
        """Return an infix representation of the composite selector.

        Returns
        -------
        str
            Repr with ``&`` or ``|``.
        """

        symbol = "&" if self.operator == "and" else "|"
        joined = f" {symbol} ".join(repr(child) for child in self.selectors)
        return f"({joined})"


@dataclass(frozen=True, repr=False)
class NotSelector(BaseSelector):
    """Selector composed with unary ``~``.

    Parameters
    ----------
    selector:
        Selector to negate.
    """

    selector: SelectorLike

    def __init__(self, selector: SelectorLike) -> None:
        """Create a negated selector.

        Parameters
        ----------
        selector:
            Selector to negate.
        """

        object.__setattr__(self, "selector_kind", "not")
        object.__setattr__(self, "selector_value", selector)
        object.__setattr__(self, "selector", selector)

    def to_target_spec(self) -> TargetSpec:
        """Convert the negated selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with a nested selector payload.
        """

        nested = (
            self.selector.to_target_spec()
            if isinstance(self.selector, BaseSelector)
            else self.selector
        )
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

    def __repr__(self) -> str:
        """Return a unary representation of the negated selector.

        Returns
        -------
        str
            Repr with ``~``.
        """

        return f"(~{self.selector!r})"


#: Node-universe and pass-qualifier kinds a gradient address cannot wrap.
_NODE_KINDS = frozenset({"grad_fn", "grad_fn_label", "grad_kind", "intervening", "without_op"})


_QUALIFIER_KINDS = frozenset({"backward_pass"})


def _any_followed_by(*selectors: Any) -> bool:
    """Return whether any operand's selector tree contains ``tl.followed_by``."""

    from .._ir.selector_eval import contains_followed_by

    return any(contains_followed_by(selector) for selector in selectors)


#: Address universes (SPEC 1.4). ``None`` is an agnostic leaf (a label, a
#: module address, ...), which is an activation whenever it stands outside
#: ``tl.grad``. ``"mixed"`` is a union whose branches sit in different
#: universes (``tl.grad(s) | s``); ``resolve_site_plan`` splits it.
SelectorUniverse: TypeAlias = Literal["activation", "gradient", "node", "qualifier", "mixed"]


#: Leaf selector kinds per universe, for selectors and target specs alike.
_ACTIVATION_KINDS = frozenset({"func", "func_transform"})


_AGNOSTIC_KINDS = frozenset(
    {
        "label",
        "module",
        "output",
        "output_at",
        "input_at",
        "contains",
        "regex",
        "predicate",
        "in_module",
        "facet",
        "site",
        "followed_by",
        "preceded_by",
        # Step qualification is a position fact (lane F42): it narrows forward
        # subjects by episode step, like the other position selectors.
        "episode_step",
    }
)


_CONNECTIVE_KINDS = frozenset({"and", "or", "not"})


def _leaf_universe(kind: str) -> SelectorUniverse | None:
    """Return the universe of one leaf selector kind."""

    if kind == "grad":
        return "gradient"
    if kind in _NODE_KINDS:
        return "node"
    if kind in _QUALIFIER_KINDS:
        return "qualifier"
    if kind in _ACTIVATION_KINDS:
        return "activation"
    if kind in _AGNOSTIC_KINDS:
        return None
    raise UnclassifiedSelectorError(
        f"selector kind {kind!r} has no universe classification; add an explicit bucket."
    )


def _connective_children(sel: Any) -> tuple[str, tuple[Any, ...]] | None:
    """Return ``(operator, children)`` for an and/or/not node, else ``None``."""

    if isinstance(sel, CompositeSelector):
        return sel.operator, tuple(sel.selectors)
    if isinstance(sel, NotSelector):
        return "not", (sel.selector,)
    if isinstance(sel, (TargetSpec, FrozenTargetSpec)) and sel.selector_kind in _CONNECTIVE_KINDS:
        value = sel.selector_value
        if sel.selector_kind == "not":
            return "not", (value,)
        children = tuple(value) if isinstance(value, (list, tuple)) else (value,)
        return str(sel.selector_kind), children
    return None


def classify_selector_universe(sel: Any) -> SelectorUniverse | None:
    """Return which address universe a selector tree belongs to (SPEC 1.4).

    Parameters
    ----------
    sel:
        Selector, target spec, frozen target spec, or bare label string.

    Returns
    -------
    SelectorUniverse | None
        ``"activation"`` (explicitly forward leaves such as ``tl.func``),
        ``"gradient"`` (``tl.grad(...)``), ``"node"`` (``tl.grad_fn`` and the
        other autograd-node selectors), ``"qualifier"``
        (``tl.in_backward_pass`` alone or joined to qualifiers), ``"mixed"``
        (a union whose branches sit in different universes), or ``None`` for an
        agnostic tree (an activation whenever it stands outside ``tl.grad``).

    Raises
    ------
    SelectorCompositionError
        When an ``&`` inside the tree joins operands that never share a site
        (deserialized trees are checked here; operators check at build time).
    UnclassifiedSelectorError
        For a selector type with no universe bucket.
    """

    if isinstance(sel, str):
        return None
    connective = _connective_children(sel)
    if connective is not None:
        operator, children = connective
        if operator == "not":
            return classify_selector_universe(children[0])
        # A plain loop keeps one interpreter frame per nesting level, so deep
        # deserialized trees classify as deep as they evaluate.
        universes: list[SelectorUniverse | None] = []
        for child in children:
            universes.append(classify_selector_universe(child))
        if operator == "and":
            if not children:
                return None
            combined = universes[0]
            for index in range(1, len(children)):
                combined = _selectors._and_universe(
                    combined, universes[index], children[index - 1], children[index]
                )
            return combined
        return _or_universe(universes)
    if isinstance(sel, (BaseSelector, TargetSpec, FrozenTargetSpec)):
        return _leaf_universe(str(sel.selector_kind))
    raise UnclassifiedSelectorError(
        f"{type(sel).__name__} has no universe classification; add an explicit bucket."
    )


def _or_universe(universes: Sequence[SelectorUniverse | None]) -> SelectorUniverse | None:
    """Combine the universes of a union's operands."""

    present = {universe for universe in universes if universe is not None}
    if not present:
        return None
    if "mixed" in present:
        return "mixed"
    if None in universes:
        present.add("activation")
    if present == {"qualifier"}:
        return "qualifier"
    # A pass qualifier standing alone in a union is a node branch.
    if "qualifier" in present:
        present = (present - {"qualifier"}) | {"node"}
    return next(iter(present)) if len(present) == 1 else "mixed"


def _classify_selector_direction(
    sel: SelectorLike,
) -> Literal["forward", "backward"] | None:
    """Return the legacy graph-direction bucket, derived from the universe.

    Kept for the hook and door modules that still read a direction until they
    take the site's universe plan; new code calls
    :func:`classify_selector_universe`.

    Parameters
    ----------
    sel:
        Selector to classify.

    Returns
    -------
    Literal["forward", "backward"] | None
        ``"backward"`` for gradient, node and qualifier trees (and unions that
        contain one), ``"forward"`` for explicitly forward leaves, ``None``
        for agnostic trees.
    """

    universe = classify_selector_universe(sel)
    if universe is None:
        return None
    return "forward" if universe == "activation" else "backward"


def _flatten_same_operator(
    operator: str, left: SelectorLike, right: SelectorLike
) -> tuple[SelectorLike, ...]:
    """Merge same-operator composite operands into one flat child tuple.

    r-b4 R27-6a: ``&``/``|`` used to nest one binary composite per operator, so
    a programmatically composed predicate (``functools.reduce(operator.or_,
    [tl.func(n) for n in names])``) built a 500-deep binary tree and blew the
    interpreter stack at trace entry. Composites are documented n-ary with
    identical evaluation for flat and nested shapes, so chained applications of
    ONE operator now accumulate a flat child tuple: depth stays constant and
    evaluation walks one level. Mixed-operator composition still nests (the
    shape is semantic there), and existing nested trees (deserialized specs)
    keep evaluating unchanged.
    """

    children: list[SelectorLike] = []
    for operand in (left, right):
        if isinstance(operand, CompositeSelector) and operand.operator == operator:
            children.extend(operand.selectors)
        else:
            children.append(operand)
    return tuple(children)


def _check_composition(
    a: Any,
    b: Any,
    *,
    operator: Literal["and", "or"] = "and",
) -> tuple[SelectorUniverse | None, bool]:
    """Validate that two selectors can be composed.

    ``&`` refuses operands from universes that never share a site
    (``selector_cross_universe_and``) and a pass qualifier joined to an
    activation (``selector_signal_mismatch``); ``|`` joins any universes into a
    multi-branch address.

    Parameters
    ----------
    a:
        Left selector (or bare label string).
    b:
        Right selector (or bare label string).
    operator:
        Which operator is composing them.

    Returns
    -------
    tuple[SelectorUniverse | None, bool]
        The composite's universe and whether it contains ``followed_by``.
        Raises when composition is invalid.
    """

    from .._ir.selector_eval import contains_followed_by, flatten_and_conjuncts

    a_memo, b_memo = _composition_memo(a), _composition_memo(b)
    a_universe, a_followed = a_memo or (classify_selector_universe(a), contains_followed_by(a))
    b_universe, b_followed = b_memo or (classify_selector_universe(b), contains_followed_by(b))
    if operator == "and":
        universe = _selectors._and_universe(a_universe, b_universe, a, b)
    else:
        universe = _or_universe([a_universe, b_universe])
    if a_followed or b_followed:
        # `&` nests, so validate the FLAT conjunction: `a & fb & b` must pass
        # exactly like `a & b & fb` and the flat three-child spec.
        conjuncts = flatten_and_conjuncts((a, b))
        direct = [c for c in conjuncts if isinstance(c, _selectors.FollowedBySelector)]
        buried = [
            c
            for c in conjuncts
            if not isinstance(c, _selectors.FollowedBySelector) and contains_followed_by(c)
        ]
        if len(direct) != 1 or buried:
            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "nested or multi-followed_by compositions are unsupported."
            )
    return universe, a_followed or b_followed


#: Instance-dict key holding an operator-built composite's memoised
#: ``(universe, contains_followed_by)``; never pickled (``__getstate__``).
_FACTS_KEY = "_tl_composition_facts"


def _composition_memo(sel: Any) -> tuple[SelectorUniverse | None, bool] | None:
    """Return an operand's memoised composition facts, or ``None``."""

    return sel.__dict__.get(_FACTS_KEY) if isinstance(sel, CompositeSelector) else None


def _memo_safe(operand: Any) -> bool:
    """Whether an operand's facts cannot drift after composition (no ``TargetSpec`` inside)."""

    while isinstance(operand, (NotSelector, _selectors.GradSelector)):
        operand = operand.selector if isinstance(operand, NotSelector) else operand.inner
    if isinstance(operand, CompositeSelector):
        return _FACTS_KEY in operand.__dict__
    return isinstance(operand, (BaseSelector, str))


def _compose(operator: Literal["and", "or"], left: Any, right: Any) -> CompositeSelector:
    """Validate ``left <operator> right`` and build its flat composite.

    The composite memoises the facts ``_check_composition`` derived, so the
    next chained operator reads them instead of re-walking every accumulated
    child (an n-step chain was quadratic). They equal a fresh flat
    classification: the ``&`` fold and the ``|`` join combine per operand.
    """

    facts = _check_composition(left, right, operator=operator)
    composite = CompositeSelector(operator, _flatten_same_operator(operator, left, right))
    if _memo_safe(left) and _memo_safe(right):
        composite.__dict__[_FACTS_KEY] = facts
    return composite
