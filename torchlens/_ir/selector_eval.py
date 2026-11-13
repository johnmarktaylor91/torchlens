"""The one selector interpreter for capture-time, post-hoc, and live matching.

Every selector-matching decision in TorchLens routes through :func:`evaluate`,
parameterized by a ``lifecycle``:

- ``"capture"``: in-flight ``RecordContext`` objects (and duck-typed layer-like
  objects) evaluated by ``trace(save=...)`` / ``record(save=...)`` predicates.
- ``"site"``: finalized ``Op`` / ``Layer`` / ``GradFn`` sites resolved by
  ``find_sites`` / ``_resolve_sites``.
- ``"live"``: capture-time hook-site proxies matched while applying live hooks.

One matching RULE per selector kind; only the *subject universe* (which label
spellings / module candidates a lifecycle exposes) differs, via the adapter
functions below. Unsupported ``(kind, lifecycle)`` pairs raise the single typed
:class:`~torchlens.intervention.errors.SelectorCapabilityError`.

Decided unified semantics (see the consolidation report for the enumerated
behavior changes):

- ``contains`` is case-INSENSITIVE in every lifecycle; ``regex`` is
  case-sensitive ``re.search``; ``label`` is an exact literal. Post-hoc
  substring/regex search runs over ``layer_label`` only (the historical
  contract); exact ``label`` keeps the wide universe including raw and short
  spellings.
- ``func`` matches the captured function name OR the normalized layer type in
  every lifecycle (live module-boundary proxies match on function name only,
  so an op-level hook never double-fires on the module-exit pseudo-site).
- ``and`` / ``or`` composites are n-ary, and they SHORT-CIRCUIT per subject in
  every lifecycle: a ``tl.where`` predicate must not rely on being invoked for
  subjects a sibling already decided. Degenerate arities keep the standard
  identity semantics in evaluation AND spec round-trips: an empty ``and``
  matches everything, an empty ``or`` matches nothing, and a unary composite
  matches exactly like its child.
- ``followed_by`` / ``preceded_by`` are capture-time-only and refuse post-hoc
  or live evaluation through the capability path — upfront (before any
  short-circuit) for post-hoc resolution and live hook attachment alike.
- ``grad_fn_label`` is its own selector kind (backward exact grad_fn label).

This module is deliberately NOT exported from ``torchlens._ir.__init__``:
importing it pulls in ``torchlens._selection.selectors``, and keeping it out
of the package init avoids an import cycle with ``ir.container``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any, Literal, cast

from .._selection.errors import (
    LiveModeLabelError,
    SelectorCapabilityError,
    SelectorCompositionError,
    SiteResolutionError,
)
from .._selection.selectors import (
    BaseSelector,
    CompositeSelector,
    FollowedBySelector,
    NotSelector,
    PrecededBySelector,
)
from .._selection.target import FrozenTargetSpec, TargetSpec
from .._utils._record_group_table import record_member
from ._selector_nodes import (
    _compile_backward_pass_capture,
    _compile_contains,
    _compile_func,
    _compile_func_transform,
    _compile_grad_fn_backward_pass,
    _compile_grad_fn_grad_kind,
    _compile_grad_fn_label,
    _compile_grad_fn_payload,
    _compile_grad_fn_without_op,
    _compile_grad_kind_capture,
    _compile_label,
    _compile_never,
    _compile_regex,
    _Compiler,
    _match_predicate,
    _Matcher,
    _preceded_by_matches,
)
from ._selector_universe import (
    _grad_fn_boundary_aliases,
    _grad_fn_forward_op,
    _is_module_scope_alias,
    _module_containment_candidates,
    _module_output_candidates,
    _output_matches,
    _predicate_callable,
    _site_label_spellings,
    _string_labels,
    live_label_error_message as live_label_error_message,
    looks_like_finalized_label as looks_like_finalized_label,
    module_address_matches as module_address_matches,
    sanitize_transform_kind as sanitize_transform_kind,
)
from .container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from .container_registry import Role

Lifecycle = Literal["capture", "site", "live"]

_LIFECYCLE_DESC: dict[str, str] = {
    "capture": "capture-time predicate evaluation",
    "site": "post-hoc site resolution",
    "live": "live hook matching",
}

#: Selector kinds that describe a forward op's identity or container position.
#: Inside ``tl.grad(...)`` (and only there) they bridge from a backward
#: ``GradFn`` site to its paired forward op; a bare one never reaches the
#: backward universe.
DIRECTION_AGNOSTIC_KINDS = frozenset(
    {
        "label",
        "func",
        "in_module",
        "module",
        "contains",
        "regex",
        "predicate",
        "func_transform",
        "output",
        "output_at",
        "input_at",
    }
)

#: Backward-only kinds that silently never match a capture-time context.
_CAPTURE_SILENT_FALSE_KINDS = frozenset(
    {"grad_fn", "grad_fn_handle", "intervening", "grad_fn_label"}
)


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------


def walk_selector(selector: Any, *, unwrap: bool = False) -> Iterator[Any]:
    """Yield every node of a selector tree in pre-order.

    Parameters
    ----------
    selector:
        Selector tree root. Non-selector nodes are yielded as-is so callers
        can classify them (for example the ``"target_spec"`` sentinel).
    unwrap:
        Whether to descend through a generic ``.selector`` attribute on
        non-selector wrapper objects (predicates produced by ``tl.when``).

    Yields
    ------
    Any
        Selector nodes and non-selector leaves. ``followed_by`` /
        ``preceded_by`` inner predicates are NOT descended into; they are
        payloads of their wrapping selector, not tree children.
    """

    from .._selection.selectors import GradSelector

    # Iterative pre-order walk -- a deep (nested-binary or
    # deserialized) selector tree must never blow the interpreter stack from a
    # diagnostic/validation walk.
    stack: list[Any] = [selector]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, CompositeSelector):
            stack.extend(reversed(node.selectors))
        elif isinstance(node, NotSelector):
            stack.append(node.selector)
        elif isinstance(node, GradSelector):
            # A gradient address is a structural wrapper: its forward site is
            # part of the tree every capability and alias check must see.
            stack.append(node.inner)
        elif unwrap and not isinstance(node, BaseSelector):
            inner = getattr(node, "selector", None)
            if inner is not None:
                stack.append(inner)


def selector_contains_kind(selector: Any, kind: str, *, unwrap: bool = False) -> bool:
    """Return whether a selector tree contains one selector kind.

    Parameters
    ----------
    selector:
        Selector tree to inspect.
    kind:
        Selector-kind name to find.
    unwrap:
        Whether to descend through wrapper objects' ``.selector`` attribute.

    Returns
    -------
    bool
        Whether any node has the requested kind.
    """

    return any(
        isinstance(node, BaseSelector) and node.selector_kind == kind
        for node in walk_selector(selector, unwrap=unwrap)
    )


def module_union_addresses(selector: Any) -> tuple[str, ...] | None:
    """Return the addresses of a selector built only from ``tl.module`` and ``|``.

    Such a selector resolves exactly to the outputs of the matching module
    passes, so it can be settled at module exit. Any other node (``&``,
    ``~``, an empty union, or a non-module kind) returns ``None``.

    Parameters
    ----------
    selector:
        Selector tree to inspect.

    Returns
    -------
    tuple[str, ...] | None
        Module addresses in tree order, or ``None`` when some node is not a
        module term or a non-empty union.
    """

    addresses: list[str] = []
    for node in walk_selector(selector):
        if isinstance(node, CompositeSelector):
            if node.operator != "or" or not node.selectors:
                return None
            continue
        if not isinstance(node, BaseSelector) or node.selector_kind != "module":
            return None
        addresses.append(str(node.selector_value))
    return tuple(addresses) if addresses else None


def module_exit_bound_addresses(selector: Any) -> tuple[str, ...] | None:
    """Return module addresses whose pass outputs contain every op a selector can match.

    A ``tl.module`` term matches only outputs of its module's passes. An ``&``
    composite matches a subset of any one conjunct, so one bounded conjunct
    bounds it; an ``|`` composite is bounded only when every disjunct is. A
    selector bounded this way can keep its deferred escrow to the ops inside
    matching module passes: post-postprocess resolution still applies the whole
    selector, so the addresses only need to cover its matches, not equal them.
    ``~``, non-module kinds, and empty composites are unbounded.

    Parameters
    ----------
    selector:
        Selector tree to inspect.

    Returns
    -------
    tuple[str, ...] | None
        Covering module addresses (optionally pass-qualified) in tree order, or
        ``None`` when some match may lie outside every module pass's outputs.
    """

    if not isinstance(selector, BaseSelector):
        return None
    # Iterative post-order: deep nested-binary trees must not hit the
    # interpreter recursion limit (see ``walk_selector``).
    bounds: dict[int, tuple[str, ...] | None] = {}
    stack: list[tuple[Any, bool]] = [(selector, False)]
    while stack:
        node, children_done = stack.pop()
        if isinstance(node, CompositeSelector) and node.selectors:
            if not children_done:
                stack.append((node, True))
                stack.extend((child, False) for child in node.selectors)
                continue
            bound = _composite_module_bound(
                node.operator, [bounds[id(child)] for child in node.selectors]
            )
        elif isinstance(node, BaseSelector) and node.selector_kind == "module":
            bound = (str(node.selector_value),)
        else:
            bound = None
        bounds[id(node)] = bound
    return bounds[id(selector)]


def _composite_module_bound(
    operator: str, child_bounds: list[tuple[str, ...] | None]
) -> tuple[str, ...] | None:
    """Combine the module bounds of a composite's children."""

    if operator == "and":
        return next((found for found in child_bounds if found is not None), None)
    if any(found is None for found in child_bounds):
        return None
    return tuple(dict.fromkeys(address for found in child_bounds for address in found or ()))


def contains_followed_by(selector: Any, *, unwrap: bool = False) -> bool:
    """Return whether a selector tree contains a ``followed_by`` selector.

    Parameters
    ----------
    selector:
        Selector tree or wrapped predicate to inspect.
    unwrap:
        Whether to descend through wrapper objects' ``.selector`` attribute.

    Returns
    -------
    bool
        Whether the tree contains ``FollowedBySelector``.
    """

    return any(
        isinstance(node, FollowedBySelector) for node in walk_selector(selector, unwrap=unwrap)
    )


def flatten_and_conjuncts(children: Sequence[Any]) -> tuple[Any, ...]:
    """Return the flat conjunct tuple of a possibly nested ``and`` composite.

    ``&`` builds nested binary composites, so ``a & fb & b`` arrives as
    ``(a & fb) & b``. Temporal validation and the retroactive split must not
    be association-sensitive: this expands nested ``and`` children in order so
    every association sees the same conjunct list as the flat n-ary spec.

    Parameters
    ----------
    children:
        Direct children of a conjunction (or any candidate conjunct list).

    Returns
    -------
    tuple[Any, ...]
        Conjuncts with nested ``and`` composites expanded, in evaluation order.
    """

    # Iterative expansion (same order as the historical recursion)
    # so a deep nested-``and`` tree cannot blow the interpreter stack here.
    flat: list[Any] = []
    stack: list[Any] = list(reversed(children))
    while stack:
        child = stack.pop()
        if isinstance(child, CompositeSelector) and child.operator == "and":
            stack.extend(reversed(child.selectors))
        else:
            flat.append(child)
    return tuple(flat)


def split_followed_by_conjunction(
    predicate: Any,
) -> tuple[FollowedBySelector, BaseSelector] | None:
    """Split a supported ``candidate & tl.followed_by(successor)`` conjunction.

    Composites are n-ary and ``&`` nests, so the conjunction is flattened
    first; the supported retroactive shape is one ``followed_by`` conjunct
    among otherwise ordinary selector conjuncts. The candidate is the single
    other conjunct, or the conjunction of all of them.

    Parameters
    ----------
    predicate:
        Candidate save predicate.

    Returns
    -------
    tuple[FollowedBySelector, BaseSelector] | None
        ``(followed_by, candidate)`` for the supported shape, else ``None``.
    """

    if not isinstance(predicate, CompositeSelector) or predicate.operator != "and":
        return None
    conjuncts = flatten_and_conjuncts(predicate.selectors)
    followed = [c for c in conjuncts if isinstance(c, FollowedBySelector)]
    others = [c for c in conjuncts if not isinstance(c, FollowedBySelector)]
    if len(followed) != 1 or not _plain_selector_conjuncts(others):
        return None
    if len(others) == 1:
        return followed[0], cast(BaseSelector, others[0])
    return followed[0], CompositeSelector("and", tuple(others))


def _plain_selector_conjuncts(conjuncts: Sequence[Any]) -> bool:
    """Return whether ``conjuncts`` is a non-empty run of lookahead-free selectors."""

    return bool(conjuncts) and all(
        isinstance(child, BaseSelector) and not contains_followed_by(child) for child in conjuncts
    )


def first_selector_kind_outside(selector: Any, *, allowed: frozenset[str]) -> str | None:
    """Return the first selector kind outside ``allowed`` in a selector tree.

    Parameters
    ----------
    selector:
        Selector or target spec to classify.
    allowed:
        Selector kinds accepted by the caller.

    Returns
    -------
    str | None
        First unsupported selector kind (``"target_spec"`` for non-selector
        nodes), or ``None`` when the whole tree is allowed.
    """

    for node in walk_selector(selector):
        if not isinstance(node, BaseSelector):
            return "target_spec"
        if node.selector_kind not in allowed:
            return str(node.selector_kind)
    return None


# ---------------------------------------------------------------------------
# Shared matching primitives
# ---------------------------------------------------------------------------


def _capability_error(kind: str, lifecycle: str) -> SelectorCapabilityError:
    """Build the typed refusal for an unsupported ``(kind, lifecycle)`` pair."""

    where = _LIFECYCLE_DESC.get(lifecycle, lifecycle)
    if kind == "followed_by":
        return SelectorCapabilityError(
            "tl.followed_by(...) is capture-time-only retroactive save sugar: "
            "trace(save=candidate & tl.followed_by(successor), lookback=N) saves matching "
            f"candidates while the forward runs. {where} has no retroactive window.",
            code="followed_by_selector_unsupported",
            selector_kind=kind,
            lifecycle=lifecycle,
        )
    if kind == "preceded_by":
        return SelectorCapabilityError(
            "tl.preceded_by(...) matches against the capture-time lookback window and is "
            f"only supported in trace(save=...) / record(save=...) predicates, not {where}.",
            code="preceded_by_selector_unsupported",
            selector_kind=kind,
            lifecycle=lifecycle,
        )
    if kind == "input_at":
        return SelectorCapabilityError(
            "tl.model_input(...) resolves saved input placeholders, but model inputs are not "
            "live hook application sites. Use trace(..., intervene=...) on downstream ops "
            "or mutate the model input before capture.",
            code="model_input_selector_not_live",
            selector_kind=kind,
            lifecycle=lifecycle,
        )
    if kind == "facet":
        return SelectorCapabilityError(
            "tl.facet(...) / tl.head(...) selectors resolve through intervention mutators "
            "(Trace.do, also as a union: do(module_a & tl.head(i) | module_b & tl.head(j), "
            "edit)) and, on a finished trace, as a bare selector via trace[selector]; not via "
            f"{where}{' (find_sites, other composite selectors)' if lifecycle == 'site' else ''}.",
            code="facet_selector_unsupported",
            selector_kind=kind,
            lifecycle=lifecycle,
        )
    if kind == "site":
        return SelectorCapabilityError(
            "tl.site(...) needs a structural site key, which does not exist on "
            f"{where}: final structural numbering is absent mid-attach. Resolve "
            "the site on the finished trace (trace[selector] / do), or pass "
            "the site selector through trace(save=/intervene=), which arms the "
            "live minter.",
            code="site_selector_not_live",
            selector_kind=kind,
            lifecycle=lifecycle,
        )
    return SelectorCapabilityError(
        f"Unsupported selector kind {kind!r} for {where}.",
        code="selector_kind_unsupported",
        selector_kind=kind,
        lifecycle=lifecycle,
    )


#: Kinds refused UPFRONT per lifecycle. The site set must stay in lockstep
#: with the site lane's compiled refusals (contract-tested); the live set
#: names the kinds invalid in EVERY hook direction — backward-only kinds stay
#: out because backward live hooks route them through the backward matcher.
_UPFRONT_UNSUPPORTED_KINDS: dict[str, frozenset[str]] = {
    "site": frozenset({"followed_by", "preceded_by", "facet"}),
    # "site" (the structural selector): live hook-attach proxies carry no
    # structural key mid-forward; final structural addressing there is
    # exactly the impossibility that creates the escrow lane (leverage
    # G-LIVE), so it refuses typed rather than guessing.
    "live": frozenset({"followed_by", "preceded_by", "input_at", "site"}),
}


def ensure_supported(selector: Any, *, lifecycle: Lifecycle) -> None:
    """Raise upfront when a selector tree contains an unsupported kind.

    Used by post-hoc site resolution and live hook attachment so an
    unsupported kind refuses before per-subject evaluation (short-circuit
    evaluation must never hide a refusal behind a non-matching sibling).

    Parameters
    ----------
    selector:
        Selector tree to validate.
    lifecycle:
        Lifecycle about to evaluate the tree.

    Raises
    ------
    SelectorCapabilityError
        If any node's kind cannot be evaluated in this lifecycle.
    """

    unsupported = _UPFRONT_UNSUPPORTED_KINDS.get(lifecycle, frozenset())
    for node in walk_selector(selector):
        if not isinstance(node, BaseSelector):
            continue
        kind = str(node.selector_kind)
        if kind in unsupported:
            raise _capability_error(kind, lifecycle)


#: Attribute under which a selector carries its compiled plans (``{lane:
#: matcher}``). Selectors are frozen dataclasses, so the slot is written with
#: ``object.__setattr__``; a plan lives exactly as long as its selector and is
#: compiled once per lane.
_PLAN_SLOT = "_tl_plans"
#: The backward lane: ``evaluate(..., lifecycle="site")`` on a ``GradFn`` site.
_GRAD_FN_LANE = "grad_fn"

#: Label-family kinds whose forward-lane plans carry the finalized-label guard.
_LABEL_GUARDED_KINDS = frozenset({"label", "contains", "regex"})
#: Lifecycle a lane's non-selector composite children are normalized under.
_LANE_LIFECYCLE: dict[str, Lifecycle] = {
    "capture": "capture",
    "site": "site",
    "live": "live",
    _GRAD_FN_LANE: "site",
}
#: Backward-only kinds that silently never match a forward site post hoc.
_SITE_SILENT_FALSE_KINDS = frozenset(
    {
        "grad_fn",
        "grad_fn_handle",
        "grad_fn_label",
        "intervening",
        "without_op",
        "grad_kind",
        "backward_pass",
    }
)


def evaluate(selector: Any, subject: Any, *, lifecycle: Lifecycle) -> bool:
    """Return whether one selector matches one subject.

    Parameters
    ----------
    selector:
        Selector, target spec, or lifecycle-accepted shorthand.
    subject:
        Capture context, finalized site, or live hook-site proxy.
    lifecycle:
        Which subject universe and capability set to evaluate under.

    Returns
    -------
    bool
        Whether the selector matches.

    Raises
    ------
    SelectorCapabilityError
        If the selector kind cannot be evaluated in this lifecycle.
    """

    normalized = (
        selector
        if isinstance(selector, BaseSelector)
        else normalize_selector_like(selector, lifecycle=lifecycle)
    )
    if lifecycle == "site" and _is_grad_fn(subject):
        return _evaluate_grad_fn_site(normalized, subject)
    return _evaluate_subject(normalized, subject, lifecycle)


def _evaluate_subject(selector: BaseSelector, subject: Any, lifecycle: str) -> bool:
    """Evaluate one selector against one forward-flavored subject."""

    return _plan(selector, lifecycle)(subject)


def _evaluate_grad_fn_site(selector: BaseSelector, site: Any) -> bool:
    """Evaluate one selector against a backward ``GradFn`` site.

    Node kinds match the GradFn itself. A bare forward kind never matches a
    GradFn: the bridge from a forward site to the GradFn behind it lives only
    in ``tl.grad(...)``, whose inner selector is evaluated against the paired
    forward op (and synthetic boundary aliases sharing the autograd identity).
    """

    return _plan(selector, _GRAD_FN_LANE)(site)


def _plan(selector: BaseSelector, lane: str) -> _Matcher:
    """Return the compiled plan for one selector on one lane, compiling once."""

    plans: dict[str, _Matcher] | None = getattr(selector, _PLAN_SLOT, None)
    if plans is None:
        plans = {}
        object.__setattr__(selector, _PLAN_SLOT, plans)
    matcher = plans.get(lane)
    if matcher is None:
        matcher = plans[lane] = _compile(selector, lane)
    return matcher


def _compile(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile one selector node into its plan for one lane."""

    kind = str(selector.selector_kind)
    compiler = _LANE_COMPILERS.get(lane, {}).get(kind)
    if compiler is None:
        compiler = _LANE_FALLBACKS.get(lane, _compile_capability_refusal)
    matcher = compiler(selector, lane)
    if kind in _LABEL_GUARDED_KINDS and lane != _GRAD_FN_LANE:
        matcher = _with_label_guard(kind, str(selector.selector_value), lane, matcher)
    return matcher


def _child_plan(child: Any, lane: str) -> _Matcher:
    """Return the plan for one composite child."""

    if isinstance(child, BaseSelector):
        return _plan(child, lane)
    return partial(_match_lazy_child, child=child, lifecycle=_LANE_LIFECYCLE.get(lane, "site"))


def _match_lazy_child(subject: Any, *, child: Any, lifecycle: Lifecycle) -> bool:
    """Evaluate a non-selector composite child through :func:`evaluate`."""

    return evaluate(child, subject, lifecycle=lifecycle)


def _compile_and(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile an n-ary conjunction (short-circuits per subject)."""

    if not isinstance(selector, CompositeSelector):
        return _LANE_FALLBACKS.get(lane, _compile_capability_refusal)(selector, lane)
    return partial(_match_all, matchers=tuple(_child_plan(c, lane) for c in selector.selectors))


def _compile_or(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile an n-ary disjunction (short-circuits per subject)."""

    if not isinstance(selector, CompositeSelector):
        return _LANE_FALLBACKS.get(lane, _compile_capability_refusal)(selector, lane)
    return partial(_match_any, matchers=tuple(_child_plan(c, lane) for c in selector.selectors))


def _compile_not(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a negation."""

    if not isinstance(selector, NotSelector):
        return _LANE_FALLBACKS.get(lane, _compile_capability_refusal)(selector, lane)
    return partial(_match_not, matcher=_child_plan(selector.selector, lane))


def _match_all(subject: Any, *, matchers: tuple[_Matcher, ...]) -> bool:
    """Return whether every child plan matches ``subject`` (short-circuit)."""

    return all(matcher(subject) for matcher in matchers)


def _match_any(subject: Any, *, matchers: tuple[_Matcher, ...]) -> bool:
    """Return whether any child plan matches ``subject`` (short-circuit)."""

    return any(matcher(subject) for matcher in matchers)


def _match_not(subject: Any, *, matcher: _Matcher) -> bool:
    """Return the complement of one child plan on ``subject``."""

    return not matcher(subject)


def _compile_site(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``site`` (structural position; address law: valid in every lane).

    Post hoc it reads the persisted ``op.graph.site_key``; on the capture lane it reads the
    streaming live minter armed by the ``tl.trace`` entry; the live hook-proxy lane refuses
    (mid-attach proxies carry no structural key, see :data:`_UPFRONT_UNSUPPORTED_KINDS`).
    """

    if lane == "site":
        return partial(_match_site_key, selector=selector)
    if lane == "capture":
        return partial(_match_site_capture, selector=selector)
    return _compile_capability_refusal(selector, lane)


def _match_site_key(subject: Any, *, selector: BaseSelector) -> bool:
    """Match the persisted ``site_key`` of a finalized site."""

    return _site_key_matches(selector, record_member(subject, "site_key", None))


def _match_site_capture(subject: Any, *, selector: BaseSelector) -> bool:
    """Match the live site key the armed minter renders for a capture context."""

    from torchlens.intervention._site_keys import active_live_minter

    minter = active_live_minter()
    if minter is None:
        raise SelectorCapabilityError(
            "tl.site(...) on the capture lane needs the live site-key "
            "minter, which tl.trace arms automatically when save=/"
            "intervene= contains a site selector. This capture surface "
            "did not arm it (tl.record and the non-PyTorch backends do not "
            "support live site keys yet). Address the finished trace "
            "post hoc (trace[selector] / do), or use tl.trace with "
            "the site selector in save=/intervene=."
        )
    return _site_key_matches(selector, minter.key_for_context(subject))


def _compile_input_at(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``input_at``: a model-input container path (never live)."""

    if lane == "live":
        return _compile_capability_refusal(selector, lane)
    return partial(
        _input_path_subject_matches,
        requested_path=tuple(selector.selector_value),
        lifecycle=lane,
    )


def _compile_predicate(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``predicate`` (``tl.where``)."""

    predicate = _predicate_callable(selector.selector_value)
    if predicate is None:
        return _refuse_predicate_payload
    if lane == "live":
        return partial(_match_predicate_live, predicate=predicate)
    return partial(_match_predicate, predicate=predicate)


def _refuse_predicate_payload(subject: Any) -> bool:
    """Refuse a predicate selector whose payload is not callable."""

    raise SiteResolutionError(
        "tl.where(...) requires a callable predicate.", code="where_predicate_not_callable"
    )


def _match_predicate_live(subject: Any, *, predicate: Callable[[Any], Any]) -> bool:
    """Return the predicate's truth on a live proxy, typed on failure."""

    try:
        return bool(predicate(subject))
    except Exception as exc:
        raise SiteResolutionError(
            "live predicate selector failed at "
            f"{getattr(subject, '_layer_label_raw', '<unknown>')}",
            code="live_predicate_failed",
        ) from exc


def _compile_preceded_by(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``preceded_by``: capture-time lookback over retained predecessors."""

    if lane != "capture" or not isinstance(selector, PrecededBySelector):
        return _compile_capability_refusal(selector, lane)
    return partial(_preceded_by_matches, inner=selector.inner)


def _compile_followed_by(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a standalone ``followed_by`` (always a typed refusal)."""

    if lane != "capture":
        return _compile_capability_refusal(selector, lane)
    return _refuse_followed_by_composition


def _refuse_followed_by_composition(subject: Any) -> bool:
    """Refuse a ``followed_by`` reached outside ``candidate & tl.followed_by(...)``."""

    raise SelectorCompositionError(
        "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
        "standalone, negated, or OR-composed followed_by selectors are unsupported.",
        code="followed_by_composition_unsupported",
    )


def _compile_capability_refusal(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a ``(kind, lane)`` pair the lane cannot evaluate."""

    return partial(_refuse_capability, kind=str(selector.selector_kind), lifecycle=lane)


def _refuse_capability(subject: Any, *, kind: str, lifecycle: str) -> bool:
    """Raise the typed refusal for an unsupported ``(kind, lifecycle)`` pair."""

    raise _capability_error(kind, lifecycle)


def _compile_grad(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad`` on a forward lane: the forward op its inner names.

    A gradient address belongs to the forward op its inner selector names;
    which universe the match lands in is the site plan's decision.
    """

    return _child_plan(selector.selector_value, lane)


def _compile_grad_fn_grad(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad`` on a GradFn: the one bridge to the paired forward op."""

    return _compile_bridged(selector.selector_value)


def _compile_bridged(inner: Any) -> _Matcher:
    """Compile a ``tl.grad`` inner selector against the op behind a GradFn.

    The one bridge from the forward universe to a backward ``GradFn`` site:
    connectives recurse; every forward kind evaluates against the paired
    pass-level op and the boundary aliases sharing its autograd identity.
    """

    selector = normalize_selector_like(inner, lifecycle="site")
    kind = str(selector.selector_kind)
    if kind in {"and", "or"} and isinstance(selector, CompositeSelector):
        matchers = tuple(_compile_bridged(child) for child in selector.selectors)
        return partial(_match_all if kind == "and" else _match_any, matchers=matchers)
    if kind == "not" and isinstance(selector, NotSelector):
        return partial(_match_not, matcher=_compile_bridged(selector.selector))
    return partial(_match_grad_fn_bridge, forward=_plan(selector, "site"))


def _with_label_guard(kind: str, value: str, lane: str, matcher: _Matcher) -> _Matcher:
    """Wrap a label-family plan with the finalized-label diagnostic its lane applies."""

    if lane == "site" or not looks_like_finalized_label(value):
        return matcher
    return partial(_match_guarded_label, kind=kind, value=value, lifecycle=lane, matcher=matcher)


def _match_guarded_label(
    subject: Any, *, kind: str, value: str, lifecycle: str, matcher: _Matcher
) -> bool:
    """Apply the finalized-label refusal for ``subject``, then the label plan."""

    _maybe_guard_label(kind, value, subject, lifecycle)
    return matcher(subject)


def _compile_module(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``module``: ops a module call produces (its output boundary)."""

    return partial(
        _match_module,
        kind="module",
        target=str(selector.selector_value),
        lifecycle=lane,
        candidates=_module_output_candidates,
    )


def _compile_in_module(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``in_module``: ops a module call contains."""

    return partial(
        _match_module,
        kind="in_module",
        target=str(selector.selector_value),
        lifecycle=lane,
        candidates=_module_containment_candidates,
    )


def _match_module(
    subject: Any,
    *,
    kind: str,
    target: str,
    lifecycle: str,
    candidates: Callable[[Any, str], tuple[Any, ...]],
) -> bool:
    """Return whether a module candidate of the subject belongs to ``target``."""

    if _is_module_scope_alias(kind, subject, lifecycle):
        return False
    return any(
        module_address_matches(candidate, target) for candidate in candidates(subject, lifecycle)
    )


def _compile_output(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``output``: an output index or semantic role."""

    return partial(_output_matches, value=selector.selector_value, lifecycle=lane)


def _compile_output_at(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``output_at``: a typed output container path."""

    return partial(_match_output_at, requested=tuple(selector.selector_value))


def _match_output_at(subject: Any, *, requested: tuple[Any, ...]) -> bool:
    """Return whether the subject's container path equals the requested path."""

    return output_path_matches(tuple(getattr(subject, "container_path", ()) or ()), requested)


def _compile_episode_step(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``episode_step`` (the ``at_step`` qualifier)."""

    return partial(_episode_step_matches, value=selector.selector_value, lifecycle=lane)


def _episode_step_matches(subject: Any, *, value: Any, lifecycle: str) -> bool:
    """Evaluate one episode-step selector (the ``at_step`` qualifier).

    Capture/live lifecycles read the LIVE step position from the armed join
    session (the stepped-module boundary hooks); ops between steps carry no
    step and never match. Evaluating outside an episode capture refuses
    typed at the first evaluation -- a step qualifier on a plain capture is
    a category error, and matching-nothing silently would be the inverse of
    the fires-at-every-step wrongness the selector exists to close. The
    post-hoc site lifecycle reads the persisted ``Op.episode_step`` stamps
    (unstamped/plain products are door-guarded at resolution entry).
    """

    steps = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if lifecycle in ("capture", "live"):
        from torchlens._capture._episode_join import active_join_session

        session = active_join_session()
        if session is None:
            from torchlens.intervention._bind_step import active_bind_decode_step

            # Capture-free door: an armed binding counts root
            # forwards, so a bound generate answers the step qualifier with
            # its decode step (prefill is step 0); the episode semantics above
            # are unchanged.
            decode_step = active_bind_decode_step()
            if decode_step is not None:
                return decode_step in steps
            from torchlens.intervention.errors import SelectorCapabilityError

            raise SelectorCapabilityError(
                "at_step(...) names an episode step, but this capture carries "
                "no episode declaration: there are no steps to qualify. "
                "Declare the episode -- tl.trace(model, x, "
                "episode=tl.options.EpisodeOptions(stepped_module=..., ...), "
                "...) -- or drop the step qualifier.",
                code="episode_step_selector_without_episode",
                remedy="declare episode= on the capture, or drop at_step()",
            )
        return session.in_step is not None and session.in_step in steps
    stamp = getattr(subject, "episode_step", None)
    return stamp is not None and stamp in steps


def _match_grad_fn_bridge(site: Any, *, forward: _Matcher) -> bool:
    """Return whether the forward plan matches the GradFn's paired op or an alias."""

    forward_op = _grad_fn_forward_op(site)
    if forward_op is not None and forward(forward_op):
        return True
    return any(forward(alias) for alias in _grad_fn_boundary_aliases(site))


def output_path_matches(saved_path: tuple[Any, ...], requested_path: tuple[Any, ...]) -> bool:
    """Return whether a captured typed path matches a user path.

    Parameters
    ----------
    saved_path:
        Captured typed output path.
    requested_path:
        User path using plain indices, keys, or field names.

    Returns
    -------
    bool
        Whether both paths address the same output leaf.
    """

    if len(saved_path) != len(requested_path):
        return False
    return all(
        _output_path_component_matches(saved_component, requested_component)
        for saved_component, requested_component in zip(saved_path, requested_path)
    )


def _output_path_component_matches(saved_component: Any, requested_component: Any) -> bool:
    """Return whether one captured path component matches a user component."""

    if isinstance(saved_component, TupleIndex):
        return saved_component.index == requested_component
    if isinstance(saved_component, (DictKey, HFKey)):
        return saved_component._key == requested_component
    if isinstance(saved_component, (NamedField, DataclassField)):
        return saved_component._name == requested_component
    return saved_component == requested_component


def _raise_for_finalized_live_label(selector_kind: str, label_value: str) -> None:
    """Reject finalized-looking label selectors during live capture."""

    if looks_like_finalized_label(label_value):
        raise LiveModeLabelError(
            live_label_error_message(label_value, selector_kind),
            code="label_selector_not_live",
        )


def _is_grad_fn(subject: Any) -> bool:
    """Return whether a site subject is a backward ``GradFn`` log."""

    from torchlens._data_classes.grad_fn import GradFn  # deferred: cycle

    return isinstance(subject, GradFn)


#: Lookup-key types whose values never compare equal to a ``str`` target.
_NEVER_EQUAL_TO_STR = frozenset({int, bool, float, type(None)})


def _label_site_index(sites: Sequence[Any]) -> dict[str, tuple[Any, ...]] | None:
    """Map each label spelling to the sites a ``label`` selector matches post hoc.

    The universe is exactly the ``"label"`` plan node (:func:`_match_label` on the site lane)
    under the ``"site"`` lifecycle: the :data:`_STRING_LABEL_ATTRS` spellings
    and the site's ``lookup_keys``. ``index.get(target, ())`` therefore equals
    the sites, in order, for which ``evaluate(label(target), site,
    lifecycle="site")`` is true.
    """

    index: dict[str, list[Any]] = {}
    for site in sites:
        if _is_grad_fn(site):
            return None
        keys = dict.fromkeys(_string_labels(site, "site"))
        for key in getattr(site, "lookup_keys", ()) or ():
            if type(key) in _NEVER_EQUAL_TO_STR:
                continue
            if type(key) is not str:
                return None
            keys[key] = None
        for key in keys:
            index.setdefault(key, []).append(site)
    return {key: tuple(matched) for key, matched in index.items()}


def _maybe_guard_label(kind: str, value: str, subject: Any, lifecycle: str) -> None:
    """Apply the finalized-label diagnostic where the lifecycle requires it."""

    if lifecycle == "live":
        _raise_for_finalized_live_label(kind, value)
        return
    if lifecycle == "capture":
        from .predicate import RecordContext

        if isinstance(subject, RecordContext):
            _raise_for_finalized_live_label(kind, value)


def module_union_matches(subject: Any, addresses: frozenset[str]) -> bool:
    """Return whether a capture subject matches a union of ``tl.module`` addresses.

    Gives the answer :func:`evaluate` gives on the capture lane for a selector
    that :func:`module_union_addresses` maps to ``addresses``: ``tl.module``
    there reads only the subject's module-output candidates, so a plain label
    candidate matches by set membership of the label or its address part (the
    string branch of :func:`module_address_matches`), and any other candidate
    spelling is compared address by address.

    Parameters
    ----------
    subject:
        Capture-time record context.
    addresses:
        Module addresses, optionally pass-qualified (``"block:2"``).

    Returns
    -------
    bool
        Whether any module-output candidate of ``subject`` matches an address.
    """

    for candidate in _module_output_candidates(subject, "capture"):
        if type(candidate) is str and not candidate.startswith("("):
            if candidate in addresses or candidate.rsplit(":", 1)[0] in addresses:
                return True
        elif any(module_address_matches(candidate, address) for address in addresses):
            return True
    return False


def _input_path_subject_matches(
    subject: Any, requested_path: tuple[Any, ...], lifecycle: str
) -> bool:
    """Return whether a subject consumes an input at ``requested_path``."""

    if lifecycle == "site" and _site_input_path_matches(subject, requested_path):
        return True
    return any(
        _occurrence_path_matches(getattr(container, "leaf_occurrences", ()) or (), requested_path)
        for container in getattr(subject, "input_containers", ()) or ()
    )


def _site_input_path_matches(subject: Any, requested_path: tuple[Any, ...]) -> bool:
    """Return whether a finalized site produced a model-input leaf at ``requested_path``."""

    trace = getattr(subject, "trace", None)
    site_labels = _site_label_spellings(subject)
    for record in getattr(trace, "_containers", {}).values():
        for snapshot in getattr(record, "snapshots", ()) or ():
            if getattr(snapshot, "role", None) != Role.MODEL_INPUT:
                continue
            occurrences = getattr(snapshot, "leaf_occurrences", ()) or ()
            if _occurrence_path_matches(occurrences, requested_path, site_labels):
                return True
    return False


def _occurrence_path_matches(
    occurrences: Any, requested_path: tuple[Any, ...], producer_labels: set[str] | None = None
) -> bool:
    """Return whether any leaf occurrence sits at ``requested_path``."""

    for occurrence in occurrences:
        if producer_labels is not None and occurrence.producer_op_label not in producer_labels:
            continue
        if output_path_matches(tuple(occurrence.path), requested_path):
            return True
    return False


#: Kinds every forward lane compiles the same way (lane differences live in
#: the compiled node's bound arguments, never in evaluation-time branching).
_COMMON_COMPILERS: dict[str, _Compiler] = {
    "and": _compile_and,
    "or": _compile_or,
    "not": _compile_not,
    "label": _compile_label,
    "contains": _compile_contains,
    "regex": _compile_regex,
    "func": _compile_func,
    "func_transform": _compile_func_transform,
    "module": _compile_module,
    "in_module": _compile_in_module,
    "site": _compile_site,
    "output": _compile_output,
    "output_at": _compile_output_at,
    "input_at": _compile_input_at,
    "predicate": _compile_predicate,
    "episode_step": _compile_episode_step,
    "preceded_by": _compile_preceded_by,
    "followed_by": _compile_followed_by,
    "grad": _compile_grad,
}
#: Per-lane compile tables: one node kind per selector form.
_LANE_COMPILERS: dict[str, dict[str, _Compiler]] = {
    "capture": {
        **_COMMON_COMPILERS,
        "grad_kind": _compile_grad_kind_capture,
        "backward_pass": _compile_backward_pass_capture,
        **dict.fromkeys(_CAPTURE_SILENT_FALSE_KINDS, _compile_never),
    },
    "site": {
        **_COMMON_COMPILERS,
        **dict.fromkeys(_SITE_SILENT_FALSE_KINDS, _compile_never),
    },
    "live": dict(_COMMON_COMPILERS),
    _GRAD_FN_LANE: {
        "and": _compile_and,
        "or": _compile_or,
        "not": _compile_not,
        "grad": _compile_grad_fn_grad,
        # A bare forward kind never reaches the backward universe.
        **dict.fromkeys(DIRECTION_AGNOSTIC_KINDS, _compile_never),
        "intervening": _compile_grad_fn_without_op,
        "without_op": _compile_grad_fn_without_op,
        "grad_fn_label": _compile_grad_fn_label,
        "grad_fn": _compile_grad_fn_payload,
        "grad_kind": _compile_grad_fn_grad_kind,
        "backward_pass": _compile_grad_fn_backward_pass,
    },
}
#: What a lane compiles for a kind its table does not list.
_LANE_FALLBACKS: dict[str, _Compiler] = {
    "capture": _compile_capability_refusal,
    "site": _compile_capability_refusal,
    "live": _compile_capability_refusal,
    _GRAD_FN_LANE: _compile_never,
}

# ---------------------------------------------------------------------------
# The one spec deserializer: a builder per target-spec kind
# ---------------------------------------------------------------------------

#: One spec builder: ``(kind, value, metadata, lifecycle)`` to the selector, or
#: ``None`` when the payload shape is not one the kind accepts (the caller
#: raises the single unsupported-kind refusal).
_SpecBuilder = Callable[[str, Any, dict[str, Any], str], "BaseSelector | None"]


def selector_from_spec(
    kind: str,
    value: Any,
    metadata: dict[str, Any] | None,
    *,
    lifecycle: Lifecycle = "site",
) -> BaseSelector:
    """Build a selector from a target-spec payload.

    Parameters
    ----------
    kind:
        Selector kind from a target spec.
    value:
        Selector payload.
    metadata:
        Selector metadata.
    lifecycle:
        Lifecycle the selector will be evaluated under. Drives only refusals:
        live matching refuses ``input_at`` and ``facet`` specs here with the
        capability message.

    Returns
    -------
    BaseSelector
        Selector matching the target spec.

    Raises
    ------
    SelectorCapabilityError
        If the kind cannot be evaluated in this lifecycle.
    SiteResolutionError
        If the payload shape is unsupported.
    """

    builder = _spec_builders().get(kind)
    selector = None if builder is None else builder(kind, value, metadata or {}, lifecycle)
    if selector is None:
        raise SiteResolutionError(
            f"Unsupported target spec selector kind {kind!r}.", code="selector_kind_unsupported"
        )
    return selector


def _spec_builders() -> dict[str, _SpecBuilder]:
    """Return the kind-to-builder table (the selector factories import lazily)."""

    from .._selection.selectors import (
        FacetSelector,
        SiteSelector,
        at_step,
        func,
        func_transform,
        grad,
        grad_fn,
        grad_fn_label,
        grad_input,
        grad_output,
        in_backward_pass,
        in_module,
        label,
        label_contains,
        label_matches,
        model_input,
        model_output,
        module,
        output_at,
        where,
        without_op,
    )

    return {
        "label": partial(_spec_from_string, factory=label),
        "grad_fn_label": partial(_spec_from_string, factory=grad_fn_label),
        "module": partial(_spec_from_string, factory=module),
        "contains": partial(_spec_from_string, factory=label_contains),
        "regex": partial(_spec_from_string, factory=label_matches),
        "in_module": partial(_spec_from_string, factory=in_module),
        "func": partial(_spec_func, factory=func),
        "func_transform": partial(_spec_func_transform, factory=func_transform),
        "output": partial(_spec_from_value, factory=model_output),
        "output_at": partial(_spec_from_value, factory=output_at),
        "input_at": partial(_spec_input_at, factory=model_input),
        "facet": partial(_spec_facet, factory=FacetSelector),
        "site": partial(_spec_site, factory=SiteSelector),
        "predicate": partial(_spec_predicate, factory=where),
        "grad_fn": partial(_spec_grad_fn, factory=grad_fn),
        "intervening": partial(_spec_constant, factory=without_op),
        "without_op": partial(_spec_constant, factory=without_op),
        "grad_kind": partial(_spec_grad_kind, factory=(grad_input, grad_output)),
        "backward_pass": partial(_spec_backward_pass, factory=in_backward_pass),
        "episode_step": partial(_spec_episode_step, factory=at_step),
        "grad": partial(_spec_grad, factory=grad),
        "not": _spec_not,
        "and": _spec_composite,
        "or": _spec_composite,
        "followed_by": _spec_lookaround,
        "preceded_by": _spec_lookaround,
    }


def _spec_from_string(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector | None:
    """Build a selector whose payload is one string literal."""

    selector = factory(str(value))
    return selector if isinstance(selector, BaseSelector) else None


def _spec_from_value(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build a selector from its payload as-is (``output``, ``output_at``)."""

    return cast(BaseSelector, factory(value))


def _spec_constant(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build a payload-free selector (``intervening`` / ``without_op``)."""

    return cast(BaseSelector, factory())


def _spec_func(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``func`` from a name or a ``{"name": ..., "output": ...}`` payload."""

    if isinstance(value, dict):
        return cast(BaseSelector, factory(str(value.get("name")), output=value.get("output")))
    return cast(BaseSelector, factory(str(value)))


def _spec_func_transform(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``func_transform`` from an optional transform kind."""

    return cast(BaseSelector, factory(None if value is None else str(value)))


def _spec_input_at(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``input_at`` (``tl.model_input``); the live lifecycle refuses it."""

    if lifecycle == "live":
        raise _capability_error(kind, lifecycle)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(BaseSelector, factory(*value))
    return cast(BaseSelector, factory(value))


def _spec_facet(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``facet`` from its name / head-index / module-address payload."""

    if lifecycle == "live":
        raise _capability_error(kind, lifecycle)
    if isinstance(value, dict):
        name = value.get("name")
        head_index = value.get("head_index")
        module_address = value.get("module_address")
        if name is not None or head_index is not None:
            return cast(
                BaseSelector,
                factory(
                    None if name is None else str(name),
                    head_index=None if head_index is None else int(head_index),
                    module_address=None if module_address is None else str(module_address),
                ),
            )
    raise SiteResolutionError(
        f"Unsupported facet selector payload {value!r}.", code="facet_selector_invalid"
    )


def _spec_site(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``site`` from its structural-key payload mapping."""

    payload = dict(value or {})
    return cast(
        BaseSelector,
        factory(
            payload.get("key"),
            module_path=payload.get("module_path"),
            op_type=payload.get("op_type"),
            slot=payload.get("slot"),
            ordinal=payload.get("ordinal"),
        ),
    )


def _spec_predicate(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector | None:
    """Build ``predicate`` (``tl.where``) from a callable payload."""

    if not callable(value):
        return None
    return cast(BaseSelector, factory(value, name_hint=metadata.get("name_hint")))


def _spec_grad_fn(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``grad_fn`` from its type / label-pattern / custom payload."""

    payload = dict(value or {})
    return cast(
        BaseSelector,
        factory(
            payload.get("type"),
            label=payload.get("grad_fn_label_pattern"),
            is_custom=payload.get("is_custom"),
        ),
    )


def _spec_grad_kind(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: tuple[Any, Any]
) -> BaseSelector:
    """Build ``grad_kind`` from the ``(grad_input, grad_output)`` factory pair."""

    grad_input, grad_output = factory
    return cast(BaseSelector, grad_input() if value == "grad_input" else grad_output())


def _spec_backward_pass(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``backward_pass`` from an integer pass index."""

    if not isinstance(value, int):
        raise SiteResolutionError(
            "backward_pass target specs require an integer pass index.",
            code="backward_pass_selector_invalid",
        )
    return cast(BaseSelector, factory(value))


def _spec_episode_step(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``episode_step`` (``tl.at_step``) from a sequence of step indices."""

    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(BaseSelector, factory(*(int(step) for step in value)))
    raise SiteResolutionError(
        "episode_step target specs require a sequence of 0-based step indices.",
        code="episode_step_selector_invalid",
        remedy="rebuild the selector with at_step(<step indices>)",
    )


def _spec_grad(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str, *, factory: Any
) -> BaseSelector:
    """Build ``grad``: a bare label string stays verbatim, else the normalized inner."""

    if not isinstance(value, str):
        value = normalize_selector_like(value, lifecycle=cast(Lifecycle, lifecycle))
    return cast(BaseSelector, factory(value))


def _spec_not(kind: str, value: Any, metadata: dict[str, Any], lifecycle: str) -> BaseSelector:
    """Build ``not`` by negating the normalized inner selector."""

    return ~normalize_selector_like(value, lifecycle=cast(Lifecycle, lifecycle))


def _spec_composite(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str
) -> BaseSelector:
    """Build an n-ary ``and`` / ``or`` from a sequence of nested selector specs."""

    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SiteResolutionError(
            f"{kind!r} target specs require a sequence of nested selectors.",
            code="composite_selector_invalid",
        )
    children = tuple(
        normalize_selector_like(child, lifecycle=cast(Lifecycle, lifecycle)) for child in value
    )
    return CompositeSelector(cast("Literal['and', 'or']", kind), cast(Any, children))


def _spec_lookaround(
    kind: str, value: Any, metadata: dict[str, Any], lifecycle: str
) -> BaseSelector:
    """Build ``followed_by`` / ``preceded_by`` (capture-time only) from an inner payload."""

    if lifecycle != "capture":
        raise _capability_error(kind, lifecycle)
    inner = value
    if isinstance(inner, str):
        raise SiteResolutionError(
            f"tl.{kind}(...) target spec carries an opaque audit payload {inner!r}; "
            "an inner predicate saved by repr cannot be reconstructed. Rebuild the "
            "selector in code, or re-save the spec with a structural selector inner "
            "such as tl.func(...).",
            code="lookback_selector_inner_opaque",
        )
    if not isinstance(inner, BaseSelector) and not callable(inner):
        inner = normalize_selector_like(inner, lifecycle="capture")
    if kind == "followed_by":
        return FollowedBySelector(inner)
    return PrecededBySelector(inner)


def _site_key_matches(selector: Any, rendered_key: Any) -> bool:
    """Match one ``site`` selector against one rendered site key (or None)."""

    from .._selection.selectors import SiteSelector

    if not isinstance(selector, SiteSelector):
        selector = selector_from_spec("site", selector.selector_value, None)
    return bool(selector.matches_key(rendered_key if isinstance(rendered_key, str) else None))


def _selector_from_site_table(site_table: Any) -> BaseSelector:
    """Lower a ``find_sites()`` _SiteTable into a label-union selector."""

    from .._selection.selectors import label

    site_labels: list[str] = []
    for site in site_table:
        site_label = getattr(site, "layer_label", None)
        if isinstance(site_label, str) and site_label not in site_labels:
            site_labels.append(site_label)
    if not site_labels:
        raise SiteResolutionError(
            "This _SiteTable resolved 0 sites, so there is nothing to "
            "address; re-run find_sites(...) with a matching query.",
            code="site_table_empty",
        )
    selector: BaseSelector = label(site_labels[0])
    for site_label in site_labels[1:]:
        selector = selector | label(site_label)
    return selector


def normalize_selector_like(selector_like: Any, *, lifecycle: Lifecycle) -> BaseSelector:
    """Normalize a selector-like input to a selector.

    Parameters
    ----------
    selector_like:
        Selector, target spec, frozen target spec, or lifecycle shorthand.
        Bare strings keep their historical per-lifecycle meaning: post-hoc
        queries treat them as substring searches (``tl.label_contains``), live hook
        targets as exact labels (``tl.label``).
    lifecycle:
        Lifecycle the selector will be evaluated under.

    Returns
    -------
    BaseSelector
        Normalized selector object.

    Raises
    ------
    SiteResolutionError
        If the input shape is unsupported.
    """

    if isinstance(selector_like, BaseSelector):
        return selector_like
    spec_selector = _spec_selector_like(selector_like, lifecycle)
    if spec_selector is not None:
        return spec_selector
    shorthand = _shorthand_selector_like(selector_like, lifecycle)
    if shorthand is not None:
        return shorthand
    # An ``Op``/``Layer`` RECORD is the most precise site spelling a user can
    # hold. Lower it to its PASS-QUALIFIED ``Op.label`` selector(s) here, so the
    # ``set_only`` do door, ``_resolve_sites(record)`` and the mutator site validation all address
    # exactly the pass(es) the record names -- never the bare ``layer_label``
    # (the LAST pass of a multi-pass layer) the live arm below would pick.
    # Lazy import: ``intervention.hooks`` imports this module.
    from ..intervention.hooks import lower_record_site_target

    lowered = lower_record_site_target(selector_like)
    if lowered is not selector_like:
        return lowered
    if lifecycle == "live" and hasattr(selector_like, "layer_label"):
        from .._selection.selectors import label

        return label(str(selector_like.layer_label))
    if lifecycle == "live":
        raise SiteResolutionError(
            f"Unsupported live hook selector {selector_like!r}.",
            code="live_hook_selector_unsupported",
        )
    raise SiteResolutionError(
        f"Unsupported site query {selector_like!r}.", code="site_query_unsupported"
    )


def _spec_selector_like(selector_like: Any, lifecycle: Lifecycle) -> BaseSelector | None:
    """Build the selector a target spec names, or ``None`` for any other input."""

    if isinstance(selector_like, TargetSpec):
        metadata: dict[str, Any] | None = selector_like.metadata
    elif isinstance(selector_like, FrozenTargetSpec):
        metadata = dict(selector_like.metadata)
    else:
        return None
    return selector_from_spec(
        selector_like.selector_kind, selector_like.selector_value, metadata, lifecycle=lifecycle
    )


def _shorthand_selector_like(selector_like: Any, lifecycle: Lifecycle) -> BaseSelector | None:
    """Lower the lifecycle shorthands, or return ``None`` for any other input."""

    if isinstance(selector_like, str):
        from .._selection.selectors import label, label_contains

        return label(selector_like) if lifecycle == "live" else label_contains(selector_like)
    if type(selector_like).__name__ == "_SiteTable":
        # find_sites() output is accepted directly (leverage B3/NEW-4): the
        # zero-match refusal RECOMMENDS log.find_sites(...), so its result
        # must be a valid address, not a dead end.
        return _selector_from_site_table(selector_like)
    if lifecycle == "capture" and callable(selector_like):
        from .._selection.selectors import where

        return where(selector_like)
    return None


__all__ = [
    "DIRECTION_AGNOSTIC_KINDS",
    "Lifecycle",
    "contains_followed_by",
    "ensure_supported",
    "evaluate",
    "first_selector_kind_outside",
    "flatten_and_conjuncts",
    "live_label_error_message",
    "looks_like_finalized_label",
    "module_address_matches",
    "normalize_selector_like",
    "output_path_matches",
    "sanitize_transform_kind",
    "selector_contains_kind",
    "selector_from_spec",
    "split_followed_by_conjunction",
    "walk_selector",
]
