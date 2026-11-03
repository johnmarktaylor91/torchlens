"""Resolve TorchLens intervention selectors against model logs."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from .._ir.selector_eval import (
    _is_grad_fn,
    _label_site_index,
    ensure_supported,
    evaluate,
    normalize_selector_like,
    selector_contains_kind,
    walk_selector,
)
from ._display import site_table_card, site_table_repr
from ._site_refusals import (
    check_site_query_arguments,
    refuse_excess_fanout,
    refuse_unmatched_plan,
)
from .errors import (
    MultiMatchWarning,
    SelectorCompositionError,
    SiteResolutionError,
)
from .module_alias_guard import refuse_trace_alias_spellings
from .selectors import (
    BaseSelector,
    CompositeSelector,
    FacetSelector,
    GradSelector,
    NotSelector,
    classify_selector_universe,
)
from .target import FrozenTargetSpec, TargetSpec

if TYPE_CHECKING:
    import pandas as pd

    from torchlens._data_classes.grad_fn import GradFn
    from torchlens._data_classes.layer import Layer
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace

SelectorInput = BaseSelector | TargetSpec | FrozenTargetSpec | str
if TYPE_CHECKING:
    Site: TypeAlias = Op | Layer | GradFn
else:
    Site: TypeAlias = Any

#: The three site universes a resolved address lands in (SPEC 1.4).
BranchUniverse: TypeAlias = Literal["activation", "gradient", "node"]
#: Legacy direction names ``_iter_sites`` still accepts from callers that have
#: not moved to universe names: forward ops and backward GradFn records.
_LEGACY_UNIVERSES: dict[str, BranchUniverse] = {"forward": "activation", "backward": "node"}


@dataclass(frozen=True)
class _SiteTable:
    """Internal record-grain result of resolving a selector (ordered, indexable).

    Not public: the public resolution door is ``trace[selector]``, which returns
    a ``tl.Selection`` of ``tl.types.Site`` rows for a set-naming selector.
    """

    _sites: tuple[Site, ...]
    selector: Any | None = None
    _signals: tuple[str, ...] | None = None

    def __len__(self) -> int:
        """Return the number of resolved sites.

        Returns
        -------
        int
            Number of layer-pass records in the table.
        """

        return len(self._sites)

    def __iter__(self) -> Iterator[Site]:
        """Iterate through resolved sites in execution order.

        Returns
        -------
        Iterator[Op]
            Iterator over layer-pass records.
        """

        return iter(self._sites)

    def __getitem__(self, idx: int | slice) -> Site | _SiteTable:
        """Return one site or a sliced site table.

        Parameters
        ----------
        idx:
            Integer index or slice.

        Returns
        -------
        Op | _SiteTable
            Single layer pass for integer indexes, table for slices.
        """

        if isinstance(idx, slice):
            return _SiteTable(
                self._sites[idx],
                selector=self.selector,
                _signals=None if self._signals is None else self._signals[idx],
            )
        return self._sites[idx]

    def _site_signals(self) -> tuple[str, ...]:
        """Return the pass each entry belongs to: ``"forward"`` or ``"backward"``.

        Entries resolved through a site plan carry their branch's signal (a
        ``tl.grad`` site is a forward op whose ``"backward"`` gradient is
        addressed); otherwise a GradFn record is ``"backward"`` and every
        other record ``"forward"``.
        """

        if self._signals is not None:
            return self._signals
        return tuple("backward" if _is_grad_fn(site) else "forward" for site in self._sites)

    def __repr__(self) -> str:
        """Return a compact table representation.

        Returns
        -------
        str
            Summary including count and first labels.
        """

        return site_table_repr(self)

    def __str__(self) -> str:
        """Show a bounded prefix of the resolved sites."""

        return site_table_card(self)

    def where(self, predicate: Callable[[Site], bool]) -> _SiteTable:
        """Filter the table with a predicate.

        Parameters
        ----------
        predicate:
            Callable receiving each layer-pass record.

        Returns
        -------
        _SiteTable
            Filtered table in original execution order.
        """

        kept = [
            (site, signal)
            for site, signal in zip(self._sites, self._site_signals(), strict=True)
            if predicate(site)
        ]
        return _SiteTable(
            tuple(site for site, _ in kept),
            selector=self.selector,
            _signals=tuple(signal for _, signal in kept),
        )

    def first(self) -> Site:
        """Return the first resolved site.

        Returns
        -------
        Op
            First layer-pass record.

        Raises
        ------
        SiteResolutionError
            If the table is empty.
        """

        if not self._sites:
            raise SiteResolutionError(
                "_SiteTable is empty; no first site is available.", code="site_table_empty"
            )
        return self._sites[0]

    def labels(self) -> tuple[str, ...]:
        """Return execution-order labels for resolved sites.

        Returns
        -------
        tuple[str, ...]
            Stable layer labels.
        """

        return tuple(
            str(
                getattr(
                    record,
                    "layer_label",
                    getattr(record, "label", "<unknown>"),
                )
            )
            for record in map(_site_record, self._sites)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas table describing resolved sites.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per resolved site.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [
            {
                "label": getattr(site, "label", None),
                "signal": signal,
                "layer_label": getattr(site, "layer_label", None),
                "call_index": getattr(site, "call_index", None),
                "step_index": getattr(site, "step_index", None),
                "raw_index": getattr(site, "raw_index", None),
                "func_name": getattr(site, "func_name", getattr(site, "name", None)),
                "module": getattr(site, "module", None),
                "modules": getattr(site, "modules", ()),
                "output_of_module_calls": getattr(site, "output_of_module_calls", ()),
            }
            for site, signal in zip(
                map(_site_record, self._sites), self._site_signals(), strict=True
            )
        ]
        return pd.DataFrame(rows)


def _resolve_sites(
    trace: Trace,
    selector: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> _SiteTable:
    """Resolve a selector or accepted shorthand against a model log.

    Parameters
    ----------
    trace:
        Model log whose captured layer-pass records should be searched.
    selector:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed. ``None`` is rejected in this MVP.

    Returns
    -------
    _SiteTable
        Ordered table of resolved sites.

    Raises
    ------
    AmbiguousSiteError
        If more sites resolve than ``max_fanout`` allows.
    SiteResolutionError
        If no site resolves or the query shape is unsupported.
    """

    check_site_query_arguments(selector, strict=strict, max_fanout=max_fanout)

    selector = _normalize_query(selector)
    plan = resolve_site_plan(trace, selector, strict=strict)
    matched = tuple(site for branch in plan for site in branch.sites)
    signals = tuple(branch.signal for branch in plan for _ in branch.sites)
    refuse_excess_fanout(selector, matched, max_fanout)
    if len(matched) > 1:
        warnings.warn(
            _multi_match_message(selector, matched),
            MultiMatchWarning,
            stacklevel=2,
        )
    return _SiteTable(matched, selector=selector, _signals=signals)


def _resolve_facet_sites(
    log: Trace,
    selector: FacetSelector,
    query: SelectorInput,
) -> tuple[Any, ...]:
    """Resolve a facet selector on a finished trace to the facets it reads.

    The facets are the ones the write doors (``attach_hooks``,
    ``do``) resolve for the same selector, each the ``Facet`` that
    ``record.facets[...]`` serves, so its ``.value`` is the slice an edit at
    those doors receives (a head's slice, an SAE latent's encoder activation).
    A selector that matches several facets keeps its broadcast; the caller's
    fan-out check and multi-match count report apply to them as to op sites.

    Parameters
    ----------
    log:
        Finished trace to read.
    selector:
        Normalized facet selector.
    query:
        The selector as the caller spelled it, for messages.

    Returns
    -------
    tuple[Facet, ...]
        One ``Facet`` per matched facet, in resolution order.

    Raises
    ------
    SiteResolutionError
        Coded ``facet_selector_no_match`` when the trace serves no such facet.
    """

    from .._semantic.facets import Facet
    from ..intervention.hooks import _resolve_facet_specs

    facets = tuple(Facet(spec) for _, spec in _resolve_facet_specs(log, selector))
    if not facets:
        first = next(iter(getattr(log, "layer_labels", ()) or ()), None)
        example = f"trace[{first!r}].facets.keys()" if first is not None else "record.facets.keys()"
        raise SiteResolutionError(
            f"facet selector {query!r} matched 0 facets on this trace: no module or op record "
            "here serves that facet",
            code="facet_selector_no_match",
            remedy=f"list the facets a record serves with {example}, then select one of them",
            selector=repr(query),
        )
    return facets


def _site_record(site: Any) -> Any:
    """Return the record a table entry describes: the home op of a ``Facet``, else itself."""

    from .._semantic.facets import Facet

    return site.spec.home if isinstance(site, Facet) else site


#: Label indexes of the active :func:`_label_index_scope`, keyed by ``id(log)``:
#: ``(log, site universe, universe length, index)``. ``None`` outside a scope.
_LABEL_INDEXES: ContextVar[dict[int, tuple[Any, Sequence[Site], int, Any]] | None] = ContextVar(
    "torchlens_label_indexes", default=None
)


@contextmanager
def _label_index_scope() -> Iterator[None]:
    """Resolve plain ``tl.label`` selectors through one label index per trace in the block.

    Inside the block, the first plain-label resolution against a trace builds
    a map from every label spelling to the sites the selector evaluator would
    match, and later plain-label resolutions against the same trace look the
    answer up instead of evaluating the selector on every op. The caller
    guarantees that no op is added, removed or relabelled inside the block;
    every other check of :func:`_resolve_sites` still runs per query. A nested
    scope reuses the outer one.

    Yields
    ------
    None
        Control while the scope is active.
    """

    if _LABEL_INDEXES.get() is not None:
        yield
        return
    token = _LABEL_INDEXES.set({})
    try:
        yield
    finally:
        _LABEL_INDEXES.reset(token)


def _indexed_label_match(
    log: Trace,
    direction: str,
    selector: BaseSelector,
) -> tuple[Site, ...] | None:
    """Return a plain-label selector's matches from the scope's index, or ``None``.

    Parameters
    ----------
    log:
        Model log being searched.
    direction:
        Resolver search universe (``"activation"``, or its legacy name
        ``"forward"``).
    selector:
        Normalized selector.

    Returns
    -------
    tuple[Site, ...] | None
        Matching sites in execution order, exactly what
        :func:`_resolve_unchecked` returns for the selector; ``None`` when no
        scope is active, the selector is not a plain label, the universe is
        not the forward ops, or the index cannot be built exactly.
    """

    indexes = _LABEL_INDEXES.get()
    if (
        indexes is None
        or direction not in ("forward", "activation")
        or str(selector.selector_kind) != "label"
    ):
        return None
    sites = _iter_sites(log, direction)
    ensure_supported(selector, lifecycle="site")
    cached = indexes.get(id(log))
    if cached is None or cached[0] is not log or cached[1] is not sites or cached[2] != len(sites):
        index = _label_site_index(sites)
        if index is None:
            return None
        cached = (log, sites, len(sites), index)
        indexes[id(log)] = cached
    return cast("tuple[Site, ...]", cached[3].get(str(selector.selector_value), ()))


def _multi_match_message(query: SelectorInput, matched: Sequence[Site]) -> str:
    """Return the multi-match warning text for a resolved site set.

    A set holding a synthetic model-output alias (``output_N``) together with
    the op it aliases does not fan out: a value edit applied at both sites
    compounds on the one returned value, so the warning says so.

    Parameters
    ----------
    query:
        Selector input as the caller spelled it.
    matched:
        Resolved sites, in execution order.

    Returns
    -------
    str
        Warning message.
    """

    records = [_site_record(site) for site in matched]
    matched_labels = {getattr(site, "layer_label", None) for site in records}
    pairs = [
        (str(getattr(site, "layer_label", None)), str(parent))
        for site in records
        if getattr(site, "is_output", False)
        for parent in (getattr(site, "parents", ()) or ())
        if parent in matched_labels
    ]
    if not pairs:
        return f"selector {query!r} matched {len(matched)} sites and will fan out."
    named = ", ".join(f"{alias!r} aliases {producer!r}" for alias, producer in pairs)
    return (
        f"selector {query!r} matched {len(matched)} sites, including a model-output alias "
        f"and the op it aliases ({named}); a value edit applied at both compounds on the "
        "same returned value instead of reaching independent sites. Narrow the selector to "
        "the producing op."
    )


def find_sites(
    log: Trace,
    query: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> _SiteTable:
    """Find matching sites and return a table.

    Parameters
    ----------
    log:
        Model log whose captured layer-pass records should be searched.
    query:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed.

    Returns
    -------
    _SiteTable
        Ordered table of resolved sites.
    """

    check_site_query_arguments(query, strict=strict, max_fanout=max_fanout)

    selector = _normalize_query(query)
    if isinstance(selector, FacetSelector):
        # Facets resolve through the write doors and _resolve_sites only; the
        # post-hoc search refuses them with the capability message.
        ensure_supported(selector, lifecycle="site")
    plan = resolve_site_plan(log, selector, strict=strict, require_match=False)
    matched = tuple(site for branch in plan for site in branch.sites)
    refuse_excess_fanout(query, matched, max_fanout)
    signals = tuple(branch.signal for branch in plan for _ in branch.sites)
    return _SiteTable(matched, selector=query, _signals=signals)


def _guard_episode_step_query(log: Trace, selector: Any) -> None:
    """Door guard for post-hoc episode-step queries (lane F42).

    A step-qualified selector resolving against a product with no step axis
    would match NOTHING silently -- the inverse of the fires-at-every-step
    wrongness the qualifier exists to close -- so the door refuses typed at
    the point of failure instead: plain captures have no steps to qualify,
    and pre-F42 episode artifacts carry no ``Op.episode_step`` stamps.

    Parameters
    ----------
    log:
        Model log the query resolves against.
    selector:
        Normalized query selector.

    Raises
    ------
    SiteResolutionError
        ``episode_step_selector_without_episode`` on plain captures;
        ``episode_step_unstamped`` on stamp-less episode artifacts.
    """

    if not isinstance(selector, BaseSelector) or not selector_contains_kind(
        selector, "episode_step", unwrap=True
    ):
        return
    from .._capture._episode_ledger import capture_kind_for

    if capture_kind_for(log) != "episode":
        raise SiteResolutionError(
            "at_step(...) names an episode step, but this product carries no "
            "episode declaration: there are no steps to qualify. Re-capture "
            "with tl.trace(model, x, episode=tl.options.EpisodeOptions(...)), or "
            "drop the step qualifier.",
            code="episode_step_selector_without_episode",
            remedy="re-capture with episode=, or drop at_step()",
        )
    if not any(getattr(op, "episode_step", None) is not None for op in log._layer_list):
        raise SiteResolutionError(
            "at_step(...) needs the per-op episode-step stamps "
            "(Op.episode_step), but no op on this episode product carries "
            "one -- a pre-stamping artifact. Re-capture (or re-save from a "
            "fresh capture) with this TorchLens version to mint the stamps.",
            code="episode_step_unstamped",
            remedy="re-capture the episode with a stamping TorchLens version",
        )


def _iter_layer_ops(log: Trace) -> Sequence[Op]:
    """Return final layer ops from a completed model log.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    Sequence[Op]
        Execution-order layer-pass records.

    Raises
    ------
    SiteResolutionError
        If the model log has not completed postprocessing.
    """

    if not getattr(log, "_tracing_finished", False):
        raise SiteResolutionError("Sites can only be resolved after the forward pass is complete.")
    return log._layer_list


def _iter_sites(log: Trace, universe: str) -> Sequence[Site]:
    """Return the candidate sites of one universe.

    Parameters
    ----------
    log:
        Model log to inspect.
    universe:
        ``"activation"`` or ``"gradient"`` (both search the forward ops: a
        gradient address names a forward op's output gradient, which exists
        before any backward pass runs) or ``"node"`` (the GradFn records of
        the backward passes). The legacy names ``"forward"`` and
        ``"backward"`` mean ``"activation"`` and ``"node"``.

    Returns
    -------
    Sequence[Site]
        Forward op sites or backward GradFn sites.

    Raises
    ------
    SiteResolutionError
        For the node universe before any backward pass ran.
    """

    resolved = _LEGACY_UNIVERSES.get(universe, universe)
    if resolved in ("activation", "gradient"):
        return _iter_layer_ops(log)
    if resolved != "node":
        raise SiteResolutionError(f"unknown site universe {universe!r}.")
    if not log._autograd_has_backward_pass:
        raise SiteResolutionError("Backward selectors require backward() to run first.")
    return tuple(log._grad_fn_logs.values())


def _resolve_unchecked(
    sites: Sequence[Site],
    query: SelectorInput,
    *,
    strict: bool,
) -> tuple[Site, ...]:
    """Resolve without zero/multi fanout validation.

    Parameters
    ----------
    sites:
        Layer-pass records to search.
    query:
        Selector input.
    strict:
        Whether strict portability rules are active.

    Returns
    -------
    tuple[Op, ...]
        Matching sites in execution order.
    """

    selector = _normalize_query(query)
    if strict and any(
        isinstance(node, BaseSelector) and node.selector_kind == "predicate"
        for node in walk_selector(selector)
    ):
        raise SiteResolutionError(
            "tl.where(...) predicate selectors are non-portable in strict mode."
        )
    ensure_supported(selector, lifecycle="site")
    return tuple(site for site in sites if evaluate(selector, site, lifecycle="site"))


def _selector_resolution_direction(query: SelectorInput) -> Literal["forward", "backward"]:
    """Return the legacy search direction of a selector query.

    Derived from the universe plan for the hook and door modules that still
    read one direction: ``"backward"`` when any branch is a gradient or node
    branch, else ``"forward"``. New code reads :func:`split_site_branches`.

    Parameters
    ----------
    query:
        Selector query to inspect.

    Returns
    -------
    Literal["forward", "backward"]
        Legacy direction.
    """

    branches = split_site_branches(query)
    return "backward" if any(b.universe != "activation" for b in branches) else "forward"


@dataclass(frozen=True)
class SelectorBranch:
    """One universe branch of a selector, before resolution against a trace.

    Attributes
    ----------
    universe:
        ``"activation"``, ``"gradient"`` or ``"node"``.
    selector:
        What the branch matches: the activation selector, the forward site a
        gradient branch names (the ``tl.grad`` wrapper removed), or the node
        selector (pass qualifiers included, as they always were).
    pass_filter:
        The branch's top-level ``tl.in_backward_pass`` qualifiers joined with
        ``&``, or ``None`` for every pass. Activation branches never carry one.
    """

    universe: BranchUniverse
    selector: Any
    pass_filter: BaseSelector | None = None

    @property
    def signal(self) -> Literal["forward", "backward"]:
        """Return the pass this branch's hooks belong to."""

        return "forward" if self.universe == "activation" else "backward"

    def accepts_pass(self, pass_index: int) -> bool:
        """Return whether a backward pass number passes this branch's filter."""

        return pass_filter_accepts(self.pass_filter, pass_index)


@dataclass(frozen=True)
class SiteBranch:
    """One universe branch of a selector resolved against a trace.

    Attributes
    ----------
    universe:
        ``"activation"``, ``"gradient"`` or ``"node"``.
    sites:
        Matched sites in execution order: forward ops for the activation and
        gradient universes (a gradient site is the op whose OUTPUT gradient is
        addressed), GradFn records for the node universe.
    pass_filter:
        Top-level backward pass qualifier, or ``None`` for every pass.
    selector:
        The branch selector that matched ``sites``.
    """

    universe: BranchUniverse
    sites: tuple[Site, ...]
    pass_filter: BaseSelector | None = None
    selector: Any = None

    @property
    def signal(self) -> Literal["forward", "backward"]:
        """Return the pass this branch's hooks belong to."""

        return "forward" if self.universe == "activation" else "backward"

    def accepts_pass(self, pass_index: int) -> bool:
        """Return whether a backward pass number passes this branch's filter."""

        return pass_filter_accepts(self.pass_filter, pass_index)


def pass_filter_accepts(pass_filter: Any, pass_index: int) -> bool:
    """Evaluate a backward-pass qualifier tree against one pass number.

    Parameters
    ----------
    pass_filter:
        ``None`` (every pass), a ``tl.in_backward_pass`` selector, or an
        ``&`` / ``|`` / ``~`` composition of them.
    pass_index:
        One-based backward pass number.

    Returns
    -------
    bool
        Whether the pass is accepted.
    """

    if pass_filter is None:
        return True
    selector = _normalize_query(pass_filter)
    if isinstance(selector, CompositeSelector):
        results = (pass_filter_accepts(child, pass_index) for child in selector.selectors)
        return all(results) if selector.operator == "and" else any(results)
    if isinstance(selector, NotSelector):
        return not pass_filter_accepts(selector.selector, pass_index)
    if selector.selector_kind == "backward_pass":
        return int(selector.selector_value) == int(pass_index)
    raise SiteResolutionError(f"{selector!r} is not a backward pass qualifier.")


def _misplaced_qualifier(selector: Any, *, code: str) -> Exception:
    """Build the refusal for a pass qualifier nested inside a gradient expression."""

    return SelectorCompositionError(
        f"{selector!r} nests a backward pass qualifier inside a gradient expression "
        "(under ~ or within a nested group); a pass qualifier filters a whole gradient "
        "branch.",
        code=code,
        remedy="join the qualifier at the top of the branch: tl.grad(s) & tl.in_backward_pass(2)",
    )


def _split_operand(operand: Any) -> tuple[Any, ...]:
    """Return an operand's children when it is a ``|`` node (selector or spec)."""

    if isinstance(operand, CompositeSelector):
        return tuple(operand.selectors) if operand.operator == "or" else ()
    if isinstance(operand, (TargetSpec, FrozenTargetSpec)) and operand.selector_kind == "or":
        return tuple(_normalize_query(operand).selector_value)
    return ()


def _flatten(operand: Any, operator: Literal["and", "or"]) -> list[Any]:
    """Flatten nested same-operator nodes into one operand list."""

    if operator == "or":
        children = _split_operand(operand)
    elif isinstance(operand, CompositeSelector) and operand.operator == "and":
        children = tuple(operand.selectors)
    elif isinstance(operand, (TargetSpec, FrozenTargetSpec)) and operand.selector_kind == "and":
        children = tuple(_normalize_query(operand).selector_value)
    else:
        children = ()
    if not children:
        return [operand]
    flat: list[Any] = []
    for child in children:
        flat.extend(_flatten(child, operator))
    return flat


def _join(operator: Literal["and", "or"], items: Sequence[Any]) -> Any:
    """Join selectors with one operator; ``None`` for none, the item for one."""

    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return CompositeSelector(operator, tuple(items))


def _unwrap_gradient(selector: Any) -> Any:
    """Return the forward-site expression a gradient-universe expression names."""

    if isinstance(selector, (TargetSpec, FrozenTargetSpec)):
        selector = _normalize_query(selector)
    if isinstance(selector, GradSelector):
        return selector.inner
    if isinstance(selector, CompositeSelector):
        return CompositeSelector(
            selector.operator, tuple(_unwrap_gradient(child) for child in selector.selectors)
        )
    if isinstance(selector, NotSelector):
        return NotSelector(_unwrap_gradient(selector.selector))
    raise _misplaced_qualifier(selector, code="selector_signal_mismatch")


def _branch_for(operand: Any) -> SelectorBranch:
    """Classify one union operand into its universe branch."""

    universe = classify_selector_universe(operand)
    if universe in (None, "activation"):
        return SelectorBranch("activation", operand)
    if universe == "qualifier":
        return SelectorBranch("node", operand, operand)
    if universe == "mixed":
        raise SelectorCompositionError(
            f"{operand!r} mixes universes below its top-level union.",
            code="selector_cross_universe_and",
            remedy="write | at the top of the address and & inside each branch",
        )
    conjuncts = _flatten(operand, "and")
    qualifiers = [c for c in conjuncts if classify_selector_universe(c) == "qualifier"]
    pass_filter = _join("and", qualifiers)
    if universe == "node":
        return SelectorBranch("node", operand, pass_filter)
    rest = [c for c in conjuncts if classify_selector_universe(c) != "qualifier"]
    inner = _join("and", [_unwrap_gradient(c) for c in rest])
    return SelectorBranch("gradient", inner, pass_filter)


def split_site_branches(query: Any) -> tuple[SelectorBranch, ...]:
    """Split a selector into its universe branches without a trace.

    The selector is read with ``|`` at the top; each operand belongs to one
    universe (an ``&`` across universes refused when it was built). Operands
    of one universe and one pass filter merge into one branch, so
    ``tl.grad(a) | tl.grad(b)`` is the single gradient branch ``tl.grad(a |
    b)`` names, while ``tl.grad(s) | s`` keeps a gradient and an activation
    branch. Bare label strings stay verbatim (each door gives them its own
    meaning).

    Parameters
    ----------
    query:
        Selector, target spec, frozen target spec, or bare label string.

    Returns
    -------
    tuple[SelectorBranch, ...]
        Branches in order of first appearance.

    Raises
    ------
    SelectorCompositionError
        For a pass qualifier nested inside a gradient expression, or a tree
        that mixes universes below its top-level union.
    """

    selector = query if isinstance(query, str) else _normalize_query(query)
    operands = _flatten(selector, "or")
    if len(operands) == 1:
        return (_branch_for(selector),)
    groups: list[tuple[SelectorBranch, list[Any]]] = []
    for operand in operands:
        branch = _branch_for(operand)
        key = (branch.universe, branch.pass_filter)
        for first, members in groups:
            if (first.universe, first.pass_filter) == key:
                members.append(branch.selector)
                break
        else:
            groups.append((branch, [branch.selector]))
    return tuple(
        SelectorBranch(first.universe, _join("or", members), first.pass_filter)
        for first, members in groups
    )


def _refuse_strict_bare_strings(selector: Any) -> None:
    """Refuse bare label strings anywhere in a strict-mode selector tree."""

    if isinstance(selector, str) or any(isinstance(node, str) for node in walk_selector(selector)):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector.",
            code="bare_string_selector_strict",
        )


def _resolve_branch_sites(
    trace: Trace, selector: BaseSelector, universe: str, *, strict: bool
) -> tuple[Site, ...]:
    """Resolve one branch selector against its universe's sites."""

    search = "node" if universe == "node" else "activation"
    indexed = _indexed_label_match(trace, search, selector)
    if indexed is not None:
        return indexed
    return _resolve_unchecked(tuple(_iter_sites(trace, search)), selector, strict=strict)


def resolve_site_plan(
    trace: Trace, selector: SelectorInput, *, strict: bool = False, require_match: bool = True
) -> tuple[SiteBranch, ...]:
    """Resolve a selector into one site set per universe branch (SPEC 1.6).

    A gradient branch resolves its forward site in the activation universe
    (same strictness and refusal texts) and keeps the matched forward ops: each
    names its output-gradient boundary, which exists once a backward runs, so
    no backward pass is required. A node branch resolves against the GradFn
    records and still requires one.

    Parameters
    ----------
    trace:
        Finished trace to resolve against.
    selector:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    require_match:
        Whether a plan matching no site at all refuses (the door rule; the
        zero-match refusal recommends ``trace[selector]``) and a bare multi-pass
        label refuses as at the activation door. ``find_sites`` passes False.

    Returns
    -------
    tuple[SiteBranch, ...]
        One branch per (universe, pass filter), in order of first appearance.

    Raises
    ------
    SiteResolutionError
        For a zero-match plan, an unsupported query, or a node branch on a
        trace with no backward pass.
    """

    if strict and isinstance(selector, str):
        _refuse_strict_bare_strings(selector)
    normalized = _normalize_query(selector)
    if isinstance(normalized, FacetSelector):
        facets = _resolve_facet_sites(trace, normalized, normalized)
        return (SiteBranch("activation", tuple(facets), None, normalized),)
    if strict:
        _refuse_strict_bare_strings(normalized)
    _guard_episode_step_query(trace, normalized)
    refuse_trace_alias_spellings(trace, normalized)
    resolved: list[SiteBranch] = []
    for planned in split_site_branches(normalized):
        branch_selector = _normalize_query(planned.selector)
        sites = _resolve_branch_sites(trace, branch_selector, planned.universe, strict=strict)
        resolved.append(SiteBranch(planned.universe, sites, planned.pass_filter, branch_selector))
    if require_match:
        refuse_unmatched_plan(normalized, resolved)
    return tuple(resolved)


def _normalize_query(query: SelectorInput) -> BaseSelector:
    """Normalize a query object to a selector.

    Parameters
    ----------
    query:
        Supported selector input.

    Returns
    -------
    BaseSelector
        Normalized selector object.

    Raises
    ------
    SiteResolutionError
        If the query shape is unsupported.
    """

    return normalize_selector_like(query, lifecycle="site")


__all__ = [
    "SelectorBranch",
    "SiteBranch",
    "_SiteTable",
    "_selector_resolution_direction",
    "find_sites",
    "pass_filter_accepts",
    "resolve_site_plan",
    "_resolve_sites",
    "split_site_branches",
]
