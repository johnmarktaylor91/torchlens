"""The one selector interpreter for capture-time, post-hoc, and live matching.

Every selector-matching decision in TorchLens routes through :func:`evaluate`,
parameterized by a ``lifecycle``:

- ``"capture"``: in-flight ``RecordContext`` objects (and duck-typed layer-like
  objects) evaluated by ``trace(save=...)`` / ``record(save=...)`` predicates.
- ``"site"``: finalized ``Op`` / ``Layer`` / ``GradFn`` sites resolved by
  ``find_sites`` / ``resolve_sites``.
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

This module is deliberately NOT exported from ``torchlens.ir.__init__``:
importing it pulls in ``torchlens.intervention.selectors``, and keeping it out
of the package init avoids an import cycle with ``ir.container``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

from ..backends import TORCH_BACKEND_NAME
from ..intervention.errors import (
    LiveModeLabelError,
    SelectorCapabilityError,
    SelectorCompositionError,
    SiteResolutionError,
)
from ..intervention.selectors import (
    BaseSelector,
    CompositeSelector,
    FollowedBySelector,
    NotSelector,
    PrecededBySelector,
)
from ..intervention.types import FrozenTargetSpec, TargetSpec
from .container import DataclassField, DictKey, HFKey, NamedField, TupleIndex

Lifecycle = Literal["capture", "site", "live"]

#: Alias address -> canonical address for the model the active door runs.
#: A module object registered under several names (``self.alias = self.block``)
#: is labeled under its canonical (first-registered) address; every door that
#: evaluates live or capture subjects publishes its model's map here so a
#: ``tl.module`` / ``tl.in_module`` target spelled with any registered name
#: selects the same module. Post-hoc subjects carry their Trace, which records
#: the map in ``Module.all_addresses``.
_ACTIVE_MODULE_ALIASES: ContextVar[Mapping[str, str] | None] = ContextVar(
    "torchlens_active_module_aliases", default=None
)

_LIFECYCLE_DESC: dict[str, str] = {
    "capture": "capture-time predicate evaluation",
    "site": "post-hoc site resolution",
    "live": "live hook matching",
}

#: Selector kinds that describe a forward op's identity or container position
#: and therefore bridge from a backward ``GradFn`` site to its paired forward op.
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

#: String-label attribute universes per lifecycle, in match order.
_STRING_LABEL_ATTRS: dict[str, tuple[str, ...]] = {
    "capture": ("label", "raw_label", "label_raw", "layer_label", "layer_label_short"),
    "site": ("layer_label", "label", "layer_label_short", "label_short", "_layer_label_raw"),
    "live": ("_layer_label_raw", "_label_raw", "layer_label"),
}

#: Substring/regex universes. Post-hoc search stays on the final label only:
#: matching raw/short spellings would make ``contains("raw")`` match every op.
_SUBSTRING_LABEL_ATTRS: dict[str, tuple[str, ...]] = {
    **_STRING_LABEL_ATTRS,
    "site": ("layer_label",),
}

_FINAL_LABEL_PATTERN = re.compile(r"(?:_\d+_\d+(?::\d+)?$|:\d+$)")


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

    # r-b4 R27-6a: iterative pre-order walk -- a deep (nested-binary or
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

    # r-b4 R27-6a: iterative expansion (same order as the historical recursion)
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
    if len(followed) != 1 or not others:
        return None
    if not all(isinstance(child, BaseSelector) for child in others):
        return None
    if any(contains_followed_by(child) for child in others):
        return None
    if len(others) == 1:
        return followed[0], cast(BaseSelector, others[0])
    return followed[0], CompositeSelector("and", tuple(others))


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


def sanitize_transform_kind(kind: object) -> str:
    """Return the transform-kind spelling used for labels.

    Parameters
    ----------
    kind:
        Transform kind or selector value.

    Returns
    -------
    str
        Lowercase spelling with underscores and dots removed.
    """

    return str(kind).lower().replace("_", "").replace(".", "")


def module_address_matches(module_pass: Any, address: str) -> bool:
    """Return whether a module candidate belongs to an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified label, ``(address, call_index)`` tuple, or tuple repr.
    address:
        Module address with or without pass qualification. A bare address
        matches every call of the module; a pass label (``"block:2"``) matches
        only that call, whichever spelling the candidate uses.

    Returns
    -------
    bool
        Whether the module candidate belongs to the requested address.
    """

    if isinstance(module_pass, tuple) and module_pass and isinstance(module_pass[0], str):
        if module_pass[0] == address:
            return True
        # Live boundary sites carry ``(address, call_index)`` tuples; a pass
        # label must compare against the same "address:call" spelling the
        # string branch below (and bind's boundary hook) uses.
        return len(module_pass) > 1 and f"{module_pass[0]}:{module_pass[1]}" == address
    module_label = str(module_pass)
    if module_label.startswith("("):
        base, sep, call = address.rpartition(":")
        if sep and call.isdigit():
            return f"'{base}', {call})" in module_label or f'"{base}", {call})' in module_label
        return f"'{address}'" in module_label or f'"{address}"' in module_label
    module_address = module_label.rsplit(":", 1)[0]
    return module_label == address or module_address == address


@contextmanager
def module_alias_scope(aliases: Mapping[str, str]) -> Iterator[None]:
    """Publish one model's alias map to module-selector evaluation.

    Parameters
    ----------
    aliases:
        Alias address -> canonical address (empty when no module is shared).

    Yields
    ------
    None
        Control while the map is active; the previous map is restored on exit.
    """

    token = _ACTIVE_MODULE_ALIASES.set(dict(aliases))
    try:
        yield
    finally:
        _ACTIVE_MODULE_ALIASES.reset(token)


def canonical_module_address(address: str, aliases: Mapping[str, str]) -> str:
    """Rewrite an alias spelling of a module address to its canonical address.

    Parameters
    ----------
    address:
        Module address, bare (``"alias"``) or pass-qualified (``"alias:2"``).
    aliases:
        Alias address -> canonical address.

    Returns
    -------
    str
        The canonical spelling with any pass qualifier kept (``"block:2"``), or
        ``address`` unchanged when it is not an alias. A pass qualifier counts
        calls of the module object, whichever registered name the forward
        called it through.
    """

    canonical = aliases.get(address)
    if canonical is not None:
        return canonical
    base, sep, call = address.rpartition(":")
    if sep and call.isdigit() and base in aliases:
        return f"{aliases[base]}:{call}"
    return address


def _site_module_aliases(subject: Any) -> Mapping[str, str]:
    """Return the alias map recorded on a post-hoc subject's Trace.

    Parameters
    ----------
    subject:
        Finalized site carrying a ``_source_trace_ref`` weakref.

    Returns
    -------
    Mapping[str, str]
        Alias address -> canonical address (empty when unavailable).
    """

    ref = getattr(subject, "_source_trace_ref", None)
    trace = ref() if callable(ref) else None
    accessor = getattr(trace, "_module_logs", None)
    return _AliasAddresses(getattr(accessor, "_alias_dict", None) or {})


class _AliasAddresses(Mapping[str, str]):
    """Read-only alias -> canonical-address view over ``ModuleAccessor._alias_dict``.

    ``trace.modules["alias"]`` resolves through that dict (built from
    ``Module.all_addresses``); this view lets post-hoc selectors resolve the
    same spelling to the same module without copying the map per evaluation.
    """

    __slots__ = ("_modules",)

    def __init__(self, modules_by_alias: Mapping[str, Any]) -> None:
        """Wrap an alias -> ``Module`` mapping.

        Parameters
        ----------
        modules_by_alias:
            Alias address -> the ``Module`` it names.
        """

        self._modules = modules_by_alias

    def __getitem__(self, alias: str) -> str:
        """Return the canonical address of the module ``alias`` names."""

        return str(self._modules[alias].address)

    def __iter__(self) -> Iterator[str]:
        """Iterate alias addresses."""

        return iter(self._modules)

    def __len__(self) -> int:
        """Return the number of alias addresses."""

        return len(self._modules)


def module_selector_target(value: Any, subject: Any, lifecycle: str) -> str:
    """Return the canonical module address a module-scoped selector names.

    Parameters
    ----------
    value:
        ``tl.module`` / ``tl.in_module`` selector value.
    subject:
        Lifecycle subject.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    str
        The target with a registered alias rewritten to the canonical address
        the subject's module candidates carry.
    """

    target = str(value)
    aliases = _ACTIVE_MODULE_ALIASES.get()
    if aliases is None and lifecycle == "site":
        aliases = _site_module_aliases(subject)
    if not aliases:
        return target
    return canonical_module_address(target, aliases)


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
    """Return whether one captured path component matches a user component.

    Parameters
    ----------
    saved_component:
        Captured typed path component.
    requested_component:
        User path component.

    Returns
    -------
    bool
        Whether both components address the same child.
    """

    if isinstance(saved_component, TupleIndex):
        return saved_component.index == requested_component
    if isinstance(saved_component, (DictKey, HFKey)):
        return saved_component.key == requested_component
    if isinstance(saved_component, (NamedField, DataclassField)):
        return saved_component.name == requested_component
    return saved_component == requested_component


def looks_like_finalized_label(label_value: str) -> bool:
    """Return whether a label literal looks postprocessed.

    Parameters
    ----------
    label_value:
        Label literal from ``tl.label`` / ``tl.contains`` / ``tl.regex``.

    Returns
    -------
    bool
        True for pass suffixes like ``:2`` or ``relu_4_27``-style names.
    """

    return bool(_FINAL_LABEL_PATTERN.search(label_value)) and not label_value.endswith("_raw")


def live_label_error_message(label_value: str, selector_kind: str = "label") -> str:
    """Build a copy-pasteable finalized-label diagnostic.

    Parameters
    ----------
    label_value:
        Finalized-looking label.
    selector_kind:
        Label-oriented selector kind for the message.

    Returns
    -------
    str
        User-facing error message.
    """

    return (
        f"tl.{selector_kind}({label_value!r}) looks like a finalized postprocess label, but "
        "live selectors run during capture before those labels exist. For post-capture "
        f'selection use tl.where(lambda p: p.layer_label == "{label_value}"). For live hooks, '
        'prefer a capture-time selector such as tl.func("relu") or '
        'tl.module("encoder.layer.4").'
    )


def _raise_for_finalized_live_label(selector_kind: str, label_value: str) -> None:
    """Reject finalized-looking label selectors during live capture.

    Parameters
    ----------
    selector_kind:
        Label-oriented selector kind.
    label_value:
        Selector literal or pattern.

    Raises
    ------
    LiveModeLabelError
        If the value names a label that exists only after postprocessing.
    """

    if looks_like_finalized_label(label_value):
        raise LiveModeLabelError(live_label_error_message(label_value, selector_kind))


def _capability_error(kind: str, lifecycle: str) -> SelectorCapabilityError:
    """Build the typed refusal for an unsupported ``(kind, lifecycle)`` pair.

    Parameters
    ----------
    kind:
        Selector kind being refused.
    lifecycle:
        Lifecycle that cannot evaluate the kind.

    Returns
    -------
    SelectorCapabilityError
        Typed capability refusal with a task-appropriate remedy.
    """

    where = _LIFECYCLE_DESC.get(lifecycle, lifecycle)
    if kind == "followed_by":
        return SelectorCapabilityError(
            "tl.followed_by(...) is capture-time-only retroactive save sugar: "
            "trace(save=candidate & tl.followed_by(successor), lookback=N) saves matching "
            f"candidates while the forward runs. {where} has no retroactive window."
        )
    if kind == "preceded_by":
        return SelectorCapabilityError(
            "tl.preceded_by(...) matches against the capture-time lookback window and is "
            f"only supported in trace(save=...) / record(save=...) predicates, not {where}."
        )
    if kind == "input_at":
        return SelectorCapabilityError(
            "tl.input_at(...) resolves saved input placeholders, but model inputs are not "
            "live hook application sites. Use trace(..., intervene=...) on downstream ops "
            "or mutate the model input before capture."
        )
    if kind == "facet":
        return SelectorCapabilityError(
            "tl.facet(...) / tl.head(...) selectors resolve through intervention mutators "
            f"(Trace.set, attach_hooks), not through {where}."
        )
    if kind == "site":
        return SelectorCapabilityError(
            "tl.site(...) needs a structural site key, which does not exist on "
            f"{where}: final structural numbering is absent mid-attach. Resolve "
            "the site on the finished trace (log.resolve_sites / do), or pass "
            "the site selector through trace(save=/intervene=), which arms the "
            "live minter."
        )
    return SelectorCapabilityError(f"Unsupported selector kind {kind!r} for {where}.")


#: Kinds refused UPFRONT per lifecycle. The site set must stay in lockstep
#: with ``_evaluate_subject``'s site refusals (contract-tested); the live set
#: names the kinds invalid in EVERY hook direction — backward-only kinds stay
#: out because backward live hooks route them through the backward matcher.
_UPFRONT_UNSUPPORTED_KINDS: dict[str, frozenset[str]] = {
    "site": frozenset({"followed_by", "preceded_by", "facet"}),
    # "site" (the C03 structural selector): live hook-attach proxies carry no
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


# ---------------------------------------------------------------------------
# Per-lifecycle subject universes
# ---------------------------------------------------------------------------


def _string_labels(
    subject: Any,
    lifecycle: str,
    attrs: dict[str, tuple[str, ...]] = _STRING_LABEL_ATTRS,
) -> tuple[str, ...]:
    """Return the label-string universe for one subject.

    Parameters
    ----------
    subject:
        Lifecycle subject (context, site, or live proxy).
    lifecycle:
        Active lifecycle key.
    attrs:
        Attribute-universe table; substring/regex matching passes the
        narrower :data:`_SUBSTRING_LABEL_ATTRS`.

    Returns
    -------
    tuple[str, ...]
        Non-``None`` label spellings, deduplicated in match order.
    """

    labels: list[str] = []
    seen: set[str] = set()
    for attr in attrs[lifecycle]:
        value = getattr(subject, attr, None)
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            labels.append(text)
    return tuple(labels)


def _maybe_guard_label(kind: str, value: str, subject: Any, lifecycle: str) -> None:
    """Apply the finalized-label diagnostic where the lifecycle requires it.

    Parameters
    ----------
    kind:
        Label-oriented selector kind (``label`` / ``contains`` / ``regex``).
    value:
        Selector literal or pattern.
    subject:
        Candidate subject.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    None
        Finalized layer objects remain valid post-capture selector inputs.
    """

    if lifecycle == "live":
        _raise_for_finalized_live_label(kind, value)
        return
    if lifecycle == "capture":
        from .predicate import RecordContext

        if isinstance(subject, RecordContext):
            _raise_for_finalized_live_label(kind, value)


def _is_module_scope_alias(kind: str, subject: Any, lifecycle: str) -> bool:
    """Return whether a module-scoped selector must skip ``subject`` as an alias.

    ``tl.module`` / ``tl.in_module`` address the ops a module call produces or
    contains. Two kinds of subject carry that module stamp without being such
    an op, and matching them would apply a value edit twice:

    - post hoc, the synthetic ``output_N`` node (``is_output``). Postprocess
      mints one per returned value; it executes nothing and re-carries its
      producer's module calls in ``output_of_module_calls`` for display only,
      and replay recomputes it from the producer;
    - live, for ``tl.in_module`` only, a module-boundary leaf whose tensor an
      op INSIDE that call produced (``_tl_boundary_inner_alias``): that op
      already matched at the op door. ``tl.module`` keeps firing at the
      boundary, where op doors never match it.

    Parameters
    ----------
    kind:
        Selector kind (``module`` or ``in_module``).
    subject:
        Site candidate.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether the module-scoped selector must not match ``subject``.
    """

    if lifecycle == "site":
        return bool(getattr(subject, "is_output", False))
    if lifecycle == "live" and kind == "in_module":
        return bool(getattr(subject, "_tl_module_boundary", False)) and bool(
            getattr(subject, "_tl_boundary_inner_alias", False)
        )
    return False


def _module_output_candidates(subject: Any, lifecycle: str) -> tuple[Any, ...]:
    """Return module-output-boundary candidates for ``tl.module``.

    Parameters
    ----------
    subject:
        Lifecycle subject.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    tuple[Any, ...]
        Module-pass candidates whose address may match.
    """

    module_outputs = tuple(getattr(subject, "output_of_module_calls", ()) or ())
    if lifecycle == "capture" and not module_outputs:
        source_trace = getattr(subject, "source_trace", None)
        if getattr(source_trace, "backend", TORCH_BACKEND_NAME) != TORCH_BACKEND_NAME:
            module_candidate = getattr(subject, "module", None)
            if module_candidate is not None:
                module_outputs = (module_candidate,)
    return module_outputs


def _module_containment_candidates(subject: Any, lifecycle: str) -> tuple[Any, ...]:
    """Return module-containment candidates for ``tl.in_module``.

    Parameters
    ----------
    subject:
        Lifecycle subject.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    tuple[Any, ...]
        Containment candidates: addresses, pass-qualified labels, or tuples.
    """

    if lifecycle == "capture":
        return _capture_module_candidates(subject)
    return tuple(getattr(subject, "output_of_module_calls", ()) or ()) + tuple(
        getattr(subject, "modules", ()) or ()
    )


def _capture_module_candidates(ctx: Any) -> tuple[Any, ...]:
    """Return module-address candidates visible on a capture context.

    Parameters
    ----------
    ctx:
        Capture-time context or layer-like object.

    Returns
    -------
    tuple[Any, ...]
        Module addresses and pass-qualified module labels.
    """

    candidates: list[str] = []
    address = getattr(ctx, "address", None)
    if address is not None:
        candidates.append(str(address))
    source_trace = getattr(ctx, "source_trace", None)
    if getattr(source_trace, "module_identity_mode", None) == "function_root":
        candidates.append("self")
        candidates.append("self:1")
    for frame in getattr(ctx, "module_stack", ()):
        frame_address = getattr(frame, "address", None)
        if frame_address is None and isinstance(frame, dict):
            frame_address = frame.get("address")
        if frame_address is None:
            continue
        frame_pass = getattr(frame, "pass_index", None)
        if frame_pass is None and isinstance(frame, dict):
            frame_pass = frame.get("pass_index")
        candidates.append(str(frame_address))
        if frame_pass is not None:
            candidates.append(f"{frame_address}:{frame_pass}")
    modules = getattr(ctx, "modules", ())
    module_ops = getattr(ctx, "output_of_module_calls", ())
    candidates.extend(
        _module_candidate_string(candidate) for candidate in tuple(modules) + tuple(module_ops)
    )
    return tuple(candidates)


def _module_candidate_string(candidate: Any) -> str:
    """Return a selector-compatible module candidate string.

    Parameters
    ----------
    candidate:
        Module candidate from an op, module call, or capture context.

    Returns
    -------
    str
        Address or pass-qualified address for :func:`module_address_matches`.
    """

    if isinstance(candidate, tuple) and candidate and isinstance(candidate[0], str):
        if len(candidate) > 1:
            return f"{candidate[0]}:{candidate[1]}"
        return candidate[0]
    return str(candidate)


def _output_matches(subject: Any, value: Any, lifecycle: str) -> bool:
    """Return whether a subject matches an output index or semantic role.

    Parameters
    ----------
    subject:
        Lifecycle subject.
    value:
        Output index or semantic role.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether the subject is the requested output.
    """

    if lifecycle == "capture":
        return getattr(subject, "output_index", None) == value
    if isinstance(value, int):
        return getattr(subject, "multi_output_index", None) == value
    return getattr(subject, "multi_output_name", None) == str(value)


def _input_path_subject_matches(
    subject: Any, requested_path: tuple[Any, ...], lifecycle: str
) -> bool:
    """Return whether a subject consumes an input at ``requested_path``.

    Parameters
    ----------
    subject:
        Lifecycle subject.
    requested_path:
        User path using plain indices, keys, or field names.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether any input-container leaf occurrence matches the path.
    """

    if lifecycle == "site":
        from .container_registry import Role

        trace = getattr(subject, "source_trace", None)
        site_labels = {
            label
            for label in (
                getattr(subject, "layer_label", None),
                getattr(subject, "layer_label_raw", None),
                getattr(subject, "_layer_label_raw", None),
            )
            if label is not None
        }
        for record in getattr(trace, "_containers", {}).values():
            for snapshot in getattr(record, "snapshots", ()) or ():
                if getattr(snapshot, "role", None) != Role.MODEL_INPUT:
                    continue
                for occurrence in getattr(snapshot, "leaf_occurrences", ()) or ():
                    if occurrence.producer_op_label not in site_labels:
                        continue
                    if output_path_matches(tuple(occurrence.path), requested_path):
                        return True
    for container in getattr(subject, "input_containers", ()) or ():
        for occurrence in getattr(container, "leaf_occurrences", ()) or ():
            if output_path_matches(tuple(occurrence.path), requested_path):
                return True
    return False


def _predicate_payload(value: Any) -> Any:
    """Validate and unpack a predicate selector payload.

    Parameters
    ----------
    value:
        Stored predicate payload: a callable or ``(callable, name_hint)``.

    Returns
    -------
    Any
        The predicate callable.

    Raises
    ------
    SiteResolutionError
        If the predicate payload is malformed.
    """

    if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
        return value[0]
    if callable(value):
        return value
    raise SiteResolutionError("tl.where(...) requires a callable predicate.")


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


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


def _is_grad_fn(subject: Any) -> bool:
    """Return whether a site subject is a backward ``GradFn`` log.

    Parameters
    ----------
    subject:
        Candidate site.

    Returns
    -------
    bool
        Whether the subject is a GradFn.
    """

    from torchlens.data_classes.grad_fn import GradFn

    return isinstance(subject, GradFn)


def _evaluate_subject(selector: BaseSelector, subject: Any, lifecycle: str) -> bool:
    """Evaluate one selector against one forward-flavored subject.

    Parameters
    ----------
    selector:
        Normalized selector.
    subject:
        Capture context, forward site, or live proxy.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    kind = str(selector.selector_kind)
    value = selector.selector_value

    if kind == "and" and isinstance(selector, CompositeSelector):
        return all(
            evaluate(child, subject, lifecycle=cast(Lifecycle, lifecycle))
            for child in selector.selectors
        )
    if kind == "or" and isinstance(selector, CompositeSelector):
        return any(
            evaluate(child, subject, lifecycle=cast(Lifecycle, lifecycle))
            for child in selector.selectors
        )
    if kind == "not" and isinstance(selector, NotSelector):
        return not evaluate(selector.selector, subject, lifecycle=cast(Lifecycle, lifecycle))

    if kind == "label":
        _maybe_guard_label(kind, str(value), subject, lifecycle)
        target = str(value)
        if target in _string_labels(subject, lifecycle):
            return True
        if lifecycle == "site":
            return target in (getattr(subject, "lookup_keys", ()) or ())
        return False
    if kind == "contains":
        _maybe_guard_label(kind, str(value), subject, lifecycle)
        if lifecycle == "live" and bool(getattr(subject, "_tl_module_boundary", False)):
            return False
        needle = str(value).lower()
        labels = _string_labels(subject, lifecycle, _SUBSTRING_LABEL_ATTRS)
        if lifecycle == "site" and ":" in needle:
            # Bare final labels never contain ':', so a pass-qualified needle
            # (the remedy the multipass_bare_label refusal teaches) can only
            # address the pass-qualified op spelling; without it, string
            # addressing of one pass of a multi-pass layer matches 0 sites.
            pass_label = getattr(subject, "label", None)
            if isinstance(pass_label, str) and pass_label not in labels:
                labels = (*labels, pass_label)
        return any(needle in label.lower() for label in labels)
    if kind == "regex":
        _maybe_guard_label(kind, str(value), subject, lifecycle)
        pattern = str(value)
        return any(
            re.search(pattern, label) is not None
            for label in _string_labels(subject, lifecycle, _SUBSTRING_LABEL_ATTRS)
        )
    if kind == "func":
        if isinstance(value, dict):
            name = value.get("name")
            output_target = value.get("output")
            if output_target is not None and not _output_matches(subject, output_target, lifecycle):
                return False
        else:
            name = value
        func_name = getattr(subject, "func_name", None)
        if func_name is not None and str(name) == str(func_name):
            return True
        if lifecycle == "live" and bool(getattr(subject, "_tl_module_boundary", False)):
            return False
        layer_type = getattr(subject, "layer_type", None)
        return layer_type is not None and str(name) == str(layer_type)
    if kind == "func_transform":
        if not bool(getattr(subject, "is_transform", False)):
            return False
        if value is None:
            return True
        transform_kind = getattr(subject, "transform_kind", None)
        if transform_kind is None:
            return False
        return sanitize_transform_kind(transform_kind) == sanitize_transform_kind(value)
    if kind in {"module", "in_module"} and _is_module_scope_alias(kind, subject, lifecycle):
        return False
    if kind == "module":
        target = module_selector_target(value, subject, lifecycle)
        return any(
            module_address_matches(candidate, target)
            for candidate in _module_output_candidates(subject, lifecycle)
        )
    if kind == "in_module":
        target = module_selector_target(value, subject, lifecycle)
        return any(
            module_address_matches(candidate, target)
            for candidate in _module_containment_candidates(subject, lifecycle)
        )
    if kind == "site":
        # C03 structural-position selector (address law: valid in every
        # lane). Post hoc it reads the persisted op.site_key; on the capture
        # lane it reads the streaming live minter armed by the tl.trace
        # entry; the live hook-proxy lane refuses upfront (mid-attach
        # proxies carry no structural key -- see _UPFRONT_UNSUPPORTED_KINDS).
        if lifecycle == "site":
            return _site_key_matches(selector, getattr(subject, "site_key", None))
        if lifecycle == "capture":
            from torchlens.intervention.site_keys import active_live_minter

            minter = active_live_minter()
            if minter is None:
                raise SelectorCapabilityError(
                    "tl.site(...) on the capture lane needs the live site-key "
                    "minter, which tl.trace arms automatically when save=/"
                    "intervene= contains a site selector. This capture surface "
                    "did not arm it (tl.record and preview backends do not "
                    "support live site keys yet). Address the finished trace "
                    "post hoc (log.resolve_sites / do), or use tl.trace with "
                    "the site selector in save=/intervene=."
                )
            return _site_key_matches(selector, minter.key_for_context(subject))
        raise _capability_error(kind, lifecycle)
    if kind == "output":
        return _output_matches(subject, value, lifecycle)
    if kind == "output_at":
        return output_path_matches(
            tuple(getattr(subject, "container_path", ()) or ()),
            tuple(value),
        )
    if kind == "input_at":
        if lifecycle == "live":
            raise _capability_error(kind, lifecycle)
        return _input_path_subject_matches(subject, tuple(value), lifecycle)
    if kind == "predicate":
        predicate = _predicate_payload(value)
        if lifecycle == "live":
            try:
                return bool(predicate(subject))
            except Exception as exc:
                raise SiteResolutionError(
                    "live predicate selector failed at "
                    f"{getattr(subject, '_layer_label_raw', '<unknown>')}"
                ) from exc
        return bool(predicate(subject))
    if kind == "episode_step":
        return _episode_step_matches(value, subject, lifecycle)
    if kind == "preceded_by":
        if lifecycle != "capture" or not isinstance(selector, PrecededBySelector):
            raise _capability_error(kind, lifecycle)
        return _preceded_by_matches(selector, subject)
    if kind == "followed_by":
        if lifecycle != "capture":
            raise _capability_error(kind, lifecycle)
        raise SelectorCompositionError(
            "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
            "standalone, negated, or OR-composed followed_by selectors are unsupported."
        )
    if lifecycle == "capture":
        if kind == "grad_kind":
            return getattr(subject, "grad_kind", None) == value
        if kind == "backward_pass":
            ctx_pass = getattr(subject, "backward_pass_index", None)
            if ctx_pass is None:
                ctx_pass = getattr(subject, "pass_index", None)
            return ctx_pass == value
        if kind in _CAPTURE_SILENT_FALSE_KINDS:
            return False
        raise _capability_error(kind, lifecycle)
    if lifecycle == "site":
        if kind in {
            "grad_fn",
            "grad_fn_handle",
            "grad_fn_label",
            "intervening",
            "without_op",
            "grad_kind",
            "backward_pass",
        }:
            return False
        raise _capability_error(kind, lifecycle)
    raise _capability_error(kind, lifecycle)


def _episode_step_matches(value: Any, subject: Any, lifecycle: str) -> bool:
    """Evaluate one episode-step selector (lane F42's step qualifier).

    Capture/live lifecycles read the LIVE step position from the armed join
    session (the stepped-module boundary hooks); ops between steps carry no
    step and never match. Evaluating outside an episode capture refuses
    typed at the first evaluation -- a step qualifier on a plain capture is
    a category error, and matching-nothing silently would be the inverse of
    the fires-at-every-step wrongness the selector exists to close. The
    post-hoc site lifecycle reads the persisted ``Op.episode_step`` stamps
    (unstamped/plain products are door-guarded at resolution entry).

    Parameters
    ----------
    value:
        The selector's ordered step tuple.
    subject:
        Capture context, live proxy, or finalized site record.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether the subject executes inside one of the named steps.
    """

    steps = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if lifecycle in ("capture", "live"):
        from torchlens.capture._episode_join import active_join_session

        session = active_join_session()
        if session is None:
            from torchlens.intervention.errors import SelectorCapabilityError

            raise SelectorCapabilityError(
                "at_step(...) names an episode step, but this capture carries "
                "no episode declaration: there are no steps to qualify. "
                "Declare the episode -- tl.trace(model, x, "
                "episode=tl.options.EpisodeSpec(stepped_module=..., ...), "
                "...) -- or drop the step qualifier.",
                code="episode_step_selector_without_episode",
                remedy="declare episode= on the capture, or drop at_step()",
            )
        return session.in_step is not None and session.in_step in steps
    stamp = getattr(subject, "episode_step", None)
    return stamp is not None and stamp in steps


def _preceded_by_matches(selector: PrecededBySelector, ctx: Any) -> bool:
    """Return whether a retained lookback predecessor matches ``selector.inner``.

    Parameters
    ----------
    selector:
        Lookback predecessor selector.
    ctx:
        Capture-time context with ``recent_ops`` populated.

    Returns
    -------
    bool
        Whether a retained (parent) predecessor matches the inner predicate.
    """

    parent_labels = set(getattr(ctx, "parent_labels_raw", ()) or getattr(ctx, "parent_labels", ()))
    recent_ops = tuple(getattr(ctx, "recent_ops", ()))
    inner = selector.inner
    if parent_labels:
        return any(
            (recent.raw_label or recent.label) in parent_labels and bool(inner(recent))  # type: ignore[operator]
            for recent in recent_ops
        )
    return any(bool(inner(recent)) for recent in recent_ops)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# GradFn (backward) site evaluation
# ---------------------------------------------------------------------------


def _evaluate_grad_fn_site(selector: BaseSelector, site: Any) -> bool:
    """Evaluate one selector against one backward ``GradFn`` site.

    Direction-agnostic kinds bridge to the paired forward op (and synthetic
    boundary aliases sharing the autograd identity); backward kinds match the
    GradFn itself.

    Parameters
    ----------
    selector:
        Normalized selector.
    site:
        GradFn site.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    kind = str(selector.selector_kind)
    if kind == "and" and isinstance(selector, CompositeSelector):
        return all(evaluate(child, site, lifecycle="site") for child in selector.selectors)
    if kind == "or" and isinstance(selector, CompositeSelector):
        return any(evaluate(child, site, lifecycle="site") for child in selector.selectors)
    if kind == "not" and isinstance(selector, NotSelector):
        return not evaluate(selector.selector, site, lifecycle="site")
    if kind in DIRECTION_AGNOSTIC_KINDS:
        if site.op is not None and _evaluate_subject(selector, site.op, "site"):
            return True
        return any(
            _evaluate_subject(selector, alias, "site") for alias in _grad_fn_boundary_aliases(site)
        )
    if kind in {"intervening", "without_op"}:
        return not site.has_op
    if kind == "grad_fn_label":
        return site.label == str(selector.selector_value)
    if kind == "grad_fn":
        value = selector.selector_value
        payload = value if isinstance(value, dict) else {}
        grad_fn_type = payload.get("type")
        label_pattern = payload.get("grad_fn_label_pattern")
        is_custom = payload.get("is_custom")
        if grad_fn_type is not None and not _grad_fn_type_matches(site, str(grad_fn_type)):
            return False
        if label_pattern is not None and str(label_pattern) not in site.label:
            return False
        return not (is_custom is not None and bool(site.is_custom) is not bool(is_custom))
    if kind == "grad_kind":
        grad_kind = str(selector.selector_value)
        field_name = "grad_inputs" if grad_kind == "grad_input" else "grad_outputs"
        return any(
            getattr(call, field_name, None) is not None for call in _grad_fn_call_values(site)
        )
    if kind == "backward_pass":
        pass_index = int(selector.selector_value)
        return any(
            getattr(call, "backward_pass_index", None) == pass_index
            for call in _grad_fn_call_values(site)
        )
    return False


def _grad_fn_boundary_aliases(site: Any) -> tuple[Any, ...]:
    """Return input/output alias ops sharing a GradFn identity with ``site``.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Any, ...]
        Synthetic input/output ops whose ``grad_fn_object_id`` equals the
        GradFn's object id, excluding the already-paired op.
    """

    trace = site.source_trace
    if trace is None:
        return ()
    paired_label = site.op_label
    aliases: list[Any] = []
    for layer in getattr(trace, "layer_list", ()):
        if getattr(layer, "layer_label", None) == paired_label:
            continue
        if getattr(layer, "grad_fn_object_id", None) != site.grad_fn_object_id:
            continue
        if not (getattr(layer, "is_input", False) or getattr(layer, "is_output", False)):
            continue
        aliases.append(layer)
    return tuple(aliases)


def _grad_fn_type_matches(site: Any, requested: str) -> bool:
    """Return whether a grad_fn_handle type matches user spelling flexibly.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    requested:
        Requested type string.

    Returns
    -------
    bool
        Whether class name or normalized type matches.
    """

    lowered = requested.lower()
    normalized = lowered.removesuffix("backward0").removesuffix("backward")
    candidates = {
        site.class_name.lower(),
        site.type.lower(),
        site.label.lower(),
    }
    return lowered in candidates or normalized in candidates


def _grad_fn_call_values(site: Any) -> tuple[Any, ...]:
    """Return GradFnCall values from dict or accessor-backed call storage.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Any, ...]
        Recorded GradFnCall values.
    """

    calls = site.calls
    if isinstance(calls, dict):
        return tuple(calls.values())
    return tuple(calls._list)


# ---------------------------------------------------------------------------
# The one spec deserializer
# ---------------------------------------------------------------------------


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

    from ..intervention.selectors import (
        FacetSelector,
        contains,
        func,
        func_transform,
        grad_fn,
        grad_fn_label,
        grad_input,
        grad_output,
        in_backward_pass,
        in_module,
        input_at,
        label,
        module,
        output,
        output_at,
        regex,
        where,
        without_op,
    )

    metadata = metadata or {}
    if kind == "label":
        return label(str(value))
    if kind == "grad_fn_label":
        return grad_fn_label(str(value))
    if kind == "func":
        if isinstance(value, dict):
            return func(str(value.get("name")), output=value.get("output"))
        return func(str(value))
    if kind == "func_transform":
        return func_transform(None if value is None else str(value))
    if kind == "module":
        return module(str(value))
    if kind == "output":
        return output(value)
    if kind == "output_at":
        return output_at(value)
    if kind == "input_at":
        if lifecycle == "live":
            raise _capability_error(kind, lifecycle)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return input_at(*value)
        return input_at(value)
    if kind == "contains":
        return contains(str(value))
    if kind == "regex":
        return regex(str(value))
    if kind == "in_module":
        selector = in_module(str(value))
        if isinstance(selector, BaseSelector):
            return selector
    if kind == "facet":
        if lifecycle == "live":
            raise _capability_error(kind, lifecycle)
        if isinstance(value, dict):
            name = value.get("name")
            head_index = value.get("head_index")
            module_address = value.get("module_address")
            if name is not None or head_index is not None:
                return FacetSelector(
                    None if name is None else str(name),
                    head_index=None if head_index is None else int(head_index),
                    module_address=None if module_address is None else str(module_address),
                )
        raise SiteResolutionError(f"Unsupported facet selector payload {value!r}.")
    if kind == "site":
        from ..intervention.selectors import SiteSelector

        payload = dict(value or {})
        return SiteSelector(
            payload.get("key"),
            module_path=payload.get("module_path"),
            op_type=payload.get("op_type"),
            slot=payload.get("slot"),
            ordinal=payload.get("ordinal"),
        )
    if kind == "predicate" and callable(value):
        return where(value, name_hint=metadata.get("name_hint"))
    if kind == "grad_fn":
        payload = dict(value or {})
        return grad_fn(
            payload.get("type"),
            label=payload.get("grad_fn_label_pattern"),
            is_custom=payload.get("is_custom"),
        )
    if kind in {"intervening", "without_op"}:
        return without_op()
    if kind == "grad_kind":
        return grad_input() if value == "grad_input" else grad_output()
    if kind == "backward_pass":
        if not isinstance(value, int):
            raise SiteResolutionError("backward_pass target specs require an integer pass index.")
        return in_backward_pass(value)
    if kind == "episode_step":
        from ..intervention.selectors import at_step

        if isinstance(value, Sequence) and not isinstance(value, str):
            return at_step(*(int(step) for step in value))
        raise SiteResolutionError(
            "episode_step target specs require a sequence of 0-based step indices.",
            code="episode_step_selector_invalid",
            remedy="rebuild the selector with at_step(<step indices>)",
        )
    if kind == "not":
        return ~normalize_selector_like(value, lifecycle=lifecycle)
    if kind in {"and", "or"}:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise SiteResolutionError(
                f"{kind!r} target specs require a sequence of nested selectors."
            )
        children = tuple(normalize_selector_like(child, lifecycle=lifecycle) for child in value)
        return CompositeSelector(cast("Literal['and', 'or']", kind), cast(Any, children))
    if kind in {"followed_by", "preceded_by"}:
        if lifecycle != "capture":
            raise _capability_error(kind, lifecycle)
        inner = value
        if isinstance(inner, str):
            raise SiteResolutionError(
                f"tl.{kind}(...) target spec carries an opaque audit payload {inner!r}; "
                "an inner predicate saved by repr cannot be reconstructed. Rebuild the "
                "selector in code, or re-save the spec with a structural selector inner "
                "such as tl.func(...)."
            )
        if not isinstance(inner, BaseSelector) and not callable(inner):
            inner = normalize_selector_like(inner, lifecycle="capture")
        if kind == "followed_by":
            return FollowedBySelector(inner)
        return PrecededBySelector(inner)
    raise SiteResolutionError(f"Unsupported target spec selector kind {kind!r}.")


def _site_key_matches(selector: Any, rendered_key: Any) -> bool:
    """Match one ``site`` selector against one rendered site key (or None)."""

    from ..intervention.selectors import SiteSelector

    if not isinstance(selector, SiteSelector):
        selector = selector_from_spec("site", selector.selector_value, None)
    return bool(selector.matches_key(rendered_key if isinstance(rendered_key, str) else None))


def _selector_from_site_table(site_table: Any) -> BaseSelector:
    """Lower a ``find_sites()`` SiteTable into a label-union selector."""

    from ..intervention.selectors import label

    site_labels: list[str] = []
    for site in site_table:
        site_label = getattr(site, "layer_label", None)
        if isinstance(site_label, str) and site_label not in site_labels:
            site_labels.append(site_label)
    if not site_labels:
        raise SiteResolutionError(
            "This SiteTable resolved 0 sites, so there is nothing to "
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
        queries treat them as substring searches (``tl.contains``), live hook
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
    if isinstance(selector_like, TargetSpec):
        return selector_from_spec(
            selector_like.selector_kind,
            selector_like.selector_value,
            selector_like.metadata,
            lifecycle=lifecycle,
        )
    if isinstance(selector_like, FrozenTargetSpec):
        return selector_from_spec(
            selector_like.selector_kind,
            selector_like.selector_value,
            dict(selector_like.metadata),
            lifecycle=lifecycle,
        )
    if isinstance(selector_like, str):
        from ..intervention.selectors import contains, label

        if lifecycle == "live":
            return label(selector_like)
        return contains(selector_like)
    if type(selector_like).__name__ == "SiteTable":
        # find_sites() output is accepted directly (leverage B3/NEW-4): the
        # zero-match refusal RECOMMENDS log.find_sites(...), so its result
        # must be a valid address, not a dead end.
        return _selector_from_site_table(selector_like)
    if lifecycle == "capture" and callable(selector_like):
        from ..intervention.selectors import where

        return where(selector_like)
    # An ``Op``/``Layer`` RECORD is the most precise site spelling a user can
    # hold (AUD-CODE 4.10, W051-REPLAY out-of-fence item 1). Lower it to its
    # PASS-QUALIFIED ``Op.label`` selector(s) here, so the ``set()`` door,
    # ``resolve_sites(record)`` and the mutator site validation all address
    # exactly the pass(es) the record names -- never the bare ``layer_label``
    # (the LAST pass of a multi-pass layer) the live arm below would pick.
    # Lazy import: ``intervention.hooks`` imports this module.
    from ..intervention.hooks import lower_record_site_target

    lowered = lower_record_site_target(selector_like)
    if lowered is not selector_like:
        return lowered
    if lifecycle == "live" and hasattr(selector_like, "layer_label"):
        from ..intervention.selectors import label

        return label(str(selector_like.layer_label))
    if lifecycle == "live":
        raise SiteResolutionError(f"Unsupported live hook selector {selector_like!r}.")
    raise SiteResolutionError(f"Unsupported site query {selector_like!r}.")


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
