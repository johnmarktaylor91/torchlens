"""The general display filter (knob N9, themes memo section 6).

Promotes the shipped ``skip_fn`` machinery (predicate hiding with
transitive edge bridging) into one declarative display channel: a closed
token vocabulary, Selection support, and mandatory rendered disclosure.
The build is polarity, safety, and disclosure -- the filtering engine is
the existing one.

v1 tokens, priority as measured:

- ``"reshapes"``: the glue family (measured 49% of distilgpt2's ops) --
  the token that actually delivers the torchview experience.
- ``"constants"``: internally-initialized ops with no input ancestry.
- ``"non_module_ops"``: torchview MIGRATION PARITY, documented as such --
  it removes 2-5% on real models because TorchLens records module
  MEMBERSHIP, so nearly every op is inside some module.

Include and exclude are mutually exclusive in v1; boundaries are
auto-exempt on token/Selection spellings; a raw predicate keeps its strict
typed boundary raise. Disclosure is blocking and RENDERED: the caption
("displayed X of Y eligible ops; Z filtered (...)"), dashed bridged edges
with midpoint "via N hidden" labels, and a bridged-reachability legend
line.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..._errors import InvalidArgumentError

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "BRIDGED_EDGE_LEGEND_LINE",
    "FILTER_TOKENS",
    "CompiledDisplayFilter",
    "DisplayFilter",
    "compile_display_filter",
]

#: The glue family: shape-plumbing ops whose hiding yields the torchview-
#: style compact picture. A CLOSED set -- additions are reviewed vocabulary
#: changes, never silent growth.
_RESHAPE_FAMILY = frozenset(
    {
        "reshape",
        "view",
        "view_as",
        "permute",
        "transpose",
        "t",
        "flatten",
        "unflatten",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "contiguous",
        "chunk",
        "split",
        "narrow",
        "select",
        "getitem",
        "unbind",
        "movedim",
        "swapaxes",
        "swapdims",
        "ravel",
        "broadcast_to",
    }
)

#: Closed v1 token vocabulary (memo section 6). ``noise`` is deferred until
#: its taxonomy is pinned; buffers/params/boolean/orphans are rejected as
#: tokens (existing knobs keep their jobs); ``module_calls`` is collapse
#: semantics, not filtering.
FILTER_TOKENS = ("constants", "non_module_ops", "reshapes")

#: The mandatory legend line on every filtered artifact.
BRIDGED_EDGE_LEGEND_LINE = (
    "dashed edge = reachability through omitted operations, not direct adjacency"
)


@dataclass(frozen=True)
class DisplayFilter:
    """One declarative display-filter request.

    Exactly one of ``exclude`` / ``include`` may be set (v1 polarity rule).
    Each accepts a closed token, a list of tokens, a same-trace Selection,
    or a raw predicate ``layer -> bool`` (the documented power spelling,
    strict boundary semantics preserved).
    """

    exclude: Any = None
    include: Any = None


@dataclass(frozen=True)
class CompiledDisplayFilter:
    """A filter compiled against one trace.

    Attributes
    ----------
    skip_fn:
        The predicate handed to the shipped hiding machinery.
    caption:
        The mandatory rendered caption ("displayed X of Y eligible ops; Z
        filtered (<rules>)").
    legend_line:
        The bridged-reachability legend line.
    eligible:
        Op population examined.
    filtered:
        Ops the filter hides.
    rules:
        Human naming of the active rules.
    """

    skip_fn: Callable[[Any], bool]
    caption: str
    legend_line: str
    eligible: int
    filtered: int
    rules: str


def _is_boundary(layer: Any) -> bool:
    """Return whether a layer is an input/output boundary (auto-exempt)."""

    return bool(getattr(layer, "is_input", False) or getattr(layer, "is_final_output", False))


def _token_predicate(token: str) -> Callable[[Any], bool]:
    """Return the membership predicate for one closed token."""

    if token == "reshapes":
        return lambda layer: str(getattr(layer, "layer_type", "")) in _RESHAPE_FAMILY
    if token == "constants":
        return lambda layer: (
            bool(getattr(layer, "is_internally_initialized", False))
            and not bool(getattr(layer, "has_input_ancestor", True))
        )
    if token == "non_module_ops":
        return lambda layer: not tuple(getattr(layer, "modules", ()) or ())
    raise InvalidArgumentError(
        f"unknown display-filter token {token!r}; the closed v1 vocabulary is "
        f"{', '.join(FILTER_TOKENS)}",
        code="display_filter_token_unknown",
        remedy="pass one of the closed tokens, a Selection, or a predicate",
        argument="display_filter",
    )


def _resolve_selection_operand(trace: Trace, selection: Any) -> tuple[Any, ...]:
    """Resolve a Selection operand to its SiteEntry rows on this trace.

    A ResolvedSelection bound to a foreign trace refuses typed -- alignment
    (``align_to``) is the one explicit door for cross-trace reuse.
    """

    from torchlens.selection import ResolvedSelection, Selection

    if isinstance(selection, ResolvedSelection):
        resolved: Any = selection
    elif isinstance(selection, Selection):
        resolved = selection.resolve(trace)
    elif hasattr(selection, "__selection__"):
        resolved = selection.__selection__().resolve(trace)
    else:  # duck-typed entries-bearing object (admitted by _normalize_rules)
        resolved = selection
    bound_trace = getattr(resolved, "_trace", None)
    if bound_trace is not None and bound_trace is not trace:
        raise InvalidArgumentError(
            "display filter received a ResolvedSelection bound to a different "
            "trace; selections filter the trace they were resolved on",
            code="display_filter_selection_foreign",
            remedy="pass the unresolved Selection, or align_to(this_trace) first",
            argument="display_filter",
        )
    if isinstance(resolved, ResolvedSelection):
        return tuple(resolved)
    return tuple(getattr(resolved, "entries", ()))


def _structural_join(trace: Trace, structural: set[str], labels: set[str]) -> None:
    """Add labels for ops whose L1 site key matches a cross-run aligned entry."""

    for op in trace.ops:
        if str(getattr(op, "site_key", "")) in structural:
            labels.add(op.layer_label)
            label = getattr(op, "label", None)
            if label:
                labels.add(label)


def _selection_labels(trace: Trace, selection: Any) -> frozenset[str]:
    """Resolve a Selection (query or resolved) to this trace's op labels."""

    entries = _resolve_selection_operand(trace, selection)
    labels: set[str] = set()
    structural: set[str] = set()
    for entry in entries:  # SiteEntry rows (family level: zero-mask sites count)
        site_key = getattr(entry, "site_key", None)
        if isinstance(site_key, tuple) and site_key:
            layer_label = str(site_key[0])
            labels.add(layer_label)
            if len(site_key) > 1 and site_key[1] is not None:
                labels.add(f"{layer_label}:{site_key[1]}")
        structural_key = getattr(entry, "structural_site_key", None)
        if structural_key:
            structural.add(str(structural_key))
    if structural:  # cross-run aligned entries also join on the L1 site key
        _structural_join(trace, structural, labels)
    return frozenset(labels)


def _normalize_rules(trace: Trace, value: Any) -> tuple[Callable[[Any], bool], str, bool]:
    """Compile one polarity operand to (match_fn, rule wording, is_predicate)."""

    if callable(value) and not isinstance(value, str):
        name = getattr(value, "__name__", type(value).__name__)
        return value, f"predicate {name}", True
    if isinstance(value, str):
        predicate = _token_predicate(value)
        return predicate, value, False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        predicates = [_token_predicate(str(token)) for token in value]
        wording = "+".join(str(token) for token in value)
        return (lambda layer: any(p(layer) for p in predicates)), wording, False
    if hasattr(value, "__selection__") or hasattr(value, "resolve") or hasattr(value, "entries"):
        labels = _selection_labels(trace, value)
        return (
            (
                lambda layer: (
                    layer.layer_label in labels or str(getattr(layer, "label", "")) in labels
                )
            ),
            "selection",
            False,
        )
    raise InvalidArgumentError(
        f"display filter operand must be a closed token, a list of tokens, a "
        f"Selection, or a predicate; received {type(value).__name__}",
        code="display_filter_operand_invalid",
        remedy=f"pass one of the closed tokens ({', '.join(FILTER_TOKENS)}), a Selection, or a callable",
        argument="display_filter",
    )


def compile_display_filter(trace: Trace, display_filter: DisplayFilter) -> CompiledDisplayFilter:
    """Compile a filter request against one trace, with counted disclosure.

    Raises
    ------
    InvalidArgumentError
        ``display_filter_polarity_conflict`` when both polarities are set or
        neither is; token/Selection/operand refusals per their sites.
    """

    if (display_filter.exclude is None) == (display_filter.include is None):
        raise InvalidArgumentError(
            "display filter takes exactly one polarity in v1: set exclude= or "
            "include=, not both and not neither",
            code="display_filter_polarity_conflict",
            remedy="pass exclude=<rules> or include=<rules>",
            argument="display_filter",
        )
    is_include = display_filter.include is not None
    operand = display_filter.include if is_include else display_filter.exclude
    match_fn, wording, is_predicate = _normalize_rules(trace, operand)

    eligible = 0
    filtered_labels: set[str] = set()
    for op in trace.ops:
        if _is_boundary(op):
            continue
        eligible += 1
        matched = bool(match_fn(op))
        hidden = (not matched) if is_include else matched
        if hidden:
            filtered_labels.add(op.layer_label)
            label = getattr(op, "label", None)
            if label:
                filtered_labels.add(label)

    def _skip(layer: Any) -> bool:
        """Decide one layer's visibility under the compiled polarity."""

        if not is_predicate and _is_boundary(layer):
            # Boundaries auto-exempt on token/Selection spellings; a raw
            # predicate keeps the shipped strict boundary raise downstream.
            return False
        matched = bool(match_fn(layer))
        return (not matched) if is_include else matched

    filtered_count = len({label for label in filtered_labels if ":" not in label}) or len(
        filtered_labels
    )
    polarity = "include" if is_include else "exclude"
    caption = (
        f"displayed {eligible - filtered_count} of {eligible} eligible ops; "
        f"{filtered_count} filtered ({polarity}: {wording})"
    )
    return CompiledDisplayFilter(
        skip_fn=_skip,
        caption=caption,
        legend_line=BRIDGED_EDGE_LEGEND_LINE,
        eligible=eligible,
        filtered=filtered_count,
        rules=f"{polarity}: {wording}",
    )
