"""Consistent, bounded candidate rows for lookup error messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._display_card import render_card
from ._display_keys import candidates_text, display_key


def candidate_keys(candidates: Sequence[str], records: Mapping[str, Any]) -> list[str]:
    """Resolve suggestions through stored records; do not parse positional suffixes."""

    return list(
        dict.fromkeys(display_key(records[key]) if key in records else key for key in candidates)
    )


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance with a single row of working storage."""

    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def lookup_message(
    problem: str, candidates: Sequence[str], remedy: str, *, query: str | None = None
) -> str:
    """Rank choices before the five-candidate cut, keeping the remedy last."""

    ranked = (
        sorted(candidates, key=lambda key: _edit_distance(query, key))
        if query is not None
        else candidates
    )
    rows = [("candidates", candidates_text(ranked))] if ranked else []
    rows.append(("remedy", remedy))
    return render_card(problem, rows)


def short_name_message(key: str, candidates: Sequence[str]) -> str:
    """Offer full parameter or buffer addresses without dumping every match."""

    return lookup_message(
        f"Ambiguous short name {key!r}.", candidates, "use a full address", query=key
    )


def module_output_message(call: Any, trace: Any) -> str:
    """Share the bounded output choices across module and endpoint refusals."""

    candidates = (
        candidate_keys(call.output_ops, trace._layer_dict_all_keys)
        if trace is not None
        else [f"module_call.output_ops[{index}]" for index in range(len(call.output_ops))]
    )
    return lookup_message(
        f"ModuleCall {call.label!r} has {len(call.output_ops)} output ops.",
        candidates,
        "select one output with module_call.output_ops[index]",
    )


def module_lookup_message(key: str, module: Any, matches: Sequence[str]) -> str:
    """Show stored display keys for an ambiguous module-local layer lookup."""

    return lookup_message(
        f"Ambiguous lookup: {key!r} matches {len(matches)} layers in module {module.address!r}.",
        candidate_keys(matches, module._source_trace._layer_dict_all_keys),
        "select one layer by its full key",
        query=key,
    )


def ambiguous_substring_message(key: str, records: Sequence[Any]) -> str:
    """Keep a substring refusal and its resolved candidate ranking together."""

    return lookup_message(
        f"Ambiguous lookup: {key!r} matches {len(records)} layers.",
        [display_key(record) for record in records],
        "Use a more specific key.",
        query=key,
    )
