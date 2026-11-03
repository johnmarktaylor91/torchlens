"""Consistent, bounded candidate rows for lookup error messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._display_card import card_row
from ._display_keys import candidates_text, display_key


def candidate_keys(candidates: Sequence[str], records: Mapping[str, Any]) -> list[str]:
    """Resolve suggestions through stored records; do not parse positional suffixes."""

    return list(
        dict.fromkeys(display_key(records[key]) if key in records else key for key in candidates)
    )


def lookup_message(problem: str, candidates: Sequence[str], remedy: str) -> str:
    """Keep the problem first, followed by at most five candidates and the remedy."""

    lines = [problem]
    if candidates:
        lines.append(card_row("candidates", candidates_text(candidates)))
    lines.append(card_row("remedy", remedy))
    return "\n".join(lines)


def short_name_message(key: str, candidates: Sequence[str]) -> str:
    """Offer full parameter or buffer addresses without dumping every match."""

    return lookup_message(f"Ambiguous short name {key!r}.", candidates, "use a full address")


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
    )
