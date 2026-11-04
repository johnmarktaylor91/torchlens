"""Bounded module and module-call record displays."""

from __future__ import annotations

from itertools import islice
from typing import Any

from .._utils._display_card import inline_names, one_line, render_card
from .._utils._display_keys import display_key
from .._utils._display_numbers import format_count, format_exact_count, plural
from .._utils.fail_open import fail_open
from ._value_state import payload_facts, resolve_record, value_line


def _output(record: Any, label: str) -> str:
    """One output site's metadata and words; never choose from a plurality."""

    output = resolve_record(record, label)
    if output is None:
        return label
    metadata, stats, honesty = payload_facts(output)
    return value_line(display_key(output), metadata, stats, honesty)


def module_repr_line(module: Any) -> str:
    """Module identity, class, call/op counts, parameters and mode."""

    count = getattr(module, "num_params", None)
    params = (
        "params not counted"
        if count is None
        else (f"{format_count(count)} params" if count else "no params")
    )
    facts: tuple[str, ...] = (
        str(getattr(module, "class_name", "?")),
        plural(getattr(module, "num_calls", 0), "call"),
        plural(len(getattr(module, "layer_labels", ()) or ()), "op"),
        params,
        "train" if getattr(module, "training", False) else "eval",
    )
    if getattr(module, "has_multiple_addresses", False):
        facts += ("(aliased)",)
    return one_line(f"Module {getattr(module, 'address', '?')}", facts)


def module_card(module: Any) -> str:
    """Module facts with one output only when the captured output is singular."""

    rows = []
    count = getattr(module, "num_params", None)
    if count:
        rows.append(("params", format_exact_count(count)))
    if getattr(module, "has_multiple_addresses", False):
        rows.append(("aliases", inline_names(getattr(module, "all_addresses", ()) or ())))
    children = getattr(module, "address_children", ()) or ()
    if children:
        rows.append(("children", inline_names(children)))
    labels: Any = getattr(module, "output_layers", ()) or ()
    if len(labels) == 1:
        rows.append(("output", _output(module, labels[0])))
    elif labels:
        rows.append(
            ("outputs", f"{format_exact_count(len(labels))} output sites; see .output_layers")
        )
    return render_card(module_repr_line(module), rows, (".calls", ".layers", ".to_pandas()"))


def module_call_repr_line(call: Any) -> str:
    """Call index stays distinct from operation pass identity."""

    cls = getattr(call, "class_name", None)
    if not cls:
        cls = fail_open(lambda: call.module.class_name, lambda _error: "?")
    facts: tuple[str, ...] = (
        f"call {getattr(call, 'call_index', '?')}",
        str(cls),
        f"{format_count(len(getattr(call, 'ops', ()) or ()))} recorded operation rows",
    )
    return one_line(f"ModuleCall {getattr(call, 'address', '?')}", facts)


def module_call_card(call: Any) -> str:
    """At most three output sites and child calls, plus exact remainders."""

    rows = []
    labels: Any = getattr(call, "output_layers", ()) or ()
    for label in islice(labels, 3):
        rows.append(("output", _output(call, label)))
    if len(labels) > 3:
        rows.append(("", f"... and {format_exact_count(len(labels) - 3)} more outputs"))
    children = getattr(call, "call_children", ()) or ()
    if children:
        rows.append(("children", inline_names(children)))
    return render_card(module_call_repr_line(call), rows, (".outs", ".to_pandas()"))
