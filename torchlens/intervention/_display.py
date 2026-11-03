"""Bounded intervention displays, separate from content identity and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._display_card import LIST_LIMIT, render_card
from .._utils._display_constructor import clipped, display_guard, value_text
from .._utils._display_numbers import plural
from .._utils.fail_open import fail_open

if TYPE_CHECKING:
    from .spec import Recipe, Rule
    from .types import CapturedArgTemplate, FrozenInterventionSpec


def _term_text(value: Any) -> str:
    """Use the designed repr for selectors and edits, not arbitrary callables."""

    if type(value).__module__ in {
        "torchlens._selection.selector_base",
        "torchlens._selection.selectors",
    } or (
        type(value).__module__ == "torchlens.intervention.types" and type(value).__name__ == "Edit"
    ):
        return fail_open(lambda: clipped(repr(value)), lambda _error: f"<{type(value).__name__}>")
    return value_text(value)


@display_guard
def rule_repr(rule: Rule) -> str:
    """Read stored rule metadata without computing its content-derived identity."""

    return clipped(f"Rule: {_rule_text(rule)}")


def _rule_text(rule: Rule) -> str:
    """Describe one target and edit without walking tensor contents."""

    return f"{_term_text(rule.target)} gets {_term_text(rule.edit)}, {rule.signal}, {rule.address_class}"


@display_guard
def rule_card(rule: Rule) -> str:
    """Show rule metadata and the already computed non-tensor identity, when available."""

    memo = rule._identity_memo
    identity = memo[1][1] if memo is not None and not rule._action_tensors else "not computed"
    return render_card(
        rule_repr(rule),
        [
            ("target", _term_text(rule.target)),
            ("edit", _term_text(rule.edit)),
            ("signal", rule.signal),
            ("address", rule.address_class),
            ("identity", identity),
        ],
        (".target", ".edit", ".rule_id"),
    )


@display_guard
def recipe_repr(recipe: Recipe, selector_type: type[Any]) -> str:
    """Use tl.when only for a single, exactly reconstructible clause."""

    if len(recipe.rules) == 1:
        rule = recipe.rules[0]
        target, edit = _term_text(rule.target), _term_text(rule.edit)
        spelling = f"tl.when({target}, {edit})"
        if (
            isinstance(rule.target, selector_type)
            and edit.startswith("tl.")
            and not any(token in spelling for token in ("<", "...", "tl.Edit("))
            and len(spelling) <= 100
        ):
            return spelling
    return f"Recipe: {plural(len(recipe.rules), 'rule')}"


@display_guard
def recipe_card(recipe: Recipe) -> str:
    """Show at most five rules and disclose the exact remainder."""

    rows = [
        (f"rule {index}", _rule_text(rule))
        for index, rule in enumerate(recipe.rules[:LIST_LIMIT], 1)
    ]
    if len(recipe.rules) > LIST_LIMIT:
        rows.append(("rules", f"... and {len(recipe.rules) - LIST_LIMIT} more rules"))
    return render_card(repr(recipe), rows, (".rules",))


@display_guard
def template_repr(template: CapturedArgTemplate) -> str:
    """Count replay arguments; literal tensors and nested records never dump."""

    return (
        f"CapturedArgTemplate: {plural(len(template.args), 'positional arg')}, "
        f"{plural(len(template.kwargs), 'keyword arg')}"
    )


def _component_text(component: Any) -> str:
    """Describe literals through metadata-only values, with other records kept shallow."""

    if type(component).__name__ in {"LiteralTensor", "LiteralValue"}:
        return value_text(component.value)
    return value_text(component)


@display_guard
def template_card(template: CapturedArgTemplate) -> str:
    """Show a bounded prefix of positional and keyword argument templates."""

    rows = [
        (f"arg {i}", _component_text(item)) for i, item in enumerate(template.args[:LIST_LIMIT])
    ]
    remaining = max(0, len(template.args) - LIST_LIMIT)
    budget = LIST_LIMIT - len(rows)
    rows.extend((clipped(key, 30), _component_text(item)) for key, item in template.kwargs[:budget])
    remaining += max(0, len(template.kwargs) - budget)
    if remaining:
        rows.append(("args", f"... and {remaining} more"))
    return render_card(template_repr(template), rows, (".args", ".kwargs", ".func_id", ".notes"))


@display_guard
def frozen_spec_repr(spec: FrozenInterventionSpec) -> str:
    """Summarize frozen intervention state without printing callbacks or record payloads."""

    return (
        f"FrozenInterventionSpec: {plural(len(spec.targets), 'target')}, "
        f"{plural(len(spec.records), 'fire')}"
    )


@display_guard
def frozen_spec_card(spec: FrozenInterventionSpec) -> str:
    """Expose counts and a safe value summary; detail fields stay directly accessible."""

    return render_card(
        frozen_spec_repr(spec),
        [
            ("edit", _term_text(spec.helper)),
            ("value", value_text(spec.value)),
            ("hooks", str(len(spec.hook_specs))),
            ("metadata", f"{len(spec.metadata)} entries"),
        ],
        (".targets", ".records", ".target_value_specs", ".hook_specs", ".metadata"),
    )


@display_guard
def edge_repr(edge: Any) -> str:
    """Describe one stored edge occurrence, without inventing short labels."""

    path = ".".join(str(component) for component in edge.arg_path) or "?"
    relation = f", {edge.view_or_copy}" if edge.view_or_copy in ("view", "copy") else ""
    return clipped(
        f"EdgeUseRecord: {edge.parent_label} -> {edge.child_label}, {edge.arg_kind} {path}{relation}"
    )


@display_guard
def edge_card(edge: Any) -> str:
    """Expose the stored occurrence address without printing nested payloads."""

    return render_card(
        edge_repr(edge),
        [
            ("parent", edge.parent_label),
            ("child", edge.child_label),
            ("argument", f"{edge.arg_kind} {value_text(edge.arg_path)}"),
            ("relation", edge.view_or_copy or "not recorded"),
        ],
        (".parent_label", ".child_label", ".child_func_call_id", ".arg_path"),
    )


@display_guard
def bound_model_repr(binding: Any) -> str:
    """Read binding metadata without computing a recipe digest."""

    status = binding._last_report.status if binding._last_report is not None else "never called"
    return clipped(
        f"BoundModel {type(binding._base_model).__name__}: "
        f"{plural(len(binding._spec.rules), 'rule')}, capture-free live lane, {status}"
    )


@display_guard
def bound_model_card(binding: Any) -> str:
    """Disclose the execution contract and read-only detail exits."""

    return render_card(
        bound_model_repr(binding),
        [
            ("execution", "values replaced after execution"),
            ("surface", ".recipe/.model read-only"),
        ],
        (".recipe", ".model", ".last_report"),
    )
