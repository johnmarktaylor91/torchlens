"""Structural checks of a recipe document, run before any value is decoded.

``read_recipe`` refuses a recipe whose keys, save level, rules or capture rules are
not the closed shape ``build_recipe`` writes; ``_placeholders_in`` finds the callable
placeholders a document carries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .contract import SAVE_LEVELS, TYPED_VALUE_KEY


def _placeholders_in(data: Any) -> list[dict[str, Any]]:
    """Return every ``callable_placeholder`` object inside encoded data."""

    found: list[dict[str, Any]] = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get(TYPED_VALUE_KEY) == "callable_placeholder":
                found.append(item)
                continue
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


#: Save levels a recipe may carry: the artifact's own, so a recipe saved with a runnable
#: trace (``level="runnable"``) reads back.
_RECIPE_SAVE_LEVELS: Final = frozenset(SAVE_LEVELS)
#: Keys every rule object carries.
_RULE_KEYS: Final = frozenset({"index", "selector", "edit"})


def _recipe_shape_problem(data: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return ``(detail, code)`` for the first structural fault of a recipe, else ``None``."""

    allowed = {
        "format",
        "format_version",
        "helper_registry_version",
        "save_level",
        "executable",
        "spec_derived",
        "append_state",
        "rules",
        "values",
        "helpers",
        "function_registry_keys",
        "capture_rules",
    }
    if set(data) - allowed:
        return "recipe contains unknown keys", "manifest_schema_invalid"
    problem = _capture_rules_problem(data.get("capture_rules", []))
    if problem is not None:
        return problem, "manifest_schema_invalid"
    if data.get("save_level") not in _RECIPE_SAVE_LEVELS:
        return f"save_level {data.get('save_level')!r} is unknown", "artifact_save_level_invalid"
    rules = data.get("rules")
    if not isinstance(rules, list):
        return "rules is missing or not an array", "manifest_schema_invalid"
    if not isinstance(data.get("values", []), list):
        return "values is not an array", "manifest_schema_invalid"
    for position, rule in enumerate(rules):
        if not isinstance(rule, dict) or not _RULE_KEYS.issubset(rule):
            return f"rule {position} lacks {sorted(_RULE_KEYS)}", "manifest_schema_invalid"
        if not isinstance(rule["edit"], dict) or not isinstance(rule["selector"], dict):
            return f"rule {position} edit or selector is not an object", "manifest_schema_invalid"
    return None


# Complexity: the closed capture-rule grammar checks selector and edit portability together.
def _capture_rules_problem(rules: Any) -> str | None:
    """Check the closed capture-rule schema before decoding any member."""

    if not isinstance(rules, list):
        return "capture_rules is not an array"
    if _placeholders_in(rules):
        return "capture_rules requires portable built-in edits"
    selector_keys = {"selector_kind", "selector_value", "strict", "slice_spec", "metadata"}
    edit_keys = {
        "name",
        "args",
        "kwargs",
        "portability",
        "signal",
        "is_batch_independent",
        "is_compatible_with_append",
        "metadata",
        "selection_recipe",
    }
    # Keep the closed wire-schema checks together so every refusal is reviewed in one place.
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or set(rule) != {"index", "selector", "edit"}:
            return f"capture_rules[{index}] has unknown or missing keys"
        if type(rule["index"]) is not int or rule["index"] != index:
            return f"capture_rules[{index}] has an invalid index"
        target, edit = rule["selector"], rule["edit"]
        if not isinstance(target, dict) or set(target) != selector_keys:
            return f"capture_rules[{index}] has an invalid selector"
        if type(target["strict"]) is not bool or not isinstance(target["metadata"], dict):
            return f"capture_rules[{index}] has invalid selector field types"
        if not isinstance(target["selector_kind"], str) or not target["selector_kind"]:
            return f"capture_rules[{index}] has an invalid selector kind"
        if not isinstance(edit, dict) or set(edit) != edit_keys or edit["portability"] != "builtin":
            return f"capture_rules[{index}] requires a built-in edit"
        if (
            not isinstance(edit["args"], list)
            or not isinstance(edit["kwargs"], dict)
            or not isinstance(edit["metadata"], dict)
            or type(edit["is_batch_independent"]) is not bool
            or type(edit["is_compatible_with_append"]) is not bool
        ):
            return f"capture_rules[{index}] has invalid edit field types"
    return None
