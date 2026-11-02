"""Read the intervention recipe (``intervention.json``) of a 3.0 artifact.

Design section 8: built-in edits come back through the helper registry, with
their factories rebuilt from name, args and kwargs; a callable placeholder
(rule level or inside an edit's arguments) loads as an audit-only edit and the
recipe's ``placeholders`` list names it, so every rerun door refuses with
``run_intervention_spec_not_persisted``. The grammar is closed: exact key sets
per recipe, rule and edit, selectors through the spec codec, values through
the strict cell decoder. This module decodes to plain values only; the
helper-registry rebuild and the ``InterventionSpec`` are the ``recipe`` step
``torchlens._io.bundle`` installs into the reader doors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .format3 import contract
from .format3.decode import CallablePlaceholder, Decoder
from .format3.validate import integrity_error, read_json_file
from .format_errors import ArtifactError

__all__ = ["DecodedRecipe", "DecodedRule", "read_recipe"]

_CORRUPT: Final = contract.REFUSAL_CODES["record_file_corrupt"]


_RECIPE_FORMAT: Final = "torchlens.intervention_recipe"
_RECIPE_KEYS: Final = frozenset(
    {
        "format", "format_version", "helper_registry_version", "save_level", "executable",
        "spec_derived", "append_state", "rules", "helpers", "function_registry_keys",
    }
)  # fmt: skip
_RULE_KEYS: Final = frozenset({"index", "direction", "selector", "edit", "metadata"})
_EDIT_KEYS: Final = frozenset(
    {
        "name", "args", "kwargs", "portability", "signal", "is_batch_independent",
        "is_compatible_with_append", "metadata", "selection_recipe", "callable",
    }
)  # fmt: skip
_PORTABILITIES: Final = ("builtin", "callable_placeholder")


@dataclass(frozen=True)
class DecodedRule:
    """One recipe rule after the grammar check and the cell decode, before any object is built.

    ``selector`` is the selector object as written (``selector_kind`` and its
    fields); ``edit`` holds the decoded edit fields (``name``, ``args``,
    ``kwargs``, ``portability``, ...) with tensor arguments still as
    ``BlobRef`` values; ``placeholders`` lists ``(position, qualname)`` for
    every callable placeholder under the edit.
    """

    position: int
    direction: str
    selector: dict[str, Any]
    edit: dict[str, Any]
    metadata: dict[str, Any]
    placeholders: tuple[tuple[int | str, str], ...]


@dataclass(frozen=True)
class DecodedRecipe:
    """The recipe file decoded to plain values; ``rehydrate`` turns it into an ``InterventionSpec``."""

    save_level: str
    executable: bool
    spec_derived: Any
    append_state: dict[str, Any]
    helpers: Any
    function_registry_keys: Any
    helper_registry_version: Any
    rules: tuple[DecodedRule, ...]

    @property
    def placeholders(self) -> list[tuple[int | str, str]]:
        return [found for rule in self.rules for found in rule.placeholders]


def _recipe_error(root: Path, detail: str, *, code: str) -> ArtifactError:
    return integrity_error(root / contract.INTERVENTION_FILE, detail, code=code)


def read_recipe(root: Path, manifest: Mapping[str, Any], *, decoder: Decoder) -> DecodedRecipe:
    """Read and decode ``intervention.json`` against the recipe grammar (contract (e)).

    Parameters
    ----------
    root, manifest:
        The validated artifact and its manifest (the recipe's digest is listed there).
    decoder:
        The artifact's cell decoder; its tensor sink records every tensor
        reference an edit argument carries.

    Every grammar violation refuses with ``bundle_metadata_integrity_refused``
    and names the rule; no selector, edit or spec object is built here.
    """

    target = root / contract.INTERVENTION_FILE
    data, raw = read_json_file(target, code=_CORRUPT, what="recipe")
    listed = manifest["files"][contract.INTERVENTION_FILE]
    if len(raw) != listed["bytes"] or hashlib.sha256(raw).hexdigest() != listed["sha256"]:
        raise _recipe_error(root, "bytes differ from the manifest's digest", code=_CORRUPT)
    if not isinstance(data, dict) or set(data) != _RECIPE_KEYS:
        raise _recipe_error(
            root, f"the recipe carries exactly {sorted(_RECIPE_KEYS)}", code=_CORRUPT
        )
    if data["format"] != _RECIPE_FORMAT or data["format_version"] != contract.FORMAT_MAJOR:
        raise _recipe_error(root, f"format must be {_RECIPE_FORMAT} version 3", code=_CORRUPT)
    if not isinstance(data["rules"], list) or not isinstance(data["executable"], bool):
        raise _recipe_error(root, "rules must be an array, executable a boolean", code=_CORRUPT)
    if data["save_level"] not in contract.SAVE_LEVELS or not isinstance(data["append_state"], dict):
        raise _recipe_error(root, "save_level unknown or append_state not an object", code=_CORRUPT)
    rules = tuple(
        _decode_rule(rule, position, decoder, root) for position, rule in enumerate(data["rules"])
    )
    return DecodedRecipe(
        save_level=data["save_level"],
        executable=data["executable"],
        spec_derived=data["spec_derived"],
        append_state=data["append_state"],
        helpers=data["helpers"],
        function_registry_keys=data["function_registry_keys"],
        helper_registry_version=data["helper_registry_version"],
        rules=rules,
    )


def _decode_rule(rule: Any, position: int, decoder: Decoder, root: Path) -> DecodedRule:
    if not isinstance(rule, dict) or set(rule) != _RULE_KEYS or rule["index"] != position:
        raise _recipe_error(root, f"rule {position} must carry {sorted(_RULE_KEYS)}", code=_CORRUPT)
    if rule["direction"] not in ("forward", "backward") or not isinstance(rule["metadata"], dict):
        raise _recipe_error(root, f"rule {position} direction or metadata mistyped", code=_CORRUPT)
    selector = rule["selector"]
    if not isinstance(selector, dict) or not isinstance(selector.get("selector_kind"), str):
        raise _recipe_error(
            root, f"rule {position} selector is not a selector object", code=_CORRUPT
        )
    edit = _decode_edit(rule["edit"], decoder, root, position)
    found = tuple(
        (p.position, p.qualname)
        for p in _placeholders_in((edit.get("args"), edit.get("kwargs"), edit.get("callable")))
    )
    if edit["portability"] == "callable_placeholder" and not found:
        raise _recipe_error(
            root, f"rule {position} placeholder edit lacks its callable", code=_CORRUPT
        )
    metadata = decoder.decode(rule["metadata"], at=f"recipe.rules[{position}].metadata")
    return DecodedRule(
        position=position,
        direction=rule["direction"],
        selector=dict(selector),
        edit=edit,
        metadata=dict(metadata),
        placeholders=found,
    )


def _decode_edit(edit: Any, decoder: Decoder, root: Path, position: int) -> dict[str, Any]:
    """Decode the edit object's cells; the helper registry rebuild happens in ``rehydrate``."""

    if not isinstance(edit, dict) or not set(edit) <= _EDIT_KEYS or "name" not in edit:
        raise _recipe_error(root, f"rule {position} edit is not an edit object", code=_CORRUPT)
    at = f"recipe.rules[{position}].edit"
    fields = {key: decoder.decode(value, at=f"{at}.{key}") for key, value in edit.items()}
    fields.setdefault("portability", "builtin")
    if not isinstance(fields["name"], str) or fields["portability"] not in _PORTABILITIES:
        raise _recipe_error(
            root, f"rule {position} edit name or portability unknown", code=_CORRUPT
        )
    args, kwargs = fields.get("args", ()), fields.get("kwargs", {})
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise _recipe_error(root, f"rule {position} edit args or kwargs mistyped", code=_CORRUPT)
    fields["args"], fields["kwargs"] = tuple(args), dict(kwargs)
    return fields


def _placeholders_in(value: Any) -> Iterator[CallablePlaceholder]:
    if isinstance(value, CallablePlaceholder):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _placeholders_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _placeholders_in(item)
