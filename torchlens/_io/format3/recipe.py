"""The intervention recipe of the 3.0 saved format: ``intervention.json``.

One grammar, two homes: a trace saved with a staged intervention spec carries
the file beside its records, and ``save_intervention`` writes it as a
``kind="intervention"`` artifact of its own. Every rule is the staged hook's
selector plus the ``Edit`` identity under the typed value rules; a tensor
argument is a tensor reference into ``blobs/recipe-<nnnn>.safetensors``; a user
callable is never written: it becomes a ``callable_placeholder`` at rule level
(integer ``position``) or in place inside a built-in edit's arguments
(``"<rule index>.<arg path>"``), and the recipe is then not executable.
Contract: ``docs/maintainers/saved_format_3.md``, section 5; design:
``docs/maintainers/saved_format_3_design.md``, section 8.

This module is duck-typed over the staged spec (``hook_specs``,
``target_value_specs``) so the writer imports nothing from the intervention
package; the intervention package rebuilds runtime specs from
:func:`read_recipe`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import torch

from .. import _json
from ..format_errors import ArtifactError
from .contract import INTERVENTION_FILE, TYPED_VALUE_KEY
from .payloads import PayloadStore, host_tensor, read_tensor, tensor_reference
from .values import ValueEncoder, decode_value

#: ``format`` of the recipe file.
RECIPE_FORMAT: Final = "torchlens.intervention_recipe"
#: ``format_version`` of the recipe file (the saved-format major).
RECIPE_FORMAT_VERSION: Final = 3
#: Save level whose recipe is never executable.
_AUDIT_LEVEL: Final = "audit"
#: Edit portability of a rule whose edit is a user callable.
PLACEHOLDER_PORTABILITY: Final = "callable_placeholder"
#: Edit name of a rule staged from a raw callable hook.
_CUSTOM_EDIT_NAME: Final = "custom"


class CallablePlaceholder(NamedTuple):
    """A callable that the recipe names but does not carry.

    Attributes
    ----------
    position:
        Rule index, or ``"<rule index>.<arg path>"`` inside edit arguments.
    qualname:
        ``module.qualname`` of the callable when it was saved.
    repr_length:
        Length of the callable's repr when it was saved.
    """

    position: int | str
    qualname: str
    repr_length: int


def _pairs_to_dict(value: Any) -> dict[str, Any]:
    """Return a dict from an ``Edit`` tuple-of-pairs field or a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    return {str(key): item for key, item in (value or ())}


def _is_edit(value: Any) -> bool:
    """Return whether a value carries the ``Edit`` identity fields."""

    return all(hasattr(value, name) for name in ("name", "args", "kwargs", "portability"))


def _selector(target: Any, encoder: _RecipeEncoder, at: str) -> dict[str, Any]:
    """Encode a staged target spec as the rule ``selector`` object."""

    slice_spec = getattr(target, "slice_spec", None)
    if dataclasses.is_dataclass(slice_spec) and not isinstance(slice_spec, type):
        slice_spec = dataclasses.asdict(slice_spec)
    selector_value = getattr(target, "selector_value", None)
    if _is_target(selector_value):
        selector_value = _selector(selector_value, encoder, f"{at}.selector_value")
    else:
        selector_value = encoder.encode(selector_value, f"{at}.selector_value")
    return {
        "selector_kind": getattr(target, "selector_kind", None),
        "selector_value": selector_value,
        "strict": bool(getattr(target, "strict", True)),
        "slice_spec": encoder.encode(slice_spec, f"{at}.slice_spec"),
        "metadata": encoder.encode(
            _pairs_to_dict(getattr(target, "metadata", {})), f"{at}.metadata"
        ),
    }


def _is_target(value: Any) -> bool:
    """Return whether a value is a staged target spec."""

    return hasattr(value, "selector_kind") and hasattr(value, "selector_value")


def _placeholder_edit(index: int, hook: Any, encoder: _RecipeEncoder) -> dict[str, Any]:
    """Return the edit object of a rule whose edit is a user callable."""

    name = _CUSTOM_EDIT_NAME
    if _is_edit(hook):
        name = str(hook.name)
        hook = getattr(hook, "factory", None) or hook
    return {
        "name": name,
        "args": [],
        "kwargs": {},
        "portability": PLACEHOLDER_PORTABILITY,
        "callable": encoder.callable_placeholder(hook, position=index),
        "signal": None,
        "is_batch_independent": False,
        "is_compatible_with_append": False,
        "metadata": {},
        "selection_recipe": None,
    }


def _edit(index: int, edit: Any, encoder: _RecipeEncoder) -> dict[str, Any]:
    """Encode a built-in edit; callables in its arguments become placeholders."""

    if not _is_edit(edit) or edit.portability != "builtin":
        return _placeholder_edit(index, edit, encoder)
    kwargs = _pairs_to_dict(edit.kwargs)
    return {
        "name": str(edit.name),
        "args": [encoder.encode(arg, f"{index}.args.{i}") for i, arg in enumerate(edit.args)],
        "kwargs": {
            key: encoder.encode(item, f"{index}.kwargs.{key}") for key, item in kwargs.items()
        },
        "portability": "builtin",
        "signal": getattr(edit, "signal", None),
        "is_batch_independent": bool(getattr(edit, "is_batch_independent", False)),
        "is_compatible_with_append": bool(getattr(edit, "is_compatible_with_append", False)),
        "metadata": encoder.encode(
            _pairs_to_dict(getattr(edit, "metadata", ())), f"{index}.metadata"
        ),
        "selection_recipe": encoder.encode(
            getattr(edit, "selection_recipe", None), f"{index}.selection_recipe"
        ),
    }


def _direction(metadata: Mapping[str, Any], target: Any) -> str:
    """Return ``backward`` for a gradient-site rule, else ``forward``."""

    created_by = str(metadata.get("created_by", ""))
    kind = str(getattr(target, "selector_kind", ""))
    return "backward" if "backward" in created_by or "grad" in kind else "forward"


def _rule(index: int, hook_spec: Any, encoder: _RecipeEncoder) -> dict[str, Any]:
    """Encode one staged hook as a recipe rule."""

    metadata = dict(getattr(hook_spec, "metadata", {}) or {})
    edit = getattr(hook_spec, "helper", None)
    if edit is None:
        edit = getattr(hook_spec, "hook", None)
    if metadata.get("facet_write"):
        # A facet write fires through a capture-bound scatter wrapper; the raw
        # edit alone would apply to the whole tensor, so the rule keeps only a
        # placeholder of the wrapper.
        edit = getattr(hook_spec, "hook", None)
        encoded_edit = _placeholder_edit(index, edit, encoder)
    else:
        encoded_edit = _edit(index, edit, encoder)
    target = getattr(hook_spec, "site_target", None)
    rule_metadata = dict(metadata)
    handle = getattr(hook_spec, "handle", None)
    if handle is not None:
        rule_metadata["handle"] = handle
    return {
        "index": index,
        "direction": _direction(metadata, target),
        "selector": _selector(target, encoder, f"{index}.selector"),
        "edit": encoded_edit,
        "metadata": encoder.encode(rule_metadata, f"{index}.metadata"),
    }


def _value_rule(index: int, value_spec: Any, encoder: _RecipeEncoder) -> dict[str, Any]:
    """Encode one staged value replacement (``trace.set``)."""

    value = getattr(value_spec, "value", None)
    encoded = (
        encoder.callable_placeholder(value, position=f"values.{index}")
        if callable(value) and not isinstance(value, torch.Tensor)
        else encoder.encode(value, f"values.{index}.value")
    )
    return {
        "index": index,
        "selector": _selector(getattr(value_spec, "site_target", None), encoder, f"values.{index}"),
        "value": encoded,
        "metadata": encoder.encode(
            dict(getattr(value_spec, "metadata", {}) or {}), f"values.{index}.metadata"
        ),
    }


class _RecipeEncoder(ValueEncoder):
    """Value encoder of the recipe: callables become placeholders, tensors chunk payloads."""

    def __init__(self, store: PayloadStore) -> None:
        super().__init__(tensor_ref=self._store_tensor, callables="placeholder")
        self._store = store
        self.placeholders: list[dict[str, Any]] = []

    def encode(self, value: Any, at: str, *, payload: bool = True) -> Any:
        """Encode one recipe value; every tensor in a recipe is a payload."""

        encoded = super().encode(value, at, payload=payload)
        self.placeholders.extend(_placeholders_in(encoded))
        return encoded

    def callable_placeholder(self, value: Any, *, position: int | str) -> dict[str, Any]:
        """Return (and record) the rule-level placeholder of a callable."""

        from_values = super().encode(value, str(position), payload=False)
        if not (isinstance(from_values, dict) and from_values.get(TYPED_VALUE_KEY)):
            return from_values
        placeholder = {**from_values, "position": position}
        self.placeholders.append(placeholder)
        return placeholder

    def _store_tensor(self, value: Any, at: str) -> dict[str, Any] | None:
        """Write one recipe tensor into ``blobs/recipe-<nnnn>.safetensors``."""

        if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
            return None
        blob, key = self._store.recipe_writer().submit(host_tensor(value))
        return tensor_reference(value, blob=blob, key=key)


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


def build_recipe(
    spec: Any,
    *,
    store: PayloadStore,
    save_level: str,
    helper_registry_version: Any,
    append_state: Mapping[str, Any],
    function_registry_keys: Sequence[Mapping[str, Any]],
    spec_derived: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the recipe document of a staged spec and the values it disclosed.

    Parameters
    ----------
    spec:
        Staged intervention spec (``hook_specs``, ``target_value_specs``).
    store:
        Payload store of the artifact (recipe tensors go to its recipe series).
    save_level:
        Resolved save level.
    helper_registry_version:
        Version of the built-in edit registry.
    append_state:
        Append provenance of the saved trace.
    function_registry_keys:
        Function registry keys of the saved trace's ops.
    spec_derived:
        Whether every rule was lowered from a capture-time predicate.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        The JSON-ready recipe and the ``unencodable`` disclosures.
    """

    encoder = _RecipeEncoder(store)
    rules = [_rule(i, hook, encoder) for i, hook in enumerate(getattr(spec, "hook_specs", ()))]
    values = [
        _value_rule(i, value_spec, encoder)
        for i, value_spec in enumerate(getattr(spec, "target_value_specs", ()))
    ]
    helpers = {
        rule["edit"]["name"]: {"name": rule["edit"]["name"], "version": helper_registry_version}
        for rule in rules
        if rule["edit"]["portability"] == "builtin"
    }
    executable = save_level != _AUDIT_LEVEL and not encoder.placeholders and not encoder.disclosures
    recipe = {
        "format": RECIPE_FORMAT,
        "format_version": RECIPE_FORMAT_VERSION,
        "helper_registry_version": helper_registry_version,
        "save_level": save_level,
        "executable": executable,
        "spec_derived": bool(spec_derived),
        "append_state": ValueEncoder().encode(dict(append_state), "append_state"),
        "rules": rules,
        "values": values,
        "helpers": list(helpers.values()),
        "function_registry_keys": [dict(entry) for entry in function_registry_keys],
    }
    return recipe, list(encoder.disclosures)


def manifest_entry(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest ``intervention`` object of a written recipe.

    Parameters
    ----------
    recipe:
        The recipe document.

    Returns
    -------
    dict[str, Any]
        ``file``, ``executable``, the rule and value counts, and the
        placeholder positions with their qualified names.
    """

    return {
        "file": INTERVENTION_FILE,
        "executable": bool(recipe["executable"]),
        "rules": len(recipe["rules"]),
        "values": len(recipe.get("values", ())),
        "placeholders": [
            {"position": item["position"], "qualname": item["qualname"]}
            for item in _placeholders_in([recipe["rules"], recipe.get("values", [])])
        ],
    }


def write_recipe(root: Path, recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``intervention.json`` under a staged artifact; return the manifest entry.

    Parameters
    ----------
    root:
        Staged artifact directory.
    recipe:
        The recipe document from :func:`build_recipe`.

    Returns
    -------
    dict[str, Any]
        The manifest ``intervention`` object.
    """

    text = json.dumps(recipe, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (root / INTERVENTION_FILE).write_text(text, encoding="utf-8")
    return manifest_entry(recipe)


def _bind(data: Any, root: Path, at: str) -> Any:
    """Decode recipe data, reading tensors and keeping placeholders typed."""

    if isinstance(data, list):
        return [_bind(item, root, f"{at}[{i}]") for i, item in enumerate(data)]
    if not isinstance(data, dict):
        return data
    tag = data.get(TYPED_VALUE_KEY)
    if tag is None:
        return {key: _bind(item, root, f"{at}[{key!r}]") for key, item in data.items()}
    if tag == "callable_placeholder":
        fields = decode_value(data, at=at)
        return CallablePlaceholder(
            fields["position"], str(fields["qualname"]), int(fields["repr_length"])
        )
    if tag == "tensor":
        tensor = read_tensor(root, blob=str(data["blob"]), key=str(data["key"]))
        return torch.view_as_complex(tensor) if data.get("complex") else tensor
    if tag in {"tuple", "set", "frozenset"}:
        items = [_bind(item, root, f"{at}[{i}]") for i, item in enumerate(data["items"])]
        return {"tuple": tuple, "set": set, "frozenset": frozenset}[tag](items)
    return decode_value(data, at=at)


def read_recipe(root: str | Path) -> dict[str, Any]:
    """Read ``intervention.json`` with its tensors; placeholders stay placeholders.

    Parameters
    ----------
    root:
        Artifact directory holding ``intervention.json``.

    Returns
    -------
    dict[str, Any]
        The decoded recipe: tensors are CPU tensors, callables are
        :class:`CallablePlaceholder` values.

    Raises
    ------
    ArtifactError
        ``bundle_load_failed`` when the file is missing, malformed or not a
        recipe of this format version.
    """

    path = Path(root) / INTERVENTION_FILE
    try:
        data = _json.read_bounded(path)
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            f"The intervention recipe {path} could not be read: {exc}",
            code="bundle_load_failed",
            remedy="the artifact is incomplete or damaged; re-save it",
        ) from exc
    if not isinstance(data, dict) or data.get("format") != RECIPE_FORMAT:
        raise ArtifactError(
            f"{path} is not a TorchLens intervention recipe.",
            code="bundle_load_failed",
            remedy="load a directory written by save_intervention or trace.save",
        )
    if data.get("format_version") != RECIPE_FORMAT_VERSION:
        raise ArtifactError(
            f"{path} has recipe format_version {data.get('format_version')!r}; this "
            f"TorchLens reads {RECIPE_FORMAT_VERSION}.",
            code="bundle_load_failed",
            remedy="re-save the recipe with this TorchLens version",
        )
    return {key: _bind(value, Path(root), key) for key, value in data.items()}


def recipe_placeholders(recipe: Mapping[str, Any]) -> list[CallablePlaceholder]:
    """Return every placeholder of a decoded recipe, in rule order.

    Parameters
    ----------
    recipe:
        Recipe from :func:`read_recipe`.

    Returns
    -------
    list[CallablePlaceholder]
        The callables the recipe names but does not carry.
    """

    found: list[CallablePlaceholder] = []
    stack: list[Any] = [recipe.get("values", []), recipe.get("rules", [])]
    while stack:
        item = stack.pop()
        if isinstance(item, CallablePlaceholder):
            found.append(item)
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(reversed(list(item)))
    return found


def trace_recipe_context(trace: Any) -> dict[str, Any]:
    """Return the recipe context fields of a live trace (append state, keys, derivation).

    Parameters
    ----------
    trace:
        The trace being saved.

    Returns
    -------
    dict[str, Any]
        ``append_state``, ``function_registry_keys`` (the keys the ops'
        argument templates carry) and ``spec_derived``.
    """

    append_records = [
        record
        for record in getattr(trace, "state_history", None) or ()
        if isinstance(record, dict) and record.get("op") == "append"
    ]
    keys = []
    for layer in getattr(trace, "_layer_list", None) or ():
        key = getattr(getattr(layer, "args_template", None), "func_id", None)
        if dataclasses.is_dataclass(key) and not isinstance(key, type):
            keys.append({"layer_label": str(layer.layer_label), "key": dataclasses.asdict(key)})
    spec = getattr(trace, "_intervention_spec", None)
    hook_specs = list(getattr(spec, "hook_specs", ()) or ())
    return {
        "append_state": {
            "is_appended": bool(getattr(trace, "_provenance_is_appended", False)),
            "append_sequence_id": int(getattr(trace, "_append_sequence_id", 0) or 0),
            "append_history": list(getattr(trace, "_append_history", None) or ()),
            "state_history": append_records,
        },
        "function_registry_keys": keys,
        "spec_derived": bool(hook_specs)
        and all(
            dict(getattr(hook, "metadata", {}) or {}).get("created_by") == "intervene_predicate"
            for hook in hook_specs
        ),
    }


def write_trace_recipe(
    root: Path,
    trace: Any,
    *,
    store: PayloadStore,
    save_level: str,
    helper_registry_version: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Write the recipe of a trace's staged spec beside its records.

    Parameters
    ----------
    root:
        Staged trace artifact directory.
    trace:
        The trace being saved.
    store:
        The artifact's payload store (call before it closes).
    save_level:
        Resolved save level.
    helper_registry_version:
        Version of the built-in edit registry.

    Returns
    -------
    tuple[dict[str, Any] | None, list[dict[str, Any]]]
        The manifest ``intervention`` object (``None`` when the trace has no
        staged rule) and the recipe's disclosures.
    """

    spec = getattr(trace, "_intervention_spec", None)
    if not has_rules(spec):
        return None, []
    recipe, disclosures = build_recipe(
        spec,
        store=store,
        save_level=save_level,
        helper_registry_version=helper_registry_version,
        **trace_recipe_context(trace),
    )
    return write_recipe(root, recipe), disclosures


def has_rules(spec: Any) -> bool:
    """Return whether a staged spec holds any rule or value replacement."""

    return bool(getattr(spec, "hook_specs", None) or getattr(spec, "target_value_specs", None))


__all__ = [
    "PLACEHOLDER_PORTABILITY",
    "RECIPE_FORMAT",
    "RECIPE_FORMAT_VERSION",
    "CallablePlaceholder",
    "build_recipe",
    "has_rules",
    "manifest_entry",
    "read_recipe",
    "recipe_placeholders",
    "trace_recipe_context",
    "write_recipe",
    "write_trace_recipe",
]
