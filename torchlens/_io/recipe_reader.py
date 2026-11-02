"""Read the intervention recipe (``intervention.json``) of a 3.0 artifact.

Design section 8: built-in edits come back through the helper registry, with
their factories rebuilt from name, args and kwargs; a callable placeholder
(rule level or inside an edit's arguments) loads as an audit-only edit and the
recipe's ``placeholders`` list names it, so every rerun door refuses with
``run_intervention_spec_not_persisted``. The grammar is closed: exact key sets
per recipe, rule and edit, selectors through the spec codec, values through
the strict cell decoder. This module is the reader lane's; ``format3/reader.py``
is its only caller.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

from .format3 import contract
from .format3.decode import CallablePlaceholder, Decoder
from .format3.validate import integrity_error, read_json_file
from .format_contract import BlobRef
from .format_errors import ArtifactError
from .paths import resolve_bundle_blobs_dir

__all__ = ["read_recipe"]

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


class _Recipe:
    """The decoder and tensor entries one recipe read shares across its rules."""

    def __init__(self, decoder: Decoder, tensor_entries: Mapping[str, Mapping[str, Any]]) -> None:
        self.decoder = decoder
        self.tensor_entries = tensor_entries


def _recipe_error(root: Path, detail: str, *, code: str) -> ArtifactError:
    return integrity_error(root / contract.INTERVENTION_FILE, detail, code=code)


def read_recipe(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    decoder: Decoder,
    tensor_entries: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Return the ``InterventionSpec`` staged from the artifact's recipe file.

    Parameters
    ----------
    root, manifest:
        The validated artifact and its manifest (the recipe's digest is listed there).
    decoder:
        The artifact's cell decoder; its tensor sink fills ``tensor_entries`` as
        edit arguments are decoded.
    tensor_entries:
        Blob id to tensor entry, read when an argument tensor is materialized.
    """

    from ..intervention.types import InterventionSpec

    artifact = _Recipe(decoder, tensor_entries)

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
    spec = InterventionSpec(metadata={})
    placeholders: list[tuple[int | str, str]] = []
    for position, rule in enumerate(data["rules"]):
        placeholders.extend(_restore_rule(spec, rule, position, artifact, root))
    spec.metadata["recipe"] = {
        "executable": data["executable"] and not placeholders,
        "spec_derived": data["spec_derived"],
        "append_state": data["append_state"],
        "helpers": data["helpers"],
        "function_registry_keys": data["function_registry_keys"],
        "helper_registry_version": data["helper_registry_version"],
        "placeholders": placeholders,
    }
    return spec


def _restore_rule(
    spec: Any, rule: Any, position: int, artifact: _Recipe, root: Path
) -> list[tuple[int | str, str]]:
    """Stage one rule on ``spec`` as a sticky hook; return the placeholders it carries."""

    if not isinstance(rule, dict) or set(rule) != _RULE_KEYS or rule["index"] != position:
        raise _recipe_error(root, f"rule {position} must carry {sorted(_RULE_KEYS)}", code=_CORRUPT)
    if rule["direction"] not in ("forward", "backward") or not isinstance(rule["metadata"], dict):
        raise _recipe_error(root, f"rule {position} direction or metadata mistyped", code=_CORRUPT)
    selector = _decode_selector(rule["selector"], root, position)
    edit, found = _decode_edit(rule["edit"], artifact, root, position)
    metadata = {
        **artifact.decoder.decode(rule["metadata"], at=f"recipe.rules[{position}].metadata"),
        "direction": rule["direction"],
        "rule_index": position,
        "loaded_from_recipe": True,
    }
    hook = edit.factory() if edit.factory is not None else edit
    spec.add_hook(selector, hook, helper=edit, handle=f"recipe-{position}", metadata=metadata)
    return found


def _decode_selector(selector: Any, root: Path, position: int) -> Any:
    from ..intervention._spec_codec import _target_spec_from_json

    if not isinstance(selector, dict) or not isinstance(selector.get("selector_kind"), str):
        raise _recipe_error(
            root, f"rule {position} selector is not a selector object", code=_CORRUPT
        )
    try:
        return _target_spec_from_json(selector)
    except (KeyError, TypeError, ValueError) as exc:
        raise _recipe_error(
            root, f"rule {position} selector is malformed: {exc}", code=_CORRUPT
        ) from exc


def _decode_edit(
    edit: Any, artifact: _Recipe, root: Path, position: int
) -> tuple[Any, list[tuple[int | str, str]]]:
    """Rebuild a built-in edit through the helper registry; a placeholder loads audit-only."""

    from ..intervention.helpers import rebuild_builtin_helper
    from ..intervention.types import Edit

    if not isinstance(edit, dict) or not set(edit) <= _EDIT_KEYS or "name" not in edit:
        raise _recipe_error(root, f"rule {position} edit is not an edit object", code=_CORRUPT)
    at = f"recipe.rules[{position}].edit"
    fields = {key: artifact.decoder.decode(value, at=f"{at}.{key}") for key, value in edit.items()}
    name, portability = fields["name"], fields.get("portability", "builtin")
    if not isinstance(name, str) or portability not in ("builtin", "callable_placeholder"):
        raise _recipe_error(
            root, f"rule {position} edit name or portability unknown", code=_CORRUPT
        )
    args = tuple(_materialize(fields.get("args", ()), artifact, root))
    kwargs = dict(_materialize(fields.get("kwargs", {}), artifact, root))
    found = [
        (p.position, p.qualname) for p in _placeholders_in((args, kwargs, fields.get("callable")))
    ]
    if portability == "callable_placeholder" and not found:
        raise _recipe_error(
            root, f"rule {position} placeholder edit lacks its callable", code=_CORRUPT
        )
    extras: dict[str, Any] = {
        "metadata": tuple(sorted(dict(fields.get("metadata") or {}).items())),
        "signal": fields.get("signal"),
        "is_batch_independent": bool(fields.get("is_batch_independent", False)),
        "is_compatible_with_append": bool(fields.get("is_compatible_with_append", False)),
        "selection_recipe": fields.get("selection_recipe"),
    }
    if found:
        extras["metadata"] = (
            *extras["metadata"],
            ("executable", False),
            ("placeholders", tuple(found)),
        )
        audit = Edit(name=name, args=args, kwargs=tuple(kwargs.items()), portability=cast(Any, "callable_placeholder"), **extras)  # fmt: skip
        return audit, found
    try:
        rebuilt = rebuild_builtin_helper(name, args, kwargs)
    except Exception as exc:  # the registry's typed refusal names the unknown builtin
        raise _recipe_error(
            root, f"rule {position} built-in edit cannot be rebuilt: {exc}", code=_CORRUPT
        ) from exc
    return replace(rebuilt, **extras), found


def _placeholders_in(value: Any) -> Iterator[CallablePlaceholder]:
    if isinstance(value, CallablePlaceholder):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _placeholders_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _placeholders_in(item)


def _materialize(value: Any, artifact: _Recipe, root: Path) -> Any:
    """Replace the tensor references under ``value`` with tensors, eagerly, through the lazy layer."""

    from .rehydrate import _build_lazy_tensor_ref

    if isinstance(value, BlobRef):
        index = dict(artifact.tensor_entries)
        ref = _build_lazy_tensor_ref(value, index, root, kind="out", payload_hints=None)
        if ref is None:
            raise _recipe_error(root, f"tensor {value.blob_id} has no usable entry", code=_CORRUPT)
        return ref.materialize(
            map_location="cpu", resolved_blobs_dir=resolve_bundle_blobs_dir(root)
        )
    if isinstance(value, list):
        return [_materialize(item, artifact, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, artifact, root) for item in value)
    if isinstance(value, dict):
        return {key: _materialize(item, artifact, root) for key, item in value.items()}
    return value
