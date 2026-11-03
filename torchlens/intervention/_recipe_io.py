"""Save and load a :class:`Recipe` as a ``.tlspec`` directory (manifest kind ``"recipe"``).

A recipe is the AUTHORED intervention: ``tl.when(selector, edit)`` clauses that
have not been lowered against any capture. It is saved on its own, with no
trace, model or fire records: ``spec.json`` carries one ``{"target", "edit"}``
payload per rule in the same codec the intervention-spec artifact uses, tensor
arguments go to ``tensors/`` sidecars, and ``manifest.json`` stamps
``kind: "recipe"`` so ``tl.load`` returns a :class:`Recipe`.

Rules with runtime-only WHERE terms (bare predicates, ``tl.where``) and opaque
callable edits have no portable form: they refuse at the ``"portable"`` level
through the shared codec and are recorded as audit-only opaque payloads at
``"audit"``, which ``tl.load`` then refuses typed (``recipe_audit_only``).
"""

from __future__ import annotations

import os
import shutil
import uuid
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Any

from .._errors import InvalidArgumentError
from .._io.manifest import TensorEntry
from .._io.tlspec import _TlSpecWriter
from .._ir.selector_eval import selector_from_spec
from .._selection.selector_base import BaseSelector
from .._selection.selectors import where
from ._spec_codec import (
    _SPEC_FILE,
    TLSPEC_FORMAT_VERSION,
    _target_spec_from_json,
    _target_spec_to_json,
    _validate_format_version,
)
from ._spec_values import (
    SaveLevel,
    _coerce_save_level,
    _deserialize_value,
    _serialize_value,
    _SerializedState,
)
from .errors import ReplayPreconditionError
from .helpers import HELPER_REGISTRY_VERSION
from .save import (
    _MANIFEST_FILE,
    _README_FILE,
    _TENSOR_DIR,
    _collect_helpers,
    _fsync_directory,
    _load_tensor_refs,
    _read_json_file,
    _reject_symlink_path,
    _spec_has_opaque,
    _write_json_file,
    _write_tensor_sidecars,
    _write_text_file,
)
from .spec import Recipe, Rule

#: Name of the rule-list object inside ``spec.json`` for a recipe artifact.
_RECIPE_KEY = "recipe"


def _write_recipe_manifest(
    *,
    path: Path,
    recipe: Recipe,
    spec_json: dict[str, Any],
    tensor_entries: list[TensorEntry],
    legacy_format_version: str,
    save_level: str,
) -> None:
    """Write the unified manifest of a standalone recipe (``kind: "recipe"``).

    The recipe carries no model and no sites, so the manifest's model fields
    describe the recipe class itself; ``save_level`` is the public-schema save
    level and ``legacy_format_version`` the intervention codec version.
    """

    if save_level == "runnable":
        raise InvalidArgumentError(
            "Recipe artifacts do not support save_level='runnable': a recipe "
            "carries no model to run",
            code="artifact_save_level_unsupported",
            remedy="use 'audit', 'executable_with_callables', or 'portable'",
            artifact_kind="recipe",
        )
    compat = {
        "format_version": spec_json.get("format_version"),
        "helper_registry_version": spec_json.get("helper_registry_version"),
        "executable": bool(spec_json.get("executable", False)),
        "digest": spec_json.get("digest"),
    }
    manifest = {
        "format_version": legacy_format_version,
        "tensor_entries": [entry.to_dict() for entry in tensor_entries],
    }
    manifest.update(
        _TlSpecWriter.build_manifest(
            kind="recipe",
            source=recipe,
            tensor_entries=tensor_entries,
            save_level=save_level,
            spec_compat_info=compat,
            intervention_compat_metadata=None,
        )
    )
    _TlSpecWriter.write_json(path, manifest)


def _write_recipe_tree(
    tmp_path: Path,
    recipe: Recipe,
    save_level: SaveLevel,
    state: _SerializedState,
    tensor_entries: list[TensorEntry],
) -> None:
    """Fill the temporary ``.tlspec`` directory: tensors, spec, manifest, README, fsync."""

    tmp_path.mkdir(parents=True)
    (tmp_path / _TENSOR_DIR).mkdir()
    rules = [_serialize_rule(rule, save_level, state) for rule in recipe.rules]
    _write_tensor_sidecars(tmp_path, state.tensor_refs, tensor_entries, write_tensor_blob_fn=None)
    payload = {_RECIPE_KEY: {"rules": rules}}
    spec_json = {
        "format_version": TLSPEC_FORMAT_VERSION,
        "helper_registry_version": HELPER_REGISTRY_VERSION,
        "save_level": save_level.value,
        "executable": save_level != SaveLevel.AUDIT and not _spec_has_opaque(payload),
        "helpers": _collect_helpers(payload),
        "digest": recipe.digest,
        **payload,
    }
    _write_json_file(tmp_path / _SPEC_FILE, spec_json)
    _write_recipe_manifest(
        path=tmp_path / _MANIFEST_FILE,
        recipe=recipe,
        spec_json=spec_json,
        tensor_entries=tensor_entries,
        legacy_format_version=TLSPEC_FORMAT_VERSION,
        save_level=save_level.value,
    )
    _write_text_file(tmp_path / _README_FILE, _readme_text(spec_json, tensor_entries))
    _fsync_directory(tmp_path / _TENSOR_DIR)
    _fsync_directory(tmp_path)


def _restore_after_failed_save(tmp_path: Path, target_path: Path, backup_path: Path | None) -> None:
    """Drop the half-written tree and put a displaced previous recipe back."""

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    if backup_path is not None and not target_path.exists() and backup_path.exists():
        try:
            os.rename(backup_path, target_path)
        except OSError as restore_exc:
            warnings.warn(
                f"Could not restore the previous recipe from its backup after a "
                f"failed save ({restore_exc}). Your prior recipe is NOT lost: it "
                f"remains at {backup_path}. Move it back to {target_path} to recover it.",
                UserWarning,
                stacklevel=3,
            )


def save_recipe(
    recipe: Recipe,
    path: str | Path,
    *,
    level: str | SaveLevel = SaveLevel.PORTABLE,
    overwrite: bool = False,
) -> None:
    """Write ``recipe`` as a ``.tlspec`` directory of manifest kind ``"recipe"``.

    Parameters
    ----------
    recipe:
        The authored recipe (``tl.when(...)`` clauses).
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level: ``"portable"`` (the default), ``"executable_with_callables"``
        or ``"audit"``. ``"runnable"`` is refused: a recipe carries no model.
    overwrite:
        Whether an existing target directory may be replaced.

    Raises
    ------
    OpaqueCallableError
        A predicate WHERE term or an opaque callable edit below ``"audit"``.
    FileExistsError
        The path exists and ``overwrite`` is ``False``.
    """

    save_level = _coerce_save_level(level)
    target_path = Path(path)
    _reject_symlink_path(target_path, context="recipe target")
    tmp_path = target_path.parent / f"tmp.{uuid.uuid4().hex}"
    backup_path: Path | None = None
    tensor_entries: list[TensorEntry] = []
    state = _SerializedState(tensor_entries=tensor_entries, tensor_refs={})
    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Recipe path already exists: {target_path}")
        _write_recipe_tree(tmp_path, recipe, save_level, state, tensor_entries)
        if target_path.exists():
            if not overwrite:
                raise FileExistsError(f"Recipe path already exists: {target_path}")
            backup_path = target_path.parent / f"{target_path.name}.bak.{uuid.uuid4().hex}"
            os.rename(target_path, backup_path)
        os.rename(tmp_path, target_path)
        _fsync_directory(target_path.parent)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
    except BaseException:
        _restore_after_failed_save(tmp_path, target_path, backup_path)
        raise


def _audit_only_recipe(spec_path: Path, save_level: object) -> ReplayPreconditionError:
    """The typed refusal for a recipe record whose payloads cannot be rebuilt."""

    return ReplayPreconditionError(
        f"{spec_path} is an audit-only recipe record (save_level={save_level!r}): a "
        "predicate WHERE term or an opaque callable edit was recorded by repr and cannot "
        "be rebuilt into a Recipe. Read spec.json for the record; to make a loadable "
        "recipe, express every WHERE term as a selector (tl.func / tl.in_module / tl.site "
        "/ tl.label) and every edit as a helper, then save at level 'portable'.",
        code="recipe_audit_only",
    )


def _manifest_tensor_entries(spec_path: Path) -> list[TensorEntry]:
    """Read the manifest's tensor entries, refusing a malformed list with the file named."""

    manifest = _read_json_file(spec_path / _MANIFEST_FILE)
    manifest_entries = manifest.get("tensor_entries", [])
    if not isinstance(manifest_entries, list):
        raise ReplayPreconditionError(
            f"{spec_path / _MANIFEST_FILE} field 'tensor_entries' must be a list, "
            f"got {type(manifest_entries).__name__}"
        )
    return [TensorEntry.from_dict(e) for e in manifest_entries]


def load_recipe(
    path: str | Path,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Recipe:
    """Rebuild a :class:`Recipe` from a ``.tlspec`` directory of kind ``"recipe"``.

    Parameters
    ----------
    path:
        Directory containing ``spec.json``, ``manifest.json`` and ``tensors/``.
    trust_custom_callables:
        Explicit permission to import custom callables recorded in an
        ``"executable_with_callables"`` save when no allowlist is supplied.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names.

    Returns
    -------
    Recipe
        The recipe with every rule's selector and edit rebuilt.

    Raises
    ------
    ReplayPreconditionError
        ``recipe_audit_only`` when the artifact holds audit-only opaque
        payloads (a predicate WHERE term or an opaque edit) that cannot be
        rebuilt; a malformed ``spec.json`` refuses with the file named.
    """

    spec_path = Path(path)
    _reject_symlink_path(spec_path, context="recipe path")
    _reject_symlink_path(spec_path / _SPEC_FILE, context="recipe spec.json")
    _reject_symlink_path(spec_path / _MANIFEST_FILE, context="recipe manifest.json")
    data = _read_json_file(spec_path / _SPEC_FILE)
    _validate_format_version(data.get("format_version"), path=spec_path)
    rules_payload = _recipe_rules_payload(data, spec_path)
    if not bool(data.get("executable", False)):
        raise _audit_only_recipe(spec_path, data.get("save_level"))
    tensors = _load_tensor_refs(spec_path, _manifest_tensor_entries(spec_path))
    rules: list[Rule] = []
    for index, rule_payload in enumerate(rules_payload):
        if not isinstance(rule_payload, dict) or "target" not in rule_payload:
            raise ReplayPreconditionError(
                f"{spec_path / _SPEC_FILE} recipe rule {index} must be an object with "
                "'target' and 'edit'"
            )
        target = _target_spec_from_json(rule_payload["target"])
        selector = selector_from_spec(
            target.selector_kind, target.selector_value, target.metadata, lifecycle="capture"
        )
        edit = _deserialize_value(
            rule_payload.get("edit"),
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
        rules.append(Rule(selector, edit))
    return Recipe(rules=tuple(rules))


def _serialize_rule(rule: Any, save_level: SaveLevel, state: _SerializedState) -> dict[str, Any]:
    """Serialize one rule as its target spec and edit payload."""

    target = rule.target if isinstance(rule.target, BaseSelector) else where(rule.target)
    return {
        "rule_id": rule.rule_id,
        "target": _target_spec_to_json(target.to_target_spec(), save_level),
        "edit": _serialize_value(rule.edit, save_level, state),
    }


def _recipe_rules_payload(data: dict[str, Any], spec_path: Path) -> list[Any]:
    """Return the rule list of a recipe ``spec.json``, refusing malformed shapes."""

    recipe_payload = data.get(_RECIPE_KEY)
    if not isinstance(recipe_payload, dict) or not isinstance(recipe_payload.get("rules"), list):
        raise ReplayPreconditionError(
            f"{spec_path / _SPEC_FILE} is missing its 'recipe.rules' list; the file is "
            "not a TorchLens recipe or was hand-edited"
        )
    return list(recipe_payload["rules"])


def _readme_text(spec_json: dict[str, Any], tensor_entries: list[TensorEntry]) -> str:
    """Render the human-readable README for a recipe artifact."""

    rules = spec_json[_RECIPE_KEY]["rules"]
    lines = [
        "# TorchLens recipe",
        "",
        f"Format version: {spec_json['format_version']}",
        f"Save level: {spec_json['save_level']}",
        f"Executable: {spec_json['executable']}",
        f"Rules: {len(rules)}",
        f"Tensor sidecars: {len(tensor_entries)}",
        "",
        "Load with `torchlens.load(path)`; the result is a `torchlens.Recipe`.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["load_recipe", "save_recipe"]
