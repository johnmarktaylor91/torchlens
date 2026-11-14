"""Save and load a :class:`Recipe` as a ``.tlspec`` directory (manifest kind ``"recipe"``).

A recipe is the AUTHORED intervention: ``tl.when(selector, edit)`` clauses that
have not been lowered against any capture. It is saved on its own, with no
trace, model or fire records, in saved format 3: ``manifest.json`` stamps
``kind: "recipe"``, ``intervention.json`` carries one rule per clause in the
intervention-recipe grammar (the selector and the built-in ``Edit`` identity;
``docs/maintainers/saved_format_3.md``, section 5), and tensor arguments go to
``blobs/recipe-<nnnn>.safetensors``. ``tl.load`` returns a :class:`Recipe`.

Rules with runtime-only WHERE terms (bare predicates, ``tl.where``) and user
callable edits have no portable form: a user callable is never written, it is a
``callable_placeholder``, so such a recipe refuses at the ``"portable"`` level
and is recorded with ``executable`` false at ``"audit"`` and
``"executable_with_callables"``, which ``tl.load`` then refuses typed
(``recipe_audit_only``).
"""

from __future__ import annotations

import os
import shutil
import uuid
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Any, NamedTuple

from .._errors import InvalidArgumentError
from .._io._durability import fsync_tree
from .._io._staging import mark_superseding_save, settle_failed_staging
from .._io.format3.contract import INTERVENTION_FILE
from .._io.format3.payloads import PayloadStore
from .._io.format3.reader import validate_artifact
from .._io.format3.recipe import RecipeHeader, build_recipe, read_recipe, write_recipe
from .._io.format3.writer import EnvelopeSpec, envelope_fields
from .._io.tlspec import _TlSpecWriter
from .._ir.selector_eval import selector_from_spec
from .._selection.selector_base import BaseSelector
from .._selection.selectors import where
from ._rerun_predicate import spec_from_recipe
from ._save import _MANIFEST_FILE, _fsync_directory, _read_json_file, _reject_symlink_path
from ._spec import Recipe, Rule
from ._spec_values import SaveLevel, _coerce_save_level
from .errors import ReplayPreconditionError


class _RuleHook(NamedTuple):
    """One authored rule in the staged-hook shape the recipe grammar encodes."""

    site_target: Any
    helper: Any
    hook: Any
    metadata: dict[str, Any]


class _RecipeRules(NamedTuple):
    """The authored rules of a recipe in the staged-spec shape the grammar reads."""

    hook_specs: tuple[_RuleHook, ...]
    target_value_specs: tuple[Any, ...] = ()


def _rule_hook(rule: Any) -> _RuleHook:
    """Return one rule as its target spec and edit."""

    target = rule.target if isinstance(rule.target, BaseSelector) else where(rule.target)
    return _RuleHook(target.to_target_spec(), rule.edit, rule.edit, {})


def _write_recipe_manifest(
    *,
    path: Path,
    recipe: Recipe,
    document: dict[str, Any],
    envelope: dict[str, Any],
) -> None:
    """Write the unified manifest of a standalone recipe (``kind: "recipe"``).

    The recipe carries no model and no sites, so the manifest's model fields
    describe the recipe class itself; the save level is read from the recipe
    document and the saved format 3 envelope fields are written last.
    """

    save_level = str(document["save_level"])
    if save_level == "runnable":
        raise InvalidArgumentError(
            "Recipe artifacts do not support save_level='runnable': a recipe "
            "carries no model to run",
            code="artifact_save_level_unsupported",
            remedy="use 'audit', 'executable_with_callables', or 'portable'",
            artifact_kind="recipe",
        )
    compat = {
        "format_version": document.get("format_version"),
        "helper_registry_version": document.get("helper_registry_version"),
        "executable": bool(document.get("executable", False)),
        "digest": recipe.digest,
    }
    manifest = _TlSpecWriter.build_manifest(
        kind="recipe",
        source=recipe,
        tensor_entries=[],
        save_level=save_level,
        spec_compat_info=compat,
        intervention_compat_metadata=None,
    )
    manifest.pop("tensor_entries", None)
    manifest.update(envelope)
    _TlSpecWriter.write_json(path, manifest)


def _write_recipe_tree(tmp_path: Path, recipe: Recipe, save_level: SaveLevel) -> None:
    """Fill the temporary ``.tlspec`` directory: recipe, tensors, manifest, fsync."""

    tmp_path.mkdir(parents=True)
    store = PayloadStore(tmp_path)
    rules = _RecipeRules(tuple(_rule_hook(rule) for rule in recipe.rules))
    document, disclosures = build_recipe(
        rules, store=store, header=RecipeHeader(save_level=save_level.value)
    )
    entry = write_recipe(tmp_path, document)
    store.close()
    envelope = envelope_fields(
        tmp_path,
        EnvelopeSpec("recipe", save_level.value, codecs=store.codecs, disclosures=disclosures),
    )
    _write_recipe_manifest(
        path=tmp_path / _MANIFEST_FILE,
        recipe=recipe,
        document=document,
        envelope={**envelope, "intervention": entry},
    )
    fsync_tree(tmp_path)


def _restore_after_failed_save(target_path: Path, backup_path: Path | None) -> None:
    """Put a displaced previous recipe back after a failed save."""

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
        A predicate WHERE term or a user callable edit at ``"portable"``.
    FileExistsError
        The path exists and ``overwrite`` is ``False``.
    """

    save_level = _coerce_save_level(level)
    target_path = Path(path)
    _reject_symlink_path(target_path, context="recipe target")
    tmp_path = target_path.parent / f"{target_path.name}.tmp.{uuid.uuid4().hex}"
    backup_path: Path | None = None
    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Recipe path already exists: {target_path}")
        mark_superseding_save(target_path, overwrite=overwrite)
        _write_recipe_tree(tmp_path, recipe, save_level)
        if target_path.exists():
            if not overwrite:
                raise FileExistsError(f"Recipe path already exists: {target_path}")
            backup_path = target_path.parent / f"{target_path.name}.bak.{uuid.uuid4().hex}"
            os.rename(target_path, backup_path)
        os.rename(tmp_path, target_path)
        _fsync_directory(target_path.parent)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
    except BaseException as exc:
        _restore_after_failed_save(target_path, backup_path)
        settle_failed_staging(tmp_path, target_path, overwrite=overwrite, primary=exc)
        raise


#: Remedy of every malformed-record refusal: the files are rewritten, never repaired.
_MALFORMED_REMEDY = (
    "re-save the recipe with Recipe.save(path); a hand-edited or foreign record is not loadable"
)


def _audit_only_recipe(spec_path: Path, save_level: object) -> ReplayPreconditionError:
    """The typed refusal for a recipe record whose rules cannot be rebuilt."""

    return ReplayPreconditionError(
        f"{spec_path} is an audit-only recipe record (save_level={save_level!r}): a "
        "predicate WHERE term or a user callable edit was recorded as a callable "
        "placeholder and cannot be rebuilt into a Recipe. Read intervention.json for the "
        "record; to make a loadable recipe, express every WHERE term as a selector "
        "(tl.func / tl.in_module / tl.site / tl.label) and every edit as a helper, then "
        "save at level 'portable'.",
        code="recipe_audit_only",
    )


def _require_rule_object(rule_payload: Any, index: int, spec_path: Path) -> None:
    """Refuse a recipe rule that is not an object carrying ``selector`` and ``edit``."""

    if not isinstance(rule_payload, dict) or not {"selector", "edit"} <= set(rule_payload):
        raise ReplayPreconditionError(
            f"{spec_path / INTERVENTION_FILE} recipe rule {index} must be an object with "
            "'selector' and 'edit'",
            code="recipe_record_malformed",
            remedy=_MALFORMED_REMEDY,
        )


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
        Directory containing ``manifest.json``, ``intervention.json`` and ``blobs/``.
    trust_custom_callables, allowed_custom_callable_modules:
        Accepted for the ``tl.load`` model-free door; a saved format 3 recipe never carries a
        callable, so neither changes what loads.

    Returns
    -------
    Recipe
        The recipe with every rule's selector and edit rebuilt.

    Raises
    ------
    ReplayPreconditionError
        ``recipe_audit_only`` for a record holding a callable placeholder (a
        predicate WHERE term or a user callable edit); ``recipe_record_malformed``
        for a malformed rule list, named.
    """

    del trust_custom_callables, allowed_custom_callable_modules
    spec_path = Path(path)
    _reject_symlink_path(spec_path, context="recipe path")
    _reject_symlink_path(spec_path / INTERVENTION_FILE, context="recipe intervention.json")
    _reject_symlink_path(spec_path / _MANIFEST_FILE, context="recipe manifest.json")
    validate_artifact(spec_path, expect_kind="recipe")
    data = _read_json_file(spec_path / INTERVENTION_FILE)
    rules_payload = _recipe_rules_payload(data, spec_path)
    if not bool(data.get("executable", False)):
        raise _audit_only_recipe(spec_path, data.get("save_level"))
    for index, rule_payload in enumerate(rules_payload):
        _require_rule_object(rule_payload, index, spec_path)
    spec = spec_from_recipe(read_recipe(spec_path))
    rules = tuple(
        Rule(
            selector_from_spec(
                hook.site_target.selector_kind,
                hook.site_target.selector_value,
                hook.site_target.metadata,
                lifecycle="capture",
            ),
            hook.helper,
        )
        for hook in spec.hook_specs
    )
    return Recipe(rules=rules)


def _recipe_rules_payload(data: dict[str, Any], spec_path: Path) -> list[Any]:
    """Return the rule list of a recipe ``intervention.json``, refusing malformed shapes."""

    rules = data.get("rules")
    if not isinstance(rules, list):
        raise ReplayPreconditionError(
            f"{spec_path / INTERVENTION_FILE} is missing its 'rules' list; the file is "
            "not a TorchLens recipe or was hand-edited",
            code="recipe_record_malformed",
            remedy=_MALFORMED_REMEDY,
        )
    return list(rules)


__all__ = ["load_recipe", "save_recipe"]
