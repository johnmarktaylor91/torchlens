"""Replay recipes saved with a non-torch trace, so validation after load re-executes.

A non-torch backend validates a trace by replaying every recorded op from its
parents' saved payloads. The runtime capture list that replay reads holds live
framework handles, so a save strips it and a loaded trace used to report
validation ``unavailable``. At ``level="executable_with_callables"`` (the level
torch already uses to ship re-execution payloads) the save now also writes one
data-only :class:`~torchlens.backends._replay_adapters.OpRecipe` per op to
``replay_recipes.json`` beside the records; the manifest's ``files`` table
digests it like every other listed file. Array values are never in the section:
parents, parameters and constants are named by label and come from the payload
blobs the artifact already holds.

On the first read of ``validation_replay_status`` after ``tl.load``, the
recipes are decoded against the backend adapter's closed op allowlist and
handed to the adapter's ``bind_recipes``, which rebuilds the capture list; the
backend's ordinary validator then runs. A saved verdict is never restored.
An op the live validator could replay but that has no recipe makes the loaded
status ``unverified`` naming that op; a framework version different from the
one that saved the trace, an unreadable section, or a missing adapter makes it
``unavailable`` with the cause in the message. A bundle saved without the
section keeps the stripped ``unavailable`` status.

The loaded section lives in a side table keyed by the loaded trace's identity
(released with the trace), so no attribute is attached to the ``Trace``.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from .._validation_status import ValidationStatus
from ..backends._replay_adapters import (
    ADAPTER_BACKENDS,
    RECIPE_FORMAT_VERSION,
    OpRecipe,
    ReplayAdapterUnavailable,
    ReplayRecipeError,
    decode_recipe,
    encode_recipe,
    get_adapter,
    validate_recipe,
)
from . import _json

__all__ = (
    "REPLAY_RECIPES_FILE",
    "REPLAY_RECIPE_SAVE_LEVEL",
    "SECTION_FORMAT",
    "SECTION_VERSION",
    "LoadedReplayRecipes",
    "attach_replay_recipes",
    "build_section",
    "installed_framework_version",
    "loaded_replay_recipe_status",
    "loaded_section",
    "write_replay_recipes",
)

#: File name of the section beside the record tables.
REPLAY_RECIPES_FILE: Final = "replay_recipes.json"
#: Format tag of the section document.
SECTION_FORMAT: Final = "torchlens.replay_recipes"
#: Version of the section document; a reader refuses any other.
SECTION_VERSION: Final = 1
#: The only save level that writes the section (torch's re-execution level).
REPLAY_RECIPE_SAVE_LEVEL: Final = "executable_with_callables"
#: Status reason of the loaded unavailable statuses (the existing stripped code).
_STRIPPED_REASON: Final = "loaded_trace_runtime_capture_stripped"
#: Per-op reason of an op that has no recipe (the existing registered code).
_NO_RECIPE_REASON: Final = "no_replayable_function"
#: Distributions whose installed version identifies each framework runtime.
_FRAMEWORK_DISTRIBUTIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "jax": ("jax",),
    "tf": ("tensorflow", "tensorflow-cpu", "tensorflow-macos", "tf-nightly"),
    "mlx": ("mlx",),
    "paddle": ("paddlepaddle", "paddlepaddle-gpu"),
    "tinygrad": ("tinygrad",),
}

#: Loaded sections keyed weakly by their trace; a row dies with its trace.
_LOADED_SECTIONS: weakref.WeakKeyDictionary[Any, LoadedReplayRecipes] = weakref.WeakKeyDictionary()


@dataclass
class LoadedReplayRecipes:
    """The section read from a loaded artifact, before and after binding.

    Attributes
    ----------
    document:
        The section document as read (kept for a re-save of the loaded trace).
    problem:
        Why the section cannot be used (unreadable, wrong format or version),
        else ``None``.
    bound:
        Whether ``bind_recipes`` already rebuilt the capture list.
    """

    document: Mapping[str, Any]
    problem: str | None = None
    bound: bool = field(default=False)

    @property
    def backend(self) -> str:
        """Backend the section was saved for."""

        return str(self.document.get("backend", ""))

    @property
    def framework_version(self) -> str:
        """Framework runtime version at save."""

        return str(self.document.get("framework_version", ""))

    @property
    def missing(self) -> tuple[Mapping[str, Any], ...]:
        """Replayable ops the save could not describe, with the cause."""

        return tuple(self.document.get("missing") or ())


def installed_framework_version(backend: str) -> str:
    """Installed version of ``backend``'s framework distribution, else ``""``.

    Read from package metadata, so the framework is never imported. The saved
    and installed strings are compared for equality only.
    """

    for distribution in _FRAMEWORK_DISTRIBUTIONS.get(backend, ()):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return ""


def _op_label(op: Any) -> str:
    """The op's label (its first output slot)."""

    label = getattr(op, "layer_label", None)
    return str(label if label is not None else getattr(op, "label", ""))


def _describe_op(trace: Any, op: Any, adapter: Any) -> tuple[str, Any]:
    """``("recipe", OpRecipe)``, ``("not_replayable", reason)`` or ``("missing", cause)``."""

    # Complexity: an adapter is framework code run inside a save; any failure
    # becomes this op's recorded cause and is reported by name after load.
    try:
        reason = adapter.can_replay(trace, op)
        if reason is not None:
            return "not_replayable", str(reason)
        recipe = adapter.recipe(trace, op)
        if recipe is None:
            return "missing", "the adapter returned no recipe for a replayable op"
        validate_recipe(recipe)
        return "recipe", recipe
    except Exception as exc:  # noqa: BLE001 - recorded in the section and named after load
        return "missing", f"{type(exc).__name__}: {exc}"


def build_section(trace: Any, adapter: Any, backend: str) -> dict[str, Any]:
    """Describe every op of a live trace through ``adapter``.

    Parameters
    ----------
    trace:
        Live trace of ``backend``.
    adapter:
        The backend's replay adapter module.
    backend:
        Backend name.

    Returns
    -------
    dict[str, Any]
        The JSON-safe section document: header, encoded recipes in graph
        order, replayable ops without a recipe (``missing``) and ops the live
        validator does not replay either (``not_replayable``).
    """

    recipes: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    not_replayable: list[dict[str, str]] = []
    covered: set[str] = set()
    for op in tuple(getattr(trace, "_layer_list", ()) or ()):
        label = _op_label(op)
        if label in covered:
            continue
        kind, value = _describe_op(trace, op, adapter)
        if kind == "recipe":
            recipes.append(encode_recipe(value))
            covered.update(value.output_slots or (label,))
        elif kind == "missing":
            missing.append({"op_label": label, "detail": value})
        else:
            not_replayable.append({"op_label": label, "detail": value})
    return {
        "format": SECTION_FORMAT,
        "version": SECTION_VERSION,
        "backend": backend,
        "framework_version": installed_framework_version(backend),
        "recipe_format_version": RECIPE_FORMAT_VERSION,
        "recipes": recipes,
        "missing": missing,
        "not_replayable": not_replayable,
    }


def _section_for_save(trace: Any, backend: str) -> Mapping[str, Any] | None:
    """The section to write for ``trace``, or ``None`` when none applies."""

    if bool(getattr(trace, "_loaded_from_bundle", False)):
        # A loaded trace re-saves the section it was loaded with, unchanged.
        section = loaded_section(trace)
        return None if section is None or section.problem else section.document
    try:
        adapter = get_adapter(backend)
    except ReplayAdapterUnavailable:
        return None
    return build_section(trace, adapter, backend)


def write_replay_recipes(root: Path, trace: Any, *, backend: str, save_level: str) -> bool:
    """Write ``replay_recipes.json`` under a staged trace artifact when it applies.

    Parameters
    ----------
    root:
        Staged artifact directory (before the manifest digests its files).
    trace:
        The trace being saved.
    backend:
        The trace's backend name.
    save_level:
        Resolved save level; only ``executable_with_callables`` writes.

    Returns
    -------
    bool
        Whether the file was written (``False`` for torch, other levels, or a
        backend without a replay adapter).
    """

    if save_level != REPLAY_RECIPE_SAVE_LEVEL or backend not in ADAPTER_BACKENDS:
        return False
    document = _section_for_save(trace, backend)
    if document is None:
        return False
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (root / REPLAY_RECIPES_FILE).write_text(text, encoding="utf-8")
    return True


def _header_problem(document: Any) -> str | None:
    """Why a read section document is unusable, else ``None``."""

    if not isinstance(document, Mapping):
        return "the section is not a JSON object"
    if document.get("format") != SECTION_FORMAT:
        return f"the section format is {document.get('format')!r}, not {SECTION_FORMAT!r}"
    if document.get("version") != SECTION_VERSION:
        return f"section version {document.get('version')!r} is not readable (expects 1)"
    if document.get("recipe_format_version") != RECIPE_FORMAT_VERSION:
        return f"recipe format {document.get('recipe_format_version')!r} is not readable"
    for key in ("recipes", "missing", "not_replayable"):
        if not isinstance(document.get(key), list):
            return f"section field {key!r} is not a list"
    return None


def attach_replay_recipes(trace: Any, root: Path, manifest: Mapping[str, Any]) -> None:
    """Remember a loaded artifact's replay recipe section for ``trace``.

    Only a file the manifest lists (and the reader already digest-checked) is
    read. Nothing is decoded or executed here; binding happens on the first
    validation-status read.

    Parameters
    ----------
    trace:
        The loaded trace.
    root:
        Artifact directory.
    manifest:
        The verified manifest.
    """

    files = manifest.get("files")
    if not isinstance(files, Mapping) or REPLAY_RECIPES_FILE not in files:
        return
    try:
        document = _json.read_bounded(root / REPLAY_RECIPES_FILE, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        section = LoadedReplayRecipes(document={}, problem=f"the section is unreadable: {exc}")
    else:
        problem = _header_problem(document)
        section = LoadedReplayRecipes(document=document if problem is None else {}, problem=problem)
    _LOADED_SECTIONS[trace] = section


def loaded_section(trace: Any) -> LoadedReplayRecipes | None:
    """The replay recipe section loaded with ``trace``, else ``None``."""

    return _LOADED_SECTIONS.get(trace)


def _allowlist(adapter: Any) -> frozenset[str] | None:
    """The adapter's closed op allowlist (``ALLOWLIST`` or ``op_allowlist()``)."""

    allowlist = getattr(adapter, "ALLOWLIST", None)
    if allowlist is None and callable(getattr(adapter, "op_allowlist", None)):
        allowlist = adapter.op_allowlist()
    return None if allowlist is None else frozenset(allowlist)


def _decode_all(
    encoded: Sequence[Any], *, backend: str, allowlist: frozenset[str]
) -> dict[str, OpRecipe]:
    """Decode every recipe, keyed by its first output slot; refuses on the first bad one."""

    recipes: dict[str, OpRecipe] = {}
    for position, data in enumerate(encoded):
        recipe = decode_recipe(data, backend=backend, allowlist=allowlist)
        if not recipe.output_slots:
            raise ReplayRecipeError("slots", f"recipe {position} has no output slot")
        recipes[recipe.output_slots[0]] = recipe
    return recipes


def _unavailable(trace: Any, backend: str, message: str) -> ValidationStatus:
    """Loaded unavailable status naming why the recipes cannot run."""

    return ValidationStatus(
        state="unavailable",
        is_available=False,
        backend=backend,
        source="loaded",
        reason=_STRIPPED_REASON,
        message=f"Replay validation is unavailable for this loaded {backend} trace: {message}",
        payload_load_status=getattr(trace, "payload_load_status", None),
    )


def _unverified_missing(trace: Any, section: LoadedReplayRecipes) -> ValidationStatus:
    """Loaded unverified status naming every replayable op that has no recipe."""

    backend = section.backend
    rows = section.missing
    labels = [str(row.get("op_label")) for row in rows]
    shown = ", ".join(labels[:8]) + (f" (+{len(labels) - 8} more)" if len(labels) > 8 else "")
    decisions = tuple(
        {
            "op_label": str(row.get("op_label")),
            "func_name": None,
            "phase": "load",
            "decision": "unverified",
            "reason": _NO_RECIPE_REASON,
            "detail": f"no replay recipe was saved for this op: {row.get('detail')}",
        }
        for row in rows
    )
    return ValidationStatus(
        state="unverified",
        is_available=False,
        backend=backend,
        source="loaded",
        reason=_NO_RECIPE_REASON,
        message=(
            f"Replay validation of this loaded {backend} trace did not run: "
            f"{len(labels)} replayable op(s) have no saved replay recipe: {shown}."
        ),
        payload_load_status=getattr(trace, "payload_load_status", None),
        num_unverified_ops=len(labels),
        num_pure_unverified_ops=len(labels),
        unverified_reason_counts={_NO_RECIPE_REASON: len(labels)},
        decisions=decisions,
    )


def _bind(trace: Any, section: LoadedReplayRecipes) -> ValidationStatus | None:
    """Decode and bind the section; ``None`` on success, else the unavailable status."""

    backend = section.backend
    try:
        adapter = get_adapter(backend)
    except ReplayAdapterUnavailable as exc:
        return _unavailable(trace, backend, f"no replay adapter is installed ({exc}).")
    allowlist = _allowlist(adapter)
    if allowlist is None:
        return _unavailable(trace, backend, "the replay adapter declares no op allowlist.")
    try:
        recipes = _decode_all(section.document["recipes"], backend=backend, allowlist=allowlist)
    except ReplayRecipeError as exc:
        return _unavailable(trace, backend, f"a saved replay recipe was refused: {exc}.")
    # Complexity: bind_recipes is adapter code reading loaded payloads; a raise
    # leaves no capture list, so the trace reports unavailable with the cause.
    try:
        adapter.bind_recipes(trace, recipes)
    except Exception as exc:  # noqa: BLE001 - reported in the status message, never swallowed
        return _unavailable(
            trace, backend, f"binding the saved replay recipes raised {type(exc).__name__}: {exc}."
        )
    section.bound = True
    return None


def loaded_replay_recipe_status(trace: Any) -> ValidationStatus | None:
    """Bind a loaded trace's replay recipes, or return why validation cannot run.

    Called where a loaded non-torch trace used to report the stripped status.

    Parameters
    ----------
    trace:
        Trace whose validation status is being read.

    Returns
    -------
    ValidationStatus or None
        ``None`` when there is no section (the old behaviour applies) or the
        recipes are bound and the backend's ordinary validator may run; else
        the ``unverified`` or ``unavailable`` status that names the cause.
    """

    section = loaded_section(trace)
    if section is None or section.bound:
        return None
    backend = str(getattr(trace, "_provenance_backend", ""))
    if section.problem is not None:
        return _unavailable(trace, backend, f"its replay recipe section {section.problem}.")
    if section.backend != backend:
        return _unavailable(
            trace, backend, f"its replay recipe section was saved for {section.backend!r}."
        )
    installed = installed_framework_version(backend)
    if section.framework_version != installed:
        return _unavailable(
            trace,
            backend,
            f"it was saved with {backend} {section.framework_version or '(unknown)'} and "
            f"this runtime has {installed or '(not installed)'}; replay runs only on the "
            "framework version that saved it.",
        )
    if section.missing:
        return _unverified_missing(trace, section)
    return _bind(trace, section)
