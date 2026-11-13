"""Replay recipes built and bound through the backend's replay adapter, around a save and a load.

At ``level="executable_with_callables"`` a save of a non-torch trace asks the
backend's replay adapter for one data-only
:class:`~torchlens.backends._replay_adapters.OpRecipe` per op and writes them,
with the parameter values the adapter reports, as the section described in
:mod:`torchlens._io._replay_recipe_codec`.

On the first read of ``validation_replay_status`` after ``tl.load``, the
recipes are decoded against the backend adapter's closed op allowlist and
handed to the adapter's ``bind_recipes`` (with the saved parameter values as
its ``param_payloads`` keyword), which rebuilds the capture list; the
backend's ordinary validator then runs. A saved verdict is never restored.
An op the live validator could replay but that has no recipe makes the loaded
status ``unverified`` naming that op; a framework version different from the
one that saved the trace, an unreadable section, or a missing adapter makes it
``unavailable`` with the cause in the message. A bundle saved without the
section keeps the stripped ``unavailable`` status.

The module sits with the other adapter-facing ``_io`` modules (save gates,
runnable binding) above the backends layer: its callers are the save body and
validation, both on that layer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from .._validation_status import ValidationStatus
from ..backends._replay_adapters import (
    ADAPTER_BACKENDS,
    RECIPE_FORMAT_VERSION,
    OpRecipe,
    ReplayRefusal,
    decode_recipe,
    encode_recipe,
    get_adapter,
    validate_recipe,
)
from ._replay_recipe_codec import (
    REPLAY_RECIPE_SAVE_LEVEL,
    REPLAY_RECIPES_FILE,
    SECTION_FORMAT,
    SECTION_VERSION,
    LoadedReplayRecipes,
    loaded_section,
    param_fields,
    resaved_document,
)
from .format3.payloads import PayloadStore

#: Layer of this module: it imports the backends layer and is imported only by
#: the save body (``_trace_sidecars`` under ``bundle``) and validation.
__tl_layer__ = "L3"

__all__ = (
    "build_section",
    "installed_framework_version",
    "loaded_replay_recipe_status",
    "write_replay_recipes",
)

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

    # A recipe is saved whenever the adapter gives one, also for an op the
    # replay session refuses: the backend validator judges that op after load
    # exactly as it did live. Without a recipe, a refused op is recorded as not
    # replayable and any other op as missing. Complexity: adapter code runs
    # inside a save; any failure becomes this op's recorded cause, named after load.
    try:
        recipe = adapter.recipe(trace, op)
        if recipe is not None:
            validate_recipe(recipe)
            return "recipe", recipe
        cause = "the adapter returned no recipe for a replayable op"
    except Exception as exc:  # noqa: BLE001 - recorded in the section and named after load
        cause = f"{type(exc).__name__}: {exc}"
    try:
        reason = adapter.can_replay(trace, op)
    except Exception as exc:  # noqa: BLE001 - recorded in the section and named after load
        return "missing", f"{cause}; can_replay raised {type(exc).__name__}: {exc}"
    return ("missing", cause) if reason is None else ("not_replayable", str(reason))


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


def _live_params(trace: Any, adapter: Any) -> Any:
    """The adapter's ``{address: array}`` for ``trace``, or the exception it raised."""

    live = getattr(adapter, "live_param_payloads", None)
    if live is None:
        return {}
    # Complexity: adapter code runs inside a save; a failure is recorded in the
    # section and its parameter ops fail closed after load.
    try:
        return dict(live(trace))
    except Exception as exc:  # noqa: BLE001 - recorded in the section
        return exc


def _section_for_save(
    trace: Any, backend: str, store: PayloadStore | None
) -> Mapping[str, Any] | None:
    """The section to write for ``trace``, or ``None`` when none applies."""

    if bool(getattr(trace, "_loaded_from_bundle", False)):
        # A loaded trace re-saves the section it was loaded with.
        section = loaded_section(trace)
        if section is None or section.problem:
            return None
        return resaved_document(section, store)
    try:
        adapter = get_adapter(backend)
    except ReplayRefusal:
        return None
    document = build_section(trace, adapter, backend)
    return {**document, **param_fields(_live_params(trace, adapter), store)}


def write_replay_recipes(
    root: Path,
    trace: Any,
    *,
    backend: str,
    save_level: str,
    store: PayloadStore | None = None,
) -> bool:
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
    store:
        The artifact's payload store, for parameter values the adapter
        reports (before the store closes).

    Returns
    -------
    bool
        Whether the file was written (``False`` for torch, other levels, or a
        backend without a replay adapter).
    """

    if save_level != REPLAY_RECIPE_SAVE_LEVEL or backend not in ADAPTER_BACKENDS:
        return False
    document = _section_for_save(trace, backend, store)
    if document is None:
        return False
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (root / REPLAY_RECIPES_FILE).write_text(text, encoding="utf-8")
    return True


def _allowlist(adapter: Any) -> frozenset[str] | None:
    """The adapter's closed op allowlist (the contract's module-level ``ALLOWLIST``)."""

    allowlist = getattr(adapter, "ALLOWLIST", None)
    return None if allowlist is None else frozenset(allowlist)


def _decode_all(
    encoded: Sequence[Any], *, backend: str, allowlist: frozenset[str]
) -> dict[str, OpRecipe]:
    """Decode every recipe, keyed by its first output slot; refuses on the first bad one."""

    # decode_recipe refuses a recipe without output slots, so slot 0 exists.
    recipes: dict[str, OpRecipe] = {}
    for data in encoded:
        recipe = decode_recipe(data, backend=backend, allowlist=allowlist)
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
    except ReplayRefusal as exc:
        return _unavailable(trace, backend, f"no replay adapter is installed ({exc}).")
    allowlist = _allowlist(adapter)
    if allowlist is None:
        return _unavailable(trace, backend, "the replay adapter declares no op allowlist.")
    try:
        recipes = _decode_all(section.document["recipes"], backend=backend, allowlist=allowlist)
    except ReplayRefusal as exc:
        return _unavailable(trace, backend, f"a saved replay recipe was refused: {exc}.")
    # Complexity: bind_recipes is adapter code reading loaded payloads; a raise
    # leaves no capture list, so the trace reports unavailable with the cause.
    # Parameter values go as a keyword only when there are any, so adapters
    # whose recipes read no parameter keep the two-argument contract.
    params = {"param_payloads": dict(section.params)} if section.params else {}
    try:
        adapter.bind_recipes(trace, recipes, **params)
    except Exception as exc:  # noqa: BLE001 - reported in the status message, never swallowed
        return _unavailable(
            trace, backend, f"binding the saved replay recipes raised {type(exc).__name__}: {exc}."
        )
    section.bound = True
    return None


def _section_problem(section: LoadedReplayRecipes, backend: str) -> str | None:
    """Why a loaded section cannot be bound on this runtime, else ``None``."""

    if section.problem is not None:
        return f"its replay recipe section {section.problem}."
    if section.backend != backend:
        return f"its replay recipe section was saved for {section.backend!r}."
    recipe_format = section.document.get("recipe_format_version")
    if recipe_format != RECIPE_FORMAT_VERSION:
        return (
            f"its replay recipe section has recipe format {recipe_format!r}, which is not readable."
        )
    installed = installed_framework_version(backend)
    if section.framework_version != installed:
        return (
            f"it was saved with {backend} {section.framework_version or '(unknown)'} and "
            f"this runtime has {installed or '(not installed)'}; replay runs only on the "
            "framework version that saved it."
        )
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
    problem = _section_problem(section, backend)
    if problem is not None:
        return _unavailable(trace, backend, problem)
    if section.missing:
        return _unverified_missing(trace, section)
    return _bind(trace, section)
