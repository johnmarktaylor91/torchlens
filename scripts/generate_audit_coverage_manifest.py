"""Generate the TorchLens Total Audit coverage manifest."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import torchlens
from torchlens import constants

MANIFEST_PATH = Path("notebooks/total_audit/_coverage_manifest.json")

LOG_TYPE_NAMES = [
    "ModelLog",
    "LayerLog",
    "LayerPassLog",
    "ModuleLog",
    "ModulePassLog",
    "ParamLog",
    "BufferLog",
    "GradFnLog",
    "GradFnPassLog",
    "Bundle",
]

FIELD_ORDER_BY_CLASS = {
    "ModelLog": constants.MODEL_LOG_FIELD_ORDER,
    "LayerLog": constants.LAYER_LOG_FIELD_ORDER,
    "LayerPassLog": constants.LAYER_PASS_LOG_FIELD_ORDER,
    "ModuleLog": constants.MODULE_LOG_FIELD_ORDER,
    "ModulePassLog": constants.MODULE_PASS_LOG_FIELD_ORDER,
    "ParamLog": constants.PARAM_LOG_FIELD_ORDER,
    "BufferLog": constants.BUFFER_LOG_FIELD_ORDER,
    "GradFnLog": constants.GRAD_FN_LOG_FIELD_ORDER,
    "GradFnPassLog": constants.GRAD_FN_PASS_LOG_FIELD_ORDER,
}

WRAPPER_COMPAT_ALIASES = [
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_saved_activations",
    "summary",
    "show_model_graph",
    "show_backward_graph",
    "load_intervention_spec",
]


def _kind_for_object(obj: Any) -> str:
    """Return a stable manifest kind for a Python object.

    Parameters
    ----------
    obj:
        Object being inventoried.

    Returns
    -------
    str
        Human-readable object kind.
    """

    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if inspect.ismethod(obj):
        return "method"
    return type(obj).__name__


def _top_level_entries() -> list[dict[str, Any]]:
    """Inventory names exported from ``torchlens.__all__``.

    Returns
    -------
    list[dict[str, Any]]
        Manifest records for top-level exported names.
    """

    entries: list[dict[str, Any]] = []
    for name in sorted(torchlens.__all__):
        obj = getattr(torchlens, name)
        entries.append(
            {
                "name": name,
                "kind": _kind_for_object(obj),
                "qualname": getattr(obj, "__qualname__", None),
                "module": getattr(obj, "__module__", None),
            }
        )
    return entries


def _iter_immediate_submodules() -> Iterable[str]:
    """Yield importable immediate ``torchlens.X`` submodule names.

    Yields
    ------
    str
        Public immediate child module name.
    """

    for module_info in pkgutil.iter_modules(torchlens.__path__):
        if not module_info.name.startswith("_"):
            yield module_info.name


def _public_names_from_module(module_name: str, module: ModuleType) -> tuple[list[str], str]:
    """Return public names from a submodule, preferring ``__all__``.

    Parameters
    ----------
    module_name:
        Fully qualified module name.
    module:
        Imported module object.

    Returns
    -------
    tuple[list[str], str]
        Public names and the source used to discover them.
    """

    if hasattr(module, "__all__"):
        return sorted(str(name) for name in module.__all__), "__all__"
    warnings.warn(
        f"{module_name} lacks __all__; manifest falls back to dir() and may miss lazy names.",
        stacklevel=2,
    )
    return sorted(name for name in dir(module) if not name.startswith("_")), "dir"


def _submodule_entries() -> dict[str, dict[str, Any]]:
    """Inventory every immediate ``torchlens.X`` submodule.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of submodule name to discovery metadata and public names.
    """

    submodules: dict[str, dict[str, Any]] = {}
    for submodule_name in sorted(_iter_immediate_submodules()):
        module_name = f"torchlens.{submodule_name}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional extras vary by env
            warnings.warn(f"Could not import {module_name}: {exc!r}", stacklevel=2)
            submodules[submodule_name] = {
                "names": [],
                "source": "import_error",
                "warning": repr(exc),
            }
            continue
        names, source = _public_names_from_module(module_name, module)
        submodules[submodule_name] = {"names": names, "source": source}
    return submodules


def _resolve_log_type(name: str) -> type[Any] | None:
    """Resolve a log type from the top-level module or ``torchlens.types``.

    Parameters
    ----------
    name:
        Class name to resolve.

    Returns
    -------
    type[Any] | None
        Resolved class object, or ``None`` if unavailable.
    """

    types_module = importlib.import_module("torchlens.types")
    candidate = getattr(types_module, name, None)
    if inspect.isclass(candidate):
        return candidate
    if name in torchlens.__all__:
        candidate = getattr(torchlens, name, None)
        if inspect.isclass(candidate):
            return candidate
    return None


def _member_kind(cls: type[Any], name: str, member: Any) -> str:
    """Classify a class member without losing descriptor type information.

    Parameters
    ----------
    cls:
        Class that owns or inherits the member.
    name:
        Member name.
    member:
        Value returned by ``inspect.getmembers``.

    Returns
    -------
    str
        Manifest kind for the member.
    """

    try:
        raw_member = inspect.getattr_static(cls, name)
    except AttributeError:
        raw_member = member
    if isinstance(raw_member, property):
        return "property"
    if isinstance(raw_member, classmethod):
        return "classmethod"
    if isinstance(raw_member, staticmethod):
        return "staticmethod"
    if inspect.isfunction(member) or inspect.ismethod(member):
        return "method"
    return "attribute"


def _class_entries() -> dict[str, list[dict[str, str]]]:
    """Inventory public members and FIELD_ORDER fields for audited log types.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Mapping of class name to member records.
    """

    classes: dict[str, list[dict[str, str]]] = {}
    for class_name in LOG_TYPE_NAMES:
        cls = _resolve_log_type(class_name)
        if cls is None:
            warnings.warn(f"Could not resolve audited log type {class_name}.", stacklevel=2)
            classes[class_name] = []
            continue

        members: dict[str, dict[str, str]] = {}
        for name, member in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            members[name] = {"name": name, "kind": _member_kind(cls, name, member)}

        for field_name in FIELD_ORDER_BY_CLASS.get(class_name, []):
            members.setdefault(str(field_name), {"name": str(field_name), "kind": "field"})

        classes[class_name] = sorted(members.values(), key=lambda entry: entry["name"])
    return classes


def _compat_alias_entries() -> list[dict[str, str]]:
    """Inventory deprecated top-level aliases protected from silent deletion.

    Returns
    -------
    list[dict[str, str]]
        Compatibility alias records.
    """

    moved_objects = getattr(torchlens, "_MOVED_OBJECTS", {})
    aliases = [
        {"name": name, "target": f"{target_module}.{target_name}", "kind": "object_alias"}
        for name, (target_module, target_name) in sorted(moved_objects.items())
    ]
    aliases.extend(
        {
            "name": name,
            "target": getattr(getattr(torchlens, name), "__module__", "unknown"),
            "kind": "wrapper_function",
        }
        for name in WRAPPER_COMPAT_ALIASES
        if hasattr(torchlens, name)
    )
    return sorted(aliases, key=lambda entry: entry["name"])


def build_manifest() -> dict[str, Any]:
    """Build the full Total Audit coverage manifest.

    Returns
    -------
    dict[str, Any]
        Manifest payload.
    """

    return {
        "schema_version": 1,
        "top_level": _top_level_entries(),
        "submodules": _submodule_entries(),
        "classes": _class_entries(),
        "compat_aliases": _compat_alias_entries(),
    }


def main() -> None:
    """Generate ``notebooks/total_audit/_coverage_manifest.json``."""

    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    public_count = (
        len(manifest["top_level"])
        + sum(len(entry["names"]) for entry in manifest["submodules"].values())
        + sum(len(entries) for entries in manifest["classes"].values())
    )
    print(f"Wrote {MANIFEST_PATH} with {public_count} public items.")
    print(f"compat_aliases: {len(manifest['compat_aliases'])}")


if __name__ == "__main__":
    main()
