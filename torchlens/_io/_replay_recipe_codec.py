"""Replay recipe section of a saved non-torch trace: its file, header, parameter values and reader.

A non-torch backend validates a trace by replaying every recorded op from its
parents' saved payloads. At ``level="executable_with_callables"`` a save writes
one data-only replay recipe per op to ``replay_recipes.json`` beside the
records (see :mod:`torchlens._io._replay_recipe_adapter`, which builds and binds
them through the backend's replay adapter); the manifest's ``files`` table
digests it like every other listed file. Array values are never in the section:
parents and constants are named by label and come from the payload blobs the
artifact already holds. A loaded trace keeps no parameter bytes, so the values
an adapter reports through ``live_param_payloads`` are stored as payload blobs
of their own; the section's ``param_payloads`` maps each parameter address to
its blob.

This module owns the document shape and the load side that needs no backend:
reading a verified artifact's section and its parameter values at ``tl.load``.
Nothing is decoded or executed at load. The loaded section lives in a side
table keyed by the loaded trace's identity (released with the trace), so no
attribute is attached to the ``Trace``.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from .._utils._numpy_bridge import tensor_from_numpy, tensor_to_numpy
from . import _json
from .format3.payloads import NUMPY_CODEC, PayloadStore, host_tensor, read_tensor
from .format_errors import ArtifactError

__all__ = (
    "PARAM_FIELD",
    "PARAM_PROBLEM_FIELD",
    "REPLAY_RECIPES_FILE",
    "REPLAY_RECIPE_SAVE_LEVEL",
    "SECTION_FORMAT",
    "SECTION_VERSION",
    "LoadedReplayRecipes",
    "attach_replay_recipes",
    "loaded_section",
    "param_fields",
    "resaved_document",
)

#: File name of the section beside the record tables.
REPLAY_RECIPES_FILE: Final = "replay_recipes.json"
#: Format tag of the section document.
SECTION_FORMAT: Final = "torchlens.replay_recipes"
#: Version of the section document; a reader refuses any other.
SECTION_VERSION: Final = 1
#: The only save level that writes the section (torch's re-execution level).
REPLAY_RECIPE_SAVE_LEVEL: Final = "executable_with_callables"
#: Section field mapping each saved parameter address to its payload blob.
PARAM_FIELD: Final = "param_payloads"
#: Section field naming why the parameter values could not be saved.
PARAM_PROBLEM_FIELD: Final = "param_payload_problem"

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
    params:
        Saved parameter values, ``{parameter address: NumPy array}``.
    """

    document: Mapping[str, Any]
    problem: str | None = None
    bound: bool = field(default=False)
    params: Mapping[str, Any] = field(default_factory=dict)

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


def _store_array(store: PayloadStore, array: np.ndarray) -> dict[str, Any]:
    """Queue one host array as a NumPy-codec payload blob; return its reference."""

    if array.dtype.kind in {"O", "S", "U", "V"}:
        raise TypeError(f"a {array.dtype} parameter cannot be stored as a payload blob")
    tensor = host_tensor(tensor_from_numpy(np.array(array, copy=True, order="C")))
    blob, key = store.submit(tensor, codec=NUMPY_CODEC)
    return {"blob": blob, "key": key, "dtype": array.dtype.str, "shape": list(array.shape)}


def param_fields(values: Any, store: PayloadStore | None) -> dict[str, Any]:
    """Section fields for parameter ``values`` (``{address: array}``), stored in ``store``.

    ``values`` is a mapping, or the exception the adapter raised producing it;
    a failure is recorded as ``param_payload_problem`` and no value is kept,
    so each parameter op then fails closed when the loaded trace validates.
    """

    if isinstance(values, BaseException):
        return {PARAM_PROBLEM_FIELD: f"{type(values).__name__}: {values}"}
    if not values:
        return {}
    if store is None:
        return {PARAM_PROBLEM_FIELD: "the save offered no payload store for parameter values"}
    try:
        refs = {
            str(address): _store_array(store, np.asarray(value))
            for address, value in values.items()
        }
    except (TypeError, ValueError) as exc:
        return {PARAM_PROBLEM_FIELD: f"{type(exc).__name__}: {exc}"}
    return {PARAM_FIELD: refs}


def _header_problem(document: Any) -> str | None:
    """Why a read section document is unusable, else ``None``."""

    if not isinstance(document, Mapping):
        return "the section is not a JSON object"
    if document.get("format") != SECTION_FORMAT:
        return f"the section format is {document.get('format')!r}, not {SECTION_FORMAT!r}"
    if document.get("version") != SECTION_VERSION:
        return f"section version {document.get('version')!r} is not readable (expects 1)"
    for key in ("recipes", "missing", "not_replayable"):
        if not isinstance(document.get(key), list):
            return f"section field {key!r} is not a list"
    if not isinstance(document.get(PARAM_FIELD, {}), Mapping):
        return f"section field {PARAM_FIELD!r} is not an object"
    return None


def _read_params(root: Path, refs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Read every saved parameter value back as a NumPy array of its saved dtype and shape."""

    params: dict[str, np.ndarray] = {}
    for address, ref in refs.items():
        if not isinstance(ref, Mapping):
            raise ValueError(f"parameter {address!r} has no blob reference")
        tensor = read_tensor(root, blob=str(ref["blob"]), key=str(ref["key"]))
        array = np.asarray(tensor_to_numpy(tensor))
        dtype = np.dtype(str(ref["dtype"]))
        if array.dtype != dtype:
            raise ValueError(f"parameter {address!r} reads as {array.dtype}, saved as {dtype}")
        params[str(address)] = array.reshape(tuple(int(n) for n in ref["shape"]))
    return params


def _read_section(root: Path) -> LoadedReplayRecipes:
    """Read and check the section file and its parameter values."""

    try:
        document = _json.read_bounded(root / REPLAY_RECIPES_FILE, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        return LoadedReplayRecipes(document={}, problem=f"the section is unreadable: {exc}")
    problem = _header_problem(document)
    if problem is not None:
        return LoadedReplayRecipes(document={}, problem=problem)
    try:
        params = _read_params(root, document.get(PARAM_FIELD) or {})
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        return LoadedReplayRecipes(
            document={}, problem=f"has unreadable parameter values ({type(exc).__name__}: {exc})"
        )
    return LoadedReplayRecipes(document=document, params=params)


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
    _LOADED_SECTIONS[trace] = _read_section(root)


def loaded_section(trace: Any) -> LoadedReplayRecipes | None:
    """The replay recipe section loaded with ``trace``, else ``None``."""

    return _LOADED_SECTIONS.get(trace)


def resaved_document(section: LoadedReplayRecipes, store: PayloadStore | None) -> Mapping[str, Any]:
    """The section a loaded trace re-saves: as loaded, its parameter values stored again."""

    document = {
        k: v for k, v in section.document.items() if k not in (PARAM_FIELD, PARAM_PROBLEM_FIELD)
    }
    kept = {k: v for k, v in section.document.items() if k == PARAM_PROBLEM_FIELD}
    return {**document, **kept, **param_fields(dict(section.params), store)}
