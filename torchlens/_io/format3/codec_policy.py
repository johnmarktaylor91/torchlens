"""Codec role policy for format 3 tensor cells (contract: payload backend boundary).

An artifact's payload columns (``out``, ``grad`` and the other PAYLOAD storage columns)
take one codec: the torch codec for a torch artifact, the numeric array codec for every
other backend (its tensors materialize as arrays). The numeric array codec also carries
raw values (the trace input and other value cells) in any artifact, and outside payload
columns of a non-torch artifact a tensor of that artifact's own backend when its
``logical_backend`` hint names it (derived gradients). A codec hint that
disagrees with the cell's backend is a typed ``record_schema_invalid`` refusal raised
before any codec is looked up. A backend outside the contract set is admitted only while it
is registered in this runtime, and its registered capabilities say whether its payloads
materialize.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache
from typing import Any

from ..._data_classes.field_policy import StorageKind
from . import contract
from .decode import CELL_CODE, cell_error
from .schema import record_schema

__all__ = [
    "backend_hint_fields",
    "backend_materializes",
    "codec_backend",
    "known_backend",
    "payload_codec",
    "payload_columns",
]

_TORCH_CODEC, _ARRAY_CODEC = contract.PAYLOAD_CODEC_IDS


@cache
def payload_columns(kind: str) -> frozenset[str]:
    """Columns of a record kind that hold tensor payloads (``out``, ``grad``, saved arguments)."""

    try:
        schema = record_schema(kind)
    except (KeyError, ValueError):
        return frozenset()
    return frozenset(n for n, c in schema.items() if c.storage == StorageKind.PAYLOAD.value)


def payload_codec(manifest_backend: str) -> str:
    """The one codec the payload cells of a ``manifest_backend`` artifact carry."""

    return _TORCH_CODEC if manifest_backend == contract.DEFAULT_BACKEND else _ARRAY_CODEC


def _cell_role(at: str) -> tuple[str, str]:
    """``(record kind, column)`` of a cell path ``kind[row].column...``."""

    kind, _, rest = at.partition("[")
    column = rest.split("].", 1)[1] if "]." in rest else ""
    return kind, re.split(r"[.\[{<]", column, maxsplit=1)[0]


def codec_backend(
    codec_id: str, hints: Mapping[str, Any] | None, at: str, *, manifest_backend: str
) -> str:
    """The backend a tensor cell belongs to, checked against the cell's role and the
    artifact's backend before any codec is looked up: a payload column takes only the
    artifact's payload codec (``payload_codec``); outside payload columns the torch codec
    is a torch tensor and the array codec a raw array, or a tensor of a non-torch
    artifact's own backend when its ``logical_backend`` hint names it; a ``logical_backend``, ``backend``
    or ``runtime`` hint that disagrees refuses ``record_schema_invalid``. Agreeing hints
    are kept."""

    kind, column = _cell_role(at)
    payload = column in payload_columns(kind)
    if payload:
        backend = manifest_backend
    elif codec_id == _TORCH_CODEC:
        backend = "torch"
    elif (
        manifest_backend != contract.DEFAULT_BACKEND
        and (hints or {}).get("logical_backend") == manifest_backend
    ):
        # A non-torch trace keeps its own tensors outside payload columns too (derived
        # gradients): the array codec hinted with the artifact's backend is that backend.
        backend = manifest_backend
    else:
        backend = "numpy"
    problem: str | None = None
    native = payload_codec(manifest_backend)
    if payload and codec_id != native:
        problem = (
            f"payload cells of a {manifest_backend} artifact take {native!r}, not {codec_id!r}"
        )
    for name in ("logical_backend", "backend", "runtime"):
        hint = (hints or {}).get(name)
        if problem is None and hint is not None and hint != backend:
            problem = f"tensor codec hint {name}={hint!r} disagrees with the {backend} payload"
    if problem is not None:
        raise cell_error(at, problem, code=CELL_CODE)
    return backend


def known_backend(name: str) -> bool:
    """Whether ``name`` is a contract backend or a backend registered in this runtime now.

    A runtime-registered backend (a third-party spec) saves and loads under its own name;
    an unregistered unknown name keeps the typed refusal.
    """

    if name in contract.BACKENDS:
        return True
    from ...backends.registry import registered_backend_specs

    return any(spec.name == name for spec in registered_backend_specs())


def backend_materializes(backend: str) -> bool:
    """Whether a registered backend rebuilds its saved payloads (torch always does)."""

    if backend == contract.DEFAULT_BACKEND:
        return True
    from ...backends.registry import registered_backend_specs

    return any(
        spec.name == backend and spec.capabilities.payload_materialization
        for spec in registered_backend_specs()
    )


_BACKEND_HINT_FIELDS = (
    "logical_dtype",
    "logical_device",
    "transport_backend",
    "transport_dtype",
    "codec_metadata",
)


def backend_hint_fields(hints: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the payload codec's entry fields from a tensor cell's codec hints."""

    hints = hints or {}
    return {name: hints.get(name) for name in _BACKEND_HINT_FIELDS}
