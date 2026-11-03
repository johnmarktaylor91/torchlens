"""Codec role policy for format 3 tensor cells (contract: payload backend boundary).

An artifact's payload columns (``out``, ``grad`` and the other PAYLOAD storage columns)
take one codec: the torch codec for a torch artifact, the numeric array codec for every
other backend (its tensors materialize as arrays). The numeric array codec also carries
raw values (the trace input and other value cells) in any artifact. A codec hint that
disagrees with the cell's backend is a typed ``record_schema_invalid`` refusal raised
before any codec is looked up.
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

__all__ = ["codec_backend", "payload_codec", "payload_columns"]

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

    return _TORCH_CODEC if manifest_backend == "torch" else _ARRAY_CODEC


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
    is a torch tensor and the array codec a raw array; a ``logical_backend``, ``backend``
    or ``runtime`` hint that disagrees refuses ``record_schema_invalid``. Agreeing hints
    are kept."""

    kind, column = _cell_role(at)
    payload = column in payload_columns(kind)
    if payload:
        backend = manifest_backend
    else:
        backend = "torch" if codec_id == _TORCH_CODEC else "numpy"
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
