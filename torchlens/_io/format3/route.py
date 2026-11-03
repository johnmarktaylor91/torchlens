"""Routing of ``bundle.load`` into the 3.0 saved format: the root test and the door dispatch.

Kept outside ``bundle.py`` and its import cycle: this module reaches only the
reader, so the loader's layering (largest cycle, in-function imports) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import torch

from ..format_errors import ArtifactError
from . import contract, reader

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ...bundle import Bundle
    from ...intervention.types import InterventionSpec
    from .. import PayloadLoadHints

_FORMAT3_READER_SHAPES: Final = frozenset({"v3", "unknown"})


def is_format3_root(bundle_path: Path, tlspec_format: str) -> bool:
    """Return whether the directory is read by the 3.0 reader.

    The reader owns roots naming the 3.0 format and unrecognized roots with a
    manifest. It validates their format, schema, version window and root files.
    Recognized 2.x artifacts refuse before this route is considered.
    """

    manifest_path = bundle_path / contract.MANIFEST_FILE
    if not (manifest_path.exists() or manifest_path.is_symlink()):
        return False
    return tlspec_format in _FORMAT3_READER_SHAPES


@dataclass(frozen=True, slots=True)
class Format3LoadRequest:
    """The load knobs one ``tl.load`` call carries to the format3 reader doors."""

    lazy: bool
    map_location: str | torch.device
    materialize_nested: bool
    payload_hints: PayloadLoadHints | None
    unknown_relations: Literal["opaque", "refuse"] = "opaque"


def load_format3(
    bundle_path: Path, request: Format3LoadRequest
) -> Trace | Bundle | InterventionSpec:
    """Dispatch a 3.0 artifact to the format3 reader door its manifest ``kind`` names
    (``unknown_relations`` reaches the bundle door only)."""

    try:
        kind = reader.read_manifest(bundle_path)["kind"]
        if kind == "intervention":
            return cast("InterventionSpec", reader.load_intervention_recipe(bundle_path))
        options: dict[str, Any] = {
            "lazy": request.lazy,
            "map_location": request.map_location,
            "materialize_nested": request.materialize_nested,
            "payload_hints": request.payload_hints,
        }
        if kind == "bundle":
            options["unknown_relations"] = request.unknown_relations
        door = reader.load_bundle if kind == "bundle" else reader.load_trace
        loaded = door(bundle_path, **options)
    except RecursionError as exc:
        # A hostile record graph nested deeply enough to blow the C stack during
        # rehydration surfaces typed, never as a raw RecursionError.
        raise ArtifactError(
            f"Failed to load bundle at {bundle_path}: record nesting exceeded the "
            "interpreter recursion limit during rehydration.",
            code="rehydrate_recursion_limit",
        ) from exc
    return cast("Trace | Bundle", loaded)
