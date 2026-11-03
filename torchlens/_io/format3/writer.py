"""The 3.0 writer: a trace or bundle to typed JSON records plus safetensors blobs.

Contract: ``docs/maintainers/saved_format_3.md`` (envelope, records, tensor
references, unencodable values, the intervention recipe, refusals). Every
function here is a stub that refuses typed until the writer lane fills it;
the signatures are the contract the reader and runnable lanes build against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from .contract import refuse_writer_not_built

_LANE_MODULE = "torchlens/_io/format3/writer.py"


def write_trace(
    trace: Any,
    path: str | Path,
    *,
    level: str = "portable",
    overwrite: bool = False,
    payload_policy: dict[str, bool] | None = None,
) -> NoReturn:
    """Write one trace as a 3.0 artifact directory.

    Parameters
    ----------
    trace:
        Live trace to persist.
    path:
        Target directory; refused when it exists unless ``overwrite``.
    level:
        One of ``torchlens._io.format3.contract.SAVE_LEVELS``.
    overwrite:
        Replace an existing artifact atomically.
    payload_policy:
        Explicit payload flags; conflicts with the level refuse typed.
    """

    refuse_writer_not_built("write_trace", _LANE_MODULE)


def write_bundle(bundle: Any, path: str | Path, *, level: str = "portable") -> NoReturn:
    """Write a bundle: one member artifact per trace plus the relation table.

    Parameters
    ----------
    bundle:
        Live bundle to persist.
    path:
        Target directory.
    level:
        Save level applied to every member.
    """

    refuse_writer_not_built("write_bundle", _LANE_MODULE)


def write_intervention_recipe(spec: Any, path: str | Path) -> NoReturn:
    """Write a staged intervention spec as the typed recipe file.

    The same grammar serves the recipe inside a trace artifact and the
    separate recipe ``save_intervention`` writes.

    Parameters
    ----------
    spec:
        Staged intervention spec (rules and built-in edits).
    path:
        Target file.
    """

    refuse_writer_not_built("write_intervention_recipe", _LANE_MODULE)
