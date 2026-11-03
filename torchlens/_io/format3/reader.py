"""The 3.0 reader: manifest first, then every record validated against the contract.

Contract: ``docs/maintainers/saved_format_3.md``. The reader is fail-closed:
it opens the manifest, checks the format and record versions, verifies every
file digest, then validates each record file (known keys only, types by the
storage-kind map, tensor references resolved against the blob index) before a
single object is built. Every function here is a stub that refuses typed until
the reader lane fills it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from .contract import refuse_reader_not_built

_LANE_MODULE = "torchlens/_io/format3/reader.py"


def read_manifest(path: str | Path) -> NoReturn:
    """Read and validate ``manifest.json`` without opening any other file.

    Parameters
    ----------
    path:
        Artifact directory.
    """

    refuse_reader_not_built("read_manifest", _LANE_MODULE)


def validate_artifact(path: str | Path) -> NoReturn:
    """Validate every record file and blob of an artifact against the contract.

    This is the checked load's first half and the save preflight's reload.

    Parameters
    ----------
    path:
        Artifact directory.
    """

    refuse_reader_not_built("validate_artifact", _LANE_MODULE)


def load_trace(path: str | Path, *, lazy: bool = False) -> NoReturn:
    """Load a trace artifact after validating it.

    Parameters
    ----------
    path:
        Artifact directory.
    lazy:
        Leave tensor payloads as references until first access.
    """

    refuse_reader_not_built("load_trace", _LANE_MODULE)


def load_bundle(path: str | Path) -> NoReturn:
    """Load a bundle artifact: members and relations.

    Parameters
    ----------
    path:
        Artifact directory.
    """

    refuse_reader_not_built("load_bundle", _LANE_MODULE)


def load_intervention_recipe(path: str | Path) -> Any:
    """Load a typed recipe file into a staged intervention spec.

    A rule whose edit is a callable placeholder loads as an audit-only edit;
    the rerun doors refuse it with the position of the missing callable.

    Parameters
    ----------
    path:
        Recipe file.
    """

    refuse_reader_not_built("load_intervention_recipe", _LANE_MODULE)
