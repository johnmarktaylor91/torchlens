"""The 3.0 runnable component: descriptor JSON plus safetensors state, no pickle.

Contract: ``docs/maintainers/saved_format_3.md``, section "Runnable". The
runnable directory keeps the descriptor, call recipe, callable reference and
initializer policy component versions of the writer contract; its state dict
and non-persistent buffers are safetensors files indexed from the manifest.
Every function here is a stub that refuses typed until the runnable lane fills it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from .contract import refuse_reader_not_built, refuse_writer_not_built

_LANE_MODULE = "torchlens/_io/format3/runnable.py"


def write_runnable(trace: Any, path: str | Path, *, include_weights: bool = True) -> NoReturn:
    """Write the runnable component of a trace into an artifact directory.

    Parameters
    ----------
    trace:
        Live trace carrying runnable state.
    path:
        Artifact directory (the component lands under ``runnable/``).
    include_weights:
        Write the state dict blobs; otherwise the descriptor alone.
    """

    refuse_writer_not_built("write_runnable", _LANE_MODULE)


def load_runnable(path: str | Path) -> NoReturn:
    """Load the runnable component after the artifact validated.

    Parameters
    ----------
    path:
        Artifact directory.
    """

    refuse_reader_not_built("load_runnable", _LANE_MODULE)
