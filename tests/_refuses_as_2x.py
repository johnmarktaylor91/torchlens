"""Shared assertion: a file saved by torchlens 2.x refuses at ``tl.load``, typed and clear.

TorchLens 3 carries no reader for 2.x files: their pickles name package paths that left the
public surface in the 3.0 rename. Every test that once proved a 2.x file loads now proves it
refuses with one typed error that names the 2.x origin and the remedy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import torchlens as tl
from torchlens.errors import ArtifactTooOldError


def assert_refuses_as_2x(path: Path | str) -> ArtifactTooOldError:
    """Load ``path`` and assert the typed 2.x refusal; return the raised error.

    Parameters
    ----------
    path:
        A saved artifact written by a torchlens 2.x release.

    Returns
    -------
    ArtifactTooOldError
        The refusal, for callers that check more fields.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ArtifactTooOldError) as excinfo:
            tl.load(str(path))
    message = str(excinfo.value)
    assert "saved by torchlens 2.x" in message, message
    assert "save it again with torchlens 2.36.x" in message, message
    assert "capture the model again" in message, message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    return excinfo.value
