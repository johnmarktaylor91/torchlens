"""A pickle that names the edit class by its 2.x spelling refuses; the new name is what 3 writes.

The class behind ``tl.Edit`` was renamed from ``HelperSpec`` to ``Edit`` with no public alias.
Torchlens 3 reads no file saved by 2.x, so the old global name refuses with the one typed 2.x
error at the bundle loader's restricted unpickler instead of being mapped to ``Edit``.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import replace

import pytest

import torchlens as tl
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.errors import ArtifactTooOldError

_OLD_GLOBAL = b"ctorchlens.intervention.types\nHelperSpec\n"
_NEW_GLOBAL = b"ctorchlens.intervention.types\nEdit\n"


def _scrubbed_edit() -> tl.Edit:
    """A builtin edit as save scrubs it (no factory callable)."""

    return replace(tl.zero_ablate(), factory=None)


def test_new_writes_pickle_the_edit_class_by_its_new_name() -> None:
    payload = pickle.dumps(_scrubbed_edit(), protocol=2)
    assert _NEW_GLOBAL in payload and b"HelperSpec" not in payload


def test_an_old_name_pickle_refuses_as_a_2x_file_through_the_bundle_unpickler() -> None:
    edit = _scrubbed_edit()
    legacy = pickle.dumps(edit, protocol=2).replace(_NEW_GLOBAL, _OLD_GLOBAL)
    assert _OLD_GLOBAL in legacy
    with pytest.raises(ArtifactTooOldError) as excinfo:
        _RenameAwareUnpickler(io.BytesIO(legacy)).load()
    assert "saved by torchlens 2.x" in str(excinfo.value)
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
