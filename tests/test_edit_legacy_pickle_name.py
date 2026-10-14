"""Artifacts written before the U00 rename of the edit-object class still load.

JMT's ruling U00 (2026-10-05): "yes, renames are hard. clean break". The class
behind ``tl.Edit`` was renamed from ``HelperSpec`` to ``Edit`` with no public
alias. Saved artifacts of tlspec 6-9 pickle it under the old global name, so the
bundle loader's restricted unpickler maps that one global to ``Edit`` (format
plumbing, never a public spelling). The new name pickles as ``Edit``.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import replace

import torchlens as tl
from torchlens._io.bundle import _RenameAwareUnpickler

_OLD_GLOBAL = b"ctorchlens.intervention.types\nHelperSpec\n"
_NEW_GLOBAL = b"ctorchlens.intervention.types\nEdit\n"


def _scrubbed_edit() -> tl.Edit:
    """A builtin edit as save scrubs it (no factory callable)."""

    return replace(tl.zero_ablate(), factory=None)


def test_new_writes_pickle_the_edit_class_by_its_new_name() -> None:
    payload = pickle.dumps(_scrubbed_edit(), protocol=2)
    assert _NEW_GLOBAL in payload and b"HelperSpec" not in payload


def test_an_old_name_pickle_loads_as_edit_through_the_bundle_unpickler() -> None:
    edit = _scrubbed_edit()
    legacy = pickle.dumps(edit, protocol=2).replace(_NEW_GLOBAL, _OLD_GLOBAL)
    assert _OLD_GLOBAL in legacy
    loaded = _RenameAwareUnpickler(io.BytesIO(legacy)).load()
    assert type(loaded) is tl.Edit
    assert loaded == edit
