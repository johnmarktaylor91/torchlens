"""Rollup LOW/INFO hardening for the in-fence _io surfaces (R10 / R60).

* ``io_role`` closed-vocabulary validation at parse (R10-17).
* Negative logical-shape dims rejected (R10-8).
* Type-keyed save memos are weak so a factory-made class is not pinned (R60-12).
"""

from __future__ import annotations

import pytest

from torchlens._io import bundle as bundle_module, runnable as runnable_module
from torchlens._io.payload_codec import _logical_shape_from_metadata
from torchlens._io.runnable_load import (
    ContextFieldInvalidError,
    _parse_input_binding,
)


@pytest.mark.smoke
def test_input_binding_io_role_closed_vocabulary() -> None:
    good = {
        "io_role": "model_input",
        "model_ref": "m",
        "model_site_position": 0,
        "container_record_id": 0,
        "container_path": [],
    }
    assert _parse_input_binding(good).io_role == "model_input"

    bad = dict(good, io_role="has_cuda")
    with pytest.raises(ContextFieldInvalidError) as excinfo:
        _parse_input_binding(bad)
    assert "io_role" in str(excinfo.value)


@pytest.mark.smoke
def test_negative_logical_shape_is_rejected() -> None:
    assert _logical_shape_from_metadata({"logical_shape": [2, 3]}) == (2, 3)
    assert _logical_shape_from_metadata({"logical_shape": [-1, 3]}) is None
    assert _logical_shape_from_metadata({"logical_shape": [-1, -1]}) is None


def test_save_memos_are_weak_keyed() -> None:
    import weakref

    assert isinstance(runnable_module._SPARSE_CORE_NODE_KINDS, weakref.WeakKeyDictionary)
    assert isinstance(runnable_module._DATACLASS_FIELD_NAMES, weakref.WeakKeyDictionary)
    assert isinstance(bundle_module._NESTED_BLOB_KINDS, weakref.WeakKeyDictionary)
