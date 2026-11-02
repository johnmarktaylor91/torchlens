"""io.2 acceptance: a 3.0 artifact loads back equal to the live trace at every save level.

Column identity: the record files carry exactly the columns the writer's level
schema names, and the loaded objects carry the same persisted keys. Value
identity: the canonical object diff of ``tests/_pickle_compare.py`` finds no
difference between the live trace and the loaded one, eager and lazy, for the
contract's worked example (the portable tiny CNN) at each level.
"""

from __future__ import annotations

import pytest
from _format3_envelope import read_manifest, read_record, save_tiny
from _pickle_compare import _diff

from torchlens._io.format3 import contract
from torchlens._io.format3.schema import level_columns

pytestmark = pytest.mark.heavy

LEVELS = tuple(level for level in contract.SAVE_LEVELS if level != "executable_with_callables")


@pytest.fixture(scope="module", params=LEVELS)
def saved(request, tmp_path_factory):
    level = request.param
    root = tmp_path_factory.mktemp(f"f3-{level}") / "trace"
    trace = save_tiny(root, level)
    return level, root, trace


def test_record_files_carry_exactly_the_level_columns(saved):
    level, root, _ = saved
    manifest = read_manifest(root)
    assert manifest["save_level"] == level
    for kind in manifest["records"]:
        payload = read_record(root, kind)
        columns = payload["fields"] if kind == "trace" else payload["columns"]
        expected = set(level_columns(kind, manifest["payload_policy"]))
        assert set(columns) == expected, kind


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_loaded_trace_equals_live(saved, lazy):
    import torchlens as tl

    level, root, live = saved
    loaded = tl.load(root, lazy=lazy)
    diffs = _diff(live, loaded, "trace", set())
    assert diffs == [], f"{level}/{'lazy' if lazy else 'eager'}:\n" + "\n".join(diffs[:60])


def test_loaded_rows_match_the_manifest(saved):
    import torchlens as tl

    _, root, _ = saved
    manifest = read_manifest(root)
    loaded = tl.load(root)
    assert len(loaded.ops) == manifest["records"]["op"]["rows"]
    assert loaded._source_bundle_path == str(root)
