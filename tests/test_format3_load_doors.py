"""io.2 acceptance, reader side: a 3.0 artifact loads back equal at every save level.

Column identity: the record files carry exactly the columns the writer's level
schema names. Value identity: saving the loaded trace again writes record tables
cell-identical to the original artifact and blob chunks with the same digests
(the writer's ``test_format3_loaded_equals_live`` proves live equals table, so
live, table and loaded agree per record kind); a lazy load materializes every
persisted tensor equal to the eager load's by ``torch.equal``, dtype, shape and
device string. The contract's worked example is the portable tiny CNN.
"""

from __future__ import annotations

import pytest
import torch
from _format3_envelope import (
    read_manifest,
    read_record,
    record_table_diffs,
    resave,
    save_tiny,
    tensor_columns,
)

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


def test_loaded_trace_resaves_identically(saved, tmp_path):
    import torchlens as tl

    level, root, _ = saved
    loaded = tl.load(root)
    again = resave(loaded, tmp_path / "again", level)
    diffs = record_table_diffs(root, again)
    assert diffs == [], f"{level}:\n" + "\n".join(diffs)


def _tensor_equal(eager: torch.Tensor, lazy: torch.Tensor) -> bool:
    return (
        eager.dtype == lazy.dtype
        and tuple(eager.shape) == tuple(lazy.shape)
        and str(eager.device) == str(lazy.device)
        and torch.equal(eager, lazy)
    )


def test_lazy_load_materializes_every_persisted_tensor(saved):
    import torchlens as tl

    level, root, _ = saved
    columns = tensor_columns(root, "op")
    if not columns:
        pytest.skip(f"{level} persists no op tensors")
    eager, lazy = tl.load(root), tl.load(root, lazy=True)
    compared = 0
    for name in columns:
        for op_eager, op_lazy in zip(eager._layer_list, lazy._layer_list, strict=True):
            value = getattr(op_eager, name)
            if isinstance(value, torch.Tensor):
                assert _tensor_equal(value, getattr(op_lazy, name)), f"{name} on {op_eager.label}"
                compared += 1
    assert compared > 0


def test_loaded_rows_match_the_manifest(saved):
    import torchlens as tl

    _, root, _ = saved
    manifest = read_manifest(root)
    loaded = tl.load(root)
    assert len(loaded._layer_list) == manifest["records"]["op"]["rows"]
    assert loaded._source_bundle_path == root
