"""CONTRACT (h): a 3.0 artifact reads back equal to the live trace it was saved from.

``docs/maintainers/saved_format_3.md`` section 9. A tiny CNN is traced once and saved
at each of the four save levels. Until the 3.0 reader is merged, the record files are
decoded with ``values.decode_value`` (the decoder the reader builds on) and compared
cell by cell with the live records:

- the column set of each record file is the persisted key set of its level;
- ``value`` and ``id_array`` cells compare equal (floats by exact repr); a ``ref`` cell
  resolves to the row of the same live record object, whose cells are compared in turn;
  a label or address string stands for the live record object that carries it;
- ``tensor_ref`` cells compare by dtype, shape and recorded device, and the payload read
  from its chunk by ``torch.equal`` with the live tensor;
- an ``unencodable`` disclosure compares its ``repr`` with the live value's ``repr``.

The writer's privacy policies are applied to the live side before comparing, each by an
exact rule: parameter barcodes are renamed through the ``param`` table's barcode column
(so ``equivalence_class`` and every barcode-keyed map compare exactly), absolute source
paths become their file names, object ids are renumbered one-to-one within each column, and the
``_source_trace_ref`` back-reference is written as null.

``_KNOWN_GAPS`` lists the columns where the current writer's value differs from the live
one (today only the runnable level's scrub). It may only shrink: a gap that closes fails
``test_known_gaps_still_differ`` until it is removed from the list.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import weakref
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes._accessor_base import Accessor
from torchlens._data_classes._state_adapter import state_items
from torchlens._io.format3 import schema as format_schema
from torchlens._io.format3.contract import TENSOR_REFERENCE_FIELDS
from torchlens._io.format3.payloads import read_tensor
from torchlens._io.format3.values import RowRef, decode_value, object_state
from torchlens._io.format3.writer import record_kind_of

_LEVELS = ("portable", "audit", "runnable", "executable_with_callables")
#: Seed containers: record kind to the live trace attribute whose order is the row order.
_SEEDS = {
    "op": "_layer_list",
    "layer": "_layer_logs",
    "module": "_module_logs",
    "param": "_param_logs",
}
#: Columns holding Python object ids, which the writer renumbers one-to-one.
_OBJECT_ID_COLUMNS = frozenset(
    {"grad_fn_object_id", "_provenance_input_object_id", "_provenance_model_object_id"}
)
#: Live attributes that name a record (a label or address string stands for it).
_RECORD_NAME_ATTRIBUTES = ("layer_label", "address", "label", "barcode", "name")
# The writer remaps each param barcode to ``param_NNNNNN`` and sorts the run of them in an
# equivalence key; the live key sorts the random raw barcodes, so its order is arbitrary.
_PARAM_TOKEN_RUN = re.compile(r"param_\d{6}(?:_param_\d{6})+")


def _sorted_param_run(match: re.Match[str]) -> str:
    return "_".join(sorted(re.findall(r"param_\d{6}", match.group(0))))


#: Computed columns the writer evaluated on the scrubbed copy (no trace link).
_COMPUTED_ON_SCRUBBED_COPY: frozenset[str] = frozenset()
#: Value objects the writer flattened to plain mappings.
_FLATTENED_VALUE_OBJECTS: frozenset[str] = frozenset()
_RUNNABLE_SCRUB = frozenset(
    {
        "layer.func_config",
        "op.func_config",
        "module.custom_attributes",
        "op.parent_params",
        "trace._buffer_initial_values",
        "trace._orphan_records",
    }
)
#: Columns whose written value differs from the live value today, per save level.
_KNOWN_GAPS: dict[str, frozenset[str]] = {
    "portable": _COMPUTED_ON_SCRUBBED_COPY | _FLATTENED_VALUE_OBJECTS,
    "audit": _COMPUTED_ON_SCRUBBED_COPY | _FLATTENED_VALUE_OBJECTS,
    "runnable": _COMPUTED_ON_SCRUBBED_COPY | _FLATTENED_VALUE_OBJECTS | _RUNNABLE_SCRUB,
    "executable_with_callables": _COMPUTED_ON_SCRUBBED_COPY | _FLATTENED_VALUE_OBJECTS,
}


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x)))
        x = nn.functional.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.fc(x) + torch.ones(3)


def _seed_rows(trace: Any, kind: str) -> list[Any]:
    container = getattr(trace, _SEEDS[kind], None)
    if container is None:
        return []
    if isinstance(container, dict):
        return list(container.values())
    listed = getattr(container, "_list", None)
    return list(listed) if isinstance(listed, list) else list(container)


class _Comparison:
    """Compare decoded record tables with the live records they were written from."""

    def __init__(
        self,
        root: Path,
        tables: dict[str, dict[str, Any]],
        policy: dict[str, bool],
        barcodes: dict[str, str],
    ) -> None:
        self.root = root
        self.tables = tables
        self.policy = policy
        self.barcodes = barcodes
        self.barcode_pattern = (
            re.compile("|".join(re.escape(code) for code in sorted(barcodes, key=len)[::-1]))
            if barcodes
            else None
        )
        self.paired: dict[tuple[str, int], int] = {}
        self.pending: deque[tuple[str, int, Any]] = deque()
        self.object_ids: dict[str, dict[int, int]] = {}
        self.mismatches: dict[str, str] = {}
        self.compared = 0
        self.column = ""

    def miss(self, text: str) -> None:
        self.mismatches.setdefault(self.column, text)

    def pair(self, kind: str, row: int, live: Any, at: str) -> None:
        known = self.paired.get((kind, row))
        if known is None:
            self.paired[(kind, row)] = id(live)
            self.pending.append((kind, row, live))
        elif known != id(live):
            self.miss(f"{at}: ref {kind}[{row}] names another live record")

    def run(self) -> None:
        while self.pending:
            kind, row, live = self.pending.popleft()
            self.compare_row(kind, row, live)

    def compare_row(self, kind: str, row: int, live: Any) -> None:
        payload = self.tables.get(kind)
        self.column = f"{kind}.<file>"
        if payload is None:
            self.miss(f"{kind}[{row}]: no records/{kind}.json")
            return
        state = dict(state_items(live))
        schema = format_schema.record_schema(kind)
        cells = payload["fields"] if kind == "trace" else payload["columns"]
        expected = format_schema.level_columns(kind, self.policy)
        if list(cells) != expected:
            self.miss(f"{kind}: columns differ from the level's key set")
        for name in expected:
            self.column = f"{kind}.{name}"
            cell = cells[name] if kind == "trace" else cells[name][row]
            at = f"{kind}[{row}].{name}"
            decoded = decode_value(cell, at=at)
            live_value = _live_cell(live, state, name, schema[name].storage)
            if name in _OBJECT_ID_COLUMNS:
                self.compare_object_id(decoded, live_value, at)
            elif name == "_source_trace_ref":
                if decoded is not None or not isinstance(live_value, (weakref.ref, type(None))):
                    self.miss(f"{at}: {decoded!r:.80} vs {live_value!r:.80}")
            else:
                self.compare(decoded, live_value, at)
            self.compared += 1

    def compare_object_id(self, decoded: Any, live: Any, at: str) -> None:
        if live is None or decoded is None:
            if live is not decoded:
                self.miss(f"{at}: {decoded!r} vs {live!r}")
            return
        renumbered = self.object_ids.setdefault(at.rsplit(".", 1)[-1], {})
        previous = renumbered.setdefault(int(live), int(decoded))
        if previous != decoded or list(renumbered.values()).count(decoded) != 1:
            self.miss(f"{at}: object id renumbering is not one-to-one")

    def translate(self, text: str) -> str:
        if os.path.isabs(text) and os.sep in text and "\n" not in text:
            return os.path.basename(text)
        if self.barcode_pattern is None:
            return text
        text = self.barcode_pattern.sub(lambda match: self.barcodes[match.group(0)], text)
        return _PARAM_TOKEN_RUN.sub(_sorted_param_run, text)

    def compare(self, decoded: Any, live: Any, at: str) -> None:
        if isinstance(live, Accessor) and isinstance(decoded, list):
            # An accessor of records is an edge: written as its member list.
            live = list(live._list)
        if isinstance(decoded, RowRef):
            if record_kind_of(live) != decoded.record:
                self.miss(f"{at}: ref {decoded} vs live {type(live).__name__}")
            elif decoded.record != "trace":
                self.pair(decoded.record, decoded.row, live, at)
            return
        if isinstance(decoded, str) and record_kind_of(live) is not None:
            names = {
                self.translate(str(getattr(live, attribute, None)))
                for attribute in _RECORD_NAME_ATTRIBUTES
            }
            if decoded not in names:
                self.miss(f"{at}: {decoded!r} names no attribute of the live record")
            return
        if isinstance(live, torch.Tensor):
            self.compare_tensor(decoded, live, at)
            return
        if isinstance(decoded, dict) and set(decoded) == {"type", "repr", "repr_length", "at"}:
            text = repr(live)
            if decoded["repr"] != text[:256] or decoded["repr_length"] != len(text):
                self.miss(f"{at}: disclosure repr differs from the live repr")
            return
        self.compare_plain(decoded, live, at)

    def compare_plain(self, decoded: Any, live: Any, at: str) -> None:
        if isinstance(live, str):
            if decoded != self.translate(live):
                self.miss(f"{at}: {decoded!r:.80} vs {live!r:.80}")
        elif isinstance(live, float) and isinstance(decoded, float):
            same = repr(decoded) == repr(live) or (math.isnan(decoded) and math.isnan(live))
            if not same:
                self.miss(f"{at}: {decoded!r} vs {live!r}")
        elif isinstance(live, (list, tuple)):
            if type(decoded) is not type(live) or len(decoded) != len(live):
                self.miss(f"{at}: {decoded!r:.80} vs {live!r:.80}")
                return
            for index, (left, right) in enumerate(zip(decoded, live, strict=True)):
                self.compare(left, right, f"{at}[{index}]")
        elif isinstance(live, dict):
            translated = {
                (self.translate(key) if isinstance(key, str) else key): value
                for key, value in live.items()
            }
            if not isinstance(decoded, dict) or list(decoded) != list(translated):
                self.miss(f"{at}: mapping keys differ")
                return
            for key, value in translated.items():
                self.compare(decoded[key], value, f"{at}[{key!r}]")
        elif _is_value_object(live) and type(decoded) is type(live):
            self.compare(object_state(decoded), object_state(live), at)
        elif decoded != live:
            self.miss(f"{at}: {decoded!r:.80} vs {live!r:.80}")

    def compare_tensor(self, decoded: Any, live: torch.Tensor, at: str) -> None:
        if not isinstance(decoded, dict) or not set(decoded) <= set(TENSOR_REFERENCE_FIELDS):
            self.miss(f"{at}: live tensor written as {decoded!r:.80}")
            return
        if decoded["dtype"] != str(live.dtype).replace("torch.", ""):
            self.miss(f"{at}: dtype {decoded['dtype']} vs {live.dtype}")
        if decoded["shape"] != list(live.shape) or decoded["device"] != str(live.device):
            self.miss(f"{at}: shape or device differs")
        if "blob" not in decoded:
            return
        loaded = read_tensor(self.root, blob=decoded["blob"], key=decoded["key"])
        if decoded.get("complex"):
            loaded = torch.view_as_complex(loaded)
        if not torch.equal(loaded, live.detach().cpu().resolve_conj().resolve_neg()):
            self.miss(f"{at}: payload differs from the live tensor")


def _is_value_object(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("torchlens") and (
        dataclasses.is_dataclass(value) or hasattr(value, "__dict__") or hasattr(value, "_fields")
    )


def _live_cell(record: Any, state: dict[str, Any], name: str, storage: str) -> Any:
    if name in state:
        return state[name]
    if storage == "computed":
        try:
            return getattr(record, name)
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
            return None
    return None


def _tables(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tables = {
        kind: json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        for kind, entry in manifest["records"].items()
    }
    return manifest, tables


def _compare_level(trace: Any, root: Path) -> _Comparison:
    manifest, tables = _tables(root)
    columns = tables["param"]["columns"]
    written = dict(zip(columns["address"], columns["barcode"], strict=True))
    barcodes = {
        str(param.barcode): str(written[param.address]) for param in _seed_rows(trace, "param")
    }
    comparison = _Comparison(root, tables, manifest["payload_policy"], barcodes)
    comparison.compare_row("trace", 0, trace)
    for kind in _SEEDS:
        rows = _seed_rows(trace, kind)
        assert tables[kind]["rows"] >= len(rows), kind
        for row, live in enumerate(rows):
            comparison.pair(kind, row, live, f"{kind} seed")
    comparison.run()
    for kind, payload in tables.items():
        if kind != "trace":
            unpaired = [
                row for row in range(payload["rows"]) if (kind, row) not in comparison.paired
            ]
            assert not unpaired, f"{kind} rows {unpaired} match no live record"
    return comparison


@pytest.fixture(scope="module")
def comparisons(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, _Comparison]]:
    base = tmp_path_factory.mktemp("format3_lel")
    torch.manual_seed(0)
    trace = tl.trace(
        _TinyCNN().eval(),
        torch.randn(2, 1, 6, 6),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        results = {}
        for level in _LEVELS:
            trace.save(base / level, level=level)
            results[level] = _compare_level(trace, base / level)
        yield results
    finally:
        trace.cleanup()


@pytest.mark.parametrize("level", _LEVELS)
def test_loaded_records_equal_the_live_records(
    comparisons: dict[str, _Comparison], level: str
) -> None:
    comparison = comparisons[level]
    unexpected = {
        column: text
        for column, text in comparison.mismatches.items()
        if column not in _KNOWN_GAPS[level]
    }
    assert not unexpected, "\n".join(unexpected[key][:240] for key in sorted(unexpected))
    assert comparison.compared >= 13 * 150


@pytest.mark.parametrize("level", _LEVELS)
def test_known_gaps_still_differ(comparisons: dict[str, _Comparison], level: str) -> None:
    closed = sorted(_KNOWN_GAPS[level] - set(comparisons[level].mismatches))
    assert not closed, f"these gaps closed; remove them from _KNOWN_GAPS[{level!r}]: {closed}"


@pytest.mark.parametrize("level", _LEVELS)
def test_barcode_renaming_is_one_to_one(comparisons: dict[str, _Comparison], level: str) -> None:
    barcodes = comparisons[level].barcodes
    assert barcodes
    assert len(set(barcodes.values())) == len(barcodes)
