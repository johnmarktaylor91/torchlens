"""The 3.0 saved-format writer: envelope, record tables, payload chunks and disclosures.

Covers the three worked examples of ``docs/maintainers/saved_format_3.md`` section 8
(a plain portable trace, a runnable save, an intervened save), every record key with
its JSON type at every save level, the absence of pickle at every level, the manifest
``disclosures`` object, blob naming and chunking, and the tlspec 12 columns the format
adds (``op.is_constant``, ``op.arg_expressions`` and the live-side settled values).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

import torchlens as tl
from torchlens._io.format3 import schema as format_schema, writer as format_writer
from torchlens._io.format3.contract import (
    ARTIFACT_KINDS,
    BLOBS_DIR,
    FORMAT_MAJOR,
    FORMAT_NAME,
    INTERVENTION_FILE,
    MANIFEST_FILE,
    MANIFEST_REQUIRED_FIELDS,
    PAYLOAD_CODEC_IDS,
    RECORD_CONTRACT_VERSION,
    RECORD_INDEX,
    TENSOR_REFERENCE_FIELDS,
    TYPED_VALUE_KEY,
    TYPED_VALUE_TAGS,
)
from torchlens._io.format3.recipe import RECIPE_FORMAT, RECIPE_FORMAT_VERSION
from torchlens._io.format3.values import decode_value
from torchlens._io.writer_contract import writer_contract_digest
from torchlens._postprocess._pipeline import PIPELINE

_PAYLOAD_FLAGS = (
    "include_outs",
    "include_grads",
    "include_saved_args",
    "include_rng_states",
    "include_buffer_values",
    "include_custom_attributes",
    "structure_only",
)
_BLOB_NAME = re.compile(rf"^{BLOBS_DIR}/({'|'.join(PAYLOAD_CODEC_IDS)})-(\d{{4}})\.safetensors$")
_BLOB_KEY = re.compile(r"^t(\d+)$")
_PICKLE_PROTOCOL_HEADER = re.compile(rb"^\x80[\x02-\x05]")
_IO2_OP_COLUMNS = (
    "is_constant",
    "arg_expressions",
    "equivalence_class",
    "func_duration",
    "edge_substitutions",
    "edge_replacement_stamps",
)


class _TinyCNN(nn.Module):
    """The tiny CNN of the worked examples (conv2d, relu, pool, linear, a constant)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x)))
        x = nn.functional.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.fc(x) + torch.ones(3)


class _Opaque:
    """A user object with no typed encoding (written as an ``unencodable`` disclosure)."""

    def __repr__(self) -> str:
        return "Opaque<" + "x" * 300 + ">"


def _input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 1, 6, 6)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(root: Path) -> dict[str, Any]:
    return _load_json(root / MANIFEST_FILE)


def _records(root: Path) -> dict[str, dict[str, Any]]:
    manifest = _manifest(root)
    return {kind: _load_json(root / entry["path"]) for kind, entry in manifest["records"].items()}


def _artifact_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _walk_json(value: Any):
    """Yield every JSON object and scalar of a parsed JSON value."""

    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Save the tiny CNN at every save level (plus grads and structure-only variants)."""

    base = tmp_path_factory.mktemp("format3_writer")
    x = _input()
    plain = tl.trace(_TinyCNN().eval(), x)
    plain.save(base / "portable")
    plain.save(base / "audit", level="audit")

    grads_model = _TinyCNN().train()
    grads_input = _input().requires_grad_(True)
    graded = tl.trace(grads_model, grads_input, capture=tl.options.CaptureOptions(save_grads="all"))
    graded.log_backward(graded[graded.output_layers[0]].out.sum())
    graded.save(base / "grads", include_saved_args=True, include_rng_states=True)

    replayable = tl.trace(_TinyCNN().eval(), x, capture=tl.options.CaptureOptions(replayable=True))
    replayable.save(base / "runnable", level="runnable")

    structure = tl.trace(
        _TinyCNN().eval(), x, capture=tl.options.CaptureOptions(structure_only=True)
    )
    structure.save(base / "structure_only")

    tl.Bundle({"plain": plain, "graded": graded}).save(base / "bundle")
    return {
        "portable": base / "portable",
        "audit": base / "audit",
        "grads": base / "grads",
        "runnable": base / "runnable",
        "structure_only": base / "structure_only",
        "bundle": base / "bundle",
    }


_TRACE_LABELS = ("portable", "audit", "grads", "runnable", "structure_only")
_LEVEL_BY_LABEL = {
    "portable": "portable",
    "audit": "audit",
    "grads": "portable",
    "runnable": "runnable",
    "structure_only": "portable",
}


def _trace_roots(saved: dict[str, Path]) -> list[tuple[str, Path]]:
    """Every trace-kind artifact, bundle members included."""

    roots = [(label, saved[label]) for label in _TRACE_LABELS]
    members = sorted((saved["bundle"] / "members").iterdir())
    roots.extend((f"bundle/{member.name}", member) for member in members)
    return roots


@pytest.mark.parametrize("label", _TRACE_LABELS)
def test_manifest_envelope(saved: dict[str, Path], label: str) -> None:
    root = saved[label]
    manifest = _manifest(root)
    missing = [name for name in MANIFEST_REQUIRED_FIELDS if name not in manifest]
    assert not missing
    assert manifest["format"] == FORMAT_NAME
    assert manifest["format_version"] == FORMAT_MAJOR
    assert manifest["tlspec_version"] == RECORD_CONTRACT_VERSION
    assert manifest["kind"] == "trace"
    assert manifest["save_level"] == _LEVEL_BY_LABEL[label]
    assert manifest["contract_digest"] == writer_contract_digest()
    assert set(manifest["codecs"]) <= set(PAYLOAD_CODEC_IDS)
    policy = manifest["payload_policy"]
    assert set(policy) == set(_PAYLOAD_FLAGS)
    assert all(type(flag) is bool for flag in policy.values())
    assert "tensor_entries" not in manifest


@pytest.mark.parametrize("label", (*_TRACE_LABELS, "bundle"))
def test_files_digest_every_file_but_the_manifest(saved: dict[str, Path], label: str) -> None:
    root = saved[label]
    manifest = _manifest(root)
    on_disk = [name for name in _artifact_files(root) if name not in (MANIFEST_FILE, "PARTIAL")]
    assert sorted(manifest["files"]) == on_disk
    for name, entry in manifest["files"].items():
        data = (root / name).read_bytes()
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.mark.parametrize("label", _TRACE_LABELS)
def test_records_index_matches_record_files(saved: dict[str, Path], label: str) -> None:
    root = saved[label]
    manifest = _manifest(root)
    known_kinds = {kind for kind, _, _ in RECORD_INDEX}
    assert set(manifest["records"]) <= known_kinds
    assert {"trace", "op", "layer", "module", "param"} <= set(manifest["records"])
    for kind, entry in manifest["records"].items():
        assert entry["path"] == f"records/{kind}.json"
        data = (root / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data)
        payload = json.loads(data)
        assert payload["record"] == kind
        assert payload["tlspec_version"] == RECORD_CONTRACT_VERSION
        if kind == "trace":
            assert entry["rows"] == 1
            assert set(payload) == {"record", "tlspec_version", "fields"}
            continue
        assert set(payload) == {"record", "tlspec_version", "rows", "columns"}
        assert payload["rows"] == entry["rows"] >= 1
        assert all(len(cells) == payload["rows"] for cells in payload["columns"].values())


def _is_tensor_tag(value: Any) -> bool:
    return isinstance(value, dict) and value.get(TYPED_VALUE_KEY) == "tensor"


def _tensor_ref_cell_ok(cell: Any) -> bool:
    """A ``tensor_ref`` cell: null, a tensor reference, or a container of them."""

    if cell is None or _is_tensor_tag(cell):
        return True
    if isinstance(cell, list):
        return all(_tensor_ref_cell_ok(item) for item in cell)
    if isinstance(cell, dict):
        tag = cell.get(TYPED_VALUE_KEY)
        if tag in ("tuple", "set", "frozenset"):
            return all(_tensor_ref_cell_ok(item) for item in cell["items"])
        if tag == "dict":
            return all(_tensor_ref_cell_ok(pair[1]) for pair in cell["items"])
        if tag is None:
            return all(_tensor_ref_cell_ok(item) for item in cell.values())
    return False


def _id_array_cell_ok(cell: Any) -> bool:
    """An ``id_array`` cell: null or a container of label strings, row indices or refs."""

    if cell is None or isinstance(cell, (str, int)):
        return True
    if isinstance(cell, list):
        return all(_id_array_cell_ok(item) for item in cell)
    if isinstance(cell, dict):
        tag = cell.get(TYPED_VALUE_KEY)
        if tag == "ref":
            return True
        if tag in ("tuple", "set", "frozenset"):
            return all(_id_array_cell_ok(item) for item in cell["items"])
        if tag == "dict":
            return all(_id_array_cell_ok(item) for pair in cell["items"] for item in pair)
        if tag is None:
            return all(_id_array_cell_ok(item) for item in cell.values())
    return False


#: Edge columns whose live value is an accessor of records; each is an id array.
_ACCESSOR_EDGE_COLUMNS = frozenset({("layer", "ops"), ("module", "ops"), ("module", "params")})


def _cell_type_problem(kind: str, name: str, json_type: str, cell: Any) -> str | None:
    at = f"{kind}.{name}"
    decode_value(cell, at=at)
    if json_type == "tensor_ref" and not _tensor_ref_cell_ok(cell):
        return f"not a tensor reference cell: {cell!r:.200}"
    if json_type == "id_array" and not _id_array_cell_ok(cell):
        return f"not an id_array cell: {cell!r:.200}"
    if json_type not in ("tensor_ref", "id_array", "value"):
        return f"JSON type {json_type!r} is never written"
    return None


@pytest.mark.parametrize("label", _TRACE_LABELS)
def test_every_record_key_present_with_its_json_type(saved: dict[str, Path], label: str) -> None:
    root = saved[label]
    policy = _manifest(root)["payload_policy"]
    wrong: dict[str, str] = {}
    for kind, payload in _records(root).items():
        schema = format_schema.record_schema(kind)
        expected = format_schema.level_columns(kind, policy)
        cells_by_column = (
            {name: [cell] for name, cell in payload["fields"].items()}
            if kind == "trace"
            else payload["columns"]
        )
        assert list(cells_by_column) == expected, kind
        for name, cells in cells_by_column.items():
            for cell in cells:
                problem = _cell_type_problem(kind, name, schema[name].json_type, cell)
                if problem:
                    wrong.setdefault(f"{kind}.{name}", problem)
    assert not wrong, "\n".join(f"{key}: {text}" for key, text in sorted(wrong.items()))


@pytest.mark.parametrize(
    "column", sorted(f"{kind}.{name}" for kind, name in _ACCESSOR_EDGE_COLUMNS)
)
def test_accessor_edge_columns_are_id_arrays(saved: dict[str, Path], column: str) -> None:
    kind, name = column.split(".")
    cells = _records(saved["portable"])[kind]["columns"][name]
    assert all(isinstance(cell, list) and _id_array_cell_ok(cell) for cell in cells), cells[0]
    assert any(cell for cell in cells), cells


@pytest.mark.parametrize("label", (*_TRACE_LABELS, "bundle"))
def test_no_pickle_at_any_level(saved: dict[str, Path], label: str) -> None:
    root = saved[label]
    for name in _artifact_files(root):
        assert not name.endswith((".pkl", ".pickle")), name
        data = (root / name).read_bytes()
        assert not _PICKLE_PROTOCOL_HEADER.match(data), name
        if name.endswith(".json"):
            for node in _walk_json(json.loads(data)):
                if isinstance(node, dict) and TYPED_VALUE_KEY in node:
                    assert node[TYPED_VALUE_KEY] in TYPED_VALUE_TAGS, (name, node)
        else:
            assert name.endswith(".safetensors"), name
            with safe_open(str(root / name), framework="pt") as handle:
                assert list(handle.keys())


@pytest.mark.parametrize("label", ("portable", "grads", "bundle"))
def test_blobs_are_codec_chunks_with_monotone_opaque_keys(
    saved: dict[str, Path], label: str
) -> None:
    roots = (
        [root for _, root in _trace_roots(saved) if root.parent.name == "members"]
        if label == "bundle"
        else [saved[label]]
    )
    for root in roots:
        _check_blob_layout(root)


def _check_blob_layout(root: Path) -> dict[str, list[str]]:
    """Assert the chunk naming and key rules; return the keys of each chunk."""

    manifest = _manifest(root)
    chunks = [name for name in _artifact_files(root) if name.startswith(f"{BLOBS_DIR}/")]
    assert chunks, root
    keys_by_chunk: dict[str, list[str]] = {}
    numbers: list[int] = []
    for chunk in chunks:
        match = _BLOB_NAME.match(chunk)
        assert match, chunk
        assert match.group(1) in manifest["codecs"]
        with safe_open(str(root / chunk), framework="pt") as handle:
            keys = list(handle.keys())
        assert all(_BLOB_KEY.match(key) for key in keys), keys
        keys_by_chunk[chunk] = keys
        numbers.extend(int(key[1:]) for key in keys)
    assert sorted(numbers) == list(range(len(numbers))), "keys are not t0..t<n-1>"
    by_codec: dict[str, list[int]] = {}
    for chunk in chunks:
        match = _BLOB_NAME.match(chunk)
        assert match is not None
        by_codec.setdefault(match.group(1), []).append(int(match.group(2)))
    assert all(sorted(series) == list(range(len(series))) for series in by_codec.values())
    for blob_id, entry in manifest["tensors"].items():
        assert entry["key"] in keys_by_chunk[entry["blob"]], blob_id
    referenced = 0
    for payload in _records(root).values():
        for node in _walk_json(payload):
            if _is_tensor_tag(node) and "blob" in node:
                assert set(node) - {TYPED_VALUE_KEY} <= set(TENSOR_REFERENCE_FIELDS)
                assert node["key"] in keys_by_chunk[node["blob"]], node
                referenced += 1
    assert referenced >= 1
    return keys_by_chunk


def test_chunks_close_at_the_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cap = 256
    monkeypatch.setenv("TORCHLENS_PAYLOAD_CHUNK_BYTES", str(cap))
    trace = tl.trace(_TinyCNN().eval(), _input())
    trace.save(tmp_path / "chunked")
    keys_by_chunk = _check_blob_layout(tmp_path / "chunked")
    assert len(keys_by_chunk) > 1
    for chunk, keys in keys_by_chunk.items():
        with safe_open(str(tmp_path / "chunked" / chunk), framework="pt") as handle:
            sizes = [
                handle.get_tensor(key).numel() * handle.get_tensor(key).element_size()
                for key in keys
            ]
        assert len(keys) == 1 or sum(sizes) <= cap, (chunk, sizes)


def test_structure_only_references_carry_no_blob(saved: dict[str, Path]) -> None:
    root = saved["structure_only"]
    manifest = _manifest(root)
    assert manifest["payload_policy"]["structure_only"] is True
    assert manifest["codecs"] == []
    assert not [name for name in _artifact_files(root) if name.startswith(f"{BLOBS_DIR}/")]
    columns = _records(root)["op"]["columns"]
    assert "out" in columns
    for cell in columns["out"]:
        assert cell is None or (_is_tensor_tag(cell) and "blob" not in cell and "key" not in cell)


def test_disclosure_summary_rules() -> None:
    assert format_writer.disclosure_summary([]) is None
    few = [{"at": f"op[{i}].x"} for i in range(3)]
    assert format_writer.disclosure_summary(few) == {
        "count": 3,
        "first": ["op[0].x", "op[1].x", "op[2].x"],
    }
    many = [{"at": f"op[{i}].x"} for i in range(40)]
    summary = format_writer.disclosure_summary(many)
    assert summary is not None
    assert summary["count"] == 40
    assert summary["first"] == [f"op[{i}].x" for i in range(16)]


@pytest.mark.parametrize("label", (*_TRACE_LABELS, "bundle"))
def test_disclosures_absent_when_every_value_encodes(saved: dict[str, Path], label: str) -> None:
    assert "disclosures" not in _manifest(saved[label])


def test_disclosures_count_and_first_paths(tmp_path: Path) -> None:
    policy = dict.fromkeys(_PAYLOAD_FLAGS, False)
    schema = format_schema.record_schema("trace")
    column = next(
        name
        for name in format_schema.level_columns("trace", policy)
        if schema[name].json_type == "value"
    )
    state = {column: [_Opaque() for _ in range(20)]}
    manifest = format_writer.write_trace_envelope(
        tmp_path,
        base_manifest={},
        scrubbed_state=state,
        tensor_refs={},
        codecs=[],
        save_level="portable",
        payload_policy=policy,
    )
    assert manifest == _manifest(tmp_path)
    summary = manifest.get("disclosures")
    assert summary == {
        "count": 20,
        "first": [f"trace.{column}[{index}]" for index in range(16)],
    }
    unencodable = [
        node
        for node in _walk_json(_records(tmp_path)["trace"])
        if isinstance(node, dict) and node.get(TYPED_VALUE_KEY) == "unencodable"
    ]
    assert len(unencodable) == 20
    for node in unencodable:
        assert node["type"].endswith("_Opaque")
        assert node["repr"] == repr(_Opaque())[:256]
        assert node["repr_length"] == len(repr(_Opaque()))
    format_writer.reload_check(tmp_path)


@pytest.mark.parametrize("label", (*_TRACE_LABELS, "bundle-members"))
def test_io2_op_columns_at_every_level(saved: dict[str, Path], label: str) -> None:
    roots = (
        [root for name, root in _trace_roots(saved) if name.startswith("bundle/")]
        if label == "bundle-members"
        else [saved[label]]
    )
    for root in roots:
        columns = _records(root)["op"]["columns"]
        for name in _IO2_OP_COLUMNS:
            assert name in columns, (root, name)
        assert all(type(flag) is bool for flag in columns["is_constant"])
        for name in ("edge_substitutions", "edge_replacement_stamps"):
            assert all(isinstance(cell, dict) for cell in columns[name]), (name, columns[name])
        assert any(cell is not None for cell in columns["equivalence_class"])
        assert any(cell is not None for cell in columns["func_duration"])


def test_is_constant_marks_the_literal_producer(saved: dict[str, Path]) -> None:
    columns = _records(saved["portable"])["op"]["columns"]
    constants = [
        label
        for label, flag in zip(columns["layer_label"], columns["is_constant"], strict=True)
        if flag
    ]
    assert any(label.startswith("ones") for label in constants), constants


#: A non-pipeline phase timing bucket name (``dispatch_forward_model``, ``ctx_build_*``).
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


@pytest.mark.parametrize("label", _TRACE_LABELS)
def test_phase_timings_keyed_by_pipeline_step_names(saved: dict[str, Path], label: str) -> None:
    timings = _records(saved[label])["trace"]["fields"]["_phase_timings"]
    assert isinstance(timings, dict) and timings
    step_names = {step.name for step in PIPELINE}
    assert step_names & set(timings)
    for key in timings:
        assert key in step_names or _SNAKE_CASE.fullmatch(key), key


def test_portable_example_envelope(saved: dict[str, Path]) -> None:
    root = saved["portable"]
    manifest = _manifest(root)
    policy = manifest["payload_policy"]
    assert policy["include_outs"] is True
    assert policy["structure_only"] is False
    assert manifest["codecs"] == ["torch_safetensors_v1"]
    op = _records(root)["op"]
    assert "out" in op["columns"]
    assert all(_is_tensor_tag(cell) and "blob" in cell for cell in op["columns"]["out"])


def test_runnable_example_envelope(saved: dict[str, Path]) -> None:
    root = saved["runnable"]
    manifest = _manifest(root)
    assert manifest["save_level"] == "runnable"
    policy = manifest["payload_policy"]
    assert not any(policy[flag] for flag in ("include_outs", "include_grads", "include_saved_args"))
    op_columns = _records(root)["op"]["columns"]
    assert "out" not in op_columns
    assert "grad" not in op_columns
    if "runnable" not in manifest:
        pytest.skip("the runnable component is written by the runnable lane's runnable.py")
    assert manifest["runnable"]["descriptor"].startswith("runnable/")


def test_intervened_example_envelope(tmp_path: Path) -> None:
    trace = tl.trace(_TinyCNN().eval(), _input(), intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    trace.save(tmp_path / "intervened")
    root = tmp_path / "intervened"
    columns = _records(root)["op"]["columns"]
    edited = [
        label
        for label, flag in zip(columns["layer_label"], columns["is_edited"], strict=True)
        if flag
    ]
    assert edited and all(label.startswith("relu") for label in edited)
    assert INTERVENTION_FILE in _manifest(root)["files"]
    recipe = _load_json(root / INTERVENTION_FILE)
    assert recipe["format"] == RECIPE_FORMAT
    assert recipe["format_version"] == RECIPE_FORMAT_VERSION
    assert recipe["executable"] is True
    assert [rule["edit"]["name"] for rule in recipe["rules"]] == ["scale"]
    assert _manifest(root)["intervention"]["executable"] is True
    assert _manifest(root)["intervention"]["file"] == INTERVENTION_FILE
    format_writer.reload_check(root)


def test_bundle_envelope(saved: dict[str, Path]) -> None:
    root = saved["bundle"]
    manifest = _manifest(root)
    assert manifest["kind"] == "bundle"
    assert manifest["kind"] in ARTIFACT_KINDS
    assert manifest["format_version"] == FORMAT_MAJOR
    assert (root / "relations.json").is_file()
    members = sorted(path.name for path in (root / "members").iterdir())
    assert len(members) == 2
    assert all(re.match(r"^\d{4}-[A-Za-z0-9_.-]+$", name) for name in members), members
    for name in members:
        member = _manifest(root / "members" / name)
        assert member["kind"] == "trace"
        assert member["format_version"] == FORMAT_MAJOR


@pytest.mark.parametrize("label", (*_TRACE_LABELS, "bundle"))
def test_staged_artifact_passes_the_reload_check(saved: dict[str, Path], label: str) -> None:
    roots = (
        [root for name, root in _trace_roots(saved) if name.startswith("bundle/")]
        if label == "bundle"
        else [saved[label]]
    )
    for root in roots:
        format_writer.reload_check(root)
