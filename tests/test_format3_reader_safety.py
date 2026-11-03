"""The 3.0 reader never executes code: exploit classes from the unpickler battery, ported to the JSON envelope.

The pickle-era battery (``tests/test_io_embedded_tensor_route_refused.py``,
``tests/test_unpickle_c_path.py``, ``tests/test_r4_metadata_unpickle_rce.py``,
``tests/test_r27_dotted_walk_rce.py``, ``tests/test_r63_construct_deny.py``,
``tests/test_io_mediated_allocation_bypass.py``, ``tests/test_r49_unpickler_weights_only_baseline.py``,
``tests/test_safe_unpickle_stream_bounds.py``) proved an unpickler refuses
gadgets. The 3.0 format has no interpreter: a crafted manifest or record cell
naming a dotted callable, a code object, an oversized blob, a tag outside the
vocabulary, a duplicate key or a wrong digest refuses with a typed
``ArtifactError`` and builds no record object, imports no module and calls no
callable. Each test plants a marker module and a spy on ``os.system`` and
asserts both untouched after the refusal.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
from _format3_envelope import (
    copy_artifact,
    first_column,
    read_manifest,
    read_record,
    redigest,
    save_tiny,
    set_cell,
    write_manifest,
    write_record,
)

from torchlens._io.format3 import contract
from torchlens._io.format_errors import ArtifactError

pytestmark = pytest.mark.heavy

KNOWN_CODES = frozenset(contract.REFUSAL_CODES.values()) | frozenset(
    contract.PROVISIONAL_REFUSALS.values()
)
MARKER_MODULE = "tl_format3_reader_marker"


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("f3-safety") / "trace"
    save_tiny(root, "portable")
    return root


@pytest.fixture
def tripwires(tmp_path, monkeypatch):
    """A marker module whose import touches a file, a spy on ``os.system`` and a record-allocation counter."""

    import torchlens._data_classes._state_adapter as adapter

    marker = tmp_path / "marker.txt"
    pkg = tmp_path / "marker_pkg"
    pkg.mkdir()
    (pkg / f"{MARKER_MODULE}.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            Path({str(marker)!r}).write_text("imported")
            def touch(*args, **kwargs):
                Path({str(marker)!r}).write_text("called")
            class Gadget:
                def __reduce__(self):
                    return (touch, ())
            """
        )
    )
    monkeypatch.syspath_prepend(str(pkg))
    sys.modules.pop(MARKER_MODULE, None)
    calls = []
    monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd) or 0)
    built = []
    original = adapter.state_new
    monkeypatch.setattr(adapter, "state_new", lambda cls: built.append(cls) or original(cls))
    yield {"marker": marker, "system_calls": calls, "built": built}
    sys.modules.pop(MARKER_MODULE, None)


def _refuses(root: Path, tripwires, *, code: str | None = None) -> ArtifactError:
    import torchlens as tl

    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    got = info.value.fields.get("code")
    assert got in KNOWN_CODES, info.value
    if code is not None:
        assert got == code, info.value
    assert not tripwires["marker"].exists(), "the reader imported or called the marker module"
    assert MARKER_MODULE not in sys.modules
    assert tripwires["system_calls"] == []
    assert not any(cls.__name__ == "Trace" for cls in tripwires["built"]), "a Trace was built"
    return info.value


def _with_cell(saved: Path, target: Path, cell, *, kind: str = "op") -> Path:
    copy = copy_artifact(saved, target)
    set_cell(copy, kind, first_column(copy, kind), 0, cell)
    return copy


def _tagged(tag: str, **fields):
    return {contract.TYPED_VALUE_KEY: tag, **fields}


# -- dotted names: record, type and enum tags name a callable or a code object -------------

DOTTED_GADGETS = [
    "os.system",
    "posix.system",
    "subprocess.Popen",
    "builtins.eval",
    "builtins.exec",
    "importlib.import_module",
    "torch.load",
    "torch.serialization.load",
    "torch.storage._load_from_bytes",
    "torch._utils._rebuild_tensor_v2",
    "torch.UntypedStorage",
    "torch._C._VariableFunctions.from_file",
    "types.CodeType",
    "types.FunctionType",
    "numpy.core.multiarray._reconstruct",
    f"{MARKER_MODULE}.touch",
    f"{MARKER_MODULE}.Gadget",
    "torchlens._io._safe_unpickle.SafeBundleUnpickler",
]


@pytest.mark.parametrize("name", DOTTED_GADGETS)
def test_record_tag_with_a_dotted_callable_refuses(saved, tmp_path, tripwires, name):
    root = _with_cell(saved, tmp_path / "rec", _tagged("record", type=name, fields={}))
    _refuses(root, tripwires)


@pytest.mark.parametrize("name", DOTTED_GADGETS)
def test_record_tag_with_arguments_refuses_without_calling(saved, tmp_path, tripwires, name):
    cell = _tagged("record", type=name, fields={"value": "touch /tmp/x", "args": ["-c", "x"]})
    _refuses(_with_cell(saved, tmp_path / "recargs", cell), tripwires)


@pytest.mark.parametrize("name", DOTTED_GADGETS)
def test_enum_tag_with_a_dotted_callable_refuses(saved, tmp_path, tripwires, name):
    _refuses(_with_cell(saved, tmp_path / "enum", _tagged("enum", type=name, name="x")), tripwires)


def test_type_tag_never_imports_the_named_module(saved, tmp_path, tripwires):
    """A ``type`` cell is a name, never a class: decoding it must not import the module."""

    import torchlens as tl

    copy = _with_cell(saved, tmp_path / "type", _tagged("type", name=f"{MARKER_MODULE}.Gadget"))
    try:
        tl.load(copy)
    except ArtifactError as exc:
        assert exc.fields.get("code") in KNOWN_CODES
    assert not tripwires["marker"].exists()
    assert MARKER_MODULE not in sys.modules
    assert tripwires["system_calls"] == []


@pytest.mark.parametrize(
    "name",
    [
        "os.system; touch x",
        "../os.system",
        "os.system()",
        "",
        "x" * 300,
        "torch.Tensor\x00os.system",
    ],
)
def test_malformed_dotted_names_refuse(saved, tmp_path, tripwires, name):
    for tag in ("record", "type", "enum"):
        fields = {"type": name, "fields": {}} if tag == "record" else {"name": name}
        if tag == "enum":
            fields = {"type": name, "name": "x"}
        _refuses(_with_cell(saved, tmp_path / f"bad-{tag}", _tagged(tag, **fields)), tripwires)


# -- tags outside the vocabulary and pickle-shaped cells ------------------------------------

FOREIGN_TAGS = [
    "pickle",
    "reduce",
    "global",
    "newobj",
    "build",
    "persid",
    "ext",
    "callable",
    "code",
    "lambda",
    "",
]


@pytest.mark.parametrize("tag", FOREIGN_TAGS)
def test_tag_outside_the_vocabulary_refuses(saved, tmp_path, tripwires, tag):
    cell = _tagged(tag, type="os.system", args=["id"], fields={}, items=[], name="x")
    _refuses(_with_cell(saved, tmp_path / "tag", cell), tripwires, code="record_schema_invalid")


def test_pickle_bytes_inside_a_bytes_cell_stay_bytes(saved, tmp_path, tripwires):
    import base64

    import torchlens as tl

    # A hand-written protocol-0 stream: GLOBAL os.system, one argument, REDUCE.
    payload = b"cos\nsystem\n(S'touch " + str(tripwires["marker"]).encode() + b"'\ntR."
    cell = _tagged("bytes", base64=base64.b64encode(payload).decode(), bytes=len(payload))
    copy = _with_cell(saved, tmp_path / "pbytes", cell)
    try:
        tl.load(copy)
    except ArtifactError as exc:
        assert exc.fields.get("code") in KNOWN_CODES
    assert tripwires["system_calls"] == []
    assert not tripwires["marker"].exists()


@pytest.mark.parametrize("tag", list(contract.TYPED_VALUE_TAGS))
def test_every_tag_with_extra_and_missing_fields_refuses(saved, tmp_path, tripwires, tag):
    _refuses(_with_cell(saved, tmp_path / "missing", _tagged(tag)), tripwires)
    _refuses(
        _with_cell(
            saved, tmp_path / "extra", _tagged(tag, __reduce__="os.system", items=[], name="x")
        ),
        tripwires,
    )


# -- allocation: oversized blob references, bytes and integers -------------------------------


def _tensor_cell(**overrides):
    cell = _tagged(
        "tensor",
        dtype="float32",
        shape=[2, 3],
        device="cpu",
        requires_grad=False,
        codec=contract.PAYLOAD_CODEC_IDS[0],
    )
    cell.update(overrides)
    return cell


OVERSIZED = [
    {"shape": [2_000_000_000, 2_000_000_000]},
    {"shape": [2**62]},
    {"shape": [1] * 65},
    {"shape": [-1]},
    {"shape": [3.5]},
    {"shape": ["3"]},
    {"shape": "3"},
    {"dtype": "qint8"},
    {"dtype": "torch.float32"},
    {"dtype": "float32; import os"},
    {"device": "cpu; import os"},
    {"device": "nosuchdevice:0"},
    {"device": "cuda:99999"},
    {"requires_grad": 1},
    {"codec": "pickle"},
    {"codec": {"id": "pickle"}},
    {"blob": "../../etc/passwd", "key": "t0"},
    {"blob": "/etc/passwd", "key": "t0"},
    {"blob": "blobs/nope.safetensors", "key": "t0"},
    {"blob": "manifest.json", "key": "t0"},
    {"key": "t0"},
    {"blob": "blobs/x.safetensors"},
]


@pytest.mark.parametrize("override", OVERSIZED, ids=lambda o: json.dumps(o)[:40])
def test_oversized_or_malformed_tensor_reference_refuses(saved, tmp_path, tripwires, override):
    _refuses(_with_cell(saved, tmp_path / "tensor", _tensor_cell(**override)), tripwires)


def test_tensor_reference_to_a_listed_blob_with_a_bomb_shape_refuses(saved, tmp_path, tripwires):
    manifest = read_manifest(saved)
    blobs = [p for p in manifest["files"] if p.startswith(contract.BLOBS_DIR + "/")]
    if not blobs:
        pytest.skip("portable level wrote no blob file")
    cell = _tensor_cell(blob=blobs[0], key="t999999999", shape=[10**9, 10**9])
    _refuses(_with_cell(saved, tmp_path / "bomb", cell), tripwires)


@pytest.mark.parametrize(
    "cell",
    [
        _tagged("bytes", base64="AAAA", bytes=2**40),
        _tagged("bytes", base64="A" * (2 * (1 << 20)), bytes=(3 * (1 << 20)) // 2),
        _tagged("bytes", base64="!!!!", bytes=3),
        _tagged("int", v="9" * 5000),
        _tagged("int", v="0x10"),
        _tagged("int", v="1e400"),
        _tagged("int", v=10**20),
        _tagged("float", v="1e400"),
        _tagged("float", v=float("inf")),
        _tagged("complex", re="1", im="2"),
    ],
    ids=lambda c: (
        c[contract.TYPED_VALUE_KEY] + ":" + str(c.get("bytes", c.get("v", c.get("re"))))[:12]
    ),
)
def test_declared_sizes_and_numeric_texts_are_bounded(saved, tmp_path, tripwires, cell):
    _refuses(_with_cell(saved, tmp_path / "bounded", cell), tripwires)


def test_nesting_deeper_than_the_cap_refuses(saved, tmp_path, tripwires):
    cell = "leaf"
    for _ in range(200):
        cell = _tagged("tuple", items=[cell])
    _refuses(_with_cell(saved, tmp_path / "deep", cell), tripwires)


def test_bare_non_finite_float_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "nan")
    payload = read_record(copy, "op")
    column = first_column(copy, "op")
    text = json.dumps(payload).replace(json.dumps(payload["columns"][column][0]), "NaN", 1)
    (copy / contract.RECORDS_DIR / "op.json").write_text(text)
    redigest(copy, f"{contract.RECORDS_DIR}/op.json")
    _refuses(copy, tripwires)


# -- duplicate keys -----------------------------------------------------------------------


def test_duplicate_json_keys_in_a_record_file_refuse(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "dupkey")
    path = copy / contract.RECORDS_DIR / "op.json"
    payload = read_record(copy, "op")
    column = first_column(copy, "op")
    text = path.read_text()
    first = json.dumps(column) + ":"
    cut = text.index(first)
    text = (
        text[:cut]
        + json.dumps(column)
        + ":"
        + json.dumps(["os.system"] * payload["rows"])
        + ","
        + text[cut:]
    )
    path.write_text(text)
    redigest(copy, f"{contract.RECORDS_DIR}/op.json")
    _refuses(copy, tripwires)


def test_duplicate_json_keys_in_the_manifest_refuse(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "dupman")
    path = copy / contract.MANIFEST_FILE
    text = path.read_text()
    assert text.startswith("{")
    path.write_text('{"format": "os.system",' + text[1:])
    _refuses(copy, tripwires)


def test_duplicate_dict_pairs_refuse(saved, tmp_path, tripwires):
    cell = _tagged("dict", items=[["a", 1], ["a", 2]])
    _refuses(_with_cell(saved, tmp_path / "duppair", cell), tripwires)


def test_blob_key_redeclared_differently_across_two_cells_refuses(saved, tmp_path, tripwires):
    # One stored tensor may sit in two cells; a second reference declaring
    # another shape for the same (chunk, key) is a forgery and refuses.
    copy = copy_artifact(saved, tmp_path / "dupblob")
    payload = read_record(copy, "op")
    ref = next(
        (
            c
            for col in payload["columns"].values()
            for c in col
            if isinstance(c, dict) and c.get(contract.TYPED_VALUE_KEY) == "tensor" and c.get("blob")
        ),
        None,
    )
    if ref is None:
        pytest.skip("portable level wrote no tensor payload")
    redeclared = dict(ref)
    redeclared["shape"] = [1, *ref["shape"]]
    set_cell(copy, "op", first_column(copy, "op"), 0, redeclared)
    _refuses(copy, tripwires)


def test_duplicate_file_entries_differing_in_case_or_dots_refuse(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "dupfile")
    manifest = read_manifest(copy)
    relpath, entry = next(iter(manifest["files"].items()))
    manifest["files"]["./" + relpath] = dict(entry)
    write_manifest(copy, manifest)
    _refuses(copy, tripwires)


# -- digests -------------------------------------------------------------------------------


def test_wrong_file_digest_refuses(saved, tmp_path, tripwires):
    for relpath in read_manifest(saved)["files"]:
        copy = copy_artifact(saved, tmp_path / ("digest-" + relpath.replace("/", "_")))
        manifest = read_manifest(copy)
        manifest["files"][relpath]["sha256"] = "f" * 64
        write_manifest(copy, manifest)
        _refuses(copy, tripwires)


@pytest.mark.parametrize(
    "relpath", ["extra.bin", "blobs/planted.safetensors", "records/extra.json"]
)
def test_unlisted_file_in_the_artifact_is_named_and_never_read(saved, tmp_path, relpath):
    """Contract section 1: an unlisted file is never opened; the load succeeds with a warning."""

    import torchlens as tl

    copy = copy_artifact(saved, tmp_path / ("unlisted-" + relpath.replace("/", "_")))
    (copy / relpath).parent.mkdir(parents=True, exist_ok=True)
    (copy / relpath).write_bytes(b"\x00")
    with pytest.warns(UserWarning, match="unreferenced extra files") as caught:
        trace = tl.load(copy)
    assert relpath in str(caught[0].message)
    assert trace is not None


def test_wrong_record_digest_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "recdigest")
    manifest = read_manifest(copy)
    manifest["records"]["op"]["sha256"] = "0" * 64
    write_manifest(copy, manifest)
    _refuses(copy, tripwires)


def test_wrong_byte_count_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "bytes")
    manifest = read_manifest(copy)
    relpath = next(iter(manifest["files"]))
    manifest["files"][relpath]["bytes"] = 2**40
    write_manifest(copy, manifest)
    _refuses(copy, tripwires)


def test_edited_record_without_redigest_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "noredigest")
    payload = read_record(copy, "op")
    payload["columns"][first_column(copy, "op")][0] = "edited"
    write_record(copy, "op", payload, redigest_it=False)
    _refuses(copy, tripwires, code=contract.REFUSAL_CODES["record_file_corrupt"])


# -- 2.x envelopes and record references ---------------------------------------------------


@pytest.mark.parametrize("marker", ["metadata.pkl", "bundle.json"])
def test_pre3_root_marker_refuses_as_below_floor(saved, tmp_path, tripwires, marker):
    copy = copy_artifact(saved, tmp_path / "pre3")
    (copy / marker).write_bytes(b"\x80\x04\x95")
    _refuses(copy, tripwires, code=contract.REFUSAL_CODES["format_older_than_floor"])


def test_manifest_without_format_key_refuses_as_below_floor(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "noformat")
    manifest = read_manifest(copy)
    del manifest["format"]
    write_manifest(copy, manifest)
    _refuses(copy, tripwires, code=contract.REFUSAL_CODES["format_older_than_floor"])


def test_tlspec_below_floor_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "oldspec")
    manifest = read_manifest(copy)
    manifest["tlspec_version"] = 10
    write_manifest(copy, manifest)
    _refuses(copy, tripwires, code=contract.REFUSAL_CODES["format_older_than_floor"])


@pytest.mark.parametrize(
    "ref",
    [
        {"record": "op", "row": 10**9},
        {"record": "op", "row": -1},
        {"record": "os.system", "row": 0},
        {"record": "trace", "row": 1},
        {"record": "op", "row": "0"},
    ],
    ids=lambda r: f"{r['record']}:{r['row']}",
)
def test_dangling_or_foreign_record_references_refuse(saved, tmp_path, tripwires, ref):
    _refuses(_with_cell(saved, tmp_path / "ref", _tagged("ref", **ref)), tripwires)


def test_record_tag_naming_a_record_class_refuses(saved, tmp_path, tripwires):
    """Record classes travel as rows and ``ref`` cells, never as inline ``record`` objects."""

    cell = _tagged("record", type="torchlens._data_classes.op.Op", fields={})
    _refuses(_with_cell(saved, tmp_path / "inline-op", cell), tripwires)


def test_symlinked_record_file_refuses(saved, tmp_path, tripwires):
    copy = copy_artifact(saved, tmp_path / "symlink")
    target = copy / contract.RECORDS_DIR / "op.json"
    real = tmp_path / "elsewhere.json"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    _refuses(copy, tripwires)


def test_tag_refusal_carries_the_ruled_cell_code(saved, tmp_path, tripwires):
    """A cell that fails its type check refuses with the ruled ``record_schema_invalid``."""

    cell = _tagged(FOREIGN_TAGS[0], type="os.system", args=["id"], fields={}, items=[], name="x")
    error = _refuses(_with_cell(saved, tmp_path / "ruled", cell), tripwires)
    assert error.fields["code"] == "record_schema_invalid"


def test_recursion_during_rehydration_refuses_typed(saved, monkeypatch):
    """A record graph that exhausts the interpreter stack surfaces typed, never raw."""

    import torchlens as tl
    from torchlens._io.format3 import reader

    def explode(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(reader, "load_trace", explode)
    with pytest.raises(ArtifactError) as info:
        tl.load(saved)
    assert info.value.fields["code"] == "rehydrate_recursion_limit"


def test_contract_backends_are_the_payload_codec_registry() -> None:
    """The closed ``backend`` vocabulary names exactly the registered backend payload codecs.

    The registry's ``numpy`` entry is the raw-input value codec, not a backend.
    """

    from torchlens._io import payload_codec
    from torchlens._io.format3 import contract

    assert set(contract.BACKENDS) == set(payload_codec._CODECS) - {"numpy"}
    assert contract.DEFAULT_BACKEND in contract.BACKENDS
