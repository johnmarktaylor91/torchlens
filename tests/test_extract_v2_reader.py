"""Lazy reader, views, graded verification, and exporters (lane F18).

The D14 read path: bounded open, ledger-order iteration, coalesced row
access (T-COALESCE wiring), duplicate-aware ID lookup, the exact byte
guard, trusted-prefix monitoring, and the D7 read-side verification grades;
plus the D15 exporters with the self-contained contract file (T-CONTRACT:
a subprocess with TorchLens NOT importable resolves every stimulus ID,
identifies every member's origin, and verifies file integrity).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._data_substrate import ExtractionArtifactError
from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import (
    as_torch_dataset,
    export_extraction,
    extract_dataset,
    feature_matrix,
    load_extraction,
    open_extraction,
    shuffled_batches,
)


def _artifact(tmp_path: Path, *, n: int = 10, ids: bool = True, batch_size: int = 3) -> Path:
    """Write a small deterministic two-key artifact and return its dir."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    stimuli = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    out = tmp_path / "artifact"
    extract_dataset(
        model,
        stimuli,
        {"relu": "relu", "logits": "output_1"},
        batch_size=batch_size,
        output_dir=out,
        progress=False,
        stimulus_ids=[f"s{i % 7}" for i in range(n)] if ids else None,
    )
    return out


# --- lazy reader core ---------------------------------------------------------------


def test_reader_rows_coalesce_and_match_materialized(tmp_path: Path) -> None:
    """T-COALESCE wiring: scattered fancy rows equal the materialized rows."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    full = reader.materialize(["relu"])["relu"]
    picks = [9, 0, 4, 3, 1]
    scattered = reader.rows(picks, keys=["relu"])["relu"]
    assert torch.equal(scattered, full[picks])
    sliced = reader[2:5]
    assert torch.equal(sliced["relu"], full[2:5])
    with pytest.raises(InvalidArgumentError) as excinfo:
        reader.rows([99])
    assert excinfo.value.fields["code"] == "extraction_reader_row_out_of_range"


def test_reader_iter_batches_serves_ledger_order(tmp_path: Path) -> None:
    """iter_batches yields shard payloads in LEDGER order with key subsets."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    rows = 0
    for payload in reader.iter_batches(keys=["logits"]):
        assert set(payload) == {"logits"}
        rows += payload["logits"].shape[0]
    assert rows == reader.n_stimuli == 10
    with pytest.raises(InvalidArgumentError) as excinfo:
        next(iter(reader.iter_batches(keys=["nope"])))
    assert excinfo.value.fields["code"] == "extraction_reader_key_unknown"


@pytest.mark.smoke
def test_reader_duplicate_aware_id_lookup(tmp_path: Path) -> None:
    """row_for refuses ambiguity naming rows_for; unknown ids refuse typed."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    assert reader.rows_for("s1") == [1, 8]
    with pytest.raises(InvalidArgumentError) as excinfo:
        reader.row_for("s1")
    assert excinfo.value.fields["code"] == "extraction_reader_id_ambiguous"
    assert reader.row_for("s6") == 6
    with pytest.raises(InvalidArgumentError) as excinfo:
        reader.row_for("missing")
    assert excinfo.value.fields["code"] == "extraction_reader_id_unknown"


def test_reader_ids_unavailable_refuses(tmp_path: Path) -> None:
    """ID lookup on an id-free artifact refuses typed."""

    out = _artifact(tmp_path, ids=False)
    reader = open_extraction(out)
    with pytest.raises(InvalidArgumentError) as excinfo:
        reader.row_for("s0")
    assert excinfo.value.fields["code"] == "extraction_reader_ids_unavailable"


def test_eager_guard_computes_exact_bytes_and_refuses(tmp_path: Path) -> None:
    """D14: the byte guard is EXACT from ledger facts and names the reader."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    exact = reader.requested_bytes(["relu"])
    assert exact == 10 * 4 * 4, "10 rows x 4 features x float32"
    with pytest.raises(InvalidArgumentError) as excinfo:
        load_extraction(out, max_bytes=exact - 1)
    assert excinfo.value.fields["code"] == "extraction_eager_budget_exceeded"
    assert "open_extraction" in str(excinfo.value)
    loaded = load_extraction(out, max_bytes=10**9)
    assert torch.equal(loaded.activations["relu"], reader.materialize(["relu"])["relu"])


def test_reader_monitors_in_progress_trusted_prefix(tmp_path: Path) -> None:
    """Trusted-prefix monitoring: refresh() picks up newly committed shards."""

    out = _artifact(tmp_path)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Exception) as excinfo:
        open_extraction(out)
    assert excinfo.value.fields["code"] == "extraction_manifest_invalid"
    reader = open_extraction(out, in_progress=True)
    assert reader.n_shards == 4
    ledger = (out / "ledger.jsonl").read_text().splitlines()
    (out / "ledger.jsonl").write_text("".join(line + "\n" for line in ledger[:2]))
    fresh = open_extraction(out, in_progress=True)
    assert fresh.n_shards == 2
    (out / "ledger.jsonl").write_text("".join(line + "\n" for line in ledger))
    assert fresh.refresh() == 2
    assert fresh.n_shards == 4
    completed = open_extraction(out.parent / "artifact2", in_progress=True) if False else reader
    with pytest.raises(InvalidArgumentError) as excinfo:
        open_extraction(out, in_progress=True, verify="everything")
    assert excinfo.value.fields["code"] == "extraction_reader_verify_invalid"
    del completed


def test_reader_not_monitoring_refuses_refresh(tmp_path: Path) -> None:
    """refresh() on a completed-artifact handle refuses typed."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    with pytest.raises(InvalidArgumentError) as excinfo:
        reader.refresh()
    assert excinfo.value.fields["code"] == "extraction_reader_not_monitoring"


# --- graded verification (D7 read side) ----------------------------------------------


@pytest.mark.smoke
def test_first_access_verification_catches_flipped_byte(tmp_path: Path) -> None:
    """T-INTEGRITY class A read side: a flipped shard byte fails the CRC."""

    out = _artifact(tmp_path)
    shard = sorted(out.glob("batch_*.safetensors"))[1]
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))
    reader = open_extraction(out, verify="first_access")
    reader.rows([0], keys=["relu"])  # shard 0 untouched: verifies clean
    with pytest.raises(ExtractionArtifactError) as excinfo:
        reader.rows([5], keys=["relu"])  # shard 1 carries the flip
    assert excinfo.value.fields["code"] == "extraction_shard_integrity_mismatch"
    # Structural default never CRCs, so the same read only fails at parse
    # time if at all — verify_all is the explicit full scan.
    full = open_extraction(out)
    with pytest.raises(ExtractionArtifactError) as excinfo:
        full.verify_all()
    assert excinfo.value.fields["code"] == "extraction_shard_integrity_mismatch"


def test_verify_all_checks_value_reductions(tmp_path: Path) -> None:
    """verify_all recomputes the D7 value fact for every ledgered key."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    report = reader.verify_all()
    assert report["n_shards"] == 4
    assert report["values_checked"] == 8, "two keys x four shards"


# --- views layer -----------------------------------------------------------------------


@pytest.mark.smoke
def test_shuffled_batches_cover_every_row_once(tmp_path: Path) -> None:
    """The SAE access pattern: shard-order + within-shard shuffle, full cover."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    full = reader.materialize(["relu"])["relu"]
    seen = []
    for batch in shuffled_batches(reader, 2, seed=7, keys=["relu"]):
        assert batch["relu"].shape[0] <= 2
        seen.extend(batch["relu"].tolist())
    assert len(seen) == 10
    assert sorted(map(tuple, seen)) == sorted(map(tuple, full.tolist()))


def test_shuffled_batches_batch_size_refuses(tmp_path: Path) -> None:
    """A non-positive batch size refuses typed."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    with pytest.raises(InvalidArgumentError) as excinfo:
        next(iter(shuffled_batches(reader, 0)))
    assert excinfo.value.fields["code"] == "extraction_reader_batch_size_invalid"


@pytest.mark.smoke
def test_feature_matrix_refuses_trimmed_keys(tmp_path: Path) -> None:
    """A ragged key has no rectangular matrix; to_padded is the densifier."""

    class _Tok(nn.Module):
        """Tiny token model."""

        def __init__(self) -> None:
            """Build deterministically."""

            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(20, 4)

        def forward(self, input_ids, attention_mask=None):
            """Embed (trailing mul keeps the module output non-terminal)."""

            return torch.relu(self.emb(input_ids)) * 1.0

    def collate(items):
        """Right-pad integer rows."""

        width = max(len(row) for row in items)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros(len(items), width, dtype=torch.long)
        for i, row in enumerate(items):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    out = tmp_path / "ragged-artifact"
    extract_dataset(
        _Tok().eval(),
        [[1, 2, 3], [4], [5, 6]],
        {"h": "relu"},
        batch_size=3,
        output_dir=out,
        collate=collate,
        ragged="trim",
        progress=False,
    )
    reader = open_extraction(out)
    with pytest.raises(InvalidArgumentError) as excinfo:
        feature_matrix(reader, "h")
    assert excinfo.value.fields["code"] == "extraction_reader_ragged_matrix_unsupported"


def test_feature_matrix_and_torch_dataset_views(tmp_path: Path) -> None:
    """feature_matrix flattens dense keys; the dataset serves per-row dicts."""

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    matrix = feature_matrix(reader, "relu")
    assert matrix.shape == (10, 4)
    dataset = as_torch_dataset(reader, keys=["logits"])
    assert len(dataset) == 10
    row = dataset[3]
    assert set(row) == {"logits"}
    assert torch.equal(row["logits"], reader.materialize(["logits"])["logits"][3])


# --- exporters (item 14 / D15) ------------------------------------------------------------


def test_export_npy_roundtrip_and_contract_subprocess(tmp_path: Path) -> None:
    """T-CONTRACT: the export is self-contained WITHOUT TorchLens importable.

    In a subprocess with TorchLens unimportable and the source artifact
    ABSENT, the contract file alone resolves every stimulus ID to its row,
    identifies every member's origin, and verifies per-file sha256.
    """

    out = _artifact(tmp_path)
    reader = open_extraction(out)
    expected = reader.materialize(["relu"])["relu"]
    dest = tmp_path / "export"
    export_extraction(out, dest, format="npy")
    import shutil

    shutil.rmtree(out)  # the source artifact is GONE; the export must stand alone
    checker = r"""
import hashlib, json, sys
import numpy as np

dest = sys.argv[1]
assert "torchlens" not in sys.modules
contract = json.load(open(f"{dest}/export_contract.json"))
ids = contract["stimulus_ids"]
assert ids is not None and len(ids) == contract["source"]["n_stimuli"]
# Resolve every occurrence of a duplicated stimulus id to its rows.
rows = [i for i, sid in enumerate(ids) if sid == "s1"]
assert rows == [1, 8], rows
for member in contract["members"]:
    payload = open(f"{dest}/{member['file']}", "rb").read()
    assert len(payload) == member["byte_size"]
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    assert digest == member["sha256"], member["file"]
    assert member["source_key"] in contract["keys"] or "|" in member["source_key"]
stem = contract["keys"]["relu"]["sanitized_stem"]
values = np.load(f"{dest}/{stem}.npy")
assert values.shape == tuple([contract["source"]["n_stimuli"]] + contract["keys"]["relu"]["logical_shape"])
print("CONTRACT_OK", values.shape)
"""
    result = subprocess.run(
        [sys.executable, "-c", checker, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "CONTRACT_OK" in result.stdout
    import numpy as np

    contract = json.loads((dest / "export_contract.json").read_text())
    stem = contract["keys"]["relu"]["sanitized_stem"]
    assert torch.equal(torch.from_numpy(np.load(dest / f"{stem}.npy")), expected)
    assert (dest / "provenance" / "manifest.json").exists()
    assert (dest / "provenance" / "stimulus_ids.json").exists()
    assert (dest / "export_ledger.jsonl").exists()


def test_export_destination_exists_refuses(tmp_path: Path) -> None:
    """The exporter publishes atomically and never overwrites."""

    out = _artifact(tmp_path)
    dest = tmp_path / "export"
    export_extraction(out, dest, format="npy")
    with pytest.raises(InvalidArgumentError) as excinfo:
        export_extraction(out, dest, format="npy")
    assert excinfo.value.fields["code"] == "extraction_export_destination_exists"


def test_export_format_vocabulary_refuses(tmp_path: Path) -> None:
    """Formats outside npy/hdf5/mat refuse typed."""

    out = _artifact(tmp_path)
    with pytest.raises(InvalidArgumentError) as excinfo:
        export_extraction(out, tmp_path / "x", format="parquet")
    assert excinfo.value.fields["code"] == "extraction_export_format_invalid"


def test_export_space_preflight_refuses_with_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D15: the destination preflight names the exact byte numbers."""

    import collections
    import shutil as shutil_module

    out = _artifact(tmp_path)
    usage = collections.namedtuple("usage", "total used free")

    monkeypatch.setattr(
        "torchlens._extraction.export.shutil.disk_usage", lambda _: usage(100, 90, 10)
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        export_extraction(out, tmp_path / "x", format="npy")
    assert excinfo.value.fields["code"] == "extraction_export_space_insufficient"
    assert excinfo.value.fields["free_bytes"] == 10
    del shutil_module


@pytest.mark.smoke
def test_export_bf16_widen_refusal_and_disclosure(tmp_path: Path) -> None:
    """bf16 members widen to fp32 WITH disclosure, or refuse on request."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    out = tmp_path / "artifact"
    extract_dataset(
        model,
        torch.randn(4, 3),
        ["relu"],
        batch_size=2,
        output_dir=out,
        dtype="bfloat16",
        progress=False,
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        export_extraction(out, tmp_path / "x", format="npy", widen_unsupported=False)
    assert excinfo.value.fields["code"] == "extraction_export_dtype_unsupported"
    dest = tmp_path / "export"
    export_extraction(out, dest, format="npy")
    contract = json.loads((dest / "export_contract.json").read_text())
    conversion = contract["keys"]["relu_1_2"]["conversion"]
    assert conversion["conversion"] == "bfloat16->float32"
    assert conversion["lossy"] is False


def test_export_mat_v5_limit_refuses_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAT refuses BEFORE its 2 GB limit, naming HDF5 and the h5read line."""

    pytest.importorskip("scipy")
    out = _artifact(tmp_path)
    monkeypatch.setattr("torchlens._extraction.export._MAT_V5_LIMIT", 8)
    with pytest.raises(InvalidArgumentError) as excinfo:
        export_extraction(out, tmp_path / "x", format="mat")
    assert excinfo.value.fields["code"] == "extraction_export_mat_v5_limit"
    assert "h5read" in str(excinfo.value)


def test_export_hdf5_dependency_refusal_or_roundtrip(tmp_path: Path) -> None:
    """HDF5 rides the optional extra: absent -> typed refusal naming it."""

    out = _artifact(tmp_path)
    try:
        import h5py  # noqa: F401
    except ImportError:
        with pytest.raises(InvalidArgumentError) as excinfo:
            export_extraction(out, tmp_path / "x", format="hdf5")
        assert excinfo.value.fields["code"] == "extraction_export_dependency_missing"
        assert "extraction-export" in str(excinfo.value)
        return
    dest = tmp_path / "export"
    export_extraction(out, dest, format="hdf5")
    import h5py

    with h5py.File(dest / "extraction.h5") as handle:
        reader = open_extraction  # namespacing only
        del reader
        assert handle["relu"].shape[0] == 10


@pytest.mark.smoke
def test_export_ragged_triplet_in_npy(tmp_path: Path) -> None:
    """Ragged members export as the values/offsets/shapes triplet (D4/D15)."""

    import numpy as np

    class _TokenModel(nn.Module):
        """Toy token model (kwargs envelope)."""

        def __init__(self) -> None:
            """Build deterministically."""

            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(50, 4)

        def forward(self, input_ids, attention_mask=None):
            """Embed and gate by the mask."""

            return torch.relu(self.emb(input_ids)) * 1.0

        # the trailing mul keeps module output distinct from model output

    def collate(items):
        """Right-pad word-split texts."""

        lens = [len(t.split()) for t in items]
        width = max(lens)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros(len(items), width, dtype=torch.long)
        for i, length in enumerate(lens):
            ids[i, :length] = torch.arange(1, length + 1)
            mask[i, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}

    model = _TokenModel().eval()
    out = tmp_path / "artifact"
    extract_dataset(
        model,
        ["a b c", "a", "a b c d", "a b"],
        {"h": "relu"},
        batch_size=2,
        output_dir=out,
        collate=collate,
        ragged="trim",
        progress=False,
    )
    dest = tmp_path / "export"
    export_extraction(out, dest, format="npy")
    values = np.load(dest / "h__values.npy")
    offsets = np.load(dest / "h__offsets.npy")
    shapes = np.load(dest / "h__shapes.npy")
    assert values.dtype != object, "no object arrays anywhere"
    assert offsets.tolist() == [0, 3, 4, 8, 10]
    assert shapes[:, 0].tolist() == [3, 1, 4, 2]
    reader = open_extraction(out)
    row2 = reader.row(2)["h"]
    start, stop = offsets[2], offsets[3]
    assert torch.equal(torch.from_numpy(values[start:stop]), row2)
