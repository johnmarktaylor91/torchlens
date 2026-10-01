"""W051-EXTRACT: one frozen layout per key, on the writer AND the reader (audit 1.3 / 2.10b).

A key frozen DENSE at batch zero under ``ragged="trim"`` used to be late-admitted
to the trimmed set when a later batch drifted, while the manifest layout stayed
``dense``; ``materialize()``/``load_extraction`` then concatenated the dense
shards and OVERWROTE them with the merged trimmed carriers, returning a subset
of the rows on a ``status=complete`` artifact. Resume re-derived the ragged
gate from an empty run state, so a continuation could commit differently shaped
shards under one key. Every path now refuses typed or rehydrates from the
manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._errors import InvalidArgumentError
from torchlens._extraction.ragged import RaggedBatch
from torchlens._extraction.reader import open_extraction
from torchlens.dataset_extraction import (
    DatasetExtractionResumeError,
    extract_dataset,
    load_extraction,
)

pytestmark = pytest.mark.smoke


class _Tok(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(50, 6)
        self.proj = nn.Linear(6, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, input_ids, attention_mask=None):  # type: ignore[no-untyped-def]
        return self.head(self.proj(self.emb(input_ids)))


_SEQS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17, 18], [19, 20, 21]]


def _collate(items: list[list[int]], *, mask_always: bool) -> dict[str, torch.Tensor]:
    width = max(len(seq) for seq in items)
    ids = torch.zeros(len(items), width, dtype=torch.long)
    mask = torch.zeros(len(items), width, dtype=torch.long)
    for row, seq in enumerate(items):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
    if not mask_always and all(len(seq) == width for seq in items):
        return {"input_ids": ids}
    return {"input_ids": ids, "attention_mask": mask}


def _mask_omitting_collate(items: list[list[int]]) -> dict[str, torch.Tensor]:
    return _collate(items, mask_always=False)


def _masked_collate(items: list[list[int]]) -> dict[str, torch.Tensor]:
    return _collate(items, mask_always=True)


class _Die(Exception):
    pass


def _dying(items: list, stop_at: int):  # type: ignore[no-untyped-def]
    for index, item in enumerate(items):
        if index == stop_at:
            raise _Die()
        yield item


def test_late_mask_shaped_drift_refuses_instead_of_late_admitting(tmp_path: Path) -> None:
    """p5_mixed_layout: batch zero equal-width without a mask, batch one ragged."""

    torch.manual_seed(0)
    model = _Tok().eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            iter(_SEQS),
            {"h": "proj"},
            batch_size=2,
            output_dir=tmp_path,
            progress=False,
            collate=_mask_omitting_collate,
            ragged="trim",
        )
    fields = excinfo.value.fields
    assert fields["code"] == "extraction_ragged_refused"
    assert fields["late_mask_shaped"] is True
    assert fields["batch_index"] == 1
    assert fields["planned_shape"] == [4, 6]
    assert "EVERY batch" in str(excinfo.value)
    # Refused BEFORE the offending shard committed: exactly the dense batch-zero shard.
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 1
    assert "layout" not in ledger[0]["keys"]["h"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["layers"]["h"]["layout"] == "dense"
    assert manifest["status"] == "in_progress"


def test_mask_shaped_from_batch_zero_stores_trimmed_and_reads_every_row(tmp_path: Path) -> None:
    """The remedy works: a mask on every batch stores the key trimmed, all rows served."""

    torch.manual_seed(0)
    model = _Tok().eval()
    extract_dataset(
        model,
        iter(_SEQS),
        {"h": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        collate=_masked_collate,
        ragged="trim",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["layers"]["h"]["layout"] == "trimmed"
    assert manifest["layers"]["h"]["pooled_per_stimulus_shape"] == [None, 6]
    reader = open_extraction(tmp_path)
    carrier = reader.materialize()["h"]
    assert isinstance(carrier, RaggedBatch)
    assert carrier.row_count == len(_SEQS) == reader.n_stimuli
    loaded = load_extraction(tmp_path).activations["h"]
    assert isinstance(loaded, RaggedBatch) and loaded.row_count == len(_SEQS)


def test_reader_refuses_manifest_layout_contradicting_the_ledger(tmp_path: Path) -> None:
    """A mixed artifact (the pre-fix writer's product) refuses at open, not after dropping rows."""

    torch.manual_seed(0)
    model = _Tok().eval()
    extract_dataset(
        model,
        iter(_SEQS),
        {"h": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        collate=_masked_collate,
        ragged="trim",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"]["h"]["layout"] = "dense"  # what the late-admission bug froze
    manifest_path.write_text(json.dumps(manifest))
    for door in (open_extraction, load_extraction):
        with pytest.raises(DatasetExtractionResumeError) as excinfo:
            door(tmp_path)
        fields = excinfo.value.fields
        assert fields["code"] == "extraction_manifest_invalid"
        assert fields["key"] == "h"
        assert fields["declared_layout"] == "dense"
        assert fields["ledgered_layout"] == "trimmed"
        assert fields["shard"] == "batch_00000.safetensors"


def test_materialize_belt_refuses_a_key_in_both_layouts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Even past the open-time check, a key seen dense AND trimmed never concatenates a subset."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    extract_dataset(
        model,
        torch.randn(4, 3),
        {"h": "relu_1_2"},
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
    )
    reader = open_extraction(tmp_path)
    dense_payloads = list(reader.iter_batches())
    values = dense_payloads[1]["h"]
    forged = RaggedBatch(values=values, offsets=torch.tensor([0, 1, 2]), row_shapes=[[4], [4]])
    monkeypatch.setattr(
        reader, "iter_batches", lambda keys=None: iter([dense_payloads[0], {"h": forged}])
    )
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        reader.materialize()
    assert excinfo.value.fields["code"] == "extraction_manifest_invalid"
    assert excinfo.value.fields["mixed_layout_keys"] == ["h"]


class _Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv1(x))).flatten(1))


def _shape_drifting_items() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(3, 8, 8),
        torch.randn(3, 8, 8),
        torch.randn(3, 16, 16),
        torch.randn(3, 16, 16),
    ]


def test_resume_rehydrates_the_ragged_gate_from_the_manifest(tmp_path: Path) -> None:
    """p6_resume_shape: interrupt-then-resume refuses exactly like the single run."""

    items = _shape_drifting_items()
    model = _Conv().eval()
    control_dir = tmp_path / "control"
    with pytest.raises(InvalidArgumentError) as control:
        extract_dataset(
            model,
            iter(items),
            {"c1": "conv1"},
            batch_size=1,
            output_dir=control_dir,
            progress=False,
        )
    assert control.value.fields["code"] == "extraction_ragged_refused"

    out = tmp_path / "resumed"
    with pytest.raises(_Die):
        extract_dataset(
            model, _dying(items, 2), {"c1": "conv1"}, batch_size=1, output_dir=out, progress=False
        )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"]["c1"]["pooled_per_stimulus_shape"] == [4, 8, 8]
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            iter(items),
            {"c1": "conv1"},
            batch_size=1,
            output_dir=out,
            progress=False,
            resume=True,
        )
    fields = excinfo.value.fields
    assert fields["code"] == "extraction_ragged_refused"
    assert fields["planned_shape"] == [4, 8, 8]
    assert fields["observed_shape"] == [4, 16, 16]
    assert fields["batch_index"] == 2
    assert json.loads((out / "manifest.json").read_text())["status"] == "in_progress"
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [row["keys"]["c1"]["per_stimulus_shape"] for row in ledger] == [[4, 8, 8], [4, 8, 8]]


def test_resume_rehydrates_from_stored_shape_on_a_manifest_without_pooled_shape(
    tmp_path: Path,
) -> None:
    """Older manifests (no pooled shape record) still gate: stored == pooled with no transform."""

    items = _shape_drifting_items()
    model = _Conv().eval()
    with pytest.raises(_Die):
        extract_dataset(
            model,
            _dying(items, 2),
            {"c1": "conv1"},
            batch_size=1,
            output_dir=tmp_path,
            progress=False,
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["layers"]["c1"]["pooled_per_stimulus_shape"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            iter(items),
            {"c1": "conv1"},
            batch_size=1,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_ragged_refused"
    assert excinfo.value.fields["planned_shape"] == [4, 8, 8]


def test_resume_continues_a_trimmed_key_trimmed(tmp_path: Path) -> None:
    """The trimmed-key set rehydrates too: a continuation never writes dense shards under it."""

    torch.manual_seed(0)
    model = _Tok().eval()
    with pytest.raises(_Die):
        extract_dataset(
            model,
            _dying(_SEQS, 4),
            {"h": "proj"},
            batch_size=2,
            output_dir=tmp_path,
            progress=False,
            collate=_masked_collate,
            ragged="trim",
        )
    extract_dataset(
        model,
        iter(_SEQS),
        {"h": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        collate=_masked_collate,
        ragged="trim",
        resume=True,
    )
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["keys"]["h"]["layout"] for row in ledger] == ["trimmed"] * 3
    reader = open_extraction(tmp_path)
    carrier = reader.materialize()["h"]
    assert isinstance(carrier, RaggedBatch) and carrier.row_count == len(_SEQS)
    # Row-exact against a clean, uninterrupted trim run.
    clean_dir = tmp_path / "clean"
    extract_dataset(
        model,
        iter(_SEQS),
        {"h": "proj"},
        batch_size=2,
        output_dir=clean_dir,
        progress=False,
        collate=_masked_collate,
        ragged="trim",
    )
    clean = open_extraction(clean_dir).materialize()["h"]
    assert torch.equal(carrier.values, clean.values)
    assert torch.equal(carrier.offsets, clean.offsets)
