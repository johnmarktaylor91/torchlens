"""Extraction v2 engine rows: the D13 store pipeline end to end (lane F18).

Covers the manifested pipeline order on real (toy-scale) runs: input-derived
row truth (T-ROWCOUNT), pool numerics against manual oracles, dtype routing
with byte-exact bf16/fp8 storage (T-FP8-BF16), the D4 ragged gate
(T-WIDTH / T-DRIFT-PREFIX / T-TRIM / T-PADSIDE), warning dedup counts,
ledger geometry facts, and the item-16 resume hardening (input-digest
replay, callable mismatch, v1 acknowledgment).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._data_substrate import read_trusted_rows
from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import (
    BatchEnvelope,
    DatasetExtractionResumeError,
    extract_dataset,
    open_extraction,
)
from torchlens.utils._torch_compat import get_cpu_float8_deterministic_fill_support

pytestmark = pytest.mark.smoke


class _TokenModel(nn.Module):
    """Toy token model taking the HF-shaped kwargs envelope."""

    def __init__(self) -> None:
        """Build the embedding + projection stack deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(50, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        """Embed, optionally offset by positions, and project."""

        hidden = self.emb(input_ids)
        if position_ids is None:
            # Real absolute-position models default to arange positions —
            # exactly why left padding without derived positions is wrong.
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        hidden = hidden + position_ids.unsqueeze(-1).float() * 0.01
        # The trailing mul keeps the proj module output distinct from the
        # model output op, so the "proj" module-path lookup stays unique.
        return self.proj(torch.relu(hidden)) * 1.0


_TEXTS = ["a b c", "a", "a b c d e", "a b"]


def _word_collate(items, *, left: bool = False):
    """Collate word-split texts into an HF-shaped mapping batch.

    Parameters
    ----------
    items:
        Text items.
    left:
        Whether to left-pad (the D5 hazard geometry).
    """

    lens = [len(text.split()) for text in items]
    width = max(lens)
    ids = torch.zeros(len(items), width, dtype=torch.long)
    mask = torch.zeros(len(items), width, dtype=torch.long)
    for i, length in enumerate(lens):
        if left:
            ids[i, width - length :] = torch.arange(1, length + 1)
            mask[i, width - length :] = 1
        else:
            ids[i, :length] = torch.arange(1, length + 1)
            mask[i, :length] = 1
    return {"input_ids": ids, "attention_mask": mask}


def _collate_right(items):
    """Right-padded word collate."""

    return _word_collate(items)


def _collate_left(items):
    """Left-padded word collate (the D5 hazard)."""

    return _word_collate(items, left=True)


# --- row truth (item 4 / T-ROWCOUNT) -------------------------------------------


def test_row_count_is_input_derived_and_refuses_drift(tmp_path: Path) -> None:
    """A collate lying about row_count refuses BEFORE any shard commits."""

    def lying_collate(items):
        """Claim one extra row."""

        return BatchEnvelope(
            args=(torch.stack(items),),
            kwargs={},
            row_count=len(items) + 1,
            mask=None,
            disclosure={"kind": "test"},
        )

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            [torch.randn(3) for _ in range(4)],
            ["relu"],
            batch_size=2,
            output_dir=tmp_path,
            collate=lying_collate,
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_row_count_mismatch"
    assert not list(tmp_path.glob("batch_*")), "refused BEFORE the first commit"


# --- pool numerics (item 8) -------------------------------------------------------


def test_pool_presets_match_manual_oracles(tmp_path: Path) -> None:
    """token_mean / last_token / cls match hand-computed masked reductions."""

    model = _TokenModel().eval()
    encoded = _collate_right(_TEXTS)
    with torch.no_grad():
        hidden = model(encoded["input_ids"], encoded["attention_mask"])
    mask = encoded["attention_mask"].bool()

    results = {}
    for preset in ("token_mean", "last_token", "cls"):
        out = extract_dataset(
            model,
            _TEXTS,
            {"h": "proj"},
            batch_size=4,
            collate=_collate_right,
            pool=preset,
            progress=False,
        )
        results[preset] = out["h"]

    manual_mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    last = mask.long().cumsum(1).argmax(1)
    manual_last = hidden[torch.arange(4), last]
    assert torch.allclose(results["token_mean"], manual_mean, atol=1e-6)
    assert torch.allclose(results["last_token"], manual_last, atol=1e-6)
    assert torch.allclose(results["cls"], hidden[:, 0], atol=1e-6)


def test_pool_kills_raggedness_across_batches(tmp_path: Path) -> None:
    """The documented raggedness-killer: pooled text harvests stay dense."""

    model = _TokenModel().eval()
    paths = extract_dataset(
        model,
        _TEXTS,
        {"h": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        collate=_collate_right,
        pool="token_mean",
        progress=False,
    )
    assert len(paths) == 2
    reader = open_extraction(tmp_path)
    assert reader.logical_shape("h") == [8]


# --- dtype routing (item 9 / T-FP8-BF16) --------------------------------------------


@pytest.mark.parametrize(
    "dtype_name",
    [
        "bfloat16",
        pytest.param(
            "float8_e4m3fn",
            marks=pytest.mark.skipif(
                not get_cpu_float8_deterministic_fill_support(),
                reason="CPU Float8 empty-fill under deterministic mode postdates the torch 2.1 floor",
            ),
        ),
        pytest.param(
            "float8_e5m2",
            marks=pytest.mark.skipif(
                not get_cpu_float8_deterministic_fill_support(),
                reason="CPU Float8 empty-fill under deterministic mode postdates the torch 2.1 floor",
            ),
        ),
    ],
)
def test_bf16_fp8_store_byte_exact_roundtrip(tmp_path: Path, dtype_name: str) -> None:
    """bf16/fp8 shards round-trip BYTE-EXACT through safetensors (D11)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    stimuli = torch.randn(4, 3)
    extract_dataset(
        model,
        stimuli,
        ["relu"],
        batch_size=4,
        output_dir=tmp_path,
        dtype=dtype_name,
        progress=False,
    )
    reader = open_extraction(tmp_path)
    stored = reader.materialize()["relu_1_2"]
    with torch.no_grad():
        expected = model(stimuli).to(getattr(torch, dtype_name))
    assert stored.dtype == getattr(torch, dtype_name)
    assert torch.equal(stored.view(torch.uint8), expected.view(torch.uint8))
    layer = reader.manifest["layers"]["relu_1_2"]
    assert layer["dtype_conversion"]["stored_dtype"] == f"torch.{dtype_name}"


# --- ragged gate (item 10) -----------------------------------------------------------


def test_ragged_refuse_preserves_only_committed_prefix(tmp_path: Path) -> None:
    """T-DRIFT-PREFIX: the refusal fires at batch 1; shard 0 stays trusted."""

    model = _TokenModel().eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            _TEXTS,
            {"h": "proj"},
            batch_size=2,
            output_dir=tmp_path,
            collate=_collate_right,
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_ragged_refused"
    rows = read_trusted_rows(tmp_path)
    assert len(rows) == 1, "exactly the committed prefix survives"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "in_progress"


def test_trim_rows_equal_solo_forwards_under_left_padding(tmp_path: Path) -> None:
    """T-TRIM x left padding: (start, extent) slicing reads TOKEN rows exactly.

    Left padding shifts each row's first valid index (extent alone would
    mis-slice), and with derived position_ids the trimmed rows must match
    per-string solo forwards.
    """

    model = _TokenModel().eval()
    with pytest.warns(UserWarning, match="derived position_ids"):
        extract_dataset(
            model,
            _TEXTS,
            {"h": "proj"},
            batch_size=4,
            output_dir=tmp_path,
            collate=_collate_left,
            ragged="trim",
            progress=False,
        )
    reader = open_extraction(tmp_path)
    for index, text in enumerate(_TEXTS):
        solo = _collate_right([text])
        with torch.no_grad():
            expected = model(solo["input_ids"])[0]
        row = reader.row(index)["h"]
        assert row.shape == expected.shape
        assert torch.allclose(row, expected, atol=1e-5), f"row {index} mis-sliced"
    rows = read_trusted_rows(tmp_path)
    geometry = rows[0]["row_geometry"]
    assert geometry["padding_side"] == "left"
    assert geometry["starts"] != [0] * len(geometry["starts"]), "left pads shift starts"


def test_row_geometry_ledgered_regardless_of_regime(tmp_path: Path) -> None:
    """T-PADSIDE: (start, extent) + padding_side ledger on DENSE artifacts too."""

    model = _TokenModel().eval()
    extract_dataset(
        model,
        _TEXTS[:2],
        {"h": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        collate=_collate_right,
        pool="token_mean",
        progress=False,
    )
    rows = read_trusted_rows(tmp_path)
    geometry = rows[0]["row_geometry"]
    assert geometry["extents"] == [3, 1]
    assert geometry["starts"] == [0, 0]
    assert geometry["padding_side"] == "right"
    assert rows[0]["position_ids_source"] == "model_default"


# --- warning dedup (item 5) ------------------------------------------------------------


def test_capture_warnings_deduplicate_with_manifest_counts(tmp_path: Path) -> None:
    """D9: repeated capture warnings surface once and count in the manifest."""

    model = _TokenModel().eval()
    with pytest.warns(UserWarning, match="derived position_ids") as record:
        extract_dataset(
            model,
            _TEXTS,
            {"h": "proj"},
            batch_size=2,
            output_dir=tmp_path,
            collate=_collate_left,
            pool="token_mean",
            progress=False,
        )
    derivation_warnings = [
        entry for entry in record if "derived position_ids" in str(entry.message)
    ]
    assert len(derivation_warnings) == 1, "disclosed once per run, both batches pad"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    warning_counts = manifest["run"]["warnings"]
    assert any("derived position_ids" in key for key in warning_counts)
    assert manifest["run"]["position_ids_sources"] == ["derived"]


# --- stimulus ids at completion (item 7) ---------------------------------------------------


def test_surplus_iterable_ids_refuse_before_completion(tmp_path: Path) -> None:
    """D2: exact ID equality is checked BEFORE completion, never after."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()

    def gen():
        """Yield three stimuli against four supplied ids."""

        for _ in range(3):
            yield torch.randn(3)

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            gen(),
            ["relu"],
            batch_size=2,
            output_dir=tmp_path,
            stimulus_ids=["a", "b", "c", "d"],
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_stimulus_ids_cardinality"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "in_progress", "a mismatch NEVER completes"


def test_duplicate_ids_are_legal_and_recorded(tmp_path: Path) -> None:
    """D2: duplicates are legal with ids_unique recorded false."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    extract_dataset(
        model,
        torch.randn(4, 3),
        ["relu"],
        batch_size=2,
        output_dir=tmp_path,
        stimulus_ids=["a", "b", "a", "c"],
        progress=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stimulus_provenance"]["ids_unique"] is False


# --- resume hardening (item 16) ---------------------------------------------------------


def _interrupt_after(tmp_path: Path, n_shards: int) -> None:
    """Reconstruct a mid-run artifact with ``n_shards`` committed."""

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "in_progress"
    manifest["totals"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    rows = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("".join(line + "\n" for line in rows[:n_shards]), encoding="utf-8")
    for line in rows[n_shards:]:
        (tmp_path / json.loads(line)["file"]).unlink()


def test_resume_input_digest_catches_reordered_iterable(tmp_path: Path) -> None:
    """Item 16: the skip-replay verifies the prefix through input_digest."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    rows = [torch.arange(3, dtype=torch.float32) + i for i in range(6)]
    extract_dataset(model, rows, ["relu"], batch_size=2, output_dir=tmp_path, progress=False)
    _interrupt_after(tmp_path, 1)
    reordered = [rows[1], rows[0]] + rows[2:]
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        extract_dataset(
            model,
            reordered,
            ["relu"],
            batch_size=2,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_input_mismatch"


def test_resume_complete_class_callable_digest_mismatch_refuses(tmp_path: Path) -> None:
    """D8: two COMPLETE-class callables with different digests refuse."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()

    def scale_two(tensor):
        """Scale by 2."""

        return tensor * 2.0

    def scale_three(tensor):
        """Scale by 3 (differs only in one constant)."""

        return tensor * 3.0

    extract_dataset(
        model,
        torch.randn(6, 3),
        ["relu"],
        batch_size=2,
        output_dir=tmp_path,
        transform=scale_two,
        progress=False,
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        extract_dataset(
            model,
            torch.randn(6, 3),
            ["relu"],
            batch_size=2,
            output_dir=tmp_path,
            transform=scale_three,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_callable_mismatch"


def test_resume_policy_matrix_names_exact_fields(tmp_path: Path) -> None:
    """Composition row: changing pool/dtype/ragged on resume names the field."""

    model = _TokenModel().eval()
    extract_dataset(
        model,
        _TEXTS,
        {"h": "proj"},
        batch_size=4,
        output_dir=tmp_path,
        collate=_collate_right,
        pool="token_mean",
        progress=False,
    )
    _interrupt_after(tmp_path, 0)
    for change, field in [
        ({"pool": "token_max"}, "pool"),
        ({"pool": "token_mean", "dtype": "float16"}, "dtype_policy"),
        ({"pool": "token_mean", "ragged": "trim"}, "ragged"),
    ]:
        with pytest.raises(DatasetExtractionResumeError) as excinfo:
            extract_dataset(
                model,
                _TEXTS,
                {"h": "proj"},
                batch_size=4,
                output_dir=tmp_path,
                collate=_collate_right,
                progress=False,
                resume=True,
                **change,
            )
        assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
        assert field in excinfo.value.fields["mismatched_fields"]
