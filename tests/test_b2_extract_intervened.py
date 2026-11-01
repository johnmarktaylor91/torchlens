"""Intervened dataset extraction (R2-08, lane EXTRACT).

``extract_dataset(..., intervene=spec)`` stores the INTERVENED activations:
on a small chain model a zero-ablation of one site leaves every upstream key
bit-equal to plain extraction, zeroes the ablated site, and changes the
downstream key, and the artifact records the counterfactual (harvest block,
per-shard fire receipts, resume-compared identity).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import extract_dataset
from torchlens._data_substrate import read_trusted_rows
from torchlens._errors import InvalidArgumentError
from torchlens._extraction.harvest_schema import HARVEST_BLOCK_KEY, validate_harvest_block
from torchlens._extraction.resume import MANIFEST_FILENAME
from torchlens.dataset_extraction import open_extraction
from torchlens.errors import DatasetExtractionResumeError

_LAYERS = {"a": "fc1", "b": "act", "c": "fc2", "d": "fc3"}


class _Chain(nn.Module):
    """Four-site chain: fc1 -> relu -> fc2 -> fc3."""

    def __init__(self) -> None:
        """Build the chain deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(6, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain; the trailing mul keeps fc3 distinct from the output."""

        return self.fc3(self.fc2(self.act(self.fc1(x)))) * 1.0


def _stimuli() -> torch.Tensor:
    """Ten deterministic stimuli."""

    generator = torch.Generator().manual_seed(1)
    return torch.randn(10, 6, generator=generator)


def _spec() -> tl.intervention.spec.InterventionSpec:
    """Zero-ablate the fc2 module output."""

    return tl.when(tl.module("fc2"), tl.zero_ablate())


def _manifest(path: Path) -> dict:
    """Read an artifact manifest."""

    return json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def test_intervened_differs_exactly_where_the_intervention_acts() -> None:
    """Upstream keys bit-equal, the ablated site zero, downstream changed."""

    model = _Chain().eval()
    clean = extract_dataset(model, _stimuli(), _LAYERS, batch_size=4, progress=False)
    edited = extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, progress=False, intervene=_spec()
    )
    assert torch.equal(edited["a"], clean["a"])
    assert torch.equal(edited["b"], clean["b"])
    assert torch.count_nonzero(edited["c"]) == 0
    assert not torch.equal(clean["c"], edited["c"])
    assert not torch.equal(clean["d"], edited["d"])
    expected_d = model.fc3.bias.detach().expand_as(edited["d"])
    assert torch.allclose(edited["d"], expected_d)


def test_intervened_disk_artifact_records_the_counterfactual(tmp_path: Path) -> None:
    """Harvest block, per-shard receipts, and the frozen sites are on disk."""

    model = _Chain().eval()
    spec = _spec()
    extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        path=tmp_path,
        progress=False,
        intervene=spec,
    )
    manifest = _manifest(tmp_path)
    harvest = validate_harvest_block(manifest[HARVEST_BLOCK_KEY])
    assert harvest.policy == "intervened"
    assert harvest.engine == "live_hook"
    assert harvest.address_class == "module"
    assert harvest.target_site_keys
    assert str(harvest.receipt_digest).startswith("sha256:")
    run_record = manifest["run"]["intervention"]
    assert run_record["spec_digest"] == spec.digest
    assert run_record["target_site_keys"] == list(harvest.target_site_keys)
    assert manifest["signature"]["model_mode"]["intervention"]["policy"] == "intervened"
    rows = read_trusted_rows(tmp_path)
    assert len(rows) == 3
    for row in rows:
        assert row["intervention"]["fire_count"] >= 1
        assert row["intervention"]["site_keys"] == list(harvest.target_site_keys)
    reader = open_extraction(tmp_path)
    assert torch.count_nonzero(reader.materialize(["c"])["c"]) == 0


def test_clean_disk_artifact_declares_policy_none(tmp_path: Path) -> None:
    """A clean harvest says so explicitly (never "written before the schema")."""

    extract_dataset(
        _Chain().eval(), _stimuli(), _LAYERS, batch_size=4, path=tmp_path, progress=False
    )
    manifest = _manifest(tmp_path)
    assert validate_harvest_block(manifest[HARVEST_BLOCK_KEY]).policy == "none"
    assert manifest["signature"]["model_mode"] == "eval_no_grad"
    assert all("intervention" not in row for row in read_trusted_rows(tmp_path))


def test_op_type_address_class_runs() -> None:
    """A ``tl.func`` (op-type class) spec executes and edits its site."""

    model = _Chain().eval()
    edited = extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=5,
        progress=False,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    assert torch.count_nonzero(edited["b"]) == 0


def _interrupt_after(path: Path, n_shards: int) -> None:
    """Reconstruct the on-disk state of a kill after ``n_shards`` commits."""

    manifest = _manifest(path)
    manifest["status"] = "in_progress"
    manifest["totals"] = None
    manifest["ledger_digest"] = None
    (path / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    ledger = path / "ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("".join(line + "\n" for line in lines[:n_shards]), encoding="utf-8")
    for line in lines[n_shards:]:
        (path / json.loads(line)["file"]).unlink()


def test_intervened_resume_continues_and_refuses_a_changed_intervention(tmp_path: Path) -> None:
    """Same spec resumes to identical rows; another spec or a clean run refuses."""

    model = _Chain().eval()
    kwargs = {"batch_size": 2, "path": tmp_path, "progress": False}
    extract_dataset(model, _stimuli(), _LAYERS, intervene=_spec(), **kwargs)
    full = open_extraction(tmp_path).materialize()
    receipt_digest = _manifest(tmp_path)[HARVEST_BLOCK_KEY]["receipt_digest"]
    _interrupt_after(tmp_path, 2)
    extract_dataset(model, _stimuli(), _LAYERS, intervene=_spec(), resume=True, **kwargs)
    resumed = open_extraction(tmp_path).materialize()
    for key, tensor in full.items():
        assert torch.equal(resumed[key], tensor)
    assert _manifest(tmp_path)[HARVEST_BLOCK_KEY]["receipt_digest"] == receipt_digest

    _interrupt_after(tmp_path, 2)
    other = tl.when(tl.module("fc1"), tl.zero_ablate())
    for intervene in (other, None):
        with pytest.raises(DatasetExtractionResumeError) as excinfo:
            extract_dataset(model, _stimuli(), _LAYERS, intervene=intervene, resume=True, **kwargs)
        assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
        assert excinfo.value.fields["mismatched_fields"] == ["model_mode"]


def test_intervene_refusals_are_typed() -> None:
    """Non-spec values, finer address classes, and coupling helpers refuse."""

    model = _Chain().eval()
    cases = [
        (lambda out, ctx: out, "extraction_intervene_invalid"),
        (tl.when(tl.label("relu_1_2"), tl.zero_ablate()), "harvest_address_class_ungated"),
        (tl.when(tl.module("fc2"), tl.mean_ablate()), "extraction_intervention_batch_dependent"),
        (
            tl.when(tl.module("fc2"), lambda out, ctx: out * 0),
            "extraction_intervention_batch_dependent",
        ),
    ]
    for intervene, code in cases:
        with pytest.raises(InvalidArgumentError) as excinfo:
            extract_dataset(model, _stimuli(), _LAYERS, progress=False, intervene=intervene)
        assert excinfo.value.fields["code"] == code


def test_intervene_with_selector_layers_refuses() -> None:
    """Selector-valued layers= plus intervene= refuses typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _Chain().eval(), _stimuli(), tl.func("relu"), progress=False, intervene=_spec()
        )
    assert excinfo.value.fields["code"] == "extraction_intervened_selector_layers_unsupported"


def test_a_batch_without_a_fire_refuses() -> None:
    """A spec that matches no site on a batch refuses (no receipt, no record)."""

    with warnings.catch_warnings(), pytest.raises(InvalidArgumentError) as excinfo:
        # The capture door's own zero-match warning precedes the refusal.
        warnings.simplefilter("ignore", UserWarning)
        extract_dataset(
            _Chain().eval(),
            _stimuli(),
            _LAYERS,
            progress=False,
            intervene=tl.when(tl.func("sigmoid"), tl.zero_ablate()),
        )
    assert excinfo.value.fields["code"] == "extraction_intervention_no_fire"


class _ShapeBranch(nn.Module):
    """Takes a second relu only on short batches (a structural branch)."""

    def __init__(self) -> None:
        """Build the site deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(6, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One relu on full batches, two on the short tail batch."""

        hidden = torch.relu(self.fc1(x))
        if x.shape[0] < 4:
            hidden = torch.relu(hidden * 2.0)
        return hidden * 1.0


def test_a_batch_firing_at_other_sites_refuses() -> None:
    """The short tail batch fires at a second relu: one artifact, one counterfactual."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _ShapeBranch().eval(),
            _stimuli(),
            {"h": "fc1"},
            batch_size=4,
            progress=False,
            intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        )
    assert excinfo.value.fields["code"] == "extraction_intervention_site_drift"
    assert excinfo.value.fields["batch_index"] == 2


def test_pre_r208_manifest_still_loads_and_resumes(tmp_path: Path) -> None:
    """A manifest without the harvest block (older writers) opens and resumes clean."""

    model = _Chain().eval()
    kwargs = {"batch_size": 2, "path": tmp_path, "progress": False}
    extract_dataset(model, _stimuli(), _LAYERS, **kwargs)
    full = open_extraction(tmp_path).materialize()
    manifest = _manifest(tmp_path)
    del manifest[HARVEST_BLOCK_KEY]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert torch.equal(open_extraction(tmp_path).materialize()["d"], full["d"])
    _interrupt_after(tmp_path, 2)
    assert HARVEST_BLOCK_KEY not in _manifest(tmp_path)
    extract_dataset(model, _stimuli(), _LAYERS, resume=True, **kwargs)
    resumed = open_extraction(tmp_path).materialize()
    assert all(torch.equal(resumed[key], full[key]) for key in full)


def test_malformed_harvest_block_refuses_typed(tmp_path: Path) -> None:
    """A torn or foreign counterfactual block never half-loads."""

    extract_dataset(
        _Chain().eval(),
        _stimuli(),
        _LAYERS,
        batch_size=4,
        path=tmp_path,
        progress=False,
        intervene=_spec(),
    )
    manifest = _manifest(tmp_path)
    manifest[HARVEST_BLOCK_KEY]["receipt_digest"] = None
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InvalidArgumentError) as excinfo:
        open_extraction(tmp_path)
    assert excinfo.value.fields["code"] == "harvest_schema_invalid"
