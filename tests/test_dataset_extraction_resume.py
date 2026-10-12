"""Resumability + self-description contract tests for :func:`torchlens.extract_dataset`.

Covers the artifact-v2 resume mechanism (extract memo D1: append-only fsynced
ledger, commit protocol, field-by-field signature compare, crash-safety proven
via a hard child-process death) and the self-describing bounded manifest (site
identity, stimulus provenance, axis semantics, dtype/device, loader
round-trip).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_substrate import ExtractionArtifactError, read_trusted_rows
from torchlens.dataset_extraction import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_V2,
    DatasetExtractionResumeError,
    load_extraction,
)


def _interrupt_after(tmp_path: Path, n_shards: int) -> None:
    """Reconstruct a mid-run artifact state: ``n_shards`` committed, in progress.

    Truncates the append-only ledger to its first ``n_shards`` rows, deletes
    the later shard files, and reverts the manifest to ``in_progress`` — the
    exact on-disk state a kill between two shard commits leaves behind.

    Parameters
    ----------
    tmp_path:
        Artifact directory.
    n_shards:
        Committed-shard count to keep.
    """

    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "in_progress"
    manifest["totals"] = None
    manifest["ledger_digest"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    rows = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("".join(line + "\n" for line in rows[:n_shards]), encoding="utf-8")
    for line in rows[n_shards:]:
        (tmp_path / json.loads(line)["file"]).unlink()


class _CountingModel(nn.Module):
    """Tiny deterministic model that counts forward invocations."""

    def __init__(self) -> None:
        """Initialize the affine layers and the call counter."""

        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.head = nn.Linear(4, 2)
        self.n_forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the affine stack and count the call.

        Parameters
        ----------
        inputs:
            Batched stimulus rows.

        Returns
        -------
        torch.Tensor
            Model logits.
        """

        self.n_forward_calls += 1
        return self.head(self.relu(self.fc(inputs)))


_LAYERS = {"relu": "relu", "logits": "output_1"}


def _stimuli(n: int = 10) -> torch.Tensor:
    """Return ``n`` distinctive stimulus rows.

    Parameters
    ----------
    n:
        Number of stimuli.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)`` float tensor whose rows identify their index.
    """

    return torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)


def test_resume_requires_output_dir() -> None:
    """``resume=True`` in in-memory mode refuses typed: there is no ledger."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            _CountingModel().eval(), _stimuli(), _LAYERS, batch_size=4, progress=False, resume=True
        )
    assert excinfo.value.fields["code"] == "extraction_resume_requires_output_dir"


def test_resume_refuses_unmanifested_shard_dir(tmp_path: Path) -> None:
    """Shards without a manifest cannot be verified, so resume refuses typed."""

    (tmp_path / "batch_00000.pt").write_bytes(b"opaque")
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            _CountingModel().eval(),
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_unmanifested_dir"


def test_resume_refuses_mismatched_signature(tmp_path: Path) -> None:
    """A resume whose run parameters differ from the artifact's refuses typed."""

    model = _CountingModel().eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=5,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
    assert "batch_size" in excinfo.value.fields["mismatched_fields"]

    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli() + 1.0,
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
    assert "stimuli" in excinfo.value.fields["mismatched_fields"]


def test_resume_of_complete_artifact_never_runs_the_model(tmp_path: Path) -> None:
    """Resuming a finished artifact returns its shard paths without a forward."""

    model = _CountingModel().eval()
    first = tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    calls_after_first = model.n_forward_calls
    again = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
    )
    assert model.n_forward_calls == calls_after_first
    assert again == first


def test_resume_recomputes_only_missing_shards(tmp_path: Path) -> None:
    """An interrupted run resumes from its ledger, recomputing only the tail."""

    model = _CountingModel().eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=2, output_dir=tmp_path, progress=False
    )
    # Reconstruct the mid-run state: 2 of 5 shards ledgered, in progress.
    _interrupt_after(tmp_path, 2)

    calls_before = model.n_forward_calls
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        resume=True,
    )
    assert model.n_forward_calls == calls_before + 3
    assert [path.name for path in paths] == [f"batch_0000{i}.safetensors" for i in range(5)]
    loaded = load_extraction(tmp_path)
    clean = tl.extract_dataset(model, _stimuli(), _LAYERS, batch_size=2, progress=False)
    for key, tensor in clean.items():
        assert torch.equal(loaded.activations[key], tensor)


def test_deleted_ledgered_shard_refuses_typed(tmp_path: Path) -> None:
    """A ledgered shard gone missing ends the trusted prefix TYPED (D1).

    The v1 engine silently "healed" the artifact by recomputing from the
    break; under the append-only v2 ledger a committed row can never be
    rewritten, so a missing or size-mismatched member is a typed refusal on
    BOTH resume and load — never a silent truncation, never a lie.
    """

    model = _CountingModel().eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=2, output_dir=tmp_path, progress=False
    )
    (tmp_path / "batch_00003.safetensors").unlink()
    with pytest.raises(ExtractionArtifactError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=2,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_ledger_prefix_broken"
    with pytest.raises(ExtractionArtifactError) as excinfo:
        load_extraction(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_ledger_prefix_broken"


def test_torn_final_ledger_line_is_dropped_not_fatal(tmp_path: Path) -> None:
    """Crash debris — a torn final ledger line — is never trusted, never fatal."""

    model = _CountingModel().eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=2, output_dir=tmp_path, progress=False
    )
    _interrupt_after(tmp_path, 2)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        ledger_path.read_text(encoding="utf-8") + '{"index": 2, "file": "batch_0',
        encoding="utf-8",
    )
    rows = read_trusted_rows(tmp_path)
    assert [row["index"] for row in rows] == [0, 1]
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        resume=True,
    )
    assert [path.name for path in paths] == [f"batch_0000{i}.safetensors" for i in range(5)]
    assert load_extraction(tmp_path).manifest["status"] == "complete"


def test_resume_with_iterable_stimuli_skips_consumed_prefix(tmp_path: Path) -> None:
    """Iterable stimuli resume by consuming exactly the ledgered prefix."""

    model = _CountingModel().eval()
    rows = list(_stimuli(7))
    tl.extract_dataset(model, rows, _LAYERS, batch_size=3, output_dir=tmp_path, progress=False)
    _interrupt_after(tmp_path, 1)

    calls_before = model.n_forward_calls
    tl.extract_dataset(
        model, rows, _LAYERS, batch_size=3, output_dir=tmp_path, progress=False, resume=True
    )
    assert model.n_forward_calls == calls_before + 2
    loaded = load_extraction(tmp_path)
    clean = tl.extract_dataset(model, _stimuli(7), _LAYERS, batch_size=3, progress=False)
    assert torch.equal(loaded.activations["relu"], clean["relu"])

    # A stimulus stream shorter than the ledger is a signature violation.
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model, rows[:2], _LAYERS, batch_size=3, output_dir=tmp_path, progress=False, resume=True
        )
    assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"


@pytest.mark.heavy
def test_hard_process_death_mid_shard_write_then_resume(tmp_path: Path) -> None:
    """A child hard-dies (``os._exit``) mid ``torch.save``; resume heals the run.

    This is the D7 crash-safety proof: the partially-written shard only ever
    bears the temp name, the manifest ledger trusts exactly the completed
    prefix, and the resumed artifact is bit-equal to a clean run.
    """

    child_source = textwrap.dedent(
        """
        import os, sys, torch
        import torchlens as tl
        from torch import nn

        out_dir = sys.argv[1]
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        stimuli = torch.arange(30, dtype=torch.float32).reshape(10, 3)

        real_save = torch.save
        calls = {"n": 0}

        def dying_save(obj, path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                with open(path, "wb") as handle:
                    handle.write(b"\\x00partial")
                os._exit(1)
            return real_save(obj, path, *args, **kwargs)

        torch.save = dying_save
        tl.extract_dataset(
            model,
            stimuli,
            {"relu": "relu", "logits": "output_1"},
            batch_size=2,
            output_dir=out_dir,
            progress=False,
            shard_format="pt",
        )
        """
    )
    repo_root = Path(tl.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", child_source, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1, result.stderr
    assert (tmp_path / "batch_00000.pt").exists()
    assert (tmp_path / "batch_00001.pt").exists()
    assert not (tmp_path / "batch_00002.pt").exists(), "partial shard bears the final name"
    assert list(tmp_path.glob("*.tmp")), "hard death should leave the temp file behind"
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "in_progress"
    trusted = read_trusted_rows(tmp_path)
    assert [row["file"] for row in trusted] == ["batch_00000.pt", "batch_00001.pt"]

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    stimuli = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    layers = {"relu": "relu", "logits": "output_1"}
    tl.extract_dataset(
        model, stimuli, layers, batch_size=2, output_dir=tmp_path, progress=False, resume=True
    )
    assert not list(tmp_path.glob("*.tmp")), "resume sweeps orphan temp files"
    loaded = load_extraction(tmp_path)
    assert loaded.manifest["status"] == "complete"
    clean = tl.extract_dataset(model, stimuli, layers, batch_size=2, progress=False)
    for key, tensor in clean.items():
        assert torch.equal(loaded.activations[key], tensor)


def test_manifest_is_self_describing(tmp_path: Path) -> None:
    """The manifest carries site identity, provenance, axes, dtype, and device."""

    model = _CountingModel().eval()
    ids = [f"stim-{index:02d}" for index in range(7)]
    tl.extract_dataset(
        model,
        _stimuli(7),
        _LAYERS,
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
        transform=lambda tensor: tensor.mean(dim=1),
        stimulus_ids=ids,
    )
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == MANIFEST_SCHEMA_V2
    assert manifest["torchlens_version"] == tl.__version__
    assert manifest["status"] == "complete"
    assert manifest["totals"] == {"n_shards": 3, "n_stimuli": 7}
    assert isinstance(manifest["ledger_digest"], str)
    assert manifest["ledger_digest"].startswith("sha256:")

    signature = manifest["signature"]
    assert signature["schema_version"] == MANIFEST_SCHEMA_V2
    assert signature["layer_plan"] == _LAYERS
    assert signature["layers_kind"] == "mapping"
    assert signature["batch_size"] == 3
    # A raw lambda is an OPAQUE step: identification-only disclosure, not
    # resume-verifiable.
    pipeline = signature["transform_pipeline"]
    assert pipeline["schema"] == "tl_transform_pipeline_v1"
    assert pipeline["steps"][0]["kind"] == "opaque"
    assert pipeline["steps"][0]["qualname"].endswith("<lambda>")
    assert pipeline["resume_verifiable"] is False
    assert signature["stimuli"]["kind"] == "tensor"
    assert signature["stimuli"]["shape"] == [7, 3]
    assert signature["stimuli"]["dtype"] == "torch.float32"
    assert signature["stimuli"]["digest"].startswith("sha256:")
    # Model identity is measured by default: a recomputable crypto digest.
    identity = signature["model_identity"]
    assert identity["level"] == "measured"
    assert identity["digest"].startswith("blake2b:")
    assert identity["algorithm_id"] == "tl_model_state_merkle"
    assert identity["n_state_entries"] == len(model.state_dict())
    assert signature["model_mode"] == "eval_no_grad"

    provenance = manifest["stimulus_provenance"]
    assert provenance["n_stimuli"] == 7
    assert provenance["ids_recorded"] is True
    assert provenance["ids_digest"] == signature["stimulus_ids_digest"]
    assert "iteration order" in provenance["order"]
    sidecar = json.loads((tmp_path / "stimulus_ids.json").read_text(encoding="utf-8"))
    assert sidecar["ids"] == ids

    relu = manifest["layers"]["relu"]
    assert relu["layer_label"] == "relu_1_2"
    assert isinstance(relu["site_key"], str) and relu["site_key"].startswith("s1|")
    assert relu["site_key_unavailable"] is None
    assert relu["captured_dtype"] == "torch.float32"
    assert relu["captured_device"] == "cpu"
    assert relu["per_stimulus_shape"] == [4]
    # The mean(dim=1) transform collapses the activation axis; the manifest
    # discloses the stored geometry separately from the captured geometry.
    assert relu["stored_per_stimulus_shape"] == []
    assert relu["batch_axis"] == 0

    rows = read_trusted_rows(tmp_path)
    assert [row["n_rows"] for row in rows] == [3, 3, 1]
    assert [row["row_start"] for row in rows] == [0, 3, 6]
    for row in rows:
        assert isinstance(row["crc32"], int)
        assert row["byte_size"] == (tmp_path / row["file"]).stat().st_size
        assert row["ids_range_digest"].startswith("sha256:")
        assert row["keys"]["relu"]["value_reduction"].startswith("0x")
        assert row["keys"]["relu"]["dtype"] == "torch.float32"


def test_load_extraction_subset_and_refusals(tmp_path: Path) -> None:
    """The loader selects layers, and refuses incomplete or unknown requests."""

    model = _CountingModel().eval()
    tl.extract_dataset(
        model, _stimuli(6), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    subset = load_extraction(tmp_path, layers=["relu"])
    assert set(subset.activations) == {"relu"}
    assert subset.activations["relu"].shape == (6, 4)

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as key_excinfo:
        load_extraction(tmp_path, layers=["nope"])
    assert key_excinfo.value.fields["code"] == "extraction_reader_key_unknown"

    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        load_extraction(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_manifest_invalid"

    manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        load_extraction(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_manifest_invalid"
