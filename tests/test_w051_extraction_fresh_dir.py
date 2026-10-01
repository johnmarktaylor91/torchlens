"""W051-EXTRACT: a fresh run into a used directory leaves no stale shards (audit 4.7),
and the DEFAULT safetensors codec survives a hard process death mid-write.
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
from torchlens._data_substrate import read_trusted_rows
from torchlens._extraction.reader import open_extraction
from torchlens.dataset_extraction import extract_dataset, load_extraction


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


@pytest.mark.smoke
def test_fresh_run_sweeps_higher_index_stale_shards(tmp_path: Path) -> None:
    """p11_stale: 4 shards, then a fresh 2-shard run -> exactly 2 shard files remain."""

    model = _model()
    extract_dataset(
        model,
        torch.randn(20, 3),
        {"h": "relu_1_2"},
        batch_size=5,
        output_dir=tmp_path,
        progress=False,
    )
    assert len(list(tmp_path.glob("batch_*.safetensors"))) == 4
    extract_dataset(
        model,
        torch.randn(10, 3),
        {"h": "relu_1_2"},
        batch_size=5,
        output_dir=tmp_path,
        progress=False,
    )
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "batch_00000.safetensors",
        "batch_00001.safetensors",
        "ledger.jsonl",
        "manifest.json",
    ]
    reader = open_extraction(tmp_path)
    assert reader.n_shards == 2 and reader.n_stimuli == 10
    assert reader.verify_all()["crc_checked"] == 2


@pytest.mark.smoke
def test_fresh_run_sweeps_the_other_native_format_too(tmp_path: Path) -> None:
    model = _model()
    extract_dataset(
        model,
        torch.randn(10, 3),
        {"h": "relu_1_2"},
        batch_size=5,
        output_dir=tmp_path,
        progress=False,
        shard_format="pt",
    )
    assert len(list(tmp_path.glob("batch_*.pt"))) == 2
    extract_dataset(
        model,
        torch.randn(5, 3),
        {"h": "relu_1_2"},
        batch_size=5,
        output_dir=tmp_path,
        progress=False,
    )
    assert not list(tmp_path.glob("batch_*.pt"))
    assert [p.name for p in sorted(tmp_path.glob("batch_*"))] == ["batch_00000.safetensors"]


@pytest.mark.heavy
def test_hard_process_death_mid_safetensors_write_then_resume(tmp_path: Path) -> None:
    """The DEFAULT codec's crash-safety proof (the pre-existing test forced ``pt``)."""

    child_source = textwrap.dedent(
        """
        import os, sys, torch
        import safetensors.torch
        import torchlens as tl
        from torch import nn

        out_dir = sys.argv[1]
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        stimuli = torch.arange(30, dtype=torch.float32).reshape(10, 3)

        real_save = safetensors.torch.save_file
        calls = {"n": 0}

        def dying_save(tensors, path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                with open(path, "wb") as handle:
                    handle.write(b"\\x00partial")
                os._exit(1)
            return real_save(tensors, path, *args, **kwargs)

        safetensors.torch.save_file = dying_save
        tl.extract_dataset(
            model,
            stimuli,
            {"relu": "relu", "logits": "output_1"},
            batch_size=2,
            output_dir=out_dir,
            progress=False,
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
    assert (tmp_path / "batch_00000.safetensors").exists()
    assert (tmp_path / "batch_00001.safetensors").exists()
    assert not (tmp_path / "batch_00002.safetensors").exists(), "partial shard bears the final name"
    assert list(tmp_path.glob("*.tmp")), "hard death should leave the temp file behind"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "in_progress"
    assert [row["file"] for row in read_trusted_rows(tmp_path)] == [
        "batch_00000.safetensors",
        "batch_00001.safetensors",
    ]

    model = _model()
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
