"""Tests for parent-owned subprocess isolation and timeouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.worker_supervisor import run_isolated_subprocess, supervise_worker


def test_supervisor_scrubs_credentials_and_enforces_timeout(tmp_path: Path) -> None:
    """A fresh argv-only child cannot see a secret and is killed at its wall cap."""

    observation = run_isolated_subprocess(
        [
            sys.executable,
            "-c",
            "import os,time; print(os.getenv('CRAWLER_SECRET_TOKEN')); time.sleep(2)",
        ],
        tmp_path / "supervisor",
        timeout_seconds=0.1,
        rss_limit_bytes=1024**3,
        base_environment={
            "PATH": "/usr/bin:/bin",
            "CRAWLER_SECRET_TOKEN": "never",  # pragma: allowlist secret
        },
    )

    assert observation.timed_out is True
    assert observation.signal_number == 9
    assert "never" not in observation.stdout_tail


def test_supervisor_accepts_only_atomic_worker_receipt(tmp_path: Path) -> None:
    """The standard worker succeeds in a fresh process and its receipt hash verifies."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.zeros(1, 4, device=device),), {})
""",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    receipt = scratch / "result" / "receipt.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "stable_id": "m_supervised",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "unknown",
                "input_spec": {"shape": [1, 4], "dtype": "float32"},
                "scratch_root": str(scratch),
                "meaningful_modes": ["train", "eval"],
                "recipe_revision": compute_recipe_revision(
                    {"recipe_type": "typed-adapter", "path": adapter.name},
                    "unbound",
                    adapter_bytes=adapter.read_bytes(),
                ),
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request,
        receipt,
        scratch,
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    assert result.observation.exit_code == 0
    assert result.receipt_error is None
    assert result.worker_receipt is not None
    assert result.worker_receipt["awards_runs"] is False
