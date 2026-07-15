"""Regression coverage for the round-2 sandbox audit and R4 source inventory."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler import worker_supervisor
from menagerie.crawler.constants import FailureStage
from menagerie.crawler.driver import (
    AuthorArtifact,
    EnvironmentBinding,
    _attempt_policy_satisfied,
    _attempts_from_supervised,
    _environment_failure,
    _supervise_environment_worker,
)
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.policy import SandboxUnavailableError, detect_os_sandbox
from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.tests.conftest import HASH, make_author_proposal
from menagerie.crawler.tests.test_slice_d_proposal_author import _ground_proposal, _make_r4
from menagerie.crawler.worker_supervisor import supervise_worker


def _typed_adapter(outside_path: Path) -> str:
    """Return a model adapter that catches a native denied write and returns a tensor.

    Parameters
    ----------
    outside_path:
        Read-only path targeted from the model's forward method.

    Returns
    -------
    str
        Complete typed adapter source.
    """

    return f"""from __future__ import annotations
import ctypes
import os
import torch

class CaughtDenial(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.open({str(outside_path)!r}.encode(), os.O_WRONLY | os.O_CREAT, 0o600)
        if descriptor >= 0:
            libc.close(descriptor)
        return value + 1

def build_model() -> object:
    return CaughtDenial()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {{}})
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux denial-audit regression")
def test_caught_os_sandbox_denial_poisons_successful_forward_receipt(tmp_path: Path) -> None:
    """A caught native write denial becomes failed:policy and cannot satisfy run award."""

    if detect_os_sandbox("Linux") is None or shutil.which("strace") is None:
        pytest.skip("working Linux sandbox denial broker is unavailable")
    outside_path = tmp_path.parent / f"{tmp_path.name}-forbidden.bin"
    outside_path.unlink(missing_ok=True)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(_typed_adapter(outside_path), encoding="utf-8")
    proposal = make_author_proposal("m_caught_denial")
    scratch = tmp_path / "scratch"
    receipt_path = scratch / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    expected_revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": adapter.name},
        proposal["source_identity"],
        adapter_bytes=adapter.read_bytes(),
    )
    proposal["recipe_revision"] = expected_revision
    proposal["proposed_facts"]["implementation"]["recipe_revision"] = expected_revision
    request_path.write_text(
        json.dumps(
            {
                "stable_id": proposal["stable_id"],
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "vision",
                "input_spec": {"shape": [1, 3, 8, 8], "dtype": "float32"},
                "scratch_root": str(scratch),
                "receipt_path": str(receipt_path),
                "meaningful_modes": ["eval"],
                "source_identity": proposal["source_identity"],
                "recipe_revision": expected_revision,
                "execution_identity": HASH,
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    assert result.observation.exit_code == 0
    assert result.worker_receipt is not None
    policy = result.worker_receipt["policy_observation"]
    assert policy["write_outside_scratch_attempted"] is True
    assert str(outside_path) in policy["write_paths"]
    assert result.worker_receipt["per_mode"]["eval"]["forward_completed"] is True
    assert result.worker_receipt["per_mode"]["eval"]["error"]["reason_code"] == (
        "write-outside-scratch"
    )
    environment = EnvironmentBinding(
        prefix=tmp_path / "env",
        python_executable=Path(sys.executable),
        family="core",
        target="linux-64",
        env_generation=HASH,
        lock_sha256=HASH,
        resolved_export_sha256=HASH,
        packages_manifest_sha256=HASH,
        python_version="3.11",
        compiler_identity="test-compiler",
        sdk_identity="test-sdk",
    )
    artifact = AuthorArtifact(proposal, {"sources": []}, tmp_path)
    attempts = _attempts_from_supervised(
        artifact,
        result,
        environment,
        HASH,
        0,
        20,
        12 * 1024**3,
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
    )

    assert len(attempts) == 1
    assert attempts[0]["result"] == "failed"
    assert attempts[0]["stage"] == "policy"
    assert attempts[0]["error"]["reason_code"] == "write-outside-scratch"
    assert _attempt_policy_satisfied(attempts, proposal, 1) is False
    assert not outside_path.exists()


def test_real_environment_interpreter_signals_sandbox_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production env-interpreter path terminalizes missing isolation honestly."""

    def no_sandbox(_system_name: str | None = None) -> None:
        """Report no working OS sandbox."""

        return None

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        """Reject any attempted unsandboxed child launch."""

        del args, kwargs
        raise AssertionError("environment worker must not launch unsandboxed")

    monkeypatch.setattr(worker_supervisor, "detect_os_sandbox", no_sandbox)
    monkeypatch.setattr(worker_supervisor.subprocess, "Popen", forbidden_popen)
    env_python = tmp_path / "exact-env" / "bin" / "python"
    with pytest.raises(SandboxUnavailableError) as captured:
        _supervise_environment_worker(
            tmp_path / "request.json",
            tmp_path / "result" / "receipt.json",
            tmp_path / "supervisor",
            env_python,
            timeout_seconds=10,
            rss_limit_bytes=1024**3,
            cwd=tmp_path,
        )

    assert str(captured.value) == FailureStage.SANDBOX_UNAVAILABLE.value
    stage, reason = _environment_failure(captured.value)
    assert (stage, reason) == ("policy", "sandbox-unavailable-v1")
    assert f"failed:{stage}" == "failed:policy"
    assert f"failed:{stage}" != "failed:runner"


def _add_archive_source(
    manifest: dict[str, Any], archive_path: Path, members: dict[str, str]
) -> None:
    """Append one deliberately mislabeled fetched archive to a source manifest.

    Parameters
    ----------
    manifest:
        Controlled-fetch manifest fixture.
    archive_path:
        CAS object path to create.
    members:
        Archive member names and text bytes.
    """

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for name, member_text in members.items():
            archive.writestr(name, member_text)
    archive_bytes = archive_path.read_bytes()
    manifest["sources"].append(
        {
            "source_id": "archive-source",
            "url": "https://example.com/supplement.zip",
            "revision": "v1",
            "content_sha256": hash_bytes(archive_bytes),
            "cas_path": str(archive_path),
            "retrieval_status": "fetched",
            "role": "introducing-paper",
            "content_kind": "paper-supplement",
        }
    )


def test_r4_inventory_uses_fetched_archive_bytes_not_author_labels(tmp_path: Path) -> None:
    """Code-bearing CAS bytes refuse R4 while a genuine no-code archive still permits it."""

    adapter_code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, adapter_code)
    _add_archive_source(
        manifest,
        tmp_path / "source-code.zip",
        {
            "upstream/src/example_net.py": (
                "import torch\n\n"
                "class ExampleNet(torch.nn.Module):\n"
                "    def __init__(self) -> None:\n"
                "        super().__init__()\n"
                "        self.conv = torch.nn.Conv2d(3, 4, 3)\n\n"
                "    def forward(self, value: torch.Tensor) -> torch.Tensor:\n"
                "        return self.conv(value)\n"
            )
        },
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )

    no_code_root = tmp_path / "no-code"
    no_code_root.mkdir()
    no_code_proposal, no_code_manifest = _ground_proposal(no_code_root)
    _make_r4(no_code_proposal, no_code_manifest, no_code_root, adapter_code)
    _add_archive_source(
        no_code_manifest,
        no_code_root / "paper-materials.zip",
        {
            "README.md": "Architecture equations and prose only.\n",
            "supplement/metrics.py": "def accuracy(expected, observed):\n    return 1.0\n",
            "supplement/plotting.c": "void plot_metrics(void) { return; }\n",
        },
    )
    report = validate_author_proposal(
        no_code_proposal,
        allowed_model_dir=no_code_root,
        source_manifest=no_code_manifest,
    )

    assert report.rung.value == "R4_REIMPLEMENT"
