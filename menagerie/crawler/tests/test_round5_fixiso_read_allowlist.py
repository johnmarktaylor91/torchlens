"""Regression coverage for round-5 Linux semantic read isolation."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import site
import sys
from pathlib import Path
from typing import Iterator

import pytest

from menagerie.crawler import worker_supervisor
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.policy import _runtime_code_path_allowed, detect_os_sandbox
from menagerie.crawler.worker_supervisor import (
    _read_path_is_allowed,
    SupervisedResult,
    supervise_worker,
)


def _tiny_adapter(constructor_body: str) -> str:
    """Return a typed torch adapter with a configurable constructor body.

    Parameters
    ----------
    constructor_body:
        Complete indented statements placed inside ``build_model``.

    Returns
    -------
    str
        Complete adapter source.
    """

    body = "\n".join(f"    {line}" for line in constructor_body.splitlines())
    return f"""from __future__ import annotations
import ctypes
import os
from pathlib import Path
import torch

class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1

def build_model() -> object:
{body}

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


@pytest.fixture
def undeclared_data_paths(tmp_path: Path) -> Iterator[tuple[tuple[Path, ...], Path]]:
    """Create hidden data inside site-packages and the repository for one test.

    Parameters
    ----------
    tmp_path:
        Pytest-owned path used to make globally unique fixture names.

    Yields
    ------
    tuple[tuple[pathlib.Path, ...], pathlib.Path]
        Four site-packages data paths and one repository data path.
    """

    site_packages = Path(site.getsitepackages()[0]).resolve()
    hidden_directory = site_packages / f".menagerie-r5-{tmp_path.name}"
    repository_root = Path(__file__).resolve().parents[3]
    repository_data = repository_root / f".menagerie-r5-{tmp_path.name}.bin"
    hidden_directory.mkdir()
    site_paths = tuple(
        hidden_directory / name for name in ("weights.pt", "weights.bin", "weights.npz", "weights")
    )
    try:
        for path in site_paths:
            path.write_bytes(b"hidden-site-packages-model-data")
        repository_data.write_bytes(b"hidden-repository-model-data")
        yield site_paths, repository_data
    finally:
        shutil.rmtree(hidden_directory, ignore_errors=True)
        repository_data.unlink(missing_ok=True)


def _supervise_adapter(
    tmp_path: Path,
    adapter_source: str,
    *,
    declared_input: Path | None = None,
) -> SupervisedResult:
    """Run one typed adapter through the production Linux supervisor.

    Parameters
    ----------
    tmp_path:
        Per-test request, result, and scratch root.
    adapter_source:
        Complete typed-adapter source.
    declared_input:
        Optional exact input path named by the immutable request.

    Returns
    -------
    SupervisedResult
        Parent observation and optional verified receipt.
    """

    adapter = tmp_path / "adapter.py"
    adapter.write_text(adapter_source, encoding="utf-8")
    source_identity = "source-round5-iso"
    recipe_revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": adapter.name},
        source_identity,
        adapter_bytes=adapter.read_bytes(),
    )
    scratch = tmp_path / "scratch"
    receipt_path = tmp_path / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    request: dict[str, object] = {
        "stable_id": "m_round5_read_allowlist",
        "recipe": {
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        "modality": "unknown",
        "input_spec": {"shape": [1, 2], "dtype": "float32"},
        "scratch_root": str(scratch),
        "meaningful_modes": ["eval"],
        "source_identity": source_identity,
        "recipe_revision": recipe_revision,
    }
    if declared_input is not None:
        request["input_contract"] = {"code_path": str(declared_input)}
        request["input_manifest"] = {
            "validated_model_root": str(tmp_path),
            "validated_input_code_path": str(declared_input),
        }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=30,
        rss_limit_bytes=12 * 1024**3,
    )


def _assert_linux_enforcement_or_closed(result: SupervisedResult) -> bool:
    """Assert fail-closed behavior when Linux enforcement is unavailable.

    Parameters
    ----------
    result:
        Completed supervisor result.

    Returns
    -------
    bool
        True when the Linux sandbox and syscall audit were available.
    """

    available = detect_os_sandbox("Linux") is not None and shutil.which("strace") is not None
    if available:
        return True
    assert result.worker_receipt is None
    assert result.receipt_error == "failed:sandbox-unavailable"
    return False


def test_hash_inventoried_package_data_and_declared_input_are_readable(tmp_path: Path) -> None:
    """Only RECORD-bound package data joins code and declared inputs in the read closure."""

    prefix = tmp_path / "environment"
    site_packages = prefix / "lib" / "python3.11" / "site-packages"
    package = site_packages / "demo_runtime"
    dist_info = site_packages / "demo_runtime-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    runtime_code = package / "__init__.py"
    package_data = package / "cacert.pem"
    hidden_data = package / "weights.npz"
    declared_input = tmp_path / "declared-input.json"
    runtime_code.write_text("VALUE = 1\n", encoding="utf-8")
    package_data.write_bytes(b"public package certificate bundle")
    hidden_data.write_bytes(b"undeclared model data")
    declared_input.write_text('{"value": 1}\n', encoding="utf-8")
    digest = base64.urlsafe_b64encode(hashlib.sha256(package_data.read_bytes()).digest()).rstrip(
        b"="
    )
    (dist_info / "RECORD").write_text(
        f"demo_runtime/cacert.pem,sha256={digest.decode('ascii')},{package_data.stat().st_size}\n",
        encoding="utf-8",
    )

    runtime_roots = (prefix, tmp_path / "verified-source")
    assert _runtime_code_path_allowed(runtime_code.resolve(), runtime_roots) is True
    assert _runtime_code_path_allowed(package_data.resolve(), runtime_roots) is True
    assert _runtime_code_path_allowed(hidden_data.resolve(), runtime_roots) is False
    assert (
        _read_path_is_allowed(
            str(declared_input),
            tmp_path,
            (),
            (declared_input,),
            runtime_roots,
        )
        is True
    )

    package_data.write_bytes(b"tampered package data")
    assert _runtime_code_path_allowed(package_data.resolve(), runtime_roots) is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux semantic-read regression")
def test_python_open_hidden_env_and_repository_data_cannot_earn_run(
    tmp_path: Path,
    undeclared_data_paths: tuple[tuple[Path, ...], Path],
) -> None:
    """Caught Python opens inside broad namespace trees still dirty the receipt."""

    site_paths, repository_data = undeclared_data_paths
    path_literals = repr([str(path) for path in (*site_paths, repository_data)])
    adapter_source = _tiny_adapter(
        f"for hidden_path in {path_literals}:\n"
        "    try:\n"
        "        Path(hidden_path).read_bytes()\n"
        "    except Exception:\n"
        "        pass\n"
        "return Tiny()"
    )

    result = _supervise_adapter(tmp_path, adapter_source)

    if not _assert_linux_enforcement_or_closed(result):
        return
    assert result.worker_receipt is not None
    policy = result.worker_receipt["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert set(map(str, (*site_paths, repository_data))).issubset(policy["checkpoint_paths"])
    assert result.worker_receipt["awards_runs"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux semantic-read regression")
def test_native_reads_hidden_env_and_repository_data_poison_successful_receipt(
    tmp_path: Path,
    undeclared_data_paths: tuple[tuple[Path, ...], Path],
) -> None:
    """Successful libc reads of hidden namespace data are parent-poisoned."""

    site_paths, repository_data = undeclared_data_paths
    path_literals = repr([str(path) for path in (*site_paths, repository_data)])
    adapter_source = _tiny_adapter(
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        f"for hidden_path in {path_literals}:\n"
        "    descriptor = libc.open(hidden_path.encode(), os.O_RDONLY)\n"
        "    if descriptor < 0:\n"
        "        raise RuntimeError(f'native hidden read failed: {hidden_path}')\n"
        "    buffer = ctypes.create_string_buffer(64)\n"
        "    if libc.read(descriptor, buffer, len(buffer)) <= 0:\n"
        "        raise RuntimeError(f'native hidden read returned no bytes: {hidden_path}')\n"
        "    libc.close(descriptor)\n"
        "return Tiny()"
    )

    result = _supervise_adapter(tmp_path, adapter_source)

    if not _assert_linux_enforcement_or_closed(result):
        return
    assert result.worker_receipt is not None
    assert result.worker_receipt["constructor_completed"] is True
    assert result.worker_receipt["per_mode"]["eval"]["forward_completed"] is True
    policy = result.worker_receipt["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert set(map(str, (*site_paths, repository_data))).issubset(policy["checkpoint_paths"])
    assert result.worker_receipt["error"]["reason_code"] == "checkpoint-read"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux semantic-read regression")
def test_declared_input_verified_source_and_environment_code_reads_remain_clean(
    tmp_path: Path,
) -> None:
    """Exact declared data and inventoried code can still earn a clean receipt."""

    declared_input = tmp_path / "declared-input.bin"
    declared_input.write_bytes(b"declared-model-input")
    adapter_source = _tiny_adapter(
        f"if Path({str(declared_input)!r}).read_bytes() != b'declared-model-input':\n"
        "    raise RuntimeError('declared input mismatch')\n"
        "if not Path(__file__).read_bytes():\n"
        "    raise RuntimeError('verified source unreadable')\n"
        "if not Path(torch.__file__).read_bytes():\n"
        "    raise RuntimeError('environment code unreadable')\n"
        "return Tiny()"
    )

    result = _supervise_adapter(tmp_path, adapter_source, declared_input=declared_input)

    if not _assert_linux_enforcement_or_closed(result):
        return
    assert result.observation.exit_code == 0
    assert result.worker_receipt is not None
    assert result.worker_receipt["constructor_completed"] is True
    assert result.worker_receipt["per_mode"]["eval"]["forward_completed"] is True
    policy = result.worker_receipt["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is False
    assert result.worker_receipt["error"] is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux semantic-read regression")
def test_missing_linux_read_audit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing syscall audit mechanism refuses launch and cannot earn a run."""

    real_which = worker_supervisor.shutil.which

    def missing_strace(command: str) -> str | None:
        """Hide only the required Linux syscall audit executable.

        Parameters
        ----------
        command:
            Executable name requested by the supervisor.

        Returns
        -------
        str | None
            ``None`` for strace and the real lookup for every other executable.
        """

        return None if command == "strace" else real_which(command)

    monkeypatch.setattr(worker_supervisor.shutil, "which", missing_strace)
    result = _supervise_adapter(tmp_path, _tiny_adapter("return Tiny()"))

    assert result.worker_receipt is None
    # R6-M3: sandbox-unavailability is a closed-taxonomy `failed:policy` terminal
    # (reason `sandbox-unavailable-v1`); the missing-audit path still fails closed.
    assert result.receipt_error == "failed:policy"
    assert result.observation.exit_code is None
