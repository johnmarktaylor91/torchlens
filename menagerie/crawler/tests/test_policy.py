"""Tests for execution-only security policy."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from menagerie.crawler.legacy_manifest_audit import (
    audit_runtime_root_grants,
    compile_execution_read_manifest,
)
from menagerie.crawler.policy import ExecutionPolicy, PolicyViolation, build_safe_environment

import base64
import hashlib
import json
import os
import shutil
import site
import sys
from typing import Any
from typing import Iterator
from menagerie.crawler import worker_supervisor
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.policy import (
    _runtime_code_path_allowed,
    _runtime_model_data_path,
    _runtime_package_data_paths,
    detect_os_sandbox,
)
from menagerie.crawler.worker_supervisor import (
    _read_path_is_allowed,
    SupervisedResult,
    supervise_worker,
)


def test_safe_environment_scrubs_credentials_and_socket_tripwire_fails(tmp_path: Path) -> None:
    """Child environment omits secrets and any socket attempt is observed and rejected."""

    safe = build_safe_environment(
        tmp_path / "safe",
        base_environment={
            "PATH": "/bin",
            "SECRET_TOKEN": "never",  # pragma: allowlist secret
        },
    )
    assert safe["PATH"] == "/bin"
    assert "SECRET_TOKEN" not in safe
    assert safe["HF_HUB_OFFLINE"] == "1"
    assert safe["__CF_USER_TEXT_ENCODING"] == f"0x{os.getuid():X}:0x0:0x0"

    with ExecutionPolicy(tmp_path / "scratch") as observation:
        with pytest.raises(PolicyViolation, match="blocked socket"):
            socket.create_connection(("example.invalid", 443))
        with socket.socket() as client:
            with pytest.raises(PolicyViolation, match="blocked socket"):
                client.connect(("example.invalid", 443))
    assert observation.network_attempted is True
    assert len(observation.socket_targets) == 2


def test_legacy_root_grant_is_audit_only(tmp_path: Path) -> None:
    """The quarantined v1 parser may characterize history but cannot reach live spawn."""

    legacy_root = tmp_path.resolve()
    manifest = compile_execution_read_manifest(
        stable_id="m_legacy_audit",
        work_id="work-legacy-audit",
        execution_identity="sha256:" + "1" * 64,
        code_manifest_identity="sha256:" + "2" * 64,
        code_members=(),
        runtime_support=((legacy_root, "runtime-root"),),
    )
    assert audit_runtime_root_grants(manifest) == (legacy_root,)


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
    hidden_directory = site_packages / (
        f".menagerie-r5-{tmp_path.parent.parent.name}-{tmp_path.name}"
    )
    repository_root = Path(__file__).resolve().parents[3]
    repository_data = repository_root / (
        f".menagerie-r5-{tmp_path.parent.parent.name}-{tmp_path.name}.bin"
    )
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
        Optional legacy author path used to prove it cannot become a grant.

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
        # An author-controlled input_contract path is descriptive only; v3 has
        # no shadow input-manifest field that can grant the worker a read.
        request["input_contract"] = {"code_path": str(declared_input)}
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


def _legacy_diagnostic(tmp_path: Path) -> dict[str, Any]:
    """Load a rejected flat-v1 diagnostic for legacy policy characterization.

    Parameters
    ----------
    tmp_path:
        Test root passed to ``_supervise_adapter``.

    Returns
    -------
    dict[str, Any]
        Non-authoritative on-disk worker diagnostic.
    """

    return json.loads((tmp_path / "result" / "receipt.json").read_text(encoding="utf-8"))


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
    inventoried_weights = package / "bundled-weights.pth"
    declared_input = tmp_path / "declared-input.json"
    runtime_code.write_text("VALUE = 1\n", encoding="utf-8")
    package_data.write_bytes(b"public package certificate bundle")
    hidden_data.write_bytes(b"undeclared model data")
    inventoried_weights.write_bytes(b"inventoried but undeclared model weights")
    declared_input.write_text('{"value": 1}\n', encoding="utf-8")
    digest = base64.urlsafe_b64encode(hashlib.sha256(package_data.read_bytes()).digest()).rstrip(
        b"="
    )
    weight_digest = base64.urlsafe_b64encode(
        hashlib.sha256(inventoried_weights.read_bytes()).digest()
    ).rstrip(b"=")
    (dist_info / "RECORD").write_text(
        (
            f"demo_runtime/cacert.pem,sha256={digest.decode('ascii')},"
            f"{package_data.stat().st_size}\n"
            f"demo_runtime/bundled-weights.pth,sha256={weight_digest.decode('ascii')},"
            f"{inventoried_weights.stat().st_size}\n"
        ),
        encoding="utf-8",
    )

    runtime_roots = (prefix, tmp_path / "verified-source")
    assert _runtime_code_path_allowed(runtime_code.resolve(), runtime_roots) is True
    assert _runtime_code_path_allowed(package_data.resolve(), runtime_roots) is True
    assert _runtime_code_path_allowed(hidden_data.resolve(), runtime_roots) is False
    assert _runtime_code_path_allowed(inventoried_weights.resolve(), runtime_roots) is False
    assert inventoried_weights.resolve() not in _runtime_package_data_paths(runtime_roots)
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
    # Round 14: model-data denial precedes ordinary allowlist success. Only the
    # compiled manifest's exact trusted standard asset may bypass this classifier.
    assert not _read_path_is_allowed(
        str(inventoried_weights),
        tmp_path,
        (),
        (inventoried_weights,),
        runtime_roots,
    )
    assert _read_path_is_allowed(
        str(inventoried_weights),
        tmp_path,
        (),
        (inventoried_weights,),
        runtime_roots,
        standard_input_asset=inventoried_weights,
    )

    package_data.write_bytes(b"tampered package data")
    assert _runtime_code_path_allowed(package_data.resolve(), runtime_roots) is False


def test_compiled_manifest_rejects_author_shaped_model_data_and_large_text(
    tmp_path: Path,
) -> None:
    """No renamed checkpoint can enter the compiled implementation capability."""

    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"hidden weights")
    large_text = tmp_path / "numeric.py"
    large_text.write_bytes(b"0" * (1024**2 + 1))
    assert _runtime_model_data_path(large_text)
    with pytest.raises(ValueError, match="forbidden kind/suffix"):
        compile_execution_read_manifest(
            stable_id="m_manifest_deny",
            work_id="work-manifest-deny",
            execution_identity="sha256:" + "1" * 64,
            code_manifest_identity="sha256:" + "2" * 64,
            code_members=((checkpoint, hash_bytes(checkpoint.read_bytes()), "python-source"),),
        )
    with pytest.raises(ValueError, match="model-data-shaped"):
        compile_execution_read_manifest(
            stable_id="m_manifest_deny",
            work_id="work-manifest-deny",
            execution_identity="sha256:" + "1" * 64,
            code_manifest_identity="sha256:" + "2" * 64,
            code_members=((large_text, hash_bytes(large_text.read_bytes()), "python-source"),),
        )


def test_compiled_manifest_rejects_symlink_and_hardlink_aliases(tmp_path: Path) -> None:
    """Accepted implementation bytes cannot escape through inode aliases."""

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    symlink = tmp_path / "symlink.py"
    symlink.symlink_to(source)
    digest = hash_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="aliased"):
        compile_execution_read_manifest(
            stable_id="m_manifest_alias",
            work_id="work-manifest-alias",
            execution_identity="sha256:" + "1" * 64,
            code_manifest_identity="sha256:" + "2" * 64,
            code_members=((symlink, digest, "python-source"),),
        )

    hardlink = tmp_path / "hardlink.py"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="aliased"):
        compile_execution_read_manifest(
            stable_id="m_manifest_alias",
            work_id="work-manifest-alias",
            execution_identity="sha256:" + "1" * 64,
            code_manifest_identity="sha256:" + "2" * 64,
            code_members=((source, digest, "python-source"),),
        )


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
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    diagnostic = _legacy_diagnostic(tmp_path)
    policy = diagnostic["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert set(map(str, (*site_paths, repository_data))).issubset(policy["checkpoint_paths"])
    assert diagnostic["awards_runs"] is False


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
    assert result.observation.exit_code == 0
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux semantic-read regression")
def test_legacy_author_declared_input_cannot_become_a_read_capability(
    tmp_path: Path,
) -> None:
    """The removed input_contract path grant fails closed even in a legacy request."""

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
    assert result.observation.exit_code == 1
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    diagnostic = _legacy_diagnostic(tmp_path)
    policy = diagnostic["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert str(declared_input) in policy["checkpoint_paths"]
    assert diagnostic["error"]["reason_code"] == "checkpoint-read"


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
