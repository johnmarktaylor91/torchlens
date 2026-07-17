"""Round-19 sealed environment-authority composition regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    compile_execution_read_manifest_v3,
    verify_execution_read_manifest_v3,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.tests.conftest import RealEnvironmentFixture


HASH = "sha256:" + "a" * 64


def test_real_hardlink_clone_binds_complete_environment_authority(
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The real Torch prefix binds hardlinks and startup/runtime content as one unit."""

    binding = real_environment_fixture.binding
    authority = binding.environment_authority
    assert authority is not None
    assert binding.python_executable == real_environment_fixture.prefix / "bin" / "python"
    assert binding.env_generation == authority.environment_generation
    assert binding.base_environment_generation == authority.base_environment_generation
    assert binding.environment_content_sha256 == authority.content_manifest_sha256
    assert binding.environment_authority_id == authority.authority_id
    assert real_environment_fixture.startup_pth in authority.startup_pth_paths
    assert any(
        entry.relative_path.endswith("__future__.py")
        for entry in authority.content_manifest.entries
    )
    assert any(entry.relative_path.endswith(".so") for entry in authority.content_manifest.entries)


def _hardlink_file(source: Path, target: Path, content: bytes) -> None:
    """Create one file and its hardlinked sealed-prefix member.

    Parameters
    ----------
    source, target:
        Outside staging file and in-prefix member path.
    content:
        Exact shared bytes.
    """

    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    os.link(source, target)


def test_hardlinked_prefix_is_one_sealed_authority_and_mutation_stales(
    tmp_path: Path,
) -> None:
    """Hardlinks, startup ``.pth``, native bytes, and metadata form one unit."""

    prefix = tmp_path / "prefix"
    staging = tmp_path / "staging"
    _hardlink_file(staging / "python", prefix / "bin" / "python", b"python")
    (prefix / "bin" / "python").chmod(0o755)
    _hardlink_file(
        staging / "future.py",
        prefix / "lib" / "python3.11" / "__future__.py",
        b"future = True\n",
    )
    _hardlink_file(
        staging / "startup.pth",
        prefix / "lib" / "python3.11" / "site-packages" / "sentinel.pth",
        b"import sentinel\n",
    )
    _hardlink_file(staging / "native.so", prefix / "lib" / "libsentinel.so", b"native")
    _hardlink_file(
        staging / "metadata.json",
        prefix / "conda-meta" / "sentinel-1-0.json",
        b"{}\n",
    )

    cache = EnvironmentAuthorityCache()
    authority = cache.bind(
        prefix=prefix,
        selected_interpreter=prefix / "bin" / "python",
        base_environment_generation=HASH,
    )

    assert authority.prefix == prefix.resolve()
    assert authority.selected_interpreter == prefix / "bin" / "python"
    assert authority.environment_generation != HASH
    assert authority.content_manifest_sha256.startswith("sha256:")
    assert all(path.stat().st_nlink > 1 for path in prefix.rglob("*") if path.is_file())
    assert cache.full_seals == 1
    cache.verify(authority)
    assert cache.cheap_validations == 1
    assert cache.full_seals == 1

    changed = prefix / "lib" / "python3.11" / "__future__.py"
    changed.unlink()
    changed.write_text("future = False\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        cache.verify(authority)
    assert cache.rehashes == 1
    assert cache.full_seals == 2


def test_outside_selected_interpreter_is_rejected_at_binding(tmp_path: Path) -> None:
    """An interpreter escape such as ``/bin/false`` fails before worker spawn."""

    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to("/bin/false")

    with pytest.raises(AuthorityDerivationError, match="selected interpreter.*prefix"):
        EnvironmentAuthorityCache().bind(
            prefix=prefix,
            selected_interpreter=prefix / "bin" / "python",
            base_environment_generation=HASH,
        )


def test_manifest_v3_rejects_changed_interpreter_association(tmp_path: Path) -> None:
    """The v3 verifier binds argv authority to exact sealed interpreter bytes."""

    prefix = tmp_path / "prefix"
    staging = tmp_path / "staging"
    interpreter = prefix / "bin" / "python"
    _hardlink_file(staging / "python", interpreter, b"python")
    interpreter.chmod(0o755)
    code = tmp_path / "adapter.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    authority = EnvironmentAuthorityCache().bind(
        prefix=prefix,
        selected_interpreter=interpreter,
        base_environment_generation=HASH,
    )
    manifest = compile_execution_read_manifest_v3(
        stable_id="m_round19",
        work_id="work-round19",
        execution_identity=HASH,
        code_manifest_identity=stable_hash(["adapter.py"]),
        environment_authority=authority,
        code_members=((code, hash_bytes(code.read_bytes()), "python-source"),),
    )
    verify_execution_read_manifest_v3(manifest)

    interpreter.unlink()
    interpreter.write_bytes(b"changed")
    interpreter.chmod(0o755)
    with pytest.raises(AuthorityDerivationError):
        verify_execution_read_manifest_v3(manifest)
