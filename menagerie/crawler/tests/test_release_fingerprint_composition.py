"""Round-21 VS2 complete cheap-fingerprint composition proof."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    EnvironmentAuthorityV1,
    verify_environment_authority,
)
from menagerie.crawler.driver import bind_materialized_environment
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    _copy_up_real_environment_member,
    hardlink_clone_tree,
)
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.test_environment_authority_composition import (
    _run_host_denial_composition,
)


def _different_same_size(content: bytes) -> bytes:
    """Return different bytes with exactly the original length.

    Parameters
    ----------
    content:
        Nonempty original file bytes.

    Returns
    -------
    bytes
        Same-size content differing in its first byte.
    """

    if not content:
        raise AssertionError("round-21 mutation member must be nonempty")
    return bytes((content[0] ^ 1,)) + content[1:]


def _restore_metadata(path: Path, before: os.stat_result) -> None:
    """Restore caller-settable mode, access time, and modification time.

    Parameters
    ----------
    path:
        Mutated regular file.
    before:
        Metadata captured before mutation.
    """

    path.chmod(stat.S_IMODE(before.st_mode))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _write_same_size_and_restore_times(path: Path, content: bytes) -> os.stat_result:
    """Write same-size bytes in place and restore ordinary timestamps.

    Parameters
    ----------
    path:
        Existing regular file to mutate.
    content:
        Replacement bytes with the exact current file size.

    Returns
    -------
    os.stat_result
        Metadata captured immediately before mutation.
    """

    before = path.stat()
    if len(content) != before.st_size:
        raise AssertionError("round-21 mutation must preserve file size")
    with path.open("r+b") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _restore_metadata(path, before)
    after = path.stat()
    assert after.st_ino == before.st_ino
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    _require_observable_change(before.st_ctime_ns, after.st_ctime_ns, "ctime_ns")
    return before


def _replace_same_size_and_restore_times(path: Path, content: bytes) -> os.stat_result:
    """Replace one path with a distinct same-size inode and restored timestamps.

    Parameters
    ----------
    path:
        Existing regular file to replace.
    content:
        Replacement bytes with the exact current file size.

    Returns
    -------
    os.stat_result
        Metadata captured immediately before replacement.
    """

    before = path.stat()
    if len(content) != before.st_size:
        raise AssertionError("round-21 replacement must preserve file size")
    with path.open("rb") as retained_inode:
        path.unlink()
        path.write_bytes(content)
        _restore_metadata(path, before)
        after = path.stat()
        assert retained_inode.read() != content
        assert after.st_ino != before.st_ino
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    _require_observable_change(before.st_ctime_ns, after.st_ctime_ns, "ctime_ns")
    return before


def _require_observable_change(before: int, after: int, field: str) -> None:
    """Require a release filesystem to expose one cheap-field transition.

    Parameters
    ----------
    before, after:
        Metadata values bracketing an ordinary mutation.
    field:
        Stable field name used in a release-gate diagnostic.
    """

    if before != after:
        return
    message = f"filesystem did not expose the required {field} transition"
    if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
        pytest.fail(f"unmet-release-gate: {message}")
    pytest.skip(message)


def _make_private_copy(
    path: Path,
    source_path: Path,
) -> tuple[bytes, os.stat_result]:
    """Break one fixture hardlink while preserving its bytes and metadata.

    Parameters
    ----------
    path, source_path:
        Hardlinked clone member and private-source member that must remain unchanged.

    Returns
    -------
    tuple[bytes, os.stat_result]
        Original bytes and post-copy metadata baseline.
    """

    return _copy_up_real_environment_member(path, source_path)


def _assert_one_stale_rehash(
    cache: EnvironmentAuthorityCache,
    authority: EnvironmentAuthorityV1,
) -> None:
    """Require one cheap mismatch to rehash, invalidate, and reject authority.

    Parameters
    ----------
    cache, authority:
        Newly sealed cache and its mutation-preceding authority.
    """

    with pytest.raises(AuthorityDerivationError, match="content seal"):
        cache.verify(authority)
    assert {
        "cheap_validations": cache.cheap_validations,
        "full_seals": cache.full_seals,
        "rehashes": cache.rehashes,
        "invalidations": cache.invalidations,
    } == {
        "cheap_validations": 1,
        "full_seals": 2,
        "rehashes": 1,
        "invalidations": 1,
    }


def _bind_clone(
    fixture: RealEnvironmentFixture,
    prefix: Path,
    cache: EnvironmentAuthorityCache,
) -> EnvironmentAuthorityV1:
    """Strictly bind one real-prefix clone through the shipped lifecycle binder.

    Parameters
    ----------
    fixture:
        Lock, export, and probe facts from the real environment fixture.
    prefix:
        Real hardlink-cloned materialized environment.
    cache:
        Fresh lifecycle-owned authority cache.

    Returns
    -------
    EnvironmentAuthorityV1
        Strict complete-prefix authority.
    """

    binding = bind_materialized_environment(
        fixture.intent,
        prefix,
        fixture.probe_results,
        authority_cache=cache,
    )
    authority = binding.environment_authority
    if authority is None:
        raise AssertionError("strict round-21 binding lacks environment authority")
    return authority


def test_cheap_fingerprint_catches_stat_preserved_mutation_without_false_staling_clone(
    tmp_path: Path,
    isolated_real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Field-complete cheap checks stale mutations but re-baseline clone churn.

    Parameters
    ----------
    tmp_path:
        Isolated clone, external target, and real award roots.
    real_environment_fixture:
        Lock-selected real Torch environment sealed by the shipped binder.
    """

    assert set(structural.ROUND21_VS2_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "T01",
        "T01-CI",
        "T02",
    }
    real_environment_fixture = isolated_real_environment_fixture
    fixture_authority = real_environment_fixture.binding.environment_authority
    if fixture_authority is None:
        raise AssertionError("real environment fixture lacks strict authority")

    primary = tmp_path / "primary"
    hardlink_clone_tree(real_environment_fixture.prefix, primary)
    verify_environment_authority(fixture_authority)
    external_target = tmp_path / "external-target.bin"
    external_original = b"round21 external seal\n"
    external_target.write_bytes(external_original)
    external_link = primary / "round21-external-target"
    external_link.symlink_to(external_target)

    clone_cache = EnvironmentAuthorityCache()
    clone_authority = _bind_clone(real_environment_fixture, primary, clone_cache)
    authority_id = clone_authority.authority_id
    generation = clone_authority.environment_generation
    second_clone = tmp_path / "second-clone"
    hardlink_clone_tree(primary, second_clone)
    clone_cache.verify(clone_authority)
    verify_environment_authority(fixture_authority)
    clone_rehashes = clone_cache.rehashes
    clone_full_seals = clone_cache.full_seals
    clone_invalidations = clone_cache.invalidations
    assert clone_cache.invalidations == 0
    assert clone_authority.authority_id == authority_id
    assert clone_authority.environment_generation == generation

    expected_sandbox = "bubblewrap" if sys.platform == "linux" else "sandbox-exec"
    award_fixture = RealEnvironmentFixture(
        source_prefix=real_environment_fixture.source_prefix,
        prefix=primary,
        binding=bind_materialized_environment(
            real_environment_fixture.intent,
            primary,
            real_environment_fixture.probe_results,
            authority_cache=clone_cache,
        ),
        intent=real_environment_fixture.intent,
        probe_results=real_environment_fixture.probe_results,
        sentinel_module=real_environment_fixture.sentinel_module,
        startup_pth=(
            primary
            / real_environment_fixture.startup_pth.relative_to(real_environment_fixture.prefix)
        ),
    )
    clone_award_root = tmp_path / "clone-award"
    clone_award_root.mkdir()
    _run_host_denial_composition(
        clone_award_root,
        award_fixture,
        expected_sandbox=expected_sandbox,
    )

    mutable_relative = next(
        entry.relative_path
        for entry in clone_authority.content_manifest.entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.startswith("include/")
        and entry.sha256 is not None
    )
    mutable = primary / mutable_relative
    original, private_status = _make_private_copy(
        mutable,
        real_environment_fixture.source_prefix / mutable_relative,
    )
    verify_environment_authority(fixture_authority)
    clone_cache.verify(clone_authority)
    assert private_status.st_nlink == 1

    stat_cache = EnvironmentAuthorityCache()
    stat_authority = _bind_clone(real_environment_fixture, primary, stat_cache)
    _write_same_size_and_restore_times(mutable, _different_same_size(original))
    _assert_one_stale_rehash(stat_cache, stat_authority)
    stale_award_root = tmp_path / "stale-award"
    stale_award_root.mkdir()
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        _run_host_denial_composition(
            stale_award_root,
            RealEnvironmentFixture(
                source_prefix=award_fixture.source_prefix,
                prefix=primary,
                binding=award_fixture.binding,
                intent=award_fixture.intent,
                probe_results=award_fixture.probe_results,
                sentinel_module=award_fixture.sentinel_module,
                startup_pth=award_fixture.startup_pth,
            ),
            expected_sandbox=expected_sandbox,
        )
    stale_public = stale_award_root / "runtime" / "mirrors" / "public"
    assert not stale_public.exists() or not any(path.is_file() for path in stale_public.rglob("*"))

    mutable.write_bytes(original)
    _restore_metadata(mutable, private_status)
    inode_cache = EnvironmentAuthorityCache()
    inode_authority = _bind_clone(real_environment_fixture, primary, inode_cache)
    _replace_same_size_and_restore_times(mutable, _different_same_size(original))
    _assert_one_stale_rehash(inode_cache, inode_authority)

    mutable.write_bytes(original)
    _restore_metadata(mutable, private_status)
    external_cache = EnvironmentAuthorityCache()
    external_authority = _bind_clone(real_environment_fixture, primary, external_cache)
    _write_same_size_and_restore_times(
        external_target,
        _different_same_size(external_original),
    )
    _assert_one_stale_rehash(external_cache, external_authority)

    assert clone_rehashes == 1
    assert clone_full_seals == 2
    assert clone_invalidations == 0
    assert clone_cache.invalidations == 1
    assert clone_authority.authority_id == authority_id
    assert clone_authority.environment_generation == generation
