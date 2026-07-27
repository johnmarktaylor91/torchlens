"""Round-21 VS3 constant-bounded authority-walk composition proof."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    RuntimeMember,
    collect_executable_closure_v3,
    compile_execution_read_manifest_v3_from_closure,
    environment_read_capability,
    verify_execution_read_manifest_v3,
)
from menagerie.crawler.driver import SupervisedForwardLane, bind_materialized_environment
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    RealEnvironmentSealCounter,
    _copy_up_real_environment_member,
    hardlink_clone_tree,
    real_environment_registry,
)
from menagerie.crawler.tests.dry_run_support import DRY_RUN_CASES, TinyModelAuthor
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    _driver,
    _paths,
    _write_jsonl,
)


def _private_mutable_member(
    fixture: RealEnvironmentFixture,
    prefix: Path,
) -> Path:
    """Return one privately copied sealed member suitable for mutation.

    Parameters
    ----------
    fixture:
        Real-prefix authority supplying the sealed member inventory.
    prefix:
        Test-owned hardlink clone to detach from the shared source inode.

    Returns
    -------
    pathlib.Path
        Private regular member whose mutation cannot alter the session fixture.
    """

    authority = fixture.binding.environment_authority
    if authority is None:
        raise AssertionError("real environment fixture lacks strict authority")
    relative = next(
        entry.relative_path
        for entry in authority.content_manifest.entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.startswith("include/")
        and entry.sha256 is not None
    )
    path = prefix / relative
    _copy_up_real_environment_member(path, fixture.source_prefix / relative)
    return path


def _compile_count_arm(
    tmp_path: Path,
    fixture: RealEnvironmentFixture,
    cache: EnvironmentAuthorityCache,
) -> None:
    """Compile and project 30 distinct real-prefix manifests in one pass.

    Parameters
    ----------
    tmp_path:
        Isolated exact model-code member root.
    fixture:
        Real-prefix authority used by the shipped v3 compiler.
    cache:
        Lifecycle-owned cache whose production counters prove constant walks.
    """

    authority = cache.authority
    if authority is None:
        raise AssertionError("real driver cache lacks active authority")
    code = tmp_path / "count-arm-model.py"
    code.write_text("VALUE = 21\n", encoding="utf-8")
    member = RuntimeMember(
        path=code.resolve(),
        sha256=hash_bytes(code.read_bytes()),
        kind="python-source",
        provenance="round21-count-arm",
    )
    before_walks = cache.cheap_tree_walks
    before_passes = cache.currentness_passes
    with cache.currentness_pass(authority) as verification:
        for index in range(30):
            closure = collect_executable_closure_v3(
                code_manifest_identity=stable_hash([f"model-{index}"]),
                environment_authority=authority,
                code_members=(member,),
                worker_members=(),
                verification_token=verification,
            )
            manifest = compile_execution_read_manifest_v3_from_closure(
                closure,
                stable_id=f"m_round21_scale_{index:02d}",
                work_id=f"work-round21-scale-{index:02d}",
                execution_identity=stable_hash({"round21-scale": index}),
                verification_token=verification,
            )
            verify_execution_read_manifest_v3(
                manifest,
                verification_token=verification,
            )
            capability = environment_read_capability(
                manifest,
                verification_token=verification,
            )
            assert capability.environment_prefix == fixture.prefix
    assert cache.currentness_passes == before_passes + 1
    assert cache.cheap_tree_walks == before_walks + 1
    assert cache.lstat_tree_walks == 2 * cache.full_seals + cache.cheap_tree_walks


def _assert_spawn_validation_catches_post_pass_mutation(
    tmp_path: Path,
    fixture: RealEnvironmentFixture,
) -> None:
    """Prove a mutation after pass validation is rejected before child spawn.

    Parameters
    ----------
    tmp_path:
        Isolated clone and strict-binding artifact root.
    fixture:
        Real environment fixture whose lock/probe facts remain authoritative.
    """

    prefix = tmp_path / "mutation-prefix"
    hardlink_clone_tree(fixture.prefix, prefix)
    mutable = _private_mutable_member(fixture, prefix)
    cache = EnvironmentAuthorityCache()
    binding = bind_materialized_environment(
        fixture.intent,
        prefix,
        fixture.probe_results,
        authority_cache=cache,
    )
    authority = binding.environment_authority
    if authority is None:
        raise AssertionError("mutation clone lacks strict authority")
    with cache.currentness_pass(authority):
        pass
    before = mutable.stat()
    changed = bytearray(mutable.read_bytes())
    if not changed:
        raise AssertionError("round-21 mutation member is empty")
    changed[0] ^= 1
    with mutable.open("r+b") as handle:
        handle.write(changed)
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(mutable, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(AuthorityDerivationError, match="content seal"):
        with cache.spawn_verification(authority):
            pytest.fail("stale spawn verification yielded a capability")
    assert cache.spawn_validations == 1
    assert cache.real_spawns == 0
    assert cache.rehashes == 1
    assert cache.invalidations == 1
    assert cache.lstat_tree_walks == 2 * cache.full_seals + cache.cheap_tree_walks


def test_pass_and_spawn_validation_walks_are_constant_bounded(
    tmp_path: Path,
    isolated_real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """One real walk per currentness pass and per actual worker spawn.

    Parameters
    ----------
    tmp_path:
        Isolated three-model campaign, count arm, and mutation clone root.
    """

    assert set(structural.ROUND21_VS3_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "T01",
        "T01-CI",
        "T02",
        "T03",
    }
    isolated_fixture = isolated_real_environment_fixture
    master = tmp_path / "scale-master.jsonl"
    deferred = tmp_path / "scale-deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": case.name, "zoo": "round21-scale", "variant": "base"}
            for case in (DRY_RUN_CASES[0], DRY_RUN_CASES[1], DRY_RUN_CASES[3])
        ],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    paths = _paths(tmp_path, snapshot)
    environments = RealEnvironmentLane(isolated_fixture)
    driver = _driver(
        tmp_path,
        snapshot,
        author=TinyModelAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=environments,
        registry=real_environment_registry(isolated_fixture),
    )

    first = driver.run()
    cache = environments.environment_authority_cache
    initial_models = scan_jsonl(paths.ledgers.models)
    initial_attempts = scan_jsonl(paths.ledgers.attempts)
    assert first.status == "complete"
    assert len({str(row["stable_id"]) for row in initial_models}) == 3
    assert len(initial_attempts) >= 3
    assert {row["status"]["code"] for row in initial_models} == {"runs"}
    assert {row["result"] for row in initial_attempts} == {"succeeded"}
    assert cache.real_spawns == len(initial_attempts)
    assert cache.spawn_validations == cache.real_spawns
    assert cache.cheap_tree_walks == cache.currentness_passes + cache.real_spawns
    assert cache.lstat_tree_walks == 2 * cache.full_seals + cache.cheap_tree_walks
    assert cache.full_seals == 1
    assert cache.rehashes == 0

    walks_after_first = cache.cheap_tree_walks
    spawns_after_first = cache.real_spawns
    second = driver.run()
    walks_after_second = cache.cheap_tree_walks
    third = driver.run()
    assert second.status == third.status == "complete"
    assert walks_after_second == walks_after_first + 1
    assert cache.cheap_tree_walks == walks_after_second + 1
    assert cache.real_spawns == spawns_after_first
    assert cache.spawn_validations == cache.real_spawns
    assert cache.cheap_tree_walks == cache.currentness_passes + cache.real_spawns
    assert cache.lstat_tree_walks == 2 * cache.full_seals + cache.cheap_tree_walks
    assert scan_jsonl(paths.ledgers.models) == initial_models
    assert scan_jsonl(paths.ledgers.attempts) == initial_attempts
    assert cache.full_seals == 1
    assert cache.rehashes == 0

    _compile_count_arm(tmp_path, isolated_fixture, cache)
    assert cache.cheap_tree_walks == cache.currentness_passes + cache.real_spawns
    _assert_spawn_validation_catches_post_pass_mutation(tmp_path, isolated_fixture)


def test_real_environment_fixture_full_seals_are_session_bounded(
    real_environment_fixture: RealEnvironmentFixture,
    real_environment_seal_counter: RealEnvironmentSealCounter,
) -> None:
    """Read-only compositions share one seal while mutators receive bounded isolates."""

    real_environment_seal_counter.assert_bounded(require_shared=True)
    counts = real_environment_seal_counter.snapshot()
    assert (
        real_environment_fixture.binding.environment_authority_cache
        is (real_environment_seal_counter.shared_caches[0])
    )
    assert counts["shared_fixtures"] == 1
    assert counts["shared_full_seals"] == 1
    assert counts["base_seals"] == 1 + counts["isolated_fixtures"]
    assert counts["observed_full_seals"] <= counts["maximum_full_seals"]
