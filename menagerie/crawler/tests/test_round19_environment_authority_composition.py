"""Round-19 sealed environment-authority composition regressions."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    RuntimeMember,
    compile_execution_read_manifest_v3,
    verify_execution_read_manifest_v3,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    SupervisedForwardLane,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.policy import detect_os_sandbox, generate_macos_sandbox_profile
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    real_environment_registry,
)
from menagerie.crawler.tests.dry_run_support import DRY_RUN_CASES, TinyModelAuthor
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
    _test_authority_context,
    _write_jsonl,
)


HASH = "sha256:" + "a" * 64

_DECLARED_ADAPTER = """from __future__ import annotations

import torch
import menagerie_round19_sentinel as round19_sentinel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from declared_model.layers import Tiny


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""

_DECLARED_MODEL = """from __future__ import annotations

import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1
"""


def _denial_adapter(undeclared: Path) -> str:
    """Return typed adapter source that catches one undeclared repository read.

    Parameters
    ----------
    undeclared:
        Exact adjacent repository file excluded from the code manifest.

    Returns
    -------
    str
        Complete trusted adapter source.
    """

    return f"""from __future__ import annotations

import ctypes
import os
import torch
import menagerie_round19_sentinel as round19_sentinel


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def _open_undeclared() -> None:
    descriptor = ctypes.CDLL(None, use_errno=True).open(
        os.fsencode({str(undeclared)!r}), os.O_RDONLY
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), {str(undeclared)!r})
    os.close(descriptor)


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    try:
        _open_undeclared()
    except OSError:
        pass
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


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


def test_real_multi_model_cache_closes_currentness_and_quarantines_mutation(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """One lifecycle cache serves repeated currentness and rejects changed prefix bytes."""

    master = tmp_path / "cache-master.jsonl"
    deferred = tmp_path / "cache-deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": case.name, "zoo": "cache-fixtures", "variant": "base"}
            for case in (DRY_RUN_CASES[0], DRY_RUN_CASES[1], DRY_RUN_CASES[3])
        ],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    paths = _paths(tmp_path, snapshot)
    environments = RealEnvironmentLane(real_environment_fixture)
    driver = _driver(
        tmp_path,
        snapshot,
        author=TinyModelAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=environments,
        registry=real_environment_registry(real_environment_fixture),
    )

    first = driver.run()
    assert first.status == "complete"
    initial_models = scan_jsonl(paths.ledgers.models)
    initial_attempts = scan_jsonl(paths.ledgers.attempts)
    assert len(initial_models) == 3
    assert {row["status"]["code"] for row in initial_models} == {"runs"}
    assert all(row["result"] == "succeeded" for row in initial_attempts)
    prior_revisions = {str(row["record_revision"]) for row in initial_models}
    prior_execution_identities = {
        str(row["execution"]["execution_identity"]) for row in initial_models
    }

    second = driver.run()
    cheap_after_second = environments.environment_authority_cache.cheap_validations
    third = driver.run()
    cheap_after_third = environments.environment_authority_cache.cheap_validations
    full_seals_while_unchanged = environments.environment_authority_cache.full_seals
    rehashes_while_unchanged = environments.environment_authority_cache.rehashes
    assert second.status == third.status == "complete"
    assert scan_jsonl(paths.ledgers.models) == initial_models
    assert scan_jsonl(paths.ledgers.attempts) == initial_attempts

    authority = real_environment_fixture.binding.environment_authority
    assert authority is not None
    relative_member = next(
        entry.relative_path
        for entry in authority.content_manifest.entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.startswith("include/")
        and entry.relative_path.endswith(".h")
        and not entry.relative_path.startswith("conda-meta/")
    )
    changed = real_environment_fixture.prefix / relative_member
    source = real_environment_fixture.source_prefix / relative_member
    original_mode = changed.stat().st_mode
    changed.unlink()
    changed.write_bytes(source.read_bytes() + b"\n# round19 cache mutation\n")
    changed.chmod(original_mode)
    try:
        mutated = driver.run()
    finally:
        changed.unlink()
        os.link(source, changed)

    cache = environments.environment_authority_cache
    final_models = scan_jsonl(paths.ledgers.models)
    final_attempts = scan_jsonl(paths.ledgers.attempts)
    latest_by_id = {str(row["stable_id"]): row for row in final_models}
    new_attempts = final_attempts[len(initial_attempts) :]
    integrity_quarantines = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "environment-integrity-quarantined"
    ]

    assert {
        "first_status": first.status,
        "cheap_second_pass": cheap_after_second > 0,
        "cheap_third_pass_increased": cheap_after_third > cheap_after_second,
        "one_full_seal_while_unchanged": full_seals_while_unchanged == 1,
        "zero_rehashes_while_unchanged": rehashes_while_unchanged == 0,
        "mutation_added_one_cheap_validation": (cache.cheap_validations == cheap_after_third + 1),
        "full_seals": cache.full_seals,
        "rehashes": cache.rehashes,
        "invalidations": cache.invalidations,
        "mutation_status": mutated.status,
        "integrity_quarantines": len(integrity_quarantines),
        "terminal_environment_revisions": all(
            row["status"]["code"] == "failed:environment" for row in latest_by_id.values()
        ),
        "new_runs": sum(row["status"]["code"] == "runs" for row in final_models) - 3,
        "stale_prior_revisions": prior_revisions.isdisjoint(
            {str(row["record_revision"]) for row in latest_by_id.values()}
        ),
        "new_environment_failures": len(new_attempts) == 3
        and all(
            row["result"] == "failed" and row["stage"] == "environment" for row in new_attempts
        ),
        "old_identity_awards": any(
            row["result"] == "succeeded"
            and row.get("identities", {}).get("execution") in prior_execution_identities
            for row in new_attempts
        ),
    } == {
        "first_status": "complete",
        "cheap_second_pass": True,
        "cheap_third_pass_increased": True,
        "one_full_seal_while_unchanged": True,
        "zero_rehashes_while_unchanged": True,
        "mutation_added_one_cheap_validation": True,
        "full_seals": 2,
        "rehashes": 1,
        "invalidations": 1,
        "mutation_status": "complete",
        "integrity_quarantines": 1,
        "terminal_environment_revisions": True,
        "new_runs": 0,
        "stale_prior_revisions": True,
        "new_environment_failures": True,
        "old_identity_awards": False,
    }


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


def _typed_artifact(
    item: Any,
    paths: Any,
    driver: Any,
    context: Any,
    adapter_source: str,
    *,
    declared_members: Optional[Mapping[str, str]] = None,
) -> AuthorArtifact:
    """Stage one typed adapter and bind its complete declared code closure.

    Parameters
    ----------
    item, paths, driver, context:
        Existing real-composition driver authority and one intake item.
    adapter_source:
        Entry-point source written as ``adapter.py``.
    declared_members:
        Additional model-package members intentionally admitted by static imports.

    Returns
    -------
    AuthorArtifact
        Rebound proposed artifact consumed by the shipped compiler.
    """

    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)
    adapter_path = artifact.model_dir / "adapter.py"
    adapter_path.write_text(adapter_source, encoding="utf-8")
    for relative, source in (declared_members or {}).items():
        member = artifact.model_dir / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(source, encoding="utf-8")
    adapter_digest = hash_bytes(adapter_path.read_bytes())
    code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
    facts = artifact.proposal["proposed_facts"]
    facts["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": adapter_digest,
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
            "code_manifest": code_manifest,
        }
    )
    facts["input_contract"]["args"][0]["shape"] = [1, 2]
    facts["modes"]["meaningful_modes"] = ["eval"]
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
    facts["evidence"]["excerpts"][0]["supports"] = sorted(
        set(facts["evidence"]["excerpts"][0]["supports"])
        | {
            "implementation.code_manifest[].path",
            "implementation.code_manifest[].sha256",
        }
    )
    artifact.proposal["verified_hashes"]["code"] = adapter_digest
    artifact.proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    return _rebind_fake_author_result(artifact)


def _run_host_denial_composition(
    tmp_path: Path,
    fixture: RealEnvironmentFixture,
    *,
    expected_sandbox: str,
) -> None:
    """Run one positive package and one caught undeclared-read denial end to end.

    Parameters
    ----------
    tmp_path:
        Isolated campaign root.
    fixture:
        Strict real hardlink-clone environment authority.
    expected_sandbox:
        Host OS sandbox kind required by this release proof.
    """

    sandbox = detect_os_sandbox()
    if sandbox is None or sandbox.kind != expected_sandbox:
        message = f"{expected_sandbox} host enforcement is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    items = driver._ordered_work(snapshot, {})
    context = _test_authority_context(snapshot, driver.config)
    environment = fixture.binding
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            environment.family: environment.env_generation,
        },
    )

    positive = _typed_artifact(
        items[0],
        paths,
        driver,
        context,
        _DECLARED_ADAPTER,
        declared_members={
            "declared_model/__init__.py": "",
            "declared_model/layers.py": _DECLARED_MODEL,
        },
    )
    undeclared = paths.work_root / items[1].stable_id / "fake-model" / "undeclared_repo.py"
    denial = _typed_artifact(
        items[1],
        paths,
        driver,
        context,
        _denial_adapter(undeclared),
    )
    undeclared.write_text("UNDECLARED_REPOSITORY_MEMBER = True\n", encoding="utf-8")
    assert all(
        row["path"] != undeclared.name
        for row in denial.proposal["proposed_facts"]["implementation"]["code_manifest"]
    )

    with CanonicalReducer(paths.ledgers, context) as reducer:

        def append_attempt(attempt: Mapping[str, Any]) -> None:
            """Persist one worker-derived attempt through the canonical reducer."""

            reducer.append_attempt(attempt)

        positive = driver._stage_author_result(items[0], positive, reducer)
        positive_gate = (
            FakeChecker().check_metadata([positive], paths.work_root, driver.config).gate
        )
        assert positive_gate is not None
        reducer.append_gate(positive_gate)
        positive_attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            positive,
            environment,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=append_attempt,
        )
        assert len(positive_attempts) == 1
        assert positive_attempts[0]["result"] == "succeeded", positive_attempts[0]["error"]
        assert positive_attempts[0]["raw_award_receipt"] is not None
        assert (
            positive_attempts[0]["parent_attestation"]["named_raw_award_receipt_sha256"]
            == positive_attempts[0]["raw_award_receipt_sha256"]
        )
        positive_argv = list(positive_attempts[0]["invocation"]["argv"])
        interpreter_index = positive_argv.index(str(environment.python_executable))
        assert positive_argv[interpreter_index : interpreter_index + 4] == [
            str(environment.python_executable),
            "-B",
            "-m",
            "menagerie.crawler.worker",
        ]
        positive_model = driver_module._assemble_run_model(
            items[0], positive, positive_attempts, [positive_gate], driver.config
        )
        decisions = driver._license_decisions(positive)
        assert positive.staged is not None
        assert set(decisions) == {claim.claim_id for claim in positive.staged.custody_claims}
        object_digests = {obj.object_id: obj.content_sha256 for obj in positive.staged.objects}
        assert all(
            decisions[claim.claim_id].content_sha256 == object_digests[claim.object_id]
            for claim in positive.staged.custody_claims
        )
        driver._authorize_and_publish_artifact(
            positive,
            positive_model,
            [positive_gate],
            reducer,
        )
        reducer.append_model(reducer.prepare_model(positive_model))
        public_mirror = paths.runtime_root / "mirrors" / "public"
        positive_public_bytes = {
            path.relative_to(public_mirror).as_posix(): path.read_bytes()
            for path in public_mirror.rglob("*")
            if path.is_file()
        }
        assert positive_public_bytes

        denial = driver._stage_author_result(items[1], denial, reducer)
        denial_gate = FakeChecker().check_metadata([denial], paths.work_root, driver.config).gate
        assert denial_gate is not None
        reducer.append_gate(denial_gate)
        denial_attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            denial,
            environment,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=append_attempt,
        )
        assert len(denial_attempts) == 1
        assert denial_attempts[0]["result"] == "failed"
        assert denial_attempts[0]["stage"] == "policy"
        assert denial_attempts[0]["error"]["reason_code"] == "checkpoint-read"
        assert str(undeclared) in denial_attempts[0]["policy_observation"]["checkpoint_paths"]

        for artifact, attempts in ((positive, positive_attempts), (denial, denial_attempts)):
            closure = driver_module._collect_worker_executable_closure(artifact, environment)
            execution_identity = driver_module._execution_identity(
                artifact,
                environment,
                closure_identity=closure.identity,
            )
            recomputed = driver_module._compile_worker_read_manifest(
                artifact,
                environment,
                execution_identity,
                closure=closure,
            )
            assert {attempt["execution_read_manifest_identity"] for attempt in attempts} == {
                recomputed.manifest_id
            }

    persisted_attempts = scan_jsonl(paths.ledgers.attempts)
    denied_attempt = next(
        row for row in persisted_attempts if row["stable_id"] == items[1].stable_id
    )
    assert str(undeclared) in denied_attempt["policy_observation"]["checkpoint_paths"]
    models = scan_jsonl(paths.ledgers.models)
    assert [(row["stable_id"], row["status"]["code"]) for row in models] == [
        (items[0].stable_id, "runs")
    ]
    assert {
        path.relative_to(public_mirror).as_posix(): path.read_bytes()
        for path in public_mirror.rglob("*")
        if path.is_file()
    } == positive_public_bytes


def test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Real bwrap/strace denial persists while a declared package earns ``runs``."""

    if sys.platform != "linux":
        message = "Linux release host is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)
    fixture = request.getfixturevalue("real_environment_fixture")
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="bubblewrap")


def test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Real Seatbelt denial persists while the same declared package earns ``runs``."""

    if sys.platform != "darwin":
        message = "macOS release host is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)
    fixture = request.getfixturevalue("real_environment_fixture")
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="sandbox-exec")


def _macos_profile_manifest(tmp_path: Path) -> tuple[Any, dict[str, Path]]:
    """Build one freshly verified four-part v3 profile fixture.

    Parameters
    ----------
    tmp_path:
        Isolated authority tree.

    Returns
    -------
    tuple[Any, dict[str, pathlib.Path]]
        Shipped v3 manifest and its named semantic members.
    """

    prefix = tmp_path / "sealed-prefix"
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python")
    interpreter.chmod(0o755)
    runtime = prefix / "lib" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("RUNTIME = True\n", encoding="utf-8")
    external = tmp_path / "external-escape.py"
    external.write_text("EXTERNAL = True\n", encoding="utf-8")
    (prefix / "lib" / "external.py").symlink_to(external)
    authority = EnvironmentAuthorityCache().bind(
        prefix=prefix,
        selected_interpreter=interpreter,
        base_environment_generation=HASH,
    )
    model = tmp_path / "model.py"
    crawler = tmp_path / "crawler_bootstrap.py"
    asset = tmp_path / "standard-asset.txt"
    request_path = tmp_path / "request.json"
    for path, content in (
        (model, "MODEL = True\n"),
        (crawler, "CRAWLER = True\n"),
        (asset, "asset\n"),
        (request_path, "{}\n"),
    ):
        path.write_text(content, encoding="utf-8")
    manifest = compile_execution_read_manifest_v3(
        stable_id="m_macos_profile",
        work_id="work-macos-profile",
        execution_identity=HASH,
        code_manifest_identity=stable_hash([model.name]),
        environment_authority=authority,
        code_members=(
            RuntimeMember(
                model,
                hash_bytes(model.read_bytes()),
                "python-source",
                "accepted-model-code-manifest",
            ),
        ),
        worker_members=(
            RuntimeMember(
                crawler,
                hash_bytes(crawler.read_bytes()),
                "python-source",
                "crawler-worker-import-closure",
            ),
        ),
        standard_input_asset=(asset, hash_bytes(asset.read_bytes()), "standard-test-asset"),
    )
    return manifest, {
        "prefix": prefix.resolve(),
        "model": model.resolve(),
        "crawler": crawler.resolve(),
        "external": external.resolve(),
        "asset": asset.resolve(),
        "request": request_path.resolve(),
        "runtime": runtime.resolve(),
    }


def test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members(
    tmp_path: Path,
) -> None:
    """Seatbelt grants only one fresh v3 prefix and exact outside capabilities."""

    manifest, members = _macos_profile_manifest(tmp_path)
    scratch = tmp_path / "scratch"
    result = tmp_path / "result"
    profile = generate_macos_sandbox_profile(
        (scratch, result),
        allowed_read_paths=(members["request"],),
        execution_read_manifest=manifest,
    )

    prefix_grant = f"(allow file-read* (subpath {json.dumps(str(members['prefix']))}))"
    assert profile.count(prefix_grant) == 1
    subpath_rules = {
        line for line in profile.splitlines() if line.startswith("(allow file-read* (subpath ")
    }
    assert subpath_rules == {
        '(allow file-read* (subpath "/System"))',
        '(allow file-read* (subpath "/usr/lib"))',
        '(allow file-read* (subpath "/Library/Apple"))',
        '(allow file-read* (subpath "/private/etc"))',
        '(allow file-read* (subpath "/dev"))',
        f"(allow file-read* (subpath {json.dumps(str(scratch.resolve()))}))",
        f"(allow file-read* (subpath {json.dumps(str(result.resolve()))}))",
        prefix_grant,
    }
    for kind in ("model", "crawler", "external", "asset", "request"):
        literal = f"(allow file-read* (literal {json.dumps(str(members[kind]))}))"
        assert profile.count(literal) == 1
    assert "(regex" not in profile
    for forbidden_root in (
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path.home().resolve() / ".local" / "lib",
    ):
        assert f"(subpath {json.dumps(str(forbidden_root))})" not in profile
    assert f"(allow file-write* (subpath {json.dumps(str(members['prefix']))}))" not in profile
    assert str(members["runtime"]) not in profile


def test_macos_profile_refuses_stale_environment_prefix_grant(
    tmp_path: Path,
) -> None:
    """A stale v3 authority cannot create a Seatbelt prefix grant."""

    manifest, members = _macos_profile_manifest(tmp_path)
    members["runtime"].write_text("RUNTIME = False\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        generate_macos_sandbox_profile((), execution_read_manifest=manifest)
