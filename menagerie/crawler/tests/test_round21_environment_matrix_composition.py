"""Round-21 exhaustive real-prefix environment-unit proof matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.policy as policy_module
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    ExecutionReadManifestV3,
    environment_read_capability,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    DriverIntegrationError,
    SupervisedForwardLane,
    bind_materialized_environment,
)
from menagerie.crawler.env_lifecycle import canonical_probe_receipt_bytes
from menagerie.crawler.envs import EnvironmentIntent, LockArtifacts
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentFixtureFactory,
    RealEnvironmentSealCounter,
    _copy_up_real_environment_member,
    hardlink_bytes,
)
from menagerie.crawler.tests.test_environment_authority_composition import (
    _typed_artifact,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    _driver,
    _paths,
    _snapshot,
    _test_authority_context,
)


_FIXED_PACKAGE_ROW = {
    "name": "menagerie-round21-fixture",
    "version": "1.0",
    "build": "proof_0",
    "url": "https://example.test/menagerie-round21-fixture-1.0-proof_0.conda",
    "sha256": "sha256:" + "2" * 64,
}


@dataclass(frozen=True)
class _CompositionObservation:
    """Persisted result of one shipped real-prefix worker composition."""

    artifact: AuthorArtifact
    attempts: tuple[dict[str, Any], ...]
    manifest: ExecutionReadManifestV3
    models: tuple[dict[str, Any], ...]
    public_files: tuple[Path, ...]
    work_root: Path


EnvironmentProof = Callable[
    [
        Path,
        RealEnvironmentFixture,
        RealEnvironmentFixtureFactory,
        RealEnvironmentSealCounter,
    ],
    None,
]


def _site_packages(prefix: Path) -> Path:
    """Return the immediate site-packages directory below a real prefix.

    Parameters
    ----------
    prefix:
        Materialized real environment root.

    Returns
    -------
    pathlib.Path
        Unique selected immediate site-packages directory.
    """

    candidates = sorted(prefix.glob("lib/python*/site-packages"))
    if not candidates:
        raise AssertionError("real prefix has no immediate site-packages directory")
    return candidates[-1]


def _adapter_source(build_body: str) -> str:
    """Return a typed tiny-model adapter with one exact constructor action.

    Parameters
    ----------
    build_body:
        Python statements inserted into ``build_model`` before model construction.

    Returns
    -------
    str
        Complete trusted adapter source.
    """

    indented = "\n".join(f"    {line}" for line in build_body.splitlines())
    return f"""from __future__ import annotations

import importlib
import os
import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def build_model() -> object:
{indented}
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


def _dynamic_import_adapter(module_name: str, assertion: str) -> str:
    """Return an adapter that imports one runtime-only sealed module.

    Parameters
    ----------
    module_name, assertion:
        Dynamic module name and expression required after import.

    Returns
    -------
    str
        Complete typed adapter source.
    """

    return _adapter_source(f"module = importlib.import_module({module_name!r})\nassert {assertion}")


def _caught_import_adapter(module_name: str) -> str:
    """Return an adapter that catches absence of one runtime-only module.

    Parameters
    ----------
    module_name:
        Dynamic module deliberately absent for the positive control.

    Returns
    -------
    str
        Complete typed adapter source.
    """

    return _adapter_source(
        f"try:\n    importlib.import_module({module_name!r})\nexcept ModuleNotFoundError:\n    pass"
    )


def _run_composition(
    root: Path,
    fixture: RealEnvironmentFixture,
    adapter_source: str,
) -> _CompositionObservation:
    """Run, persist, reduce, and independently recompile one real worker attempt.

    Parameters
    ----------
    root:
        Isolated campaign root for one matrix cell.
    fixture:
        Strict real-prefix binding compiled by the shipped binder.
    adapter_source:
        Trusted model adapter executed by the shipped supervisor and worker.

    Returns
    -------
    _CompositionObservation
        Persisted attempt/model/public observations and independently rebuilt manifest.
    """

    root.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)
    driver = _driver(root, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    context = _test_authority_context(snapshot, driver.config)
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            fixture.binding.family: fixture.binding.env_generation,
        },
    )
    artifact = _typed_artifact(item, paths, driver, context, adapter_source)
    attempts: tuple[dict[str, Any], ...]
    with CanonicalReducer(paths.ledgers, context) as reducer:

        def append_attempt(attempt: Mapping[str, Any]) -> None:
            """Persist one worker-derived attempt through the canonical reducer."""

            reducer.append_attempt(attempt)

        staged = driver._stage_author_result(item, artifact, reducer)
        gate = FakeChecker().check_metadata([staged], paths.work_root, driver.config).gate
        if gate is None:
            raise AssertionError("matrix composition checker did not return its production gate")
        reducer.append_gate(gate)
        attempts = tuple(
            dict(attempt)
            for attempt in SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
                staged,
                fixture.binding,
                1,
                paths.work_root,
                worker_lock_path=paths.worker_lock,
                worker_lease_path=paths.worker_lease,
                run_id=driver.config.run_id,
                attempt_sink=append_attempt,
            )
        )
        if len(attempts) != 1:
            raise AssertionError("matrix composition did not produce exactly one mode attempt")

        closure = driver_module._collect_worker_executable_closure(staged, fixture.binding)
        execution_identity = driver_module._execution_identity(
            staged,
            fixture.binding,
            closure_identity=closure.identity,
        )
        manifest = driver_module._compile_worker_read_manifest(
            staged,
            fixture.binding,
            execution_identity,
            closure=closure,
        )
        if attempts[0]["execution_read_manifest_identity"] != manifest.manifest_id:
            raise AssertionError("persisted attempt disagrees with independent shipped compilation")

        if attempts[0]["result"] == "succeeded":
            model = driver_module._assemble_run_model(
                item,
                staged,
                attempts,
                [gate],
                driver.config,
            )
            decisions = driver._license_decisions(staged)
            if staged.staged is None:
                raise AssertionError("successful matrix artifact was not staged")
            if set(decisions) != {claim.claim_id for claim in staged.staged.custody_claims}:
                raise AssertionError("successful matrix artifact lacks complete license decisions")
            driver._authorize_and_publish_artifact(staged, model, [gate], reducer)
            reducer.append_model(reducer.prepare_model(model))

    public_root = paths.runtime_root / "mirrors" / "public"
    return _CompositionObservation(
        artifact=staged,
        attempts=attempts,
        manifest=manifest,
        models=tuple(dict(row) for row in scan_jsonl(paths.ledgers.models)),
        public_files=tuple(sorted(path for path in public_root.rglob("*") if path.is_file())),
        work_root=paths.work_root,
    )


def _assert_award(observation: _CompositionObservation) -> None:
    """Require one canonical ``runs`` award and published object.

    Parameters
    ----------
    observation:
        Completed shipped composition.
    """

    attempt = observation.attempts[0]
    assert attempt["result"] == "succeeded", attempt.get("error")
    assert attempt["raw_award_receipt"] is not None
    assert [model["status"]["code"] for model in observation.models] == ["runs"]
    assert observation.public_files


def _assert_checkpoint_poison(
    observation: _CompositionObservation,
    checkpoint_path: Path,
) -> None:
    """Require exact checkpoint poison with no model or public bytes.

    Parameters
    ----------
    observation:
        Completed shipped negative composition.
    checkpoint_path:
        Exact in-prefix model-data path that the trusted adapter read.
    """

    attempt = observation.attempts[0]
    assert attempt["result"] == "failed"
    assert attempt["stage"] == "policy"
    assert attempt["error"]["reason_code"] == "checkpoint-read"
    assert str(checkpoint_path) in attempt["policy_observation"]["checkpoint_paths"]
    assert attempt["raw_award_receipt"] is None
    assert observation.models == ()
    assert observation.public_files == ()


def _assert_stale_before_spawn(
    observation: _CompositionObservation,
    fixture: RealEnvironmentFixture,
) -> None:
    """Require a changed sealed prefix to invalidate before another child starts.

    Parameters
    ----------
    observation:
        Prior positive control whose staged artifact is reused.
    fixture:
        Mutated real-prefix authority that must refuse the next spawn.
    """

    cache = fixture.binding.environment_authority_cache
    if cache is None:
        raise AssertionError("matrix fixture lacks its lifecycle authority cache")
    prior_spawns = cache.real_spawns
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            observation.artifact,
            fixture.binding,
            1,
            observation.work_root,
            run_id="round21-stale-refusal",
        )
    assert cache.invalidations == 1
    assert cache.real_spawns == prior_spawns


def _proof_e01(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E01: a sealed runtime-only dynamic import awards without a forecast member."""

    del factory, counter
    observation = _run_composition(
        tmp_path,
        shared,
        _dynamic_import_adapter(
            shared.sentinel_module,
            "module.INTERPRETER_SENTINEL == 'round19-selected-prefix'",
        ),
    )
    _assert_award(observation)
    code_paths = observation.artifact.proposal["proposed_facts"]["implementation"]["code_manifest"]
    assert [row["path"] for row in code_paths] == ["adapter.py"]
    assert shared.sentinel_module not in json.dumps(code_paths)


def _proof_e02(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E02: a same-size, restored-mtime non-conda mutation quarantines authority."""

    del shared, counter
    fixture = factory(None)
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(
            fixture.sentinel_module,
            "module.INTERPRETER_SENTINEL == 'round19-selected-prefix'",
        ),
    )
    _assert_award(observation)
    authority = fixture.binding.environment_authority
    cache = fixture.binding.environment_authority_cache
    if authority is None or cache is None:
        raise AssertionError("E02 lacks its sealed authority")
    relative = next(
        entry.relative_path
        for entry in authority.content_manifest.entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.startswith("include/")
        and entry.relative_path.endswith(".h")
    )
    member = fixture.prefix / relative
    original, private_status = _copy_up_real_environment_member(
        member,
        fixture.source_prefix / relative,
    )
    cache.verify(authority)
    replacement = bytes([original[0] ^ 1]) + original[1:]
    member.write_bytes(replacement)
    member.chmod(stat.S_IMODE(private_status.st_mode))
    os.utime(member, ns=(private_status.st_atime_ns, private_status.st_mtime_ns))
    _assert_stale_before_spawn(observation, fixture)
    assert cache.rehashes == 2


def _replacement_intent(root: Path, fixture: RealEnvironmentFixture) -> EnvironmentIntent:
    """Create independently expected lock/export inputs for the fixed E03 package row.

    Parameters
    ----------
    root, fixture:
        New committed-artifact root and original exact real fixture.

    Returns
    -------
    EnvironmentIntent
        Replacement intent derived from original committed rows plus one fixed row.
    """

    original = json.loads(fixture.intent.lock.export_bytes)
    rows = [*original["packages"], dict(_FIXED_PACKAGE_ROW)]
    fields = ("name", "version", "build", "url", "sha256")
    rows.sort(key=lambda row: tuple(str(row[field]) for field in fields))
    export_bytes = (
        json.dumps({"packages": rows}, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    lock_bytes = fixture.intent.lock.lock_bytes.rstrip() + (
        "\n"
        + _FIXED_PACKAGE_ROW["url"]
        + "#"
        + _FIXED_PACKAGE_ROW["sha256"].removeprefix("sha256:")
        + "\n"
    ).encode("utf-8")
    root.mkdir(parents=True)
    target = "round21-replacement-host"
    lock_path = root / f"{target}.lock"
    export_path = root / f"{target}.resolved.json"
    export_hash_path = root / f"{target}.resolved.sha256"
    lock_path.write_bytes(lock_bytes)
    export_path.write_bytes(export_bytes)
    export_hash_path.write_text(f"{hash_bytes(export_bytes)}\n", encoding="utf-8")
    (root / f"{target}.probes.json").write_bytes(
        canonical_probe_receipt_bytes(fixture.probe_results)
    )
    return replace(
        fixture.intent,
        lock=LockArtifacts(
            target=target,
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=export_hash_path,
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            declared_export_hash=hash_bytes(export_bytes),
        ),
        generation=None,
    )


def _proof_e03(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E03: package metadata and package bytes force an expected strict rebuild."""

    del shared
    fixture = factory(None)
    authority = fixture.binding.environment_authority
    cache = fixture.binding.environment_authority_cache
    if authority is None or cache is None:
        raise AssertionError("E03 lacks its old sealed authority")
    site_packages = _site_packages(fixture.prefix)
    module = site_packages / "menagerie_round21_package.py"
    module.write_text("PACKAGE_SENTINEL = 'expected-replacement'\n", encoding="utf-8")
    metadata = fixture.prefix / "conda-meta" / "menagerie-round21-fixture-1.0-proof_0.json"
    metadata.write_text(
        json.dumps(_FIXED_PACKAGE_ROW, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AuthorityDerivationError, match="content seal"):
        cache.verify(authority)
    assert cache.invalidations == 1
    with pytest.raises(DriverIntegrationError, match="declared resolved export"):
        bind_materialized_environment(
            fixture.intent,
            fixture.prefix,
            fixture.probe_results,
            authority_cache=EnvironmentAuthorityCache(),
        )

    replacement_intent = _replacement_intent(tmp_path / "replacement-artifacts", fixture)
    replacement_cache = EnvironmentAuthorityCache()
    replacement_binding = bind_materialized_environment(
        replacement_intent,
        fixture.prefix,
        fixture.probe_results,
        authority_cache=replacement_cache,
    )
    counter.record_replacement(replacement_cache)
    assert replacement_binding.base_environment_generation != authority.base_environment_generation
    assert replacement_binding.environment_content_sha256 != authority.content_manifest_sha256
    assert replacement_binding.env_generation != authority.environment_generation
    rebuilt = RealEnvironmentFixture(
        source_prefix=fixture.source_prefix,
        prefix=fixture.prefix,
        binding=replacement_binding,
        intent=replacement_intent,
        probe_results=fixture.probe_results,
        sentinel_module=fixture.sentinel_module,
        startup_pth=fixture.startup_pth,
    )
    observation = _run_composition(
        tmp_path / "rebuilt-award",
        rebuilt,
        _dynamic_import_adapter(
            "menagerie_round21_package",
            "module.PACKAGE_SENTINEL == 'expected-replacement'",
        ),
    )
    _assert_award(observation)


def _proof_e04(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E04: a post-seal regular module addition refuses before spawn."""

    del shared, counter
    fixture = factory(None)
    module_name = "menagerie_round21_postseal_regular"
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _caught_import_adapter(module_name),
    )
    _assert_award(observation)
    (_site_packages(fixture.prefix) / f"{module_name}.py").write_text(
        "POSTSEAL = True\n", encoding="utf-8"
    )
    _assert_stale_before_spawn(observation, fixture)


def _proof_e05(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E05: an internal sealed symlink binds its text/target and awards."""

    del shared, counter
    link_name = "menagerie_round21_internal_link.py"

    def configure(root: Path, prefix: Path) -> None:
        """Add an internal module symlink before the shipped seal."""

        del root
        (_site_packages(prefix) / link_name).symlink_to("menagerie_round19_sentinel.py")

    fixture = factory(configure)
    authority = fixture.binding.environment_authority
    if authority is None:
        raise AssertionError("E05 lacks its sealed authority")
    relative = (_site_packages(fixture.prefix) / link_name).relative_to(fixture.prefix).as_posix()
    entry = next(
        item for item in authority.content_manifest.entries if item.relative_path == relative
    )
    assert entry.entry_type == "symlink"
    assert entry.link_text == "menagerie_round19_sentinel.py"
    assert (
        entry.resolved_target_relative_path
        == (_site_packages(fixture.prefix) / "menagerie_round19_sentinel.py")
        .relative_to(fixture.prefix)
        .as_posix()
    )
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(
            link_name.removesuffix(".py"),
            "module.INTERPRETER_SENTINEL == 'round19-selected-prefix'",
        ),
    )
    _assert_award(observation)


def _proof_e06(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E06: an exact external regular-file symlink projects once and awards."""

    del shared, counter
    link_name = "menagerie_round21_external_link.py"
    state: dict[str, Path] = {}

    def configure(root: Path, prefix: Path) -> None:
        """Add one exact external Python target and in-prefix symlink."""

        target = root / "external" / "round21_external_target.py"
        target.parent.mkdir(parents=True)
        target.write_text("EXTERNAL_SENTINEL = 'sealed-external'\n", encoding="utf-8")
        (_site_packages(prefix) / link_name).symlink_to(target)
        state["target"] = target.resolve()

    fixture = factory(configure)
    target = state["target"]
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(
            link_name.removesuffix(".py"),
            "module.EXTERNAL_SENTINEL == 'sealed-external'",
        ),
    )
    _assert_award(observation)
    authority = fixture.binding.environment_authority
    if authority is None:
        raise AssertionError("E06 lacks its sealed authority")
    assert [record.path for record in authority.external_targets].count(target) == 1
    capability = environment_read_capability(observation.manifest)
    assert capability.exact_member_paths.count(target) == 1
    mounts = policy_module._linux_minimal_read_mounts(  # noqa: SLF001
        (str(fixture.binding.python_executable),),
        Path.cwd(),
        capability.exact_member_paths,
    )
    assert mounts.count(target) == 1
    profile = policy_module.generate_macos_sandbox_profile(
        (), execution_read_manifest=observation.manifest
    )
    assert profile.count(f"(allow file-read* (literal {json.dumps(str(target))}))") == 1


def _proof_e07(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E07: restored-size/mtime external target mutation stales before spawn."""

    del shared, counter
    link_name = "menagerie_round21_external_mutable.py"
    original = b"EXTERNAL_SENTINEL = 'original-bytes'\n"
    replacement = b"EXTERNAL_SENTINEL = 'mutated-bytes!'\n"
    assert len(replacement) == len(original)
    state: dict[str, Path] = {}

    def configure(root: Path, prefix: Path) -> None:
        """Add one mutable external target and its sealed prefix link."""

        target = root / "external" / "round21_external_mutable.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        (_site_packages(prefix) / link_name).symlink_to(target)
        state["target"] = target

    fixture = factory(configure)
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(
            link_name.removesuffix(".py"),
            "module.EXTERNAL_SENTINEL == 'original-bytes'",
        ),
    )
    _assert_award(observation)
    target = state["target"]
    before = target.stat()
    target.write_bytes(replacement)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    _assert_stale_before_spawn(observation, fixture)


def _proof_e08(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E08: a post-seal unrecorded external symlink cannot widen authority."""

    del shared, counter
    fixture = factory(None)
    module_name = "menagerie_round21_postseal_external"
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _caught_import_adapter(module_name),
    )
    _assert_award(observation)
    authority = fixture.binding.environment_authority
    if authority is None:
        raise AssertionError("E08 lacks its sealed authority")
    assert authority.external_targets == ()
    target = tmp_path / "unrecorded-external.py"
    target.write_text("UNRECORDED = True\n", encoding="utf-8")
    (_site_packages(fixture.prefix) / f"{module_name}.py").symlink_to(target)
    assert authority.external_targets == ()
    _assert_stale_before_spawn(observation, fixture)


def _proof_e09(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E09: retargeting an internal sealed symlink refuses before spawn."""

    del shared, counter
    link_name = "menagerie_round21_retarget.py"

    def configure(root: Path, prefix: Path) -> None:
        """Add equal-length internal targets and one sealed module symlink."""

        site = _site_packages(prefix)
        hardlink_bytes(
            root / "round21-overlay" / "menagerie_round21_target_a.py",
            site / "menagerie_round21_target_a.py",
            b"VALUE = 'a'\n",
        )
        hardlink_bytes(
            root / "round21-overlay" / "menagerie_round21_target_b.py",
            site / "menagerie_round21_target_b.py",
            b"VALUE = 'b'\n",
        )
        (site / link_name).symlink_to("menagerie_round21_target_a.py")

    fixture = factory(configure)
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(link_name.removesuffix(".py"), "module.VALUE == 'a'"),
    )
    _assert_award(observation)
    link = _site_packages(fixture.prefix) / link_name
    link.unlink()
    link.symlink_to("menagerie_round21_target_b.py")
    _assert_stale_before_spawn(observation, fixture)


def _proof_e10(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E10: an external directory escape fails strict bind after a real award control."""

    del shared, counter
    fixture = factory(None)
    observation = _run_composition(
        tmp_path / "award",
        fixture,
        _dynamic_import_adapter(
            fixture.sentinel_module,
            "module.INTERPRETER_SENTINEL == 'round19-selected-prefix'",
        ),
    )
    _assert_award(observation)
    outside = tmp_path / "external-directory"
    outside.mkdir()
    (_site_packages(fixture.prefix) / "round21_directory_escape").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(AuthorityDerivationError, match="non-file target"):
        bind_materialized_environment(
            fixture.intent,
            fixture.prefix,
            fixture.probe_results,
            authority_cache=EnvironmentAuthorityCache(),
        )
    with pytest.raises(AuthorityDerivationError, match="selected interpreter.*prefix"):
        EnvironmentAuthorityCache().bind(
            prefix=fixture.prefix,
            selected_interpreter=Path("/bin/false"),
            base_environment_generation=str(fixture.binding.base_environment_generation),
        )


def _proof_e11(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E11: the sealed textual immediate startup ``.pth`` stays clean and awards."""

    del factory, counter
    authority = shared.binding.environment_authority
    if authority is None:
        raise AssertionError("E11 lacks its sealed authority")
    assert shared.startup_pth in authority.startup_pth_paths
    observation = _run_composition(
        tmp_path,
        shared,
        _adapter_source(
            "assert os.environ.get('MENAGERIE_ROUND19_PTH_SENTINEL') == 'sealed-startup'"
        ),
    )
    _assert_award(observation)
    assert (
        observation.attempts[0]["policy_observation"]["checkpoint_or_weight_read_attempted"]
        is False
    )


def _proof_e12(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E12: reading an in-prefix ``.pt`` poisons despite sealed-prefix authority."""

    del shared, counter
    state: dict[str, Path] = {}

    def configure(root: Path, prefix: Path) -> None:
        """Add one hardlinked checkpoint payload before the shipped seal."""

        checkpoint = _site_packages(prefix) / "menagerie_round21_checkpoint.pt"
        hardlink_bytes(
            root / "round21-overlay" / checkpoint.name,
            checkpoint,
            b"trusted-model-checkpoint-bytes\n",
        )
        state["checkpoint"] = checkpoint

    fixture = factory(configure)
    checkpoint = state["checkpoint"]
    observation = _run_composition(
        tmp_path / "poison",
        fixture,
        _adapter_source(f"with open({str(checkpoint)!r}, 'rb') as handle:\n    handle.read()"),
    )
    _assert_checkpoint_poison(observation, checkpoint)


def _proof_e13(
    tmp_path: Path,
    shared: RealEnvironmentFixture,
    factory: RealEnvironmentFixtureFactory,
    counter: RealEnvironmentSealCounter,
) -> None:
    """E13: a binary immediate ``.pth`` remains checkpoint poison, not startup config."""

    del shared, counter
    state: dict[str, Path] = {}

    def configure(root: Path, prefix: Path) -> None:
        """Add one binary immediate site-packages ``.pth`` before sealing."""

        checkpoint = _site_packages(prefix) / "menagerie_round21_binary.pth"
        hardlink_bytes(
            root / "round21-overlay" / checkpoint.name,
            checkpoint,
            b"\x00round21-binary-pth\n",
        )
        state["checkpoint"] = checkpoint

    fixture = factory(configure)
    checkpoint = state["checkpoint"]
    authority = fixture.binding.environment_authority
    if authority is None:
        raise AssertionError("E13 lacks its sealed authority")
    assert checkpoint in authority.startup_pth_paths
    observation = _run_composition(
        tmp_path / "poison",
        fixture,
        _adapter_source(f"with open({str(checkpoint)!r}, 'rb') as handle:\n    handle.read()"),
    )
    _assert_checkpoint_poison(observation, checkpoint)


_ENVIRONMENT_ENTRY_PROOF_REGISTRY: tuple[tuple[str, EnvironmentProof], ...] = (
    ("E01", _proof_e01),
    ("E05", _proof_e05),
    ("E06", _proof_e06),
    ("E10", _proof_e10),
    ("E11", _proof_e11),
)
_ENVIRONMENT_CHANGE_PROOF_REGISTRY: tuple[tuple[str, EnvironmentProof], ...] = (
    ("E02", _proof_e02),
    ("E03", _proof_e03),
    ("E04", _proof_e04),
    ("E07", _proof_e07),
    ("E08", _proof_e08),
    ("E09", _proof_e09),
)
_ENVIRONMENT_POLICY_PROOF_REGISTRY: tuple[tuple[str, EnvironmentProof], ...] = (
    ("E12", _proof_e12),
    ("E13", _proof_e13),
)
_ROUND21_ENVIRONMENT_MATRIX = tuple(
    sorted(
        (
            *_ENVIRONMENT_ENTRY_PROOF_REGISTRY,
            *_ENVIRONMENT_CHANGE_PROOF_REGISTRY,
            *_ENVIRONMENT_POLICY_PROOF_REGISTRY,
        ),
        key=lambda item: item[0],
    )
)


@pytest.mark.parametrize(
    ("cell_id", "proof"),
    _ROUND21_ENVIRONMENT_MATRIX,
    ids=[cell_id for cell_id, _proof in _ROUND21_ENVIRONMENT_MATRIX],
)
def test_round21_environment_unit_matrix(
    cell_id: str,
    proof: EnvironmentProof,
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    isolated_real_environment_factory: RealEnvironmentFixtureFactory,
    real_environment_seal_counter: RealEnvironmentSealCounter,
) -> None:
    """Execute every registry-derived E01--E13 cell as a real composition.

    Parameters
    ----------
    cell_id, proof:
        Exact closed matrix identifier and its real-prefix proof callable.
    tmp_path:
        Per-cell scratch root deleted immediately by the crawler test fixture.
    real_environment_fixture:
        Sole shared read-only strict real-prefix fixture.
    isolated_real_environment_factory:
        One-use disk-bounded private clone builder for destructive cells.
    real_environment_seal_counter:
        Session accounting for shared, isolated, and expected replacement seals.
    """

    expected_ids = tuple(f"E{index:02d}" for index in range(1, 14))
    registered_ids = tuple(value for value, _registered_proof in _ROUND21_ENVIRONMENT_MATRIX)
    assert registered_ids == expected_ids
    assert len(registered_ids) == len(set(registered_ids))
    assert cell_id in registered_ids
    assert set(structural.ROUND21_VS4_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "P04",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "T01",
        "T01-CI",
        "T02",
        "T03",
    }
    proof(
        tmp_path,
        real_environment_fixture,
        isolated_real_environment_factory,
        real_environment_seal_counter,
    )
