"""Full-contract synthetic fixtures for crawler Slice A tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator, NoReturn, Optional

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, ProposedAuthorResult
from menagerie.crawler.authority import (
    AuthorityContext,
    EnvironmentAuthorityCache,
    build_authority_context,
    completion_line_for_raw_award_receipt,
    derive_parent_attestation,
    raw_award_receipt_sha256,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3 as ATTEMPT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3 as AUTHOR_PROPOSAL_SCHEMA_VERSION,
    AUTHOR_PROMPT_NAME,
    CHECKER_PROMPT_NAME,
    GATE_SCHEMA_VERSION_V3 as GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3 as MODEL_SCHEMA_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    SourceRung,
    EnvironmentPhase,
)
from menagerie.crawler.driver import EnvironmentBinding, bind_materialized_environment
from menagerie.crawler.env_lifecycle import (
    LifecycleResult,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    canonical_probe_receipt_bytes,
    installed_package_inventory_bytes,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    ExportCheck,
    IntentProbes,
    LockArtifacts,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicensedArtifact,
    RedistributionClass,
    classify_redistribution,
    recompute_license_decision,
)
from menagerie.crawler.metadata import (
    authored_fact_leaves,
    recompute_accepted_identities,
)
from menagerie.crawler.proposal import ProposalValidationReport
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
)
from menagerie.crawler.standard_inputs import ASSET_ROOT
from menagerie.crawler.worker_supervisor import SupervisedResult, SupervisorObservation
from menagerie.crawler.policy import detect_os_sandbox

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64
NOW = "2026-07-14T12:00:00Z"

_MINIMAL_REAL_SITE_MEMBERS = (
    "_pytest",
    "attr",
    "attrs",
    "filelock",
    "fsspec",
    "iniconfig",
    "jinja2",
    "jsonschema",
    "jsonschema_specifications",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "pluggy",
    "py.py",
    "pytest",
    "referencing",
    "rpds",
    "sympy",
    "torchgen",
    "typing_extensions.py",
)
_MINIMAL_REAL_DIST_INFO_PREFIXES = (
    "filelock-",
    "fsspec-",
    "jinja2-",
    "MarkupSafe-",
    "mpmath-",
    "networkx-",
    "numpy-",
    "sympy-",
    "torch-",
    "typing_extensions-",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RELEASE_SPEC_PATH = _REPOSITORY_ROOT / "menagerie/crawler/envs/specs/round21-release.yml"
_RELEASE_PROBE_CONTRACT_PATH = _RELEASE_SPEC_PATH.with_name("round21-release.probes.json")
_ROUND21_LINUX_REGISTRY_PATH = Path(__file__).with_name("round21_linux_real_nodes.json")
_ROUND21_MACOS_REGISTRY_PATH = Path(__file__).with_name("round21_macos_real_nodes.json")
_ROUND21_RELEASE_REGISTRY_PATHS = {
    "linux-64": _ROUND21_LINUX_REGISTRY_PATH,
    "osx-arm64": _ROUND21_MACOS_REGISTRY_PATH,
}
_ROUND21_RELEASE_MARKERS = {
    "linux-64": "round21_linux_real",
    "osx-arm64": "round21_macos_real",
}
_ROUND21_RELEASE_COLLECTED: dict[str, set[str]] = {"linux-64": set(), "osx-arm64": set()}
_ROUND21_RELEASE_PASSED: dict[str, set[str]] = {"linux-64": set(), "osx-arm64": set()}
_ROUND21_RELEASE_SKIPPED: dict[str, set[str]] = {"linux-64": set(), "osx-arm64": set()}
_ROUND21_RELEASE_XFAILED: dict[str, set[str]] = {"linux-64": set(), "osx-arm64": set()}
_ROUND21_RELEASE_FAILED: dict[str, set[str]] = {"linux-64": set(), "osx-arm64": set()}
_ROUND21_RELEASE_CONTENT_DIGEST: Optional[str] = None
_ROUND21_RELEASE_PROBE_RESULTS: tuple[ProbeResult, ...] = ()

_FACT_KEYS = (
    "identity",
    "taxonomy",
    "external_metadata",
    "website",
    "people_and_origin",
    "dates",
    "citation",
    "licenses",
    "source_resolution",
    "evidence",
    "implementation",
    "input_contract",
    "modes",
    "fidelity",
)


@dataclass(frozen=True)
class RealEnvironmentFixture:
    """Session hardlink clone and strict production binding used by compositions."""

    source_prefix: Path
    prefix: Path
    binding: EnvironmentBinding
    intent: EnvironmentIntent
    probe_results: tuple[ProbeResult, ...]
    sentinel_module: str
    startup_pth: Path


RealEnvironmentConfigurator = Callable[[Path, Path], None]
RealEnvironmentFixtureFactory = Callable[
    [Optional[RealEnvironmentConfigurator]], RealEnvironmentFixture
]


@dataclass
class RealEnvironmentSealCounter:
    """Session accounting for shared and isolated real-prefix fixture seals."""

    shared_caches: list[EnvironmentAuthorityCache] = field(default_factory=list)
    isolated_caches: list[EnvironmentAuthorityCache] = field(default_factory=list)
    replacement_caches: list[EnvironmentAuthorityCache] = field(default_factory=list)
    base_seals: int = 0

    def record(self, cache: EnvironmentAuthorityCache, *, shared: bool) -> None:
        """Record one fixture cache immediately after its shipped strict bind.

        Parameters
        ----------
        cache:
            Production authority cache that performed the fixture's strict bind.
        shared:
            Whether the cache belongs to the sole session-shared fixture.
        """

        if cache.full_seals != 1:
            raise AssertionError("a real fixture bind must perform exactly one initial full seal")
        if shared and self.shared_caches:
            raise AssertionError("the pytest session attempted a second shared real fixture")
        caches = self.shared_caches if shared else self.isolated_caches
        caches.append(cache)
        self.base_seals += cache.full_seals

    def record_replacement(self, cache: EnvironmentAuthorityCache) -> None:
        """Record one independently expected replacement-generation seal.

        Parameters
        ----------
        cache:
            Production authority cache that strictly bound a rebuilt generation.
        """

        if cache.full_seals != 1:
            raise AssertionError("a replacement bind must perform exactly one full seal")
        self.replacement_caches.append(cache)

    def snapshot(self) -> dict[str, int]:
        """Return deterministic fixture-seal counts for composition assertions.

        Returns
        -------
        dict[str, int]
            Shared/isolated fixture counts and observed production full-seal totals.
        """

        shared_full_seals = sum(cache.full_seals for cache in self.shared_caches)
        isolated_full_seals = sum(cache.full_seals for cache in self.isolated_caches)
        isolated_rehashes = sum(cache.rehashes for cache in self.isolated_caches)
        replacement_full_seals = sum(cache.full_seals for cache in self.replacement_caches)
        return {
            "shared_fixtures": len(self.shared_caches),
            "isolated_fixtures": len(self.isolated_caches),
            "base_seals": self.base_seals,
            "shared_full_seals": shared_full_seals,
            "isolated_full_seals": isolated_full_seals,
            "isolated_rehashes": isolated_rehashes,
            "replacement_caches": len(self.replacement_caches),
            "replacement_full_seals": replacement_full_seals,
            "observed_full_seals": (
                shared_full_seals + isolated_full_seals + replacement_full_seals
            ),
            "maximum_full_seals": (
                len(self.shared_caches)
                + len(self.isolated_caches)
                + isolated_rehashes
                + len(self.replacement_caches)
            ),
        }

    def assert_bounded(self, *, require_shared: bool = False) -> None:
        """Assert one shared seal and only recorded mutation re-seals per isolate.

        Parameters
        ----------
        require_shared:
            Whether the caller requires the session-shared fixture to have been used.
        """

        counts = self.snapshot()
        expected_shared = 1 if require_shared else counts["shared_fixtures"]
        if counts["shared_fixtures"] > 1 or counts["shared_fixtures"] != expected_shared:
            raise AssertionError("the pytest session must own at most one shared real fixture")
        if counts["base_seals"] != counts["shared_fixtures"] + counts["isolated_fixtures"]:
            raise AssertionError("every real fixture must perform exactly one initial base seal")
        if counts["shared_full_seals"] != counts["shared_fixtures"]:
            raise AssertionError("the shared real environment was re-sealed")
        if counts["isolated_full_seals"] != (
            counts["isolated_fixtures"] + counts["isolated_rehashes"]
        ):
            raise AssertionError("an isolated real fixture performed an unexplained full seal")
        if counts["replacement_full_seals"] != counts["replacement_caches"]:
            raise AssertionError("a replacement generation performed an unexplained full seal")
        if counts["observed_full_seals"] > counts["maximum_full_seals"]:
            raise AssertionError("real fixture full seals exceeded the session bound")


class RealEnvironmentLane(SequentialEnvironmentLifecycle):
    """Expose a prebuilt strict real prefix through the production lifecycle type."""

    def __init__(self, fixture: RealEnvironmentFixture) -> None:
        """Store the already-probed hardlink clone used by real compositions.

        Parameters
        ----------
        fixture:
            Strictly bound session clone and its durable lifecycle artifacts.
        """

        self.fixture = fixture
        self.events: list[str] = []
        self._active = fixture.prefix
        self._authority_cache = fixture.binding.environment_authority_cache
        if self._authority_cache is None:
            raise AssertionError("real fixture lane requires its shipped strict-bind cache")

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: Any,
    ) -> LifecycleResult:
        """Invoke the driver callback with the prebuilt real prefix and probes.

        Parameters
        ----------
        intent:
            Registry intent. It must refer to the fixture's lifecycle artifacts.
        use:
            Driver callback that performs model work while the prefix is active.

        Returns
        -------
        LifecycleResult
            Minimal lifecycle result marker for tests.
        """

        if intent.lock != self.fixture.intent.lock or intent.probes != self.fixture.intent.probes:
            raise AssertionError("real fixture lane received a non-fixture environment intent")
        self.events.append(f"use:{intent.name}")
        use(self.fixture.prefix, self.fixture.probe_results)
        return LifecycleResult(
            intent=intent.name,
            target=intent.lock.target,
            export_sha256=str(intent.lock.declared_export_hash),
            probe_results=self.fixture.probe_results,
            disk_before=0,
            disk_after_create=0,
            disk_after_teardown=0,
            disk_recovery_checked=False,
        )


def real_environment_registry(
    fixture: RealEnvironmentFixture,
    *,
    intent_name: str = "core",
) -> EnvironmentRegistry:
    """Return a registry mapping production routes to the real fixture intent.

    Parameters
    ----------
    fixture:
        Strictly bound real clone whose artifacts define the sole test intent.
    intent_name:
        Intent name used by ordinary PyTorch routing.

    Returns
    -------
    EnvironmentRegistry
        Minimal registry whose routed intent strict-binds to the fixture prefix.
    """

    intent = EnvironmentIntent(
        name=intent_name,
        phase=fixture.intent.phase,
        framework=fixture.intent.framework,
        description=fixture.intent.description,
        split_guidance=fixture.intent.split_guidance,
        channels=fixture.intent.channels,
        dependencies=fixture.intent.dependencies,
        probes=fixture.intent.probes,
        lock=fixture.intent.lock,
        generation=fixture.intent.generation,
    )
    return EnvironmentRegistry(
        intents={intent_name: intent},
        phase_order=(EnvironmentPhase.PYTORCH,),
        small_set_target=True,
        hard_cap=None,
        global_split_guidance="round19 real fixture registry",
    )


def _real_environment_failure(message: str) -> NoReturn:
    """Fail a release gate or skip an unavailable optional local composition.

    Parameters
    ----------
    message:
        Exact unmet real-environment prerequisite.
    """

    if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
        pytest.fail(f"unmet-release-gate: {message}")
    pytest.skip(message)
    raise AssertionError("pytest.skip returned unexpectedly")


def _real_environment_source() -> Path:
    """Return the explicitly selected lock-built conda-family base prefix."""

    release_gate = os.environ.get("MENAGERIE_RELEASE_GATE") == "1"
    if release_gate and not os.environ.get("MENAGERIE_PLATFORM_LOCK"):
        _real_environment_failure("MENAGERIE_PLATFORM_LOCK is unavailable")
    value = os.environ.get("MENAGERIE_REAL_ENV_PREFIX")
    if not value and not release_gate:
        value = os.environ.get("CONDA_PREFIX")
    if not value:
        _real_environment_failure("MENAGERIE_REAL_ENV_PREFIX/CONDA_PREFIX is unavailable")
    prefix = Path(str(value)).resolve()
    if not (prefix / "conda-meta").is_dir() or not (prefix / "bin" / "python").exists():
        _real_environment_failure(f"selected prefix is not a materialized conda env: {prefix}")
    return prefix


def _release_probe_contract() -> IntentProbes:
    """Load the committed target-neutral release probe contract.

    Returns
    -------
    IntentProbes
        Exact import and export probes shared by every release host.
    """

    try:
        raw = json.loads(_RELEASE_PROBE_CONTRACT_PATH.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        _real_environment_failure(f"release probe contract is unavailable: {exc}")
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "imports",
        "export_checks",
        "source_build",
    }:
        _real_environment_failure("release probe contract has an invalid schema")
    imports = raw.get("imports")
    checks = raw.get("export_checks")
    if (
        raw.get("schema_version") != "menagerie.crawler.release-probes.v1"
        or not isinstance(imports, list)
        or len(imports) < 3
        or not all(isinstance(value, str) and value for value in imports)
        or not isinstance(checks, list)
        or raw.get("source_build") != []
    ):
        _real_environment_failure("release probe contract is incomplete")
    export_checks = []
    for raw_check in checks:
        if (
            not isinstance(raw_check, dict)
            or set(raw_check) != {"module", "attribute"}
            or not all(
                isinstance(raw_check.get(key), str) and raw_check.get(key)
                for key in ("module", "attribute")
            )
        ):
            _real_environment_failure("release export-check contract is malformed")
        export_checks.append(ExportCheck(str(raw_check["module"]), str(raw_check["attribute"])))
    return IntentProbes(tuple(imports), tuple(export_checks), ())


def _observe_release_probes(prefix: Path, probes: IntentProbes) -> tuple[ProbeResult, ...]:
    """Run the committed release probes under the selected prefix interpreter.

    Parameters
    ----------
    prefix:
        Lock-built prefix or its hardlink clone.
    probes:
        Committed target-neutral probe contract.

    Returns
    -------
    tuple[ProbeResult, ...]
        Canonical successful observations with stable details.
    """

    program = (
        "import importlib, json; "
        f"imports={list(probes.imports)!r}; "
        f"checks={[(check.module, check.attribute) for check in probes.export_checks]!r}; "
        "modules={name:importlib.import_module(name) for name in imports}; "
        "print(json.dumps({'exports':{name+'.'+attribute:"
        "str(getattr(modules[name],attribute)) for name,attribute in checks}},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            (str(prefix / "bin/python"), "-B", "-c", program),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        observation = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        stderr = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
        detail = f": {stderr[-1000:]}" if isinstance(stderr, str) and stderr else ""
        _real_environment_failure(f"committed release probes failed: {exc}{detail}")
    exports = observation.get("exports") if isinstance(observation, dict) else None
    if not isinstance(exports, dict):
        _real_environment_failure("release export observations are unavailable")
    results = [ProbeResult(f"import:{name}", True, f"imported {name}") for name in probes.imports]
    for check in probes.export_checks:
        key = f"{check.module}.{check.attribute}"
        value = exports.get(key)
        if not isinstance(value, str) or not value:
            _real_environment_failure(f"release export observation is absent: {key}")
        results.append(
            ProbeResult(f"export:{check.module}:{check.attribute}", True, f"{key}={value}")
        )
    return tuple(results)


def _committed_fixture_intent(
    prefix: Path,
    *,
    observe_probes: bool = True,
) -> tuple[EnvironmentIntent, tuple[ProbeResult, ...]]:
    """Load and independently verify the release fixture's committed lock family.

    Parameters
    ----------
    prefix:
        Hardlink clone of the clean lock-materialized prefix.
    observe_probes:
        Whether this intentionally executable fixture must rerun live probes.

    Returns
    -------
    tuple[EnvironmentIntent, tuple[ProbeResult, ...]]
        Strict binder intent and freshly observed committed probes.
    """

    lock_value = os.environ.get("MENAGERIE_PLATFORM_LOCK")
    if not lock_value:
        _real_environment_failure("MENAGERIE_PLATFORM_LOCK is unavailable")
    lock_path = Path(lock_value).resolve()
    lock_root = (_REPOSITORY_ROOT / "menagerie/crawler/envs/locks").resolve()
    if lock_path.parent != lock_root or not lock_path.name.startswith("round19-"):
        _real_environment_failure("release lock is not a committed platform artifact")
    export_path = lock_path.with_suffix(".resolved.json")
    export_hash_path = lock_path.with_suffix(".resolved.sha256")
    provenance_path = lock_path.with_suffix(".provenance.json")
    probe_receipt_path = lock_path.with_suffix(".probes.json")
    platform_target = lock_path.name.removeprefix("round19-").removesuffix(".lock")
    required = (
        _RELEASE_SPEC_PATH,
        _RELEASE_PROBE_CONTRACT_PATH,
        lock_path,
        export_path,
        export_hash_path,
        provenance_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        _real_environment_failure(
            "committed release artifacts are unavailable: "
            + ", ".join(path.name for path in missing)
        )
    if platform_target == "linux-64" and not probe_receipt_path.is_file():
        _real_environment_failure("committed Linux release probe receipt is unavailable")
    if platform_target != "linux-64" and probe_receipt_path.exists():
        _real_environment_failure("non-Linux release probes must be hosted observations")

    lock_bytes = lock_path.read_bytes()
    export_bytes = export_path.read_bytes()
    try:
        parse_exact_lock(lock_bytes)
        canonical_export = parse_resolved_export(export_bytes)
    except Exception as exc:
        _real_environment_failure(f"committed release lock family is invalid: {exc}")
    declared_export_hash = export_hash_path.read_text(encoding="utf-8").strip()
    if canonical_export != export_bytes or hash_bytes(export_bytes) != declared_export_hash:
        _real_environment_failure("committed resolved export digest is invalid")
    if installed_package_inventory_bytes(prefix) != export_bytes:
        _real_environment_failure(
            "created-prefix package inventory differs from the committed resolved export"
        )

    probes = _release_probe_contract()
    committed_probe_results: tuple[ProbeResult, ...] | None = None
    if probe_receipt_path.is_file():
        try:
            committed_probe_results = parse_probe_receipt_bytes(
                probes, probe_receipt_path.read_bytes()
            )
        except Exception as exc:
            _real_environment_failure(f"committed release probe receipt is invalid: {exc}")
    observed_probe_results = (
        _observe_release_probes(prefix, probes)
        if observe_probes or committed_probe_results is None
        else committed_probe_results
    )
    if committed_probe_results is not None and observed_probe_results != committed_probe_results:
        _real_environment_failure("live release probes differ from the committed receipt")

    try:
        provenance = json.loads(provenance_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        _real_environment_failure(f"release provenance is invalid: {exc}")
    artifact_target = lock_path.stem
    required_digests = {
        "spec_sha256": hash_bytes(_RELEASE_SPEC_PATH.read_bytes()),
        "probe_contract_sha256": hash_bytes(_RELEASE_PROBE_CONTRACT_PATH.read_bytes()),
        "lock_sha256": hash_bytes(lock_bytes),
        "resolved_export_sha256": hash_bytes(export_bytes),
    }
    if probe_receipt_path.is_file():
        required_digests["probe_receipt_sha256"] = hash_bytes(probe_receipt_path.read_bytes())
    committed_probe_state = provenance.get("probe_observation")
    expected_hosted_probe_state = (
        isinstance(committed_probe_state, dict)
        and committed_probe_state.get("committed_on_linux") is False
        and committed_probe_state.get("producer") == "hosted macOS release CI attestation"
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != "menagerie.crawler.release-lock-provenance.v1"
        or provenance.get("target") != platform_target
        or any(provenance.get(key) != value for key, value in required_digests.items())
        or (platform_target == "linux-64" and provenance.get("probe_receipt_sha256") is None)
        or (platform_target == "osx-arm64" and not expected_hosted_probe_state)
    ):
        _real_environment_failure("release provenance does not bind the committed lock family")
    clean_create = provenance.get("clean_create")
    if not isinstance(clean_create, dict):
        _real_environment_failure("release provenance lacks clean-create state")
    if platform_target == "linux-64" and clean_create.get("validated") is not True:
        _real_environment_failure("release provenance lacks native clean-create validation")
    if platform_target == "osx-arm64" and clean_create.get("validation_host") != (
        "hosted macOS release CI"
    ):
        _real_environment_failure("macOS clean-create validation is not delegated to hosted CI")

    spec = json.loads(_RELEASE_SPEC_PATH.read_bytes())
    if not isinstance(spec, dict):
        _real_environment_failure("release specification is malformed")
    channels = spec.get("channels")
    dependencies = spec.get("dependencies")
    if (
        not isinstance(channels, list)
        or not all(isinstance(value, str) and value for value in channels)
        or not isinstance(dependencies, list)
        or not all(isinstance(value, str) and value for value in dependencies)
    ):
        _real_environment_failure("release specification dependencies are malformed")
    intent = EnvironmentIntent(
        name="core",
        phase=EnvironmentPhase.PYTORCH,
        framework="pytorch",
        description="Round-21 committed-lock release fixture",
        split_guidance="release-proof-only",
        channels=tuple(channels),
        dependencies=tuple(dependencies),
        probes=probes,
        lock=LockArtifacts(
            target=artifact_target,
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=export_hash_path,
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            declared_export_hash=declared_export_hash,
        ),
        generation=None,
    )
    return intent, observed_probe_results


def _fixture_intent(
    artifact_root: Path,
    prefix: Path,
    *,
    observe_probes: bool = True,
) -> tuple[EnvironmentIntent, tuple[ProbeResult, ...]]:
    """Create strict lifecycle artifacts from the selected lock-built installation.

    Parameters
    ----------
    artifact_root, prefix:
        Outside-prefix artifact directory and hardlink clone.
    observe_probes:
        Whether an intentionally executable fixture must rerun live probes.

    Returns
    -------
    tuple[EnvironmentIntent, tuple[ProbeResult, ...]]
        Exact strict binder contract and durable successful probes.
    """

    if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
        return _committed_fixture_intent(prefix, observe_probes=observe_probes)

    artifact_root.mkdir(parents=True)
    package_bytes = installed_package_inventory_bytes(prefix)
    package_value = json.loads(package_bytes)
    package_rows = package_value["packages"]
    lock_bytes = (
        "@EXPLICIT\n"
        + "".join(
            f"{row['url']}#{str(row['sha256']).removeprefix('sha256:')}\n" for row in package_rows
        )
    ).encode("utf-8")
    target = "round19-real-host"
    lock_path = artifact_root / f"{target}.lock"
    export_path = artifact_root / f"{target}.resolved.json"
    export_hash_path = artifact_root / f"{target}.resolved.sha256"
    lock_path.write_bytes(lock_bytes)
    export_path.write_bytes(package_bytes)
    export_hash_path.write_text(f"{hash_bytes(package_bytes)}\n", encoding="utf-8")
    probes = IntentProbes(("torch", "menagerie_round19_sentinel"), (), ())
    probe_results = tuple(
        ProbeResult(f"import:{name}", True, f"real clone imported {name}")
        for name in probes.imports
    )
    (artifact_root / f"{target}.probes.json").write_bytes(
        canonical_probe_receipt_bytes(probe_results)
    )
    intent = EnvironmentIntent(
        name="core",
        phase=EnvironmentPhase.PYTORCH,
        framework="pytorch",
        description="Round-19 lock-built hardlink-clone release fixture",
        split_guidance="fixture-only",
        channels=("conda-forge",),
        dependencies=("python", "pytorch"),
        probes=probes,
        lock=LockArtifacts(
            target=target,
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=export_hash_path,
            lock_bytes=lock_bytes,
            export_bytes=package_bytes,
            declared_export_hash=hash_bytes(package_bytes),
        ),
        generation=None,
    )
    return intent, probe_results


def hardlink_clone_tree(source: Path, destination: Path) -> None:
    """Create one checked same-filesystem hardlink tree clone.

    Parameters
    ----------
    source, destination:
        Existing source tree and new destination directory.
    """

    destination.mkdir()
    subprocess.run(
        ("cp", "-al", f"{source}/.", str(destination)),
        check=True,
        capture_output=True,
        text=True,
    )


def hardlink_bytes(source: Path, destination: Path, content: bytes) -> None:
    """Create one file and a hardlinked destination with the same bytes.

    Parameters
    ----------
    source, destination:
        Outside staging file and exact hardlinked destination.
    content:
        Exact bytes shared by both paths.
    """

    source.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    os.link(source, destination)


def _copy_real_environment_member(source: Path, destination: Path) -> None:
    """Privately copy one selected real-environment member without following symlinks.

    Parameters
    ----------
    source, destination:
        Existing trusted member and exact destination path in the minimal private source.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ("cp", "-a", str(source), str(destination)),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _real_environment_failure(f"minimal real-environment member copy failed: {exc}")


def _linux_native_dependency_closure(
    source: Path,
    roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return trusted environment-local ELF dependencies needed by minimal Python/Torch.

    Parameters
    ----------
    source:
        Selected real conda-family prefix.
    roots:
        Executables and native extensions whose recursive ``ldd`` closure is required.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Canonical lexical dependency paths beneath ``source``.
    """

    pending = list(roots)
    inspected: set[Path] = set()
    dependencies: set[Path] = set()
    while pending:
        member = pending.pop()
        resolved = member.resolve(strict=True)
        if resolved in inspected:
            continue
        inspected.add(resolved)
        try:
            completed = subprocess.run(
                ("ldd", str(member)),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _real_environment_failure(f"minimal native dependency probe failed: {exc}")
        for line in completed.stdout.splitlines():
            if "=>" not in line:
                continue
            value = line.split("=>", 1)[1].split("(", 1)[0].strip()
            if not value.startswith("/"):
                continue
            dependency = Path(os.path.normpath(value))
            if not dependency.is_relative_to(source):
                continue
            dependencies.add(dependency)
            pending.append(dependency)
    return tuple(sorted(dependencies, key=lambda path: path.relative_to(source).as_posix()))


def _minimal_real_environment_source(source: Path, private_source: Path) -> None:
    """Build a small executable real prefix with sealed external native dependencies.

    Parameters
    ----------
    source, private_source:
        Selected full conda environment and empty per-test private-source directory.
    """

    site_candidates = sorted(source.glob("lib/python*/site-packages"))
    if not site_candidates:
        _real_environment_failure("selected prefix has no immediate site-packages directory")
    site_packages = site_candidates[-1]
    python_root = site_packages.parent
    private_site = private_source / site_packages.relative_to(source)

    interpreter = source / "bin" / "python"
    interpreter_target = interpreter.resolve(strict=True)
    _copy_real_environment_member(interpreter, private_source / interpreter.relative_to(source))
    _copy_real_environment_member(
        interpreter_target,
        private_source / interpreter_target.relative_to(source),
    )
    for member in sorted(python_root.iterdir(), key=lambda path: path.name):
        if member != site_packages:
            _copy_real_environment_member(member, private_source / member.relative_to(source))
    _copy_real_environment_member(source / "ssl", private_source / "ssl")

    torch_root = site_packages / "torch"
    if not torch_root.is_dir():
        _real_environment_failure("selected prefix has no real Torch package")
    for member in sorted(torch_root.iterdir(), key=lambda path: path.name):
        if member.name != "lib":
            _copy_real_environment_member(member, private_site / "torch" / member.name)
    copied_torch_symlinks = tuple(
        path
        for path in sorted(torch_root.rglob("*"), key=lambda path: str(path))
        if path.is_symlink() and path.relative_to(torch_root).parts[0] != "lib"
    )
    for symlink in copied_torch_symlinks:
        target = symlink.resolve(strict=True)
        if not target.is_relative_to(source):
            _real_environment_failure(f"Torch runtime symlink escapes selected prefix: {symlink}")
        destination = private_source / target.relative_to(source)
        if not destination.exists() and not destination.is_symlink():
            _copy_real_environment_member(target, destination)
    for name in _MINIMAL_REAL_SITE_MEMBERS:
        member = site_packages / name
        if not member.exists():
            _real_environment_failure(f"selected prefix lacks minimal runtime member: {name}")
        _copy_real_environment_member(member, private_site / name)
    for member in sorted(site_packages.glob("*.dist-info"), key=lambda path: path.name):
        if member.name.startswith(_MINIMAL_REAL_DIST_INFO_PREFIXES):
            _copy_real_environment_member(member, private_site / member.name)

    metadata = source / "conda-meta"
    if not metadata.is_dir():
        _real_environment_failure("selected prefix lacks real conda metadata")
    _copy_real_environment_member(metadata, private_source / "conda-meta")
    header = next(
        (
            path
            for path in sorted((source / "include").rglob("*.h"))
            if path.is_file() and path.stat().st_size > 0
        ),
        None,
    )
    if header is None:
        _real_environment_failure("selected prefix lacks a nonempty real native header")
    _copy_real_environment_member(header, private_source / header.relative_to(source))

    global_deps = torch_root / "lib" / "libtorch_global_deps.so"
    torch_shm_manager = torch_root / "bin" / "torch_shm_manager"
    torch_shm_manager_target = (
        torch_shm_manager.resolve(strict=True) if torch_shm_manager.is_file() else None
    )
    numpy_native_extensions = tuple(sorted((site_packages / "numpy").rglob("*.so")))
    if not numpy_native_extensions:
        _real_environment_failure("selected prefix has no real NumPy native extension")
    native_roots = (
        interpreter_target,
        global_deps,
        *((torch_shm_manager_target,) if torch_shm_manager_target is not None else ()),
        *tuple(sorted((python_root / "lib-dynload").glob("*.so"))),
        *tuple(sorted(torch_root.glob("_C*.so"))),
        *tuple(sorted((site_packages / "markupsafe").glob("*.so"))),
        *numpy_native_extensions,
        *tuple(sorted((site_packages / "numpy.libs").glob("*.so*"))),
    )
    dependencies = {
        global_deps,
        source / "lib" / "libgcc_s.so.1",
        source / "lib" / "libstdc++.so.6",
        *((torch_shm_manager_target,) if torch_shm_manager_target is not None else ()),
        *_linux_native_dependency_closure(source, native_roots),
    }
    for member in sorted(dependencies, key=lambda path: path.relative_to(source).as_posix()):
        destination = private_source / member.relative_to(source)
        if destination.exists() or destination.is_symlink():
            continue
        _copy_real_environment_member(member.resolve(strict=True), destination)


def _linux_reflinks_supported(source: Path, root: Path) -> bool:
    """Return whether GNU ``cp`` can require a real reflink in the fixture filesystem.

    Parameters
    ----------
    source, root:
        Selected full prefix and isolated fixture root used for the one-file probe.

    Returns
    -------
    bool
        ``True`` only when ``--reflink=always`` succeeds without a copy fallback.
    """

    probe = root / "reflink-probe"
    try:
        completed = subprocess.run(
            (
                "cp",
                "-a",
                "--reflink=always",
                str((source / "bin" / "python").resolve(strict=True)),
                str(probe),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    finally:
        probe.unlink(missing_ok=True)
    return completed.returncode == 0


def _private_real_environment_source(source: Path, root: Path) -> Path:
    """Create a disk-bounded source whose inodes are private to one mutating test.

    Parameters
    ----------
    source, root:
        Selected immutable environment and isolated fixture root.

    Returns
    -------
    pathlib.Path
        Private source used for the mutating test's required hardlink clone.
    """

    private_source = root / "private-source"
    private_source.mkdir()
    if sys.platform == "linux" and not _linux_reflinks_supported(source, root):
        _minimal_real_environment_source(source, private_source)
        return private_source
    command = (
        ("cp", "-a", "-c", f"{source}/.", str(private_source))
        if sys.platform == "darwin"
        else ("cp", "-a", "--reflink=always", f"{source}/.", str(private_source))
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        _real_environment_failure(f"private real-environment source copy failed: {exc}")
    return private_source


def _copy_up_real_environment_member(
    member: Path,
    source_member: Path,
) -> tuple[bytes, os.stat_result]:
    """Detach one sealed member before mutation and prove its source bytes stay unchanged.

    Parameters
    ----------
    member, source_member:
        Test-owned hardlink and the private-source member whose bytes must remain intact.

    Returns
    -------
    tuple[bytes, os.stat_result]
        Original bytes and the detached member's post-copy metadata baseline.
    """

    if not member.samefile(source_member):
        raise AssertionError("mutation member is not linked to its private source")
    original = source_member.read_bytes()
    before = member.stat()
    member.unlink()
    member.write_bytes(original)
    member.chmod(stat.S_IMODE(before.st_mode))
    os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = member.stat()
    if after.st_nlink != 1 or member.samefile(source_member):
        raise AssertionError("mutation member copy-up did not create a private inode")
    if source_member.read_bytes() != original:
        raise AssertionError("mutation member copy-up changed its private source")
    return original, after


def _build_real_environment_fixture(
    root: Path,
    counter: RealEnvironmentSealCounter,
    *,
    shared: bool,
    configure: Optional[RealEnvironmentConfigurator] = None,
) -> RealEnvironmentFixture:
    """Hardlink-clone, probe, and strictly seal one real Torch conda prefix.

    Parameters
    ----------
    root:
        Session-shared or function-isolated fixture root.
    counter:
        Session counter receiving the production cache after strict binding.
    shared:
        Whether to clone the immutable base directly or first make private inodes.
    configure:
        Optional test-owned pre-seal mutation applied only to an isolated clone.

    Returns
    -------
    RealEnvironmentFixture
        Strict binding and artifacts backed by one production authority cache.
    """

    source = _real_environment_source()
    clone_source = source if shared else _private_real_environment_source(source, root)
    prefix = root / "prefix"
    try:
        hardlink_clone_tree(clone_source, prefix)
    except (OSError, subprocess.CalledProcessError) as exc:
        _real_environment_failure(f"hardlink clone failed: {exc}")
    site_candidates = sorted(prefix.glob("lib/python*/site-packages"))
    if not site_candidates:
        _real_environment_failure("clone has no immediate site-packages directory")
    site_packages = site_candidates[-1]
    overlay = root / "overlay"
    overlay.mkdir()
    sentinel_source = overlay / "menagerie_round19_sentinel.py"
    sentinel_source.write_text(
        "INTERPRETER_SENTINEL = 'round19-selected-prefix'\n", encoding="utf-8"
    )
    os.link(sentinel_source, site_packages / sentinel_source.name)
    pth_source = overlay / "menagerie_round19_startup.pth"
    pth_source.write_text(
        "import os, sys; os.environ['MENAGERIE_ROUND19_PTH_SENTINEL']='sealed-startup'; "
        "os.environ['OPENSSL_CONF']=os.path.join(sys.prefix,'ssl','openssl.cnf')\n",
        encoding="utf-8",
    )
    startup_pth = site_packages / pth_source.name
    os.link(pth_source, startup_pth)
    if configure is not None:
        if shared:
            raise AssertionError("the session-shared real fixture cannot be configured")
        configure(root, prefix)
    interpreter = prefix / "bin" / "python"
    binary_immediate_pth = any(
        b"\x00" in path.read_bytes() for path in site_packages.glob("*.pth") if path.is_file()
    )
    if not binary_immediate_pth:
        try:
            completed = subprocess.run(
                (
                    str(interpreter),
                    "-B",
                    "-c",
                    "import json, os, sys, torch, menagerie_round19_sentinel as s; "
                    "print(json.dumps({'prefix':sys.prefix,'torch':torch.__version__,"
                    "'sentinel':s.INTERPRETER_SENTINEL,'pth':"
                    "os.environ.get('MENAGERIE_ROUND19_PTH_SENTINEL')}))",
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            observation = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            stderr = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
            detail = f": {stderr[-1000:]}" if isinstance(stderr, str) and stderr else ""
            _real_environment_failure(f"clone interpreter/Torch probe failed: {exc}{detail}")
        if Path(str(observation["prefix"])).resolve() != prefix.resolve():
            _real_environment_failure("clone interpreter does not report the clone prefix")
        if observation.get("sentinel") != "round19-selected-prefix":
            _real_environment_failure("environment-only sentinel is not importable")
        if observation.get("pth") != "sealed-startup":
            _real_environment_failure("sealed startup .pth effect is absent")
    if interpreter.resolve() == Path(sys.executable).resolve():
        _real_environment_failure("selected interpreter resolves to the driver interpreter")
    if detect_os_sandbox() is None:
        _real_environment_failure("required host sandbox/audit tools are unavailable")
    regular_files = [path for path in prefix.rglob("*") if path.is_file() and not path.is_symlink()]
    if not regular_files or any(path.stat().st_nlink <= 1 for path in regular_files):
        _real_environment_failure("hardlink clone contains a non-hardlinked regular file")
    if not any(path.suffix in {".so", ".dylib", ".pyd"} for path in regular_files):
        _real_environment_failure("clone has no real native extension/library")
    if not any(
        "dist-info" in path.parts or path.parent.name == "conda-meta" for path in regular_files
    ):
        _real_environment_failure("clone has no package metadata")
    intent, probe_results = _fixture_intent(
        root / "artifacts",
        prefix,
        observe_probes=not binary_immediate_pth,
    )
    authority_cache = EnvironmentAuthorityCache()
    binding = bind_materialized_environment(
        intent,
        prefix,
        probe_results,
        authority_cache=authority_cache,
    )
    if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
        global _ROUND21_RELEASE_CONTENT_DIGEST, _ROUND21_RELEASE_PROBE_RESULTS
        if binding.environment_authority is None:
            _real_environment_failure("release fixture has no sealed environment authority")
        _ROUND21_RELEASE_CONTENT_DIGEST = binding.environment_authority.content_manifest_sha256
        _ROUND21_RELEASE_PROBE_RESULTS = probe_results
    counter.record(authority_cache, shared=shared)
    return RealEnvironmentFixture(
        source_prefix=clone_source,
        prefix=prefix,
        binding=binding,
        intent=intent,
        probe_results=probe_results,
        sentinel_module="menagerie_round19_sentinel",
        startup_pth=startup_pth,
    )


@pytest.fixture(scope="session")
def real_environment_seal_counter() -> Iterator[RealEnvironmentSealCounter]:
    """Yield session accounting and enforce its seal bound during teardown.

    Yields
    ------
    RealEnvironmentSealCounter
        Mutable session counter populated only by real fixture strict binds.
    """

    counter = RealEnvironmentSealCounter()
    yield counter
    counter.assert_bounded()


@pytest.fixture(autouse=True)
def _clean_crawler_test_scratch(tmp_path: Path) -> Iterator[None]:
    """Remove crawler per-test scratch promptly, including retained failed-test trees."""

    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def real_environment_fixture(
    tmp_path_factory: pytest.TempPathFactory,
    real_environment_seal_counter: RealEnvironmentSealCounter,
) -> RealEnvironmentFixture:
    """Build the sole session-shared, read-only real environment binding."""

    root = tmp_path_factory.mktemp("round19-shared-real-environment")
    return _build_real_environment_fixture(root, real_environment_seal_counter, shared=True)


@pytest.fixture
def isolated_real_environment_fixture(
    tmp_path: Path,
    real_environment_seal_counter: RealEnvironmentSealCounter,
) -> RealEnvironmentFixture:
    """Build a fresh private-source clone and seal for one mutating composition."""

    root = tmp_path / "isolated-real-environment"
    root.mkdir()
    return _build_real_environment_fixture(root, real_environment_seal_counter, shared=False)


@pytest.fixture
def isolated_real_environment_factory(
    tmp_path: Path,
    real_environment_seal_counter: RealEnvironmentSealCounter,
) -> RealEnvironmentFixtureFactory:
    """Return a one-use builder for a pre-seal configured private real prefix.

    Returns
    -------
    RealEnvironmentFixtureFactory
        Disk-bounded factory using the existing private-source and hardlink-clone path.
    """

    used = False

    def build(
        configure: Optional[RealEnvironmentConfigurator] = None,
    ) -> RealEnvironmentFixture:
        """Build one isolated fixture with an optional pre-seal configurator.

        Parameters
        ----------
        configure:
            Test-owned callback invoked after the standard sentinel and startup ``.pth``
            are linked but before the shipped strict binder seals the prefix.

        Returns
        -------
        RealEnvironmentFixture
            Fresh private real-prefix fixture.
        """

        nonlocal used
        if used:
            raise AssertionError("an isolated real-environment factory is one-use")
        used = True
        root = tmp_path / "isolated-real-environment"
        root.mkdir()
        return _build_real_environment_fixture(
            root,
            real_environment_seal_counter,
            shared=False,
            configure=configure,
        )

    return build


def make_worker_result_v3_mapping(
    diagnostic: dict[str, Any],
    *,
    raw_award_receipt: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the sole synthetic worker-result.v3 wrapper mapping.

    Parameters
    ----------
    diagnostic:
        Nested worker-receipt.v1 diagnostic facts.
    raw_award_receipt:
        Optional success-only raw award receipt.
    Returns
    -------
    dict[str, Any]
        Production-shaped closed outer v3 wrapper.
    """

    normalized = deepcopy(diagnostic)
    normalized.pop("receipt_sha256", None)
    defaults: dict[str, Any] = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": "m_worker_result_fixture",
        "source_identity": "source-worker-result-fixture",
        "recipe_revision": HASH,
        "observed_recipe_revision": HASH,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": HASH,
        "observed_input_asset_sha256": None,
        "execution_identity": HASH,
        "seed": 0,
        "input_seed": 0,
        "mode": "eval",
        "device": "cpu",
        "framework": "pytorch",
        "awards_runs": False,
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "per_mode": {},
        "declared_meaningful_modes": ["eval"],
        "detected_meaningful_modes": ["eval"],
        "meaningful_modes": ["eval"],
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "policy_observation": {
            "network_attempted": False,
            "socket_targets": [],
            "checkpoint_or_weight_read_attempted": False,
            "checkpoint_paths": [],
            "write_outside_scratch_attempted": False,
            "write_paths": [],
            "credentials_present": False,
            "torchlens_import_attempted": False,
            "cache_read_attempted": False,
        },
        "error": None,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    raw_digest = (
        raw_award_receipt_sha256(raw_award_receipt) if raw_award_receipt is not None else None
    )
    outer_payload = {
        "result_version": "menagerie.crawler.worker-result.v3",
        "raw_award_receipt": deepcopy(raw_award_receipt),
        "raw_award_receipt_sha256": raw_digest,
        "diagnostic": normalized,
    }
    return {**outer_payload, "result_sha256": stable_hash(outer_payload)}


def make_supervised_worker_result_v3(
    observation: SupervisorObservation,
    diagnostic: dict[str, Any],
    *,
    raw_award_receipt: Optional[dict[str, Any]] = None,
    parent_attestation: Optional[dict[str, Any]] = None,
    receipt_error: Optional[str] = None,
    unattested_partial: Optional[dict[str, Any]] = None,
) -> SupervisedResult:
    """Build the sole synthetic live supervised v3 result fixture.

    Parameters
    ----------
    observation:
        Parent-owned process observation.
    diagnostic:
        Nested worker-receipt.v1 diagnostic facts.
    raw_award_receipt:
        Optional success-only raw award receipt.
    parent_attestation:
        Optional parent-owned v2 attestation.
    receipt_error:
        Optional supervisor loader error.
    unattested_partial:
        Optional non-awarding partial-result reference.

    Returns
    -------
    SupervisedResult
        Production-shaped outer v3 wrapper and matching supervisor projections.
    """

    outer = make_worker_result_v3_mapping(
        diagnostic,
        raw_award_receipt=raw_award_receipt,
    )
    raw_digest = outer["raw_award_receipt_sha256"]
    success_attestation = (
        str(parent_attestation["attestation_sha256"])
        if raw_award_receipt is not None and parent_attestation is not None
        else None
    )
    return SupervisedResult(
        observation=observation,
        worker_receipt=outer,
        receipt_error=receipt_error,
        success_attestation_sha256=success_attestation,
        raw_award_receipt=deepcopy(raw_award_receipt),
        raw_award_receipt_sha256=raw_digest,
        parent_attestation=deepcopy(parent_attestation),
        unattested_partial=deepcopy(unattested_partial),
    )


def make_licensed_artifact_fixture(
    mirrors: MirrorStore,
    content: bytes,
    *,
    staged_path: Path,
    origin: ArtifactOrigin,
    evidence: tuple[LicenseEvidence, ...],
    media_type: str = "application/octet-stream",
) -> LicensedArtifact:
    """Materialize explicit legacy mirror data for license-sweep unit fixtures.

    This test-only constructor carries no production authorization meaning. Tests of
    canonical publication use artifact transactions and reducer authorization directly.

    Parameters
    ----------
    mirrors, content, staged_path, origin, evidence, media_type:
        Explicit fixture storage and evidence inputs.

    Returns
    -------
    LicensedArtifact
        Data-only legacy manifest row used by checkpoint/license unit tests.
    """

    redistribution = classify_redistribution(evidence)
    public = redistribution is RedistributionClass.PUBLIC_OK
    manifest = mirrors.put(
        content,
        mirror_class=MirrorClass.PUBLIC if public else MirrorClass.PRIVATE,
        retention_class=(
            RetentionClass.DURABLE_PUBLIC if public else RetentionClass.RESTRICTED_PRIVATE
        ),
        origin=origin,
        media_type=media_type,
    )
    return LicensedArtifact(
        staged_path=staged_path,
        manifest=manifest,
        decision=recompute_license_decision(manifest.content_sha256, evidence),
    )


def make_authority_context(
    stable_ids: Any,
    *,
    snapshot_id: str = "snapshot-test",
    snapshot_sha256: str = HASH,
) -> AuthorityContext:
    """Build the mandatory production-shaped authority context for tests.

    Parameters
    ----------
    stable_ids:
        Iterable of stable model identifiers admitted by the synthetic intake.
    snapshot_id, snapshot_sha256:
        Exact synthetic or materialized intake snapshot identity.

    Returns
    -------
    AuthorityContext
        Context derived from exact shipped contract bytes and synthetic intake rows.
    """

    rows = tuple({"stable_id": str(stable_id)} for stable_id in stable_ids)
    return build_authority_context(
        active_intake_snapshot_id=snapshot_id,
        active_intake_snapshot_sha256=snapshot_sha256,
        intake_rows=rows,
        author_model="claude",
        author_version="test",
        checker_model="codex",
        checker_version="test",
        environment_generations={"env-test": HASH},
    )


def _checker_prompt_hash() -> str:
    """Return the exact frozen checker prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / f"{CHECKER_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _author_prompt_hash() -> str:
    """Return the exact frozen author prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / f"{AUTHOR_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _model_facts(model: dict[str, Any]) -> dict[str, Any]:
    """Extract proposal fact roots from a synthetic canonical model."""

    return {key: model[key] for key in _FACT_KEYS}


def _bind_model_identities(model: dict[str, Any]) -> None:
    """Populate synthetic model identity claims from its exact accepted facts."""

    identities = recompute_accepted_identities(
        _model_facts(model),
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["evidence"]["evidence_identity"] = identities.evidence
    model["implementation"]["recipe_revision"] = identities.recipe
    # Recipe includes evidence/implementation fields, so recompute once after embedding
    # the first-pass derived values.
    identities = recompute_accepted_identities(
        _model_facts(model),
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["implementation"]["recipe_revision"] = identities.recipe
    model["accuracy_gate"]["vet_identity"] = identities.vet
    model["accuracy_gate"]["prompt_sha256"] = _checker_prompt_hash()


def make_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-1",
    execution_identity: str = HASH,
    mode: Optional[str] = "eval",
) -> dict[str, Any]:
    """Build a complete valid forward attempt.

    Parameters
    ----------
    stable_id:
        Attempt model ID.
    attempt_id:
        Immutable attempt ID.
    execution_identity:
        Execution identity bound to the receipt.
    mode:
        Forward mode.

    Returns
    -------
    dict[str, Any]
        Complete attempt.v2 payload.
    """

    model: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "work_id": f"work-{stable_id}",
        "stable_id": stable_id,
        "attempt_no": 1,
        "parent_attempt_id": None,
        "actor": "worker",
        "stage": "forward",
        "mode": mode,
        "started_at": NOW,
        "finished_at": NOW,
        "result": "succeeded",
        "attempted_rungs": ["R1_LIBRARY"],
        "retries": {
            "stage_attempt": 1,
            "root_cause_repeat": 0,
            "author_round": 0,
            "gate_round": 0,
        },
        "identities": {
            "source": HASH,
            "evidence": HASH,
            "recipe": HASH,
            "environment": HASH,
            "execution": execution_identity,
            "runner": HASH,
            "author_prompt": HASH,
            "checker_prompt": HASH,
        },
        "environment": {
            "family": "core",
            "target": "test",
            "env_id": "env-test",
            "lock_sha256": HASH,
            "resolved_export_sha256": HASH,
            "python": "3.11",
            "packages_manifest_sha256": HASH,
            "compiler_identity": "test-compiler",
            "sdk_identity": "test-sdk",
            "authority_epoch": None,
            "base_environment_generation": None,
            "environment_content_sha256": None,
            "environment_authority_id": None,
            "selected_interpreter_relative_path": None,
            "selected_interpreter_digest": None,
            "external_escape_records": [],
        },
        "host": {
            "machine_id": "machine-test",
            "os": "linux",
            "os_build": "test",
            "architecture": "x86_64",
            "cpu": "test-cpu",
            "ram_bytes": 1024,
            "accelerator": None,
            "accelerator_runtime": None,
        },
        "invocation": {
            "argv": ["python", "worker.py"],
            "cwd": "/scratch",
            "safe_env": {"OFFLINE": "1"},
            "seed": 0,
            "device": "cpu",
            "mode": mode,
            "network_policy": "offline",
            "timeout_seconds": 300,
            "rss_limit_bytes": 1024,
            "scratch_limit_bytes": 1024,
        },
        "worker_receipt": {
            "present": True,
            "receipt_sha256": HASH,
            "observed_recipe_revision": HASH,
            "observed_adapter_sha256": None,
            "observed_code_manifest_sha256": HASH,
            "observed_input_asset_sha256": hash_bytes((ASSET_ROOT / "image.ppm").read_bytes()),
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": True,
            "forward_started": True,
            "forward_completed": True,
            "mode": mode,
            "input_signature": {
                "tree": {
                    "args": {"tuple": [{"leaf": 0}]},
                    "kwargs": {},
                },
                "leaves": [
                    {
                        "path": "input.args[0]",
                        "kind": "tensor",
                        "shape": [1, 3, 8, 8],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "output_signature": {
                "tree": {"leaf": 0},
                "leaves": [
                    {
                        "path": "output",
                        "kind": "tensor",
                        "shape": [1, 2],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "output_value_sha256": HASH,
            "input_kind": "standard-image",
            "input_asset": (
                f"standard:image.ppm:{hash_bytes((ASSET_ROOT / 'image.ppm').read_bytes())}"
            ),
            "input_note": "canonical test image",
            "parameter_count_total": 2,
            "parameter_count_trainable": 2,
            "native_framework": "pytorch",
            "delegated_method": "forward",
            "constructor_seconds": 0.1,
            "forward_seconds": 0.1,
        },
        "supervisor_observation": {
            "exit_code": 0,
            "signal": None,
            "wall_seconds": 0.1,
            "cpu_seconds": 0.1,
            "peak_rss_bytes": 128,
            "stdout_sha256": HASH,
            "stdout_bytes": 0,
            "stdout_tail": "",
            "stdout_completion_line": None,
            "stderr_sha256": HASH,
            "stderr_bytes": 0,
            "stderr_tail": "",
            "full_log_local_path": "/logs/test",
            "full_log_retention": "campaign",
        },
        "policy_observation": {
            "network_attempted": False,
            "socket_targets": [],
            "checkpoint_or_weight_read_attempted": False,
            "checkpoint_paths": [],
            "write_outside_scratch_attempted": False,
            "write_paths": [],
            "credentials_present": False,
            "torchlens_import_attempted": False,
            "cache_read_attempted": False,
        },
        "error": None,
        "defer_evidence": None,
        "capability_observation": None,
    }
    reference = make_model(stable_id, accepted=True)
    facts = _model_facts(reference)
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["identities"].update(
        {
            "source": identities.source,
            "evidence": identities.evidence,
            "recipe": identities.recipe,
            "checker_prompt": _checker_prompt_hash(),
        }
    )
    receipt = model["worker_receipt"]
    receipt["receipt_sha256"] = None
    receipt["observed_recipe_revision"] = identities.recipe
    raw_receipt = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": f"nonce-{attempt_id}",
        "request_sha256": HASH,
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "execution_identity": execution_identity,
        "recipe_revision": identities.recipe,
        "code_manifest_identity": HASH,
        "input_identity": hash_bytes((ASSET_ROOT / "image.ppm").read_bytes()),
        "requested_mode": mode,
        "observation": deepcopy(receipt),
    }
    completion_line = completion_line_for_raw_award_receipt(raw_receipt)
    completion_bytes = (completion_line + "\n").encode("utf-8")
    observation = model["supervisor_observation"]
    observation["stdout_sha256"] = hash_bytes(completion_bytes)
    observation["stdout_bytes"] = len(completion_bytes)
    # The public record keeps only the parent-attested TorchLens marker; arbitrary
    # worker stdout belongs in the gitignored local diagnostic sidecar.
    observation["stdout_completion_line"] = completion_line
    model.update(
        {
            "execution_read_manifest_identity": HASH,
            "raw_award_receipt": raw_receipt,
            "raw_award_receipt_sha256": raw_award_receipt_sha256(raw_receipt),
            "parent_attestation": derive_parent_attestation(
                raw_receipt,
                completion_line,
                observation,
                started_at=NOW,
                finished_at=NOW,
            ),
            "unattested_partial": None,
        }
    )
    return model


def make_failed_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-1",
    stage: str = "source",
    reason_code: str = "identity-unresolved",
) -> dict[str, Any]:
    """Build one reducer-valid failed attempt for terminal-evidence tests.

    Parameters
    ----------
    stable_id, attempt_id:
        Exact model and immutable attempt identities.
    stage, reason_code:
        Closed failure stage and reason.

    Returns
    -------
    dict[str, Any]
        Complete failed attempt payload with redacted diagnostics.
    """

    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    diagnostic = {
        "redaction": "externally-controlled-text-v1",
        "content_sha256": HASH,
        "local_path": f".crawl-local/diagnostics/{attempt_id}.json",
        "diagnostic_key": "$.error.message",
    }
    mode = "eval" if stage == "forward" else None
    attempt.update(
        {
            "actor": "driver",
            "stage": stage,
            "mode": mode,
            "result": "failed",
            "environment": None,
            "error": {
                "stage": stage,
                "reason_code": reason_code,
                "exception_type": "builtins.RuntimeError",
                "message": diagnostic,
                "traceback": None,
                "no_traceback_reason": "synthetic failure has no traceback",
                "native_crash": False,
                "root_cause_fingerprint": HASH,
                "details": {},
            },
        }
    )
    attempt["identities"]["environment"] = None
    attempt["identities"]["execution"] = None
    attempt["invocation"].update({"mode": mode, "network_policy": "not-invoked"})
    attempt["worker_receipt"].update(
        {
            "present": False,
            "receipt_sha256": None,
            "observed_recipe_revision": None,
            "observed_adapter_sha256": None,
            "observed_code_manifest_sha256": None,
            "observed_input_asset_sha256": None,
            "constructor_started": False,
            "constructor_completed": False,
            "input_completed": False,
            "forward_started": False,
            "forward_completed": False,
            "mode": mode,
            "input_signature": None,
            "output_signature": None,
            "output_value_sha256": None,
            "input_kind": None,
            "input_asset": None,
            "input_note": "worker was not invoked for this synthetic failure",
            "parameter_count_total": None,
            "parameter_count_trainable": None,
            "native_framework": None,
            "delegated_method": None,
        }
    )
    attempt["supervisor_observation"].update(
        {
            "exit_code": None,
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "stdout_sha256": None,
            "stdout_bytes": 0,
            "stdout_completion_line": None,
            "stderr_sha256": None,
            "stderr_bytes": 0,
            "full_log_local_path": "driver-observed",
        }
    )
    attempt["raw_award_receipt"] = None
    attempt["raw_award_receipt_sha256"] = None
    parent_attestation = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": f"driver-{attempt_id}",
        "request_sha256": HASH,
        "completion_line_sha256": None,
        "named_raw_award_receipt_sha256": None,
        "exit_code": None,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 0,
        "stdout_sha256": hash_bytes(b""),
        "stderr_sha256": hash_bytes(b""),
        "started_at": NOW,
        "finished_at": NOW,
    }
    parent_attestation["attestation_sha256"] = stable_hash(parent_attestation)
    attempt["parent_attestation"] = parent_attestation
    attempt["unattested_partial"] = None
    return attempt


def rebind_attempt_raw_proof(attempt: dict[str, Any]) -> dict[str, Any]:
    """Rebuild exact raw receipt and parent proof after a test mutates an attempt."""

    receipt = attempt["worker_receipt"]
    raw_receipt = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": f"nonce-{attempt['attempt_id']}",
        "request_sha256": HASH,
        "stable_id": attempt["stable_id"],
        "work_id": attempt["work_id"],
        "execution_identity": attempt["identities"]["execution"],
        "recipe_revision": attempt["identities"]["recipe"],
        "code_manifest_identity": receipt["observed_code_manifest_sha256"],
        "input_identity": receipt["observed_input_asset_sha256"] or HASH,
        "requested_mode": attempt["mode"],
        "observation": deepcopy(receipt),
    }
    line = completion_line_for_raw_award_receipt(raw_receipt)
    completion_bytes = (line + "\n").encode("utf-8")
    supervisor = attempt["supervisor_observation"]
    supervisor["stdout_completion_line"] = line
    supervisor["stdout_sha256"] = hash_bytes(completion_bytes)
    supervisor["stdout_bytes"] = len(completion_bytes)
    attempt["raw_award_receipt"] = raw_receipt
    attempt["raw_award_receipt_sha256"] = raw_award_receipt_sha256(raw_receipt)
    attempt["parent_attestation"] = derive_parent_attestation(
        raw_receipt,
        line,
        supervisor,
        started_at=str(attempt["started_at"]),
        finished_at=str(attempt["finished_at"]),
    )
    return attempt


def rebind_nonaward_parent_proof(attempt: dict[str, Any]) -> dict[str, Any]:
    """Rebuild exact parent-only proof after a fixture becomes non-awarding.

    Parameters
    ----------
    attempt:
        Current-v3 failed or observed attempt fixture.

    Returns
    -------
    dict[str, Any]
        The same fixture with closed empty-stream parent attestation.
    """

    parent = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": f"driver-{attempt['attempt_id']}",
        "request_sha256": HASH,
        "completion_line_sha256": None,
        "named_raw_award_receipt_sha256": None,
        "exit_code": attempt["supervisor_observation"].get("exit_code"),
        "signal": attempt["supervisor_observation"].get("signal"),
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": attempt["supervisor_observation"].get("peak_rss_bytes"),
        "stdout_sha256": attempt["supervisor_observation"].get("stdout_sha256") or hash_bytes(b""),
        "stderr_sha256": attempt["supervisor_observation"].get("stderr_sha256") or hash_bytes(b""),
        "started_at": attempt["started_at"],
        "finished_at": attempt["finished_at"],
    }
    parent["attestation_sha256"] = stable_hash(parent)
    attempt["parent_attestation"] = parent
    return attempt


def bind_terminal_attempts(model: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind a terminal model fixture to exact canonical attempt observations.

    Parameters
    ----------
    model:
        Non-run model fixture to update in place.
    attempts:
        Ordered canonical attempts supporting its terminal status.

    Returns
    -------
    dict[str, Any]
        The updated model fixture.
    """

    from menagerie.crawler.authority import derive_terminal_observation

    attempt_ids = [str(attempt["attempt_id"]) for attempt in attempts]
    model["status"]["attempt_ids"] = attempt_ids
    model["execution"]["accepted_attempt_ids"] = []
    model["observed"] = derive_terminal_observation(
        attempts,
        stable_id=str(model["stable_id"]),
        work_id=str(attempts[0]["work_id"]) if attempts else "not-applicable",
    )
    model["modes"]["per_mode_run"] = {
        str(attempt["mode"]): {
            "attempt_id": attempt["attempt_id"],
            "status": attempt["result"],
        }
        for attempt in attempts
        if attempt.get("mode") in model["modes"]["meaningful_modes"]
    }
    return model


def make_gate(
    stable_ids: Optional[list[str]] = None,
    *,
    gate_id: str = "gate-1",
    gate_kind: str = "metadata_batch",
    vet_identity: Optional[str] = None,
    fidelity_identity: Optional[str] = None,
) -> dict[str, Any]:
    """Build a complete valid metadata-batch or fidelity gate.

    Parameters
    ----------
    stable_ids:
        Item stable IDs. Metadata defaults to ten IDs; fidelity uses one.
    gate_id:
        Immutable gate ID.
    gate_kind:
        ``metadata_batch`` or ``fidelity``.
    vet_identity, fidelity_identity:
        Item identities.

    Returns
    -------
    dict[str, Any]
        Complete gate.v2 payload.
    """

    if stable_ids is None:
        stable_ids = [f"m_{index}" for index in range(10)]
    fidelity_required = gate_kind == "fidelity"
    items: list[dict[str, Any]] = []
    for stable_id in stable_ids:
        model = make_model(stable_id, accepted=True)
        item_vet_identity = str(vet_identity or model["accuracy_gate"]["vet_identity"])
        items.append(
            {
                "work_id": f"work-{stable_id}",
                "campaign_root_work_id": f"work-{stable_id}",
                "stable_id": stable_id,
                "family_representative_id": stable_id,
                "fidelity_identity": fidelity_identity if fidelity_required else None,
                "vet_identity": item_vet_identity,
                "verified_hashes": {
                    "proposal": HASH,
                    "source_manifest": HASH,
                    "evidence": HASH,
                    "code": None,
                    "source_to_code_map": HASH,
                    "family_template": None,
                },
                "integrity": {
                    "verdict": "accurate",
                    "hash_mismatches": [],
                    "excerpt_discrepancies": [],
                    "locator_failures": [],
                },
                "verdict": "accurate",
                "field_checks": [
                    {
                        "field": field,
                        "verdict": "accurate",
                        "evidence_ids": ["evidence-1"],
                        "checked_source_ids": ["source-1"],
                        "reason": "supported",
                        "required_repair": None,
                    }
                    for field in authored_fact_leaves(
                        _model_facts(model), schema_version=MODEL_SCHEMA_VERSION
                    )
                ],
                "fidelity": {
                    "required": fidelity_required,
                    "verdict": "match" if fidelity_required else "not-applicable",
                    "material_checks": [],
                    "unsupported_choices": [],
                    "contradictions": [],
                    "omissions": [],
                    "permanent_scar": False,
                },
                "rung_check": {
                    "selected_rung": "R1_LIBRARY",
                    "highest_applicable": "R1_LIBRARY",
                    "verdict": "accurate",
                    "findings": [],
                },
                "unsupported_claims": [],
                "required_repairs": [],
                "confidence": "high",
                "terminal_disposition": None,
            }
        )
    proposal = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_id": gate_id,
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "gate_kind": gate_kind,
        "batch_size": len(items),
        "gate_round": 1,
        "gate_identity": HASH,
        "checker": {
            "provider": "openai",
            "model": "codex",
            "version": "test",
            "prompt_sha256": _checker_prompt_hash(),
            "started_at": NOW,
            "finished_at": NOW,
        },
        "items": items,
        "result_envelope_sha256": HASH,
        "author_result_schema_identity": HASH,
        "dispatcher_identity": HASH,
    }
    proposal["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in proposal.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return proposal


def _citation() -> dict[str, Any]:
    """Return complete cited-work metadata.

    Returns
    -------
    dict[str, Any]
        Citation block.
    """

    return {
        "status": "present",
        "title": "Example Model",
        "authors": ["A. Author"],
        "year": 2020,
        "venue": "TestConf",
        "arxiv_id": None,
        "doi": None,
        "openreview_id": None,
        "url": "https://example.com/paper",
        "bibtex": None,
        "source_evidence_ids": ["evidence-1"],
    }


def make_model(
    stable_id: str = "m_example",
    *,
    accepted: bool = False,
    status_code: str = "runs",
    attempt_id: str = "attempt-1",
) -> dict[str, Any]:
    """Build a complete valid model revision.

    Parameters
    ----------
    stable_id:
        Model stable ID.
    accepted:
        Whether to populate every gated source-read block.
    status_code:
        Closed terminal status code.
    attempt_id:
        Per-mode accepted forward attempt.

    Returns
    -------
    dict[str, Any]
        Complete model.v3 payload with syntactically valid reducer-owned authority fields.
    """

    source = {
        "source_id": "source-1",
        "role": "implementation",
        "kind": "repository",
        "url": "https://example.com/model",
        "revision_kind": "commit",
        "revision": "abc123",
        "locator": "model.py",
        "content_sha256": HASH,
        "byte_count": 100,
        "media_type": "text/x-python",
        "retrieved_at": NOW,
        "fetch_recipe": "https-get",
        "mirror_class": "public",
        "mirror_digest": HASH,
    }
    metadata_blocks: dict[str, Any]
    if accepted:
        citation = _citation()
        metadata_blocks = {
            "taxonomy": {
                "family": "ExampleNet",
                "domains": ["vision"],
                "tasks": ["classification"],
                "modalities": ["vision"],
                "era": "modern",
                "architecture_tags": ["CNN"],
                "novel_ops": [],
            },
            "external_metadata": {
                "modality": ["vision"],
                "architecture_class": ["CNN"],
                "domain": ["computer vision"],
                "task": ["classification"],
                "field": "machine learning",
                "subfield": "computer vision",
                "paradigm": ["supervised"],
                "lineage": [],
                "predecessors": [],
                "tags": ["example"],
                "keywords": ["cnn"],
                "venue": "TestConf",
                "family": "ExampleNet",
                "era": "modern",
                "year": 2020,
                "country": "US",
                "authors": ["A. Author"],
                "institution": ["Example Lab"],
                "citation": citation,
                "license": "Apache-2.0",
                "key_contribution": "A grounded example.",
                "description": "A small source-grounded example network.",
                "original_framework": "pytorch",
                "run_framework": "pytorch",
                "modes": {
                    "meaningful_modes": ["eval"],
                    "train_eval_divergence": "none",
                },
            },
            "website": {
                "kind": "family-representative",
                "tagline": "A compact example network",
                "description": "A source-grounded example. It is used for integrity tests.",
                "key_contribution": "A grounded example.",
                "voice_version": "v1",
                "family_grounding_id": "grounding-1",
                "template_source_model_id": None,
                "variant_parameter_input_line": None,
                "template_hash": None,
            },
            "people_and_origin": {
                "authors": ["A. Author"],
                "labs": ["Example Lab"],
                "institutions": ["Example Institute"],
                "origin_countries": ["US"],
                "country_basis": "institution affiliation",
                "country_confidence": "high",
                "country_note": "Grounded in the paper.",
            },
            "dates": {
                "year": 2020,
                "year_basis": "paper publication",
                "first_public_date": "2020-01-01",
                "first_public_date_basis": "repository release",
            },
            "citation": citation,
            "licenses": {
                "code": {
                    "spdx": "Apache-2.0",
                    "status": "declared",
                    "source_id": "source-1",
                    "locator": "LICENSE",
                    "evidence_ids": ["evidence-1"],
                },
                "paper_text": {"status": "linked-not-redistributed", "source_id": "source-1"},
                "weights": {"status": "not-used"},
                "data": {
                    "spdx": None,
                    "status": "not-applicable",
                    "source_id": None,
                    "evidence_ids": [],
                },
                "redistribution_class": "public-compatible",
            },
        }
    else:
        metadata_blocks = {
            "taxonomy": None,
            "external_metadata": None,
            "website": None,
            "people_and_origin": None,
            "dates": None,
            "citation": None,
            "licenses": None,
        }
    kind = status_code.split(":", 1)[0]
    stage = status_code.split(":", 1)[1] if kind == "failed" else None
    reason = "identity-unresolved" if stage == "source" else None
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "record_seq": 1,
        "record_revision": HASH,
        "parent_revision": None,
        "created_at": NOW,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": "accepted" if accepted else "pending",
        "family_variant_derivation": None,
        "intake": {
            "snapshot_id": "snapshot-1",
            "snapshot_sha256": HASH,
            "legacy_row_sha256": None,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": [],
            "discovery_sources": ["master_catalog"],
        },
        "identity": {
            "canonical_name": "ExampleNet",
            "aliases": [],
            "acronym": None,
            "variant": "base",
            "variant_scope": "family",
            "family_representative_id": stable_id,
            "duplicate_of": None,
            "alias_of": None,
        },
        **metadata_blocks,
        "source_resolution": {
            "rung": "R1_LIBRARY",
            "decision": "official implementation",
            "rung_evidence": "source-1",
            "sufficiency_gap": None,
            "searched_at": NOW,
            "attempted_rungs": [
                {
                    "rung": "R1_LIBRARY",
                    "result": "selected",
                    "reason_code": "available",
                    "evidence_ids": ["evidence-1"],
                }
            ],
            "search_report": {
                "queries": ["ExampleNet implementation"],
                "places_checked": ["web"],
                "links_checked": ["https://example.com/model"],
                "languages_checked": ["en"],
                "archives_checked": [],
                "started_at": NOW,
                "finished_at": NOW,
                "conclusion": "Official implementation found.",
            },
            "mandatory_link_status": "ok",
            "primary_source_id": "source-1",
            "sources": [source],
        },
        "evidence": {
            "excerpts": [
                {
                    "evidence_id": "evidence-1",
                    "source_id": "source-1",
                    "locator": "README:1",
                    "text": "ExampleNet is a small convolutional network.",
                    "text_sha256": hash_bytes(b"ExampleNet is a small convolutional network."),
                    "supports": ["identity.canonical_name"],
                    "family_level": True,
                    "disposition": "supporting",
                    "license_disposition": "short-excerpt-committed",
                }
            ],
            "coverage": {
                "all_agent_fields_have_support": accepted,
                "missing_support": [] if accepted else ["authored_metadata"],
                "family_grounding_complete": accepted,
            },
            "evidence_identity": HASH,
            "family_grounding_path": None,
        },
        "implementation": {
            "original_framework": "pytorch",
            "run_framework": "pytorch",
            "native_object_type": "torch.nn.Module",
            "native_call_method": "forward",
            "transparent_forward_adapter": True,
            "recipe_type": "declarative-library",
            "code_path": None,
            "code_sha256": None,
            "builder_symbol": None,
            "dummy_call_symbol": None,
            "library_recipe": {
                "distribution": "example",
                "version": "1.0",
                "artifact_sha256": HASH,
                "module": "example",
                "symbol": "ExampleNet",
                "kwargs": {"weights": None},
                "pretrained_disable_fields": ["weights"],
            },
            "upstream_files": [],
            "patches": [],
            "source_to_code_map": [],
            "declared_choices": [],
            "initialization": {
                "policy": "random",
                "pretrained_disabled": True,
                "source_specified_choices": [],
            },
            "mode": "eval",
            "device_policy": "cpu",
            "required_construct_asset": None,
            "recipe_revision": HASH,
            "torchlens_import_static_check": "passed",
        },
        "input_contract": {
            "builder_symbol": "make_dummy_call",
            "seed": 0,
            "semantic_description": "One small RGB image.",
            "source_basis": ["evidence-1"],
            "smallest_valid_probe_rationale": "Smallest valid spatial extent.",
            "args": [
                {
                    "path": "args[0]",
                    "kind": "tensor",
                    "semantic_role": "image",
                    "shape": [1, 3, 8, 8],
                    "dtype": "float32",
                    "device_policy": "cpu",
                    "distribution": "normal",
                    "constraints": [],
                    "source_evidence_ids": ["evidence-1"],
                }
            ],
            "kwargs": [],
            "non_tensor_values": [],
            "masks_state_and_control": [],
            "expected_output_semantics": "class scores",
        },
        "observed": {
            "parameter_count_total": 2,
            "parameter_count_trainable": 2,
            "native_framework": "pytorch",
            "delegated_method": "forward",
            "output_signature": {
                "tree": {"leaf": 0},
                "leaves": [
                    {
                        "path": "output",
                        "kind": "tensor",
                        "shape": [1, 2],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "input_kind": "standard-image",
            "input_asset": (
                f"standard:image.ppm:{hash_bytes((ASSET_ROOT / 'image.ppm').read_bytes())}"
            ),
            "input_note": "canonical test image",
            "constructor_seconds": 0.1,
            "forward_seconds": 0.1,
            "peak_rss_bytes": 128,
            "measurement_attempt_ids": [attempt_id],
            "snippet": "driver-owned isolated forward",
            "snippet_sha256": stable_hash("driver-owned isolated forward"),
        },
        "modes": {
            "meaningful_modes": ["eval"],
            "per_mode_run": {"eval": {"attempt_id": attempt_id, "status": "succeeded"}},
            "train_eval_divergence": "none",
            "divergence_evidence": "single meaningful mode",
        },
        "fidelity": {
            "required": False,
            "reason": "R1 official library",
            "verdict": None,
            "fidelity_identity": None,
            "gate_id": None,
            "current": kind == "runs",
            "permanent_scar": False,
            "deviations": [],
        },
        "accuracy_gate": {
            "required": True,
            "vet_identity": HASH if accepted else None,
            "gate_id": "gate-1" if accepted else None,
            "verdict": "accurate" if accepted else None,
            "current": accepted,
            "checker_model": "codex",
            "checker_version": "test",
            "prompt_sha256": HASH,
        },
        "execution": {
            "execution_identity": HASH,
            "environment_id": "env-test",
            "env_generation": HASH,
            "accepted_attempt_ids": [attempt_id],
            "confirmation_policy": "single-mechanical",
            "network_attempted": False,
            "checkpoint_accessed": False,
            "last_verified_at": NOW,
            "current": kind == "runs",
        },
        "status": {
            "kind": kind,
            "code": status_code,
            "stage": stage,
            "reason_code": reason,
            # Failed diagnostics must be absent or sidecar-redacted at the reducer boundary.
            "detail": None,
            "traceback": None,
            "no_traceback_reason": "no Python exception" if kind == "failed" else None,
            "attempted_rungs": ["R1_LIBRARY"],
            "retries": {
                "source": 0,
                "fetch": 0,
                "evidence": 0,
                "author": 0,
                "gate": 0,
                "environment": 0,
                "import": 0,
                "constructor": 0,
                "input": 0,
                "forward": 0,
                "fidelity": 0,
            },
            "environment": "env-test",
            "timestamp": NOW,
            "attempt_ids": [attempt_id],
            "root_cause_fingerprint": HASH if kind == "failed" else None,
            "supersedes_revision": None,
            "human_review": {
                "required": False,
                "reason": None,
                "queue": None,
                "requested_at": None,
            },
        },
        "provenance": {
            "author_model": "claude",
            "author_version": "test",
            "author_prompt_sha256": HASH,
            "checker_model": "codex",
            "checker_version": "test",
            "producer_run_id": "run-test",
            "machine_id": "machine-test",
        },
        "budget": {
            "author_sessions_used": 0,
            "author_sessions_max": 3,
            "gate_rounds_used": 0,
            "run_revisions_used": 1,
            "explicit_grants": [],
        },
        "flags": [],
        "notes": "",
        "scar_history": [],
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": accepted,
            "evidence_coverage_complete": accepted,
            "accuracy_gate_current": accepted,
            "required_fidelity_current": True,
            "execution_current": kind == "runs",
            "family_template_valid": True,
            "release_eligible": accepted and kind == "runs",
            "issues": (
                []
                if accepted and kind == "runs"
                else ["authored-metadata-pending"]
                if kind == "runs"
                else [status_code]
            ),
        },
        "dependency_vector": {
            "intake_snapshot_id": "snapshot-1",
            "intake_snapshot_sha256": HASH,
            "intake_item_sha256": HASH,
            "author_result_schema_identity": "pending-untrusted",
            "author_dispatcher_identity": "pending-untrusted",
            "author_prompt_identity": "pending-untrusted",
            "checker_prompt_identity": "pending-untrusted",
            "terminal_rule_identity": "pending-untrusted",
            "status_proof_identity": "pending-untrusted",
            "source_manifest_identity": "pending-untrusted",
            "proposal_identity": "pending-untrusted",
            "author_result_identity": "pending-untrusted",
            "checker_gate_identity": "pending-untrusted",
            "recipe_revision": HASH,
            "runner_identity": "pending-untrusted",
            "award_closure_identity": "pending-untrusted",
            "environment_generation": HASH,
            "artifact_transaction_id": "not-applicable",
            "representative_revision": "not-applicable",
            "publication_policy_identity": "pending-untrusted",
            "accepted_attempt_ids": [attempt_id],
            "artifact_claim_ids": [],
        },
        "artifact_authority": {
            "state": "not-applicable",
            "transaction_id": "not-applicable",
            "committed_event_id": "not-applicable",
            "authorization_id": "not-applicable",
            "reconstruction_sha256": "not-applicable",
            "claim_ids": [],
        },
        "family_authority": {
            "binding_state": "ordinary",
            "representative_stable_id": stable_id,
            "representative_revision": "not-applicable",
            "representative_gate_id": "not-applicable",
            "representative_proposal_id": "not-applicable",
            "variant_token": "not-applicable",
            "template_source_revision": "not-applicable",
            "derivation_rule_identity": "not-applicable",
        },
    }
    if accepted:
        model["evidence"]["excerpts"][0]["supports"] = list(
            authored_fact_leaves(_model_facts(model), schema_version=MODEL_SCHEMA_VERSION)
        )
        _bind_model_identities(model)
    return model


def make_author_proposal(stable_id: str = "m_example") -> dict[str, Any]:
    """Build a complete staged author proposal.

    Parameters
    ----------
    stable_id:
        Proposed model ID.

    Returns
    -------
    dict[str, Any]
        Complete author-proposal.v3 payload.
    """

    model = make_model(stable_id, accepted=True)
    fact_keys = (
        "identity",
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
        "implementation",
        "input_contract",
        "modes",
        "fidelity",
    )
    proposal = {
        "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": "proposal-1",
        "proposal_sha256": HASH,
        "work_id": f"work-{stable_id}",
        "campaign_id": f"campaign-{stable_id}",
        "stable_id": stable_id,
        "intake_snapshot_id": "snapshot-test",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": stable_hash({"stable_id": stable_id}),
        "source_manifest_identity": HASH,
        "dispatcher_identity": HASH,
        "created_at": NOW,
        "author": {
            "provider": "anthropic",
            "model": "claude",
            "version": "test",
            "prompt_sha256": _author_prompt_hash(),
        },
        "source_identity": HASH,
        "evidence_identity": HASH,
        "recipe_revision": HASH,
        "fidelity_identity": None,
        "vet_identity": HASH,
        "verified_hashes": {
            "source_manifest": HASH,
            "evidence": HASH,
            "code": None,
            "source_to_code_map": HASH,
            "family_template": None,
        },
        "proposed_facts": {key: deepcopy(model[key]) for key in fact_keys},
    }
    identities = recompute_accepted_identities(
        proposal["proposed_facts"],
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="current",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "fidelity_identity": identities.fidelity,
            "vet_identity": identities.vet,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return proposal


def bind_handoff_execution(
    proposal: dict[str, Any],
    *,
    context: AuthorityContext,
    work_id: str,
    campaign_id: str,
    source_manifest_identity: str,
) -> dict[str, Any]:
    """Bind a proposal into one schema-valid executable deferral handoff.

    Parameters
    ----------
    proposal:
        Complete proposal fixture to bind in place.
    context:
        Active authority context supplying intake and dispatcher identities.
    work_id, campaign_id, source_manifest_identity:
        Exact terminal-result associations.

    Returns
    -------
    dict[str, Any]
        Closed handoff execution mapping.
    """

    stable_id = str(proposal["stable_id"])
    proposal.update(
        {
            "work_id": work_id,
            "campaign_id": campaign_id,
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
            "source_manifest_identity": source_manifest_identity,
            "dispatcher_identity": context.author_dispatcher_identity,
        }
    )
    proposal["author"]["prompt_sha256"] = context.author_prompt_identity
    proposal["verified_hashes"]["source_manifest"] = source_manifest_identity
    implementation = proposal["proposed_facts"]["implementation"]
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    code_manifest_identity = stable_hash(implementation.get("code_manifest") or [])
    handoff = {
        "proposal": proposal,
        "proposal_sha256": proposal["proposal_sha256"],
        "code_manifest_identity": code_manifest_identity,
        "source_manifest_identity": source_manifest_identity,
    }
    handoff["handoff_sha256"] = stable_hash(handoff)
    return handoff


def make_proposed_artifact(
    proposal: dict[str, Any], source_manifest: dict[str, Any], model_dir: Path
) -> Any:
    """Wrap an injected proposal in the mandatory typed author-result arm."""

    from menagerie.crawler.driver import AuthorArtifact  # noqa: PLC0415

    raw_result = {
        "result_id": f"result-{proposal['stable_id']}",
        "result_sha256": HASH,
        "stable_id": proposal["stable_id"],
        "work_id": proposal["work_id"],
        "campaign_id": proposal.get("campaign_id", proposal["work_id"]),
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": source_manifest.get("manifest_sha256", HASH),
        "intake_snapshot_id": proposal.get("intake_snapshot_id", "snapshot-test"),
        "intake_snapshot_sha256": proposal.get("intake_snapshot_sha256", HASH),
        "intake_item_sha256": proposal.get("intake_item_sha256", HASH),
        "created_at": proposal.get("created_at", NOW),
    }
    binding = AuthorResultBinding(raw_result=raw_result, **raw_result)
    report = ProposalValidationReport(
        stable_id=str(proposal["stable_id"]),
        rung=SourceRung(str(proposal["proposed_facts"]["source_resolution"]["rung"])),
        code_path=None,
        supported_claims=frozenset(),
    )
    result = ProposedAuthorResult(binding=binding, proposal=proposal, validation_report=report)
    return AuthorArtifact(result, source_manifest, model_dir)


def make_operational_event() -> dict[str, Any]:
    """Build a complete usage-pause operational event.

    Returns
    -------
    dict[str, Any]
        Complete operational-event.v1 payload.
    """

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": "event-1",
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "created_at": NOW,
        "event_kind": "usage-pause",
        "status": "paused:usage-limit",
        "provider": "anthropic",
        "observed_response": "limit reached",
        "reset_at": "2026-07-14T13:00:00Z",
        "queued_work_counts": {"author": 2},
        "current_environment": "env-test",
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {"wakeup": "scheduled"},
    }


def make_shutdown_interruption_event() -> dict[str, Any]:
    """Build the frozen operational-only shutdown interruption fixture.

    Returns
    -------
    dict[str, Any]
        Complete pre-spawn worker-shutdown-interrupted event.
    """

    event = make_operational_event()
    event.update(
        {
            "event_kind": "worker-shutdown-interrupted",
            "status": "interrupted:shutdown",
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "current_environment": None,
            "details": {
                "invocation_id": "invocation-1",
                "admission_boundary": "pre-spawn",
                "stable_id": "m_example",
                "work_id": "work-m_example",
                "execution_identity": HASH,
                "request_identity": None,
                "lease_id": None,
                "child_pid": None,
                "child_start_token": None,
                "child_pgid": None,
                "signal": None,
                "parent_observation": None,
                "partial_receipt": None,
            },
        }
    )
    return event


@pytest.fixture
def valid_model() -> dict[str, Any]:
    """Return a valid accepted model payload.

    Returns
    -------
    dict[str, Any]
        Accepted model payload.
    """

    return make_model(accepted=True)


def _round21_release_target_from_lock() -> str:
    """Return the release target selected by ``MENAGERIE_PLATFORM_LOCK``.

    Returns
    -------
    str
        Canonical target basename, currently ``linux-64`` or ``osx-arm64``.
    """

    lock_value = os.environ.get("MENAGERIE_PLATFORM_LOCK", "round19-linux-64.lock")
    name = Path(lock_value).name
    if name.startswith("round19-") and name.endswith(".lock"):
        return name.removeprefix("round19-").removesuffix(".lock")
    return "linux-64"


def _round21_release_nodes(target: str) -> tuple[str, ...]:
    """Load one exact checked-in release proof node set.

    Parameters
    ----------
    target:
        Platform target for the registry.

    Returns
    -------
    tuple[str, ...]
        Full parameterized pytest node IDs in canonical registry order.
    """

    registry_path = _ROUND21_RELEASE_REGISTRY_PATHS.get(target)
    if registry_path is None:
        raise pytest.UsageError(f"unmet-release-gate: unsupported release target {target!r}")
    payload = json.loads(registry_path.read_bytes())
    if not isinstance(payload, dict) or payload.get("target") != target:
        raise pytest.UsageError(f"unmet-release-gate: invalid {target} release proof registry")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(node, str) for node in nodes):
        raise pytest.UsageError(f"unmet-release-gate: empty {target} release proof registry")
    if len(nodes) != len(set(nodes)):
        raise pytest.UsageError(f"unmet-release-gate: duplicate {target} release proof node")
    return tuple(nodes)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply release markers from exact checked-in registries.

    Parameters
    ----------
    items:
        Fully expanded pytest items before marker-expression deselection.
    """

    for target, marker_name in _ROUND21_RELEASE_MARKERS.items():
        expected = frozenset(_round21_release_nodes(target))
        for item in items:
            if item.nodeid in expected:
                item.add_marker(getattr(pytest.mark, marker_name))
                _ROUND21_RELEASE_COLLECTED[target].add(item.nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record terminal outcomes for registered release nodes.

    Parameters
    ----------
    report:
        One pytest setup/call/teardown phase report.
    """

    for target in _ROUND21_RELEASE_REGISTRY_PATHS:
        if report.nodeid not in frozenset(_round21_release_nodes(target)):
            continue
        if report.skipped:
            destination = (
                _ROUND21_RELEASE_XFAILED[target]
                if hasattr(report, "wasxfail")
                else _ROUND21_RELEASE_SKIPPED[target]
            )
            destination.add(report.nodeid)
        elif report.failed:
            _ROUND21_RELEASE_FAILED[target].add(report.nodeid)
        elif report.when == "call" and report.passed:
            _ROUND21_RELEASE_PASSED[target].add(report.nodeid)


def _round21_release_commit() -> str:
    """Return the CI-provided or repository-observed release commit.

    Returns
    -------
    str
        Full Git commit SHA.
    """

    provided = os.environ.get("GITHUB_SHA")
    if provided:
        return provided
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=_REPOSITORY_ROOT,
    )
    return completed.stdout.strip()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Fail closed and emit the release pass/not-skip attestation.

    Parameters
    ----------
    session:
        Completed pytest session whose exit status may be made fail-closed.
    exitstatus:
        Pytest's pre-attestation session result.
    """

    attestation_value = os.environ.get("MENAGERIE_RELEASE_ATTESTATION")
    if not attestation_value:
        return
    target = _round21_release_target_from_lock()
    expected = frozenset(_round21_release_nodes(target))
    collected = frozenset(_ROUND21_RELEASE_COLLECTED[target])
    passed = frozenset(_ROUND21_RELEASE_PASSED[target])
    skipped = frozenset(_ROUND21_RELEASE_SKIPPED[target])
    xfailed = frozenset(_ROUND21_RELEASE_XFAILED[target])
    failed = frozenset(_ROUND21_RELEASE_FAILED[target])
    lock_path = Path(os.environ.get("MENAGERIE_PLATFORM_LOCK", ""))
    export_path = lock_path.with_suffix(".resolved.json")
    provenance_path = lock_path.with_suffix(".provenance.json")
    complete = (
        os.environ.get("MENAGERIE_RELEASE_GATE") == "1"
        and collected == expected
        and passed == expected
        and not skipped
        and not xfailed
        and not failed
        and int(exitstatus) == int(pytest.ExitCode.OK)
        and _ROUND21_RELEASE_CONTENT_DIGEST is not None
        and lock_path.is_file()
        and export_path.is_file()
        and provenance_path.is_file()
    )
    sandbox = detect_os_sandbox()
    prefix = Path(os.environ.get("MENAGERIE_REAL_ENV_PREFIX", ""))
    attestation = {
        "schema_version": "menagerie.crawler.release-proof-attestation.v1",
        "status": "passed" if complete else "unmet-release-gate",
        "target": target,
        "commit_sha": _round21_release_commit(),
        "lock_path": lock_path.as_posix(),
        "lock_sha256": hash_bytes(lock_path.read_bytes()) if lock_path.is_file() else None,
        "resolved_export_sha256": (
            hash_bytes(export_path.read_bytes()) if export_path.is_file() else None
        ),
        "provenance_sha256": (
            hash_bytes(provenance_path.read_bytes()) if provenance_path.is_file() else None
        ),
        "environment_content_digest": _ROUND21_RELEASE_CONTENT_DIGEST,
        "probe_results": [
            {"name": result.name, "passed": result.passed, "detail": result.detail}
            for result in _ROUND21_RELEASE_PROBE_RESULTS
        ],
        "host": {"system": platform.system(), "machine": platform.machine()},
        "sandbox": None if sandbox is None else sandbox.kind,
        "selected_interpreter": str((prefix / "bin/python").resolve()),
        "expected_nodes": sorted(expected),
        "collected_nodes": sorted(collected),
        "passed_nodes": sorted(passed),
        "skipped_nodes": sorted(skipped),
        "xfailed_nodes": sorted(xfailed),
        "failed_nodes": sorted(failed),
    }
    attestation_path = Path(attestation_value)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = attestation_path.with_suffix(f"{attestation_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(attestation_path)
    if not complete:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
