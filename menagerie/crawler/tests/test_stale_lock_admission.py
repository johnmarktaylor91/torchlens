"""Superseded-lock detection and pre-normalization capacity withholding.

Both defects here killed the 2026-08-04 pilot rung's admission honesty:

* Lock artifacts are the environment lane's own solve residue, rewritten on
  every run. When ``segmentation-models-pytorch`` was declared in ``core``
  AFTER that run's solve, the residue export (121 packages, no smp) kept being
  read as the routed environment's truth, so every smp model was refused with
  "no package row carries that name" -- a false permanent wall, because the
  very next solve of the declaration installs it. A lock whose resolved export
  no longer satisfies its own intent's declared dependencies is now
  ``superseded``: an unknown inventory, never a stale certainty.

* Full-config Mixtral (m5915, ~31B-parameter lower bound) terminalized
  ``failed:runner`` / ``protocol-violation`` on a transformers version drift
  inside ``_normalize_artifact_modes`` before any size check ran, so
  ``capacity-deferrals`` reported zero. The size is knowable from the authored
  recipe alone, so the capacity withhold now runs in the author lane, before
  normalization can refuse for reasons of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.constants import (
    EnvironmentPhase,
    OperationalEventKind,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_admission import _routed_environment_packages
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.environment_coverage import (
    CoverageBasis,
    CoverageVerdict,
    assess_environment_coverage,
    find_covering_intents,
)
from menagerie.crawler.envs import LockArtifacts
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    HOST_MEMORY_ENV_VAR,
    capacity_deferral_path,
    load_capacity_deferral_rows,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.recipe import RecipeError, bind_library_artifact_digest
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.tests.test_host_capacity import MIXTRAL_RECIPE

pytestmark = pytest.mark.smoke

#: The pilot residue in miniature: the core solve that ran BEFORE the
#: declaration gained segmentation-models-pytorch, in the closed exact
#: resolved-export row format (``env_lifecycle._PACKAGE_FIELDS``).
_RESIDUE_PACKAGES: list[dict[str, str]] = [
    {
        "name": "pytorch",
        "version": "2.13.0",
        "build": "cpu_generic_0",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/pytorch-2.13.0.conda",
        "sha256": "sha256:" + "a" * 64,
    },
    {
        "name": "timm",
        "version": "1.0.24",
        "build": "pyhd8ed1ab_0",
        "url": "https://conda.anaconda.org/conda-forge/noarch/timm-1.0.24.conda",
        "sha256": "sha256:" + "b" * 64,
    },
    {
        "name": "transformers",
        "version": "5.14.1",
        "build": "pyhd8ed1ab_0",
        "url": "https://conda.anaconda.org/conda-forge/noarch/transformers-5.14.1.conda",
        "sha256": "sha256:" + "c" * 64,
    },
]

#: The declaration as it stands AFTER the residue solve: smp joined core.
_CURRENT_DEPENDENCIES = (
    "python>=3.11,<3.13",
    "pytorch>=2.3",
    "timm",
    "transformers",
    "segmentation-models-pytorch",
)


def _lock(
    tmp_path: Path,
    *,
    packages: list[dict[str, str]],
    declared_dependencies: tuple[str, ...],
) -> LockArtifacts:
    """Return complete, hash-consistent lock artifacts for one synthetic solve."""

    export_bytes = json.dumps({"packages": packages}).encode("utf-8")
    return LockArtifacts(
        target="osx-arm64",
        lock_path=tmp_path / "osx-arm64.lock",
        export_path=tmp_path / "osx-arm64.resolved.json",
        export_hash_path=tmp_path / "osx-arm64.resolved.sha256",
        lock_bytes=b"@EXPLICIT\n",
        export_bytes=export_bytes,
        declared_export_hash=hash_bytes(export_bytes),
        declared_dependencies=declared_dependencies,
    )


# -- 1. the lock status itself ------------------------------------------------


def test_a_residue_lock_missing_a_new_declaration_is_superseded(tmp_path: Path) -> None:
    """The exact pilot state: spec declares smp, the last solve's export lacks it."""

    lock = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=_CURRENT_DEPENDENCIES,
    )
    # python is declared but genuinely absent from the miniature residue too;
    # restrict the declaration to the interesting difference for a sharp assert.
    lock_smp_only = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=("pytorch>=2.3", "segmentation-models-pytorch"),
    )
    assert lock.status == "superseded"
    assert lock_smp_only.status == "superseded"
    assert lock_smp_only.missing_declared_dependency() == "segmentation-models-pytorch"


def test_a_lock_satisfying_its_declaration_stays_locked(tmp_path: Path) -> None:
    """Version bounds and channel prefixes are stripped; names are what count."""

    lock = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=("conda-forge::pytorch>=2.3", "timm", "transformers"),
    )
    assert lock.status == "locked"
    assert lock.missing_declared_dependency() is None


def test_version_drift_alone_never_supersedes(tmp_path: Path) -> None:
    """A solver picking a different version is an outcome, not a stale spec."""

    lock = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=("transformers==4.57.1",),
    )
    assert lock.status == "locked"


def test_an_unreadable_export_is_not_read_as_superseded(tmp_path: Path) -> None:
    """Downstream export parsing owns that refusal; supersession needs proof."""

    export_bytes = b"\xff\xfenot json"
    lock = LockArtifacts(
        target="osx-arm64",
        lock_path=tmp_path / "osx-arm64.lock",
        export_path=tmp_path / "osx-arm64.resolved.json",
        export_hash_path=tmp_path / "osx-arm64.resolved.sha256",
        lock_bytes=b"@EXPLICIT\n",
        export_bytes=export_bytes,
        declared_export_hash=hash_bytes(export_bytes),
        declared_dependencies=_CURRENT_DEPENDENCIES,
    )
    assert lock.status == "locked"


# -- 2. admission reads unknown, not the residue ------------------------------


def _registry(lock: LockArtifacts) -> Any:
    """Duck-typed registry exposing exactly what admission reads."""

    return SimpleNamespace(intents={"core": SimpleNamespace(lock=lock)})


def test_routed_packages_from_a_superseded_lock_are_no_inventory(tmp_path: Path) -> None:
    """Admission must not treat last run's solve residue as this run's truth."""

    superseded = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=_CURRENT_DEPENDENCIES,
    )
    current = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=("pytorch>=2.3", "timm", "transformers"),
    )
    assert _routed_environment_packages(_registry(superseded), "core") == ()
    assert [row["name"] for row in _routed_environment_packages(_registry(current), "core")] == [
        "pytorch",
        "timm",
        "transformers",
    ]


def _smp_implementation() -> dict[str, Any]:
    """Return the archived m9304 pin in miniature."""

    return {
        "recipe_type": "declarative-library",
        "library_recipe": {
            "distribution": "segmentation_models_pytorch",
            "module": "segmentation_models_pytorch",
            "symbol": "DeepLabV3Plus",
            "version": "0.5.0",
            "kwargs": {},
            "pretrained_disable_fields": [],
            "pretrained_fields_absent": True,
        },
    }


def test_an_smp_model_passes_admission_once_the_residue_is_superseded(
    tmp_path: Path,
) -> None:
    """The full m9304/m9617 path: coverage admits, and the digest binds null.

    Against the residue inventory the digest binding refuses (the tripwire,
    unchanged); against the superseded lock's honest no-inventory the model is
    admitted with a null digest and reaches the environment lane, whose fresh
    solve of the current declaration installs smp.
    """

    implementation = _smp_implementation()
    with pytest.raises(RecipeError, match="does not install distribution"):
        bind_library_artifact_digest(dict(implementation), tuple(_RESIDUE_PACKAGES))

    superseded = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=_CURRENT_DEPENDENCIES,
    )
    routed = _routed_environment_packages(_registry(superseded), "core")
    assessment = assess_environment_coverage(
        implementation,
        routed_intent="core",
        routed_packages=routed,
        registry=_registry(superseded),
    )
    assert assessment.verdict is CoverageVerdict.NOT_ASSESSABLE
    assert not assessment.deferred
    binding_target = _smp_implementation()
    changed = bind_library_artifact_digest(binding_target, routed)
    assert changed is True
    assert binding_target["library_recipe"]["artifact_sha256"] is None


def test_covering_intents_never_cite_a_superseded_inventory(tmp_path: Path) -> None:
    """Stale locked-inventory evidence downgrades to the declared-dependency basis."""

    superseded = _lock(
        tmp_path,
        packages=_RESIDUE_PACKAGES,
        declared_dependencies=_CURRENT_DEPENDENCIES,
    )
    registry = SimpleNamespace(
        intents={
            "core": SimpleNamespace(
                lock=superseded, dependencies=list(_CURRENT_DEPENDENCIES)
            )
        }
    )
    covering = find_covering_intents("transformers", registry, exclude=None)
    assert [item.intent for item in covering] == ["core"]
    assert covering[0].basis is CoverageBasis.DECLARED_DEPENDENCY


# -- 3. Mixtral defers on size before normalization can refuse ----------------


class _StubArtifact:
    """Stand-in exposing only the proposal the capacity withhold reads."""

    def __init__(self, implementation: dict[str, Any]) -> None:
        self._implementation = implementation

    @property
    def proposal(self) -> dict[str, Any]:
        """Return a proposal carrying exactly the authored implementation facts."""

        return {"proposed_facts": {"implementation": self._implementation}}


class _StubLedger:
    """Operational ledger stub capturing appended events."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record one operational event."""

        self.records.append(event)
        return event


def _work_item(stable_id: str, name: str) -> WorkItem:
    """Return a scheduled work item for one synthetic roster row."""

    return WorkItem(
        intake=IntakeItem(
            stable_id=stable_id,
            name=name,
            zoo="huggingface_transformers",
            variant="",
            discovery_source="master_catalog",
            legacy_row_sha256=f"sha256:{'0' * 64}",
            preserved_legacy_flags=(),
        ),
        route=IntentRoute(stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _driver(tmp_path: Path) -> Any:
    """Duck-typed driver exposing exactly what the withhold touches."""

    records_root = tmp_path / "records"
    driver = SimpleNamespace(
        dependencies=SimpleNamespace(clock=lambda: "2026-08-04T12:00:00.000000Z"),
        paths=SimpleNamespace(
            work_root=tmp_path / "work",
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl"),
        ),
        config=SimpleNamespace(
            campaign_id="pilot",
            run_id="crawler-run",
            machine_id="mymini",
        ),
    )
    driver._record_host_capacity_deferral = MethodType(
        CrawlerDriver._record_host_capacity_deferral, driver
    )
    return driver


@pytest.fixture(autouse=True)
def _small_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the host to 16 GiB so the withhold decision is deterministic."""

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(16 * 2**30))


def test_mixtral_is_withheld_before_normalization(tmp_path: Path) -> None:
    """The archived m5915 recipe defers on size, and the refusal is durable.

    On the rung this model terminalized on a transformers version drift inside
    normalization; the size check must not depend on normalization surviving.
    """

    driver = _driver(tmp_path)
    item = _work_item("m5915", "mixtral")
    artifact: Any = _StubArtifact({"library_recipe": MIXTRAL_RECIPE})
    operational = _StubLedger()

    withheld = CrawlerDriver._withhold_for_host_capacity(driver, item, artifact, operational)

    assert withheld is True
    records_root = tmp_path / "records"
    rows = load_capacity_deferral_rows([capacity_deferral_path(records_root)])
    assert len(rows) == 1
    assert rows[0]["stable_id"] == "m5915"
    assert rows[0]["disposition"] == CAPACITY_DEFERRAL_DISPOSITION
    assert len(operational.records) == 1
    event = operational.records[0]
    assert event["event_kind"] == OperationalEventKind.CAMPAIGN_HEALTH.value
    assert event["details"]["disposition"] == "host-capacity-deferred"
    assert event["details"]["deferred"][0]["stable_id"] == "m5915"
    assert event["event_id"].startswith("host-capacity-")


def test_a_model_within_capacity_is_not_withheld(tmp_path: Path) -> None:
    """No ledger row and no event for a recipe declaring no oversized stack."""

    driver = _driver(tmp_path)
    item = _work_item("m9304", "smp_DeepLabV3Plus")
    artifact: Any = _StubArtifact(_smp_implementation())
    operational = _StubLedger()

    withheld = CrawlerDriver._withhold_for_host_capacity(driver, item, artifact, operational)

    assert withheld is False
    assert not capacity_deferral_path(tmp_path / "records").exists()
    assert operational.records == []
