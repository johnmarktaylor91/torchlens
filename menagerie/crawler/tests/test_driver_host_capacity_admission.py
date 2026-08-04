"""Driver-side pre-admission capacity gate tests.

Exercises ``CrawlerDriver._admit_within_host_capacity`` directly against the
real Mixtral recipe: an oversized model must be withheld from the environment
lane, recorded in the deferral ledger, and announced on the operational ledger,
while every other model passes through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.constants import (
    EnvironmentPhase,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    HOST_MEMORY_ENV_VAR,
    capacity_deferral_path,
    load_capacity_deferral_rows,
)
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.tests.test_host_capacity import (
    MIXTRAL_RECIPE,
    NAMED_SYMBOL_RECIPE,
)


@dataclass
class _StubArtifact:
    """Minimal stand-in exposing only the proposal the capacity gate reads."""

    recipe: Any

    @property
    def proposal(self) -> dict[str, Any]:
        """Return a proposal carrying exactly the authored implementation facts."""

        return {"proposed_facts": {"implementation": self.recipe}}


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
            zoo="discovered-pytorch",
            variant="",
            discovery_source="master_catalog",
            legacy_row_sha256=f"sha256:{'0' * 64}",
            preserved_legacy_flags=(),
        ),
        route=IntentRoute(
            stable_id=stable_id,
            intent="torch-core",
            phase=EnvironmentPhase.PYTORCH,
        ),
    )


def _driver(records_root: Path) -> Any:
    """Return a duck-typed driver exposing exactly what the gate touches."""

    return SimpleNamespace(
        dependencies=SimpleNamespace(clock=lambda: "2026-08-04T12:00:00.000000Z"),
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        ),
        config=SimpleNamespace(
            campaign_id="c1-mech",
            run_id="crawler-run",
            machine_id="mymini",
        ),
    )


@pytest.fixture(autouse=True)
def _small_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the host to 16 GiB so the gate's decision is deterministic."""

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(16 * 2**30))


def test_oversized_model_is_withheld_and_recorded(tmp_path: Path) -> None:
    """Mixtral never reaches the environment lane, and the refusal is durable."""

    records_root = tmp_path / "records"
    driver = _driver(records_root)
    mixtral = _work_item("m5915", "Mixtral 8x7B")
    dla = _work_item("m4334", "DLA-60")
    artifacts: Any = {
        "m5915": _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        "m4334": _StubArtifact({"library_recipe": NAMED_SYMBOL_RECIPE}),
    }
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(
        driver, (mixtral, dla), artifacts, operational
    )

    assert [item.stable_id for item in admitted] == ["m4334"]
    rows = load_capacity_deferral_rows([capacity_deferral_path(records_root)])
    assert len(rows) == 1
    assert rows[0]["stable_id"] == "m5915"
    assert rows[0]["disposition"] == CAPACITY_DEFERRAL_DISPOSITION
    assert rows[0]["capacity"]["estimate"]["parameter_count_lower_bound"] > 8.59e9
    assert rows[0]["capacity"]["threshold"]["host_physical_memory_bytes"] == 16 * 2**30

    assert len(operational.records) == 1
    event = operational.records[0]
    assert event["event_kind"] == OperationalEventKind.CAMPAIGN_HEALTH.value
    assert event["status"] == OperationalEventStatus.HEALTHY.value
    assert event["details"]["disposition"] == "host-capacity-deferred"
    assert event["details"]["deferred"][0]["stable_id"] == "m5915"
    assert event["event_id"].startswith("host-capacity-")


def test_a_wave_with_nothing_oversized_is_untouched(tmp_path: Path) -> None:
    """No ledger and no event when every model fits."""

    records_root = tmp_path / "records"
    driver = _driver(records_root)
    dla = _work_item("m4334", "DLA-60")
    artifacts: Any = {"m4334": _StubArtifact({"library_recipe": NAMED_SYMBOL_RECIPE})}
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(
        driver, (dla,), artifacts, operational
    )

    assert admitted == (dla,)
    assert not capacity_deferral_path(records_root).exists()
    assert operational.records == []


def test_a_bigger_host_admits_the_same_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog becomes runnable on larger hardware with no code change."""

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(512 * 2**30))
    records_root = tmp_path / "records"
    driver = _driver(records_root)
    mixtral = _work_item("m5915", "Mixtral 8x7B")
    artifacts: Any = {"m5915": _StubArtifact({"library_recipe": MIXTRAL_RECIPE})}
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(
        driver, (mixtral,), artifacts, operational
    )

    assert admitted == (mixtral,)
    assert not capacity_deferral_path(records_root).exists()
    assert operational.records == []


def test_an_empty_wave_never_reads_the_host(tmp_path: Path) -> None:
    """The gate is free when there is no work to admit."""

    driver = _driver(tmp_path / "records")
    operational: Any = _StubLedger()
    assert (
        CrawlerDriver._admit_within_host_capacity(driver, (), {}, operational) == ()
    )
    assert operational.records == []
