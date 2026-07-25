"""Round-21 clean-clone handoff artifact-authority identity matrix."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.artifact_transactions import ArtifactEventKind
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver import (
    AuthorArtifact,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverIntegrationError,
)
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.test_slice_f_driver import (
    BothDeferredRealAuthor,
    DisabledAuthor,
    FakeAuthor,
    FakeChecker,
    FakeEnvironments,
    FakeForward,
    FakeNotifier,
    TypedAdapterPatch,
    _copy_clean_clone_handoff_authority,
    _driver,
    _paths,
    _snapshot,
    _terminal_fake_author_result,
    apply_typed_adapter_patch,
    finalize_typed_adapter_patch,
    test_linux_handoff_attempts_both_deferred_statuses_and_supersedes,
)

_CHANGED_HANDOFF_ADAPTER = """from __future__ import annotations

import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 2


def build_model() -> object:
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""


class _ChangedDeferredRealAuthor(FakeAuthor):
    """Create a second valid handoff proposal with changed adapter bytes."""

    def author(
        self,
        item: driver_module.WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: driver_module.AuthorityContext,
    ) -> AuthorArtifact:
        """Return changed same-work authority through the ordinary author contract."""

        artifact = super().author(item, work_root, config, context)
        apply_typed_adapter_patch(artifact, TypedAdapterPatch(source=_CHANGED_HANDOFF_ADAPTER))
        rebound = finalize_typed_adapter_patch(artifact, config)
        return _terminal_fake_author_result(rebound, platform="cuda")


def _make_clean_clone(
    tmp_path: Path,
) -> tuple[Any, driver_module.DriverPaths]:
    """Create one canonical-only clone containing a finalized real handoff proposal."""

    source_root = tmp_path / "source-campaign"
    source_root.mkdir()
    snapshot = _snapshot(source_root, count=1)
    result = _driver(source_root, snapshot, author=BothDeferredRealAuthor()).run()
    assert result.status == "complete"
    source_paths = _paths(source_root, snapshot)
    clone_snapshot, clone_paths = _copy_clean_clone_handoff_authority(
        source_paths,
        snapshot,
        tmp_path / "clean-clone",
    )
    shutil.rmtree(source_root)
    assert not source_root.exists()
    return clone_snapshot, clone_paths


def _handoff_driver(
    paths: driver_module.DriverPaths,
) -> tuple[CrawlerDriver, DisabledAuthor, FakeChecker, FakeForward, FakeEnvironments]:
    """Build a selected-handoff driver whose active lanes expose any precheck escape."""

    author = DisabledAuthor()
    checker = FakeChecker()
    forward = FakeForward()
    environments = FakeEnvironments(paths.runtime_root / "forbidden-environments")
    dependencies = DriverDependencies(
        author,
        checker,
        forward,
        environments,
        FakeNotifier(),
        lambda: driver_module.utc_now(),
    )
    driver = CrawlerDriver(
        paths,
        DriverConfig(
            target="linux-x86_64-cuda",
            only_status="deferred:*",
            run_id="round21-handoff-identity",
            machine_id="linux-machine",
            review_checkpoint_at=None,
            progress_milestones=(),
        ),
        dependencies,
        registry=driver_module.load_environment_registry(target="linux-x86_64-cuda"),
    )
    return driver, author, checker, forward, environments


def _assert_no_lane_or_canonical_write(
    paths: driver_module.DriverPaths,
    before: dict[Path, bytes],
    author: DisabledAuthor,
    checker: FakeChecker,
    forward: FakeForward,
    environments: FakeEnvironments,
) -> None:
    """Assert one handoff refusal preceded every active lane and canonical append."""

    assert {path: path.read_bytes() if path.exists() else b"" for path in before} == before
    assert author.calls == 0
    assert checker.metadata_calls == 0
    assert checker.fidelity_calls == 0
    assert forward.calls == {}
    assert environments.events == []
    assert not environments.root.exists()
    assert not paths.worker_lease.exists()


def _proof_h01(tmp_path: Path, fixture: RealEnvironmentFixture) -> None:
    """Retain valid two-status clean-clone authority and third-run byte no-op."""

    test_linux_handoff_attempts_both_deferred_statuses_and_supersedes(tmp_path, fixture)


def _proof_h02(tmp_path: Path, fixture: RealEnvironmentFixture) -> None:
    """Reject old authority before it can label changed proposal and adapter bytes."""

    del fixture
    clone_root = tmp_path / "clean-clone"
    clone_root.mkdir()
    snapshot = _snapshot(clone_root, count=1)
    first = _driver(clone_root, snapshot, author=BothDeferredRealAuthor()).run()
    assert first.status == "complete"
    paths = _paths(clone_root, snapshot)
    driver, _author, _checker, _forward, _environments = _handoff_driver(paths)
    context = driver_module.build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=driver.config.author_model,
        author_version=driver.config.author_version,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    driver._intake_snapshot = snapshot
    driver._authority_context = context
    guarded = (
        paths.ledgers.models,
        paths.ledgers.attempts,
        paths.ledgers.gates,
        paths.ledgers.artifacts,
    )
    with (
        JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational,
        CanonicalReducer(paths.ledgers, context) as reducer,
    ):
        work = driver._ordered_work(snapshot, reducer.current_records, {})
        assert len(work) == 1
        item = work[0]
        changed = _ChangedDeferredRealAuthor().author(
            item,
            paths.work_root,
            driver.config,
            context,
        )
        changed = driver._stage_author_result(item, changed, reducer)
        prior = deepcopy(reducer.current_records[item.stable_id]["artifact_authority"])
        assert str(changed.staged.transaction_id) != prior["transaction_id"]
        before = {path: path.read_bytes() if path.exists() else b"" for path in guarded}
        with pytest.raises(
            DriverIntegrationError,
            match="^retained-artifact-authority-mismatch$",
        ):
            driver._terminalize(
                item,
                changed,
                "deferred:needs-cuda",
                None,
                "changed handoff proposal",
                (),
                reducer,
                operational,
                {"status": "running"},
            )
    assert {path: path.read_bytes() if path.exists() else b"" for path in guarded} == before


def _proof_h03(tmp_path: Path, fixture: RealEnvironmentFixture) -> None:
    """Reject a selected deferral whose named transaction has zero finals."""

    del fixture
    _snapshot_value, paths = _make_clean_clone(tmp_path)
    rows = scan_jsonl(paths.ledgers.artifacts)
    retained = [
        row
        for row in rows
        if row["event_kind"]
        not in {ArtifactEventKind.PUBLISHED.value, ArtifactEventKind.PRIVATE_COMMITTED.value}
    ]
    paths.ledgers.artifacts.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in retained)
    )
    driver, author, checker, forward, environments = _handoff_driver(paths)
    guarded = (
        paths.ledgers.models,
        paths.ledgers.attempts,
        paths.ledgers.gates,
        paths.ledgers.artifacts,
    )
    before = {path: path.read_bytes() if path.exists() else b"" for path in guarded}
    with pytest.raises(DriverIntegrationError, match="^handoff-authority-unavailable$"):
        driver.run()
    _assert_no_lane_or_canonical_write(
        paths,
        before,
        author,
        checker,
        forward,
        environments,
    )


def _proof_h04(tmp_path: Path, fixture: RealEnvironmentFixture) -> None:
    """Reject a selected deferral whose named transaction has ambiguous finals."""

    del fixture
    _snapshot_value, paths = _make_clean_clone(tmp_path)
    rows = scan_jsonl(paths.ledgers.artifacts)
    final = next(
        row
        for row in rows
        if row["event_kind"]
        in {ArtifactEventKind.PUBLISHED.value, ArtifactEventKind.PRIVATE_COMMITTED.value}
    )
    paths.ledgers.artifacts.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in (*rows, final))
    )
    driver, author, checker, forward, environments = _handoff_driver(paths)
    guarded = (
        paths.ledgers.models,
        paths.ledgers.attempts,
        paths.ledgers.gates,
        paths.ledgers.artifacts,
    )
    before = {path: path.read_bytes() if path.exists() else b"" for path in guarded}
    with pytest.raises(DriverIntegrationError, match="^handoff-authority-unavailable$"):
        driver.run()
    _assert_no_lane_or_canonical_write(
        paths,
        before,
        author,
        checker,
        forward,
        environments,
    )


_ROUND21_HANDOFF_AUTHORITY_MATRIX: tuple[
    tuple[str, Callable[[Path, RealEnvironmentFixture], None]], ...
] = (
    ("H01", _proof_h01),
    ("H02", _proof_h02),
    ("H03", _proof_h03),
    ("H04", _proof_h04),
)


@pytest.mark.parametrize(
    ("cell_id", "proof"),
    _ROUND21_HANDOFF_AUTHORITY_MATRIX,
    ids=[cell_id for cell_id, _proof in _ROUND21_HANDOFF_AUTHORITY_MATRIX],
)
def test_round21_handoff_authority_identity_matrix(
    cell_id: str,
    proof: Callable[[Path, RealEnvironmentFixture], None],
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Execute every exact H01--H04 clean-clone handoff identity control."""

    expected_ids = tuple(f"H{index:02d}" for index in range(1, 5))
    registered_ids = tuple(value for value, _registered_proof in _ROUND21_HANDOFF_AUTHORITY_MATRIX)
    assert registered_ids == expected_ids
    assert cell_id in registered_ids
    assert set(structural.ROUND21_VS6_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
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
    proof(tmp_path, real_environment_fixture)
