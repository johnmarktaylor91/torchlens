"""Test-only end-to-end dry-run support for the crawler CLI acceptance gate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.authority import EnvironmentAuthorityCache
from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.driver import (
    AuthorArtifact,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverPaths,
    SupervisedForwardLane,
    WorkItem,
)
from menagerie.crawler.env_lifecycle import (
    LifecycleResult,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    parse_probe_receipt_bytes,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    LockArtifacts,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.tests.conftest import NOW, _committed_fixture_intent
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeNotifier,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
)


@dataclass(frozen=True)
class DryRunCase:
    """One tiny real model and its expected source-gated observations."""

    name: str
    modality: tuple[str, ...]
    shape: tuple[int, ...]
    divergence: str
    adapter_source: str
    fidelity_required: bool = False


_MLP_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyMLP()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 4, device=device),), {})
"""

_CONV_STATISTICAL_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStatisticalConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout2d(p=0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(torch.relu(self.norm(self.conv(value))))
        return value.mean(dim=(2, 3))


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStatisticalConvNet()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(2, 3, 8, 8, device=device),), {})
"""

_STRUCTURAL_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStructuralBranch(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return value.mean(dim=(2, 3))
        return value.flatten(1)


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStructuralBranch()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {})
"""

_CONV_STABLE_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStableConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value).mean(dim=(2, 3))


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStableConvNet()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {})
"""

DRY_RUN_CASES = (
    DryRunCase("DryRunMLP", ("unknown",), (1, 4), "none", _MLP_SOURCE),
    DryRunCase(
        "DryRunStatisticalConvNet",
        ("vision",),
        (2, 3, 8, 8),
        "statistical",
        _CONV_STATISTICAL_SOURCE,
    ),
    DryRunCase(
        "DryRunStructuralBranch",
        ("vision",),
        (1, 3, 8, 8),
        "structural",
        _STRUCTURAL_SOURCE,
        fidelity_required=True,
    ),
    DryRunCase("DryRunStableConvNet", ("vision",), (1, 3, 8, 8), "none", _CONV_STABLE_SOURCE),
)

DRY_RUN_ITEMS = (
    *((case.name, "base") for case in DRY_RUN_CASES),
    ("DryRunMLP", "calibration-1"),
    ("DryRunMLP", "calibration-2"),
    ("DryRunStatisticalConvNet", "calibration-1"),
    ("DryRunStatisticalConvNet", "calibration-2"),
    ("DryRunStableConvNet", "calibration-1"),
    ("DryRunStableConvNet", "calibration-2"),
)
DRY_RUN_PHASE = EnvironmentPhase.PYTORCH


class TinyModelAuthor(FakeAuthor):
    """Extend the canned author with runnable typed adapters for the tiny corpus."""

    def __init__(self) -> None:
        """Index the frozen tiny corpus by intake name."""

        super().__init__()
        self._cases = {case.name: case for case in DRY_RUN_CASES}

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return a canned accepted proposal bound to one real tiny adapter.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory driver work item, output root, configuration, and frozen authority context.

        Returns
        -------
        AuthorArtifact
            Authenticated author result containing the selected tiny adapter.
        """

        artifact = super().author(item, work_root, config, context)
        case = self._cases[item.intake.name]
        adapter_path = artifact.model_dir / "adapter.py"
        adapter_bytes = case.adapter_source.encode("utf-8")
        adapter_path.write_bytes(adapter_bytes)

        proposal = artifact.proposal
        proposal["proposal_id"] = f"proposal-{item.stable_id}"
        proposal["proposal_sha256"] = stable_hash(
            {"stable_id": item.stable_id, "adapter_sha256": hash_bytes(adapter_bytes)}
        )
        facts = proposal["proposed_facts"]
        facts["identity"]["canonical_name"] = case.name
        if case.name == "DryRunMLP":
            facts["identity"]["canonical_name"] = "MLP"
            facts["taxonomy"]["family"] = "MLP"
            facts["external_metadata"]["family"] = "MLP"
            facts["external_metadata"]["architecture_class"] = ["MLP"]
        facts["external_metadata"]["modality"] = list(case.modality)
        facts["external_metadata"]["modes"] = {
            "meaningful_modes": ["train", "eval"],
            "train_eval_divergence": case.divergence,
        }
        facts["modes"] = {
            "meaningful_modes": ["train", "eval"],
            "per_mode_run": {},
            "train_eval_divergence": case.divergence,
            "divergence_evidence": "expected tiny-model train/eval behavior",
        }
        implementation = facts["implementation"]
        implementation.update(
            {
                "recipe_type": "typed-adapter",
                "code_path": "adapter.py",
                "code_sha256": hash_bytes(adapter_bytes),
                "builder_symbol": "build_model",
                "dummy_call_symbol": "make_dummy_call",
                "library_recipe": None,
            }
        )
        code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
        implementation["code_manifest"] = code_manifest
        facts["evidence"]["excerpts"][0]["supports"] = sorted(
            set(facts["evidence"]["excerpts"][0]["supports"])
            | {
                "implementation.code_manifest[].path",
                "implementation.code_manifest[].sha256",
            }
        )
        contract = facts["input_contract"]
        contract["args"][0].update(
            {
                "semantic_role": "features" if case.name == "DryRunMLP" else "image",
                "shape": list(case.shape),
            }
        )
        proposal["verified_hashes"]["code"] = hash_bytes(adapter_bytes)
        proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
        fidelity_identity = stable_hash({"stable_id": item.stable_id, "kind": "dry-run-fidelity"})
        proposal["fidelity_identity"] = fidelity_identity
        resolution = facts["source_resolution"]
        resolution["rung"] = "R3_PORT"
        resolution["attempted_rungs"] = [
            {
                "rung": rung,
                "result": "selected" if rung == "R3_PORT" else "unavailable",
                "reason_code": "documented-search",
                "evidence_ids": ["evidence-1"],
            }
            for rung in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT")
        ]
        excerpt = facts["evidence"]["excerpts"][0]
        port_text = (
            f"{excerpt['text']} The staged source code is a faithful port of the documented "
            f"forward architecture and input contract for {case.name}, using "
            f"{' '.join(case.modality)} modality with {case.divergence} train eval divergence. "
            f"The grounded family is {facts['taxonomy']['family']}."
        )
        port_bytes = port_text.encode("utf-8")
        excerpt.update(
            {
                "locator": f"bytes:0-{len(port_bytes)}",
                "text": port_text,
                "text_sha256": hash_bytes(port_bytes),
            }
        )
        implementation["source_to_code_map"] = [
            {
                "material_item": "documented forward architecture",
                "source_id": "source-1",
                "source_locator": excerpt["locator"],
                "evidence_ids": ["evidence-1"],
                "code_path": "adapter.py",
                "code_locator": "complete typed adapter",
                "disposition": "faithful-port",
            }
        ]
        excerpt["supports"] = sorted(
            set(excerpt["supports"])
            | {
                "implementation.source_to_code_map[].code_locator",
                "implementation.source_to_code_map[].code_path",
                "implementation.source_to_code_map[].disposition",
                "implementation.source_to_code_map[].evidence_ids[]",
                "implementation.source_to_code_map[].material_item",
                "implementation.source_to_code_map[].source_id",
                "implementation.source_to_code_map[].source_locator",
            }
        )
        source_row = resolution["sources"][0]
        source_row.update(
            {
                "content_sha256": hash_bytes(port_bytes),
                "byte_count": len(port_bytes),
            }
        )
        manifest_row = artifact.source_manifest["sources"][0]
        Path(str(manifest_row["cas_path"])).write_bytes(port_bytes)
        manifest_row.update(
            {
                "content_sha256": hash_bytes(port_bytes),
                "byte_count": len(port_bytes),
            }
        )
        source_manifest_identity = stable_hash(artifact.source_manifest["sources"])
        artifact.source_manifest["manifest_sha256"] = source_manifest_identity
        proposal["source_manifest_identity"] = source_manifest_identity
        proposal["verified_hashes"]["source_manifest"] = source_manifest_identity
        facts["fidelity"].update(
            {
                "required": True,
                "reason": (
                    "dry-run typed port exercises structural fidelity"
                    if case.fidelity_required
                    else "dry-run typed port requires canonical fidelity"
                ),
                "verdict": None,
                "fidelity_identity": fidelity_identity,
                "gate_id": None,
                "current": False,
            }
        )
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class AllSourceFailureAuthor(TinyModelAuthor):
    """Inject a deterministic all-source-failure terminal partition for acceptance testing."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Fail source resolution before any proposal or model attempt can be accepted.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory author-lane inputs retained to implement the live protocol exactly.

        Raises
        ------
        RuntimeError
            Always, to create the explicit dry-run negative acceptance partition.
        """

        del item, work_root, config, context
        raise RuntimeError("injected dry-run all-source failure")


class MaterializedDryRunEnvironment(SequentialEnvironmentLifecycle):
    """Expose one already-provisioned prefix through the production lifecycle protocol."""

    def __init__(
        self,
        prefix: Path,
        intent: EnvironmentIntent,
        probe_results: tuple[ProbeResult, ...],
    ) -> None:
        """Store exact prefix and committed lifecycle artifacts.

        Parameters
        ----------
        prefix, intent, probe_results:
            Real materialized prefix plus its exact lock/export/probe contract.
        """

        self.prefix = prefix
        self.intent = intent
        self.probe_results = probe_results
        self._active = prefix
        self._authority_cache = EnvironmentAuthorityCache()

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: Any,
    ) -> LifecycleResult:
        """Give the driver the real prefix so it performs strict production binding.

        Parameters
        ----------
        intent:
            Routed intent, which must equal the artifact-backed fixture intent.
        use:
            Driver callback that strictly calls ``bind_materialized_environment`` before work.

        Returns
        -------
        LifecycleResult
            Durable lifecycle observation for the already-materialized fixture.
        """

        if intent != self.intent:
            raise AssertionError("dry-run environment intent differs from fixture artifacts")
        use(self.prefix, self.probe_results)
        return LifecycleResult(
            intent=intent.name,
            target=intent.lock.target,
            export_sha256=str(intent.lock.declared_export_hash),
            probe_results=self.probe_results,
            disk_before=0,
            disk_after_create=0,
            disk_after_teardown=0,
            disk_recovery_checked=False,
        )


def _materialized_environment(
    prefix: Path,
) -> tuple[MaterializedDryRunEnvironment, EnvironmentRegistry]:
    """Load the real fixture prefix's exact lock/export/probe artifacts.

    Parameters
    ----------
    prefix:
        Session hardlink-clone prefix selected explicitly by the CLI.

    Returns
    -------
    tuple[MaterializedDryRunEnvironment, EnvironmentRegistry]
        Production lifecycle lane and single-intent routing registry.
    """

    resolved_prefix = prefix.resolve(strict=True)
    if not (resolved_prefix / "conda-meta").is_dir():
        raise ValueError(
            f"dry-run environment is not a materialized conda prefix: {resolved_prefix}"
        )
    if os.environ.get("MENAGERIE_PLATFORM_LOCK"):
        intent, probe_results = _committed_fixture_intent(resolved_prefix)
        registry = EnvironmentRegistry(
            intents={intent.name: intent},
            phase_order=(DRY_RUN_PHASE,),
            small_set_target=True,
            hard_cap=None,
            global_split_guidance="single committed-lock real fixture environment",
        )
        return MaterializedDryRunEnvironment(resolved_prefix, intent, probe_results), registry
    artifact_root = resolved_prefix.parent / "artifacts"
    target = "round19-real-host"
    lock_path = artifact_root / f"{target}.lock"
    export_path = artifact_root / f"{target}.resolved.json"
    export_hash_path = artifact_root / f"{target}.resolved.sha256"
    probe_path = artifact_root / f"{target}.probes.json"
    required = (lock_path, export_path, export_hash_path, probe_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"dry-run environment fixture artifacts are missing: {missing}")
    probes = IntentProbes(("torch", "menagerie_round19_sentinel"), (), ())
    probe_results = parse_probe_receipt_bytes(probes, probe_path.read_bytes())
    lock = LockArtifacts(
        target=target,
        lock_path=lock_path,
        export_path=export_path,
        export_hash_path=export_hash_path,
        lock_bytes=lock_path.read_bytes(),
        export_bytes=export_path.read_bytes(),
        declared_export_hash=export_hash_path.read_text(encoding="utf-8").strip(),
    )
    intent = EnvironmentIntent(
        name="core",
        phase=DRY_RUN_PHASE,
        framework="pytorch",
        description="Round-19 real-prefix acceptance dry-run fixture",
        split_guidance="fixture-only",
        channels=("conda-forge",),
        dependencies=("python", "pytorch"),
        probes=probes,
        lock=lock,
        generation=None,
    )
    registry = EnvironmentRegistry(
        intents={intent.name: intent},
        phase_order=(DRY_RUN_PHASE,),
        small_set_target=True,
        hard_cap=None,
        global_split_guidance="single real fixture environment",
    )
    return MaterializedDryRunEnvironment(resolved_prefix, intent, probe_results), registry


class RecordingNotifier(FakeNotifier):
    """Persist fake notification calls so separate CLI processes remain observable."""

    def __init__(self, path: Path) -> None:
        """Bind the append-only fake notification log."""

        super().__init__()
        self.path = path

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Capture and append one ASCII notification summary."""

        delivered = super().notify(summary, idempotency_key=idempotency_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes({"summary": summary}) + b"\n")
        return delivered


def create_dry_run_snapshot(campaign_root: Path) -> IntakeSnapshot:
    """Create or reload the immutable four-architecture dry-run intake snapshot."""

    sources = campaign_root / "intake-sources"
    sources.mkdir(parents=True, exist_ok=True)
    master = sources / "master.jsonl"
    deferred = sources / "deferred.jsonl"
    rows = [
        {"name": name, "zoo": "dry-run-fixtures", "variant": variant}
        for name, variant in DRY_RUN_ITEMS
    ]
    master.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    deferred.write_bytes(b"")
    return create_intake_snapshot(master, deferred, campaign_root / "intake")


def dry_run_paths(campaign_root: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated runtime and canonical ledger paths for one dry-run campaign."""

    records = campaign_root / "records"
    return DriverPaths(
        campaign_root / "runtime",
        snapshot.root,
        LedgerPaths(
            records / "models" / "current-shard.jsonl",
            records / "attempts" / "local.jsonl",
            records / "gates" / "current-shard.jsonl",
        ),
    )


def build_dry_run_driver(
    repo_root: Path,
    campaign_root: Path,
    environment_prefix: Path,
    *,
    review_checkpoint_at: Optional[int],
    progress_milestones: tuple[int, ...],
    run_id: str,
) -> CrawlerDriver:
    """Build the real driver with only author/checker/notifier lanes deterministic.

    Parameters
    ----------
    repo_root, campaign_root, environment_prefix:
        Checked-out source, disposable campaign, and explicit real environment prefix.
    review_checkpoint_at, progress_milestones, run_id:
        Ordinary driver checkpoint, notification, and invocation configuration.

    Returns
    -------
    CrawlerDriver
        Driver using the shipped supervisor/compiler and strict materialized-environment binder.
    """

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    environments, registry = _materialized_environment(environment_prefix)
    author = (
        AllSourceFailureAuthor()
        if os.environ.get("MENAGERIE_DRY_RUN_INJECT_ALL_SOURCE_FAILURE") == "1"
        else TinyModelAuthor()
    )
    dependencies = DriverDependencies(
        author=author,
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=30, cwd=repo_root),
        environments=environments,
        notifier=RecordingNotifier(campaign_root / "notifications.jsonl"),
        clock=lambda: NOW,
    )
    config = DriverConfig(
        target="osx-arm64",
        run_id=run_id,
        machine_id=f"dry-run-{environment_prefix.name}",
        review_checkpoint_at=review_checkpoint_at,
        progress_milestones=progress_milestones,
        notify_command=None,
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
    )
    return CrawlerDriver(paths, config, dependencies, registry=registry)


def read_notification_summaries(path: Path) -> tuple[str, ...]:
    """Read persisted fake notification summaries in append order."""

    if not path.is_file():
        return ()
    return tuple(
        str(json.loads(line)["summary"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
