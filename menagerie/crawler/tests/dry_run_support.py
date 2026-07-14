"""Test-only end-to-end dry-run support for the crawler CLI acceptance gate."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from menagerie.crawler.driver import (
    AuthorArtifact,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverPaths,
    SupervisedForwardLane,
    WorkItem,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_recipe_revision,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.tests.conftest import NOW
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeEnvironments,
    FakeNotifier,
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
    return TinyMLP()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 4, device=device),), {})
"""

_CONV_STATISTICAL_SOURCE = """from __future__ import annotations

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
    return TinyStatisticalConvNet()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(2, 3, 8, 8, device=device),), {})
"""

_STRUCTURAL_SOURCE = """from __future__ import annotations

import torch


class TinyStructuralBranch(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return value.mean(dim=(2, 3))
        return value.flatten(1)


def build_model() -> object:
    return TinyStructuralBranch()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {})
"""

_CONV_STABLE_SOURCE = """from __future__ import annotations

import torch


class TinyStableConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value).mean(dim=(2, 3))


def build_model() -> object:
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


class TinyModelAuthor(FakeAuthor):
    """Extend the canned author with runnable typed adapters for the tiny corpus."""

    def __init__(self) -> None:
        """Index the frozen tiny corpus by intake name."""

        super().__init__()
        self._cases = {case.name: case for case in DRY_RUN_CASES}

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a canned accepted proposal bound to one real tiny adapter."""

        artifact = super().author(item, work_root, config)
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
        recipe_revision = compute_recipe_revision(
            {"recipe_type": "typed-adapter", "path": adapter_path.name},
            str(proposal["source_identity"]),
            adapter_bytes=adapter_bytes,
        )
        proposal["recipe_revision"] = recipe_revision
        implementation["recipe_revision"] = recipe_revision
        contract = facts["input_contract"]
        contract["code_path"] = "adapter.py"
        contract["args"][0].update(
            {
                "semantic_role": "features" if case.name == "DryRunMLP" else "image",
                "shape": list(case.shape),
            }
        )
        proposal["verified_hashes"]["code"] = hash_bytes(adapter_bytes)
        if case.fidelity_required:
            fidelity_identity = stable_hash(
                {"stable_id": item.stable_id, "kind": "dry-run-fidelity"}
            )
            proposal["fidelity_identity"] = fidelity_identity
            facts["source_resolution"]["rung"] = "R3_PORT"
            facts["fidelity"].update(
                {
                    "required": True,
                    "reason": "dry-run typed port exercises the fidelity lane",
                    "verdict": None,
                    "fidelity_identity": fidelity_identity,
                    "gate_id": None,
                    "current": False,
                }
            )
        return AuthorArtifact(proposal, artifact.source_manifest, artifact.model_dir)


class RecordingNotifier(FakeNotifier):
    """Persist fake notification calls so separate CLI processes remain observable."""

    def __init__(self, path: Path) -> None:
        """Bind the append-only fake notification log."""

        super().__init__()
        self.path = path

    def notify(self, summary: str) -> bool:
        """Capture and append one ASCII notification summary."""

        delivered = super().notify(summary)
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
    *,
    review_checkpoint_at: Optional[int],
    progress_milestones: tuple[int, ...],
    run_id: str,
) -> CrawlerDriver:
    """Build the real driver with only out-of-scope external lanes replaced by fakes."""

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    dependencies = DriverDependencies(
        author=TinyModelAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=30, cwd=repo_root),
        environments=FakeEnvironments(campaign_root / "current-interpreter-envs"),
        notifier=RecordingNotifier(campaign_root / "notifications.jsonl"),
        clock=lambda: NOW,
    )
    config = DriverConfig(
        target="osx-arm64",
        run_id=run_id,
        machine_id=f"dry-run-{Path(sys.executable).name}",
        review_checkpoint_at=review_checkpoint_at,
        progress_milestones=progress_milestones,
        notify_command=None,
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
    )
    return CrawlerDriver(paths, config, dependencies)


def read_notification_summaries(path: Path) -> tuple[str, ...]:
    """Read persisted fake notification summaries in append order."""

    if not path.is_file():
        return ()
    return tuple(
        str(json.loads(line)["summary"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
