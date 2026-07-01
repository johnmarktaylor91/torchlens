"""SLURM cluster runner helpers for RAM-heavy menagerie validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from menagerie.catalog import CATALOG_DB, CatalogRow
from menagerie.ledger import (
    ENV_VERIFICATION_DB,
    LEGACY_UNKNOWN,
    VerificationRun,
    VerificationTarget,
    _resolve_verification_db,
    append_verification_run,
    connect as connect_ledger,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_ARTIFACT_ROOT = Path("/tmp/torchlens_menagerie_cluster")
CATALOG_COLUMNS = (
    "model_id",
    "display_index",
    "stable_id",
    "name",
    "variant",
    "family",
    "family_normalized",
    "domain",
    "zoo",
    "constructor_call",
    "input_shape",
    "input_dtype",
    "era",
    "verified",
    "notes",
    "source",
    "recipe_revision_sha256",
    "input_is_real",
    "verification_expectation",
    "quarantine",
)
TERMINAL_STATUSES = frozenset(
    {
        "passed",
        "failed",
        "skipped",
        "timeout",
        "not_applicable",
        "deferred",
        "error",
        "install_failed",
        "env_unavailable",
        "oom",
        "native_crash",
        "killed",
    }
)
SLURM_TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }
)
# Total physical RAM on the local workstation, in GiB. Retained for the legacy
# kwarg name and tier-sizing call sites.
LOCAL_RAM_THRESHOLD_GB = 125.0
# LOCAL-FIRST cluster-routing threshold (GiB). A model routes to the SHARED axon
# cluster ONLY with HARD measured evidence it cannot fit locally: a measured peak
# RSS at/above this usable-RAM threshold, or a prior LOCAL run that exhausted
# (essentially) this much RAM and was still killed. The 10 GiB gap below the
# 125 GiB total leaves OS/headroom. axon is a SHARED cluster for the whole lab;
# never preemptively send an unmeasured / merely large-LOOKING model there.
LOCAL_FIRST_CLUSTER_THRESHOLD_GB = 115.0
# A LOCAL worker-memory-cap kill only counts as genuine "can't fit locally"
# evidence when the cap it blew through was near full local RAM. A model killed
# at a SMALL cap (an early-sweep protective cap, e.g. 30 GiB) proves nothing
# about 115 GiB feasibility, so such kills must NOT escalate to the cluster.
LOCAL_RAM_FAILURE_CAP_FLOOR_GB = 105.0
# Local statuses that constitute a RAM-related failure (escalate to cluster). A
# bare ``native_crash`` is deliberately EXCLUDED: a native crash is ambiguous
# (often a code/wrapper bug, not RAM exhaustion), so shipping it to the shared
# cluster would waste a node on a non-RAM failure. Only unambiguous resource kills
# (the OOM killer / an explicit kill) count; a genuine RAM native-crash surfaces
# as ``oom``/``killed`` or as a near-full-RAM ``failed:memory_cap`` instead.
LOCAL_RAM_FAILURE_STATUSES = frozenset({"oom", "killed"})
MB_PER_GB = 1024
GIANT_HEURISTIC_PATTERNS = (
    "depth_pro",
    "efficientdet_d5",
    "efficientdet_d6",
    "efficientdet_d7",
    "eva_giant",
    "eva02_enormous",
    "vit_so400m",
    "beit_large_patch16_512",
    "mixture-of-experts",
    "mixture of experts",
    "moe",
    "longcat",
    "deepseek_vl",
    "outetts",
    "ettin",
)


@dataclass(frozen=True)
class GiantRegistryEntry:
    """Static first-contact routing record for a RAM-heavy model.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    name:
        Human-readable model name.
    measured_peak_rss_mb:
        Campaign-measured peak RSS in MB when known.
    node_mem_gb:
        Initial SLURM memory tier to request.
    reason:
        Evidence or conservative rationale for cluster routing.
    worker_memory_cap_gb:
        Per-model validator RSS cap. Defaults to ``node_mem_gb - 10``.
    partition:
        Optional preferred SLURM partition for this size tier.
    force_cluster:
        Whether the static seed routes to cluster even before ledger evidence.
    """

    stable_id: str
    name: str
    measured_peak_rss_mb: int | None
    node_mem_gb: int
    reason: str
    worker_memory_cap_gb: int | None = None
    partition: str | None = None
    force_cluster: bool = True


@dataclass(frozen=True)
class NodeTier:
    """SLURM memory tier used for right-sized dispatch.

    Parameters
    ----------
    mem_gb:
        Requested SLURM memory in GiB.
    worker_memory_cap_gb:
        Validator worker RSS cap in GiB.
    partition:
        SLURM partition for this tier.
    max_peak_rss_gb:
        Largest measured peak normally assigned to this tier.
    gpu:
        Whether this tier requests one GPU via SLURM ``--gres``.
    """

    mem_gb: int
    worker_memory_cap_gb: int
    partition: str
    max_peak_rss_gb: int
    gpu: bool = False


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster connection and SLURM defaults.

    Parameters
    ----------
    host:
        SSH host or alias.
    account:
        SLURM account.
    partition:
        Default SLURM partition.
    remote_repo:
        Repository path on the cluster.
    remote_artifact_root:
        Artifact root on the cluster.
    remote_home:
        Absolute remote home directory used to expand ``~`` in SLURM ``#SBATCH``
        directives, which SLURM does not expand itself. Resolved once per
        dispatch via ``ssh <host> 'echo $HOME'`` when left ``None``.
    remote_pixi_bin:
        Remote path where the pixi binary is staged. The cluster nodes do not
        ship pixi on ``PATH``, so the local pixi binary is rsync'd here once per
        dispatch and the sbatch script invokes it by absolute path.
    pixi_env:
        Committed pixi lock prefix under ``menagerie/locks``.
    cpus_per_task:
        CPUs requested for each array task.
    time_limit:
        SLURM time limit.
    array_concurrency:
        Maximum concurrent array tasks.
    node_tiers:
        Ordered memory tiers. The default uses 180 -> 250 -> 500 -> 1000 GiB.
    gpu_node_tier:
        Placeholder GPU tier for future CUDA-required cluster rows. Disabled in
        practice while ``REQUIRES_CUDA`` is empty pending confirmed stable IDs.
    sbatch_wait_timeout_sec:
        Maximum wall-clock seconds to wait for a blocking ``sbatch --wait`` command.
    """

    host: str = "axon"
    account: str = "nklab"
    partition: str = "nklab"
    remote_repo: str = "~/projects/torchlens"
    remote_artifact_root: str = "~/projects/torchlens/.cluster_runner"
    remote_home: str | None = None
    remote_pixi_bin: str = "~/.cache/torchlens/bin/pixi"
    pixi_env: str = "cluster_giants"
    cpus_per_task: int = 8
    time_limit: str = "12:00:00"
    array_concurrency: int = 4
    node_tiers: tuple[NodeTier, ...] = (
        NodeTier(180, 170, "nklab", 130),
        NodeTier(250, 230, "u19moc3", 210),
        NodeTier(500, 480, "naplab", 420),
        NodeTier(1000, 960, "naplab", 900),
    )
    # TODO(Q3): confirm a40/l40 partition+account+gres syntax before enabling
    # cluster-gpu dispatch with non-empty REQUIRES_CUDA rows.
    gpu_node_tier: NodeTier = NodeTier(180, 170, "TODO_Q3_GPU_PARTITION", 130, gpu=True)
    sbatch_wait_timeout_sec: float = 43_200.0


@dataclass(frozen=True)
class ClusterAssignment:
    """One stable ID assigned to one SLURM array task.

    Parameters
    ----------
    campaign_id:
        Campaign key used for fail-loud merge idempotency.
    attempt_id:
        Attempt key used for fail-loud merge idempotency.
    assignment_id:
        Stable assignment key unique within a campaign attempt.
    stable_id:
        Durable model identity.
    array_index:
        SLURM array task index.
    node_mem_gb:
        Requested SLURM memory in GiB.
    worker_memory_cap_gb:
        Validator worker RSS cap in GiB.
    partition:
        SLURM partition.
    reason:
        Routing evidence.
    expected_row_count:
        Expected result rows for this task.
    timeout_sec:
        Validator timeout for this assignment.
    input_scale:
        Validator input scale for this assignment.
    gpu:
        Whether this assignment uses a GPU SLURM tier.
    """

    campaign_id: str
    attempt_id: str
    assignment_id: str
    stable_id: str
    array_index: int
    node_mem_gb: int
    worker_memory_cap_gb: int
    partition: str
    reason: str
    expected_row_count: int = 1
    timeout_sec: float = 14400.0
    input_scale: float = 1.0
    gpu: bool = False


@dataclass(frozen=True)
class DispatchResult:
    """Result of preparing and optionally submitting a cluster campaign.

    Parameters
    ----------
    campaign_id:
        Campaign key.
    attempt_id:
        Attempt key.
    assignments:
        Submitted task assignments.
    local_artifact_dir:
        Local dispatch artifact directory.
    remote_artifact_dir:
        Remote dispatch artifact directory.
    sbatch_job_ids:
        Parsed SLURM job IDs. Empty for dry-run/no-parse submissions.
    commands:
        Commands executed or prepared.
    dry_run:
        Whether commands were prepared without execution.
    """

    campaign_id: str
    attempt_id: str
    assignments: tuple[ClusterAssignment, ...]
    local_artifact_dir: Path
    remote_artifact_dir: str
    sbatch_job_ids: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    dry_run: bool = False


@dataclass(frozen=True)
class MergeReport:
    """Summary of an idempotent cluster result merge.

    Parameters
    ----------
    campaign_id:
        Campaign key.
    attempt_id:
        Attempt key.
    inserted:
        Number of new verification rows inserted.
    duplicates:
        Number of already-imported identical result rows skipped.
    assignments:
        Number of assignments verified against expected counts and checksums.
    run_id_collisions:
        Number of detected run-ID collisions. A non-zero value is only used for
        diagnostics before failing loud.
    """

    campaign_id: str
    attempt_id: str
    inserted: int
    duplicates: int
    assignments: int
    run_id_collisions: int = 0


@dataclass(frozen=True)
class ClusterResultRow:
    """One exported cluster verification row plus merge keys.

    Parameters
    ----------
    campaign_id:
        Campaign key.
    attempt_id:
        Attempt key.
    assignment_id:
        Assignment key.
    run:
        Verification ledger row produced by the worker.
    """

    campaign_id: str
    attempt_id: str
    assignment_id: str
    run: VerificationRun


@dataclass(frozen=True)
class ResourceRoute:
    """Operational routing decision for one validation row.

    Parameters
    ----------
    lane:
        Route lane: ``local-cpu``, ``local-gpu``, ``cluster-gpu``, or
        ``cluster-ram``.
    device:
        Explicit worker device, either ``cpu`` or ``cuda``.
    cluster:
        Whether the row should be dispatched to the cluster.
    reason:
        Human-readable routing reason.
    """

    lane: str
    device: str
    cluster: bool
    reason: str


# Catalog-confirmed hard-CUDA stable IDs. These rows have no honest CPU validation
# path (Triton/CUDA kernels or package-level CUDA assertions), so local-first means
# local GPU first when the row fits the workstation GPU, not CPU.
REQUIRES_CUDA: dict[str, str] = {
    "m4921": "FLA gated_deltanet requires CUDA/Triton kernels",
    "m4922": "FLA gated_deltanet2 requires CUDA/Triton kernels",
    "m4928": "FLA gated_deltaproduct requires CUDA/Triton kernels",
    "m4932": "FLA GLA requires CUDA/Triton kernels",
    "m5624": "lightweight-GAN discriminator hard-requires CUDA",
    "m5625": "lightweight-GAN generator hard-requires CUDA",
    "m5626": "lightweight-GAN simple decoder hard-requires CUDA",
    "m11955": "lightweight-GAN discriminator hard-requires CUDA",
    "m11956": "lightweight-GAN generator hard-requires CUDA",
}


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class ClusterMergeConflict(RuntimeError):
    """Raised when a cluster merge key maps to non-identical payloads."""


class ClusterResultIntegrityError(RuntimeError):
    """Raised when cluster result counts or checksums do not match the manifest."""


class ClusterJobFailed(RuntimeError):
    """Raised when a SLURM job was submitted and ran but exited non-zero.

    This is distinct from a transport/submit failure: the job reached the
    cluster, was accepted by sbatch (a job ID was returned), and then failed
    while running. Callers MUST surface this as an honest job/validation failure
    rather than masking it behind a benign ``cluster_unavailable`` skip.

    Parameters
    ----------
    job_ids:
        Parsed SLURM job IDs for the failed submissions.
    detail:
        Human-readable failure detail (return code, stderr/stdout tail).
    dispatch:
        Best-effort dispatch context so callers can still collect any honest
        result rows the worker wrote before failing.
    """

    def __init__(
        self,
        job_ids: Sequence[str],
        detail: str,
        dispatch: DispatchResult | None = None,
    ) -> None:
        self.job_ids = tuple(job_ids)
        self.detail = detail
        self.dispatch = dispatch
        super().__init__(f"cluster job(s) {','.join(self.job_ids) or '<unknown>'} failed: {detail}")


# Static first-contact routing seeds. Under the LOCAL-FIRST policy only entries
# with ``force_cluster=True`` preemptively route to the SHARED axon cluster before
# any measurement -- and that flag is reserved for the few GENUINE can't-fit
# giants whose MEASURED peak RSS is at/above local RAM (>=115 GiB). Every entry
# whose measured peak fits locally (<115 GiB) keeps ``force_cluster=False`` so it
# validates LOCALLY first; it is retained only for its tier-sizing metadata and
# escalates to the cluster on its own merits (a measured >=115 GiB peak or a local
# RAM failure) via the ledger, never preemptively. Measured peaks below reflect
# the campaign max on axon (the four forced giants exceeded local RAM there).
GIANT_REGISTRY: dict[str, GiantRegistryEntry] = {
    "m920": GiantRegistryEntry(
        "m920",
        "Ettin-decoder-1b",
        103 * MB_PER_GB,
        180,
        "axon peak ~103 GiB; fits locally",
        force_cluster=False,
    ),
    "m2064": GiantRegistryEntry(
        "m2064",
        "OuteTTS",
        106 * MB_PER_GB,
        180,
        "axon peak ~106 GiB; fits locally",
        force_cluster=False,
    ),
    "m3635": GiantRegistryEntry(
        "m3635",
        "beit_large_patch16_512",
        48 * MB_PER_GB,
        180,
        "axon peak about 48 GiB",
        force_cluster=False,
    ),
    "m4165": GiantRegistryEntry(
        "m4165",
        "deepseek_vl_hybrid",
        104 * MB_PER_GB,
        180,
        "axon peak ~104 GiB; fits locally",
        force_cluster=False,
    ),
    "m4246": GiantRegistryEntry(
        "m4246", "depth_pro", 183 * MB_PER_GB, 250, "axon peak ~183 GiB; exceeds local RAM"
    ),
    "m4494": GiantRegistryEntry(
        "m4494",
        "effdet_efficientdet_d5",
        100 * MB_PER_GB,
        180,
        "axon peak about 100 GiB",
        force_cluster=False,
    ),
    "m4495": GiantRegistryEntry(
        "m4495",
        "effdet_efficientdet_d5",
        100 * MB_PER_GB,
        180,
        "axon peak about 100 GiB",
        force_cluster=False,
    ),
    "m4523": GiantRegistryEntry(
        "m4523",
        "effdet_tf_efficientdet_d5",
        100 * MB_PER_GB,
        180,
        "axon peak about 100 GiB",
        force_cluster=False,
    ),
    "m4524": GiantRegistryEntry(
        "m4524",
        "effdet_tf_efficientdet_d5_ap",
        100 * MB_PER_GB,
        180,
        "axon peak about 100 GiB",
        force_cluster=False,
    ),
    "m4525": GiantRegistryEntry(
        "m4525",
        "effdet_tf_efficientdet_d6",
        128 * MB_PER_GB,
        180,
        "axon peak ~128 GiB; exceeds local RAM",
    ),
    "m4526": GiantRegistryEntry(
        "m4526",
        "effdet_tf_efficientdet_d6",
        183 * MB_PER_GB,
        250,
        "axon peak ~183 GiB; exceeds local RAM",
    ),
    "m4527": GiantRegistryEntry(
        "m4527",
        "effdet_tf_efficientdet_d7",
        228 * MB_PER_GB,
        250,
        "axon peak ~228 GiB; exceeds local RAM",
    ),
    "m4797": GiantRegistryEntry(
        "m4797",
        "eva02_enormous",
        94 * MB_PER_GB,
        180,
        "axon peak ~94 GiB; fits locally",
        force_cluster=False,
    ),
    "m4808": GiantRegistryEntry(
        "m4808",
        "eva_giant_560",
        80 * MB_PER_GB,
        180,
        "axon peak about 80 GiB",
        force_cluster=False,
    ),
    "m5187": GiantRegistryEntry(
        "m5187",
        "gigagan_unet_upsampler",
        90 * MB_PER_GB,
        180,
        "axon peak ~90 GiB; fits locally",
        force_cluster=False,
    ),
    "m5651": GiantRegistryEntry(
        "m5651",
        "longcat_flash",
        82 * MB_PER_GB,
        180,
        "axon peak ~82 GiB; fits locally",
        force_cluster=False,
    ),
    "m11112": GiantRegistryEntry(
        "m11112",
        "vit_so400m_896",
        98 * MB_PER_GB,
        180,
        "axon peak about 98 GiB",
        force_cluster=False,
    ),
    # Escalated-on-local-OOM giants, axon-MEASURED 2026-06-28 (completed runs ->
    # true peaks, not cap-at-kill artifacts). Seeded so they right-size on first
    # contact (tier 250/250/500) instead of re-deriving + re-OOMing every run.
    "m9025": GiantRegistryEntry(
        "m9025",
        "samvit_large_patch16",
        134 * MB_PER_GB,
        180,
        "axon-measured ~134 GiB; exceeds local RAM",
    ),
    "m4598": GiantRegistryEntry(
        "m4598",
        "efficientnet_l2",
        145 * MB_PER_GB,
        180,
        "axon-measured ~145 GiB; exceeds local RAM",
    ),
    "m9024": GiantRegistryEntry(
        "m9024",
        "samvit_huge_patch16",
        213 * MB_PER_GB,
        250,
        "axon-measured ~213 GiB; exceeds local RAM",
    ),
}


def default_command_runner(
    command: Sequence[str], *, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with captured text output.

    Parameters
    ----------
    command:
        Command arguments.
    timeout:
        Optional subprocess timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed command.
    """

    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


CLUSTER_PIXI_BIN_ENV = "TORCHLENS_CLUSTER_PIXI_BIN"
DEFAULT_CLUSTER_PIXI_BIN = Path("/tmp/tlpixi-cluster/bin/pixi")


def _local_pixi_bin() -> Path:
    """Return the pixi binary to stage on the cluster.

    The cluster runs an old userland (glibc 2.17, OpenSSL 1.0.x) that cannot load
    the workstation's glibc/OpenSSL-3 pixi build (it fails with
    ``libssl.so.3: cannot open shared object file``). A cluster-compatible,
    self-contained pixi build (the official ``x86_64-unknown-linux-musl``
    release) must be staged instead. Discovery order:

    1. ``$TORCHLENS_CLUSTER_PIXI_BIN`` -- explicit override.
    2. ``/tmp/tlpixi-cluster/bin/pixi`` -- conventional musl staging path.
    3. ``envs.pixi_bin()`` -- the local island pixi, as a last resort (works only
       when the workstation and cluster share a compatible userland).

    Returns
    -------
    pathlib.Path
        Local pixi executable path to rsync to the cluster.
    """

    override = os.environ.get(CLUSTER_PIXI_BIN_ENV)
    if override and Path(override).exists():
        return Path(override)
    if DEFAULT_CLUSTER_PIXI_BIN.exists():
        return DEFAULT_CLUSTER_PIXI_BIN
    from menagerie import envs

    return envs.pixi_bin()


def resolve_remote_home(
    config: ClusterConfig,
    command_runner: CommandRunner = default_command_runner,
) -> str:
    """Return the absolute remote home directory for ``config.host``.

    SLURM does not expand ``~`` or ``$HOME`` inside ``#SBATCH`` directives, so a
    literal absolute home is resolved once per dispatch and substituted into the
    log paths. The resolved value is cached on ``config.remote_home`` by callers.

    Parameters
    ----------
    config:
        Cluster defaults.
    command_runner:
        Injectable command runner for the ``ssh ... echo $HOME`` probe.

    Returns
    -------
    str
        Absolute remote home directory (no trailing slash).
    """

    if config.remote_home:
        return config.remote_home.rstrip("/")
    result = _run_cluster_command(("ssh", config.host, "echo $HOME"), command_runner, timeout=60.0)
    home = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if not home or not home.startswith("/"):
        raise RuntimeError(
            f"could not resolve absolute remote home for host {config.host!r}: {home!r}"
        )
    return home.rstrip("/")


def _expand_remote_path(path: str, remote_home: str) -> str:
    """Expand a leading ``~`` / ``$HOME`` against an absolute remote home.

    Parameters
    ----------
    path:
        Remote path that may begin with ``~`` or ``$HOME``.
    remote_home:
        Absolute remote home directory.

    Returns
    -------
    str
        Path with a leading home reference replaced by ``remote_home``.
    """

    home = remote_home.rstrip("/")
    if path == "~" or path == "$HOME":
        return home
    if path.startswith("~/"):
        return f"{home}/{path[2:]}"
    if path.startswith("$HOME/"):
        return f"{home}/{path[len('$HOME/') :]}"
    return path


def _path_relative_to_repo(path: Path) -> str | None:
    """Return a path's POSIX location relative to the local repo, or ``None``.

    The local repo (``REPO_ROOT``) is rsync'd to the cluster each dispatch, so a
    ledger path under the local repo root is present on the cluster -- but at the
    REMOTE repo root, not the local absolute path. This returns the relative
    component so callers can re-root it under ``config.remote_repo``. A path
    outside the repo (for example a ``/tmp`` smoke ledger) returns ``None``.

    Parameters
    ----------
    path:
        Absolute local verification-db path.

    Returns
    -------
    str | None
        POSIX relative path under the repo, or ``None`` when outside the repo.
    """

    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def load_catalog_rows_ro(
    db_path: Path = CATALOG_DB, stable_ids: Sequence[str] = ()
) -> list[CatalogRow]:
    """Load catalog rows from an existing SQLite snapshot in read-only mode.

    Parameters
    ----------
    db_path:
        Existing catalog database snapshot.
    stable_ids:
        Optional stable IDs to restrict.

    Returns
    -------
    list[CatalogRow]
        Catalog rows ordered by ``model_id``.

    Raises
    ------
    FileNotFoundError
        If the catalog snapshot does not exist.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"catalog snapshot not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    clauses: list[str] = []
    params: list[str] = []
    if stable_ids:
        placeholders = ",".join("?" for _ in stable_ids)
        clauses.append(f"stable_id IN ({placeholders})")
        params.extend(stable_ids)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT {', '.join(CATALOG_COLUMNS)} FROM models {where} ORDER BY model_id"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_catalog_row_from_sql(row) for row in rows]


def is_giant(
    row: CatalogRow | Mapping[str, object],
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None = None,
    *,
    local_ram_threshold_gb: float = LOCAL_FIRST_CLUSTER_THRESHOLD_GB,
) -> bool:
    """Return whether a row must be routed to the SHARED axon cluster.

    LOCAL-FIRST policy (axon is a shared lab cluster; use it only when a model
    genuinely cannot run locally). A model routes to the cluster ONLY with HARD
    MEASURED evidence it cannot fit in local RAM:

    * its MEASURED peak RSS (from ANY prior attempt, local or cluster) is at/above
      the usable-local-RAM threshold, OR
    * it RAM-FAILED on a prior LOCAL attempt -- an OOM / resource kill, or a
      worker-memory-cap kill at a cap near full local RAM.

    Everything else -- unmeasured/unknown-size models, models with a measured peak
    BELOW the threshold, and anything that merely LOOKS large by name, parameter
    count, or input shape -- routes LOCAL. An unmeasured model is never
    preemptively sent to the cluster: if it OOMs locally the ledger records it and
    it escalates on a later run (the intended escalation path). Size ESTIMATES
    (param count / FLOPs / name patterns) are deliberately NOT a routing signal --
    only measured RAM evidence counts. The static :data:`GIANT_REGISTRY` may still
    force-route the rare genuine can't-fit model before its first measurement.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.
    ledger:
        Ledger connection/path, or a stable-ID to peak-RSS mapping.
    local_ram_threshold_gb:
        Usable local-RAM RSS threshold in GiB. A measured peak at/above this is
        proof the model cannot fit locally.

    Returns
    -------
    bool
        Whether the model should be handled by the cluster runner.
    """

    stable_id = _row_value(row, "stable_id")
    _status, peak_rss_mb = _latest_status_and_peak(stable_id, ledger)
    threshold_mb = int(local_ram_threshold_gb * MB_PER_GB)
    # (1) Hard evidence: a measured peak at/above usable local RAM. The peak may
    # have been measured on the cluster (e.g. a 228 GiB effdet_d7 run) -- that is
    # still genuine proof the model cannot fit in local RAM.
    measured_peak_mb = _max_measured_peak_mb(stable_id, ledger, latest_peak_mb=peak_rss_mb)
    if measured_peak_mb is not None and measured_peak_mb >= threshold_mb:
        return True
    # (2) Hard evidence: the model RAM-failed on a prior LOCAL attempt (OOM /
    # memory native crash / kill, or a memory-cap kill near full local RAM).
    if _had_local_ram_failure(stable_id, ledger):
        return True
    # (3) Narrow static seed for a genuine can't-fit model with no measurement yet.
    entry = GIANT_REGISTRY.get(stable_id)
    if entry is not None and entry.force_cluster:
        return True
    # Otherwise LOCAL-FIRST: unmeasured, small-peak, or merely large-looking.
    return False


def route_giants(
    rows: Sequence[CatalogRow],
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None = None,
    *,
    local_ram_threshold_gb: float = LOCAL_FIRST_CLUSTER_THRESHOLD_GB,
) -> tuple[CatalogRow, ...]:
    """Return rows that should be routed to the cluster.

    Parameters
    ----------
    rows:
        Candidate catalog rows.
    ledger:
        Ledger connection/path, or a stable-ID to peak-RSS mapping.
    local_ram_threshold_gb:
        Local machine RSS threshold in GiB.

    Returns
    -------
    tuple[CatalogRow, ...]
        Cluster-routed rows in input order.
    """

    return tuple(
        row
        for row in rows
        if is_giant(row, ledger=ledger, local_ram_threshold_gb=local_ram_threshold_gb)
    )


def requires_cuda(
    row: CatalogRow | Mapping[str, object],
    *,
    requires_cuda_set: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a row is known to require CUDA to validate.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.
    requires_cuda_set:
        Stable-ID mapping of catalog-confirmed CUDA-required rows.

    Returns
    -------
    bool
        ``True`` only for explicitly cataloged hard-CUDA rows.
    """

    active_set = REQUIRES_CUDA if requires_cuda_set is None else requires_cuda_set
    return _row_value(row, "stable_id") in active_set


def gpu_mem_fit_estimate(
    row: CatalogRow | Mapping[str, object],
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None = None,
) -> int:
    """Return a coarse GPU memory fit estimate in bytes.

    This estimate is operational routing input only. It never validates a model
    and never escalates a CPU-eligible row; callers use it only after
    ``requires_cuda(row)`` is already true.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.
    ledger:
        Optional ledger evidence. Currently unused; reserved for future measured
        CUDA memory estimates without changing the public helper shape.

    Returns
    -------
    int
        Coarse estimated GPU memory requirement in bytes.
    """

    del ledger
    haystack = " ".join(
        _row_value(row, field) for field in ("name", "family", "family_normalized", "notes")
    )
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bmk])\b", haystack, flags=re.IGNORECASE)
    if match is None:
        return 1 * 1024**3
    value = float(match.group(1))
    suffix = match.group(2).casefold()
    multiplier = {"b": 1_000_000_000, "m": 1_000_000, "k": 1_000}[suffix]
    param_bytes = int(value * multiplier * 4)
    return max(1 * 1024**3, param_bytes * 3)


def route_resources(
    row: CatalogRow | Mapping[str, object],
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None = None,
    *,
    local_gpu_vram_bytes: int | None = None,
    config: ClusterConfig | None = None,
) -> ResourceRoute:
    """Return the operational resource route for one validation row.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.
    ledger:
        Ledger evidence used by the existing RAM-giant router.
    local_gpu_vram_bytes:
        Probed local GPU VRAM in bytes, or ``None`` when no local GPU is usable.
    config:
        Cluster config placeholder for future GPU tier policy. Accepted to keep
        routing call sites explicit; currently unused.

    Returns
    -------
    ResourceRoute
        Explicit lane, worker device, cluster flag, and reason.
    """

    del config
    if requires_cuda(row):
        required_bytes = gpu_mem_fit_estimate(row, ledger)
        if (
            local_gpu_vram_bytes is not None
            and local_gpu_vram_bytes > 0
            and required_bytes <= int(0.8 * local_gpu_vram_bytes)
        ):
            return ResourceRoute(
                "local-gpu",
                "cuda",
                False,
                f"requires_cuda; estimated_vram_bytes={required_bytes}",
            )
        return ResourceRoute(
            "cluster-gpu",
            "cuda",
            True,
            f"requires_cuda; estimated_vram_bytes={required_bytes}",
        )
    if is_giant(row, ledger=ledger):
        return ResourceRoute("cluster-ram", "cpu", True, "measured_or_static_ram_giant")
    return ResourceRoute("local-cpu", "cpu", False, "local_first_default")


def probe_local_gpu_vram_bytes(command_runner: CommandRunner | None = None) -> int | None:
    """Return local GPU VRAM bytes using ``nvidia-smi`` without importing torch.

    Parameters
    ----------
    command_runner:
        Injectable command runner for tests. Defaults to the module subprocess
        runner when omitted.

    Returns
    -------
    int | None
        First GPU total memory in bytes, or ``None`` when unavailable.
    """

    command = (
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    )
    try:
        if command_runner is None:
            result = default_command_runner(command, timeout=2.0)
        else:
            result = command_runner(command)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    first_line = (result.stdout or "").splitlines()[0:1]
    if not first_line:
        return None
    try:
        mib = int(first_line[0].strip())
    except ValueError:
        return None
    return mib * 1024**2


def poll_cluster_terminal(
    dispatch: DispatchResult,
    *,
    config: ClusterConfig | None = None,
    command_runner: CommandRunner = default_command_runner,
    poll_interval_sec: float = 30.0,
    timeout_sec: float | None = None,
) -> bool:
    """Poll SLURM until every dispatched array job reaches a terminal state.

    Parameters
    ----------
    dispatch:
        Cluster dispatch metadata carrying sbatch job IDs.
    config:
        Cluster connection defaults.
    command_runner:
        Injectable command runner for ``ssh sacct``.
    poll_interval_sec:
        Sleep interval between polls.
    timeout_sec:
        Maximum wall-clock polling duration. Defaults to the cluster config's
        blocking sbatch wait timeout.

    Returns
    -------
    bool
        ``True`` only when every observed state is terminal before timeout.
    """

    if not dispatch.sbatch_job_ids:
        # No submitted job IDs (e.g. a job-ID parse failure) is NOT terminal: there
        # is no accounting evidence that anything finished, so treating it as
        # terminal would let collect attribute every missing artifact as a failure.
        return False
    active_config = config or ClusterConfig()
    deadline = time.monotonic() + (
        active_config.sbatch_wait_timeout_sec if timeout_sec is None else timeout_sec
    )
    while True:
        try:
            states_by_job = _sacct_states_by_job(
                dispatch.sbatch_job_ids, active_config, command_runner
            )
            # Require an observed TERMINAL state for EVERY submitted job ID. A job
            # not yet visible in SLURM accounting (routine sacct lag) produces NO
            # line -> NOT terminal -> stay pending. This prevents a partial sacct
            # view (job A COMPLETED while job B is still running / not yet in the
            # accounting DB) from being read as fully terminal, which would stamp
            # job B's still-running tasks failed:cluster_task_failed.
            if all(
                states_by_job.get(job_id)
                and all(state in SLURM_TERMINAL_STATES for state in states_by_job[job_id])
                for job_id in dispatch.sbatch_job_ids
            ):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, poll_interval_sec))


def _sacct_states_by_job(
    job_ids: Sequence[str],
    config: ClusterConfig,
    command_runner: CommandRunner,
) -> dict[str, tuple[str, ...]]:
    """Return observed SLURM states grouped by SUBMITTED job ID from ``sacct``.

    Each sacct JobID is mapped back to the submitted job ID it belongs to so a
    submitted ID is "observed" iff at least one of its (allocation or array-task)
    lines appears. A job not yet propagated to the SLURM accounting DB produces NO
    line, so it is simply ABSENT from the returned mapping -- the caller treats an
    absent submitted ID as non-terminal (still pending), never as terminal.

    Parameters
    ----------
    job_ids:
        SLURM job IDs to inspect.
    config:
        Cluster connection defaults.
    command_runner:
        Injectable command runner.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Observed terminal/non-terminal state names keyed by submitted job ID
        (array-task JobIDs ``<jobid>_<index>`` fold back onto ``<jobid>``). Only
        submitted IDs with at least one observed line appear.
    """

    submitted = tuple(job_ids)
    command = (
        "ssh",
        config.host,
        "sacct -X -j " + ",".join(submitted) + " --noheader --parsable2 --format=JobID,State",
    )
    result = command_runner(command)
    states_by_job: dict[str, list[str]] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        observed_job_id = parts[0].split()[0]
        # Array tasks report as ``<jobid>_<index>``; fold them onto the submitted
        # base job ID so the base is counted as observed.
        base_job_id = observed_job_id.split("_", 1)[0].split(".", 1)[0]
        if base_job_id in submitted:
            target = base_job_id
        elif observed_job_id in submitted:
            target = observed_job_id
        else:
            # A JobID we did not submit (defensive); skip it.
            continue
        state = parts[1].split()[0].upper()
        states_by_job.setdefault(target, []).append(state)
    return {job_id: tuple(states) for job_id, states in states_by_job.items()}


def node_tier_for_row(
    row: CatalogRow | Mapping[str, object],
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None = None,
    *,
    config: ClusterConfig | None = None,
) -> NodeTier:
    """Return the right-sized SLURM memory tier for a row.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.
    ledger:
        Ledger connection/path, or a stable-ID to peak-RSS mapping.
    config:
        Cluster defaults.

    Returns
    -------
    NodeTier
        Selected memory tier.
    """

    active_config = config or ClusterConfig()
    if requires_cuda(row):
        return active_config.gpu_node_tier
    stable_id = _row_value(row, "stable_id")
    entry = GIANT_REGISTRY.get(stable_id)
    status, measured_peak_mb = _latest_status_and_peak(stable_id, ledger)
    max_measured_peak_mb = _max_measured_peak_mb(stable_id, ledger, latest_peak_mb=measured_peak_mb)
    if _had_local_ram_failure(stable_id, ledger):
        if max_measured_peak_mb is not None and max_measured_peak_mb > 0:
            measured_peak_mb = max_measured_peak_mb
        else:
            # No measured peak: scale by the trait heuristic and ladder up one tier
            # per repeat OOM. NEVER jump straight to the largest tier -- a model that
            # OOM'd on a ~125GB box needs the next tier up, not a 1TB node.
            return _laddered_escalation_tier(row, stable_id, ledger, active_config)
    if (
        status == "oom"
        and entry is None
        and _looks_like_unregistered_moe_monster(row)
        and _oom_run_count(stable_id, ledger) >= 2
    ):
        return _largest_tier(active_config)
    if measured_peak_mb is None and entry is not None:
        measured_peak_mb = entry.measured_peak_rss_mb
    if entry is not None and measured_peak_mb is None:
        return _tier_from_entry(entry, active_config)
    if measured_peak_mb is None:
        measured_peak_mb = _heuristic_peak_mb(row)
    peak_gb = max(1, (measured_peak_mb + MB_PER_GB - 1) // MB_PER_GB)
    for tier in active_config.node_tiers:
        if peak_gb <= tier.max_peak_rss_gb:
            return tier
    return active_config.node_tiers[-1]


def pending_assignments_for_resume(
    assignments: Sequence[ClusterAssignment],
    targets: Mapping[str, VerificationTarget],
    *,
    ledger_db: Path | None = None,
) -> tuple[ClusterAssignment, ...]:
    """Filter assignments using the ledger as the only completion source.

    Parameters
    ----------
    assignments:
        Candidate cluster assignments.
    targets:
        Current identity targets keyed by stable ID.
    ledger_db:
        Verification ledger path.

    Returns
    -------
    tuple[ClusterAssignment, ...]
        Assignments lacking a current terminal ledger row.
    """

    completed = ledger_completed_stable_ids(targets, ledger_db=ledger_db)
    return tuple(assignment for assignment in assignments if assignment.stable_id not in completed)


def ledger_completed_stable_ids(
    targets: Mapping[str, VerificationTarget],
    *,
    ledger_db: Path | None = None,
) -> set[str]:
    """Return stable IDs completed for the current ledger identity tuple.

    Parameters
    ----------
    targets:
        Current identity targets keyed by stable ID.
    ledger_db:
        Verification ledger path.

    Returns
    -------
    set[str]
        Stable IDs with matching terminal rows.
    """

    if not targets:
        return set()
    with connect_ledger(_resolve_verification_db(ledger_db)) as conn:
        conn.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS temp_cluster_targets(
                stable_id TEXT PRIMARY KEY,
                recipe_revision_sha256 TEXT NOT NULL,
                torchlens_source_hash TEXT NOT NULL,
                env_hash TEXT NOT NULL,
                lock_hash TEXT NOT NULL,
                device_requested TEXT NOT NULL,
                scope TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM temp_cluster_targets")
        conn.executemany(
            """
            INSERT INTO temp_cluster_targets(
                stable_id,
                recipe_revision_sha256,
                torchlens_source_hash,
                env_hash,
                lock_hash,
                device_requested,
                scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    stable_id,
                    target.recipe_revision_sha256,
                    target.torchlens_source_hash,
                    target.env_hash,
                    target.lock_hash,
                    target.device_requested,
                    target.scope,
                )
                for stable_id, target in targets.items()
            ),
        )
        rows = conn.execute(
            """
            SELECT current_verification.stable_id
            FROM current_verification
            JOIN temp_cluster_targets AS target
              ON target.stable_id = current_verification.stable_id
             AND target.recipe_revision_sha256 = current_verification.recipe_revision_sha256
             AND target.torchlens_source_hash = current_verification.torchlens_source_hash
             AND target.env_hash = current_verification.env_hash
             AND target.lock_hash = current_verification.lock_hash
             AND target.device_requested = current_verification.device_requested
             AND target.scope = current_verification.scope
            WHERE current_verification.status IN (
                'passed',
                'failed',
                'skipped',
                'timeout',
                'not_applicable',
                'deferred',
                'error',
                'install_failed',
                'env_unavailable',
                'oom',
                'native_crash',
                'killed'
            )
              AND NOT (
                  current_verification.torchlens_source_hash = ?
                  OR current_verification.lock_hash = ?
              )
            """,
            (LEGACY_UNKNOWN, LEGACY_UNKNOWN),
        ).fetchall()
    return {str(row["stable_id"]) for row in rows}


def dispatch_giants(
    stable_ids: Sequence[str],
    *,
    catalog_db: Path = CATALOG_DB,
    ledger_db: Path | None = None,
    repo_root: Path = REPO_ROOT,
    local_artifact_root: Path = CLUSTER_ARTIFACT_ROOT,
    config: ClusterConfig | None = None,
    command_runner: CommandRunner = default_command_runner,
    campaign_id: str | None = None,
    attempt_id: str | None = None,
    timeout_by_id: Mapping[str, float] | None = None,
    input_scale_by_id: Mapping[str, float] | None = None,
    dry_run: bool = False,
    wait: bool = True,
) -> DispatchResult:
    """Prepare and submit a SLURM array for giant model validation.

    Parameters
    ----------
    stable_ids:
        Stable IDs to dispatch.
    catalog_db:
        Existing catalog snapshot. It is copied and opened read-only by workers.
    ledger_db:
        Local verification ledger used for routing estimates.
    repo_root:
        Local repository root to rsync.
    local_artifact_root:
        Local dispatch artifact root.
    config:
        Cluster defaults.
    command_runner:
        Injectable subprocess runner for rsync/ssh/sbatch calls.
    campaign_id:
        Optional campaign key. Defaults to a timestamped key.
    attempt_id:
        Optional attempt key. Defaults to ``"attempt-1"``.
    timeout_by_id:
        Optional per-stable-ID validator timeout overrides.
    input_scale_by_id:
        Optional per-stable-ID input-scale overrides.
    dry_run:
        Prepare artifacts and commands without executing them.
    wait:
        Whether ``sbatch`` should block with ``--wait``. Defaults to the legacy
        blocking behavior.

    Returns
    -------
    DispatchResult
        Dispatch artifact and submission metadata.
    """

    if not stable_ids:
        raise ValueError("stable_ids must not be empty")
    active_config = config or ClusterConfig()
    resolved_campaign = campaign_id or f"cluster-{utc_now().replace(':', '').replace('+', 'Z')}"
    resolved_attempt = attempt_id or "attempt-1"
    artifact_dir = local_artifact_root / resolved_campaign / resolved_attempt
    artifact_dir.mkdir(parents=True, exist_ok=True)
    catalog_snapshot = artifact_dir / "catalog.db"
    _copy_catalog_snapshot(catalog_db, catalog_snapshot)
    rows = load_catalog_rows_ro(catalog_snapshot, stable_ids=stable_ids)
    rows_by_id = {row.stable_id: row for row in rows}
    missing = sorted(set(stable_ids).difference(rows_by_id))
    if missing:
        raise ValueError(f"stable IDs missing from catalog snapshot: {missing!r}")
    resolved_ledger_db = _resolve_verification_db(ledger_db)
    assignments = tuple(
        _assignment_for_row(
            row=rows_by_id[stable_id],
            index=index,
            ledger_db=resolved_ledger_db,
            config=active_config,
            campaign_id=resolved_campaign,
            attempt_id=resolved_attempt,
            timeout_sec=(timeout_by_id or {}).get(stable_id, 14400.0),
            input_scale=(input_scale_by_id or {}).get(stable_id, 1.0),
        )
        for index, stable_id in enumerate(stable_ids)
    )
    assignment_path = artifact_dir / "assignments.json"
    write_assignment_manifest(assignments, assignment_path)
    remote_artifact_dir = (
        f"{active_config.remote_artifact_root.rstrip('/')}/{resolved_campaign}/{resolved_attempt}"
    )
    # Resolve the absolute remote home once so SLURM #SBATCH log directives are
    # not written with a literal '~' (which SLURM does not expand, causing the
    # job to fail instantly with no .err log). Dry-run renders without SSH.
    if dry_run:
        remote_home = active_config.remote_home or "$HOME"
    else:
        remote_home = resolve_remote_home(active_config, command_runner)
    sbatch_paths = _write_sbatch_scripts(
        assignments,
        artifact_dir=artifact_dir,
        config=active_config,
        remote_artifact_dir=remote_artifact_dir,
        verification_db=resolved_ledger_db,
        remote_home=remote_home,
    )
    commands = _dispatch_commands(
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        remote_artifact_dir=remote_artifact_dir,
        config=active_config,
        sbatch_paths=sbatch_paths,
        wait=wait,
    )
    sbatch_job_ids: list[str] = []
    if not dry_run:
        # Setup commands are everything before the per-tier sbatch submissions;
        # derive the count instead of hardcoding it so adding setup steps (e.g.
        # the pixi-bootstrap rsync) cannot silently desynchronize the split.
        setup_command_count = len(commands) - len(sbatch_paths)
        for command in commands[:setup_command_count]:
            _run_cluster_command(command, command_runner, timeout=None)
        for command in commands[setup_command_count:]:
            try:
                result, job_id = _run_sbatch_command(
                    command,
                    command_runner,
                    timeout=active_config.sbatch_wait_timeout_sec if wait else None,
                    wait=wait,
                )
            except ClusterJobFailed as error:
                error.dispatch = DispatchResult(
                    campaign_id=resolved_campaign,
                    attempt_id=resolved_attempt,
                    assignments=assignments,
                    local_artifact_dir=artifact_dir,
                    remote_artifact_dir=remote_artifact_dir,
                    sbatch_job_ids=tuple(sbatch_job_ids),
                    commands=tuple(tuple(cmd) for cmd in commands),
                    dry_run=dry_run,
                )
                raise
            if job_id is not None:
                sbatch_job_ids.append(job_id)
    return DispatchResult(
        campaign_id=resolved_campaign,
        attempt_id=resolved_attempt,
        assignments=assignments,
        local_artifact_dir=artifact_dir,
        remote_artifact_dir=remote_artifact_dir,
        sbatch_job_ids=tuple(sbatch_job_ids),
        commands=tuple(tuple(command) for command in commands),
        dry_run=dry_run,
    )


def render_sbatch_script(
    assignments: Sequence[ClusterAssignment],
    *,
    config: ClusterConfig,
    remote_artifact_dir: str,
    verification_db: Path | None = None,
    remote_home: str | None = None,
) -> str:
    """Render a self-contained SLURM array script.

    Parameters
    ----------
    assignments:
        Cluster assignments.
    config:
        Cluster defaults.
    remote_artifact_dir:
        Remote directory containing dispatch artifacts.
    verification_db:
        Verification ledger path to export and pass to the worker.
    remote_home:
        Absolute remote home used to expand ``~`` for the SLURM ``#SBATCH``
        ``--output`` / ``--error`` directives, which SLURM does not expand.
        Defaults to ``config.remote_home`` when omitted.

    Returns
    -------
    str
        Bash sbatch script text.
    """

    if not assignments:
        raise ValueError("assignments must not be empty")
    resolved_home = remote_home or config.remote_home
    if not resolved_home:
        raise ValueError(
            "remote_home is required to render SLURM #SBATCH log paths; SLURM does "
            "not expand '~' or '$HOME' in #SBATCH directives"
        )
    indexes = ",".join(str(assignment.array_index) for assignment in assignments)
    default_tier = assignments[0]
    manifest = f"{remote_artifact_dir}/assignments.json"
    resolved_verification_db = _resolve_verification_db(verification_db)
    remote_repo = _expand_remote_path(config.remote_repo, resolved_home)
    # SLURM #SBATCH directives are parsed before the job shell runs and do not
    # perform tilde/$HOME expansion, so the log paths must be absolute.
    log_dir = _expand_remote_path(f"{remote_artifact_dir}/logs", resolved_home)
    # The worker writes its verification ledger on the cluster node. A ledger
    # path under the rsync'd repo is present on the node (mapped to the REMOTE
    # repo root); any other local path (e.g. a /tmp smoke ledger) is not, so it
    # is redirected to a node-local path under the remote artifact dir. The
    # emitted path is always absolute -- bash does NOT expand a leading '~'
    # inside the double-quoted export/CLI values. Results are rsync'd back and
    # merged into the caller's ledger regardless, so this redirect loses no rows.
    repo_relative = _path_relative_to_repo(Path(resolved_verification_db))
    if repo_relative is not None:
        worker_verification_db = f"{remote_repo}/{repo_relative}"
    else:
        worker_verification_db = _expand_remote_path(
            f"{remote_artifact_dir}/worker_ledger/{Path(resolved_verification_db).name}",
            resolved_home,
        )
    lock_hash = compute_lock_hash_for_env(config.pixi_env)
    pixi_bin = _expand_remote_path(config.remote_pixi_bin, resolved_home)
    gres_line = "#SBATCH --gres=gpu:1\n" if default_tier.gpu else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=tl_cluster_giants
#SBATCH --account={config.account}
#SBATCH --partition={default_tier.partition or config.partition}
#SBATCH --mem={default_tier.node_mem_gb}G
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --time={config.time_limit}
{gres_line.rstrip()}
#SBATCH --array={indexes}%{config.array_concurrency}
#SBATCH --output={log_dir}/giant_%A_%a.log
#SBATCH --error={log_dir}/giant_%A_%a.err

set -euo pipefail
cd {config.remote_repo}
mkdir -p {remote_artifact_dir}/logs {remote_artifact_dir}/results
mkdir -p "$(dirname "{worker_verification_db}")"
export {ENV_VERIFICATION_DB}="{worker_verification_db}"
# The pixi env install prefix and package cache must live on a filesystem that
# supports file locking (flock). The cluster $HOME is on NFS/Lustre where flock
# fails with ENOLCK ("No locks available", os error 37), so default both to
# node-local /tmp. PIXI_CACHE_DIR moves the global package cache off NFS.
PROJECT_ROOT="${{TORCHLENS_CLUSTER_PIXI_ROOT:-/tmp/torchlens-cluster-pixi-$USER}}"
export PIXI_CACHE_DIR="${{PIXI_CACHE_DIR:-/tmp/pixi-cache-$USER}}"
# The lock hash is baked in at render time from the committed, rsync'd lock
# files (byte-identical on the node). Computing it here avoids invoking a bare
# `python` before the pixi env exists -- the cluster login/compute node ships
# only Python 2.7 as `python` and has no `python3`, so a remote
# `python -m menagerie.cluster_runner lock-hash` raised a SyntaxError and failed
# the job before any validation ran.
LOCK_HASH="{lock_hash}"
PROJECT_DIR="$PROJECT_ROOT/{config.pixi_env}-${{LOCK_HASH:0:16}}"
mkdir -p "$PROJECT_DIR"
cp "menagerie/locks/{config.pixi_env}.pixi.toml" "$PROJECT_DIR/pixi.toml"
cp "menagerie/locks/{config.pixi_env}.pixi.lock" "$PROJECT_DIR/pixi.lock"
# The cluster nodes do not ship pixi on PATH; it is staged at PIXI_BIN by the
# dispatch setup. Invoke it by absolute path.
PIXI_BIN="{pixi_bin}"
"$PIXI_BIN" install --manifest-path "$PROJECT_DIR/pixi.toml" --locked
"$PIXI_BIN" run --manifest-path "$PROJECT_DIR/pixi.toml" --frozen -- \\
    python -u -m menagerie.cluster_runner worker \\
    --assignment-manifest {manifest} \\
    --task-index "$SLURM_ARRAY_TASK_ID" \\
    --repo-root {config.remote_repo} \\
    --result-dir {remote_artifact_dir}/results \\
    --verification-db "{worker_verification_db}"
"""


def write_assignment_manifest(assignments: Sequence[ClusterAssignment], path: Path) -> None:
    """Write dispatch assignments as JSON.

    Parameters
    ----------
    assignments:
        Assignments to write.
    path:
        Destination JSON path.
    """

    if not assignments:
        raise ValueError("assignments must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_id": assignments[0].campaign_id,
        "attempt_id": assignments[0].attempt_id,
        "assignments": [asdict(assignment) for assignment in assignments],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_assignment_manifest(path: Path) -> tuple[ClusterAssignment, ...]:
    """Load dispatch assignments from JSON.

    Parameters
    ----------
    path:
        Assignment manifest path.

    Returns
    -------
    tuple[ClusterAssignment, ...]
        Loaded assignments.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ClusterAssignment(**item) for item in payload["assignments"])


def run_worker_assignment(
    assignment_manifest: Path,
    task_index: int,
    *,
    repo_root: Path = REPO_ROOT,
    result_dir: Path,
    verification_db: Path | None = None,
    command_runner: CommandRunner = default_command_runner,
) -> ClusterResultRow:
    """Run one cluster assignment and export its result row.

    Parameters
    ----------
    assignment_manifest:
        Assignment manifest path.
    task_index:
        SLURM array task index.
    repo_root:
        Repository root on the worker host.
    result_dir:
        Directory where result JSONL and host-contract files are written.
    verification_db:
        Verification ledger path used by the child validator and latest-row read.
    command_runner:
        Injectable command runner.

    Returns
    -------
    ClusterResultRow
        Exported cluster result row.
    """

    assignments = load_assignment_manifest(assignment_manifest)
    by_index = {assignment.array_index: assignment for assignment in assignments}
    if task_index not in by_index:
        raise ValueError(f"no assignment for task index {task_index}")
    assignment = by_index[task_index]
    result_dir.mkdir(parents=True, exist_ok=True)
    record_host_contract(result_dir / f"{assignment.assignment_id}.host.json")
    resolved_verification_db = _resolve_verification_db(verification_db)
    command = [
        sys.executable,
        "-u",
        "-m",
        "menagerie.validate_menagerie",
        "--stable-ids",
        assignment.stable_id,
        "--jobs",
        "1",
        "--worker-memory-cap-gb",
        str(assignment.worker_memory_cap_gb),
        "--timeout-sec",
        str(assignment.timeout_sec),
        "--input-scale",
        str(assignment.input_scale),
        "--min-free-gb",
        "10",
        # The worker IS the cluster node; validate the giant in-place. Without
        # --runner local the child validator defaults to --runner auto and
        # re-routes the giant straight back to the cluster (an infinite nested
        # dispatch that records env_unavailable instead of validating).
        # --base-env-only stops it re-routing into a pixi island env too.
        "--runner",
        "local",
        "--base-env-only",
        "--out-dir",
        str(result_dir / assignment.stable_id),
        "--db",
        str(assignment_manifest.parent / "catalog.db"),
        "--no-build-catalog",
        "--verification-db",
        str(resolved_verification_db),
    ]
    if assignment.gpu:
        command.extend(("--device", "cuda"))
    previous_verification_db = os.environ.get(ENV_VERIFICATION_DB)
    os.environ[ENV_VERIFICATION_DB] = str(resolved_verification_db)
    try:
        try:
            command_runner(command)
        except subprocess.CalledProcessError as exc:
            # C1: surface the child validator's captured stdout/stderr instead of
            # swallowing it. Without this, a giant/cluster failure shows only a bare
            # "returned non-zero exit status N" in the slurm .err -- the real reason
            # (OOM, dependency skip, etc.) is captured by capture_output=True and lost.
            if exc.stdout:
                sys.stderr.write(
                    "=== child validator stdout (surfaced by C1) ===\n" + exc.stdout + "\n"
                )
            if exc.stderr:
                sys.stderr.write(
                    "=== child validator stderr (surfaced by C1) ===\n" + exc.stderr + "\n"
                )
            sys.stderr.flush()
            row = _append_child_failure_run(
                assignment,
                catalog_db=assignment_manifest.parent / "catalog.db",
                ledger_db=resolved_verification_db,
                exc=exc,
            )
            result = ClusterResultRow(
                campaign_id=assignment.campaign_id,
                attempt_id=assignment.attempt_id,
                assignment_id=assignment.assignment_id,
                run=row,
            )
            result_path = result_dir / f"{assignment.assignment_id}.jsonl"
            write_result_rows_jsonl((result,), result_path)
            write_result_manifest(
                (result,), result_dir / f"{assignment.assignment_id}.manifest.json"
            )
            return result
    finally:
        if previous_verification_db is None:
            os.environ.pop(ENV_VERIFICATION_DB, None)
        else:
            os.environ[ENV_VERIFICATION_DB] = previous_verification_db
    row = latest_verification_run_for_stable_id(
        assignment.stable_id,
        ledger_db=resolved_verification_db,
    )
    result = ClusterResultRow(
        campaign_id=assignment.campaign_id,
        attempt_id=assignment.attempt_id,
        assignment_id=assignment.assignment_id,
        run=row,
    )
    result_path = result_dir / f"{assignment.assignment_id}.jsonl"
    write_result_rows_jsonl((result,), result_path)
    write_result_manifest((result,), result_dir / f"{assignment.assignment_id}.manifest.json")
    return result


def _catalog_row_for_assignment(stable_id: str, catalog_db: Path) -> CatalogRow:
    """Return the catalog row for a cluster worker assignment.

    Parameters
    ----------
    stable_id:
        Durable model identity from the assignment manifest.
    catalog_db:
        Snapshot catalog database staged beside the assignment manifest.

    Returns
    -------
    CatalogRow
        Matching catalog row.
    """

    with sqlite3.connect(catalog_db) as conn:
        row = conn.execute(
            """
            SELECT model_id, display_index, stable_id, name, variant, family,
                   family_normalized, domain, zoo, constructor_call, input_shape,
                   input_dtype, era, verified, notes, source, recipe_revision_sha256,
                   input_is_real, verification_expectation, quarantine
            FROM models
            WHERE stable_id = ?
            """,
            (stable_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"no catalog row for stable_id {stable_id}")
    return CatalogRow(
        model_id=int(row[0]),
        display_index=int(row[1]),
        stable_id=str(row[2]),
        name=str(row[3]),
        variant=str(row[4]),
        family=str(row[5]),
        family_normalized=str(row[6]),
        domain=str(row[7]),
        zoo=str(row[8]),
        constructor_call=str(row[9]),
        input_shape=str(row[10]),
        input_dtype=str(row[11]),
        era=str(row[12]),
        verified=bool(row[13]),
        notes=str(row[14]),
        source=str(row[15]),
        recipe_revision_sha256=str(row[16]),
        input_is_real=bool(row[17]),
        verification_expectation=str(row[18]),
        quarantine=bool(row[19]),
    )


def _child_failure_message(exc: subprocess.CalledProcessError) -> str:
    """Return an honest ledger message for a failed child validator process.

    Parameters
    ----------
    exc:
        Child validator process failure with captured output.

    Returns
    -------
    str
        Compact message containing the return code plus captured stdout/stderr.
    """

    parts = [f"child validator failed with exit code {exc.returncode}"]
    if exc.stdout:
        parts.append("stdout:\n" + exc.stdout.strip())
    if exc.stderr:
        parts.append("stderr:\n" + exc.stderr.strip())
    return "\n\n".join(parts)


def _append_child_failure_run(
    assignment: ClusterAssignment,
    *,
    catalog_db: Path,
    ledger_db: Path,
    exc: subprocess.CalledProcessError,
) -> VerificationRun:
    """Append and return a failed ledger row for a crashed worker child.

    Parameters
    ----------
    assignment:
        Cluster assignment being executed.
    catalog_db:
        Snapshot catalog database staged beside the assignment manifest.
    ledger_db:
        Verification ledger path used by the child validator.
    exc:
        Child validator process failure.

    Returns
    -------
    VerificationRun
        Appended failed row.
    """

    row = _catalog_row_for_assignment(assignment.stable_id, catalog_db)
    started_at = utc_now()
    finished_at = utc_now()
    run = VerificationRun(
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        name=row.name,
        zoo=row.zoo,
        variant=row.variant,
        scope="forward",
        status="failed",
        forward_pass=None,
        backward_pass=None,
        backward_na_reason=None,
        metadata_ok=None,
        n_ops=None,
        graph_shape_hash=None,
        svg_sha256=None,
        torchlens_version=LEGACY_UNKNOWN,
        torch_version=LEGACY_UNKNOWN,
        python_version=sys.version.split()[0],
        device_requested="cuda" if assignment.gpu else "cpu",
        device_actual=None,
        env_hash=os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH", LEGACY_UNKNOWN),
        lock_hash=os.environ.get("TORCHLENS_MENAGERIE_LOCK_HASH", LEGACY_UNKNOWN),
        torchlens_source_hash=os.environ.get("TORCHLENS_SOURCE_HASH", LEGACY_UNKNOWN),
        input_scale=assignment.input_scale,
        runner_host=socket.gethostname(),
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=0.0,
        peak_rss_mb=None,
        error_class="failed:cluster_job_failed",
        error_message=_child_failure_message(exc),
    )
    with connect_ledger(ledger_db) as conn:
        append_verification_run(conn, run)
    return latest_verification_run_for_stable_id(row.stable_id, ledger_db=ledger_db)


def latest_verification_run_for_stable_id(
    stable_id: str, *, ledger_db: Path | None = None
) -> VerificationRun:
    """Return the latest verification run for a stable ID.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger_db:
        Verification ledger path.

    Returns
    -------
    VerificationRun
        Latest verification run.
    """

    with connect_ledger(_resolve_verification_db(ledger_db)) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM current_verification
            WHERE stable_id = ?
            """,
            (stable_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"no verification row for stable_id {stable_id}")
    return _verification_run_from_row(row)


def current_real_verification_run_for_stable_id(
    stable_id: str, *, ledger_db: Path | None = None
) -> VerificationRun:
    """Return the current CASCADE-SUPPRESSED verification run for a stable ID.

    Reads the ``current_verification_real`` view so a frozen batch-cascade
    artifact never shadows the model's real current row. Used to reconstruct the
    manifest row for an incrementally-skipped (provably current + passed) model.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger_db:
        Verification ledger path.

    Returns
    -------
    VerificationRun
        Current cascade-suppressed verification run.
    """

    with connect_ledger(_resolve_verification_db(ledger_db)) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM current_verification_real
            WHERE stable_id = ?
            """,
            (stable_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"no real verification row for stable_id {stable_id}")
    return _verification_run_from_row(row)


def write_result_rows_jsonl(rows: Sequence[ClusterResultRow], path: Path) -> None:
    """Write exported cluster result rows as JSONL.

    Parameters
    ----------
    rows:
        Result rows.
    path:
        Destination JSONL path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_cluster_result_payload(row), sort_keys=True) + "\n")


def load_result_rows_jsonl(path: Path) -> tuple[ClusterResultRow, ...]:
    """Load exported cluster result rows from JSONL.

    Parameters
    ----------
    path:
        Source JSONL path.

    Returns
    -------
    tuple[ClusterResultRow, ...]
        Result rows.
    """

    rows: list[ClusterResultRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                ClusterResultRow(
                    campaign_id=str(payload["campaign_id"]),
                    attempt_id=str(payload["attempt_id"]),
                    assignment_id=str(payload["assignment_id"]),
                    run=VerificationRun(**payload["run"]),
                )
            )
    return tuple(rows)


def write_result_manifest(rows: Sequence[ClusterResultRow], path: Path) -> None:
    """Write expected counts and checksums for exported result rows.

    Parameters
    ----------
    rows:
        Result rows.
    path:
        Destination manifest path.
    """

    if not rows:
        raise ValueError("rows must not be empty")
    grouped: dict[tuple[str, str, str], list[ClusterResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.campaign_id, row.attempt_id, row.assignment_id)].append(row)
    assignments = [
        {
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "assignment_id": assignment_id,
            "expected_row_count": len(group),
            "result_checksum": _assignment_result_checksum(group),
        }
        for (campaign_id, attempt_id, assignment_id), group in sorted(grouped.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"assignments": assignments}, indent=2, sort_keys=True), "utf-8")


def merge_cluster_results(
    result_rows_path: Path,
    result_manifest_path: Path,
    *,
    local_ledger_db: Path | None = None,
) -> MergeReport:
    """Merge cluster results into the local ledger idempotently and fail-loud.

    Parameters
    ----------
    result_rows_path:
        JSONL rows exported by cluster workers.
    result_manifest_path:
        Manifest carrying per-assignment expected row counts and checksums.
    local_ledger_db:
        Local verification ledger path.

    Returns
    -------
    MergeReport
        Merge counts.

    Raises
    ------
    ClusterMergeConflict
        If an existing merge key or run ID maps to a different payload.
    ClusterResultIntegrityError
        If expected counts or checksums do not match the rows.
    """

    rows = load_result_rows_jsonl(result_rows_path)
    expected = _load_result_expectations(result_manifest_path)
    _verify_result_expectations(rows, expected)
    if not rows:
        raise ClusterResultIntegrityError("result rows must not be empty")
    campaign_ids = {row.campaign_id for row in rows}
    attempt_ids = {row.attempt_id for row in rows}
    if len(campaign_ids) != 1 or len(attempt_ids) != 1:
        raise ClusterResultIntegrityError("result rows must share one campaign and attempt")
    inserted = 0
    duplicates = 0
    run_id_collisions = 0
    with connect_ledger(_resolve_verification_db(local_ledger_db)) as conn:
        _initialize_cluster_merge_tables(conn)
        for row in rows:
            checksum = _verification_run_checksum(row.run)
            existing = _existing_merge_row(conn, row)
            if existing is not None:
                if existing["row_checksum"] != checksum:
                    raise ClusterMergeConflict(
                        f"conflicting payload for {row.campaign_id}/"
                        f"{row.attempt_id}/{row.assignment_id}/{row.run.run_id}"
                    )
                _assert_imported_run_present(conn, row.run, checksum)
                duplicates += 1
                continue
            if _verification_run_exists(conn, row.run.run_id):
                run_id_collisions += 1
                raise ClusterMergeConflict(
                    f"run_id {row.run.run_id} already exists without a matching cluster import"
                )
            append_verification_run(conn, row.run)
            conn.execute(
                """
                INSERT INTO cluster_result_imports(
                    campaign_id,
                    attempt_id,
                    assignment_id,
                    source_run_id,
                    row_checksum,
                    imported_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row.campaign_id,
                    row.attempt_id,
                    row.assignment_id,
                    row.run.run_id,
                    checksum,
                    utc_now(),
                ),
            )
            _assert_imported_run_present(conn, row.run, checksum)
            inserted += 1
    return MergeReport(
        campaign_id=next(iter(campaign_ids)),
        attempt_id=next(iter(attempt_ids)),
        inserted=inserted,
        duplicates=duplicates,
        assignments=len(expected),
        run_id_collisions=run_id_collisions,
    )


def collect_cluster_results(
    dispatch_result: DispatchResult,
    *,
    config: ClusterConfig | None = None,
    command_runner: CommandRunner = default_command_runner,
    local_result_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Collect and verify runner-host-stamped cluster result artifacts.

    Parameters
    ----------
    dispatch_result:
        Dispatch metadata returned by :func:`dispatch_giants`.
    config:
        Cluster defaults.
    command_runner:
        Injectable command runner for rsync.
    local_result_dir:
        Optional destination directory. Defaults under the dispatch artifact directory.
    dry_run:
        Skip rsync and only combine already-present local artifacts.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Aggregate result JSONL path and aggregate result-manifest path.
    """

    active_config = config or ClusterConfig()
    result_dir = local_result_dir or (dispatch_result.local_artifact_dir / "results")
    result_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        command_runner(
            (
                "rsync",
                "-az",
                f"{active_config.host}:{dispatch_result.remote_artifact_dir.rstrip('/')}/results/",
                str(result_dir).rstrip("/") + "/",
            )
        )
    rows = _load_and_verify_collected_results(result_dir)
    aggregate_rows = dispatch_result.local_artifact_dir / "cluster_results.jsonl"
    aggregate_manifest = dispatch_result.local_artifact_dir / "cluster_results.manifest.json"
    write_result_rows_jsonl(rows, aggregate_rows)
    write_result_manifest(rows, aggregate_manifest)
    return aggregate_rows, aggregate_manifest


@dataclass(frozen=True)
class CollectedClusterResults:
    """Per-model cluster collection outcome tolerant of partial array failures.

    A SLURM array job is a per-task contract, not an all-or-nothing one: when one
    task fails ``sbatch --wait`` returns non-zero for the whole array, but every
    other task may have validated and written an honest result artifact. This
    record separates the two so the caller attributes results PER-MODEL by each
    task's own outcome -- never by the array-job-level return code.

    Parameters
    ----------
    present_assignments:
        Assignments whose task wrote a valid, verified per-model result row.
    missing_assignments:
        Assignments whose task produced no valid result (failed, OOM'd, crashed,
        or never started). These get an honest per-model ``failed:*`` row.
    result_rows_path:
        Aggregate JSONL of the present result rows, or ``None`` when none were
        present (write inputs to :func:`merge_cluster_results` require >= 1 row).
    result_manifest_path:
        Aggregate manifest for the present result rows, or ``None`` when none were
        present.
    result_dir:
        Local directory the per-task result artifacts were collected into.
    log_dir:
        Local directory the per-task ``logs/`` artifacts were collected into.
        Used to read a failed task's ``.err`` log for its honest message.
    """

    present_assignments: tuple[ClusterAssignment, ...]
    missing_assignments: tuple[ClusterAssignment, ...]
    result_rows_path: Path | None
    result_manifest_path: Path | None
    result_dir: Path
    log_dir: Path


def collect_cluster_results_partial(
    dispatch_result: DispatchResult,
    *,
    config: ClusterConfig | None = None,
    command_runner: CommandRunner = default_command_runner,
    local_result_dir: Path | None = None,
    local_log_dir: Path | None = None,
    dry_run: bool = False,
) -> CollectedClusterResults:
    """Collect per-model cluster results, tolerating partial array failures.

    Unlike :func:`collect_cluster_results` (which fails loud when ANY expected
    artifact is absent), this collects whatever valid per-task result artifacts
    came back and reports the remainder as missing assignments. A partial array
    failure -- some tasks validated, some failed -- is the normal case this
    handles: the present tasks keep their real per-model status and the missing
    ones are surfaced honestly by the caller, NEVER cascaded from the array-job
    return code.

    Each present artifact is still verified individually against its own
    per-task manifest (count + checksum), so a present-but-corrupt result is
    treated as missing rather than silently trusted -- the tripwire stays armed
    for the rows it can see.

    Parameters
    ----------
    dispatch_result:
        Dispatch metadata returned by :func:`dispatch_giants`.
    config:
        Cluster defaults.
    command_runner:
        Injectable command runner for rsync.
    local_result_dir:
        Optional destination directory for per-task ``results/`` artifacts.
    local_log_dir:
        Optional destination directory for per-task ``logs/`` artifacts (the
        ``giant_<jobid>_<taskidx>.err`` files used to recover a failed task's
        honest message).
    dry_run:
        Skip rsync and only combine already-present local artifacts.

    Returns
    -------
    CollectedClusterResults
        Present (validated) and missing assignments with collection paths.
    """

    active_config = config or ClusterConfig()
    result_dir = local_result_dir or (dispatch_result.local_artifact_dir / "results")
    log_dir = local_log_dir or (dispatch_result.local_artifact_dir / "logs")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        remote = dispatch_result.remote_artifact_dir.rstrip("/")
        # rsync the results and logs trees independently. A missing remote logs
        # tree (e.g. nothing failed) must not abort result collection, so the
        # logs rsync is best-effort.
        command_runner(
            (
                "rsync",
                "-az",
                f"{active_config.host}:{remote}/results/",
                str(result_dir).rstrip("/") + "/",
            )
        )
        try:
            command_runner(
                (
                    "rsync",
                    "-az",
                    f"{active_config.host}:{remote}/logs/",
                    str(log_dir).rstrip("/") + "/",
                )
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Logs are only needed to enrich a failed task's message; their
            # absence never blocks attributing the present results.
            pass
    present_rows, present_assignment_ids = _load_present_collected_results(
        result_dir, dispatch_result.assignments
    )
    present_assignments = tuple(
        assignment
        for assignment in dispatch_result.assignments
        if assignment.assignment_id in present_assignment_ids
    )
    missing_assignments = tuple(
        assignment
        for assignment in dispatch_result.assignments
        if assignment.assignment_id not in present_assignment_ids
    )
    rows_path: Path | None = None
    manifest_path: Path | None = None
    if present_rows:
        rows_path = dispatch_result.local_artifact_dir / "cluster_results.jsonl"
        manifest_path = dispatch_result.local_artifact_dir / "cluster_results.manifest.json"
        write_result_rows_jsonl(present_rows, rows_path)
        write_result_manifest(present_rows, manifest_path)
    return CollectedClusterResults(
        present_assignments=present_assignments,
        missing_assignments=missing_assignments,
        result_rows_path=rows_path,
        result_manifest_path=manifest_path,
        result_dir=result_dir,
        log_dir=log_dir,
    )


def read_task_error_log(
    log_dir: Path,
    job_ids: Sequence[str],
    array_index: int,
    *,
    max_chars: int = 600,
) -> str | None:
    """Return the tail of a failed array task's ``.err`` log, if present.

    SLURM writes per-task error logs as ``giant_<jobid>_<taskidx>.err`` (from the
    ``#SBATCH --error`` directive). This recovers the task's OWN failure detail so
    a missing-result model surfaces its real error, not the generic batch message.

    Parameters
    ----------
    log_dir:
        Local directory the ``logs/`` tree was collected into.
    job_ids:
        Candidate SLURM array job IDs (``%A``).
    array_index:
        SLURM array task index (``%a``).
    max_chars:
        Maximum trailing characters to return.

    Returns
    -------
    str | None
        Compact non-empty error tail, or ``None`` when no log is available.
    """

    candidates: list[Path] = []
    for job_id in job_ids:
        candidates.append(log_dir / f"giant_{job_id}_{array_index}.err")
    # Fall back to any matching task index when the job ID is unknown/mismatched.
    candidates.extend(sorted(log_dir.glob(f"giant_*_{array_index}.err")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        compact = " ".join(text.splitlines()[-12:]).strip()
        if len(compact) > max_chars:
            compact = compact[-max_chars:]
        return compact or None
    return None


def _load_present_collected_results(
    result_dir: Path, assignments: Sequence[ClusterAssignment]
) -> tuple[tuple[ClusterResultRow, ...], set[str]]:
    """Load and verify whichever per-task result artifacts are present.

    Each ``<assignment_id>.manifest.json`` / ``.jsonl`` pair is loaded and
    verified against its own manifest independently. A pair that is absent,
    unreadable, or fails its own integrity check is treated as missing (the
    assignment is simply absent from the returned ID set) rather than aborting
    collection of the valid siblings.

    Parameters
    ----------
    result_dir:
        Directory containing per-task ``*.jsonl`` and ``*.manifest.json`` files.
    assignments:
        Dispatched assignments. Only artifacts whose assignment ID is among these
        are accepted, so a stray file cannot inject an unexpected row.

    Returns
    -------
    tuple[tuple[ClusterResultRow, ...], set[str]]
        Verified present result rows and the set of present assignment IDs.
    """

    known_ids = {assignment.assignment_id for assignment in assignments}
    present_rows: list[ClusterResultRow] = []
    present_ids: set[str] = set()
    for manifest_path in sorted(result_dir.glob("*.manifest.json")):
        rows_path = manifest_path.with_suffix("").with_suffix(".jsonl")
        if not rows_path.exists():
            continue
        try:
            rows = load_result_rows_jsonl(rows_path)
            expectations = _load_result_expectations(manifest_path)
            _verify_result_expectations(rows, expectations)
        except (OSError, json.JSONDecodeError, KeyError, ClusterResultIntegrityError):
            # A corrupt or self-inconsistent artifact is NOT trusted; the task is
            # left missing so the caller surfaces an honest per-model failure.
            continue
        artifact_ids = {row.assignment_id for row in rows}
        if not artifact_ids.issubset(known_ids):
            continue
        present_rows.extend(rows)
        present_ids.update(artifact_ids)
    return tuple(present_rows), present_ids


def record_host_contract(path: Path) -> dict[str, str]:
    """Record host facts needed to interpret cluster rows.

    Parameters
    ----------
    path:
        Destination JSON path.

    Returns
    -------
    dict[str, str]
        Recorded host contract.
    """

    contract = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "glibc": ".".join(platform.libc_ver()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cuda_driver": _optional_command_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    return contract


def compute_lock_hash_for_env(pixi_env: str, *, locks_dir: Path | None = None) -> str:
    """Return a SHA-256 hash for a committed pixi manifest and lock.

    Parameters
    ----------
    pixi_env:
        Lock prefix under ``menagerie/locks``.
    locks_dir:
        Optional locks directory.

    Returns
    -------
    str
        SHA-256 digest.
    """

    resolved_locks = locks_dir or Path(__file__).resolve().parent / "locks"
    manifest = resolved_locks / f"{pixi_env}.pixi.toml"
    lock = resolved_locks / f"{pixi_env}.pixi.lock"
    if not manifest.exists() or not lock.exists():
        raise FileNotFoundError(f"missing committed pixi lock inputs for {pixi_env!r}")
    digest = sha256()
    digest.update(manifest.read_bytes())
    digest.update(b"\0")
    digest.update(lock.read_bytes())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the cluster-runner command parser.

    Returns
    -------
    argparse.ArgumentParser
        CLI parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--assignment-manifest", type=Path, required=True)
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    worker.add_argument("--result-dir", type=Path, required=True)
    worker.add_argument("--verification-db", type=Path)
    lock_hash = subparsers.add_parser("lock-hash")
    lock_hash.add_argument("--pixi-env", default="cluster_giants")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cluster-runner CLI.

    Parameters
    ----------
    argv:
        Optional argument list.

    Returns
    -------
    int
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    if args.command == "worker":
        run_worker_assignment(
            args.assignment_manifest,
            args.task_index,
            repo_root=args.repo_root,
            result_dir=args.result_dir,
            verification_db=args.verification_db,
        )
        return 0
    if args.command == "lock-hash":
        print(compute_lock_hash_for_env(args.pixi_env))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


def _catalog_row_from_sql(row: Sequence[Any]) -> CatalogRow:
    """Build a catalog row from SQLite values."""

    return CatalogRow(
        model_id=int(row[0]),
        display_index=int(row[1]),
        stable_id=str(row[2]),
        name=str(row[3]),
        variant=str(row[4]),
        family=str(row[5]),
        family_normalized=str(row[6]),
        domain=str(row[7]),
        zoo=str(row[8]),
        constructor_call=str(row[9]),
        input_shape=str(row[10]),
        input_dtype=str(row[11]),
        era=str(row[12]),
        verified=bool(row[13]),
        notes=str(row[14]),
        source=str(row[15]),
        recipe_revision_sha256=str(row[16]),
        input_is_real=bool(row[17]),
        verification_expectation=str(row[18]),
        quarantine=bool(row[19]),
    )


def _row_value(row: CatalogRow | Mapping[str, object], field: str) -> str:
    """Return a string field from a catalog row or mapping."""

    if isinstance(row, Mapping):
        return str(row.get(field, ""))
    return str(getattr(row, field))


def _latest_status_and_peak(
    stable_id: str,
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None,
) -> tuple[str | None, int | None]:
    """Return latest ledger status and peak RSS for a stable ID."""

    if ledger is None:
        return None, None
    if isinstance(ledger, Mapping):
        peak = ledger.get(stable_id)
        return None, int(peak) if peak is not None else None
    if isinstance(ledger, Path):
        with connect_ledger(ledger) as conn:
            return _latest_status_and_peak(stable_id, conn)
    row = ledger.execute(
        """
        SELECT status, peak_rss_mb
        FROM current_verification
        WHERE stable_id = ?
        """,
        (stable_id,),
    ).fetchone()
    if row is None:
        return None, None
    return str(row["status"]), None if row["peak_rss_mb"] is None else int(row["peak_rss_mb"])


def _oom_run_count(
    stable_id: str,
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None,
) -> int:
    """Return the number of OOM ledger rows for a stable ID.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger:
        Ledger connection/path, peak-RSS mapping, or ``None``.

    Returns
    -------
    int
        Count of OOM verification rows for the stable ID.
    """

    if ledger is None or isinstance(ledger, Mapping):
        return 0
    if isinstance(ledger, Path):
        with connect_ledger(ledger) as conn:
            return _oom_run_count(stable_id, conn)
    row = ledger.execute(
        """
        SELECT COUNT(*)
        FROM verification_runs
        WHERE stable_id = ?
          AND status = 'oom'
        """,
        (stable_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _max_measured_peak_mb(
    stable_id: str,
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None,
    *,
    latest_peak_mb: int | None = None,
) -> int | None:
    """Return the largest measured peak RSS (MB) for a stable ID, any host.

    A measured peak is HARD non-fit evidence regardless of where it was measured:
    a cluster-measured 228 GiB peak proves the model cannot fit in local RAM just
    as well as a local measurement would. The max over ALL historical runs is used
    so that a later, smaller cluster-validated peak (the latest terminal row) does
    not erase the earlier proof that the model is too large for the workstation.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger:
        Ledger connection/path, peak-RSS mapping, or ``None``.
    latest_peak_mb:
        Already-known latest peak (from :func:`_latest_status_and_peak`), used as
        a floor and as the only source for the mapping/None ledger forms.

    Returns
    -------
    int | None
        Largest measured peak RSS in MB, or ``None`` when never measured.
    """

    if ledger is None or isinstance(ledger, Mapping):
        return latest_peak_mb
    if isinstance(ledger, Path):
        with connect_ledger(ledger) as conn:
            return _max_measured_peak_mb(stable_id, conn, latest_peak_mb=latest_peak_mb)
    row = ledger.execute(
        """
        SELECT MAX(peak_rss_mb) AS max_peak
        FROM verification_runs
        WHERE stable_id = ?
          AND peak_rss_mb IS NOT NULL
        """,
        (stable_id,),
    ).fetchone()
    history_peak = None if row is None or row["max_peak"] is None else int(row["max_peak"])
    candidates = [value for value in (latest_peak_mb, history_peak) if value is not None]
    return max(candidates) if candidates else None


def _had_local_ram_failure(
    stable_id: str,
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None,
) -> bool:
    """Return whether a stable ID RAM-failed on a prior LOCAL attempt.

    A LOCAL RAM failure is HARD evidence the model cannot run on the workstation
    and must escalate to the cluster. It is either:

    * a LOCAL run with an OOM / resource-kill status, or
    * a LOCAL ``failed:memory_cap`` run whose worker memory cap was NEAR full local
      RAM (>= :data:`LOCAL_RAM_FAILURE_CAP_FLOOR_GB`). A memory-cap kill at a SMALL
      protective cap (e.g. an early-sweep 30 GiB cap) is NOT evidence the model
      needs >115 GiB, so it must NOT escalate -- that was the original
      over-routing trap.

    "Local" means a run whose ``runner_host`` equals ``socket.gethostname()`` --
    this workstation (the shared cluster nodes use distinct ``ax*`` hostnames).
    Mapping / ``None`` ledger forms carry no per-run host or status detail and so
    report no failure.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger:
        Ledger connection/path, peak-RSS mapping, or ``None``.

    Returns
    -------
    bool
        Whether a qualifying local RAM failure exists.
    """

    if ledger is None or isinstance(ledger, Mapping):
        return False
    if isinstance(ledger, Path):
        with connect_ledger(ledger) as conn:
            return _had_local_ram_failure(stable_id, conn)
    local_host = socket.gethostname()
    rows = ledger.execute(
        """
        SELECT status, error_class, error_message
        FROM verification_runs
        WHERE stable_id = ?
          AND runner_host = ?
        """,
        (stable_id, local_host),
    ).fetchall()
    cap_floor_gb = LOCAL_RAM_FAILURE_CAP_FLOOR_GB
    for row in rows:
        status = str(row["status"]) if row["status"] is not None else ""
        if status in LOCAL_RAM_FAILURE_STATUSES:
            return True
        error_class = str(row["error_class"]) if row["error_class"] is not None else ""
        if error_class == "failed:memory_cap":
            cap_gb = _parse_worker_cap_gb(row["error_message"])
            if cap_gb is not None and cap_gb >= cap_floor_gb:
                return True
    return False


def _parse_worker_cap_gb(error_message: object) -> float | None:
    """Return the worker-memory-cap GiB embedded in a memory-cap message.

    Parameters
    ----------
    error_message:
        Verification ``error_message`` text, possibly ``None``.

    Returns
    -------
    float | None
        Parsed ``--worker-memory-cap-gb`` value, or ``None`` when absent.
    """

    if not error_message:
        return None
    match = re.search(r"worker-memory-cap-gb=([\d.]+)", str(error_message))
    return float(match.group(1)) if match else None


def _matches_first_contact_heuristic(row: CatalogRow | Mapping[str, object]) -> bool:
    """Return whether row metadata is too risky for local first contact.

    NOTE: as of the LOCAL-FIRST routing policy this is NO LONGER part of the
    cluster-routing DECISION in :func:`is_giant` -- size estimates over-route a
    shared cluster. It is retained only as a conservative tier-sizing input for
    models already routed to the cluster (see :func:`node_tier_for_row`).
    """

    haystack = " ".join(
        _row_value(row, field)
        for field in ("name", "family", "family_normalized", "domain", "zoo", "notes")
    ).casefold()
    if any(pattern in haystack for pattern in GIANT_HEURISTIC_PATTERNS):
        return True
    if _param_count_is_giant(haystack):
        return True
    return _input_shape_is_large(_row_value(row, "input_shape"))


def _looks_like_unregistered_moe_monster(row: CatalogRow | Mapping[str, object]) -> bool:
    """Return whether an unregistered row looks like a large MoE family.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping.

    Returns
    -------
    bool
        Whether metadata names a mixture-of-experts family.
    """

    haystack = " ".join(
        _row_value(row, field)
        for field in ("name", "family", "family_normalized", "domain", "zoo", "notes")
    ).casefold()
    return any(
        pattern in haystack for pattern in ("moe", "mixture-of-experts", "mixture of experts")
    )


def _param_count_is_giant(haystack: str) -> bool:
    """Return whether text contains a giant model-size marker."""

    for value, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*(b|bn|billion|m|mm)\b", haystack):
        count = float(value)
        if suffix in {"b", "bn", "billion"} and count >= 1.0:
            return True
        if suffix in {"m", "mm"} and count >= 400.0:
            return True
    return False


def _input_shape_is_large(input_shape: str) -> bool:
    """Return whether an input-shape string suggests unsafe local first contact."""

    dims = [int(value) for value in re.findall(r"\d+", input_shape)]
    if len(dims) < 3:
        return False
    product = 1
    for dim in dims:
        product *= max(1, dim)
    return max(dims) >= 512 and product >= 512 * 512 * 3


def _heuristic_peak_mb(row: CatalogRow | Mapping[str, object]) -> int:
    """Return a conservative first-contact peak estimate."""

    haystack = " ".join(
        _row_value(row, field) for field in ("name", "family", "family_normalized", "notes")
    ).casefold()
    if "moe" in haystack or "longcat" in haystack or "deepseek" in haystack:
        return 360 * MB_PER_GB
    if _param_count_is_giant(haystack):
        return 220 * MB_PER_GB
    if _input_shape_is_large(_row_value(row, "input_shape")):
        return 180 * MB_PER_GB
    return 160 * MB_PER_GB


def _tier_from_entry(entry: GiantRegistryEntry, config: ClusterConfig) -> NodeTier:
    """Return a tier matching a static registry entry."""

    partition = entry.partition
    worker_cap = entry.worker_memory_cap_gb
    for tier in config.node_tiers:
        if tier.mem_gb >= entry.node_mem_gb:
            return NodeTier(
                mem_gb=entry.node_mem_gb,
                worker_memory_cap_gb=worker_cap or max(1, entry.node_mem_gb - 10),
                partition=partition or tier.partition,
                max_peak_rss_gb=tier.max_peak_rss_gb,
            )
    tier = config.node_tiers[-1]
    return NodeTier(
        mem_gb=tier.mem_gb,
        worker_memory_cap_gb=worker_cap or tier.worker_memory_cap_gb,
        partition=partition or tier.partition,
        max_peak_rss_gb=tier.max_peak_rss_gb,
    )


def _largest_tier(config: ClusterConfig) -> NodeTier:
    """Return the largest configured memory tier.

    Parameters
    ----------
    config:
        Cluster defaults.

    Returns
    -------
    NodeTier
        Configured tier with the largest requested memory.
    """

    return max(config.node_tiers, key=lambda tier: tier.mem_gb)


def _laddered_escalation_tier(
    row: CatalogRow | Mapping[str, object],
    stable_id: str,
    ledger: sqlite3.Connection | Path | Mapping[str, int] | None,
    config: ClusterConfig,
) -> NodeTier:
    """Right-size a cluster node for a model that RAM-failed locally with no peak.

    Scales by the conservative trait heuristic (:func:`_heuristic_peak_mb`, 160-360
    GiB) to the SMALLEST fitting tier, then steps up one tier per *additional* prior
    OOM so a genuinely-larger model ratchets up the ladder instead of re-OOM-ing on
    the same tier. Reaches the largest tier only after exhausting the smaller ones --
    it never jumps straight to max.

    Parameters
    ----------
    row:
        Catalog row or row-like mapping (for the trait heuristic).
    stable_id:
        Durable model identity (for the OOM-count ladder step).
    ledger:
        Ledger connection/path, peak-RSS mapping, or ``None``.
    config:
        Cluster defaults (supplies the node-tier ladder).

    Returns
    -------
    NodeTier
        The laddered, right-sized tier (never larger than necessary).
    """

    tiers = config.node_tiers
    heuristic_gb = max(1, (_heuristic_peak_mb(row) + MB_PER_GB - 1) // MB_PER_GB)
    base_index = next(
        (index for index, tier in enumerate(tiers) if heuristic_gb <= tier.max_peak_rss_gb),
        len(tiers) - 1,
    )
    extra_steps = max(0, _oom_run_count(stable_id, ledger) - 1)
    return tiers[min(base_index + extra_steps, len(tiers) - 1)]


def _assignment_for_row(
    *,
    row: CatalogRow,
    index: int,
    ledger_db: Path,
    config: ClusterConfig,
    campaign_id: str,
    attempt_id: str,
    timeout_sec: float = 14400.0,
    input_scale: float = 1.0,
) -> ClusterAssignment:
    """Build one cluster assignment for a row."""

    tier = node_tier_for_row(row, ledger=ledger_db, config=config)
    entry = GIANT_REGISTRY.get(row.stable_id)
    return ClusterAssignment(
        campaign_id=campaign_id,
        attempt_id=attempt_id,
        assignment_id=f"{campaign_id}:{attempt_id}:{index}:{row.stable_id}",
        stable_id=row.stable_id,
        array_index=index,
        node_mem_gb=tier.mem_gb,
        worker_memory_cap_gb=tier.worker_memory_cap_gb,
        partition=tier.partition,
        reason=(
            REQUIRES_CUDA[row.stable_id]
            if row.stable_id in REQUIRES_CUDA
            else entry.reason
            if entry is not None
            else "cold-start heuristic"
        ),
        timeout_sec=timeout_sec,
        input_scale=input_scale,
        gpu=tier.gpu,
    )


def _copy_catalog_snapshot(source: Path, destination: Path) -> None:
    """Copy an existing catalog database snapshot."""

    if not source.exists():
        raise FileNotFoundError(f"catalog database must already exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _dispatch_commands(
    *,
    repo_root: Path,
    artifact_dir: Path,
    remote_artifact_dir: str,
    config: ClusterConfig,
    sbatch_paths: Sequence[Path],
    wait: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """Return rsync, mkdir, and sbatch commands for dispatch.

    Parameters
    ----------
    repo_root:
        Local repository root.
    artifact_dir:
        Local artifact directory.
    remote_artifact_dir:
        Remote artifact directory.
    config:
        Cluster defaults.
    sbatch_paths:
        Local sbatch scripts to submit.
    wait:
        Whether to include ``--wait`` in sbatch submission commands.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        Commands in execution order.
    """

    remote = f"{config.host}:{config.remote_repo.rstrip('/')}/"
    artifact_remote = f"{config.host}:{remote_artifact_dir.rstrip('/')}/"
    pixi_bin_dir = config.remote_pixi_bin.rsplit("/", 1)[0]
    setup_commands = (
        (
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".research",
            "--exclude",
            "__pycache__",
            "--exclude",
            "menagerie/data/verification.db*",
            str(repo_root).rstrip("/") + "/",
            remote,
        ),
        ("ssh", config.host, f"mkdir -p {remote_artifact_dir}/logs {remote_artifact_dir}/results"),
        ("rsync", "-az", str(artifact_dir).rstrip("/") + "/", artifact_remote),
        # The cluster nodes do not ship pixi on PATH, so stage the local pixi
        # binary to a known remote path and invoke it by absolute path in the
        # sbatch script. -p preserves the executable bit.
        ("ssh", config.host, f"mkdir -p {pixi_bin_dir}"),
        (
            "rsync",
            "-az",
            str(_local_pixi_bin()),
            f"{config.host}:{config.remote_pixi_bin}",
        ),
    )
    sbatch_prefix = "sbatch --wait" if wait else "sbatch"
    sbatch_commands = tuple(
        ("ssh", config.host, f"{sbatch_prefix} {remote_artifact_dir}/{sbatch_path.name}")
        for sbatch_path in sbatch_paths
    )
    return (*setup_commands, *sbatch_commands)


def _write_sbatch_scripts(
    assignments: Sequence[ClusterAssignment],
    *,
    artifact_dir: Path,
    config: ClusterConfig,
    remote_artifact_dir: str,
    verification_db: Path | None = None,
    remote_home: str | None = None,
) -> tuple[Path, ...]:
    """Write one sbatch script per memory/partition tier.

    Parameters
    ----------
    assignments:
        Cluster assignments.
    artifact_dir:
        Local artifact directory.
    config:
        Cluster defaults.
    remote_artifact_dir:
        Remote artifact directory.
    verification_db:
        Verification ledger path to export in each sbatch script.
    remote_home:
        Absolute remote home used to expand ``~`` in SLURM ``#SBATCH`` log
        directives.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Written sbatch script paths.
    """

    groups: dict[tuple[str, int, bool], list[ClusterAssignment]] = defaultdict(list)
    for assignment in assignments:
        groups[(assignment.partition, assignment.node_mem_gb, assignment.gpu)].append(assignment)
    paths: list[Path] = []
    for (partition, mem_gb, gpu), group in sorted(groups.items(), key=lambda item: item[0]):
        suffix = "_gpu" if gpu else ""
        path = artifact_dir / f"cluster_runner_{partition}_{mem_gb}g{suffix}.sbatch"
        path.write_text(
            render_sbatch_script(
                group,
                config=config,
                remote_artifact_dir=remote_artifact_dir,
                verification_db=verification_db,
                remote_home=remote_home,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)


def _parse_sbatch_job_id(stdout: str) -> str | None:
    """Parse an sbatch job ID from command output."""

    match = re.search(r"Submitted batch job\s+(\S+)", stdout)
    return match.group(1) if match else None


def _row_get(row: sqlite3.Row, column: str) -> object:
    """Return a column from a SQLite row, or ``None`` when the column is absent.

    Defensive accessor for forward/backward-compatible reads of optional columns
    (e.g. the nullable machine/hardware fields, which a legacy or partial row may
    lack). A real ``NULL`` and a missing column both read as ``None``.

    Parameters
    ----------
    row:
        SQLite row.
    column:
        Column name.

    Returns
    -------
    object
        Column value, or ``None`` when the column is not present on the row.
    """

    return row[column] if column in row.keys() else None


def _verification_run_from_row(row: sqlite3.Row) -> VerificationRun:
    """Build a verification run from a SQLite row."""

    return VerificationRun(
        stable_id=str(row["stable_id"]),
        recipe_revision_sha256=str(row["recipe_revision_sha256"]),
        name=str(row["name"]),
        zoo=str(row["zoo"]),
        variant=str(row["variant"]),
        scope=row["scope"],
        status=row["status"],
        forward_pass=row["forward_pass"],
        backward_pass=row["backward_pass"],
        backward_na_reason=row["backward_na_reason"],
        metadata_ok=row["metadata_ok"],
        n_ops=row["n_ops"],
        graph_shape_hash=row["graph_shape_hash"],
        svg_sha256=row["svg_sha256"],
        torchlens_version=str(row["torchlens_version"]),
        torch_version=str(row["torch_version"]),
        python_version=str(row["python_version"]),
        device_requested=str(row["device_requested"]),
        device_actual=row["device_actual"],
        env_hash=row["env_hash"],
        lock_hash=str(row["lock_hash"]),
        torchlens_source_hash=str(row["torchlens_source_hash"]),
        input_scale=row["input_scale"],
        runner_host=row["runner_host"],
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
        duration_sec=float(row["duration_sec"]),
        peak_rss_mb=row["peak_rss_mb"],
        error_class=row["error_class"],
        error_message=row["error_message"],
        run_id=str(row["run_id"]),
        machine_cpu_model=_row_get(row, "machine_cpu_model"),  # type: ignore[arg-type]
        machine_cpu_cores_physical=_row_get(row, "machine_cpu_cores_physical"),  # type: ignore[arg-type]
        machine_cpu_cores_logical=_row_get(row, "machine_cpu_cores_logical"),  # type: ignore[arg-type]
        machine_total_ram_gb=_row_get(row, "machine_total_ram_gb"),  # type: ignore[arg-type]
        machine_gpu_models=_row_get(row, "machine_gpu_models"),  # type: ignore[arg-type]
        machine_gpu_count=_row_get(row, "machine_gpu_count"),  # type: ignore[arg-type]
        machine_platform=_row_get(row, "machine_platform"),  # type: ignore[arg-type]
        machine_torch_num_threads=_row_get(row, "machine_torch_num_threads"),  # type: ignore[arg-type]
    )


def _cluster_result_payload(row: ClusterResultRow) -> dict[str, object]:
    """Return a JSON-compatible cluster result payload."""

    return {
        "campaign_id": row.campaign_id,
        "attempt_id": row.attempt_id,
        "assignment_id": row.assignment_id,
        "run": asdict(row.run),
    }


def _verification_run_checksum(run: VerificationRun) -> str:
    """Return a stable checksum for a verification run payload."""

    payload = json.dumps(asdict(run), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _assignment_result_checksum(rows: Sequence[ClusterResultRow]) -> str:
    """Return a stable checksum for all rows in one assignment."""

    digest = sha256()
    for row in sorted(rows, key=lambda item: item.run.run_id):
        digest.update(_verification_run_checksum(row.run).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_result_expectations(path: Path) -> dict[tuple[str, str, str], tuple[int, str]]:
    """Load expected result counts and checksums."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expectations: dict[tuple[str, str, str], tuple[int, str]] = {}
    for item in payload["assignments"]:
        key = (str(item["campaign_id"]), str(item["attempt_id"]), str(item["assignment_id"]))
        expectations[key] = (int(item["expected_row_count"]), str(item["result_checksum"]))
    return expectations


def _verify_result_expectations(
    rows: Sequence[ClusterResultRow],
    expectations: Mapping[tuple[str, str, str], tuple[int, str]],
) -> None:
    """Verify result rows against expected counts and checksums."""

    grouped: dict[tuple[str, str, str], list[ClusterResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.campaign_id, row.attempt_id, row.assignment_id)].append(row)
    if set(grouped) != set(expectations):
        raise ClusterResultIntegrityError(
            f"result assignment keys {sorted(grouped)} do not match manifest {sorted(expectations)}"
        )
    for key, group in grouped.items():
        expected_count, expected_checksum = expectations[key]
        actual_checksum = _assignment_result_checksum(group)
        if len(group) != expected_count:
            raise ClusterResultIntegrityError(
                f"{key} expected {expected_count} rows, got {len(group)}"
            )
        if actual_checksum != expected_checksum:
            raise ClusterResultIntegrityError(f"{key} checksum mismatch")


def _load_and_verify_collected_results(result_dir: Path) -> tuple[ClusterResultRow, ...]:
    """Load and verify per-task result artifacts from a directory.

    Parameters
    ----------
    result_dir:
        Directory containing per-task ``*.jsonl`` and ``*.manifest.json`` files.

    Returns
    -------
    tuple[ClusterResultRow, ...]
        Verified result rows.
    """

    all_rows: list[ClusterResultRow] = []
    manifest_paths = sorted(result_dir.glob("*.manifest.json"))
    if not manifest_paths:
        raise ClusterResultIntegrityError(f"no result manifests found in {result_dir}")
    for manifest_path in manifest_paths:
        rows_path = manifest_path.with_suffix("").with_suffix(".jsonl")
        if not rows_path.exists():
            raise ClusterResultIntegrityError(f"missing rows for manifest {manifest_path}")
        rows = load_result_rows_jsonl(rows_path)
        expectations = _load_result_expectations(manifest_path)
        _verify_result_expectations(rows, expectations)
        all_rows.extend(rows)
    if not all_rows:
        raise ClusterResultIntegrityError(f"no result rows found in {result_dir}")
    return tuple(all_rows)


def _initialize_cluster_merge_tables(conn: sqlite3.Connection) -> None:
    """Create cluster merge idempotency tables."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_result_imports(
            campaign_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            assignment_id TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            row_checksum TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            PRIMARY KEY(campaign_id, attempt_id, assignment_id, source_run_id)
        )
        """
    )


def _existing_merge_row(conn: sqlite3.Connection, row: ClusterResultRow) -> sqlite3.Row | None:
    """Return an existing merge row for a cluster result key."""

    return conn.execute(
        """
        SELECT row_checksum
        FROM cluster_result_imports
        WHERE campaign_id = ?
          AND attempt_id = ?
          AND assignment_id = ?
          AND source_run_id = ?
        """,
        (row.campaign_id, row.attempt_id, row.assignment_id, row.run.run_id),
    ).fetchone()


def _verification_run_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    """Return whether a verification run ID already exists.

    Parameters
    ----------
    conn:
        SQLite connection.
    run_id:
        Verification run ID.

    Returns
    -------
    bool
        Whether the run ID is present in the ledger.
    """

    row = conn.execute("SELECT 1 FROM verification_runs WHERE run_id = ?", (run_id,)).fetchone()
    return row is not None


def _assert_imported_run_present(
    conn: sqlite3.Connection, run: VerificationRun, checksum: str
) -> None:
    """Assert an imported cluster assignment has its ledger row.

    Parameters
    ----------
    conn:
        SQLite connection.
    run:
        Imported verification run payload.
    checksum:
        Expected run checksum.

    Raises
    ------
    ClusterResultIntegrityError
        If the import marker exists without a ledger row.
    ClusterMergeConflict
        If the ledger row has a different payload.
    """

    existing = conn.execute(
        "SELECT * FROM verification_runs WHERE run_id = ?", (run.run_id,)
    ).fetchone()
    if existing is None:
        raise ClusterResultIntegrityError(
            f"cluster import for {run.run_id} has no verification ledger row"
        )
    existing_checksum = _verification_run_checksum(_verification_run_from_row(existing))
    if existing_checksum != checksum:
        raise ClusterMergeConflict(f"run_id {run.run_id} already exists with different payload")


def _run_cluster_command(
    command: Sequence[str],
    command_runner: CommandRunner,
    *,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    """Run one cluster transport command with a timeout when supported.

    Parameters
    ----------
    command:
        Command arguments.
    command_runner:
        Injectable command runner.
    timeout:
        Optional timeout in seconds for the default runner.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed command.
    """

    if command_runner is default_command_runner:
        return default_command_runner(command, timeout=timeout)
    return command_runner(command)


def _run_sbatch_command(
    command: Sequence[str],
    command_runner: CommandRunner,
    *,
    timeout: float | None,
    wait: bool = True,
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    """Run a blocking ``sbatch --wait`` command and classify its outcome.

    A non-zero return code from ``sbatch --wait`` is ambiguous: it can mean the
    submission itself was rejected (a genuine transport/submit failure) or that
    the job was accepted, ran, and then failed. The two are disambiguated by
    whether sbatch printed a ``Submitted batch job N`` line: if it did, the job
    reached the cluster and ran, so the non-zero status is an honest job/
    validation failure (``ClusterJobFailed``), NEVER a benign cluster-unavailable
    skip. If it did not, the original transport error is re-raised so the caller
    records a legitimate cluster-unavailable row.

    Parameters
    ----------
    command:
        sbatch transport command.
    command_runner:
        Injectable command runner.
    timeout:
        Optional timeout in seconds for the default runner.
    wait:
        Whether the command is a blocking ``sbatch --wait`` submission.

    Returns
    -------
    tuple[subprocess.CompletedProcess[str], str | None]
        The completed process and the parsed SLURM job ID (or ``None``).

    Raises
    ------
    ClusterJobFailed
        If sbatch accepted the job (a job ID was printed) but it exited non-zero.
    subprocess.CalledProcessError
        If the submission itself failed before a job ID was assigned.
    """

    try:
        result = _run_cluster_command(command, command_runner, timeout=timeout)
    except subprocess.CalledProcessError as error:
        stdout = error.stdout or ""
        job_id = _parse_sbatch_job_id(stdout)
        if job_id is not None:
            stderr = (error.stderr or "").strip()
            tail = stderr or stdout.strip() or repr(error.cmd)
            label = "sbatch --wait" if wait else "sbatch"
            raise ClusterJobFailed(
                (job_id,),
                f"{label} returncode={error.returncode}; {tail}",
            ) from error
        raise
    return result, _parse_sbatch_job_id(result.stdout)


def _optional_command_output(command: Sequence[str]) -> str:
    """Return command output or an empty string when unavailable."""

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


if __name__ == "__main__":
    raise SystemExit(main())
