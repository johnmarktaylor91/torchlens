"""One result format for every capture benchmark suite.

Every runner in this package writes a single JSON document of the shape built by
:func:`new_document`: the configuration, the environment, package versions, the
measurement protocol, and a list of cells. A cell is one (suite, case, arm, subject)
measurement. Cells that could not run keep their row with a non-``ok`` status and a
reason, so unsupported and refused cells stay visible in every table generated from
the JSON. :func:`validate_document` is the contract the table and chart generators
rely on.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.host_label import benchmark_host_label  # noqa: E402

SCHEMA_ID = "torchlens.bench.capture.v1"
SUITES = ("capture", "A", "B", "fixtures", "panel", "census", "aa")
STATUSES = ("ok", "unsupported", "refused", "error", "timeout", "noisy", "mismatch")
NON_OK_NEEDS_REASON = frozenset(STATUSES) - {"ok"}
GIT_SHA_ENV = "TL_BENCH_GIT_SHA"
PEER_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "nnsight",
    "transformer_lens",
    "thingsvision",
    "captum",
    "numpy",
    "psutil",
)
PROTOCOL_DEFAULTS = {"warmups": 5, "samples": 50, "processes": 3, "fresh_subprocess": True}
REQUIRED_CELL_KEYS = ("cell_id", "suite", "case", "arm", "subject", "status")


def package_versions(packages: tuple[str, ...] = PEER_PACKAGES) -> dict[str, str | None]:
    """Return installed versions for the benchmark's packages (``None`` when absent).

    Parameters
    ----------
    packages:
        Distribution names to look up.

    Returns
    -------
    dict[str, str | None]
        Version string per package, ``None`` for a package that is not installed.
    """

    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name.replace("_", "-"))
        except metadata.PackageNotFoundError:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    try:
        versions["torchlens"] = metadata.version("torchlens")
    except metadata.PackageNotFoundError:
        versions["torchlens"] = None
    versions["git_sha"] = os.environ.get(GIT_SHA_ENV) or None
    return versions


def environment_snapshot(device: str = "cpu") -> dict[str, Any]:
    """Describe the machine and runtime with a neutral host label.

    Parameters
    ----------
    device:
        Benchmark device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict[str, Any]
        Host label, platform, CPU and thread facts, load average, and the device name.
    """

    import torch

    try:
        load_1m: float | None = os.getloadavg()[0]
    except OSError:
        load_1m = None
    gpu = None
    if device == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "hostname": benchmark_host_label(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "load_average_1m": load_1m,
        "device": device,
        "gpu_name": gpu,
    }


def new_document(
    suite: str,
    config: dict[str, Any],
    *,
    device: str = "cpu",
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a result document for one suite run.

    Parameters
    ----------
    suite:
        One of :data:`SUITES`.
    config:
        The run's configuration (command-line options, selected cases and arms).
    device:
        Benchmark device.
    protocol:
        Measurement protocol overrides on top of :data:`PROTOCOL_DEFAULTS`.

    Returns
    -------
    dict[str, Any]
        A document with an empty ``cells`` list.
    """

    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {SUITES}")
    return {
        "schema": SCHEMA_ID,
        "suite": suite,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": dict(config),
        "environment": environment_snapshot(device),
        "versions": package_versions(),
        "protocol": {**PROTOCOL_DEFAULTS, **(protocol or {})},
        "aa_noise": None,
        "cells": [],
    }


def make_cell(
    suite: str,
    case: str,
    arm: str,
    subject: str,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    """Build one cell row.

    Parameters
    ----------
    suite, case, arm, subject:
        The cell's coordinates; ``subject`` names the model or fixture.
    status:
        One of :data:`STATUSES`. Every status but ``ok`` needs a ``reason`` field.
    **fields:
        Further cell data: ``reason``, ``keeps``, ``contract``, ``timing``, ``memory``,
        ``exact``, ``counts`` and so on.

    Returns
    -------
    dict[str, Any]
        The cell row.
    """

    cell = {
        "cell_id": f"{suite}/{case}/{arm}/{subject}",
        "suite": suite,
        "case": case,
        "arm": arm,
        "subject": subject,
        "status": status,
    }
    cell.update(fields)
    return cell


def _cell_problems(index: int, cell: Any) -> list[str]:
    """Return the contract violations of one cell."""

    if not isinstance(cell, dict):
        return [f"cell {index} is not an object"]
    problems = [f"cell {index} lacks {key!r}" for key in REQUIRED_CELL_KEYS if key not in cell]
    status = cell.get("status")
    if status not in STATUSES:
        problems.append(f"cell {index} has unknown status {status!r}")
    elif status in NON_OK_NEEDS_REASON and not cell.get("reason"):
        problems.append(f"cell {index} ({cell.get('cell_id')}) is {status} without a reason")
    if cell.get("suite") == "B" and status == "ok" and not cell.get("keeps"):
        problems.append(f"cell {index} ({cell.get('cell_id')}) is a suite B row without 'keeps'")
    return problems


def validate_document(doc: Any) -> list[str]:
    """Check a result document against the format contract.

    Parameters
    ----------
    doc:
        A parsed result document.

    Returns
    -------
    list[str]
        Human-readable violations; empty when the document conforms.
    """

    if not isinstance(doc, dict):
        return ["document is not an object"]
    problems: list[str] = []
    if doc.get("schema") != SCHEMA_ID:
        problems.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA_ID!r}")
    if doc.get("suite") not in SUITES:
        problems.append(f"unknown suite {doc.get('suite')!r}")
    for key in ("config", "environment", "versions", "protocol"):
        if not isinstance(doc.get(key), dict):
            problems.append(f"missing object {key!r}")
    cells = doc.get("cells")
    if not isinstance(cells, list):
        return [*problems, "missing list 'cells'"]
    seen: set[str] = set()
    for index, cell in enumerate(cells):
        problems.extend(_cell_problems(index, cell))
        if isinstance(cell, dict):
            key = f"{cell.get('cell_id')}#{cell.get('process_index', 0)}"
            if key in seen:
                problems.append(f"duplicate cell {key}")
            seen.add(key)
    return problems
