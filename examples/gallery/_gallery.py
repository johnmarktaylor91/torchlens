"""Shared pieces of the Reproduction Gallery entries: the stamp, the report and line counts.

Every entry script in this folder states one cited claim, measures it with public
TorchLens calls, checks it against oracles fixed before the first run, runs controls that
must fail, saves a portable artifact, reloads it and replays one measurement from the
reloaded copy. The entry passes only when every oracle passes, every control behaves as
declared and the replay from the reloaded artifact matches; otherwise the script prints
``VERDICT: did not pass`` and the entry moves to the next batch. Rules are never moved to
make an entry pass.

The summary every entry writes (``<entry>.json`` in ``--out``) holds the claim, the pinned
inputs, each oracle and control with its rule, measurement and outcome, the artifact and
its replay, the basis-grid row (core verbs used and line counts) and the stamp: the exact
software, machine and inputs the verdict was measured on.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import torchlens as tl

PASSED = "passed"
NOT_PASSED = "did not pass"

#: The basis grid's columns: the core verbs a Gallery entry can use.
VERBS = ("capture", "select", "read", "do", "compare", "validate", "save")

REGION_START = "# region: "
REGION_END = "# endregion"


@dataclass
class Check:
    """One oracle or control: its rule, what was measured and whether the rule held.

    Parameters
    ----------
    rule:
        The rule, fixed before the first run.
    measured:
        The measured numbers (JSON-serialisable).
    passed:
        Whether the rule held. For a control this means the control behaved as declared
        (the wrong solution failed, the shuffled side dropped, the refusal fired).
    """

    rule: str
    measured: Any
    passed: bool


@dataclass
class Entry:
    """Everything an entry reports; ``report`` prints it and writes the JSON summary.

    Parameters
    ----------
    name:
        Entry slug (the script's file stem).
    claim:
        The cited claim, with its source and version.
    pinned:
        Pinned inputs: model ids and revisions, data hashes, seeds, dtypes.
    oracles:
        Oracle name to ``Check``.
    controls:
        Control name to ``Check``.
    verbs:
        The basis-grid verbs the entry's TorchLens analysis uses (a subset of ``VERBS``).
    architectures:
        Architectures the entry ran on in this run.
    measured:
        Further reported, non-gating numbers.
    artifact:
        The saved artifact and its replay (see ``artifact_record``).
    lines:
        Basis-grid line counts (see ``count_region_lines``).
    """

    name: str
    claim: str
    pinned: dict[str, Any]
    oracles: dict[str, Check] = field(default_factory=dict)
    controls: dict[str, Check] = field(default_factory=dict)
    verbs: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    measured: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)
    lines: dict[str, Any] = field(default_factory=dict)

    def verdict(self) -> str:
        """Return ``PASSED`` when every oracle and control held and the replay matched.

        Returns
        -------
        str
            ``PASSED`` or ``NOT_PASSED``.
        """

        checks = [*self.oracles.values(), *self.controls.values()]
        replay_ok = bool(self.artifact.get("replay", {}).get("matched", False))
        ok = bool(checks) and all(check.passed for check in checks) and replay_ok
        return PASSED if ok else NOT_PASSED


def git_sha() -> str:
    """Return the checkout's commit, ``GALLERY_GIT_SHA`` when set, else ``"unknown"``.

    Returns
    -------
    str
        A commit sha or ``"unknown"``.
    """

    if os.environ.get("GALLERY_GIT_SHA"):
        return os.environ["GALLERY_GIT_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def stamp(device: str, started: float) -> dict[str, Any]:
    """The software and machine the verdict was measured on.

    Parameters
    ----------
    device:
        The torch device the entry ran on.
    started:
        ``time.time()`` at the start of the run.

    Returns
    -------
    dict
        TorchLens version and commit, torch, python, platform, device, threads, wall time.
    """

    versions: dict[str, str] = {}
    for distribution in ("transformers", "torchvision", "rsatoolbox", "numpy"):
        try:  # installed metadata: some packages (rsatoolbox 0.3.2) have no __version__
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    machine = platform.processor() or platform.machine()
    if device.startswith("cuda") and torch.cuda.is_available():
        machine = torch.cuda.get_device_name(torch.device(device))
    return {
        "torchlens": tl.__version__,
        "commit": git_sha(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "machine": machine,
        "torch_threads": torch.get_num_threads(),
        "libraries": versions,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
    }


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file or, for a directory, of its files in sorted order.

    Parameters
    ----------
    path:
        A file or directory.

    Returns
    -------
    str
        Hex digest.
    """

    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def path_bytes(path: Path) -> int:
    """Size of a file or of every file under a directory.

    Parameters
    ----------
    path:
        A file or directory.

    Returns
    -------
    int
        Total bytes.
    """

    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def artifact_record(
    path: Path,
    replay: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Reload a saved artifact with ``tl.load`` and replay one measurement from it.

    Parameters
    ----------
    path:
        The saved ``.tlspec`` (trace or bundle).
    replay:
        Called with the reloaded object; returns a dict holding at least ``matched``
        (bool) and the numbers it compared.

    Returns
    -------
    dict
        File name, sha256, size and the replay result.
    """

    loaded = tl.load(path)
    result = replay(loaded)
    if "matched" not in result:
        raise ValueError("a replay result must say whether it matched")
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "bytes": path_bytes(path),
        "loaded_type": type(loaded).__name__,
        "replay": result,
    }


def count_region_lines(source: Path, region: str) -> int:
    """Count the load-bearing lines of one marked region of a script.

    A region starts at a line ``# region: <name>`` and ends at the next ``# endregion``.
    Blank lines, comment lines and docstring lines are not counted; imports are.

    Parameters
    ----------
    source:
        The script.
    region:
        The region name.

    Returns
    -------
    int
        Non-blank, non-comment, non-docstring lines inside the region.
    """

    lines = source.read_text("utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == REGION_START + region)
    except StopIteration as exc:
        raise ValueError(f"{source.name} has no region {region!r}") from exc
    count, in_doc = 0, False
    for line in lines[start + 1 :]:
        text = line.strip()
        if text == REGION_END:
            return count
        if in_doc:
            in_doc = not (text.endswith('"""') or text.endswith("'''"))
            continue
        if text.startswith(('"""', "'''")):
            closed = len(text) > 3 and (text.endswith('"""') or text.endswith("'''"))
            in_doc = not closed
            continue
        if text and not text.startswith("#"):
            count += 1
    raise ValueError(f"{source.name} region {region!r} has no end")


def _jsonable(value: Any) -> Any:
    """Convert checks and tuples to plain JSON values."""

    if isinstance(value, Check):
        return {"rule": value.rule, "measured": _jsonable(value.measured), "passed": value.passed}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if value == value else "nan"
    return value


def report(entry: Entry, out_dir: Path | None, device: str, started: float) -> dict[str, Any]:
    """Print the entry's claim, checks and verdict; write ``<entry>.json`` in ``out_dir``.

    Parameters
    ----------
    entry:
        The finished entry.
    out_dir:
        Output directory, or ``None`` to print only.
    device:
        The torch device the entry ran on.
    started:
        ``time.time()`` at the start of the run.

    Returns
    -------
    dict
        The JSON summary.
    """

    summary = {
        "entry": entry.name,
        "claim": entry.claim,
        "pinned": entry.pinned,
        "oracles": entry.oracles,
        "controls": entry.controls,
        "measured": entry.measured,
        "artifact": entry.artifact,
        "basis": {
            "verbs": [verb for verb in VERBS if verb in entry.verbs],
            "lines": entry.lines,
            "architectures": list(entry.architectures),
        },
        "verdict": entry.verdict(),
        "stamp": stamp(device, started),
    }
    summary = _jsonable(summary)
    print(f"CLAIM: {entry.claim}")
    for kind, checks in (("ORACLE", entry.oracles), ("CONTROL", entry.controls)):
        for name, check in checks.items():
            outcome = "pass" if check.passed else "FAIL"
            print(f"{kind} {name}: {outcome}; rule: {check.rule}")
            print(f"  measured: {json.dumps(_jsonable(check.measured), sort_keys=True)}")
    replay = entry.artifact.get("replay", {})
    print(f"ARTIFACT: {entry.artifact.get('file')} replay matched: {replay.get('matched')}")
    print(f"BASIS: verbs {summary['basis']['verbs']}; lines {entry.lines}")
    print(f"VERDICT: {summary['verdict']}")
    print(f"STAMP: {json.dumps(summary['stamp'], sort_keys=True)}")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        target = Path(out_dir) / f"{entry.name}.json"
        target.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary
