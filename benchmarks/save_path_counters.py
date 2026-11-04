"""Work done by one ``trace.save`` on the default portable path, as deterministic counters.

The save-time baseline the 3.0 format writer is measured against. Counters gate; wall
clock is context only (perf doctrine: counters, never wall clock, gate).

Counters for one save of a fixed capture:

``metadata_pickle_dumps``
    ``dump_canonical_metadata`` passes over the scrubbed state (the pure-Python
    canonical pickler is the dominant save cost, so a second pass doubles it).
``pickler_save_calls``
    ``save`` calls inside the canonical pickler (objects the walk visits).
``verify_reloads``
    Restricted-unpickler ``load`` calls during the save: the write/read-symmetry dry
    run of the exact metadata bytes before they are written.
``fsync_calls``
    ``os.fsync`` calls (every written file, then the directories, then the parent
    after the publish rename).
``read_opens`` / ``write_opens``
    Python-level ``open`` calls under the destination's parent directory, by mode.
    Tensor blobs are written by the safetensors extension and do not appear here;
    they appear in the byte counters.
``files_written`` / ``bytes_written``
    Files in the published bundle and their total size; ``bytes_by_part`` splits the
    size into ``manifest``, ``metadata``, ``blobs`` and ``other``.

``python_calls`` (total calls under ``cProfile``) and the wall time of two unprofiled
saves (an A/A pair) are reported as information and never gate.

Run ``python benchmarks/save_path_counters.py --out OUT.json``; one JSON document is
printed and written to ``--out``.
"""

from __future__ import annotations

import argparse
import builtins
import cProfile
import io
import json
import os
import pstats
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GATED_COUNTERS = (
    "metadata_pickle_dumps",
    "pickler_save_calls",
    "verify_reloads",
    "fsync_calls",
    "read_opens",
    "write_opens",
    "files_written",
    "bytes_written",
)


@contextmanager
def _patched(
    owner: Any, name: str, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Iterator[None]:
    own = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, wrap(original))
    try:
        yield
    finally:
        if own:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _counting(counts: Counter[str], key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[key] += 1
            return original(*args, **kwargs)

        return counted

    return wrap


def _open_counter(
    counts: Counter[str], root: Path
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    root_text = str(root)

    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def counted(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, os.PathLike)) and os.fspath(file).startswith(root_text):
                counts["write_opens" if any(c in mode for c in "wax+") else "read_opens"] += 1
            return original(file, mode, *args, **kwargs)

        return counted

    return wrap


@contextmanager
def counting(root: Path) -> Iterator[Counter[str]]:
    """Count save-path work while the block runs; opens are counted only under ``root``."""

    from torchlens._io import _canonical_pickle
    from torchlens._io.bundle import _RenameAwareUnpickler

    counts: Counter[str] = Counter()
    pickler = _canonical_pickle._CanonicalMetadataPickler
    with ExitStack() as stack:
        stack.enter_context(_patched(pickler, "dump", _counting(counts, "metadata_pickle_dumps")))
        stack.enter_context(_patched(pickler, "save", _counting(counts, "pickler_save_calls")))
        stack.enter_context(
            _patched(_RenameAwareUnpickler, "load", _counting(counts, "verify_reloads"))
        )
        stack.enter_context(_patched(os, "fsync", _counting(counts, "fsync_calls")))
        stack.enter_context(_patched(builtins, "open", _open_counter(counts, root)))
        stack.enter_context(_patched(io, "open", _open_counter(counts, root)))
        yield counts


def _part(relative: Path) -> str:
    if relative.name == "manifest.json":
        return "manifest"
    if relative.name == "metadata.pkl":
        return "metadata"
    if relative.parts and relative.parts[0] == "blobs":
        return "blobs"
    return "other"


def bundle_bytes(path: Path) -> dict[str, Any]:
    """Return the file count, total bytes and per-part bytes of a published bundle."""

    by_part: Counter[str] = Counter()
    files = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files += 1
            by_part[_part(item.relative_to(path))] += item.stat().st_size
    return {
        "files_written": files,
        "bytes_written": sum(by_part.values()),
        "bytes_by_part": {k: by_part.get(k, 0) for k in ("manifest", "metadata", "blobs", "other")},
    }


def build_capture() -> Any:
    """Capture the benchmark's ``tinynet`` cell (the ``aux_save`` row's model and input)."""

    import torchlens as tl
    from benchmarks.perf_models import load_model_and_input

    torch.manual_seed(0)
    model, x = load_model_and_input("tinynet", "cpu")
    return tl.trace(model, x)


def measure(trace: Any, workdir: Path) -> dict[str, Any]:
    """Save ``trace`` once under the counters and return the counter document."""

    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "counted.tlspec"
    with counting(workdir) as counts:
        trace.save(str(target))
    result: dict[str, Any] = {key: counts.get(key, 0) for key in GATED_COUNTERS}
    result.update(bundle_bytes(target))
    return result


def _python_calls(trace: Any, workdir: Path) -> int:
    profiler = cProfile.Profile()
    profiler.enable()
    trace.save(str(workdir / "profiled.tlspec"))
    profiler.disable()
    return int(pstats.Stats(profiler).total_calls)  # type: ignore[attr-defined]


def _wall_ms(trace: Any, workdir: Path, tag: str) -> float:
    started = time.perf_counter()
    trace.save(str(workdir / f"timed_{tag}.tlspec"))
    return round((time.perf_counter() - started) * 1000.0, 3)


def run() -> dict[str, Any]:
    """Measure the counters twice (they must agree) plus the information-only context."""

    import warnings

    trace = build_capture()
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workdir = Path(tmp)
        first = measure(trace, workdir / "a")
        second = measure(trace, workdir / "b")
        info = {
            "python_calls": _python_calls(trace, workdir),
            "wall_ms_aa_pair": [_wall_ms(trace, workdir, "1"), _wall_ms(trace, workdir, "2")],
        }
    return {
        "schema": "torchlens.save_path_counters.v1",
        "model": "tinynet",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "counters": first,
        "repeat_agrees": first == second,
        "information": info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    document = run()
    text = json.dumps(document, indent=2, sort_keys=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if document["repeat_agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
