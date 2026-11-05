"""Work done by one ``trace.save`` on the default portable path, as deterministic counters.

The save path is saved format 3 (``docs/maintainers/saved_format_3.md``): typed JSON
record tables plus safetensors chunks, no pickle at any level. Counters gate; wall
clock is context only (perf doctrine: counters, never wall clock, gate).

Counters for one save of a fixed capture:

``metadata_writes``
    ``write_trace_envelope`` calls: the one pass that encodes the scrubbed state into
    the record tables and writes them with the manifest (a second pass doubles the
    dominant save cost).
``encoder_calls``
    ``ValueEncoder.encode`` calls inside that pass: one per record cell that is not
    a plain JSON scalar (``format3.values.plain_cell`` cells are written as they
    are, without an encoder call).
``verify_reloads``
    ``preflight_reload`` calls during the save: the late.21 read-back of the staged
    record tables through the bounded JSON reader and typed decoder before publish.
``fsync_calls``
    ``os.fsync`` calls (every written file, then the directories, then the parent
    after the publish rename).
``read_opens`` / ``write_opens``
    Python-level ``open`` calls under the destination's parent directory, by mode.
    Tensor blobs are written by the safetensors extension and do not appear here;
    they appear in the byte counters.
``files_written`` / ``bytes_written``
    Files in the published bundle and their total size with each wall-clock timing
    float (``TIMING_FLOAT_PATHS``) counted at ``TIMING_FLOAT_WIDTH`` bytes; every other
    byte counts as written. ``bytes_by_part`` splits the size into ``manifest``,
    ``metadata`` (the ``records/`` tables), ``blobs`` and ``other``;
    ``timing_floats_by_file`` counts the timing floats found per record table.

``python_calls`` (total calls under ``cProfile``), the raw byte count of the bundle and
the wall time of two unprofiled saves (an A/A pair) are reported as information and
never gate.

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
    "metadata_writes",
    "encoder_calls",
    "verify_reloads",
    "fsync_calls",
    "read_opens",
    "write_opens",
    "files_written",
    "bytes_written",
)


#: Wall-clock timing floats in the record tables, by file and key path (``*`` matches
#: one list index or object key). The pre-3 pickle wrote every float at a fixed width,
#: so timings never moved its size; saved format 3 writes floats by exact repr
#: (``docs/maintainers/saved_format_3.md``), whose text length moves with the measured
#: value. Only these floats are counted at ``TIMING_FLOAT_WIDTH``; every other byte,
#: other floats included, counts as written, so any non-timing change moves the count.
TIMING_FLOAT_PATHS: dict[str, tuple[str, ...]] = {
    "records/op.json": ("columns.func_duration.*.fields.value",),
    "records/module.json": (
        "columns.forward_duration.*.fields.value",
        "columns.func_calls_duration.*.fields.value",
        "columns.total_forward_duration.*.fields.value",
        "columns.total_func_calls_duration.*.fields.value",
    ),
    "records/module_call.json": (
        "columns.forward_duration.*.fields.value",
        "columns.func_calls_duration.*.fields.value",
    ),
    "records/trace.json": (
        "fields._cost_forward_duration.fields.value",
        "fields._cost_func_calls_duration.fields.value",
        "fields._phase_timings.*.total_s",
        "fields._provenance_setup_duration.fields.value",
        "fields._provenance_cleanup_duration.fields.value",
        "fields._provenance_start_time",
        "fields._provenance_end_time",
    ),
}
#: Bytes each timing float counts for (a binary double's width).
TIMING_FLOAT_WIDTH = 8


def _matches(node: Any, steps: list[str]) -> Iterator[Any]:
    """Yield the values under ``node`` at the dotted ``steps`` (``*``: any one child)."""

    if not steps:
        yield node
        return
    head, rest = steps[0], steps[1:]
    if isinstance(node, dict):
        children = list(node.values()) if head == "*" else [node[head]] if head in node else []
    elif isinstance(node, list) and head == "*":
        children = list(node)
    else:
        children = []
    for child in children:
        yield from _matches(child, rest)


def timing_float_slack(relative: str, text: str) -> tuple[int, int]:
    """Return (bytes the timing floats of one file take beyond their fixed width, count)."""

    patterns = TIMING_FLOAT_PATHS.get(relative)
    if not patterns:
        return 0, 0
    document = json.loads(text)
    floats = [
        value
        for pattern in patterns
        for value in _matches(document, pattern.split("."))
        if isinstance(value, float)
    ]
    slack = sum(len(json.dumps(value)) - TIMING_FLOAT_WIDTH for value in floats)
    return slack, len(floats)


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

    from torchlens._io import bundle
    from torchlens._io.format3.values import ValueEncoder

    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        stack.enter_context(
            _patched(bundle, "write_trace_envelope", _counting(counts, "metadata_writes"))
        )
        stack.enter_context(_patched(ValueEncoder, "encode", _counting(counts, "encoder_calls")))
        stack.enter_context(
            _patched(bundle, "preflight_reload", _counting(counts, "verify_reloads"))
        )
        stack.enter_context(_patched(os, "fsync", _counting(counts, "fsync_calls")))
        stack.enter_context(_patched(builtins, "open", _open_counter(counts, root)))
        stack.enter_context(_patched(io, "open", _open_counter(counts, root)))
        yield counts


def _part(relative: Path) -> str:
    if relative.name == "manifest.json":
        return "manifest"
    if relative.parts and relative.parts[0] == "records":
        return "metadata"
    if relative.parts and relative.parts[0] == "blobs":
        return "blobs"
    return "other"


def bundle_bytes(path: Path) -> dict[str, Any]:
    """Return the file count, timing-normalised total and per-part bytes, and timing floats.

    ``raw_bytes_written`` is the bundle's size as written (information only: it moves
    with the measured timings).
    """

    by_part: Counter[str] = Counter()
    timing_floats: dict[str, int] = {}
    files = raw = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files += 1
            relative = item.relative_to(path)
            size = item.stat().st_size
            raw += size
            slack, found = 0, 0
            if relative.as_posix() in TIMING_FLOAT_PATHS:
                slack, found = timing_float_slack(
                    relative.as_posix(), item.read_text(encoding="utf-8")
                )
                timing_floats[relative.as_posix()] = found
            by_part[_part(relative)] += size - slack
    return {
        "files_written": files,
        "bytes_written": sum(by_part.values()),
        "bytes_by_part": {k: by_part.get(k, 0) for k in ("manifest", "metadata", "blobs", "other")},
        "timing_floats_by_file": timing_floats,
        "raw_bytes_written": raw,
    }


def build_capture() -> Any:
    """Capture the benchmark's ``tinynet`` cell (the ``aux_save`` row's model and input).

    The capture's seed and name are pinned. An unseeded capture draws a random seed,
    and the record tables write the seed as decimal text, so ``metadata`` bytes would
    move with its digit count between processes. The
    automatic name counts captures in the process (``tinynet_1``, ``tinynet_12``), so
    its length depends on what ran before. Pinning both leaves the save path as is.
    """

    import torchlens as tl
    from benchmarks.perf_models import load_model_and_input
    from torchlens.options import CaptureOptions

    torch.manual_seed(0)
    model, x = load_model_and_input("tinynet", "cpu")
    return tl.trace(model, x, capture=CaptureOptions(seed=0, name="tinynet"))


def measure(trace: Any, workdir: Path) -> dict[str, Any]:
    """Save ``trace`` once under the counters and return the counter document."""

    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "counted.tlspec"
    with counting(workdir) as counts:
        trace.save(str(target))
    result: dict[str, Any] = {key: counts.get(key, 0) for key in GATED_COUNTERS}
    sizes = bundle_bytes(target)
    sizes.pop("raw_bytes_written")
    result.update(sizes)
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
            "raw_bytes_written": bundle_bytes(workdir / "a" / "counted.tlspec")[
                "raw_bytes_written"
            ],
        }
    return {
        "schema": "torchlens.save_path_counters.v2",
        "model": "tinynet",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
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
