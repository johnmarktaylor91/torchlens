"""One-time recording sweep over the postprocess axes matrix.

Runs every axis of ``tests/support/postprocess_axes.py`` with the combined
read+write audit in RECORD mode and dumps the per-step observation unions as
JSON. The output is EVIDENCE for seeding the declared ``reads`` sets and the
four-category findings review (design-ppdag-v3 §2.4/§8) — declarations are
hand-reviewed against step source before landing, never auto-regenerated.

Usage (env is set by the script itself):
    python tools/record_postprocess_matrix.py out.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

os.environ["TORCHLENS_POSTPROCESS_ASSERTIONS"] = "1"
os.environ["TORCHLENS_POSTPROCESS_WRITE_AUDIT"] = "record"
os.environ["TORCHLENS_POSTPROCESS_READ_AUDIT"] = "record"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    """Run the matrix and dump per-step and per-axis observations."""

    out_path = sys.argv[1] if len(sys.argv) > 1 else "postprocess_matrix.json"
    from support.postprocess_axes import iter_axes

    import torchlens._postprocess as pp

    per_axis: dict[str, dict[str, dict[str, list[str]]]] = {}
    failures: dict[str, str] = {}
    union: dict[str, dict[str, set[str]]] = {}
    started = time.process_time()
    for axis_name, axis_fn in iter_axes():
        pp.RECORDED_STEP_WRITES.clear()
        pp.RECORDED_STEP_READS.clear()
        pp.RECORDED_STEP_CLONE_READS.clear()
        pp.RECORDED_STEP_EFFECTIVE_WRITES.clear()
        try:
            trace = axis_fn()
        except Exception:
            failures[axis_name] = traceback.format_exc()
            print(f"[matrix] AXIS FAILED: {axis_name}", file=sys.stderr)
            continue
        axis_record: dict[str, dict[str, list[str]]] = {}
        for channel, sink in (
            ("writes", pp.RECORDED_STEP_WRITES),
            ("reads", pp.RECORDED_STEP_READS),
            ("clone_reads", pp.RECORDED_STEP_CLONE_READS),
            ("effective_writes", pp.RECORDED_STEP_EFFECTIVE_WRITES),
        ):
            for step, columns in sink.items():
                axis_record.setdefault(step, {}).setdefault(channel, []).extend(sorted(columns))
                union.setdefault(step, {}).setdefault(channel, set()).update(columns)
        per_axis[axis_name] = axis_record
        if trace is not None:
            trace.cleanup()
        print(f"[matrix] done: {axis_name}")
    elapsed = time.process_time() - started
    payload = {
        "per_axis": per_axis,
        "union": {
            step: {channel: sorted(columns) for channel, columns in channels.items()}
            for step, channels in union.items()
        },
        "failures": failures,
        "cpu_seconds": elapsed,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"[matrix] wrote {out_path}; cpu={elapsed:.1f}s; failures={len(failures)}")


if __name__ == "__main__":
    main()
