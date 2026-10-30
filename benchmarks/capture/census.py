"""The "what hooks miss" census over the classics corpus.

For every corpus entry this traces the model, counts the operations that PyTorch module
hooks cannot reach, forward-replay validates it, and names the boundaries the capture
cannot see into. A module forward hook sees a module's output; a forward pre-hook sees
the tensors passed into a module call; a hook on the root module sees the model output.
An operation whose output is none of those is reachable only by editing ``forward``.

Two reachability counts are kept so the reader can pick the generous one:

* ``unreachable_output_hooks``: not a module output and not a model output;
* ``unreachable_any_hook``: additionally not fed into any module call (so a pre-hook
  on the consumer cannot see it either). This is the headline share.

Input, output and buffer nodes are not computations and are left out of the
denominator. A validation failure is a finding and is kept by model, with the
failure summary; nothing is filtered out.

Run one entry in-process (``--entry ID``) or the whole corpus, each entry in a fresh
subprocess (``--jobs N`` at a time), writing one result document::

    python benchmarks/capture/census.py --out census.json --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import gc
import json
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.capture import schema  # noqa: E402
from benchmarks.host_label import redact_local_paths  # noqa: E402

FAMILY_ORDER = ("attention", "recurrent", "convolutional", "dense", "other")
ATTENTION_MODULES = {
    "MultiheadAttention",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
}
RECURRENT_MODULES = {"LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell"}
ATTENTION_FUNCS = {"scaled_dot_product_attention", "multi_head_attention_forward"}
MATMUL_FUNCS = {"matmul", "bmm", "einsum", "__matmul__", "baddbmm"}
RECURRENT_FUNCS = {"lstm", "gru", "rnn_tanh", "rnn_relu", "lstm_cell", "gru_cell", "rnn_tanh_cell"}
CONV_FUNCS = {
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
}
DENSE_FUNCS = {"linear", "addmm", "mm", "bilinear"}
# Single calls whose internal arithmetic runs inside one kernel: the capture records the
# call and its output, never the gates, scores or intermediate sums inside it.
FUSED_BOUNDARIES = {
    "scaled_dot_product_attention": "fused attention kernel",
    "_native_multi_head_attention": "fused attention kernel",
    "multi_head_attention_forward": "attention routine (decomposes unless the fast path runs)",
    "lstm": "fused recurrent kernel (gates not exposed)",
    "gru": "fused recurrent kernel (gates not exposed)",
    "rnn_tanh": "fused recurrent kernel",
    "rnn_relu": "fused recurrent kernel",
    "lstm_cell": "fused recurrent cell",
    "gru_cell": "fused recurrent cell",
    "rnn_tanh_cell": "fused recurrent cell",
    "embedding_bag": "fused gather-reduce kernel",
    "layer_norm": "fused normalization kernel",
    "group_norm": "fused normalization kernel",
    "batch_norm": "fused normalization kernel",
    "instance_norm": "fused normalization kernel",
    "ctc_loss": "fused loss kernel",
    "cross_entropy": "fused loss kernel",
}
ENTRY_TIMEOUT_S = 600


def classify_reachability(op: Any) -> dict[str, bool]:
    """Say which hook kinds can reach one operation's output.

    Parameters
    ----------
    op:
        A captured operation (anything with the ``Op`` reachability fields).

    Returns
    -------
    dict[str, bool]
        ``computation`` (counted at all), ``output_hook`` (a module or root forward
        hook sees it) and ``any_hook`` (an output hook or a consumer's pre-hook sees it).
    """

    computation = not (
        getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or getattr(op, "is_buffer", False)
    )
    output_hook = bool(getattr(op, "is_module_output", False)) or bool(
        getattr(op, "is_output_parent", False)
    )
    pre_hook = bool(getattr(op, "input_to_module_calls", None))
    return {
        "computation": computation,
        "output_hook": output_hook,
        "any_hook": output_hook or pre_hook,
    }


def classify_family(module_types: Iterable[str], func_names: Iterable[str]) -> str:
    """Assign an architecture family by a fixed priority rule.

    Attention wins over recurrence, recurrence over convolution, convolution over a
    dense (linear) network; anything else is ``other`` (energy models, logic nets,
    hand-written arithmetic).

    Parameters
    ----------
    module_types:
        Class names of the model's modules.
    func_names:
        Function names of the captured operations.

    Returns
    -------
    str
        One of :data:`FAMILY_ORDER`.
    """

    mods = set(module_types)
    funcs = set(func_names)
    if mods & ATTENTION_MODULES or funcs & ATTENTION_FUNCS:
        return "attention"
    if "softmax" in funcs and funcs & MATMUL_FUNCS:
        return "attention"
    if mods & RECURRENT_MODULES or funcs & RECURRENT_FUNCS:
        return "recurrent"
    if funcs & CONV_FUNCS:
        return "convolutional"
    if funcs & DENSE_FUNCS or "Linear" in mods:
        return "dense"
    return "other"


def census_counts(ops: Iterable[Any]) -> dict[str, Any]:
    """Count reachable and unreachable operations and the fused boundaries.

    Parameters
    ----------
    ops:
        The trace's operations.

    Returns
    -------
    dict[str, Any]
        Counts, shares, the function names of unreachable operations, and the fused or
        opaque boundary calls seen.
    """

    total = unreachable_out = unreachable_any = 0
    missed: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    funcs: Counter[str] = Counter()
    for op in ops:
        reach = classify_reachability(op)
        if not reach["computation"]:
            continue
        name = str(getattr(op, "func_name", None) or "none")
        total += 1
        funcs[name] += 1
        if name in FUSED_BOUNDARIES:
            boundaries[name] += 1
        if not reach["output_hook"]:
            unreachable_out += 1
        if not reach["any_hook"]:
            unreachable_any += 1
            missed[name] += 1
    return {
        "computations": total,
        "unreachable_output_hooks": unreachable_out,
        "unreachable_any_hook": unreachable_any,
        "share_unreachable_output_hooks": unreachable_out / total if total else None,
        "share_unreachable_any_hook": unreachable_any / total if total else None,
        "missed_funcs": dict(missed.most_common()),
        "fused_boundaries": {name: [n, FUSED_BOUNDARIES[name]] for name, n in boundaries.items()},
        "func_counts": dict(funcs.most_common()),
    }


def _opaque_calls(ops: Iterable[Any]) -> dict[str, int]:
    """Count calls the capture records only as one opaque node (custom autograd, extensions)."""

    opaque: Counter[str] = Counter()
    for op in ops:
        qual = str(getattr(op, "func_qualname", "") or "")
        name = str(getattr(op, "func_name", "") or "")
        if name == "apply" or "Function.apply" in qual or qual.startswith("torch.ops."):
            opaque[qual or name] += 1
    return dict(opaque)


def _validate(entry: Any, model: Any, inputs: Any) -> dict[str, Any]:
    """Forward-replay validate one entry exactly as the corpus test does."""

    import torch

    import torchlens as tl
    from tests.classics_corpus._loader import VALIDATION_SEED
    from torchlens.utils.arg_handling import normalize_input_args
    from torchlens.validation import last_validation_failure

    with torch.no_grad():
        model(*normalize_input_args(copy.deepcopy(inputs), model))
    start = time.perf_counter()
    verdict = tl.validate(
        model, inputs, scope="forward", random_seed=VALIDATION_SEED, validate_metadata=True
    )
    result: dict[str, Any] = {"passed": bool(verdict), "seconds": time.perf_counter() - start}
    if not verdict:
        failure = last_validation_failure()
        result["failure"] = failure.summary() if failure is not None else "no failure record"
    return result


PANEL_PREFIX = "panel:"


def _resolve(
    entry_id: str, build_entry: Any, corpus_entries: Any
) -> tuple[Any, dict[str, Any], Any]:
    """Return the entry, its descriptive fields and a zero-argument builder.

    ``panel:<name>`` names a smoke-runnable real-model panel entry (or an operation
    fixture) instead of a corpus entry; it is censused the same way.
    """

    if entry_id.startswith(PANEL_PREFIX):
        from benchmarks.capture.fixtures import build_fixture
        from benchmarks.capture.panel import build_panel_entry

        name = entry_id[len(PANEL_PREFIX) :]
        builder = build_fixture if name.startswith("fx-") else build_panel_entry
        return None, {"tier": "panel", "module": name}, lambda: builder(name)[:2]
    entry = next(e for e in corpus_entries() if e.id == entry_id)
    return entry, {"tier": entry.tier, "module": entry.module}, lambda: build_entry(entry)


def run_entry(entry_id: str) -> dict[str, Any]:
    """Census one corpus entry in this process and return its cell.

    Parameters
    ----------
    entry_id:
        Corpus entry id.

    Returns
    -------
    dict[str, Any]
        A census cell (status ``ok`` or ``error`` with the reason).
    """

    import torch

    import torchlens as tl
    from tests.classics_corpus._loader import build_entry, corpus_entries

    torch.use_deterministic_algorithms(True)
    entry, base, build = _resolve(entry_id, build_entry, corpus_entries)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, inputs = build()
            start = time.perf_counter()
            trace = tl.trace(model, inputs)
            trace_s = time.perf_counter() - start
            ops = list(trace.ops)
            counts = census_counts(ops)
            family = classify_family(
                (type(m).__name__ for m in model.modules()), counts["func_counts"]
            )
            opaque = _opaque_calls(ops)
            del trace, ops
            gc.collect()
            validation = _validate(entry, model, inputs)
    except Exception as exc:  # noqa: BLE001 - every failure is a recorded finding
        reason = redact_local_paths(f"{type(exc).__name__}: {exc}"[:800], REPO_ROOT)
        return schema.make_cell(
            "census", "hooks_miss", "torchlens", entry_id, "error", reason=reason, **base
        )
    return schema.make_cell(
        "census",
        "hooks_miss",
        "torchlens",
        entry_id,
        "ok",
        family=family,
        trace_seconds=trace_s,
        counts=counts,
        opaque_calls=opaque,
        validation=validation,
        **base,
    )


def _run_subprocess(entry_id: str, timeout: int) -> dict[str, Any]:
    """Census one entry in a fresh interpreter and parse its cell."""

    cmd = [sys.executable, str(Path(__file__).resolve()), "--entry", entry_id]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return schema.make_cell(
            "census", "hooks_miss", "torchlens", entry_id, "timeout", reason=f"over {timeout} s"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = redact_local_paths((proc.stdout + proc.stderr)[-800:], REPO_ROOT)
    return schema.make_cell(
        "census",
        "hooks_miss",
        "torchlens",
        entry_id,
        "error",
        reason=f"rc={proc.returncode}: {tail}",
    )


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate census cells by architecture family.

    Parameters
    ----------
    cells:
        Census cells.

    Returns
    -------
    dict[str, Any]
        Per family: models, models with a number, median and pooled unreachable share,
        replay pass count and rate, and the failing model ids.
    """

    by_family: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_family.setdefault(cell.get("family", "not_captured"), []).append(cell)
    out: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        ok = [r for r in rows if r["status"] == "ok"]
        shares = sorted(
            r["counts"]["share_unreachable_any_hook"] for r in ok if r["counts"]["computations"]
        )
        comp = sum(r["counts"]["computations"] for r in ok)
        missed = sum(r["counts"]["unreachable_any_hook"] for r in ok)
        passed = [r for r in ok if r["validation"]["passed"]]
        out[family] = {
            "models": len(rows),
            "captured": len(ok),
            "median_share_unreachable_any_hook": shares[len(shares) // 2] if shares else None,
            "pooled_share_unreachable_any_hook": missed / comp if comp else None,
            "models_with_any_unreachable": sum(1 for s in shares if s > 0),
            "replay_passed": len(passed),
            "replay_pass_rate": len(passed) / len(ok) if ok else None,
            "replay_failed": sorted(r["subject"] for r in ok if not r["validation"]["passed"]),
            "not_captured": sorted(r["subject"] for r in rows if r["status"] != "ok"),
        }
    return out


def run_corpus(
    out: Path, jobs: int, timeout: int, tier: str | None, limit: int | None
) -> dict[str, Any]:
    """Census every selected entry, each in a fresh subprocess, and write the document."""

    from tests.classics_corpus._loader import corpus_entries

    if tier == "panel":
        from benchmarks.capture.panel import smoke_panel

        probes = ["fx-custom_autograd-100-small", "fx-attention_fused-100-small"]
        ids = [PANEL_PREFIX + name for name in (*smoke_panel(), *probes)]
    else:
        ids = [e.id for e in corpus_entries(tier)][: limit or None]
    doc = schema.new_document(
        "census",
        {"tier": tier, "limit": limit, "jobs": jobs, "timeout_s": timeout, "entries": len(ids)},
    )
    doc["protocol"] = {
        "fresh_subprocess": True,
        "deterministic_algorithms": True,
        "validation": "forward+metadata",
    }
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for cell in pool.map(lambda i: _run_subprocess(i, timeout), ids):
            doc["cells"].append(cell)
            print(f"{cell['subject']}: {cell['status']}", flush=True)
    doc["summary"] = summarize(doc["cells"])
    out.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", help="census one entry in this process and print its cell")
    parser.add_argument("--out", type=Path, default=Path("census.json"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=ENTRY_TIMEOUT_S)
    parser.add_argument("--tier", choices=("smoke", "comprehensive", "panel"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.entry:
        print(json.dumps(run_entry(args.entry), default=str), flush=True)
        return
    doc = run_corpus(args.out, args.jobs, args.timeout, args.tier, args.limit)
    print(json.dumps(doc["summary"], indent=1))
    problems = schema.validate_document(doc)
    if problems:
        raise SystemExit("result document violates the format: " + "; ".join(problems[:5]))


if __name__ == "__main__":
    main()
