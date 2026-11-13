"""Call-time floor instrument: calls per function output for each call-time section.

The capture-core counters (``capture_core_cost.py``) count calls per op by
phase. This module splits the forward of one ``tl.trace`` further, into the
sections the call-time lanes work on, and divides by the number of function
outputs the trace logged. It also runs the decision measurement (wall time,
shadow mode; :mod:`benchmarks.capture_core_floor_shadow`).

Counter names are ``floor.<model>.<section>_per_output``; the sections are
listed in :data:`SECTIONS` and documented in ``capture_core_floor.md``. They
partition every Python and C call made inside ``_capture/run.py::_run_forward``
except ``label_read_stamp``, which is an overlay (calls inside the label read
and stamp, wherever they happen, also counted in their section).

How calls are attributed (deterministic; no timing):

* the forward opens ``model_code``: the model's own code and torch's module
  call machinery;
* the function wrapper (``wrapped_func``) opens ``wrapper_outside_emit``;
* TorchLens code reached from ``model_code`` outside the function wrapper opens
  ``module_handling`` (the forward runner, the session forward, module entry
  and exit records); the user forward that ``decorated_forward`` calls, and a
  module call (``nn.Module.__call__``) TorchLens makes, return to ``model_code``;
* the emit (``log_function_output_tensors``, or the thin emit ``thin_emit``)
  opens ``emit_rest``; inside it the first marker of :data:`SECTION_MARKERS`
  reached opens that section, which keeps every call beneath it (the outermost
  section wins).

A lane that moves a section's work into a new function adds the function's
name to that section's markers in the same change; otherwise the work shows
up in ``emit_rest`` and the section's ``hits`` count falls to zero, which
``check_markers`` reports.

Usage::

    python benchmarks/capture_core_floor.py --counters --out sections.json
    python benchmarks/capture_core_floor.py --wall --out wall.json
    python benchmarks/capture_core_floor.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: The five full models and the module-call model, as REPLAN section 4 (N0) names them.
FLOOR_MODELS = ("chain1000", "varied200", "loopcell100", "resnet18", "gpt2_tiny4", "mod_relu200")

#: Every section, in report order. ``label_read_stamp`` is an overlay.
SECTIONS = (
    "arg_scan",
    "call_site",
    "autograd",
    "output_facts",
    "payload",
    "emit_rest",
    "wrapper_outside_emit",
    "module_handling",
    "model_code",
)
OVERLAYS = ("label_read_stamp",)

#: ``section -> function names`` whose call inside the emit opens the section.
#: Names are matched within the ``torchlens`` package, in any module, so a
#: function that moves between modules keeps its section.
SECTION_MARKERS: dict[str, tuple[str, ...]] = {
    # Argument scan, parents, versions, parameters, slot map, source resolution.
    "arg_scan": (
        "_extract_arg_tensors_and_params",
        "_fast_lane_arg_tensors",
        "get_arg_tensors_for_resolution",
        "split_non_tensor_args",
        "planned_slot_call",
        "split_call_arguments",
        "_bind_call_alias_labels",
        "_repin_rebound_receiver",
        "resolve_or_late_index_buffer_address",
        "log_source_tensor",
        "log_closure_constant_if_known",
        "snapshot_arg_slots",
        "parents_from_slots",
        "_set_arg_template_fields",
        "_build_graph_relationship_fields",
        "_build_param_fields",
        "witness_from_slots",
        "extract_salient_args",
    ),
    # Call-site capture: the frame walk, the site lookup, the location.
    "call_site": ("_set_call_site_fields", "_get_code_context"),
    # The output walk and autograd saved-tensor statistics.
    "autograd": (
        "_output_entries_with_autograd_stats",
        "_partition_output_entries_with_autograd_stats",
    ),
    # Shape, dtype, memory, gradient function, in-place and edit facts of one output.
    "output_facts": (
        "_read_output_shape_dtype_memory",
        "pop_op_grad_fn",
        "_metadata_shape",
        "get_memory_amount_from_metadata",
        "recorded_op_grad_fn",
        "pop_user_grad_fn",
        "pop_same_object_mutation",
        "_pop_tensor_live_fire_results",
        "_label_version_baseline",
    ),
    # The saved payload: copy, save budget, duplicate checks.
    "payload": (
        "_save_activation_fields",
        "_stream_activation_fields",
        "_save_call_args_fields",
    ),
}

#: ``(file suffix, function name)`` of the structural markers.
_RUN_FORWARD = ("_capture/run.py", "_run_forward")
_WRAPPER = ("backends/torch/wrappers.py", "wrapped_func")
_EMIT_NAME = "log_function_output_tensors"
_THIN_EMIT = ("backends/torch/_thin_emit.py", "thin_emit")
_MODULE_FORWARD = ("backends/torch/model_prep.py", "decorated_forward")
_OUTPUT = ("backends/torch/_ops_activations.py", "_log_output_tensor_info")
_LABEL = ("_utils/_tensor_label.py", ("get_tensor_label", "set_tensor_label"))

#: torch's module-call entry points: reached from TorchLens code, they start the
#: model's own code (the root module call the forward runner makes).
_MODULE_CALL = ("torch/nn/modules/module.py", ("__call__", "_wrapped_call_impl", "_call_impl"))

# Code kinds the classifier returns.
_K_OTHER, _K_TL, _K_RUN, _K_WRAP, _K_EMIT, _K_MODFWD, _K_MODCALL = range(7)


def _package_dir() -> str:
    import torchlens

    return os.path.dirname(os.path.abspath(torchlens.__file__)) + os.sep


def _make_classifier() -> Callable[[Any], tuple[int, str | None, bool, bool]]:
    """Return ``code -> (kind, section marker, is label marker, is output marker)``."""

    package = _package_dir()
    by_name = {name: section for section, names in SECTION_MARKERS.items() for name in names}

    def classify(code: Any) -> tuple[int, str | None, bool, bool]:
        filename = os.path.abspath(code.co_filename)
        if not filename.startswith(package):
            path = filename.replace("\\", "/")
            if code.co_name in _MODULE_CALL[1] and path.endswith(_MODULE_CALL[0]):
                return _K_MODCALL, None, False, False
            return _K_OTHER, None, False, False
        name = code.co_name
        path = filename.replace("\\", "/")
        kind = _K_TL
        if name == _RUN_FORWARD[1] and path.endswith(_RUN_FORWARD[0]):
            kind = _K_RUN
        elif name == _WRAPPER[1] and path.endswith(_WRAPPER[0]):
            kind = _K_WRAP
        elif name == _EMIT_NAME or (name == _THIN_EMIT[1] and path.endswith(_THIN_EMIT[0])):
            kind = _K_EMIT
        elif name == _MODULE_FORWARD[1] and path.endswith(_MODULE_FORWARD[0]):
            kind = _K_MODFWD
        label = name in _LABEL[1] and path.endswith(_LABEL[0])
        output = name == _OUTPUT[1] and path.endswith(_OUTPUT[0])
        return kind, by_name.get(name), label, output

    return classify


def _next_section(
    current: str, kind: int, marker: str | None, frame: Any, kinds: Callable[[Any], int]
) -> str:
    """Return the section a call of ``frame`` runs in, given the caller's section."""

    if current == "outside":
        return "model_code" if kind == _K_RUN else current
    if current == "model_code":
        if kind == _K_WRAP:
            return "wrapper_outside_emit"
        if kind == _K_EMIT:
            return "emit_rest"
        return "module_handling" if kind not in (_K_OTHER, _K_MODCALL) else current
    if current == "module_handling":
        if kind == _K_MODCALL:
            return "model_code"
        back = frame.f_back
        if kind == _K_OTHER and back is not None and kinds(back.f_code) == _K_MODFWD:
            return "model_code"
        return current
    if current == "wrapper_outside_emit":
        return "emit_rest" if kind == _K_EMIT else current
    if current == "emit_rest" and marker is not None:
        return marker
    return current


def count_sections(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``fn`` (one ``tl.trace``) and count its forward's calls by section.

    Returns
    -------
    tuple
        ``fn``'s result and ``{"outputs", "calls", "hits"}``: function outputs
        logged, calls per section (and overlay), and marker calls that opened
        each emit section.
    """

    classify = _make_classifier()
    cache: dict[int, tuple[Any, tuple[int, str | None, bool, bool]]] = {}
    calls = dict.fromkeys(("outside", *SECTIONS, *OVERLAYS), 0)
    hits = dict.fromkeys(SECTION_MARKERS, 0)
    outputs = [0]
    stack = ["outside"]
    marks: list[Any] = []
    label_frame: list[Any] = [None]

    def info(code: Any) -> tuple[int, str | None, bool, bool]:
        entry = cache.get(id(code))
        if entry is None or entry[0] is not code:
            entry = (code, classify(code))
            cache[id(code)] = entry
        return entry[1]

    def kind_of(code: Any) -> int:
        return info(code)[0]

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event == "call":
            kind, marker, label, output = info(frame.f_code)
            current = stack[-1]
            section = _next_section(current, kind, marker, frame, kind_of)
            if section != current:
                stack.append(section)
                marks.append(frame)
                if section in hits and current == "emit_rest":
                    hits[section] += 1
            if output and section != "outside":
                outputs[0] += 1
            if label and label_frame[0] is None and section != "outside":
                label_frame[0] = frame
            calls[section] += 1
            if label_frame[0] is not None:
                calls["label_read_stamp"] += 1
        elif event == "c_call":
            calls[stack[-1]] += 1
            if label_frame[0] is not None:
                calls["label_read_stamp"] += 1
        elif event == "return":
            if label_frame[0] is frame:
                label_frame[0] = None
            if marks and marks[-1] is frame:
                marks.pop()
                stack.pop()

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = fn()
    finally:
        sys.setprofile(previous)
    calls.pop("outside")
    return result, {"outputs": outputs[0], "calls": calls, "hits": hits}


def measure_sections(model: Any, x: Any) -> dict[str, Any]:
    """Count one warmed trace of ``model`` by section, on one thread, collector off.

    The caller has already traced the model at least twice (the counts are
    steady from the third trace; see ``capture_core_cost.measure_model``).
    """

    from _hermetic_counters import quiet_gc

    import torchlens as tl

    with quiet_gc():
        trace, record = count_sections(lambda: tl.trace(model, x))
    record["rows"] = len(trace)
    del trace
    return record


def flatten(document: dict[str, Any]) -> dict[str, float]:
    """Return the floor counters of a measurement as ``name -> number``.

    Reads the ``floor`` block of each model record (written by
    ``capture_core_cost.py`` or by ``--counters`` here).
    """

    flat: dict[str, float] = {}
    for model, record in document.get("models", {}).items():
        floor = record.get("floor") if isinstance(record, dict) else None
        if not floor or not floor.get("outputs"):
            continue
        outputs = floor["outputs"]
        flat[f"floor.{model}.outputs"] = outputs
        total = sum(floor["calls"][name] for name in SECTIONS)
        flat[f"floor.{model}.forward_total_per_output"] = total / outputs
        for name, count in floor["calls"].items():
            flat[f"floor.{model}.{name}_per_output"] = count / outputs
        for name, count in floor["hits"].items():
            flat[f"floor.{model}.hits.{name}_per_output"] = count / outputs
    return flat


def check_markers(flat: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """Name every section whose markers fired at the baseline and fire no more.

    A section the baseline saw that now records no marker call has lost its
    markers (a renamed or replaced function) and its counter would read low.
    """

    problems = []
    for name, value in sorted(baseline.items()):
        if name.startswith("floor.") and ".hits." in name and value > 0:
            if flat.get(name, 0) == 0:
                section = name.split(".hits.")[1].removesuffix("_per_output")
                problems.append(
                    f"{name}: no marker of section {section!r} fired; add the function that now "
                    f"does this work to SECTION_MARKERS[{section!r}]"
                )
    return problems


def _measure_one(name: str) -> dict[str, Any]:
    """Warm and count one model in this process."""

    import torch

    import torchlens as tl
    from benchmarks.capture_core_cost import build

    torch.set_num_threads(1)
    model, x = build(name)
    tl.trace(model, x)
    tl.trace(model, x)
    return measure_sections(model, x)


def _counters(names: tuple[str, ...]) -> dict[str, Any]:
    """Count each model in a fresh interpreter with ``PYTHONHASHSEED=0``."""

    env = dict(os.environ, PYTHONHASHSEED="0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Address-space randomization off makes heap addresses repeat between
    # processes. The allocator can still hand a new tensor a storage address
    # that is in the label stamp's alias index in one run and not the next,
    # which moves a few calls per trace (``_register_storage_alias``); the
    # comparison reports those and passes them within the bars' tolerance.
    setarch = shutil.which("setarch")
    prefix = [setarch, platform.machine(), "-R"] if setarch else []
    models: dict[str, Any] = {}
    for name in names:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "one.json"
            script = str(Path(__file__).resolve())
            command = [*prefix, sys.executable, script, "--one", name, "--out", str(out)]
            done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
            if out.exists():
                models[name] = {"floor": json.loads(out.read_text(encoding="utf-8"))}
            else:
                models[name] = {"error": f"exited {done.returncode}: {done.stderr[-1500:]}"}
        print(f"counted {name}: {'error' if 'error' in models[name] else 'ok'}", file=sys.stderr)
    return {"tool": "capture_core_floor", "address_randomization": not prefix, "models": models}


def _compare(first: dict[str, float], second: dict[str, float]) -> tuple[list[str], list[str]]:
    """Return the counters that differ at all, and those beyond the bars' tolerance."""

    from benchmarks import capture_core_bars as bars_mod

    differ = []
    for name in sorted(set(first) | set(second)):
        a, b = first.get(name), second.get(name)
        if a != b:
            relative = abs(a - b) / max(abs(a), abs(b)) if a is not None and b is not None else 1.0
            differ.append(f"{name}: {a} against {b} (relative {relative:.2g})")
    return differ, bars_mod.compare(first, second)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the JSON document here")
    parser.add_argument("--models", default=",".join(FLOOR_MODELS), help="comma-separated models")
    parser.add_argument("--counters", action="store_true", help="section counters per output")
    parser.add_argument("--wall", action="store_true", help="the shadow decision measurement")
    parser.add_argument("--repeats", type=int, default=9, help="with --wall: traces per kind")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return 0 on success."""

    args = _parser().parse_args(argv)
    if args.compare:
        a, b = (flatten(json.loads(p.read_text(encoding="utf-8"))) for p in args.compare)
        differ, beyond = _compare(a, b)
        print(f"{len(a)} counters; {len(differ)} differ; {len(beyond)} beyond the tolerance")
        print("\n".join(differ + [f"BEYOND {line}" for line in beyond]))
        return 1 if beyond or not a else 0
    names = tuple(args.models.split(","))
    if args.one:
        document: dict[str, Any] = _measure_one(args.one)
    elif args.wall:
        from benchmarks import capture_core_floor_shadow as shadow

        document = shadow.run(names, args.repeats)
    else:
        document = _counters(names)
    text = json.dumps(document, indent=1, default=str) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    errors = [n for n, r in document.get("models", {}).items() if "error" in r]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
