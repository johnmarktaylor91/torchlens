"""Decision measurement in shadow mode: wall time of the call-time floor per function output.

Each model is traced under the real wrapper (fast lane included) three ways,
timing the forward (``_capture/run.py::_run_forward``) and the whole trace:

- bare: the model called directly, with logging off;
- production: today's emit;
- shadow: every call-time observation of the H2 list runs first through
  today's helper functions (consuming and stamping nothing), then today's emit
  runs; each part is timed. The forward less the time inside today's emit is
  the forward with every call-time observation executed and the rest of
  today's emit removed, with module handling behaving as in production.

Observation sections (wall, shadow mode): ``arg_scan`` (argument scan,
parents, parameters, source resolution), ``call_site``, ``module_stack``,
``autograd`` (output partition and saved-tensor statistics),
``output_facts`` and ``payload`` (the product's save path on a scratch dict).
``forward_outside`` is the forward less today's emit less the observations:
the wrapper, module handling and the model's own code.

The shadow emit replaces today's emit at every attribute of a loaded TorchLens
module bound to it (:func:`swap_emit`): the function wrapper calls it through
``wrappers``, but the fast lane passes ``_ops_emission``'s own global to
``_call_plan.commit_fast_call``, and other entry points hold their own imported
names. Patching one binding leaves the others' calls unmeasured.

Every per-output figure divides by the function outputs of a production trace
(calls of ``_log_output_tensor_info``). The minimum over ``repeats`` traces is
reported. This is the instrument of stage 2 REPLAN sections 2 and 3, adapted
from its ``decision_measurement_rp.py``; nothing here edits product state
beyond in-process patches that are undone after each model.
"""

from __future__ import annotations

import gc
import platform
import sys
import time
import traceback
import warnings
from typing import Any

import torch

import torchlens as tl
from benchmarks import capture_core_cost as cost
from benchmarks.capture_core_floor import count_sections

T = torch.Tensor
P = torch.nn.Parameter
perf = time.perf_counter
OBSERVATION_SECTIONS = (
    "arg_scan",
    "call_site",
    "module_stack",
    "autograd",
    "output_facts",
    "payload",
)


class _Probe:
    """Today's call-time helpers, run as observations; patched in only while measuring."""

    def __init__(self) -> None:
        import torchlens._capture.run as run
        import torchlens.backends.torch.wrappers as wrappers
        from torchlens import _state
        from torchlens._capture.arg_positions import _normalize_func_name
        from torchlens._capture_state_helpers import _is_uninitialized_param
        from torchlens._utils._internal_read import internal_scalar_read
        from torchlens._utils._tensor_label import get_tensor_label
        from torchlens._utils._torch_compat import tensor_version_or_none
        from torchlens._utils.introspection import (
            _get_code_context,
            get_arg_tensors_for_resolution,
        )
        from torchlens._utils.tensor_utils import get_memory_amount_from_metadata
        from torchlens.backends.torch._ops_activations import (
            _metadata_shape,
            _save_activation_fields,
        )
        from torchlens.backends.torch._ops_autograd import (
            _partition_output_entries_with_autograd_stats,
        )
        from torchlens.backends.torch._ops_capture_records import (
            _snapshot_exhaustive_module_stack,
        )
        from torchlens.backends.torch._ops_exhaustive import _output_should_be_logged
        from torchlens.backends.torch._ops_predicate_events import (
            _extract_arg_tensors_and_params,
        )
        from torchlens.backends.torch._user_grad_fn import recorded_op_grad_fn
        from torchlens.backends.torch.buffer_writes import resolve_or_late_index_buffer_address
        from torchlens.backends.torch.sources import (
            log_closure_constant_if_known,
            log_source_tensor,
        )

        self.run, self.wrappers, self.state = run, wrappers, _state
        self.normalize, self.uninitialized = _normalize_func_name, _is_uninitialized_param
        self.scalar_read, self.label, self.version = (
            internal_scalar_read,
            get_tensor_label,
            tensor_version_or_none,
        )
        self.code_context, self.resolution = _get_code_context, get_arg_tensors_for_resolution
        self.memory, self.shape, self.save = (
            get_memory_amount_from_metadata,
            _metadata_shape,
            _save_activation_fields,
        )
        self.partition, self.modules = (
            _partition_output_entries_with_autograd_stats,
            _snapshot_exhaustive_module_stack,
        )
        self.should_log, self.extract = _output_should_be_logged, _extract_arg_tensors_and_params
        self.grad_fn, self.buffer_address = (
            recorded_op_grad_fn,
            resolve_or_late_index_buffer_address,
        )
        self.closure_constant, self.source = log_closure_constant_if_known, log_source_tensor
        self.real_run_forward = run._run_forward
        self.real_emit = wrappers.log_function_output_tensors
        self.bound_shadow_emit = self.shadow_emit  # one object, so the swap back finds it
        self.forward_times: list[float] = []
        self.sections = [0.0] * len(OBSERVATION_SECTIONS)
        self.split = [0.0, 0.0]  # observations, today's emit
        self.notes: dict[str, str] = {}

    def timed_run_forward(self, *args: Any, **kwargs: Any) -> Any:
        """``_run_forward`` with its wall time appended to ``forward_times``."""

        start = perf()
        try:
            return self.real_run_forward(*args, **kwargs)
        finally:
            self.forward_times.append(perf() - start)

    def _see(self, key: Any, value: Any, seen: dict[str, Any]) -> None:
        """Parent, argument-slot and parameter observations of one argument value."""

        if isinstance(value, P):
            meta = getattr(value, "_tl", None)
            barcode = getattr(meta, "param_barcode", None)
            seen["params"].append((key, id(value), tuple(value.shape), barcode))
        elif isinstance(value, T):
            label = self.label(value)
            if label is None:
                seen["unlabelled"].append((key, tuple(value.shape), value.dtype, value.device))
            else:
                seen["positions"][key] = label
                seen["parents"].append(label)
                seen["versions"][label] = self.version(value)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, T):
                    self._see((key, index), item, seen)

    def _arguments(self, trace: Any, layer_type: str, args: tuple, kwargs: dict) -> dict:
        """Argument scan with source resolution, as today's helpers do it."""

        self.extract(layer_type, args, kwargs)
        for tensor in self.resolution(args, kwargs):
            if isinstance(tensor, P) or self.label(tensor) is not None:
                continue
            address = self.buffer_address(trace, tensor)
            if address is not None:
                self.source(trace, tensor, "buffer", address)
            else:
                self.closure_constant(trace, tensor)
        seen: dict[str, Any] = {
            "positions": {},
            "parents": [],
            "versions": {},
            "unlabelled": [],
            "params": [],
        }
        with self.scalar_read():
            for index, value in enumerate(args):
                self._see(index, value, seen)
            for key, value in kwargs.items():
                self._see(key, value, seen)
        seen["witness"] = self.state.completeness_witness_is_shadow()
        return seen

    def _payload(self, trace: Any, out: T, label: str, raw_index: int, facts: tuple) -> Any:
        """The saved activation through the product's own save path, on a scratch dict."""

        to_save = trace._layer_nums_to_save
        if not (to_save == "all" or raw_index in to_save):
            return None
        func_name, is_inplace, shape, dtype, memory, arg_copies, kwarg_copies = facts
        scratch = {
            "func_name": func_name,
            "is_inplace": is_inplace,
            "output_device": trace._options_output_device,
            "detach_saved_activations": trace._options_detach_saved_activations,
            "annotations": {},
            "_label_raw": label,
            "_layer_label_raw": label,
            "raw_index": raw_index,
            "shape": shape,
            "dtype": dtype,
            "activation_memory": memory,
            "out": None,
            "transformed_out": None,
            "transformed_out_shape": None,
            "transformed_out_dtype": None,
            "transformed_activation_memory": None,
            "has_saved_activation": False,
            "has_saved_args": False,
            "saved_args": None,
            "saved_kwargs": None,
            "activation_transform": trace._options_activation_transform,
        }
        if "payload_fallback" not in self.notes:
            try:
                transform = trace._options_activation_transform
                self.save(trace, scratch, out, arg_copies, kwarg_copies, transform)
                return scratch
            except Exception:  # noqa: BLE001 - recorded; the fallback is a plain clone
                self.notes["payload_fallback"] = traceback.format_exc()[-600:]
        with self.scalar_read():
            scratch["out"] = out.detach().clone()
        return scratch

    def _output(self, trace: Any, out: T, label: str, raw_index: int, call: tuple) -> None:
        """Facts and payload of one output; adds to the output and payload sections."""

        start = perf()
        is_inplace = self.label(out) is not None
        with self.scalar_read():
            grad_fn = self.grad_fn(None, out)
            if out.dtype == torch.bool and out.ndim == 0:
                out.item()
        grad_cls = type(grad_fn) if grad_fn is not None else None
        _ = None if grad_cls is None else f"{grad_cls.__module__}.{grad_cls.__qualname__}"
        shape = self.shape(out, op_label=label)
        memory = self.memory(out, shape, out.dtype)
        _ = (out.device, getattr(getattr(out, "_tl", None), "address", None))
        middle = perf()
        func_name, arg_copies, kwarg_copies = call
        facts = (func_name, is_inplace, shape, out.dtype, memory, arg_copies, kwarg_copies)
        self._payload(trace, out, label, raw_index, facts)
        self.sections[4] += middle - start
        self.sections[5] += perf() - middle

    def observe(self, trace: Any, func: Any, func_name: str, args: tuple, kwargs: dict,
                arg_copies: Any, kwarg_copies: Any, out_orig: Any, exec_ctx: Any,
                bottom: bool, cid: int) -> None:  # fmt: skip
        """Every call-time observation of one call, consuming and stamping nothing."""

        if isinstance(out_orig, T) and (
            self.uninitialized(out_orig) or not self.should_log(out_orig, bottom)
        ):
            return
        t0 = perf()
        layer_type = self.normalize(func_name)
        self._arguments(trace, layer_type, args, kwargs)
        t1 = perf()
        cache = getattr(trace, "_code_context_cache", None)
        if cache is None:
            cache = {}
            trace._code_context_cache = cache
        self.code_context(
            trace.num_context_lines,
            source_loading_enabled=trace._options_save_source_context,
            disable_col_offset=False,
            context_cache=cache,
            call_site_out=[],
        )
        t2 = perf()
        self.modules(trace)
        t3 = perf()
        entries = self.partition(trace, out_orig)
        t4 = perf()
        for index, value in enumerate((t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            self.sections[index] += value
        workspace = trace._raw_graph_ws
        ahead = 0
        for entry in entries:
            out = entry.value
            if not self.should_log(out, bottom):
                continue
            ahead += 1  # today's emit, which runs next, advances the counters
            raw_index = workspace.layer_counter + ahead
            type_index = workspace.raw_layer_type_counter[layer_type] + ahead
            label = f"{layer_type}_{type_index}_{raw_index}_raw"
            self._output(trace, out, label, raw_index, (func_name, arg_copies, kwarg_copies))

    def shadow_emit(self, *args: Any) -> Any:
        """Observations first, then today's emit; both timed."""

        start = perf()
        self.observe(*args)
        middle = perf()
        try:
            return self.real_emit(*args)
        finally:
            self.split[0] += middle - start
            self.split[1] += perf() - middle

    def traces(self, model: Any, x: Any, shadow: bool, repeats: int) -> list[tuple]:
        """Trace ``repeats`` times; return ``(forward, observations, emit, sections, total)``."""

        self.run._run_forward = self.timed_run_forward
        if shadow:
            swapped = swap_emit(self.real_emit, self.bound_shadow_emit)
            self.notes["emit_bindings"] = ", ".join(swapped)
        rows = []
        try:
            for _ in range(repeats):
                self.forward_times.clear()
                self.sections[:] = [0.0] * len(OBSERVATION_SECTIONS)
                self.split[:] = [0.0, 0.0]
                gc.collect()
                start = perf()
                tl.trace(model, x)
                total = perf() - start
                forward = self.forward_times[-1]
                rows.append((forward, self.split[0], self.split[1], list(self.sections), total))
        finally:
            self.run._run_forward = self.real_run_forward
            swap_emit(self.bound_shadow_emit, self.real_emit)
        return rows


def emit_bindings(emit: Any) -> list[tuple[Any, str]]:
    """Every ``(module, attribute)`` of a loaded TorchLens module whose value is ``emit``."""

    found: list[tuple[Any, str]] = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", None)
        namespace = getattr(module, "__dict__", None)
        if not isinstance(name, str) or not isinstance(namespace, dict):
            continue
        if name == "torchlens" or name.startswith("torchlens."):
            found.extend((module, key) for key, value in list(namespace.items()) if value is emit)
    return found


def swap_emit(current: Any, replacement: Any) -> list[str]:
    """Rebind every TorchLens module attribute bound to ``current`` to ``replacement``.

    Returns the ``module.attribute`` names rebound, sorted.
    """

    bindings = emit_bindings(current)
    for module, key in bindings:
        setattr(module, key, replacement)
    return sorted(f"{module.__name__}.{key}" for module, key in bindings)


def _bare(model: Any, x: Any, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = perf()
        model(x)
        times.append(perf() - start)
    return min(times)


def measure_model(probe: _Probe, name: str, repeats: int) -> dict[str, Any]:
    """Wall figures of one model, microseconds per function output (minimum of ``repeats``)."""

    model, x = cost.build(name)
    tl.trace(model, x)
    tl.trace(model, x)
    _, counted = count_sections(lambda: tl.trace(model, x))
    outputs = counted["outputs"]
    us = 1e6 / outputs
    production = probe.traces(model, x, False, repeats)
    shadow = probe.traces(model, x, True, repeats)
    best = min(shadow, key=lambda row: row[0] - row[2])
    less_emit = best[0] - best[2]
    sections = {key: value * us for key, value in zip(OBSERVATION_SECTIONS, best[3], strict=True)}
    return {
        "outputs": outputs,
        "whole_trace_ms": min(row[4] for row in production) * 1e3,
        "whole_trace_us_per_output": min(row[4] for row in production) * us,
        "forward_us_per_output": min(row[0] for row in production) * us,
        "forward_less_todays_emit_us_per_output": less_emit * us,
        "observations_us_per_output": best[1] * us,
        "todays_emit_us_per_output": best[2] * us,
        "sections_us_per_output": sections,
        "forward_outside_us_per_output": (less_emit - best[1]) * us,
        "shadow_whole_trace_ms": min(row[4] for row in shadow) * 1e3,
        "bare_us_per_output": _bare(model, x, repeats) * us,
    }


def run(names: tuple[str, ...], repeats: int = 9) -> dict[str, Any]:
    """Measure every named model in this process; one thread, warnings silenced."""

    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    models: dict[str, Any] = {}
    probe: _Probe | None = None
    for name in names:
        try:
            if probe is None:
                tl.trace(*cost.build(name))  # installs the wrappers before the probe binds
                probe = _Probe()
            models[name] = measure_model(probe, name, repeats)
        except Exception as error:  # noqa: BLE001 - one model must not void the rest
            models[name] = {"error": f"{type(error).__name__}: {error}"}
            models[name]["traceback"] = traceback.format_exc()[-1500:]
        print(f"wall {name}: {'error' if 'error' in models[name] else 'ok'}", flush=True)
    return {
        "tool": "capture_core_floor_shadow",
        "repeats": repeats,
        "environment": {**cost.environment(), "machine": platform.machine()},
        "models": models,
        "notes": {} if probe is None else probe.notes,
    }
