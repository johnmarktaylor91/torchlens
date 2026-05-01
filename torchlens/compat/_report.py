"""Runtime compatibility reporting for model/input pairs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import inspect
import multiprocessing
import threading
from typing import Any, Literal

import torch
from torch import nn

Status = Literal["pass", "known_broken", "scope", "not_tested"]
Severity = Literal["ok", "info", "warning", "error"]


@dataclass(frozen=True)
class CompatRow:
    """One row in the TorchLens compatibility truth table.

    Parameters
    ----------
    key:
        Stable machine-readable row key.
    label:
        Human-readable row label.
    status:
        Compatibility status for this model/input pair.
    severity:
        User-facing severity for the row.
    detected:
        Whether the row's condition was detected in the supplied model/input.
    details:
        Explanation of the finding.
    suggestion:
        Suggested workaround or next action.
    """

    key: str
    label: str
    status: Status
    severity: Severity
    detected: bool
    details: str
    suggestion: str = ""


@dataclass(frozen=True)
class CompatReport:
    """Compatibility report returned by :func:`torchlens.compat.report`.

    Parameters
    ----------
    model_type:
        Qualified type name for the inspected model.
    torch_version:
        PyTorch version used for the inspection.
    rows:
        Truth-table rows for the inspected model/input pair.
    """

    model_type: str
    torch_version: str
    rows: tuple[CompatRow, ...]

    def row(self, key: str) -> CompatRow:
        """Return one report row by stable key.

        Parameters
        ----------
        key:
            Row key to look up.

        Returns
        -------
        CompatRow
            Matching report row.

        Raises
        ------
        KeyError
            If ``key`` is not present in this report.
        """

        for report_row in self.rows:
            if report_row.key == key:
                return report_row
        raise KeyError(key)

    def to_markdown(self) -> str:
        """Render this report as a GitHub-flavored Markdown table.

        Returns
        -------
        str
            Markdown representation of the report.
        """

        lines = [
            f"### TorchLens compatibility report for `{self.model_type}`",
            "",
            f"- PyTorch: `{self.torch_version}`",
            f"- Rows: {len(self.rows)}",
            "",
            "| Row | Status | Severity | Detected | Details | Suggestion |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(row.label),
                        f"`{row.status}`",
                        f"`{row.severity}`",
                        "yes" if row.detected else "no",
                        _escape_markdown(row.details),
                        _escape_markdown(row.suggestion),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def show(self) -> str:
        """Render this report as a fixed-width text table.

        Returns
        -------
        str
            Text table suitable for terminals and notebook display.
        """

        headers = ("Row", "Status", "Severity", "Detected", "Details")
        body = [
            (
                row.label,
                row.status,
                row.severity,
                "yes" if row.detected else "no",
                row.details,
            )
            for row in self.rows
        ]
        widths = _column_widths([headers, *body])
        lines = [
            f"TorchLens compatibility report for {self.model_type}",
            f"PyTorch: {self.torch_version}",
            "",
            _format_table_line(headers, widths),
            _format_table_line(tuple("-" * width for width in widths), widths),
        ]
        lines.extend(_format_table_line(row, widths) for row in body)
        return "\n".join(lines)


def report(model: nn.Module, input: Any) -> CompatReport:  # noqa: A002
    """Probe a model/input pair against TorchLens compatibility rows.

    Parameters
    ----------
    model:
        Model or wrapper to inspect.
    input:
        Example model input. The report inspects the input tree but does not
        execute ``model(input)``.

    Returns
    -------
    CompatReport
        Structured compatibility report.
    """

    rows = (
        _hf_transformers_row(model),
        _accelerate_dispatch_row(model),
        _accelerate_offload_row(model),
        _bitsandbytes_row(model),
        _tied_parameters_row(model),
        _multi_gpu_rng_row(),
        _data_parallel_row(model),
        _ddp_row(model),
        _fsdp_row(model),
        _deepspeed_row(model),
        _torch_compile_row(model),
        _fx_row(model),
        _lightning_row(model),
        _functorch_row(model),
        _quantized_row(model, input),
        _device_context_row(),
        _single_thread_row(),
    )
    return CompatReport(
        model_type=_qualified_type_name(model),
        torch_version=str(torch.__version__),
        rows=rows,
    )


def _escape_markdown(value: str) -> str:
    """Escape table separators in Markdown cells.

    Parameters
    ----------
    value:
        Cell value.

    Returns
    -------
    str
        Escaped cell value.
    """

    return value.replace("|", "\\|").replace("\n", " ")


def _column_widths(rows: Sequence[Sequence[str]]) -> tuple[int, ...]:
    """Compute fixed-width table column sizes.

    Parameters
    ----------
    rows:
        Table rows.

    Returns
    -------
    tuple[int, ...]
        Width for each column.
    """

    column_count = len(rows[0])
    return tuple(max(len(row[index]) for row in rows) for index in range(column_count))


def _format_table_line(row: Sequence[str], widths: Sequence[int]) -> str:
    """Format one fixed-width table row.

    Parameters
    ----------
    row:
        Cell values.
    widths:
        Column widths.

    Returns
    -------
    str
        Formatted row.
    """

    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _qualified_type_name(value: Any) -> str:
    """Return a stable qualified type name.

    Parameters
    ----------
    value:
        Object to identify.

    Returns
    -------
    str
        ``module.qualname`` for ``value``'s type.
    """

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _iter_tensors(value: Any, seen: set[int] | None = None) -> Iterator[torch.Tensor]:
    """Yield tensors from a nested input tree.

    Parameters
    ----------
    value:
        Object that may contain tensors.
    seen:
        Object ids already visited.

    Yields
    ------
    torch.Tensor
        Tensors reachable through builtin containers.
    """

    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item, seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_tensors(item, seen)


def _iter_modules(model: nn.Module) -> Iterable[nn.Module]:
    """Yield modules from ``model`` with a defensive fallback.

    Parameters
    ----------
    model:
        Module to inspect.

    Returns
    -------
    Iterable[nn.Module]
        Module iterator.
    """

    try:
        return tuple(model.modules())
    except Exception:
        return (model,)


def _class_identity(value: Any) -> str:
    """Return lowercase class/module identity text for heuristic checks.

    Parameters
    ----------
    value:
        Object to identify.

    Returns
    -------
    str
        Lowercase qualified type identity.
    """

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}".lower()


def _model_class_contains(model: nn.Module, needles: Sequence[str]) -> bool:
    """Return whether a model class identity contains any needle.

    Parameters
    ----------
    model:
        Model to inspect.
    needles:
        Lowercase substrings to search for.

    Returns
    -------
    bool
        True if any needle matches.
    """

    identity = _class_identity(model)
    return any(needle in identity for needle in needles)


def _hf_transformers_row(model: nn.Module) -> CompatRow:
    """Build the Hugging Face Transformers wrapper row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _model_class_contains(model, ("transformers.", "pretrainedmodel")) or hasattr(
        model, "config"
    )
    details = (
        "Hugging Face-style module detected; eager forward capture is supported when the "
        "model is not compiled, offloaded, or sharded."
        if detected
        else "No Hugging Face Transformers wrapper detected."
    )
    suggestion = (
        "Use torchlens.compat.from_huggingface for offline-first loading when helpful."
        if detected
        else ""
    )
    return CompatRow(
        "hf_transformers",
        "HF Transformers wrapper",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        suggestion,
    )


def _accelerate_dispatch_row(model: nn.Module) -> CompatRow:
    """Build the Accelerate device-map dispatch row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    device_map = getattr(model, "hf_device_map", None)
    detected = bool(device_map)
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Accelerate device_map dispatch detected; parameters may materialize on different "
        "devices during forward, so TorchLens cannot validate a single coherent eager trace."
        if detected
        else "No Accelerate device_map dispatch detected."
    )
    return CompatRow(
        "accelerate_device_map_auto",
        "Accelerate device_map='auto'",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Run on a single materialized device before logging." if detected else "",
    )


def _accelerate_offload_row(model: nn.Module) -> CompatRow:
    """Build the Accelerate CPU/disk offload row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = False
    for module in _iter_modules(model):
        hook = getattr(module, "_hf_hook", None)
        if hook is not None and (
            bool(getattr(hook, "offload", False)) or getattr(hook, "execution_device", None)
        ):
            detected = True
            break
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Accelerate CPU/disk offload hooks detected; lazy parameter movement can bypass "
        "TorchLens' assumptions about tensor identity and device placement."
        if detected
        else "No Accelerate CPU/disk offload hooks detected."
    )
    return CompatRow(
        "accelerate_cpu_disk_offload",
        "Accelerate CPU/disk offload",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Disable offload or log a fully materialized copy." if detected else "",
    )


def _bitsandbytes_row(model: nn.Module) -> CompatRow:
    """Build the bitsandbytes quantization row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = bool(getattr(model, "is_loaded_in_8bit", False)) or bool(
        getattr(model, "is_loaded_in_4bit", False)
    )
    if not detected:
        detected = any("bitsandbytes" in _class_identity(module) for module in _iter_modules(model))
    status: Status = "known_broken" if detected else "pass"
    details = (
        "bitsandbytes 8-bit/4-bit modules detected; custom parameter wrappers and kernels "
        "are outside TorchLens' dense eager tensor contract."
        if detected
        else "No bitsandbytes 8-bit/4-bit modules detected."
    )
    return CompatRow(
        "bitsandbytes_8bit_4bit",
        "bitsandbytes 8-bit/4-bit",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Log an unquantized reference model when exact activation metadata is required."
        if detected
        else "",
    )


def _tied_parameters_row(model: nn.Module) -> CompatRow:
    """Build the tied/shared parameter row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    seen: dict[int, str] = {}
    duplicates: list[str] = []
    try:
        named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:
        named_parameters = tuple(model.named_parameters())
    except Exception:
        named_parameters = ()
    for name, parameter in named_parameters:
        param_id = id(parameter)
        if param_id in seen:
            duplicates.append(f"{seen[param_id]}={name}")
        else:
            seen[param_id] = name
    detected = bool(duplicates)
    details = (
        "Shared parameter objects detected; TorchLens tracks parameter identity and should "
        f"preserve tied-edge metadata ({', '.join(duplicates[:3])})."
        if detected
        else "No tied/shared parameter objects detected."
    )
    return CompatRow(
        "tied_parameters",
        "Tied/shared parameters",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _multi_gpu_rng_row() -> CompatRow:
    """Build the multi-GPU RNG row.

    Returns
    -------
    CompatRow
        Report row.
    """

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    detected = device_count > 1
    details = (
        f"{device_count} CUDA devices visible; TorchLens snapshots/restores RNG state for all "
        "CUDA devices via torch.cuda.get_rng_state_all()."
        if detected
        else f"{device_count} CUDA device(s) visible; multi-GPU RNG replay was not exercised."
    )
    return CompatRow(
        "multi_gpu_rng",
        "Multi-GPU RNG",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _data_parallel_row(model: nn.Module) -> CompatRow:
    """Build the ``nn.DataParallel`` row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = isinstance(model, nn.DataParallel)
    status: Status = "known_broken" if detected else "pass"
    details = (
        "nn.DataParallel detected. TorchLens' process-global capture state is incompatible "
        "with threaded replica execution; unwrap model.module before logging."
        if detected
        else "nn.DataParallel not detected."
    )
    return CompatRow(
        "data_parallel",
        "nn.DataParallel",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Call torchlens.log_forward_pass(model.module, x)." if detected else "",
    )


def _ddp_row(model: nn.Module) -> CompatRow:
    """Build the DistributedDataParallel row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _model_class_contains(model, ("distributeddataparallel",))
    details = (
        "DistributedDataParallel detected; TorchLens unwraps the rank-local .module and "
        "captures that eager module."
        if detected
        else "DistributedDataParallel not detected."
    )
    return CompatRow(
        "distributed_data_parallel",
        "DistributedDataParallel",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _fsdp_row(model: nn.Module) -> CompatRow:
    """Build the FSDP row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _model_class_contains(model, ("fullyshardeddataparallel", "fsdp"))
    status: Status = "scope" if detected else "pass"
    details = (
        "FSDP detected; sharded parameter materialization is outside TorchLens' launch scope."
        if detected
        else "FSDP not detected."
    )
    return CompatRow(
        "fsdp",
        "FSDP",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log a rank-local unsharded copy before FSDP wrapping." if detected else "",
    )


def _deepspeed_row(model: nn.Module) -> CompatRow:
    """Build the DeepSpeed row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _model_class_contains(model, ("deepspeed", "deepspeedengine"))
    status: Status = "scope" if detected else "pass"
    details = (
        "DeepSpeed engine detected; ZeRO/offload execution is outside TorchLens' launch scope."
        if detected
        else "DeepSpeed not detected."
    )
    return CompatRow(
        "deepspeed",
        "DeepSpeed",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the underlying eager module outside DeepSpeed." if detected else "",
    )


def _torch_compile_row(model: nn.Module) -> CompatRow:
    """Build the ``torch.compile`` row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _model_class_contains(model, ("optimizedmodule", "_dynamo"))
    status: Status = "scope" if detected else "pass"
    details = (
        "torch.compile OptimizedModule detected; compiled graph capture is outside TorchLens' "
        "primary scope."
        if detected
        else "torch.compile wrapper not detected."
    )
    return CompatRow(
        "torch_compile",
        "torch.compile",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the original eager model, or use torchlens.bridge.depyf for compiled-code context."
        if detected
        else "",
    )


def _fx_row(model: nn.Module) -> CompatRow:
    """Build the FX GraphModule row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    graph_module_type = getattr(torch.fx, "GraphModule", None)
    detected = graph_module_type is not None and isinstance(model, graph_module_type)
    status: Status = "scope" if detected else "pass"
    details = (
        "torch.fx.GraphModule detected; TorchLens launch support targets eager nn.Module "
        "execution rather than FX IR parity."
        if detected
        else "FX GraphModule not detected."
    )
    return CompatRow(
        "fx_graph_module",
        "FX GraphModule",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the pre-FX eager module or use torchlens.compat.from_fx for migration help."
        if detected
        else "",
    )


def _lightning_row(model: nn.Module) -> CompatRow:
    """Build the Lightning training-step row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = callable(getattr(model, "training_step", None))
    is_training = bool(getattr(model, "training", False))
    status: Status = "known_broken" if detected and is_training else "pass"
    details = (
        "LightningModule training_step detected while the module is in training mode; mid-loop "
        "trainer capture is not a supported TorchLens entry point."
        if detected and is_training
        else "Lightning training_step not detected in an active training-mode model."
    )
    return CompatRow(
        "lightning_training_step",
        "Lightning training_step mid-loop",
        status,
        "error" if detected and is_training else "ok",
        detected and is_training,
        details,
        "Use torchlens.callbacks.lightning.LayerProfilerCallback or log a plain forward."
        if detected and is_training
        else "",
    )


def _functorch_row(model: nn.Module) -> CompatRow:
    """Build the vmap/functorch row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _forward_source_contains(model, ("vmap", "functorch", "torch.func"))
    status: Status = "known_broken" if detected else "pass"
    details = (
        "forward source references vmap/functorch; TorchLens skips logging inside active "
        "functorch transforms and will produce an incomplete log."
        if detected
        else "No static vmap/functorch marker detected in forward source."
    )
    return CompatRow(
        "vmap_functorch",
        "vmap/functorch",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Log the non-vmap module separately when a complete operation trace is required."
        if detected
        else "",
    )


def _forward_source_contains(model: nn.Module, needles: Sequence[str]) -> bool:
    """Return whether ``model.forward`` source contains any marker.

    Parameters
    ----------
    model:
        Model to inspect.
    needles:
        Source substrings to search for.

    Returns
    -------
    bool
        True if source was available and a marker matched.
    """

    try:
        source = inspect.getsource(model.forward)
    except (OSError, TypeError):
        return False
    source_lower = source.lower()
    return any(needle in source_lower for needle in needles)


def _quantized_row(model: nn.Module, input_value: Any) -> CompatRow:
    """Build the quantized tensor/model row.

    Parameters
    ----------
    model:
        Model to inspect.
    input_value:
        Input tree to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    quantized_input = any(
        getattr(tensor, "is_quantized", False) for tensor in _iter_tensors(input_value)
    )
    quantized_module = any(_is_quantized_module(module) for module in _iter_modules(model))
    detected = quantized_input or quantized_module
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Quantized tensors/modules detected. tensor_nanequal no longer crashes on quantized "
        "tensors, but full quantized-model activation metadata remains best-effort."
        if detected
        else "No quantized tensors or quantized nn modules detected."
    )
    return CompatRow(
        "quantized_tensor",
        "Quantized tensors/modules",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Use a float reference model for bugs involving exact activation validation."
        if detected
        else "",
    )


def _is_quantized_module(module: nn.Module) -> bool:
    """Return whether a module appears to come from PyTorch quantization namespaces.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    bool
        True for quantized/QAT module classes.
    """

    identity = _class_identity(module)
    prefixes = (
        "torch.ao.nn.quantized",
        "torch.nn.quantized",
        "torch.ao.nn.intrinsic.quantized",
        "torch.ao.nn.qat",
        "torch.nn.qat",
    )
    return any(identity.startswith(prefix) for prefix in prefixes)


def _device_context_row() -> CompatRow:
    """Build the DeviceContext factory row.

    Returns
    -------
    CompatRow
        Report row.
    """

    return CompatRow(
        "device_context_factory",
        "DeviceContext factory injection",
        "pass",
        "ok",
        False,
        "Factory functions honor active torch.device(...) contexts during active logging.",
        "",
    )


def _single_thread_row() -> CompatRow:
    """Build the single-thread design row.

    Returns
    -------
    CompatRow
        Report row.
    """

    current_process = multiprocessing.current_process().name
    current_thread = threading.current_thread().name
    detected = current_process != "MainProcess" or current_thread != "MainThread"
    status: Status = "known_broken" if detected else "pass"
    details = (
        f"Running in {current_process}/{current_thread}; TorchLens capture is single-threaded "
        "and process-global."
        if detected
        else "Running in MainProcess/MainThread; this matches TorchLens' single-thread design."
    )
    return CompatRow(
        "single_thread_design",
        "Single-thread design",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Run capture from the main process and main thread." if detected else "",
    )


__all__ = ["CompatReport", "CompatRow", "Severity", "Status", "report"]
