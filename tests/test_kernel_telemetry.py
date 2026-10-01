"""L3 trailing-lane gates for detachable CUDA kernel telemetry."""

from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from _source_corpus import module_ast
from torch import nn

import torchlens as tl
from torchlens import kernel_telemetry as telemetry
from torchlens._io import FieldPolicy
from torchlens._io.prerelease import registered_prerelease_fields
from torchlens.kernel_telemetry import KernelLaunch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TELEMETRY_MODULE = "torchlens.kernel_telemetry"
_REAL_CUDA_ROWS = (
    "one ATen call to many kernels",
    "several ATen calls to one fused launch",
    "multiple streams",
    "asynchronous launches",
    "memory copies",
    "profiler warmup and schedule behavior",
    "capture exception teardown",
)
_DOCUMENTED_UNSTABLE_TOKENS = {
    "KernelLaunch",
    "AtenOp.gpu_kernels",
    "Op.gpu_kernels",
    "launch_name",
    "device",
    "stream",
    "duration",
    "runtime_correlation",
    "attribution_status",
    "attributed",
    "unavailable",
    "ambiguous",
    "unattributed",
}


class _CudaTelemetryModel(nn.Module):
    """Small matrix model that launches CUDA work when placed on CUDA."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run matrix multiplication followed by a pointwise operation."""

        return torch.relu(value @ value)


@pytest.mark.smoke
def test_documented_unstable_kernel_surface_matches_glossary_index() -> None:
    """Every telemetry spelling has the exact no-shim unstable marker."""

    glossary = (_REPO_ROOT / "docs/reference/glossary.md").read_text(encoding="utf-8")
    indexed = glossary.split("<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:START -->", 1)[1].split(
        "<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:END -->", 1
    )[0]
    assert set(re.findall(r"`([^`]+)`", indexed)) == _DOCUMENTED_UNSTABLE_TOKENS
    surface_rows = [line for line in indexed.splitlines() if line.startswith("|")][2:]
    assert surface_rows
    assert all("unstable -- no deprecation shim owed" in row for row in surface_rows)


def _event(
    name: str,
    category: str,
    timestamp: float,
    duration: float,
    *,
    correlation: int | None = None,
    stream: int | None = None,
    device: int | None = None,
    phase: str = "X",
) -> dict[str, object]:
    """Build one compact synthetic Chrome-trace event.

    Parameters
    ----------
    name
        Event name.
    category
        Exact Chrome category.
    timestamp
        Event start coordinate.
    duration
        Event duration coordinate.
    correlation
        Optional Kineto runtime correlation id.
    stream
        Optional CUDA stream id.
    device
        Optional CUDA device id.
    phase
        Chrome event phase.

    Returns
    -------
    dict[str, object]
        Synthetic event mapping.
    """

    args: dict[str, object] = {}
    if correlation is not None:
        args["correlation"] = correlation
    if stream is not None:
        args["stream"] = stream
    if device is not None:
        args["device"] = device
    return {
        "name": name,
        "cat": category,
        "ph": phase,
        "ts": timestamp,
        "dur": duration,
        "pid": 1,
        "tid": 2,
        "args": args,
    }


def _synthetic_matrix() -> tuple[list[Mapping[str, object]], dict[str, int]]:
    """Return a synthetic event graph covering the required correlation shapes.

    Returns
    -------
    tuple[list[Mapping[str, object]], dict[str, int]]
        Chrome events and marker-to-primitive sequence mapping.
    """

    marker_a = "torchlens::aten::1"
    marker_b = "torchlens::aten::2"
    events: list[Mapping[str, object]] = [
        _event(marker_a, "user_annotation", 10, 10),
        _event(marker_b, "user_annotation", 30, 10),
        # Correlation 101 occurs inside A and launches two asynchronous rows.
        _event("cudaLaunchKernel", "cuda_runtime", 12, 1, correlation=101),
        _event("kernel_a", "kernel", 60, 4, correlation=101, stream=7, device=0),
        _event("copy_a", "gpu_memcpy", 70, 2, correlation=101, stream=8, device=0),
        # Correlation 202 is observed in both markers: one fused launch has
        # an exact many-to-many relation, not a name-derived guess.
        _event("cudaLaunchKernel", "cuda_runtime", 16, 1, correlation=202),
        _event("cudaLaunchKernel", "cuda_runtime", 34, 1, correlation=202),
        _event("fused_kernel", "kernel", 80, 5, correlation=202, stream=9, device=0),
        # Profiler warmup/schedule residue without a retained marker is kept
        # as unattributed evidence rather than silently discarded.
        _event("warmup_kernel", "kernel", 2, 1, correlation=303, stream=6, device=0),
    ]
    return events, {marker_a: 11, marker_b: 22}


@pytest.mark.smoke
def test_synthetic_kineto_join_covers_required_async_and_many_to_many_shapes() -> None:
    """Runtime correlation, not names or timestamps, owns device attribution."""

    events, markers = _synthetic_matrix()
    payload = telemetry._payload_from_chrome_events(events, markers, telemetry_available=True)

    assert payload._available
    assert [launch.launch_name for launch in payload._launches] == [
        "kernel_a",
        "copy_a",
        "fused_kernel",
        "warmup_kernel",
    ]
    assert [launch.stream for launch in payload._launches] == [7, 8, 9, 6]
    assert payload._relations == ((11, 0), (11, 1), (11, 2), (22, 2))
    assert payload._launches[0].attribution_status == "attributed"
    assert payload._launches[-1].attribution_status == "unattributed"
    assert payload._launches[0].duration == 4.0


@pytest.mark.smoke
def test_marker_begin_end_and_capture_exception_shape_close_cleanly() -> None:
    """A begin/end marker still correlates when capture exits exceptionally."""

    marker = "torchlens::aten::exception"
    begin = _event(marker, "user_annotation", 5, 0, phase="B")
    end = _event(marker, "user_annotation", 9, 0, phase="E")
    runtime = _event("cudaLaunchKernel", "cuda_runtime", 7, 1, correlation=44)
    kernel = _event("kernel", "kernel", 20, 1, correlation=44, stream=1, device=0)
    payload = telemetry._payload_from_chrome_events(
        [begin, runtime, end, kernel], {marker: 4}, telemetry_available=True
    )
    assert payload._relations == ((4, 0),)
    assert payload._launches[0].attribution_status == "attributed"


@pytest.mark.smoke
def test_unavailable_session_is_typed_and_never_fabricates_a_launch() -> None:
    """Unavailable CUDA produces one fact-free disclosure, never a zero claim."""

    payload = telemetry._payload_from_chrome_events(
        (), {"torchlens::aten::1": 3, "torchlens::aten::2": 5}, telemetry_available=False
    )
    assert not payload._available
    assert payload._relations == ((3, 0), (5, 0))
    assert payload._launches == (
        KernelLaunch(
            launch_name=None,
            device=None,
            stream=None,
            duration=None,
            runtime_correlation=None,
            attribution_status="unavailable",
        ),
    )


@pytest.mark.smoke
def test_scoped_marker_instrumentation_restores_exact_core_functions() -> None:
    """The adapter restores both monkeypatched seams after a capture exception."""

    from torchlens.backends.torch import _aten_capture

    prepare = _aten_capture._prepare_aten_call
    finish = _aten_capture._finish_aten_call
    with pytest.raises(RuntimeError, match="planned"), telemetry._instrument_aten_markers():
        assert _aten_capture._prepare_aten_call is not prepare
        assert _aten_capture._finish_aten_call is not finish
        raise RuntimeError("planned")
    assert _aten_capture._prepare_aten_call is prepare
    assert _aten_capture._finish_aten_call is finish


@pytest.mark.smoke
def test_cpu_host_records_unavailable_views_and_drop_gated_annotation(tmp_path: Path) -> None:
    """CPU capture discloses NOT-RUN; plain v8 saves persist the telemetry section."""

    if torch.cuda.is_available():
        pytest.skip("CPU-only NOT-RUN disclosure row applies only without CUDA")

    trace = telemetry._profile_trace_with_cuda_kernels(
        lambda: tl.trace(_CudaTelemetryModel(), torch.ones(2, 2))
    )
    primitive_rows = trace._primitive_op_profile.primitive_ops
    assert primitive_rows
    assert all(row.gpu_kernels[0].attribution_status == "unavailable" for row in primitive_rows)
    assert trace.ops[0].gpu_kernels[0].attribution_status == "unavailable"
    assert set(_REAL_CUDA_ROWS) == {
        "one ATen call to many kernels",
        "several ATen calls to one fused launch",
        "multiple streams",
        "asynchronous launches",
        "memory copies",
        "profiler warmup and schedule behavior",
        "capture exception teardown",
    }

    default_path = tmp_path / "default.tlspec"
    tl.save(trace, default_path)
    loaded = tl.load(default_path)
    # tlspec v8: the telemetry annotation section persists on a plain save.
    assert "_kernel_telemetry" in loaded.annotations
    telemetry._bind_trace_telemetry(loaded)
    assert loaded.ops[0].gpu_kernels[0].attribution_status == "unavailable"


@pytest.mark.smoke
def test_kernel_rows_persist_and_registrations_are_retired() -> None:
    """tlspec v8: telemetry rows declare KEEP directly; registrations retired."""

    inventory = registered_prerelease_fields()
    assert "KernelLaunch" not in inventory
    assert "_TelemetryPayload" not in inventory
    # v8 telemetry keys retired; the standing row is C01's gated sidecar key.
    assert inventory.get("Trace.annotations", ()) in ((), ("sidecar",))
    assert set(KernelLaunch.PORTABLE_STATE_SPEC.values()) == {FieldPolicy.KEEP}
    assert set(telemetry._TelemetryPayload.PORTABLE_STATE_SPEC.values()) == {FieldPolicy.KEEP}


def _imported_modules(path: Path) -> set[str]:
    """Return absolute import targets in one Python file.

    Parameters
    ----------
    path
        Python source file.

    Returns
    -------
    set[str]
        Imported module names.
    """

    tree = module_ast(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.heavy
def test_kernel_telemetry_lane_is_detachable_from_package_and_other_tests() -> None:
    """No package module or non-governance test imports the shed-lane module."""

    package_offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(_imported_modules(path))
        for path in (_REPO_ROOT / "torchlens").rglob("*.py")
        if path.name != "kernel_telemetry.py" and _TELEMETRY_MODULE in _imported_modules(path)
    }
    test_offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(_imported_modules(path))
        for path in (_REPO_ROOT / "tests").rglob("*.py")
        if path.name
        not in {
            Path(__file__).name,
            "test_legacy_artifact.py",
            "test_prebump_roundtrip_acceptance.py",
            "test_prerelease_registrar.py",
        }
        and _TELEMETRY_MODULE in _imported_modules(path)
    }
    assert not package_offenders
    assert not test_offenders
    assert importlib.import_module(_TELEMETRY_MODULE) is telemetry


@pytest.mark.heavy
@pytest.mark.optional
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/CUPTI matrix NOT-RUN-DISCLOSED")
def test_real_cuda_cupti_correlation_matrix() -> None:
    """Run the real CUDA/CUPTI correlation gate when a device leg exists."""

    device = torch.device("cuda")
    trace = telemetry._profile_trace_with_cuda_kernels(
        lambda: tl.trace(_CudaTelemetryModel().to(device), torch.ones(32, 32, device=device))
    )
    launches = [
        launch
        for row in trace._primitive_op_profile.primitive_ops
        for launch in row.gpu_kernels
        if launch.attribution_status == "attributed"
    ]
    assert launches
    assert all(launch.runtime_correlation is not None for launch in launches)
    assert all(launch.duration is not None and launch.duration >= 0 for launch in launches)
