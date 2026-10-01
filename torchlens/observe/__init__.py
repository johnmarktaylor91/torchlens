"""Observe kit (megasprint lane F24): memory truth + diagnostics substrates.

Built on TorchLens's OWN captured records (the sibling ``torchlens.debug``
kit holds the tool-shaped diagnostics: ``check_determinism``,
``bisect_nan_backward``, the isolated-rerun harness). This package owns:

- the pass-level-peak publication gate (``_peaks``): no surface prints or
  divides by ``forward_peak_memory`` / ``backward_peak_memory`` without
  checking the recorded measurement backend;
- the categorized, module-contained memory timeline v2 artifact
  (``_timeline``), its deterministic dependency-free SVG renderer
  (``_svg``), and the opt-in estimated liveness view (``_liveness``);
- the device-memory sampling provider interface behind
  ``CaptureOptions(track_device_memory=...)`` (``_device_memory``).

Access spelling is ``import torchlens.observe`` for now; root-facade routing
is an F35 registration fragment. Every spelling here is DOCUMENTED-UNSTABLE
pending naming-session ratification.
"""

from __future__ import annotations

from ._device_memory import (
    CudaAllocatorProvider,
    DeviceMemoryProvider,
    DeviceMemoryReading,
    DeviceMemorySample,
    device_memory_samples,
)
from ._liveness import DEATH_RULES, estimated_liveness
from ._peaks import PassPeakFacts, format_pass_peak, pass_peak_facts
from ._svg import render_timeline_svg, write_timeline_svg
from ._timeline import SCHEMA_ID, memory_timeline_v2, module_rollup

__all__ = [
    "CudaAllocatorProvider",
    "DEATH_RULES",
    "DeviceMemoryProvider",
    "DeviceMemoryReading",
    "DeviceMemorySample",
    "PassPeakFacts",
    "SCHEMA_ID",
    "device_memory_samples",
    "estimated_liveness",
    "format_pass_peak",
    "memory_timeline_v2",
    "module_rollup",
    "pass_peak_facts",
    "render_timeline_svg",
    "write_timeline_svg",
]
