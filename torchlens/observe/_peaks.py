"""The pass-level-peak publication gate (observe item 4).

``Trace.forward_peak_memory`` / ``backward_peak_memory`` are real runtime
measurements with BACKEND-DEPENDENT meaning: CUDA reports the device
allocator peak; CPU reports a process-RSS endpoint delta (non-reproducible --
measured 151.0 -> 113.7 MB on identical runs, and 332 KB while retaining
93.7 MB); MPS reports an allocator delta. No surface may print or divide by
a pass-level peak without checking ``forward_memory_backend`` -- this module
is the ONE formatting door, and the source lint in
``tests/test_observe_kit_peaks.py`` keeps display surfaces routed through it.

Two rules the gate enforces by construction:

- cpu/mps values render LABELED with their source-specific meaning and are
  never called a bare "peak"; they are marked ratio-ineligible (a ratio over
  a non-reproducible RSS delta is noise presented as measurement).
- a CUDA ``0`` renders as "did not exceed the pre-existing high-water mark",
  which is what the counter actually says -- never as "used 0 bytes".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PassPeakFacts", "format_pass_peak", "pass_peak_facts"]


@dataclass(frozen=True)
class PassPeakFacts:
    """Typed facts about one pass-level peak measurement.

    Parameters
    ----------
    which:
        ``"forward"`` or ``"backward"``.
    backend:
        The recorded measurement backend (``"cuda"`` / ``"cpu"`` / ``"mps"``
        / ``"unknown"`` / ``None``).
    value_bytes:
        Raw measured value in bytes, or ``None`` when never measured.
    meaning:
        What the number IS on this backend (never the bare word "peak" for
        cpu/mps).
    ratio_eligible:
        Whether the value may participate in any derived ratio or division.
        ``True`` ONLY for a positive CUDA device peak.
    caveat:
        The mandatory caveat a renderer must keep beside the value, or ``""``.
    """

    which: str
    backend: str | None
    value_bytes: int | None
    meaning: str
    ratio_eligible: bool
    caveat: str


def pass_peak_facts(trace: Any, which: str = "forward") -> PassPeakFacts:
    """Return the typed publication facts for one pass-level peak.

    Parameters
    ----------
    trace:
        Trace whose measurement should be described.
    which:
        ``"forward"`` or ``"backward"``.

    Returns
    -------
    PassPeakFacts
        Backend-checked facts; renderers never read the raw fields directly.
    """

    backend = getattr(trace, f"{which}_memory_backend", None)
    peak = getattr(trace, f"{which}_peak_memory", None)
    if not backend or backend == "unknown" or peak is None:
        return PassPeakFacts(
            which=which,
            backend=backend if isinstance(backend, str) else None,
            value_bytes=None,
            meaning="not measured on this capture",
            ratio_eligible=False,
            caveat="",
        )
    value = int(peak)
    if backend == "cuda":
        if value == 0:
            return PassPeakFacts(
                which=which,
                backend=backend,
                value_bytes=0,
                meaning=("did not exceed the process's pre-existing CUDA high-water mark"),
                ratio_eligible=False,
                caveat=("a CUDA 0 means no high-water advance, not zero allocation"),
            )
        return PassPeakFacts(
            which=which,
            backend=backend,
            value_bytes=value,
            meaning="CUDA device allocator peak",
            ratio_eligible=True,
            caveat="",
        )
    if backend == "cpu":
        return PassPeakFacts(
            which=which,
            backend=backend,
            value_bytes=value,
            meaning="process RSS growth (NOT a tensor peak)",
            ratio_eligible=False,
            caveat=(
                "host RSS endpoint delta; non-reproducible across identical runs "
                "and 0 can mean the forward fit in already-resident heap headroom"
            ),
        )
    if backend == "mps":
        return PassPeakFacts(
            which=which,
            backend=backend,
            value_bytes=value,
            meaning="MPS allocator delta (NOT a device peak)",
            ratio_eligible=False,
            caveat="cheap allocator-delta basis; 0 can mean already-resident reuse",
        )
    return PassPeakFacts(
        which=which,
        backend=str(backend),
        value_bytes=value,
        meaning=f"{backend}-basis measurement",
        ratio_eligible=False,
        caveat="unrecognized measurement backend; treat as labeled raw telemetry",
    )


def format_pass_peak(trace: Any, which: str = "forward") -> str:
    """Render one pass-level peak line through the publication gate.

    Parameters
    ----------
    trace:
        Trace whose measurement should be rendered.
    which:
        ``"forward"`` or ``"backward"``.

    Returns
    -------
    str
        One honest human-readable line (basis always named; CUDA-0 rendered
        as a high-water fact; cpu/mps rendered under their real meaning).
    """

    from ..utils.display import human_readable_size

    facts = pass_peak_facts(trace, which)
    prefix = f"Live {which}-memory peak"
    if facts.value_bytes is None:
        return f"{prefix}: unavailable ({facts.meaning})"
    if facts.backend == "cuda" and facts.value_bytes == 0:
        return f"{prefix}: 0 B advance (cuda basis; {facts.meaning})"
    size_text = human_readable_size(facts.value_bytes)
    line = f"{prefix}: {size_text} measured ({facts.backend} basis; {facts.meaning})"
    if facts.caveat:
        line += f" -- {facts.caveat}"
    return line
