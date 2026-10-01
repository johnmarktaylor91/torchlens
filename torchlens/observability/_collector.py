"""Module-tier history collector, Route A (explorer item 6; D15-D17, D21).

One real discovery forward catalogs the selected module outputs while
returning the user's actual result; persistent observers then reduce
DETACHED views under ``no_grad`` at the module return -- the collector never
enters fastlog and needs no capture-core change (D15). No raw retention
after reduction, no autograd reference surviving step commit, anywhere.

Transfer invariant (D16): reductions write fixed-width device vectors into
a preallocated staging buffer at precomputed site offsets; host transfers
are BATCHED and per-phase-per-device bounded, independent of site count --
never a per-site ``.item()`` or ``.to('cpu')``.

Both step spellings ship with provenance recorded (D17): ``optimizer=``
infers the boundary from the shipped optimizer post-step hook and stamps
``implicit`` (recording ``unknown`` for the unscale/clip status it cannot
determine -- it never guesses); the explicit ``step()`` transaction is
authoritative and REQUIRED for resume.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .. import _state
from ._artifact import CommittedBlock, HistoryWriter, RamRing
from ._chassis import EventStream, ObserverEvent
from ._errors import WatchLifecycleError, WatchPlanError
from ._kernels import (
    DEFAULT_DESCRIPTOR,
    SPINE_SLOTS,
    HistogramDescriptor,
    histogram_result_from_vector,
    merge_spine_vectors,
    sketch_vector,
    spine_result_from_vector,
    spine_vector,
)
from ._schema import (
    ObservationRecord,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
    validate_step_order,
)

__tl_layer__ = "L5"

#: Streams Route A can collect. ``activation_grad`` rides the F24 observer
#: seam (backward hooks), not this collector.
ROUTE_A_STREAMS = ("activation", "param", "param_grad", "param_delta")

#: Update-channel sample size (D11): deterministic seeded fp32 subvector of
#: min(numel, 4096) indices per parameter; full coverage degrades gracefully
#: to EXACT with ``estimated=False``.
UPDATE_SAMPLE_SIZE = 4096

#: Bytes per observation row estimate for the plan table: spine slots at
#: float64 plus identity/disclosure strings, and the sketch cells at int64.
_SPINE_ROW_BYTES = SPINE_SLOTS * 8 + 64

_BlockKey = tuple[str, str, str]


@dataclass(frozen=True)
class PlanRow:
    """One plan-table row (D21): elements per logged step, ALONGSIDE bytes."""

    site_id: str
    stream: str
    elements_per_step: int
    bytes_per_step: int
    cadence: int
    sketch_cadence: int | None
    note: str = ""


@dataclass(frozen=True)
class WatchPlan:
    """The resolved collection plan: rows, totals, and both cost columns.

    Bytes predict storage; ELEMENTS predict time -- they differ by orders of
    magnitude (D21), so both columns print.
    """

    rows: tuple[PlanRow, ...]
    total_elements_per_step: int
    total_bytes_per_step: int

    def format_table(self) -> str:
        """Render the plan table (ASCII; both cost columns)."""

        header = (
            f"{'site':40s} {'stream':12s} {'elements/step':>14s} "
            f"{'bytes/step':>11s} {'cadence':>7s} {'sketch':>6s} note"
        )
        lines = [header, "-" * len(header)]
        for row in self.rows:
            sketch = "-" if row.sketch_cadence is None else str(row.sketch_cadence)
            lines.append(
                f"{row.site_id:40s} {row.stream:12s} {row.elements_per_step:>14d} "
                f"{row.bytes_per_step:>11d} {row.cadence:>7d} {sketch:>6s} {row.note}"
            )
        lines.append(
            f"TOTAL elements/step={self.total_elements_per_step} "
            f"bytes/step={self.total_bytes_per_step}"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class WatchSettings:
    """Collector plumbing knobs: sketch tier, identity, retention, events.

    The constructor keeps the what-to-watch surface (``sites`` / ``streams``
    / ``cadence``) flat; the rarer knobs ride here. ``sketch_cadence`` sits
    beside ``descriptor`` because both configure the same sketch tier.
    """

    sketch_cadence: int | Mapping[str, int] | None = None
    descriptor: HistogramDescriptor = DEFAULT_DESCRIPTOR
    run_id: str | None = None
    segment_id: str | None = None
    output_dir: str | None = None
    ram_capacity: int | None = None
    ram_policy: str = "refuse"
    event_stream: EventStream | None = None
    update_sample_size: int = UPDATE_SAMPLE_SIZE


@dataclass(frozen=True)
class StepTruth:
    """Per-step optimizer-truth disclosures for the explicit step spelling.

    Defaults state exactly what an unadorned ``step()`` can attest: the
    optimizer step is assumed ``applied`` and the unscale/clip statuses stay
    ``"unknown"`` unless the caller can vouch for them -- the collector
    never guesses (D17).
    """

    applied: bool = True
    scale: float | None = None
    unscaled: str = "unknown"
    clipped: str = "unknown"
    micro_batches: int | None = None


def _site_seed(run_id: str, site_id: str) -> int:
    """Deterministic private seed for the update channel (D11/D13).

    Derived from run/site identity via sha256 -- never touches model or
    global RNG.
    """

    digest = hashlib.sha256(f"{run_id}|{site_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**62)


def _batch_to_cpu(
    entries: list[tuple[_BlockKey, torch.Tensor]],
) -> list[tuple[_BlockKey, torch.Tensor]]:
    """Move keyed device vectors to CPU in ONE stacked transfer per device.

    This is the D16 chokepoint for the parameter phases: per-site ``.cpu()``
    calls are banned; vectors group by device, stack, and cross in one copy
    per device per phase.
    """

    by_device: dict[torch.device, list[tuple[_BlockKey, torch.Tensor]]] = {}
    for key, vec in entries:
        by_device.setdefault(vec.device, []).append((key, vec))
    results: list[tuple[_BlockKey, torch.Tensor]] = []
    for device_entries in by_device.values():
        stacked = torch.stack([vec for _key, vec in device_entries]).cpu()
        for row_index, (key, _vec) in enumerate(device_entries):
            results.append((key, stacked[row_index]))
    return results


class _SiteSlot:
    """Internal per-site staging bookkeeping."""

    def __init__(self, record: SiteRecord, staging_index: int) -> None:
        self.record = record
        self.staging_index = staging_index


class HistoryCollector:
    """Route-A module-tier collector over one model.

    Lifecycle: ``discover(...)`` (one real forward) -> ``attach(...)`` ->
    train with either step spelling -> ``detach()``. Every hook handle is
    removed on every exit path; all collector arithmetic runs under
    ``torch.no_grad()`` + ``pause_logging()`` so a concurrent TorchLens
    capture never records collector ops.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sites: Iterable[str] | None = None,
        streams: Iterable[str] = ("activation",),
        cadence: int | Mapping[str, int] = 1,
        settings: WatchSettings | None = None,
    ) -> None:
        settings = settings if settings is not None else WatchSettings()
        self.settings = settings
        self.model = model
        self.requested_sites = None if sites is None else tuple(sites)
        self.streams = tuple(streams)
        for stream in self.streams:
            if stream not in ROUTE_A_STREAMS:
                raise WatchPlanError(
                    f"stream {stream!r} is not collectable by the Route-A "
                    f"module-tier collector (streams: {ROUTE_A_STREAMS}). "
                    "activation_grad rides the F24 observer seam (backward "
                    "hooks), not this collector.",
                    code="watch_plan_invalid",
                    stream=stream,
                    remedy=f"Use streams from {ROUTE_A_STREAMS}.",
                )
        self.cadences = self._normalize_cadence(cadence, default=1)
        self.sketch_cadences = (
            {}
            if settings.sketch_cadence is None
            else self._normalize_cadence(settings.sketch_cadence, default=None)
        )
        self.descriptor = settings.descriptor
        self.run_id = settings.run_id or uuid.uuid4().hex
        self.segment_id = settings.segment_id or uuid.uuid4().hex
        self.event_stream = settings.event_stream
        self.update_sample_size = int(settings.update_sample_size)
        self.run = RunRecord(
            run_id=self.run_id,
            segment_id=self.segment_id,
            descriptor=settings.descriptor,
            package_versions={"torch": torch.version.__version__},
            cadences=self.cadences,
        )
        ring_capacity = settings.ram_capacity if settings.ram_capacity is not None else 256
        self.ring = RamRing(
            capacity=ring_capacity,
            policy=settings.ram_policy,
            disk_backed=settings.output_dir is not None,
        )
        self.writer = (
            HistoryWriter(settings.output_dir, self.run)
            if settings.output_dir is not None
            else None
        )
        self._module_refs: dict[int, torch.nn.Module] = {}
        self._module_sites: dict[int, _SiteSlot] = {}
        self._param_sites: dict[str, _SiteSlot] = {}
        self._sites_by_id: dict[str, SiteRecord] = {}
        self._plan: WatchPlan | None = None
        self._hook_handles: list[Any] = []
        self._optimizer_handles: list[Any] = []
        self._attached = False
        self._optimizer: torch.optim.Optimizer | None = None
        self._staging: torch.Tensor | None = None
        self._staging_touched: list[bool] = []
        self._sketch_staging: torch.Tensor | None = None
        self._staging_overflow: list[tuple[int, torch.Tensor]] = []
        self._block_open = False
        self._block_step: int | None = None
        self._block_provenance = "implicit"
        self._block_truth: dict[str, Any] = {}
        self._block_micro = 0
        self._block_spines: dict[_BlockKey, torch.Tensor] = {}
        self._block_sketches: dict[_BlockKey, torch.Tensor] = {}
        self._block_grad_scale: dict[_BlockKey, str] = {}
        self._last_step: int | None = None
        self._implicit_counter = 0
        self._update_samples: dict[str, torch.Tensor] = {}
        self._sample_index_cache: dict[str, torch.Tensor] = {}

    def _normalize_cadence(
        self, cadence: int | Mapping[str, int] | None, *, default: int | None
    ) -> dict[str, int]:
        """Normalize cadence input to a per-stream mapping (D21)."""

        if cadence is None:
            return {}
        if isinstance(cadence, int):
            return dict.fromkeys(self.streams, cadence)
        result = {stream: int(default) for stream in self.streams if default is not None}
        for stream, value in cadence.items():
            if stream not in ROUTE_A_STREAMS:
                raise WatchPlanError(
                    f"cadence names unknown stream {stream!r}.",
                    code="watch_plan_invalid",
                    stream=stream,
                    remedy=f"Use streams from {ROUTE_A_STREAMS}.",
                )
            result[stream] = int(value)
        return result

    # -- discovery ---------------------------------------------------------

    def discover(self, *args: Any, **kwargs: Any) -> Any:
        """Run ONE real discovery forward; returns the user's actual result.

        Catalogs the selected module outputs (shape/dtype/device), the
        selected parameters, builds the plan table, and preallocates the
        per-phase staging buffers. Zero-match selections refuse at plan
        time: a warning at step 0 of a ten-hour run is not disclosure (D6).
        """

        output, seen = self._run_discovery_forward(*args, **kwargs)
        device = self._catalog_module_sites(seen)
        self._catalog_param_sites()
        if not self._sites_by_id:
            raise WatchPlanError(
                "The site selection matched NOTHING. Zero-match selectors "
                "refuse at plan time: a warning at step 0 of a ten-hour run "
                "is not disclosure (D6).",
                code="watch_plan_empty",
                requested=self.requested_sites,
                remedy=(
                    "Pass sites= matching module paths (prefix match) or "
                    "parameter names, or sites=None for all leaf modules."
                ),
            )
        self._allocate_staging(device)
        self._plan = self._build_plan()
        if self.writer is not None:
            for site in self._sites_by_id.values():
                self.writer.add_site(site)
        return output

    def _run_discovery_forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, dict[int, tuple[str, tuple[int, ...], str, str]]]:
        """One real forward under one-shot catalog hooks; returns (output, catalog)."""

        selected = self._select_modules()
        seen: dict[int, tuple[str, tuple[int, ...], str, str]] = {}
        handles = []

        def _catalog_hook(path: str) -> Any:
            """Build a one-shot forward hook that catalogs one module's output."""

            def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                """Record the module's output shape/dtype/device on first fire."""

                del inputs
                if id(module) in seen or not isinstance(output, torch.Tensor):
                    return
                seen[id(module)] = (
                    path,
                    tuple(output.shape),
                    str(output.dtype).removeprefix("torch."),
                    str(output.device),
                )

            return hook

        try:
            for path, module in selected:
                handles.append(module.register_forward_hook(_catalog_hook(path)))
            output = self.model(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return output, seen

    def _catalog_module_sites(self, seen: dict[int, tuple[str, tuple[int, ...], str, str]]) -> str:
        """Register module activation sites from the catalog; returns the device."""

        device = "cpu"
        if "activation" not in self.streams:
            return device
        module_by_path = dict(self._select_modules())
        for staging_index, (module_id, (path, shape, dtype, dev)) in enumerate(seen.items()):
            site = SiteRecord(
                site_id=f"module:{path}",
                kind="module",
                display_label=path,
                module_path=path,
                shape=shape,
                dtype=dtype,
                numel=math.prod(shape) if shape else 1,
            )
            self._module_sites[module_id] = _SiteSlot(site, staging_index)
            self._module_refs[module_id] = module_by_path[path]
            self._sites_by_id[site.site_id] = site
            device = dev
        return device

    def _catalog_param_sites(self) -> None:
        """Register parameter sites for the selected param-family streams."""

        if not any(s.startswith("param") for s in self.streams):
            return
        for name, param in self.model.named_parameters():
            if self.requested_sites is not None and not any(
                name.startswith(prefix) for prefix in self.requested_sites
            ):
                continue
            site = SiteRecord(
                site_id=f"param:{name}",
                kind="param",
                display_label=name,
                param_name=name,
                shape=tuple(param.shape),
                dtype=str(param.dtype).removeprefix("torch."),
                numel=param.numel(),
            )
            self._param_sites[name] = _SiteSlot(site, -1)
            self._sites_by_id[site.site_id] = site

    def _allocate_staging(self, device: str) -> None:
        """Preallocate the per-phase module-site staging buffers (D16)."""

        n_module_sites = len(self._module_sites)
        if not n_module_sites:
            return
        self._staging = torch.zeros(
            (n_module_sites, SPINE_SLOTS), dtype=torch.float64, device=device
        )
        self._staging_touched = [False] * n_module_sites
        cells = 2 * self.descriptor.bins_per_side + 8
        self._sketch_staging = torch.zeros(
            (n_module_sites, cells), dtype=torch.int64, device=device
        )

    def _select_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """Resolve the module selection (prefix match; None = leaf modules)."""

        selected: list[tuple[str, torch.nn.Module]] = []
        for path, module in self.model.named_modules():
            if not path:
                continue
            if self.requested_sites is None:
                if next(module.children(), None) is None:
                    selected.append((path, module))
            elif any(path == p or path.startswith(p) for p in self.requested_sites):
                selected.append((path, module))
        return selected

    def _build_plan(self) -> WatchPlan:
        """Build the D21 plan table: elements AND bytes per logged step."""

        rows: list[PlanRow] = []
        total_elements = 0
        total_bytes = 0
        sketch_cells = 2 * self.descriptor.bins_per_side + 8
        for site in self._sites_by_id.values():
            streams = (
                ("activation",)
                if site.kind == "module"
                else tuple(s for s in self.streams if s.startswith("param"))
            )
            for stream in streams:
                if stream not in self.streams:
                    continue
                cadence = self.cadences.get(stream, 1)
                sketch = self.sketch_cadences.get(stream)
                numel = site.numel or 0
                if stream == "param_delta":
                    elements = min(numel, self.update_sample_size)
                    note = "sampled update channel (D11)" if numel > elements else "exact"
                else:
                    elements = numel
                    note = ""
                bytes_per = _SPINE_ROW_BYTES + (sketch_cells * 8 if sketch else 0)
                rows.append(
                    PlanRow(
                        site_id=site.site_id,
                        stream=stream,
                        elements_per_step=elements // cadence,
                        bytes_per_step=bytes_per // cadence,
                        cadence=cadence,
                        sketch_cadence=sketch,
                        note=note,
                    )
                )
                total_elements += elements // cadence
                total_bytes += bytes_per // cadence
        return WatchPlan(
            rows=tuple(rows),
            total_elements_per_step=total_elements,
            total_bytes_per_step=total_bytes,
        )

    @property
    def plan(self) -> WatchPlan:
        """The resolved plan table; refuses before discovery."""

        if self._plan is None:
            raise WatchLifecycleError(
                "No plan yet: run discover(*inputs) first (one real forward "
                "that also returns your model output).",
                code="watch_lifecycle_invalid",
                remedy="Call collector.discover(inputs) before reading the plan.",
            )
        return self._plan

    # -- attach / detach -----------------------------------------------------

    def attach(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Install the persistent observers (and the implicit step spelling).

        Parameter streams need the optimizer boundary; requesting them
        without ``optimizer=`` refuses typed at attach time rather than
        producing silently absent series.
        """

        if self._plan is None:
            raise WatchLifecycleError(
                "attach() before discover(): the plan (and the staging "
                "geometry) comes from the discovery forward.",
                code="watch_lifecycle_invalid",
                remedy="Call collector.discover(inputs) first.",
            )
        if self._attached:
            raise WatchLifecycleError(
                "This collector is already attached; one collector, one hook "
                "set (no parallel hook stacks, ever).",
                code="watch_lifecycle_invalid",
                remedy="Call detach() first, or use the existing attachment.",
            )
        param_streams = [s for s in self.streams if s.startswith("param")]
        if param_streams and optimizer is None:
            raise WatchPlanError(
                f"streams {param_streams} need the optimizer-step boundary; "
                "without optimizer= their phase truth (pre/post step) is "
                "unknowable and the series would be silently absent.",
                code="watch_plan_invalid",
                streams=tuple(param_streams),
                remedy="Pass attach(optimizer=...) or drop the param streams.",
            )
        self._optimizer = optimizer
        try:
            for module_id, slot in self._module_sites.items():
                module = self._module_refs[module_id]
                self._hook_handles.append(module.register_forward_hook(self._observer_hook(slot)))
            if optimizer is not None:
                self._optimizer_handles.append(
                    optimizer.register_step_pre_hook(self._optimizer_pre_hook)
                )
                self._optimizer_handles.append(
                    optimizer.register_step_post_hook(self._optimizer_post_hook)
                )
        except BaseException:
            self.detach()
            raise
        self._attached = True

    def detach(self) -> None:
        """Remove every hook handle; idempotent; runs on every exit path."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for handle in self._optimizer_handles:
            handle.remove()
        self._optimizer_handles.clear()
        self._optimizer = None
        self._attached = False

    def __enter__(self) -> HistoryCollector:
        """Context form: detach is guaranteed on every exit path."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Detach all hooks regardless of how the block exits."""

        self.detach()

    # -- observation ---------------------------------------------------------

    def _observer_hook(self, slot: _SiteSlot) -> Any:
        """Build the persistent activation observer for one site.

        Reduces a DETACHED view under no_grad at the module return; writes
        the fixed-width vectors into the preallocated staging buffer at the
        site's precomputed offset (D16). Re-fires of the same site within
        one forward (recurrent/reused modules) go to a device-side overflow
        list and merge after the batched phase transfer -- never a per-site
        host copy. The raw tensor is never retained.
        """

        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            """Reduce this site's output into the staging buffer (or overflow)."""

            del module, inputs
            if not isinstance(output, torch.Tensor):
                return
            if not self._should_observe("activation"):
                return
            staging = self._staging
            touched = self._staging_touched
            if staging is None or not touched:
                return
            with torch.no_grad(), _state.pause_logging():
                detached = output.detach()
                vec = spine_vector(detached)
                index = slot.staging_index
                if touched[index]:
                    self._staging_overflow.append((index, vec))
                else:
                    staging[index] = vec.to(staging.device)
                    touched[index] = True
                if self._sketch_scheduled("activation") and self._sketch_staging is not None:
                    self._sketch_staging[index] += sketch_vector(detached, self.descriptor).to(
                        self._sketch_staging.device
                    )

        return hook

    def _should_observe(self, stream: str) -> bool:
        """True when the stream is scheduled for the current block."""

        if not self._block_open and self._optimizer is None:
            return False
        step = self._block_step if self._block_step is not None else self._implicit_counter
        cadence = self.cadences.get(stream, 1)
        return step % cadence == 0

    def _sketch_scheduled(self, stream: str) -> bool:
        """True when the sketch tier is scheduled for the current block."""

        sketch_cadence = self.sketch_cadences.get(stream)
        if not sketch_cadence:
            return False
        step = self._block_step if self._block_step is not None else self._implicit_counter
        return step % sketch_cadence == 0

    # -- step lifecycle --------------------------------------------------------

    @contextmanager
    def step(
        self,
        global_step: int,
        *,
        truth: StepTruth | None = None,
        new_segment: bool = False,
    ) -> Iterator[None]:
        """The explicit step transaction (authoritative; REQUIRED for resume).

        Brackets one training step; commits the StepBlock atomically at
        exit. ``truth=`` carries the per-step optimizer-truth disclosures
        (:class:`StepTruth`); omitted, the defaults attest only what a bare
        bracket can. Duplicate/decreasing steps refuse without
        ``new_segment=True``.
        """

        truth = truth if truth is not None else StepTruth()
        if self._block_open:
            raise WatchLifecycleError(
                "A step transaction is already open; steps never nest (one step axis, ever).",
                code="watch_step_conflict",
                remedy="Close the open step before starting the next.",
            )
        if new_segment:
            self.segment_id = uuid.uuid4().hex
            self._last_step = None
        validate_step_order(self._last_step, global_step, same_segment=True)
        self.ring.will_admit()
        self._open_block(global_step, provenance="explicit")
        self._block_truth = {
            "optimizer_status": "applied" if truth.applied else "skipped",
            "scale": truth.scale,
            "unscaled": truth.unscaled,
            "clipped": truth.clipped,
            "micro_batches": truth.micro_batches,
        }
        try:
            yield
        finally:
            self._commit_block()

    def _open_block(self, global_step: int, *, provenance: str) -> None:
        """Open one StepBlock and reset the staging buffers."""

        self._block_open = True
        self._block_step = global_step
        self._block_provenance = provenance
        self._block_micro = 0
        self._block_spines.clear()
        self._block_sketches.clear()
        self._block_grad_scale.clear()
        self._staging_overflow.clear()
        if self._staging is not None:
            self._staging.zero_()
            self._staging_touched = [False] * len(self._staging_touched)
        if self._sketch_staging is not None:
            self._sketch_staging.zero_()

    def _flush_forward_staging(self) -> None:
        """The batched host transfer for the forward phase (D16).

        One staging copy, one touched copy, one sketch copy, and one stacked
        overflow copy per device -- bounded, independent of site count.
        """

        if self._staging is None:
            return
        touched = list(self._staging_touched)
        overflow = self._staging_overflow
        self._staging_overflow = []
        if not any(touched) and not overflow:
            return
        # copy=True: on a CPU model .cpu() would ALIAS the staging storage,
        # and the zero_() below would wipe the stored views.
        staged = self._staging.to("cpu", copy=True)
        sketches = (
            self._sketch_staging.to("cpu", copy=True) if self._sketch_staging is not None else None
        )
        overflow_cpu = self._overflow_by_index(overflow)
        for slot in self._module_sites.values():
            if touched[slot.staging_index]:
                self._fold_staged_site(slot, staged, sketches, overflow_cpu)
        self._staging.zero_()
        self._staging_touched = [False] * len(self._staging_touched)
        if self._sketch_staging is not None:
            self._sketch_staging.zero_()
        self._block_micro += 1

    @staticmethod
    def _overflow_by_index(
        overflow: list[tuple[int, torch.Tensor]],
    ) -> dict[int, list[torch.Tensor]]:
        """Group re-fire overflow vectors per site after ONE stacked transfer."""

        overflow_cpu: dict[int, list[torch.Tensor]] = {}
        if overflow:
            stacked = torch.stack([vec for _index, vec in overflow]).cpu()
            for row, (index, _vec) in enumerate(overflow):
                overflow_cpu.setdefault(index, []).append(stacked[row])
        return overflow_cpu

    def _fold_staged_site(
        self,
        slot: _SiteSlot,
        staged: torch.Tensor,
        sketches: torch.Tensor | None,
        overflow_cpu: dict[int, list[torch.Tensor]],
    ) -> None:
        """Merge one touched site's staged row (+ overflow re-fires) into the block."""

        index = slot.staging_index
        key = (slot.record.site_id, "activation", "forward")
        vec = staged[index]
        for extra in overflow_cpu.get(index, ()):
            vec = merge_spine_vectors(vec, extra)
        self._fold_spines([(key, vec)])
        if sketches is not None and bool(sketches[index].any().item()):
            if key in self._block_sketches:
                self._block_sketches[key] += sketches[index]
            else:
                self._block_sketches[key] = sketches[index].clone()

    def _fold_spines(self, entries: list[tuple[_BlockKey, torch.Tensor]]) -> None:
        """Fold CPU spine vectors into the open block's accumulators."""

        for key, vec in entries:
            if key in self._block_spines:
                self._block_spines[key] = merge_spine_vectors(self._block_spines[key], vec)
            else:
                self._block_spines[key] = vec

    def _optimizer_pre_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """S-B: param grads + pre-step params + update-channel sample."""

        del optimizer, args, kwargs
        if not self._block_open and self._optimizer is not None:
            # Implicit spelling: the block opens at the first optimizer
            # boundary activity after the last close (D17).
            self._open_block(self._implicit_counter, provenance="implicit")
        self._flush_forward_staging()
        grad_entries: list[tuple[_BlockKey, torch.Tensor]] = []
        param_entries: list[tuple[_BlockKey, torch.Tensor]] = []
        sketch_entries: list[tuple[_BlockKey, torch.Tensor]] = []
        with torch.no_grad(), _state.pause_logging():
            observe_grad = "param_grad" in self.streams and self._should_observe("param_grad")
            observe_param = "param" in self.streams and self._should_observe("param")
            observe_delta = "param_delta" in self.streams and self._should_observe("param_delta")
            sketch_grad = self._sketch_scheduled("param_grad")
            for name, param in self.model.named_parameters():
                slot = self._param_sites.get(name)
                if slot is None:
                    continue
                site_id = slot.record.site_id
                if observe_grad and param.grad is not None:
                    key = (site_id, "param_grad", "pre_step")
                    grad_entries.append((key, spine_vector(param.grad.detach())))
                    self._block_grad_scale[key] = "unknown"
                    if sketch_grad:
                        sketch_entries.append(
                            (key, sketch_vector(param.grad.detach(), self.descriptor))
                        )
                if observe_param:
                    key = (site_id, "param", "pre_step")
                    param_entries.append((key, spine_vector(param.detach())))
                if observe_delta:
                    indices = self._sample_indices(name, param)
                    self._update_samples[name] = (
                        param.detach().reshape(-1)[indices].to(torch.float32).clone()
                    )
            self._fold_spines(_batch_to_cpu(grad_entries))
            self._fold_spines(_batch_to_cpu(param_entries))
            for key, vec in _batch_to_cpu(sketch_entries):
                if key in self._block_sketches:
                    self._block_sketches[key] += vec
                else:
                    self._block_sketches[key] = vec

    def _optimizer_post_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """S-C: post-step params + the update channel; closes implicit blocks."""

        del optimizer, args, kwargs
        param_entries: list[tuple[_BlockKey, torch.Tensor]] = []
        delta_entries: list[tuple[_BlockKey, torch.Tensor]] = []
        with torch.no_grad(), _state.pause_logging():
            observe_param = "param" in self.streams and self._should_observe("param")
            skipped = self._block_truth.get("optimizer_status") == "skipped"
            for name, param in self.model.named_parameters():
                slot = self._param_sites.get(name)
                if slot is None:
                    continue
                site_id = slot.record.site_id
                if observe_param:
                    param_entries.append(
                        ((site_id, "param", "post_step"), spine_vector(param.detach()))
                    )
                before = self._update_samples.pop(name, None)
                if before is not None and not skipped:
                    indices = self._sample_indices(name, param)
                    after = param.detach().reshape(-1)[indices].to(torch.float32)
                    delta = (after - before).to(torch.float64)
                    delta_entries.append(((site_id, "param_delta", "step"), spine_vector(delta)))
            self._fold_spines(_batch_to_cpu(param_entries))
            self._fold_spines(_batch_to_cpu(delta_entries))
        if self._block_provenance == "implicit" and self._block_open:
            self._block_truth.setdefault("optimizer_status", "applied")
            self._commit_block()
            self._implicit_counter += 1

    def _sample_indices(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Deterministic seeded sample indices for one parameter (D11).

        Cached per site: the seed derives from run/site identity via a
        PRIVATE generator that never touches model or global RNG (D13).
        """

        cached = self._sample_index_cache.get(name)
        if cached is not None:
            return cached
        numel = param.numel()
        if numel <= self.update_sample_size:
            indices = torch.arange(numel, device=param.device)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_site_seed(self.run_id, name))
            raw = torch.randint(
                0, numel, (self.update_sample_size,), generator=generator, device="cpu"
            )
            indices = torch.unique(raw).to(param.device)
        self._sample_index_cache[name] = indices
        return indices

    def _commit_block(self) -> None:
        """Commit the open StepBlock atomically (D6)."""

        if not self._block_open:
            return
        self._flush_forward_staging()
        step = self._block_step if self._block_step is not None else 0
        truth = self._block_truth
        skipped = truth.get("optimizer_status") == "skipped"
        block = StepBlockRecord(
            segment_id=self.segment_id,
            global_step=step,
            provenance=self._block_provenance,
            optimizer_status=truth.get("optimizer_status", "unknown"),
            scale=truth.get("scale"),
            unscaled=truth.get("unscaled", "unknown"),
            clipped=truth.get("clipped", "unknown"),
            micro_batches=truth.get("micro_batches") or (self._block_micro or None),
        )
        observations: list[ObservationRecord] = []
        for key, vec in self._block_spines.items():
            site_id, stream, phase = key
            if stream == "param_delta" and skipped:
                # A skipped optimizer step records NO fake update (D7/D11).
                continue
            spine = spine_result_from_vector(vec, "float64")
            sketch = None
            if key in self._block_sketches:
                sketch = histogram_result_from_vector(self._block_sketches[key], self.descriptor)
            site = self._sites_by_id[site_id]
            numel = site.numel or 0
            estimated = stream == "param_delta" and numel > spine.count_total
            observations.append(
                ObservationRecord(
                    global_step=step,
                    site_id=site_id,
                    stream=stream,
                    phase=phase,
                    presence="observed",
                    spine=spine,
                    sketch=sketch,
                    grad_scale=self._block_grad_scale.get(key),
                    estimated=estimated,
                    sample_size=spine.count_total if stream == "param_delta" else None,
                )
            )
        committed = CommittedBlock(
            block=block,
            observations=tuple(observations),
            step_lo=step,
            step_hi=step,
        )
        self.ring.admit(committed)
        if self.writer is not None:
            self.writer.append_block(committed)
        if self.event_stream is not None:
            self.event_stream.publish(
                ObserverEvent(
                    key="history.step_committed",
                    kind="scalar",
                    value=float(len(observations)),
                    global_step=step,
                    accepted_step_id=step,
                    axis_provenance=self._block_provenance,
                    scope="collector",
                )
            )
        self._last_step = step
        self._block_open = False
        self._block_step = None
        self._block_truth = {}


__all__ = [
    "ROUTE_A_STREAMS",
    "UPDATE_SAMPLE_SIZE",
    "HistoryCollector",
    "PlanRow",
    "StepTruth",
    "WatchPlan",
    "WatchSettings",
]
