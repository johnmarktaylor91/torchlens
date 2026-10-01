"""Memory-planned whole-model extraction: plan, table, run, npz (F20).

The brainpipe centerpiece (memo section 3): a two-phase plan-then-run API
whose extraction-PLAN table prints measured bytes, measured peaks, and
engine-multiplied cost before anything runs. The RUNNER is the shipped
:func:`torchlens.dataset_extraction.extract_dataset` (batched, atomic
shards, manifest, resume, model identity); this module adds the PLANNER --
two real probes, per-site linear byte fits, budget arithmetic that refuses
with numbers instead of silently shrinking (D-10), and full-signature plan
keying (D-15) -- plus npz interchange with the Net2Brain file-naming
contract (D-21).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
Design law highlights honored here:

* D-8: single-pass-primary. One ``save='all'``-shaped pass with eager
  transform-and-release beats chunking for the reduce-at-capture case;
  layer chunking is a later fallback, never a floor remedy.
* D-10: batch size belongs to the user. The planner REFUSES with
  arithmetic; it never silently shrinks a batch or drops a site.
* D-11: engine is trace-only at launch. ``engine="fastlog"`` refuses typed
  until the definitional parity suite exists (an engine swap changes what
  "whole model" MEANS); ``engine="auto"`` is dead.
* D-13: ``inference_only=True`` is the sweep's default probe/run posture.
* D-15: two probes (batch b and 2b), per-site linear fit, full
  input-signature keying; a signature mismatch is a typed refusal with the
  re-plan remedy, never a silent overrun.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._errors import InvalidArgumentError

__tl_layer__ = "L6"

__all__ = [
    "ExtractionPlan",
    "PlanSite",
    "export_npz",
    "extraction_plan",
    "parse_bytes",
]


_BYTE_UNITS: dict[str, int] = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_bytes(value: Any) -> int:
    """Parse a human-unit byte budget (D-26; twin of ``format_bytes``).

    Parameters
    ----------
    value:
        ``int`` (bytes, returned as-is), or a string such as ``"2GB"`` /
        ``"8 GiB"`` / ``"512 MB"`` (case-insensitive, optional space;
        decimal units are powers of 10, binary ``*iB`` units powers of 2).

    Returns
    -------
    int
        The budget in bytes.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        If the value is not a positive byte count or a recognized unit
        string (code ``byte_budget_invalid``).
    """

    if isinstance(value, bool):
        raise InvalidArgumentError(
            f"A byte budget cannot be a bool ({value!r}).",
            code="byte_budget_invalid",
            remedy="pass an int byte count or a string such as '8 GiB'",
        )
    if isinstance(value, int):
        if value <= 0:
            raise InvalidArgumentError(
                f"A byte budget must be positive; got {value}.",
                code="byte_budget_invalid",
                remedy="pass a positive byte count",
            )
        return value
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        for unit in sorted(_BYTE_UNITS, key=len, reverse=True):
            if text.endswith(unit):
                number_text = text[: -len(unit)]
                try:
                    number = float(number_text)
                except ValueError:
                    break
                if number <= 0:
                    break
                return int(number * _BYTE_UNITS[unit])
    raise InvalidArgumentError(
        f"Unrecognized byte budget: {value!r}.",
        code="byte_budget_invalid",
        remedy="pass an int byte count or a string such as '512 MB' / '8 GiB'",
    )


def _format_bytes(num_bytes: int | float | None) -> str:
    """Render a byte count for the plan table (``None`` renders unknown)."""

    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:,.1f} TiB"


@dataclass(frozen=True, slots=True)
class PlanSite:
    """One requested site's measured plan row (memo section 3.2).

    Parameters
    ----------
    label:
        Final layer label (the runner's addressing key).
    site_key:
        L1 structural site key where derivable, else ``None``.
    shape:
        Observed output shape at the probe batch.
    dtype:
        Observed output dtype string.
    bytes_fixed:
        Batch-independent byte intercept from the two-point linear fit.
    bytes_per_stimulus:
        Byte slope per stimulus from the two-point fit.
    batch_invariant:
        Whether the site's bytes did NOT scale with batch (position
        embeddings, mask broadcasts -- over-predicted 8x by naive total
        scaling, memo D-15).
    transformed_shape:
        Post-transform shape at the probe batch, when a transform ran.
    transformed_bytes:
        Post-transform bytes at the probe batch, when a transform ran.
    transform_seconds:
        Measured live transform time at the probe batch (D-15: the probe
        executes the transform chain; post-transform bytes and time are
        measured, never derived).
    included:
        Whether the site is in the run's save set.
    exclusion_reason:
        Actionable reason when excluded; never silently dropped.
    """

    label: str
    site_key: str | None
    shape: tuple[int, ...]
    dtype: str
    bytes_fixed: int
    bytes_per_stimulus: float
    batch_invariant: bool
    transformed_shape: tuple[int, ...] | None
    transformed_bytes: int | None
    transform_seconds: float | None
    included: bool
    exclusion_reason: str | None

    def bytes_at_batch(self, batch_size: int) -> int:
        """Return fitted retained bytes for one batch of ``batch_size``."""

        if self.transformed_bytes is not None:
            # Transform output measured at the probe batch; scale by ratio
            # unless batch-invariant.
            if self.batch_invariant:
                return self.transformed_bytes
            per_stimulus = self.transformed_bytes / max(self.shape[0], 1)
            return int(per_stimulus * batch_size)
        if self.batch_invariant:
            return self.bytes_fixed
        return int(self.bytes_fixed + self.bytes_per_stimulus * batch_size)


@dataclass(frozen=True)
class ExtractionPlan:
    """Immutable measured plan for one whole-model extraction sweep.

    Produced by :func:`extraction_plan`; consumed by :meth:`run`. The plan
    is valid ONLY for its input signature (D-15): shapes, dtype, batch
    size, transform identity, and model class all key the plan, and
    :meth:`run` refuses a drifted request typed instead of silently
    over-running.
    """

    sites: tuple[PlanSite, ...]
    batch_size: int
    n_stimuli: int
    input_signature: tuple[Any, ...]
    transform_repr: str | None
    engine: str
    memory_budget_bytes: int | None
    safety_margin: float
    probe_batch_sizes: tuple[int, int]
    probe_peak_pairs: tuple[dict[str, Any] | None, dict[str, Any] | None]
    predicted_peak_bytes: int | None
    peak_basis: str
    model_class: str
    _model_ref: Any = field(repr=False, compare=False, default=None)
    _transform: Any = field(repr=False, compare=False, default=None)

    @property
    def included_sites(self) -> tuple[PlanSite, ...]:
        """Return the sites in the run's save set."""

        return tuple(site for site in self.sites if site.included)

    @property
    def planned_passes(self) -> int:
        """Return exact planned forwards: warm-up + two probes (already run) + batches."""

        return 3 + math.ceil(self.n_stimuli / self.batch_size)

    @property
    def retained_bytes_per_batch(self) -> int:
        """Return fitted retained bytes for one batch across included sites."""

        return sum(site.bytes_at_batch(self.batch_size) for site in self.included_sites)

    @property
    def total_output_bytes(self) -> int:
        """Return fitted final artifact bytes across all stimuli."""

        per_stimulus = self.retained_bytes_per_batch / max(self.batch_size, 1)
        return int(per_stimulus * self.n_stimuli)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable dict form of the plan (object-first surface)."""

        return {
            "sites": [
                {
                    "label": site.label,
                    "site_key": site.site_key,
                    "shape": list(site.shape),
                    "dtype": site.dtype,
                    "batch_invariant": site.batch_invariant,
                    "bytes_at_batch": site.bytes_at_batch(self.batch_size),
                    "transformed_shape": (
                        list(site.transformed_shape) if site.transformed_shape else None
                    ),
                    "transform_seconds": site.transform_seconds,
                    "included": site.included,
                    "exclusion_reason": site.exclusion_reason,
                }
                for site in self.sites
            ],
            "batch_size": self.batch_size,
            "n_stimuli": self.n_stimuli,
            "planned_passes": self.planned_passes,
            "engine": self.engine,
            "transform": self.transform_repr,
            "retained_bytes_per_batch": self.retained_bytes_per_batch,
            "total_output_bytes": self.total_output_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "safety_margin": self.safety_margin,
            "probe_batch_sizes": list(self.probe_batch_sizes),
            "probe_peak_pairs": list(self.probe_peak_pairs),
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "peak_basis": self.peak_basis,
            "input_signature": [
                list(s) if isinstance(s, tuple) else s for s in self.input_signature
            ],
            "model_class": self.model_class,
        }

    def to_pandas(self) -> Any:
        """Return the per-site rows as a pandas DataFrame.

        Raises
        ------
        ImportError
            If pandas is not installed (an optional dependency).
        """

        import pandas as pd

        return pd.DataFrame(self.to_dict()["sites"])

    def table(self) -> str:
        """Render the extraction-PLAN table (memo section 3.2).

        One row per REQUESTED site -- included, excluded, or refused, never
        silently dropped -- and a footer with the probe's own measured cost,
        exact planned passes, the peak PAIR with its backend named, and
        cumulative output bytes. Unknown never renders as zero.
        """

        header = f"{'site':<28} {'shape':<20} {'dtype':<10} {'bytes/batch':>12} {'flags':<24}"
        lines = [header, "-" * len(header)]
        for site in self.sites:
            flags: list[str] = []
            if site.batch_invariant:
                flags.append("batch-invariant")
            if site.transformed_shape is not None:
                flags.append("transformed")
            if not site.included:
                flags.append(f"EXCLUDED: {site.exclusion_reason}")
            lines.append(
                f"{site.label:<28} {str(tuple(site.shape)):<20} {site.dtype:<10} "
                f"{_format_bytes(site.bytes_at_batch(self.batch_size)):>12} "
                f"{'; '.join(flags):<24}"
            )
        lines.append("-" * len(header))
        pair_texts = []
        for batch, pair in zip(self.probe_batch_sizes, self.probe_peak_pairs, strict=True):
            if pair is None:
                pair_texts.append(f"b={batch}: unmeasured")
            else:
                pair_texts.append(
                    f"b={batch}: live {_format_bytes(pair.get('live'))} / "
                    f"resident {_format_bytes(pair.get('resident'))} "
                    f"[{pair.get('backend', 'unknown')}]"
                )
        lines.extend(
            [
                f"engine: {self.engine} (trace-only at launch; fastlog gated on parity)",
                f"planned passes: {self.planned_passes} "
                f"(1 warm-up + 2 probe forwards + {self.planned_passes - 3} batches)",
                f"probe peak pairs: {'; '.join(pair_texts)}",
                f"predicted run peak ({self.peak_basis}): "
                f"{_format_bytes(self.predicted_peak_bytes)}",
                f"retained per batch: {_format_bytes(self.retained_bytes_per_batch)}",
                f"total output: {_format_bytes(self.total_output_bytes)} "
                f"across {self.n_stimuli} stimuli",
                f"memory budget: {_format_bytes(self.memory_budget_bytes)} "
                f"(safety margin {self.safety_margin:.0%})",
                f"batch-invariant sites: {sum(1 for s in self.sites if s.batch_invariant)}",
                f"excluded sites: {sum(1 for s in self.sites if not s.included)}",
            ]
        )
        return "\n".join(lines)

    def _check_signature(self, stimuli: Any) -> None:
        """Refuse a run whose stimuli drift from the planned signature."""

        probe_item_shape, probe_dtype = self.input_signature[0], self.input_signature[1]
        if isinstance(stimuli, torch.Tensor):
            item_shape = tuple(stimuli.shape[1:])
            dtype = str(stimuli.dtype)
            if item_shape != tuple(probe_item_shape) or dtype != probe_dtype:
                raise InvalidArgumentError(
                    f"This plan was probed for per-stimulus shape "
                    f"{tuple(probe_item_shape)} / {probe_dtype} but run() received "
                    f"{item_shape} / {dtype}. A plan is valid only for its input "
                    f"signature (sequence length alone was measured to shift the "
                    f"peak 6.1x on gpt2).",
                    code="extraction_plan_signature_mismatch",
                    remedy="re-plan with probe inputs matching the run stimuli",
                )

    def run(  # noqa: PLR0913 -- the single-pass door onto extract_dataset: each kwarg forwards one memo-specified public knob (D-8)
        self,
        stimuli: Any,
        *,
        stimulus_ids: Any = None,
        output_dir: str | Path | None = None,
        resume: bool = False,
        device: Any = None,
        progress: bool = True,
    ) -> Any:
        """Execute the planned sweep through the shipped extraction runner.

        Single-pass-primary (D-8): one batched extraction pass over the
        included sites with the planned transform; disk mode writes the
        atomic self-describing artifact with resume support.

        Parameters
        ----------
        stimuli:
            Tensor with leading stimulus dimension, or iterable of items.
        stimulus_ids:
            Optional per-stimulus identifiers (disk mode).
        output_dir:
            Artifact directory (disk mode) or ``None`` for in-memory.
        resume:
            Continue an interrupted artifact (disk mode).
        device:
            Optional device for model and stimuli.
        progress:
            Whether to show batch progress.

        Returns
        -------
        Any
            The extraction result (in-memory mapping or shard paths).

        Raises
        ------
        torchlens.errors.InvalidArgumentError
            On a signature drift (code ``extraction_plan_signature_mismatch``)
            or when the planning model is no longer alive
            (code ``extraction_plan_model_gone``).
        """

        model = self._model_ref() if callable(self._model_ref) else None
        if model is None:
            raise InvalidArgumentError(
                "The model this plan was probed against is no longer alive.",
                code="extraction_plan_model_gone",
                remedy="keep the model referenced, or re-plan with a live model",
            )
        self._check_signature(stimuli)
        from .dataset_extraction import extract_dataset

        kwargs: dict[str, Any] = {}
        if stimulus_ids is not None:
            kwargs["stimulus_ids"] = stimulus_ids
        return extract_dataset(
            model,
            stimuli,
            layers=[site.label for site in self.included_sites],
            batch_size=self.batch_size,
            device=device,
            output_dir=output_dir,
            transform=self._transform,
            progress=progress,
            resume=resume,
            **kwargs,
        )


def _probe_once(
    model: nn.Module,
    inputs: torch.Tensor,
    transform: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Run ONE real probe capture and measure per-site facts live.

    Parameters
    ----------
    model:
        Model under plan.
    inputs:
        Probe batch.
    transform:
        Optional unary transform (or coerced pipeline callable) executed
        LIVE per site so post-transform bytes and time are measured, not
        derived (D-15).

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, Any] | None]
        Per-label site facts and the capture's peak pair.
    """

    from . import options as tl_options
    from .user_funcs import trace

    log = trace(model, inputs, capture=tl_options.CaptureOptions(inference_only=True))
    facts: dict[str, dict[str, Any]] = {}
    for op in log.ops:
        if not getattr(op, "has_saved_activation", False):
            continue
        label = str(op.layer_label)
        payload = op.out
        if not isinstance(payload, torch.Tensor):
            continue
        entry: dict[str, Any] = {
            "label": label,
            "site_key": getattr(op, "site_key", None),
            "shape": tuple(payload.shape),
            "dtype": str(payload.dtype).removeprefix("torch."),
            "bytes": int(payload.numel() * payload.element_size()),
            "transformed_shape": None,
            "transformed_bytes": None,
            "transform_seconds": None,
        }
        if transform is not None:
            start = time.perf_counter()
            transformed = transform(payload)
            entry["transform_seconds"] = time.perf_counter() - start
            if isinstance(transformed, torch.Tensor):
                entry["transformed_shape"] = tuple(transformed.shape)
                entry["transformed_bytes"] = int(transformed.numel() * transformed.element_size())
        facts[label] = entry
    pair = log.forward_peak_memory_pair
    log.cleanup()
    return facts, pair


def _build_plan_sites(
    labels: list[str],
    facts_1: dict[str, dict[str, Any]],
    facts_2: dict[str, dict[str, Any]],
    probe_b: int,
) -> list[PlanSite]:
    """Fit per-site plan rows from the two measured probes (D-15).

    Parameters
    ----------
    labels:
        Requested site labels (every one lands a row -- included, or
        excluded with an actionable reason, never silently dropped).
    facts_1:
        Per-site facts from the probe at batch ``probe_b``.
    facts_2:
        Per-site facts from the probe at batch ``2 * probe_b``.
    probe_b:
        The first probe's batch size.

    Returns
    -------
    list[PlanSite]
        One row per requested site with the linear byte fit and
        batch-invariance flag.
    """

    plan_sites: list[PlanSite] = []
    for label in labels:
        fact_1 = facts_1.get(label)
        fact_2 = facts_2.get(label)
        if fact_1 is None:
            plan_sites.append(
                PlanSite(
                    label=label,
                    site_key=None,
                    shape=(),
                    dtype="unknown",
                    bytes_fixed=0,
                    bytes_per_stimulus=0.0,
                    batch_invariant=False,
                    transformed_shape=None,
                    transformed_bytes=None,
                    transform_seconds=None,
                    included=False,
                    exclusion_reason=(
                        "site not observed on the probe forward (check the label "
                        "or capture with save= covering it)"
                    ),
                )
            )
            continue
        bytes_1 = fact_1["bytes"]
        bytes_2 = fact_2["bytes"] if fact_2 is not None else bytes_1
        batch_invariant = bytes_2 == bytes_1
        slope = 0.0 if batch_invariant else (bytes_2 - bytes_1) / probe_b
        intercept = bytes_1 if batch_invariant else int(bytes_1 - slope * probe_b)
        plan_sites.append(
            PlanSite(
                label=label,
                site_key=fact_1["site_key"],
                shape=fact_1["shape"],
                dtype=fact_1["dtype"],
                bytes_fixed=intercept,
                bytes_per_stimulus=slope,
                batch_invariant=batch_invariant,
                transformed_shape=fact_1["transformed_shape"],
                transformed_bytes=fact_1["transformed_bytes"],
                transform_seconds=fact_1["transform_seconds"],
                included=True,
                exclusion_reason=None,
            )
        )
    return plan_sites


def extraction_plan(  # noqa: PLR0913 -- the memo-specified public planner surface (brainpipe D-10/D-11/D-15): every kwarg is a named plan input
    model: nn.Module,
    probe_inputs: torch.Tensor,
    *,
    sites: Any = "all",
    n_stimuli: int,
    batch_size: int,
    transform: Any = None,
    memory_budget: Any = None,
    safety_margin: float = 0.15,
    engine: str = "trace",
) -> ExtractionPlan:
    """Build a measured extraction plan from two real probe captures (D-15).

    Runs the model twice -- at the probe batch size ``b`` and at ``2b`` --
    fits per-site retained bytes linearly in batch size (detecting
    batch-invariant sites, which naive total-scaling over-predicts 8x),
    executes the transform chain live per site so post-transform bytes and
    time are MEASURED, and prices the run's peak from the probes' own peak
    pairs. The plan's table (:meth:`ExtractionPlan.table`) prints all of it
    before anything runs.

    Parameters
    ----------
    model:
        Model to plan against (held weakly; keep it alive through run()).
    probe_inputs:
        One REAL probe batch (leading stimulus dimension >= 1). The plan is
        valid only for this input signature.
    sites:
        ``"all"`` (every saved-activation site on the observed path) or an
        iterable of layer labels to include.
    n_stimuli:
        Total stimuli the run will consume.
    batch_size:
        The run's batch size. Belongs to the user (D-10): the planner
        refuses over-budget plans with arithmetic, never shrinks this.
    transform:
        Optional unary transform executed live at the probe.
    memory_budget:
        Optional budget accepted by :func:`parse_bytes`.
    safety_margin:
        Fraction of the budget held back (default 0.15).
    engine:
        ``"trace"`` only at launch (D-11).

    Returns
    -------
    ExtractionPlan
        The frozen, printable, runnable plan.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        On a non-trace engine (code ``extraction_engine_unsupported``), an
        over-budget plan (code ``extraction_plan_over_budget``), or invalid
        probe inputs (code ``extraction_probe_invalid``).
    """

    import weakref

    if engine != "trace":
        raise InvalidArgumentError(
            f"engine={engine!r} is not available: fastlog records a strict "
            "subset of trace's sites (175 of 389 on ResNet-50), so an engine "
            "swap changes what sites='all' MEANS; it is gated on the "
            "definitional parity suite, and engine='auto' is dead (D-11).",
            code="extraction_engine_unsupported",
            remedy="use engine='trace'",
        )
    if not isinstance(probe_inputs, torch.Tensor) or probe_inputs.ndim < 1:
        raise InvalidArgumentError(
            "probe_inputs must be a tensor with a leading stimulus dimension.",
            code="extraction_probe_invalid",
            remedy="pass one real probe batch, e.g. stimuli[:4]",
        )
    if batch_size <= 0 or n_stimuli <= 0:
        raise InvalidArgumentError(
            f"batch_size and n_stimuli must be positive; got {batch_size} and {n_stimuli}.",
            code="extraction_probe_invalid",
            remedy="pass positive batch_size and n_stimuli",
        )

    budget_bytes = parse_bytes(memory_budget) if memory_budget is not None else None

    probe_b = int(probe_inputs.shape[0])
    doubled = torch.cat([probe_inputs, probe_inputs], dim=0)
    # Warm-up capture, unmeasured: the first capture in a process pays
    # one-time wrapper/import allocations that would contaminate the
    # two-point resident fit (a 100x first-reading skew was measured on a
    # cold process). Its facts and pair are discarded.
    _probe_once(model, probe_inputs, transform)
    facts_1, pair_1 = _probe_once(model, probe_inputs, transform)
    facts_2, pair_2 = _probe_once(model, doubled, transform)

    requested: list[str] | None
    if isinstance(sites, str) and sites == "all":
        requested = None
    else:
        requested = [str(site) for site in sites]
    labels = list(facts_1) if requested is None else requested
    plan_sites = _build_plan_sites(labels, facts_1, facts_2, probe_b)

    # Peak prediction: linear fit of the probes' RESIDENT peaks vs batch,
    # evaluated at the run batch. Honest basis disclosure: this predicts
    # the measured instrumented route (engine, selection, transform,
    # signature), never a model-global constant (memo s3.3).
    predicted_peak: int | None = None
    peak_basis = "unmeasured"
    residents: list[tuple[int, int]] = [
        (batch, int(pair["resident"]))
        for batch, pair in ((probe_b, pair_1), (2 * probe_b, pair_2))
        if pair is not None and pair.get("resident") is not None
    ]
    backend = (pair_1 or {}).get("backend", "unknown")
    if len(residents) == 2 and residents[1][1] >= residents[0][1] > 0:
        slope = (residents[1][1] - residents[0][1]) / (residents[1][0] - residents[0][0])
        intercept = residents[0][1] - slope * residents[0][0]
        predicted_peak = int(max(float(residents[1][1]), intercept + slope * batch_size))
        peak_basis = f"two-point resident fit, {backend}"
    elif residents:
        # A non-monotone or zero pair cannot be fit honestly (a warm CPU
        # allocator legitimately reads 0 growth); the max reading is a
        # disclosed lower-bound estimate, never a fabricated fit.
        predicted_peak = int(max(reading for _, reading in residents))
        peak_basis = f"max probe resident reading (fit rejected: non-monotone), {backend}"

    signature = (
        tuple(probe_inputs.shape[1:]),
        str(probe_inputs.dtype),
        batch_size,
        repr(transform) if transform is not None else None,
        type(model).__qualname__,
    )
    plan = ExtractionPlan(
        sites=tuple(plan_sites),
        batch_size=batch_size,
        n_stimuli=n_stimuli,
        input_signature=signature,
        transform_repr=repr(transform) if transform is not None else None,
        engine=engine,
        memory_budget_bytes=budget_bytes,
        safety_margin=safety_margin,
        probe_batch_sizes=(probe_b, 2 * probe_b),
        probe_peak_pairs=(pair_1, pair_2),
        predicted_peak_bytes=predicted_peak,
        peak_basis=peak_basis,
        model_class=type(model).__qualname__,
        _model_ref=weakref.ref(model),
        _transform=transform,
    )
    # The fitted retained bytes per batch are a deterministic FLOOR on the
    # run's peak (the payloads must exist to be written); the RSS reading is
    # environment-noisy (a warm allocator legitimately reads 0 growth), so
    # the floor keeps the budget arithmetic honest either way.
    fitted_retained = plan.retained_bytes_per_batch
    if predicted_peak is None or fitted_retained > predicted_peak:
        predicted_peak = fitted_retained
        peak_basis = f"fitted retained-bytes floor (resident reading: {peak_basis})"
        object.__setattr__(plan, "predicted_peak_bytes", predicted_peak)
        object.__setattr__(plan, "peak_basis", peak_basis)

    if budget_bytes is not None and predicted_peak is not None:
        usable = int(budget_bytes * (1.0 - safety_margin))
        if predicted_peak > usable:
            raise InvalidArgumentError(
                f"The planned run's predicted peak ({_format_bytes(predicted_peak)}, "
                f"basis: {peak_basis}) exceeds the usable budget "
                f"({_format_bytes(usable)} = {_format_bytes(budget_bytes)} minus the "
                f"{safety_margin:.0%} safety margin) at batch_size={batch_size}. "
                "The batch size is yours (D-10): the planner never shrinks it "
                "silently. Layer chunking is NOT offered: it cannot reduce the "
                "capture base.",
                code="extraction_plan_over_budget",
                remedy=(
                    "reduce batch_size (the only lever that reduces the capture "
                    "base), raise memory_budget, or reduce with a transform"
                ),
            )
    return plan


def export_npz(
    source: Any,
    path: str | Path,
    *,
    layout: str = "consolidated",
    compatibility: str | None = None,
    stimulus_ids: list[str] | None = None,
) -> list[Path]:
    """Export extracted features as npz interchange (D-21).

    Parameters
    ----------
    source:
        In-memory extraction mapping (``{site_label: tensor}``) as returned
        by the in-memory runner, or an extraction artifact directory.
    path:
        Output file (consolidated) or directory (per-stimulus).
    layout:
        ``"consolidated"`` (default): ONE npz with one array per site plus
        a JSON sidecar manifest so numpy-only consumers get provenance.
        ``"per_stimulus"``: one npz PER STIMULUS containing one array per
        site -- Net2Brain's loader layout.
    compatibility:
        ``"net2brain"`` (with ``layout="per_stimulus"``) enforces their
        FILE-NAMING CONTRACT: their encoding loader globs and sorts
        LEXICOGRAPHICALLY, so filenames are zero-padded in stimulus order
        and the sidecar records the name-to-stimulus-id map -- get this
        wrong and rows silently misalign against brain data (memo D-21).
    stimulus_ids:
        Optional per-stimulus identifiers recorded in the sidecar.

    Returns
    -------
    list[pathlib.Path]
        The written npz paths (sidecar excluded).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        On an unknown layout/compatibility combination or non-numeric
        payloads (code ``npz_export_invalid``). Object arrays never write:
        every save uses ``allow_pickle=False`` semantics.
    """

    if layout not in {"consolidated", "per_stimulus"}:
        raise InvalidArgumentError(
            f"Unknown npz layout: {layout!r}.",
            code="npz_export_invalid",
            remedy="use layout='consolidated' or layout='per_stimulus'",
        )
    if compatibility not in {None, "net2brain"}:
        raise InvalidArgumentError(
            f"Unknown npz compatibility mode: {compatibility!r}.",
            code="npz_export_invalid",
            remedy="use compatibility=None or compatibility='net2brain'",
        )
    if compatibility == "net2brain" and layout != "per_stimulus":
        raise InvalidArgumentError(
            "compatibility='net2brain' requires layout='per_stimulus' (their "
            "encoding loader consumes per-stimulus files).",
            code="npz_export_invalid",
            remedy="pass layout='per_stimulus' with compatibility='net2brain'",
        )

    if isinstance(source, (str, Path)):
        from .dataset_extraction import load_extraction

        loaded = load_extraction(source)
        mapping: Any = loaded.activations
    else:
        mapping = source
    arrays: dict[str, np.ndarray] = {}
    for label, value in dict(mapping).items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.dtype == object:
            raise InvalidArgumentError(
                f"Site {label!r} holds an object array; npz interchange is "
                "numeric-only (allow_pickle=False).",
                code="npz_export_invalid",
                remedy="export numeric tensors only",
            )
        arrays[str(label)] = array

    if not arrays:
        raise InvalidArgumentError(
            "There are no site arrays to export.",
            code="npz_export_invalid",
            remedy="run the extraction first",
        )

    target = Path(path)
    if layout == "consolidated":
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **arrays)  # type: ignore[arg-type]  # numpy stub quirk
        sidecar: dict[str, Any] = {
            "format": "torchlens.npz.v1",
            "layout": "consolidated",
            "sites": sorted(arrays),
            "stimulus_ids": stimulus_ids,
        }
        target.with_suffix(".manifest.json").write_text(json.dumps(sidecar, indent=2))
        return [target]
    return _write_per_stimulus_npz(arrays, target, compatibility, stimulus_ids)


def _write_per_stimulus_npz(
    arrays: dict[str, np.ndarray],
    target: Path,
    compatibility: str | None,
    stimulus_ids: list[str] | None,
) -> list[Path]:
    """Write the per-stimulus npz layout with the D-21 naming contract.

    Zero-padded filenames sort lexicographically in stimulus order (the
    Net2Brain loader globs + sorts; misalignment against brain data is
    silent otherwise) and the sidecar records the name-to-stimulus-id map.

    Parameters
    ----------
    arrays:
        Validated numeric site arrays with a shared leading stimulus axis.
    target:
        Output directory.
    compatibility:
        ``None`` or ``"net2brain"`` (validated by the caller).
    stimulus_ids:
        Optional per-stimulus identifiers.

    Returns
    -------
    list[pathlib.Path]
        The written npz paths (sidecar excluded).
    """

    row_counts = {array.shape[0] for array in arrays.values()}
    if len(row_counts) != 1:
        raise InvalidArgumentError(
            f"per-stimulus layout requires one consistent stimulus count across "
            f"sites; got row counts {sorted(row_counts)}.",
            code="npz_export_invalid",
            remedy="export sites with identical stimulus counts",
        )
    n_stimuli = row_counts.pop()
    if stimulus_ids is not None and len(stimulus_ids) != n_stimuli:
        raise InvalidArgumentError(
            f"stimulus_ids has {len(stimulus_ids)} entries but the arrays hold "
            f"{n_stimuli} stimuli.",
            code="npz_export_invalid",
            remedy="pass one id per stimulus",
        )
    target.mkdir(parents=True, exist_ok=True)
    width = max(5, len(str(n_stimuli - 1)))
    written: list[Path] = []
    name_to_id: dict[str, str | None] = {}
    for index in range(n_stimuli):
        name = f"stimulus_{index:0{width}d}.npz"
        per_site = {label: array[index] for label, array in arrays.items()}
        file_path = target / name
        np.savez(file_path, **per_site)
        written.append(file_path)
        name_to_id[name] = stimulus_ids[index] if stimulus_ids is not None else None
    per_stim_sidecar: dict[str, Any] = {
        "format": "torchlens.npz.v1",
        "layout": "per_stimulus",
        "compatibility": compatibility,
        "sites": sorted(arrays),
        "n_stimuli": n_stimuli,
        "name_to_stimulus_id": name_to_id,
        "naming_contract": (
            "filenames sort lexicographically in stimulus order; consumers that "
            "glob+sort (Net2Brain) read rows aligned"
        ),
    }
    (target / "manifest.json").write_text(json.dumps(per_stim_sidecar, indent=2))
    return written
