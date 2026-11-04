"""Controls for an ablation claim: does its effect survive other baselines, directions and sites?

"Ablating X moves the metric, so X matters" is the most common intervention claim and
the easiest to get wrong. Three reruns test it with the shipped edit primitives:

- **Baseline family.** The same site ablated with zeros (``zero_ablate``), with the donor
  rows' mean (``mean_ablate``) and with coherent donor rows (``resample_rows_from``). A
  zero fill is off-distribution; an effect that only the zero fill produces measures the
  fill, not the information the site carries.
- **Direction family.** When the claim names a direction, projecting it out is compared
  with projecting out N random directions of the same norm. The rank p-value is
  ``(1 + #{random effects at least as large}) / (N + 1)``.
- **Site family.** The claimed ablation rerun at named neighbouring sites (for example the
  pre-norm and post-norm tensors of the same stream), where a real effect should hold.

A control row survives when its effect has the claimed effect's sign and at least
``min_ratio`` of its size; the direction family survives when ``p <= alpha``. The claim
survives when every family that ran survives.

This module is the private implementation of the controls helper (walk card recipes.4,
ruled 3 Nov 2026). Its public spelling is pending, so it is reachable only as
``torchlens.mechinterp._controls.run_controls``; its argument refusals are plain
``ValueError`` until a typed code is ruled with the name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .._user_funcs import trace as _capture
from ..intervention.helpers import mean_ablate, project_out, zero_ablate
from ..intervention.population import population
from ..intervention.spec import when
from ..intervention.stochastic import resample_rows_from
from ..intervention.types import Edit
from ..options import CaptureOptions

__all__ = ("ControlRow", "ControlsReport", "FamilyVerdict", "run_controls")

#: The ablation baselines, in report order.
BASELINES = ("zero", "mean", "resample")


@dataclass(frozen=True)
class ControlRow:
    """One rerun: which family, which control, its metric effect and its survival.

    ``survives`` is ``None`` for the random-direction rows (they are the null the
    claimed direction is ranked against) and for the claimed edit's own row.
    """

    family: str
    name: str
    effect: float
    survives: bool | None


@dataclass(frozen=True)
class FamilyVerdict:
    """Whether one control family survived, and why in one line."""

    family: str
    survives: bool
    reason: str


@dataclass(frozen=True)
class ControlsReport:
    """Every control rerun of one ablation claim and the verdict per family.

    Attributes
    ----------
    clean:
        The metric on the unedited run.
    effect:
        The claimed ablation's effect: edited metric minus clean metric.
    baseline:
        The claimed ablation's baseline (``"zero"``, ``"mean"`` or ``"resample"``).
    rows:
        One row per rerun, in run order.
    families:
        One verdict per family that ran.
    direction_p:
        The direction family's rank p-value, or ``None`` when no direction was given.
    """

    clean: float
    effect: float
    baseline: str
    rows: tuple[ControlRow, ...]
    families: tuple[FamilyVerdict, ...]
    direction_p: float | None = None

    @property
    def survives(self) -> bool:
        """``True`` when every family that ran survived."""

        return bool(self.families) and all(family.survives for family in self.families)

    def __str__(self) -> str:
        """Plain-text table of the reruns and the per-family verdicts."""

        lines = [
            f"Controls for a {self.baseline} ablation: clean metric {self.clean:.4g}, "
            f"effect {self.effect:+.4g}",
            f"{'family':<10} {'control':<14} {'effect':>10}  survives",
        ]
        for row in self.rows:
            mark = "-" if row.survives is None else ("yes" if row.survives else "no")
            lines.append(f"{row.family:<10} {row.name:<14} {row.effect:>+10.4g}  {mark}")
        for family in self.families:
            verdict = "survives" if family.survives else "does not survive"
            lines.append(f"{family.family}: {verdict} ({family.reason})")
        lines.append("verdict: " + ("survives" if self.survives else "does not survive"))
        return "\n".join(lines)


@dataclass(frozen=True)
class _Settings:
    """The validated knobs of one controls run."""

    axis: int
    baseline: str
    feature_axis: int | None
    n_directions: int
    seed: int
    min_ratio: float
    alpha: float


class _Runner:
    """Reruns the model under one edit at a time and measures the metric's change."""

    def __init__(self, model: nn.Module, inputs: Any, metric: Callable[[Any], Any], donors: Any):
        self.model = model
        self.inputs = inputs
        self.metric = metric
        self.donors = inputs if donors is None else donors
        self._donor_values: dict[int, torch.Tensor] = {}
        self.clean = self._measure(None)

    def _measure(self, intervene: Any) -> float:
        """Capture once (optionally edited) and return the metric as a float."""

        with torch.no_grad():
            trace = _capture(self.model, self.inputs, intervene=intervene)
        try:
            return _scalar(self.metric(trace))
        finally:
            trace.cleanup()

    def effect(self, site: Any, edit: Edit) -> float:
        """Metric change when ``edit`` is applied at ``site``."""

        return self._measure(when(site, edit)) - self.clean

    def donor_value(self, site: Any) -> torch.Tensor:
        """The donor run's value at ``site`` (refuses a site that fires other than once)."""

        key = id(site)
        if key not in self._donor_values:
            fired: list[torch.Tensor] = []

            def _grab(out: Any, *, ctx: Any) -> Any:
                """Keep a detached copy of the fired value."""

                if isinstance(out, torch.Tensor):
                    fired.append(out.detach().clone())
                return out

            with torch.no_grad():
                trace = _capture(
                    self.model, self.donors, capture=CaptureOptions(hooks={site: _grab})
                )
            trace.cleanup()
            if len(fired) != 1:
                raise ValueError(
                    f"the controls need a site that names one tensor; {site!r} fired "
                    f"{len(fired)} times on the donor run. Pass a selector that matches "
                    "exactly one op (a label, or a module whose output is one tensor)."
                )
            self._donor_values[key] = fired[0]
        return self._donor_values[key]


def _scalar(value: Any) -> float:
    """Read a metric result as a float; refuse anything with more than one element."""

    tensor = torch.as_tensor(value).detach()
    if tensor.numel() != 1:
        raise ValueError(
            f"the metric must return one number, got {tuple(tensor.shape)} values. "
            "Reduce it to a scalar (for example .mean()) inside the metric."
        )
    return float(tensor.reshape(()))


def _validate(settings: _Settings, direction: torch.Tensor | None) -> None:
    """Refuse settings the controls cannot honestly run with."""

    problems = []
    if settings.baseline not in BASELINES:
        problems.append(f"baseline must be one of {BASELINES}, got {settings.baseline!r}")
    if settings.n_directions < 1:
        problems.append(f"n_directions must be at least 1, got {settings.n_directions}")
    if not 0.0 < settings.min_ratio <= 1.0:
        problems.append(f"min_ratio must be in (0, 1], got {settings.min_ratio}")
    if not 0.0 < settings.alpha < 1.0:
        problems.append(f"alpha must be in (0, 1), got {settings.alpha}")
    if direction is not None and settings.feature_axis is None:
        problems.append("a direction needs feature_axis= (the site axis it lives on)")
    if direction is not None and (direction.ndim != 1 or not bool(direction.norm() > 0)):
        problems.append("direction must be a 1-D tensor with nonzero norm")
    if problems:
        raise ValueError("; ".join(problems))


def _baseline_edit(runner: _Runner, name: str, site: Any, settings: _Settings) -> Edit:
    """Build the ``name`` ablation of ``site``, reading donor rows when it needs them."""

    if name == "zero":
        return zero_ablate()
    donor = runner.donor_value(site)
    if name == "mean":
        return mean_ablate(donor, dim=settings.axis)
    rows = population(donor.movedim(settings.axis, 0), origin="controls donor rows")
    return resample_rows_from(rows, seed=settings.seed, axis=settings.axis)


def _agrees(effect: float, claimed: float, min_ratio: float) -> bool:
    """Same sign as the claimed effect and at least ``min_ratio`` of its size."""

    return claimed != 0.0 and effect * claimed > 0.0 and abs(effect) >= min_ratio * abs(claimed)


def _agreement_family(
    family: str, rows: list[ControlRow], claimed: float, min_ratio: float
) -> FamilyVerdict:
    """Summarise a family whose rows each must agree with the claimed effect."""

    failed = [row.name for row in rows if row.survives is False]
    if not failed:
        return FamilyVerdict(family, True, f"every control keeps at least {min_ratio:g} of it")
    return FamilyVerdict(
        family,
        False,
        f"{', '.join(failed)} keep{'s' if len(failed) == 1 else ''} under {min_ratio:g} "
        f"of the effect {claimed:+.4g} or flip its sign",
    )


def _baseline_family(
    runner: _Runner, site: Any, claimed: float, settings: _Settings
) -> tuple[list[ControlRow], FamilyVerdict]:
    """Rerun the site under the two baselines the claim did not use."""

    rows = [ControlRow("baseline", settings.baseline, claimed, None)]
    for name in BASELINES:
        if name == settings.baseline:
            continue
        effect = runner.effect(site, _baseline_edit(runner, name, site, settings))
        survives = _agrees(effect, claimed, settings.min_ratio)
        rows.append(ControlRow("baseline", name, effect, survives))
    return rows, _agreement_family("baseline", rows, claimed, settings.min_ratio)


def _random_directions(direction: torch.Tensor, count: int, seed: int) -> list[torch.Tensor]:
    """``count`` seeded Gaussian directions with ``direction``'s shape, dtype and norm."""

    generator = torch.Generator().manual_seed(seed)
    norm = direction.detach().float().norm()
    out = []
    for _ in range(count):
        draw = torch.randn(direction.shape, generator=generator)
        out.append((draw * (norm / draw.norm())).to(direction.dtype))
    return out


def _direction_family(
    runner: _Runner, site: Any, direction: torch.Tensor, settings: _Settings
) -> tuple[list[ControlRow], FamilyVerdict, float]:
    """Rank projecting out ``direction`` against projecting out matched random directions."""

    axis = settings.feature_axis
    claimed = runner.effect(site, project_out(direction, dim=axis))
    nulls = [
        runner.effect(site, project_out(draw, dim=axis))
        for draw in _random_directions(direction, settings.n_directions, settings.seed)
    ]
    at_least = sum(1 for value in nulls if abs(value) >= abs(claimed))
    p_value = (1 + at_least) / (len(nulls) + 1)
    survives = p_value <= settings.alpha and claimed != 0.0
    rows = [ControlRow("direction", "claimed", claimed, survives)]
    rows += [ControlRow("direction", f"random {i}", v, None) for i, v in enumerate(nulls)]
    reason = (
        f"p = {p_value:.3g} against {len(nulls)} random directions of the same norm "
        f"({at_least} at least as large); alpha {settings.alpha:g}"
    )
    return rows, FamilyVerdict("direction", survives, reason), p_value


def _site_family(
    runner: _Runner, sites: Mapping[str, Any], claimed: float, settings: _Settings
) -> tuple[list[ControlRow], FamilyVerdict]:
    """Rerun the claimed ablation at each named neighbouring site."""

    rows = []
    for name, other in sites.items():
        effect = runner.effect(other, _baseline_edit(runner, settings.baseline, other, settings))
        survives = _agrees(effect, claimed, settings.min_ratio)
        rows.append(ControlRow("site", str(name), effect, survives))
    return rows, _agreement_family("site", rows, claimed, settings.min_ratio)


def _run_families(
    runner: _Runner,
    site: Any,
    claimed: float,
    direction: torch.Tensor | None,
    sites: Mapping[str, Any] | None,
    settings: _Settings,
) -> tuple[list[ControlRow], list[FamilyVerdict], float | None]:
    """Run the baseline family and the direction and site families that were asked for."""

    rows, verdict = _baseline_family(runner, site, claimed, settings)
    families = [verdict]
    direction_p = None
    if direction is not None:
        direction_rows, verdict, direction_p = _direction_family(runner, site, direction, settings)
        rows += direction_rows
        families.append(verdict)
    if sites:
        site_rows, verdict = _site_family(runner, sites, claimed, settings)
        rows += site_rows
        families.append(verdict)
    if claimed == 0.0:
        families = [FamilyVerdict("claim", False, "the claimed ablation does not move the metric")]
    return rows, families, direction_p


def run_controls(  # noqa: PLR0913 -- the claim plus each family's stated knobs
    model: nn.Module,
    inputs: Any,
    site: Any,
    metric: Callable[[Any], Any],
    *,
    axis: int,
    baseline: str = "zero",
    donors: Any = None,
    direction: torch.Tensor | None = None,
    feature_axis: int | None = None,
    n_directions: int = 20,
    sites: Mapping[str, Any] | None = None,
    seed: int = 0,
    min_ratio: float = 0.5,
    alpha: float = 0.05,
) -> ControlsReport:
    """Rerun an ablation claim under the baseline, direction and site controls.

    Parameters
    ----------
    model, inputs:
        The model and the input every run uses; the model runs once per control.
    site:
        Selector of the ablated tensor; it must fire exactly once.
    metric:
        ``Trace -> number``: the quantity the claim is about.
    axis:
        The site value's batch axis, which ``mean`` averages over and ``resample`` draws
        rows along. Explicit, never guessed from rank.
    baseline:
        The ablation the claim used: ``"zero"``, ``"mean"`` or ``"resample"``.
    donors:
        Input whose rows feed the mean and resample baselines (default ``inputs``).
    direction, feature_axis:
        Optional 1-D direction the claim is about and the site axis it lives on; runs
        the direction family against ``n_directions`` matched random directions.
    sites:
        Optional ``name -> selector`` of neighbouring sites; runs the site family.
    seed:
        Seed of the random directions and the resample draw.
    min_ratio, alpha:
        Share of the claimed effect a baseline or site control must keep (same sign),
        and the largest direction-family p-value that survives.

    Returns
    -------
    ControlsReport
        Every rerun, the verdict per family and ``survives`` overall.
    """

    settings = _Settings(axis, baseline, feature_axis, n_directions, seed, min_ratio, alpha)
    _validate(settings, direction)
    runner = _Runner(model, inputs, metric, donors)
    claimed = runner.effect(site, _baseline_edit(runner, baseline, site, settings))
    rows, families, direction_p = _run_families(runner, site, claimed, direction, sites, settings)
    report_rows, verdicts = tuple(rows), tuple(families)
    return ControlsReport(runner.clean, claimed, baseline, report_rows, verdicts, direction_p)
