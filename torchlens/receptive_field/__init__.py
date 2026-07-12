"""Lazy public namespace for TorchLens receptive-field analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._errors import (
    AmbiguousCallError,
    AmbiguousInputError,
    AmbiguousTargetError,
    AmbiguousPassError,
    BackendUnsupportedError,
    ReceptiveFieldError,
    NoInfluencePathError,
    ReceptiveFieldUnavailableError,
    ReceptiveFieldValidationError,
)
from ._types import (
    GradientReceptiveField,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldDirection,
    ReceptiveFieldProfile,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
    ReceptiveFieldValidationStatus,
    ReceptiveFieldViolation,
)
from ._rules import ReceptiveFieldRule, ReceptiveFieldRuleContext, register_rf_rule, rules
from ._validation import cross_validate
from ._view import ReceptiveFieldView
from ._viz import node_spec


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


@dataclass(frozen=True)
class EmpiricalAdjointCheck:
    """One sampled comparison of receptive and projective empirical derivatives."""

    source_label: str
    target_label: str
    source_unit: tuple[int, ...]
    target_unit: tuple[int, ...]
    passed: bool | None
    receptive_value: float | None
    projective_value: float | None
    message: str | None = None


@dataclass(frozen=True)
class ReceptiveFieldVerification:
    """Combined containment and empirical-adjoint diagnostic report."""

    containment: tuple[ReceptiveFieldValidation, ...]
    empirical_adjoint: tuple[EmpiricalAdjointCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every definitive containment and adjoint check passed."""

        containment_passed = all(result.passed for result in self.containment)
        adjoint_passed = all(result.passed is not False for result in self.empirical_adjoint)
        return containment_passed and adjoint_passed


def _op_by_label(trace: Trace, label: str) -> Op | None:
    """Resolve one canonical operation from an exact pass-qualified label.

    Parameters
    ----------
    trace:
        Trace owning the operation.
    label:
        Exact pass-qualified operation label.

    Returns
    -------
    Op or None
        Canonical captured operation when present.
    """

    return next((op for op in trace.layer_list if op.label == label), None)


def _empirical_adjoint_checks(
    trace: Trace,
    containment: tuple[ReceptiveFieldValidation, ...],
    *,
    atol: float,
    rtol: float,
) -> tuple[EmpiricalAdjointCheck, ...]:
    """Compare sampled saved-graph VJP rows and double-VJP columns.

    Parameters
    ----------
    trace:
        Backward-ready trace owning the checked endpoints.
    containment:
        Completed containment checks whose empirical receptive gradients supply
        the sampled Jacobian entries.
    atol, rtol:
        Floating-point comparison tolerances for this diagnostic only.

    Returns
    -------
    tuple[EmpiricalAdjointCheck, ...]
        One reported comparison or graceful skip per receptive empirical result.
    """

    from ._gradient import gradient_for_unit
    from ._gradient_forward import projective_gradient_for_unit

    checks: list[EmpiricalAdjointCheck] = []
    for validation in containment:
        for role, receptive in validation.gradient.items():
            far_label = next(
                (op.label for op in trace.layer_list if str(op.io_role or op.label) == role),
                "",
            )
            support = torch.nonzero(receptive.support_mask, as_tuple=False)
            if validation.direction is ReceptiveFieldDirection.RECEPTIVE:
                source = _op_by_label(trace, far_label)
                target = _op_by_label(trace, validation.op_label)
                source_unit = (
                    ()
                    if support.numel() == 0
                    else tuple(int(value) for value in support[0].tolist())
                )
                target_unit = validation.unit
            else:
                source = _op_by_label(trace, validation.op_label)
                target = _op_by_label(trace, far_label)
                source_unit = validation.unit
                target_unit = (
                    ()
                    if support.numel() == 0
                    else tuple(int(value) for value in support[0].tolist())
                )
            if source is None or target is None or support.numel() == 0:
                checks.append(
                    EmpiricalAdjointCheck(
                        source_label="" if source is None else source.label,
                        target_label="" if target is None else target.label,
                        source_unit=source_unit,
                        target_unit=target_unit,
                        passed=None,
                        receptive_value=None,
                        projective_value=None,
                        message="No supported source element was available for an adjoint sample.",
                    )
                )
                continue
            try:
                if validation.direction is ReceptiveFieldDirection.RECEPTIVE:
                    projective = projective_gradient_for_unit(
                        source,
                        source_unit,
                        target=target,
                        atol=0.0,
                        rtol=0.0,
                        retain_graph=True,
                    )
                    receptive_probe = receptive
                else:
                    receptive_probe = gradient_for_unit(
                        target,
                        target_unit,
                        source=source,
                        atol=0.0,
                        rtol=0.0,
                        retain_graph=True,
                    )
                    projective = receptive
            except (BackendUnsupportedError, ReceptiveFieldError) as exc:
                checks.append(
                    EmpiricalAdjointCheck(
                        source_label=source.label,
                        target_label=target.label,
                        source_unit=source_unit,
                        target_unit=validation.unit,
                        passed=None,
                        receptive_value=None,
                        projective_value=None,
                        message=str(exc),
                    )
                )
                continue
            if not isinstance(projective, GradientReceptiveField) or not isinstance(
                receptive_probe, GradientReceptiveField
            ):
                checks.append(
                    EmpiricalAdjointCheck(
                        source_label=source.label,
                        target_label=target.label,
                        source_unit=source_unit,
                        target_unit=validation.unit,
                        passed=None,
                        receptive_value=None,
                        projective_value=None,
                        message="The selected target produced a non-scalar projective result mapping.",
                    )
                )
                continue
            receptive_value = receptive_probe.grad[source_unit]
            projective_value = projective.grad[target_unit]
            checks.append(
                EmpiricalAdjointCheck(
                    source_label=source.label,
                    target_label=target.label,
                    source_unit=source_unit,
                    target_unit=validation.unit,
                    passed=bool(
                        torch.allclose(receptive_value, projective_value, atol=atol, rtol=rtol)
                    ),
                    receptive_value=float(receptive_value.item()),
                    projective_value=float(projective_value.item()),
                )
            )
    return tuple(checks)


def verify(
    trace: Trace,
    *,
    empirical_adjoint_atol: float = 1e-6,
    empirical_adjoint_rtol: float = 1e-5,
    **kwargs: object,
) -> ReceptiveFieldVerification:
    """Run containment and sampled empirical-adjoint RF diagnostics.

    Parameters
    ----------
    trace:
        Backward-ready trace to inspect.
    empirical_adjoint_atol, empirical_adjoint_rtol:
        Non-negative floating-point comparison tolerances used only for the
        reported equality of two empirical derivative probes.
    **kwargs:
        Keyword arguments accepted by :func:`cross_validate`.

    Returns
    -------
    ReceptiveFieldVerification
        Containment results together with one empirical-adjoint report for each
        sampled receptive gradient.
    """

    if empirical_adjoint_atol < 0 or empirical_adjoint_rtol < 0:
        raise ValueError("empirical_adjoint_atol and empirical_adjoint_rtol must be non-negative.")
    containment = tuple(cross_validate(trace, **kwargs))  # type: ignore[arg-type]
    return ReceptiveFieldVerification(
        containment=containment,
        empirical_adjoint=_empirical_adjoint_checks(
            trace,
            containment,
            atol=empirical_adjoint_atol,
            rtol=empirical_adjoint_rtol,
        ),
    )


def self_check(
    trace: Trace,
    *,
    empirical_adjoint_atol: float = 1e-6,
    empirical_adjoint_rtol: float = 1e-5,
    **kwargs: object,
) -> ReceptiveFieldVerification:
    """Alias :func:`verify` for interactive RF self-consistency diagnostics."""

    return verify(
        trace,
        empirical_adjoint_atol=empirical_adjoint_atol,
        empirical_adjoint_rtol=empirical_adjoint_rtol,
        **kwargs,
    )


__all__ = [
    "AmbiguousCallError",
    "AmbiguousInputError",
    "AmbiguousTargetError",
    "AmbiguousPassError",
    "BackendUnsupportedError",
    "EmpiricalAdjointCheck",
    "GradientReceptiveField",
    "GridLayout",
    "ReceptiveField",
    "ReceptiveFieldAlignment",
    "ReceptiveFieldAxis",
    "ReceptiveFieldBox",
    "ReceptiveFieldBoxAxis",
    "ReceptiveFieldDirection",
    "ReceptiveFieldError",
    "NoInfluencePathError",
    "ReceptiveFieldProfile",
    "ReceptiveFieldRule",
    "ReceptiveFieldRuleContext",
    "ReceptiveFieldStatus",
    "ReceptiveFieldUnavailableError",
    "ReceptiveFieldValidation",
    "ReceptiveFieldValidationError",
    "ReceptiveFieldValidationStatus",
    "ReceptiveFieldVerification",
    "ReceptiveFieldView",
    "ReceptiveFieldViolation",
    "register_rf_rule",
    "cross_validate",
    "node_spec",
    "self_check",
    "verify",
    "rules",
]
