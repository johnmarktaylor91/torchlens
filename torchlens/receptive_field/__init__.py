"""Lazy public namespace for TorchLens receptive-field analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from ..utils.tensor_utils import layer_grad_tolerances_for_dtype
from ._errors import (
    AmbiguousCallError,
    AmbiguousInputError,
    AmbiguousPassError,
    AmbiguousTargetError,
    BackendUnsupportedError,
    NoInfluencePathError,
    ReceptiveFieldConfigurationError,
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
    ReceptiveFieldValidationError,
)
from ._rules import (
    ReceptiveFieldRule,
    ReceptiveFieldRuleContext,
    register_rf_rule,
    rules as _registered_rules,
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
from ._validation import cross_validate
from ._view import ReceptiveFieldView

# Importing the package executes built-in rule decorators once, before any public
# descriptor, projective, or validation path can reach the geometry engines.
from .rules import __all__ as _builtin_rule_modules  # noqa: F401

rules = _registered_rules


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from ._viz import node_spec


def __getattr__(name: str) -> object:
    """Resolve the Pillow-backed drawing helper ``node_spec`` on first access.

    ``_viz`` imports Pillow at module level, and forward validation imports
    this package for its error types, so an eager import would make
    ``tl.validate`` fail where Pillow is not installed.

    Parameters
    ----------
    name:
        Attribute requested from the package.

    Returns
    -------
    object
        The resolved attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily resolved attribute.
    """

    if name == "node_spec":
        from ._viz import node_spec

        return node_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    def verdict(self) -> ReceptiveFieldValidationStatus:
        """Return the tri-state verdict, never conflating unarmed with wrong.

        ``FAIL`` reports a real violation: a containment ``FAIL`` or a
        definitive empirical-adjoint mismatch. ``INDETERMINATE`` means at
        least one containment check could not be evaluated (typically a trace
        captured without ``requires_grad`` inputs, ``backward_ready=True``,
        and ``save_mode="reference"``) and no check failed. ``PASS`` requires
        every containment check to pass with no adjoint mismatch; adjoint
        samples that were structurally unavailable (``passed is None``) never
        substitute for a failed or unarmed containment check.
        """

        containment_failed = any(
            result.status is ReceptiveFieldValidationStatus.FAIL for result in self.containment
        )
        adjoint_failed = any(check.passed is False for check in self.empirical_adjoint)
        if containment_failed or adjoint_failed:
            return ReceptiveFieldValidationStatus.FAIL
        if not self.containment or any(
            result.status is ReceptiveFieldValidationStatus.INDETERMINATE
            for result in self.containment
        ):
            return ReceptiveFieldValidationStatus.INDETERMINATE
        return ReceptiveFieldValidationStatus.PASS

    @property
    def passed(self) -> bool:
        """Return whether the verdict is ``PASS``.

        ``INDETERMINATE`` stays ``False`` — an unarmed tripwire never reads
        as a pass — and is distinguishable from ``FAIL`` via ``verdict``.
        """

        return self.verdict is ReceptiveFieldValidationStatus.PASS


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
    atol: float | None,
    rtol: float | None,
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
        ``None`` derives per compared-gradient dtype via
        ``layer_grad_tolerances_for_dtype`` (R13 consumer wiring); an
        explicit float applies to every dtype unchanged.

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
                    # A single source selects a single field, never the role mapping.
                    receptive_probe = cast(
                        GradientReceptiveField,
                        gradient_for_unit(
                            target,
                            target_unit,
                            source=source,
                            atol=0.0,
                            rtol=0.0,
                            retain_graph=True,
                        ),
                    )
                    projective = receptive
            except (BackendUnsupportedError, ReceptiveFieldError) as exc:
                checks.append(
                    EmpiricalAdjointCheck(
                        source_label=source.label,
                        target_label=target.label,
                        source_unit=source_unit,
                        target_unit=target_unit,
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
                        target_unit=target_unit,
                        passed=None,
                        receptive_value=None,
                        projective_value=None,
                        message="The selected target produced a non-scalar projective result mapping.",
                    )
                )
                continue
            # The adjoint identity is a SIGNED equality: dL/dx via the
            # receptive probe must equal the same partial via the projective
            # probe, sign included. ``grad`` stores magnitudes for
            # influence-set semantics, so compare the retained signed values
            # (falling back to magnitudes only for legacy results predating
            # ``signed_grad``, where sign disagreements were invisible).
            receptive_tensor = (
                receptive_probe.signed_grad
                if receptive_probe.signed_grad is not None
                else receptive_probe.grad
            )
            projective_tensor = (
                projective.signed_grad if projective.signed_grad is not None else projective.grad
            )
            receptive_value = receptive_tensor[source_unit]
            projective_value = projective_tensor[target_unit]
            checks.append(
                EmpiricalAdjointCheck(
                    source_label=source.label,
                    target_label=target.label,
                    source_unit=source_unit,
                    target_unit=target_unit,
                    passed=bool(
                        torch.allclose(
                            receptive_value,
                            projective_value,
                            atol=(
                                atol
                                if atol is not None
                                else layer_grad_tolerances_for_dtype(receptive_value.dtype)[1]
                            ),
                            rtol=(
                                rtol
                                if rtol is not None
                                else layer_grad_tolerances_for_dtype(receptive_value.dtype)[0]
                            ),
                            equal_nan=True,
                        )
                    ),
                    receptive_value=float(receptive_value.item()),
                    projective_value=float(projective_value.item()),
                )
            )
    return tuple(checks)


def verify(
    trace: Trace,
    *,
    empirical_adjoint_atol: float | None = None,
    empirical_adjoint_rtol: float | None = None,
    **kwargs: object,
) -> ReceptiveFieldVerification:
    """Run containment and sampled empirical-adjoint RF diagnostics.

    Scope (R74/75-7): every empirical probe here — gradient containment and
    both sides of the adjoint equality — backpropagates through the ONE
    autograd graph built during the wrapped capture forward. The oracle is
    therefore independent of the geometric DERIVATION (it catches TorchLens
    indexing, sampling, and rule bugs) but shares its root with capture: a
    hypothetical capture-time forward corruption would deceive geometry and
    gradients alike, so PASS here does not re-attest capture fidelity. Capture
    fidelity is owned by the ``torchlens.validation`` replay tripwire, which
    re-executes ops against independently recomputed inputs.

    Parameters
    ----------
    trace:
        Backward-ready trace to inspect.
    empirical_adjoint_atol, empirical_adjoint_rtol:
        Non-negative floating-point comparison tolerances used only for the
        reported equality of two empirical derivative probes. ``None``
        (default) derives the pair per compared-gradient dtype via
        ``layer_grad_tolerances_for_dtype`` (fp32 resolves to the legacy
        layer-grad constants).
    **kwargs:
        Keyword arguments accepted by :func:`cross_validate`.

    Returns
    -------
    ReceptiveFieldVerification
        Containment results together with one empirical-adjoint report for each
        sampled receptive gradient.
    """

    if (empirical_adjoint_atol is not None and empirical_adjoint_atol < 0) or (
        empirical_adjoint_rtol is not None and empirical_adjoint_rtol < 0
    ):
        raise ReceptiveFieldConfigurationError(
            "empirical_adjoint_atol and empirical_adjoint_rtol must be non-negative."
        )
    containment = tuple(cross_validate(trace, **kwargs))  # type: ignore[arg-type]
    if not any(key in kwargs for key in ("direction", "inputs", "source", "target")):
        # Sweep the projective direction too: exact-box corner cross-checks
        # there are what expose a spurious-nonempty forward claim, which
        # receptive-only containment is structurally unable to see.
        containment += tuple(
            cross_validate(trace, direction="projective", **kwargs)  # type: ignore[arg-type]
        )
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
    empirical_adjoint_atol: float | None = None,
    empirical_adjoint_rtol: float | None = None,
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
    "ReceptiveFieldConfigurationError",
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
