"""Consolidated validation entry point for TorchLens 2.0."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..backends import BackendName, BackendUnsupportedError, resolve_backend_spec
from ..errors import TorchLensWarning
from ..options import CaptureOptions
from ..utils.display import user_stacklevel
from .backward import validate_backward_pass

if TYPE_CHECKING:
    from ..receptive_field._types import ReceptiveFieldValidation


def _rss_high_water_bytes() -> int | None:
    """Return the process RSS high-water mark in bytes, or ``None`` off-POSIX.

    Returns
    -------
    int | None
        ``ru_maxrss`` scaled to bytes (kilobytes on Linux, bytes on macOS),
        or ``None`` when the ``resource`` module is unavailable.
    """

    try:
        import resource
    except ImportError:
        return None
    import sys as _sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if _sys.platform == "darwin" else int(peak) * 1024


@dataclass(frozen=True)
class InterventionValidationReport:
    """Five-axis intervention validation result.

    Parameters
    ----------
    invariance:
        Whether a baseline forward validation succeeds.
    specificity:
        Whether hook targets can be represented without ambiguity.
    completeness:
        Whether the validation exercised all requested axes.
    consistency:
        Whether repeated validation decisions agree.
    locality:
        Whether intervention checks stay local to the requested model/input.
    details:
        Human-readable axis details.
    """

    invariance: bool
    specificity: bool
    completeness: bool
    consistency: bool
    locality: bool
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return the aggregate pass/fail result.

        Returns
        -------
        bool
            True when all axes pass.
        """

        return all(
            (
                self.invariance,
                self.specificity,
                self.completeness,
                self.consistency,
                self.locality,
            )
        )

    def __bool__(self) -> bool:
        """Return the aggregate pass/fail result for truth-value checks.

        Returns
        -------
        bool
            True when all axes pass.
        """

        return self.passed


def _raise_backward_only(name: str, scope: str) -> None:
    """Raise a per-scope keyword visibility error.

    Parameters
    ----------
    name:
        Keyword name.
    scope:
        Requested validation scope.

    Raises
    ------
    TypeError
        Always raised.
    """

    raise TypeError(f"{name} only valid for scope='backward'")


def _validate_scope_keywords(
    scope: str,
    *,
    loss_fn: Callable[[Any], torch.Tensor] | None,
    atol: float | None,
    rtol: float | None,
    validate_layer_grads: bool | None,
    layer_grad_atol: float | None,
    layer_grad_rtol: float | None,
) -> None:
    """Validate that backward-only keywords are scoped correctly.

    Parameters
    ----------
    scope:
        Requested validation scope.
    loss_fn:
        Optional backward loss function.
    atol:
        Backward absolute tolerance (``None`` = dtype-derived default).
    rtol:
        Backward relative tolerance (``None`` = dtype-derived default).
    validate_layer_grads:
        Backward layer-gradient validation flag, or None when omitted.
    layer_grad_atol:
        Backward layer-gradient absolute tolerance.
    layer_grad_rtol:
        Backward layer-gradient relative tolerance.
    """

    if scope == "backward":
        return
    if loss_fn is not None:
        _raise_backward_only("loss_fn", scope)
    if atol is not None:
        _raise_backward_only("atol", scope)
    if rtol is not None:
        _raise_backward_only("rtol", scope)
    if validate_layer_grads is not None:
        _raise_backward_only("validate_layer_grads", scope)
    if layer_grad_atol is not None:
        _raise_backward_only("layer_grad_atol", scope)
    if layer_grad_rtol is not None:
        _raise_backward_only("layer_grad_rtol", scope)


def _intervention_report(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    *,
    random_seed: int | None,
    verbose: bool,
    validate_metadata: bool,
) -> InterventionValidationReport:
    """Build an honesty-preserving intervention validation report.

    Parameters
    ----------
    model:
        Model to validate.
    input_args:
        Positional model input.
    input_kwargs:
        Keyword model input.
    random_seed:
        Optional random seed for forward validation.
    verbose:
        Whether the underlying validation should emit diagnostics.
    validate_metadata:
        Whether metadata invariant checks should run.

    Returns
    -------
    InterventionValidationReport
        Structured intervention validation result whose non-baseline axes stay
        false until real intervention-specific checks exist.
    """

    from ..user_funcs import validate_forward_pass

    forward_ok = validate_forward_pass(
        model,
        input_args,
        input_kwargs=input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )
    return InterventionValidationReport(
        invariance=forward_ok,
        specificity=False,
        completeness=False,
        consistency=False,
        locality=False,
        details={
            "invariance": "forward validation passed"
            if forward_ok
            else "forward validation failed",
            "specificity": (
                "not evaluated: intervention validation did not inspect selector "
                "specificity on this path"
            ),
            "completeness": ("not evaluated: only baseline forward validation ran on this path"),
            "consistency": (
                "not evaluated: intervention cross-run consistency is not implemented here"
            ),
            "locality": ("not evaluated: intervention-locality checks did not run on this path"),
        },
    )


def _gradient_ready_value(value: Any) -> Any:
    """Clone floating tensor inputs with autograd enabled for RF probing.

    Parameters
    ----------
    value:
        Arbitrarily nested model input value.

    Returns
    -------
    Any
        Structure-equivalent input with floating tensors made gradient-ready.
    """

    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if torch.is_floating_point(cloned) or torch.is_complex(cloned):
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, tuple):
        return tuple(_gradient_ready_value(item) for item in value)
    if isinstance(value, list):
        return [_gradient_ready_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _gradient_ready_value(item) for key, item in value.items()}
    return value


def _validate_receptive_field_scope(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    *,
    random_seed: int | None,
    validate_metadata: bool,
    backend: BackendName | None,
) -> list[ReceptiveFieldValidation]:
    """Capture and sample both RF containment directions at layer centers.

    Parameters
    ----------
    model:
        Model to capture.
    input_args, input_kwargs:
        Model inputs.
    random_seed:
        Optional deterministic capture seed.
    validate_metadata:
        Whether always-on metadata contracts should run on the captured trace.
    backend:
        Explicit backend selector.

    Returns
    -------
    list[ReceptiveFieldValidation]
        Sampled receptive and projective tri-state results.
    """

    from ..receptive_field._validation import validate_receptive_field_trace
    from ..user_funcs import trace
    from .invariants import check_metadata_invariants

    ready_args = _gradient_ready_value(input_args)
    ready_kwargs = _gradient_ready_value(input_kwargs)
    captured = trace(
        model,
        ready_args,
        input_kwargs=ready_kwargs,
        save_mode="reference",
        capture=CaptureOptions(
            layers_to_save="all",
            backward_ready=True,
            random_seed=random_seed,
        ),
        backend=backend,
    )
    if validate_metadata:
        check_metadata_invariants(captured)
    receptive = validate_receptive_field_trace(captured, direction="receptive")
    projective = validate_receptive_field_trace(captured, direction="projective")
    return [*receptive, *projective]


def validate(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    scope: str,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    validate_layer_grads: bool | None = None,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
    backend: BackendName | None = None,
) -> bool | InterventionValidationReport | list[ReceptiveFieldValidation]:
    """Validate a model/input pair for a requested TorchLens scope.

    Parameters
    ----------
    model:
        Model to validate.
    input_args:
        Positional model input.
    input_kwargs:
        Keyword model input.
    scope:
        Validation scope: ``"forward"``, ``"backward"``, ``"saved"``,
        ``"intervention"``, or ``"receptive_field"``.
    random_seed:
        Optional random seed for forward-like validation.
    verbose:
        Whether validators should emit diagnostics.
    validate_metadata:
        Whether metadata invariant checks should run for forward-like scopes.
    loss_fn:
        Backward-only loss function.
    atol:
        Backward-only absolute tolerance. ``None`` (default) derives per
        gradient dtype in ``validate_backward_pass``.
    rtol:
        Backward-only relative tolerance. ``None`` derives per dtype
        likewise.
    validate_layer_grads:
        Backward-only layer-gradient validation flag. Omission enables honest
        captured-gradient validation by default for backward scope.
    layer_grad_atol:
        Backward-only layer-gradient absolute tolerance.
    layer_grad_rtol:
        Backward-only layer-gradient relative tolerance.
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.

    Returns
    -------
    bool | InterventionValidationReport | list[ReceptiveFieldValidation]
        Validation pass/fail for forward, backward, and saved scopes, or an
        intervention validation report for ``scope="intervention"``.

    Notes
    -----
    A forward/saved ``False`` also emits one :class:`TorchLensWarning`
    summarizing the failure; the full structured record is available from
    :func:`torchlens.validation.last_validation_failure` and
    :func:`torchlens.validation.get_validation_diagnostics`.

    A metadata-invariant failure on the forward/saved scopes (a
    :class:`~torchlens.validation.invariants.MetadataInvariantError` raised by
    the invariant pass) is ONE more way the verdict is ``False``: the bool
    contract holds for every failure class (W051-HONESTY L9 -- stale-parent
    models returned ``False`` while a worker-thread escape raised through the
    same door, so callers branching on the bool missed the second class). The
    invariant itself is untouched and still records its identity on the side
    channel; the lower-level :func:`torchlens.validation.validate_forward_pass`
    door keeps raising it.
    """

    normalized_scope = scope.lower()
    valid_scopes = {"forward", "backward", "saved", "intervention", "receptive_field"}
    if normalized_scope not in valid_scopes:
        raise ValueError(f"scope must be one of {sorted(valid_scopes)!r}.")
    _validate_scope_keywords(
        normalized_scope,
        loss_fn=loss_fn,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )
    if normalized_scope == "backward":
        spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward validation."
            )
        return validate_backward_pass(
            model,
            input_args,
            input_kwargs=input_kwargs,
            loss_fn=loss_fn,
            validate_metadata=validate_metadata,
            random_seed=random_seed,
            atol=atol,
            rtol=rtol,
            validate_layer_grads=(True if validate_layer_grads is None else validate_layer_grads),
            layer_grad_atol=layer_grad_atol,
            layer_grad_rtol=layer_grad_rtol,
        )
    if normalized_scope == "receptive_field":
        spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support receptive-field validation."
            )
        return _validate_receptive_field_scope(
            model,
            input_args,
            input_kwargs,
            random_seed=random_seed,
            validate_metadata=validate_metadata,
            backend=backend,
        )

    from ..user_funcs import validate_forward_pass

    if normalized_scope in {"forward", "saved"}:
        # R33-2: validation is the product's largest transient peak and had no
        # instrumentation. Record cheap peak observations around the run and
        # publish them through ``last_validation_peak_memory()``; measurement
        # only, never part of the verdict. R33 follow-up: the probes are an
        # OPT-IN (TORCHLENS_VALIDATE_PEAK_MEMORY=1), mirroring the
        # measure_python_peak_memory capture-side design -- the
        # torch.cuda.is_available() probe can trigger driver init on some
        # setups and none of it feeds the verdict, so default validate calls
        # pay nothing. last_validation_peak_memory() already returns None for
        # the empty off-state.
        from ..utils.env_flags import closed_bool_env
        from .diagnostics import _LAST_RUN_PEAKS

        peaks_enabled = closed_bool_env("TORCHLENS_VALIDATE_PEAK_MEMORY")
        _LAST_RUN_PEAKS.clear()
        rss_before = _rss_high_water_bytes() if peaks_enabled else None
        cuda_armed = peaks_enabled and torch.cuda.is_available() and torch.cuda.is_initialized()
        cuda_peak_before = 0
        if cuda_armed:
            # R36-2: snapshot the peak instead of reset_peak_memory_stats,
            # which clobbers the caller's process-wide high-water counter.
            # A run that stays under the pre-existing peak honestly reads 0.
            cuda_peak_before = int(torch.cuda.max_memory_allocated())
        from .invariants import MetadataInvariantError

        try:
            try:
                passed = validate_forward_pass(
                    model,
                    input_args,
                    input_kwargs=input_kwargs,
                    random_seed=random_seed,
                    verbose=verbose,
                    validate_metadata=validate_metadata,
                    backend=backend,
                )
            except MetadataInvariantError as invariant_error:
                # L9: the bool contract covers EVERY failure class. The
                # invariant pass already recorded its identity on the side
                # channel before re-raising (validation/core.py); make sure a
                # record exists for the R67 warning below, then settle False.
                _record_invariant_failure_if_missing(invariant_error)
                passed = False
        finally:
            rss_after = _rss_high_water_bytes() if peaks_enabled else None
            if rss_before is not None and rss_after is not None:
                _LAST_RUN_PEAKS["host_rss_peak_delta_bytes"] = max(0, rss_after - rss_before)
            if cuda_armed:
                cuda_peak_after = int(torch.cuda.max_memory_allocated())
                _LAST_RUN_PEAKS["cuda_peak_allocated_bytes"] = (
                    cuda_peak_after if cuda_peak_after > cuda_peak_before else 0
                )
            # R33 (r7 b6-opus, 3rd round): validate's high-water phase sits on
            # the WRONG side of the internal trims -- the comparison/report
            # phase reallocates AFTER second_trace.cleanup() trimmed, so
            # ~180-225 MB of freed glibc arena stayed resident per call
            # (measured: manual malloc_trim recovered it). One more trim at
            # the END of the scope; correctness-neutral (releases only
            # allocator-free pages, ~1 ms next to a multi-second validate),
            # and after the peak reads above, which are high-water maxima the
            # trim cannot lower.
            from ..data_classes.cleanup import _trim_host_allocator

            _trim_host_allocator()
        if passed is False:
            # R67: the bare ``False`` used to be silent while the rich
            # structured diagnosis sat unreferenced in the module side channel.
            # One warning names WHAT failed and WHERE the full record lives.
            from .diagnostics import last_validation_failure

            failure = last_validation_failure()
            detail = f" {failure.summary()}" if failure is not None else ""
            warnings.warn(
                f"tl.validate FAILED for scope={normalized_scope!r}.{detail} Full "
                "structured diagnosis: "
                "torchlens.validation.last_validation_failure() / "
                "get_validation_diagnostics().",
                TorchLensWarning,
                stacklevel=user_stacklevel(),
            )
        return passed
    return _intervention_report(
        model,
        input_args,
        input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def _record_invariant_failure_if_missing(invariant_error: BaseException) -> None:
    """Record a metadata-invariant failure on the side channel when none is present."""

    from .diagnostics import (
        CHECK_METADATA_INVARIANT,
        ValidationFailure,
        last_validation_failure,
        record_validation_failure,
    )

    if last_validation_failure() is not None:
        return
    record_validation_failure(
        None,
        ValidationFailure(
            check=CHECK_METADATA_INVARIANT,
            message=f"{type(invariant_error).__name__}: {invariant_error}",
        ),
    )


__all__ = ["InterventionValidationReport", "validate"]
