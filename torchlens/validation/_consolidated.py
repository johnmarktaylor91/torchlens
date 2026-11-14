"""Consolidated validation entry point for TorchLens 2.0."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from torchlens.backends.registry import BackendUnsupportedError

from .._entry_refusals import refuse_validation_scope
from .._literals import OutputDeviceLiteral
from .._options_validation import _validate_output_device
from .._save_budget import SaveBudgetOption, resolve_save_budget
from .._utils._display_card import render_card
from .._utils.display import user_stacklevel
from .._utils.rng import restores_rng_streams
from ..backends._backend_names import TORCH_BACKEND_NAME
from ..backends.registry import BackendName, resolve_backend_spec
from ..errors import TorchLensWarning
from ..options import CaptureOptions
from ._backward import validate_backward_pass
from ._diagnostics import (
    CHECK_METADATA_INVARIANT,
    ValidationFailure,
    last_validation_failure,
    record_validation_failure,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
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
    is_invariant:
        Whether a baseline forward validation succeeds.
    is_specific:
        Whether hook targets can be represented without ambiguity.
    is_complete:
        Whether the validation exercised all requested axes.
    is_consistent:
        Whether repeated validation decisions agree.
    is_local:
        Whether intervention checks stay local to the requested model/input.
    details:
        Human-readable axis details.

    Examples
    --------
    >>> report = tl.validate(model, x, scope="intervention")
    >>> isinstance(report, InterventionValidationReport)
    True
    >>> report.is_invariant
    True
    """

    is_invariant: bool
    is_specific: bool
    is_complete: bool
    is_consistent: bool
    is_local: bool
    details: dict[str, Any]

    def __repr__(self) -> str:
        """Bounded identity card: the five verdicts lead, ``details`` is counted."""

        axes = (
            self.is_invariant,
            self.is_specific,
            self.is_complete,
            self.is_consistent,
            self.is_local,
        )
        verdict = "passed" if self.passed else "failed"
        return f"InterventionValidationReport: {verdict}, {sum(axes)} of 5 axes passed"

    def __str__(self) -> str:
        """Show every recorded axis verdict without recomputing a check."""

        return render_card(
            repr(self),
            [
                (label, "passed" if getattr(self, name) else "failed")
                for label, name in (
                    ("invariant", "is_invariant"),
                    ("specific", "is_specific"),
                    ("complete", "is_complete"),
                    ("agreement", "is_consistent"),
                    ("local", "is_local"),
                )
            ],
            (".details",),
        )

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
                self.is_invariant,
                self.is_specific,
                self.is_complete,
                self.is_consistent,
                self.is_local,
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


#: Scopes whose validation runs the saving validator capture that honors the
#: ``output_device`` / ``save_budget`` save options.
_SAVE_OPTION_SCOPES = frozenset({"forward", "saved", "intervention"})
_FORWARD_HARNESS_SCOPES = frozenset({"forward", "saved"})


def _validate_save_option_keywords(
    scope: str,
    *,
    output_device: OutputDeviceLiteral,
    save_budget: SaveBudgetOption,
) -> None:
    """Validate the validator capture's save options and their scope.

    The values are checked with the same validators the capture entry uses,
    BEFORE any forward runs, so an invalid value raises the capture's own typed
    error. A non-default value on a scope whose validation does not run the
    saving validator capture is refused rather than silently ignored.

    Parameters
    ----------
    scope:
        Normalized validation scope.
    output_device:
        Requested ``output_device`` for the validator capture.
    save_budget:
        Requested ``save_budget`` for the validator capture.

    Raises
    ------
    InvalidArgumentError
        If either value is invalid (``output_device_invalid`` /
        ``save_budget_invalid``).
    TypeError
        If a non-default value is passed for ``scope="backward"`` or
        ``scope="receptive_field"``.
    """

    _validate_output_device(output_device)
    resolve_save_budget(save_budget)
    if scope in _SAVE_OPTION_SCOPES:
        return
    allowed = "scope='forward', 'saved', or 'intervention'"
    if output_device != "same":
        raise TypeError(f"output_device only valid for {allowed}")
    if save_budget != "auto":
        raise TypeError(f"save_budget only valid for {allowed}")


def _save_option_kwargs(
    output_device: OutputDeviceLiteral,
    save_budget: SaveBudgetOption,
) -> dict[str, Any]:
    """Return the save-option keywords to forward, omitting default values.

    Parameters
    ----------
    output_device:
        Requested ``output_device``.
    save_budget:
        Requested ``save_budget``.

    Returns
    -------
    dict[str, Any]
        Only the non-default options, so a default call forwards exactly the
        keywords it always did.
    """

    forwarded: dict[str, Any] = {}
    if output_device != "same":
        forwarded["output_device"] = output_device
    if save_budget != "auto":
        forwarded["save_budget"] = save_budget
    return forwarded


def _refuse_forward_only_keywords(
    scope: str,
    num_threads: int | None,
    trace_observer: Callable[[Trace | None], None] | None,
) -> None:
    """Refuse ``num_threads`` and ``trace_observer`` outside the forward/saved scopes.

    Parameters
    ----------
    scope:
        Normalized validation scope.
    num_threads, trace_observer:
        The requested keywords; ``None`` means omitted.

    Raises
    ------
    InvalidArgumentError
        ``validation_scope_invalid``, naming the keyword, when either is passed
        for ``scope="backward"``, ``"intervention"`` or ``"receptive_field"``.
    """

    if scope in _FORWARD_HARNESS_SCOPES:
        return
    for name, value in (("num_threads", num_threads), ("trace_observer", trace_observer)):
        if value is not None:
            refuse_validation_scope(scope, _FORWARD_HARNESS_SCOPES, keyword=name)


def _intervention_report(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    *,
    seed: int | None,
    verbose: bool,
    validate_metadata: bool,
    save_options: dict[str, Any] | None = None,
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
    seed:
        Optional random seed for forward validation.
    verbose:
        Whether the underlying validation should emit diagnostics.
    validate_metadata:
        Whether metadata invariant checks should run.
    save_options:
        Non-default ``output_device`` / ``save_budget`` keywords for the
        validator capture, or ``None``.

    Returns
    -------
    InterventionValidationReport
        Structured intervention validation result whose non-baseline axes stay
        false until real intervention-specific checks exist.
    """

    from .._user_public_impls import validate_forward_pass

    forward_ok = validate_forward_pass(
        model,
        input_args,
        input_kwargs=input_kwargs,
        seed=seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
        **(save_options or {}),
    )
    return InterventionValidationReport(
        is_invariant=forward_ok,
        is_specific=False,
        is_complete=False,
        is_consistent=False,
        is_local=False,
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


@dataclass(frozen=True)
class _ReceptiveFieldRunOptions:
    """Capture settings of one receptive-field validation run."""

    seed: int | None
    validate_metadata: bool
    backend: BackendName | None


def _validate_receptive_field_scope(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    run: _ReceptiveFieldRunOptions,
) -> list[ReceptiveFieldValidation]:
    """Capture and sample both RF containment directions at layer centers.

    Parameters
    ----------
    model:
        Model to capture.
    input_args, input_kwargs:
        Model inputs.
    run:
        Optional deterministic capture seed, whether always-on metadata
        contracts run on the captured trace, and the explicit backend selector.

    Returns
    -------
    list[ReceptiveFieldValidation]
        Sampled receptive and projective tri-state results.
    """

    from .._user_public_impls import trace
    from ..receptive_field._validation import validate_receptive_field_trace  # deferred: cycle
    from ._invariants import check_metadata_invariants

    ready_args = _gradient_ready_value(input_args)
    ready_kwargs = _gradient_ready_value(input_kwargs)
    captured = trace(
        model,
        ready_args,
        input_kwargs=ready_kwargs,
        capture=CaptureOptions(backward_ready=True, seed=run.seed, save_mode="reference"),
        save="all",
        backend=run.backend,
    )
    if run.validate_metadata:
        check_metadata_invariants(captured)
    receptive = validate_receptive_field_trace(captured, direction="receptive")
    projective = validate_receptive_field_trace(captured, direction="projective")
    return [*receptive, *projective]


@contextlib.contextmanager
def _validation_peak_probe() -> Iterator[None]:
    """Record opt-in peak-memory observations around one forward/saved validation.

    Validation is the product's largest transient peak. Cheap peak observations
    are published through ``last_validation_peak_memory()``; measurement only,
    never part of the verdict. The probes are OPT-IN
    (``TORCHLENS_VALIDATE_PEAK_MEMORY=1``), mirroring the capture-side
    ``measure_python_peak_memory`` design: the ``torch.cuda.is_available()``
    probe can trigger driver init on some setups, so default calls pay nothing.
    """

    from .._data_classes.cleanup import _trim_host_allocator  # deferred: startup cost
    from .._utils.env_flags import closed_bool_env  # deferred: startup cost
    from ._diagnostics import _LAST_RUN_PEAKS  # deferred: startup cost

    peaks_enabled = closed_bool_env("TORCHLENS_VALIDATE_PEAK_MEMORY")
    _LAST_RUN_PEAKS.clear()
    rss_before = _rss_high_water_bytes() if peaks_enabled else None
    cuda_armed = peaks_enabled and torch.cuda.is_available() and torch.cuda.is_initialized()
    # Snapshot the peak instead of reset_peak_memory_stats, which clobbers the
    # caller's process-wide high-water counter; a run under it honestly reads 0.
    cuda_peak_before = int(torch.cuda.max_memory_allocated()) if cuda_armed else 0
    try:
        yield
    finally:
        rss_after = _rss_high_water_bytes() if peaks_enabled else None
        if rss_before is not None and rss_after is not None:
            _LAST_RUN_PEAKS["host_rss_peak_delta_bytes"] = max(0, rss_after - rss_before)
        if cuda_armed:
            cuda_peak_after = int(torch.cuda.max_memory_allocated())
            _LAST_RUN_PEAKS["cuda_peak_allocated_bytes"] = (
                cuda_peak_after if cuda_peak_after > cuda_peak_before else 0
            )
        # The comparison/report phase reallocates AFTER the internal trims, so
        # freed glibc arena would stay resident per call; one more trim at the
        # end, after the peak reads (high-water maxima the trim cannot lower).
        _trim_host_allocator()


def _validate_forward_scope(
    model: nn.Module,
    inputs: Any,
    input_kwargs: dict[str, Any] | None,
    *,
    scope: str,
    forward_options: dict[str, Any],
) -> bool:
    """Run the forward/saved validation, settling every failure class as ``False``.

    A metadata-invariant failure is ONE more way the verdict is ``False``: the
    invariant pass already recorded its identity on the side channel before
    re-raising (``validation/_core.py``); a record is made sure to exist for the
    R67 warning, then the verdict settles ``False``.

    Parameters
    ----------
    model, inputs, input_kwargs:
        Model and inputs to validate.
    scope:
        Normalized scope, ``"forward"`` or ``"saved"``.
    forward_options:
        Keywords forwarded to ``validate_forward_pass``, ``verbose`` included;
        its ``verbose`` also governs the failure report.

    Returns
    -------
    bool
        The validation verdict.
    """

    from .._user_public_impls import validate_forward_pass  # deferred: cycle
    from ._invariants import MetadataInvariantError  # deferred: startup cost

    with _validation_peak_probe():
        try:
            passed = validate_forward_pass(
                model, inputs, input_kwargs=input_kwargs, **forward_options
            )
        except MetadataInvariantError as invariant_error:
            _record_invariant_failure_if_missing(invariant_error)
            passed = False
    if passed is False:
        _report_validation_failure(scope, forward_options["verbose"])
    return passed


@restores_rng_streams
def validate(
    model: nn.Module,
    inputs: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    scope: str = "forward",
    seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    validate_layer_grads: bool | None = None,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
    backend: BackendName | None = None,
    output_device: OutputDeviceLiteral = "same",
    save_budget: SaveBudgetOption = "auto",
    num_threads: int | None = None,
    trace_observer: Callable[[Trace | None], None] | None = None,
) -> bool | InterventionValidationReport | list[ReceptiveFieldValidation]:
    """Validate a model/input pair for a requested TorchLens scope.

    Parameters
    ----------
    model:
        Model to validate.
    inputs:
        Positional model input.
    input_kwargs:
        Keyword model input.
    scope:
        Validation scope: ``"forward"`` (default), ``"backward"``, ``"saved"``,
        ``"intervention"``, or ``"receptive_field"``. ``"receptive_field"``
        captures with ``save_mode="reference"``, so a model whose forward
        writes its own buffers in place (a train-mode BatchNorm's running
        statistics) raises ``MutatedReferenceError``; validate it in eval mode.
    seed:
        Optional random seed for forward-like validation. Like ``tl.trace``, the call
        restores the caller's global RNG streams on return and on error.
    verbose:
        Whether validators should emit diagnostics.
    validate_metadata:
        Whether metadata invariant checks should run for forward-like scopes.
    loss_fn:
        Backward-only loss function.
    atol:
        Backward-only absolute tolerance. ``None`` (default) derives per gradient
        dtype in ``validate_backward_pass``.
    rtol:
        Backward-only relative tolerance. ``None`` derives per dtype likewise.
    validate_layer_grads:
        Backward-only layer-gradient validation flag. Omission enables honest
        captured-gradient validation by default for backward scope. The root
        module call is never compared, so a model whose forward calls no child
        module returns False with a ``RuntimeWarning``.
    layer_grad_atol:
        Backward-only layer-gradient absolute tolerance.
    layer_grad_rtol:
        Backward-only layer-gradient relative tolerance.
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.
    output_device:
        Where the validator capture keeps its saved activations, with the
        same values and default as ``CaptureOptions.output_device``
        (``"same"``, ``"cpu"``, or ``"cuda"``). ``"cpu"`` lets a GPU model
        validate with its activations held in host memory; replay still runs
        each op on its original device. Saved function arguments (the
        replay inputs) other than parameters are kept there too and moved
        back to each op's device for its check, so the GPU keeps only the
        parameters and one forward's transients. Honored by the
        ``"forward"``, ``"saved"``, and ``"intervention"`` scopes; a
        non-default value on another scope raises ``TypeError``.
    save_budget:
        Per-device ceiling on the validator capture's retained bytes, with
        the same values and default as ``CaptureOptions.save_budget``
        (``"auto"``, a float fraction in ``(0, 1]``, an int byte cap, or
        ``None`` to disable). Exceeding it raises
        :class:`~torchlens.errors.SaveBudgetExceededError` exactly as the
        capture does. Same scopes as ``output_device``.
    num_threads:
        Torch intra-op threads for every validation forward (ground truth,
        capture, retrace, replay); the prior count is restored on return and on
        error. ``None`` keeps the process setting. Forward and saved scopes,
        torch backend only.
    trace_observer:
        Called exactly once per forward/saved run with the validation ``Trace``
        after replay and before cleanup, or with ``None`` when a precondition
        refusal returns ``False`` before any capture; not called when the call
        raises. It may read public fields (``len(trace.ops)``,
        ``trace.provenance.fingerprint``, ``trace.validation_replay_status``);
        the trace is cleaned up after it returns, so keep values, not the
        trace. An exception it raises propagates after teardown restores
        state. It never changes the verdict. Same scopes as ``num_threads``.

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
    :class:`~torchlens.validation._invariants.MetadataInvariantError` raised by
    the invariant pass) is ONE more way the verdict is ``False``: the bool
    contract holds for every failure class (a stale-parent model returning
    ``False`` while a worker-thread escape raised through the same door would
    make callers branching on the bool miss the second class). The
    invariant itself is untouched and still records its identity on the side
    channel; the lower-level :func:`torchlens.validation.validate_forward_pass`
    door keeps raising it.

    Examples
    --------
    >>> tl.validate(model, x, scope="forward")
    True
    >>> tl.validate(model, x, scope="saved")
    True
    """

    normalized_scope = scope.lower()
    valid_scopes = {"forward", "backward", "saved", "intervention", "receptive_field"}
    if normalized_scope not in valid_scopes:
        refuse_validation_scope(scope, valid_scopes)
    _validate_scope_keywords(
        normalized_scope,
        loss_fn=loss_fn,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )
    _validate_save_option_keywords(
        normalized_scope, output_device=output_device, save_budget=save_budget
    )
    _refuse_forward_only_keywords(normalized_scope, num_threads, trace_observer)
    save_options = _save_option_kwargs(output_device, save_budget)
    if normalized_scope == "backward":
        spec = resolve_backend_spec(backend, model, inputs, input_kwargs)
        if spec.name != TORCH_BACKEND_NAME or not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward validation."
            )
        return validate_backward_pass(
            model,
            inputs,
            input_kwargs=input_kwargs,
            loss_fn=loss_fn,
            validate_metadata=validate_metadata,
            seed=seed,
            atol=atol,
            rtol=rtol,
            validate_layer_grads=(True if validate_layer_grads is None else validate_layer_grads),
            layer_grad_atol=layer_grad_atol,
            layer_grad_rtol=layer_grad_rtol,
        )
    if normalized_scope == "receptive_field":
        spec = resolve_backend_spec(backend, model, inputs, input_kwargs)
        if spec.name != TORCH_BACKEND_NAME or not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support receptive-field validation."
            )
        return _validate_receptive_field_scope(
            model,
            inputs,
            input_kwargs,
            _ReceptiveFieldRunOptions(seed, validate_metadata, backend),
        )

    if normalized_scope in {"forward", "saved"}:
        return _validate_forward_scope(
            model,
            inputs,
            input_kwargs,
            scope=normalized_scope,
            forward_options={
                "verbose": verbose,
                "seed": seed,
                "validate_metadata": validate_metadata,
                "backend": backend,
                "num_threads": num_threads,
                "trace_observer": trace_observer,
                **save_options,
            },
        )
    return _intervention_report(
        model,
        inputs,
        input_kwargs,
        seed=seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
        save_options=save_options,
    )


def _report_validation_failure(normalized_scope: str, verbose: bool) -> None:
    """Warn once that validation failed, naming the check and op; print it when verbose.

    R67: the bare ``False`` used to be silent while the rich structured
    diagnosis sat unreferenced in the module side channel. One warning names
    WHAT failed and WHERE the full record lives. ``verbose=True`` also prints
    the same line through the validators' verbose channel (stdout), because a
    failure decided at a site that prints nothing (a perturbation-insensitive
    edge, an unverified replay) otherwise left a verbose run with no output.
    """

    from ._diagnostics import last_validation_failure

    failure = last_validation_failure()
    detail = f" {failure.summary()}" if failure is not None else ""
    if verbose:
        print(f"tl.validate FAILED for scope={normalized_scope!r}.{detail}")
    warnings.warn(
        f"tl.validate FAILED for scope={normalized_scope!r}.{detail} Full "
        "structured diagnosis: "
        "torchlens.validation.last_validation_failure() / "
        "get_validation_diagnostics().",
        TorchLensWarning,
        stacklevel=user_stacklevel(),
    )


def _record_invariant_failure_if_missing(invariant_error: BaseException) -> None:
    """Record a metadata-invariant failure on the side channel when none is present."""

    if last_validation_failure() is not None:
        return
    record_validation_failure(
        None,
        ValidationFailure(
            check_name=CHECK_METADATA_INVARIANT,
            message=f"{type(invariant_error).__name__}: {invariant_error}",
        ),
    )


__all__ = ["InterventionValidationReport", "validate"]
