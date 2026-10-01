"""Convenience constructor for value-sweep intervention bundles."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._deprecations import MISSING, MissingType
from .._errors import ArgumentTypeError, InvalidArgumentError
from ..bundle import Bundle
from ..errors import TorchLensWarning
from .hooks import HookContext
from .selectors import BaseSelector, func, label
from .spec import InterventionSpec, when
from .types import HelperSpec

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

SWEEP_NAME = "sweep"


def sweep(
    model: nn.Module,
    x: Any,
    at: str | BaseSelector | Callable[[Any], bool] | MissingType = MISSING,
    values: Iterable[Any] | MissingType = MISSING,
    *,
    input_kwargs: dict[Any, Any] | None = None,
    names: Sequence[str] | None = None,
    include_baseline: bool = False,
    **trace_kwargs: Any,
) -> Bundle:
    """Capture one intervened trace per swept replacement value.

    Parameters
    ----------
    model:
        PyTorch module to capture.
    x:
        Positional input argument or argument container passed to ``tl.trace``.
    at:
        Intervention site target. Strings match either an exact TorchLens label
        or a function name; selectors and predicate callables are used directly.
    values:
        Replacement values to sweep at ``at``.
    input_kwargs:
        Optional keyword inputs passed to ``model.forward``.
    names:
        Optional Bundle member names. When omitted, names are derived from
        ``SWEEP_NAME`` and the value index.
    include_baseline:
        Whether to ALSO capture one PRISTINE (un-intervened) trace first,
        named ``"baseline"`` and set as the Bundle baseline — without it a
        sweep bundle has no comparison anchor and ``most_changed`` raises
        ``BaselineUndeterminedError`` (F03 ledger memo item 2). The default
        ``False`` keeps the legacy member cardinality and DISCLOSES the
        missing baseline at construction with one coded warning.
    **trace_kwargs:
        Additional keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    Bundle
        Bundle containing one trace per swept value (plus the pristine
        ``"baseline"`` member first, under ``include_baseline=True``).

    Raises
    ------
    ValueError
        If no values are provided, names do not match values, a member name
        collides with ``"baseline"``, or an explicit ``intervene`` argument
        is supplied.
    TypeError
        If ``at`` cannot be used as a capture-time intervention predicate.
    """

    # Resolve positional `values` (may be MISSING if it was skipped when param was keyword-only)
    if values is MISSING:
        raise ArgumentTypeError(
            "sweep() is missing its required values iterable",
            code="sweep_values_missing",
            remedy="pass a non-empty iterable as values",
            argument="values",
        )

    if "intervene" in trace_kwargs:
        raise InvalidArgumentError(
            "sweep() received intervene even though it constructs its own intervention",
            code="sweep_intervention_conflict",
            remedy="remove intervene and express the target with the at argument",
            argument="intervene",
        )

    # C03 spec door: the ONE immutable InterventionSpec is accepted unchanged
    # as sweep values -- one member per spec, each captured with
    # intervene=<that spec>. Site and action already live inside each spec,
    # so at= must be omitted, and mixing spec and plain values is refused
    # (a plain value NEEDS at=, a spec FORBIDS it; guessing per-element
    # would silently re-target half the sweep).
    from collections.abc import Iterable as _EarlyIterable
    from typing import cast as _cast

    early_values = list(_cast("_EarlyIterable[Any]", values))
    spec_count = sum(isinstance(value, InterventionSpec) for value in early_values)
    if spec_count:
        if spec_count != len(early_values):
            raise InvalidArgumentError(
                f"sweep() received {spec_count} InterventionSpec values mixed "
                f"with {len(early_values) - spec_count} plain replacement "
                "values; a sweep is either one site swept over plain values "
                "(at=..., values=[v1, v2]) or one member per spec "
                "(values=[spec1, spec2])",
                code="sweep_spec_values_mixed",
                remedy="pass all-spec values without at=, or all-plain values with at=",
                argument="values",
            )
        if at is not MISSING:
            raise InvalidArgumentError(
                "sweep(at=..., values=[InterventionSpec, ...]) conflicts: each "
                "spec already names its own WHERE and action, so at= has "
                "nothing sound to address",
                code="sweep_spec_at_conflict",
                remedy="drop at= when sweeping over InterventionSpec values",
                argument="at",
            )
        return _sweep_over_specs(
            model,
            x,
            early_values,
            input_kwargs=input_kwargs,
            names=names,
            include_baseline=include_baseline,
            **trace_kwargs,
        )

    if at is MISSING:
        raise ArgumentTypeError(
            "sweep() is missing its required at site target",
            code="sweep_site_missing",
            remedy="pass at as a label, selector, or predicate callable",
            argument="at",
        )
    else:
        resolved_at = at

    # At this point both resolved_at and values are fully resolved (not MISSING).
    from typing import cast

    resolved_at_typed = cast("str | BaseSelector | Callable[[Any], bool]", resolved_at)

    # early_values already materialized the iterable once (generators would
    # be exhausted by a second pass).
    swept_values = early_values
    if not swept_values:
        raise InvalidArgumentError(
            "sweep() received an empty values iterable",
            code="sweep_values_empty",
            remedy="pass at least one replacement value",
            argument="values",
        )
    if names is not None and len(names) != len(swept_values):
        raise InvalidArgumentError(
            f"sweep() received {len(names)} names for {len(swept_values)} values",
            code="sweep_names_length_mismatch",
            remedy="pass exactly one name per replacement value or omit names",
            argument="names",
        )

    site = _coerce_sweep_site(resolved_at_typed)
    member_names = list(names) if names is not None else _default_member_names(len(swept_values))
    traces = {}
    from ..user_funcs import trace as _trace

    baseline = _mint_pristine_baseline(
        model,
        x,
        member_names=member_names,
        include_baseline=include_baseline,
        input_kwargs=input_kwargs,
        **trace_kwargs,
    )
    for member_name, value in zip(member_names, swept_values, strict=True):
        # C03 (ledger memo item 1): the swept value rides a TYPED builtin
        # helper spec whose args carry the value, never an anonymous closure
        # -- before this, a swept member's only value linkage was its default
        # name (measured amnesia: byte-identical audit rows across members).
        traces[member_name] = _trace(
            model,
            x,
            input_kwargs=input_kwargs,
            intervene=when(site, sweep_replace(value)),
            **trace_kwargs,
        )
    return _assemble_sweep_bundle(
        baseline,
        traces,
        params={
            "values": [repr(value) for value in swept_values],
            "include_baseline": include_baseline,
        },
    )


def _sweep_over_specs(
    model: nn.Module,
    x: Any,
    specs: list[InterventionSpec],
    *,
    names: Sequence[str] | None,
    include_baseline: bool = False,
    **trace_kwargs: Any,
) -> Bundle:
    """Capture one intervened trace per swept InterventionSpec (C03 spec door).

    ``input_kwargs`` rides ``**trace_kwargs`` (it is forwarded verbatim to
    every capture alongside the other trace kwargs).
    """

    if not specs:
        raise InvalidArgumentError(
            "sweep() received an empty values iterable",
            code="sweep_values_empty",
            remedy="pass at least one replacement value",
            argument="values",
        )
    if names is not None and len(names) != len(specs):
        raise InvalidArgumentError(
            f"sweep() received {len(names)} names for {len(specs)} values",
            code="sweep_names_length_mismatch",
            remedy="pass exactly one name per replacement value or omit names",
            argument="names",
        )
    member_names = list(names) if names is not None else _default_member_names(len(specs))
    from ..user_funcs import trace as _trace

    baseline = _mint_pristine_baseline(
        model,
        x,
        member_names=member_names,
        include_baseline=include_baseline,
        # One extra frame (sweep -> _sweep_over_specs) vs the at/values path:
        # the absence disclosure must land on the USER's sweep() call site.
        _warn_stacklevel=4,
        **trace_kwargs,
    )
    traces = {}
    for member_name, member_spec in zip(member_names, specs, strict=True):
        traces[member_name] = _trace(
            model,
            x,
            intervene=member_spec,
            **trace_kwargs,
        )
    return _assemble_sweep_bundle(
        baseline,
        traces,
        params={
            "spec_digests": [spec.spec_digest for spec in specs],
            "include_baseline": include_baseline,
        },
    )


def _mint_pristine_baseline(
    model: nn.Module,
    x: Any,
    *,
    member_names: Sequence[str],
    include_baseline: bool,
    _warn_stacklevel: int = 3,
    **trace_kwargs: Any,
) -> Trace | None:
    """Capture the PRISTINE baseline member, or disclose its absence.

    ``include_baseline=False`` (the legacy cardinality) emits ONE coded
    construction-time warning: a sweep bundle without a pristine member has
    no comparison anchor, so ``most_changed`` raises
    ``BaselineUndeterminedError`` — the measured item-2 defect was that this
    surprise arrived only at comparison time. ``_warn_stacklevel`` keeps the
    disclosure attributed to the USER's ``sweep()`` call site on every door
    (the spec door adds one interior frame).
    """

    if not include_baseline:
        warnings.warn(
            TorchLensWarning(
                "sweep() built a bundle with NO pristine baseline member: "
                "every member is intervened, so baseline-anchored reads "
                "(most_changed, output_delta ...) will refuse. Pass "
                "include_baseline=True to mint one un-intervened 'baseline' "
                "capture first, or add one with bundle.add later.",
                code="sweep_baseline_absent",
            ),
            stacklevel=_warn_stacklevel,
        )
        return None
    if "baseline" in set(member_names):
        raise InvalidArgumentError(
            "sweep(include_baseline=True) reserves the member name 'baseline' "
            "for the pristine capture, but names= also carries 'baseline'",
            code="sweep_baseline_name_collision",
            remedy="rename the swept member or drop include_baseline",
            argument="names",
        )
    from ..user_funcs import trace as _trace

    return _trace(model, x, **trace_kwargs)


def _assemble_sweep_bundle(
    baseline: Trace | None,
    traces: dict[str, Trace],
    *,
    params: dict[str, Any],
) -> Bundle:
    """Assemble the sweep Bundle with lineage anchors and a chronology row."""

    members: dict[str, Trace] = {}
    if baseline is not None:
        members["baseline"] = baseline
    members.update(traces)
    bundle = Bundle(members, baseline="baseline" if baseline is not None else None)
    operation = bundle._record_bundle_operation("sweep", member_names=tuple(members), params=params)
    for member_name in traces:
        bundle._member_construction[member_name] = {
            "origin": "swept",
            "operation_id": operation.operation_id,
        }
    return bundle


def sweep_replace(value: Any) -> HelperSpec:
    """Typed builtin replacement helper carrying one swept value.

    Same replacement semantics the historical sweep closure applied (scalar
    broadcast fill, tensor ``expand_as`` where broadcastable), with the value
    declared in the helper's portable ``args`` so audit rows, saved specs,
    and provenance joins can distinguish swept members by content.

    Parameters
    ----------
    value:
        Scalar or tensor replacement value.

    Returns
    -------
    HelperSpec
        Built-in-compatible forward helper spec.
    """

    return HelperSpec(
        helper_name="sweep_replace",
        args=(value,),
        factory=lambda: _replacement_hook(value),
        batch_independent=True,
    )


def _coerce_sweep_site(param: str | BaseSelector | Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Normalize a sweep site target to a capture-time predicate.

    Parameters
    ----------
    param:
        String, selector, or predicate target.

    Returns
    -------
    Callable[[Any], bool]
        Predicate suitable for ``tl.when``.

    Raises
    ------
    TypeError
        If ``param`` is not a supported target.
    """

    if isinstance(param, str):
        return label(param) | func(param)
    if callable(param):
        return param
    raise ArgumentTypeError(
        f"sweep() site target has unsupported type {type(param).__name__}",
        code="sweep_site_type_invalid",
        remedy="pass a string, selector, or predicate callable as at",
        argument="at",
        received_type=type(param).__name__,
    )


def _replacement_hook(value: Any) -> Callable[..., torch.Tensor]:
    """Create a hook that replaces an out with one swept value.

    Parameters
    ----------
    value:
        Scalar or tensor replacement value.

    Returns
    -------
    Callable[..., torch.Tensor]
        Runtime hook callable.
    """

    def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Replace ``out`` with ``value`` aligned to ``out`` metadata."""

        del hook
        replacement = _value_to_tensor(value, out)
        if replacement.shape == torch.Size([]):
            return torch.zeros_like(out) + replacement
        if replacement.shape == out.shape:
            return replacement.clone()
        try:
            return replacement.expand_as(out).clone()
        except RuntimeError:
            return replacement

    return _hook


def _value_to_tensor(value: Any, out: torch.Tensor) -> torch.Tensor:
    """Convert one swept value to an out-compatible tensor.

    Parameters
    ----------
    value:
        Scalar or tensor replacement value.
    out:
        Captured output tensor whose dtype and device should be matched.

    Returns
    -------
    torch.Tensor
        Replacement tensor on the same device and dtype as ``out``.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device=out.device, dtype=out.dtype)
    return torch.as_tensor(value, device=out.device, dtype=out.dtype)


def _default_member_names(count: int) -> list[str]:
    """Return default Bundle member names for a sweep.

    Parameters
    ----------
    count:
        Number of swept values.

    Returns
    -------
    list[str]
        Stable member names.
    """

    return [f"{SWEEP_NAME}_{index}" for index in range(count)]


__all__ = ["SWEEP_NAME", "sweep"]
