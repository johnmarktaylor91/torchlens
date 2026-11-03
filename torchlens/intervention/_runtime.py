"""Runtime context ownership for TorchLens intervention execution."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from types import SimpleNamespace
from typing import Any

import torch

from .. import _state
from .._capture.arg_positions import _normalize_func_name
from .._data_classes._site_key import call_instance_id, render_site_key, site_axis
from .._data_classes._trace_missing_attrs import is_live_trace
from .._errors import InvalidArgumentError
from .._ir.intervention import FireResult
from .._record_type import record_type
from .._state import pause_logging
from .._utils.arg_handling import copy_arg_tree
from ..backends.torch._tl import copy_replacement_meta
from ._audit import build_fire_record
from ._hooks_support import gradient_inner
from ._module_boundary import (
    _apply_module_boundary_live_hooks as _apply_module_boundary_live_hooks,
    _peek_module_intervention_parent_labels as _peek_module_intervention_parent_labels,
    _peek_tensor_live_fire_results as _peek_tensor_live_fire_results,
    _record_module_intervention_parent_labels as _record_module_intervention_parent_labels,
    _record_tensor_live_fire_results as _record_tensor_live_fire_results,
    _replace_tensor_outputs as _replace_tensor_outputs,
)
from ._runtime_gradient import (
    _active_backward_pass_number,
    _current_injection_rule,
    _first_replaced_tuple_index,
    _hook_display_name,
    _hook_state,
    _HookReentrancyGuard,
    _live_run_ctx,
    _next_gradient_call_index,
    _record_backward_selector_fire,
    _trace_backward_pass_number,
)
from ._runtime_probes import (
    _apply_inplace_replacement_to_mutated_storage,
    _content_probe_mutated,
    _is_inplace_style_func_name,
    _tensor_content_probe,
    _tensor_version,
    _tuple_content_probes,
    _tuple_probes_mutated,
    _tuple_versions,
)
from ._runtime_sites import (
    _forward_site_label,
    _helper_routes_inputs,
    _hook_call_inputs_for_site,
    _input_splice_module_scope,
    _is_plain_module_selector,
    _op_output_slot,
    _selector_uses_only_provisional_fields,
    _site_name,
    _whole_call_site_view,
)
from ._types import FireRecord
from .errors import HookSignatureError, HookValueError
from .hooks import (
    HookContext,
    NormalizedHookEntry,
    _target_operands,
    live_backward_selector_matches,
    live_selector_matches_site,
    make_hook_context,
    split_site_branches,
)


@contextmanager
def active_intervention_context(
    *,
    intervention_spec: Any | None,
    hook_plan: Any | None,
) -> Iterator[None]:
    """Temporarily install a rerun/live intervention context in global state.

    Parameters
    ----------
    intervention_spec:
        Active intervention spec for the capture.
    hook_plan:
        Normalized hook entries consumed by live wrapper dispatch.

    Yields
    ------
    None
        Control while the context is installed.
    """

    # grind-r5 b7 R55 (sol HIGH, poss. REOPENED b2:C10): restores unwind
    # through a SPLICEABLE entry list (the rng-monitor _PATCH_STACKS
    # standard), not blind save/restore. The old unconditional restore
    # re-published a DEAD context when two overlapping contexts unwound out
    # of stack order (two-thread probe left thread A's spec/plan live after
    # BOTH finally blocks ran), poisoning every later operation with a stale
    # intervention. A non-top exit now splices its link out (the entry above
    # inherits its predecessor) and only the top exit writes the globals.
    # Cross-thread VISIBILITY of the process-global slot remains bounded by
    # the single-threaded-by-design capture contract (concurrent captures
    # refuse at admission; this hot-path manager takes no lock by doctrine).
    entry = _InterventionContextEntry(
        intervention_spec,
        hook_plan,
        _state._active_intervention_spec,
        _state._active_hook_plan,
    )
    _CONTEXT_ENTRIES.append(entry)
    _state._active_intervention_spec = intervention_spec
    _state._active_hook_plan = hook_plan
    try:
        yield
    finally:
        if _CONTEXT_ENTRIES and _CONTEXT_ENTRIES[-1] is entry:
            _CONTEXT_ENTRIES.pop()
            _state._active_intervention_spec = entry.previous_spec
            _state._active_hook_plan = entry.previous_plan
        else:
            for index in range(len(_CONTEXT_ENTRIES) - 1, -1, -1):
                if _CONTEXT_ENTRIES[index] is entry:
                    if index + 1 < len(_CONTEXT_ENTRIES):
                        above = _CONTEXT_ENTRIES[index + 1]
                        above.previous_spec = entry.previous_spec
                        above.previous_plan = entry.previous_plan
                    del _CONTEXT_ENTRIES[index]
                    break


class _InterventionContextEntry:
    """One live ``active_intervention_context`` publication, spliceable."""

    __slots__ = ("hook_plan", "previous_plan", "previous_spec", "spec")

    def __init__(self, spec: Any, hook_plan: Any, previous_spec: Any, previous_plan: Any) -> None:
        self.spec = spec
        self.hook_plan = hook_plan
        self.previous_spec = previous_spec
        self.previous_plan = previous_plan


_CONTEXT_ENTRIES: list[_InterventionContextEntry] = []


HOOK_REENTRANCY_GUARD = _HookReentrancyGuard()


def _execute_hook(
    hook_callable: Any,
    out: torch.Tensor,
    hook_context: HookContext,
    *,
    allow_shape_change: bool = False,
) -> torch.Tensor:
    """Run and validate one hook callable under ``pause_logging()``.

    Parameters
    ----------
    hook_callable:
        User or helper hook callable.
    out:
        Current out tensor at the hook site.
    hook_context:
        Metadata snapshot passed as the keyword-only ``ctx`` argument.
    allow_shape_change:
        Escape hatch allowing dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Replacement out tensor.

    Raises
    ------
    HookSignatureError
        If the callable cannot be invoked with the hook signature.
    HookValueError
        If the callable returns an incompatible value (``None`` keeps ``out``).
    """

    try:
        inspect.signature(hook_callable).bind(out, ctx=hook_context)
    except TypeError as exc:
        _refuse_gradient_edit_on_forward(hook_callable, hook_context)
        raise HookSignatureError(
            f"hook {hook_context.name!r} could not be called at "
            f"{_site_name(hook_context)} with signature (out, *, ctx)"
        ) from exc
    injection_trace = getattr(_state, "_active_trace", None)
    _injection_armed = injection_trace is not None and (
        getattr(injection_trace, "_tl_injection_state", None) or {}
    ).get("armed", False)
    if _injection_armed:
        # F01 log_injections stage 0-1: record the torch calls THIS hook
        # firing executes as anchored injected ops. The recorder installs
        # INSIDE the paused window, so the main capture journal and the
        # live site-key minter never see these calls -- injected ops
        # consume no global label counter and no site-key cohort ordinal
        # by construction (the misfire test pins both halves).
        from ._injection import InjectionAnchor, injection_recorder, next_firing_index

        site = hook_context.op
        site_get = site.get if hasattr(site, "get") else lambda key, d=None: getattr(site, key, d)
        rule_id = (getattr(injection_trace, "_tl_injection_state", None) or {}).get(
            "current_rule"
        ) or f"adhoc:{hook_context.name}"
        recorder = injection_recorder(
            injection_trace,
            InjectionAnchor(
                host_label=str(site_get("label") or site_get("raw_label") or "unknown"),
                host_site_key=site_get("site_key"),
                host_pass=int(site_get("pass_number") or 1),
                spec_rule_id=rule_id,
                firing_index=next_firing_index(injection_trace, rule_id),
            ),
        )
        with HOOK_REENTRANCY_GUARD, pause_logging(), recorder:
            result = hook_callable(out, ctx=hook_context)
    else:
        with HOOK_REENTRANCY_GUARD, pause_logging():
            result = hook_callable(out, ctx=hook_context)
    return validate_hook_output(
        result,
        out,
        hook_context=hook_context,
        allow_shape_change=allow_shape_change,
    )


#: Keyword arguments of the normalized gradient-tuple callback every backward
#: edit is mounted as (``hooks._adapt_tensor_backward_hook`` and the
#: ``grad_fn_handle`` helpers).
_GRADIENT_CALLBACK_PARAMETERS = frozenset(
    {"grad_output", "grad_fn_handle", "call_index", "run_ctx"}
)


def _refuse_gradient_edit_on_forward(hook_callable: Any, hook_context: HookContext) -> None:
    """Refuse a gradient edit reached by a forward-only engine (replay, push).

    An autograd-node edit is normalized to the gradient-tuple callback, whose
    ``grad_output``, ``grad_fn_handle``, ``call_index`` and ``run_ctx``
    arguments a forward value can never supply (this runs only after the
    forward bind failed). The replay engine recomputes forward values only,
    so such an edit has no gradient to edit there.

    Raises
    ------
    InvalidArgumentError
        ``replay_gradient_edit_unsupported`` when the callable is a gradient
        callback and the context is a forward one; otherwise returns.
    """

    if hook_context.signal != "forward":
        return
    try:
        parameters = inspect.signature(hook_callable).parameters
    except (TypeError, ValueError):
        return
    if not _GRADIENT_CALLBACK_PARAMETERS.issubset(parameters):
        return
    site = _site_name(hook_context)
    address = hook_context.op.get("address")
    module = str(address) if isinstance(address, str) and address else "<module>"
    verb = hook_context.name
    engine = hook_context.run_state.get("engine") or "forward"
    raise InvalidArgumentError(
        f"The gradient edit {verb!r} at site {site!r} reached the {engine} engine, "
        "which recomputes forward values only and never runs a backward pass, so "
        "the edit has no gradient to edit there. Nothing was changed. Remedy: "
        "capture the backward pass live and edit the gradient there: "
        f"trace = tl.trace(model, x, capture=tl.options.CaptureOptions("
        f"backward_ready=True), intervene=tl.when(tl.grad(tl.module({module!r})), edit)); "
        "trace.backward(loss)",
        code="replay_gradient_edit_unsupported",
        remedy=(
            "tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True), "
            f"intervene=tl.when(tl.grad(tl.module({module!r})), edit)) then "
            "trace.backward(loss)"
        ),
        argument="value_or_hook",
        verb=verb,
        site=site,
        engine=engine,
    )


def validate_hook_output(
    result: Any,
    out: torch.Tensor,
    *,
    hook_context: HookContext | None = None,
    allow_shape_change: bool = False,
) -> torch.Tensor:
    """Validate a hook return value against the input out metadata.

    Parameters
    ----------
    result:
        Hook return value.
    out:
        Original out tensor.
    hook_context:
        Optional context for error messages.
    allow_shape_change:
        If true, allow dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Validated replacement tensor.

    Raises
    ------
    HookValueError
        If the return value is not ``None`` and not a compatible tensor.
    """

    # R67: these are user-payload refusals. Say "intervention replacement" (the
    # vocabulary the user typed via intervene=/tl.when), name the helper/site,
    # and stamp structured fields so a partial record is diagnosable from the
    # public record alone.
    helper_name = getattr(hook_context, "name", None)
    site = _site_name(hook_context)

    def _payload_refusal(problem: str, expected: object, got: object) -> HookValueError:
        """Build one structured intervention-payload refusal."""

        helper_text = f" (helper {helper_name!r})" if helper_name else ""
        return HookValueError(
            f"intervention replacement{helper_text} {problem} at {site}; "
            f"expected {expected}, got {got}. Fix the replacement tensor passed "
            "to the intervene= clause.",
            code="intervention_replacement_invalid",
            site=site,
            slot="intervene",
            helper=helper_name,
            expected=str(expected),
            got=str(got),
        )

    if result is None:
        # A hook is ``fn(tensor, *, ctx) -> Tensor | None``: None leaves the
        # value unchanged (an observer, or one callable on a mixed site).
        return out
    if not isinstance(result, torch.Tensor):
        raise _payload_refusal("returned a non-tensor", "torch.Tensor", type(result).__name__)
    if not allow_shape_change:
        if result.dtype != out.dtype:
            raise _payload_refusal("has the wrong dtype", out.dtype, result.dtype)
        if result.device != out.device:
            raise _payload_refusal("is on the wrong device", out.device, result.device)
        if tuple(result.shape) != tuple(out.shape):
            raise _payload_refusal("has the wrong shape", tuple(out.shape), tuple(result.shape))
    result = _copy_reused_live_hook_result(out, result)
    _copy_tl_replacement_attrs(out, result)
    return result


def _copy_reused_live_hook_result(out: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Copy a hook result that is already a labeled live capture object.

    The commit path OVERWRITES the result's live label metadata in place
    (:func:`_copy_tl_replacement_attrs` -> ``copy_replacement_meta``). When a
    hook returns a REUSED tensor object -- another live op's current-session
    output, or one shared object fired at 2+ matched sites -- each fire stamps
    its own site label on the same object and the LAST fire steals it:
    consumers executing afterwards record the last site as parent, the earlier
    site's children vanish (byte-identical payloads make the misattributed
    graph validate clean), and chained edits at the orphaned site become
    silent no-ops. A labeled foreign object is therefore CLONED (autograd
    graph preserved) so the site label lands on a distinct object, mirroring
    the raw-module-hook rewire guard. The site's own pass-through result is
    exempt (the in-place carve-out).
    """

    if result is out:
        return result
    from torchlens.backends.torch import _tl as _tl_meta

    if _tl_meta.get(result) is None:
        return result
    # The copy is TorchLens-internal bookkeeping the user's program never
    # executed; it must not enter the captured graph as a spurious clone op
    # (this runs OUTSIDE _execute_hook's paused window).
    with pause_logging():
        return result.clone()


def _copy_tl_replacement_attrs(source: torch.Tensor, replacement: torch.Tensor) -> None:
    """Copy TorchLens tensor metadata from an original out to a replacement.

    Parameters
    ----------
    source:
        Original activation supplied to a user hook.
    replacement:
        Tensor returned by the hook.

    Returns
    -------
    None
        The replacement tensor is annotated in place when PyTorch permits
        dynamic tensor attributes.
    """

    if replacement is source:
        return
    with suppress(Exception):
        copy_replacement_meta(source, replacement)


def _apply_live_hooks(
    out: torch.Tensor,
    *,
    site: Any,
    container_path: tuple[Any, ...] = (),
    call_args: tuple[Any, ...] = (),
    call_kwargs: dict[str, Any] | None = None,
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, tuple[FireResult, ...]]:
    """Apply active live post-hooks to one capture-time output tensor.

    Parameters
    ----------
    out:
        Tensor returned by the decorated torch function after in-place safe-copy.
    site:
        Capture-time site proxy for selector matching and hook context.
    container_path:
        Stable path inside a multi-output container.
    call_args:
        Original positional call inputs for input-routed helpers.
    call_kwargs:
        Original keyword call inputs for input-routed helpers.
    call_input_snapshots:
        Optional pre-execution positional and keyword input snapshots for
        in-place ops whose post-hook input route would otherwise observe
        already-mutated live tensors.

    Returns
    -------
    tuple[torch.Tensor, tuple[FireResult, ...]]
        Original or hook-replaced tensor plus typed fire results for the
        corresponding capture event.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return out, ()
    _stamp_live_site_identity(site)

    current_out = out
    fire_results: list[FireResult] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.signal != "forward":
            continue
        if normalized_entry.metadata.get("timing", "post") != "post":
            continue
        module_scope = _input_splice_module_scope(normalized_entry.site_target, normalized_entry)
        if module_scope is not None and not getattr(site, "_tl_module_boundary", False):
            if _is_plain_module_selector(normalized_entry.site_target):
                continue
            raise HookValueError(
                "splice_module(input_route='in') with a module-scoped op selector is ambiguous. "
                "Use tl.module(...)/tl.in_module(...) alone for one module-call splice, or use "
                "an op selector without tl.in_module(...) for op-level splicing."
            )
        if not live_selector_matches_site(
            normalized_entry.site_target,
            _whole_call_site_view(site) if module_scope is not None else site,
        ):
            continue

        hook_args, hook_kwargs = _hook_call_inputs_for_site(
            normalized_entry,
            site=site,
            call_args=call_args,
            call_kwargs=call_kwargs,
            call_input_snapshots=call_input_snapshots,
        )
        run_state = _live_run_ctx()
        hook_context = make_hook_context(
            name=_hook_display_name(normalized_entry),
            timing="post",
            signal="forward",
            op=site,
            state=_hook_state(run_state, getattr(site, "_layer_label_raw", None), normalized_entry),
            run_context=run_state,
            args=hook_args or (current_out,),
            kwargs=hook_kwargs,
        )
        previous_notes = tuple(hook_context.run_state.get("ledger_notes", ()))
        pre_hook_shape = tuple(current_out.shape)
        pre_hook_dtype = str(current_out.dtype)
        version_before = _tensor_version(current_out)
        content_before = _tensor_content_probe(current_out)
        # F01 log_injections: a pre-lowered plan entry that carries its spec
        # rule id (the tl.module(...) lowering at the capture entry) anchors
        # this firing's injected ops to that PERSISTED id; entries without one
        # leave the door's own ``current_rule`` untouched (AUD-CODE 2.3b).
        entry_rule_id = normalized_entry.metadata.get("rule_id")
        stamped_state = (
            _armed_injection_state(_state._active_trace) if isinstance(entry_rule_id, str) else None
        )
        with _current_injection_rule(stamped_state, entry_rule_id):
            result = _execute_hook(
                normalized_entry.normalized_callable,
                current_out,
                hook_context,
                allow_shape_change=bool(normalized_entry.metadata.get("allow_shape_change", False)),
            )
        result, replaced = _apply_inplace_replacement_to_mutated_storage(
            result,
            current_out=current_out,
            site=site,
            call_args=call_args,
            call_kwargs=call_kwargs,
        )
        if (
            not replaced
            and result is current_out
            and (
                (version_before is not None and _tensor_version(current_out) != version_before)
                or _content_probe_mutated(content_before, _tensor_content_probe(current_out))
            )
        ):
            # An in-place-mutating HOOK (``out.mul_(0); return out``) is a
            # genuine value change: recording replaced=False minted ZERO
            # replacement evidence, so validation later failed forward replay
            # in a capture-bug shape on a genuine intervention, and an
            # unvalidated trace carried the false no-replacement claim. The
            # content probe closes the ``.data``-alias channel the version
            # counter cannot see (fresh counter on the alias impl).
            replaced = True
        record = _build_live_fire_record(
            normalized_entry,
            site=site,
            container_path=container_path,
            previous_notes=previous_notes,
            run_ctx=hook_context.run_state,
            replaced=replaced,
        )
        _append_active_spec_records([record])
        metadata = normalized_entry.metadata
        fire_results.append(
            FireResult(
                plan_id=str(
                    metadata.get(
                        "plan_id", metadata.get("hook_id", _hook_display_name(normalized_entry))
                    )
                ),
                site_label=site._layer_label_raw,
                fired_at_capture_index=int(getattr(site, "raw_index", 0) or 0),
                pre_hook_shape=pre_hook_shape,
                post_hook_shape=tuple(result.shape),
                pre_hook_dtype=pre_hook_dtype,
                post_hook_dtype=str(result.dtype),
                replaced=replaced,
                fire_record=record,
            )
        )
        if normalized_entry.metadata.get("zero_match_ledger") == "intervene_selector":
            trace = _state._active_trace
            if trace is not None:
                trace._tl_intervene_selector_fire_count = (
                    int(getattr(trace, "_tl_intervene_selector_fire_count", 0)) + 1
                )
        current_out = result
    return current_out, tuple(fire_results)


def _armed_injection_state(trace: Any) -> dict[str, Any] | None:
    """The trace's consolidated injection state when ``log_injections`` is armed."""

    from ._injection import peek_injection_state

    state = peek_injection_state(trace)
    if isinstance(state, dict) and state.get("armed", False):
        return state
    return None


def snapshot_call_inputs_for_inplace_intervention_site(
    *,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    trace: Any,
    func_call_id: int,
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Snapshot call inputs when an in-place op has a matching input-routed hook.

    Parameters
    ----------
    func_name:
        Decorated torch function name about to execute.
    args:
        Positional call inputs before execution.
    kwargs:
        Keyword call inputs before execution.
    trace:
        Active trace, used for predicate-intervention options.
    func_call_id:
        Function-call id allocated by the torch wrapper.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]] | None
        Recursive tensor clones of the call inputs when a post-execution hook
        may read an in-place op's semantic inputs, otherwise ``None``.
    """

    if not _is_inplace_style_func_name(func_name):
        return None
    if not _has_matching_input_routed_intervention(
        func_name=func_name,
        trace=trace,
        func_call_id=func_call_id,
    ):
        return None
    with pause_logging():
        return tuple(copy_arg_tree(arg) for arg in args), {
            key: copy_arg_tree(value) for key, value in kwargs.items()
        }


def _has_matching_input_routed_intervention(
    *,
    func_name: str,
    trace: Any,
    func_call_id: int,
) -> bool:
    """Return whether the current in-place op needs pre-execution input snapshots.

    Parameters
    ----------
    func_name:
        Decorated torch function name.
    trace:
        Active trace.
    func_call_id:
        Function-call id allocated by the torch wrapper.

    Returns
    -------
    bool
        Whether any active post-hook or ``tl.when`` predicate hook routes the
        current op's inputs through the post-hook execution path.
    """

    site = _provisional_inplace_site(func_name, trace, func_call_id)
    hook_plan = _state._active_hook_plan
    if hook_plan:
        for entry in hook_plan:
            normalized_entry = _coerce_hook_entry(entry)
            if _entry_needs_input_snapshot(normalized_entry, site):
                return True

    options = getattr(trace, "_predicate_save_options", None)
    intervene = None if options is None else getattr(options, "intervene", None)
    if intervene is None:
        return False
    selector = getattr(intervene, "selector", None)
    decision = getattr(intervene, "decision", None)
    hook = None if decision is None else getattr(decision, "hook", None)
    if selector is None or hook is None:
        return False
    if not _helper_routes_inputs(hook):
        return False
    activation = [
        target for universe, target in split_site_branches(selector) if universe == "activation"
    ]
    if not activation:
        return False
    return _selector_may_match_provisional_site(activation[0], site)


def _entry_needs_input_snapshot(entry: NormalizedHookEntry, site: Any) -> bool:
    """Return whether an active hook entry needs an in-place input snapshot.

    Parameters
    ----------
    entry:
        Normalized hook entry.
    site:
        Provisional current-op site.

    Returns
    -------
    bool
        Whether the entry is a matching post-forward input-routed hook.
    """

    if entry.signal != "forward":
        return False
    if entry.metadata.get("timing", "post") != "post":
        return False
    if not _helper_routes_inputs(entry.helper_spec):
        return False
    return _selector_may_match_provisional_site(entry.site_target, site)


def _selector_may_match_provisional_site(selector: Any, site: Any) -> bool:
    """Return a conservative selector result for a pre-execution site.

    The provisional site intentionally lacks output-dependent metadata.  A
    failed predicate or a selector that depends on unavailable raw-label data
    therefore means "may match", so input snapshots are never skipped on a
    false negative.

    Parameters
    ----------
    selector:
        Selector-like target to evaluate.
    site:
        Provisional pre-execution live site.

    Returns
    -------
    bool
        Whether the selector may match once the operation has executed.
    """

    try:
        matched = live_selector_matches_site(selector, site)
    except Exception:
        return True
    if matched:
        return True
    return not _selector_uses_only_provisional_fields(selector)


def _provisional_inplace_site(func_name: str, trace: Any, func_call_id: int) -> Any:
    """Build a selector-matchable site for a not-yet-executed in-place op.

    Parameters
    ----------
    func_name:
        Decorated torch function name.
    trace:
        Active trace.
    func_call_id:
        Function-call id allocated by the torch wrapper.

    Returns
    -------
    Any
        Minimal live-site proxy for pre-execution selector matching.
    """

    from ..backends.torch._ops_capture_records import _snapshot_exhaustive_module_stack

    layer_type = _normalize_func_name(func_name)
    modules = tuple(_snapshot_exhaustive_module_stack(trace))
    raw_index = trace._raw_graph_ws.layer_counter + 1
    type_index = trace._raw_graph_ws.raw_layer_type_counter.get(layer_type, 0) + 1
    raw_label = f"{layer_type}_{type_index}_{raw_index}_raw"
    return SimpleNamespace(
        layer_label=raw_label,
        _layer_label_raw=raw_label,
        _label_raw=raw_label,
        raw_index=raw_index,
        layer_type=layer_type,
        func_name=func_name,
        func_call_id=func_call_id,
        container_path=(),
        module=modules[-1] if modules else None,
        modules=modules,
        output_of_module_calls=(),
        output_of_modules=(),
        _tl_module_boundary=False,
        is_transform=False,
        transform_kind=None,
        transform_chain=(),
        transform_config={},
        is_inplace=True,
        _tl_input_snapshot=False,
        call_index=1,
        lookup_keys=[],
    )


def _apply_live_backward_hooks(
    grad_input: tuple[torch.Tensor | None, ...] | None,
    grad_output: tuple[torch.Tensor | None, ...] | None,
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]:
    """Apply active grad_fn_handle post-hook helpers.

    Parameters
    ----------
    grad_input:
        Current autograd grad_input tuple.
    grad_output:
        Autograd grad_output tuple.
    grad_fn_handle:
        GradFn site for selector matching.
    call_index:
        One-based grad_fn_handle call index.

    Returns
    -------
    tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]
        Mutated grad_input tuple, or None when no helper mutates it, plus
        immutable fire records for every matching helper.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan or grad_input is None:
        return None, ()

    current = grad_input
    mutated = False
    fire_records: list[FireRecord] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.universe != "node":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target,
            grad_fn_handle,
            call_index,
            grad_input=current,
            grad_output=grad_output,
        ):
            continue
        previous = current
        versions_before = _tuple_versions(current)
        probes_before = _tuple_content_probes(current)
        with HOOK_REENTRANCY_GUARD, pause_logging():
            result = normalized_entry.normalized_callable(
                current,
                grad_output=grad_output,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                run_ctx=_live_run_ctx(),
            )
        if result is not None:
            current = _validate_grad_tuple(result, current, grad_fn_handle=grad_fn_handle)
            mutated = True
        fire_records.append(
            _build_live_backward_fire_record(
                normalized_entry,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                grad_kind="grad_input",
                inplace_mutated=_tuple_versions(previous) != versions_before
                or _tuple_probes_mutated(probes_before, _tuple_content_probes(previous)),
                timing="post",
                previous=previous,
                current=current,
            )
        )
        _record_backward_selector_fire(normalized_entry)
    _append_active_spec_records(fire_records)
    return (current if mutated else None), tuple(fire_records)


def _apply_live_backward_prehooks(
    grad_input: tuple[torch.Tensor | None, ...],
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]:
    """Apply active AccumulateGrad prehook helpers.

    Parameters
    ----------
    grad_input:
        Current autograd prehook grad_input tuple.
    grad_fn_handle:
        GradFn site for selector matching.
    call_index:
        One-based grad_fn_handle call index expected for the matching post-hook.

    Returns
    -------
    tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]
        Mutated grad_input tuple, or None when no helper mutates it, plus
        immutable fire records for every matching helper.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return None, ()

    current = grad_input
    mutated = False
    fire_records: list[FireRecord] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.universe != "node":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target,
            grad_fn_handle,
            call_index,
            grad_input=current,
            grad_output=None,
        ):
            continue
        previous = current
        versions_before = _tuple_versions(current)
        probes_before = _tuple_content_probes(current)
        with HOOK_REENTRANCY_GUARD, pause_logging():
            result = normalized_entry.normalized_callable(
                current,
                grad_output=None,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                run_ctx=_live_run_ctx(),
            )
        if result is not None:
            current = _validate_grad_tuple(result, current, grad_fn_handle=grad_fn_handle)
            mutated = True
        fire_records.append(
            _build_live_backward_fire_record(
                normalized_entry,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                grad_kind="grad_input",
                inplace_mutated=_tuple_versions(previous) != versions_before
                or _tuple_probes_mutated(probes_before, _tuple_content_probes(previous)),
                timing="pre",
                previous=previous,
                current=current,
            )
        )
        _record_backward_selector_fire(normalized_entry)
    _append_active_spec_records(fire_records)
    return (current if mutated else None), tuple(fire_records)


def apply_live_gradient_hooks(
    grad: torch.Tensor,
    *,
    trace: Any,
    op: Any,
) -> tuple[torch.Tensor, tuple[FireRecord, ...]]:
    """Apply the gradient-universe edits addressed to one op's output gradient.

    Called from the one TorchLens boundary that observes the gradient of a
    forward op's output (the tensor hook or output-slot pre-hook that saves
    ``Op.grad``), BEFORE the save, so the saved ``.grad`` is the edit chain's
    final value (SPEC 3.7). The returned tensor is what autograd passes into
    the op's backward function, so an edit here changes every gradient
    upstream of the op (SPEC 1.1: output-gradient semantics, never the
    node's ``grad_input``).

    Parameters
    ----------
    grad:
        Gradient of the op's output arriving at the boundary.
    trace:
        Trace whose published backward hook plan applies.
    op:
        Final forward op record whose output this gradient belongs to.

    Returns
    -------
    tuple[torch.Tensor, tuple[FireRecord, ...]]
        Edited gradient (``grad`` itself when nothing fired or changed it)
        plus one fire record per matching entry, in declared order.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan or _state.active_capture()[0] is not trace:
        return grad, ()
    pass_index = _trace_backward_pass_number(trace)
    current = grad
    fire_records: list[FireRecord] = []
    candidates = _gradient_site_ops(trace, op)
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.universe != "gradient":
            continue
        matched = next(
            (
                candidate
                for candidate in candidates
                if _gradient_entry_matches(trace, normalized_entry, candidate, pass_index)
            ),
            None,
        )
        if matched is None:
            continue
        current, record = _fire_gradient_entry(
            normalized_entry, current, trace=trace, op=matched, pass_index=pass_index
        )
        fire_records.append(record)
        _record_backward_selector_fire(normalized_entry)
    _append_active_spec_records(fire_records)
    return current, tuple(fire_records)


def _fire_gradient_entry(
    entry: NormalizedHookEntry,
    grad: torch.Tensor,
    *,
    trace: Any,
    op: Any,
    pass_index: int | None,
) -> tuple[torch.Tensor, FireRecord]:
    """Run one gradient-universe hook on one output gradient and record it.

    Parameters
    ----------
    entry:
        Matching gradient-universe hook entry.
    grad:
        Current gradient (after the earlier entries of the chain).
    trace:
        Trace being backwarded.
    op:
        Forward op whose output gradient this is.
    pass_index:
        Active one-based backward pass index.

    Returns
    -------
    tuple[torch.Tensor, FireRecord]
        Gradient after this hook and its fire record.
    """

    site_label = _forward_site_label(op)
    tuple_index = _op_output_slot(op)
    run_state = _live_run_ctx()
    call_index = _next_gradient_call_index(run_state, entry, site_label)
    context = make_hook_context(
        name=_hook_display_name(entry),
        timing="post",
        signal="backward",
        op=op,
        state=_hook_state(run_state, site_label, entry),
        run_context=run_state,
        args=(grad,),
        kwargs={
            "grad_kind": "grad_output",
            "backward_pass_number": pass_index,
            "tuple_index": tuple_index,
            "call_index": call_index,
        },
    )
    result, replaced = _run_gradient_hook(entry, grad, context)
    return result, _gradient_fire_record(entry, site_label, context, replaced=replaced)


def _run_gradient_hook(
    entry: NormalizedHookEntry, grad: torch.Tensor, context: HookContext
) -> tuple[torch.Tensor, bool]:
    """Call one gradient-universe hook and report whether it changed the gradient.

    Parameters
    ----------
    entry:
        Matching gradient-universe hook entry.
    grad:
        Current gradient (after the earlier entries of the chain).
    context:
        Hook context of this fire.

    Returns
    -------
    tuple[torch.Tensor, bool]
        Gradient after the hook, and whether the hook replaced the gradient or
        mutated it in place.
    """

    version_before = _tensor_version(grad)
    probe_before = _tensor_content_probe(grad)
    with HOOK_REENTRANCY_GUARD, pause_logging():
        result = entry.normalized_callable(grad, ctx=context)
    if result is None:
        result = grad
    else:
        result = validate_hook_output(
            result,
            grad,
            hook_context=context,
            allow_shape_change=False,
        )
    replaced = (
        result is not grad
        or (version_before is not None and _tensor_version(grad) != version_before)
        or _content_probe_mutated(probe_before, _tensor_content_probe(grad))
    )
    return result, replaced


def _gradient_fire_record(
    entry: NormalizedHookEntry, site_label: str, context: HookContext, *, replaced: bool
) -> FireRecord:
    """Return the fire record of one gradient-universe hook fire.

    Parameters
    ----------
    entry:
        Gradient-universe hook entry that fired.
    site_label:
        Forward site label of the op whose output gradient fired.
    context:
        Hook context of the fire; its kwargs carry the pass, call and slot.
    replaced:
        Whether the hook replaced or mutated the gradient.

    Returns
    -------
    FireRecord
        Live backward fire record.
    """

    return build_fire_record(
        target_label=site_label,
        call_label=site_label,
        func_call_id=None,
        container_path=(),
        engine="live",
        edit=entry.helper_spec,
        site_label=site_label,
        timing="post",
        signal="backward",
        edit_name=_hook_display_name(entry),
        backward_pass_number=context.kwargs["backward_pass_number"],
        call_index=context.kwargs["call_index"],
        grad_kind="grad_output",
        tuple_index=context.kwargs["tuple_index"],
        replaced=replaced,
    )


def _gradient_site_ops(trace: Any, op: Any) -> list[Any]:
    """Return the forward records whose output gradient is this op's output gradient.

    An output layer is an identity wrapper of its parent: its gradient is the
    parent's output gradient (the read side stores it on both), so
    ``tl.grad("output_1")`` addresses the same boundary.

    Parameters
    ----------
    trace:
        Trace being backwarded.
    op:
        Final forward op whose output gradient arrived.

    Returns
    -------
    list[Any]
        ``op`` followed by its output-layer children.
    """

    ops = [op]
    if not getattr(op, "is_output_parent", False):
        return ops
    layer_dict = trace._layer_dict_all_keys if is_live_trace(trace) else {}
    for child_label in getattr(op, "children", ()) or ():
        child = layer_dict.get(child_label)
        if child is not None and getattr(child, "is_output", False):
            ops.append(child)
    return ops


def _gradient_entry_matches(
    trace: Any,
    entry: NormalizedHookEntry,
    op: Any,
    pass_index: int | None,
) -> bool:
    """Return whether a gradient-universe entry addresses this op in this pass.

    The inner selector of ``tl.grad(inner)`` resolves in the activation
    universe against the finished forward op (the same lifecycle a post-hoc
    activation door uses); ``tl.in_backward_pass(n)`` joined to it filters the
    pass. Per-op answers of the selector part are cached per trace.

    Parameters
    ----------
    trace:
        Trace being backwarded.
    entry:
        Gradient-universe hook entry.
    op:
        Final forward op record.
    pass_index:
        Active one-based backward pass index.

    Returns
    -------
    bool
        Whether the entry fires on this op's output gradient.
    """

    return bool(_gradient_target_matches(trace, entry.site_target, op, pass_index))


def _gradient_target_matches(trace: Any, target: Any, op: Any, pass_index: int | None) -> bool:
    """Evaluate one gradient-branch target tree against an op and a pass.

    Parameters
    ----------
    trace:
        Trace being backwarded (owns the per-op cache).
    target:
        ``tl.grad(inner)``, a pass qualifier, or an ``&``/``|`` of these.
    op:
        Final forward op record.
    pass_index:
        Active one-based backward pass index.

    Returns
    -------
    bool
        Whether the target matches.
    """

    inner = gradient_inner(target)
    if inner is not None:
        return _inner_selector_matches_op(inner, op)
    # Live selectors and the stored TargetSpec a sticky rule is re-normalised
    # from read the same way (a recipe stores ``tl.in_backward_pass(2)`` as
    # kind ``"backward_pass"`` with the pass index as its value).
    if getattr(target, "selector_kind", None) == "backward_pass":
        return pass_index == getattr(target, "pass_index", target.selector_value)
    operator, operands = _target_operands(target)
    if operator == "not":
        return not _gradient_target_matches(trace, operands[0], op, pass_index)
    if operator is not None:
        results = (_gradient_target_matches(trace, child, op, pass_index) for child in operands)
        return all(results) if operator == "and" else any(results)
    return False


def _inner_selector_matches_op(inner: Any, op: Any) -> bool:
    """Return whether a forward selector matches one finished op (cached per run).

    Parameters
    ----------
    inner:
        Forward-universe selector-like (a label string resolves as an exact
        label, like every live hook target).
    op:
        Final forward op record.

    Returns
    -------
    bool
        Whether the selector matches the op.
    """

    from .._ir.selector_eval import normalize_selector_like
    from .._selection.resolver import _resolve_unchecked

    # Session-only cache in the run state (last_run is never persisted).
    cache: dict[tuple[str, str], bool] = _live_run_ctx().setdefault(
        "_tl_gradient_site_match_cache", {}
    )
    op_label = str(getattr(op, "label", ""))
    # Keyed by the canonical selector repr, never id(): a recycled id of a
    # dead selector must not inherit another selector's answers.
    key = (repr(inner), op_label)
    cached = cache.get(key)
    if cached is not None:
        return cached
    selector = normalize_selector_like(inner, lifecycle="live")
    matched = bool(_resolve_unchecked((op,), selector, strict=False))
    cache[key] = matched
    return matched


def _validate_grad_tuple(
    result: Any,
    reference: tuple[torch.Tensor | None, ...],
    *,
    grad_fn_handle: Any,
) -> tuple[torch.Tensor | None, ...]:
    """Validate a grad_fn_handle helper return tuple.

    Parameters
    ----------
    result:
        Helper return value.
    reference:
        Original grad tuple.
    grad_fn_handle:
        GradFn used in diagnostics.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        Validated tuple.
    """

    if not isinstance(result, tuple):
        raise HookValueError(
            f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
            f"{type(result).__name__}; expected tuple or None"
        )
    if len(result) != len(reference):
        raise HookValueError(
            f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
            f"{len(result)} gradients; expected {len(reference)}"
        )
    for index, (candidate, expected) in enumerate(zip(result, reference, strict=True)):
        if expected is None:
            if candidate is not None:
                raise HookValueError(
                    "backward helper at "
                    f"{getattr(grad_fn_handle, 'label', '<unknown>')} returned a tensor for "
                    f"slot {index}; expected None"
                )
            continue
        if candidate is None:
            continue
        if not isinstance(candidate, torch.Tensor):
            raise HookValueError(
                f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
                f"{type(candidate).__name__} for slot {index}; expected torch.Tensor or None"
            )
        if candidate.dtype != expected.dtype:
            raise HookValueError(
                f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
                f"dtype {candidate.dtype} for slot {index}; expected {expected.dtype}"
            )
        if candidate.device != expected.device:
            raise HookValueError(
                f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
                f"device {candidate.device} for slot {index}; expected {expected.device}"
            )
        if tuple(candidate.shape) != tuple(expected.shape):
            raise HookValueError(
                f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
                f"shape {tuple(candidate.shape)} for slot {index}; expected "
                f"{tuple(expected.shape)}"
            )
    return result


def _coerce_hook_entry(entry: Any) -> NormalizedHookEntry:
    """Return a normalized hook entry or raise a deterministic type error.

    Parameters
    ----------
    entry:
        Hook-plan entry from runtime state.

    Returns
    -------
    NormalizedHookEntry
        Valid normalized entry.
    """

    if isinstance(entry, NormalizedHookEntry):
        return entry
    raise HookValueError("live hook execution requires a normalized hook plan entry")


class _LiveSiteMinter:
    """Incremental ``site_key_v1`` minter for the LIVE door (F02 D6 coordinate).

    The replay door reads the postprocess-minted ``Op.site_key``; the live
    door fires BEFORE its op is journaled, so it PEEKS the key the postprocess
    minter will assign: consume every op already journaled on the capture's
    ``capture_events.op_events`` in execution order under the same cohort rule as
    :class:`~torchlens._data_classes._site_key.SiteKeyMinter` (pass-qualified
    innermost call instance, pass-free module site, layer type, output
    slot), then report ``seen + 1`` for the firing op without consuming it
    (it is consumed when its own record lands). Both doors therefore derive
    ONE logical firing coordinate, and a stochastic edit draws the same
    donor at capture time and on replay (fable51 audit AUD-CODE 3.8: the raw
    ordinal label shifted whenever an unrelated upstream op was added).

    DISCLOSED RESIDUAL: postprocess prunes orphan ops (they consume no
    ordinals, SF-63); orphan-ness is not knowable live, so a to-be-pruned op
    that PRECEDES the firing op inside the same cohort shifts the live
    ordinal by one relative to replay. The pass component derives from the
    innermost module call's pass (functional, module-free recurrence is
    grouped only at postprocess and stays a residual).
    """

    __slots__ = ("_seen", "_consumed", "_first_raw_index_by_call")

    def __init__(self) -> None:
        self._seen: dict[tuple[str, tuple[str, ...], str, int | None], int] = {}
        self._consumed = 0
        self._first_raw_index_by_call: dict[int, int] = {}

    @staticmethod
    def _cohort(
        modules: Any, layer_type: str, slot: int | None
    ) -> tuple[str, tuple[str, ...], str, int | None]:
        """Return the SiteKeyMinter cohort tuple for one op position."""

        return (call_instance_id(modules), site_axis(modules), str(layer_type), slot)

    @staticmethod
    def _journal_output_slot(record: Any) -> int | None:
        """Read one journaled op's ``multi_output_index`` (flat event or decomposed record)."""

        output = getattr(record, "output", None)
        slot = getattr(output, "multi_output_index", None) if output is not None else None
        if slot is None:
            slot = getattr(record, "multi_output_index", None)
        return slot if isinstance(slot, int) else None

    def consume(self, trace: Any) -> None:
        """Advance over every op journaled since the last call, in execution order."""

        events = getattr(getattr(trace, "capture_events", None), "op_events", None)
        if events is None:
            try:
                workspace = trace._raw_graph_ws
            except AttributeError:
                workspace = None
            records = getattr(workspace, "raw_layer_dict", None) or {}
            events = [records[label] for label in list(records)]
        journal = list(events)
        for record in journal[self._consumed :]:
            modules = getattr(record, "modules", None) or ()
            cohort = self._cohort(
                modules, record_type(record, ""), self._journal_output_slot(record)
            )
            self._seen[cohort] = self._seen.get(cohort, 0) + 1
        self._consumed = len(journal)

    def output_slot(self, site: Any) -> int | None:
        """Derive the firing op's ``multi_output_index`` from its live facts.

        A bare tensor output (empty container path) has no slot; inside a
        container the slot is the op's position among the call's outputs,
        which the wrapper reserves as consecutive raw indexes.
        """

        if not getattr(site, "container_path", ()):
            return None
        raw_index = getattr(site, "raw_index", None)
        call_id = getattr(site, "func_call_id", None)
        if not isinstance(raw_index, int) or call_id is None:
            return None
        first = self._first_raw_index_by_call.setdefault(call_id, raw_index)
        return raw_index - first

    def peek(self, modules: Any, layer_type: str, slot: int | None) -> str:
        """Return the key the postprocess minter will assign to the firing op."""

        cohort = self._cohort(modules, layer_type, slot)
        ordinal = self._seen.get(cohort, 0) + 1
        return render_site_key(site_axis(modules), str(layer_type), slot, ordinal)


def _stamp_live_site_identity(site: Any) -> None:
    """Stamp ``site_key`` + module-call ``pass_index`` on a live op site proxy.

    Exhaustive-mode op sites only: module-boundary splice sites carry no op
    record, and predicate-mode (``tl.record``) captures have no replay door
    to agree with, so both keep the label fallback of the derived-seed
    coordinate.
    Idempotent (a proxy stamped once is left alone).
    """

    if getattr(site, "site_key", None) or getattr(site, "_tl_module_boundary", False):
        return
    trace = _state._active_trace
    if trace is None or trace._options_mode != "exhaustive":
        return
    # The minter rides the session-only run context (last_run, FieldPolicy
    # DROP) rather than its own Trace attr: no schema/ownership row, and the
    # portable-state gate never sees it.
    run_ctx = _live_run_ctx()
    minter = run_ctx.get("site_minter")
    if minter is None:
        minter = _LiveSiteMinter()
        run_ctx["site_minter"] = minter
    minter.consume(trace)
    modules = list(getattr(site, "modules", None) or ())
    if not modules:
        from ..backends.torch._ops_capture_records import _snapshot_exhaustive_module_stack

        modules = _snapshot_exhaustive_module_stack(trace)
    site.site_key = minter.peek(modules, record_type(site, "") or "", minter.output_slot(site))
    if getattr(site, "pass_number", None) is None:
        innermost = modules[-1] if modules else None
        pass_number = innermost[1] if isinstance(innermost, tuple) and len(innermost) == 2 else 1
        site.pass_number = int(pass_number) if isinstance(pass_number, int) else 1


def _build_live_fire_record(
    entry: NormalizedHookEntry,
    *,
    site: Any,
    container_path: tuple[Any, ...],
    previous_notes: tuple[Any, ...],
    run_ctx: dict[str, Any],
    replaced: bool,
) -> FireRecord:
    """Build a record for one live hook fire.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    site:
        Capture-time site proxy.
    container_path:
        Output path for the hooked tensor.
    previous_notes:
        Operation-history notes present before hook execution.
    run_ctx:
        Shared hook run context after execution.
    replaced:
        Whether the hook returned a replacement tensor object.

    Returns
    -------
    FireRecord
        Immutable fire record appended to the eventual layer pass.
    """

    return build_fire_record(
        target_label=site._layer_label_raw,
        call_label=site._layer_label_raw,
        func_call_id=site.func_call_id,
        container_path=container_path,
        engine="live",
        edit=entry.helper_spec,
        site_label=site._layer_label_raw,
        timing="post",
        signal="forward",
        edit_name=_hook_display_name(entry),
        run_ctx=run_ctx,
        previous_notes=previous_notes,
        replaced=replaced,
    )


def _build_live_backward_fire_record(
    entry: NormalizedHookEntry,
    *,
    grad_fn_handle: Any,
    call_index: int,
    grad_kind: str,
    timing: str,
    previous: tuple[torch.Tensor | None, ...],
    current: tuple[torch.Tensor | None, ...],
    inplace_mutated: bool = False,
) -> FireRecord:
    """Build an audit record for one live backward hook fire.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    grad_fn_handle:
        Backward site receiving the hook.
    call_index:
        One-based callback index for this grad_fn.
    grad_kind:
        Gradient tuple kind mutated by the hook.
    timing:
        Execution timing for the callback site.
    previous:
        Tuple before this helper ran.
    current:
        Tuple after this helper ran.
    inplace_mutated:
        Whether the hook mutated a grad slot IN PLACE (version-counter
        evidence): a hook editing a tensor and returning ``None`` is a
        genuine value change and must not record ``replaced=False``.

    Returns
    -------
    FireRecord
        Immutable backward fire record.
    """

    tuple_index = _first_replaced_tuple_index(previous, current)
    label = str(getattr(grad_fn_handle, "label", ""))
    pass_index = _active_backward_pass_number(grad_fn_handle)
    return build_fire_record(
        target_label=label,
        call_label=f"{label}:{call_index}" if label else str(call_index),
        func_call_id=None,
        container_path=(),
        engine="live",
        edit=entry.helper_spec,
        site_label=label,
        timing=timing,  # type: ignore[arg-type]
        signal="backward",
        edit_name=_hook_display_name(entry),
        backward_pass_number=pass_index,
        call_index=call_index,
        grad_kind=grad_kind,  # type: ignore[arg-type]
        tuple_index=tuple_index,
        replaced=tuple_index is not None or current is not previous or inplace_mutated,
    )


def _append_active_spec_records(records: list[FireRecord]) -> None:
    """Append live fire records to the active intervention spec ledger.

    Parameters
    ----------
    records:
        Fire records produced by the current live callback.

    Returns
    -------
    None
        Mutates the active intervention spec when one is installed.
    """

    if not records:
        return
    # Attested coupling (lane F42): this is the ONE funnel every live
    # FireRecord passes through, so the armed coupling session of an
    # intervened episode capture attributes fires here -- independent of
    # whether a spec ledger is installed.
    from .._capture._episode_coupling import active_coupling_session

    coupling = active_coupling_session()
    if coupling is not None:
        coupling.note_fire_records(records)
    spec = _state._active_intervention_spec
    if spec is not None and hasattr(spec, "records"):
        spec.records.extend(records)


def do(trace: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a one-shot intervention operation to a model log.

    Parameters
    ----------
    trace:
        Trace-like object that will eventually receive the operation.
    *args:
        Positional arguments forwarded to ``log.do``.
    **kwargs:
        Keyword arguments forwarded to ``log.do``.

    Returns
    -------
    Any
        Operation result from ``log.do``.
    """

    return trace.do(*args, **kwargs)


__all__ = [
    "HOOK_REENTRANCY_GUARD",
    "_HookReentrancyGuard",
    "_apply_live_hooks",
    "_apply_live_backward_hooks",
    "_apply_live_backward_prehooks",
    "_execute_hook",
    "apply_live_gradient_hooks",
    "active_intervention_context",
    "do",
    "validate_hook_output",
]
