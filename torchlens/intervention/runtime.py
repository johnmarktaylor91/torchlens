"""Runtime context ownership for TorchLens intervention execution."""

from __future__ import annotations

import inspect
import time
import warnings
import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from types import SimpleNamespace
from typing import Any

import torch

from .. import _state
from .._state import pause_logging
from ..backends.torch._tl import clear_tensor_label, copy_replacement_meta, get_tensor_label
from ..capture.arg_positions import _normalize_func_name
from ..ir.intervention import FireResult
from ..utils.arg_handling import copy_arg_tree
from .errors import HookSignatureError, HookValueError
from .hooks import (
    HookContext,
    NormalizedHookEntry,
    live_backward_selector_matches,
    live_selector_matches_site,
    make_hook_context,
    make_live_site_proxy,
)
from .types import FireRecord


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


class _HookReentrancyGuard:
    """Track recursive TorchLens hook execution depth in global state."""

    def __init__(self) -> None:
        """Initialise an inactive re-entrancy guard."""

        self.depth = 0
        self.active_log_id: int | None = None

    @property
    def active(self) -> bool:
        """Return whether hook execution is currently active.

        Returns
        -------
        bool
            Whether at least one hook is on the call stack.
        """

        return self.depth > 0

    def __enter__(self) -> _HookReentrancyGuard:
        """Enter hook execution.

        Returns
        -------
        _HookReentrancyGuard
            This guard.
        """

        self.depth += 1
        _state._hook_reentrancy_depth = self.depth
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Leave hook execution.

        Parameters
        ----------
        exc_type:
            Exception type, if any.
        exc:
            Exception value, if any.
        traceback:
            Exception traceback, if any.
        """

        self.depth = max(0, self.depth - 1)
        _state._hook_reentrancy_depth = self.depth
        if self.depth == 0:
            self.active_log_id = None


HOOK_REENTRANCY_GUARD = _HookReentrancyGuard()


def _execute_hook(
    hook_callable: Any,
    out: torch.Tensor,
    hook_context: HookContext,
    *,
    force_shape_change: bool = False,
) -> torch.Tensor:
    """Run and validate one hook callable under ``pause_logging()``.

    Parameters
    ----------
    hook_callable:
        User or helper hook callable.
    out:
        Current out tensor at the hook site.
    hook_context:
        Metadata snapshot passed as the keyword-only ``hook`` argument.
    force_shape_change:
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
        If the callable returns ``None`` or an incompatible value.
    """

    try:
        inspect.signature(hook_callable).bind(out, hook=hook_context)
    except TypeError as exc:
        raise HookSignatureError(
            f"hook {hook_context.name!r} could not be called at "
            f"{_site_name(hook_context)} with signature (out, *, hook)"
        ) from exc
    with HOOK_REENTRANCY_GUARD, pause_logging():
        result = hook_callable(out, hook=hook_context)
    return validate_hook_output(
        result,
        out,
        hook_context=hook_context,
        force_shape_change=force_shape_change,
    )


def validate_hook_output(
    result: Any,
    out: torch.Tensor,
    *,
    hook_context: HookContext | None = None,
    force_shape_change: bool = False,
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
    force_shape_change:
        If true, allow dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Validated replacement tensor.

    Raises
    ------
    HookValueError
        If the return value is invalid.
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
        raise _payload_refusal("returned None", "torch.Tensor", None)
    if not isinstance(result, torch.Tensor):
        raise _payload_refusal("returned a non-tensor", "torch.Tensor", type(result).__name__)
    if force_shape_change:
        result = _copy_reused_live_hook_result(out, result)
        _copy_tl_replacement_attrs(out, result)
        return result
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

    current_out = out
    fire_results: list[FireResult] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "forward":
            continue
        if normalized_entry.metadata.get("timing", "post") != "post":
            continue
        module_scope = _input_splice_module_scope(normalized_entry.site_target, normalized_entry)
        if module_scope is not None and not getattr(site, "_tl_module_boundary", False):
            if _is_plain_module_selector(normalized_entry.site_target):
                continue
            raise HookValueError(
                "splice_module(input='in') with a module-scoped op selector is ambiguous. "
                "Use tl.module(...)/tl.in_module(...) alone for one module-call splice, or use "
                "an op selector without tl.in_module(...) for op-level splicing."
            )
        if not live_selector_matches_site(normalized_entry.site_target, site):
            continue

        hook_args, hook_kwargs = _hook_call_inputs_for_site(
            normalized_entry,
            site=site,
            call_args=call_args,
            call_kwargs=call_kwargs,
            call_input_snapshots=call_input_snapshots,
        )
        hook_context = make_hook_context(
            name=_hook_display_name(normalized_entry),
            timing="post",
            direction="forward",
            layer_log=site,
            run_ctx=_live_run_ctx(),
            args=hook_args or (current_out,),
            kwargs=hook_kwargs,
        )
        previous_notes = tuple(hook_context.run_ctx.get("ledger_notes", ()))
        pre_hook_shape = tuple(current_out.shape)
        pre_hook_dtype = str(current_out.dtype)
        version_before = _tensor_version(current_out)
        content_before = _tensor_content_probe(current_out)
        result = _execute_hook(
            normalized_entry.normalized_callable,
            current_out,
            hook_context,
            force_shape_change=bool(normalized_entry.metadata.get("force_shape_change", False)),
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
            run_ctx=hook_context.run_ctx,
            replaced=replaced,
        )
        _append_active_spec_records([record])
        fire_results.append(
            FireResult(
                plan_id=str(
                    normalized_entry.metadata.get(
                        "plan_id",
                        normalized_entry.metadata.get(
                            "hook_id", _hook_display_name(normalized_entry)
                        ),
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


def _tensor_version(value: Any) -> int | None:
    """Return a tensor's in-place mutation counter, or ``None`` when unreadable.

    ``Tensor._version`` is the cheap autograd version counter; inference-mode
    tensors (no counter) and exotic subclasses read as ``None``, which callers
    treat as "no in-place evidence" rather than a refusal.
    """

    if not isinstance(value, torch.Tensor):
        return None
    try:
        return int(value._version)
    except Exception:
        return None


def _tuple_versions(values: tuple[torch.Tensor | None, ...]) -> tuple[int | None, ...]:
    """Version counters for one grad tuple, ``None`` per non-tensor slot."""

    return tuple(_tensor_version(value) for value in values)


_CONTENT_PROBE_SAMPLES = 8
"""Bounded per-fire sample width for the storage-alias mutation probe."""


def _tensor_content_probe(value: Any) -> tuple[Any, ...] | None:
    """Bounded strided content sample witnessing storage-alias mutations.

    ``Tensor._version`` misses mutations routed through a DIFFERENT impl over
    the same storage: ``.data`` mints a storage-sharing alias with a FRESH
    version counter, so ``out.data.mul_(0); return out`` changed execution
    with the counter witness reading "no mutation" (the incomplete half of
    2289e56c). Identity returns therefore pair the counter with this O(1)
    sample -- numel plus up to :data:`_CONTENT_PROBE_SAMPLES` evenly strided
    elements. Wholesale in-place edits (zeroing, scaling) are caught; a
    mutation confined to unsampled elements remains a documented residual,
    with replay validation the fail-closed authority. ``None`` (non-tensor /
    unreadable / exotic subclass) reads as "no evidence", never a refusal.
    """

    if not isinstance(value, torch.Tensor):
        return None
    from .._state import pause_logging

    try:
        with pause_logging():
            numel = int(value.numel())
            if numel == 0:
                return (0, ())
            flat = value.detach().reshape(-1)
            count = min(_CONTENT_PROBE_SAMPLES, numel)
            step = max(1, numel // count)
            sample = flat[::step][:count].tolist()
        return (numel, tuple(sample))
    except Exception:
        return None


def _content_probe_mutated(before: tuple[Any, ...] | None, after: tuple[Any, ...] | None) -> bool:
    """NaN-aware inequality between two content probes (missing = no evidence)."""

    if before is None or after is None:
        return False
    numel_before, sample_before = before
    numel_after, sample_after = after
    if numel_before != numel_after or len(sample_before) != len(sample_after):
        return True
    for left, right in zip(sample_before, sample_after, strict=True):
        if left != right and not (left != left and right != right):  # NaN == NaN here
            return True
    return False


def _tuple_content_probes(
    values: tuple[torch.Tensor | None, ...],
) -> tuple[tuple[Any, ...] | None, ...]:
    """Content probes for one grad tuple, ``None`` per non-tensor slot."""

    return tuple(_tensor_content_probe(value) for value in values)


def _tuple_probes_mutated(
    before: tuple[tuple[Any, ...] | None, ...],
    after: tuple[tuple[Any, ...] | None, ...],
) -> bool:
    """Whether any grad-tuple slot's content probe changed (NaN-aware)."""

    if len(before) != len(after):
        return True
    return any(
        _content_probe_mutated(left, right) for left, right in zip(before, after, strict=True)
    )


def _apply_inplace_replacement_to_mutated_storage(
    result: torch.Tensor,
    *,
    current_out: torch.Tensor,
    site: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, bool]:
    """Make a live output replacement effective for discarded in-place returns.

    Parameters
    ----------
    result:
        Validated hook result.
    current_out:
        Output tensor supplied to the hook.
    site:
        Capture-time site metadata.
    call_args:
        Live positional inputs after the operation executed.
    call_kwargs:
        Live keyword inputs after the operation executed.

    Returns
    -------
    tuple[torch.Tensor, bool]
        Effective output and whether execution accepted the replacement.
    """

    if result is current_out:
        return result, False
    if not bool(getattr(site, "is_inplace", False)):
        return result, True

    func_name = str(getattr(site, "func_name", ""))
    destination: torch.Tensor | None = None
    if _is_inplace_style_func_name(func_name) and call_args:
        candidate = call_args[0]
        if isinstance(candidate, torch.Tensor):
            destination = candidate
    elif call_kwargs is not None:
        candidate = call_kwargs.get("out")
        if isinstance(candidate, torch.Tensor):
            destination = candidate

    safe_to_copy = destination is not None and (
        tuple(destination.shape) == tuple(result.shape)
        and destination.dtype == result.dtype
        and destination.device == result.device
    )
    if safe_to_copy:
        assert destination is not None
        try:
            with pause_logging():
                destination.copy_(result)
        except Exception:
            safe_to_copy = False
    if safe_to_copy:
        return result, True

    warnings.warn(
        f"Refused an output replacement for in-place operation {func_name!r} because "
        "TorchLens could not safely copy it into the mutated tensor storage; execution "
        "continues with the original output and the fire record is marked replaced=False.",
        UserWarning,
        stacklevel=3,
    )
    return current_out, False


def _apply_module_boundary_live_hooks(
    out_orig: Any,
    *,
    module_address: str,
    module_call_index: int,
    module_type: str,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> Any:
    """Apply module-boundary live hooks to module forward outputs.

    Parameters
    ----------
    out_orig:
        Raw module forward output.
    module_address:
        TorchLens module address.
    module_call_index:
        One-based module call index.
    module_type:
        Module type name.
    call_args:
        Original module positional inputs.
    call_kwargs:
        Original module keyword inputs.

    Returns
    -------
    Any
        Module output with any tensor replacements applied.
    """

    trace = _state._active_trace
    predicate_options = getattr(trace, "_predicate_save_options", None)
    predicate_intervene = getattr(predicate_options, "intervene", None)
    predicate_selector = getattr(predicate_intervene, "selector", None)
    if predicate_selector is not None:
        from ..ir.selector_eval import selector_contains_kind
        from .selectors import BaseSelector

        if not isinstance(predicate_selector, BaseSelector) or not selector_contains_kind(
            predicate_selector, "module"
        ):
            predicate_intervene = None
    else:
        predicate_intervene = None
    if not _state._active_hook_plan and predicate_intervene is None:
        return out_orig
    module_call = (module_address, module_call_index)
    replacements: dict[tuple[Any, ...], torch.Tensor] = {}
    for out, container_path in _iter_tensor_outputs(out_orig):
        site = make_live_site_proxy(
            _layer_label_raw=f"{module_address}:{module_call_index}",
            func_name=module_type,
            layer_type=module_type.lower(),
            tensor=out,
            func_call_id=0,
            container_path=container_path,
            fields={
                "raw_index": 0,
                "module": module_call,
                "modules": (module_call,),
                "output_of_module_calls": (module_call,),
                "_tl_module_boundary": True,
            },
        )
        setattr(site, "_tl_module_boundary", True)
        hooked, fire_results = _apply_live_hooks(
            out,
            site=site,
            container_path=container_path,
            call_args=call_args,
            call_kwargs=call_kwargs,
        )
        all_fire_results = list(fire_results)
        if predicate_intervene is not None and trace is not None:
            from ..backends.torch.ops import _record_predicate_intervention_spec
            from ..capture.predicates import _evaluate_intervene_op
            from .hooks import normalize_hook_plan

            assert predicate_options is not None
            decision = _evaluate_intervene_op(site, predicate_options)
            if decision is not None:
                _record_predicate_intervention_spec(trace, site, decision)
                hook_entries = normalize_hook_plan(
                    decision.hook,
                    default_site_target=predicate_selector,
                    direction=decision.direction,
                )
                with active_intervention_context(
                    intervention_spec=getattr(trace, "_intervention_spec", None),
                    hook_plan=hook_entries,
                ):
                    hooked, predicate_fire_results = _apply_live_hooks(
                        hooked,
                        site=site,
                        container_path=container_path,
                        call_args=call_args,
                        call_kwargs=call_kwargs,
                    )
                all_fire_results.extend(predicate_fire_results)
                if predicate_fire_results:
                    trace._tl_intervene_selector_fire_count = int(
                        getattr(trace, "_tl_intervene_selector_fire_count", 0)
                    ) + len(predicate_fire_results)
        fire_results = tuple(all_fire_results)
        if fire_results:
            if hooked is not out:
                parent_label = get_tensor_label(out)
                if parent_label is not None:
                    _record_module_intervention_parent_labels(hooked, (parent_label,), trace)
                clear_tensor_label(hooked)
            _record_tensor_live_fire_results(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked
    if not replacements:
        return out_orig
    return _replace_tensor_outputs(out_orig, replacements)


_MODULE_INTERVENTION_PARENTS_ATTR = "_tl_module_intervention_parent_labels"
_MODULE_INTERVENTION_PARENTS_TABLE = "_tl_module_intervention_parents_by_id"


def _record_tensor_live_fire_results(
    tensor: torch.Tensor, fire_results: tuple[FireResult, ...]
) -> None:
    """Attach module-boundary fire results to a tensor, never dropping them silently.

    A replacement tensor that rejects dynamic attributes used to swallow the
    evidence (bare ``except: pass``), so the module exit reran with no
    intervention provenance and the fresh value was misclassified as an
    ``internal_source``. Delegate to the op-level setter, which falls back to
    the storage-owned side table and raises a typed ``CompatibilityError``
    only when NEITHER channel is writable.

    Parameters
    ----------
    tensor:
        Tensor that received a live module-boundary hook.
    fire_results:
        Fire results emitted by the live hook dispatcher.
    """

    from ..backends.torch._ops_interventions import _set_tensor_live_fire_results

    _set_tensor_live_fire_results(tensor, fire_results)


def _peek_tensor_live_fire_results(tensor: torch.Tensor) -> tuple[FireResult, ...]:
    """Return (without consuming) live fire results attached to ``tensor``.

    Checks the plain attribute first, then the storage-owned side table the
    robust setter falls back to for attr-rejecting replacement tensors. The
    module-exit consumer gates its replacement-vs-internal-source
    classification on this peek, so it must see both channels.

    Parameters
    ----------
    tensor:
        Tensor about to be classified at a module exit.
    """

    try:
        fire_results = tuple(getattr(tensor, "_tl_live_fire_results", ()) or ())
    except Exception:
        fire_results = ()
    if fire_results:
        return fire_results
    from ..backends.torch import _ops_interventions as intervention_state

    try:
        with pause_logging():
            storage = tensor.untyped_storage()
        records = getattr(storage, intervention_state._LIVE_FIRE_RESULTS_STORAGE_ATTR, None)
    except Exception:
        return ()
    if not isinstance(records, dict):
        return ()
    entry = records.get(id(tensor))
    if entry is not None and entry[0]() is tensor:
        return tuple(entry[1])
    return ()


def _record_module_intervention_parent_labels(
    tensor: torch.Tensor,
    parent_labels: tuple[str, ...],
    trace: Any,
) -> None:
    """Record the replaced-parent labels for a module-boundary replacement.

    Falls back to a trace-scoped identity-keyed table (weakly guarded against
    id reuse, dying with the capture) when the replacement tensor rejects
    dynamic attributes, so the boundary op minted at module exit keeps its
    dataflow parents instead of silently losing them.

    Parameters
    ----------
    tensor:
        Replacement tensor produced by a live module-boundary hook.
    parent_labels:
        Raw labels of the replaced module-output tensors.
    trace:
        Active trace owning the fallback table (``None`` tolerated; the loss
        is then disclosed with a warning rather than swallowed).
    """

    labels = tuple(parent_labels)
    try:
        setattr(tensor, _MODULE_INTERVENTION_PARENTS_ATTR, labels)
        return
    except Exception:
        pass
    if trace is None:
        warnings.warn(
            "TorchLens could not record intervention parent provenance for a "
            "module-boundary replacement tensor (dynamic attributes rejected and "
            "no active trace); the replacement op will carry no parents.",
            stacklevel=2,
        )
        return
    table = trace.__dict__.setdefault(_MODULE_INTERVENTION_PARENTS_TABLE, {})
    table[id(tensor)] = (weakref.ref(tensor), labels)


def _peek_module_intervention_parent_labels(tensor: torch.Tensor, trace: Any) -> tuple[str, ...]:
    """Return the recorded replaced-parent labels for ``tensor``, if any.

    Parameters
    ----------
    tensor:
        Module-output tensor being classified at a module exit.
    trace:
        Active trace whose fallback table is consulted when the tensor
        carries no attribute.
    """

    try:
        labels = tuple(getattr(tensor, _MODULE_INTERVENTION_PARENTS_ATTR, ()) or ())
    except Exception:
        labels = ()
    if labels:
        return labels
    table = getattr(trace, _MODULE_INTERVENTION_PARENTS_TABLE, None) if trace is not None else None
    if not isinstance(table, dict):
        return ()
    entry = table.get(id(tensor))
    if entry is not None and entry[0]() is tensor:
        return tuple(entry[1])
    return ()


def _iter_tensor_outputs(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[torch.Tensor, tuple[Any, ...]]]:
    """Yield tensor leaves and their paths from a module output structure.

    Parameters
    ----------
    value:
        Module output value to traverse.
    path:
        Current container path prefix.

    Yields
    ------
    tuple[torch.Tensor, tuple[Any, ...]]
        Tensor leaf and stable path.
    """

    if isinstance(value, torch.Tensor):
        yield value, path
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensor_outputs(item, (*path, key))


def _replace_tensor_outputs(value: Any, replacements: dict[tuple[Any, ...], torch.Tensor]) -> Any:
    """Return ``value`` with tensor leaves replaced by path.

    Parameters
    ----------
    value:
        Original module output structure.
    replacements:
        Mapping from tensor leaf path to replacement tensor.

    Returns
    -------
    Any
        Output structure with replacements applied.
    """

    if () in replacements:
        return replacements[()]
    if isinstance(value, tuple):
        rebuilt_items = tuple(
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        )
        if _is_namedtuple_instance(value):
            return tuple.__new__(type(value), rebuilt_items)
        return type(value)(rebuilt_items)
    if isinstance(value, list):
        return [
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_tensor_outputs_by_child(item, replacements, (key,))
            for key, item in value.items()
        }
    return value


def _replace_tensor_outputs_by_child(
    value: Any, replacements: dict[tuple[Any, ...], torch.Tensor], prefix: tuple[Any, ...]
) -> Any:
    """Return a child value with replacements beneath ``prefix`` applied.

    Parameters
    ----------
    value:
        Child value to rebuild.
    replacements:
        Full replacement mapping.
    prefix:
        Path prefix for ``value`` within its parent.

    Returns
    -------
    Any
        Child value with matching replacements applied.
    """

    child_replacements = {
        path[len(prefix) :]: replacement
        for path, replacement in replacements.items()
        if path[: len(prefix)] == prefix
    }
    if not child_replacements:
        return value
    return _replace_tensor_outputs(value, child_replacements)


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance.

    Parameters
    ----------
    value:
        Candidate container.

    Returns
    -------
    bool
        Whether ``value`` is a tuple with ``_fields`` metadata.
    """

    fields = getattr(type(value), "_fields", None)
    return isinstance(value, tuple) and isinstance(fields, tuple)


def _hook_call_inputs_for_site(
    entry: NormalizedHookEntry,
    *,
    site: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any] | None,
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the positional and keyword inputs for one hook fire.

    Parameters
    ----------
    entry:
        Normalized hook entry being executed.
    site:
        Live site proxy.
    call_args:
        Captured positional inputs.
    call_kwargs:
        Captured keyword inputs.
    call_input_snapshots:
        Optional pre-execution input snapshots captured for in-place ops.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Hook positional and keyword inputs.
    """

    if not _helper_routes_inputs(entry.helper_spec):
        return call_args, dict(call_kwargs or {})
    if not getattr(site, "_tl_module_boundary", False):
        if bool(getattr(site, "_tl_input_snapshot", False)) and call_input_snapshots is not None:
            return call_input_snapshots
        if bool(getattr(site, "is_inplace", False)):
            warnings.warn(
                "An input-routed intervention fired on an in-place operation without a "
                "pre-mutation input snapshot; the intervention may be seeing post-mutation "
                "inputs. This can occur with out= aliasing or a conservative pre-gate miss.",
                UserWarning,
                stacklevel=3,
            )
        return call_args, dict(call_kwargs or {})
    return call_args, dict(call_kwargs or {})


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


def _is_inplace_style_func_name(func_name: str) -> bool:
    """Return whether ``func_name`` follows TorchLens' in-place op naming rules.

    Parameters
    ----------
    func_name:
        Decorated torch function name.

    Returns
    -------
    bool
        Whether the function is a candidate in-place op before execution.
    """

    return (
        (func_name.endswith("_") and not func_name.startswith("__"))
        or func_name.startswith("__i")
        or func_name in {"__setitem__", "__delitem__"}
    )


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
    if getattr(decision, "direction", "forward") not in {"forward", "both"}:
        return False
    return _selector_may_match_provisional_site(selector, site)


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

    if entry.metadata.get("direction", "forward") != "forward":
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


def _selector_uses_only_provisional_fields(selector: Any) -> bool:
    """Return whether a selector can be faithfully evaluated before execution.

    Parameters
    ----------
    selector:
        Selector-like target to inspect.

    Returns
    -------
    bool
        Whether a false pre-execution match result is definitive.
    """

    try:
        normalized = getattr(selector, "selector_kind", None)
        # A provisional op explicitly carries ``_tl_module_boundary=False`` and
        # ``output_of_module_calls=()``. Therefore ``tl.module`` is definitively
        # false here; its eventual true match is evaluated by the module-exit hook.
        if normalized in {"func", "module", "in_module"}:
            return True
        if normalized in {"and", "or"}:
            return all(
                _selector_uses_only_provisional_fields(child) for child in selector.selectors
            )
        if normalized == "not":
            return _selector_uses_only_provisional_fields(selector.selector)
    except Exception:
        return False
    return False


def _helper_routes_inputs(helper: Any | None) -> bool:
    """Return whether a helper consumes call inputs from ``hook.args``.

    Parameters
    ----------
    helper:
        Helper spec-like object.

    Returns
    -------
    bool
        Whether the helper is ``splice_module(input='in')``.
    """

    if helper is None or getattr(helper, "name", None) != "splice_module":
        return False
    metadata = dict(getattr(helper, "metadata", ()))
    return metadata.get("input") == "in"


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

    from ..backends.torch.ops import _snapshot_exhaustive_module_stack

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


def _input_splice_module_scope(site_target: Any, entry: NormalizedHookEntry) -> str | None:
    """Return the module address for input-spliced module-scoped selectors.

    Parameters
    ----------
    site_target:
        Selector-like hook target.
    entry:
        Normalized hook entry.

    Returns
    -------
    str | None
        Module address when the entry is an input-spliced module scope.
    """

    helper = entry.helper_spec
    if helper is None or helper.name != "splice_module":
        return None
    helper_metadata = dict(helper.metadata)
    if helper_metadata.get("input") != "in":
        return None
    return _module_scope_address(site_target)


def _module_scope_address(site_target: Any) -> str | None:
    """Return a module address when a selector tree contains module scope.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    str | None
        First module address found in the selector tree.
    """

    kind = getattr(site_target, "selector_kind", None)
    if kind in {"module", "in_module"}:
        value = getattr(site_target, "selector_value", None)
        return None if value is None else str(value)
    selectors = getattr(site_target, "selectors", None)
    if selectors is not None:
        for selector in selectors:
            address = _module_scope_address(selector)
            if address is not None:
                return address
    selector = getattr(site_target, "selector", None)
    if selector is not None:
        return _module_scope_address(selector)
    return None


def _is_plain_module_selector(site_target: Any) -> bool:
    """Return whether a selector is exactly a module boundary or containment selector.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    bool
        Whether the selector is exactly ``tl.module`` or ``tl.in_module``.
    """

    return getattr(site_target, "selector_kind", None) in {"module", "in_module"}


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
        if normalized_entry.metadata.get("direction", "forward") != "backward":
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
        if normalized_entry.metadata.get("direction", "forward") != "backward":
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


def _record_backward_selector_fire(entry: NormalizedHookEntry) -> None:
    """Increment the deferred zero-match ledger for one backward selector fire.

    Parameters
    ----------
    entry:
        Normalized backward hook entry that matched the live GradFn site.
    """

    if entry.metadata.get("created_by") != "intervene_backward_selector":
        return
    trace = _state._active_trace
    if trace is None:
        return
    trace._tl_intervene_selector_fire_count = (
        int(getattr(trace, "_tl_intervene_selector_fire_count", 0)) + 1
    )


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


def _live_run_ctx() -> dict[str, Any]:
    """Return the shared run context for the active model log.

    Returns
    -------
    dict[str, Any]
        Mutable context shared by hooks in this live run.
    """

    trace = _state._active_trace
    if trace is None:
        return {}
    run_ctx = getattr(trace, "last_run", None)
    if run_ctx is None:
        run_ctx = {"engine": "live", "timestamp": time.monotonic()}
        trace.last_run = run_ctx
    else:
        run_ctx.setdefault("engine", "live")
        run_ctx.setdefault("timestamp", time.monotonic())
    return run_ctx


def _hook_display_name(entry: NormalizedHookEntry) -> str:
    """Return a stable display name for a hook entry.

    Parameters
    ----------
    entry:
        Normalized hook entry.

    Returns
    -------
    str
        Helper name or callable qualname.
    """

    if entry.helper_spec is not None:
        return entry.helper_spec.name
    return getattr(entry.normalized_callable, "__qualname__", "user_hook")


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

    from .audit import build_fire_record

    return build_fire_record(
        target_label=site._layer_label_raw,
        call_label=site._layer_label_raw,
        func_call_id=site.func_call_id,
        container_path=container_path,
        engine="live",
        helper=entry.helper_spec,
        site_label=site._layer_label_raw,
        timing="post",
        direction="forward",
        helper_name=_hook_display_name(entry),
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

    from .audit import build_fire_record

    tuple_index = _first_replaced_tuple_index(previous, current)
    label = str(getattr(grad_fn_handle, "label", ""))
    pass_index = _active_backward_pass_index(grad_fn_handle)
    return build_fire_record(
        target_label=label,
        call_label=f"{label}:{call_index}" if label else str(call_index),
        func_call_id=None,
        container_path=(),
        engine="live",
        helper=entry.helper_spec,
        site_label=label,
        timing=timing,  # type: ignore[arg-type]
        direction="backward",
        helper_name=_hook_display_name(entry),
        backward_pass_index=pass_index,
        call_index=call_index,
        grad_kind=grad_kind,  # type: ignore[arg-type]
        tuple_index=tuple_index,
        replaced=tuple_index is not None or current is not previous or inplace_mutated,
    )


def _first_replaced_tuple_index(
    previous: tuple[torch.Tensor | None, ...],
    current: tuple[torch.Tensor | None, ...],
) -> int | None:
    """Return the first tuple slot whose object identity changed.

    Parameters
    ----------
    previous:
        Tuple before helper execution.
    current:
        Tuple after helper execution.

    Returns
    -------
    int | None
        First changed slot index, or ``None`` when object identities match.
    """

    for index, (before, after) in enumerate(zip(previous, current, strict=False)):
        if before is not after:
            return index
    return None


def _active_backward_pass_index(grad_fn_handle: Any) -> int | None:
    """Return the active trace backward pass index for a grad_fn handle.

    Parameters
    ----------
    grad_fn_handle:
        Runtime GradFn record or compatible test stub.

    Returns
    -------
    int | None
        Active one-based backward pass index, when available.
    """

    trace = getattr(grad_fn_handle, "source_trace", None)
    if trace is None:
        return None
    value = getattr(trace, "_active_backward_pass_index", None)
    return int(value) if isinstance(value, int) else None


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
    spec = _state._active_intervention_spec
    if spec is not None and hasattr(spec, "records"):
        spec.records.extend(records)


def _site_name(hook_context: HookContext | None) -> str:
    """Return a readable site name for hook diagnostics.

    Parameters
    ----------
    hook_context:
        Optional hook context.

    Returns
    -------
    str
        Layer label or ``"<unknown site>"``.
    """

    if hook_context is None:
        return "<unknown site>"
    layer_label = hook_context.layer_log.get("layer_label")
    if layer_label is None:
        return "<unknown site>"
    return str(layer_label)


def do(log: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a one-shot intervention operation to a model log.

    Parameters
    ----------
    log:
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

    return log.do(*args, **kwargs)


__all__ = [
    "HOOK_REENTRANCY_GUARD",
    "_HookReentrancyGuard",
    "_apply_live_hooks",
    "_apply_live_backward_hooks",
    "_apply_live_backward_prehooks",
    "_execute_hook",
    "active_intervention_context",
    "do",
    "validate_hook_output",
]
