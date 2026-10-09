"""Steered, shape-varying support for the guarded fast live engine.

The fast live session (``_fast_run._FastLiveSession``) runs the model's native
forward and collects the saved sites. This module adds the three pieces a
steered rerun needs:

* the staged intervention spec applied at REAL module boundaries through the
  same ``_apply_module_boundary_live_hooks`` door the capture lane fires at
  every module exit (same selector matching, same ``_execute_hook`` gate,
  same ``FireResult`` records);
* shape-varied admission: an input whose rank, dtype and device match the
  capture but whose size differs is admitted, and the run is then guarded by
  the ordered call fingerprint (``Trace._raw_call_fingerprint``) rather than by
  size equality;
* the honesty rule for a shape-varied run: metadata the native forward did not
  refresh (unsaved ops' shapes and sizes) reads ``None``, TorchLens's existing
  not-available spelling, never the capture-time numbers.

Only plain ``tl.module(address)`` boundary targets are applied here; every
other staged target refuses typed so the legacy door falls back to capture.
"""

from __future__ import annotations

import threading
import warnings
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from . import _runnable_execution as _execution
from ._errors import TorchLensWarning
from ._runnable_execution import _INPUT_CHECK_UNAVAILABLE, _contract_check
from .errors import RunCapabilityUnavailableError
from .runnable import ContractCheck, RunnableErrorCode

#: Op fields a native forward cannot refresh for unsaved ops; after a
#: shape-varied run they read ``None`` instead of the capture-time numbers.
_UNREFRESHED_SHAPE_FIELDS: tuple[str, ...] = (
    "shape",
    "transformed_out_shape",
    "activation_memory",
    "transformed_activation_memory",
)


def fast_live_input_admission(
    trace: Any,
    input_args: Any,
    input_kwargs: Any,
    *,
    allow_shape_change: bool,
) -> tuple[ContractCheck | None | Any, bool]:
    """Check the runtime inputs against the captured input boundary.

    Parameters
    ----------
    trace:
        Live trace whose input ops carry the recorded shape and dtype.
    input_args, input_kwargs:
        Runtime inputs.
    allow_shape_change:
        Whether a size difference is admitted (rank and dtype must still match).

    Returns
    -------
    tuple[ContractCheck | None | Any, bool]
        The first failed check (``None`` when every check passed, or the
        ``_INPUT_CHECK_UNAVAILABLE`` sentinel when nothing could be verified),
        and whether any admitted input size differs from the capture.
    """

    try:
        return _input_admission_checks(
            trace, input_args, input_kwargs, allow_shape_change=allow_shape_change
        )
    except Exception:  # noqa: BLE001 -- a broken guard must refuse, never fail open
        return _INPUT_CHECK_UNAVAILABLE, False


def _input_admission_checks(
    trace: Any,
    input_args: Any,
    input_kwargs: Any,
    *,
    allow_shape_change: bool,
) -> tuple[ContractCheck | None | Any, bool]:
    """Run the admission checks; exceptions surface to the caller's sentinel arm."""

    input_labels = list(getattr(trace, "input_layers", ()) or ())
    if not input_labels:
        return None, False
    layer_dict = getattr(trace, "layer_dict_all_keys", None) or {}
    # Module attribute, not a bound name: the guard-failure fault injection
    # patches ``_runnable_execution._live_runtime_input_leaves``.
    leaves = _execution._live_runtime_input_leaves(input_args, input_kwargs)
    if leaves is None:
        return _INPUT_CHECK_UNAVAILABLE, False
    if len(leaves) != len(input_labels):
        return (
            _contract_check(
                "input_arity",
                False,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                f"Runtime input tree carries {len(leaves)} tensor leaves; "
                f"the capture recorded {len(input_labels)}.",
            ),
            False,
        )
    shape_changed = False
    for label, value in zip(input_labels, leaves, strict=True):
        op = layer_dict.get(label)
        if op is None:
            return _INPUT_CHECK_UNAVAILABLE, False
        expected_shape = tuple(op.shape) if op.shape is not None else None
        expected_dtype = str(op.dtype) if op.dtype is not None else None
        actual_shape = tuple(value.shape)
        if expected_dtype is not None and str(value.dtype) != expected_dtype:
            return (
                _contract_check(
                    f"input_dtype:slot:{label}",
                    False,
                    RunnableErrorCode.INPUT_DTYPE_MISMATCH,
                    f"Runtime input dtype {value.dtype} does not match {expected_dtype}.",
                    affected_op_labels=(label,),
                ),
                False,
            )
        if expected_shape is None or actual_shape == expected_shape:
            continue
        if not allow_shape_change or len(actual_shape) != len(expected_shape):
            return (
                _contract_check(
                    f"input_shape:slot:{label}",
                    False,
                    RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                    f"Runtime input shape {actual_shape} does not match {expected_shape}.",
                    affected_op_labels=(label,),
                ),
                False,
            )
        shape_changed = True
    return None, shape_changed


def _staged_user_hook_specs(trace: Any) -> list[Any]:
    """Return the staged hook specs a run must apply (engine-owned residue excluded)."""

    spec = trace._intervention_spec
    return [
        hook_spec
        for hook_spec in getattr(spec, "hook_specs", ())
        if not (getattr(hook_spec, "metadata", None) or {}).get("selection_do_engine_owned")
    ]


def _staged_entries(trace: Any) -> tuple[Any, ...]:
    """Return the trace's staged spec object followed by its hook and value entries."""

    spec = trace._intervention_spec
    if spec is None:
        return ()
    return (
        spec,
        *(getattr(spec, "hook_specs", None) or ()),
        *(getattr(spec, "target_value_specs", None) or ()),
    )


def module_boundary_plan(trace: Any) -> tuple[list[Any], tuple[str, ...]]:
    """Normalize the staged spec into a module-boundary hook plan.

    Returns
    -------
    tuple[list[Any], tuple[str, ...]]
        The normalized hook entries (with unique plan ids) and the module
        addresses they target; both empty when nothing is staged.

    Raises
    ------
    RunCapabilityUnavailableError
        ``fast_rerun_target_unsupported`` when a staged entry is not a plain
        ``tl.module(address)`` boundary target, or stages a value replacement.
    """

    from .intervention.hooks import normalize_hooks_from_spec
    from .intervention.rerun import _assign_unique_plan_ids
    from .intervention.runtime import _is_plain_module_selector

    spec = trace._intervention_spec
    if spec is None or (not _staged_user_hook_specs(trace) and not spec.target_value_specs):
        return [], ()
    if spec.target_value_specs:
        raise RunCapabilityUnavailableError(
            "The guarded fast engine applies module-boundary hooks only; this trace "
            f"stages {len(spec.target_value_specs)} value replacement(s).",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_rerun_target_unsupported",
        )
    hook_plan = _assign_unique_plan_ids(normalize_hooks_from_spec(spec))
    addresses: list[str] = []
    for entry in hook_plan:
        target = entry.site_target
        if (
            getattr(target, "selector_kind", None) != "module"
            or not _is_plain_module_selector(target)
            or not isinstance(getattr(target, "selector_value", None), str)
        ):
            raise RunCapabilityUnavailableError(
                "The guarded fast engine applies plain tl.module(address) boundary "
                f"targets only; staged target {target!r} needs the capture engine.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_rerun_target_unsupported",
            )
        addresses.append(str(target.selector_value).rsplit(":", 1)[0])
    return hook_plan, tuple(dict.fromkeys(addresses))


def _strip_pass(label: Any) -> str:
    """Return a call or op label without its ``:<index>`` suffix."""

    text = str(label)
    head, sep, tail = text.rpartition(":")
    return head if sep and tail.isdigit() else text


def graph_fired_addresses(trace: Any) -> frozenset[str]:
    """Return the module addresses whose boundary fired an intervention in the recorded graph.

    A module-boundary fire record names the module call (``address:index``) as
    its site, so the address is the call label without its index.
    """

    fired: set[str] = set()
    for op in getattr(trace, "layer_list", ()):
        for record in getattr(op, "interventions", None) or ():
            for attr in ("site_label", "target_label"):
                label = getattr(record, attr, None)
                if isinstance(label, str) and label:
                    fired.add(_strip_pass(label))
    return frozenset(fired)


def require_graph_reflects_plan(trace: Any, hook_plan: list[Any]) -> None:
    """Refuse the fast engine while the trace's graph does not show the staged plan.

    The fast engine refreshes saved values and leaves the recorded graph alone,
    so the graph must already show exactly the staged plan. A staged entry whose
    module never fired in the graph (hooks attached after a plain capture) would
    leave a trace whose values are steered but whose ops show no intervention,
    and a fire the graph records for an entry no longer staged (hooks cleared or
    detached after a steered capture) would leave replacement ops on a plain
    run. The capture engine rewrites the graph on that first rerun; every later
    rerun of the same trace is eligible here.

    Raises
    ------
    RunCapabilityUnavailableError
        ``fast_rerun_graph_unsteered`` naming the staged entries the graph lacks,
        or the recorded fires no staged entry accounts for.
    """

    fired = graph_fired_addresses(trace)
    planned = {str(entry.site_target.selector_value).rsplit(":", 1)[0] for entry in hook_plan}
    unstaged = sorted(fired - planned)
    if unstaged:
        raise RunCapabilityUnavailableError(
            "This trace's recorded graph carries intervention fires that no staged entry "
            f"accounts for ({', '.join(unstaged)}); the capture engine reruns once to record "
            "the current plan.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_rerun_graph_unsteered",
        )
    if not hook_plan:
        return
    from .intervention.rerun import _hook_plan_identifier

    missing = []
    for entry in hook_plan:
        address = str(entry.site_target.selector_value).rsplit(":", 1)[0]
        if address not in fired:
            missing.append(f"{_hook_plan_identifier(entry)}@{address}")
    if missing:
        raise RunCapabilityUnavailableError(
            "Staged intervention entries have not fired in this trace's recorded graph "
            f"({', '.join(missing)}); the capture engine reruns once to record them.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_rerun_graph_unsteered",
        )


class SteerPlan:
    """Per-session steering state: the hook plan, its targets, and run counters."""

    def __init__(self, trace: Any, model: nn.Module) -> None:
        """Lower the staged spec and resolve its module targets against ``model``."""

        self.hook_plan, self.addresses = module_boundary_plan(trace)
        self.spec = trace._intervention_spec if self.hook_plan else None
        self.staged = _staged_entries(trace)
        modules = dict(model.named_modules())
        for address in self.addresses:
            if modules.get(address) is None:
                raise RunCapabilityUnavailableError(
                    f"Staged intervention target {address!r} is absent from the live model.",
                    code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                    detection_stage="fast_rerun_target_unsupported",
                )
        self.modules = {address: modules[address] for address in self.addresses}
        require_graph_reflects_plan(trace, self.hook_plan)
        self.pass_counts: Counter[str] = Counter()
        self.fired: Counter[str] = Counter()
        self.fire_count = 0

    def follows(self, trace: Any) -> bool:
        """Return whether ``trace`` still stages exactly the entries this plan lowered.

        The staged spec is mutable (``attach_hooks``, ``clear_hooks``), so a
        cached session compares the entry objects themselves; the plan holds
        them, so an identity match cannot be a reused address.
        """

        current = _staged_entries(trace)
        return len(current) == len(self.staged) and all(
            left is right for left, right in zip(current, self.staged)
        )

    def refuse_changed_helpers(self) -> None:
        """Refuse a run whose staged helper tensors changed since they were staged.

        Raises
        ------
        SpecMutationError
            ``helper_tensor_changed_since_capture``, as the capture rerun raises.
        """

        if self.spec is not None:
            from .intervention._helper_fingerprint import refuse_changed_staged_helpers

            refuse_changed_staged_helpers(self.spec, door="rerun")

    def reset(self) -> None:
        """Clear the per-run counters."""

        self.pass_counts.clear()
        self.fired.clear()
        self.fire_count = 0

    def context(self) -> Any:
        """Return the intervention context manager for one run."""

        from contextlib import nullcontext

        from .intervention.runtime import active_intervention_context

        if not self.hook_plan:
            return nullcontext()
        return active_intervention_context(intervention_spec=self.spec, hook_plan=self.hook_plan)

    def apply(self, address: str, module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        """Fire the staged boundary hooks at one module exit; return the new output or None."""

        from .intervention._module_boundary import (
            _apply_module_boundary_live_hooks,
            _iter_tensor_outputs,
            _peek_tensor_live_fire_results,
        )

        self.pass_counts[address] += 1
        hooked = _apply_module_boundary_live_hooks(
            output,
            module_address=address,
            module_call_index=self.pass_counts[address],
            module_type=type(module).__name__,
            call_args=tuple(args),
            call_kwargs={},
        )
        for leaf, _path in _iter_tensor_outputs(hooked):
            for result in _peek_tensor_live_fire_results(leaf):
                self.fired[str(result.plan_id)] += 1
                self.fire_count += 1
        return None if hooked is output else hooked

    def unfired_plan_ids(self) -> tuple[str, ...]:
        """Return the plan ids that fired nowhere during the last run."""

        from .intervention.rerun import _hook_plan_identifier

        planned = Counter(_hook_plan_identifier(entry) for entry in self.hook_plan)
        unfired: list[str] = []
        for plan_id, count in planned.items():
            unfired.extend([plan_id] * max(0, count - self.fired[plan_id]))
        return tuple(unfired)

    def warn_unfired(self) -> tuple[str, ...]:
        """Emit the rerun contract's ``rerun_zero_fire`` warning for unfired entries."""

        unfired = self.unfired_plan_ids()
        if unfired:
            warnings.warn(
                TorchLensWarning(
                    "Rerun hook plan entries fired at zero sites on the new inputs: "
                    f"{list(unfired)!r}. The rerun completed, but those interventions were "
                    "no-ops. Remedy: resolve the target sites against the rerun trace "
                    "(trace.resolve_sites) before re-applying",
                    code="rerun_zero_fire",
                    unfired_plan_ids=list(unfired),
                ),
                stacklevel=4,
            )
        return unfired


def session_is_active(session_ref: weakref.ReferenceType[Any]) -> Callable[[], bool]:
    """Return the activity test of a session's persistent hooks.

    True only while the session runs on its owner thread, so hooks left on the
    model by a cached session stay inert for captures and other sessions.
    """

    def active() -> bool:
        session = session_ref()
        return (
            session is not None
            and bool(session.active)
            and threading.get_ident() == session.owner_thread_id
        )

    return active


def install_steer_hooks(plan: SteerPlan, session_ref: weakref.ReferenceType[Any]) -> list[Any]:
    """Register one persistent forward hook per steered module.

    The hooks are installed BEFORE the session's collection hooks so torch
    runs them first and the collected site value is the post-intervention
    value, exactly as capture saves it. They are inert unless the session is
    active on the owner thread.
    """

    handles: list[Any] = []
    active = session_is_active(session_ref)
    for address, module in plan.modules.items():

        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            output: Any,
            *,
            address: str = address,
            ref: weakref.ReferenceType[Any] = session_ref,
        ) -> Any:
            """Apply the staged boundary hooks for this address during an active run."""

            session = ref()
            if session is None or not active():
                return None
            return session.steer_plan.apply(address, module, args, output)

        handles.append(module.register_forward_hook(hook))
    return handles


def clear_unrefreshed_shape_metadata(trace: Any, refreshed_labels: frozenset[str]) -> int:
    """Set unsaved ops' shape and size metadata to ``None`` after a shape-varied run.

    Returns the number of ops whose fields were cleared.
    """

    cleared = 0
    for op in trace.layer_list:
        if op.label in refreshed_labels:
            continue
        for field_name in _UNREFRESHED_SHAPE_FIELDS:
            if getattr(op, field_name, None) is not None:
                setattr(op, field_name, None)
        cleared += 1
    return cleared


def refusal_code(exc: BaseException) -> str:
    """Return the typed code of a fast-engine refusal for the rerun ledger.

    The code is the error's stable code, followed by ``:<stage>`` when the
    error names a detection stage or a failed contract check, so the ledger
    says WHICH guard sent the run back to the capture engine.
    """

    fields = getattr(exc, "fields", None)
    fields = fields if isinstance(fields, Mapping) else {}
    code = getattr(exc, "code", None) or fields.get("code")
    if not isinstance(code, str) or not code:
        code = type(exc).__name__
    stage = getattr(exc, "detection_stage", None) or fields.get("detection_stage")
    if not stage:
        check = getattr(exc, "contract_check", None) or fields.get("contract_check")
        stage = getattr(check, "name", None)
    return f"{code}:{stage}" if isinstance(stage, str) and stage else code


def output_dtype_and_rank(value: Any) -> tuple[str | None, int | None]:
    """Return ``(dtype, rank)`` of a tensor, or ``(None, None)`` for non-tensors."""

    if isinstance(value, torch.Tensor):
        return str(value.dtype), value.ndim
    return None, None
