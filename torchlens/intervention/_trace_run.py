"""Trace run functions reached from the records through ``_data_classes._doors.intervention``.

``Trace.run`` and ``trace.autograd.disarm`` are the thin methods;
the ``_runnable`` / ``_fast_run`` / ``_fast_live_steer`` dispatch lives here.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from torch import nn

from .._capture._episode_coupling import refuse_coupled_replay
from .._capture._episode_join import refuse_broken_join_claim
from .._capture.outcome import (
    CaptureStatus,
    outcome_for,
    refuse_streamed_trace_run,
    require_capture_capability,
)
from .._capture.projectors import RefreshProjector
from .._capture.structure_only import require_structure_only_capability
from .._data_classes._staged_spec import _staged_user_entries
from .._data_classes._trace_rerun_state import (
    mark_stateful_run_warning_emitted,
    stateful_run_warning_was_emitted,
)
from .._data_classes.trace import _USE_STORED_TRANSFORM
from .._deprecations import MISSING, MissingType
from .._errors import InvalidArgumentError, KeywordConflictError, RecordBindingError
from .._fast_live_steer import refusal_code
from .._fast_run import (
    _FastLiveSession,
    _FastSparseSession,
    close_fast_run_session,
    live_source_model,
)
from .._runnable import (
    DivergencePolicy,
    PathFaithfulness,
    RunnableErrorCode,
    RunProvider,
    RunResult,
)
from .._runnable.execution import (
    raise_analysis_run_unavailable,
    refuse_truncated_rerun,
    run_live_trace,
    run_loaded_sparse_trace,
)
from .._runnable.state import validate_run_seed
from .._runnable.transaction import _LiveRunOptions, _resolve_run_save_labels
from .._runnable.verification import _run_fork_name
from .._trace_state import TraceState
from ..backends.torch._trace_backward import _refuse_stats_capture
from ..backends.torch.backward import disarm_triggers as _disarm_triggers_impl
from ..errors import (
    PathDivergenceError,
    RunCapabilityUnavailableError,
    RuntimeSignatureDriftError,
    StateBindingError,
)
from ..options import ReplayOptions, merge_replay_options
from ._intervened_rerun_guard import refuse_unpersisted_intervention_rerun
from ._zero_fire import check_zero_fire_policy, zero_fire_policy_scope
from .errors import EngineDispatchError, PendingValueEditsWarning
from .rerun import run as _rerun_impl

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = [
    "disarm",
    "run",
]


def run_fast_loaded_trace(trace: Any, inputs: Any, *, seed: int | None) -> RunResult:
    """Run or initialize the explicit verify-once loaded sparse fast path.

    ``trace`` is the loaded sparse source Trace, ``inputs`` the runtime input tree
    and ``seed`` is pinned for the lifetime of the fast session. Returns the first
    ordinary verified result or a later compiled fast result.
    """

    session = trace.__dict__.get("_fast_run_session")
    if isinstance(session, _FastSparseSession):
        return session.run(inputs, seed=seed)
    result = run_loaded_sparse_trace(
        trace,
        inputs,
        seed=seed,
        on_divergence=DivergencePolicy.RAISE,
    )
    if result.report.path_faithfulness is not PathFaithfulness.VERIFIED:
        raise RunCapabilityUnavailableError(
            "fast=True requires a fully verified first sparse run; this artifact settled "
            f"{result.report.path_faithfulness.value}.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_verify_once",
        )
    trace.__dict__["_fast_run_session"] = _FastSparseSession.build(
        trace,
        result.trace,
        seed=seed,
    )
    return result


def run_fast_live_trace(
    trace: Any, inputs: Any, *, seed: int | None, until: Any = None
) -> RunResult:
    """Run the explicit native-forward targeted-collection live fast path.

    The result is the cached session's own fork; the source trace stays untouched.
    """

    model = live_source_model(trace)
    if model is None:
        raise RunCapabilityUnavailableError(
            "The live Trace no longer retains its source model: the trace "
            "holds it only weakly, so live-run availability depends on the "
            "caller keeping a strong reference (an inline-constructed model "
            "is collected at the first gc pass after capture). Keep the "
            "model alive, or save/load a runnable artifact instead.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            provider=RunProvider.LIVE,
        )
    session = trace.__dict__.get("_fast_run_session")
    if (
        not isinstance(session, _FastLiveSession)
        or session.model_ref() is not model
        or not session.steer_plan.follows(trace)
    ):
        if hasattr(session, "close"):
            session.close()
        result_fork = trace._fork_trace(name=_run_fork_name(trace))
        session = _FastLiveSession(trace, model, target=result_fork)
        trace.__dict__["_fast_run_session"] = session
    return session.run(inputs, seed=seed, until=until)


def try_guarded_fast_rerun(
    trace: Any, model: nn.Module, inputs: Any, *, output_transform: Any = None
) -> tuple[bool, str | None]:
    """Run the legacy intervened rerun through the guarded fast engine when eligible.

    Parameters
    ----------
    trace:
        Live trace carrying the staged spec; this door refreshes it in place on success.
    model:
        The model the caller passed (or the trace's retained live model).
    inputs:
        Transformed forward input.
    output_transform:
        The resolved rerun output transform, applied to the native output for
        ``trace.transformed_output`` exactly as the capture engine does.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` when the fast engine ran; ``(False, code)`` with the
        typed refusal code when the caller must fall back to the capture
        engine. A refused session is closed so the fallback's rebuilt trace
        starts a fresh plan next time; the trace is never poisoned here
        because the fallback replaces its state.
    """

    # The legacy door leaves no TorchLens hook on the user's model between
    # calls (the rerun hook-staging contract), so the session is built for the
    # run and closed after it; the explicit fast=True door keeps its cached
    # session as before.
    close_fast_run_session(trace)
    session: _FastLiveSession | None = None
    try:
        session = _FastLiveSession(trace, model)
        session.poison_on_divergence = False
        # The legacy door returns the refreshed Trace, not a run verdict; the
        # hooked-run ceiling belongs to the explicit run(inputs=, fast=True) door.
        session.reports_hooked_ceiling = False
        session.output_transform = output_transform
        session.run(inputs, seed=None)
    except (RunCapabilityUnavailableError, PathDivergenceError, RuntimeSignatureDriftError) as exc:
        return False, refusal_code(exc)
    finally:
        if session is not None:
            session.close()
    return True, None


def _apply_run_save_selection(result: Any, selected: Any) -> Any:
    """Apply the run-time save= retention selection to one settled RunResult.

    Retention/presentation ONLY (L4 sec 4): verification breadth is untouched
    (every contract check already ran at settlement), the attestation aggregate
    is unchanged, and input/output boundary nodes keep their payloads exactly
    like the capture-time selection machinery. ``None`` (save= omitted) is
    today's behavior byte-for-byte.
    """

    if selected is None:
        return result
    for layer in result.trace._layer_list:
        if layer.type in ("input", "output"):
            continue
        if layer.layer_label in selected:
            continue
        RefreshProjector._clear_payload(layer)
    return result


def _refuse_state_compromised_live_run(trace: Any) -> None:
    """Refuse live/fast execution after a failed declared-state restore (L4 5.4).

    The session-scoped STATE-COMPROMISED latch means a prior run()'s restore
    failed mid-bracket, so the LIVE MODEL's declared state is unknown -- every
    door that reads the live model (default live, fast live, legacy rerun)
    refuses typed. Loaded-sparse runs of a saved artifact stay legal: they
    execute against staged clones and never read the live model.
    """
    try:
        runnable_state = trace._runnable
    except AttributeError:
        runnable_state = None
    compromised = getattr(runnable_state, "state_compromised", None)
    if compromised is None:
        return

    raise StateBindingError(
        "A prior run() failed while RESTORING this trace's live model state "
        f"(stopped at state entry {compromised.get('state_dict_name')!r}), so the "
        "live model's declared state is unknown and live/fast execution would "
        "misreport. Remedy: reload known-good weights onto the model (or "
        "re-capture / re-stage state), then run again",
        code="run_state_restore_failed",
        detection_stage="state_restore",
        **dict(compromised),
    )


def _warn_stateful_live_run_once(trace: Any, model: nn.Module) -> None:
    """Warn once when a live rerun has an obvious model-state mutation risk.

    Parameters
    ----------
    trace:
        Source Trace carrying the once-only warning marker.
    model:
        Live model that is about to be re-executed.
    """

    if stateful_run_warning_was_emitted(trace):
        return
    if not model.training:
        return
    running_stat_risk: tuple[str, tuple[str, ...]] | None = None
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm) or not module.training:
            continue
        buffers = getattr(module, "_buffers", {})
        running_stat_buffers = tuple(
            name
            for name in ("running_mean", "running_var", "num_batches_tracked")
            if buffers.get(name) is not None
        )
        if bool(getattr(module, "track_running_stats", False)) and running_stat_buffers:
            running_stat_risk = (module_name or "<root>", running_stat_buffers)
            break
    if running_stat_risk is None:
        return
    module_name, buffer_names = running_stat_risk
    warnings.warn(
        "run() detected training-mode BatchNorm running-stat buffers "
        f"{', '.join(buffer_names)} on module {module_name!r}; re-executing the live model can "
        "mutate them. Use eval() for immutable feature extraction or clone the model explicitly "
        "when isolated training-mode state is required.",
        UserWarning,
        # 4: this helper, run(), the thin Trace.run shell, then the caller's site.
        stacklevel=4,
    )
    mark_stateful_run_warning_emitted(trace)


def _warn_pending_value_edits_on_new_input_run(
    trace: Any, *, staged_spec_applies: bool = False
) -> None:
    """Disclose that value-edits on this trace do not apply to a new-input run.

    There are two legitimate intervention paths: (1) edit a SAVED value and
    push the effect downstream on the captured DAG (``do()`` or direct
    writes), and (2) intervene on a FRESH execution (``engine="rerun"``
    with a model and input, or a new capture with ``intervene=``). A
    ``run(inputs=...)`` on a trace carrying path-1 edits is a fresh execution:
    the edits say nothing about the new inputs, so the run is coherent but the
    edits have NO effect on it. That combination tripped a real user flow, so
    it is disclosed plainly instead of silently returning an un-edited
    verified run.

    Parameters
    ----------
    trace:
        Live Trace about to be re-executed on new inputs.
    staged_spec_applies:
        True when this run is the live provider (fast or not) applying the
        trace's staged intervention spec. The capture-time or rerun firing
        audit and the ``RERUN_PROPAGATED`` state are then that intervention's
        own trail, not an inert value edit, so only ``do()`` propagation and
        direct writes disclose.
    """

    state = getattr(trace, "state", None)
    audit = getattr(trace, "intervention_audit", None) or ()
    if staged_spec_applies:
        has_value_edits = (
            state in {TraceState.REPLAY_PROPAGATED, TraceState.DIRECT_WRITE_DIRTY}
            or any(isinstance(row, dict) and row.get("door") == "do" for row in audit)
            or bool(trace._has_direct_writes)
        )
    else:
        has_value_edits = (
            state
            in {
                TraceState.REPLAY_PROPAGATED,
                TraceState.RERUN_PROPAGATED,
                TraceState.DIRECT_WRITE_DIRTY,
            }
            or bool(audit)
            or bool(trace._has_direct_writes)
        )
    if not has_value_edits:
        return
    warnings.warn(
        "run(inputs=...) is a FRESH execution of the live model, but this trace "
        "carries value-edits (do()/push/direct writes) applied to its SAVED "
        "values -- those edits do NOT apply to a new-input run and its result "
        "reflects the un-edited model. To intervene on new inputs, rerun with "
        "hooks (do(..., engine='rerun', model=..., inputs=...)) or capture the new "
        "input with intervene=.",
        PendingValueEditsWarning,
        # 4: this helper, run(), the thin Trace.run shell, then the caller's site.
        stacklevel=4,
    )


def run(  # noqa: PLR0913  (the thin Trace.run shell keeps this documented call shape; the fold removes the duplicate site)
    trace: Trace,
    model: Any = None,
    x: Any = None,
    *,
    inputs: Any | MissingType = MISSING,
    seed: int | None = None,
    fast: bool = False,
    carry_state: bool = False,
    until: Any = None,
    save: Any = None,
    on_divergence: Literal["raise", "return"] = "raise",
    chunk_paths: Any | None = None,
    replay: ReplayOptions | None = None,
    input_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    on_zero_fire: str | MissingType = MISSING,
) -> Trace | RunResult:
    """Body of ``Trace.run``; the method carries the documented contract."""

    zero_fire_policy = None if on_zero_fire is MISSING else check_zero_fire_policy(on_zero_fire)
    if seed is not None:
        # r77 nit + r79 hardening: validate ``seed`` at the run door so junk
        # raises the typed precondition lane instead of escaping as torch's
        # raw ``RuntimeError``. r79 extends the r77 non-int check to the two
        # escapes r78 found: ``bool`` (an int subclass that
        # ``Generator.manual_seed`` rejects) and an int outside torch's
        # accepted long range (pybind overflow). The failed call is
        # transactional either way (global torch RNG untouched).
        validate_run_seed(seed)
    # R06: the settled-outcome gate is the FIRST authority on run(). The
    # outcome sidecar lives in ``__dict__`` and survives cleanup()-husking,
    # while the ``trace._runnable`` read below trips the husk
    # ``TraceCleanedUpError`` (and the analysis-load arm its capability
    # refusal) BEFORE any N-gate could evaluate -- so callers branching on
    # ``fields["code"] == "N3"`` (the documented contract) missed both husk
    # classes. Pre-refuse only the statuses EVERY provider row refuses
    # (FAILED / ABORTED_NONFINITE / UNKNOWN); HALTED stays with the
    # provider-specific gates because the loaded-sparse row allows it.
    settled_outcome = outcome_for(trace)
    if settled_outcome is None:
        # A cleanup()-husked trace loses the stamped sidecar; the outcome
        # property derives the honest UNKNOWN the structural lattice
        # mandates for it.
        settled_outcome = getattr(trace, "outcome", None)
    if settled_outcome is not None and settled_outcome.status in (
        CaptureStatus.FAILED,
        CaptureStatus.ABORTED_NONFINITE,
        CaptureStatus.UNKNOWN,
    ):
        require_capture_capability(trace, "live_replay")
        require_structure_only_capability(trace, "live_replay")
        _refuse_stats_capture(trace, "live_replay")
    # Episode join gate (lane F40c), AFTER the R06 outcome authority:
    # whole-episode replay is an episode-dependent claim across every
    # join, so a MEASURED broken join (exogenous or declared crossing)
    # refuses typed. Narrow by design: unmeasured/legacy episode
    # artifacts keep their shipped replay behavior.
    refuse_broken_join_claim(trace)
    # Coupled-episode replay gate (lane F42), beside the join gate: a
    # COUPLED product's ledger attests one PERTURBED execution, and no
    # run provider re-arms the capture-time intervention, so no engine
    # can derive a fresh ledger -- the verdict's refusal arm. Uncoupled
    # episode products keep their shipped replay behavior.
    refuse_coupled_replay(trace)
    refuse_streamed_trace_run(trace)  # IO-8: a to_disk trace's payloads are its bundle
    refuse_truncated_rerun(trace)
    readiness = trace._runnable.readiness
    loaded_provider = getattr(readiness, "provider", None)
    use_unified_provider = inputs is not MISSING or (
        not isinstance(model, nn.Module)
        and loaded_provider
        in {
            RunProvider.LOADED_SPARSE,
            RunProvider.LOADED_ANALYSIS,
        }
    )
    if use_unified_provider:
        # C03 honesty gate, narrowed to the loaded providers: a loaded artifact
        # has no model to install the staged intervention spec on, and silently
        # returning an un-intervened run was a measured silent wrong answer. The
        # live provider (fast or not) applies the spec. D1 carve-out: sticky hooks
        # the selection-do() plan attached are engine-owned residue of an edit
        # already applied to the saved values; their inertness on a new-input run
        # is the PendingValueEditsWarning disclosure below, never a refusal.
        staged_user_hooks, staged_value_specs = _staged_user_entries(trace)
        loaded_run = loaded_provider in {RunProvider.LOADED_SPARSE, RunProvider.LOADED_ANALYSIS}
        if (staged_user_hooks or staged_value_specs) and loaded_run:
            raise EngineDispatchError(
                "run(inputs=...) on a loaded artifact cannot apply this trace's "
                f"staged intervention spec ({len(staged_user_hooks)} sticky hook(s), "
                f"{len(staged_value_specs)} value replacement(s) staged); running "
                "would silently return an un-intervened result. Run the live trace "
                "(trace.run(inputs=...) applies the spec there) or capture fresh with "
                "tl.trace(model, x, intervene=...); to run without it, detach the "
                "staged spec first (trace.clear_hooks() or detach_hooks(site=/handle=)).",
                code="run_staged_spec_unapplied",
                remedy="run the live trace or tl.trace(..., intervene=...), or detach "
                "the staged hooks before run(inputs=...) on the loaded artifact",
            )
        if inputs is not MISSING:
            if model is not None or x is not None:
                raise KeywordConflictError(
                    "Pass inputs= without the legacy model/x arguments",
                    code="run_legacy_arguments_conflict",
                    remedy="pass only inputs= on the unified run surface",
                )
            run_inputs = inputs
        else:
            if x is not None:
                raise KeywordConflictError(
                    "Loaded sparse run accepts one input tree",
                    code="run_legacy_arguments_conflict",
                    remedy="pass one input tree, preferably via inputs=",
                )
            run_inputs = model
        if chunk_paths is not None or replay is not None:
            raise KeywordConflictError(
                "Sparse/unified run does not accept legacy rerun options",
                code="run_legacy_options_conflict",
                remedy="drop chunk_paths/replay from the unified run call",
            )
        if fast and DivergencePolicy(on_divergence) is not DivergencePolicy.RAISE:
            raise InvalidArgumentError(
                "fast=True always fails closed and requires on_divergence='raise'",
                code="run_fast_divergence_policy_invalid",
                remedy="use on_divergence='raise' with fast=True, or drop fast=",
                argument="on_divergence",
            )
        if fast and carry_state:
            raise InvalidArgumentError(
                "carry_state=True cannot combine with fast=True: fast mode's "
                "cached-oracle contract already forbids declared-state mutation",
                code="run_fast_carry_state_unsupported",
                remedy="drop carry_state= (fast mode never restores state it "
                "forbids mutating) or drop fast=",
                argument="carry_state",
            )
        if (
            fast
            and until is not None
            and loaded_provider in {RunProvider.LOADED_SPARSE, RunProvider.LOADED_ANALYSIS}
        ):
            # The live fast tier stops its native forward at a saved
            # module; a loaded artifact's fast tier compiles the
            # full recorded path.
            raise InvalidArgumentError(
                "until= combines with fast=True only on a live trace; a loaded "
                "artifact's fast tier compiles the full recorded path",
                code="run_fast_until_unsupported",
                remedy="drop until= or drop fast=, or run the live trace",
                argument="until",
            )
        if fast and save is not None:
            raise RunCapabilityUnavailableError(
                "run-time save= does not compose with fast=True yet: the fast "
                "tier's hook site-set is scoped by the ORIGINAL capture save= "
                "selection (an explicit matrix entry is required; silence "
                "would be undefined behavior)",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_run_save_pending",
                remedy="drop save= with fast=True, or re-capture with the desired save= selection",
            )
        run_save_labels: Any = None
        if save is not None:
            run_save_labels = _resolve_run_save_labels(trace, save, until)
        if carry_state and loaded_provider in {
            RunProvider.LOADED_SPARSE,
            RunProvider.LOADED_ANALYSIS,
        }:
            raise InvalidArgumentError(
                "carry_state=True requires a live model: the loaded provider "
                "mutates STAGED CLONES, never a live model, so there is no "
                "model for state to carry into (a silent no-op would "
                "misreport what persisted)",
                code="run_carry_state_requires_live_model",
                remedy="drop carry_state= on loaded traces, or run the live model",
                argument="carry_state",
            )
        if loaded_provider is RunProvider.LOADED_SPARSE:
            # N3 (loaded-sparse split): HALTED stays ALLOWED here -- the
            # loaded provider executes exactly the recorded taken-path
            # prefix DAG under pause_logging(), so the live-replay failure
            # mode cannot occur; failed/unproven captures still refuse.
            require_capture_capability(trace, "loaded_sparse_run")
            require_structure_only_capability(trace, "live_replay")
            _refuse_stats_capture(trace, "live_replay")
            if fast:
                return run_fast_loaded_trace(trace, run_inputs, seed=seed)
            return _apply_run_save_selection(
                run_loaded_sparse_trace(
                    trace,
                    run_inputs,
                    seed=seed,
                    on_divergence=DivergencePolicy(on_divergence),
                    until=until,
                ),
                run_save_labels,
            )
        if loaded_provider is RunProvider.LOADED_ANALYSIS:
            # R06: a HALTED analysis load refuses N5 first -- the generic
            # analysis refusal's remedy ("save a runnable artifact") is a
            # dead end N4 forbids for halted captures.
            require_capture_capability(trace, "live_replay")
            require_structure_only_capability(trace, "live_replay")
            _refuse_stats_capture(trace, "live_replay")
            raise_analysis_run_unavailable(trace)
        # N3/N5 (live provider): the live run -- fast=True included --
        # re-drives the full native forward, which halted/failed/unproven
        # captures cannot honor.
        require_capture_capability(trace, "live_replay")
        require_structure_only_capability(trace, "live_replay")
        _refuse_stats_capture(trace, "live_replay")
        _refuse_state_compromised_live_run(trace)
        try:
            staged_spec = trace._intervention_spec
        except AttributeError:
            staged_spec = None
        # The live provider applies the staged spec (fast or not); a previous
        # live run's own RERUN_PROPAGATED trail is a fresh execution, not a
        # value edit (repeated training steps through run(inputs=...)).
        _warn_pending_value_edits_on_new_input_run(
            trace,
            staged_spec_applies=bool(
                getattr(staged_spec, "hook_specs", ())
                or getattr(staged_spec, "target_value_specs", ())
            )
            or (getattr(trace, "last_run", None) or {}).get("engine") in {"guarded_fast", "live"},
        )
        source_ref = trace._source_model_ref
        live_model = source_ref() if source_ref is not None else None
        if live_model is not None:
            _warn_stateful_live_run_once(trace, live_model)

        if fast:
            from .._fast_live_steer import refuse_unfired_fast_run

            with zero_fire_policy_scope(zero_fire_policy):
                fast_result = run_fast_live_trace(trace, run_inputs, seed=seed, until=until)
                refuse_unfired_fast_run(fast_result.trace)
            return fast_result

        # The staged spec (sticky hooks, value replacements) applies on the live
        # forward; the fork carries the fire receipts and zero-fire facts. The
        # planner is reached through importlib: it is a leaf over the rerun engine.
        run_staged = importlib.import_module("torchlens.intervention._run_staged")
        live_options = _LiveRunOptions(
            carry_state=carry_state,
            until=until,
            staged_plan=run_staged.build_staged_run_plan(trace),
        )
        with zero_fire_policy_scope(zero_fire_policy):
            live_result = run_live_trace(
                trace, run_inputs, seed=seed, on_divergence=on_divergence, options=live_options
            )
        return _apply_run_save_selection(live_result, run_save_labels)

    if fast:
        raise KeywordConflictError(
            "fast=True is available only with the unified inputs= surface",
            code="run_fast_requires_inputs",
            remedy="call trace.run(inputs=..., fast=True) instead of the legacy surface",
            argument="fast",
        )
    if carry_state:
        raise KeywordConflictError(
            "carry_state= is a unified-run option and cannot mix with the "
            "legacy run surface (which never snapshot-restores state)",
            code="run_legacy_options_conflict",
            remedy="call trace.run(inputs=..., carry_state=True) instead of the legacy surface",
            argument="carry_state",
        )
    if until is not None:
        raise KeywordConflictError(
            "until= is a unified-run option and cannot mix with the legacy run surface",
            code="run_legacy_options_conflict",
            remedy="call trace.run(inputs=..., until=...) instead of the legacy surface",
            argument="until",
        )
    if save is not None:
        raise KeywordConflictError(
            "save= is a unified-run option and cannot mix with the legacy run surface",
            code="run_legacy_options_conflict",
            remedy="call trace.run(inputs=..., save=...) instead of the legacy surface",
            argument="save",
        )
    _refuse_state_compromised_live_run(trace)
    refuse_unpersisted_intervention_rerun(trace, loaded_provider)

    # N3/N5 (legacy live rerun surface): same live-provider rule.
    require_capture_capability(trace, "live_replay")
    require_structure_only_capability(trace, "live_replay")
    _refuse_stats_capture(trace, "live_replay")

    run_model: nn.Module | None
    if isinstance(model, nn.Module):
        run_model = model
        user_input = x
    else:
        source_ref = trace._source_model_ref
        user_input = model
        if x is not None:
            raise KeywordConflictError(
                "Pass either run(model, x) or run(new_user_input), not both",
                code="run_legacy_arguments_conflict",
                remedy="pass run(model, x) or run(new_user_input), never both forms",
            )
        transformed_input = _apply_rerun_transform(trace, user_input, transform=input_transform)
        run_model = source_ref() if source_ref is not None else None
        if run_model is None:
            raise RecordBindingError(
                "This Trace does not retain a live model reference",
                code="run_source_model_collected",
                remedy="pass the model explicitly as trace.run(model, input)",
            )
    replay_options = merge_replay_options(replay=replay)
    if isinstance(model, nn.Module):
        transformed_input = _apply_rerun_transform(trace, user_input, transform=input_transform)
    _warn_stateful_live_run_once(trace, run_model)

    resolved_output_transform = _resolve_rerun_output_transform(trace, output_transform)
    # Guarded fast engine first for a steered rerun: a native forward with
    # the staged spec applied at real module boundaries, saving the saved
    # sites and the boundary ops, guarded by the sealed call fingerprint.
    # Any typed refusal (ineligible save scope or target, a real
    # divergence) falls back to the capture rerun below, with the code in
    # ``last_run``. A trace with nothing staged keeps the capture rerun.
    from .._fast_live_steer import stages_user_entries

    fast_refused: str | None = None
    with zero_fire_policy_scope(zero_fire_policy):
        if (
            chunk_paths is None
            and not replay_options.append
            and replay_options.chunk_size is None
            and stages_user_entries(trace)
        ):
            from .._fast_live_steer import refuse_unfired_fast_run

            fast_done, fast_refused = try_guarded_fast_rerun(
                trace, run_model, transformed_input, output_transform=resolved_output_transform
            )
            if fast_done:
                trace.input = user_input
                refuse_unfired_fast_run(trace)
                return trace
        result = _rerun_impl(
            trace,
            run_model,
            transformed_input,
            replay=replay_options,
            chunk_paths=chunk_paths,
            output_transform=resolved_output_transform,
            fast_refused=fast_refused,
        )
    # Atomic swap rebuilds Trace state; restore raw_input to the new
    # user-supplied value so visualization / save-load report the
    # current input rather than the prior trace's.
    result.input = user_input
    return result


def _apply_rerun_transform(
    trace: Trace,
    user_input: Any,
    *,
    transform: Callable[[Any], Any] | bool | object,
) -> Any:
    """Body of ``Trace._apply_rerun_transform``; the method carries the documented contract."""
    try:
        stored_transform = trace._transform
    except AttributeError:
        stored_transform = None
    if transform is _USE_STORED_TRANSFORM and stored_transform is not None:
        return stored_transform(user_input)
    if transform is False:
        return user_input
    if callable(transform):
        return transform(user_input)
    return user_input


def _resolve_rerun_output_transform(
    trace: Trace,
    output_transform: Callable[[Any], Any] | bool | object,
) -> Callable[[Any], Any] | None:
    """Body of ``Trace._resolve_rerun_output_transform``; the method carries the documented contract."""
    try:
        stored_transform = trace._output_transform
    except AttributeError:
        stored_transform = None
    if output_transform is _USE_STORED_TRANSFORM:
        return stored_transform
    if output_transform is False:
        return None
    if callable(output_transform):
        return output_transform
    return None


def disarm(trace: Trace) -> None:
    """Body of ``trace.autograd.disarm``; the method carries the documented contract."""
    _disarm_triggers_impl(trace)
