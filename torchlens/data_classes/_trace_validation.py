"""Trace validation and replay mixin."""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Union, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from ..validation.status import ValidationReplayStatus
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._deprecations import MISSING, MissingType
from .._errors import InvalidArgumentError, KeywordConflictError, RecordBindingError
from ..options import ReplayOptions, merge_replay_options
from ..runnable import DivergencePolicy, RunProvider, RunResult
from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    _scrub_per_op_equivalence_lists,
    _substitute_conditional_branch_edges,
    cleanup,
)
from .op import Op


def _refuse_loaded_backward_capture(trace: "Trace", entry_point: str) -> None:
    """Refuse live backward capture on a bundle-loaded analysis trace.

    WT1 A-IV item 21 (lane A08): a loaded trace carries no live autograd
    handles tied to its recorded forward, so ``log_backward`` used to hook
    whatever FOREIGN graph the caller's loss came from, half-mutate the trace
    (backward passes and grad_fns from an unrelated forward), and die with an
    untyped ``AttributeError`` -- after which ``draw_backward`` silently
    rendered that wrong graph. Refuse typed BEFORE any mutation.
    """

    if not getattr(trace, "_loaded_from_bundle", False):
        return
    from ..errors import RunCapabilityUnavailableError
    from ..runnable import RunnableErrorCode

    raise RunCapabilityUnavailableError(
        f"{entry_point}() is unavailable on a bundle-loaded Trace: the loaded "
        "artifact carries no live autograd graph tied to its recorded "
        "forward, so a backward here would capture whatever unrelated graph "
        "the loss came from and record a wrong backward pass. Re-capture "
        "live (tl.trace(model, x, capture=CaptureOptions(backward_ready="
        "True))) and call log_backward on that Trace, or inspect the "
        "backward metadata this artifact already persisted.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="loaded_backward_capture",
    )


def _materialize_layer_mirrors_for_removed(
    trace: "Trace",
    removed_entries: Iterable[Op],
) -> None:
    """Materialize Layer mirror fields whose representative op is being removed.

    A Layer's M8 mirror descriptors read through to its first-pass op; husking
    that op would otherwise change what a still-held Layer reads. Materializing
    first preserves the dict-era post-removal surface (the copies existed at
    this point in the dict era). Layers whose representative op survives keep
    mirroring — their reads are unaffected by the removal.
    """

    layer_logs = trace.__dict__.get("layer_logs")
    if not layer_logs:
        return
    from .layer import _layer_rep_op, materialize_layer_mirrors

    removed_ids = {id(entry) for entry in removed_entries}
    for layer_log in layer_logs.values():
        rep = _layer_rep_op(layer_log)
        if rep is not None and id(rep) in removed_ids:
            materialize_layer_mirrors(layer_log)


_USE_STORED_TRANSFORM = object()
_JAX_VALIDATION_REPLAY_BACKEND = "jax"
_MLX_VALIDATION_REPLAY_BACKEND = "mlx"
_TINYGRAD_VALIDATION_REPLAY_BACKEND = "tinygrad"


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
    from ..capture.projectors import RefreshProjector

    for layer in result.trace.layer_list:
        if layer.layer_type in ("input", "output"):
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

    runnable_state = getattr(trace, "_runnable", None)
    compromised = getattr(runnable_state, "state_compromised", None)
    if compromised is None:
        return
    from ..errors import StateBindingError

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

    if trace.__dict__.get("_stateful_run_warning_emitted", False):
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
    import warnings

    module_name, buffer_names = running_stat_risk
    warnings.warn(
        "run() detected training-mode BatchNorm running-stat buffers "
        f"{', '.join(buffer_names)} on module {module_name!r}; re-executing the live model can "
        "mutate them. Use eval() for immutable feature extraction or clone the model explicitly "
        "when isolated training-mode state is required.",
        UserWarning,
        stacklevel=3,
    )
    trace.__dict__["_stateful_run_warning_emitted"] = True


def _warn_pending_value_edits_on_new_input_run(trace: Any) -> None:
    """Disclose that value-edits on this trace do not apply to a new-input run.

    There are two legitimate intervention paths: (1) edit a SAVED value and
    push the effect downstream on the captured DAG (``do()``/``push_from``/
    direct writes), and (2) intervene on a FRESH execution (``engine="rerun"``
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
    """

    from .._trace_state import TraceState

    state = getattr(trace, "state", None)
    has_value_edits = (
        state
        in {
            TraceState.REPLAY_PROPAGATED,
            TraceState.RERUN_PROPAGATED,
            TraceState.DIRECT_WRITE_DIRTY,
        }
        or bool(getattr(trace, "intervention_audit", None))
        or bool(trace._has_direct_writes)
    )
    if not has_value_edits:
        return
    import warnings

    from ..intervention.errors import PendingValueEditsWarning

    warnings.warn(
        "run(inputs=...) is a FRESH execution of the live model, but this trace "
        "carries value-edits (do()/push/direct writes) applied to its SAVED "
        "values -- those edits do NOT apply to a new-input run and its result "
        "reflects the un-edited model. To intervene on new inputs, rerun with "
        "hooks (do(..., engine='rerun', model=..., x=...)) or capture the new "
        "input with intervene=.",
        PendingValueEditsWarning,
        stacklevel=3,
    )


def _loaded_non_torch_validation_replay_unavailable(trace: Any) -> bool:
    """Return whether loaded non-torch replay validation cannot run.

    Parameters
    ----------
    trace:
        Trace-like object being checked.

    Returns
    -------
    bool
        True for loaded non-torch traces whose backend runtime replay
        capture lists were stripped by portable save.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    backend = str(getattr(trace, "backend", "torch"))
    if backend == _MLX_VALIDATION_REPLAY_BACKEND:
        return True
    if backend == _JAX_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "jax_equation_captures", ()))
    if backend == _TINYGRAD_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "tinygrad_uop_captures", ()))
    return False


class TraceValidationMixin(_TraceMixinBase):
    """``Trace`` validation surface: forward replay, backward checks, and re-capture."""

    def save_new_outs(
        self: "Trace",
        model: torch.nn.Module,
        input_args: torch.Tensor | list[Any],
        input_kwargs: dict[Any, Any] | None = None,
        layers_to_save: str | list[str] = "all",
        grad_layers_to_save: str | list[str] | None = "all",
        random_seed: int | None = None,
        backward_ready: bool | None = None,
        _run_until_plan: Any | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only outs.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed, backward_ready:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.save_new_outs`.
        """
        from .._capture_state_helpers import unwrap_compiled_model
        from ..capture.outcome import require_capture_capability
        from ..capture.trace import save_new_outs as _impl

        # N3/N5: a live refresh re-drives the FULL native forward against the
        # recorded graph, which a halted/failed/unproven capture cannot honor.
        require_capture_capability(self, "live_replay")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "live_replay")
        model = unwrap_compiled_model(model)

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            backward_ready=backward_ready,
            _run_until_plan=_run_until_plan,
        )

    def validate_forward_pass(
        self: "Trace",
        ground_truth_output_tensors: list[torch.Tensor] | torch.Tensor,
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationReplayStatus"]:
        """Validate saved outs against ground-truth model outputs.

        Parameters
        ----------
        ground_truth_output_tensors:
            Ground-truth model outputs. A bare tensor -- what
            ``model(x)`` naturally produces for a single-output model -- is
            normalized to ``[tensor]``: iterating it directly counted the
            tensor's ROWS as expected outputs, so a byte-correct capture
            reported a validation FAILURE (``"1 logged vs <batch> expected"``)
            for a caller-arity slip (r7 b1-opus R08-3). A tripwire that fails
            on correct captures trains users to ignore it.
        verbose, validate_metadata:
            Forwarded unchanged to
            :func:`torchlens.validation.core.validate_saved_outs`.

        Returns
        -------
        bool or ValidationReplayStatus
            ``True`` if validation succeeds. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status instead of a pass/fail bool.
        """
        from ..backends import get_backend_spec
        from ..capture.outcome import require_capture_capability
        from ..runnable import refuse_poisoned_trace

        if isinstance(ground_truth_output_tensors, torch.Tensor):
            ground_truth_output_tensors = [ground_truth_output_tensors]

        refuse_poisoned_trace(self, "validation")
        # N2: refusing ENTRY for failed/unproven captures is not a check
        # exemption -- the tripwire bodies stay byte-untouched, the halted
        # exemption neither widens nor narrows, and legacy UNATTESTED
        # artifacts deliberately keep entry OPEN (the tripwire stays armed).
        require_capture_capability(self, "validation_entry")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "validation_entry")
        status = self.validation_replay_status
        if bool(getattr(self, "_loaded_from_bundle", False)) and not status.available:
            setattr(self, "_validation_replay_status", status)
            return status
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        validation_result = spec.validate_trace(
            self,
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )
        if spec.name == "torch":
            from ..validation.status import ValidationReplayStatus

            if isinstance(validation_result, ValidationReplayStatus):
                setattr(self, "_validation_replay_status", validation_result)
                if validation_result.state in {"passed", "failed"}:
                    return validation_result.passed
        return validation_result

    @property
    def validation_replay_status(self: "Trace") -> "ValidationReplayStatus":
        """Return replay-validation availability or last completed result.

        Returns
        -------
        ValidationReplayStatus
            Status object distinguishing live replay validation from loaded
            traces whose runtime replay captures were stripped during save.
        """

        from ..backends import get_backend_spec
        from ..validation.status import ValidationReplayStatus

        cached_status = getattr(self, "_validation_replay_status", None)
        if isinstance(cached_status, ValidationReplayStatus):
            return cached_status
        backend = str(getattr(self, "backend", "torch"))
        if _loaded_non_torch_validation_replay_unavailable(self):
            return ValidationReplayStatus.unavailable_loaded_runtime_stripped(
                backend=backend,
                payload_load_status=getattr(self, "payload_load_status", None),
            )
        spec = get_backend_spec(backend)
        if not spec.capabilities.validation_replay:
            return ValidationReplayStatus.unavailable_unsupported(backend=backend)
        return ValidationReplayStatus.available_live(backend=backend)

    def push(
        self: "Trace",
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Push the edit downstream through the recorded graph (DAG replay).

        Parameters
        ----------
        replay:
            Grouped replay options (``ReplayOptions``: ``strict``, ``hooks``,
            ``differentiable``).

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "live_replay")
        replay_options = merge_replay_options(replay=replay)

        from ..intervention.replay import push as _impl

        return _impl(self, replay=replay_options)

    def push_from(
        self: "Trace",
        site: Any,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Push downstream from a pre-mutated site.

        Parameters
        ----------
        site:
            Layer pass or selector resolving to one origin. The origin's
            current out is preserved and used as the override.
        replay:
            Grouped replay options (``ReplayOptions``).

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "live_replay")
        replay_options = merge_replay_options(replay=replay)

        from ..intervention.replay import push_from as _impl

        return _impl(self, site, replay=replay_options)

    def run(
        self: "Trace",
        model: Any = None,
        x: Any = None,
        *,
        inputs: Any | MissingType = MISSING,
        seed: int | None = None,
        fast: bool = False,
        carry_state: bool = False,
        until: Any = None,
        save: Any = None,
        on_divergence: DivergencePolicy = DivergencePolicy.RAISE,
        chunk_paths: Any | None = None,
        replay: ReplayOptions | None = None,
        transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    ) -> "Trace | RunResult":
        """Execute this Trace through its live or loaded provider.

        Parameters
        ----------
        model:
            Model to execute through TorchLens decorated wrappers. When omitted,
            the live model captured by this ``Trace`` is reused if still
            available -- the trace holds it weakly, so this requires the
            caller to have kept a strong reference (see Raises).
        x:
            Forward input. If ``model`` is omitted, the first positional argument
            is treated as the new user input.
        inputs:
            Unified provider input tree. Supplying this keyword returns a
            transactional :class:`RunResult` and leaves this Trace unchanged.
        seed:
            Optional deterministic live refresh, random-state, and runtime RNG seed.
        fast:
            Explicit stateful static-loop mode. The live provider runs native ``forward``
            with targeted module/function collection and a per-call path/shape guard. The
            loaded provider performs one ordinary verified run, then reuses staged state
            and compiled argument binders. Unlike the default transactional provider, later
            fast iterations reuse one result Trace in place.
        carry_state:
            PROVISIONAL spelling (documented-unstable). Default ``False``: the
            live provider snapshot-restores the model's declared state (named
            parameters, registered buffers, alias topology preserved) around
            the run, so repeated ``run()`` calls leave the model bit-identical.
            ``True`` skips the restore: declared-state mutations persist on the
            live model (episode/rollout state accumulation). The report
            discloses the choice (``report.state_carried``); verification is
            untouched -- the NEXT run from mutated state faces every gate as
            usual. Refuses typed with ``fast=True`` and on loaded providers
            (staged clones have no live model for state to carry into).
        on_divergence:
            Strict divergence behavior or the sole poison-return opt-in.
        chunk_paths:
            Optional explicit tensor leaf paths to split.
        strict:
            Whether graph-shape divergence should raise instead of warn.
        transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit input
            transform callable for this run.
        output_transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit output
            transform callable for this run.

        Returns
        -------
        Trace or RunResult
            A unified transactional result for ``inputs=`` and loaded sparse
            providers. Legacy ``run(model, x)`` intervention reruns retain their
            compatibility return until that surface is migrated.

        Raises
        ------
        RunCapabilityUnavailableError
            When no provider is available: an analysis-only loaded Trace, or a
            live Trace whose source model has been garbage-collected (see
            Notes).

        Notes
        -----
        A live-provider run re-executes the retained model object. The Trace
        holds that model by WEAK reference (``tl.release_model()`` doctrine:
        capturing never extends the model's lifetime), so live-provider
        availability depends on the CALLER still holding the model. If the
        caller's last strong reference is dropped -- including the inline
        ``tl.trace(Model(), x)`` idiom, where the trace is the only holder --
        any garbage-collection pass makes a later ``run()`` refuse typed with
        ``run_capability_unavailable``. Keep a reference to the model (or save
        and load a runnable artifact) when ``run()`` must stay available.

        TorchLens warns once when it detects training-mode BatchNorm
        running-stat buffers, which the forward pass can mutate. Custom mutable
        attributes such as caches and user counters cannot be detected
        generically. Clone the model explicitly when isolated state is
        required. Live-state mutation can also change the captured graph and
        trigger the normal graph-change tripwire.
        """

        if seed is not None:
            # r77 nit + r79 hardening: validate ``seed`` at the run door so junk
            # raises the typed precondition lane instead of escaping as torch's
            # raw ``RuntimeError``. r79 extends the r77 non-int check to the two
            # escapes r78 found: ``bool`` (an int subclass that
            # ``Generator.manual_seed`` rejects) and an int outside torch's
            # accepted long range (pybind overflow). The failed call is
            # transactional either way (global torch RNG untouched).
            from .._runnable_state import validate_run_seed

            validate_run_seed(seed)
        # R06: the settled-outcome gate is the FIRST authority on run(). The
        # outcome sidecar lives in ``__dict__`` and survives cleanup()-husking,
        # while the ``self._runnable`` read below trips the husk
        # ``TraceCleanedUpError`` (and the analysis-load arm its capability
        # refusal) BEFORE any N-gate could evaluate -- so callers branching on
        # ``fields["code"] == "N3"`` (the documented contract) missed both husk
        # classes. Pre-refuse only the statuses EVERY provider row refuses
        # (FAILED / ABORTED_NONFINITE / UNKNOWN); HALTED stays with the
        # provider-specific gates because the loaded-sparse row allows it.
        from ..capture.outcome import CaptureStatus, outcome_for, require_capture_capability

        settled_outcome = outcome_for(self)
        if settled_outcome is None:
            # A cleanup()-husked trace loses the stamped sidecar; the outcome
            # property derives the honest UNKNOWN the structural lattice
            # mandates for it.
            settled_outcome = getattr(self, "outcome", None)
        if settled_outcome is not None and settled_outcome.status in (
            CaptureStatus.FAILED,
            CaptureStatus.ABORTED_NONFINITE,
            CaptureStatus.UNKNOWN,
        ):
            require_capture_capability(self, "live_replay")
            from ..capture.structure_only import require_structure_only_capability

            require_structure_only_capability(self, "live_replay")
        if self.__dict__.get("_run_truncation_skipped_raw_labels") is not None:
            # L4 3.3.2: a truncated result cannot be re-run (any provider). The
            # poison bit alone does NOT refuse a re-run -- the transaction's
            # inherited-status leg would silently settle UNVERIFIABLE instead,
            # which is exactly the silence the capability table forbids.
            from ..errors import RunCapabilityUnavailableError
            from ..runnable import RunnableErrorCode

            raise RunCapabilityUnavailableError(
                "This Trace is a TRUNCATED run product (until=): it covers only "
                "the executed prefix, so re-running it would replay a cut "
                "forward. Run the ORIGINAL source trace instead (with or "
                "without until=).",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="truncated_result_rerun",
            )
        readiness = self._runnable.readiness
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
            if fast and until is not None:
                raise InvalidArgumentError(
                    "until= cannot combine with fast=True this release (the fast "
                    "tier compiles the full recorded path)",
                    code="run_fast_until_unsupported",
                    remedy="drop until= or drop fast=",
                    argument="until",
                )
            if fast and save is not None:
                from ..errors import RunCapabilityUnavailableError
                from ..runnable import RunnableErrorCode

                raise RunCapabilityUnavailableError(
                    "run-time save= does not compose with fast=True yet: the fast "
                    "tier's hook site-set is scoped by the ORIGINAL capture save= "
                    "selection (an explicit matrix entry is required; silence "
                    "would be undefined behavior)",
                    code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                    detection_stage="fast_run_save_pending",
                    remedy="drop save= with fast=True, or re-capture with the "
                    "desired save= selection",
                )
            run_save_labels: Any = None
            if save is not None:
                from .._runnable_execution import _resolve_run_until_plan

                # Reuse the static-form resolver: same accepted family (labels,
                # module addresses, 'saved'), same typed refusals (pre-S4
                # predicate gate, junk forms, unknown-site lookup feedback).
                save_plan = _resolve_run_until_plan(self, save)
                run_save_labels = frozenset(save_plan.requested_layer_labels)
                if until is not None:
                    until_preview = _resolve_run_until_plan(self, until)
                    executed_raw = frozenset(until_preview.executed_raw_labels)
                    executed_final = {
                        layer.layer_label
                        for layer in self.layer_list
                        if layer._layer_label_raw in executed_raw
                    }
                    outside = sorted(run_save_labels - executed_final)
                    if outside:
                        raise InvalidArgumentError(
                            "run-time save= selection must lie inside the "
                            "until= executed window; out-of-window site(s): "
                            f"{', '.join(outside)}",
                            code="run_until_form_invalid",
                            remedy="widen until= to cover the save= sites, or "
                            "drop the out-of-window save= sites",
                            argument="save",
                        )
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
            from ..capture.outcome import require_capture_capability

            if loaded_provider is RunProvider.LOADED_SPARSE:
                # N3 (loaded-sparse split): HALTED stays ALLOWED here -- the
                # loaded provider executes exactly the recorded taken-path
                # prefix DAG under pause_logging(), so the live-replay failure
                # mode cannot occur; failed/unproven captures still refuse.
                require_capture_capability(self, "loaded_sparse_run")
                from ..capture.structure_only import require_structure_only_capability

                require_structure_only_capability(self, "live_replay")
                if fast:
                    from .._fast_run import run_fast_loaded_trace

                    return run_fast_loaded_trace(self, run_inputs, seed=seed)
                from .._runnable_execution import run_loaded_sparse_trace

                return _apply_run_save_selection(
                    run_loaded_sparse_trace(
                        self,
                        run_inputs,
                        seed=seed,
                        on_divergence=on_divergence,
                        until=until,
                    ),
                    run_save_labels,
                )
            if loaded_provider is RunProvider.LOADED_ANALYSIS:
                # R06: a HALTED analysis load refuses N5 first -- the generic
                # analysis refusal's remedy ("save a runnable artifact") is a
                # dead end N4 forbids for halted captures.
                require_capture_capability(self, "live_replay")
                from ..capture.structure_only import require_structure_only_capability

                require_structure_only_capability(self, "live_replay")
                from .._runnable_execution import raise_analysis_run_unavailable

                raise_analysis_run_unavailable(self)
            # N3/N5 (live provider): the live run -- fast=True included --
            # re-drives the full native forward, which halted/failed/unproven
            # captures cannot honor.
            require_capture_capability(self, "live_replay")
            from ..capture.structure_only import require_structure_only_capability

            require_structure_only_capability(self, "live_replay")
            from .._runnable_execution import _LiveRunOptions, run_live_trace

            _refuse_state_compromised_live_run(self)
            _warn_pending_value_edits_on_new_input_run(self)
            source_ref = getattr(self, "_source_model_ref", None)
            live_model = source_ref() if source_ref is not None else None
            if live_model is not None:
                _warn_stateful_live_run_once(self, live_model)

            if fast:
                from .._fast_run import run_fast_live_trace

                return run_fast_live_trace(self, run_inputs, seed=seed)

            return _apply_run_save_selection(
                run_live_trace(
                    self,
                    run_inputs,
                    seed=seed,
                    on_divergence=on_divergence,
                    options=_LiveRunOptions(carry_state=carry_state, until=until),
                ),
                run_save_labels,
            )

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
        _refuse_state_compromised_live_run(self)

        # N3/N5 (legacy live rerun surface): same live-provider rule.
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "live_replay")

        run_model: nn.Module | None
        if isinstance(model, nn.Module):
            run_model = model
            user_input = x
        else:
            source_ref = getattr(self, "_source_model_ref", None)
            user_input = model
            if x is not None:
                raise KeywordConflictError(
                    "Pass either run(model, x) or run(new_user_input), not both",
                    code="run_legacy_arguments_conflict",
                    remedy="pass run(model, x) or run(new_user_input), never both forms",
                )
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)
            run_model = source_ref() if source_ref is not None else None
            if run_model is None:
                raise RecordBindingError(
                    "This Trace does not retain a live model reference",
                    code="run_source_model_collected",
                    remedy="pass the model explicitly as trace.run(model, input)",
                )
        replay_options = merge_replay_options(replay=replay)
        if isinstance(model, nn.Module):
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)
        _warn_stateful_live_run_once(self, run_model)

        from ..intervention.rerun import run as _impl

        resolved_output_transform = self._resolve_rerun_output_transform(output_transform)
        result = _impl(
            self,
            run_model,
            transformed_input,
            replay=replay_options,
            chunk_paths=chunk_paths,
            output_transform=resolved_output_transform,
        )
        # Atomic swap rebuilds Trace state; restore raw_input to the new
        # user-supplied value so visualization / save-load report the
        # current input rather than the prior trace's.
        result.raw_input = user_input
        return result

    def _apply_rerun_transform(
        self: "Trace",
        user_input: Any,
        *,
        transform: Callable[[Any], Any] | bool | object,
    ) -> Any:
        """Apply the stored or explicit input transform for ``rerun``.

        Parameters
        ----------
        user_input:
            New user input supplied to ``rerun``.
        transform:
            Sentinel to reuse the stored transform, ``False`` to bypass, or an
            explicit callable to use for this rerun.

        Returns
        -------
        Any
            Model-ready rerun input.
        """

        stored_transform = getattr(self, "_transform", None)
        if transform is _USE_STORED_TRANSFORM and stored_transform is not None:
            return stored_transform(user_input)
        if transform is False:
            return user_input
        if callable(transform):
            return transform(user_input)
        return user_input

    def _resolve_rerun_output_transform(
        self: "Trace",
        output_transform: Callable[[Any], Any] | bool | object,
    ) -> Callable[[Any], Any] | None:
        """Resolve the output transform callable for ``rerun``.

        Parameters
        ----------
        output_transform:
            Sentinel to reuse the stored output transform, ``False`` to bypass,
            or an explicit callable to use for this rerun.

        Returns
        -------
        Callable[[Any], Any] | None
            Output transform to apply to the fresh model output, or ``None``.
        """

        stored_transform = getattr(self, "_output_transform", None)
        if output_transform is _USE_STORED_TRANSFORM:
            return stored_transform
        if output_transform is False:
            return None
        if callable(output_transform):
            return output_transform
        return None

    def check_metadata_invariants(self: "Trace") -> bool:
        """Run metadata invariant checks on this completed model log.

        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
        # N2: refusing ENTRY for failed/unproven captures is not a check
        # exemption -- the tripwire bodies stay byte-untouched, and legacy
        # UNATTESTED artifacts deliberately keep entry OPEN.
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "validation_entry")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "validation_entry")
        from ..validation.invariants import check_metadata_invariants as _impl

        return _impl(self)

    def cleanup(self: "Trace") -> None:
        """Delete log data, break cycles, and free cached GPU memory.

        Returns
        -------
        None
            This method mutates the model log in place.
        """
        return cleanup(self)

    def release_param_refs(self: "Trace", *, allow_iter_rehydrate: bool = False) -> None:
        """Release live ``nn.Parameter`` references held by ParamLogs.

        Parameters
        ----------
        allow_iter_rehydrate:
            If ``True``, iterating ``param_logs`` may lazily restore live
            references from the source model. Public explicit releases leave
            this disabled.

        Returns
        -------
        None
            This method mutates ParamLogs in place.
        """
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = False
        for param_log in self.param_logs.values():
            param_log.release_param_ref()
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = allow_iter_rehydrate

    def _postprocess(
        self: "Trace",
        output_tensors: list[torch.Tensor],
        output_tensor_addresses: list[str],
    ) -> None:
        """Run postprocessing on a completed raw capture pass.

        Parameters
        ----------
        output_tensors:
            Output tensors returned by the model.
        output_tensor_addresses:
            Hierarchical addresses for those outputs.
        """
        from ..postprocess import postprocess as _impl

        self._postprocessing_active = True
        try:
            return _impl(
                self,
                output_tensors=output_tensors,
                output_tensor_addresses=output_tensor_addresses,
            )
        finally:
            self._postprocessing_active = False

    def _run_and_log_inputs_through_model(
        self: "Trace",
        model: torch.nn.Module,
        input_args: torch.Tensor | list[Any],
        input_kwargs: dict[Any, Any] | None = None,
        layers_to_save: str | list[str | int] | None = "all",
        grad_layers_to_save: str | list[str | int] | None = "all",
        random_seed: int | None = None,
        postprocess: bool = True,
        reservation_resume: object | None = None,
    ) -> Any:
        """Run a forward pass and capture it into this model log.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.run_and_log_inputs_through_model`.
        reservation_resume:
            Continuation token from the caller's own live ``capture_reservation``
            claim (the recorder pass); a bare nested entry without it refuses.
        """
        from ..capture.trace import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            postprocess=postprocess,
            reservation_resume=reservation_resume,
        )

    def log_backward(self: "Trace", loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` while capturing first-class backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This model log, for chaining.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec
        from ..capture.outcome import require_capture_capability

        # N3: the backward projection assumes a structurally complete captured
        # forward graph; HALTED stays allowed (the autograd graph of a halted
        # capture IS the captured prefix).
        require_capture_capability(self, "backward")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "backward_grads")
        _refuse_loaded_backward_capture(self, "log_backward")
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import log_backward as _impl

        return cast("Trace", _impl(self, loss, **backward_kwargs))

    def backward(self: "Trace", loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` and populate this Trace with backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This Trace, for chaining.
        """

        return self.log_backward(loss, **backward_kwargs)

    def recording_backward(self: "Trace") -> Any:
        """Return a context manager that captures user-managed backward calls.

        Returns
        -------
        Any
            Backward recording context manager.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "backward")
        from ..capture.structure_only import require_structure_only_capability

        require_structure_only_capability(self, "backward_grads")
        _refuse_loaded_backward_capture(self, "recording_backward")
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import recording_backward as _impl

        return _impl(self)

    def disarm_triggers(self: "Trace") -> None:
        """Detach this Trace from global autograd backward interception.

        Returns
        -------
        None
            Future plain ``loss.backward()`` or ``torch.autograd.*`` calls will
            not record into this Trace.
        """
        from ..backends.torch.backward import disarm_triggers as _impl

        _impl(self)

    def _remove_log_entry(
        self: "Trace",
        log_entry: Op,
        remove_references: bool = True,
        replacement_labels: dict[str, str] | None = None,
    ) -> None:
        """Remove a single layer-pass entry and scrub graph references.

        Parameters
        ----------
        log_entry:
            Entry to remove.
        remove_references:
            Whether to scrub all graph references to the removed entry.
        replacement_labels:
            Optional removed-label -> survivor substitutions for merge-style
            removals: conditional references repoint to the survivor instead
            of being dropped.
        """
        tensor_label = _label_for_reference_removal(log_entry, self._tracing_finished)
        _materialize_layer_mirrors_for_removed(self, (log_entry,))
        if remove_references:
            _remove_log_entry_references(self, tensor_label, replacement_labels)
        _clear_entry_attributes(log_entry)

    def _batch_remove_log_entries(
        self: "Trace",
        entries_to_remove: Iterable[Op],
        remove_references: bool = True,
        replacement_labels: dict[str, str] | None = None,
    ) -> None:
        """Remove multiple layer-pass entries using single-pass filtering.

        Parameters
        ----------
        entries_to_remove:
            Entries to remove.
        remove_references:
            Whether to scrub all graph references to the removed entries.
        replacement_labels:
            Optional removed-label -> survivor substitutions for merge-style
            removals: conditional references repoint to the survivor instead
            of being dropped.
        """
        entries_to_remove = list(entries_to_remove)
        removal_ids = {id(entry) for entry in entries_to_remove}
        surviving_entries = [entry for entry in self if id(entry) not in removal_ids]
        _materialize_layer_mirrors_for_removed(self, entries_to_remove)

        labels_to_remove = set()
        for entry in entries_to_remove:
            labels_to_remove.add(_label_for_reference_removal(entry, self._tracing_finished))

        if not remove_references:
            for entry in entries_to_remove:
                _clear_entry_attributes(entry)
            return

        _scrub_conditional_fields_after_removal(
            self, labels_to_remove, surviving_entries, replacement_labels
        )

        for field_name in _LIST_FIELDS_TO_CLEAN:
            collection = getattr(self, field_name)
            collection[:] = [label for label in collection if label not in labels_to_remove]

        self.conditional_branch_edges = _substitute_conditional_branch_edges(
            self.conditional_branch_edges, labels_to_remove, replacement_labels
        )

        for param_group, tensor_labels in list(self.layers_with_params.items()):
            self.layers_with_params[param_group] = [
                label for label in tensor_labels if label not in labels_to_remove
            ]
        self.layers_with_params = {
            param_group: tensor_labels
            for param_group, tensor_labels in self.layers_with_params.items()
            if len(tensor_labels) > 0
        }

        for _equiv_group, equivalent_label_set in list(self.op_equivalence_classes.items()):
            equivalent_label_set -= labels_to_remove
        self.op_equivalence_classes = {
            equiv_group: equivalent_label_set
            for equiv_group, equivalent_label_set in self.op_equivalence_classes.items()
            if len(equivalent_label_set) > 0
        }

        _scrub_per_op_equivalence_lists(surviving_entries, labels_to_remove)

        for entry in entries_to_remove:
            _clear_entry_attributes(entry)
