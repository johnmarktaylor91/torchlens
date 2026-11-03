"""Trace validation, run and backward mixin: thin methods over the record-rebuild destinations.

The validation, run and backward bodies live in ``validation/_trace_checks.py``,
``intervention/_trace_run.py`` and ``backends/torch/_trace_backward.py``; the methods here keep the
public signatures and docstrings and forward through ``_doors``. The log-entry removal helpers and
the capture-side private methods stay here until the fold step pulls the shell into ``trace.py``.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, Union

import torch

if TYPE_CHECKING:
    from .._runnable import RunResult
    from ..validation.status import ValidationStatus
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._deprecations import MISSING, MissingType
from ..options import ReplayOptions
from . import _doors
from ._trace_pristine import derive_pristine
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

#: Sentinel default of ``Trace.run``'s ``input_transform`` / ``output_transform``: reuse the stored
#: transform. Defined here (L1) so the thin method's signature and ``intervention/_trace_run.py``
#: compare against the same object; the fold step moves it with the method.
_USE_STORED_TRANSFORM = object()


def _stateful_run_warning_was_emitted(trace: Any) -> bool:
    """Whether ``trace`` already carried its once-only stateful-run warning."""

    return bool(trace.__dict__.get("_stateful_run_warning_emitted", False))


def _mark_stateful_run_warning_emitted(trace: Any) -> None:
    """Record on ``trace`` that its once-only stateful-run warning was emitted."""

    trace.__dict__["_stateful_run_warning_emitted"] = True


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

    layer_logs = trace.__dict__.get("_layer_logs")
    if not layer_logs:
        return
    from .layer import _layer_rep_op, materialize_layer_mirrors

    removed_ids = {id(entry) for entry in removed_entries}
    for layer_log in layer_logs.values():
        rep = _layer_rep_op(layer_log)
        if rep is not None and id(rep) in removed_ids:
            materialize_layer_mirrors(layer_log)


class TraceValidationMixin(_TraceMixinBase):
    """``Trace`` validation surface: forward replay, backward checks, and re-capture."""

    def save_new_outs(
        self: "Trace",
        model: torch.nn.Module,
        input_args: torch.Tensor | list[Any],
        input_kwargs: dict[Any, Any] | None = None,
        layers_to_save: str | list[str] = "all",
        grad_layers_to_save: str | list[str] | None = "all",
        seed: int | None = None,
        backward_ready: bool | None = None,
        _run_until_plan: Any | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only outs.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, seed, backward_ready:
            Forwarded unchanged to
            :func:`torchlens._capture.trace.save_new_outs`.
        """
        return _doors.validation()._trace_checks.save_new_outs(
            self,
            model,
            input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            seed=seed,
            backward_ready=backward_ready,
            _run_until_plan=_run_until_plan,
        )

    def validate_forward_pass(
        self: "Trace",
        ground_truth_output_tensors: list[torch.Tensor] | torch.Tensor,
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationStatus"]:
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
        bool or ValidationStatus
            ``True`` if validation succeeds. A torch replay that passes on final
            outputs whose every element is non-finite returns ``False`` with a
            ``degenerate_forward`` failure recorded. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status instead of a pass/fail bool.

        Notes
        -----
        Every op replays under its recorded RNG state; like ``tl.trace``, the
        call restores the caller's global RNG streams on return and on error.
        """
        return _doors.validation()._trace_checks.validate_forward_pass(
            self, ground_truth_output_tensors, verbose=verbose, validate_metadata=validate_metadata
        )

    @property
    def validation_replay_status(self: "Trace") -> "ValidationStatus":
        """Return replay-validation availability or last completed result.

        Returns
        -------
        ValidationStatus
            Status object distinguishing live replay validation from loaded
            traces whose runtime replay captures were stripped during save.
        """
        return _doors.validation()._trace_checks.validation_replay_status(self)

    @property
    def is_pristine(self: "Trace") -> bool | None:
        """Whether nothing ever intervened on this trace, read from its saved evidence.

        Derived at read time from records that persist with the trace (the
        intervention audit, the operation history, mutation states, direct
        writes and op intervention marks), so it survives ``Trace.save`` /
        ``tl.load`` and no reset makes a touched trace read pristine again.
        It is a reading, never a verdict: it does not change validation or
        any run report. Distinct from ``Trace.state`` (``TraceState.PRISTINE``
        is an operational state an assignment can set) and from validation's
        pristine-torch oracle (unwrapped torch functions).

        Returns
        -------
        bool | None
            ``False`` when any touch evidence is found; ``None`` when the
            trace cannot carry the evidence (an artifact older than tlspec 9,
            or a non-torch backend), which means no evidence and is never a
            pristine reading; otherwise ``True``.
        """

        return derive_pristine(self)

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
        return _doors.intervention()._trace_run.push(self, replay=replay)

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
        return _doors.intervention()._trace_run.push_from(self, site, replay=replay)

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
        on_divergence: Literal["raise", "return"] = "raise",
        chunk_paths: Any | None = None,
        replay: ReplayOptions | None = None,
        input_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        on_zero_fire: str | MissingType = MISSING,
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
            A tuple or list spreads as positional arguments, as ``tl.trace``
            binds ``input_args``, on every provider and with ``fast=True``
            (on a live trace a one-parameter forward takes a multi-element
            sequence whole); any other value is the single argument, and
            ``{"args": [...], "kwargs": {...}}`` binds both.
        seed:
            Optional deterministic live refresh, random-state, and runtime RNG seed.
        fast:
            Explicit stateful static-loop mode. The live provider runs native ``forward``
            with targeted module/function collection and a per-call path/shape guard. The
            loaded provider performs one ordinary verified run, then reuses staged state
            and compiled argument binders. Both keep the source Trace untouched; unlike the
            default transactional provider, later fast iterations refresh one result Trace.
            With ``until=`` a live fast run cuts the native forward right after that saved site.
        carry_state:
            PROVISIONAL spelling (documented-unstable). Default ``False``: the
            live provider snapshot-restores the model's declared state (named
            parameters, registered buffers, alias topology preserved) around
            the run, so repeated ``run()`` calls leave the model bit-identical.
            ``True`` skips the restore: declared-state mutations persist on the
            live model (episode/rollout state accumulation). The report
            discloses the choice (``report.has_carried_state``); verification is
            untouched -- the NEXT run from mutated state faces every gate as
            usual. Refuses typed with ``fast=True`` and on loaded providers
            (staged clones have no live model for state to carry into).
        on_divergence:
            Strict divergence behavior or the sole poison-return opt-in.
        chunk_paths:
            Optional explicit tensor leaf paths to split.
        strict:
            Whether graph-shape divergence should raise instead of warn.
        input_transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit input
            transform callable for this run.
        output_transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit output
            transform callable for this run.
        on_zero_fire:
            What a staged hook entry that fired at zero sites on this run does
            after the fact is recorded (``last_run["hooks_unfired"]``):
            ``"warn"`` (default) emits a ``rerun_zero_fire`` coded warning,
            ``"error"`` raises ``rerun_zero_fire``, ``"disclose"`` records only.
            The same knob as ``spec.bind`` and ``tl.trace``. Typed refusals for
            an edit that cannot fire on a rerun (a post-capture-only predicate, a
            same-input rerun that fired nowhere) come first, whatever the knob
            says (spelling DOCUMENTED-UNSTABLE).

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
        InvalidArgumentError
            ``trace_option_unsupported`` for a Trace captured with ``storage=tl.to_disk(...)``: a
            run would re-capture every payload into RAM; capture again to a new path and run that.

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
        return _doors.intervention()._trace_run.run(
            self,
            model,
            x,
            inputs=inputs,
            seed=seed,
            fast=fast,
            carry_state=carry_state,
            until=until,
            save=save,
            on_divergence=on_divergence,
            chunk_paths=chunk_paths,
            replay=replay,
            input_transform=input_transform,
            output_transform=output_transform,
            on_zero_fire=on_zero_fire,
        )

    def check_metadata_invariants(self: "Trace") -> bool:
        """Run metadata invariant checks on this completed model log.

        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
        return _doors.validation()._trace_checks.check_metadata_invariants(self)

    def cleanup(self: "Trace") -> None:
        """Delete log data, break cycles, and free cached GPU memory.

        Returns
        -------
        None
            This method mutates the model log in place.
        """
        return cleanup(self)

    def _release_param_refs(self: "Trace", *, allow_iter_rehydrate: bool = False) -> None:
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
        if hasattr(self._param_logs, "_rehydrate_on_iter"):
            self._param_logs._rehydrate_on_iter = False
        for param_log in self._param_logs.values():
            param_log._release_param_ref()
        if hasattr(self._param_logs, "_rehydrate_on_iter"):
            self._param_logs._rehydrate_on_iter = allow_iter_rehydrate

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
        from .._postprocess import postprocess as _impl

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
            :func:`torchlens._capture.trace.run_and_log_inputs_through_model`.
        reservation_resume:
            Continuation token from the caller's own live ``capture_reservation``
            claim (the recorder pass); a bare nested entry without it refuses.
        """
        from .._capture.run import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            seed=random_seed,
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
        return _doors.backends_torch()._trace_backward.log_backward(self, loss, **backward_kwargs)

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

    def capture_backward(self: "Trace") -> Any:
        """Return a context manager that captures user-managed backward calls.

        Returns
        -------
        Any
            Backward recording context manager.
        """
        return _doors.backends_torch()._trace_backward.capture_backward(self)

    def disarm_triggers(self: "Trace") -> None:
        """Detach this Trace from global autograd backward interception.

        Returns
        -------
        None
            Future plain ``loss.backward()`` or ``torch.autograd.*`` calls will
            not record into this Trace.
        """
        _doors.intervention()._trace_run.disarm_triggers(self)

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

        self._graph_conditional_branch_edges = _substitute_conditional_branch_edges(
            self._graph_conditional_branch_edges, labels_to_remove, replacement_labels
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

        for _equiv_group, equivalent_label_set in list(self._graph_op_equivalence_classes.items()):
            equivalent_label_set -= labels_to_remove
        self._graph_op_equivalence_classes = {
            equiv_group: equivalent_label_set
            for equiv_group, equivalent_label_set in self._graph_op_equivalence_classes.items()
            if len(equivalent_label_set) > 0
        }

        _scrub_per_op_equivalence_lists(surviving_entries, labels_to_remove)

        for entry in entries_to_remove:
            _clear_entry_attributes(entry)
