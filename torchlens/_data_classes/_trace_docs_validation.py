"""Full docstrings of the ``Trace`` validation, run and backward verbs methods.

The methods on ``Trace`` carry a one-line summary; ``_trace_docs.attach_trace_docs`` binds the
text here as their ``__doc__`` when ``trace.py`` finishes defining the class, so ``help()``
and notebooks show the full reference while ``trace.py`` stays inside its file cap.
"""

from __future__ import annotations

__all__ = ["DOCS"]

DOCS: dict[str, str] = {
    "save_new_outs": (
        """Re-run the model with new inputs, saving only outs.
        
        Parameters
        ----------
        model, input_args, input_kwargs, save, save_grads, seed, backward_ready:
            Forwarded unchanged to
            :func:`torchlens._capture.trace.save_new_outs`.
        """
    ),
    "validate_forward_pass": (
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
    ),
    "validation_replay_status": (
        """Return replay-validation availability or last completed result.
        
        Returns
        -------
        ValidationStatus
            Status object distinguishing live replay validation from loaded
            traces whose runtime replay captures were stripped during save.
        """
    ),
    "is_pristine": (
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
    ),
    "push": (
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
    ),
    "push_from": (
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
    ),
    "run": (
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
            binds ``inputs``, on every provider and with ``fast=True``
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
    ),
    "check_metadata_invariants": (
        """Run metadata invariant checks on this completed model log.
        
        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
    ),
    "cleanup": (
        """Delete log data, break cycles, and free cached GPU memory.
        
        Returns
        -------
        None
            This method mutates the model log in place.
        """
    ),
    "_release_param_refs": (
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
    ),
    "_postprocess": (
        """Run postprocessing on a completed raw capture pass.
        
        Parameters
        ----------
        output_tensors:
            Output tensors returned by the model.
        output_tensor_addresses:
            Hierarchical addresses for those outputs.
        """
    ),
    "_run_and_log_inputs_through_model": (
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
    ),
    "log_backward": (
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
    ),
    "backward": (
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
    ),
    "capture_backward": (
        """Return a context manager that captures user-managed backward calls.
        
        Returns
        -------
        Any
            Backward recording context manager.
        """
    ),
    "disarm_triggers": (
        """Detach this Trace from global autograd backward interception.
        
        Returns
        -------
        None
            Future plain ``loss.backward()`` or ``torch.autograd.*`` calls will
            not record into this Trace.
        """
    ),
    "_remove_log_entry": (
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
    ),
    "_batch_remove_log_entries": (
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
    ),
}
