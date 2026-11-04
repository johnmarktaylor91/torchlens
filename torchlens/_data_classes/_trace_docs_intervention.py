"""Full docstrings of the ``Trace`` intervention verbs, hooks and forks methods.

The methods on ``Trace`` carry a one-line summary; ``_trace_docs.attach_trace_docs`` binds the
text here as their ``__doc__`` when ``trace.py`` finishes defining the class, so ``help()``
and notebooks show the full reference while ``trace.py`` stays inside its file cap.
"""

from __future__ import annotations

__all__ = ["DOCS"]

DOCS: dict[str, str] = {
    "injected_ops": (
        """Injected-op records from ``log_injections=True`` hooks (F01 stage 1).
        
        The injected half of the query split: computation performed INSIDE
        intervention hooks, recorded as anchored ``InjectedOp`` records
        clustering under their intervention (never inside the model's module
        hierarchy). Empty tuple when the option was never armed or no hook
        executed torch calls. Session-only at stage 1 (save refuses typed,
        naming lane F44). Spelling DOCUMENTED-UNSTABLE.
        """
    ),
    "model_ops": (
        """The MODEL half of the injections query split (F01 stage 1).
        
        Exactly the trace's ordinary executed op records (``layer_list``);
        injected ops never enter this family, the label counters, or the
        site-key cohort ordinals, so this split is a disclosure, not a
        filter. Spelling DOCUMENTED-UNSTABLE.
        """
    ),
    "save_intervention": (
        """Save this log's intervention recipe to a ``.tlspec`` directory.
        
        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        allow_direct_writes:
            Whether executable saves may proceed after direct out writes.
        overwrite:
            Whether an existing destination may be replaced.
        """
    ),
    "recipe": (
        """Return an immutable snapshot of this log's intervention recipe.
        
        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """
    ),
    "clear_edits": (
        """Detach sticky hooks by site or handle.
        
        Parameters
        ----------
        site:
            Optional selector-like target. When provided, all sticky hooks for
            that target are removed.
        handle:
            Optional hook handle returned by a hook ``do``.
        strict:
            Whether no-op detach requests should raise.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        
        Returns
        -------
        Trace
            This model log, with a stale recipe if hooks were removed.

        Notes
        -----
        Removal takes effect at once, like a torch hook handle: a rerun or
        backward pass in progress skips the removed hooks from the next hook
        site it reaches, and the site that is firing when this is called
        finishes its queued hooks first.
        """
    ),
    "do": (
        """Apply an intervention and dispatch to replay, rerun, or set-only.
        
        ``do`` is the one editing verb: ``(site, value)`` stages a value,
        ``(site, hook)`` stages a sticky hook, and a ``Recipe`` (``tl.when``)
        stages every rule it carries; the selected engine then propagates.
        ``engine="set_only"`` stages without running (a ``Recipe`` keeps its
        hooks sticky for the next ``run``). ``clear_edits`` removes staged
        edits by site or handle.
        
        The site decides which tensor the edit changes: a bare site edits the
        activation, ``tl.grad(site)`` the gradient of the site's output, and
        ``tl.grad(site) | site`` both. A gradient edit needs a backward pass:
        with ``model=`` and ``inputs=`` the forward reruns on a new trace with the
        edit armed (the audit row says ``"armed"``) and the next
        ``result.backward(loss)`` applies it; ``engine="set_only"`` arms it on
        this trace instead. Replay re-executes forward ops only and refuses a
        gradient edit (``replay_gradient_edit_unsupported``).
        
        ``do`` is the one editing verb on a trace. A non-callable value sets
        the site and propagates it; a hook or helper is attached as a sticky
        rule and propagated; a mapping of sites to edits applies several edits
        in one propagation; an ``InterventionSpec`` applies its rules.
        ``InterventionOptions(engine="set_only")`` stages without
        propagating, and ``InterventionOptions(differentiable=True)`` replays
        onto a new fork whose outputs stay differentiable.
        
        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional value, hook or helper for the ``(site, edit)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        inputs:
            Input required when ``engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``: ``engine``
            in ``"auto"``/``"replay"``/``"rerun"``/``"set_only"``,
            ``confirm_mutation``, ``strict``, ``allow_changed_params``,
            ``differentiable``).
        loss:
            Callable of the returned trace giving a scalar loss, for a plan
            with a gradient site: after arming, the door runs
            ``result.backward(loss(result))`` so the gradient edit fires inside
            this call. Refuses ``do_loss_not_callable`` for a tensor or any
            other non-callable and ``do_loss_without_gradient_site`` when no
            site is a gradient site.
        
        Returns
        -------
        Trace
            This model log after the selected propagation engine runs, or, for
            a gradient plan rerun, the new trace holding the armed edits.
        """
    ),
    "armed_edits": (
        """The gradient rules in this trace's recipe, as ``Rule`` clauses.
        
        Read-only. Each entry is ``Rule(target=tl.grad(...), edit=...)`` with
        ``signal == "backward"``; the rules stay armed (sticky) until
        ``clear_edits(...)`` or their handle's ``remove()``.
        """
    ),
    "fork": (
        """Create a copy-on-write intervention fork of this log.
        
        Tensor payloads and sealed metadata columns are SHARED with this
        trace (the dominant bytes on real models), but the fork is not
        near-free: isolating every mutation surface retains on the order of
        ~60 gc-tracked objects / ~13 KB of small allocations per op
        (measured; see ``_trace_fork``). The fork settles a DERIVED capture
        outcome (UNATTESTED for a complete parent), never this trace's
        attested settle stamp -- a fork is the sanctioned mutation surface.
        
        Parameters
        ----------
        name:
            Optional name for the forked log.
        
        Returns
        -------
        Trace
            Forked model log.
        """
    ),
    "_fork_trace": (
        """Build a copy-on-write fork of this Trace (the M11 builder).
        
        Parameters
        ----------
        name:
            Optional fork name.
        
        Returns
        -------
        Trace
            Fork sharing this trace's frozen storage, with every mutation
            surface isolated (see ``_trace_fork.build_fork``).
        """
    ),
}
