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
    "set": (
        """Set a site out recipe without propagating it.
        
        Parameters
        ----------
        site:
            Selector-like target for the out to replace.
        value:
            Static replacement tensor or one-shot callable accepting the
            matched out and returning a replacement tensor.
        direction:
            Signal direction to replace: ``"forward"``, ``"backward"``, or
            ``"both"``.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        
        Returns
        -------
        Trace
            This model log, with a stale intervention recipe.
        """
    ),
    "attach_hooks": (
        """Attach sticky hooks to the current intervention spec.
        
        The site decides which tensor a hook sees: a bare site mounts on the
        activation, ``tl.grad(site)`` on the gradient of the site's output
        (fired by ``trace.backward(loss)``), and ``tl.grad(site) | site`` on
        both. Attaching is registration only: nothing runs until the next
        forward (activation hooks) or backward (gradient hooks).
        
        Raw PyTorch ``register_forward_hook`` remains supported for users who
        need module-local replacement logic outside this API; during active
        TorchLens captures, returned replacement tensors are instrumented so the
        graph can continue through downstream ops.
        
        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        hook:
            Optional hook for the ``(site, hook)`` input shape.
        *extra_hooks:
            Additional hooks to compose at ``hooks_or_site`` in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        
        Returns
        -------
        Any
            Scoped removable hook handle.
        """
    ),
    "remove": (
        """Remove the most recent legacy-returned hook attachment.
        
        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """
    ),
    "__enter__": (
        """Enter a legacy scoped hook attachment.
        
        Returns
        -------
        Trace
            This log, acting as the most recent hook handle.
        """
    ),
    "__exit__": (
        """Clean up a legacy scoped hook attachment.
        
        Parameters
        ----------
        exc_type:
            Exception type, if the body raised.
        exc:
            Exception value, if the body raised.
        traceback:
            Exception traceback, if the body raised.
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
            Optional hook handle returned by ``attach_hooks``.
        strict:
            Whether no-op detach requests should raise.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        
        Returns
        -------
        Trace
            This model log, with a stale recipe if hooks were removed.
        """
    ),
    "clear_hooks": (
        """Clear all sticky hooks from the current intervention spec.
        
        Parameters
        ----------
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        
        Returns
        -------
        Trace
            This model log with hook specs cleared and marked stale.
        """
    ),
    "do": (
        """Apply an intervention and dispatch to replay, rerun, or set-only.
        
        The site decides which tensor the edit changes: a bare site edits the
        activation, ``tl.grad(site)`` the gradient of the site's output, and
        ``tl.grad(site) | site`` both. A gradient edit needs a backward pass:
        with ``model=`` and ``x=`` the forward reruns on a new trace with the
        edit armed (the audit row says ``"armed"``) and the next
        ``result.backward(loss)`` applies it; ``engine="set_only"`` arms it on
        this trace instead. Replay re-executes forward ops only and refuses a
        gradient edit (``replay_gradient_edit_unsupported``).
        
        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional hook for the ``(site, hook)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        inputs:
            Input required when ``engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``: ``engine``
            in ``"auto"``/``"replay"``/``"rerun"``/``"set_only"``,
            ``confirm_mutation``, ``strict``).
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
