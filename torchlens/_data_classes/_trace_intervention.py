"""Trace intervention mixin: the thin shell of the intervention verbs.

Every method forwards to a function taking the trace first. The bodies live in
``torchlens.intervention._trace_edits`` (``do``, ``set``, ``remove``,
``clear_edits``, ``save_intervention`` and their helpers), ``_trace_selection_edits``
(selection-kind ``do``), ``_trace_hooks`` (``attach_hooks``, ``clear_hooks``),
reached through the ``_doors.intervention`` door, and in the L1 modules
``_trace_intervention_records`` (spec state and history records) and
``_fork_lineage`` (the fork verb). The COW fork builder itself is
``_trace_fork.build_fork``.
"""

from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from ..intervention.types import FrozenInterventionSpec, InterventionSpec, TargetSpec
from ..options import InterventionOptions
from . import _doors, _fork_lineage, _trace_intervention_records
from ._trace_fork import _STREAM_DERIVED_GUARD_FIELDS, _ForkMemo, _memoized_deep_copy, build_fork
from ._trace_grad_sites import armed_edits, armed_summary_line, settle_armed_gradient_edits
from ._trace_intervention_records import _DoTransaction

# The re-exports stay until the fold removes this module: the torch backward door
# reads ``settle_armed_gradient_edits`` from here by name, and tests import the rest.
__all__ = [
    "_STREAM_DERIVED_GUARD_FIELDS",
    "TraceInterventionMixin",
    "_ForkMemo",
    "_memoized_deep_copy",
    "settle_armed_gradient_edits",
]


class TraceInterventionMixin(_TraceMixinBase):
    """``Trace`` intervention surface: spec save/load, fork, replay, and rerun."""

    @property
    def injected_ops(self: "Trace") -> tuple[Any, ...]:
        """Injected-op records from ``log_injections=True`` hooks (F01 stage 1).

        The injected half of the query split: computation performed INSIDE
        intervention hooks, recorded as anchored ``InjectedOp`` records
        clustering under their intervention (never inside the model's module
        hierarchy). Empty tuple when the option was never armed or no hook
        executed torch calls. Session-only at stage 1 (save refuses typed,
        naming lane F44). Spelling DOCUMENTED-UNSTABLE.
        """

        return _doors.intervention()._trace_edits.injected_ops(self)

    @property
    def model_ops(self: "Trace") -> tuple[Any, ...]:
        """The MODEL half of the injections query split (F01 stage 1).

        Exactly the trace's ordinary executed op records (``layer_list``);
        injected ops never enter this family, the label counters, or the
        site-key cohort ordinals, so this split is a disclosure, not a
        filter. Spelling DOCUMENTED-UNSTABLE.
        """

        return tuple(self._layer_list)

    def save_intervention(
        self: "Trace",
        path: str | Path,
        *,
        level: str = "executable_with_callables",
        allow_direct_writes: bool = False,
        overwrite: bool = False,
    ) -> None:
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

        return _doors.intervention()._trace_edits.save_intervention(
            self, path, level=level, allow_direct_writes=allow_direct_writes, overwrite=overwrite
        )

    @cached_property
    def recipe(self: "Trace") -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def _history_site_payload(self: "Trace", site: Any) -> Any:
        """Forward to ``_trace_intervention_records.history_site_payload``."""

        return _trace_intervention_records.history_site_payload(site)

    def set(
        self: "Trace",
        site: Any,
        value: Any,
        *,
        direction: str = "forward",
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
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

        return _doors.intervention()._trace_edits.set_site(
            self, site, value, direction=direction, strict=strict, confirm_mutation=confirm_mutation
        )

    def attach_hooks(
        self: "Trace",
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
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

        return _doors.intervention()._trace_hooks.attach_hooks(
            self,
            hooks_or_site,
            hook,
            *extra_hooks,
            strict=strict,
            prepend=prepend,
            confirm_mutation=confirm_mutation,
        )

    def _refuse_chunked_batch_coupling(self: "Trace", entries: list[Any]) -> None:
        """Forward to ``_doors.intervention()._edit_guards.refuse_chunked_batch_coupling``."""

        return _doors.intervention()._edit_guards.refuse_chunked_batch_coupling(self, entries)

    def remove(self: "Trace") -> None:
        """Remove the most recent legacy-returned hook attachment.

        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """

        return _doors.intervention()._trace_edits.remove(self)

    def __enter__(self: "Trace") -> "Trace":
        """Enter a legacy scoped hook attachment.

        Returns
        -------
        Trace
            This log, acting as the most recent hook handle.
        """

        return self

    def __exit__(self: "Trace", exc_type: Any, exc: Any, traceback: Any) -> None:
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

        self.remove()

    def clear_edits(
        self: "Trace",
        site: Any = None,
        handle: Any = None,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
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

        return _doors.intervention()._trace_edits.clear_edits(
            self, site, handle, strict=strict, confirm_mutation=confirm_mutation
        )

    def clear_hooks(self: "Trace", *, confirm_mutation: bool = False) -> "Trace":
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

        return _doors.intervention()._trace_hooks.clear_hooks(
            self, confirm_mutation=confirm_mutation
        )

    @property
    def _graph_edges(self: "Trace") -> tuple[Any, ...]:
        """Forward to ``_doors.intervention()._trace_edits.graph_edges``."""

        return _doors.intervention()._trace_edits.graph_edges(self)

    def _settle_selection_do(
        self: "Trace", txn: "_DoTransaction", selected_engine: str, mutation_kind: str
    ) -> None:
        """Forward to ``_doors.intervention()._trace_edits.settle_selection_do``."""

        return _doors.intervention()._trace_edits.settle_selection_do(
            self, txn, selected_engine, mutation_kind
        )

    def do(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        inputs: Any = None,
        intervention: InterventionOptions | None = None,
        loss: Callable[["Trace"], torch.Tensor] | None = None,
    ) -> "Trace":
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

        return _doors.intervention()._trace_edits.do(
            self,
            hooks_or_site,
            value_or_hook,
            model=model,
            inputs=inputs,
            intervention=intervention,
            loss=loss,
        )

    def _do_with_options(  # noqa: PLR0913 -- do()'s own arguments, after it merges the options and enters the changed-parameter scope
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        model: nn.Module | None,
        x: Any,
        intervention_options: InterventionOptions,
    ) -> "Trace":
        """Forward to ``_doors.intervention()._trace_edits.do_with_options``."""

        return _doors.intervention()._trace_edits.do_with_options(
            self,
            hooks_or_site,
            value_or_hook,
            model=model,
            x=x,
            intervention_options=intervention_options,
        )

    @property
    def armed_edits(self: "Trace") -> tuple[Any, ...]:
        """The gradient rules in this trace's recipe, as ``Rule`` clauses.

        Read-only. Each entry is ``Rule(target=tl.grad(...), edit=...)`` with
        ``signal == "backward"``; the rules stay armed (sticky) until
        ``clear_edits(...)`` or their handle's ``remove()``.
        """

        return armed_edits(self)

    def _armed_summary_line(self: "Trace") -> str | None:
        """Summary text while an armed gradient rule has not fired (or ``None``)."""

        return armed_summary_line(self)

    def _record_do_intervention_event(
        self: "Trace",
        txn: _DoTransaction,
        *,
        status: str | None,
        error: str | None = None,
        staged_only: bool = False,
        extra_fields: dict[str, Any] | None = None,
        event_row: bool = False,
    ) -> dict[str, Any]:
        """Forward to ``_doors.intervention()._trace_edits.record_do_intervention_event``."""

        return _doors.intervention()._trace_edits.record_do_intervention_event(
            self,
            txn,
            status=status,
            error=error,
            staged_only=staged_only,
            extra_fields=extra_fields,
            event_row=event_row,
        )

    def fork(self: "Trace", name: str | None = None) -> "Trace":
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

        return _fork_lineage.fork_trace(self, name)

    def _record_operation(self: "Trace", op: str, **payload: Any) -> None:
        """Forward to ``_trace_intervention_records.record_operation``."""

        return _trace_intervention_records.record_operation(self, op, **payload)

    def _warn_if_root_mutation(self: "Trace", *, confirm_mutation: bool) -> None:
        """Forward to ``_doors.intervention()._edit_guards.warn_if_root_mutation``."""

        return _doors.intervention()._edit_guards.warn_if_root_mutation(
            self, confirm_mutation=confirm_mutation
        )

    def _select_do_engine(self: "Trace", engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Forward to ``_doors.intervention()._edit_guards.select_do_engine``."""

        return _doors.intervention()._edit_guards.select_do_engine(self, engine, model=model, x=x)

    def _apply_do_mutation(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_edits.apply_do_mutation``."""

        return _doors.intervention()._trace_edits.apply_do_mutation(
            self,
            hooks_or_site,
            value_or_hook,
            engine=engine,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )

    @staticmethod
    def _is_selection_batch(value: Any) -> bool:
        """Forward to ``_doors.intervention()._trace_selection_edits.is_selection_batch``."""

        return _doors.intervention()._trace_selection_edits.is_selection_batch(value)

    @staticmethod
    def _selection_batch_kind(selection: Any) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.selection_batch_kind``."""

        return _doors.intervention()._trace_selection_edits.selection_batch_kind(selection)

    def _apply_selection_batch_do(
        self: "Trace",
        pairs: Any,
        *,
        engine: str,
        strict: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_batch_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_batch_do(
            self, pairs, engine=engine, strict=strict
        )

    def _batch_act_plan(
        self: "Trace", index: int, selection: Any, edit: Any
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.batch_act_plan``."""

        return _doors.intervention()._trace_selection_edits.batch_act_plan(
            self, index, selection, edit
        )

    def _apply_cross_kind_batch_do(
        self: "Trace",
        pairs: list[tuple[Any, Any]],
        kinds: list[str],
        *,
        engine: str,
        strict: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_cross_kind_batch_do``."""

        return _doors.intervention()._trace_selection_edits.apply_cross_kind_batch_do(
            self, pairs, kinds, engine=engine, strict=strict
        )

    def _apply_selection_do(
        self: "Trace",
        selection: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_do(
            self, selection, edit, engine=engine, strict=strict
        )

    def _attach_selection_plan_hooks(
        self: "Trace",
        hook_items: list[dict[str, Any]],
        *,
        strict: bool,
    ) -> tuple[str, ...]:
        """Forward to ``_doors.intervention()._trace_selection_edits.attach_selection_plan_hooks``."""

        return _doors.intervention()._trace_selection_edits.attach_selection_plan_hooks(
            self, hook_items, strict=strict
        )

    def _apply_selection_edge_do(
        self: "Trace",
        selection: Any,
        lifted: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
    ) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_edge_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_edge_do(
            self, selection, lifted, edit, engine=engine, strict=strict
        )

    def _apply_selection_param_do(
        self: "Trace",
        selection: Any,
        lifted: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
    ) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_param_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_param_do(
            self, selection, lifted, edit, engine=engine, strict=strict
        )

    def _apply_leaf_selection_edits(self: "Trace", leaf_items: list[dict[str, Any]]) -> None:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_leaf_selection_edits``."""

        return _doors.intervention()._trace_selection_edits.apply_leaf_selection_edits(
            self, leaf_items
        )

    def _validate_supplied_model_matches_capture(self: "Trace", model: nn.Module) -> None:
        """Forward to ``_doors.intervention()._trace_edits.validate_supplied_model_matches_capture``."""

        return _doors.intervention()._trace_edits.validate_supplied_model_matches_capture(
            self, model
        )

    def _fork_trace(
        self: "Trace",
        *,
        name: str | None,
    ) -> "Trace":
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

        return build_fork(self, name=name)

    def _next_fork_name(self: "Trace") -> str:
        """Forward to ``_fork_lineage.next_fork_name``."""

        return _fork_lineage.next_fork_name(self)

    def _rebind_fork_owner_refs(self: "Trace") -> None:
        """Forward to ``_fork_lineage.rebind_fork_owner_refs``."""

        return _fork_lineage.rebind_fork_owner_refs(self)

    def _recipe_is_clean(self: "Trace") -> bool:
        """Forward to ``_trace_intervention_records.recipe_is_clean``."""

        return _trace_intervention_records.recipe_is_clean(self)

    def _ensure_intervention_spec(self: "Trace") -> InterventionSpec:
        """Forward to ``intervention._trace_edits.ensure_intervention_spec``."""

        return _doors.intervention()._trace_edits.ensure_intervention_spec(self)

    def _mark_intervention_spec_mutated(self: "Trace", *, outs_stale: bool = True) -> None:
        """Forward to ``_trace_intervention_records.mark_intervention_spec_mutated``."""

        return _trace_intervention_records.mark_intervention_spec_mutated(
            self, outs_stale=outs_stale
        )

    def _validate_intervention_site(self: "Trace", site: Any, *, strict: bool) -> None:
        """Forward to ``_doors.intervention()._trace_edits.validate_intervention_site``."""

        return _doors.intervention()._trace_edits.validate_intervention_site(
            self, site, strict=strict
        )

    def _target_spec_from_site(self: "Trace", site: Any, *, strict: bool) -> TargetSpec:
        """Forward to ``_doors.intervention()._trace_edits.target_spec_from_site``."""

        return _doors.intervention()._trace_edits.target_spec_from_site(self, site, strict=strict)
