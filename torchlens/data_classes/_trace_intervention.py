"""Trace intervention mixin.

The fork itself is the M11 copy-on-write builder in ``_trace_fork``;
this mixin owns the public intervention surface (set/attach/detach/do/
fork dispatch, spec management, history records).
"""

import copy
import sys
import time
import uuid
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._errors import InvalidArgumentError
from .._trace_state import TraceState
from ..intervention.types import (
    FrozenInterventionSpec,
    InterventionSpec,
    TargetSpec,
)
from ..options import InterventionOptions, ReplayOptions, merge_intervention_options
from ._trace_fork import (
    _STREAM_DERIVED_GUARD_FIELDS,
    _ForkMemo,
    _memoized_deep_copy,
    build_fork,
)

__all__ = [
    "_STREAM_DERIVED_GUARD_FIELDS",
    "TraceInterventionMixin",
    "_ForkMemo",
    "_memoized_deep_copy",
]


class _DoTransaction(NamedTuple):
    """Identity of one ``do()`` transaction, minted once at the door.

    ``transaction_start`` is the monotonic bound scoping this transaction's
    FireRecords; ``audit_rows_before`` detects whether the mutation step
    already appended its own audit row.
    """

    hooks_or_site: Any
    value_or_hook: Any
    engine: str
    transaction_start: float
    audit_rows_before: int


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

        from ..intervention.injection import injected_ops

        return injected_ops(self)

    @property
    def model_ops(self: "Trace") -> tuple[Any, ...]:
        """The MODEL half of the injections query split (F01 stage 1).

        Exactly the trace's ordinary executed op records (``layer_list``);
        injected ops never enter this family, the label counters, or the
        site-key cohort ordinals, so this split is a disclosure, not a
        filter. Spelling DOCUMENTED-UNSTABLE.
        """

        return tuple(self.layer_list)

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

        from ..intervention.save import save_intervention
        from ..runnable import refuse_poisoned_trace

        refuse_poisoned_trace(self, "intervention export")
        save_intervention(
            self,
            path,
            level=level,
            allow_direct_writes=allow_direct_writes,
            overwrite=overwrite,
        )

    @cached_property
    def intervention_spec(self: "Trace") -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def _history_site_payload(self: "Trace", site: Any) -> Any:
        """Return a stable site payload for ``state_history`` records.

        Parameters
        ----------
        site:
            Original selector-like site payload supplied to a mutator.

        Returns
        -------
        Any
            Plain layer labels for direct label targets, otherwise a stable
            repr-style string for human-readable history records.
        """

        del self
        if isinstance(site, str):
            return site
        selector_kind = getattr(site, "selector_kind", None)
        selector_value = getattr(site, "selector_value", None)
        if selector_kind == "label" and isinstance(selector_value, str):
            return selector_value
        return repr(site)

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

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        if direction not in {"forward", "backward", "both"}:
            raise InvalidArgumentError(
                "set(..., direction=...) must be 'forward', 'backward', or 'both'; "
                f"received {direction!r}",
                code="intervention_direction_invalid",
                remedy="pass direction='forward', 'backward', or 'both'",
                argument="direction",
            )
        if direction in {"backward", "both"}:

            def _backward_set_hook(grad: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return a static or callable gradient replacement."""

                del hook
                if callable(value):
                    return cast(torch.Tensor, value(grad))
                return cast(torch.Tensor, value)

            self.attach_hooks(
                site,
                _backward_set_hook,
                direction="backward",
                strict=strict,
                confirm_mutation=True,
            )
            if direction == "backward":
                self._record_operation(
                    "set",
                    site=self._history_site_payload(site),
                    value_kind=type(value).__name__,
                    strict=strict,
                    callable=callable(value),
                    direction=direction,
                )
                return self
        from ..intervention.hooks import is_facet_target

        if is_facet_target(site):

            def _facet_replacement_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return a static or callable facet replacement slice.

                Parameters
                ----------
                out:
                    Facet slice supplied by the facet hook wrapper.
                hook:
                    Hook context supplied by TorchLens.

                Returns
                -------
                torch.Tensor
                    Replacement facet slice.
                """

                del hook
                if callable(value):
                    return cast(torch.Tensor, value(out))
                return cast(torch.Tensor, value)

            self.attach_hooks(
                site,
                _facet_replacement_hook,
                direction="forward",
                strict=strict,
                confirm_mutation=True,
            )
            self._record_operation(
                "set",
                site=self._history_site_payload(site),
                value_kind=type(value).__name__,
                strict=strict,
                callable=callable(value),
                facet_scatter=True,
                direction="forward",
            )
            return self
        self._validate_intervention_site(site, strict=strict)
        metadata = {"created_by": "set_callable_one_shot"} if callable(value) else {}
        self._ensure_intervention_spec().add_set(
            self._target_spec_from_site(site, strict=strict),
            value,
            metadata=metadata,
        )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "set",
            site=self._history_site_payload(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
            direction="forward",
        )
        return self

    def attach_hooks(
        self: "Trace",
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
        direction: str | None = None,
    ) -> Any:
        """Attach sticky hooks to the current intervention spec.

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
        direction:
            Optional signal direction override: ``"forward"``, ``"backward"``,
            or ``"both"``.

        Returns
        -------
        Any
            Scoped removable hook handle.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import HookSignatureError
        from ..intervention.handles import HookHandle
        from ..intervention.hooks import normalize_hook_plan
        from ..intervention.spec import (
            InterventionSpec as PublicInterventionSpec,
            entries_from_spec,
        )

        if direction is not None and direction not in {"forward", "backward", "both"}:
            raise InvalidArgumentError(
                "attach_hooks(..., direction=...) must be 'forward', 'backward', or 'both'; "
                f"received {direction!r}",
                code="intervention_direction_invalid",
                remedy="pass direction='forward', 'backward', 'both', or None",
                argument="direction",
            )
        if isinstance(hooks_or_site, PublicInterventionSpec):
            # C03 spec door: the ONE immutable spec is accepted unchanged.
            # Its rules already carry site, action, and direction, so extra
            # call arguments have nothing sound to bind to.
            if hook is not None or extra_hooks or direction is not None:
                raise InvalidArgumentError(
                    "attach_hooks(spec) takes the InterventionSpec alone: its "
                    "rules already carry the site, action, and direction, so "
                    "extra hook/direction arguments have nothing to bind to",
                    code="spec_door_extra_arguments",
                    remedy="pass only the spec (build clauses with tl.when and "
                    "merge them), or use the (site, hook) call shape without a spec",
                    argument="hooks_or_site",
                )
            entries = entries_from_spec(hooks_or_site, door="attach_hooks")
        elif extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)],
                direction=cast(Any, direction),
                allow_replay_only_site_targets=True,
            )
        else:
            entries = normalize_hook_plan(
                hooks_or_site,
                hook,
                direction=cast(Any, direction),
                allow_replay_only_site_targets=True,
            )
        from ..intervention.hooks import expand_facet_hook_entries

        entries = expand_facet_hook_entries(self, entries)
        self._refuse_chunked_batch_coupling(entries)
        for entry in entries:
            self._validate_intervention_site(entry.site_target, strict=strict)
        spec = self._ensure_intervention_spec()
        handle_ids: list[str] = []
        for entry in entries:
            handle_id = f"hook-{uuid.uuid4().hex}"
            handle_ids.append(handle_id)
            metadata = dict(entry.metadata)
            if metadata.get("facet_write"):
                # Facet-slice entries MUST store the scatter wrapper built by
                # expand_facet_hook_entries as the fire-time hook: storing the raw
                # helper would drop the wrapper, and rerun normalization would then
                # apply the helper to the whole home tensor instead of the selected
                # facet slice. The raw helper is kept in ``helper=`` as provenance.
                stored_hook: Any = entry.normalized_callable
            else:
                stored_hook = (
                    entry.helper_spec
                    if entry.helper_spec is not None
                    else entry.normalized_callable
                )
            spec.add_hook(
                self._target_spec_from_site(entry.site_target, strict=strict),
                stored_hook,
                helper=entry.helper_spec,
                handle=handle_id,
                metadata=metadata,
                prepend=prepend,
            )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "attach_hooks",
            hook_count=len(entries),
            sites=tuple(self._history_site_payload(entry.site_target) for entry in entries),
            strict=strict,
            prepend=prepend,
            handles=tuple(handle_ids),
            direction=direction,
        )
        self._last_hook_handle_ids = tuple(handle_ids)
        return HookHandle(self, tuple(handle_ids), confirm_mutation=confirm_mutation)

    def _refuse_chunked_batch_coupling(self: "Trace", entries: list[Any]) -> None:
        """Refuse batch-coherent edits on a chunked-forward capture (D34).

        The capture door already refuses ``chunk_size`` with ``hooks=`` /
        ``intervene=``; the surviving hole is a batch-coherent helper applied
        in REPLAY to a trace whose capture ran in forward chunks: the hook
        would see one CHUNK and permute or average it as if it were the whole
        batch -- a silently wrong experiment. Whole-batch visibility is not
        provable from the chunk metadata in v1, so the guard refuses typed.
        An ordinary (unchunked) minibatch is a real batch and stays legal.

        Parameters
        ----------
        entries:
            Normalized hook entries about to attach.
        """

        if not getattr(self, "chunked_forward", False):
            return
        for entry in entries:
            helper = getattr(entry, "helper_spec", None)
            if helper is None:
                continue
            if dict(helper.metadata).get("batch_coherent"):
                raise InvalidArgumentError(
                    f"edit {helper.helper_name!r} reads or exchanges the WHOLE "
                    "batch at fire time, but this trace was captured in forward "
                    "chunks (chunked_forward=True): each fire would see one "
                    "chunk and treat it as the full batch",
                    code="chunked_replay_batch_coupling",
                    remedy="re-capture without chunk_size for whole-batch "
                    "edits, or apply a per-row edit (resample_rows_from) whose "
                    "value never couples rows",
                    argument="hooks_or_site",
                )

    def remove(self: "Trace") -> None:
        """Remove the most recent legacy-returned hook attachment.

        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """

        for handle_id in self._last_hook_handle_ids:
            self.detach_hooks(handle=handle_id, confirm_mutation=True)
        self._last_hook_handle_ids = ()

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

    def detach_hooks(
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

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import SpecMutationError

        if site is None and handle is None:
            if strict:
                raise SpecMutationError("detach_hooks requires a site or handle in strict mode.")
            return self

        target_spec = None
        if site is not None:
            self._validate_intervention_site(site, strict=strict)
            target_spec = self._target_spec_from_site(site, strict=strict)

        handle_values = (
            tuple(getattr(handle, "handle_ids", (str(handle),))) if handle is not None else (None,)
        )
        removed = 0
        for handle_value in handle_values:
            removed += self._ensure_intervention_spec().remove_hook(
                site_target=target_spec,
                handle=handle_value,
            )
        if removed == 0 and strict:
            raise SpecMutationError("detach_hooks did not match any sticky hooks.")
        if removed > 0:
            self._mark_intervention_spec_mutated()
        self._record_operation(
            "detach_hooks",
            site=self._history_site_payload(site) if site is not None else None,
            handle=str(handle) if handle is not None else None,
            removed=removed,
            strict=strict,
        )
        return self

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

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        self._ensure_intervention_spec().clear()
        self._mark_intervention_spec_mutated()
        self._record_operation("clear_hooks")
        return self

    @property
    def edges(self: "Trace") -> tuple[Any, ...]:
        """Return this trace's dataflow edge family (finalized edge views).

        One ``EdgeUseRecord`` per parent->child occurrence (parallel edges
        first-class), in execution order; rows are the identity-stable
        provenance records themselves (immutable finalized views). Requires
        an ``intervention_ready`` capture — otherwise refuses typed
        (``edge_provenance_unavailable``). DOCUMENTED-UNSTABLE spelling
        pending naming-session ratification.
        """

        from ..selection import _trace_edge_records

        return _trace_edge_records(self)

    def _settle_selection_do(
        self: "Trace", txn: "_DoTransaction", selected_engine: str, mutation_kind: str
    ) -> None:
        """Record a leaf-site selection edit and quarantine episode evidence.

        Selection edits propagate (or deliberately do not) inside the
        mutation step; no hook targets exist to push. A non-staged edit
        perturbed this product's values, so inherited episode step evidence
        no longer describes it (lane F42).
        """

        staged_only = selected_engine == "set_only" and mutation_kind not in (
            "selection_replayed",
            "selection_set",
        )
        self._record_do_intervention_event(txn, status=None, staged_only=staged_only)
        if not staged_only:
            from ..capture._episode_coupling import (
                quarantine_episode_after_perturbed_replay,
            )

            quarantine_episode_after_perturbed_replay(self)

    def do(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        x: Any = None,
        intervention: InterventionOptions | None = None,
        direction: str | None = None,
    ) -> "Trace":
        """Apply an intervention and dispatch to replay, rerun, or set-only.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional hook for the ``(site, hook)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``: ``engine``
            in ``"auto"``/``"replay"``/``"rerun"``/``"set_only"``,
            ``confirm_mutation``, ``strict``).
        direction:
            Optional signal direction override for hook-style mutations.

        Returns
        -------
        Trace
            This model log after the selected propagation engine runs.
        """

        from ..intervention.errors import EngineDispatchError

        if type(hooks_or_site).__name__ == "SiteTable":
            # find_sites() output is a first-class address (leverage B3/NEW-4):
            # the zero-match refusal recommends find_sites, so its result must
            # be accepted here, lowered to the exact resolved site labels.
            from ..ir.selector_eval import normalize_selector_like

            hooks_or_site = normalize_selector_like(hooks_or_site, lifecycle="live")
        intervention_options = merge_intervention_options(intervention=intervention)
        engine_value = intervention_options.engine
        confirm_mutation_value = intervention_options.confirm_mutation
        strict_value = intervention_options.strict

        if engine_value not in {"auto", "replay", "rerun", "set_only"}:
            raise InvalidArgumentError(
                "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'; "
                f"received {engine_value!r}",
                code="intervention_engine_invalid",
                remedy="pass engine='auto', 'replay', 'rerun', or 'set_only'",
                argument="engine",
            )

        selected_engine = self._select_do_engine(engine_value, model=model, x=x)
        if selected_engine == "rerun":
            if model is None:
                raise EngineDispatchError("do(..., engine='rerun') requires model= and x=.")
            self._validate_supplied_model_matches_capture(model)
        # C03 fire-evidence rule: ONE transaction envelope per do() attempt,
        # written on success, no-fire, AND failure -- intervention_audit is
        # never empty after an attempt. The monotonic bound scopes exactly
        # this transaction's FireRecords (the ONE builder stamps every one).
        txn = _DoTransaction(
            hooks_or_site=hooks_or_site,
            value_or_hook=value_or_hook,
            engine=selected_engine,
            transaction_start=time.monotonic(),
            audit_rows_before=len(self.intervention_audit),
        )
        try:
            mutation_kind, attached_handles = self._apply_do_mutation(
                hooks_or_site,
                value_or_hook,
                engine=selected_engine,
                strict=strict_value,
                confirm_mutation=confirm_mutation_value,
                direction=direction,
            )
        except BaseException as exc:
            self._record_do_intervention_event(txn, status="error", error=repr(exc))
            raise
        self._record_operation(
            "do",
            mutation_kind=mutation_kind,
            engine=selected_engine,
            requested_engine=engine_value,
            model_supplied=model is not None,
            x_supplied=x is not None,
            strict=strict_value,
            direction=direction,
        )

        if (
            mutation_kind in ("selection_replayed", "selection_set", "region_replayed")
            or selected_engine == "set_only"
        ):
            self._settle_selection_do(txn, selected_engine, mutation_kind)
            return self
        try:
            if selected_engine == "replay":
                result = self.push(replay=ReplayOptions(strict=strict_value))
            else:
                assert model is not None
                result = self.run(model, x, replay=ReplayOptions(strict=strict_value))
        except BaseException as exc:
            # A failed do() must not leave its sticky hooks attached: the
            # next push would silently re-fire them (contamination). Detach
            # exactly the hooks THIS call attached, then re-raise.
            for handle_id in attached_handles:
                self.detach_hooks(handle=handle_id, confirm_mutation=True)
            self._record_do_intervention_event(txn, status="error", error=repr(exc))
            raise
        self._record_do_intervention_event(txn, status=None)
        # Lane F42: the push/rerun replayed a perturbed cone -- inherited
        # episode step evidence no longer describes the edited product.
        from ..capture._episode_coupling import quarantine_episode_after_perturbed_replay

        quarantine_episode_after_perturbed_replay(result)
        return result

    def _record_do_intervention_event(
        self: "Trace",
        txn: _DoTransaction,
        *,
        status: str | None,
        error: str | None = None,
        staged_only: bool = False,
    ) -> None:
        """Write the ONE InterventionEvent envelope for one do() transaction.

        Parameters
        ----------
        txn:
            The transaction identity minted at the do() door (WHERE/edit
            arguments, selected engine, FireRecord time bound, and the
            audit-row watermark).
        status:
            Explicit status (``"error"``), or ``None`` to derive
            fired/no_fire from the transaction's FireRecords.
        error:
            Stringified failure for error rows.
        staged_only:
            Whether the edit was staged without any propagation attempt.
        """

        from ..intervention.audit import (
            fire_records_since,
            record_intervention_event,
            rules_payload,
            site_keys_for_labels,
        )
        from ..intervention.spec import InterventionSpec as PublicInterventionSpec, _canonical_repr

        hooks_or_site = txn.hooks_or_site
        value_or_hook = txn.value_or_hook
        engine = txn.engine
        fires = fire_records_since(self, txn.transaction_start) if status != "error" else []
        fired_labels = tuple(
            dict.fromkeys(record.call_label or record.target_label for record in fires)
        )
        if isinstance(hooks_or_site, PublicInterventionSpec):
            rules = rules_payload(hooks_or_site)
            edit_names = tuple(str(rule["action"]) for rule in rules)
            selection_repr = " | ".join(str(rule["where"]) for rule in rules)
            fired_action_reprs = {
                _canonical_repr(record.helper) for record in fires if record.helper is not None
            }
            zero_fire_rule_ids = tuple(
                str(rule["rule_id"])
                for rule in rules
                if str(rule["action"]) not in fired_action_reprs
            )
        elif self._is_selection_batch(hooks_or_site):
            # One transactional envelope for the whole batch (D33): per-pair
            # edits named in order, plus the kind-specific staged-store slots
            # disclosure so cross-kind completion needs no schema change.
            rules = ()
            edit_names = tuple(
                str(
                    getattr(
                        pair_edit,
                        "helper_name",
                        getattr(pair_edit, "__name__", type(pair_edit).__name__),
                    )
                )
                for _sel, pair_edit in hooks_or_site
            )
            selection_repr = " ; ".join(repr(sel) for sel, _pair_edit in hooks_or_site)
            zero_fire_rule_ids = ()
        else:
            rules = ()
            edit_name = getattr(
                value_or_hook,
                "helper_name",
                getattr(value_or_hook, "__name__", type(value_or_hook).__name__),
            )
            edit_names = (str(edit_name),)
            selection_repr = self._history_site_payload(hooks_or_site)
            if not isinstance(selection_repr, str):
                selection_repr = repr(selection_repr)
            zero_fire_rule_ids = ()
        if status is None:
            resolved_status = "fired" if fires else "no_fire"
        else:
            resolved_status = status
        extra: dict[str, Any] = {}
        if staged_only:
            extra["staged_only"] = True
        if self._is_selection_batch(hooks_or_site):
            extra["selection_batch"] = {
                "pairs": len(hooks_or_site),
                "kinds": ["ACT"] * len(hooks_or_site),
                # Kind-specific staged-store slots (D33): carried NOW so the
                # OPEN cross-kind completion item needs no schema change.
                "staged_stores": {"act": len(hooks_or_site), "param": 0, "edge": 0},
            }
        record_intervention_event(
            self,
            lane="set_only" if engine == "set_only" else engine,  # type: ignore[arg-type]
            door="do",
            edit_names=edit_names,
            selection_repr=str(selection_repr),
            status=resolved_status,  # type: ignore[arg-type]
            fire_count=len(fires),
            site_keys=site_keys_for_labels(self, fired_labels),
            rules=rules,
            zero_fire_rule_ids=zero_fire_rule_ids,
            error=error,
            extra=extra or None,
            append_audit_row=len(self.intervention_audit) == txn.audit_rows_before,
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

        fork = self._fork_trace(name=name)
        self._record_operation("fork", source_id=id(self), name=fork.trace_label)
        return fork

    def _record_operation(self: "Trace", op: str, **payload: Any) -> None:
        """Append a structured operation record to ``state_history``.

        Parameters
        ----------
        op:
            Operation name.
        **payload:
            Operation-specific metadata.

        Returns
        -------
        None
            The history list is mutated in place.
        """

        self.state_history.append(
            {
                "op": op,
                "spec_revision": self._spec_revision,
                "timestamp": time.monotonic(),
                **payload,
            }
        )

    def _warn_if_root_mutation(self: "Trace", *, confirm_mutation: bool) -> None:
        """Emit the once-per-root mutate-in-place warning when appropriate.

        Parameters
        ----------
        confirm_mutation:
            Whether the caller explicitly accepted in-place mutation.
        """

        if confirm_mutation or self.parent_run is not None or self._warned_mutate_in_place:
            return
        from ..intervention.errors import MutateInPlaceWarning
        from ..options import suppress_mutate_warnings

        if suppress_mutate_warnings.is_suppressed:
            return
        warnings.warn(
            "MutateInPlaceWarning: Trace mutators modify root logs in place. "
            "Use log.fork(...) for isolated edits or pass confirm_mutation=True.",
            MutateInPlaceWarning,
            stacklevel=3,
        )
        self._warned_mutate_in_place = True

    def _select_do_engine(self: "Trace", engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Resolve the concrete ``do`` engine from caller arguments.

        Parameters
        ----------
        engine:
            Requested engine name.
        model:
            Optional model supplied for rerun.
        x:
            Optional input supplied for rerun.

        Returns
        -------
        str
            Concrete engine name.

        Raises
        ------
        EngineDispatchError
            If the engine cannot be inferred from an incomplete model/input pair.
        """

        from ..intervention.errors import EngineDispatchError

        if engine != "auto":
            if engine == "rerun" and (model is None or x is None):
                raise EngineDispatchError(
                    "do(..., engine='rerun') requires both model= and x=. "
                    "Pass both, or use engine='replay' if full rerun is not intended."
                )
            return engine
        if (model is None) != (x is None):
            raise EngineDispatchError(
                "do(engine='auto') needs both model= and x= for rerun, or neither for "
                "replay. Pass both, or use engine='replay' if rerun is not intended."
            )
        return "rerun" if model is not None else "replay"

    def _apply_do_mutation(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
        direction: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Apply the mutation part of ``do`` and report its kind.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional value or hook.
        engine:
            Concrete engine selected by ``_select_do_engine``.
        strict:
            Whether selector checks should be strict.
        confirm_mutation:
            Whether root mutation warnings should be suppressed.
        direction:
            Optional signal direction override.

        Returns
        -------
        tuple[str, tuple[str, ...]]
            The mutation kind (``"set"``, ``"attach_hooks"``, or a
            ``"selection_*"`` kind) and the sticky-hook handle ids THIS call
            attached (for the caller's failure cleanup).
        """

        from ..intervention.spec import InterventionSpec as PublicInterventionSpec
        from ..selection import ResolvedSelection, Selection

        if isinstance(hooks_or_site, PublicInterventionSpec):
            # C03 spec door: do(spec) applies the whole immutable spec through
            # the attach path (one preflight, all rules or none), then the
            # caller's selected engine propagates.
            if value_or_hook is not None:
                raise InvalidArgumentError(
                    "do(spec, edit) conflicts: the InterventionSpec already "
                    "carries its actions, so a second edit argument has "
                    "nothing to bind to",
                    code="spec_door_extra_arguments",
                    remedy="pass only the spec to do(), or use the "
                    "(site, edit) call shape without a spec",
                    argument="value_or_hook",
                )
            handle = self.attach_hooks(
                hooks_or_site,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "attach_hooks", tuple(handle.handle_ids)

        # A RegionTarget (F01) is the region-as-a-unit door: the replay
        # lowering substitutes the region's derived exit values without
        # replaying the interior. Distinct from the TraceSlice lift below,
        # which keeps its shipped member-site family semantics. sys.modules
        # gate keeps ordinary do() free of the regions import.
        regions_module = sys.modules.get("torchlens.intervention.regions")
        if regions_module is not None and isinstance(hooks_or_site, regions_module.RegionTarget):
            self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
            return (
                regions_module.apply_region_do(
                    self, hooks_or_site, value_or_hook, engine=engine, strict=strict
                ),
                (),
            )
        # A TraceSlice targets its member family: lift it to the whole-site
        # QUERY (re-resolved on THIS trace by site name, so a slice built on
        # the source log addresses the same sites on a fork). sys.modules
        # gate keeps ordinary do() free of any slice import.
        slice_module = sys.modules.get("torchlens.trace_slice")
        if slice_module is not None and isinstance(hooks_or_site, slice_module.TraceSlice):
            hooks_or_site = hooks_or_site.__selection__()
        if self._is_selection_batch(hooks_or_site):
            self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
            if value_or_hook is not None:
                raise InvalidArgumentError(
                    "do([(selection, edit), ...], edit) conflicts: the batch "
                    "pairs already carry their edits, so a second edit "
                    "argument has nothing to bind to",
                    code="selection_batch_pair_invalid",
                    remedy="pass only the (selection, edit) pair list",
                    argument="value_or_hook",
                )
            return self._apply_selection_batch_do(
                hooks_or_site, engine=engine, strict=strict, direction=direction
            )
        if isinstance(hooks_or_site, (Selection, ResolvedSelection)):
            self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
            return self._apply_selection_do(
                hooks_or_site,
                value_or_hook,
                engine=engine,
                strict=strict,
                direction=direction,
            )
        if value_or_hook is not None and (engine == "set_only" or not callable(value_or_hook)):
            self.set(
                hooks_or_site,
                value_or_hook,
                direction=direction or "forward",
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set", ()
        handle = self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            direction=direction,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks", tuple(handle.handle_ids)

    @staticmethod
    def _is_selection_batch(value: Any) -> bool:
        """Whether a ``do()`` input is a selection-batch pair list (D33).

        A non-empty list/tuple of 2-item pairs with at least one explicit
        ``Selection``/``ResolvedSelection``/``TraceSlice`` in a pair's first
        slot. Legacy ``(site, hook)`` pair lists (selector sites) keep the
        shipped door untouched.
        """

        if not isinstance(value, (list, tuple)) or not value:
            return False
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
            return False
        from ..selection import ResolvedSelection, Selection

        slice_module = sys.modules.get("torchlens.trace_slice")
        slice_type = slice_module.TraceSlice if slice_module is not None else ()
        return any(
            isinstance(item[0], (Selection, ResolvedSelection, slice_type)) for item in value
        )

    def _apply_selection_batch_do(
        self: "Trace",
        pairs: Any,
        *,
        engine: str,
        strict: bool,
        direction: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Apply a selection-batch ``do([(selection, edit), ...])`` (D33, v1).

        ONE atomic transaction: every pair resolves and derives its masked
        edit first, ALL hooks attach together (a refused site mid-plan
        detaches everything already attached), the per-pair ACT audit rows
        append only after every attachment succeeded, and the caller's single
        engine pass propagates the whole batch in ONE replay. Donor sharing
        across clauses is plan OBJECT identity (D8): one ``SamplingPlan``
        reused in several pairs carries one content-digest ``donor_group_id``;
        distinct equal-content plan objects are disambiguated here by the
        batch normalizer (``assign_batch_donor_groups``,
        clause-selection-keyed suffixes; ``per_firing`` never suffixed)
        so kwargs coincidence never shares a group
        while a rerun of the same declared batch still reproduces its draws.

        v1 scope (capability-reported, never silently narrowed): ACT
        selections on interior sites only. Cross-kind batches (PARAM/EDGE)
        refuse typed -- edge substitution commits and pushes inside its
        per-entry loop today, so cross-kind atomicity does not exist to be
        reused; the completion item stays OPEN and is never relabeled
        complete. Leaf sites (inputs/buffers) commit-then-propagate per site
        and cannot join one hook transaction yet.

        Returns
        -------
        tuple[str, tuple[str, ...]]
            ``("selection_hooks", attached_handle_ids)``.
        """

        from ..selection import ResolvedSelection, Selection, _lift, build_selection_do_plan

        slice_module = sys.modules.get("torchlens.trace_slice")
        slice_type = slice_module.TraceSlice if slice_module is not None else ()
        normalized: list[tuple[Any, Any]] = []
        kinds: list[str] = []
        for index, (selection, edit) in enumerate(pairs):
            if isinstance(selection, slice_type):
                selection = selection.__selection__()
            if not isinstance(selection, (Selection, ResolvedSelection)):
                raise InvalidArgumentError(
                    f"selection-batch do() pair {index} carries a "
                    f"{type(selection).__name__} target; a batch mixing "
                    "Selections with legacy selector sites would run two "
                    "different engines under one call",
                    code="selection_batch_pair_invalid",
                    remedy="make every pair's first slot a Selection (lift "
                    "producers with .__selection__()), or use the legacy "
                    "(site, hook) list without Selections",
                    argument="hooks_or_site",
                )
            lifted = _lift(selection)
            kind = getattr(lifted, "kind", "ACT")
            kinds.append(kind)
            normalized.append((selection, edit))
        non_act = [f"pair {index}: {kind}" for index, kind in enumerate(kinds) if kind != "ACT"]
        if non_act:
            raise InvalidArgumentError(
                "selection-batch do() v1 is ACT-only; this batch carries "
                f"[{', '.join(non_act)}]. Capability report: ACT selections "
                "attach as ONE transaction; PARAM and EDGE selections commit "
                "and push inside their own doors today, so a cross-kind batch "
                "cannot be made atomic yet (the completion item is OPEN, not "
                "silently narrowed)",
                code="selection_batch_cross_kind",
                remedy="apply PARAM/EDGE selections in their own do() calls, one per kind",
                argument="hooks_or_site",
            )
        from ..intervention.stochastic import assign_batch_donor_groups

        normalized = assign_batch_donor_groups(normalized)
        plans: list[tuple[Any, list[dict[str, Any]], dict[str, Any]]] = []
        for index, (selection, edit) in enumerate(normalized):
            resolved, plan, audit = build_selection_do_plan(self, selection, edit)
            leaf_sites = [item["op"].label for item in plan if item["is_leaf"]]
            if leaf_sites:
                raise InvalidArgumentError(
                    f"selection-batch do() pair {index} addresses leaf sites "
                    f"{leaf_sites!r} (inputs/buffers): leaf edits commit and "
                    "propagate per site and cannot join one hook transaction "
                    "in v1",
                    code="selection_batch_leaf_unsupported",
                    remedy="apply leaf-site edits through single-selection "
                    "do() calls; batch interior sites only",
                    argument="hooks_or_site",
                )
            plans.append((resolved, plan, audit))
        all_hook_items = [item for _resolved, plan, _audit in plans for item in plan]
        attached = self._attach_selection_plan_hooks(
            all_hook_items, strict=strict, direction=direction
        )
        for _resolved, _plan, audit in plans:
            self.intervention_audit.append(audit)
        return "selection_hooks", attached

    def _apply_selection_do(
        self: "Trace",
        selection: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
        direction: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Apply a Selection-targeted edit under the mask-application contract.

        The selection resolves against this trace. Interior sites (sites with
        a replayable func) get the edit attached to the hook plan with the
        engine-owned edit-then-scatter wrapper (element masks) or unchanged
        (whole-site short-circuit). LEAF sites (inputs/buffers; no func to
        replay) get the edited value computed NOW from the saved value under
        the same scatter contract, committed transactionally, and propagated
        with origin-preserving replay. An audit record (query repr + resolve
        digest + per-site relations) is appended to ``intervention_audit``.
        Empty resolutions attach nothing (emptiness is disclosure, never an
        error).

        Returns
        -------
        tuple[str, tuple[str, ...]]
            The mutation kind -- ``"selection_hooks"`` (interior sites;
            caller runs the engine), ``"selection_replayed"`` (leaf sites
            already propagated), or ``"selection_set"`` (leaf values
            committed without propagation) -- plus the sticky-hook handle
            ids attached for the hook kind (empty otherwise).
        """

        from ..intervention.errors import EngineDispatchError
        from ..selection import _lift, build_selection_do_plan

        lifted = _lift(selection)
        if lifted is not None and lifted.kind == "EDGE":
            return (
                self._apply_selection_edge_do(
                    selection, lifted, edit, engine=engine, strict=strict
                ),
                (),
            )
        if lifted is not None and lifted.kind == "PARAM":
            return (
                self._apply_selection_param_do(
                    selection, lifted, edit, engine=engine, strict=strict
                ),
                (),
            )
        resolved, plan, audit = build_selection_do_plan(self, selection, edit)
        leaf_items = [item for item in plan if item["is_leaf"]]
        hook_items = [item for item in plan if not item["is_leaf"]]
        if leaf_items and hook_items:
            raise EngineDispatchError(
                "a selection plan mixing leaf sites (inputs/buffers) and interior "
                "sites cannot propagate as ONE replay transaction in v1: interior "
                "recomputation reads captured consumed values, so the leaf edit "
                "would be silently discarded at the interior site. Split the "
                "selection by site class."
            )
        if not plan:
            self.intervention_audit.append(audit)
            return "selection_replayed", ()
        if leaf_items:
            if engine not in ("replay", "set_only"):
                raise EngineDispatchError(
                    "leaf-site selection edits (inputs/buffers) ride the replay "
                    "engine (or set_only); for rerun, pass the modified input as x=."
                )
            self._apply_leaf_selection_edits(leaf_items)
            if engine == "replay":
                from ..intervention.replay import push_from

                for item in leaf_items:
                    push_from(self, item["op"], replay=ReplayOptions(strict=strict))
                self.intervention_audit.append(audit)
                return "selection_replayed", ()
            self.intervention_audit.append(audit)
            return "selection_set", ()
        attached = self._attach_selection_plan_hooks(hook_items, strict=strict, direction=direction)
        self.intervention_audit.append(audit)
        return "selection_hooks", attached

    def _attach_selection_plan_hooks(
        self: "Trace",
        hook_items: list[dict[str, Any]],
        *,
        strict: bool,
        direction: str | None,
    ) -> tuple[str, ...]:
        """Attach one selection plan's hooks transactionally.

        A refused site mid-plan must not leave earlier sites armed: on any
        failure the hooks this plan already attached are detached before the
        exception propagates. Returns the attached handle ids.
        """

        from ..intervention.selectors import label as label_selector

        attached: list[str] = []
        try:
            for item in hook_items:
                handle = self.attach_hooks(
                    label_selector(item["op"].label),
                    item["edit"],
                    strict=strict,
                    confirm_mutation=True,
                    direction=direction,
                )
                attached.extend(handle.handle_ids)
        except BaseException:
            for handle_id in attached:
                self.detach_hooks(handle=handle_id, confirm_mutation=True)
            raise
        # These sticky hooks are the do() plan's edit-then-scatter residue of
        # an edit applied NOW (the caller propagates and appends the audit
        # record), not user-staged future-run material, so the run(inputs=...)
        # staged-spec honesty gate must skip them: a value-edited fork gets the
        # PendingValueEditsWarning disclosure at the run door, never a refusal
        # (D1 2026-08-19).
        attached_ids = set(attached)
        for hook_spec in self._ensure_intervention_spec().hook_specs:
            if hook_spec.handle in attached_ids:
                hook_spec.metadata["selection_do_engine_owned"] = True
        self._mark_intervention_spec_mutated()
        return tuple(attached)

    def _apply_selection_edge_do(
        self: "Trace",
        selection: Any,
        lifted: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
    ) -> str:
        """Apply one EDGE-kind selection edit through the edge-substitution path."""

        from ..selection import ResolvedSelection, SelectionError

        if edit is None:
            raise ValueError(
                "do(selection, edit) requires an edit: pass an Edit/HelperSpec, "
                "a hook callable, or a replacement tensor."
            )
        if isinstance(lifted, ResolvedSelection):
            if lifted._trace is not self:
                raise SelectionError(
                    "the resolved edge selection is bound to a different trace.",
                    code="selection_trace_mismatch",
                )
            resolved_edges = lifted
        else:
            resolved_edges = lifted.resolve(self)
        from ..intervention.edge_substitution import apply_edge_substitution_do

        payload = apply_edge_substitution_do(
            self, resolved_edges, edit, engine=engine, strict=strict
        )
        self.intervention_audit.append(
            {
                "kind": "EDGE",
                "selection_repr": repr(selection),
                "resolve_digest": resolved_edges.resolve_digest,
                "edit": getattr(edit, "helper_name", getattr(edit, "__name__", "value")),
                **payload,
            }
        )
        return "selection_replayed"

    def _apply_selection_param_do(
        self: "Trace",
        selection: Any,
        lifted: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
    ) -> str:
        """Apply one PARAM-kind selection edit through parameter substitution.

        The edit is applied "as if" the parameter were changed: every
        consumption of the parameter is substituted at its derived occurrence
        address on the replay engine, and the live parameter object is never
        written (decided 2026-08-17, superseding the D3 typed-refusal
        default on the replay path; rerun/set_only keep refusing typed).
        """

        from ..selection import ResolvedSelection, SelectionError

        if edit is None:
            raise ValueError(
                "do(selection, edit) requires an edit: pass an Edit/HelperSpec, "
                "a hook callable, or a replacement tensor."
            )
        if isinstance(lifted, ResolvedSelection):
            if lifted._trace is not self:
                raise SelectionError(
                    "the resolved parameter selection is bound to a different trace.",
                    code="selection_trace_mismatch",
                )
            resolved_params = lifted
        else:
            resolved_params = lifted.resolve(self)
        from ..intervention.param_substitution import apply_param_substitution_do

        payload = apply_param_substitution_do(
            self, resolved_params, edit, engine=engine, strict=strict
        )
        self.intervention_audit.append(
            {
                "kind": "PARAM",
                "selection_repr": repr(selection),
                "resolve_digest": resolved_params.resolve_digest,
                "edit": getattr(edit, "helper_name", getattr(edit, "__name__", "value")),
                **payload,
            }
        )
        return "selection_replayed"

    def _apply_leaf_selection_edits(self: "Trace", leaf_items: list[dict[str, Any]]) -> None:
        """Compute + commit leaf-site edited values under the scatter contract.

        The edit hook computes its full replacement from the SAVED leaf value
        (helpers stay mask-oblivious); the engine wrapper scatters selected
        elements onto a fresh tensor. Commit rides the transactional replay
        commit helper (snapshot + rollback), minting FireRecords so
        disclosure matches the hook path.
        """

        from ..intervention.audit import build_fire_record
        from ..intervention.hooks import make_hook_context
        from ..intervention.replay import _commit_replay_updates, _replay_site_key
        from ..intervention.types import HelperSpec
        from ..selection import _apply_invalid

        pending_updates: dict[str, torch.Tensor] = {}
        pending_records: dict[str, list[Any]] = {}
        for item in leaf_items:
            op = item["op"]
            derived = item["edit"]
            saved = op.out
            if not isinstance(saved, torch.Tensor):
                raise _apply_invalid(
                    "not_maskable",
                    f"leaf site {op.label!r} has no saved tensor value to edit.",
                    site=op.label,
                )
            if isinstance(derived, HelperSpec):
                if derived.factory is None:
                    raise _apply_invalid(
                        "not_maskable",
                        f"edit {derived.helper_name!r} has no runtime factory.",
                        site=op.label,
                    )
                hook_callable = derived.factory()
                helper_spec: HelperSpec | None = derived
                helper_name = derived.helper_name
            else:
                hook_callable = derived
                helper_spec = None
                helper_name = getattr(derived, "__name__", "hook")
            context = make_hook_context(
                name=helper_name,
                timing="post",
                direction="forward",
                layer_log=op,
                run_ctx={},
                args=(saved,),
                kwargs={},
            )
            applied = hook_callable(saved, hook=context)
            if not isinstance(applied, torch.Tensor):
                raise _apply_invalid(
                    "not_maskable",
                    f"edit at leaf site {op.label!r} produced a non-tensor "
                    f"({type(applied).__name__}).",
                    site=op.label,
                )
            pending_updates[_replay_site_key(op)] = applied
            pending_records[_replay_site_key(op)] = [
                build_fire_record(
                    target_label=op.layer_label,
                    call_label=op.label,
                    func_call_id=op.func_call_id,
                    container_path=tuple(op.container_path or ()),
                    engine="replay",
                    helper=helper_spec,
                    site_label=op.layer_label,
                    timing="post",
                    direction="forward",
                    helper_name=helper_name,
                    run_ctx=context.run_ctx if hasattr(context, "run_ctx") else None,
                    replaced=applied is not saved,
                )
            ]
        _commit_replay_updates(self, pending_updates, pending_records)

    def _validate_supplied_model_matches_capture(self: "Trace", model: nn.Module) -> None:
        """Validate rerun model evidence against the captured source model.

        Parameters
        ----------
        model:
            Candidate model for rerun.

        Raises
        ------
        ModelMismatchError
            Fail-closed on the root entry-point fact (absent fact or a
            non-``module_call`` root, F41), then if available class or
            weight-fingerprint evidence differs.
        """

        from ..intervention.errors import ModelMismatchError
        from ..user_funcs import _fingerprint_model_weights, _qualname_for_model

        # F41 (foldA D10/D11): the root entry-point fact joins this gate
        # FAIL-CLOSED -- it is written unconditionally on every capture, so
        # absence (a legacy artifact) refuses rather than silently skipping,
        # and a non-module_call root refuses rather than re-running
        # ``forward`` under a capture of a different entry point (supplying
        # ``owner`` for a capture of ``owner.generate`` would pass the
        # class+weights checks and run ``__call__`` under a report that can
        # say verified -- the silent-wrongness door the ruling closes).
        # Interim posture: bound-method captures refuse rerun/append; never
        # widened past the ruling.
        root_fact = getattr(self, "root_entry_point", None)
        if root_fact is None:
            raise ModelMismatchError(
                "rerun/append identity gate is fail-closed on the root "
                "entry-point fact, and this trace carries none (a legacy "
                "artifact saved before the fact was written unconditionally). "
                "The gate cannot prove the supplied model is the captured "
                "entry point. Remedy: re-capture with a current TorchLens "
                "(every capture writes Trace.root_entry_point), or use the "
                "replay engine, which re-executes recorded ops and needs no "
                "live model",
                code="root_entry_point_unavailable",
            )
        root_kind = str(root_fact).partition(":")[0]
        if root_kind != "module_call":
            bound_hint = (
                "this capture's root is the bound method "
                f"{str(root_fact).partition(':')[2]!r}, and re-running a "
                "supplied module would execute forward/__call__ -- a "
                "DIFFERENT entry point -- under a report that could claim "
                "fidelity. Bound-method captures refuse rerun/append in this "
                "release (interim posture, foldA D11). "
                if root_kind == "bound_method"
                else (
                    f"this capture's root entry point is {root_fact!r}, which "
                    "rerun cannot re-execute (rerun runs the supplied "
                    "module's forward). "
                )
            )
            raise ModelMismatchError(
                "rerun/append cannot re-execute this capture's entry point: "
                + bound_hint
                + "Remedy: use the replay engine (re-executes recorded ops, "
                "no live model), or re-capture from the plain module root "
                "(tl.trace(model, ...)) if module-forward rerun is what you "
                "want",
                code="rerun_entry_point_unsupported",
                root_entry_point=str(root_fact),
            )

        expected_class = getattr(self, "model_class_qualname", None)
        actual_class = _qualname_for_model(model)
        if expected_class is not None and actual_class != expected_class:
            raise ModelMismatchError(
                "Supplied model class does not match captured model class: "
                f"expected {expected_class!r}, got {actual_class!r}."
            )

        expected_fingerprint = getattr(self, "param_hash_quick", None)
        if expected_fingerprint is None:
            return
        actual_fingerprint = _fingerprint_model_weights(model)
        if actual_fingerprint != expected_fingerprint:
            raise ModelMismatchError(
                "Supplied model weight fingerprint does not match captured model weights."
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
        """Return a deterministic default fork name for this parent log."""

        base_name = self.trace_label or "trace"
        fork_count = sum(
            1
            for record in self.state_history
            if isinstance(record, dict) and record.get("op") == "fork"
        )
        return f"{base_name}_fork_{fork_count + 1}"

    def _rebind_fork_owner_refs(self: "Trace") -> None:
        """Rebind weak owner references on child objects to this trace.

        Used by the fork builder and by the rerun-refresh paths, which
        replace record state wholesale and must re-point every child's
        owner reference (and drop stale facet caches) afterwards.
        """

        for layer_pass in self.layer_list:
            layer_pass.source_trace = self
            del layer_pass.facets
        for layer_log in self.layer_logs.values():
            layer_log.source_trace = self
        for module_log in self.modules:
            module_log._source_trace = self
            module_log.__dict__.pop("_facets_cache", None)
            for module_call in module_log.calls.values():
                module_call._source_trace = self

    def _recipe_is_clean(self: "Trace") -> bool:
        """Return whether propagated outs match the current spec revision.

        Returns
        -------
        bool
            ``True`` when the current out recipe revision equals the
            mutable intervention spec revision.
        """

        return self._spec_revision == self._out_recipe_revision

    def _ensure_intervention_spec(self: "Trace") -> InterventionSpec:
        """Return the mutable intervention spec, creating one if needed.

        Returns
        -------
        InterventionSpec
            Mutable intervention recipe owned by this log.
        """

        if self._intervention_spec is None:
            self._intervention_spec = InterventionSpec()
        return self._intervention_spec

    def _mark_intervention_spec_mutated(self: "Trace") -> None:
        """Invalidate cached frozen views and mark the spec stale.

        Returns
        -------
        None
            This model log is mutated in place.
        """

        self._spec_revision += 1
        self.__dict__.pop("intervention_spec", None)
        self.__dict__.pop("_frozen_intervention_spec", None)
        self.__dict__.pop("_cached_frozen_intervention_spec", None)
        self.state = TraceState.SPEC_STALE

    def _validate_intervention_site(self: "Trace", site: Any, *, strict: bool) -> None:
        """Validate that a mutator site resolves on this log.

        Parameters
        ----------
        site:
            Selector-like target to validate.
        strict:
            Whether selector resolution should be strict.

        Returns
        -------
        None
            Raises when the site cannot resolve.
        """

        from ..intervention.errors import SiteResolutionError
        from ..intervention.resolver import _selector_resolution_direction

        # L1 narrowing (r3): only a typed direction-classification refusal may
        # fall through to the strict forward resolution below (which raises its
        # own typed refusal); any other exception is a bug and propagates.
        try:
            if _selector_resolution_direction(site) == "backward":
                return
        except SiteResolutionError:
            pass
        max_fanout = max(1, len(self.layer_list))
        self.resolve_sites(site, strict=strict, max_fanout=max_fanout)

    def _target_spec_from_site(self: "Trace", site: Any, *, strict: bool) -> TargetSpec:
        """Convert a selector-like site to a mutable target spec.

        Parameters
        ----------
        site:
            Selector-like target, target spec, or layer pass.
        strict:
            Whether the resulting target should carry strict resolution.

        Returns
        -------
        TargetSpec
            Mutable target spec stored in the intervention recipe.
        """

        from ..intervention.hooks import lower_record_site_target

        if isinstance(site, TargetSpec):
            target = copy.copy(site)
            target.strict = strict or target.strict
            return target
        # An ``Op``/``Layer`` RECORD lowers to its PASS-QUALIFIED label
        # selector(s) first (AUD-CODE 4.10, W051-REPLAY out-of-fence item 2):
        # an ``Op`` is exactly its own pass (``site.label``, never the bare
        # ``layer_label`` = the LAST pass of a multi-pass layer); a multi-pass
        # ``Layer`` is the explicit all-passes composite. Selector-likes pass
        # through unchanged.
        site = lower_record_site_target(site)
        if hasattr(site, "to_target_spec"):
            target = site.to_target_spec()
            target.strict = strict or target.strict
            return cast("TargetSpec", target)
        if hasattr(site, "layer_label"):
            return TargetSpec("label", str(site.layer_label), strict=strict)
        return TargetSpec("label", site, strict=strict)
