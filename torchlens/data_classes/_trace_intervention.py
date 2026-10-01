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
from typing import TYPE_CHECKING, Any, cast

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


class TraceInterventionMixin(_TraceMixinBase):
    """``Trace`` intervention surface: spec save/load, fork, replay, and rerun."""

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

        if direction is not None and direction not in {"forward", "backward", "both"}:
            raise InvalidArgumentError(
                "attach_hooks(..., direction=...) must be 'forward', 'backward', or 'both'; "
                f"received {direction!r}",
                code="intervention_direction_invalid",
                remedy="pass direction='forward', 'backward', 'both', or None",
                argument="direction",
            )
        if extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)],
                direction=cast(Any, direction),
            )
        else:
            entries = normalize_hook_plan(hooks_or_site, hook, direction=cast(Any, direction))
        from ..intervention.hooks import expand_facet_hook_entries

        entries = expand_facet_hook_entries(self, entries)
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
        mutation_kind = self._apply_do_mutation(
            hooks_or_site,
            value_or_hook,
            engine=selected_engine,
            strict=strict_value,
            confirm_mutation=confirm_mutation_value,
            direction=direction,
        )
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

        if mutation_kind in ("selection_replayed", "selection_set"):
            # Leaf-site selection edits propagate (or deliberately do not)
            # inside the mutation step; no hook targets exist to push.
            return self
        if selected_engine == "set_only":
            return self
        if selected_engine == "replay":
            return self.push(replay=ReplayOptions(strict=strict_value))
        assert model is not None
        return self.run(model, x, replay=ReplayOptions(strict=strict_value))

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
    ) -> str:
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
        str
            ``"set"``, ``"attach_hooks"``, or ``"selection_hooks"``.
        """

        from ..selection import ResolvedSelection, Selection

        # A TraceSlice targets its member family: lift it to the whole-site
        # QUERY (re-resolved on THIS trace by site name, so a slice built on
        # the source log addresses the same sites on a fork). sys.modules
        # gate keeps ordinary do() free of any slice import.
        slice_module = sys.modules.get("torchlens.trace_slice")
        if slice_module is not None and isinstance(hooks_or_site, slice_module.TraceSlice):
            hooks_or_site = hooks_or_site.__selection__()
        if isinstance(hooks_or_site, (Selection, ResolvedSelection)):
            self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
            return self._apply_selection_do(
                hooks_or_site,
                value_or_hook,
                engine=engine,
                strict=strict,
                direction=direction,
            )
        if engine == "set_only" and value_or_hook is not None:
            self.set(
                hooks_or_site,
                value_or_hook,
                direction=direction or "forward",
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        if value_or_hook is not None and not callable(value_or_hook):
            self.set(
                hooks_or_site,
                value_or_hook,
                direction=direction or "forward",
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            direction=direction,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks"

    def _apply_selection_do(
        self: "Trace",
        selection: Any,
        edit: Any,
        *,
        engine: str,
        strict: bool,
        direction: str | None,
    ) -> str:
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
        str
            ``"selection_hooks"`` (interior sites; caller runs the engine),
            ``"selection_replayed"`` (leaf sites already propagated), or
            ``"selection_set"`` (leaf values committed without propagation).
        """

        from ..intervention.errors import EngineDispatchError
        from ..intervention.selectors import label as label_selector
        from ..selection import _lift, build_selection_do_plan

        lifted = _lift(selection)
        if lifted is not None and lifted.kind == "EDGE":
            return self._apply_selection_edge_do(
                selection, lifted, edit, engine=engine, strict=strict
            )
        if lifted is not None and lifted.kind == "PARAM":
            return self._apply_selection_param_do(
                selection, lifted, edit, engine=engine, strict=strict
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
            return "selection_replayed"
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
                return "selection_replayed"
            self.intervention_audit.append(audit)
            return "selection_set"
        for item in hook_items:
            self.attach_hooks(
                label_selector(item["op"].label),
                item["edit"],
                strict=strict,
                confirm_mutation=True,
                direction=direction,
            )
        self.intervention_audit.append(audit)
        return "selection_hooks"

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
        written (JMT ruling 2026-08-17, superseding the D3 typed-refusal
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

        from ..intervention.hooks import make_hook_context
        from ..intervention.replay import _commit_replay_updates, _replay_site_key
        from ..intervention.types import FireRecord, HelperSpec
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
                FireRecord(
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
            If available class or weight-fingerprint evidence differs.
        """

        from ..intervention.errors import ModelMismatchError
        from ..user_funcs import _fingerprint_model_weights, _qualname_for_model

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

        if isinstance(site, TargetSpec):
            target = copy.copy(site)
            target.strict = strict or target.strict
            return target
        if hasattr(site, "to_target_spec"):
            target = site.to_target_spec()
            target.strict = strict or target.strict
            return cast("TargetSpec", target)
        if hasattr(site, "layer_label"):
            return TargetSpec("label", str(site.layer_label), strict=strict)
        return TargetSpec("label", site, strict=strict)
