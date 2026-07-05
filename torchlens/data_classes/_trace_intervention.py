"""Trace intervention mixin."""

import copy
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
import time
import uuid
import weakref
import warnings
from typing import TYPE_CHECKING, Any, Set, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from .trace import Trace

from .. import _state
from .._deprecations import MISSING, MissingType
from .._trace_state import TraceState
from ..intervention.types import (
    ForkFieldPolicy,
    FrozenInterventionSpec,
    InterventionSpec,
    MODEL_LOG_FIELD_FORK_POLICY,
    TargetSpec,
)
from ..options import InterventionOptions, merge_intervention_options
from .layer import Layer, OpAccessor
from .op import Op
from ._state_adapter import state_items, state_new, state_restore


class TraceInterventionMixin:
    def save_intervention(
        self,
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

        save_intervention(
            self,
            path,
            level=level,
            allow_direct_writes=allow_direct_writes,
            overwrite=overwrite,
        )

    @cached_property
    def intervention_spec(self) -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def set(
        self,
        site: Any,
        value: Any,
        *,
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
                strict=strict,
                confirm_mutation=True,
            )
            self._record_operation(
                "set",
                site=repr(site),
                value_kind=type(value).__name__,
                strict=strict,
                callable=callable(value),
                facet_scatter=True,
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
            site=repr(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
        )
        return self

    def attach_hooks(
        self,
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
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

        Returns
        -------
        Any
            Scoped removable hook handle.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import HookSignatureError
        from ..intervention.handles import HookHandle
        from ..intervention.hooks import normalize_hook_plan

        if extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)]
            )
        else:
            entries = normalize_hook_plan(hooks_or_site, hook)
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
            spec.add_hook(
                self._target_spec_from_site(entry.site_target, strict=strict),
                entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
                helper=entry.helper_spec,
                handle=handle_id,
                metadata=metadata,
                prepend=prepend,
            )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "attach_hooks",
            hook_count=len(entries),
            sites=tuple(repr(entry.site_target) for entry in entries),
            strict=strict,
            prepend=prepend,
            handles=tuple(handle_ids),
        )
        self._last_hook_handle_ids = tuple(handle_ids)
        scoped_handle = HookHandle(self, tuple(handle_ids), confirm_mutation=confirm_mutation)
        if extra_hooks:
            return scoped_handle
        return self

    def remove(self) -> None:
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

    def __enter__(self) -> "Trace":
        """Enter a legacy scoped hook attachment.

        Returns
        -------
        Trace
            This log, acting as the most recent hook handle.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
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
        self,
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
            site=repr(site) if site is not None else None,
            handle=str(handle) if handle is not None else None,
            removed=removed,
            strict=strict,
        )
        return self

    def clear_hooks(self, *, confirm_mutation: bool = False) -> "Trace":
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

    def do(
        self,
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        x: Any = None,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        intervention: InterventionOptions | None = None,
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
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        strict:
            Whether selector and propagation checks should raise.

        Returns
        -------
        Trace
            This model log after the selected propagation engine runs.
        """

        from ..intervention.errors import EngineDispatchError

        intervention_options = merge_intervention_options(
            intervention=intervention,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
        )
        engine_value = intervention_options.engine
        confirm_mutation_value = intervention_options.confirm_mutation
        strict_value = intervention_options.strict

        if engine_value not in {"auto", "replay", "rerun", "set_only"}:
            raise ValueError(
                "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'."
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
        )
        self._record_operation(
            "do",
            mutation_kind=mutation_kind,
            engine=selected_engine,
            requested_engine=engine_value,
            model_supplied=model is not None,
            x_supplied=x is not None,
            strict=strict_value,
        )

        if selected_engine == "set_only":
            return self
        if selected_engine == "replay":
            return self.push(strict=strict_value)
        assert model is not None
        return self.run(model, x, strict=strict_value)

    def fork(self, name: str | None = None) -> "Trace":
        """Create a copy-on-write intervention fork of this log.

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

    def _record_operation(self, op: str, **payload: Any) -> None:
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

    def _warn_if_root_mutation(self, *, confirm_mutation: bool) -> None:
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

    def _select_do_engine(self, engine: str, *, model: nn.Module | None, x: Any) -> str:
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
        self,
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
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

        Returns
        -------
        str
            ``"set"`` or ``"attach_hooks"``.
        """

        if engine == "set_only" and value_or_hook is not None:
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        if value_or_hook is not None and not callable(value_or_hook):
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks"

    def _validate_supplied_model_matches_capture(self, model: nn.Module) -> None:
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
        self,
        *,
        name: str | None,
        deep_copy_layer_labels: Set[str] | None = None,
    ) -> "Trace":
        """Build a forked Trace with policy-driven field handling.

        Parameters
        ----------
        name:
            Optional fork name.
        deep_copy_layer_labels:
            Optional layer labels that require full Op field copies. When
            provided, other Op shells are still forked but their fields use a
            shallow copy to avoid deep-copying untouched replay context.

        Returns
        -------
        Trace
            Forked log whose mutable containers are independent.
        """

        fork = state_new(type(self))
        fork_state = {
            field_name: self._fork_model_field(field_name, value)
            for field_name, value in state_items(self)
        }
        state_restore(fork, fork_state)
        fork.parent_run = weakref.ref(self)
        fork.trace_label = name or self._next_fork_name()
        fork._intervention_spec = copy.deepcopy(self._ensure_intervention_spec())
        fork.state_history = copy.deepcopy(self.state_history)
        fork.relationship_evidence = copy.deepcopy(self.relationship_evidence)
        fork._out_recipe_revision = self._out_recipe_revision
        fork._spec_revision = self._spec_revision
        fork.state = self.state
        fork._warned_mutate_in_place = False
        fork._warned_direct_write = False

        layer_map = fork._fork_layer_ops_from(
            self,
            deep_copy_layer_labels=deep_copy_layer_labels,
        )
        fork._rebuild_fork_layer_collections(self, layer_map)
        fork._rebind_fork_owner_refs()
        _state._register_log(fork)
        return fork

    def _next_fork_name(self) -> str:
        """Return a deterministic default fork name for this parent log."""

        base_name = self.trace_label or "trace"
        fork_count = sum(
            1
            for record in self.state_history
            if isinstance(record, dict) and record.get("op") == "fork"
        )
        return f"{base_name}_fork_{fork_count + 1}"

    def _fork_model_field(self, field_name: str, value: Any) -> Any:
        """Apply the Trace fork policy to a single field.

        Parameters
        ----------
        field_name:
            Field being copied.
        value:
            Current field value.

        Returns
        -------
        Any
            Field value for the fork.
        """

        policy = MODEL_LOG_FIELD_FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _fork_layer_ops_from(
        self,
        parent: "Trace",
        *,
        deep_copy_layer_labels: Set[str] | None = None,
    ) -> dict[int, Op]:
        """Fork every Op and return an old-object-id map.

        Parameters
        ----------
        parent:
            Parent log whose layer ops are being forked.
        deep_copy_layer_labels:
            Optional layer labels that require normal fork policy copies. Ops
            outside this set receive distinct shells with shallow-copied fields.

        Returns
        -------
        dict[int, Op]
            Mapping from ``id(parent_pass)`` to forked pass.
        """

        layer_map: dict[int, Op] = {}
        fork_equivalent_ops = self.op_equivalence_classes
        for parent_pass in parent.layer_list:
            fork_pass = state_new(Op)
            deep_copy_fields = (
                deep_copy_layer_labels is None or parent_pass.layer_label in deep_copy_layer_labels
            )
            state_restore(
                fork_pass,
                {
                    field_name: self._fork_layer_pass_field(
                        field_name,
                        value,
                        deep_copy=deep_copy_fields,
                    )
                    for field_name, value in state_items(parent_pass)
                },
            )
            fork_pass.source_trace = self
            eq_type = getattr(fork_pass, "equivalence_class", None)
            if eq_type in fork_equivalent_ops:
                fork_pass.equivalent_ops = fork_equivalent_ops[eq_type]
            object.__setattr__(fork_pass, "_construction_done", True)
            layer_map[id(parent_pass)] = fork_pass
        return layer_map

    def _fork_layer_pass_field(self, field_name: str, value: Any, *, deep_copy: bool = True) -> Any:
        """Apply the Op fork policy to a single field.

        Parameters
        ----------
        field_name:
            Op field being copied.
        value:
            Current field value.
        deep_copy:
            Whether to honor the normal copy policy. False shares field values
            for untouched differentiable-replay ops while still creating a
            distinct Op shell.

        Returns
        -------
        Any
            Field value for the forked pass.
        """

        if field_name == "_source_trace_ref":
            return None
        if not deep_copy:
            return self._copy_shallow_fork_value(value)
        policy = Op.FIELD_FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _rebuild_fork_layer_collections(self, parent: "Trace", layer_map: dict[int, Op]) -> None:
        """Rebuild layer lookup containers so they point at forked ops.

        Parameters
        ----------
        parent:
            Parent log whose containers are being mirrored.
        layer_map:
            Mapping from parent pass object id to forked pass.
        """

        def remap_pass(value: Any) -> Any:
            """Map a parent pass object to its forked counterpart.

            Parameters
            ----------
            value:
                Candidate parent-layer object or another value.

            Returns
            -------
            Any
                Forked layer pass when ``value`` is known, otherwise ``value``.
            """
            return layer_map.get(id(value), value)

        self.layer_list = [remap_pass(layer) for layer in parent.layer_list]
        self.layer_dict_main_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_main_keys.items()
        )
        self.layer_dict_all_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_all_keys.items()
        )
        fork_layer_logs: dict[str, Layer] = OrderedDict()
        for label, parent_layer in parent.layer_logs.items():
            fork_layer_log = state_new(type(parent_layer))
            state_restore(
                fork_layer_log,
                {key: self._copy_fork_value(value) for key, value in state_items(parent_layer)},
            )
            fork_layer_log.source_trace = self
            fork_layer_log.ops = OpAccessor(
                OrderedDict(
                    (call_index, remap_pass(layer_pass))
                    for call_index, layer_pass in parent_layer.ops.items()
                )
            )
            if getattr(fork_layer_log, "equivalence_class", None) in self.op_equivalence_classes:
                fork_layer_log.equivalent_ops = self.op_equivalence_classes[
                    fork_layer_log.equivalence_class
                ]
            fork_layer_logs[label] = fork_layer_log
        self.layer_logs = fork_layer_logs

    def _rebind_fork_owner_refs(self) -> None:
        """Rebind weak owner references on forked child objects to this fork."""

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

    @staticmethod
    def _copy_fork_value(value: Any) -> Any:
        """Copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.

        Returns
        -------
        Any
            Fork-safe copy.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            return copy.copy(value)

    @staticmethod
    def _copy_shallow_fork_value(value: Any) -> Any:
        """Shallow-copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.

        Returns
        -------
        Any
            Lightweight fork-safe copy for replay-unaffected Op fields.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return value
        try:
            return copy.copy(value)
        except Exception:
            return value

    @staticmethod
    def _default_fork_policy(value: Any) -> ForkFieldPolicy:
        """Choose a conservative fork policy for fields outside policy tables.

        Parameters
        ----------
        value:
            Field value without an explicit policy.

        Returns
        -------
        ForkFieldPolicy
            Default share/copy decision.
        """

        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return ForkFieldPolicy.FORK_SHARE
        if isinstance(value, torch.Tensor) or callable(value):
            return ForkFieldPolicy.FORK_SHARE
        return ForkFieldPolicy.FORK_COPY

    def _recipe_is_clean(self) -> bool:
        """Return whether propagated outs match the current spec revision.

        Returns
        -------
        bool
            ``True`` when the current out recipe revision equals the
            mutable intervention spec revision.
        """

        return self._spec_revision == self._out_recipe_revision

    def _ensure_intervention_spec(self) -> InterventionSpec:
        """Return the mutable intervention spec, creating one if needed.

        Returns
        -------
        InterventionSpec
            Mutable intervention recipe owned by this log.
        """

        if self._intervention_spec is None:
            self._intervention_spec = InterventionSpec()
        return self._intervention_spec

    def _mark_intervention_spec_mutated(self) -> None:
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

    def _validate_intervention_site(self, site: Any, *, strict: bool) -> None:
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

        from ..intervention.resolver import _selector_resolution_direction

        try:
            if _selector_resolution_direction(site) == "backward":
                return
        except Exception:
            pass
        max_fanout = max(1, len(self.layer_list))
        self.resolve_sites(site, strict=strict, max_fanout=max_fanout)

    def _target_spec_from_site(self, site: Any, *, strict: bool) -> TargetSpec:
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
