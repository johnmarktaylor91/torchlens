"""Reversible forward-pre-hook input provenance for PyTorch eager capture."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.modules.module as torch_module

from ... import _state
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.module import HookInfo
from ...data_classes.prehook import (
    ModuleInputSnapshot,
    PreHookEffect,
    TensorInputObservation,
    TypedInputPath,
)
from ...ir import PreHookProvenanceEvent
from ._tl import get_tensor_label

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

_INCOMPLETE_VERSION = "tensor_version_unavailable"
_INCOMPLETE_BYPASS = "registration_interposition_bypassed"
_IMMUTABLE_LEAF_TYPES = (str, bytes, int, float, bool, complex, type(None))


@dataclass(frozen=True)
class _TensorState:
    """Cheap live tensor observation used for transition comparison."""

    path: TypedInputPath
    value: torch.Tensor
    object_id: int
    version: int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    source_op: str | None


@dataclass(frozen=True)
class _StateObservation:
    """Metadata-only observation of one logical args/kwargs state."""

    structure: tuple[Any, ...]
    tensors: tuple[_TensorState, ...]
    leaves: tuple[tuple[TypedInputPath, str, Any], ...]
    incomplete_reasons: tuple[str, ...]


@dataclass
class _InvocationToken:
    """Mutable correlation token for one re-entrant module invocation."""

    module: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    before_snapshot: ModuleInputSnapshot
    before_observation: _StateObservation
    effects: list[PreHookEffect] = field(default_factory=list)
    incomplete_reasons: set[str] = field(default_factory=set)
    bound: bool = False
    event_emitted: bool = False


@dataclass(frozen=True)
class _RegistryReplacement:
    """One identity-guarded registry value replacement."""

    registry: dict[int, Callable[..., Any]]
    hook_id: int
    original: Callable[..., Any]
    wrapper: Callable[..., Any]


@dataclass(frozen=True)
class _InstanceAttributeReplacement:
    """One identity-guarded instance attribute replacement."""

    owner: object
    name: str
    owned_before: bool
    original: Any
    wrapper: Any


@dataclass
class PreHookProvenanceLedger:
    """Strong session-only ledger for all reversible interposition."""

    trace: "Trace"
    model: nn.Module
    modules: tuple[nn.Module, ...]
    prepared_ids: frozenset[int]
    local_registration_original: Callable[..., Any]
    global_registration_original: Callable[..., Any]
    registry_replacements: list[_RegistryReplacement] = field(default_factory=list)
    attribute_replacements: list[_InstanceAttributeReplacement] = field(default_factory=list)
    invocation_stacks: dict[int, list[_InvocationToken]] = field(default_factory=dict)
    known_wrappers: set[int] = field(default_factory=set)
    # Bypass evidence is intentionally sticky: once registry interposition was
    # evaded, later private edits cannot prove that every executed hook was seen.
    bypassed_modules: set[int] = field(default_factory=set)
    global_bypass_seen: bool = False
    active: bool = True

    def current_token(self, module: nn.Module) -> _InvocationToken | None:
        """Return the active LIFO token for ``module`` when present."""

        stack = self.invocation_stacks.get(id(module), ())
        return stack[-1] if stack else None


def install_prehook_provenance(
    trace: "Trace",
    model: nn.Module,
    *,
    forward_hook_wrapper_factory: Callable[[nn.Module, Callable[..., Any]], Callable[..., Any]]
    | None = None,
) -> PreHookProvenanceLedger:
    """Install the complete reversible pre-hook provenance transaction.

    Parameters
    ----------
    trace:
        Active TorchLens capture trace.
    model:
        Prepared root module.
    forward_hook_wrapper_factory:
        Existing TorchLens output-replacement wrapper factory. Supplying it
        migrates forward-hook wrapping into the same restoration transaction.

    Returns
    -------
    PreHookProvenanceLedger
        Installed transaction ledger.

    Raises
    ------
    Exception
        Any installation error is re-raised after rolling back partial work.
    """

    modules = tuple(model.modules())
    ledger = PreHookProvenanceLedger(
        trace=trace,
        model=model,
        modules=modules,
        prepared_ids=frozenset(id(module) for module in modules),
        local_registration_original=nn.Module.register_forward_pre_hook,
        global_registration_original=torch_module.register_module_forward_pre_hook,
    )
    trace._prehook_provenance_ledger = ledger
    try:
        _interpose_registration_apis(ledger)
        for module in modules:
            _wrap_registry_hooks(ledger, module._forward_pre_hooks, module=module, scope="module")
            if forward_hook_wrapper_factory is not None:
                _wrap_existing_forward_hooks(ledger, module, forward_hook_wrapper_factory)
        _wrap_registry_hooks(
            ledger,
            torch_module._global_forward_pre_hooks,
            module=None,
            scope="global",
        )
        _refresh_observers(ledger)
    except Exception:
        rollback_prehook_provenance(trace)
        raise
    return ledger


def rollback_prehook_provenance(trace: "Trace") -> None:
    """Roll back all TorchLens pre-hook interposition by identity three-way merge.

    Parameters
    ----------
    trace:
        Trace that may own a provenance ledger.
    """

    ledger = getattr(trace, "_prehook_provenance_ledger", None)
    if not isinstance(ledger, PreHookProvenanceLedger) or not ledger.active:
        return
    ledger.active = False
    if nn.Module.register_forward_pre_hook is _local_registration_wrapper_for(ledger):
        nn.Module.register_forward_pre_hook = ledger.local_registration_original  # type: ignore[method-assign]
    if torch_module.register_module_forward_pre_hook is _global_registration_wrapper_for(ledger):
        torch_module.register_module_forward_pre_hook = ledger.global_registration_original
    for attribute_replacement in reversed(ledger.attribute_replacements):
        current = getattr(attribute_replacement.owner, "__dict__", {}).get(
            attribute_replacement.name
        )
        if current is not attribute_replacement.wrapper:
            continue
        if attribute_replacement.owned_before:
            setattr(
                attribute_replacement.owner,
                attribute_replacement.name,
                attribute_replacement.original,
            )
        else:
            getattr(attribute_replacement.owner, "__dict__", {}).pop(
                attribute_replacement.name, None
            )
    for registry_replacement in reversed(ledger.registry_replacements):
        current = registry_replacement.registry.get(registry_replacement.hook_id)
        if current is registry_replacement.wrapper:
            registry_replacement.registry[registry_replacement.hook_id] = (
                registry_replacement.original
            )
    trace.__dict__.pop("_prehook_provenance_ledger", None)


def _local_registration_wrapper_for(ledger: PreHookProvenanceLedger) -> Callable[..., Any] | None:
    """Return the installed local registration wrapper stored in the ledger."""

    for replacement in ledger.attribute_replacements:
        if replacement.owner is nn.Module and replacement.name == "register_forward_pre_hook":
            return replacement.wrapper
    return None


def _global_registration_wrapper_for(ledger: PreHookProvenanceLedger) -> Callable[..., Any] | None:
    """Return the installed global registration wrapper stored in the ledger."""

    for replacement in ledger.attribute_replacements:
        if (
            replacement.owner is torch_module
            and replacement.name == "register_module_forward_pre_hook"
        ):
            return replacement.wrapper
    return None


def _interpose_registration_apis(ledger: PreHookProvenanceLedger) -> None:
    """Interpose PyTorch's public forward-pre-hook registration APIs."""

    original_local = ledger.local_registration_original
    original_global = ledger.global_registration_original

    @wraps(original_local)
    def register_forward_pre_hook(
        module: nn.Module,
        hook: Callable[..., Any],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> Any:
        """Wrap hooks only for modules in this prepared model set."""

        if not ledger.active or id(module) not in ledger.prepared_ids:
            return original_local(module, hook, prepend=prepend, with_kwargs=with_kwargs)
        wrapped = _make_pre_hook_wrapper(
            ledger,
            original=hook,
            scope="module",
            hook_id=-1,
            with_kwargs=with_kwargs,
        )
        handle = original_local(module, wrapped, prepend=prepend, with_kwargs=with_kwargs)
        hook_id = int(handle.id)
        _set_wrapper_registration_id(wrapped, hook_id)
        ledger.registry_replacements.append(
            _RegistryReplacement(module._forward_pre_hooks, hook_id, hook, wrapped)
        )
        ledger.known_wrappers.add(id(wrapped))
        _ensure_observer(ledger, module)
        return handle

    @wraps(original_global)
    def register_module_forward_pre_hook(hook: Callable[..., Any]) -> Any:
        """Wrap a newly registered process-global hook."""

        if not ledger.active:
            return original_global(hook)
        wrapped = _make_pre_hook_wrapper(
            ledger,
            original=hook,
            scope="global",
            hook_id=-1,
            with_kwargs=False,
        )
        handle = original_global(wrapped)
        hook_id = int(handle.id)
        _set_wrapper_registration_id(wrapped, hook_id)
        ledger.registry_replacements.append(
            _RegistryReplacement(torch_module._global_forward_pre_hooks, hook_id, hook, wrapped)
        )
        ledger.known_wrappers.add(id(wrapped))
        for prepared_module in ledger.modules:
            _ensure_observer(ledger, prepared_module)
        return handle

    _replace_attribute(ledger, nn.Module, "register_forward_pre_hook", register_forward_pre_hook)
    _replace_attribute(
        ledger,
        torch_module,
        "register_module_forward_pre_hook",
        register_module_forward_pre_hook,
    )


def _replace_attribute(
    ledger: PreHookProvenanceLedger,
    owner: object,
    name: str,
    wrapper: Any,
) -> None:
    """Replace one attribute and record exact prior ownership/value."""

    namespace = getattr(owner, "__dict__")
    owned_before = name in namespace
    original = namespace.get(name) if owned_before else getattr(owner, name)
    setattr(owner, name, wrapper)
    ledger.attribute_replacements.append(
        _InstanceAttributeReplacement(owner, name, owned_before, original, wrapper)
    )


def _wrap_registry_hooks(
    ledger: PreHookProvenanceLedger,
    registry: dict[int, Callable[..., Any]],
    *,
    module: nn.Module | None,
    scope: Literal["global", "module"],
    bypass: bool = False,
) -> bool:
    """Replace unrecognized hook values in place at their existing IDs.

    Returns
    -------
    bool
        Whether at least one unrecognized hook was wrapped.
    """

    with_kwargs_registry = getattr(module, "_forward_pre_hooks_with_kwargs", {}) if module else {}
    wrapped_any = False
    for hook_id, hook in list(registry.items()):
        if id(hook) in ledger.known_wrappers:
            continue
        wrapped = _make_pre_hook_wrapper(
            ledger,
            original=hook,
            scope=scope,
            hook_id=int(hook_id),
            with_kwargs=bool(hook_id in with_kwargs_registry),
        )
        registry[hook_id] = wrapped
        ledger.registry_replacements.append(
            _RegistryReplacement(registry, int(hook_id), hook, wrapped)
        )
        ledger.known_wrappers.add(id(wrapped))
        wrapped_any = True
        if bypass:
            if module is None:
                ledger.global_bypass_seen = True
            else:
                ledger.bypassed_modules.add(id(module))
    return wrapped_any


def _wrap_existing_forward_hooks(
    ledger: PreHookProvenanceLedger,
    module: nn.Module,
    factory: Callable[[nn.Module, Callable[..., Any]], Callable[..., Any]],
) -> None:
    """Move existing forward-hook wrapping under the reversible ledger."""

    for hook_id, hook in list(module._forward_hooks.items()):
        if id(hook) in ledger.known_wrappers:
            continue
        wrapped = factory(module, hook)
        module._forward_hooks[hook_id] = wrapped
        ledger.registry_replacements.append(
            _RegistryReplacement(module._forward_hooks, int(hook_id), hook, wrapped)
        )
        ledger.known_wrappers.add(id(wrapped))


def _refresh_observers(ledger: PreHookProvenanceLedger) -> None:
    """Install observers only on modules with effective local/global pre-hooks."""

    global_hooks_exist = bool(torch_module._global_forward_pre_hooks)
    for module in ledger.modules:
        if global_hooks_exist or bool(module._forward_pre_hooks):
            _ensure_observer(ledger, module)


def refresh_registration_bypasses(trace: "Trace", module: nn.Module | None = None) -> None:
    """Scan for private/pre-bound registration bypasses and downgrade attribution.

    Parameters
    ----------
    trace:
        Active capture trace.
    module:
        Imminent module invocation. ``None`` scans every prepared module.
    """

    ledger = getattr(trace, "_prehook_provenance_ledger", None)
    if not isinstance(ledger, PreHookProvenanceLedger) or not ledger.active:
        return
    global_wrapped = _wrap_registry_hooks(
        ledger,
        torch_module._global_forward_pre_hooks,
        module=None,
        scope="global",
        bypass=True,
    )
    targets = ledger.modules if module is None else (module,)
    local_wrapped = False
    for target in targets:
        local_wrapped = (
            _wrap_registry_hooks(
                ledger,
                target._forward_pre_hooks,
                module=target,
                scope="module",
                bypass=True,
            )
            or local_wrapped
        )
    if module is None or global_wrapped:
        _refresh_observers(ledger)
    elif local_wrapped:
        _ensure_observer(ledger, module)


def _ensure_observer(ledger: PreHookProvenanceLedger, module: nn.Module) -> None:
    """Install one instance-local outer-call observer and root forward binder."""

    if any(
        replacement.owner is module and replacement.name == "_call_impl"
        for replacement in ledger.attribute_replacements
    ):
        return
    namespace = module.__dict__
    owned_before = "_call_impl" in namespace
    original_owned = namespace.get("_call_impl")
    prior_call_impl = module._call_impl

    @wraps(prior_call_impl)
    def observed_call_impl(*args: Any, **kwargs: Any) -> Any:
        """Observe the real ``Module.__call__`` boundary without running hooks."""

        if (
            not ledger.active
            or not _state._logging_enabled
            or _state._active_trace is not ledger.trace
        ):
            return prior_call_impl(*args, **kwargs)
        if getattr(ledger.trace, "capture_mode", None) == "fast":
            return prior_call_impl(*args, **kwargs)
        refresh_registration_bypasses(ledger.trace, module)
        with _paused_logging():
            before_observation = _observe_state(args, kwargs)
            before_snapshot = _snapshot_state(
                args,
                kwargs,
                before_observation,
                trace=ledger.trace,
                reuse_source_payload=True,
            )
        token = _InvocationToken(
            module=module,
            args=args,
            kwargs=kwargs,
            before_snapshot=before_snapshot,
            before_observation=before_observation,
        )
        if ledger.global_bypass_seen or id(module) in ledger.bypassed_modules:
            token.incomplete_reasons.add(_INCOMPLETE_BYPASS)
        stack = ledger.invocation_stacks.setdefault(id(module), [])
        stack.append(token)
        try:
            return prior_call_impl(*args, **kwargs)
        finally:
            if stack and stack[-1] is token:
                stack.pop()
            elif token in stack:
                stack.remove(token)
            if token.effects and not token.event_emitted:
                _emit_token_event(ledger, token, address=_module_address(module), call_index=None)

    module._call_impl = observed_call_impl  # type: ignore[method-assign]
    ledger.attribute_replacements.append(
        _InstanceAttributeReplacement(
            module,
            "_call_impl",
            owned_before,
            original_owned,
            observed_call_impl,
        )
    )
    if module is ledger.model:
        _ensure_root_forward_binder(ledger)


def _ensure_root_forward_binder(ledger: PreHookProvenanceLedger) -> None:
    """Install the reversible root ``forward`` binder required by the design."""

    model = ledger.model
    if any(
        replacement.owner is model and replacement.name == "forward"
        for replacement in ledger.attribute_replacements
    ):
        return
    namespace = model.__dict__
    owned_before = "forward" in namespace
    original_owned = namespace.get("forward")
    prior_forward = model.forward

    @wraps(prior_forward)
    def root_forward(*args: Any, **kwargs: Any) -> Any:
        """Bind the current root token before the real forward body starts."""

        bind_invocation(ledger.trace, model, "self", 1, args, kwargs)
        return prior_forward(*args, **kwargs)

    model.forward = root_forward
    ledger.attribute_replacements.append(
        _InstanceAttributeReplacement(model, "forward", owned_before, original_owned, root_forward)
    )


def bind_invocation(
    trace: "Trace",
    module: nn.Module,
    address: str,
    call_index: int,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Bind the current invocation token to its actual ModuleCall identity."""

    ledger = getattr(trace, "_prehook_provenance_ledger", None)
    if not isinstance(ledger, PreHookProvenanceLedger) or not ledger.active:
        return
    token = ledger.current_token(module)
    if token is None:
        if (
            not module._forward_pre_hooks
            and not torch_module._global_forward_pre_hooks
            and not ledger.global_bypass_seen
            and id(module) not in ledger.bypassed_modules
        ):
            return
        refresh_registration_bypasses(trace, module)
        bypassed = ledger.global_bypass_seen or id(module) in ledger.bypassed_modules
        if bypassed and args is not None and kwargs is not None:
            with _paused_logging():
                observation = _observe_state(args, kwargs)
                snapshot = _snapshot_state(args, kwargs, observation)
            reasons = tuple(sorted({_INCOMPLETE_BYPASS, *observation.incomplete_reasons}))
            trace.capture_events.pre_hook_events.append(
                PreHookProvenanceEvent(
                    address=address or "self",
                    call_index=call_index,
                    inputs_before_pre_hooks=None,
                    inputs_after_pre_hooks=_snapshot_with_reasons(snapshot, reasons),
                    effects=(),
                    capture_complete=False,
                    incomplete_reasons=reasons,
                )
            )
        return
    if token.bound or not token.effects:
        return
    token.bound = True
    _emit_token_event(ledger, token, address=address or "self", call_index=call_index)


def _emit_token_event(
    ledger: PreHookProvenanceLedger,
    token: _InvocationToken,
    *,
    address: str,
    call_index: int | None,
) -> None:
    """Finalize snapshot B from the wrapper logical state and append one event."""

    with _paused_logging():
        after_observation = _observe_state(token.args, token.kwargs)
        if _fingerprint(token.before_observation) == _fingerprint(after_observation):
            after_snapshot = token.before_snapshot
        else:
            after_snapshot = _snapshot_state(token.args, token.kwargs, after_observation)
    reasons = tuple(sorted(token.incomplete_reasons | set(after_observation.incomplete_reasons)))
    before_snapshot = _snapshot_with_reasons(token.before_snapshot, reasons)
    after_snapshot = _snapshot_with_reasons(after_snapshot, reasons)
    ledger.trace.capture_events.pre_hook_events.append(
        PreHookProvenanceEvent(
            address=address,
            call_index=call_index,
            inputs_before_pre_hooks=before_snapshot,
            inputs_after_pre_hooks=after_snapshot if call_index is not None else None,
            effects=tuple(token.effects),
            capture_complete=not reasons,
            incomplete_reasons=reasons,
        )
    )
    token.event_emitted = True


def _snapshot_with_reasons(
    snapshot: ModuleInputSnapshot,
    reasons: tuple[str, ...],
) -> ModuleInputSnapshot:
    """Return ``snapshot`` with merged completeness reasons."""

    merged = tuple(sorted(set(snapshot.incomplete_reasons) | set(reasons)))
    if merged == snapshot.incomplete_reasons:
        return snapshot
    return ModuleInputSnapshot(
        _args=snapshot._args,
        _kwargs=snapshot._kwargs,
        structure=snapshot.structure,
        tensor_observations=snapshot.tensor_observations,
        capture_complete=not merged,
        incomplete_reasons=merged,
        summary=snapshot.summary,
    )


def _make_pre_hook_wrapper(
    ledger: PreHookProvenanceLedger,
    *,
    original: Callable[..., Any],
    scope: Literal["global", "module"],
    hook_id: int,
    with_kwargs: bool,
) -> Callable[..., Any]:
    """Return a transparent wrapper that records one hook transition."""

    registration_id = [hook_id]
    hook_info = _hook_info(original)

    @wraps(original)
    def wrapped(*hook_args: Any, **hook_kwargs: Any) -> Any:
        """Call the exact user hook and record its normalized transition."""

        module = hook_args[0] if hook_args and isinstance(hook_args[0], nn.Module) else None
        if module is None:
            return original(*hook_args, **hook_kwargs)
        token = ledger.current_token(module)
        if token is None or not ledger.active:
            return original(*hook_args, **hook_kwargs)
        actual_args = tuple(hook_args[1]) if len(hook_args) > 1 else token.args
        actual_kwargs = dict(hook_args[2]) if with_kwargs and len(hook_args) > 2 else token.kwargs
        token.args = actual_args
        token.kwargs = actual_kwargs
        with _paused_logging():
            before = _observe_state(token.args, token.kwargs)
        result: Any = None
        inputs_after_side_effects: _StateObservation | None = None
        outcome: Literal["none", "returned", "invalid_return", "raised"] = "none"
        exception_type = None
        exception_message = None
        try:
            result = original(*hook_args, **hook_kwargs)
            if result is not None:
                with _paused_logging():
                    inputs_after_side_effects = _observe_state(actual_args, actual_kwargs)
                outcome = "returned"
                if with_kwargs:
                    if isinstance(result, tuple) and len(result) == 2:
                        token.args = result[0]
                        token.kwargs = result[1]
                    else:
                        outcome = "invalid_return"
                else:
                    token.args = result if isinstance(result, tuple) else (result,)
            return result
        except Exception as exc:
            outcome = "raised"
            exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            exception_message = str(exc)
            raise
        finally:
            with _paused_logging():
                after = _observe_state(token.args, token.kwargs)
                changed, kinds, paths, reasons = _compare_observations(before, after)
                if inputs_after_side_effects is not None:
                    _, side_effect_kinds, side_effect_paths, side_effect_reasons = (
                        _compare_observations(before, inputs_after_side_effects)
                    )
                    kinds = tuple(sorted(set(kinds) | set(side_effect_kinds)))
                    paths = tuple(sorted(set(paths) | set(side_effect_paths), key=repr))
                    reasons = tuple(sorted(set(reasons) | set(side_effect_reasons)))
                    changed = True if kinds else (None if reasons else False)
            token.incomplete_reasons.update(reasons)
            token.effects.append(
                PreHookEffect(
                    ordinal=len(token.effects) + 1,
                    scope=scope,
                    registration_id=registration_id[0],
                    hook=hook_info,
                    with_kwargs=with_kwargs,
                    outcome=outcome,
                    changed=changed,
                    change_kinds=kinds,
                    changed_paths=paths,
                    before_fingerprint=_fingerprint(before),
                    after_fingerprint=_fingerprint(after),
                    incomplete_reasons=reasons,
                    exception_type=exception_type,
                    exception_message=exception_message,
                )
            )

    setattr(wrapped, "_torchlens_registration_id_cell", registration_id)
    return wrapped


def _set_wrapper_registration_id(wrapper: Callable[..., Any], hook_id: int) -> None:
    """Set the real PyTorch ID after delegated registration allocates it."""

    cell = getattr(wrapper, "_torchlens_registration_id_cell", None)
    if isinstance(cell, list) and cell:
        cell[0] = hook_id


def _hook_info(hook: Callable[..., Any]) -> HookInfo:
    """Build portable descriptive metadata for one original user hook."""

    name = getattr(hook, "__name__", type(hook).__name__)
    qualname = getattr(hook, "__qualname__", name)
    module_name = getattr(hook, "__module__", "")
    full_qualname = f"{module_name}.{qualname}" if module_name else qualname
    source_location = None
    try:
        source_file = inspect.getsourcefile(hook) or inspect.getfile(hook)
        source_line = inspect.getsourcelines(hook)[1]
    except (OSError, TypeError):
        pass
    else:
        source_location = FuncCallLocation(
            file=source_file,
            line_number=source_line,
            func_name=full_qualname,
            source_loading_enabled=False,
        )
    return HookInfo(name=name, qualname=full_qualname, source_location=source_location)


@contextmanager
def _paused_logging() -> Iterator[None]:
    """Pause TorchLens logging around provenance-only tensor operations."""

    with _state.pause_logging():
        yield


def _observe_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _StateObservation:
    """Capture structure, identity, version, and metadata without tensor copies."""

    structure: list[Any] = []
    tensors: list[_TensorState] = []
    leaves: list[tuple[TypedInputPath, str, Any]] = []
    reasons: set[str] = set()
    _walk_value(args, ("args",), structure, tensors, leaves, reasons)
    _walk_value(kwargs, ("kwargs",), structure, tensors, leaves, reasons)
    return _StateObservation(
        structure=tuple(structure),
        tensors=tuple(tensors),
        leaves=tuple(leaves),
        incomplete_reasons=tuple(sorted(reasons)),
    )


def _walk_value(
    value: Any,
    path: TypedInputPath,
    structure: list[Any],
    tensors: list[_TensorState],
    leaves: list[tuple[TypedInputPath, str, Any]],
    reasons: set[str],
) -> None:
    """Walk supported input containers in deterministic typed-path order."""

    if isinstance(value, torch.Tensor):
        try:
            version = int(value._version)
        except (AttributeError, RuntimeError):
            version = None
            reasons.add(_INCOMPLETE_VERSION)
        structure.append((path, "tensor"))
        tensors.append(
            _TensorState(
                path=path,
                value=value,
                object_id=id(value),
                version=version,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                layout=str(value.layout),
                requires_grad=bool(value.requires_grad),
                source_op=get_tensor_label(value),
            )
        )
        return
    if isinstance(value, tuple):
        structure.append((path, "tuple", len(value)))
        for index, item in enumerate(value):
            _walk_value(item, (*path, index), structure, tensors, leaves, reasons)
        return
    if isinstance(value, list):
        structure.append((path, "list", len(value)))
        for index, item in enumerate(value):
            _walk_value(item, (*path, index), structure, tensors, leaves, reasons)
        return
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        structure.append((path, "mapping", tuple((type(key).__name__, repr(key)) for key in keys)))
        for key in keys:
            _walk_value(value[key], (*path, ("key", key)), structure, tensors, leaves, reasons)
        return
    if isinstance(value, _IMMUTABLE_LEAF_TYPES):
        structure.append((path, "immutable", type(value).__name__))
        leaves.append((path, "immutable", (type(value).__name__, value)))
        return
    structure.append((path, "object", type(value).__qualname__))
    leaves.append((path, "object", (type(value).__qualname__, id(value))))


def _compare_observations(
    before: _StateObservation,
    after: _StateObservation,
) -> tuple[bool | None, tuple[str, ...], tuple[TypedInputPath, ...], tuple[str, ...]]:
    """Compare two observations using the decided tri-state detection semantics."""

    kinds: set[str] = set()
    paths: set[TypedInputPath] = set()
    if before.structure != after.structure:
        kinds.add("structure")
        before_paths = {entry[0] for entry in before.structure}
        after_paths = {entry[0] for entry in after.structure}
        paths.update(before_paths ^ after_paths)
    before_tensors = {state.path: state for state in before.tensors}
    after_tensors = {state.path: state for state in after.tensors}
    for path in before_tensors.keys() | after_tensors.keys():
        left = before_tensors.get(path)
        right = after_tensors.get(path)
        if left is None or right is None:
            kinds.add("structure")
            paths.add(path)
            continue
        if left.object_id != right.object_id:
            kinds.add("identity")
            paths.add(path)
            continue
        if left.version is not None and right.version is not None and left.version != right.version:
            kinds.add("in_place_tensor")
            paths.add(path)
        left_metadata = (left.shape, left.dtype, left.device, left.layout, left.requires_grad)
        right_metadata = (right.shape, right.dtype, right.device, right.layout, right.requires_grad)
        if left_metadata != right_metadata:
            kinds.add("tensor_metadata")
            paths.add(path)
    before_leaves = {path: value for path, _kind, value in before.leaves}
    after_leaves = {path: value for path, _kind, value in after.leaves}
    for path in before_leaves.keys() | after_leaves.keys():
        if before_leaves.get(path) != after_leaves.get(path):
            kinds.add("value_or_identity")
            paths.add(path)
    reasons = tuple(sorted(set(before.incomplete_reasons) | set(after.incomplete_reasons)))
    changed: bool | None = True if kinds else (None if reasons else False)
    return changed, tuple(sorted(kinds)), tuple(sorted(paths, key=repr)), reasons


def _fingerprint(observation: _StateObservation) -> tuple[Any, ...]:
    """Return a compact metadata-only fingerprint for one observation."""

    tensor_bits = tuple(
        (
            state.path,
            state.object_id,
            state.version,
            state.shape,
            state.dtype,
            state.device,
            state.layout,
            state.requires_grad,
        )
        for state in observation.tensors
    )
    return observation.structure, tensor_bits, observation.leaves


def _snapshot_state(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    observation: _StateObservation,
    *,
    trace: "Trace | None" = None,
    reuse_source_payload: bool = False,
) -> ModuleInputSnapshot:
    """Create one truthful detached snapshot, deduplicating tensor aliases."""

    tensor_copies: dict[int, Any] = {}
    payload_origins: dict[int, str] = {}
    copy_failures: set[int] = set()
    states_by_id = {state.object_id: state for state in observation.tensors}

    def clone(value: Any) -> Any:
        """Clone tensors and recursively preserve supported input containers."""

        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id in tensor_copies:
                return tensor_copies[tensor_id]
            state = states_by_id[tensor_id]
            if reuse_source_payload and trace is not None and state.source_op is not None:
                event = trace.capture_events.op_event_by_label_raw.get(state.source_op)
                payload = event.output.tensor.payload if event is not None else None
                if isinstance(payload, torch.Tensor) and payload is not value:
                    tensor_copies[tensor_id] = payload
                    payload_origins[tensor_id] = "immutable_producer_snapshot"
                    return payload
            try:
                copied = value.detach().clone()  # noqa: detach - provenance snapshot copy
            except Exception:
                copied = None
                copy_failures.add(tensor_id)
            tensor_copies[tensor_id] = copied
            payload_origins[tensor_id] = "dedicated_pre_hook_snapshot"
            return copied
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    cloned_args = clone(args)
    cloned_kwargs = clone(kwargs)
    alias_groups: dict[int, int] = {}
    tensor_observations: list[TensorInputObservation] = []
    reasons = set(observation.incomplete_reasons)
    for state in observation.tensors:
        alias_group = alias_groups.setdefault(state.object_id, len(alias_groups) + 1)
        payload = tensor_copies.get(state.object_id)
        if state.object_id in copy_failures:
            reasons.add("tensor_payload_copy_failed")
        tensor_observations.append(
            TensorInputObservation(
                path=state.path,
                source_op=state.source_op,
                object_id=state.object_id,
                version=state.version,
                shape=state.shape,
                dtype=state.dtype,
                device=state.device,
                layout=state.layout,
                requires_grad=state.requires_grad,
                payload=payload,
                payload_status="saved" if payload is not None else "unsaved",
                alias_group=alias_group,
                payload_origin=payload_origins.get(state.object_id, "dedicated_pre_hook_snapshot"),
            )
        )
    return ModuleInputSnapshot(
        _args=cloned_args,
        _kwargs=cloned_kwargs,
        structure=observation.structure,
        tensor_observations=tuple(tensor_observations),
        capture_complete=not reasons,
        incomplete_reasons=tuple(sorted(reasons)),
        summary=f"args={len(args)}, kwargs={len(kwargs)}, tensors={len(observation.tensors)}",
    )


def _module_address(module: nn.Module) -> str:
    """Return the prepared module address for interrupted event diagnostics."""

    meta = getattr(module, "_tl", None)
    address = getattr(meta, "address", None)
    return "self" if not address else str(address)
