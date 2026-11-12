"""Opt-in prepared-model cache: a model keeps its built ``forward`` wrappers between captures.

Outside :func:`kept_prepared` nothing changes: every capture builds its session
``forward`` wrappers for the model's submodules at entry and puts the prior
``forward`` back at cleanup.

Inside ``with kept_prepared(model):`` the wrapper objects the first capture
built are kept aside (never on the model) and the next capture of the same
model installs those same objects instead of building new ones, after an
identity check: the module tree record (which modules, under which names, of
which classes), every parameter object, and, per wrapped module, that its
instance ``forward`` is still the one the wrapper was built over and its class
``forward`` is the one wrapped. Any mismatch builds fresh wrappers, which the
block keeps from then on. Session cleanup still puts every prior ``forward``
back after each capture, on every path, so between captures and after the block
the model holds no TorchLens callable, exactly as outside the block. Every other
part of preparation and cleanup (parameter stamps and ``requires_grad`` restore,
buffer stamps, tensor-label clearing, module metadata) runs on every capture.

The wrappers read the active capture at call time
(:func:`.model_prep.module_forward_decorator`), so one wrapper object serves any
number of captures.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torch import nn

from ..._utils._model_walk import current_tree_record
from ._scope_facts_cache import cached_scope_facts

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ["kept_prepared", "install_session_forwards"]


@dataclass(slots=True)
class _KeptPreparation:
    """The wrappers one model keeps between the captures of one block.

    ``installed`` holds ``(module, prior, wrapper, class_forward)`` per wrapped
    submodule, as the session install recorded them plus the class ``forward``
    seen at install. ``tree_record`` and ``params`` are the identity key;
    ``params`` holds, per module (root included), its ``_parameters`` mapping
    and the parameter objects it held.
    """

    model_ref: weakref.ref[nn.Module]
    depth: int = 0
    tree_record: Any = None
    params: tuple[tuple[Any, tuple[Any, ...]], ...] = ()
    installed: list[tuple[nn.Module, Any, Any, Any]] = field(default_factory=list)
    no_instance: Any = None


#: ``id(model) -> kept preparation`` for every model inside a ``kept_prepared`` block.
_KEPT: dict[int, _KeptPreparation] = {}


def _class_forward(module: nn.Module) -> Any:
    """Return the ``forward`` the module's class defines (a plain function, unbound)."""

    return getattr(type(module), "forward", None)


def _slot_modules(model: nn.Module, kept: _KeptPreparation) -> list[nn.Module]:
    """Return the root followed by every kept submodule, the order ``params`` uses."""

    return [model, *(entry[0] for entry in kept.installed)]


def _param_slots(model: nn.Module, kept: _KeptPreparation) -> tuple[Any, ...]:
    """Return ``(registry, parameters)`` for the root and every kept submodule."""

    slots = []
    for module in _slot_modules(model, kept):
        registry = module.__dict__.get("_parameters")
        slots.append((registry, tuple(registry.values()) if registry is not None else ()))
    return tuple(slots)


def _same_params(kept: _KeptPreparation, model: nn.Module) -> bool:
    """Return whether every module holds the same registry and parameter objects, in order.

    A module's parameters are what its ``_parameters`` registry holds, so this
    sees a parameter replaced, added or removed anywhere in the tree.
    """

    for module, (registry, params) in zip(_slot_modules(model, kept), kept.params, strict=True):
        live = module.__dict__.get("_parameters")
        if live is not registry or (live is not None and len(live) != len(params)):
            return False
        if live is not None:
            for held, now in zip(params, live.values(), strict=True):
                if held is not now:
                    return False
    return True


def _kept_still_valid(kept: _KeptPreparation, model: nn.Module) -> bool:
    """Return whether the kept wrappers may serve this capture of ``model``.

    A ``None`` tree record means the tree is walked live (it cannot be
    described), so nothing is reused for it.
    """

    record = current_tree_record(model)
    if record is None or record is not kept.tree_record:
        return False
    if not _same_params(kept, model):
        return False
    for module, prior, _wrapper, class_forward in kept.installed:
        if module.__dict__.get("forward", kept.no_instance) is not prior:
            return False
        if _class_forward(module) is not class_forward:
            return False
    return True


def _reinstall(trace: Trace, model: nn.Module, kept: _KeptPreparation) -> None:
    """Install the kept wrappers for this capture, as the session install would.

    The wrappers' module boundary was registered, process-wide, by the install
    that built them. The install record goes to the session as usual, so its
    cleanup puts every prior ``forward`` back.
    """

    workspace = trace._module_capture_ws
    workspace.stateful_module_ids, workspace.stateful_module_addresses = cached_scope_facts(model)
    record = workspace.session_forward_wrappers
    for module, prior, wrapper, _class in kept.installed:
        module.__dict__["forward"] = wrapper
        record.append((module, prior, wrapper))


def install_session_forwards(
    trace: Trace,
    model: nn.Module,
    install: Callable[[Trace, nn.Module], None],
    no_instance: Any,
) -> None:
    """Install this capture's submodule ``forward`` wrappers, reusing kept ones in a block.

    Outside a :func:`kept_prepared` block this is ``install(trace, model)``, the
    session install. Inside one, valid kept wrappers are installed again;
    otherwise fresh ones are built and kept for the captures that follow.
    ``no_instance`` is the install record's marker for a module that had no
    instance ``forward``.
    """

    kept = _KEPT.get(id(model))
    if kept is None or kept.model_ref() is not model:
        install(trace, model)
        return
    if kept.installed and _kept_still_valid(kept, model):
        _reinstall(trace, model, kept)
        return
    kept.installed, kept.tree_record, kept.params = [], None, ()
    install(trace, model)
    record = current_tree_record(model)
    if record is None:
        return  # an undescribable tree gets fresh wrappers every capture
    kept.installed = [
        (module, prior, wrapper, _class_forward(module))
        for module, prior, wrapper in trace._module_capture_ws.session_forward_wrappers
    ]
    kept.tree_record = record
    kept.params = _param_slots(model, kept)
    kept.no_instance = no_instance


@contextmanager
def kept_prepared(model: nn.Module) -> Iterator[nn.Module]:
    """Keep ``model``'s built ``forward`` wrappers for the captures made inside the block.

    Re-entering the block for the same model nests; the outermost exit drops
    the kept wrappers. The model itself is left as after any capture: the
    session cleanup has already put every prior ``forward`` back.

    Parameters
    ----------
    model
        The model captured repeatedly inside the block.

    Yields
    ------
    nn.Module
        ``model`` itself.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"kept_prepared expects an nn.Module, got {type(model).__name__}")
    kept = _KEPT.get(id(model))
    if kept is None or kept.model_ref() is not model:
        kept = _KeptPreparation(model_ref=weakref.ref(model))
        _KEPT[id(model)] = kept
    kept.depth += 1
    try:
        yield model
    finally:
        kept.depth -= 1
        if kept.depth == 0 and _KEPT.get(id(model)) is kept:
            del _KEPT[id(model)]
