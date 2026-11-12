"""Opt-in prepared-model cache: a model keeps its session preparation between captures.

Outside :func:`kept_prepared` nothing changes: every capture installs its
session ``forward`` wrappers on the model's submodules at entry and puts them
back at cleanup, so a model between captures holds no TorchLens callable.

Inside ``with kept_prepared(model):`` the wrappers one capture installed stay on
the model until the next capture of the same model, which reuses them only
after an identity check: the module tree record (which modules, under which
names, of which classes), every parameter object, and, per wrapped module, that
its ``forward`` is still the kept wrapper and its class ``forward`` is the one
wrapped. Any mismatch puts the kept wrappers back and prepares afresh. Every
other part of preparation and cleanup (parameter stamps and ``requires_grad``
restore, buffer stamps, tensor-label clearing, module metadata) still runs on
every capture. When the block exits, on every path, the kept wrappers are put
back, so the model is left exactly as after a capture outside the block.

The wrappers read the active capture at call time, so a call between captures
passes straight through to the original ``forward``, and a deep copy or pickle
taken between captures reduces to the prior ``forward`` exactly as it does
during a capture (:class:`._session_forwards._SessionForward`).
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from torch import nn

from ..._utils._model_walk import current_tree_record, model_parameters
from ._scope_facts_cache import cached_scope_facts

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ["kept_prepared", "install_session_forwards"]


@dataclass(slots=True)
class _KeptPreparation:
    """The preparation one model keeps between the captures of one block.

    ``installed`` holds ``(module, prior, wrapper, class_forward)`` per wrapped
    submodule, as the session install recorded them plus the class ``forward``
    seen at install. ``tree_record`` and ``params`` are the identity key.
    """

    model_ref: weakref.ref[nn.Module]
    depth: int = 0
    tree_record: Any = None
    params: tuple[nn.Parameter, ...] = ()
    installed: list[tuple[nn.Module, Any, Any, Any]] = field(default_factory=list)
    restore: Callable[[Any], None] | None = None


#: ``id(model) -> kept preparation`` for every model inside a ``kept_prepared`` block.
_KEPT: dict[int, _KeptPreparation] = {}


def _class_forward(module: nn.Module) -> Any:
    """Return the ``forward`` the module's class defines, without binding it."""

    return inspect.getattr_static(type(module), "forward", None)


def _same_params(kept: tuple[nn.Parameter, ...], live: tuple[nn.Parameter, ...]) -> bool:
    """Return whether two parameter tuples hold the same objects in the same order."""

    if len(kept) != len(live):
        return False
    return all(kept[index] is live[index] for index in range(len(kept)))


def _kept_still_valid(kept: _KeptPreparation, model: nn.Module) -> bool:
    """Return whether the kept wrappers may serve this capture of ``model``.

    A ``None`` tree record means the tree is walked live (it cannot be
    described), so nothing is reused for it.
    """

    record = current_tree_record(model)
    if record is None or record is not kept.tree_record:
        return False
    if not _same_params(kept.params, model_parameters(model)):
        return False
    for module, _prior, wrapper, class_forward in kept.installed:
        if module.__dict__.get("forward") is not wrapper:
            return False
        if _class_forward(module) is not class_forward:
            return False
    return True


def _put_back(kept: _KeptPreparation) -> None:
    """Put back every ``forward`` the kept wrappers replaced, as the session cleanup does.

    The session's own restore runs over the kept install record, so a module
    whose ``forward`` no longer holds the kept wrapper keeps the newer value.
    """

    installed, kept.installed = kept.installed, []
    kept.tree_record = None
    kept.params = ()
    if not installed or kept.restore is None:
        return
    record = [(module, prior, wrapper) for module, prior, wrapper, _cls in installed]
    kept.restore(
        SimpleNamespace(_module_capture_ws=SimpleNamespace(session_forward_wrappers=record))
    )


def _rearm_session(trace: Trace, model: nn.Module) -> None:
    """Run the per-capture part of the wrapper install for reused wrappers.

    The wrappers' module boundary was registered, process-wide, by the install
    that built them.
    """

    workspace = trace._module_capture_ws
    workspace.stateful_module_ids, workspace.stateful_module_addresses = cached_scope_facts(model)
    workspace.session_forward_wrappers = []


def install_session_forwards(
    trace: Trace,
    model: nn.Module,
    install: Callable[[Trace, nn.Module], None],
    restore: Callable[[Any], None],
) -> None:
    """Install this capture's submodule ``forward`` wrappers, reusing kept ones in a block.

    Outside a :func:`kept_prepared` block this is ``install(trace, model)``.
    ``restore`` is the session cleanup's wrapper restore; the block uses it to
    put the kept wrappers back.
    Inside one, valid kept wrappers are reused; otherwise the kept ones are put
    back, ``install`` runs, and the block takes the new wrappers over from the
    session (the session's cleanup then has nothing of them to restore).
    """

    kept = _KEPT.get(id(model))
    if kept is None or kept.model_ref() is not model:
        install(trace, model)
        return
    if kept.installed and _kept_still_valid(kept, model):
        _rearm_session(trace, model)
        return
    _put_back(kept)
    kept.restore = restore
    install(trace, model)
    workspace = trace._module_capture_ws
    record = current_tree_record(model)
    if record is None:
        return  # an undescribable tree is prepared afresh every capture
    kept.installed = [
        (module, prior, wrapper, _class_forward(module))
        for module, prior, wrapper in workspace.session_forward_wrappers
    ]
    kept.tree_record = record
    kept.params = model_parameters(model)
    workspace.session_forward_wrappers = []


@contextmanager
def kept_prepared(model: nn.Module) -> Iterator[nn.Module]:
    """Keep ``model``'s capture preparation between the captures made inside the block.

    Re-entering the block for the same model nests; the outermost exit puts the
    kept wrappers back. ``requires_grad`` and every per-capture observation are
    restored or re-read on every capture, inside the block as outside it.

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
        if kept.depth == 0:
            try:
                _put_back(kept)
            finally:
                if _KEPT.get(id(model)) is kept:
                    del _KEPT[id(model)]
