"""The stateful-module scope facts, kept per unchanged model and verified before reuse.

:func:`._module_state.stateful_scope_facts` decides which modules own state
(a non-``None`` parameter or buffer of their own or below them) and spells
their prepared addresses. Every capture session used to decide it afresh with
a post-order walk. The answer depends on three things only, and this module
reuses the last answer while all three hold:

- the module tree, through the recorded walk of
  :func:`~torchlens._utils._model_walk.current_tree_record` (compared by
  identity, so any added, removed, replaced, renamed or moved submodule
  discards the cached facts);
- each module's own-state bit: whether its ``_parameters`` or ``_buffers``
  holds a non-``None`` value, read live on every call;
- each stateful module's prepared address, read live on every call.

Values are never read. The cache is a weak dictionary on the model and holds
modules through weak references only, so it keeps nothing alive.
"""

from __future__ import annotations

import weakref
from typing import Any

from torch import nn

from ..._utils._model_walk import current_tree_record, model_modules
from ..._utils._tensor_label import get_module_meta
from ._module_state import stateful_modules

#: Model -> ``(record, bits, refs, raw_addresses, ids, addresses)``.
_FACTS: weakref.WeakKeyDictionary[nn.Module, tuple[Any, ...]] = weakref.WeakKeyDictionary()


def _own_state_bits(modules: tuple[nn.Module, ...]) -> tuple[bool, ...]:
    """Return, per module, whether it registers a non-``None`` parameter or buffer itself."""

    bits: list[bool] = []
    for module in modules:
        owns = False
        for registry in (module._parameters, module._buffers):
            for name in registry:
                if registry[name] is not None:
                    owns = True
                    break
            if owns:
                break
        bits += (owns,)
    return tuple(bits)


def _raw_address(module: nn.Module) -> Any:
    """Return the module's prepared address as stored (``None`` when it has none)."""

    return getattr(get_module_meta(module), "address", None)


def _addresses_hold(refs: tuple[Any, ...], raw: tuple[Any, ...]) -> bool:
    """Return whether every stateful module still has the address the facts spelled."""

    for index in range(len(refs)):
        module = refs[index]()
        if module is None or _raw_address(module) != raw[index]:
            return False
    return True


def cached_scope_facts(model: nn.Module) -> tuple[frozenset[int], frozenset[str]]:
    """Return what ``stateful_scope_facts(model)`` returns, reused while its inputs hold."""

    record = current_tree_record(model)
    if record is None:
        _FACTS.pop(model, None)
        return _decide(model)[4:]
    bits = _own_state_bits(model_modules(model))
    cached = _FACTS.get(model)
    if (
        cached is not None
        and cached[0] is record
        and cached[1] == bits
        and _addresses_hold(cached[2], cached[3])
    ):
        return cached[4], cached[5]
    facts = (record, bits, *_decide(model)[2:])
    _FACTS[model] = facts
    return facts[4], facts[5]


def _decide(model: nn.Module) -> tuple[Any, ...]:
    """Decide the facts in full: ``(None, None, refs, raw_addresses, ids, addresses)``."""

    modules = stateful_modules(model)
    refs = tuple(weakref.ref(module) for module in modules.values())
    raw = tuple(_raw_address(module) for module in modules.values())
    addresses = frozenset(str(address) for address in raw if address)
    return None, None, refs, raw, frozenset(modules), addresses


__all__ = ["cached_scope_facts"]
