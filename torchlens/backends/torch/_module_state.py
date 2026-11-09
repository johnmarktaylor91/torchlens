"""Which modules own state, the fact call-site grouping (``callsite_v2``) scopes on.

A module is stateful when it, or any module below it, registers a parameter or a
buffer that is not ``None``. Only a stateful module is a grouping scope: each of its
calls runs every op inside it once more, so those ops get one pass per module call.
A stateless module (``nn.ReLU``, ``nn.Dropout``, ``nn.MaxPool2d``) is read as the
function it is, grouped by where the stateful code around it called it
(``docs/guides/grouping.md``).

The set is decided once per capture from what the model declares, never from which
weights one capture happened to read, so grouping cannot flip between captures of
the same model. ``UninitializedParameter`` slots of a lazy module count as state.
"""

from __future__ import annotations

from torch import nn

from ..._utils._tensor_label import get_module_meta


def _owns_state(module: nn.Module) -> bool:
    """Whether ``module`` itself registers a non-None parameter or buffer."""

    own = (*module._parameters.values(), *module._buffers.values())
    return any(value is not None for value in own)


def stateful_modules(model: nn.Module) -> dict[int, nn.Module]:
    """Return every stateful module in ``model``'s tree, keyed by ``id``.

    Iterative post-order over ``_modules`` (deep models do not hit the recursion
    limit); a module registered under several names is decided once.
    """

    verdict: dict[int, bool] = {}
    keep: dict[int, nn.Module] = {}
    stack: list[tuple[nn.Module, bool]] = [(model, False)]
    while stack:
        module, children_done = stack.pop()
        key = id(module)
        if children_done:
            children = (child for child in module._modules.values() if child is not None)
            stateful = _owns_state(module) or any(verdict[id(child)] for child in children)
            verdict[key] = stateful
            if stateful:
                keep[key] = module
            continue
        if key in verdict:
            continue
        verdict[key] = False  # placeholder: a cyclic registration reads as stateless
        stack.append((module, True))
        stack.extend(
            (child, False)
            for child in module._modules.values()
            if child is not None and id(child) not in verdict
        )
    return keep


def stateful_scope_facts(model: nn.Module) -> tuple[frozenset[int], frozenset[str]]:
    """The stateful modules' ids and their prepared addresses (the module-stack spelling)."""

    modules = stateful_modules(model)
    addresses: set[str] = set()
    for module in modules.values():
        meta = get_module_meta(module)
        address = getattr(meta, "address", None)
        if address:
            addresses.add(str(address))
    return frozenset(modules), frozenset(addresses)
