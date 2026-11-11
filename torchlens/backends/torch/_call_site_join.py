"""Join a cached call site's code part with the current op's modules.

The call-site table (``_ops_call_site_fields``) keys a site on code only: code
objects, their files and offsets, and, in place of each module the walk reads
(the module of a ``_call_impl`` frame, a crossed stateless module call), a slot
numbered by first appearance. The modules themselves come per op from the
module calls the module stack holds. Two ops with equal keys therefore have the
same code context and the same call-site chain up to the module tokens: the
chain's collapse compares tokens only for equality, the key fixes which slots
are equal, and a module token never equals a code token. So the cached chain
is a template: its module tokens are replaced by the current op's modules.

The join also writes the chain under the raw-frames key the build would have
written (``_call_site.cached_chain``), so the code-context cache holds exactly
the entries a build writes, in the same order.
"""

from typing import Any

from ..._utils._call_site import module_id_scope_entry

__all__ = ("join", "make_template", "slots")

#: The tag of a module token: ``module_id_scope_entry(id) == ((_MODULE, id), 0)``.
_MODULE = module_id_scope_entry(0)[0][0]

#: ``(chain positions, raw-frames cache key, raw positions)``; each position is
#: ``(index, slot)``. The key is ``None`` when the chain is empty.
Template = tuple[tuple[tuple[int, int], ...], Any, tuple[tuple[int, int], ...]]


def slots(modules: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(distinct modules in first-appearance order, each module's slot)`` of ``modules``."""

    index: dict[int, int] = {}
    pattern = []
    for module_id in modules:
        slot = index.get(module_id)
        if slot is None:
            slot = len(index)
            index[module_id] = slot
        pattern.append(slot)
    return tuple(index), tuple(pattern)


def _positions(entries: Any, slot: dict[int, int]) -> tuple[tuple[int, int], ...] | None:
    """``(index, slot)`` of every module token in ``entries``; ``None`` if one is not a slot."""

    found = []
    for index, (token, _instruction) in enumerate(entries):
        if isinstance(token, tuple) and len(token) == 2 and token[0] == _MODULE:
            module_slot = slot.get(token[1])
            if module_slot is None:
                return None
            found.append((index, module_slot))
    return tuple(found)


def make_template(
    chain: tuple[Any, ...], modules: tuple[int, ...], raw_key: Any
) -> Template | None:
    """The template of a chain built for an op whose distinct modules are ``modules``.

    ``raw_key`` is the raw-frames cache key the build looked the chain up under
    (``None`` when the chain is empty). ``None`` when the chain cannot be joined:
    a module token that is not one of ``modules``, or a chain whose key the
    build did not report.
    """

    slot = {module_id: index for index, module_id in enumerate(modules)}
    chain_positions = _positions(chain, slot)
    if chain_positions is None:
        return None
    if raw_key is None:
        return ((), None, ()) if not chain else None
    raw_positions = _positions(raw_key[1], slot)
    if raw_positions is None:
        return None
    return chain_positions, raw_key, raw_positions


def join(
    chain: tuple[Any, ...], template: Template, modules: tuple[int, ...], cache: dict[Any, Any]
) -> tuple[Any, ...]:
    """``chain`` with its module tokens replaced by ``modules``, cached as a build caches it."""

    chain_positions, raw_key, raw_positions = template
    if raw_key is None:
        return chain
    raw = list(raw_key[1])
    for index, slot in raw_positions:
        raw[index] = ((_MODULE, modules[slot]), 0)
    key = (raw_key[0], tuple(raw))
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    out = list(chain)
    for index, slot in chain_positions:
        out[index] = ((_MODULE, modules[slot]), 0)
    joined = tuple(out)
    cache[key] = joined
    return joined
