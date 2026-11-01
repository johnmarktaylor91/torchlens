"""Bounded scan of the plain tensors a module holds outside ``register_buffer``.

Model preparation stamps every plain tensor a module keeps in its attributes and
containers (list, tuple, dict items, bounded nesting) with a buffer address, so a
forward that reads one gets a buffer source. The walk is bounded per module (depth,
stamped tensors, items examined) and every cut is disclosed through
:func:`warn_held_scan_truncated`. Split from ``buffer_writes.py`` under the R43
file-size ratchet; the names are the same.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

_HELD_SCAN_MAX_DEPTH = 4
"""Container levels below a module attribute that the held-tensor scan descends.

Level 1 is an item of the attribute's own container (``attr.<i>`` / ``attr[<key>]``);
a container found at level 4 is not entered. Mirrors ``_FIDELITY_MAX_DEPTH``.
"""

_HELD_SCAN_MAX_OBJECTS = 4096
"""Plain tensors the held-tensor scan stamps per module.

Mirrors ``_robustness._ITER_TENSORS_MAX_NODES``. Only tensors count: a tensor-free
container (a merge table of string pairs, a deep config dict) never trips the bound.
"""

_HELD_SCAN_MAX_VISITS = 1 << 16
"""Container items the held-tensor scan examines per module, of any type.

A hard cost cap independent of what the containers hold, so a 200k-entry scalar table
costs at most this many type checks. Reaching it with items left unexamined ends the
module's scan and is disclosed as a cut naming the container where it ended: a tensor
past that point is missed. Only a module whose containers hold this many items reaches
it, so ordinary tensor-free holders stay silent.
"""

_HELD_KEY_MAX_CHARS = 40
_HELD_KEY_ESCAPES = (("%", "%25"), (".", "%2E"), (":", "%3A"))


def _render_held_key(key: Any) -> str:
    """Render one dict key as a dot-free, colon-free address segment.

    ``repr(key)`` reads like the Python access (``cache['cpu']``, ``cache[1]``), so a str
    key never collides with an int that prints the same. ``%``, ``.`` and ``:`` are
    percent-escaped because ``.`` separates the owning module from the buffer name and
    ``:`` separates an address from a pass number. A hostile ``__repr__`` falls back to
    the key's type name; long renderings are cut to ``_HELD_KEY_MAX_CHARS``.

    Parameters
    ----------
    key:
        Dict key.

    Returns
    -------
    str
        Escaped rendering without brackets.
    """

    try:
        rendered = repr(key)
    except Exception:  # noqa: BLE001 - a hostile __repr__ must not break capture setup
        rendered = f"<{type(key).__name__}>"
    for raw, escaped in _HELD_KEY_ESCAPES:
        rendered = rendered.replace(raw, escaped)
    if len(rendered) > _HELD_KEY_MAX_CHARS:
        rendered = rendered[: _HELD_KEY_MAX_CHARS - 1] + "~"
    return rendered


class _HeldScan:
    """Bounded, cycle-guarded preorder walk of one module's held containers.

    Shared across the module's attributes so the bounds are per module. Names are
    reserved in scan order; a repeat gets ``#2``, ``#3``, ... so no tensor is dropped.
    A bound cut is recorded only when a plain tensor is actually found past it: a
    container at the depth bound is searched (detect only, never stamped), and the
    object bound is recorded at the first tensor beyond the stamped ones. A depth cut
    whose tensors are all stamped elsewhere in the module (or are its registered
    buffers) is dropped in ``finish``, since every read of them resolves. Every item
    examined, in either mode, counts against ``_HELD_SCAN_MAX_VISITS``; reaching it with
    items left ends the module's scan and records a visit-cap cut naming the container.
    """

    def __init__(self, truncations: list[str] | None, module_address: str) -> None:
        self.prefix = f"{module_address}." if module_address else ""
        self.taken: set[str] = set()
        self.stamped = 0
        self.stamped_ids: set[int] = set()
        self.depth_cuts: list[tuple[str, list[int]]] = []
        self.visits = 0
        self.entered: set[int] = set()
        self.truncations = truncations
        self.exhausted = False

    def reserve(self, name: str) -> str:
        """Return ``name``, or ``name#<n>`` with the smallest free ``n >= 2``; reserve it."""

        unique = name
        ordinal = 2
        while unique in self.taken:
            unique = f"{name}#{ordinal}"
            ordinal += 1
        self.taken.add(unique)
        return unique

    def _cut(self, note: str) -> None:
        """Record one bound cut for the caller's disclosure."""

        if self.truncations is not None:
            self.truncations.append(self.prefix + note)

    def _visit(self, holder: str) -> bool:
        """Count one examined item of ``holder``; at the visit cap record the cut and stop.

        Called only once an item has been fetched, so reaching the cap here always leaves
        that item (and anything after it) unexamined.
        """

        if self.visits >= _HELD_SCAN_MAX_VISITS:
            if not self.exhausted:
                self._cut(f"{holder} (visit cap {_HELD_SCAN_MAX_VISITS} items)")
            self.exhausted = True
            return False
        self.visits += 1
        return True

    def stamp(self, tensor: torch.Tensor) -> None:
        """Remember a yielded tensor so a depth cut holding only it is not disclosed."""

        self.stamped_ids.add(id(tensor))

    def finish(self) -> None:
        """Record the depth cuts that hide at least one tensor not stamped by this module."""

        for name, found in self.depth_cuts:
            if any(found_id not in self.stamped_ids for found_id in found):
                self._cut(f"{name} (depth bound {_HELD_SCAN_MAX_DEPTH})")

    def admit(self, name: str) -> bool:
        """Count one plain tensor to stamp; at the object bound record the cut and stop."""

        if self.stamped >= _HELD_SCAN_MAX_OBJECTS:
            self.exhausted = True
            self._cut(f"{name} (object bound {_HELD_SCAN_MAX_OBJECTS})")
            return False
        self.stamped += 1
        return True

    def held_tensor_ids(self, container: Any, name: str) -> list[int]:
        """Detect-only search: ids of the plain tensors ``container`` holds at any depth.

        Nothing is stamped or entered for the main walk; items share the visit cap.
        """

        found: list[int] = []
        seen = {id(container)}
        stack = [_held_items(container)]
        while stack:
            entry = next(stack[-1], None)
            if entry is None:
                stack.pop()
                continue
            if not self._visit(name):
                break
            item = entry[1]
            if _is_plain_tensor(item):
                found.append(id(item))
            elif _is_enterable(item) and id(item) not in seen and id(item) not in self.entered:
                seen.add(id(item))
                stack.append(_held_items(item))
        return found

    def walk(self, container: Any, path: str, depth: int) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield plain tensors under ``container`` (named ``path``, items at ``depth``).

        Iterative with lazy item iterators, so a huge dict costs nothing past the bounds
        and the depth bound is independent of Python's recursion limit.
        """

        self.entered.add(id(container))
        stack: list[tuple[str, int, Iterator[tuple[Any, Any]]]] = [
            (path, depth, _held_items(container))
        ]
        while stack and not self.exhausted:
            parent, level, items = stack[-1]
            entry = next(items, None)
            if entry is None:
                stack.pop()
                continue
            if not self._visit(parent):
                return
            (key, is_key), item = entry
            is_tensor = _is_plain_tensor(item)
            if not is_tensor and not (_is_enterable(item) and id(item) not in self.entered):
                continue
            segment = _render_held_key(key) if is_key else str(key)
            name = self.reserve(f"{parent}[{segment}]")
            if is_tensor:
                if self.admit(name):
                    self.stamp(item)
                    yield name, item
            elif level < _HELD_SCAN_MAX_DEPTH:
                self.entered.add(id(item))
                stack.append((name, level + 1, _held_items(item)))
            else:
                found = self.held_tensor_ids(item, name)
                if found:
                    self.depth_cuts.append((name, found))


def _is_plain_tensor(item: Any) -> bool:
    """True for a tensor that is not a Parameter (Parameters have their own sources)."""

    return isinstance(item, torch.Tensor) and not isinstance(item, nn.Parameter)


def _is_enterable(item: Any) -> bool:
    """True for a non-empty list/tuple/dict; str, bytes and numbers are leaves."""

    return isinstance(item, (list, tuple, dict)) and bool(item)


def _held_items(container: Any) -> Iterator[tuple[tuple[Any, bool], Any]]:
    """Yield ``((key_or_index, is_dict_key), item)`` lazily for a list/tuple/dict."""

    if isinstance(container, dict):
        for key, item in container.items():
            yield (key, True), item
    else:
        for index, item in enumerate(container):
            yield (index, False), item


def iter_module_held_plain_tensors(
    module: nn.Module, truncations: list[str] | None = None, module_address: str = ""
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the plain tensors a module holds outside its parameter/buffer registries.

    A module-held plain tensor exists before the forward and is read from module
    state, so model preparation stamps it as a buffer source exactly like a
    registered buffer (its reads then root at a ``buffer`` node instead of
    dangling). Holder shapes and their address suffixes:

    * a public tensor attribute: ``name``;
    * an item of a list/tuple attribute: ``name.<index>`` (the original spelling);
    * a dict value, e.g. timm's eval-mode attention-bias cache filled by an earlier
      forward: ``name[<key>]`` with the key rendered by ``_render_held_key``
      (``attention_bias_cache['cpu']``), so the address's last ``.`` still splits the
      owning module from the buffer name;
    * nested list/tuple/dict values below those: ``[<index>]`` / ``[<key>]`` appended
      per level (``cache['outer']['cpu']``, ``lists['k'][0]``, ``pairs.0[1]``).

    Names are never dropped: a rendering that repeats an earlier name gets ``#2``,
    ``#3``, ... in scan order. Below the attribute's own list/tuple items the scan is
    bounded (``_HELD_SCAN_MAX_DEPTH`` levels and ``_HELD_SCAN_MAX_OBJECTS`` tensors
    per module, cycle-guarded). A depth or object cut is described in ``truncations``
    only when a plain tensor actually lies past it (and, for a depth cut, is not
    stamped elsewhere in the module), so tensor-free containers never produce one.
    Cost is capped at ``_HELD_SCAN_MAX_VISITS`` examined items per module; reaching
    that cap with items left ends the module's scan and is described too, naming the
    container where it ended (a tensor past that point is missed).

    Private (``_``) and ``tl_`` attributes and Parameters are never yielded.
    A tensor created inside ``forward`` is not module state at preparation time
    and is never seen here.

    Parameters
    ----------
    module:
        Module whose own ``__dict__`` is scanned (no recursion into submodules).
    truncations:
        Optional collector for one description per bound that cut the scan.
    module_address:
        The module's address, prefixed to each ``truncations`` description.

    Yields
    ------
    tuple[str, torch.Tensor]
        Address suffix relative to the module, and the held tensor.
    """

    scan = _HeldScan(truncations, module_address)
    scan.taken.update(name for name in module.__dict__ if not name.startswith(("_", "tl_")))
    for _, registered in module.named_buffers(recurse=False):
        scan.stamp(registered)
    for attr_name, attr_val in module.__dict__.items():
        if attr_name.startswith(("_", "tl_")):
            continue
        if isinstance(attr_val, torch.Tensor):
            if not isinstance(attr_val, nn.Parameter):
                scan.stamp(attr_val)
                yield attr_name, attr_val
        elif isinstance(attr_val, dict) and attr_val:
            yield from scan.walk(attr_val, attr_name, 1)
        elif isinstance(attr_val, (list, tuple)):
            # Top-level items keep the original unbounded ``name.<index>`` scan; only
            # containers below them enter the bounded walk.
            scan.entered.add(id(attr_val))
            for index, item in enumerate(attr_val):
                name = scan.reserve(f"{attr_name}.{index}")
                if isinstance(item, torch.Tensor):
                    if not isinstance(item, nn.Parameter):
                        scan.stamp(item)
                        yield name, item
                elif _is_enterable(item) and id(item) not in scan.entered:
                    yield from scan.walk(item, name, 2)
    scan.finish()
