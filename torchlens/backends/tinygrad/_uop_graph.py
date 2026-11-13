"""UOp-graph walks the tinygrad capture records ops from.

Extracted from ``backend`` so that module stays under its size ceiling.
:func:`_data_flow_uops` picks the UOps a capture records: those whose value
reaches an output as data.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from hashlib import blake2b
from typing import Any, cast


def _unique_uops(outputs: Sequence[Any]) -> tuple[Any, ...]:
    """Return UOps reachable from outputs in topological order.

    Parameters
    ----------
    outputs
        tinygrad output tensors.

    Returns
    -------
    tuple[Any, ...]
        Unique UOps in first-seen topological order.
    """

    seen: set[int] = set()
    ordered: list[Any] = []
    for output in outputs:
        for uop in cast(Any, output).uop.toposort():
            if id(uop) in seen:
                continue
            seen.add(id(uop))
            ordered.append(uop)
    return tuple(ordered)


def _src(uop: Any, edit_sites: Mapping[int, Any]) -> tuple[Any, ...]:
    """The UOps a capture treats as ``uop``'s inputs.

    An edit's replacement UOp (a key of ``edit_sites``) stands for the call
    result it replaced: its inputs are that result's inputs, so the edit's
    own ops are never walked and stay out of the capture.
    """

    return tuple(getattr(edit_sites.get(id(uop), uop), "src", ()) or ())


def edited_unique_uops(outputs: Sequence[Any], edit_sites: Mapping[int, Any]) -> tuple[Any, ...]:
    """Return the UOps of an edited forward in the order of the unedited forward.

    Labels are reserved in this order, so an edited capture labels each op as
    the plain capture of the same forward does. Each reachable UOp is mapped
    to its unedited counterpart (each replacement back to the result it
    replaced, everything downstream rebuilt over those); tinygrad interns
    UOps, so the counterparts are the plain forward's own UOps and their
    :func:`_unique_uops` order is the plain order.

    Parameters
    ----------
    outputs
        tinygrad output tensors of the edited forward.
    edit_sites
        ``id(replacement UOp) -> replaced result UOp``.

    Returns
    -------
    tuple[Any, ...]
        Reachable UOps, the edits' own ops excluded, in plain-forward order.
    """

    real = _edited_postorder(outputs, edit_sites)
    plain = _plain_counterparts(real, edit_sites)
    order: dict[int, int] = {}
    for output in outputs:
        for uop in plain[id(cast(Any, output).uop)].toposort():
            order.setdefault(id(uop), len(order))
    return tuple(sorted(real, key=lambda uop: order[id(plain[id(uop)])]))


def _edited_postorder(outputs: Sequence[Any], edit_sites: Mapping[int, Any]) -> list[Any]:
    """The UOps reachable from ``outputs`` through :func:`_src`, inputs before users."""

    real: list[Any] = []
    seen: set[int] = set()
    stack: list[tuple[Any, bool]] = [(cast(Any, output).uop, False) for output in outputs]
    while stack:
        uop, done = stack.pop()
        if done:
            real.append(uop)
        elif id(uop) not in seen:
            seen.add(id(uop))
            stack.append((uop, True))
            stack.extend((src, False) for src in _src(uop, edit_sites) if id(src) not in seen)
    return real


def _plain_counterparts(real: Sequence[Any], edit_sites: Mapping[int, Any]) -> dict[int, Any]:
    """``id(edited UOp) -> the UOp the unedited forward made in its place``.

    ``real`` must list inputs before users, so every input is mapped first.
    """

    plain: dict[int, Any] = {}
    for uop in real:
        base = edit_sites.get(id(uop), uop)
        src = tuple(getattr(base, "src", ()) or ())
        rebuilt = tuple(plain.get(id(item), item) for item in src)
        changed = any(a is not b for a, b in zip(rebuilt, src, strict=True))
        plain[id(uop)] = base.replace(src=rebuilt) if changed else base
    return plain


def _is_materializable_uop(uop: Any) -> bool:
    """Return whether a UOp can be saved as a tinygrad Tensor payload.

    A structural test: nothing is realized here (realize-once capture takes
    every payload from one later evaluation). The UOp must give a tinygrad
    Tensor with a shape and a dtype that has a host format. The host-only
    ``NPY`` device holds numpy-backed buffers and has no renderer: its buffers
    are readable, but a view over one (a ``RESHAPE`` before the copy to the
    compute device) has no value of its own until copied, so it is not a
    payload.

    Parameters
    ----------
    uop
        Candidate tinygrad UOp.

    Returns
    -------
    bool
        True when tinygrad can expose shape and host payload for ``uop``.
    """

    try:
        from tinygrad import Tensor

        tensor = Tensor(uop)
        tuple(tensor.shape)
        device = str(tensor.device)
    except Exception:
        return False
    if getattr(tensor.dtype.base, "fmt", None) is None:
        return False
    return not device.startswith("NPY") or _uop_name(uop) == "BUFFER"


def _uop_name(uop: Any) -> str:
    """Return a stable tinygrad UOp name.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Operation name without the ``Ops.`` prefix.
    """

    op = getattr(uop, "op", None)
    return str(getattr(op, "name", op)).removeprefix("Ops.")


def _uop_signature(uop: Any, memo: dict[int, str] | None = None) -> str:
    """Return a structural UOp signature string.

    Parameters
    ----------
    uop
        tinygrad UOp.
    memo
        Signatures already computed in this call, by UOp id. Each source
        enters as a digest of its own signature, computed once, so a value
        the graph reuses is not re-expanded at every use (the fully expanded
        string doubles per reuse: exponential time and memory in depth).

    Returns
    -------
    str
        Operation/dtype/arg signature over the source signatures' digests.
    """

    memo = {} if memo is None else memo
    if id(uop) not in memo:
        src = getattr(uop, "src", ()) or ()
        kids = ",".join(
            blake2b(_uop_signature(c, memo).encode(), digest_size=16).hexdigest() for c in src
        )
        memo[id(uop)] = (
            f"{_uop_name(uop)}:{getattr(uop, 'dtype', None)}:{getattr(uop, 'arg', None)}[{kids}]"
        )
    return memo[id(uop)]


def _op_func_name(observed_names: Sequence[str], op_name: str) -> str:
    """The function name an op is recorded under.

    The observer reports the callable tinygrad passed to ``Tensor._apply_uop``.
    tinygrad routes most ops through internal helpers (``_mop`` for movement
    ops, ``_rop`` for reductions, a ``lambda`` for elementwise math), and those
    names do not say what the op computes. The last observed name that is not
    such a helper is used; otherwise the UOp's op name (``add``, ``reshape``,
    ``reduce``).

    Parameters
    ----------
    observed_names
        Names the observer recorded for the UOp, in call order.
    op_name
        The UOp's op name (:func:`_uop_name`).

    Returns
    -------
    str
        The recorded function name.
    """

    informative = [name for name in observed_names if not name.startswith(("_", "<"))]
    return informative[-1] if informative else op_name.lower()


def _data_flow_uops(
    uops: Sequence[Any],
    outputs: Sequence[Any],
    source_ids: Collection[int],
    *,
    boundary_ids: Collection[int] = frozenset(),
    edit_sites: Mapping[int, Any] | None = None,
) -> tuple[Any, ...]:
    """Return the UOps whose values reach an output as data, in ``uops`` order.

    The walk starts at the outputs and follows ``src`` edges through
    materializable UOps only, stopping at the capture's sources (``source_ids``,
    the input UOps). A non-materializable UOp ends the walk: it carries no
    tensor value, so what feeds it is not data of the forward. That leaves out
    the scalar ``CONST`` elements of a movement op's shape vector and the
    storage ``BUFFER`` behind an input, which a plain topological sort reaches
    but which no recorded op consumes. The same rule decides which ops torch
    keeps: a value no output depends on and no input produces is an orphan.

    Parameters
    ----------
    uops
        UOps reachable from the outputs, in topological order (:func:`_unique_uops`).
    outputs
        tinygrad output tensors.
    source_ids
        ``id()`` of every source UOp (the capture's inputs).
    boundary_ids
        ``id()`` of UOps recorded whole: kept when reached, but the walk does
        not continue into what produced them (a pre-forward random weight's
        generator subgraph, see ``_realize_once``).
    edit_sites
        ``id(replacement UOp) -> replaced result UOp`` of an edited capture;
        the walk passes from a replacement to the replaced result's inputs.

    Returns
    -------
    tuple[Any, ...]
        The data-flow UOps, sources excluded, each materializable.
    """

    materializable: dict[int, bool] = {}
    kept: set[int] = set()
    stack = [cast(Any, output).uop for output in outputs]
    while stack:
        uop = stack.pop()
        key = id(uop)
        if key in kept or key in source_ids:
            continue
        if key not in materializable:
            materializable[key] = _is_materializable_uop(uop)
        if not materializable[key]:
            continue
        kept.add(key)
        if key not in boundary_ids:
            stack.extend(_src(uop, edit_sites or {}))
    return tuple(uop for uop in uops if id(uop) in kept)
