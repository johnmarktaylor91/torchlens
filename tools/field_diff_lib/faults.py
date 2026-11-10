"""Planted faults and the mutation probe.

A planted fault changes one thing in a live trace before it is dumped, so a
comparison of a clean dump against the planted one must report it. The
faults are the proof that the differential can see a changed value, a
changed alias and a missing field.
"""

from __future__ import annotations

from typing import Any

import torch

from .canonical import tensor_hash

#: The case every fault is planted in.
PLANT_CASE = "two_relu"

#: Fault name -> what it changes.
FAULTS: dict[str, str] = {
    "value": "one saved activation is changed in place (content only)",
    "meta": "one metadata field of one op is overridden",
    "alias": "one op's saved output is replaced by another op's equal tensor (identity only)",
    "missing": "one field of one op is deleted, so reading it raises",
    "absent": "one field name is left out of the dump of every record",
}

_FIELD = "func_name"


def _relu_ops(trace: Any) -> list[Any]:
    """Return the relu ops of the fault case, in execution order."""

    ops = [op for op in trace._layer_list if str(op.label).startswith("relu")]  # noqa: SLF001
    if len(ops) < 2:
        raise RuntimeError(f"fault case needs two relu ops, found {[op.label for op in ops]}")
    return ops


def plant(fault: str, trace: Any) -> frozenset[str]:
    """Plant one fault in a live trace.

    Parameters
    ----------
    fault:
        A key of ``FAULTS``.
    trace:
        The live trace of the fault case.

    Returns
    -------
    frozenset[str]
        Field names the dump must leave out (non-empty only for ``absent``).
    """

    if fault == "absent":
        return frozenset({_FIELD})
    first, second = _relu_ops(trace)[:2]
    if fault == "value":
        with torch.no_grad():
            first.out.add_(1.0)
    elif fault == "meta":
        setattr(first, _FIELD, "planted")
    elif fault == "alias":
        if tensor_hash(first.out) != tensor_hash(second.out):
            raise RuntimeError("alias fault needs two equal saved outputs")
        setter = getattr(second, "_internal_set", None)
        if setter is not None:
            setter("out", first.out)
        else:
            second.out = first.out
    elif fault == "missing":
        delattr(first, _FIELD)
    else:
        raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
    return frozenset()


def mutation_probe(sites: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    """Write to each saved output in turn and list the fields whose content follows.

    Storage sharing is compared structurally by the dump; this probe shows it
    behaviourally. For every ``ops[...].out`` tensor, one is added in place,
    every tensor read by the dump is hashed again, the changed fields are
    recorded, and the write is undone.

    Parameters
    ----------
    sites:
        ``(field path, tensor)`` for every tensor the dump read.

    Returns
    -------
    dict[str, Any]
        ``written field -> sorted list of fields whose content changed``.
    """

    unique: dict[int, tuple[list[str], torch.Tensor]] = {}
    for site, tensor in sites:
        unique.setdefault(id(tensor), ([], tensor))[0].append(site)
    result: dict[str, Any] = {}
    for site, tensor in sites:
        if not site.endswith(".out") or not site.startswith("ops[") or site in result:
            continue
        if tensor.layout is not torch.strided or not tensor.dtype.is_floating_point:
            result[site] = "<not probed>"
            continue
        before = {key: tensor_hash(entry[1]) for key, entry in unique.items()}
        try:
            with torch.no_grad():
                tensor.add_(1.0)
            changed = [
                name
                for key, (names, other) in unique.items()
                if tensor_hash(other) != before[key]
                for name in names
            ]
            with torch.no_grad():
                tensor.sub_(1.0)
        except Exception as error:  # noqa: BLE001 - a refused write is the finding
            result[site] = f"<write raises {type(error).__name__}>"
            continue
        result[site] = sorted(set(changed))
    return result
