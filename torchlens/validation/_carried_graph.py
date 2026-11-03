"""Typed refusal for a stock backward that reaches a graph an earlier backward freed.

A model that keeps non-leaf tensors with autograd history across forward calls
(a key table computed from an embedding at construction, a hidden state kept
from the previous call) makes every backward after the first fail inside torch
with ``RuntimeError: Trying to backward through the graph a second time``.
That is the model's own backward contract, not a capture defect, so the
backward validator names it through one stable code instead of letting the
raw torch error escape.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

import torch
from torch import nn

from ..errors import ValidationError

#: Fragment of torch's message for a backward through freed saved tensors,
#: unchanged since torch 1.x.
FREED_GRAPH_MARKER = "backward through the graph a second time"

_SCAN_DEPTH = 3
_REPORT_LIMIT = 3


def is_freed_graph_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is torch's freed-saved-tensors backward error."""

    return isinstance(exc, RuntimeError) and FREED_GRAPH_MARKER in str(exc)


def carried_graph_attributes(model: nn.Module, *, limit: int = _REPORT_LIMIT) -> list[str]:
    """Return dotted paths of module attributes that hold tensors with autograd history.

    Parameters and buffers are autograd leaves and never listed; plain
    attributes, and lists, tuples and dicts of them to a small depth, are.
    """

    found: list[str] = []
    for address, module in model.named_modules():
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            path = f"{address}.{name}" if address else name
            _collect_graph_tensors(value, path, found, depth=0, limit=limit)
            if len(found) >= limit:
                return found
    return found


def _collect_graph_tensors(
    value: Any, path: str, found: list[str], *, depth: int, limit: int
) -> None:
    """Append to ``found`` the path of every tensor inside ``value`` that carries an autograd graph."""
    if len(found) >= limit:
        return
    if isinstance(value, torch.Tensor):
        if value.grad_fn is not None:
            found.append(path)
        return
    if depth >= _SCAN_DEPTH:
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_graph_tensors(item, f"{path}[{index}]", found, depth=depth + 1, limit=limit)
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_graph_tensors(item, f"{path}[{key!r}]", found, depth=depth + 1, limit=limit)


def rethrow_freed_graph(model: nn.Module, exc: BaseException) -> NoReturn:
    """Re-raise ``exc``, typed as ``backward_validation_carried_graph`` when the model proves it.

    The typed refusal needs evidence: torch's freed-saved-tensors error AND a
    module attribute holding a tensor with autograd history. Without both the
    original exception propagates unchanged, so a capture defect that produced
    the same torch error is never relabelled as the model's doing.
    """

    carried = carried_graph_attributes(model) if is_freed_graph_error(exc) else []
    if not carried:
        raise exc
    raise ValidationError(
        "validate_backward_pass refused: the backward reached an autograd graph an "
        "earlier backward already freed, and the model carries autograd history "
        "across forward calls in module attributes "
        f"{carried} (a tensor computed from parameters at construction or in a "
        "previous forward and reused). Remedy: detach the carried tensors or "
        "compute them inside forward, or validate a freshly constructed model",
        code="backward_validation_carried_graph",
        carried_graph_attributes=list(carried),
    ) from exc


@contextmanager
def refusing_carried_graph(model: nn.Module) -> Iterator[None]:
    """Run a backward arm, typing torch's freed-graph error when ``model`` proves it.

    Both backward-validator arms (the stock pass and the captured candidate)
    run inside this context; any other exception propagates unchanged.
    """

    try:
        yield
    except RuntimeError as exc:
        rethrow_freed_graph(model, exc)
