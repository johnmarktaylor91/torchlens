"""Deferred resolution for thin rows: observe at the call, resolve when the row fills.

The thin emit (:mod:`._thin_emit`) serves a plain admitted call. For the deferrable
families the call-time step records only what the resolver needs, read while the
call's state exists, under one private fields key (:data:`DEFERRED`); the row's fill
(:meth:`._thin_record._ThinRow._fill`, the first need or the settle) pops the key and
runs each resolver on its recorded inputs before anything reads the fields. Every
resolver is the builder's own expression on the recorded objects, so the values are
the ones today's emit writes at the call:

- the argument template: ``_set_arg_template_fields`` observes the facts
  (``observe_template``: labels, layouts, literal values, copies of unlabeled
  tensors, the function key) and :func:`resolve_arg_template_fields` resolves them;
- the function config: ``extract_salient_args`` over ``record_call_values`` of the
  arguments (tensor shapes and dtypes, copied containers) when the layer type's
  extractor reads arguments (a type with no extractor writes ``{}`` at the call).

Rows with equal template inputs share one observation through the template key memo
of ``observe_template``; the function config is not memoized, because equal records
can hold values of different types (``1``, ``1.0``, ``True``) that its extractors
keep. This module imports nothing from the capture import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..._capture._recorded_tensor import UnrecordableArgument, record_call_values
from ..._capture.salient_args import extract_salient_args, salient_reads_args
from ._arg_template_resolve import resolve_arg_template_fields

__all__ = ("DEFERRED", "defer_arg_template", "resolve_deferred", "set_function_config")

DEFERRED = "_thin_deferred"
"""The private fields key holding a thin row's deferred resolutions until its fill."""

_Resolution = tuple[Callable[[dict[str, Any], Any], None], Any]


def _defer(fields: dict[str, Any], resolution: _Resolution) -> None:
    """Add one resolution to the row's deferred list."""

    pending: list[_Resolution] | None = fields.get(DEFERRED)
    if pending is None:
        fields[DEFERRED] = [resolution]
    else:
        pending.append(resolution)


def resolve_deferred(fields: dict[str, Any]) -> None:
    """Run and drop a thin row's deferred resolutions, in the order the call recorded them."""

    pending = fields.pop(DEFERRED, None)
    if pending:
        for resolve, recorded in pending:
            resolve(fields, recorded)


def defer_arg_template(fields: dict[str, Any], observed: tuple[Any, ...], parts: Any) -> None:
    """Defer ``resolve_arg_template_fields(fields, observed, parts)`` to the row's fill."""

    _defer(fields, (_resolve_template, (observed, parts)))


def _resolve_template(fields: dict[str, Any], recorded: tuple[Any, Any]) -> None:
    """Write the argument template fields from the facts observed at the call."""

    observed, parts = recorded
    resolve_arg_template_fields(fields, observed, parts)


def set_function_config(fields: dict[str, Any], call: tuple[Any, ...], defer: bool) -> None:
    """Write ``func_config`` now, or record its inputs and resolve it at the row's fill.

    ``call`` is ``(layer_type, func_name, args, kwargs)``; the parameter shapes are
    ``fields["param_shapes"]``. A layer type with no extractor (its config is ``{}``)
    and a value the recorder cannot replace resolve now.
    """

    layer_type, func_name, args, kwargs = call
    param_shapes: list[tuple[int, ...]] | None = fields.get("param_shapes")
    if defer and salient_reads_args(layer_type):
        try:
            recorded_args, recorded_kwargs = record_call_values(args, kwargs)
        except UnrecordableArgument:
            pass
        else:
            recorded = (layer_type, func_name, recorded_args, recorded_kwargs, param_shapes)
            _defer(fields, (_resolve_config, recorded))
            return
    fields["func_config"] = extract_salient_args(layer_type, func_name, args, kwargs, param_shapes)


def _resolve_config(fields: dict[str, Any], recorded: tuple[Any, ...]) -> None:
    """Write ``func_config`` from the call values recorded at the call."""

    fields["func_config"] = extract_salient_args(*recorded)
