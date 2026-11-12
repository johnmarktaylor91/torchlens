"""Patching a model input leaf: the prebuilt patching helpers' path for input-homed facets.

Live hooks never fire at input sites, so a whole-tensor facet whose home is a
model input is patched by rebuilding the input tree with the leaf at the
capture-recorded input address (``Op.io_role``) replaced, then rerunning.
:mod:`torchlens._semantic.patching` calls ``_model_input_home_role`` to detect
such a facet and ``_patch_input_leaf`` to rebuild the input.

Every patched rerun (hook path or input path) goes through ``_rerun_with_inputs``,
which hands the rerun the same forward keyword inputs (``trace_kwargs["input_kwargs"]``)
the clean and corrupted baselines were traced with.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .._input_coerce import _coerce_inputs
from .._input_route import _RoutedInputs
from .._record_type import record_type
from .._utils._record_group_table import record_member
from .._utils.arg_handling import normalize_input_args
from .._utils.introspection import INPUT_SEARCH_DEPTH_LIMIT, get_vars_of_type_from_obj
from ..backends.torch.backend import _get_input_arg_names
from ..intervention._trace_run import rerun_in_place


def _rerun_with_inputs(
    log: Any, model: nn.Module, inputs: Any, input_kwargs: dict[Any, Any] | None
) -> Any:
    """Rerun ``log`` in place on ``inputs`` with the baselines' forward keyword inputs.

    The rerun entry takes one input slot and coerces it with no keyword inputs, so a
    patched rerun used to drop the ``input_kwargs`` the baselines were traced with
    (``use_cache=False`` on a Hugging Face LM): the forward took another code path,
    diverged from the corrupted graph, and the patch hook resolved to no site. When
    keyword inputs are present they ride a :class:`_RoutedInputs` carrier, coerced as
    the capture door coerces them; ``log.input`` keeps the caller's input as before.

    Parameters
    ----------
    log:
        Fork of the corrupted baseline, rerun in place.
    model:
        Model to execute.
    inputs:
        The (possibly patched) positional input of the corrupted run.
    input_kwargs:
        Forward keyword inputs of the baselines, or ``None``.

    Returns
    -------
    Any
        The rerun trace.
    """

    if not input_kwargs:
        return rerun_in_place(log, model, inputs)
    routed = _RoutedInputs(*_coerce_inputs(model, inputs, dict(input_kwargs)))
    result = rerun_in_place(log, model, routed)
    result.input = inputs
    return result


def _input_op_by_label(log: Any, label: str) -> Any | None:
    """Return the trace's input op whose label is ``label``, or ``None``."""

    return next((op for op in getattr(log, "input_ops", ()) if str(op.label) == label), None)


def _model_input_home_role(log: Any, facet_name: str, address: str) -> str | None:
    """Return the recorded input address when a whole-tensor facet homes on a model input.

    Parameters
    ----------
    log:
        Trace holding the facet.
    facet_name:
        Facet to inspect.
    address:
        Module address exposing the facet.

    Returns
    -------
    str | None
        The home input op's hierarchical ``io_role`` address (for example
        ``"input.x.0.nested"``), or ``None`` when the home is a regular op
        (live hooks handle it) or no facet spec is reachable.

    Raises
    ------
    ValueError
        If a facet homed on a model input is not the identity view of the home
        tensor (patching the raw input would silently write outside the facet),
        or if the home input op cannot be bound to a recorded input address --
        falling back to the hook path would silently run an UNPATCHED
        counterfactual, because live hooks never fire at input sites.
    """

    try:
        spec = log.modules[address].facets[facet_name].spec
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return None
    home = getattr(spec, "home", None)
    if home is None or record_type(home, None) != "input":
        return None
    home_label = str(getattr(home, "label", ""))
    home_op = _input_op_by_label(log, home_label)
    if home_op is None:
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} homes on model input "
            f"{home_label!r}, which is not among this trace's input ops; the input "
            "patch cannot be bound and live hooks never fire at input sites."
        )
    if tuple(getattr(spec, "transforms", ()) or ()) or not bool(spec.write_mask().all()):
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} is a transformed or sliced view of "
            f"model input {home_label!r}. Sliced input facets cannot be patched: live hooks "
            "never fire at input sites, and whole-input replacement would write outside the "
            "facet."
        )
    io_role = record_member(home_op, "io_role", None)
    if not io_role:
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} homes on model input "
            f"{home_label!r}, but that op records no io_role input address; the input "
            "patch cannot be bound and live hooks never fire at input sites."
        )
    return str(io_role)


def _resolve_input_leaf_by_role(
    x: Any, model: nn.Module, io_role: str
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the leaf of ``x`` recorded at input address ``io_role``, plus all leaves.

    Mirrors capture-time input flattening exactly (see
    ``TorchBackend._fetch_label_move_input_tensors``): positional args are
    normalized against the model's forward signature, each arg's tensors are
    discovered with the same bounded traversal, addresses are
    ``input.<argname>[.<path>]``, and a repeated tensor object keeps only its
    first address -- so the recorded ``io_role`` binds by ADDRESS, never by
    ordinal position in some other walk order.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        The target leaf and every distinct tensor leaf of the tree.

    Raises
    ------
    ValueError
        If no leaf of ``x`` sits at the recorded address (fail closed: a
        silent fallback would rerun an unpatched counterfactual).
    """

    args = normalize_input_args(x, model)
    arg_names = _get_input_arg_names(model, args)
    target: torch.Tensor | None = None
    leaves: list[torch.Tensor] = []
    seen: set[int] = set()
    for arg, arg_name in zip(args, arg_names):
        for tensor, addr, _addr_full in get_vars_of_type_from_obj(
            arg,
            torch.Tensor,
            search_depth=INPUT_SEARCH_DEPTH_LIMIT,
            return_addresses=True,
        ):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            leaves.append(tensor)
            tensor_addr = f"input.{arg_name}" + (f".{addr}" if addr else "")
            if tensor_addr == io_role:
                target = tensor
    if target is None:
        raise ValueError(
            f"Input tree has no tensor leaf at recorded input address {io_role!r}; "
            "the input-homed facet patch cannot be bound to a rerun leaf."
        )
    return target, leaves


def _patch_input_leaf(
    x: Any,
    model: nn.Module,
    io_role: str,
    patch: Callable[[torch.Tensor], torch.Tensor],
) -> Any:
    """Return the input tree with the leaf at address ``io_role`` patched everywhere.

    The leaf is resolved by the capture-recorded hierarchical input address
    (``Op.io_role``), never by ordinal position: capture flattens inputs in
    BFS container order while a naive tree walk visits leaves in DFS order, so
    ordinal indexing silently patched the WRONG leaf on mixed-nesting inputs.
    Because capture dedupes a repeated tensor object into ONE input op,
    patching that op replaces the tensor at EVERY site it appears. Container
    types (Mapping subclasses, namedtuples, nested objects) are preserved by
    rebuilding through ``copy.deepcopy`` with every unpatched tensor leaf
    passed through by identity.

    Parameters
    ----------
    x:
        Model input tree exactly as passed to the baseline trace.
    model:
        Model whose forward signature determines positional-arg naming.
    io_role:
        Recorded hierarchical input address of the leaf to patch.
    patch:
        Callable producing the replacement tensor for the selected leaf.

    Returns
    -------
    Any
        Rebuilt input tree with the addressed leaf replaced at every site.

    Raises
    ------
    ValueError
        If no leaf sits at the recorded address, the patched tensor changes
        shape, or the tree cannot be rebuilt.
    """

    target, leaves = _resolve_input_leaf_by_role(x, model, io_role)
    patched = patch(target)
    if tuple(patched.shape) != tuple(target.shape):
        raise ValueError(
            f"Input patch changed the leaf shape from {tuple(target.shape)} to "
            f"{tuple(patched.shape)}; input patching must preserve shape."
        )
    memo: dict[int, Any] = {id(leaf): leaf for leaf in leaves}
    memo[id(target)] = patched
    try:
        return copy.deepcopy(x, memo)
    except Exception as exc:
        raise ValueError(
            f"Could not rebuild the input tree around patched leaf {io_role!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
