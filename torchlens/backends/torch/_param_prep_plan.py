"""Parameter records prepared once per unchanged model, verified before each reuse.

Every capture builds one ``Param`` record per unique parameter and stamps the
parameter with a fresh barcode. Most of that work reads facts that stay the
same between two traces of an unchanged model: each parameter's address, the
module that owns it, its shape, dtype, element count and memory, and the
alias addresses of tied parameters and shared modules. This module keeps
those facts per model as a plan and hands the plan back to the next trace
only after checking it against the live model.

The plan is keyed on:

- the module tree, through the recorded walk of
  :func:`~torchlens._utils._model_walk.current_tree_record` (compared by
  identity, so a submodule added, removed, replaced, renamed or moved since the
  plan was made discards it);
- each module's prepared address, read live;
- each parameter slot: its name and the identity of the parameter in it, in
  registry order (a parameter swapped, added, removed or set to ``None``
  discards the plan);
- each unique parameter's class, dtype, layout, device type, shape and
  ``requires_grad``.

Nothing else is cached. The barcode, the stamp, the ``Param`` record, the
optimizer membership and the forced-gradient flag are made fresh by every
trace, and parameter values are never read, so a value changed between traces
is seen exactly as before. A plan is not made while any parameter is lazy or
not strided (a sparse tensor's memory depends on its values).

The plan holds the model weakly (it is the key of a weak dictionary) and every
parameter through a weak reference, so it keeps no model or tensor alive.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..._utils._model_walk import current_tree_record
from ..._utils._tensor_label import get_module_meta

if TYPE_CHECKING:
    from ..._data_classes.param import Param

#: Model -> its parameter plan. Weak in the model; the plan holds parameters weakly.
_PLANS: weakref.WeakKeyDictionary[nn.Module, ParamPlan] = weakref.WeakKeyDictionary()

# Positions in a unique-parameter entry of :attr:`ParamPlan.uniques`.
REF, ADDRESS, MODULE_ADDRESS, NAME, SHAPE, DTYPE, NUMEL, MEMORY = range(8)
EXTRA_ADDRESSES, EXTRA_MODULE_ADDRESSES, PID = 8, 9, 10
_CLS, _DEVICE_TYPE, _REQUIRES_GRAD = 11, 12, 13


class ParamPlan:
    """What one trace's parameter preparation found, for the next trace to reuse.

    Attributes
    ----------
    record:
        The recorded tree walk the plan was made over (identity is the key).
    addresses:
        Each walked module's prepared address, in walk order.
    names, refs:
        Every non-``None`` parameter slot in walk and registry order: its name
        and a weak reference to the parameter in it.
    uniques:
        One entry per unique parameter, in the order its record was made.
    pid_to_address:
        ``id(parameter)`` -> address, valid only while ``refs`` verify.
    """

    __slots__ = ("addresses", "names", "pid_to_address", "record", "refs", "uniques")

    def __init__(self, record: tuple[Any, ...], addresses: tuple[str, ...]) -> None:
        """Start an empty plan over ``record``."""

        self.record = record
        self.addresses = addresses
        self.names: tuple[str, ...] = ()
        self.refs: tuple[Any, ...] = ()
        self.uniques: tuple[tuple[Any, ...], ...] = ()
        self.pid_to_address: dict[int, str] = {}


def prepared_address(module: nn.Module) -> str:
    """Return a prepared module's TorchLens address (``""`` when unprepared)."""

    meta = get_module_meta(module)
    return "" if meta is None or meta.address is None else meta.address


def _slots(modules: tuple[nn.Module, ...]) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """Return every non-``None`` parameter slot's name and a weak reference to its parameter."""

    names: list[str] = []
    refs: list[Any] = []
    for module in modules:
        params = module._parameters
        for name in params:
            param = params[name]
            if param is not None:
                names += (name,)
                refs += (weakref.ref(param),)
    return tuple(names), tuple(refs)


def _unique_entry(param: nn.Parameter, log: Param) -> tuple[Any, ...] | None:
    """Return the plan entry for one unique parameter, or ``None`` if it cannot be planned."""

    if getattr(log, "_lazy_at_prep", False) or param.layout is not torch.strided:
        return None
    address = log.address
    module_address = log.module_address
    return (
        weakref.ref(param),
        address,
        module_address,
        log.name,
        log.shape,
        log.dtype,
        log.num_params,
        int(log.param_memory),
        tuple(name for name in log.all_addresses if name != address),
        tuple(name for name in log.all_module_addresses if name != module_address),
        id(param),
        param.__class__,
        param.device.type,
        log.is_trainable,
    )


def record_param_plan(
    model: nn.Module,
    modules: tuple[nn.Module, ...],
    stamped: list[nn.Parameter],
    param_logs: dict[str, Param],
) -> None:
    """Make ``model``'s plan from the records a full preparation just built.

    ``stamped`` and ``param_logs`` list the unique parameters in the same
    order (one stamp per record). When the tree is walked live, a parameter
    cannot be weakly referenced or planned, no plan is kept.
    """

    _PLANS.pop(model, None)
    record = current_tree_record(model)
    if record is None or len(stamped) != len(param_logs):
        return
    try:
        plan = ParamPlan(record, tuple(prepared_address(module) for module in modules))
        plan.names, plan.refs = _slots(modules)
        uniques = []
        for param, log in zip(stamped, param_logs.values(), strict=True):
            entry = _unique_entry(param, log)
            if entry is None:
                return
            uniques.append(entry)
    except TypeError:
        return  # a parameter that cannot be weakly referenced: prepare in full each time
    plan.uniques = tuple(uniques)
    plan.pid_to_address = {entry[PID]: entry[ADDRESS] for entry in uniques}
    _PLANS[model] = plan


def _slots_unchanged(plan: ParamPlan, modules: tuple[nn.Module, ...]) -> bool:
    """Return whether every module's address and parameter slots are the planned ones."""

    names, refs, addresses = plan.names, plan.refs, plan.addresses
    total = len(names)
    if len(modules) != len(addresses):
        return False
    position = 0
    for index in range(len(modules)):
        module = modules[index]
        if prepared_address(module) != addresses[index]:
            return False
        params = module._parameters
        for name in params:
            param = params[name]
            if param is None:
                continue
            if position == total or names[position] != name or refs[position]() is not param:
                return False
            position += 1
    return position == total


def _uniques_unchanged(plan: ParamPlan) -> bool:
    """Return whether each unique parameter's class, dtype, layout, device, shape and flag hold."""

    for entry in plan.uniques:
        param = entry[REF]()
        if (
            param is None
            or param.__class__ is not entry[_CLS]
            or param.dtype is not entry[DTYPE]
            or param.layout is not torch.strided
            or param.requires_grad is not entry[_REQUIRES_GRAD]
            or param.shape != entry[SHAPE]
            or param.device.type != entry[_DEVICE_TYPE]
        ):
            return False
    return True


def verified_param_plan(model: nn.Module, modules: tuple[nn.Module, ...]) -> ParamPlan | None:
    """Return ``model``'s plan when every key still holds on the live model, else ``None``.

    A plan that fails its check is discarded, so the full preparation that
    follows makes a new one.
    """

    plan = _PLANS.get(model)
    if plan is None:
        return None
    if (
        current_tree_record(model) is not plan.record
        or not _slots_unchanged(plan, modules)
        or not _uniques_unchanged(plan)
    ):
        _PLANS.pop(model, None)
        return None
    return plan


def planned_models() -> int:
    """Return how many models hold a parameter plan (for tests)."""

    return len(_PLANS)


__all__ = [
    "ADDRESS",
    "DTYPE",
    "EXTRA_ADDRESSES",
    "EXTRA_MODULE_ADDRESSES",
    "MEMORY",
    "MODULE_ADDRESS",
    "NAME",
    "NUMEL",
    "PID",
    "REF",
    "SHAPE",
    "ParamPlan",
    "planned_models",
    "prepared_address",
    "record_param_plan",
    "verified_param_plan",
]
