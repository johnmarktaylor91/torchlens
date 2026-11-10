"""One capture's ``Param`` records and parameter stamps.

Capture never changes a parameter's ``requires_grad``: a frozen parameter
stays frozen, so the captured forward runs the same autograd program as an
eager forward (torch picks kernels from it, e.g. the attention paths).
Parameter provenance does not need ``grad_fn``: a parameter read is resolved
by identity at the op boundary from the stamped barcode. The user's flag is
still recorded on ``_tl.requires_grad_before_capture`` (``Param.is_trainable``,
and ``_cleanup_model_session`` restores it if the forward itself changed it).
The one exception is a capture inside ``frozen_parameter_gradients()``: an
analysis that backpropagates through its own capture (edge attribution
patching) needs a gradient path through frozen weights, so there floating
leaf parameters are set to require grad for the session and restored after.

The first trace of a model walks every parameter in full and leaves a plan
(:mod:`._param_prep_plan`); a later trace whose plan verifies against the live
model replays it, drawing barcodes, stamping parameters and building records
in the same order the full walk does.

The caller passes the ``Param`` and ``ParamAccessor`` classes in, which keeps
this module out of the record classes' import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from ..._quantities import Bytes
from ..._utils._model_walk import model_modules
from ..._utils._tensor_label import set_param_meta
from ..._utils.hashing import make_random_barcode
from ..._utils.tensor_utils import (
    _is_uninitialized_lazy_tensor as _is_uninitialized_param,
    get_memory_amount,
)
from . import _param_prep_plan as _plan
from ._module_aliases import _add_shared_module_aliases
from .frozen_parameter_gradients import (
    _FROZEN_PARAMETER_GRADIENTS,
    _forces_frozen_parameter_gradients,
)

if TYPE_CHECKING:
    from ..._data_classes.param import Param, ParamAccessor
    from ..._data_classes.trace import Trace


def _optimized_param_ids(optimizer: Any) -> set[int] | None:
    """Return the ids of the parameters ``optimizer`` holds (``None`` without one)."""

    if optimizer is None:
        return None
    ids: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            ids.add(id(p))
    return ids


def _new_param_log(
    record_type: type[Param],
    trace: Trace,
    param: nn.Parameter,
    slot: tuple[str, str, str],
    optimized: set[int] | None,
) -> Param:
    """Stamp one unique parameter and build its record (the full walk's step).

    ``slot`` is ``(address, module_address, name)``.
    """

    address, module_address, name = slot
    # Lazy modules (nn.LazyLinear etc.) hold UninitializedParameter until the
    # first forward; shape/numel/dtype-kind access raises. Register the
    # identity now with deferred geometry; postprocess step 15 finalizes it.
    lazy = _is_uninitialized_param(param)
    requires_grad_before = param.requires_grad
    if not lazy and _forces_frozen_parameter_gradients(trace, param):
        param.requires_grad = True
    barcode = make_random_barcode()
    set_param_meta(
        param, barcode=barcode, address=address, requires_grad_before=requires_grad_before
    )
    param_log = record_type(
        module_address=module_address,
        name=name,
        shape=() if lazy else tuple(param.shape),
        dtype=param.dtype,
        num_params=0 if lazy else param.numel(),
        param_memory=Bytes(0) if lazy else get_memory_amount(param),
        trainable=requires_grad_before,
        address=address,
        barcode=barcode,
        has_optimizer=None if optimized is None else id(param) in optimized,
    )
    param_log._param_ref = param
    if lazy:
        param_log._lazy_at_prep = True  # type: ignore[attr-defined]
    return param_log


def _walk_params(
    record_type: type[Param],
    trace: Trace,
    modules: tuple[nn.Module, ...],
    optimized: set[int] | None,
) -> tuple[dict[str, Param], dict[int, str], list[nn.Parameter]]:
    """Walk every module's parameters: one record per unique tensor, aliases on it."""

    param_logs: dict[str, Param] = {}
    pid_to_address: dict[int, str] = {}
    # Every parameter this prep STAMPS, so cleanup clears stamps from this
    # inventory: a param popped from ``_parameters`` mid-forward escapes a
    # re-traversal but never this list (session-scoped strong refs).
    stamped: list[nn.Parameter] = []
    for module in modules:
        address = _plan.prepared_address(module)
        module_address = address or "self"
        for name, param in module._parameters.items():
            if param is None:
                continue
            param_address = f"{address}.{name}" if address else name
            existing = pid_to_address.get(id(param))
            if existing is not None:  # a tied parameter: one record per tensor
                param_log = param_logs[existing]
                if param_address not in param_log.all_addresses:
                    param_log.all_addresses.append(param_address)
                if module_address not in param_log.all_module_addresses:
                    param_log.all_module_addresses.append(module_address)
                continue
            pid_to_address[id(param)] = param_address
            param_logs[param_address] = _new_param_log(
                record_type, trace, param, (param_address, module_address, name), optimized
            )
            stamped.append(param)
    return param_logs, pid_to_address, stamped


def _replay(
    record_types: tuple[type[Param], type[ParamAccessor]],
    trace: Trace,
    plan: _plan.ParamPlan,
    optimized: set[int] | None,
) -> None:
    """Stamp and record every unique parameter from a verified plan."""

    record_type, accessor_type = record_types
    forcing = _FROZEN_PARAMETER_GRADIENTS.get()
    param_logs: dict[str, Param] = {}
    stamped: list[nn.Parameter] = []
    for entry in plan.uniques:
        param = entry[_plan.REF]()
        address = entry[_plan.ADDRESS]
        requires_grad_before = param.requires_grad
        if forcing and _forces_frozen_parameter_gradients(trace, param):
            param.requires_grad = True
        barcode = make_random_barcode()
        set_param_meta(
            param, barcode=barcode, address=address, requires_grad_before=requires_grad_before
        )
        param_log = record_type(
            module_address=entry[_plan.MODULE_ADDRESS],
            name=entry[_plan.NAME],
            shape=entry[_plan.SHAPE],
            dtype=entry[_plan.DTYPE],
            num_params=entry[_plan.NUMEL],
            param_memory=entry[_plan.MEMORY],
            trainable=requires_grad_before,
            address=address,
            barcode=barcode,
            has_optimizer=None if optimized is None else entry[_plan.PID] in optimized,
        )
        param_log._param_ref = param
        if entry[_plan.EXTRA_ADDRESSES]:
            param_log.all_addresses.extend(entry[_plan.EXTRA_ADDRESSES])
        if entry[_plan.EXTRA_MODULE_ADDRESSES]:
            param_log.all_module_addresses.extend(entry[_plan.EXTRA_MODULE_ADDRESSES])
        param_logs[address] = param_log
        stamped.append(param)
    trace._param_log_by_pid = dict(plan.pid_to_address)
    trace._session_param_inventory = stamped
    trace._param_logs = accessor_type(param_logs)


def create_session_param_logs(
    record_types: tuple[type[Param], type[ParamAccessor]],
    trace: Trace,
    model: nn.Module,
    optimizer: Any = None,
) -> None:
    """Create ``Param`` objects and stamp each parameter's session metadata.

    ``record_types`` is ``(Param, ParamAccessor)``.
    """

    if not hasattr(trace, "_param_log_by_pid"):
        raise AttributeError("Trace._param_log_by_pid must be initialized before param logging.")
    optimized = _optimized_param_ids(optimizer)
    modules = model_modules(model)
    plan = _plan.verified_param_plan(model, modules)
    if plan is not None:
        _replay(record_types, trace, plan, optimized)
        return
    param_logs, pid_to_address, stamped = _walk_params(record_types[0], trace, modules, optimized)
    _add_shared_module_aliases(model, param_logs, pid_to_address)
    trace._param_log_by_pid = pid_to_address
    trace._session_param_inventory = stamped
    trace._param_logs = record_types[1](param_logs)
    _plan.record_param_plan(model, modules, stamped, param_logs)


__all__ = ["create_session_param_logs"]
