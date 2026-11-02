"""Fused attention fast-path parity: run the module with the user's grad flags.

Capture forces ``requires_grad=True`` on every floating prepared Parameter so
intermediate tensors carry ``grad_fn`` provenance (``model_prep``). Two torch
modules choose their kernel by that flag: ``nn.MultiheadAttention`` and
``nn.TransformerEncoderLayer`` take their fused inference fast path only when
the module is in eval mode, grad mode is on and NO tensor argument (the inputs
and the module's projection weights and biases) requires grad. Under the forced
flag both fell back to the decomposed path, so the captured output differed
from eager at rounding level and the forward-validation ground-truth check
failed, as it should: capture had changed the computation. For exactly that
module call, when eager's gate would pass, this module puts the user's own
``requires_grad=False`` back on the module's prepared Parameters so capture
takes the path eager takes (one fused op, bit-identical output), and re-forces
the flags on exit. It is the companion of ``param_mutation``'s frozen-receiver
context: the same forced flag, the in-place write case there, the kernel
choice here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
from torch import nn

from ... import _state
from ..._utils.tensor_leaves import iter_tensors
from ._tl import get_param_meta
from .param_mutation import is_prepared_parameter_receiver

#: Modules whose fused inference fast path is gated on ``requires_grad``.
FUSED_FASTPATH_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
)


def fused_fastpath_flag_context(
    trace: Any,
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AbstractContextManager[None]:
    """Run one fused-attention module forward with the flags eager would see.

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    module:
        The module whose ``forward`` is about to run.
    args:
        Positional arguments of the forward call.
    kwargs:
        Keyword arguments of the forward call.

    Returns
    -------
    AbstractContextManager[None]
        A context that restores the user's ``requires_grad=False`` on the
        module's prepared Parameters for the call when eager's fast-path gate
        (eval mode, grad on, no tensor argument requiring grad) would pass
        and only the forced flags hold it shut; otherwise a null context, so
        a trainable module or a grad-requiring input keeps the decomposed
        path on both sides.
    """

    if not isinstance(module, FUSED_FASTPATH_MODULE_TYPES) or module.training:
        return nullcontext()
    if not torch.is_grad_enabled():
        return nullcontext()
    for operand in (*args, *kwargs.values()):
        if any(tensor.requires_grad for tensor in iter_tensors(operand)):
            return nullcontext()
    forced = _forced_frozen_parameters(trace, module)
    if not forced:
        return nullcontext()
    return _user_flags_restored(forced)


def _forced_frozen_parameters(trace: Any, module: nn.Module) -> list[nn.Parameter]:
    """Return the module's prepared Parameters the user froze and prep forced on.

    Empty when any Parameter of the module requires grad by the user's own
    flag: eager's gate then fails too, and both sides take the decomposed path.
    """

    from ._completeness_origins import internal_scalar_read

    forced: list[nn.Parameter] = []
    with internal_scalar_read():
        for parameter in module.parameters():
            meta = get_param_meta(parameter)
            if meta is None or not is_prepared_parameter_receiver(trace, parameter):
                if parameter.requires_grad:
                    return []
                continue
            if meta.requires_grad_before_capture is not False:
                return []
            if parameter.requires_grad and parameter.is_leaf:
                forced.append(parameter)
    return forced


@contextmanager
def _user_flags_restored(parameters: list[nn.Parameter]) -> Iterator[None]:
    """Hold ``requires_grad=False`` on ``parameters`` for the block, then re-force.

    A Parameter that stopped being a leaf inside the block (an in-place write of
    a grad-requiring operand, as eager leaves it) keeps its read-only flag.
    """

    from ._completeness_origins import internal_scalar_read

    with _state.pause_logging(), internal_scalar_read():
        for parameter in parameters:
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        with _state.pause_logging(), internal_scalar_read():
            for parameter in parameters:
                if parameter.is_leaf:
                    parameter.requires_grad_(True)
