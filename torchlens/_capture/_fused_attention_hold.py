"""Hold torch's fused attention fast path off for one captured forward.

``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer`` pick a fused native
kernel from ``requires_grad`` and grad mode; capture holds the public switch off
so the forward, its replays and the validation ground truth record the same
unfused attention program. Moved out of ``_capture_state_helpers`` (file-size
ceiling); ``prepare_compiled_capture`` records the held addresses on
``CompiledCapturePrep.fused_attention_held``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn

from .._utils._torch_compat import (
    get_mha_fastpath_enabled,
    get_mha_fastpath_switch_support,
    set_mha_fastpath_enabled,
)

_FUSED_ATTENTION_FASTPATH_MODULES: tuple[type[nn.Module], ...] = (
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
)
"""Module types with a fused inference fast path gated on ``requires_grad`` under grad mode."""


def fused_attention_fastpath_module_addresses(model: Any) -> tuple[str, ...]:
    """Return the dotted addresses of ``model``'s fused-fast-path attention modules.

    Parameters
    ----------
    model:
        Any object; a non-``nn.Module`` has no such modules.

    Returns
    -------
    tuple[str, ...]
        Addresses (``"self"`` for the root) of every ``nn.MultiheadAttention`` and
        ``nn.TransformerEncoderLayer`` instance reachable from ``model``.
    """

    if not isinstance(model, nn.Module):
        return ()
    return tuple(
        name or "self"
        for name, module in model.named_modules()
        if isinstance(module, _FUSED_ATTENTION_FASTPATH_MODULES)
    )


@contextmanager
def hold_fused_attention_fastpath_off(model: Any) -> Iterator[tuple[str, ...]]:
    """Hold torch's fused attention fast path off for one forward of ``model``.

    ``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer`` run a fused native
    kernel (``_native_multi_head_attention`` / ``_transformer_encoder_layer_fwd``)
    in eval mode only while, among other gates, no input or parameter requires a
    gradient under grad mode. The fused kernel is one opaque op: the attention
    internals (projections, ``scaled_dot_product_attention``) would not appear in
    the trace, and which program ran would depend on the ambient grad mode and on
    the input's ``requires_grad``. Holding the public switch (``torch.backends.mha``)
    off makes the capture forward, its replays and the validation ground truth run
    the SAME unfused program in every case, so the recorded ops are always the
    attention internals. The capture records the hold as a
    ``fused_attention_fastpath_held_off`` capture advisory.

    The ``training``-flag fallback of ``force_mha_slow_path`` is deliberately NOT
    used: in eval mode it would enable the modules' dropout. Where torch lacks the
    switch (``HAS_MHA_FASTPATH_SWITCH`` False) nothing is held and the forward
    runs as torch decides.

    Parameters
    ----------
    model:
        Model about to run; a non-module or a model without such modules holds
        nothing.

    Yields
    ------
    tuple[str, ...]
        Addresses of the modules whose fast path is held off; empty when nothing
        is held.
    """

    addresses = fused_attention_fastpath_module_addresses(model)
    if not addresses or not get_mha_fastpath_switch_support():
        yield ()
        return
    prior = get_mha_fastpath_enabled()
    if prior is None:
        yield ()
        return
    set_mha_fastpath_enabled(False)
    try:
        yield addresses
    finally:
        set_mha_fastpath_enabled(prior)
