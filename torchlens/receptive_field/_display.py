"""Bounded choices for receptive-field endpoint refusals."""

from __future__ import annotations

from typing import Any

from .._utils._display_errors import lookup_message, module_output_message
from .._utils._display_keys import display_key


def _pass_message(layer: Any) -> str:
    """Name executed passes by their stored display keys."""

    return lookup_message(
        f"Layer {display_key(layer)!r} has {layer.num_passes} passes.",
        [display_key(op) for op in layer.ops.values()],
        "select a pass with layer.ops[pass_number]",
    )


def _output_message(call: Any, trace: Any) -> str:
    """Offer a bounded set of the call's existing output keys."""

    return module_output_message(call, trace)


def _call_message(module: Any) -> str:
    """Keep module-call addresses accessible without dumping all calls."""

    return lookup_message(
        f"Module {module.address!r} has {module.num_calls} calls.",
        [call.label for call in module.calls],
        "select one call with module.calls[index]",
    )
