"""The axis lists. ``default_axes()`` is what the test module and the subset tool cross.

Add an axis value by appending a row to the list in its module; a new axis
module exports its list and is joined here.
"""

from __future__ import annotations

from ..generator import Axes, DeviceSpec, OptionsSpec
from . import backward, checkpointing, concurrency, control_flow, dtypes_cpu, hooks, inputs

DEVICES: tuple[DeviceSpec, ...] = (DeviceSpec("cpu", "cpu", "the CPU leg every cell runs on"),)

OPTIONS: tuple[OptionsSpec, ...] = (
    OptionsSpec("default", "tl.trace with its defaults (every output saved)"),
    # wave-3 fills
    *backward.OPTIONS,
    *hooks.OPTIONS,
)


def default_axes() -> Axes:
    """Assemble the axis lists in row order."""

    return Axes(
        models=(
            *control_flow.MODELS,
            *inputs.MODELS,
            *dtypes_cpu.MODELS,
            # wave-3 fills
            *backward.MODELS,
            *hooks.MODELS,
            *checkpointing.MODELS,
            *concurrency.MODELS,
        ),
        inputs=inputs.INPUTS,
        dtypes=dtypes_cpu.DTYPES,
        devices=DEVICES,
        options=OPTIONS,
    )
