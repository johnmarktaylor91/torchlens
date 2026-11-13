"""The overview header's module-reuse note (collapse 5: disclose, don't switch).

An unrolled overview render says how many modules ran their calls as passes and that
``rolled=True`` draws the loop. A module's calls are passes when one layer has ops in two
of its calls: a module that owns state (a Siamese encoder, an RNN cell, a shared block),
or a stateless module called from one loop line. A stateless module called from several
lines (``layer1.0.relu`` in a ResNet block) is called twice but its calls are separate
layers, so rolling folds nothing there and the note does not count it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .request import ResolvedRenderRequest


def _calls_are_passes(module: Any) -> bool:
    """Whether one layer has ops in two different calls of ``module``."""

    seen: dict[str, int] = {}
    for index in module.calls:
        call = module.calls.get(index)
        for label in call.ops if call is not None else ():
            layer = str(label).rsplit(":", 1)[0]
            first = seen.setdefault(layer, index)
            if first != index:
                return True
    return False


def module_reuse_note(trace: Trace, request: ResolvedRenderRequest) -> str | None:
    """Disclose module reuse on an unrolled overview, never switching modes (collapse 5).

    The collapse panel ratified "disclose, don't switch" for weight-shared
    models: the default ``vis_mode`` stays, and an overview render (a smart
    collapse mode) says how many modules ran their calls as passes and that
    ``rolled=True`` draws the loop, short enough not to clip. Nested modules
    of a counted module are not counted again.

    Returns
    -------
    str | None
        The note, or ``None`` when the render is rolled, not an overview, or
        no module's calls are passes.
    """

    overview = request.collapse in ("auto", "max") or (
        isinstance(request.collapse, float) and request.collapse > 0.0
    )
    if request.vis_mode != "unrolled" or not overview:
        return None
    reused = {
        str(module.address): int(module.num_calls)
        for module in trace.modules
        if str(getattr(module, "address", "")) not in ("", "self")
        and int(getattr(module, "num_calls", 1) or 1) > 1
        and _calls_are_passes(module)
    }
    outermost = {
        address: calls
        for address, calls in reused.items()
        if not any(address.startswith(f"{other}.") for other in reused)
    }
    if not outermost:
        return None
    count = len(outermost)
    calls = sum(outermost.values())
    noun = "module" if count == 1 else "modules"
    return f"{count:,} reused {noun} ({calls:,} calls); rolled=True draws the loop"
