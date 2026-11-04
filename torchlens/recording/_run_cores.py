"""Retention of sealed run cores across the passes of one Recorder.

A :class:`~torchlens._capture.session.CapturedRunCore` carries, besides its
event spine, the pass's projection facts: the retained records, the output
tensors with their addresses and labels, and a replay copy of the pass's
capture-events journal. ``RecordingProjector.project`` reads the output facts
from the LAST core only and the capture-events copy from the FIRST core that
has one; on every other core those facts are dead weight that grows with the
pass count. A ``generate()`` forward returns its KV cache among its outputs,
so a recorder that kept every pass's output tensors held every earlier decode
step's cache: quadratic in generated tokens, gigabytes for a few megabytes of
saved activations. Each pass therefore trims the facts the previous cores no
longer serve, keeping the projection identical.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._capture.session import CapturedRunCore

#: Projection facts the projector reads from the last core only.
_LAST_CORE_FACTS: tuple[str, ...] = ("output_tensors", "output_tensor_addresses", "output_labels")


def _without_facts(core: CapturedRunCore, names: tuple[str, ...], empty: Any) -> CapturedRunCore:
    """Return ``core`` with each named projection fact replaced by ``empty``."""

    facts = dict(core.projection_facts)
    if not any(facts.get(name) for name in names):
        return core
    facts.update(dict.fromkeys(names, empty))
    return dataclasses.replace(core, projection_facts=MappingProxyType(facts))


def retain_run_core(cores: list[Any], core: CapturedRunCore | None) -> None:
    """Append one pass's sealed core, trimming facts earlier cores no longer serve.

    Parameters
    ----------
    cores:
        The recorder's cores in pass order; mutated in place.
    core:
        The pass's sealed core, or ``None`` when the pass sealed nothing.
    """

    if core is None:
        return
    if cores:
        cores[-1] = _without_facts(cores[-1], _LAST_CORE_FACTS, ())
        if any(earlier.projection_facts.get("capture_events") is not None for earlier in cores[:1]):
            core = _without_facts(core, ("capture_events",), None)
    cores.append(core)
