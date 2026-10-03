"""The turnkey HF ``generate()`` steer wrapper -- BIND-SIDE ONLY (F01).

FoldA D12 amendment note 4: the wrapper returns model outputs plus the
binding's ``.last_report``, NEVER a Trace. Anti-harness rule (foldB s7 F01
delta, ratified verbatim in the lane report): "TorchLens may drive computation
only where every choice the driver makes is itself recorded and checkable in
the artifact." The wrapper formats, runs, snapshots, relates; every injected
choice is a recorded InterventionSpec firing; the wrapper never grows
retry/stop/tool logic.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binding import BindReport
from .spec import InterventionSpec


@dataclass(frozen=True)
class SteerResult:
    """One steered generation: the model's own outputs plus the bind ledger.

    Parameters
    ----------
    outputs:
        Exactly what the base model's ``generate`` returned (never wrapped,
        never a Trace).
    report:
        The binding's out-of-band :class:`BindReport` for this generation.
    """

    outputs: Any
    report: BindReport


def steer_generate(
    model: Any,
    inputs: Any,
    spec: InterventionSpec,
    *,
    on_zero_fire: str = "error",
    **generate_kwargs: Any,
) -> SteerResult:
    """Run one real HF ``generate`` with the spec's edits held live.

    Composition sugar over ``spec.bind(model).generate(...)``: one binding is
    constructed, one generation runs capture-free with every firing recorded,
    and the result pairs the model's own outputs with the ledger. All bind
    refusals (preflight anchors, zero-fire fail-closed, missing ``generate``,
    serial re-entrancy) apply unchanged.

    Parameters
    ----------
    model:
        The base ``nn.Module`` with a real ``generate`` method.
    inputs:
        First positional argument forwarded to ``generate`` (input ids).
    spec:
        The immutable intervention spec to hold across the generation.
    on_zero_fire:
        Zero-fire settlement policy forwarded to ``spec.bind`` (FOLD-A3
        fail-closed default).
    **generate_kwargs:
        Forwarded verbatim to the model's ``generate``.

    Returns
    -------
    SteerResult
        ``outputs`` (the model's own return value) and ``report`` (the
        binding's ledger). Never a Trace.
    """

    binding = spec.bind(model, on_zero_fire=on_zero_fire)
    outputs = binding.generate(inputs, **generate_kwargs)
    report = binding.last_report
    if report is None:
        # unreachable in practice: generate() always settles a report
        raise RuntimeError("the binding returned without settling .last_report")
    return SteerResult(outputs=outputs, report=report)


__all__ = ["SteerResult", "steer_generate"]
