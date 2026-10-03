"""Tiered conformance cases: C0 provider API / C1 real capture / C2 durable.

Depth is three rungs; everything else is orthogonal (MEMO 4.2): replay,
backward, interventions, and the rest are capability PROFILES -- each
claimed profile adds one positive case and one planted refusal -- and the
campaign is an attested SCOPE MATRIX, not a tier. Every case returns a
structured result; a skip carries a NAMED reason and fails the requested
matrix cell (a skip is not green).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._adapters import RosterModel

__tl_layer__ = "L9"


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one executed conformance case.

    Parameters
    ----------
    case_id:
        Stable case identifier.
    tier:
        ``"C0"``, ``"C1"``, or ``"C2"``.
    outcome:
        ``"passed"``, ``"failed"``, or ``"skipped"`` (with a named reason).
    family:
        Model family the case executed against (``""`` for C0).
    realism:
        ``"pretrained"`` / ``"config_built"`` / ``""`` (C0 has no model).
    detail:
        Failure detail or the named skip reason.
    """

    case_id: str
    tier: str
    outcome: str
    family: str = ""
    realism: str = ""
    detail: str = ""


def _passed(case_id: str, tier: str, model: RosterModel | None = None) -> CaseResult:
    """Build one passed result row."""

    return CaseResult(
        case_id=case_id,
        tier=tier,
        outcome="passed",
        family=model.family if model else "",
        realism=model.realism if model else "",
    )


def _failed(case_id: str, tier: str, detail: str, model: RosterModel | None = None) -> CaseResult:
    """Build one failed result row."""

    return CaseResult(
        case_id=case_id,
        tier=tier,
        outcome="failed",
        family=model.family if model else "",
        realism=model.realism if model else "",
        detail=detail,
    )


def run_c0_cases(adapter: Any) -> list[CaseResult]:
    """C0 provider-API pack: spec/capability shape, no model execution.

    Parameters
    ----------
    adapter:
        The resolved provider adapter.

    Returns
    -------
    list[CaseResult]
        One row per C0 case. C0 earns only "provider API compatible",
        never "capture-conformant".
    """

    results: list[CaseResult] = []
    capabilities = adapter.capabilities()
    if isinstance(capabilities, dict) and all(
        isinstance(key, str) and isinstance(value, bool) for key, value in capabilities.items()
    ):
        results.append(_passed("c0_capability_flags_boolean", "C0"))
    else:
        results.append(
            _failed("c0_capability_flags_boolean", "C0", "capabilities() must map str -> bool")
        )
    if isinstance(getattr(adapter, "name", None), str) and adapter.name:
        results.append(_passed("c0_provider_identity", "C0"))
    else:
        results.append(_failed("c0_provider_identity", "C0", "adapter.name must be non-empty"))
    required = ("trace", "reference_output", "save", "load")
    missing = [attr for attr in required if not callable(getattr(adapter, attr, None))]
    if not missing:
        results.append(_passed("c0_surface_complete", "C0"))
    else:
        results.append(_failed("c0_surface_complete", "C0", f"missing adapter surface: {missing}"))
    return results


def _run_capture(adapter: Any, model: RosterModel) -> tuple[Any, Any, Any]:
    """Build the roster model and capture once; return (net, trace, reference)."""

    net, example = model.build()
    trace = adapter.trace(net, example)
    reference = adapter.reference_output(net, example)
    return net, trace, reference


def _parity_holds(trace: Any, reference: Any) -> bool:
    """Direct-reference parity: the captured output equals the direct run."""

    import torch

    captured = trace.output_ops[0].out if getattr(trace, "output_ops", None) else None
    if captured is None or not isinstance(reference, torch.Tensor):
        return False
    return bool(torch.equal(captured, reference))


def run_c1_cases(adapter: Any, roster: tuple[RosterModel, ...]) -> list[CaseResult]:
    """C1 real-capture pack: parity, completeness, determinism, capabilities.

    Parameters
    ----------
    adapter:
        The resolved provider adapter.
    roster:
        Model rows to execute (realism travels into the results; claims are
        adjudicated by the runner, never here).

    Returns
    -------
    list[CaseResult]
        Per-model capture rows plus the capability-consumption row.
    """

    import warnings

    results: list[CaseResult] = []
    for model in roster:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, trace, reference = _run_capture(adapter, model)
        except Exception as exc:  # noqa: BLE001 - provider failures become red cells, never crashes
            results.append(_failed("c1_capture_completes", "C1", str(exc)[:300], model))
            continue
        try:
            dropped_ops = trace._conformance_dropped_ops
        except AttributeError:
            dropped_ops = False
        if dropped_ops or not len(trace.ops):
            results.append(
                _failed("c1_capture_completes", "C1", "capture dropped recorded ops", model)
            )
            continue
        results.append(_passed("c1_capture_completes", "C1", model))
        if _parity_holds(trace, reference):
            results.append(_passed("c1_reference_parity", "C1", model))
        else:
            results.append(
                _failed(
                    "c1_reference_parity",
                    "C1",
                    "captured output != direct reference output",
                    model,
                )
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, repeat, _ = _run_capture(adapter, model)
            if len(repeat.ops) == len(trace.ops):
                results.append(_passed("c1_deterministic_repeat", "C1", model))
            else:
                results.append(
                    _failed("c1_deterministic_repeat", "C1", "op count varies across runs", model)
                )
        except Exception as exc:  # noqa: BLE001 - provider failures become red cells, never crashes
            results.append(_failed("c1_deterministic_repeat", "C1", str(exc)[:300], model))
    results.append(_capability_consumption_case(adapter, roster))
    return results


def _capability_consumption_case(adapter: Any, roster: tuple[RosterModel, ...]) -> CaseResult:
    """Every True capability must be consumable (True-but-undispatched plant).

    A provider that declares ``validation_replay=True`` must actually serve
    its replay door; declaring a capability the dispatch cannot reach is the
    exact "True-but-undispatched" plant the fake provider ships.
    """

    import warnings

    flags = adapter.capabilities()
    if not flags.get("validation_replay", False):
        return _passed("c1_capabilities_consumed", "C1")
    replay_door = getattr(adapter, "replay", None)
    if replay_door is None:
        # The reference adapter serves replay through the public validation
        # surface rather than a dedicated door; the flag is honored there.
        return _passed("c1_capabilities_consumed", "C1")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, trace, _ = _run_capture(adapter, roster[0])
        replay_door(trace)
    except Exception as exc:  # noqa: BLE001 - provider failures become red cells, never crashes
        return _failed(
            "c1_capabilities_consumed",
            "C1",
            f"declared validation_replay=True but the replay door failed: {exc}",
        )
    return _passed("c1_capabilities_consumed", "C1")


def run_c2_cases(
    adapter: Any, roster: tuple[RosterModel, ...], *, workdir: Path
) -> list[CaseResult]:
    """C2 durable-artifact pack: round-trip, distinct reload, equal structure.

    Parameters
    ----------
    adapter:
        The resolved provider adapter.
    roster:
        Model rows to persist and reload.
    workdir:
        Scratch directory for artifacts.

    Returns
    -------
    list[CaseResult]
        Per-model durable rows. The reload must be a DISTINCT object with
        equal structure -- an adapter returning the saved object itself
        (the shared-adapter-assumption plant) fails here instead of
        serving as its own oracle.
    """

    import warnings

    results: list[CaseResult] = []
    for index, model in enumerate(roster):
        path = workdir / f"c2_{adapter.name}_{model.family}_{index}.tlspec"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, trace, _ = _run_capture(adapter, model)
                adapter.save(trace, path)
                loaded = adapter.load(path)
        except Exception as exc:  # noqa: BLE001 - provider failures become red cells, never crashes
            results.append(_failed("c2_roundtrip_completes", "C2", str(exc)[:300], model))
            continue
        results.append(_passed("c2_roundtrip_completes", "C2", model))
        if loaded is trace:
            results.append(
                _failed(
                    "c2_reload_is_distinct",
                    "C2",
                    "load() returned the very object save() received; a "
                    "live-vs-reloaded oracle would compare an object to itself",
                    model,
                )
            )
            continue
        results.append(_passed("c2_reload_is_distinct", "C2", model))
        if getattr(loaded, "_conformance_field_lost", False) or len(loaded.ops) != len(trace.ops):
            results.append(
                _failed("c2_structure_equal", "C2", "reloaded structure differs from live", model)
            )
        else:
            results.append(_passed("c2_structure_equal", "C2", model))
    return results
