"""The ONE persistent input-provenance record (quickstart memo D5).

``InputProvenance`` is the rich runtime view; it serializes INTO the shipped
``ResolvedPreprocessing`` field shape (source / identifier / verified /
config / description) and rides the existing KEEP-registered
``Trace.input_preprocessor`` slot on the ``tl.save``/``tl.load`` bundle path,
so a synthesized trace can never masquerade as a real one after a handoff --
with ZERO new persisted schema fields (the C07 census is closed; this lane
mints none).

Loading re-hydrates the rich view from the record's ``config`` payload via
:func:`provenance_from_record`. Every spelling is DOCUMENTED-UNSTABLE pending
the naming sprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..data_classes.trace import ResolvedPreprocessing
from ._plan import ORIGIN_BY_RUNG, RUNG_GOLD, InputPlan, tensor_leaves

__tl_layer__ = "L2"

#: Version stamp for the config payload (bump = additive, load stays lenient).
_PROVENANCE_VERSION = 1

#: The binding trust sentence for inference (memo D6, verbatim in docs).
EXACT_PATH_CAVEAT = (
    "Successful inference proves that this exact synthesized call ran under "
    "the stated execution policy and its capture verified. It does not prove "
    "canonical preprocessing, representative values, a stable graph for "
    "other data, or another input's control-flow path."
)

#: Caveat attached to every synthesized-value record (rungs 2-3): same-shape
#: different-value inputs can change the GRAPH, not just the numbers
#: (measured: fasterrcnn 1091 vs 1083 ops), so the record never says "only
#: the values are fake".
SYNTHETIC_GRAPH_CAVEAT = (
    "Values are synthesized: value-dependent control flow may take a "
    "different path (and produce a different graph) than your real data."
)


@dataclass(frozen=True)
class TensorFacts:
    """Shape/dtype/device disclosure for one direct tensor argument."""

    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class InputProvenance:
    """Persistent provenance for how a capture's input came to exist.

    Attached for EVERY rung, including gold (memo D5): a record that only
    exists on suspicious traces would itself be a signal that can be lost in
    a handoff.

    Attributes
    ----------
    origin:
        ``"user"`` (rung 1), ``"declared"`` (rung 2), or ``"inferred"``
        (rung 3).
    rung:
        The ladder rung, 1-3 (rung 4 / meta is reserved behind D8).
    values_semantic:
        True only when the values are caller-supplied real data. Gates
        derived-semantics claims (memo D7).
    call_structure:
        Normalized call description: positional count, keyword names, and
        whether non-tensor arguments are present.
    tensors:
        Per-tensor shape/dtype/device facts (direct arguments; nested
        containers are the capture record's job).
    recipes:
        Synthesis recipe rows (rungs 2-3): recipe kind, low, high, dtype,
        seed. ``None`` on the gold rung.
    strategy:
        Inference strategy that produced the winning input (rung 3).
    attempts:
        Bounded probe diary (rung 3): ``(shape-or-None, outcome)`` rows.
    constraining_op / flexible_dims:
        Inference evidence (rung 3), when known.
    evidence_grade:
        ``verified_exact`` / ``empirically_flexible`` / ``under_determined``
        / ``unsupported`` -- exact-verification evidence held separately
        from authoritative-canonical evidence (memo D5/D10).
    execution_policy:
        ``"caller_mode"`` (tl.trace keeps the caller's flags) or
        ``"eval_no_grad_restored"`` (summary/render pin and restore).
    caveats:
        Rendered disclosure sentences (the D6 trust sentence and the
        synthetic-graph caveat ride here on non-gold rungs).
    lazy_transitions:
        Module addresses whose lazy parameters materialized during the ONE
        captured forward (disclosure of the sanctioned permanent mutation).
    preprocessing:
        Nested preprocessing provenance (e.g. the HF tokenizer record when a
        prompt string was the gold input).
    d8_meta / cache_status / cache_key_version:
        Reserved slots (memo section 7). Unused in wave 1.
    """

    origin: str
    rung: int
    values_semantic: bool
    call_structure: dict[str, Any]
    tensors: tuple[TensorFacts, ...]
    recipes: tuple[dict[str, Any], ...] | None = None
    strategy: str | None = None
    attempts: tuple[tuple[Any, str], ...] | None = None
    constraining_op: str | None = None
    flexible_dims: tuple[int, ...] | None = None
    evidence_grade: str | None = None
    execution_policy: str = "caller_mode"
    caveats: tuple[str, ...] = ()
    lazy_transitions: tuple[str, ...] = ()
    preprocessing: dict[str, Any] | None = None
    d8_meta: dict[str, Any] | None = None
    cache_status: str | None = None
    cache_key_version: str | None = None

    def describe(self) -> str:
        """Return the one-line rendered disclosure for human surfaces."""

        if self.rung == RUNG_GOLD:
            return "input: user-provided (real values)"
        shapes = ", ".join("x".join(str(d) for d in facts.shape) for facts in self.tensors)
        if self.origin == "declared":
            return f"input: synthesized from input_size= ({shapes}); values are random, disclosed"
        return (
            f"input: shape inferred ({shapes}), values synthesized; "
            f"strategy={self.strategy or 'unknown'}; disclosed"
        )


def _tensor_facts(plan: InputPlan) -> tuple[TensorFacts, ...]:
    """Derive per-tensor disclosure rows from a concrete plan."""

    return tuple(
        TensorFacts(shape=tuple(leaf.shape), dtype=str(leaf.dtype), device=str(leaf.device))
        for leaf in tensor_leaves(plan)
    )


def _call_structure(plan: InputPlan) -> dict[str, Any]:
    """Derive the normalized call-structure disclosure from a plan."""

    non_tensor = any(
        not isinstance(value, torch.Tensor)
        for value in (*plan.input_args, *plan.input_kwargs.values())
    )
    return {
        "n_positional": len(plan.input_args),
        "keyword_names": sorted(plan.input_kwargs),
        "has_non_tensor_args": non_tensor,
    }


def provenance_for_plan(  # noqa: PLR0913 -- each kwarg is one NAMED evidence field on the persisted InputProvenance record; packing them would hide the disclosure schema
    plan: InputPlan,
    *,
    recipes: tuple[dict[str, Any], ...] | None = None,
    strategy: str | None = None,
    attempts: tuple[tuple[Any, str], ...] | None = None,
    constraining_op: str | None = None,
    flexible_dims: tuple[int, ...] | None = None,
    evidence_grade: str | None = None,
    execution_policy: str = "caller_mode",
    lazy_transitions: tuple[str, ...] = (),
    preprocessing: dict[str, Any] | None = None,
) -> InputProvenance:
    """Build the provenance record for a resolved plan.

    Non-gold rungs automatically carry the D6 trust sentence and the
    synthetic-graph caveat; the gold rung carries none.
    """

    gold = plan.rung == RUNG_GOLD
    caveats: tuple[str, ...] = () if gold else (SYNTHETIC_GRAPH_CAVEAT, EXACT_PATH_CAVEAT)
    return InputProvenance(
        origin=ORIGIN_BY_RUNG[plan.rung],
        rung=plan.rung,
        values_semantic=gold,
        call_structure=_call_structure(plan),
        tensors=_tensor_facts(plan),
        recipes=recipes,
        strategy=strategy,
        attempts=attempts,
        constraining_op=constraining_op,
        flexible_dims=flexible_dims,
        evidence_grade=evidence_grade,
        execution_policy=execution_policy,
        caveats=caveats,
        lazy_transitions=lazy_transitions,
        preprocessing=preprocessing,
    )


def _json_safe(value: Any) -> Any:
    """Coerce provenance payload values to JSON/pickle-portable primitives."""

    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def provenance_to_record(provenance: InputProvenance) -> ResolvedPreprocessing:
    """Serialize the rich view into the shipped ``ResolvedPreprocessing`` shape.

    The full payload rides ``config`` under the versioned
    ``input_provenance`` key; ``verified`` keeps its shipped meaning
    restricted to this record's claim: True only for caller-authoritative
    real input.
    """

    config: dict[str, Any] = {
        "input_provenance_version": _PROVENANCE_VERSION,
        "input_provenance": _json_safe(asdict(provenance)),
    }
    return ResolvedPreprocessing(
        source="torchlens.quickstart.input_resolver",
        identifier=f"rung{provenance.rung}:{provenance.origin}",
        verified=provenance.values_semantic,
        config=config,
        description=provenance.describe(),
    )


def provenance_from_record(record: Any) -> InputProvenance | None:
    """Re-hydrate the rich view from a (possibly loaded) preprocessing record.

    Returns ``None`` for records that do not carry the versioned payload
    (legacy tokenizer/transform records, foreign sources) -- absent
    provenance on a legacy trace means caller-authoritative history, never
    an error.
    """

    config = getattr(record, "config", None)
    if not isinstance(config, dict):
        return None
    payload = config.get("input_provenance")
    if not isinstance(payload, dict):
        return None
    known = set(InputProvenance.__dataclass_fields__)
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            continue
        kwargs[key] = _rehydrate_field(key, value)
    try:
        return InputProvenance(**kwargs)
    except TypeError:
        return None


def _rehydrate_field(key: str, value: Any) -> Any:
    """Restore tuple-typed provenance fields from their JSON list forms."""

    if key == "tensors" and isinstance(value, list):
        return tuple(
            TensorFacts(
                shape=tuple(row.get("shape", ())),
                dtype=str(row.get("dtype", "")),
                device=str(row.get("device", "")),
            )
            for row in value
            if isinstance(row, dict)
        )
    if key == "attempts" and isinstance(value, list):
        return tuple(
            (tuple(row[0]) if isinstance(row[0], list) else row[0], str(row[1]))
            for row in value
            if isinstance(row, (list, tuple)) and len(row) == 2
        )
    if key in {"recipes"} and isinstance(value, list):
        return tuple(entry for entry in value if isinstance(entry, dict))
    if key in {"flexible_dims", "caveats", "lazy_transitions"} and isinstance(value, list):
        return tuple(value)
    return value


def trace_input_provenance(trace: Any) -> InputProvenance | None:
    """Read a trace's input provenance (the public reader).

    Parameters
    ----------
    trace:
        A live or loaded ``Trace``.

    Returns
    -------
    InputProvenance | None
        The rich provenance view, or ``None`` on legacy traces captured
        before the resolver existed (absent record = caller-authoritative
        gold history).
    """

    return provenance_from_record(getattr(trace, "input_preprocessor", None))
