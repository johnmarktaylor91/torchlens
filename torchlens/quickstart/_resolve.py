"""The ONE input resolver behind summary / render / trace (quickstart D1-D4).

Layer 3 of the four-layer split: applies the rung law, synthesizes declared
inputs, runs the inference search for the zero-argument rung (consuming its
exact verified trace, never recapturing -- memo D9 in-call reuse), and builds
the provenance record every rung carries.

A failed explicit size never falls through to inference; a failed inference
never falls through to a stock tensor (memo D2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .._errors import ShapeInferenceError
from ._grammar import SYNTHESIS_SEED, parse_input_size, synthesize
from ._plan import (
    RUNG_DECLARED,
    RUNG_GOLD,
    RUNG_INFERRED,
    InputPlan,
    normalize_gold_args,
    resolve_rung,
)
from ._provenance import InputProvenance, provenance_for_plan

__tl_layer__ = "L3"


@dataclass(frozen=True)
class ResolvedInputs:
    """A settled resolution: the concrete plan plus its provenance record."""

    plan: InputPlan
    provenance: InputProvenance


def _is_hf_model(model: nn.Module) -> bool:
    """Best-effort check for a Hugging Face ``PreTrainedModel`` (no import)."""

    for klass in type(model).__mro__:
        if klass.__module__.startswith("transformers.") and klass.__name__ == "PreTrainedModel":
            return True
    return False


def _inference_remedies(model: nn.Module) -> str:
    """Return the D10 ordered-remedy text, population-aware for HF models."""

    remedies = (
        "(1) pass the real forward call -- tl.<verb>(model, x) or "
        "tl.<verb>(model, input_kwargs={...}); "
        "(2) pass input_size= with your shape -- a flat tuple, a sequence of "
        "tuples, or a mapping of forward keyword names to shapes, with "
        "torchlens.quickstart.InputSpec for dtype/value-recipe hints; "
        "(3) run torchlens.debug.infer_input_shape(model, ...) directly with "
        "hints and read its attempt diary"
    )
    if _is_hf_model(model):
        remedies = (
            "for this Hugging Face model a real prompt string is a BETTER "
            "answer than input_size= -- tl.<verb>(model, 'The quick brown fox') "
            "tokenizes with the model's own tokenizer; otherwise " + remedies
        )
    return remedies


def _grade_for_inference(result: Any) -> str:
    """Map an inference result onto the D5 evidence-grade vocabulary.

    ``verified_exact``: the winning input ran and verified, and no dimension
    claims flexibility beyond the batch axis. ``empirically_flexible``: the
    verified input carries flexible dims (other sizes are expected to run,
    none of them canonical). The ``under_determined`` grade is minted by the
    confirmation-probe path in the inference module itself (B9).
    """

    grade = getattr(result, "evidence_grade", None)
    if isinstance(grade, str) and grade:
        return grade
    flexible = tuple(getattr(result, "flexible_dims", ()) or ())
    nonbatch_flexible = tuple(dim for dim in flexible if dim != 0)
    return "empirically_flexible" if nonbatch_flexible else "verified_exact"


def _resolve_inferred(model: nn.Module, verb: str) -> ResolvedInputs:
    """Run the zero-argument rung: inference search + verified-trace reuse."""

    from ..debug._infer_input_shape import infer_input_shape

    result = infer_input_shape(model, return_trace=True)
    if not result.found or result.example_input is None:
        raise ShapeInferenceError(
            f"tl.{verb}(model) could not infer a runnable input: "
            f"{result.reason or 'no candidate verified'}. "
            f"{result.message} The bounded probe diary rides "
            f"exc.fields['attempts'].",
            code="input_inference_failed",
            remedy=_inference_remedies(model),
            reason=result.reason,
            attempts=list(result.attempts),
            strategy=result.strategy,
        )
    example = result.example_input
    if isinstance(example, dict):
        plan = InputPlan(
            rung=RUNG_INFERRED,
            input_args=(),
            input_kwargs=dict(example),
            verified_trace=result.trace,
        )
    elif isinstance(example, (list, tuple)):
        plan = InputPlan(rung=RUNG_INFERRED, input_args=tuple(example), verified_trace=result.trace)
    else:
        plan = InputPlan(rung=RUNG_INFERRED, input_args=(example,), verified_trace=result.trace)
    recipe_row: tuple[dict[str, Any], ...] | None = None
    if result.value_range is not None:
        kind, low, high = result.value_range
        recipe_row = (
            {
                "recipe": str(kind),
                "low": float(low),
                "high": float(high),
                "dtype": str(result.dtype) if result.dtype is not None else None,
                "seed": None,
            },
        )
    provenance = provenance_for_plan(
        plan,
        recipes=recipe_row,
        strategy=result.strategy,
        attempts=tuple(result.attempts),
        constraining_op=result.constraining_op,
        flexible_dims=tuple(result.flexible_dims or ()),
        evidence_grade=_grade_for_inference(result),
        execution_policy="eval_no_grad_restored",
    )
    return ResolvedInputs(plan=plan, provenance=provenance)


def _resolve_declared(model: nn.Module, input_size: Any) -> ResolvedInputs:
    """Run the declared rung: grammar, static facts, deterministic synthesis."""

    parsed = parse_input_size(input_size, model)
    positional, keyword = synthesize(parsed, model, seed=SYNTHESIS_SEED)
    plan = InputPlan(rung=RUNG_DECLARED, input_args=positional, input_kwargs=keyword)
    recipes = tuple(
        {
            "recipe": slot.recipe,
            "low": slot.low,
            "high": slot.high,
            "dtype": str(slot.dtype),
            "seed": SYNTHESIS_SEED,
            "keyword": slot.keyword,
        }
        for slot in (*parsed.positional, *parsed.keyword)
    )
    provenance = provenance_for_plan(plan, recipes=recipes)
    return ResolvedInputs(plan=plan, provenance=provenance)


def resolve_inputs(
    model: nn.Module,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
    input_size: Any = None,
    *,
    verb: str = "trace",
) -> ResolvedInputs:
    """Resolve the three-rung ladder to a concrete plan plus provenance.

    Parameters
    ----------
    model:
        The model about to be captured.
    input_args:
        The caller's real positional input, when supplied (rung 1).
    input_kwargs:
        The caller's real keyword input, when supplied (rung 1).
    input_size:
        The declared-shape argument (rung 2).
    verb:
        The public verb name, used verbatim in teaching refusals.

    Returns
    -------
    ResolvedInputs
        Concrete plan (with the inference search's verified trace attached
        on rung 3) and the provenance record for the trace.

    Raises
    ------
    torchlens._errors.ArgumentConflictError
        ``input_rung_conflict`` on a mixed-rung call.
    torchlens._errors.InvalidArgumentError
        Grammar and dtype-fact refusals from the declared rung.
    torchlens._errors.ShapeInferenceError
        ``input_inference_failed`` with the bounded diary and the ordered
        remedies when the zero-argument rung cannot verify an input.
    """

    rung = resolve_rung(input_args, input_kwargs, input_size)
    if rung == RUNG_GOLD:
        plan = normalize_gold_args(input_args, input_kwargs)
        return ResolvedInputs(plan=plan, provenance=provenance_for_plan(plan))
    if rung == RUNG_DECLARED:
        return _resolve_declared(model, input_size)
    return _resolve_inferred(model, verb)


def attach_provenance(trace: Any, provenance: InputProvenance) -> None:
    """Attach a SYNTHESIZED-rung provenance record to a finished trace.

    Gold-rung records deliberately never persist here: the shipped
    ``input_preprocessor`` field means "automatic preprocessing was applied"
    (summary sections, model_profile modality inference, and the autoroute
    contract tests all read it that way), and an absent quickstart payload
    already reads as gold everywhere in the capability gate -- so a gold
    record would disturb a shipped contract to encode information absence
    already encodes. Gold provenance still travels in-memory on quickstart
    results (RenderResult.provenance).

    Nests any pre-existing preprocessing record (e.g. the HF tokenizer
    provenance written by the string-input bridge) inside the input record
    rather than clobbering it (memo D4: a supported HF string is a gold
    input with tokenizer provenance nested in the record).
    """

    from dataclasses import asdict, replace

    from ._provenance import provenance_to_record

    if provenance.values_semantic:
        return
    existing = getattr(trace, "input_preprocessor", None)
    if existing is not None and getattr(existing, "source", "") != (
        "torchlens.quickstart.input_resolver"
    ):
        try:
            nested = asdict(existing)
        except TypeError:
            nested = {"description": repr(existing)}
        provenance = replace(provenance, preprocessing=nested)
    trace.input_preprocessor = provenance_to_record(provenance)
