"""Intervened dataset extraction (R2-08; spellings DOCUMENTED-UNSTABLE).

``extract_dataset(..., intervene=spec)`` runs every batch through the live
capture door with the spec armed (``tl.trace(..., intervene=spec)``), so the
stored rows are the INTERVENED activations. The run is a counterfactual
harvest under the ``tl_counterfactual_harvest_v1`` schema
(:mod:`.harvest_schema`): its identity (engine, address class, rule
identities) joins the resume signature, batch zero FREEZES the structural
site keys the spec fired at, every later batch must fire at exactly those
sites, and each committed shard ledgers its own fire receipt. A batch with
no fire refuses typed: a counterfactual record without fire evidence is not
a record.

Only address classes whose execution gate is certified run
(:data:`.harvest_schema.EXECUTABLE_ADDRESS_CLASSES`); finer classes refuse
typed and are never coarsened. Every helper must declare
``batch_independent=True`` (the append gate's rule): a helper that couples
batch rows would make stimulus ``i``'s row depend on its batch-mates, which
breaks the row-is-stimulus contract the artifact states.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .._errors import InvalidArgumentError
from .harvest_schema import (
    ADDRESS_CLASSES,
    HarvestSpec,
    harvest_block,
    require_executable,
)

__tl_layer__ = "L5"

__all__ = [
    "INTERVENTION_ENGINE",
    "InterventionPlan",
    "attest_batch_receipt",
    "batch_receipt",
    "coerce_intervention",
    "final_harvest_block",
    "refuse_intervened_selector_layers",
]

#: The one engine intervened extraction runs: the live capture door's hooks.
INTERVENTION_ENGINE = "live_hook"

#: Selector kinds mapped to the harvest address class they address, finest
#: first (the worst member of a composite selector wins).
_STRUCTURAL_KIND_CLASSES: tuple[tuple[str, str], ...] = (
    ("site", "op"),
    ("output", "op"),
    ("func", "type"),
    ("func_transform", "type"),
    ("module", "module"),
    ("in_module", "module"),
)


@dataclasses.dataclass(frozen=True)
class InterventionPlan:
    """The resolved intervention of one extraction run.

    Attributes
    ----------
    spec:
        The caller's :class:`~torchlens.intervention.spec.InterventionSpec`.
    address_class:
        The harvest address class (the finest any rule addresses).
    spec_digest:
        Content digest over the ordered rule identities at resolution time.
    rules:
        Per-rule identity rows (rule id, WHERE/action renderings, direction).
    """

    spec: Any
    address_class: str
    spec_digest: str
    rules: tuple[dict[str, str], ...]

    def signature_record(self) -> dict[str, Any]:
        """Return the resume-compared identity of this intervention.

        Returns
        -------
        dict[str, Any]
            JSON-portable record: policy, engine, address class, spec
            digest, and the rule rows.
        """

        return {
            "policy": "intervened",
            "engine": INTERVENTION_ENGINE,
            "address_class": self.address_class,
            "spec_digest": self.spec_digest,
            "rules": [dict(rule) for rule in self.rules],
        }


def _rule_address_class(rule: Any) -> str:
    """Map one rule's WHERE term to its harvest address class.

    Parameters
    ----------
    rule:
        One :class:`~torchlens.intervention.spec.InterventionRule`.

    Returns
    -------
    str
        A member of :data:`.harvest_schema.ADDRESS_CLASSES`.
    """

    if rule.address_class == "value_dependent":
        return "selection"
    if rule.address_class == "recorded":
        return "label"
    from ..ir.selector_eval import selector_contains_kind

    for kind, address_class in _STRUCTURAL_KIND_CLASSES:
        if selector_contains_kind(rule.where, kind, unwrap=True):
            return address_class
    return "selection"


def _rule_helper(rule: Any) -> Any:
    """Return the rule's declared helper, or ``None`` for an opaque action.

    Parameters
    ----------
    rule:
        One intervention rule.

    Returns
    -------
    Edit | None
        The helper spec carrying the batch-independence declaration.
    """

    from ..intervention.types import Edit, InterventionDecision

    action = rule.action
    if isinstance(action, InterventionDecision):
        action = action.hook
    return action if isinstance(action, Edit) else None


def _refuse_batch_dependent(rule: Any) -> None:
    """Refuse a rule whose helper does not declare batch independence.

    Parameters
    ----------
    rule:
        One intervention rule.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_intervention_batch_dependent``.
    """

    helper = _rule_helper(rule)
    if helper is not None and bool(helper.batch_independent):
        return
    name = helper.helper_name if helper is not None else rule.action_repr
    raise InvalidArgumentError(
        f"Intervention action {name!r} does not declare batch_independent=True. "
        "Extraction stores one row per stimulus, so an edit whose value reads "
        "the other rows of its batch (or an opaque callable that cannot say) "
        "would make each stored row depend on which stimuli shared its batch.",
        code="extraction_intervention_batch_dependent",
        remedy=(
            "use a batch-independent helper (for example tl.zero_ablate(), or "
            "tl.mean_ablate(source) with a precomputed source tensor)"
        ),
        action=str(name),
    )


def coerce_intervention(intervene: Any) -> InterventionPlan | None:
    """Validate the ``intervene=`` slot and resolve its harvest identity.

    Parameters
    ----------
    intervene:
        ``None`` (clean extraction) or an
        :class:`~torchlens.intervention.spec.InterventionSpec` (the
        ``tl.when(...)`` product).

    Returns
    -------
    InterventionPlan | None
        The resolved plan, or ``None`` for a clean run.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_intervene_invalid`` for anything but a non-empty spec;
        ``harvest_address_class_ungated`` for an uncertified address class;
        ``extraction_intervention_batch_dependent`` for a coupling helper.
    """

    if intervene is None:
        return None
    from ..intervention.spec import InterventionSpec

    if not isinstance(intervene, InterventionSpec) or not intervene.rules:
        raise InvalidArgumentError(
            f"extract_dataset(intervene=) takes a non-empty InterventionSpec; "
            f"got {type(intervene).__name__}. The spec carries the rule "
            "identities the artifact records and resume compares.",
            code="extraction_intervene_invalid",
            remedy="build it with tl.when(selector, helper), e.g. "
            "tl.when(tl.module('fc1'), tl.zero_ablate())",
            value_type=type(intervene).__name__,
        )
    classes = [_rule_address_class(rule) for rule in intervene.rules]
    address_class = max(classes, key=ADDRESS_CLASSES.index)
    probe = HarvestSpec(
        policy="intervened",
        engine=INTERVENTION_ENGINE,
        address_class=address_class,
        target_site_keys=("<unresolved>",),
        receipt_digest="<unresolved>",
    )
    require_executable(probe)
    for rule in intervene.rules:
        _refuse_batch_dependent(rule)
    rules = tuple(
        {
            "rule_id": str(rule.rule_id),
            "where": str(rule.where_repr),
            "action": str(rule.action_repr),
            "direction": str(rule.direction),
        }
        for rule in intervene.rules
    )
    return InterventionPlan(
        spec=intervene,
        address_class=address_class,
        spec_digest=str(intervene.spec_digest),
        rules=rules,
    )


def refuse_intervened_selector_layers() -> None:
    """Refuse ``intervene=`` combined with selector-valued ``layers=``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_intervened_selector_layers_unsupported``.
    """

    raise InvalidArgumentError(
        "intervene= runs with string or mapping layers= only: the selector "
        "plan freezes against a clean batch-zero capture, so its site "
        "attestation cannot yet see the intervened graph.",
        code="extraction_intervened_selector_layers_unsupported",
        remedy="name the sites with a list or {key: layer} mapping of layer lookups",
    )


def batch_receipt(trace: Any) -> dict[str, Any]:
    """Read one intervened capture's fire receipt off its ops.

    Parameters
    ----------
    trace:
        The batch's completed capture.

    Returns
    -------
    dict[str, Any]
        ``fire_count`` (fire records minted), ``site_keys`` (sorted unique
        structural keys of the fired ops), and ``event_digest`` (the capture
        door's audit digest for this transaction, when recorded).
    """

    from ..intervention.audit import site_keys_for_labels

    fired: list[str] = []
    fire_count = 0
    for label in trace.op_labels:
        records = getattr(trace.ops[label], "interventions", None) or ()
        if records:
            fired.append(label)
            fire_count += len(records)
    audit = [
        row
        for row in getattr(trace, "intervention_audit", ())
        if isinstance(row, Mapping) and row.get("kind") == "ACT"
    ]
    return {
        "fire_count": fire_count,
        "site_keys": sorted(set(site_keys_for_labels(trace, tuple(fired)))),
        "event_digest": audit[-1].get("resolve_digest") if audit else None,
    }


def attest_batch_receipt(
    receipt: dict[str, Any], run_state: dict[str, Any], batch_index: int
) -> None:
    """Freeze batch zero's fired sites; refuse no-fire and drifted batches.

    Parameters
    ----------
    receipt:
        This batch's :func:`batch_receipt`.
    run_state:
        Mutable per-run state (holds the frozen ``intervention_sites``).
    batch_index:
        Zero-based batch index.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_intervention_no_fire`` when the spec fired nowhere;
        ``extraction_intervention_site_drift`` when it fired at other sites
        than batch zero did.
    """

    if receipt["fire_count"] == 0:
        raise InvalidArgumentError(
            f"The intervention fired at no site on batch {batch_index}; an "
            "intervened artifact whose rows were never edited would state a "
            "counterfactual that never happened.",
            code="extraction_intervention_no_fire",
            remedy="check the spec's selector against trace.ops of one batch "
            "(it must match a site every batch executes)",
            batch_index=batch_index,
        )
    frozen = run_state.get("intervention_sites")
    if frozen is None:
        run_state["intervention_sites"] = list(receipt["site_keys"])
        return
    if list(receipt["site_keys"]) != list(frozen):
        raise InvalidArgumentError(
            f"The intervention fired at sites {receipt['site_keys']} on batch "
            f"{batch_index}, but batch zero froze {frozen}; one artifact "
            "records one counterfactual.",
            code="extraction_intervention_site_drift",
            remedy="extract stimuli whose forward reaches the same sites, or "
            "split them into separate artifacts",
            batch_index=batch_index,
            frozen_site_keys=list(frozen),
            observed_site_keys=list(receipt["site_keys"]),
        )


def _receipt_digest(plan: InterventionPlan, sites: list[str], fire_counts: list[int]) -> str:
    """Digest the run's fire evidence (spec identity, sites, per-shard fires).

    Parameters
    ----------
    plan:
        The resolved intervention.
    sites:
        Frozen structural site keys.
    fire_counts:
        Per-shard fire counts in ledger order.

    Returns
    -------
    str
        ``sha256:``-prefixed hex digest.
    """

    payload = json.dumps(
        {
            "spec_digest": plan.spec_digest,
            "site_keys": list(sites),
            "fire_counts": [int(count) for count in fire_counts],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def final_harvest_block(
    plan: InterventionPlan | None, sites: list[str] | None, fire_counts: list[int]
) -> dict[str, Any]:
    """Build the manifest's ``tl_counterfactual_harvest_v1`` block.

    Parameters
    ----------
    plan:
        The resolved intervention, or ``None`` for a clean run (which
        declares ``policy="none"`` explicitly).
    sites:
        Frozen structural site keys (``None`` on a clean run).
    fire_counts:
        Per-shard fire counts in ledger order.

    Returns
    -------
    dict[str, Any]
        ``{HARVEST_BLOCK_KEY: payload}``, validated by construction.
    """

    if plan is None:
        return harvest_block(HarvestSpec())
    return harvest_block(
        HarvestSpec(
            policy="intervened",
            engine=INTERVENTION_ENGINE,
            address_class=plan.address_class,
            target_site_keys=tuple(sites or ()),
            receipt_digest=_receipt_digest(plan, list(sites or ()), fire_counts),
        )
    )
