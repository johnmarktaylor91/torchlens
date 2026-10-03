"""The ONE immutable public intervention spec (C03 substrate).

Surgery memo section 3.1: ``tl.when(...)`` is promoted to return the
immutable, public :class:`InterventionSpec` -- one multi-clause noun accepted
unchanged by ``fork.do``, ``fork.attach_hooks``, ``intervene=``, Bundle
operations, generalized ``sweep``, save/load, and (F01) ``spec.bind(model)``.
Lane options (``strict``, ``log_injections``, fast tiers) never enter the
spec: a spec means the same experiment in every lane.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification (the naming sprint owns ``InterventionSpec`` / rule-noun final
names). The class is deliberately named after the megaplan mission line; the
internal mutable sticky-hook recipe keeps its historical
``torchlens.intervention.types.InterventionSpec`` identity -- the two never
meet in one namespace (this module imports the internal one under an alias).

THE ADDRESS LAW (surgery 3.1, normative, teachable in four lines)::

    structural address   (module path, op type, site key)   valid in EVERY lane
    recorded address     (labels, Selections, between)      valid everywhere
                                                            EXCEPT a first capture
    value-dependent test (predicate reading runtime state)  valid only where
                                                            something really runs

No lane may silently drop a rule it cannot implement: the replay door
preflights the whole spec and refuses, BY NAME, rules that need a
runtime-only predicate (:func:`refuse_unreplayable_rules`).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .._errors import ArgumentTypeError, InvalidArgumentError
from .predicates import as_intervention_decision
from .types import HelperDirection, HelperSpec, InterventionDecision

AddressClass = Literal["structural", "recorded", "value_dependent"]

#: Selector kinds addressable on a FIRST capture of a model (no recorded
#: numbering required): structural positions only.
_STRUCTURAL_SELECTOR_KINDS: frozenset[str] = frozenset(
    {"func", "module", "in_module", "func_transform", "site", "output"}
)

#: Selector kinds that read recorded/final numbering (labels, ordinals,
#: recorded paths): valid everywhere except the capture that MINTS them.
_RECORDED_SELECTOR_KINDS: frozenset[str] = frozenset(
    {
        "label",
        "contains",
        "regex",
        "output_at",
        "input_at",
        "followed_by",
        "preceded_by",
        "intervening",
        "grad_fn",
        "grad_kind",
        "without_op",
    }
)

_ADDRESS_CLASS_RANK: dict[str, int] = {
    "structural": 0,
    "recorded": 1,
    "value_dependent": 2,
}


def classify_where(where: Any) -> AddressClass:
    """Classify a WHERE term under the address law (worst class wins).

    Parameters
    ----------
    where:
        Selector or predicate callable naming the rule's target.

    Returns
    -------
    AddressClass
        ``"structural"``, ``"recorded"``, or ``"value_dependent"``. Composite
        selectors take the most restrictive class of any member; bare
        callables (and ``tl.where`` predicate selectors) classify
        ``"value_dependent"`` fail-closed -- their body is opaque, so no lane
        may assume it reads structure only.
    """

    from .selectors import BaseSelector

    if isinstance(where, BaseSelector):
        from ..ir.selector_eval import selector_contains_kind

        if selector_contains_kind(where, "predicate", unwrap=True):
            return "value_dependent"
        worst: AddressClass = "structural"
        for kind in _RECORDED_SELECTOR_KINDS:
            if selector_contains_kind(where, kind, unwrap=True):
                worst = "recorded"
                break
        return worst
    return "value_dependent"


def _canonical_repr(value: Any) -> str:
    """Render a WHERE/action term canonically for rule identity.

    ``HelperSpec`` renders as its portable ``(name, args, kwargs)`` identity
    -- two helpers differing only by an argument (``noise(std=0.1)`` vs
    ``noise(std=0.9)``) render differently BY CONSTRUCTION, which is what
    makes rule ids distinguishable (ledger memo D1d). Opaque callables render
    as their qualified name with an ``@opaque`` marker: the payload is
    undeclarable, and the record says so rather than guessing.
    """

    if isinstance(value, HelperSpec):
        return f"helper:{value.helper_name}:{value.args!r}:{value.kwargs!r}"
    if isinstance(value, InterventionDecision):
        return f"decision:{value.action}:{_canonical_repr(value.hook)}:direction={value.direction}"
    from .selectors import BaseSelector

    if isinstance(value, BaseSelector):
        return f"selector:{value!r}"
    if callable(value):
        qualname = getattr(value, "__qualname__", type(value).__name__)
        return f"callable:{qualname}@opaque"
    return f"value:{value!r}"


@dataclass(frozen=True)
class InterventionRule:
    """One immutable WHERE/ACTION clause of an :class:`InterventionSpec`.

    Parameters
    ----------
    where:
        Callable selector or predicate naming the target (the WHERE term).
    action:
        Helper spec, callable transform, or intervention decision (the raw
        user spelling, preserved for audit).
    direction:
        Optional signal-direction override for the action.
    """

    where: Any
    action: Any
    direction: HelperDirection | None = None
    rule_id: str = field(default="", compare=False)
    address_class: AddressClass = field(default="value_dependent", compare=False)
    decision: InterventionDecision | None = field(default=None, compare=False, repr=False)
    where_repr: str = field(default="", compare=False, repr=False)
    action_repr: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the clause and derive its identity facts.

        Raises
        ------
        ArgumentTypeError
            If the WHERE term is not callable (``intervention_where_invalid``)
            or the action is unsupported (existing predicate-door codes).
        """

        if not callable(self.where):
            raise ArgumentTypeError(
                "InterventionSpec rule WHERE term must be a callable selector or "
                f"predicate; received {type(self.where).__name__}. Structural "
                "spellings (valid in every lane): tl.func(...), tl.in_module(...), "
                "tl.site(...). Recorded spellings: tl.label(...), Selections.",
                code="intervention_where_invalid",
                remedy="pass a selector or predicate callable as the WHERE term",
                argument="where",
            )
        decision = as_intervention_decision(self.action, direction=self.direction)
        where_repr = _canonical_repr(self.where)
        action_repr = _canonical_repr(self.action)
        digest = hashlib.sha256(
            f"{where_repr}||{action_repr}||{self.direction}".encode()
        ).hexdigest()[:10]
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "where_repr", where_repr)
        object.__setattr__(self, "action_repr", action_repr)
        object.__setattr__(self, "rule_id", f"r-{digest}")
        object.__setattr__(self, "address_class", classify_where(self.where))

    @property
    def payload_fidelity(self) -> Literal["declared", "opaque"]:
        """Audit fidelity of this rule's action payload (ledger memo D1c)."""

        return "opaque" if "@opaque" in self.action_repr else "declared"


@dataclass(frozen=True)
class InterventionSpec:
    """The immutable multi-clause intervention spec (the ONE public noun).

    The same object is accepted unchanged by every intervention door.
    Callable as a capture-time ``intervene=`` predicate: evaluating the spec
    on one op context returns the single matching rule's decision, and MORE
    than one matching rule refuses typed (``spec_rules_overlap``) -- implicit
    same-site composition is exactly the silent-drop class the spec exists to
    kill; ``tl.compose`` is the explicit spelling for chained values.
    """

    rules: tuple[InterventionRule, ...] = ()

    def __post_init__(self) -> None:
        """Refuse duplicate rules typed at construction.

        Raises
        ------
        InvalidArgumentError
            ``spec_rules_duplicate`` when two clauses carry the same rule
            identity (same WHERE, action, and direction).
        """

        seen: dict[str, int] = {}
        for index, rule in enumerate(self.rules):
            if not isinstance(rule, InterventionRule):
                raise ArgumentTypeError(
                    "InterventionSpec rules must be InterventionRule clauses; "
                    f"clause {index} is {type(rule).__name__}",
                    code="intervention_rule_type_invalid",
                    remedy="build clauses with tl.when(...) and merge specs",
                    argument="rules",
                )
            if rule.rule_id in seen:
                raise InvalidArgumentError(
                    f"duplicate intervention rule {rule.rule_id} (clauses "
                    f"{seen[rule.rule_id]} and {index} carry the same WHERE, "
                    "action, and direction); a spec is a SET of distinct "
                    "experiments -- repeat-application is not expressible by "
                    "duplication",
                    code="spec_rules_duplicate",
                    remedy="drop the duplicate clause (one rule already "
                    "applies at every matching site)",
                    argument="rules",
                )
            seen[rule.rule_id] = index

    # ------------------------------------------------------------------
    # capture-door compatibility: the spec IS an intervene= predicate
    # ------------------------------------------------------------------
    def __call__(self, ctx: Any) -> InterventionDecision | None:
        """Evaluate the spec as a capture-time intervention predicate.

        Parameters
        ----------
        ctx:
            ``RecordContext`` for the current candidate op.

        Returns
        -------
        InterventionDecision | None
            The single matching rule's decision, or ``None``.

        Raises
        ------
        InvalidArgumentError
            ``spec_rules_overlap`` when more than one rule matches one op --
            firing one would silently drop the other, and implicit value
            chaining is never assumed (``tl.compose`` is the explicit
            spelling).
        """

        rule = self.match(ctx)
        if rule is None:
            return None
        # Leverage B7: thread the matched rule's provenance through the
        # decision so the capture-door staging site persists the USER'S
        # EXPRESSION, never only the lowered per-site label targets.
        decision = rule.decision
        if decision is not None and decision.rule_id is None:
            decision = dataclasses.replace(
                decision,
                rule_id=rule.rule_id,
                where_repr=rule.where_repr,
            )
        return decision

    def match(self, ctx: Any) -> InterventionRule | None:
        """Return the single rule matching one op context, or ``None``.

        The rule-attributing form of :meth:`__call__` (the capture door reads
        the decision; the F01 bind engine needs the RULE for per-rule fire
        counts and zero-fire settlement). Both spellings share this one
        matcher, so the overlap law holds identically in every lane.

        Parameters
        ----------
        ctx:
            ``RecordContext`` (or context-shaped subject) for the candidate op.

        Raises
        ------
        InvalidArgumentError
            ``spec_rules_overlap`` when more than one rule matches one op.
        """

        matched: list[InterventionRule] = []
        for rule in self.rules:
            if rule.where(ctx):
                matched.append(rule)
        if not matched:
            return None
        if len(matched) > 1:
            names = ", ".join(rule.rule_id for rule in matched)
            raise InvalidArgumentError(
                f"intervention rules [{names}] all match op "
                f"{getattr(ctx, 'label', ctx)!r}; a spec never implicitly "
                "chains two edits at one site",
                code="spec_rules_overlap",
                remedy="compose the actions explicitly with tl.compose(...), "
                "or narrow the WHERE terms so at most one rule matches each op",
                argument="rules",
            )
        return matched[0]

    # ------------------------------------------------------------------
    # single-rule sugar compatibility with the historical when() closure
    # ------------------------------------------------------------------
    @property
    def selector(self) -> Any | None:
        """The sole rule's WHERE term (``None`` on multi-clause specs)."""

        if len(self.rules) == 1:
            return self.rules[0].where
        return None

    @property
    def decision(self) -> InterventionDecision | None:
        """The sole rule's normalized decision (``None`` on multi-clause specs)."""

        if len(self.rules) == 1:
            return self.rules[0].decision
        return None

    @property
    def spec_digest(self) -> str:
        """Stable content digest over the ordered rule identities."""

        payload = "|".join(rule.rule_id for rule in self.rules)
        return "spec-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def address_class(self) -> AddressClass:
        """Most restrictive address class over all rules (worst wins)."""

        worst = "structural"
        for rule in self.rules:
            if _ADDRESS_CLASS_RANK[rule.address_class] > _ADDRESS_CLASS_RANK[worst]:
                worst = rule.address_class
        return worst  # type: ignore[return-value]

    def merge(self, *others: InterventionSpec) -> InterventionSpec:
        """Return the multi-clause spec joining this spec's rules and others'.

        Per-rule IDs are content-derived and PRESERVED by construction; a
        duplicate clause refuses typed (``spec_rules_duplicate``).
        """

        rules = list(self.rules)
        for index, other in enumerate(others):
            if not isinstance(other, InterventionSpec):
                raise ArgumentTypeError(
                    f"merge() takes InterventionSpec operands; operand {index} "
                    f"is {type(other).__name__}",
                    code="intervention_spec_type_invalid",
                    remedy="build each clause with tl.when(...) before merging",
                    argument="others",
                )
            rules.extend(other.rules)
        return InterventionSpec(rules=tuple(rules))

    def __and__(self, other: InterventionSpec) -> InterventionSpec:
        """``spec & spec`` sugar for :meth:`merge` (one experiment, more clauses)."""

        return self.merge(other)

    def door_pairs(self) -> list[tuple[Any, Any]]:
        """Lower the spec to ``(site, action)`` pairs for the hook-plan doors."""

        return [(rule.where, rule.action) for rule in self.rules]

    def bind(self, model: Any, *, on_zero_fire: str = "error") -> Any:
        """Bind this spec to a model as a capture-free live executor (F01).

        Surgery memo 3.3: the result is a bound intervention executor -- a
        serial, non-reentrant, capture-free callable (never an ``nn.Module``)
        that transparently returns the base model's own output, supports real
        HF ``generate``, retains ``.last_report``, and atomically installs and
        removes its runtime state on success or exception.

        Parameters
        ----------
        model:
            The base ``nn.Module`` (validated through the one audited
            model-door funnel; bindings do not nest).
        on_zero_fire:
            Zero-fire settlement policy (FOLD-A3 default): ``"error"``
            fails closed after the call when any rule never fired (the report
            is retained on ``.last_report``); ``"disclose"`` records the
            zero-fire rules in the report without raising.

        Returns
        -------
        BoundInterventionExecutor
            The bound executor (``torchlens.intervention.binding``).
        """

        from .binding import bind_spec_to_model

        return bind_spec_to_model(self, model, on_zero_fire=on_zero_fire)


def when(
    condition: Callable[[Any], bool],
    action: Any,
    *,
    direction: HelperDirection | None = None,
) -> InterventionSpec:
    """Build a one-clause immutable :class:`InterventionSpec`.

    The promoted public constructor (surgery memo 3.1): the returned spec is
    the durable noun every intervention door accepts. It remains directly
    usable as a capture-time ``intervene=`` predicate, and single-clause
    specs keep the historical ``.selector`` / ``.decision`` surface.

    Parameters
    ----------
    condition:
        Selector or predicate evaluated against each op context (the WHERE
        term; see the address law in the module docstring).
    action:
        Helper spec (``tl.scale``, ``tl.noise``, ...), callable transform, or
        ``InterventionDecision``.
    direction:
        Optional signal direction override.

    Returns
    -------
    InterventionSpec
        Immutable one-clause spec.
    """

    return InterventionSpec(rules=(InterventionRule(condition, action, direction),))


def refuse_unreplayable_rules(spec: InterventionSpec, *, door: str) -> None:
    """Preflight the whole spec on a recorded-graph door; refuse BY NAME.

    Surgery 3.1: no lane may silently drop a rule it cannot implement. The
    replay/attach doors operate on the recorded graph, so a rule whose WHERE
    term is a runtime-only predicate (``value_dependent``) has nothing sound
    to resolve against -- recorded payloads may be unsaved, and a predicate
    body may read state that only exists mid-forward.

    Parameters
    ----------
    spec:
        The whole spec (preflighted before ANY rule attaches).
    door:
        Human door name for the message (``"do"`` / ``"attach_hooks"``).

    Raises
    ------
    InvalidArgumentError
        ``spec_rule_unreplayable`` naming every refused rule.
    """

    refused = [rule for rule in spec.rules if rule.address_class == "value_dependent"]
    if not refused:
        return
    names = ", ".join(f"{rule.rule_id} ({rule.where_repr})" for rule in refused)
    raise InvalidArgumentError(
        f"{door}(spec) cannot apply rule(s) [{names}]: their WHERE terms are "
        "runtime-only predicates (value-dependent tests are valid only where "
        "something really runs), and silently dropping a rule is never an "
        "option",
        code="spec_rule_unreplayable",
        remedy="re-express the WHERE term as a structural or recorded "
        "selector (tl.func / tl.in_module / tl.site / tl.label), or run the "
        "spec on a real execution lane: tl.trace(model, x, intervene=spec)",
        argument="spec",
    )


def entries_from_spec(spec: InterventionSpec, *, door: str) -> list[Any]:
    """Lower a public spec to normalized hook-plan entries for one door.

    Preflights the WHOLE spec first (:func:`refuse_unreplayable_rules`), so a
    refused rule attaches nothing. Every entry carries the rule's identity in
    its metadata -- the audit builder and the persistence path read it back,
    which is what makes two members differing only by an action argument
    distinguishable end to end.

    Parameters
    ----------
    spec:
        The public immutable spec.
    door:
        Door name for refusal messages (``"do"`` / ``"attach_hooks"``).

    Returns
    -------
    list
        ``NormalizedHookEntry`` rows in rule order.
    """

    from dataclasses import replace as dc_replace

    from .hooks import normalize_hook_plan

    refuse_unreplayable_rules(spec, door=door)
    entries: list[Any] = []
    for rule in spec.rules:
        rule_entries = normalize_hook_plan(
            rule.where,
            rule.action,
            direction=rule.direction,
            allow_replay_only_site_targets=True,
        )
        for entry in rule_entries:
            merged = dict(entry.metadata)
            merged.update(
                {
                    "spec_rule_id": rule.rule_id,
                    "spec_digest": spec.spec_digest,
                    "spec_where_repr": rule.where_repr,
                    "spec_action_repr": rule.action_repr,
                    "spec_address_class": rule.address_class,
                    "spec_payload_fidelity": rule.payload_fidelity,
                }
            )
            entries.append(dc_replace(entry, metadata=merged))
    return entries


__all__ = [
    "AddressClass",
    "InterventionRule",
    "InterventionSpec",
    "classify_where",
    "entries_from_spec",
    "refuse_unreplayable_rules",
    "when",
]
