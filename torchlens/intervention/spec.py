"""The ONE immutable public intervention spec (C03 substrate).

Surgery memo section 3.1: ``tl.when(...)`` is promoted to return the
immutable, public :class:`InterventionSpec` -- one multi-clause noun accepted
unchanged by ``fork.do``, ``intervene=``, Bundle
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

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .._errors import ArgumentTypeError, InvalidArgumentError
from .._selection.selectors import BaseSelector, CompositeSelector
from ._content_identity import (
    collect_tensors,
    render_helper,
    render_value,
    tensor_version,
)
from ._rule_match import _match_rules, _rule_decision
from ._site_signal import (
    _selector_children,
    addresses_autograd_nodes as addresses_autograd_nodes,
    is_gradient_site as is_gradient_site,
    site_signal,
)
from .predicates import as_intervention_decision
from .types import Edit, InterventionDecision

AddressClass = Literal["structural", "recorded", "value_dependent"]

#: The pass a site addresses: an activation site is ``"forward"``, a gradient
#: (``tl.grad``) or autograd-node site is ``"backward"``, a union of both is
#: ``"both"``. Derived from the site, never passed as a keyword.
SiteSignal = Literal["forward", "backward", "both"]


#: Decision renderings and the rule-id digest print a constant where the
#: removed ``direction`` keyword and ``Rule.direction`` field used to print
#: (always ``None`` for site-addressed rules), so renderings and rule ids
#: minted before the removal keep their value.
_RULE_ID_TAIL = "None"

#: Values the removed ``Rule.direction`` field took; a third positional
#: argument spelled with one of them is the old field, refused typed.
_REMOVED_DIRECTION_VALUES: frozenset[str] = frozenset({"forward", "backward", "both"})

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


def classify_where(target: Any) -> AddressClass:
    """Classify a WHERE term under the address law (worst class wins).

    Parameters
    ----------
    target:
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

    if isinstance(target, BaseSelector):
        from .._ir.selector_eval import selector_contains_kind

        if selector_contains_kind(target, "predicate", unwrap=True):
            return "value_dependent"
        worst: AddressClass = "structural"
        for kind in _RECORDED_SELECTOR_KINDS:
            if selector_contains_kind(target, kind, unwrap=True):
                worst = "recorded"
                break
        return worst
    return "value_dependent"


def refuse_invalid_loss(loss: Any, *, has_gradient_site: bool, door: str) -> None:
    """Refuse a ``loss=`` argument that cannot run the backward it asks for.

    ``loss=`` is a callable of the returned trace (or each swept member),
    never a tensor: under rerun semantics a tensor from the source graph
    would differentiate the wrong execution.

    Parameters
    ----------
    loss:
        The caller's ``loss=`` value (``None`` means not given).
    has_gradient_site:
        Whether any site of the call addresses the backward pass.
    door:
        Door name for the message (``"do"`` / ``"sweep"``).

    Raises
    ------
    InvalidArgumentError
        ``do_loss_not_callable`` for a tensor or any other non-callable;
        ``do_loss_without_gradient_site`` when no site is a gradient site.
    """

    if loss is None:
        return
    if not callable(loss):
        raise InvalidArgumentError(
            f"{door}(loss=...) takes a callable of the returned trace; a tensor from "
            "the source trace would differentiate the wrong execution "
            f"(received {type(loss).__name__})",
            code="do_loss_not_callable",
            remedy="pass loss=lambda result: result['output_1'].out.sum(), or call "
            "result.backward(loss) yourself on the returned trace",
            argument="loss",
        )
    if not has_gradient_site:
        raise InvalidArgumentError(
            f"{door}(loss=...) runs a backward pass for a gradient edit, but no site "
            "in this call is a gradient site",
            code="do_loss_without_gradient_site",
            remedy="address the gradient with tl.grad(site), or drop loss= for an activation edit",
            argument="loss",
        )


def split_site_by_signal(site: Any) -> tuple[Any | None, Any | None]:
    """Split a site into its activation half and its backward half.

    A union mixing ``tl.grad(s)`` (or node selectors) with activation operands
    is split at its top-level ``|``; each operand belongs to one pass.

    Parameters
    ----------
    site:
        Site the caller passed to a door.

    Returns
    -------
    tuple[Any | None, Any | None]
        ``(forward_site, backward_site)``; a half is ``None`` when the site
        has no operand in that pass.
    """

    signal = site_signal(site)
    if signal == "forward":
        return site, None
    if signal == "backward":
        return None, site
    forward: list[Any] = []
    backward: list[Any] = []
    for child in _selector_children(site):
        child_forward, child_backward = split_site_by_signal(child)
        if child_forward is not None:
            forward.append(child_forward)
        if child_backward is not None:
            backward.append(child_backward)
    return _join_union(forward), _join_union(backward)


def _join_union(operands: list[Any]) -> Any | None:
    """Rejoin union operands (``None`` for none, the operand itself for one)."""

    if not operands:
        return None
    if len(operands) == 1:
        return operands[0]
    return CompositeSelector("or", tuple(operands))


def _canonical_repr(value: Any, *, exact: bool = False) -> str:
    """Render a WHERE/action term canonically for rule identity.

    ``Edit`` renders as its portable ``(name, args, kwargs)`` identity
    -- two helpers differing only by an argument (``noise(std=0.1)`` vs
    ``noise(std=0.9)``) render differently BY CONSTRUCTION, which is what
    makes rule ids distinguishable (ledger memo D1d). Tensor arguments render
    by their FULL current content (dtype, shape, hash of every element), so
    two large vectors differing in one element are distinct rules and an
    in-place edit is visible on the next read. Opaque callables render as
    their qualified name with an ``@opaque`` marker: the payload is
    undeclarable, and the record says so rather than guessing.

    Parameters
    ----------
    value:
        WHERE or action term.
    exact:
        Re-hash tensor bytes even when their version counters are unchanged.

    Returns
    -------
    str
        Canonical rendering.
    """

    if isinstance(value, Edit):
        return render_helper(value, exact=exact)
    if isinstance(value, InterventionDecision):
        hook_repr = _canonical_repr(value.hook, exact=exact)
        return f"decision:{value.action}:{hook_repr}:direction={_RULE_ID_TAIL}"
    if isinstance(value, BaseSelector):
        return f"selector:{value!r}"
    if callable(value):
        qualname = getattr(value, "__qualname__", type(value).__name__)
        return f"callable:{qualname}@opaque"
    return f"value:{render_value(value, exact=exact)}"


@dataclass(frozen=True)
class Rule:
    """One immutable WHERE/ACTION clause of an :class:`InterventionSpec`.

    Parameters
    ----------
    where:
        Callable selector or predicate naming the target (the WHERE term).
    action:
        Helper spec, callable transform, or intervention decision (the raw
        user spelling, preserved for audit).

    The site decides the pass: :attr:`signal` is derived from the target
    (``tl.grad(site)`` edits the gradient, a bare site the activation,
    ``tl.grad(site) | site`` both) and is never a field. The removed
    ``direction`` field refuses typed (``intervention_direction_invalid``).
    """

    target: Any
    edit: Any
    address_class: AddressClass = field(default="value_dependent", compare=False)
    decision: InterventionDecision | None = field(default=None, compare=False, repr=False)
    where_repr: str = field(default="", compare=False, repr=False)
    _action_tensors: tuple[torch.Tensor, ...] = field(
        default=(), init=False, compare=False, repr=False
    )
    _identity_memo: tuple[tuple[int | None, ...], tuple[str, str]] | None = field(
        default=None, init=False, compare=False, repr=False
    )
    #: The activation half of the target (``None`` for a gradient-only or
    #: node-only target): the only part a forward predicate may match.
    _forward_target: Any = field(default=None, init=False, compare=False, repr=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> Rule:
        """Refuse the removed ``direction`` field before the dataclass init runs.

        Parameters
        ----------
        *args:
            Positional constructor arguments (``target``, ``edit``, ...).
        **kwargs:
            Keyword constructor arguments.

        Returns
        -------
        Rule
            The uninitialised instance the dataclass ``__init__`` fills.

        Raises
        ------
        InvalidArgumentError
            ``intervention_direction_invalid`` when ``direction=`` is passed,
            or a third positional argument is ``"forward"``, ``"backward"``
            or ``"both"`` (the old field's slot).
        """

        positional_stale = len(args) > 2 and args[2] in _REMOVED_DIRECTION_VALUES
        if "direction" in kwargs or positional_stale:
            value = args[2] if positional_stale else kwargs["direction"]
            raise InvalidArgumentError(
                f"Rule(direction={value!r}) is removed: the site decides the pass. "
                "A bare site edits the activation, tl.grad(site) the gradient, and "
                "tl.grad(site) | site both. Nothing was built.",
                code="intervention_direction_invalid",
                remedy="drop direction= and address the pass on the target: "
                "Rule(tl.grad(site), edit) for the gradient, Rule(site, edit) for "
                "the activation",
                argument="direction",
            )
        return super().__new__(cls)

    def __post_init__(self) -> None:
        """Validate the clause and derive its identity facts.

        Raises
        ------
        ArgumentTypeError
            If the WHERE term is not callable (``intervention_where_invalid``)
            or the action is unsupported (existing predicate-door codes).
        """

        if not callable(self.target):
            raise ArgumentTypeError(
                "InterventionSpec rule WHERE term must be a callable selector or "
                f"predicate; received {type(self.target).__name__}. Structural "
                "spellings (valid in every lane): tl.func(...), tl.in_module(...), "
                "tl.site(...). Recorded spellings: tl.label(...), Selections.",
                code="intervention_where_invalid",
                remedy="pass a selector or predicate callable as the WHERE term",
                argument="where",
            )
        decision = as_intervention_decision(self.edit)
        action_tensors: list[torch.Tensor] = []
        collect_tensors(self.edit, action_tensors)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "where_repr", _canonical_repr(self.target))
        object.__setattr__(self, "_action_tensors", tuple(action_tensors))
        object.__setattr__(self, "address_class", classify_where(self.target))
        object.__setattr__(self, "_forward_target", split_site_by_signal(self.target)[0])

    def _identity(self, *, exact: bool = False) -> tuple[str, str]:
        """Return ``(action_repr, rule_id)`` for the action's CURRENT content.

        The helper keeps a live reference to the user's tensors, so identity
        is derived on read. The result is cached against the tensors' version
        counters: an in-place torch edit (``v.add_(...)``) bumps the counter
        and re-derives on the next read, while repeated reads on the per-op
        hot path cost one counter comparison per tensor argument.

        Parameters
        ----------
        exact:
            Re-hash every tensor even when no counter moved (catches writes
            through ``.data`` or a NumPy view); refreshes the cache.

        Returns
        -------
        tuple[str, str]
            The canonical action rendering and the ``r-...`` rule id.
        """

        versions = tuple(tensor_version(tensor) for tensor in self._action_tensors)
        memo = self._identity_memo
        if not exact and memo is not None and memo[0] == versions and None not in versions:
            return memo[1]
        action_repr = _canonical_repr(self.edit, exact=exact)
        digest = hashlib.sha256(
            f"{self.where_repr}||{action_repr}||{_RULE_ID_TAIL}".encode()
        ).hexdigest()[:10]
        identity = (action_repr, f"r-{digest}")
        object.__setattr__(self, "_identity_memo", (versions, identity))
        return identity

    @property
    def rule_id(self) -> str:
        """Content-derived rule identity over WHERE and the current action."""

        return self._identity()[1]

    @property
    def signal(self) -> SiteSignal:
        """The pass this rule edits, derived from its target (not persisted)."""

        return site_signal(self.target)

    @property
    def action_repr(self) -> str:
        """Canonical rendering of the action over its current tensor content."""

        return self._identity()[0]

    def __repr__(self) -> str:
        """Render the clause with its current rule identity."""

        return (
            f"Rule(target={self.target!r}, edit={self.edit!r}, "
            f"signal={self.signal!r}, rule_id={self.rule_id!r}, "
            f"address_class={self.address_class!r})"
        )

    @property
    def payload_fidelity(self) -> Literal["declared", "opaque"]:
        """Audit fidelity of this rule's action payload (ledger memo D1c)."""

        return "opaque" if "@opaque" in self.action_repr else "declared"


@dataclass(frozen=True)
class Recipe:
    """The immutable multi-clause intervention spec (the ONE public noun).

    The same object is accepted unchanged by every intervention door.
    Callable as a capture-time ``intervene=`` predicate: evaluating the spec
    on one op context returns the single matching rule's decision, and MORE
    than one matching rule refuses typed (``spec_rules_overlap``) -- implicit
    same-site composition is exactly the silent-drop class the spec exists to
    kill; ``tl.compose`` is the explicit spelling for chained values.
    """

    rules: tuple[Rule, ...] = ()

    def __post_init__(self) -> None:
        """Refuse duplicate rules typed at construction.

        Raises
        ------
        InvalidArgumentError
            ``spec_rules_duplicate`` when two clauses carry the same rule
            identity (same WHERE and action).
        """

        seen: dict[str, int] = {}
        for index, rule in enumerate(self.rules):
            if not isinstance(rule, Rule):
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
                    f"{seen[rule.rule_id]} and {index} carry the same WHERE "
                    "and action); a spec is a SET of distinct "
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
        return _rule_decision(rule)

    def match(self, ctx: Any) -> Rule | None:
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

        return _match_rules(self.rules, ctx)

    # ------------------------------------------------------------------
    # single-rule sugar compatibility with the historical when() closure
    # ------------------------------------------------------------------
    @property
    def selector(self) -> Any | None:
        """The sole rule's WHERE term (``None`` on multi-clause specs)."""

        if len(self.rules) == 1:
            return self.rules[0].target
        return None

    @property
    def decision(self) -> InterventionDecision | None:
        """The sole rule's normalized decision (``None`` on multi-clause specs)."""

        if len(self.rules) == 1:
            return self.rules[0].decision
        return None

    @property
    def digest(self) -> str:
        """Content digest over the ordered rule identities, read NOW.

        Every read re-hashes tensor arguments (once each), so the digest
        names the content a run would use at this moment, including edits
        that bypass the version counter.
        """

        payload = "|".join(rule._identity(exact=True)[1] for rule in self.rules)
        return "spec-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def address_class(self) -> AddressClass:
        """Most restrictive address class over all rules (worst wins)."""

        worst = "structural"
        for rule in self.rules:
            if _ADDRESS_CLASS_RANK[rule.address_class] > _ADDRESS_CLASS_RANK[worst]:
                worst = rule.address_class
        return worst  # type: ignore[return-value]

    def merge(self, *others: Recipe) -> Recipe:
        """Return the multi-clause spec joining this spec's rules and others'.

        Per-rule IDs are content-derived and PRESERVED by construction; a
        duplicate clause refuses typed (``spec_rules_duplicate``).
        """

        rules = list(self.rules)
        for index, other in enumerate(others):
            if not isinstance(other, Recipe):
                raise ArgumentTypeError(
                    f"merge() takes InterventionSpec operands; operand {index} "
                    f"is {type(other).__name__}",
                    code="intervention_spec_type_invalid",
                    remedy="build each clause with tl.when(...) before merging",
                    argument="others",
                )
            rules.extend(other.rules)
        return Recipe(rules=tuple(rules))

    def __and__(self, other: Recipe) -> Recipe:
        """``spec & spec`` sugar for :meth:`merge` (one experiment, more clauses)."""

        return self.merge(other)

    def door_pairs(self) -> list[tuple[Any, Any]]:
        """Lower the spec to ``(site, action)`` pairs for the hook-plan doors."""

        return [(rule.target, rule.edit) for rule in self.rules]

    def bind(self, model: Any, *, on_zero_fire: str = "warn") -> Any:
        """Bind this spec to a model as a capture-free live executor (F01).

        Surgery memo 3.3: the result is a bound intervention executor -- a
        serial, non-reentrant, capture-free callable (never an ``nn.Module``)
        that transparently returns the base model's own output, supports real
        HF ``generate``, retains ``.last_report``, and atomically installs and
        removes its runtime state on success or exception. A bound executor
        reads the helper's tensor at each call; in-place changes apply to the
        next call, and ``.last_report.helper_tensor_versions`` records each
        tensor's version counter.

        Parameters
        ----------
        model:
            The base ``nn.Module`` (validated through the one audited
            model-door funnel; bindings do not nest).
        on_zero_fire:
            Zero-fire settlement policy, the same knob at every intervention
            door. A rule that never fired is always recorded on
            ``.last_report.zero_fire_rule_ids``; ``"warn"`` (default) also
            emits a ``bind_zero_fire`` coded warning, ``"error"`` raises
            ``bind_zero_fire`` after the call, and ``"disclose"`` records
            only.

        Returns
        -------
        BoundModel
            The bound executor (``torchlens.intervention.binding``).
        """

        from .binding import bind_spec_to_model

        return bind_spec_to_model(self, model, on_zero_fire=on_zero_fire)


def when(condition: Callable[[Any], bool], action: Any) -> Recipe:
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
        Helper spec (``tl.scale``, ``tl.add_noise``, ...), callable transform, or
        ``InterventionDecision``. The condition decides which tensor it edits:
        ``tl.when(tl.grad("relu_3"), tl.scale(0))`` edits the gradient.

    Returns
    -------
    InterventionSpec
        Immutable one-clause spec.
    """

    return Recipe(rules=(Rule(condition, action),))


def refuse_unreplayable_rules(spec: Recipe, *, door: str) -> None:
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


def entries_from_spec(spec: Recipe, *, door: str) -> list[Any]:
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
            rule.target,
            rule.edit,
            allow_replay_only_site_targets=True,
        )
        for entry in rule_entries:
            merged = dict(entry.metadata)
            merged.update(
                {
                    "spec_rule_id": rule.rule_id,
                    "spec_digest": spec.digest,
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
    "Rule",
    "Recipe",
    "classify_where",
    "entries_from_spec",
    "refuse_unreplayable_rules",
    "when",
]
