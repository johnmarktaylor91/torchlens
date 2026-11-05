"""The overlap-law rule matcher shared by every intervention door.

``Recipe.match`` and ``Recipe.__call__`` delegate here, and the module-boundary
door evaluates the rules that address a module through the same matcher, so the
``spec_rules_overlap`` refusal holds identically everywhere.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .spec import Rule
    from .types import InterventionDecision


def _match_rules(rules: Iterable[Rule], ctx: Any) -> Rule | None:
    """Return the single rule among ``rules`` whose forward WHERE matches ``ctx``.

    The one matcher behind :meth:`Recipe.match`, shared with doors that
    evaluate only a subset of a spec's rules at one site (the module-boundary
    door evaluates the rules that address a module), so the overlap law holds
    identically everywhere.

    Parameters
    ----------
    rules:
        Candidate rules, in spec order.
    ctx:
        ``RecordContext`` (or context-shaped subject) for the candidate site.

    Returns
    -------
    Rule | None
        The matching rule, or ``None``.

    Raises
    ------
    InvalidArgumentError
        ``spec_rules_overlap`` when more than one rule matches one site.
    """

    matched: list[Rule] = []
    for rule in rules:
        # A forward predicate sees only each rule's activation half: a
        # tl.grad(...) or autograd-node half edits the backward pass and is
        # mounted there by the capture and bind doors.
        forward_target = rule._forward_target
        if forward_target is not None and forward_target(ctx):
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


def _rule_decision(rule: Rule) -> InterventionDecision | None:
    """Return a matched rule's decision carrying the rule's provenance.

    Leverage B7: the matched rule's identity rides on the decision so the
    capture-door staging site persists the USER'S EXPRESSION (and its rule
    id), never only the lowered per-site label targets.

    Parameters
    ----------
    rule:
        The rule that matched one site.

    Returns
    -------
    InterventionDecision | None
        The rule's normalized decision with ``rule_id`` and ``where_repr`` set.
    """

    decision = rule.decision
    if decision is not None and decision.rule_id is None:
        decision = dataclasses.replace(
            decision,
            rule_id=rule.rule_id,
            where_repr=rule.where_repr,
        )
    return decision
