"""``Bundle.add_alternative``: merge a steered re-run of the user's harness.

TorchLens owns no harness control flow, so it never pushes a steer into
later calls. The user runs their own harness twice, as is and with one call
steered, and hands the second run's bundle to the first:
``b.add_alternative(second_run, steered=("draft",))``. TorchLens pairs the
calls by member name (never by order), checks the pairing, adds every
second-run member as ``prefix + name``, writes one ``alternative_of`` row per
paired call, copies the second run's own relation rows onto the copies and
records one ``add`` operation. The report says, for every call, whether it
was steered and whether its prompt and completion changed, each from which
token (see :mod:`torchlens.bundle._token_divergence`).

Checks run before anything changes, so a refusal leaves the bundle as it
was; a failure while merging restores it. The tripwire refuses a call that
was not steered, saw the same prompt and still wrote different text: the run
carries noise TorchLens cannot separate from the steer's effect.
Spellings documented-unstable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors.episode import BundleRelationError
from ._relations import MemberRelationRow
from ._token_divergence import TokenDivergence, capture_edit_rows, divergence

if TYPE_CHECKING:
    from . import Bundle

#: Edit names whose effect depends on token position or decode step: one
#: teacher-forced forward does not reproduce the steered generation.
_POSITION_EDITS = ("at_positions", "at_step")


@dataclass(frozen=True)
class AlternativeCall:
    """One call of an ``AlternativeReport`` (documented-unstable).

    Attributes
    ----------
    call:
        The call's member name in the original run.
    member:
        The second run's copy in this bundle (``prefix + call``), ``None``
        for a ``dropped`` call.
    status:
        ``steered``, ``unchanged``, ``absorbed`` (prompt changed, completion
        did not), ``changed`` (both changed), ``added`` (only in the second
        run) or ``dropped`` (only in the original).
    prompt_divergence, completion_divergence:
        As on ``TokenDivergence``; ``None`` for unpaired calls.
    """

    call: str
    member: str | None
    status: str
    prompt_divergence: int | None
    completion_divergence: int | None


@dataclass(frozen=True)
class AlternativeReport:
    """Result of ``Bundle.add_alternative`` (documented-unstable).

    Attributes
    ----------
    prefix:
        The prefix naming the second run's copies.
    steered:
        The declared steered calls.
    calls:
        One ``AlternativeCall`` per call of either run: the original's order,
        then calls only the second run made.
    uncopied_rows:
        Second-run relation rows of unknown namespaced kinds, not copied
        (their endpoints are unreadable without their kind's schema).
    """

    prefix: str
    steered: tuple[str, ...]
    calls: tuple[AlternativeCall, ...]
    uncopied_rows: int = 0

    def table(self) -> str:
        """Return the report as a fixed-width text table, one line per call."""

        def cell(value: int | None, paired: bool) -> str:
            """Render one divergence cell (``-`` for an unpaired call)."""

            return "-" if not paired else str(value)

        lines = [f"{'call':16} {'status':10} {'prompt diverges at':>19} {'output diverges at':>19}"]
        for row in self.calls:
            paired = row.status not in ("added", "dropped")
            lines.append(
                f"{row.call:16} {row.status:10} "
                f"{cell(row.prompt_divergence, paired):>19} "
                f"{cell(row.completion_divergence, paired):>19}"
            )
        return "\n".join(lines)


def _alternative_refusal(
    message: str, remedy: str, *, code: str, **fields: Any
) -> BundleRelationError:
    """Build one typed ``add_alternative`` refusal."""

    return BundleRelationError(message, code=code, remedy=remedy, **fields)


def _steered_names(steered: Any) -> tuple[str, ...]:
    """Normalize ``steered=`` to a non-empty tuple of distinct call names."""

    names = (steered,) if isinstance(steered, str) else steered
    if (
        not isinstance(names, Sequence)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise TypeError(
            f"add_alternative steered= takes a call name or a non-empty sequence of names, "
            f"got {steered!r}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"add_alternative steered= repeats a call name: {list(names)!r}")
    return tuple(names)


def _check_steering(original: Bundle, second: Bundle, steered: tuple[str, ...]) -> None:
    """Refuse a declared call that is absent, unsteered or position-dependent, or an undeclared steer."""

    for name in steered:
        for side, bundle in (("original", original), ("second_run", second)):
            if name not in bundle.names:
                raise _alternative_refusal(
                    f"add_alternative: the steered call {name!r} is not a member of the "
                    f"{side.replace('_', ' ')}; members: {bundle.names}.",
                    "name each steered call as the harness names its member in both runs",
                    call=name,
                    side=side,
                    code="alternative_target_missing",
                )
        rows = capture_edit_rows(second[name])
        if not any(row.get("sites") for row in rows):
            raise _alternative_refusal(
                f"add_alternative: {name!r} in the second run was not captured under a "
                "spec that fired, so its activations are not those of the steered call.",
                "capture the steered call with tl.trace(model, ids, intervene=spec), the "
                "spec it generated under",
                call=name,
                code="alternative_target_unsteered",
            )
        edits = " ".join(f"{row.get('edit', '')} {row.get('selection_repr', '')}" for row in rows)
        if any(marker in edits for marker in _POSITION_EDITS):
            raise _alternative_refusal(
                f"add_alternative: {name!r}'s spec edits chosen positions or decode steps; "
                "one teacher-forced forward does not reproduce that generation.",
                "use a position-uniform spec, or capture the generation as an episode "
                "(docs/reference/episode_capture.md)",
                call=name,
                code="alternative_capture_position_dependent",
            )
    for name in second.names:
        if name not in steered and capture_edit_rows(second[name]):
            raise _alternative_refusal(
                f"add_alternative: {name!r} in the second run was captured under a spec "
                f"but is not declared in steered={list(steered)!r}.",
                "declare every steered call in steered=, or capture the others plainly",
                call=name,
                code="alternative_steer_undeclared",
            )


def _status(name: str, steered: tuple[str, ...], report: TokenDivergence) -> str:
    """Return a paired call's status, refusing an unchanged call that diverged."""

    if name in steered:
        return "steered"
    if report.prompt_same and not report.completion_same:
        raise _alternative_refusal(
            f"add_alternative: {name!r} was not steered and saw the same prompt in both "
            f"runs, but its completion differs from token {report.completion_divergence}. "
            "The runs carry noise TorchLens cannot separate from the steer's effect; "
            "nothing was added.",
            "seed each call (not once per harness), use deterministic algorithms and keep "
            "the batch the same, then re-run",
            call=name,
            completion_divergence=report.completion_divergence,
            code="alternative_unchanged_call_diverged",
        )
    if report.prompt_same:
        return "unchanged"
    return "absorbed" if report.completion_same else "changed"


def _pair_calls(
    original: Bundle, second: Bundle, steered: tuple[str, ...], prefix: str
) -> tuple[AlternativeCall, ...]:
    """Return one ``AlternativeCall`` per call of either run (checks included)."""

    calls: list[AlternativeCall] = []
    for name in original.names:
        if name not in second.names:
            calls.append(AlternativeCall(name, None, "dropped", None, None))
            continue
        report = divergence((name, prefix + name), (original[name], second[name]))
        status = _status(name, steered, report)
        calls.append(
            AlternativeCall(
                name,
                prefix + name,
                status,
                report.prompt_divergence,
                report.completion_divergence,
            )
        )
    for name in second.names:
        if name not in original.names:
            calls.append(AlternativeCall(name, prefix + name, "added", None, None))
    return tuple(calls)


def _renamed_rows(second: Bundle, prefix: str) -> tuple[list[dict[str, Any]], int]:
    """Return the second run's rows re-addressed to the copies, plus the opaque-row count."""

    rows: list[dict[str, Any]] = []
    opaque = 0
    for row in second.member_relations:
        if not isinstance(row, MemberRelationRow):
            opaque += 1
            continue
        payload = row.to_payload()
        for key in ("from", "to", "member"):
            if key in payload:
                payload[key] = prefix + str(payload[key])
        rows.append(payload)
    return rows, opaque


def _merge(
    bundle: Bundle, second: Bundle, report: AlternativeReport, rows: list[dict[str, Any]]
) -> None:
    """Add the copies, relate them, record the operation; restore the bundle on failure."""

    saved = (
        bundle._members.copy(),
        dict(bundle._member_construction),
        bundle._member_relations,
        list(bundle._operations),
    )
    try:
        for name in second.names:
            bundle.add(second[name], names=report.prefix + name)
        pairs: list[dict[str, Any]] = [
            {"kind": "alternative_of", "from": call.member, "to": call.call, "params": {}}
            for call in report.calls
            if call.status not in ("added", "dropped")
        ]
        if pairs or rows:
            bundle.relate(*pairs, *rows)
        bundle._record_bundle_operation(
            "add",
            member_names=[report.prefix + name for name in second.names],
            params={
                "alternative": {
                    "prefix": report.prefix,
                    "steered": list(report.steered),
                    "calls": [
                        {
                            "call": call.call,
                            "member": call.member,
                            "status": call.status,
                            "prompt_divergence": call.prompt_divergence,
                            "completion_divergence": call.completion_divergence,
                        }
                        for call in report.calls
                    ],
                }
            },
        )
    except BaseException:
        members, construction, relations, operations = saved
        bundle._members = members
        bundle._member_construction = construction
        bundle._member_relations = relations
        bundle._operations = operations
        bundle._supergraph = None
        raise


def _bundle_add_alternative(
    self: Bundle,
    second_run: Bundle,
    *,
    steered: str | Sequence[str],
    prefix: str = "cf_",
) -> AlternativeReport:
    """Merge a re-run of the user's harness with some calls steered; report what changed.

    Exposed as the dynamic method ``Bundle.add_alternative`` (spelling
    documented-unstable).

    Parameters
    ----------
    self:
        The first run's bundle; it receives the copies.
    second_run:
        The bundle the same harness built with the ``steered`` calls captured
        under their specs.
    steered:
        The steered call name, or a sequence of names.
    prefix:
        Prefix naming each second-run member's copy (default ``"cf_"``).

    Returns
    -------
    AlternativeReport
        Per call: status and the first differing prompt and completion token.

    Raises
    ------
    TypeError
        When ``second_run`` is not another Bundle or ``steered`` is not a name
        or a sequence of names.
    BundleRelationError
        ``alternative_target_missing``, ``alternative_target_unsteered``,
        ``alternative_capture_position_dependent``,
        ``alternative_steer_undeclared``, ``alternative_member_name_taken``,
        ``alternative_unchanged_call_diverged`` or
        ``bundle_token_divergence_unavailable`` (a paired call without a
        bound completion range). The bundle is unchanged on every refusal.
    """

    from . import Bundle

    if not isinstance(second_run, Bundle) or second_run is self:
        raise TypeError("add_alternative takes the second run's Bundle, another Bundle than self")
    if not isinstance(prefix, str) or not prefix:
        raise TypeError(f"add_alternative prefix= must be a non-empty string, got {prefix!r}")
    names = _steered_names(steered)
    _check_steering(self, second_run, names)
    taken = [prefix + name for name in second_run.names if prefix + name in self.names]
    if taken:
        raise _alternative_refusal(
            f"add_alternative: the copies {taken} would reuse existing member names.",
            "pass another prefix=, for example prefix='cf2_'",
            names=taken,
            code="alternative_member_name_taken",
        )
    calls = _pair_calls(self, second_run, names, prefix)
    rows, opaque = _renamed_rows(second_run, prefix)
    report = AlternativeReport(prefix=prefix, steered=names, calls=calls, uncopied_rows=opaque)
    _merge(self, second_run, report, rows)
    return report
