"""Episode join arms (lane F40c): live + settlement ``step_join``.

``step_join`` grades the OBSERVABLE join between one step's output and the
next step's input -- direct continuation, a transform observed inside the
capture, a declared crossing, or a failed/unchecked join with a reason. It
grades the join; it never claims to know whether the cause was a tool, a
human, a file, a network, or a process (trace-verb verdict, section 2).

Three arms (foldA MEMO s5 item 8; FORK-4 selects the DEFAULT's shape later,
BOTH fork arms are built):

- DEFAULT (``feed="open"``, ``on_feed_break="disclose"``): return the one
  Trace, mark the broken join and break step in the header's ``step_join``
  envelope, and refuse every episode-dependent claim across it (step-series
  reads, whole-episode replay, the blessing fold, escalation). Ops, values,
  graph, audit, and per-segment replay remain usable.
- FORK-4 arm (b) (``on_feed_break="refuse"``): the settlement write raises
  typed ``episode_feed_break_exogenous`` carrying the settled product on
  ``exc.partial_log`` -- recoverable partial evidence, nothing discarded.
- STRICT (``EpisodeSpec(feed="closed")``): for an undeclared crossing the
  capture HALTS at the next step entry -- the first point where the break is
  observable (one-step detection latency; a live check cannot see a break
  before the broken input arrives). A DECLARED crossing under strict mode
  stops BEFORE entering the crossing step.

The live check runs through ordinary ``nn`` forward pre/post hooks on the
stepped module, armed for exactly one capture; every tensor read inside the
hooks runs under ``pause_logging`` so the measurement is invisible to the
recorded graph (the F40a four-clause declaration invariant is preserved).
Measurement FAILURES never raise -- they degrade the affected join to
``unchecked`` with a reason; only the deliberate strict-arm refusals raise.

Persisted product: the ``episode_step_join_v1`` envelope in the ledger
header's reserved ``step_join`` slot (C07X entry-dark slot; no grammar bump).
An absent envelope reads as UNMEASURED on every surface -- a pre-F40c or
failed-measurement artifact can never present an unmeasured join as measured.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session; the
refusal codes are the stable branch surface. Doc of record:
``docs/reference/episode_capture.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from ._episode_derivation import _declaration_error

if TYPE_CHECKING:
    import torch.nn as nn

    from ..options import EpisodeSpec
    from ._episode_ledger import EpisodeLedgerHeader, ResolvedEpisode

__all__ = [
    "ENTRY_SNAPSHOT_ELEMENT_CEILING",
    "STEP_JOIN_GRADES",
    "STEP_JOIN_SCHEMA",
    "EpisodeJoinSession",
    "active_episode_step",
    "active_join_session",
    "check_series_join_claim",
    "default_break_teaching",
    "refuse_broken_join_claim",
    "settle_step_join",
    "step_join_break",
    "step_join_envelope_for",
    "validate_step_join_envelope",
]

#: Schema tag stamped on the persisted step_join envelope.
STEP_JOIN_SCHEMA = "episode_step_join_v1"

#: Closed join-grade vocabulary (trace-verb verdict section 2). ``None`` in
#: the grades column marks a join that does not exist (the prefill row and
#: rows that never started); it is not a member of this vocabulary.
STEP_JOIN_GRADES = frozenset({"continuous", "transformed", "declared", "exogenous", "unchecked"})

#: Grades that BREAK the closed-episode claim (the join is not one feed).
_BREAK_GRADES = frozenset({"exogenous", "declared"})

#: Element ceiling for integer entry snapshots; larger entries fall to the
#: digest basis (position counting needs the ids in host memory, and a
#: megabyte-scale per-step host copy is not a diagnostic-tier cost).
ENTRY_SNAPSHOT_ELEMENT_CEILING = 1_048_576

_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "claim",
        "basis",
        "grades",
        "break_step",
        "live_break_step",
        "exogenous_positions",
        "reasons",
    }
)

#: The one armed join session of the running capture (captures are
#: single-threaded by design; ``armed_capture`` sets and clears it). Lane
#: F42's step-qualified selectors and the coupling fire attribution read the
#: live step position through the accessors below, never the session object.
_ACTIVE_JOIN_SESSION: EpisodeJoinSession | None = None


def active_join_session() -> EpisodeJoinSession | None:
    """Return the armed join session of the running episode capture."""

    return _ACTIVE_JOIN_SESSION


def active_episode_step() -> int | None:
    """Return the 0-based episode step the running capture is inside.

    ``None`` outside an episode capture AND between stepped-module calls
    inside one (root-loop ops before/after/between steps belong to no step;
    lane F42 attributes them as outside-step facts, never guesses a row).
    """

    session = _ACTIVE_JOIN_SESSION
    return None if session is None else session.in_step


def default_break_teaching(step: int, exogenous_positions: int | None) -> str:
    """The DEFAULT-arm teaching text (trace-verb verdict, verbatim draft).

    ``tl.conversation(...)`` and ``split_episode()`` are the verdict's named
    chain spellings (future surfaces); the naming sprint may polish the
    wording, the semantics are pinned.
    """

    if exogenous_positions is not None:
        cause = f"{exogenous_positions} positions entered this loop from outside the capture"
    else:
        cause = "content entered this loop from outside the capture"
    return (
        f"Step {step}'s input does not continue step {step - 1}'s output -- "
        f"{cause}. The trace is intact and every recorded op is real, but "
        "this is not one replayable episode: it is chain-shaped at this "
        "join. Token-series reads and whole-episode replay are refused; "
        "ops, values, graph, and per-segment replay are unaffected. Express "
        "the run as a chain: `tl.conversation(...)` with the exogenous span "
        "at a tool join, or `split_episode()` for attested segments. To "
        "refuse at capture time instead, declare `feed='closed'`."
    )


def _declared_crossing_teaching(step: int) -> str:
    """Teaching text for claims refused across a DECLARED crossing."""

    return (
        f"Step {step}'s input crosses a DECLARED exogenous boundary "
        f"(EpisodeSpec.crossings includes step {step}). The trace is intact "
        "and the crossing is disclosed, but a declared crossing is still not "
        "one closed feed: token-series reads, whole-episode replay, and the "
        "blessing fold are refused across it; ops, values, graph, and "
        "per-segment replay are unaffected."
    )


def closed_violation_teaching(step: int, *, settled: bool = False) -> str:
    """The STRICT-arm teaching text (trace-verb verdict, verbatim draft).

    ``settled=True`` is the residual arm where the live check could not
    affirm the break (a single-position injection is live-indistinguishable
    from a model emission without the settlement evidence column), so the
    violation is raised at settlement over a completed capture; the rows
    sentence is replaced with the measured truth instead of misstating it.
    """

    head = (
        f"`feed='closed'` was declared; step {step} received input that "
        f"does not continue step {step - 1}'s output. "
    )
    if settled:
        middle = (
            "The break was measured at settlement (it was not observable at "
            "the step entry), so the capture had already completed; every "
            "row is recorded and the closed-feed declaration is violated. "
        )
    else:
        middle = (
            f"Capture stopped at the break: rows 0-{step - 1} complete, row "
            f"{step} interrupted, later rows absent. "
        )
    return (
        head
        + middle
        + ("Re-run without `feed='closed'` to capture across the break with a disclosed join.")
    )


def _join_error(message: str, *, code: str, **payload: Any) -> Exception:
    """Build the typed join refusal (import-local to keep the leaf light)."""

    from ..errors.episode import EpisodeJoinError

    return EpisodeJoinError(message, code=code, **payload)


def _tensor_digest(value: Any) -> str:
    """``sha256:`` content digest over dtype + shape + raw bytes.

    Identical construction to the ledger's per-step slice digest so a
    digest-kind evidence column and a boundary snapshot of the same payload
    compare equal.
    """

    import torch

    source = value.detach().cpu().reshape(-1)
    flat = torch.empty(source.shape, dtype=source.dtype)
    flat.copy_(source)
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(str(tuple(value.shape)).encode("utf-8"))
    if flat.numel():
        hasher.update(flat.view(torch.uint8).numpy().tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _first_tensor(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None) -> Any | None:
    """The step's ENTRY EVIDENCE: the first tensor among the call arguments.

    Disclosed basis choice: the stepped call's first positional (then
    keyword) tensor is the step input on every shipped realism shape
    (``input_ids`` for LMs, the carried sample for diffusion/fixed-point
    roots). A call with no tensor argument degrades the join to
    ``unchecked``, never to a guess.
    """

    import torch

    for value in args:
        if isinstance(value, torch.Tensor):
            return value
    for value in (kwargs or {}).values():
        if isinstance(value, torch.Tensor):
            return value
    return None


def _first_tensor_leaf(output: Any) -> Any | None:
    """First tensor leaf of a step output (tensor / tuple / list / mapping)."""

    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for value in output.values():
            found = _first_tensor_leaf(value)
            if found is not None:
                return found
        return None
    if isinstance(output, (tuple, list)):
        for value in output:
            found = _first_tensor_leaf(value)
            if found is not None:
                return found
    return None


@dataclass
class _StepBoundary:
    """Per-step boundary snapshot taken by the live hooks."""

    entry_ids: tuple[int, ...] | None = None
    entry_shape: tuple[int, ...] | None = None
    entry_digest: str | None = None
    exit_digest: str | None = None
    unavailable_reason: str | None = None


def _suffix_alignment(cur: Sequence[int], expected: Sequence[int], wildcard_tail: int) -> int:
    """Longest L with ``cur[:L]`` equal to the last L positions of ``expected``.

    The final ``wildcard_tail`` positions of ``expected`` match any value
    (the live check does not know the model's emission before settlement).
    Every shipped feed shape is one rule instance: append feed (``cur ==
    expected``), last-token KV feed (``cur == expected[-1:]``), and sliding
    context windows (``cur`` a suffix of ``expected``) all align fully;
    mid-loop injection leaves exactly the injected tail unaligned.
    """

    n_expected = len(expected)
    wild_from = n_expected - wildcard_tail
    for length in range(min(len(cur), n_expected), 0, -1):
        offset = n_expected - length
        if all(cur[i] == expected[offset + i] or (offset + i) >= wild_from for i in range(length)):
            return length
    return 0


def _entry_rows(boundary: _StepBoundary) -> list[tuple[int, ...]] | None:
    """Split an integer entry snapshot into per-batch-row id sequences.

    Rank 1 is one row; rank 2 is ``[batch, positions]``. Higher ranks have
    no defensible generic position axis and fall to the digest basis.
    """

    if boundary.entry_ids is None or boundary.entry_shape is None:
        return None
    shape = boundary.entry_shape
    if len(shape) == 1:
        return [boundary.entry_ids]
    if len(shape) == 2:
        batch, width = shape
        ids = boundary.entry_ids
        return [tuple(ids[row * width : (row + 1) * width]) for row in range(batch)]
    return None


def _grade_token_join(
    prev: _StepBoundary,
    cur: _StepBoundary,
    emitted: Sequence[int] | None,
) -> tuple[str, int] | None:
    """Grade one join on the token basis; ``None`` when the basis is absent.

    ``emitted`` is step k-1's evidence-column emission (one id per batch
    row) when settlement has it; without it the emission position is a
    wildcard (the live rule).

    Returns
    -------
    tuple[str, int] | None
        ``(grade, exogenous_position_count)`` or ``None`` to fall through
        to the next basis.
    """

    prev_rows = _entry_rows(prev)
    cur_rows = _entry_rows(cur)
    if prev_rows is None or cur_rows is None or len(prev_rows) != len(cur_rows):
        return None
    per_row_emitted: list[tuple[int, ...]] | None = None
    if emitted is not None:
        if len(emitted) == len(cur_rows):
            per_row_emitted = [(int(token),) for token in emitted]
        elif len(cur_rows) == 1:
            per_row_emitted = [tuple(int(token) for token in emitted)]
        else:
            return None
    exogenous = 0
    for row_index, cur_row in enumerate(cur_rows):
        if per_row_emitted is not None:
            expected = prev_rows[row_index] + per_row_emitted[row_index]
            wildcard_tail = 0
        else:
            expected = prev_rows[row_index] + (0,)
            wildcard_tail = 1
        exogenous += len(cur_row) - _suffix_alignment(cur_row, expected, wildcard_tail)
    return ("continuous", 0) if exogenous == 0 else ("exogenous", exogenous)


class EpisodeJoinSession:
    """Live join-measurement session, armed on the stepped module.

    Session-only (never persisted); one session per ``tl.trace`` call.
    ``arm()`` registers the boundary hooks and is idempotent -- a rescue
    re-run resets the recorded boundaries through ``reset()`` instead of
    accumulating two captures' steps.
    """

    def __init__(
        self,
        stepped_module: nn.Module,
        *,
        feed: str,
        on_feed_break: str,
        crossings: tuple[int, ...],
    ) -> None:
        self._stepped_module = stepped_module
        self.feed = feed
        self.on_feed_break = on_feed_break
        self.crossings = frozenset(crossings)
        self.boundaries: list[_StepBoundary] = []
        self.live_grades: dict[int, str] = {}
        self.live_break_step: int | None = None
        #: 0-based step currently executing inside the stepped module, or
        #: ``None`` between calls (lane F42 reads it via
        #: :func:`active_episode_step` for live step attribution).
        self.in_step: int | None = None
        self._handles: list[Any] = []

    # -- lifecycle ---------------------------------------------------------

    def arm(self) -> None:
        """Register the boundary hooks (idempotent; re-arming resets state)."""

        self.reset()
        if self._handles:
            return
        module = self._stepped_module
        self._handles.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(self._post_hook, with_kwargs=True))

    def reset(self) -> None:
        """Drop recorded boundaries (a fresh capture pass is starting)."""

        self.boundaries = []
        self.live_grades = {}
        self.live_break_step = None
        self.in_step = None

    def disarm(self) -> None:
        """Remove the boundary hooks; the session keeps its measurements."""

        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -- hooks --------------------------------------------------------------

    def _pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Record a step-entry boundary and live-grade the join INTO it.

        Snapshot failure degrades the boundary (``unavailable_reason``),
        never raises; the STRICT arm's typed halts (declared-crossing stop,
        undeclared-crossing violation) are the only exceptions allowed out.
        """

        from .. import _state

        boundary = _StepBoundary()
        try:
            with _state.pause_logging():
                self._snapshot_entry(boundary, args, kwargs)
        except Exception as exc:  # noqa: BLE001 - measurement failure degrades, never raises
            boundary.unavailable_reason = f"entry_snapshot_failed:{type(exc).__name__}"
        self.boundaries.append(boundary)
        step = len(self.boundaries) - 1
        self.in_step = step
        if step == 0:
            return
        # STRICT arm, declared crossing: stop BEFORE entering the crossing
        # step (the crossing is declared, so no observation is needed).
        if self.feed == "closed" and step in self.crossings:
            raise _join_error(
                f"`feed='closed'` with a declared crossing at step {step}: "
                f"capture stops before entering the crossing step. Rows "
                f"0-{step - 1} are complete; row {step} marks the declared "
                "stop. Re-run without `feed='closed'` to capture across the "
                "declared crossing with a disclosed join.",
                code="episode_declared_crossing_stop",
                break_step=step,
            )
        grade = self._grade_live(step)
        self.live_grades[step] = grade
        if grade == "exogenous" and self.live_break_step is None:
            self.live_break_step = step
        # STRICT arm, undeclared crossing: halt at the next step entry --
        # the first point where the break is observable (one-step latency).
        if self.feed == "closed" and grade == "exogenous":
            raise _join_error(
                closed_violation_teaching(step),
                code="episode_feed_closed_violation",
                break_step=step,
            )

    def _post_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> None:
        """Record the step's exit digest on its open entry boundary.

        Measurement failure degrades the boundary, never raises; a post
        fire with no recorded entry (pre-hook halted the step) is a no-op.
        """

        from .. import _state

        self.in_step = None
        if not self.boundaries:
            return
        boundary = self.boundaries[-1]
        try:
            with _state.pause_logging():
                leaf = _first_tensor_leaf(output)
                if leaf is not None:
                    boundary.exit_digest = _tensor_digest(leaf)
        except Exception as exc:  # noqa: BLE001 - measurement failure degrades, never raises
            if boundary.unavailable_reason is None:
                boundary.unavailable_reason = f"exit_snapshot_failed:{type(exc).__name__}"

    # -- measurement ---------------------------------------------------------

    def _snapshot_entry(
        self, boundary: _StepBoundary, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Snapshot the step's first tensor entry onto ``boundary``.

        Every entry gets a content digest; small integer entries (token-id
        shaped: <= 2-D, at most ``ENTRY_SNAPSHOT_ELEMENT_CEILING`` elements)
        additionally record exact ids + shape so the settlement re-grade can
        prove prefix continuation, not just byte identity.
        """

        import torch

        entry = _first_tensor(args, kwargs)
        if entry is None:
            boundary.unavailable_reason = "no_tensor_entry"
            return
        is_integer = entry.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
        if is_integer and entry.numel() <= ENTRY_SNAPSHOT_ELEMENT_CEILING and entry.dim() <= 2:
            boundary.entry_ids = tuple(int(v) for v in entry.detach().cpu().reshape(-1).tolist())
            boundary.entry_shape = tuple(entry.shape)
        boundary.entry_digest = _tensor_digest(entry)

    def _grade_live(self, step: int) -> str:
        """Grade the join INTO ``step`` with live evidence only.

        One-step latency by construction: the join between step k-1 and
        step k is graded at step k's ENTRY, the first point where the
        broken input is observable. The emission position is a wildcard
        (settlement re-grades exactly against the evidence column).
        """

        prev = self.boundaries[step - 1]
        cur = self.boundaries[step]
        if step in self.crossings:
            return "declared"
        if prev.unavailable_reason is not None or cur.unavailable_reason is not None:
            return "unchecked"
        token_grade = _grade_token_join(prev, cur, emitted=None)
        if token_grade is not None:
            return token_grade[0]
        if cur.entry_digest is not None and cur.entry_digest == prev.exit_digest:
            return "continuous"
        if self.feed == "closed":
            # STRICT semantics: a closed feed declares DIRECT continuation;
            # an entry that is neither the prior exit nor a token
            # continuation violates the declaration even if a captured
            # transform produced it.
            return "exogenous"
        # Open feed: the entry may be a captured transform of the prior
        # output; the settlement graph witness decides transformed vs
        # exogenous. Provisional, never persisted as a claim.
        return "unchecked"


# ---------------------------------------------------------------------------
# Settlement grading
# ---------------------------------------------------------------------------


def _graph_connected(trace: Any, address: str, from_call: int, to_call: int) -> bool | None:
    """Whether captured dataflow connects call ``from_call`` to ``to_call``.

    Forward BFS from the earlier stepped call's output ops through recorded
    ``children`` edges until an op of the later call is reached. ``None``
    when the witness cannot run (missing call records), never a guess.
    """

    try:
        # ModuleCallAccessor iterates 1-based call-index KEYS; integer
        # subscripts are 0-based positional record reads.
        calls_accessor = trace.modules[address].calls
        source_labels = list(calls_accessor[from_call].output_ops or ())
        target = {str(label) for label in (calls_accessor[to_call].ops or ())}
    except (LookupError, AttributeError, TypeError, ValueError, IndexError):
        return None
    if not source_labels or not target:
        return None

    # The walk is OP-level and pass-qualified throughout (the pass-blind
    # Layer walk conflates recurrence passes: a bare multi-pass label's
    # children span every pass). ModuleCall.ops/output_ops entries and
    # Op.label are always pass-qualified; Op.children spell single-pass
    # neighbours bare, so each child resolves through the trace lookup and
    # is re-keyed by its own Op.label.
    def _resolve(label: str) -> Any | None:
        """Resolve a (possibly bare) label to its single Op, else ``None``."""

        try:
            ops = trace[label].ops
        except (LookupError, AttributeError, TypeError, ValueError):
            return None
        return ops[0] if len(ops) == 1 else None

    seen: set[str] = set()
    frontier = [str(label) for label in source_labels]
    while frontier:
        label = frontier.pop()
        op = _resolve(label)
        if op is None:
            continue
        op_label = str(op.label)
        if op_label in seen:
            continue
        seen.add(op_label)
        if op_label in target:
            return True
        frontier.extend(str(child) for child in (op.children or ()) if str(child) not in seen)
    return False


def _settle_one_join(
    trace: Any,
    resolved: Any,
    session: EpisodeJoinSession,
    step: int,
    emitted: Sequence[int] | None,
) -> tuple[str, str, int | None, str | None]:
    """Grade one join at settlement: ``(grade, basis, exogenous, reason)``."""

    if step in session.crossings:
        return ("declared", "declared", None, None)
    if step - 1 >= len(session.boundaries) or step >= len(session.boundaries):
        return ("unchecked", "none", None, "boundary_not_observed")
    prev = session.boundaries[step - 1]
    cur = session.boundaries[step]
    for boundary in (prev, cur):
        if boundary.unavailable_reason is not None:
            return ("unchecked", "none", None, boundary.unavailable_reason)
    token_grade = _grade_token_join(prev, cur, emitted)
    if token_grade is not None:
        grade, exogenous = token_grade
        basis = "tokens" if emitted is not None else "tokens_live"
        return (grade, basis, exogenous if grade == "exogenous" else None, None)
    if cur.entry_digest is not None and cur.entry_digest == prev.exit_digest:
        return ("continuous", "digest_direct", None, None)
    # Closed tri-state verdict table over the graph witness.
    return {
        True: ("transformed", "graph", None, None),
        False: ("exogenous", "graph", None, None),
        None: ("unchecked", "none", None, "graph_witness_unavailable"),
    }[_graph_connected(trace, resolved.address, step - 1, step)]


def settle_step_join(
    trace: Any,
    resolved: Any,
    row_statuses: Sequence[str],
    evidence_by_row: Sequence[Any] | None,
) -> dict[str, Any] | None:
    """Derive the persisted ``episode_step_join_v1`` envelope at settlement.

    Returns ``None`` when no live session ran (the measurement cannot be
    reconstructed post hoc; the absent envelope IS the unmeasured
    disclosure). Joins into rows that never started grade ``None`` (no join
    exists); everything else grades from the closed vocabulary, with
    measurement failures landing as ``unchecked`` + reason, never a guess.
    """

    session = getattr(resolved, "join_session", None)
    if session is None:
        return None
    grades: list[str | None] = [None]
    basis: dict[str, str] = {}
    exogenous_positions: dict[str, int] = {}
    reasons: dict[str, str] = {}
    tokens_column: Sequence[Any] | None = None
    if evidence_by_row is not None and resolved.step_output_kind == "tokens":
        tokens_column = evidence_by_row
    for step in range(1, len(row_statuses)):
        if row_statuses[step] == "absent":
            grades.append(None)
            continue
        emitted = None
        if tokens_column is not None and step - 1 < len(tokens_column):
            candidate = tokens_column[step - 1]
            if isinstance(candidate, tuple):
                emitted = candidate
        grade, join_basis, exogenous, reason = _settle_one_join(
            trace, resolved, session, step, emitted
        )
        grades.append(grade)
        basis[str(step)] = join_basis
        if exogenous is not None:
            exogenous_positions[str(step)] = exogenous
        if reason is not None:
            reasons[str(step)] = reason
    break_step = next((index for index, grade in enumerate(grades) if grade in _BREAK_GRADES), None)
    return {
        "schema": STEP_JOIN_SCHEMA,
        "claim": "measured",
        "basis": basis,
        "grades": grades,
        "break_step": break_step,
        "live_break_step": session.live_break_step,
        "exogenous_positions": exogenous_positions,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Envelope validation (fail-closed at load; C07X entry-dark slot discipline)
# ---------------------------------------------------------------------------


def validate_step_join_envelope(envelope: Mapping[str, Any], n_rows: int) -> None:
    """Validate one persisted envelope against the v1 schema, FAIL-CLOSED.

    Raises
    ------
    ValueError
        On any geometry/vocabulary violation (callers convert into
        ``episode_ledger_incoherent``; the quarantine row).
    """

    if set(envelope) != _ENVELOPE_KEYS:
        raise ValueError(
            "episode step_join envelope must carry exactly the keys "
            f"{sorted(_ENVELOPE_KEYS)}, got {sorted(envelope)}"
        )
    if envelope["schema"] != STEP_JOIN_SCHEMA:
        raise ValueError(
            f"episode step_join envelope schema {envelope['schema']!r} is not {STEP_JOIN_SCHEMA!r}"
        )
    if envelope["claim"] != "measured":
        raise ValueError(
            "episode step_join envelope claim must be 'measured' (an "
            "unmeasured join is spelled by the ABSENT envelope, never a token)"
        )
    _validate_envelope_grades(envelope, n_rows)
    _validate_envelope_mappings(envelope, n_rows)


def _validate_envelope_grades(envelope: Mapping[str, Any], n_rows: int) -> None:
    """The grades-column geometry arm of the v1 envelope validation."""

    grades = envelope["grades"]
    if not isinstance(grades, list) or len(grades) != n_rows:
        raise ValueError(
            "episode step_join grades must list one entry per ledger row "
            f"({n_rows}), got {grades!r}"
        )
    if n_rows and grades[0] is not None:
        raise ValueError("episode step_join grades[0] must be null (the prefill has no join)")
    for index, grade in enumerate(grades):
        if grade is not None and grade not in STEP_JOIN_GRADES:
            raise ValueError(
                f"episode step_join grade {grade!r} at step {index} is outside "
                f"the closed vocabulary {sorted(STEP_JOIN_GRADES)}"
            )
    for slot in ("break_step", "live_break_step"):
        value = envelope[slot]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"episode step_join {slot} must be a non-negative int or null")
    expected_break = next(
        (index for index, grade in enumerate(grades) if grade in _BREAK_GRADES), None
    )
    if envelope["break_step"] != expected_break:
        raise ValueError(
            f"episode step_join break_step {envelope['break_step']!r} does not "
            f"match the first break grade at {expected_break!r}"
        )


def _validate_envelope_mappings(envelope: Mapping[str, Any], n_rows: int) -> None:
    """The per-join mapping slots arm of the v1 envelope validation."""

    for mapping_slot, value_type in (
        ("basis", str),
        ("exogenous_positions", int),
        ("reasons", str),
    ):
        mapping_value = envelope[mapping_slot]
        if not isinstance(mapping_value, Mapping):
            raise ValueError(f"episode step_join {mapping_slot} must be a mapping")
        for key, value in mapping_value.items():
            if not isinstance(key, str) or not key.isdigit() or not (0 < int(key) < n_rows):
                raise ValueError(
                    f"episode step_join {mapping_slot} keys must be in-range "
                    f"step strings, got {key!r}"
                )
            if not isinstance(value, value_type) or isinstance(value, bool):
                raise ValueError(
                    f"episode step_join {mapping_slot} values must be "
                    f"{value_type.__name__}, got {value!r}"
                )


# ---------------------------------------------------------------------------
# Claim guards (the DEFAULT arm's refusal surfaces)
# ---------------------------------------------------------------------------


def step_join_envelope_for(subject: Any) -> Mapping[str, Any] | None:
    """Read the persisted envelope from a Trace-like or a ledger header."""

    header: Mapping[str, Any] | None = None
    annotations = getattr(subject, "annotations", None)
    if isinstance(annotations, dict):
        payload = annotations.get("episode")
        if isinstance(payload, Mapping):
            candidate = payload.get("header")
            if isinstance(candidate, Mapping):
                header = candidate
    if header is None:
        candidate = getattr(subject, "step_join", None)
        if isinstance(candidate, Mapping):
            header = {"step_join": candidate}
    if header is None:
        return None
    envelope = header.get("step_join")
    return envelope if isinstance(envelope, Mapping) else None


def step_join_break(subject: Any) -> tuple[int, str] | None:
    """Return ``(break_step, grade)`` for a measured broken join, else ``None``."""

    envelope = step_join_envelope_for(subject)
    if envelope is None:
        return None
    break_step = envelope.get("break_step")
    if break_step is None:
        return None
    grades = envelope.get("grades")
    grade = grades[break_step] if isinstance(grades, list) and break_step < len(grades) else None
    return (int(break_step), str(grade))


def _raise_break_refusal(envelope: Mapping[str, Any], break_step: int, grade: str) -> NoReturn:
    """Raise the one typed refusal for a measured broken join.

    Each branch names its literal ``code=`` (contract-lockstep scanner law):
    a declared crossing refuses ``episode_join_declared_crossing``, every
    other broken grade refuses ``episode_feed_break_exogenous``.
    """

    if grade == "declared":
        raise _join_error(
            _declared_crossing_teaching(break_step),
            code="episode_join_declared_crossing",
            break_step=break_step,
        )
    positions = envelope.get("exogenous_positions")
    count = positions.get(str(break_step)) if isinstance(positions, Mapping) else None
    raise _join_error(
        default_break_teaching(break_step, count if isinstance(count, int) else None),
        code="episode_feed_break_exogenous",
        break_step=break_step,
    )


def check_series_join_claim(subject: Any) -> None:
    """Guard an episode-SERIES claim (step-series reads, escalation).

    The strong claim door: a measured broken join refuses with the verbatim
    teaching text; an UNCHECKED join or an absent envelope refuses
    ``episode_join_unmeasured`` -- an unmeasured cross-step join can never
    support a series claim (the F40a claim-off honesty, now typed).
    """

    envelope = step_join_envelope_for(subject)
    if envelope is None:
        raise _join_error(
            "this episode's cross-step joins are UNMEASURED (the ledger "
            "carries no step_join envelope: a pre-measurement artifact or a "
            "failed live measurement), so the step series is not one "
            "attested feed. Per-row reads are unaffected. Re-capture with "
            "this TorchLens to measure the joins.",
            code="episode_join_unmeasured",
        )
    broken = step_join_break(subject)
    if broken is not None:
        break_step, grade = broken
        _raise_break_refusal(envelope, break_step, grade)
    grades = envelope.get("grades")
    if isinstance(grades, list):
        for index, grade in enumerate(grades):
            if grade == "unchecked":
                reasons = envelope.get("reasons")
                reason = reasons.get(str(index)) if isinstance(reasons, Mapping) else None
                raise _join_error(
                    f"the join into step {index} is UNCHECKED "
                    f"({reason or 'unrecorded reason'}): its continuity was "
                    "not measurable, so the step series is not one attested "
                    "feed. Per-row reads are unaffected.",
                    code="episode_join_unmeasured",
                    step=index,
                )


def refuse_broken_join_claim(subject: Any) -> None:
    """Guard an episode-dependent claim on measured broken joins ONLY.

    The gate behind WHOLE-EPISODE replay (``Trace.run``), the blessing fold
    (``derive_episode_status``), and escalation. Narrow by design: fires only
    on a MEASURED broken join (exogenous or declared). Unmeasured/legacy
    episode artifacts keep their shipped behavior on these surfaces -- the
    strict unmeasured refusal lives on the series claim door
    (:func:`check_series_join_claim`) -- and this gate never widens another
    owner's refusal surface.
    """

    envelope = step_join_envelope_for(subject)
    if envelope is None:
        return
    broken = step_join_break(subject)
    if broken is None:
        return
    break_step, grade = broken
    _raise_break_refusal(envelope, break_step, grade)


def _validated_join_declaration(spec: EpisodeSpec, declared_steps: int | None) -> tuple[int, ...]:
    """Validate the join-arm declaration triple (lane F40c).

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` for a feed/on_feed_break token
        outside its closed vocabulary or a malformed crossings declaration
        (non-int, step 0 -- the prefill's entry is the capture input, not a
        join -- or a step at/beyond the declared count).
    """

    if spec.feed not in ("open", "closed"):
        raise _declaration_error(
            f"EpisodeSpec.feed={spec.feed!r} is outside the closed vocabulary "
            "['closed', 'open']: 'open' (default) captures across feed breaks "
            "and marks them in the measured step_join envelope; 'closed' is "
            "the opt-in strict arm that halts at the break.",
            code="episode_declaration_invalid",
        )
    if spec.on_feed_break not in ("disclose", "refuse"):
        raise _declaration_error(
            f"EpisodeSpec.on_feed_break={spec.on_feed_break!r} is outside the "
            "closed vocabulary ['disclose', 'refuse']: 'disclose' (default) "
            "returns the one Trace with the break marked and episode-"
            "dependent claims refused across it; 'refuse' raises typed at "
            "settlement with the settled product on exc.partial_log.",
            code="episode_declaration_invalid",
        )
    try:
        crossings = tuple(int(step) for step in spec.crossings)
    except (TypeError, ValueError):
        raise _declaration_error(
            "EpisodeSpec.crossings must be an iterable of step indices (the "
            "declared exogenous entries), got "
            f"{type(spec.crossings).__name__}.",
            code="episode_declaration_invalid",
        ) from None
    for step in crossings:
        if step < 1:
            raise _declaration_error(
                f"EpisodeSpec.crossings declares step {step}; crossings are "
                "JOINS into steps 1+ (step 0 is the prefill, whose entry is "
                "the capture input, not a cross-step join).",
                code="episode_declaration_invalid",
            )
        if declared_steps is not None and step >= declared_steps:
            raise _declaration_error(
                f"EpisodeSpec.crossings declares step {step} but only "
                f"{declared_steps} steps are declared (steps 0-"
                f"{declared_steps - 1}).",
                code="episode_declaration_invalid",
            )
    return crossings


def _raise_settled_break_arm(
    resolved: ResolvedEpisode, header: EpisodeLedgerHeader, join_envelope: Mapping[str, Any]
) -> None:
    """Raise the strict / FORK-4(b) settlement refusal on a measured break."""

    broken = step_join_break(header)
    if broken is None or broken[1] != "exogenous":
        return
    break_step = broken[0]
    if resolved.feed == "closed":
        # Residual strict arm: the live check could not affirm this break at
        # the step entry (e.g. a single-position injection is
        # live-indistinguishable from an emission); the violation settles
        # typed over the completed capture.
        raise _join_error(
            closed_violation_teaching(break_step, settled=True),
            code="episode_feed_closed_violation",
            break_step=break_step,
        )
    if resolved.on_feed_break == "refuse":
        # FORK-4 arm (b): typed refusal carrying recoverable partial
        # evidence, built beside the disclose default (the ruling only
        # selects which shape is the DEFAULT).
        count = join_envelope.get("exogenous_positions", {}).get(str(break_step))
        raise _join_error(
            default_break_teaching(break_step, count if isinstance(count, int) else None),
            code="episode_feed_break_exogenous",
            break_step=break_step,
        )


def _live_step_join_envelope(
    resolved: ResolvedEpisode, n_rows: int, started: int
) -> dict[str, Any] | None:
    """Build the FAILED-path envelope from the live session's grades only.

    The settlement evidence column never existed (the capture failed), so
    grades are the live one-step-latency measurements, disclosed on the
    ``live`` basis; joins into rows that never started grade ``None``.
    ``None`` when no session ran (the absent envelope IS the unmeasured
    disclosure).
    """

    session = resolved.join_session
    if session is None or n_rows == 0:
        return None
    grades: list[str | None] = [None]
    basis: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for step in range(1, n_rows):
        if step >= started:
            grades.append(None)
            continue
        if step in session.crossings:
            grades.append("declared")
        elif step in session.live_grades:
            grades.append(session.live_grades[step])
        else:
            grades.append("unchecked")
            reasons[str(step)] = "not_graded_live"
        basis[str(step)] = "live"
    break_step = next(
        (index for index, grade in enumerate(grades) if grade in ("exogenous", "declared")),
        None,
    )
    return {
        "schema": STEP_JOIN_SCHEMA,
        "claim": "measured",
        "basis": basis,
        "grades": grades,
        "break_step": break_step,
        "live_break_step": session.live_break_step,
        "exogenous_positions": {},
        "reasons": reasons,
    }


@contextmanager
def armed_capture(
    episode_resolved: Any, run_capture: Callable[[], Any]
) -> Iterator[Callable[[], Any]]:
    """Yield the capture callable with the live join hooks armed (F40c).

    Arms the resolved episode's join session for exactly one capture and
    yields a callable that resets the session's recorded boundaries per
    pass -- a rescue re-run replays the capture, and two passes must never
    read as one episode's steps. Disarm rides the ``finally`` so no boundary
    hook survives the capture, success or failure (the M(oracles) item 8
    purity cell); a failed session keeps its measurements for the
    failed-path ledger. Without a join session the callable passes through
    untouched.
    """

    global _ACTIVE_JOIN_SESSION

    join_session = getattr(episode_resolved, "join_session", None)
    if join_session is None:
        yield run_capture
        return
    # Attested coupling (lane F42): the fire-attribution session arms and
    # clears in lockstep with the join session it reads step positions from.
    coupling_session = getattr(episode_resolved, "coupling_session", None)
    from ._episode_coupling import _set_active_coupling

    def _run_capture_with_join_reset() -> Any:
        """Run one capture pass with the join session's boundaries reset."""

        join_session.reset()
        if coupling_session is not None:
            coupling_session.reset()
        return run_capture()

    join_session.arm()
    _ACTIVE_JOIN_SESSION = join_session
    _set_active_coupling(coupling_session)
    try:
        yield _run_capture_with_join_reset
    finally:
        _ACTIVE_JOIN_SESSION = None
        _set_active_coupling(None)
        join_session.disarm()
