"""Declarative user-named pattern folding, v1 (collapse memo D11 / K5).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

The one hiddenlayer idea with no TorchLens equivalent: the user NAMES an
abstraction ("ConvBnRelu") and every matched site folds under that label --
instances with different weights fold together, which the automatic
equivalence-only run fold must refuse. Credit: hiddenlayer (Waleed Abdulla)
for the idea and the ``>`` path syntax; the code, the ``{k}`` repetition
sugar, acyclic name-inlining composability, and the honesty core are ours.

v1 grammar (typed AST, never raw strings past the parser):

- a pattern is a LINEAR path of atoms separated by ``>``;
- an atom is a canonical op-family name (``conv2d``, ``relu``, ``add``) or
  a module class name (``Conv2d``, ``BatchNorm2d``), optionally with an
  exact repetition ``{k}`` (parser sugar, expanded flat);
- an atom naming an EARLIER declared pattern inlines flat at parse time
  (acyclic by construction: only earlier names are visible), so the matcher
  sees only expanded typed linear paths.

The honesty core hiddenlayer never had:

- single entry, single exit: any external edge touching a match's interior
  refuses THAT instance;
- protected landmark ops (junctions) inside a match refuse the instance;
- equal-priority ambiguous extensions refuse with a diagnostic (silent
  tie-breaking in a NAMING feature teaches users their pattern applied
  where it did not);
- refusals are COUNTED AND DISCLOSED ("ConvBnRelu: 46 of 49 sites folded;
  3 refused: an external edge enters the interior");
- unknown tokens get near-miss suggestions from the trace; a no-match
  pattern warns once.

No uniformity claim, ever: each instance is its own chip with its own
stats. ``collapse="none"`` + patterns is the supported pattern-only view
(hiddenlayer's default experience); combining patterns with automatic
collapse refuses typed in v1 (the chip-atomicity integration across band /
score / ceiling / schedule / plan-cache identity is the named follow-on,
carried with plumbing: the typed matched-region records below are the
seam). Default off; the curated ``"idiomatic"`` preset (the conv-bn-relu
family) is available, also off.
"""

from __future__ import annotations

import difflib
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from .._errors import InvalidArgumentError
from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel
from ._condensed_flow import JUNCTION_FUNC_NAMES
from ._segment_descriptors import _make_op_segment_descriptor
from .collapse_plan import SegmentDescriptor

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .collapse_plan import RenderContext

_ATOM_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?:\{(\d+)\})?$")

#: Curated preset (memo D11): the conv-bn-relu idiom family. Off by default;
#: selected with ``fold_patterns="idiomatic"``.
IDIOMATIC_PATTERNS: Mapping[str, str] = {
    "ConvBnRelu": "conv2d > batch_norm > relu",
    "ConvBn": "conv2d > batch_norm",
    "ConvRelu": "conv2d > relu",
    "LinearRelu": "linear > relu",
}


@dataclass(frozen=True)
class PatternAtom:
    """One expanded step of a pattern path.

    Parameters
    ----------
    token:
        Canonical op-family name (matched case-insensitively against
        ``Op.func_name``) or module class name (matched exactly against the
        op's innermost module class).
    """

    token: str


@dataclass(frozen=True)
class PatternSpec:
    """One parsed, fully expanded user pattern (typed AST record).

    Parameters
    ----------
    name:
        User-declared pattern name (the chip label).
    atoms:
        Expanded linear path: references inlined, ``{k}`` sugar expanded.
    source:
        Original declaration text, for diagnostics.
    """

    name: str
    atoms: tuple[PatternAtom, ...]
    source: str


@dataclass(frozen=True)
class PatternMatch:
    """One matched region (typed record; the grammar-v2/chip-run-fold seam).

    Parameters
    ----------
    spec:
        The pattern that matched.
    op_labels:
        Pass-qualified member op labels in path order.
    """

    spec: PatternSpec
    op_labels: tuple[str, ...]


@dataclass
class PatternFoldReport:
    """Per-draw pattern folding disclosure (memo D11: counted, never silent).

    Parameters
    ----------
    folded:
        Accepted matches per pattern name.
    refused:
        Refused instance counts per pattern name, keyed by reason.
    unknown_tokens:
        Atom tokens matching nothing in the trace, with near-miss
        suggestions drawn from the trace's op families and module classes.
    no_match:
        Pattern names that matched zero sites.
    """

    folded: dict[str, list[PatternMatch]] = field(default_factory=dict)
    refused: dict[str, dict[str, int]] = field(default_factory=dict)
    unknown_tokens: dict[str, tuple[str, ...]] = field(default_factory=dict)
    no_match: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return the human-facing one-line-per-pattern disclosure."""

        lines = []
        names = sorted(set(self.folded) | set(self.refused) | set(self.no_match))
        for name in names:
            folded = len(self.folded.get(name, ()))
            refusals = self.refused.get(name, {})
            refused = sum(refusals.values())
            total = folded + refused
            line = f"{name}: {folded} of {total} sites folded"
            if refusals:
                reasons = "; ".join(
                    f"{count_} refused: {reason}" for reason, count_ in sorted(refusals.items())
                )
                line += f"; {reasons}"
            if name in self.no_match:
                line += " (no sites matched)"
            lines.append(line)
        return " | ".join(lines)


def parse_patterns(declarations: Mapping[str, str]) -> tuple[PatternSpec, ...]:
    """Parse pattern declarations into expanded typed specs.

    Declarations are processed in mapping order; an atom naming an earlier
    pattern inlines flat (acyclic name-inlining: later or self references
    are ordinary tokens, matched -- and near-miss-diagnosed -- against the
    trace).

    Raises
    ------
    InvalidArgumentError
        ``pattern_syntax_invalid`` for an empty path, a malformed atom, or
        a ``{k}`` repetition below 1; ``pattern_name_invalid`` for an empty
        name.
    """

    specs: list[PatternSpec] = []
    by_name: dict[str, PatternSpec] = {}
    for name, text in declarations.items():
        if not str(name).strip():
            raise InvalidArgumentError(
                f"pattern name must be non-empty; received {name!r}",
                code="pattern_name_invalid",
                remedy="declare patterns as {'Name': 'atom > atom > ...'}",
            )
        atoms: list[PatternAtom] = []
        tokens = [token.strip() for token in str(text).split(">")]
        if not any(tokens):
            raise InvalidArgumentError(
                f"pattern {name!r} has an empty path: {text!r}",
                code="pattern_syntax_invalid",
                remedy="declare a linear path such as 'conv2d > batch_norm > relu'",
            )
        for token in tokens:
            match = _ATOM_PATTERN.match(token)
            if match is None:
                raise InvalidArgumentError(
                    f"pattern {name!r} has a malformed atom {token!r}",
                    code="pattern_syntax_invalid",
                    remedy=(
                        "atoms are op-family or module-class names, optionally "
                        "with exact repetition: 'conv2d{3}'"
                    ),
                )
            base, repeat_text = match.group(1), match.group(2)
            repeat = int(repeat_text) if repeat_text else 1
            if repeat < 1:
                raise InvalidArgumentError(
                    f"pattern {name!r} repeats atom {base!r} {repeat} times",
                    code="pattern_syntax_invalid",
                    remedy="exact repetition {k} requires k >= 1",
                )
            referenced = by_name.get(base)
            for _ in range(repeat):
                if referenced is not None:
                    atoms.extend(referenced.atoms)
                else:
                    atoms.append(PatternAtom(token=base))
        spec = PatternSpec(name=str(name), atoms=tuple(atoms), source=str(text))
        specs.append(spec)
        by_name[spec.name] = spec
    return tuple(specs)


def resolve_pattern_request(
    fold_patterns: object,
) -> tuple[PatternSpec, ...]:
    """Resolve the public ``fold_patterns=`` value to parsed specs.

    Accepts ``None`` (off), the string ``"idiomatic"`` (curated preset), a
    mapping of declarations, or an already-parsed spec sequence.
    """

    if fold_patterns is None:
        return ()
    if fold_patterns == "idiomatic":
        return parse_patterns(IDIOMATIC_PATTERNS)
    if isinstance(fold_patterns, Mapping):
        return parse_patterns(fold_patterns)
    if isinstance(fold_patterns, Sequence) and all(
        isinstance(item, PatternSpec) for item in fold_patterns
    ):
        return tuple(fold_patterns)
    raise InvalidArgumentError(
        f"fold_patterns must be None, 'idiomatic', a name->path mapping, or "
        f"parsed PatternSpec records; received {type(fold_patterns).__name__}",
        code="pattern_request_invalid",
        remedy="pass fold_patterns={'ConvBnRelu': 'conv2d > batch_norm > relu'}",
    )


def _op_tokens(op: Op) -> set[str]:
    """Return the atom tokens ``op`` satisfies (op family + module class)."""

    tokens: set[str] = set()
    func_name = str(getattr(op, "func_name", "") or "").lower()
    if func_name:
        tokens.add(func_name)
    trace = op.trace
    for module_call in getattr(op, "modules", ()) or ():
        address = str(module_call).rsplit(":", 1)[0]
        try:
            module = trace.modules[address] if trace is not None else None
        except KeyError:
            module = None
        class_name = str(getattr(module, "class_name", "") or "")
        if class_name:
            tokens.add(class_name)
    return tokens


def _atom_matches(op: Op, atom: PatternAtom, token_cache: dict[str, set[str]]) -> bool:
    """Return whether ``op`` satisfies ``atom`` (case-preserving for classes)."""

    tokens = token_cache.get(op.label)
    if tokens is None:
        tokens = _op_tokens(op)
        token_cache[op.label] = tokens
    return atom.token in tokens or atom.token.lower() in tokens


def match_patterns(
    trace: Trace,
    context: RenderContext,
    specs: Sequence[PatternSpec],
) -> tuple[dict[str, SegmentDescriptor], PatternFoldReport]:
    """Match specs on the trace's executed op chain; return chips + report.

    Matching walks ops in execution order per spec, in declaration order
    (declaration order is priority; an op claimed by an earlier chip cannot
    join a later one -- counted as a refusal, never silent). Every accepted
    instance becomes one K4-family chip labeled
    ``PATTERN '<name>' -- N ops`` (K5 grammar: never ``(xN)``, never
    ``+N more``).
    """

    report = PatternFoldReport()
    segments: dict[str, SegmentDescriptor] = {}
    claimed: set[str] = set()
    token_cache: dict[str, set[str]] = {}
    ops = [cast("Op", op) for op in trace.ops]
    # Relationship collections hold bare labels while ``Op.label`` is
    # pass-qualified; index both spellings so extension can follow children.
    labels_index: dict[str, Op] = {}
    for op in ops:
        labels_index.setdefault(op.label, op)
        labels_index.setdefault(op.label.rsplit(":", 1)[0], op)
    trace_tokens: set[str] = set()
    for op in ops:
        trace_tokens.update(_op_tokens(op))
    for spec in specs:
        if not spec.atoms:
            continue
        for token in dict.fromkeys(
            atom.token for atom in spec.atoms if atom.token not in trace_tokens
        ):
            report.unknown_tokens[token] = tuple(
                difflib.get_close_matches(token, sorted(trace_tokens), n=3)
            )
        matches = _match_one_spec(spec, (ops, labels_index, claimed, token_cache), report)
        for index, match_record in enumerate(matches):
            descriptor = _pattern_descriptor(trace, context, match_record, index)
            segments[descriptor.name] = descriptor
    _warn_pattern_report(report)
    return segments, report


def _match_one_spec(
    spec: PatternSpec,
    scan: tuple[Sequence[Op], Mapping[str, Op], set[str], dict[str, set[str]]],
    report: PatternFoldReport,
) -> list[PatternMatch]:
    """Match one spec over the op chain; record folds/refusals on the report.

    Accepted instances claim their ops (declaration order is priority);
    every refusal is counted by reason -- never silent (memo D11).
    ``scan`` is the shared ``(ops, labels_index, claimed, token_cache)``
    matcher state (one tuple: the scan travels together).
    """

    ops, labels_index, claimed, token_cache = scan
    matches: list[PatternMatch] = []
    refused: dict[str, int] = {}
    for op in ops:
        if op.label in claimed or not _atom_matches(op, spec.atoms[0], token_cache):
            continue
        instance, reason = _extend_match(op, spec, labels_index, claimed, token_cache)
        if instance is None:
            if reason is not None:
                refused[reason] = refused.get(reason, 0) + 1
            continue
        accept_reason = _instance_honesty_refusal(instance, labels_index)
        if accept_reason is not None:
            refused[accept_reason] = refused.get(accept_reason, 0) + 1
            continue
        match_record = PatternMatch(spec=spec, op_labels=instance)
        matches.append(match_record)
        claimed.update(instance)
    if matches:
        report.folded[spec.name] = matches
    if refused:
        report.refused[spec.name] = refused
    if not matches:
        report.no_match.append(spec.name)
    return matches


def _extend_match(
    anchor: Op,
    spec: PatternSpec,
    labels_index: Mapping[str, Op],
    claimed: set[str],
    token_cache: dict[str, set[str]],
) -> tuple[tuple[str, ...] | None, str | None]:
    """Extend one anchor into a full linear match, or refuse with a reason.

    Extension follows dataflow children. More than one child satisfying the
    next atom is an equal-priority ambiguous tie and refuses THAT instance
    with a diagnostic (memo D11) -- silent tie-breaking in a naming feature
    teaches users their pattern applied where it did not.
    """

    members = [anchor.label]
    current = anchor
    for atom in spec.atoms[1:]:
        candidates = [
            labels_index[child]
            for child in current.children
            if child in labels_index
            and labels_index[child].label not in claimed
            and labels_index[child].label not in members
            and _atom_matches(labels_index[child], atom, token_cache)
        ]
        if not candidates:
            return None, None
        if len(candidates) > 1:
            return None, (
                f"ambiguous continuation at {current.label}: "
                f"{len(candidates)} children match '{atom.token}'"
            )
        current = candidates[0]
        members.append(current.label)
    return tuple(members), None


def _instance_honesty_refusal(
    instance: tuple[str, ...],
    labels_index: Mapping[str, Op],
) -> str | None:
    """Return the refusal reason for one candidate instance, or ``None``.

    The honesty core (memo D11): single entry, single exit -- any external
    edge touching the interior refuses the instance -- and protected
    landmark junctions inside the match refuse it.
    """

    member_set = set(instance)

    def is_external(label: str) -> bool:
        """Return whether ``label`` is a real dataflow node outside the match.

        Buffer version nodes (BatchNorm running stats and friends) are
        policy-hidden state reads, not dataflow the chip could misrepresent,
        so they never block a match.
        """

        op = labels_index.get(label)
        if op is None:
            return False
        if bool(getattr(op, "is_buffer", False)):
            return False
        return op.label not in member_set

    interior_reason = _interior_refusal(instance, labels_index, is_external)
    if interior_reason is not None:
        return interior_reason
    if len(instance) < 2:
        return None
    if any(is_external(child) for child in labels_index[instance[0]].children):
        return "an external edge leaves the interior"
    if any(is_external(parent) for parent in labels_index[instance[-1]].parents):
        return "an external edge enters the interior"
    return None


def _interior_refusal(
    instance: tuple[str, ...],
    labels_index: Mapping[str, Op],
    is_external: Callable[[str], bool],
) -> str | None:
    """Return the first interior honesty violation, or ``None`` (memo D11)."""

    for label in instance[1:-1]:
        op = labels_index[label]
        if str(getattr(op, "func_name", "") or "").lower() in JUNCTION_FUNC_NAMES:
            return "a protected landmark junction is inside the match"
        if any(is_external(parent) for parent in op.parents):
            return "an external edge enters the interior"
        if any(is_external(child) for child in op.children):
            return "an external edge leaves the interior"
    return None


def _pattern_descriptor(
    trace: Trace,
    context: RenderContext,
    match_record: PatternMatch,
    index: int,
) -> SegmentDescriptor:
    """Return the K4-family chip descriptor for one accepted instance."""

    base = _make_op_segment_descriptor(trace, context, match_record.op_labels)
    label = f"PATTERN '{match_record.spec.name}' -- {len(match_record.op_labels)} ops"
    spanned_note = base.label.split(" -- spans ", 1)
    if len(spanned_note) == 2:
        # Keep the R19-5 spanned-modules disclosure the base builder derived.
        label = f"{label} -- spans {spanned_note[1]}"
    return SegmentDescriptor(
        name=f"__pattern__{match_record.spec.name}_{index}__{base.name}",
        kind=base.kind,
        label=label,
        ops=base.ops,
        owner=base.owner,
        num_ops=base.num_ops,
        num_buffers=base.num_buffers,
        num_params=base.num_params,
    )


def _warn_pattern_report(report: PatternFoldReport) -> None:
    """Emit the counted per-draw pattern disclosures (never silent)."""

    if report.unknown_tokens:
        suggestions = "; ".join(
            f"{token!r}" + (f" (did you mean {', '.join(close)}?)" if close else "")
            for token, close in sorted(report.unknown_tokens.items())
        )
        warnings.warn(
            TorchLensWarning(
                f"pattern atoms matched nothing in this trace: {suggestions}",
                code="pattern_token_unknown",
            ),
            stacklevel=user_stacklevel(),
        )
    if report.no_match:
        warnings.warn(
            TorchLensWarning(
                "patterns matched zero sites: " + ", ".join(sorted(report.no_match)),
                code="pattern_no_match",
            ),
            stacklevel=user_stacklevel(),
        )
    if report.refused:
        warnings.warn(
            TorchLensWarning(
                f"pattern folding refused sites (honesty checks): {report.summary()}",
                code="pattern_fold_refusals",
            ),
            stacklevel=user_stacklevel(),
        )
