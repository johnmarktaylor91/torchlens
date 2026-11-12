"""Step 5's per-bool work over plain lists: classification and event linking.

Both bodies of step 5 (the column algorithm in ``_conditional_columns`` and the
facade body in ``control_flow``) read the terminal bools' inputs first, call
these functions, and write the results back their own way. The functions read
no op: classification takes each bool's witnessed consumer locations and its
captured frames, linking takes the classifications and the dense events.

Bools with the same consumer locations and the same frame objects classify
identically (``ast_branches.classify_bool`` reads only the location and the
cached file index), so each distinct pair is classified once. The frames are
keyed by the identity of their ``FuncCallLocation`` objects and pinned for the
pass, so a key never outlives the frames it names.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from . import ast_branches

__all__ = (
    "BRANCH_CONTEXT_KINDS",
    "BoolClasses",
    "classify_terminal_bools",
    "link_bools_to_events",
)

#: Bool context kinds that take part in a conditional branch.
BRANCH_CONTEXT_KINDS = frozenset({"if_test", "elif_test", "ifexp"})


class BoolClasses(NamedTuple):
    """Phase 5a and 5b results for the terminal bools, in execution order."""

    #: Filename -> cached AST index (``None`` when unparseable), first-seen order.
    file_indexes: dict[str, ast_branches.FileIndex | None]
    #: Observed structural conditional keys, first-seen order.
    conditional_keys: list[ast_branches.ConditionalKey]
    #: Raw bool label -> its deduplicated branch-participating classifications.
    classifications: dict[str, list[ast_branches.BoolClassification]]
    #: Per bool: ``(context kind, wrapper kind, is a terminal conditional bool)``.
    outputs: list[tuple[str, str | None, bool]]


def classify_terminal_bools(
    bools: list[tuple[str, Any]],
    consumer_locations: dict[str, tuple[tuple[str, int], ...]],
) -> BoolClasses:
    """Index the files and classify every terminal bool (phases 5a and 5b).

    Parameters
    ----------
    bools:
        ``(raw label, captured frames)`` per terminal scalar bool, in execution order.
    consumer_locations:
        Raw bool label -> the user source locations that consumed it on the host.

    Returns
    -------
    BoolClasses
        The file indexes, observed keys, classifications and per-bool outputs.
    """

    file_indexes: dict[str, ast_branches.FileIndex | None] = {}
    ordered_keys: dict[ast_branches.ConditionalKey, None] = {}
    classifications: dict[str, list[ast_branches.BoolClassification]] = {}
    outputs: list[tuple[str, str | None, bool]] = []
    memo: dict[Any, tuple[ast_branches.BoolClassification, list[Any]]] = {}
    pinned: list[Any] = []  # keeps every keyed frame alive, so no id is reused
    for label, frames in bools:
        locations = consumer_locations[label] if label in consumer_locations else ()
        key = (locations, tuple(map(id, frames)))
        if key in memo:
            primary, branch = memo[key]
        else:
            pinned += [frames]
            _index_files(file_indexes, locations, frames)
            primary, branch = memo[key] = _classify_one(locations, frames)
        outputs += [(primary.kind, primary.wrapper_kind, bool(branch))]
        classifications[label] = branch
        for classification in branch:
            conditional_key = classification.conditional_key
            assert conditional_key is not None  # mypy narrowing
            if conditional_key not in ordered_keys:
                ordered_keys[conditional_key] = None
    return BoolClasses(file_indexes, list(ordered_keys), classifications, outputs)


def _index_files(
    file_indexes: dict[str, ast_branches.FileIndex | None],
    locations: tuple[tuple[str, int], ...],
    frames: Any,
) -> None:
    """Add the AST index of every consumer file, then every frame file, not yet seen."""

    for filename, _line_number in locations:
        if filename not in file_indexes:
            file_indexes[filename] = ast_branches.get_file_index(filename)
    for frame in frames:
        if frame.file not in file_indexes:
            file_indexes[frame.file] = ast_branches.get_file_index(frame.file)


def _classify_one(
    locations: tuple[tuple[str, int], ...], frames: Any
) -> tuple[ast_branches.BoolClassification, list[ast_branches.BoolClassification]]:
    """Return one bool's primary classification and its branch classifications.

    Every witnessed consumer location is classified, not just the first known
    one: a bool consumed by ``assert`` or ``while`` and later by an ``if`` test
    is still a conditional bool, and one bool gating several ``if`` statements
    yields one classification per gated conditional. The runtime frame fallback
    (column-precise) applies only when no consumer location produced a
    branch-participating classification.
    """

    observed: list[ast_branches.BoolClassification] = []
    for filename, line_number in locations:
        location_classification = ast_branches.classify_bool(filename, line_number, None)
        if location_classification.kind != "unknown":
            observed.append(location_classification)
    branch = _dedup_branch_classifications(observed)
    frame_classification: ast_branches.BoolClassification | None = None
    for frame in reversed(frames):
        candidate = ast_branches.classify_bool(frame.file, frame.line_number, frame.col_offset)
        if candidate.kind != "unknown":
            frame_classification = candidate
            break
    if not branch:
        if frame_classification is not None:
            observed.append(frame_classification)
            branch = _dedup_branch_classifications([frame_classification])
    elif (
        frame_classification is not None
        and frame_classification.kind in BRANCH_CONTEXT_KINDS
        and frame_classification.conditional_key is not None
        and frame_classification.conditional_key not in {c.conditional_key for c in branch}
    ):
        # Creation-site conflict: the bool op was created inside the test span of
        # one conditional while the witnessed consumer line attributes it to a
        # DIFFERENT conditional (for example a formatter-wrapped multi-line nested
        # ternary, where the interpreter reports the inner ternary's line for the
        # outer test's ``__bool__``). Linking either key could cross-wire a foreign
        # bool into a conditional's public record, so fail closed for this bool.
        observed = []
        branch = []
    if branch:
        return branch[0], branch
    if observed:
        return observed[0], branch
    return ast_branches.BoolClassification("unknown", None, None, None), branch


def _dedup_branch_classifications(
    classifications: list[ast_branches.BoolClassification],
) -> list[ast_branches.BoolClassification]:
    """Return the branch-participating classifications, deduplicated in order.

    Kept are classifications whose kind is branch-participating and whose
    conditional key is present, deduplicated by ``(conditional_key,
    branch_test_kind)`` preserving first-seen order.
    """

    deduplicated: list[ast_branches.BoolClassification] = []
    seen: set[tuple[ast_branches.ConditionalKey, str | None]] = set()
    for classification in classifications:
        if (
            classification.kind not in BRANCH_CONTEXT_KINDS
            or classification.conditional_key is None
        ):
            continue
        identity_key = (classification.conditional_key, classification.branch_test_kind)
        if identity_key in seen:
            continue
        seen.add(identity_key)
        deduplicated.append(classification)
    return deduplicated


def link_bools_to_events(
    classifications: dict[str, list[ast_branches.BoolClassification]],
    events_by_key: dict[ast_branches.ConditionalKey, Any],
) -> dict[str, int | None]:
    """Link every bool to each event it gates; return each bool's first event id.

    One bool may gate several conditionals (predicate reuse), so every
    branch-participating classification appends its bool to the matching
    event's ``bool_layers`` (and the index-aligned ``_bool_layers_raw``) at its
    first occurrence and records that index under the arm whose test it
    evaluated in ``_arm_bool_indices``, in first-occurrence order. The returned
    map is raw bool label -> the first linked event's id (the scalar
    ``terminal_conditional_id``) or ``None``; ``events_by_key`` maps each
    structural key to its dense phase-5c event. Cognitive complexity is over 15
    on purpose (once per bool): the table updates are inlined, so a repeated
    bool costs no call.
    """

    first_ids: dict[str, int | None] = {}
    tables: dict[int, tuple[dict[str, int], list[int], dict[str, Any]]] = {}
    for bool_label, branch in classifications.items():
        first_id: int | None = None
        for classification in branch:
            conditional_key = classification.conditional_key
            if conditional_key is None or conditional_key not in events_by_key:
                continue
            event = events_by_key[conditional_key]
            event_id = event.id
            if first_id is None:
                first_id = event_id
            if event_id not in tables:
                tables[event_id] = _event_table(event)
            positions, size, arm_seen = tables[event_id]
            if bool_label in positions:
                bool_index = positions[bool_label]
            else:
                bool_index = size[0]
                size[0] = bool_index + 1
                positions[bool_label] = bool_index
                bool_layers = event.bool_layers
                bool_layers += [bool_label]
                raw_layers = event._bool_layers_raw
                raw_layers += [bool_label]
            arm_kind = classification.branch_test_kind or "then"
            arm_indices_by_kind = event._arm_bool_indices
            if arm_kind not in arm_indices_by_kind:
                arm_indices_by_kind[arm_kind] = []
            arm_indices = arm_indices_by_kind[arm_kind]
            if arm_kind not in arm_seen:
                arm_seen[arm_kind] = dict.fromkeys(arm_indices)
            seen = arm_seen[arm_kind]
            if bool_index not in seen:
                seen[bool_index] = None
                arm_indices += [bool_index]
        first_ids[bool_label] = first_id
    return first_ids


def _event_table(event: Any) -> tuple[dict[str, int], list[int], dict[str, Any]]:
    """Return one event's ``(first index per bool, [size], seen indices per arm)`` tables."""

    positions: dict[str, int] = {}
    for index, label in enumerate(event.bool_layers):
        if label not in positions:
            positions[label] = index
    return positions, [len(event.bool_layers)], {}
