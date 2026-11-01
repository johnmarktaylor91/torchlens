"""Group-membership and tensor-role parse rules for ``collective_boundary_v1``.

Split from ``_boundary_rules`` (which keeps the closed vocabularies, the shared
``_refuse`` constructor and the remaining rule groups); ``_evidence`` imports
these rule groups from here and runs them in the same fixed order. Shared
parameters follow ``_boundary_rules``: ``entry`` is one boundary journal entry,
``kind`` its validated kind, ``where`` the diagnostic message prefix and
``source`` the merge input origin carried on the refusal payload.
"""

from __future__ import annotations

from typing import Any

from ..distributed._ledger import membership_digest_for_ranks
from ._boundary_rules import _ROLE_NAMES, _is_strict_int, _refuse
from ._enums import TENSORLESS_KINDS

__all__ = ("check_group_coherence", "check_group_membership", "check_roles")


def check_group_membership(entry: dict[str, Any], where: str, source: str) -> list[Any]:
    """Check the group record's rank list and this rank's place in it.

    Returns
    -------
    list[Any]
        The recorded ``global_ranks``: distinct integers containing
        ``my_global_rank``.

    Raises
    ------
    MergeInputError
        On a missing group record or rank list, a missing ``my_global_rank``,
        a non-integer or duplicate member, or a rank outside its own group.
    """

    group = entry.get("group")
    if not isinstance(group, dict) or not isinstance(group.get("global_ranks"), list):
        raise _refuse(f"{where} has no group membership record", source=source)
    if not isinstance(group.get("my_global_rank"), int):
        raise _refuse(f"{where} has no my_global_rank", source=source)
    global_ranks: list[Any] = group["global_ranks"]
    if any(not _is_strict_int(rank) for rank in global_ranks):
        raise _refuse(f"{where} group membership contains a non-integer rank", source=source)
    if len(set(global_ranks)) != len(global_ranks):
        raise _refuse(f"{where} group membership contains duplicate ranks", source=source)
    if group["my_global_rank"] not in global_ranks:
        raise _refuse(
            f"{where} claims rank {group['my_global_rank']} outside its group membership",
            source=source,
        )
    return global_ranks


def check_group_coherence(
    group: dict[str, Any],
    global_ranks: list[Any],
    correlation: dict[str, Any],
    where: str,
    source: str,
) -> None:
    """Check that the group record's redundant fields agree with its rank list.

    ``my_group_rank``, ``size`` and the correlation key's membership digest
    are each derivable from ``global_ranks``; a value that disagrees is a
    second, contradictory membership claim.

    Parameters
    ----------
    group:
        The entry's group record, already shape-checked by
        :func:`check_group_membership`.
    global_ranks:
        The recorded member ranks.
    correlation:
        The entry's correlation key, already shape-checked.
    where, source:
        The diagnostic message prefix and the merge input origin.

    Raises
    ------
    MergeInputError
        On a non-integer or mispositioned ``my_group_rank``, a ``size`` other
        than the member count, a non-string backend, or a membership digest
        bound to a different rank list.
    """

    _check_group_rank(group, global_ranks, where, source)
    # ``size`` is written by the recorder as len(global_ranks) and was never
    # validated: a forged size is a second, contradictory membership claim.
    size = group.get("size")
    if not _is_strict_int(size) or size != len(global_ranks):
        raise _refuse(
            f"{where} group size {size!r} does not equal its recorded "
            f"membership of {len(global_ranks)} rank(s)",
            source=source,
        )
    backend = group.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise _refuse(f"{where} group backend is not a string or null", source=source)
    _check_membership_digest_binding(correlation, global_ranks, where, source)


def _check_group_rank(
    group: dict[str, Any], global_ranks: list[Any], where: str, source: str
) -> None:
    """Check that ``my_group_rank`` is null or this rank's position in ``global_ranks``.

    Parameters
    ----------
    group:
        The entry's shape-checked group record.
    global_ranks:
        The recorded member ranks.
    where, source:
        The diagnostic message prefix and the merge input origin.

    Raises
    ------
    MergeInputError
        On a non-integer ``my_group_rank``, or one that is not the position
        of ``my_global_rank`` in the recorded rank list.
    """

    my_group_rank = group.get("my_group_rank")
    if my_group_rank is not None and not _is_strict_int(my_group_rank):
        raise _refuse(f"{where} my_group_rank is not an integer or null", source=source)
    # The group rank is definitionally the position of my_global_rank in the
    # recorded (c10d-ordered) rank list -- the writer reads it from
    # ``dist.get_group_rank`` over the same group whose
    # ``get_process_group_ranks`` list it records. A permuted or out-of-range
    # value rebinds this rank's slice pairings in the gather/scatter/all_to_all
    # witness derivation (and used to reach the engine's list indexing as a raw
    # IndexError), so incoherence is tamper, never honest evidence (R18-2).
    if my_group_rank is not None and my_group_rank != global_ranks.index(group["my_global_rank"]):
        raise _refuse(
            f"{where} my_group_rank {my_group_rank} does not equal the position "
            f"of my_global_rank {group['my_global_rank']} in its own recorded "
            f"group membership {global_ranks}",
            source=source,
        )


def _check_membership_digest_binding(
    correlation: dict[str, Any], global_ranks: list[Any], where: str, source: str
) -> None:
    """Check that the correlation key's membership digest names this rank list.

    Parameters
    ----------
    correlation:
        The entry's shape-checked correlation key.
    global_ranks:
        The recorded member ranks.
    where, source:
        The diagnostic message prefix and the merge input origin.

    Raises
    ------
    MergeInputError
        When the digest is not ``membership_digest_for_ranks(global_ranks)``.
    """

    # The membership digest is definitionally sha256(sorted(global_ranks)) and
    # freely recomputable. A digest bound to a DIFFERENT membership rebinds this
    # boundary's correlation joins, lifetime ordinals, and pre-join audit row to
    # another communicator while the presence/relation checks keep reading the
    # rank list -- the two views are attacker-separable unless tied here.
    if correlation["membership_digest"] != membership_digest_for_ranks(global_ranks):
        raise _refuse(
            f"{where} membership_digest does not equal the digest of its own "
            f"recorded group membership {sorted(int(r) for r in global_ranks)}",
            source=source,
        )


def check_roles(entry: dict[str, Any], kind: Any, where: str, source: str) -> None:
    """Check the tensor-role list: cardinality against the kind, then each role.

    Raises
    ------
    MergeInputError
        On a missing roles list, an empty list on a tensor-carrying kind, or
        the first malformed role entry (see :func:`_check_role_entry`).
    """

    roles = entry.get("roles")
    if not isinstance(roles, list):
        raise _refuse(f"{where} has no roles list", source=source)
    # Role cardinality vs boundary kind (b6-opus-R18-1): deleting the roles
    # record from EVERY member of a join used to vacuously satisfy the
    # set-of-shapes agreement checks (asymmetric deletion was caught; uniform
    # corruption -- the merge threat model -- was the escape). Every
    # tensor-carrying kind records at least one role on a successful call, so
    # an empty record refuses here, at the one chokepoint merge time and load
    # rederivation share.
    if not roles and kind not in TENSORLESS_KINDS:
        raise _refuse(
            f"{where} is a tensor-carrying {kind} boundary with zero tensor "
            "roles; a successful collective of this kind always records at "
            "least one role, so an empty or deleted roles record is not "
            "honest evidence",
            source=source,
        )
    seen_role_indexes: dict[str, set[int]] = {}
    for role_index, role in enumerate(roles):
        _check_role_entry(role, role_index, seen_role_indexes, where, source)


def _check_role_entry(
    role: Any,
    role_index: int,
    seen_role_indexes: dict[str, set[int]],
    where: str,
    source: str,
) -> None:
    """Check one role entry's name, shape and per-role-unique tensor position.

    Parameters
    ----------
    role:
        One element of the entry's roles list.
    role_index:
        Its position in that list (diagnostic).
    seen_role_indexes:
        Tensor positions already seen per exact role name; updated in place.
    where, source:
        The diagnostic message prefix and the merge input origin.

    Raises
    ------
    MergeInputError
        On a non-mapping role, a role name outside the closed vocabulary, a
        shape that is not a list of non-negative integers, a missing or
        negative index, or an index repeated within one role name.
    """

    if not isinstance(role, dict):
        raise _refuse(f"{where} role entry {role_index} is not a mapping", source=source)
    if role.get("role") not in _ROLE_NAMES:
        raise _refuse(
            f"{where} role entry {role_index} has role {role.get('role')!r} "
            "outside the closed vocabulary",
            source=source,
        )
    shape = role.get("shape")
    if not isinstance(shape, list) or any(not _is_strict_int(dim) or dim < 0 for dim in shape):
        raise _refuse(
            f"{where} role entry {role_index} has no well-formed shape "
            "(a list of non-negative integers)",
            source=source,
        )
    _check_role_index(role, role_index, seen_role_indexes, where, source)


def _check_role_index(
    role: dict[str, Any],
    role_index: int,
    seen_role_indexes: dict[str, set[int]],
    where: str,
    source: str,
) -> None:
    """Check one role entry's tensor position: non-negative and unique per role name.

    Parameters
    ----------
    role:
        One role entry, already checked to be a mapping with a known name.
    role_index:
        Its position in the roles list (diagnostic).
    seen_role_indexes:
        Tensor positions already seen per exact role name; updated in place.
    where, source:
        The diagnostic message prefix and the merge input origin.

    Raises
    ------
    MergeInputError
        On a missing or negative index, or an index repeated within one role
        name.
    """

    # ``index`` is the recorder's tensor POSITION (input position for
    # contribution/contribution_destination, output position for
    # destination). Positions are unique per exact role name; they are NOT
    # required dense 0..n-1 because a tensor serving as both contribution
    # and destination keeps its input position and leaves a hole in the
    # destination positions -- an honest recorder shape (R18-1 ii).
    index_value = role.get("index")
    if not _is_strict_int(index_value) or index_value < 0:
        raise _refuse(
            f"{where} role entry {role_index} has no non-negative integer index",
            source=source,
        )
    used = seen_role_indexes.setdefault(str(role.get("role")), set())
    if index_value in used:
        raise _refuse(
            f"{where} role entry {role_index} duplicates index {index_value} "
            f"within role {role.get('role')!r}; the recorder emits each "
            "tensor position at most once per role",
            source=source,
        )
    used.add(index_value)
