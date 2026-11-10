"""Behavioural sequences of the capture core, asserted at the base commit.

The sequences live in ``tools/field_diff_lib/sequences.py``. Their full
outcome (every result, error and warning, in order) is compared between two
commits by ``tools/field_differential.py compare``. This file asserts the part
of that outcome that is stated in words: what holds today and has to keep
holding, and the metadata cycles that exist today.

Each cycle is listed in ``CYCLES_AT_BASE`` with the ownership lane that has to
remove it. That lane deletes the entry in the commit that removes the cycle;
the test fails both when a listed cycle is gone and when a new one appears.

The sequences run once, in a process of their own, so that objects kept alive
by the test runner cannot change what reference counting frees.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "field_differential.py"

pytestmark = pytest.mark.slow

SEQUENCE_NAMES = (
    "1_read_orders",
    "2_mutate_before_read",
    "3_source_changes",
    "4_arguments",
    "5_forks",
    "6_serialisation",
    "7_facets_bulk",
    "8_release",
)

#: Release scenarios in which nothing taken from the trace stays referenced.
NOTHING_KEPT = (
    "no_record_taken",
    "records_taken_and_dropped",
    "all_fields_read",
    "facets_read",
    "to_pandas_read",
    "repr_read",
    "fork_dropped",
    "pickle_copy_dropped",
    "loaded_copy_dropped",
    "backward_capture",
    "backward_capture_all_fields_read",
)

#: Scenario -> (objects only the cycle collector frees today, lanes that remove the cycle, holder).
#: A scenario that is not listed has no such object. The lane named here deletes its entry.
CYCLES_AT_BASE: dict[str, tuple[frozenset[str], tuple[str, ...], str]] = {
    "all_fields_read": (
        frozenset({"buffer", "module", "module_call"}),
        ("B3b",),
        "module cells, module facet caches and buffer versions hold records",
    ),
    "backward_capture_all_fields_read": (
        frozenset({"module", "module_call"}),
        ("B3b",),
        "module cells and module facet caches hold records",
    ),
}

#: The lanes that may own a cycle, with what each removes.
OWNERSHIP_LANES = {
    "B1": "the op list and label dictionaries",
    "B2": "Layer state and accessors",
    "B3a": "the facet cache and gradient records",
    "B3b": "module cells and buffer versions",
    "B4": "fork translation and serialised state",
    "B5": "activation: records hold their trace",
}

#: Trace fields whose value depends on what was read before them, today.
READ_ORDER_DEPENDENT_AT_BASE = frozenset({"trace/annotations", "trace/health_facts"})

#: Bulk results that depend on what was read before them, today (the trace's printed summary).
READ_ORDER_DEPENDENT_TEXT_AT_BASE = frozenset({"/repr", "/str"})

COPIES = ("loaded", "pickled", "deepcopied")
DOORS = ("other_door", "fork", *COPIES)


def _plain(value: Any) -> Any:
    """Turn the canonical form of mappings and tuples back into plain containers."""

    if isinstance(value, list):
        return [_plain(member) for member in value]
    if not isinstance(value, dict):
        return value
    if "__dict__" in value and isinstance(value.get("items"), list):
        return _plain(
            {
                key if isinstance(key, str) else json.dumps(key, sort_keys=True): entry
                for key, entry in value["items"]
            }
        )
    if "__tuple__" in value:
        return _plain(value["__tuple__"])
    return {key: _plain(entry) for key, entry in value.items()}


def _raised(outcome: Any) -> str | None:
    """Return the error type a step raised, or ``None``."""

    return outcome.get("__raises__") if isinstance(outcome, dict) else None


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """Run every sequence once in a fresh process; return ``sequence -> its log``."""

    out = tmp_path_factory.mktemp("sequences")
    command = [sys.executable, str(TOOL), "dump", "--out", str(out), "--what", "sequences"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return {
        name: json.loads((out / f"sequence-{name}.json").read_text()) for name in SEQUENCE_NAMES
    }


@pytest.fixture(scope="module")
def steps(logs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return ``sequence -> step name -> outcome`` with plain containers."""

    return {
        name: {step["step"]: _plain(step["outcome"]) for step in log["steps"]}
        for name, log in logs.items()
    }


def test_every_sequence_runs_to_its_end(logs: dict[str, dict[str, Any]]) -> None:
    """No sequence stops early: an error inside a step is an outcome, not a stop."""

    stopped = {
        name: log["sequence_raises"] for name, log in logs.items() if "sequence_raises" in log
    }
    assert stopped == {}
    assert all(log["steps"] for log in logs.values())


def test_read_order_moves_only_the_known_fields(steps: dict[str, dict[str, Any]]) -> None:
    """Reading one field family first, last or after another moves no other field's value."""

    moved = {name: set(found) for name, found in steps["1_read_orders"].items() if ".moved" in name}
    assert len(moved) == 21
    unexpected = {
        name: sorted(found - READ_ORDER_DEPENDENT_AT_BASE)
        for name, found in moved.items()
        if found - READ_ORDER_DEPENDENT_AT_BASE
    }
    assert unexpected == {}


def test_overrides_and_annotations_survive_every_door(steps: dict[str, dict[str, Any]]) -> None:
    """A value set before the first read is the value read later, through each door and copy."""

    found = steps["2_mutate_before_read"]
    for door in (*DOORS, "plain_door"):
        assert found[f"{door}.op.func_name"] == "H3_OVERRIDE", door
        assert found[f"{door}.layer.func_name"] == "H3_LAYER_OVERRIDE", door
    assert found["list_door.op.func_name"] == "H3_OVERRIDE"
    for door in ("other_door", *COPIES):
        assert found[f"{door}.trace.name"] == "h3_renamed", door
    for door in DOORS:
        assert found[f"{door}.op.annotations"] == {"user": {"data": {"h3": 1}}}, door


def test_deleted_op_field_stays_deleted_in_the_live_trace(
    steps: dict[str, dict[str, Any]],
) -> None:
    """A deleted op field cannot be deleted twice and stays deleted through another door."""

    found = steps["2_mutate_before_read"]
    assert _raised(found["op.delete"]) is None
    for step in ("op.delete_again", "op.read_deleted", "layer.delete_again"):
        assert _raised(found[step]) == "AttributeError", step
    for door in ("other_door", "fork"):
        assert _raised(found[f"{door}.op.deleted_shape"]) == "AttributeError", door


def test_deletion_outcomes_that_differ_by_door_today(steps: dict[str, dict[str, Any]]) -> None:
    """Pin what deletion does today where the doors disagree (lane B2 moves the tombstones).

    A deleted layer field reads its value again through every door, and a
    deleted op field reads as an empty tuple in a loaded, pickled or copied
    trace. Neither may change without the saved-format and public-value rules
    being consulted.
    """

    found = steps["2_mutate_before_read"]
    assert _raised(found["layer.delete"]) is None
    assert found["layer.read_deleted"] == [6, 4]
    for door in DOORS:
        assert found[f"{door}.layer.deleted_shape"] == [6, 4], door
    for door in COPIES:
        assert found[f"{door}.op.deleted_shape"] == [], door


def test_new_attributes(steps: dict[str, dict[str, Any]]) -> None:
    """An op refuses a new attribute; a layer and a trace take one, and a save then refuses."""

    found = steps["2_mutate_before_read"]
    for step in ("new.op.set", "new.op.read", "new.op.read_other_door"):
        assert _raised(found[step]) == "AttributeError", step
    for step in ("new.layer.read", "new.trace.read", "new.layer.read_other_door"):
        assert found[step] == "h3 note", step
    assert found["new.fork.layer.read"] == "h3 note"
    assert found["new.fork.trace.read"] == "h3 note"
    assert _raised(found["new.save"]) == "ArtifactError"


def test_source_change_after_capture_reaches_only_the_code_context(
    steps: dict[str, dict[str, Any]],
) -> None:
    """Editing or deleting the model's file after capture changes no field but ``code_context``."""

    found = steps["3_source_changes"]

    def fields(state: str) -> dict[str, Any]:
        read = found[f"{state}.source_fields"]
        assert _raised(read) is None, state
        return {key: value for key, value in read.items() if not key.endswith(".source_trace")}

    unchanged = fields("unchanged")
    assert unchanged
    for state in ("edited", "deleted"):
        changed = fields(state)
        assert set(changed) == set(unchanged), state
        differing = sorted(key for key in unchanged if unchanged[key] != changed[key])
        assert all(key.endswith(".code_context") for key in differing), (state, differing)


def test_saved_arguments_are_the_ones_passed_at_the_call(
    steps: dict[str, dict[str, Any]],
) -> None:
    """A list mutated after the call returns is saved as it was passed."""

    found = steps["4_arguments"]
    assert found["save_args=False.arguments"]["ops[cat_1_2:1].has_saved_args"] is False
    saved = found["save_args=True.arguments"]
    assert saved["ops[cat_1_2:1].has_saved_args"] is True
    passed = saved["ops[cat_1_2:1].saved_args"][0]
    assert [member.get("__tensor__") for member in passed] == ["Tensor", "Tensor"]
    assert saved["ops[cat_1_2:1].saved_kwargs"] == {"dim": 0}


def test_forks_and_copies_run_cold_and_warm(steps: dict[str, dict[str, Any]]) -> None:
    """Forks, edits, reruns and every copy work on a cold trace and on a fully read one."""

    for sequence in ("5_forks", "6_serialisation"):
        raised = {name: _raised(out) for name, out in steps[sequence].items() if _raised(out)}
        assert raised == {}, sequence
    for copy in COPIES:
        assert steps["6_serialisation"][f"cold_vs_warm.{copy}"] == {}, copy


def test_bulk_readers_and_field_reads_do_not_change_each_other(
    steps: dict[str, dict[str, Any]],
) -> None:
    """Tables and facets read the same before and after single-field reads; no reader refuses."""

    found = steps["7_facets_bulk"]
    assert found["bulk.refusals"] == {}
    assert set(found["bulk.before_vs_after_field_reads"]) <= READ_ORDER_DEPENDENT_TEXT_AT_BASE
    assert found["fields.before_vs_after_bulk"] == {}


def test_a_trace_with_no_record_held_is_freed_by_reference_counting(
    steps: dict[str, dict[str, Any]],
) -> None:
    """With the collector off, dropping the trace frees it whenever no record is kept."""

    found = steps["8_release"]
    for scenario in NOTHING_KEPT:
        assert "trace" in found[scenario]["freed_when_trace_dropped"], scenario
    for scenario, outcome in found.items():
        if scenario != "chained_read":
            assert outcome["never_freed"] == [], scenario


def test_kept_records_are_freed_when_dropped(steps: dict[str, dict[str, Any]]) -> None:
    """Records kept past the trace die by reference counting when they are dropped.

    A kept ``Op`` or ``Layer`` keeps its trace and the saved activations, so
    those are freed at the same moment and not when the trace variable goes.
    """

    found = steps["8_release"]
    every = {"activation", "buffer", "layer", "module", "module_call", "op", "param", "trace"}
    for scenario in ("op_held", "layer_held"):
        kept = set(found[scenario]["freed_when_kept_dropped"])
        assert {"activation", "trace"} <= kept, scenario
        assert found[scenario]["freed_only_by_collector"] == [], scenario
    for scenario in ("every_door_held", "every_door_held_reversed"):
        assert set(found[scenario]["freed_when_kept_dropped"]) == every, scenario
        assert found[scenario]["freed_only_by_collector"] == [], scenario


def test_metadata_cycles_are_the_listed_ones(steps: dict[str, dict[str, Any]]) -> None:
    """Only the collector frees exactly the objects listed in ``CYCLES_AT_BASE``.

    A lane that removes a cycle deletes its entry; an object that appears here
    without an entry is a new cycle.
    """

    found = steps["8_release"]
    measured = {
        scenario: frozenset(outcome["freed_only_by_collector"])
        for scenario, outcome in found.items()
        if scenario != "chained_read" and outcome["freed_only_by_collector"]
    }
    expected = {scenario: names for scenario, (names, _, _) in CYCLES_AT_BASE.items()}
    assert measured == expected
    for scenario, (_, lanes, holder) in CYCLES_AT_BASE.items():
        assert lanes and set(lanes) <= set(OWNERSHIP_LANES), scenario
        assert holder, scenario


def test_chained_read_returns_the_saved_tensor(steps: dict[str, dict[str, Any]]) -> None:
    """A record read from a trace nobody else holds keeps that trace, so ``.out`` is the tensor."""

    outcome = steps["8_release"]["chained_read"]
    assert _raised(outcome) is None
    assert outcome["__tensor__"] == "Tensor"
