"""Full validation of NON-identity edited forks (AUD-CODE 2.9).

``validate_forward_pass`` on ANY pushed fork with ``save_arg_values=True``
false-FAILED at the first downstream op (``argument_logging: parent not
logged as an argument``) because the argument-logging check compared the
child's capture-time ``saved_args`` snapshot against the PUSHED
``parent.out``. Consequently the tier-(ii) ``edge_intervention_boundary``
check was reachable in a passing validation only for IDENTITY edits.

The fix records the capture-time content digest of every site's out the
FIRST time the replay engine overwrites it and answers the same equality
question against that digest. The tests below pin that the check is
STILL ARMED on forks (tampered snapshots and swapped positions fail), not
skipped.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.replay import REPLAY_CAPTURE_DIGESTS_KEY
from torchlens.validation._edge_boundary import _check_edge_intervention_boundary

pytestmark = pytest.mark.smoke


class _Two(nn.Module):
    """Two branches joined by an add: the add has two distinct parents."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.fc1(x))
        b = torch.tanh(self.fc2(x))
        return torch.add(a, b) * 2.0


@pytest.fixture(scope="module")
def two():
    torch.manual_seed(0)
    model = _Two().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _parts(model: _Two, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return torch.relu(model.fc1(x)), torch.tanh(model.fc2(x))


def _add_op(trace: tl.Trace):
    return next(op for op in trace.layer_list if op.func_name == "add")


def _edge(trace: tl.Trace, position: int):
    add = _add_op(trace)
    return next(
        e
        for e in trace.edges
        if e.child_label in (add.label, add.layer_label) and e.arg_path == (position,)
    )


def _validate(fork: tl.Trace, ground_truth: torch.Tensor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fork.validate_forward_pass(ground_truth)


def _last_failure(fork: tl.Trace):
    failure = getattr(fork, "_last_validation_failure", None)
    return None if failure is None else (failure.check, failure.op_label)


def test_act_fork_validates_against_hand_truth(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    assert _validate(fork, 2.0 * b) is True
    assert _validate(fork, model(x)) is False  # the wrong claim still fails
    assert _last_failure(fork) == ("ground_truth", "output_1")


def test_non_identity_edge_fork_validates_and_reaches_boundary(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(_edge(trace, 0).__selection__(), tl.zero_ablate())
    assert _validate(fork, 2.0 * b) is True
    add_label = _add_op(trace).label
    reached = [
        d
        for d in fork.validation_replay_status.decisions
        if d.get("decision") == "edge_intervention_boundary" and d.get("op_label") == add_label
    ]
    assert reached, "boundary check must be reached in a PASSING validation of a real edit"


def test_param_fork_validates_against_hand_truth(two) -> None:
    model, x, trace = two
    fork = trace.fork()
    fork.do(tl.params("fc2.bias"), tl.zero_ablate())
    with torch.no_grad():
        expected = 2.0 * (torch.relu(model.fc1(x)) + torch.tanh(x @ model.fc2.weight.T))
    assert _validate(fork, expected) is True


def test_tampered_snapshot_at_recomputed_parent_slot_fails(two) -> None:
    """The digest path is a REAL comparison: a corrupted saved_args snapshot fails."""

    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    child = fork[_add_op(trace).label].ops[0]
    saved = list(child.saved_args)
    saved[0] = saved[0] + 1.0
    child._internal_set("saved_args", saved)
    assert _validate(fork, 2.0 * b) is False
    assert _last_failure(fork) == ("argument_logging", child.label)


def test_swapped_parent_positions_on_recomputed_child_fails(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    child = fork[_add_op(trace).label].ops[0]
    positions = child.parent_arg_positions["args"]
    child._internal_set(
        "parent_arg_positions",
        {"args": {0: positions[1], 1: positions[0]}, "kwargs": {}},
    )
    assert _validate(fork, 2.0 * b) is False
    assert _last_failure(fork) == ("argument_logging", child.label)


def test_recomputed_parent_without_digest_is_unverified_never_validated(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    digests = dict(fork.last_run[REPLAY_CAPTURE_DIGESTS_KEY])
    assert "relu_1_2:1" in digests
    digests["relu_1_2:1"] = None  # recomputed, but no recordable capture digest
    fork.last_run[REPLAY_CAPTURE_DIGESTS_KEY] = digests
    result = _validate(fork, 2.0 * b)
    assert result is not True
    status = fork.validation_replay_status
    assert status.state == "unverified"
    assert "replay_recomputed_parent_unattested" in dict(status.unverified_reason_counts or {})


def test_stamp_digest_mismatch_fails_the_boundary_and_the_validation(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    edge = _edge(trace, 0)
    fork.do(edge.__selection__(), tl.zero_ablate())
    child = fork[_add_op(trace).label].ops[0]
    key = (edge.arg_kind, edge.arg_path)
    store = dict(child.edge_substitutions)
    store[key] = {**store[key], "value": store[key]["value"] + 1.0}
    child._internal_set("edge_substitutions", store)
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_stamp_mismatch"
    assert _validate(fork, 2.0 * b) is False


def test_source_trace_and_fork_of_pushed_fork_validate(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    grandchild = fork.fork()
    assert REPLAY_CAPTURE_DIGESTS_KEY in grandchild.last_run
    assert _validate(grandchild, 2.0 * b) is True
    assert _validate(trace, model(x)) is True
    assert REPLAY_CAPTURE_DIGESTS_KEY not in (trace.last_run or {})
