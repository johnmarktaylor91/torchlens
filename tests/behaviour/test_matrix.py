"""The behaviour net: cases times feature pairs, one test per oracle.

Cell universe (see ``README.md``):

* the cheap corpus cases (the manifest's smoke tier) times the core pairs,
  unmarked (mid tier);
* every pair of the 21 features, spread round-robin over the cheap cases so
  each pair runs on one real model per session, ``slow``;
* every other case (the rest of the corpus and the toy models) times the core
  pairs, ``slow``.

Test ids are ``case-featureA-featureB``. A cell passes when each oracle
passes, is a typed refusal or is not applicable by design; a failure is a
capture, validation or persistence finding, never something to loosen.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from tests.behaviour.cases import Case, _draws_random_numbers, all_cases, smoke_cases
from tests.behaviour.features import FEATURES, Feature, feature
from tests.behaviour.harness import BASE_ORACLES, CellResult, cell_id

MATRIX_PATH = Path(__file__).resolve().parents[1] / "interaction_matrix.json"

#: Pairs every case runs: one per door the net must keep honest (selective
#: capture with lookup, rerun with export, a capture edit with a post-hoc edit,
#: disk payloads with rendering, the sparse recorder with capture-time reporting,
#: bundles with saved-scope validation). Each composes or is independent in
#: the interaction matrix.
CORE_PAIRS: tuple[tuple[str, str], ...] = (
    ("save", "index"),
    ("rerun", "export"),
    ("intervene", "fork_do"),
    ("storage", "draw"),
    ("record", "reporting"),
    ("bundle", "validate"),
)

Cell = tuple[Case, tuple[Feature, ...]]


def _pairs(names: Iterable[tuple[str, str]]) -> list[tuple[Feature, ...]]:
    return [(feature(a), feature(b)) for a, b in names]


def _all_pairs() -> list[tuple[Feature, ...]]:
    return [tuple(pair) for pair in itertools.combinations(FEATURES, 2)]


def _cells(cases: Iterable[Case], pairs: Iterable[tuple[Feature, ...]]) -> Iterator[Cell]:
    for case in cases:
        for pair in pairs:
            yield case, pair


def _sweep_cells() -> Iterator[Cell]:
    """Every feature pair once, round-robin over the cheap cases, core pairs excluded."""

    cheap = smoke_cases()
    core = {tuple(sorted(pair)) for pair in CORE_PAIRS}
    for index, pair in enumerate(_all_pairs()):
        if tuple(sorted(f.name for f in pair)) in core:
            continue
        yield cheap[index % len(cheap)], pair


def _param(cell: Cell, *marks: pytest.MarkDecorator) -> object:
    """One cell as a pytest param: its id, its tier marks and its xdist group."""

    case, pair = cell
    name = cell_id(case, pair)
    return pytest.param(cell, id=name, marks=[*marks, pytest.mark.xdist_group(name=name)])


#: ``TORCHLENS_BEHAVIOUR_UNIVERSE`` narrows the universe for a targeted run:
#: ``cheap`` (the mid tier), ``sweep`` (every pair once), ``core`` (the core
#: pairs over every other case) or ``all`` (default).
UNIVERSE_PARTS = ("cheap", "sweep", "core")


def _universe() -> list[object]:
    wanted = os.environ.get("TORCHLENS_BEHAVIOUR_UNIVERSE", "all")
    parts = set(UNIVERSE_PARTS) if wanted == "all" else {wanted}
    core = _pairs(CORE_PAIRS)
    cheap = smoke_cases()
    rest = [case for case in all_cases() if case not in cheap]
    params: list[object] = []
    if "cheap" in parts:
        params += [_param(cell) for cell in _cells(cheap, core)]
    if "sweep" in parts:
        params += [_param(cell, pytest.mark.slow) for cell in _sweep_cells()]
    if "core" in parts:
        params += [_param(cell, pytest.mark.slow) for cell in _cells(rest, core)]
    return params


UNIVERSE = _universe()


def _assert_ok(result: CellResult, oracle: str) -> None:
    outcome = result.oracle(oracle)
    assert outcome.ok, f"{result.cell_id} {oracle}: {outcome.status}: {outcome.detail}"


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_trace(cell_result: CellResult) -> None:
    """Capture finishes with ops and the outcome the features declare."""

    _assert_ok(cell_result, "trace")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_validate(cell_result: CellResult) -> None:
    """Default forward validation with metadata invariants passes."""

    _assert_ok(cell_result, "validate")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_save(cell_result: CellResult) -> None:
    """A portable save succeeds."""

    _assert_ok(cell_result, "save")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_load(cell_result: CellResult) -> None:
    """The saved artifact loads."""

    _assert_ok(cell_result, "load")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_equality(cell_result: CellResult) -> None:
    """The loaded trace equals the live one through the public surface."""

    _assert_ok(cell_result, "equality")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_forward_match(cell_result: CellResult) -> None:
    """The trace's output equals a plain forward (or the replay a feature declares)."""

    _assert_ok(cell_result, "forward_match")


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_feature_checks(cell_result: CellResult) -> None:
    """Each active feature's own promise holds on the live and loaded traces."""

    checks = [name for name in cell_result.outcomes if name.startswith("feature:")]
    for name in checks:
        _assert_ok(cell_result, name)


def test_feature_axis_matches_interaction_matrix() -> None:
    """The net's features are exactly the interaction matrix's features."""

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert {f.name for f in FEATURES} == set(matrix["features"])
    assert len(FEATURES) == 21


def test_core_pairs_are_admitted_by_the_matrix() -> None:
    """Every core pair composes or is independent in the interaction matrix."""

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    status = {tuple(sorted((c["a"], c["b"]))): c["status"] for c in matrix["cells"]}
    for pair in CORE_PAIRS:
        assert status[tuple(sorted(pair))] in {"composes", "independent"}, pair


def test_oracles_are_the_six_the_net_promises() -> None:
    """The base oracle list is the one the README documents."""

    assert BASE_ORACLES == ("trace", "validate", "save", "load", "equality", "forward_match")


def test_stochastic_tag_names_samplers_not_training_gated_ops() -> None:
    """The derived tag holds for samplers that always draw and not for eval-deterministic ops."""

    assert _draws_random_numbers("func:rand")  # hand list
    assert _draws_random_numbers("func:exponential_")  # Gumbel-softmax noise
    assert _draws_random_numbers("func:uniform_")
    for gated in (
        "func:lstm",
        "func:gru",
        "func:rnn_tanh",
        "func:dropout_",
        "func:feature_dropout",
    ):
        assert not _draws_random_numbers(gated), gated
    assert not _draws_random_numbers("func:relu")
    assert not _draws_random_numbers("dtype:float32")
