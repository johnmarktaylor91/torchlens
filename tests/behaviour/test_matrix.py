"""The behaviour net: cases times feature pairs, one test per cell.

Cell universe (see ``README.md``):

* the cheap corpus cases (the manifest's smoke tier) times the core pairs,
  ``heavy`` (each cell takes 1 to 7 s);
* every pair of the interaction-matrix features, spread round-robin over the cheap cases so
  each pair runs on one real model per session, ``slow``;
* every other case (the rest of the corpus and the toy models) times the core
  pairs, ``slow``.

Each cell is one test, ``test_cell[case-featureA-featureB]``, that judges every
oracle (the six base oracles, then each active feature's own check) and never
stops at the first unsatisfied one: a failing cell reports every failing oracle
and its reason in one message. A cell passes when each oracle passes, is a
typed refusal or is not applicable by design; a failure is a capture,
validation or persistence finding, never something to loosen.
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
from tests.behaviour.harness import BASE_ORACLES, CellResult, Outcome, cell_id

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


#: The smoke hook. Cell ids listed here run in the smoke tier instead of
#: ``heavy`` (the two tier marks are exclusive; ``tests/test_marker_lint.py``).
#: Today it holds the cheap cells the smoke-tier selection chose (the two
#: cheapest core pairs plus two more). To change it, list the ids and set
#: ``_SMOKE_TIER`` to the smoke marker spelled in this file (``pytest.mark``
#: followed by ``.smoke``): the smoke file list is a text scan of each test
#: file's own source, so the marker must be spelled here, never through an
#: alias in a conftest. Only cheap cells can join; an id the universe does not
#: hold, or one outside the cheap cells, raises at import.
SMOKE_CELLS: frozenset[str] = frozenset(
    {
        "Avalanche_multi-task_simple_MLP_compact-record-reporting",
        "Avalanche_multi-task_simple_MLP_compact-save-index",
        "BindsNET_McCullochPitts_(threshold_node)-record-reporting",
        "BindsNET_McCullochPitts_(threshold_node)-save-index",
        "BindsNET_McCullochPitts_(threshold_node)-storage-draw",
        "COMBO_(Conservative_Model-Based_Offline_RL)-record-reporting",
        "COMBO_(Conservative_Model-Based_Offline_RL)-save-index",
        "DIAYN_PolicyNetwork_(skill-conditioned_SAC_actor)-record-reporting",
        "DIAYN_PolicyNetwork_(skill-conditioned_SAC_actor)-save-index",
        "Exphormer-GPS-VirtualNode-record-reporting",
        "Exphormer-GPS-VirtualNode-save-index",
        "IQL_(Implicit_Q-Learning,_rlkit_actor)-record-reporting",
        "IQL_(Implicit_Q-Learning,_rlkit_actor)-save-index",
        "MemN2N_(End-To-End_Memory_Networks)-record-reporting",
        "MemN2N_(End-To-End_Memory_Networks)-save-index",
        "Neural_Gas-record-reporting",
        "Neural_Gas-save-index",
        "PufferLib_NMMO3_observation_encoder_(multi-hot_tile_CNN_+_player_embed)-record-reporting",
        "PufferLib_NMMO3_observation_encoder_(multi-hot_tile_CNN_+_player_embed)-save-index",
        "Sionna-NeuralReceiver-record-reporting",
        "Sionna-NeuralReceiver-save-index",
        "Tree_parity_machine-intervene-fork_do",
        "Tree_parity_machine-record-reporting",
        "Tree_parity_machine-save-index",
        "Wavelet_Kernel_Network_(WKN)-record-reporting",
        "Wavelet_Kernel_Network_(WKN)-save-index",
        "aenet-PyTorch-record-reporting",
        "aenet-PyTorch-save-index",
    }
)
_SMOKE_TIER: pytest.MarkDecorator | None = pytest.mark.smoke


def _param(cell: Cell, *marks: pytest.MarkDecorator) -> object:
    """One cell as a pytest param: its id, its tier marks and its xdist group."""

    case, pair = cell
    name = cell_id(case, pair)
    return pytest.param(cell, id=name, marks=[*marks, pytest.mark.xdist_group(name=name)])


def _cheap_tier(cell: Cell) -> pytest.MarkDecorator:
    """The tier mark of a cheap cell: ``heavy``, or smoke when the hook names it."""

    if _SMOKE_TIER is not None and cell_id(*cell) in SMOKE_CELLS:
        return _SMOKE_TIER
    return pytest.mark.heavy


#: ``TORCHLENS_BEHAVIOUR_UNIVERSE`` narrows the universe for a targeted run:
#: ``cheap`` (heavy), ``sweep`` (every pair once), ``core`` (the core
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
        cheap_cells = list(_cells(cheap, core))
        _check_smoke_cells({cell_id(*cell) for cell in cheap_cells})
        params += [_param(cell, _cheap_tier(cell)) for cell in cheap_cells]
    if "sweep" in parts:
        params += [_param(cell, pytest.mark.slow) for cell in _sweep_cells()]
    if "core" in parts:
        params += [_param(cell, pytest.mark.slow) for cell in _cells(rest, core)]
    return params


def _check_smoke_cells(cheap_ids: set[str]) -> None:
    """Refuse a smoke hook entry that names no cheap cell (a typo would drop it silently)."""

    unknown = sorted(SMOKE_CELLS - cheap_ids)
    if unknown:
        raise ValueError(f"SMOKE_CELLS names ids that are not cheap cells: {unknown[:5]}")


UNIVERSE = _universe()


@pytest.mark.parametrize("cell_result", UNIVERSE, indirect=True)
def test_cell(cell_result: CellResult) -> None:
    """Every oracle of the cell holds; a failure names each failing oracle.

    The base oracles: capture finishes with ops and the outcome the features
    declare (``trace``), default forward validation with metadata invariants
    passes (``validate``), a portable save succeeds (``save``), the artifact
    loads (``load``), the loaded trace equals the live one through the public
    surface (``equality``), and the output equals a plain forward or the replay
    a feature declares (``forward_match``). Then each active feature's own
    promise on the live and loaded traces (``feature:<name>``).
    """

    failures = cell_result.failures()
    assert not failures, f"{cell_result.cell_id}: {len(failures)} oracle(s) failed\n" + "\n".join(
        failures
    )


def test_feature_axis_matches_interaction_matrix() -> None:
    """The net's features are exactly the interaction matrix's features."""

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert {f.name for f in FEATURES} == set(matrix["features"])
    assert len(FEATURES) == len(matrix["features"])


def test_core_pairs_are_admitted_by_the_matrix() -> None:
    """Every core pair composes or is independent in the interaction matrix."""

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    status = {tuple(sorted((c["a"], c["b"]))): c["status"] for c in matrix["cells"]}
    for pair in CORE_PAIRS:
        assert status[tuple(sorted(pair))] in {"composes", "independent"}, pair


def test_oracles_are_the_six_the_net_promises() -> None:
    """The base oracle list is the one the README documents."""

    assert BASE_ORACLES == ("trace", "validate", "save", "load", "equality", "forward_match")


def test_cell_verdict_names_every_unsatisfied_oracle() -> None:
    """A cell's failures list every failing or unreached oracle and nothing satisfied."""

    result = CellResult(
        "toy-a-b",
        {
            "trace": Outcome("pass"),
            "validate": Outcome("fail", "replay differs"),
            "save": Outcome("refused", "typed", "TorchLensError", "code"),
            "load": Outcome("not_applicable", "by design"),
            "equality": Outcome("pass"),
            "feature:a": Outcome("fail", "promise broken"),
            "feature:b": Outcome("pass"),
        },
    )
    assert result.failures() == [
        "validate: fail: replay differs",
        "forward_match: not_run: the cell did not reach this oracle",
        "feature:a: fail: promise broken",
    ]


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
