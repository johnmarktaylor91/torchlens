"""Brute-force recurrent-site oracle for parameter-free loop boundaries.

The expected partitions below are enumerated from source-level call sites and
loop iterations, independently of TorchLens grouping metadata.  The oracle
checks every pair of captured calls: two calls share a layer if and only if
their hand-enumerated ground-truth site identifiers match.  This is stronger
than comparing pass-count multisets, which cannot detect swapped or crossed
memberships.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import example_models
import pytest
import torch
import torch.nn as nn

from tests.support.structural_grouper import legacy_structural_grouper  # noqa: F401
from torchlens import trace as trace_fn
from torchlens.validation.invariants import check_metadata_invariants

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")


class _BoundaryOracleModel(nn.Module):
    """Execute one adversarial boundary program selected by name."""

    def __init__(self, case: str, first: int = 3, second: int = 2) -> None:
        """Initialize every persistent site used by the oracle programs.

        Parameters
        ----------
        case:
            Program selected by :meth:`forward`.
        first:
            First loop's iteration count.
        second:
            Second loop's iteration count.
        """
        super().__init__()
        self.case = case
        self.first = first
        self.second = second
        self.emb = nn.Linear(4, 4)
        self.shared = nn.Linear(4, 4)
        self.body = nn.Linear(4, 4)
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        self.x_site = nn.Linear(4, 4)
        self.y_site = nn.Linear(4, 4)
        self.chain_sites = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
        self.norm = nn.LayerNorm(4)
        self.norm_a = nn.LayerNorm(4)
        self.norm_b = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected adversarial program.

        Parameters
        ----------
        x:
            Four-feature input tensor.

        Returns
        -------
        torch.Tensor
            Program output.
        """
        if self.case == "unary_post_ln":
            state = self.emb(x)
            for _ in range(self.first):
                state = self.norm(torch.tanh(state))
            return state
        if self.case == "masked_scan":
            state = self.emb(x)
            mask = torch.sigmoid(x)
            for _ in range(self.first):
                state = self.body(state * mask) + state
            return state
        if self.case == "parallel_combiner":
            shared = torch.tanh(self.shared(x))
            first = shared
            for _ in range(self.first):
                first = torch.tanh(first + self.enc(first))
            second = shared
            for _ in range(self.second):
                second = torch.tanh(second + self.dec(second))
            return first + second
        if self.case == "residual_two_site":
            state = self.emb(x)
            for _ in range(self.first):
                state = state + self.attn(state)
                state = state + self.mlp(state)
            return state
        if self.case == "peel":
            state = self.emb(x)
            for _ in range(self.first):
                state = torch.tanh(state + self.enc(state))
            for _ in range(self.second):
                state = torch.tanh(state + self.dec(state))
            return state
        if self.case == "post_ln_tap":
            state = self.emb(x)
            tap = state
            for _ in range(self.first):
                tap = state + self.attn(state)
                state = self.norm(tap)
            for _ in range(self.second):
                tap = tap + self.dec(tap)
            return tap
        if self.case == "post_ln":
            state = self.emb(x)
            for _ in range(self.first):
                state = self.norm(state + self.attn(state))
            return state
        if self.case == "nested_outer":
            state = self.emb(x)
            for _ in range(self.first):
                for _ in range(self.second):
                    state = torch.tanh(state + self.enc(state))
                state = torch.tanh(state + self.dec(state))
            return state
        if self.case == "same_key_prelude":
            state = self.emb(x)
            prelude = state + self.attn(state)
            first = prelude
            second = prelude
            for _ in range(self.first):
                first = first + self.x_site(first)
            for _ in range(self.second):
                second = second + self.y_site(second)
            return first * second
        if self.case == "shared_flank_norm":
            state = self.emb(x)
            for _ in range(self.first):
                state = self.norm_a(state + self.attn(state))
            for _ in range(self.second):
                state = self.norm_b(state + self.attn(state))
            return state
        if self.case == "parallel_accumulators":
            backbone = self.emb(x)
            first = x
            second = x
            for _ in range(self.first):
                backbone = self.shared(backbone)
                first = first + backbone
                second = second + backbone
            return first * second
        if self.case == "albert":
            state = x
            for _ in range(self.first):
                state = torch.tanh(self.shared(state))
            for _ in range(self.second):
                state = torch.tanh(self.shared(state))
            return state
        if self.case == "distinct_submodule":
            state = x
            for _ in range(self.first):
                state = torch.tanh(self.enc(state))
            for _ in range(self.second):
                state = torch.tanh(self.dec(state))
            return state
        if self.case == "nested_shared":
            state = x
            for _ in range(self.first):
                for _ in range(self.second):
                    state = torch.tanh(self.shared(state) + self.body(state))
            return state
        if self.case == "chained":
            state = self.emb(x)
            counts = (self.first, self.second, 2)
            for site, iterations in zip(self.chain_sites, counts):
                for _ in range(iterations):
                    state = torch.tanh(state + site(state))
            return state
        raise ValueError(f"Unknown boundary oracle case: {self.case}")


class _RepeatedFusedRecurrent(nn.Module):
    """Call one fused recurrent module repeatedly with state feedback."""

    def __init__(self, kind: type[nn.RNNBase], calls: int) -> None:
        """Initialize the selected fused recurrent module.

        Parameters
        ----------
        kind:
            ``nn.RNN``, ``nn.GRU``, or ``nn.LSTM``.
        calls:
            Number of repeated calls.
        """
        super().__init__()
        self.recurrent = kind(4, 4, batch_first=True)
        self.calls = calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed each call's state into the next call.

        Parameters
        ----------
        x:
            Batched input sequence.

        Returns
        -------
        torch.Tensor
            Final sequence output.
        """
        state: Any = None
        for _ in range(self.calls):
            x, state = self.recurrent(x) if state is None else self.recurrent(x, state)
        return x


class _RepeatedCell(nn.Module):
    """Call one RNN/GRU/LSTM cell repeatedly for slot-partition checks."""

    def __init__(self, kind: str, calls: int) -> None:
        """Initialize the selected recurrent cell.

        Parameters
        ----------
        kind:
            ``rnncell``, ``grucell``, or ``lstmcell``.
        calls:
            Number of repeated cell calls.
        """
        super().__init__()
        cell_types = {
            "rnncell": nn.RNNCell,
            "grucell": nn.GRUCell,
            "lstmcell": nn.LSTMCell,
        }
        self.kind = kind
        self.calls = calls
        self.cell = cell_types[kind](4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cell loop and return its final hidden state.

        Parameters
        ----------
        x:
            Batched feature input.

        Returns
        -------
        torch.Tensor
            Final hidden state.
        """
        hidden = torch.zeros_like(x)
        if self.kind != "lstmcell":
            for _ in range(self.calls):
                hidden = self.cell(x, hidden)
            return hidden
        cell_state = torch.zeros_like(x)
        for _ in range(self.calls):
            hidden, cell_state = self.cell(x, (hidden, cell_state))
        return hidden + cell_state * 0


def _stem(layer_label: str) -> str:
    """Return the operation stem of a finalized layer label.

    Parameters
    ----------
    layer_label:
        Finalized TorchLens layer label.

    Returns
    -------
    str
        Operation stem without layer/pass suffixes.
    """
    return layer_label.rsplit("_", 2)[0]


def _ops_with_stem(traced: Any, stem: str) -> list[Any]:
    """Return captured ops of one stem in raw execution order.

    Parameters
    ----------
    traced:
        Captured TorchLens trace.
    stem:
        Operation stem to retain.

    Returns
    -------
    list[Any]
        Matching operations in trace order.
    """
    return [op for op in traced if _stem(op.layer_label) == stem]


def _assert_brute_force_partition(
    traced: Any,
    stem: str,
    truth_sites: Sequence[Hashable],
) -> None:
    """Compare every call pair against an independently enumerated partition.

    Parameters
    ----------
    traced:
        Captured TorchLens trace.
    stem:
        Operation stem whose partition should be checked.
    truth_sites:
        Source-level recurrent-site identifier for each call in execution order.

    Returns
    -------
    None
        Raises on any false union or false split.
    """
    ops = _ops_with_stem(traced, stem)
    assert len(ops) == len(truth_sites), (
        f"{stem}: captured {len(ops)} calls, truth enumerates {len(truth_sites)}"
    )
    for first, second in itertools.combinations_with_replacement(range(len(ops)), 2):
        implementation_same = ops[first].layer_label == ops[second].layer_label
        truth_same = truth_sites[first] == truth_sites[second]
        assert implementation_same == truth_same, (
            f"{stem} calls {first}/{second}: implementation layers "
            f"{ops[first].layer_label!r}/{ops[second].layer_label!r}, "
            f"truth sites {truth_sites[first]!r}/{truth_sites[second]!r}"
        )


def _trace_and_check(
    case: str,
    first: int,
    second: int,
    truth: Mapping[str, Sequence[Hashable]],
) -> None:
    """Trace one boundary model and check every requested truth partition.

    Parameters
    ----------
    case:
        Boundary program name.
    first:
        First loop's iteration count.
    second:
        Second loop's iteration count.
    truth:
        Operation stem to independently enumerated site sequence.

    Returns
    -------
    None
        Raises when grouping or metadata invariants diverge from truth.
    """
    torch.manual_seed(0)
    traced = trace_fn(_BoundaryOracleModel(case, first, second), torch.randn(1, 4))
    for stem, truth_sites in truth.items():
        _assert_brute_force_partition(traced, stem, truth_sites)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize(
    ("case", "first", "second", "truth"),
    [
        pytest.param("unary_post_ln", 3, 0, {"tanh": ("body",) * 3}, id="disputed-a"),
        pytest.param("unary_post_ln", 2, 0, {"tanh": ("body",) * 2}, id="n-equals-2"),
        pytest.param(
            "masked_scan",
            3,
            0,
            {"mul": ("mask",) * 3, "add": ("scan",) * 3},
            id="disputed-b",
        ),
        pytest.param(
            "parallel_combiner",
            2,
            2,
            {"add": ("enc", "enc", "dec", "dec", "join")},
            id="disputed-c",
        ),
        pytest.param(
            "residual_two_site",
            3,
            0,
            {"add": ("attn", "mlp") * 3},
            id="disputed-d-two-site-residual",
        ),
        pytest.param(
            "peel",
            1,
            2,
            {"add": ("first", "second", "second"), "tanh": ("motif",) * 3},
            id="disputed-e-n1-peel",
        ),
        pytest.param(
            "post_ln_tap",
            2,
            2,
            {"add": ("post-ln", "post-ln", "tail", "tail")},
            id="f1-post-ln-tap",
        ),
        pytest.param("post_ln", 3, 0, {"add": ("post-ln",) * 3}, id="f2-post-ln"),
        pytest.param(
            "nested_outer",
            2,
            2,
            {"add": ("inner", "inner", "outer", "inner", "inner", "outer")},
            id="f3-nested-outer",
        ),
        pytest.param(
            "same_key_prelude",
            2,
            2,
            {"add": ("prelude", "x", "x", "y", "y")},
            id="f4-shared-prelude",
        ),
        pytest.param(
            "shared_flank_norm",
            2,
            2,
            {"add": ("norm-a", "norm-a", "norm-b", "norm-b")},
            id="f5-shared-flank-2x2",
        ),
        pytest.param(
            "shared_flank_norm",
            3,
            2,
            {"add": ("norm-a",) * 3 + ("norm-b",) * 2},
            id="f5-shared-flank-3x2",
        ),
        pytest.param(
            "parallel_accumulators",
            2,
            0,
            {"add": ("first", "second") * 2},
            id="interleaved-accumulators-n2",
        ),
        pytest.param(
            "parallel_accumulators",
            3,
            0,
            {"add": ("first", "second") * 3},
            id="interleaved-accumulators-n3",
        ),
        pytest.param(
            "albert",
            3,
            2,
            {"linear": ("shared",) * 5, "tanh": ("shared",) * 5},
            id="albert-5-of-5",
        ),
        pytest.param(
            "distinct_submodule",
            3,
            2,
            {
                "linear": ("enc",) * 3 + ("dec",) * 2,
                "tanh": ("enc",) * 3 + ("dec",) * 2,
            },
            id="distinct-submodule",
        ),
        pytest.param(
            "nested_shared",
            2,
            3,
            {"add": ("nested",) * 6, "tanh": ("nested",) * 6},
            id="nested-loop",
        ),
        pytest.param(
            "chained",
            2,
            3,
            {
                "add": ("first",) * 2 + ("second",) * 3 + ("third",) * 2,
                "tanh": ("first",) * 2 + ("second",) * 3 + ("third",) * 2,
            },
            id="three-chained-loops",
        ),
    ],
)
def test_boundary_partition_matches_brute_force_truth(
    case: str,
    first: int,
    second: int,
    truth: Mapping[str, Sequence[Hashable]],
) -> None:
    """Pin disputed, F1-F5, parallel, reuse, nested, and chained partitions."""
    _trace_and_check(case, first, second, truth)


@pytest.mark.parametrize(
    ("model_type", "truth"),
    [
        pytest.param(
            example_models.NestedParamFreeLoops,
            {"sin": ("inner-site",) * 12},
            id="pure-nested-4x3",
        ),
        pytest.param(
            example_models.SequentialParamFreeLoops,
            {
                "sin": ("first-loop",) * 3 + ("second-loop",) * 3,
                "cos": ("first-loop",) * 3 + ("second-loop",) * 3,
            },
            id="pure-sequential-3-plus-3",
        ),
    ],
)
def test_pure_param_free_partition_matches_brute_force_truth(
    model_type: type[nn.Module], truth: Mapping[str, Sequence[Hashable]]
) -> None:
    """Pin A's historical nested/sequential pure-op topology at exact membership."""
    torch.manual_seed(0)
    traced = trace_fn(model_type(), torch.randn(1, 4))
    for stem, truth_sites in truth.items():
        _assert_brute_force_partition(traced, stem, truth_sites)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize(
    ("kind", "stem", "slots"),
    [
        pytest.param("rnncell", "rnntanhcell", 1, id="RNNCell"),
        pytest.param("grucell", "grucell", 1, id="GRUCell"),
        pytest.param("lstmcell", "lstmcell", 2, id="LSTMCell"),
    ],
)
def test_recurrent_cell_slots_match_brute_force_truth(kind: str, stem: str, slots: int) -> None:
    """Pin N=2 RNN/GRU/LSTM cell calls to one two-pass layer per output slot."""
    calls = 2
    traced = trace_fn(_RepeatedCell(kind, calls), torch.randn(2, 4))
    ops = _ops_with_stem(traced, stem)
    layers = {op.layer_label for op in ops}
    assert len(ops) == calls * slots
    assert len(layers) == slots
    assert all(sum(op.layer_label == layer for op in ops) == calls for layer in layers)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize(
    ("kind", "stem", "slots"),
    [
        pytest.param(nn.RNN, "rnntanh", 2, id="RNN"),
        pytest.param(nn.GRU, "gru", 2, id="GRU"),
        pytest.param(nn.LSTM, "lstm", 3, id="LSTM"),
    ],
)
def test_fused_recurrent_slots_match_brute_force_truth(
    kind: type[nn.RNNBase], stem: str, slots: int
) -> None:
    """Pin fused RNN/GRU/LSTM calls to one three-pass layer per output slot."""
    calls = 3
    traced = trace_fn(_RepeatedFusedRecurrent(kind, calls), torch.randn(1, 5, 4))
    ops = _ops_with_stem(traced, stem)
    layers = {op.layer_label for op in ops}
    assert len(ops) == calls * slots
    assert len(layers) == slots
    assert all(sum(op.layer_label == layer for op in ops) == calls for layer in layers)
    assert check_metadata_invariants(traced)
