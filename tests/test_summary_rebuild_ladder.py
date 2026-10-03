"""F08 auto-ladder acceptance: rungs, folds, elision, multi-pass, partition.

Summary memo 3.1/3.4 build requirements: the identity partition holds at
every depth/fold/elision; the strict fold signature splits on trainability
and dtype drift; elision is deterministic, protected, totals-conserving;
the multi-pass row model renders one row per layer at module grain and one
row per pass at op grain.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report._factcore import factcore
from torchlens.report._summary_ladder import (
    DEFAULT_ROW_BUDGET,
    _ModelIndex,
    build_hybrid,
    build_tree,
    resolve_view,
)


class _FiveOpToy(nn.Module):
    """Two coalescing leaves, one orphan functional op, one getitem."""

    def __init__(self) -> None:
        """Two Linears; the add and slice are functional orphans."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fc1 -> relu -> fc2 -> residual add over a slice."""

        a = self.fc1(x)
        return self.fc2(torch.relu(a)) + a[:, :8]


class _Block(nn.Module):
    """One Linear + one orphan relu (a multi-item block for folding)."""

    def __init__(self, width: int) -> None:
        """One square Linear."""

        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then functional relu."""

        return torch.relu(self.lin(x))


class _Repeated(nn.Module):
    """Six identical sibling blocks (a fold run) plus a head."""

    def __init__(self) -> None:
        """Sequential of six blocks and one head Linear."""

        super().__init__()
        self.blocks = nn.Sequential(*[_Block(4) for _ in range(6)])
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blocks then head."""

        return self.head(self.blocks(x))


class _ThreePass(nn.Module):
    """One module called three times (multi-pass layer grouping)."""

    def __init__(self) -> None:
        """One square Linear reused three times."""

        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three passes through the same module."""

        for _ in range(3):
            x = self.linear(x)
        return x


def _assert_partition(trace: tl.Trace, view) -> None:
    """The identity-partition invariant (memo 3.1), all three families."""

    core = factcore(trace)
    owned_ops = [label for row in view.rows for label in row.owned_ops]
    owned_ops += list(view.root_owned_ops)
    assert len(owned_ops) == len(set(owned_ops)) == core.counts.compute_ops
    owned_params = [path for row in view.rows for path in row.owned_params]
    owned_params += list(view.root_owned_params)
    assert len(owned_params) == len(set(owned_params))
    owned_param_total = sum(row.params_owned for row in view.rows) + view.root_params_owned
    assert owned_param_total == (core.params.total or 0)
    flops_total = sum(row.flops_owned for row in view.rows) + view.root_flops_owned
    assert flops_total == int(core.compute.partition_total)


def test_hybrid_coalesces_and_shows_orphans() -> None:
    """The 5-op toy: every executed op once, orphan add visible, 5 rows."""

    trace = tl.trace(_FiveOpToy(), torch.randn(2, 8))
    view = resolve_view(trace)
    assert view.rung == "hybrid"
    kinds = {row.name: row.kind for row in view.rows}
    assert kinds["fc1"] == "coalesced"
    assert kinds["fc2"] == "coalesced"
    assert any(row.kind == "op" and row.name.startswith("add") for row in view.rows)
    assert any(row.kind == "op" and row.name.startswith("relu") for row in view.rows)
    _assert_partition(trace, view)


def test_execution_order_interleaves_orphans() -> None:
    """Hybrid rows follow execution order (relu between fc1 and fc2)."""

    trace = tl.trace(_FiveOpToy(), torch.randn(2, 8))
    view = resolve_view(trace)
    names = [row.name for row in view.rows]
    assert names.index("fc1") < names.index("relu_1_2") < names.index("fc2")


def test_fold_groups_identical_siblings() -> None:
    """Six identical blocks fold to one representative owning all six."""

    trace = tl.trace(_Repeated(), torch.randn(2, 4))
    index = _ModelIndex(trace)
    rows = build_tree(index, 2)
    folds = [row for row in rows if row.kind == "fold"]
    assert len(folds) == 1
    assert folds[0].fold_count == 6
    assert folds[0].params_owned == 6 * (4 * 4 + 4)


def test_strict_fold_signature_splits_on_trainability_and_dtype() -> None:
    """Memo 3.4: freeze one block, cast another -- the fold splits visibly."""

    model = _Repeated()
    for parameter in model.blocks[0].parameters():
        parameter.requires_grad_(False)
    model.blocks[3].lin = model.blocks[3].lin.to(torch.float64)

    class _Caster(nn.Module):
        """Wrap to keep dtypes consistent through block 3."""

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x.to(torch.float64)).to(torch.float32)

    model.blocks[3] = _Caster(model.blocks[3])
    trace = tl.trace(model, torch.randn(2, 4))
    index = _ModelIndex(trace)
    rows = build_tree(index, 2)
    fold_counts = sorted(row.fold_count for row in rows if row.kind == "fold")
    # Blocks 1..2 and 4..5 fold; blocks 0 (frozen) and 3 (dtype) stand alone.
    assert fold_counts == [2, 2]


def test_multipass_one_row_at_module_grain() -> None:
    """A layer with three passes renders ONE row owning all three events."""

    trace = tl.trace(_ThreePass(), torch.randn(2, 3))
    view = resolve_view(trace)
    linear_rows = [row for row in view.rows if "linear" in (row.address or row.name)]
    assert len(linear_rows) == 1
    assert linear_rows[0].passes == 3
    assert len(linear_rows[0].owned_ops) == 3
    _assert_partition(trace, view)


def test_multipass_one_row_per_pass_at_op_grain() -> None:
    """level='op' renders one row per pass with pass-qualified names."""

    trace = tl.trace(_ThreePass(), torch.randn(2, 3))
    view = resolve_view(trace, level="op")
    names = [row.name for row in view.rows]
    assert len(view.rows) == 3
    assert any(":" in name for name in names)
    _assert_partition(trace, view)


def test_flat_sequential_never_collapses_to_containers() -> None:
    """The structural cliff: 120 flat leaves fit the budget via elision."""

    flat = nn.Sequential(*[nn.Linear(8, 8) for _ in range(120)])
    for index, module in enumerate(flat):
        if index % 2:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    trace = tl.trace(flat, torch.randn(1, 8))
    view = resolve_view(trace)
    assert view.body_row_count <= DEFAULT_ROW_BUDGET
    assert view.body_row_count >= 8  # the min-floor: never three opaque rows
    assert any(row.kind == "elision" for row in view.rows)
    elision = next(row for row in view.rows if row.kind == "elision")
    assert elision.elided_count > 0
    assert ".." in elision.name  # the address disclosure
    _assert_partition(trace, view)


def test_elision_is_deterministic() -> None:
    """The elision rung is a pure function of (trace, args)."""

    flat = nn.Sequential(*[nn.Linear(8, 8) for _ in range(80)])
    trace = tl.trace(flat, torch.randn(1, 8))
    first = resolve_view(trace, fold_repeats=False)
    second = resolve_view(trace, fold_repeats=False)
    assert [row.row_id for row in first.rows] == [row.row_id for row in second.rows]
    assert first.disclosure == second.disclosure


@pytest.mark.smoke
def test_first_and_last_rows_survive_elision() -> None:
    """Input-adjacent and output-adjacent rows are protected (memo 3.4)."""

    flat = nn.Sequential(*[nn.Linear(8, 8) for _ in range(80)])
    trace = tl.trace(flat, torch.randn(1, 8))
    view = resolve_view(trace, fold_repeats=False)
    assert view.rows[0].kind != "elision"
    assert view.rows[-1].kind != "elision"


@pytest.mark.smoke
def test_explicit_depth_and_budget_are_honored() -> None:
    """depth= pins the tree rung; max_rows= re-budgets the ladder."""

    trace = tl.trace(_Repeated(), torch.randn(2, 4))
    depth_one = resolve_view(trace, level="module", depth=1)
    assert depth_one.rung == "tree"
    assert depth_one.depth == 1
    wide = resolve_view(trace, max_rows=500)
    assert wide.body_row_count <= 500
    _assert_partition(trace, wide)


def test_unexecuted_module_row_owns_its_params() -> None:
    """A3: declared-but-never-ran params are owned by a visible row."""

    class _Dead(nn.Module):
        """One live Linear, one dead Linear."""

        def __init__(self) -> None:
            super().__init__()
            self.live = nn.Linear(4, 4)
            self.dead = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.live(x)

    trace = tl.trace(_Dead(), torch.randn(2, 4))
    view = resolve_view(trace)
    dead_rows = [row for row in view.rows if row.address == "dead"]
    assert dead_rows and dead_rows[0].kind == "unexecuted"
    assert dead_rows[0].params_owned == 3 * 3 + 3
    _assert_partition(trace, view)


def test_hybrid_row_convention_matches_derived_budget() -> None:
    """The vgg19-derived budget constant is 48 (memo 3.4, derived not chosen)."""

    assert DEFAULT_ROW_BUDGET == 48


def test_tree_interior_of_fold_owns_nothing() -> None:
    """A fold's rendered interior displays values but owns no identities."""

    trace = tl.trace(_Repeated(), torch.randn(2, 4))
    index = _ModelIndex(trace)
    rows = build_tree(index, 3)
    fold_positions = [i for i, row in enumerate(rows) if row.kind == "fold"]
    assert fold_positions
    for position in fold_positions:
        for row in rows[position + 1 :]:
            if row.depth <= rows[position].depth:
                break
            assert row.owned_ops == ()
            assert row.owned_params == ()


def test_hybrid_counts_every_pass_event_once() -> None:
    """Hybrid ownership covers pass events exactly once (no double count)."""

    trace = tl.trace(_ThreePass(), torch.randn(2, 3))
    index = _ModelIndex(trace)
    rows = build_hybrid(index)
    owned = [label for row in rows for label in row.owned_ops]
    assert len(owned) == len(set(owned)) == factcore(trace).counts.compute_ops
