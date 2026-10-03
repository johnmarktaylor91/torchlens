"""Hardening tests for loop/recurrence grouping (round-21).

Locks the fixes for the round-20 adversarial findings:

* Multi-output calls (LSTMCell h/c, fused RNN/GRU/LSTM output/state slots) must
  produce one recurrent layer PER OUTPUT SLOT, never interleaved passes of one
  layer (review F3, review's fused-module generalization, seq2seq equivalence_symmetry).
* Grouping must be invariant to sibling capture order / ``data_children`` tuple
  order (review H1) -- the load-bearing metamorphic property.
* The ``recurrence_anchored`` bypass must work for its canonical two-call case
  (review F1).
* Param-free ops interleaved between weight-tied loops must respect loop
  boundaries; shared weights in structurally-disjoint branches must not induce
  spurious recurrence in independent ops (review F2, review M1).
"""

import dataclasses
import random
from collections import OrderedDict, defaultdict

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.postprocess.loop_grouping_adapter import (
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from torchlens.validation.invariants import check_metadata_invariants

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_passes(traced) -> "OrderedDict[str, int]":
    """Return ordered mapping of layer_label -> number of grouped passes."""
    layers: OrderedDict[str, list] = OrderedDict()
    for op in traced:
        layers.setdefault(op.layer_label, []).append(op)
    return OrderedDict((label, len(ops)) for label, ops in layers.items())


def _layers_of_type(passes: "OrderedDict[str, int]", stem: str) -> dict[str, int]:
    """Filter layer pass counts whose label starts with ``stem``."""
    return {label: count for label, count in passes.items() if label.startswith(stem)}


def _assert_structurally_coherent(traced) -> None:
    """Per-layer structural invariants: pass bijection and consistent counts."""
    layers: dict[str, list] = defaultdict(list)
    for op in traced:
        layers[op.layer_label].append(op)
    for label, ops in layers.items():
        n = len(ops)
        pass_indices = sorted(op.pass_index for op in ops)
        assert pass_indices == list(range(1, n + 1)), (
            f"layer {label}: pass_index {pass_indices} is not a bijection onto 1..{n}"
        )
        assert {op.num_passes for op in ops} == {n}, (
            f"layer {label}: num_passes disagree with member count {n}"
        )
        for op in ops:
            assert op.num_passes == len(op.recurrent_ops), (
                f"op {op.label}: num_passes != len(recurrent_ops)"
            )


# ---------------------------------------------------------------------------
# Direct-grouper order-invariance fuzz (LOAD-BEARING, review H1)
# ---------------------------------------------------------------------------


class _GraphBuilder:
    """Small builder for direct RecurrenceGroupingGraph fuzz inputs."""

    def __init__(self) -> None:
        self.order = 0
        self.node_specs: dict[str, dict] = {}
        self.raw: list[str] = []
        self.sources: list[str] = []
        self._children: dict[str, list[str]] = defaultdict(list)

    def add(
        self,
        label: str,
        eqkey: str,
        parents: tuple = (),
        uses_params: bool = False,
        params: tuple = (),
        anchored: bool = False,
        source: bool = False,
        slot: "int | None" = None,
    ) -> str:
        self.order += 1
        self.node_specs[label] = {
            "raw_order": self.order,
            "eqkey": eqkey,
            "parents": tuple(parents),
            "uses_params": uses_params,
            "params": tuple(params),
            "anchored": anchored,
            "slot": slot,
        }
        self.raw.append(label)
        if source:
            self.sources.append(label)
        for parent in parents:
            self._children[parent].append(label)
        return label

    def build(self) -> RecurrenceGroupingGraph:
        by_key: dict[str, list[str]] = defaultdict(list)
        for label, spec in self.node_specs.items():
            by_key[spec["eqkey"]].append(label)
        nodes = {}
        for label, spec in self.node_specs.items():
            nodes[label] = RecurrenceNode(
                label=label,
                raw_order=spec["raw_order"],
                equivalence_key=spec["eqkey"],
                equivalent_labels=tuple(sorted(by_key[spec["eqkey"]])),
                data_parents=spec["parents"],
                data_children=tuple(self._children[label]),
                layer_label=label,
                recurrent_labels=(label,),
                uses_params=spec["uses_params"],
                func_name="f_" + spec["eqkey"],
                param_barcodes=spec["params"],
                output_slot=spec["slot"],
                recurrence_anchored=spec["anchored"],
            )
        return RecurrenceGroupingGraph(
            nodes=nodes,
            raw_labels=tuple(self.raw),
            source_labels=tuple(self.sources),
            eligible_labels=tuple(self.raw),
        )


def _random_dag(seed: int, n_ops: int = 12, n_keys: int = 3, param_frac: float = 0.3):
    rng = random.Random(seed)
    builder = _GraphBuilder()
    inp = builder.add("inp", "input", source=True)
    labels = [inp]
    for i in range(n_ops):
        k = rng.choice([1, 1, 2])
        parents = rng.sample(labels, min(k, len(labels)))
        key = f"K{rng.randrange(n_keys)}"
        uses = rng.random() < param_frac
        params = (f"P{rng.randrange(2)}",) if uses else ()
        anchored = uses and rng.random() < 0.3
        labels.append(
            builder.add(
                f"op{i}",
                key,
                parents=parents,
                uses_params=uses,
                params=params,
                anchored=anchored,
            )
        )
    builder.add("out", "output", parents=rng.sample(labels, min(2, len(labels))))
    return builder


def _partition_signature(graph: RecurrenceGroupingGraph) -> frozenset:
    assignments = group_recurrent_nodes(graph)
    layers: dict[str, set[str]] = defaultdict(set)
    for label, assignment in assignments.items():
        layers[assignment.layer_label].add(label)
    return frozenset(frozenset(members) for members in layers.values())


def _permuted(graph: RecurrenceGroupingGraph, rng: "random.Random | None"):
    """Reverse (rng None) or shuffle every node's edge/equivalence orderings."""
    new_nodes = {}
    for label, node in graph.nodes.items():
        parents = list(node.data_parents)
        children = list(node.data_children)
        equivalents = list(node.equivalent_labels)
        if rng is None:
            parents.reverse()
            children.reverse()
            equivalents.reverse()
        else:
            rng.shuffle(parents)
            rng.shuffle(children)
            rng.shuffle(equivalents)
        new_nodes[label] = dataclasses.replace(
            node,
            data_parents=tuple(parents),
            data_children=tuple(children),
            equivalent_labels=tuple(equivalents),
        )
    return dataclasses.replace(graph, nodes=new_nodes)


def test_children_order_permutation_invariance() -> None:
    """LOAD-BEARING: the layer partition is invariant to edge-tuple ordering.

    Reversing or randomly permuting ``data_children`` / ``data_parents`` /
    ``equivalent_labels`` must never change the grouping partition -- grouping
    must be a well-defined function of the captured DAG, not of sibling capture
    order (base tree diverged on 34-38% of random DAGs).
    """
    diverged = []
    for seed in range(300):
        for param_frac in (0.0, 0.6):
            graph = _random_dag(seed, param_frac=param_frac).build()
            base_sig = _partition_signature(graph)
            if _partition_signature(_permuted(graph, None)) != base_sig:
                diverged.append((seed, param_frac, "reverse"))
            elif _partition_signature(_permuted(graph, random.Random(seed + 999))) != base_sig:
                diverged.append((seed, param_frac, "shuffle"))
    assert not diverged, f"order-sensitive groupings: {diverged[:5]} (of {len(diverged)})"


class _OrderA(nn.Module):
    def forward(self, x):
        op0 = torch.relu(x)
        op1 = torch.relu(op0)
        op2 = op1 * op0
        op3 = x * op0
        return op2 + op3.sum() * 0


class _OrderB(nn.Module):
    """Same graph as _OrderA; only the independent sibling op3 moves earlier."""

    def forward(self, x):
        op0 = torch.relu(x)
        op3 = x * op0
        op1 = torch.relu(op0)
        op2 = op1 * op0
        return op2 + op3.sum() * 0


def test_sibling_statement_order_invariance_real_models() -> None:
    """Mathematically identical models group identically regardless of statement order.

    OrderB used to fabricate a 2-pass relu "loop" for ``relu(relu(x))`` and merge
    two muls with different parents as recurrent passes.
    """
    x = torch.randn(3, 3)
    passes_a = _layer_passes(trace_fn(_OrderA(), x))
    passes_b = _layer_passes(trace_fn(_OrderB(), x))

    # Identical multiset of (type-stem, num_passes): labels may renumber.
    def type_profile(passes):
        profile = defaultdict(list)
        for label, count in passes.items():
            profile[label.rsplit("_", 2)[0]].append(count)
        return {stem: sorted(counts) for stem, counts in profile.items()}

    assert type_profile(passes_a) == type_profile(passes_b)
    # Guard integrity: a straight functional chain is NEVER a multi-pass loop.
    for passes in (passes_a, passes_b):
        relus = _layers_of_type(passes, "relu")
        assert len(relus) == 2 and set(relus.values()) == {1}, relus
        muls = _layers_of_type(passes, "mul")
        assert len(muls) == 3 and set(muls.values()) == {1}, muls


def test_straight_chain_never_grouped() -> None:
    """``tanh(tanh(x))`` (bare functional) is a chain, not a 2-pass loop."""

    class Chain(nn.Module):
        def forward(self, x):
            return torch.tanh(torch.tanh(x))

    passes = _layer_passes(trace_fn(Chain(), torch.randn(2, 4)))
    tanhs = _layers_of_type(passes, "tanh")
    assert len(tanhs) == 2 and set(tanhs.values()) == {1}, tanhs


# ---------------------------------------------------------------------------
# Multi-output cells: per-output-slot layers (review F3 + review fused + capprov)
# ---------------------------------------------------------------------------


class _TiedCellLoop(nn.Module):
    """Run a cell N times with tied weights; supports multi-state cells."""

    def __init__(self, kind: str, n_steps: int) -> None:
        super().__init__()
        self.kind = kind
        self.n_steps = n_steps
        if kind == "linear":
            self.cell = nn.Linear(4, 4)
        elif kind == "rnncell":
            self.cell = nn.RNNCell(4, 4)
        elif kind == "grucell":
            self.cell = nn.GRUCell(4, 4)
        elif kind == "lstmcell":
            self.cell = nn.LSTMCell(4, 4)

    def forward(self, x):
        out = x + 0.0
        if self.kind == "linear":
            for _ in range(self.n_steps):
                out = self.cell(out)
        elif self.kind in ("rnncell", "grucell"):
            h = torch.zeros(x.shape[0], 4)
            for _ in range(self.n_steps):
                h = self.cell(out, h)
            out = out + h.sum() * 0
        else:
            h = torch.zeros(x.shape[0], 4)
            c = torch.zeros(x.shape[0], 4)
            for _ in range(self.n_steps):
                h, c = self.cell(out, (h, c))
            out = out + h.sum() * 0
        return out


_CELL_LAYER_STEMS = {
    "linear": "linear",
    "rnncell": "rnntanhcell",
    "grucell": "grucell",
    "lstmcell": "lstmcell",
}
_CELL_SLOTS = {"linear": 1, "rnncell": 1, "grucell": 1, "lstmcell": 2}


@pytest.mark.parametrize("kind", ["linear", "rnncell", "grucell", "lstmcell"])
@pytest.mark.parametrize("n_steps", [0, 1, 2, 3, 4, 5, 6])
def test_n_iterations_yield_n_passes_per_slot(kind: str, n_steps: int) -> None:
    """N loop iterations => exactly N passes for EVERY output slot's layer.

    LSTMCell used to report 2N passes for one interleaved h/c layer (and {1, 3}
    for N=2); each co-output slot must instead be its own N-pass layer, matching
    the torch.max values/indices precedent.
    """
    traced = trace_fn(_TiedCellLoop(kind, n_steps), torch.randn(2, 4))
    stem = _CELL_LAYER_STEMS[kind]
    cell_layers = _layers_of_type(_layer_passes(traced), stem)
    if n_steps == 0:
        assert cell_layers == {}
        return
    expected_layer_count = _CELL_SLOTS[kind]
    assert len(cell_layers) == expected_layer_count, cell_layers
    assert set(cell_layers.values()) == {n_steps}, cell_layers
    _assert_structurally_coherent(traced)


class _RepeatedFusedRNN(nn.Module):
    """Call a fused multi-output recurrent module N times (state unused)."""

    def __init__(self, kind, n_calls: int) -> None:
        super().__init__()
        self.rnn = kind(4, 4, batch_first=True)
        self.n_calls = n_calls

    def forward(self, x):
        for _ in range(self.n_calls):
            x = self.rnn(x)[0]
        return x


class _RepeatedFusedRNNUsedState(nn.Module):
    """Call a fused recurrent module N times, feeding the state back in."""

    def __init__(self, kind, n_calls: int) -> None:
        super().__init__()
        self.rnn = kind(4, 4, batch_first=True)
        self.n_calls = n_calls

    def forward(self, x):
        h = None
        for _ in range(self.n_calls):
            x, h = self.rnn(x) if h is None else self.rnn(x, h)
        return x


_FUSED_STEMS = {nn.RNN: "rnntanh", nn.GRU: "gru", nn.LSTM: "lstm"}
_FUSED_SLOTS = {nn.RNN: 2, nn.GRU: 2, nn.LSTM: 3}


@pytest.mark.parametrize("kind", [nn.RNN, nn.GRU, nn.LSTM], ids=["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("used_state", [False, True], ids=["unused-state", "used-state"])
def test_fused_multi_output_modules_per_slot(kind, used_state: bool) -> None:
    """Repeated fused nn.RNN/GRU/LSTM: one N-pass layer per output slot.

    These used to collapse the sequence-output and hidden-state slots (and cell
    state for LSTM) into one 2N/3N-pass layer, tripping the equivalence_symmetry
    metadata invariant. Covers used AND unused state outputs.
    """
    n_calls = 3
    model_cls = _RepeatedFusedRNNUsedState if used_state else _RepeatedFusedRNN
    traced = trace_fn(model_cls(kind, n_calls), torch.randn(1, 5, 4))
    stem = _FUSED_STEMS[kind]
    fused_layers = _layers_of_type(_layer_passes(traced), stem)
    assert len(fused_layers) == _FUSED_SLOTS[kind], fused_layers
    assert set(fused_layers.values()) == {n_calls}, fused_layers

    # No recurrent group may span two equivalent_ops sets / output slots, and
    # pass i of every slot layer must come from module call i (co-outputs of one
    # call share parents and pass numbers, and never share a layer).
    slot_ops: dict[str, list] = defaultdict(list)
    for op in traced:
        if op.layer_label in fused_layers:
            slot_ops[op.layer_label].append(op)
    for label, ops in slot_ops.items():
        equivalence_sets = {frozenset(op.equivalent_ops) for op in ops}
        assert len(equivalence_sets) == 1, f"layer {label} spans multiple equivalent_ops sets"
        by_pass = sorted(ops, key=lambda op: op.pass_index)
        raw_orders = [op.raw_index for op in by_pass]
        assert raw_orders == sorted(raw_orders), (
            f"layer {label}: pass numbering does not follow call order"
        )
    # Call k's co-outputs share an identical parent set and identical pass_index.
    call_signature: dict[frozenset, set[int]] = defaultdict(set)
    for ops in slot_ops.values():
        for op in ops:
            call_signature[frozenset(op.parents)].add(op.pass_index)
    for _parents, pass_indices in call_signature.items():
        assert len(pass_indices) == 1, (
            f"co-outputs of one call carry different pass numbers: {pass_indices}"
        )
    assert check_metadata_invariants(traced)


def test_co_outputs_of_single_call_not_recurrent() -> None:
    """Same-shape co-outputs of ONE call never become passes of one layer."""

    class SingleCall(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTMCell(4, 4)

        def forward(self, x):
            h, c = self.cell(x, (torch.zeros(x.shape[0], 4), torch.zeros(x.shape[0], 4)))
            return h + c

    traced = trace_fn(SingleCall(), torch.randn(2, 4))
    cell_layers = _layers_of_type(_layer_passes(traced), "lstmcell")
    assert len(cell_layers) == 2 and set(cell_layers.values()) == {1}, cell_layers

    class ChunkOnce(nn.Module):
        def forward(self, x):
            a, b = torch.chunk(x, 2, dim=1)
            return a + b

    traced = trace_fn(ChunkOnce(), torch.randn(2, 4))
    chunk_layers = _layers_of_type(_layer_passes(traced), "chunk")
    assert set(chunk_layers.values()) == {1}, chunk_layers


# ---------------------------------------------------------------------------
# recurrence_anchored N=2 bypass (review F1)
# ---------------------------------------------------------------------------


class _ReusedActivation(nn.Module):
    def __init__(self, act: nn.Module, n_calls: int) -> None:
        super().__init__()
        self.act = act
        self.n_calls = n_calls

    def forward(self, x):
        for _ in range(self.n_calls):
            x = self.act(x)
        return x


@pytest.mark.parametrize("n_calls", [2, 3, 4])
@pytest.mark.parametrize("act", [nn.ReLU, nn.Tanh], ids=["relu", "tanh"])
def test_reused_module_two_calls_grouped(act, n_calls: int) -> None:
    """A reused single-op module applied N times is one N-pass layer, incl. N=2.

    The anchored bypass was dead for N=2: refinement split the 2-member group
    before the merge guard could honor ``recurrence_anchored``.
    """
    traced = trace_fn(_ReusedActivation(act(), n_calls), torch.randn(2, 4))
    stem = "relu" if act is nn.ReLU else "tanh"
    act_layers = _layers_of_type(_layer_passes(traced), stem)
    assert act_layers == {next(iter(act_layers)): n_calls}, act_layers


# ---------------------------------------------------------------------------
# Loop-boundary respect for param-free ops (review F2 + review M1)
# ---------------------------------------------------------------------------


class _ChainedTiedLoops(nn.Module):
    """Loop A (3x tied) then loop B (2x tied), tanh interleaved in both."""

    def __init__(self) -> None:
        super().__init__()
        self.A = nn.Linear(4, 4)
        self.B = nn.Linear(4, 4)

    def forward(self, x):
        h = x
        for _ in range(3):
            h = torch.tanh(self.A(h))
        for _ in range(2):
            h = torch.tanh(self.B(h))
        return h


def test_param_free_op_respects_loop_boundary() -> None:
    """The shared activation splits 3/2 with its surrounding tied loops, not 5."""
    traced = trace_fn(_ChainedTiedLoops(), torch.randn(2, 4))
    passes = _layer_passes(traced)
    assert sorted(_layers_of_type(passes, "linear").values()) == [2, 3]
    assert sorted(_layers_of_type(passes, "tanh").values()) == [2, 3], passes
    _assert_structurally_coherent(traced)


class _SharedCellTwoLoops(nn.Module):
    """One tied cell used in two structurally-disjoint loops on parallel streams."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x):
        a = x
        for _ in range(2):
            a = torch.tanh(self.cell(a))
        b = torch.relu(x)
        for _ in range(3):
            b = torch.tanh(self.cell(b))
        return a.sum() + b.sum()


def test_shared_weight_disjoint_loops_no_spurious_recurrence() -> None:
    """Independent terminal reductions on parallel streams stay single-pass.

    A weight shared across two disjoint loops used to cascade through param-free
    neighbors and group ``a.sum()``/``b.sum()`` as 2-pass recurrence.
    """
    traced = trace_fn(_SharedCellTwoLoops(), torch.randn(2, 4))
    passes = _layer_passes(traced)
    sums = _layers_of_type(passes, "sum")
    assert len(sums) == 2 and set(sums.values()) == {1}, sums
    # The tied cell is one 5-pass layer (weight-tying doctrine)...
    assert sorted(_layers_of_type(passes, "linear").values()) == [5]
    # ...but the per-stream tanh groups never merge across the parallel streams.
    assert sorted(_layers_of_type(passes, "tanh").values()) == [2, 3], passes
    _assert_structurally_coherent(traced)


class _InterleavedTiedChain(nn.Module):
    """Regression: tied weights with alternating interleaved ops keep grouping.

    RecurrentParamsSimple-style contract: the mul after tied passes 1/3 and the
    log after tied passes 2/4 are legitimate 2-pass layers because the seeds
    chain THROUGH the tied loop (unlike parallel disjoint streams).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)

    def forward(self, x):
        x = x + 1
        x = self.fc1(x)
        x = x * 2
        x = self.fc1(x)
        x = torch.log(x)
        x = torch.tan(x)
        x = self.fc1(x)
        x = x * 2
        x = self.fc1(x)
        x = torch.log(x)
        return x


def test_interleaved_tied_chain_keeps_alternating_groups() -> None:
    """Chained tied-weight interleaving still groups alternating param-free ops."""
    traced = trace_fn(_InterleavedTiedChain(), torch.randn(5, 5))
    passes = _layer_passes(traced)
    assert sorted(_layers_of_type(passes, "linear").values()) == [4]
    assert sorted(_layers_of_type(passes, "mul").values()) == [2], passes
    assert sorted(_layers_of_type(passes, "log").values()) == [2], passes


# ---------------------------------------------------------------------------
# Rolled-graph pass-count coherence (structural lock)
# ---------------------------------------------------------------------------


def test_rolled_graph_pass_count_coherence() -> None:
    """Structural invariants hold across the round's marquee recurrence models."""

    class MaxLoop(nn.Module):
        def forward(self, x):
            for _ in range(3):
                values, indices = torch.max(x, dim=1)
                x = x + values.unsqueeze(1) * 0 + indices.unsqueeze(1) * 0.0
            return x

    class LSTMCellLoop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTMCell(4, 5)

        def forward(self, x):
            h = torch.zeros(1, 5)
            c = torch.zeros(1, 5)
            for t in range(x.shape[1]):
                h, c = self.cell(x[:, t, :], (h, c))
            return h

    for model, x in (
        (MaxLoop(), torch.randn(2, 4)),
        (LSTMCellLoop(), torch.randn(1, 4, 4)),
        (_ChainedTiedLoops(), torch.randn(2, 4)),
        (_SharedCellTwoLoops(), torch.randn(2, 4)),
    ):
        traced = trace_fn(model, x)
        _assert_structurally_coherent(traced)
        assert check_metadata_invariants(traced)
    # torch.max values/indices stay two parallel 3-pass layers (the precedent
    # multi-output slots must follow).
    passes = _layer_passes(trace_fn(MaxLoop(), torch.randn(2, 4)))
    assert sorted(_layers_of_type(passes, "max").values()) == [3, 3], passes
