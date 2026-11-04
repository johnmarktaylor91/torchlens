"""Geometric fields through transformer building blocks (GPT-2 first).

Each test here fails at the base: the op had no rule (``split``, casts,
``masked_fill``), a whole-input envelope (``matmul``, ``softmax``, fused
attention), a mixed-axis reshape with no exact map, a same-object dropout
payload the gradient probe could not reach, or a refusal of set queries whose
descriptor has no windowed axis. Causal masking narrows the sequence axis in
``.of()`` answers: position ``i`` is reached by positions ``0..i`` and reaches
positions ``i..``; the gradient goldens (``cross_validate``) pin that the
geometric answer still contains every nonzero derivative.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as functional
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field._types import ReceptiveFieldValidationStatus

cross_validate = importlib.import_module("torchlens.receptive_field._validation").cross_validate

_ARMED = tl.options.CaptureOptions(backward_ready=True, save_mode="reference")
_SEQ = 5


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in rule pack while preserving registry isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _op(trace, name: str):
    """Return the last captured operation with the requested function name."""

    return next(item for item in reversed(trace._layer_list) if item.func_name == name)


def _only_output(trace, address: str):
    """Return the only output op of the module at ``address``."""

    module = trace.modules[address]
    (label,) = module.calls[0].output_ops
    return trace.ops[label]


def _row(selection, op):
    """Return the one entry of ``selection`` at ``op``."""

    rows = [entry for entry in selection if f"{entry.key[0]}:{entry.key[1]}" == op.label]
    assert len(rows) == 1, f"expected one row for {op.label}, got {len(rows)}"
    return rows[0]


def _cells(mask: torch.Tensor) -> list[list[int]]:
    """Sorted coordinates of the selected cells."""

    return sorted(mask.nonzero().tolist())


def _positions(mask: torch.Tensor, axis: int = 1) -> list[int]:
    """Positions along ``axis`` with at least one selected cell."""

    return [pos for pos in range(mask.shape[axis]) if bool(mask.movedim(axis, 0)[pos].any())]


def _all_pass(results) -> None:
    """Every gradient-inside-geometric golden passes."""

    assert results, "cross_validate produced no result"
    bad = [r for r in results if r.status is not ReceptiveFieldValidationStatus.PASS]
    assert not bad, [(r.op_label, r.status, r.violations[:3]) for r in bad]


# ---------------------------------------------------------------------------
# Axis bookkeeping ops: split / chunk / unbind, casts, reshape blocks.
# ---------------------------------------------------------------------------


class _Pick(nn.Module):
    """Return one piece of a split-family op along ``dim=1``."""

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how = how

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.how == "split":
            return x.split(2, dim=1)[2]
        if self.how == "chunk":
            return torch.chunk(x, 3, dim=1)[1]
        if self.how == "split_sizes":
            return x.split([1, 2, 3], dim=1)[1]
        return x.unbind(1)[2]


@pytest.mark.parametrize(
    ("how", "unit", "expected"),
    [
        ("split", (0, 0, 1), [0, 4, 1]),
        ("chunk", (0, 1, 2), [0, 3, 2]),
        ("split_sizes", (0, 1, 0), [0, 2, 0]),
        ("unbind", (0, 1), [0, 2, 1]),
    ],
)
def test_split_family_offsets_land_on_the_right_parent_slice(how, unit, expected) -> None:
    """A unit of chunk ``k`` comes from exactly one input cell at the chunk's offset."""

    trace = tl.trace(_Pick(how), torch.randn(1, 6, 3))
    try:
        region = trace.output_ops[0].receptive_field.of([unit])
        assert _cells(_row(region, trace.input_ops[0]).mask) == [expected]
    finally:
        trace.cleanup()


class _Casts(nn.Module):
    """Dtype casts, ``type`` and a comparison mask applied with ``masked_fill``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.half().float().type(torch.float32)
        return y.masked_fill(y.lt(0.0), 0.0)


def test_casts_and_masked_fill_are_pointwise() -> None:
    """Casts and ``masked_fill`` keep each unit on its own input cell."""

    trace = tl.trace(_Casts(), torch.randn(2, 3, 4))
    try:
        region = trace.output_ops[0].receptive_field.of([(1, 2, 3)])
        assert _cells(_row(region, trace.input_ops[0]).mask) == [[1, 2, 3]]
    finally:
        trace.cleanup()


class _SplitHeads(nn.Module):
    """GPT-2's head split: ``view`` into (heads, head_dim) then ``permute``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(1, _SEQ, 2, 4).permute(0, 2, 1, 3)


def test_head_split_reshape_maps_one_unit_to_one_feature() -> None:
    """Unit (head h, position p, d) comes from feature ``h * 4 + d`` at position ``p``."""

    trace = tl.trace(_SplitHeads(), torch.randn(1, _SEQ, 8))
    try:
        region = trace.output_ops[0].receptive_field.of([(0, 1, 3, 2)])
        assert _cells(_row(region, trace.input_ops[0]).mask) == [[0, 3, 6]]
        both = trace.output_ops[0].receptive_field.of([(0, 1, 3, 2), (0, 0, 3, 1)])
        assert _cells(_row(both, trace.input_ops[0]).mask) == [[0, 3, 1], [0, 3, 6]]
    finally:
        trace.cleanup()


class _MergeHeads(nn.Module):
    """GPT-2's head merge: ``permute`` back then ``view`` into the model width."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).contiguous().view(1, _SEQ, 8)


def test_head_merge_reshape_projects_one_feature_to_one_unit() -> None:
    """Feature ``h * 4 + d`` at position ``p`` reaches exactly (p, h * 4 + d)."""

    trace = tl.trace(_MergeHeads(), torch.randn(1, 2, _SEQ, 4))
    try:
        region = trace.input_ops[0].projective_field.of([(0, 1, 3, 2)])
        assert _cells(_row(region, trace.output_ops[0]).mask) == [[0, 3, 6]]
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Batched products and causal attention.
# ---------------------------------------------------------------------------


class _Product(nn.Module):
    """A batched matrix product of two inputs."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


def test_batched_matmul_keeps_rows_and_columns_per_operand() -> None:
    """``out[b, i, j]`` reads row ``i`` of the left operand and column ``j`` of the right."""

    a = torch.randn(1, 3, 4, requires_grad=True)
    b = torch.randn(1, 4, 5, requires_grad=True)
    trace = tl.trace(_Product(), (a, b), capture=_ARMED)
    try:
        out = trace.output_ops[0]
        left, right = trace.input_ops
        region = out.receptive_field.of([(0, 1, 2)])
        assert _cells(_row(region, left).mask) == [[0, 1, k] for k in range(4)]
        assert _cells(_row(region, right).mask) == [[0, k, 2] for k in range(4)]
        _all_pass(cross_validate(trace, ops=[out], units=(0, 1, 2), inputs=[left, right]))
    finally:
        trace.cleanup()


class _CausalAttention(nn.Module):
    """One head of causal self-attention, eager (masked softmax) or fused (``is_causal``)."""

    def __init__(self, *, fused: bool, width: int = 8) -> None:
        super().__init__()
        self.fused = fused
        self.q = nn.Linear(4, width)
        self.k = nn.Linear(4, width)
        self.v = nn.Linear(4, width)
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.fused:
            heads = [part.unsqueeze(1) for part in (q, k, v)]
            return functional.scaled_dot_product_attention(*heads, is_causal=True).squeeze(1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        future = torch.ones(_SEQ, _SEQ, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return weights @ v


@pytest.mark.parametrize("fused", [False, True], ids=["masked_softmax", "sdpa_is_causal"])
def test_causal_attention_narrows_to_the_prefix_in_both_directions(fused: bool) -> None:
    """Position 2 is reached by inputs 0..2 and input 3 reaches outputs 3..4; gradients agree."""

    torch.manual_seed(0)
    trace = tl.trace(_CausalAttention(fused=fused), torch.randn(1, _SEQ, 4, requires_grad=True))
    try:
        out = trace.output_ops[0]
        source = trace.input_ops[0]
        receptive = out.receptive_field.of([(0, 2, j) for j in range(8)])
        assert _positions(_row(receptive, source).mask) == [0, 1, 2]
        projective = source.projective_field.of([(0, 3, j) for j in range(4)])
        assert _positions(_row(projective, out).mask) == [3, 4]
    finally:
        trace.cleanup()
    torch.manual_seed(0)
    armed = tl.trace(
        _CausalAttention(fused=fused), torch.randn(1, _SEQ, 4, requires_grad=True), capture=_ARMED
    )
    try:
        out = armed.output_ops[0]
        _all_pass(cross_validate(armed, ops=[out], units=(0, 2, 1), inputs=list(armed.input_ops)))
    finally:
        armed.cleanup()


def test_unmasked_attention_set_query_covers_the_whole_sequence() -> None:
    """Without a mask the set query answers the whole sequence instead of refusing."""

    class _Plain(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return functional.scaled_dot_product_attention(x, x, x)

    trace = tl.trace(_Plain(), torch.randn(2, 2, 3, 2))
    try:
        region = trace.output_ops[0].receptive_field.of([(1, 0, 1, 0)])
        assert _cells(_row(region, trace.input_ops[0]).mask) == [
            [1, 0, p, d] for p in range(3) for d in range(2)
        ]
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Gradient probe through a same-object dropout payload.
# ---------------------------------------------------------------------------


def test_gradient_probe_reaches_through_eval_dropout() -> None:
    """Eval dropout returns its input; the probe uses the live tensor, not the logging clone."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 4)).eval()
    trace = tl.trace(model, torch.randn(1, 4, requires_grad=True), capture=_ARMED)
    try:
        dropout = _only_output(trace, "1")
        last = trace.output_ops[0]
        region = dropout.projective_field.of([(0, 1)], within=last.label, method="gradient")
        assert _row(region, last).num_selected == 4
        back = last.receptive_field.of([(0, 2)], within=dropout.label, method="gradient")
        assert _row(back, dropout).num_selected == 4
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Hugging Face models on tiny configs.
# ---------------------------------------------------------------------------


def _gpt2(attention: str):
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=64,
        n_positions=16,
        use_cache=True,
        attn_implementation=attention,
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    ids = torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(0))
    return model, ids


@pytest.mark.heavy
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_gpt2_geometric_set_query_is_causal(attention: str) -> None:
    """Block 0's MLP at position 3 reaches block 1's attention output at positions 3 and 4."""

    model, ids = _gpt2(attention)
    trace = tl.trace(model, ids)
    try:
        seed = _only_output(trace, "transformer.h.0.mlp.act")
        target = _only_output(trace, "transformer.h.1.attn.c_proj")
        units = [(0, 3, j) for j in range(4)]
        reached = seed.projective_field.of(units, within="transformer.h.1.attn.c_proj")
        assert _positions(_row(reached, target).mask) == [3, 4]
        logits = trace.ops[trace.modules["lm_head"].calls[0].output_ops[0]]
        back = logits.receptive_field.of([(0, 2, 0)], within="transformer.h.0.mlp")
        mlp = _only_output(trace, "transformer.h.0.mlp")
        assert _positions(_row(back, mlp).mask) == [0, 1, 2]
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_llama_style_geometric_set_query_is_causal() -> None:
    """Rotary embeddings, grouped-query attention and ``is_causal`` keep the causal prefix."""

    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=16,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(0))
    trace = tl.trace(model, ids)
    try:
        seed = _only_output(trace, "model.layers.0.mlp")
        target = _only_output(trace, "model.layers.1.self_attn.o_proj")
        reached = seed.projective_field.of([(0, 3, 0)], within="model.layers.1.self_attn.o_proj")
        assert _positions(_row(reached, target).mask) == [3, 4]
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_bert_style_geometric_set_query_covers_the_sequence() -> None:
    """Bidirectional attention: one position reaches every position of the next block."""

    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=16,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.BertModel(config).eval()
    ids = torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(0))
    trace = tl.trace(model, ids)
    try:
        seed = _only_output(trace, "encoder.layer.0.output")
        target = _only_output(trace, "encoder.layer.1.attention.output.dense")
        reached = seed.projective_field.of(
            [(0, 3, 0)], within="encoder.layer.1.attention.output.dense"
        )
        assert _positions(_row(reached, target).mask) == list(range(_SEQ))
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Index ops the transformer families reach for: ellipsis slices and gather.
# ---------------------------------------------------------------------------


def _input_ops(trace) -> list:
    """The captured input operations, in call order."""

    return [item for item in trace._layer_list if item.is_input]


class _TailSlice(nn.Module):
    """Slice the trailing axis behind an ellipsis, the rotary-embedding spelling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2]


def test_ellipsis_slice_keeps_a_one_to_one_map() -> None:
    """``x[..., :2]`` reaches one parent cell; at base the key padded twice and went unknown."""

    torch.manual_seed(0)
    trace = tl.trace(_TailSlice(), torch.randn(1, 2, 3, 4).requires_grad_(True), capture=_ARMED)
    try:
        sliced = _op(trace, "__getitem__")
        (source,) = _input_ops(trace)
        reached = sliced.receptive_field.of([(0, 1, 2, 1)], within=source.label)
        assert _cells(_row(reached, source).mask) == [[0, 1, 2, 1]]
        _all_pass(cross_validate(trace, ops=[sliced.label]))
    finally:
        trace.cleanup()


class _GatherRows(nn.Module):
    """Gather one source row per output row, the embedding-lookup spelling."""

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, index)


def test_gather_widens_only_the_gathered_axis_of_its_source() -> None:
    """The index operand stays pointwise; the source widens along the gathered axis only."""

    torch.manual_seed(0)
    index = torch.tensor([[2, 0, 1]]).unsqueeze(-1).expand(1, 3, 4).contiguous()
    source = torch.randn(1, 3, 4).requires_grad_(True)
    trace = tl.trace(_GatherRows(), (source, index), capture=_ARMED)
    try:
        gathered = _op(trace, "gather")
        source_op, index_op = _input_ops(trace)
        reached = gathered.receptive_field.of([(0, 1, 3)], within=[source_op.label, index_op.label])
        assert _cells(_row(reached, source_op).mask) == [[0, 0, 3], [0, 1, 3], [0, 2, 3]]
        assert _cells(_row(reached, index_op).mask) == [[0, 1, 3]]
    finally:
        trace.cleanup()


class _TwoBranches(nn.Module):
    """Two independent heads over one input, with no path between them."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left(x), self.right(x)


def test_gradient_probe_on_an_unreachable_target_answers_empty() -> None:
    """A sibling-branch target answers an empty mask instead of raising, as geometry does."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoBranches(), torch.randn(1, 4), capture=_ARMED)
    try:
        seed = _only_output(trace, "left")
        target = _only_output(trace, "right")
        for method in ("geometric", "gradient"):
            reached = seed.projective_field.of([(0, 0)], within="right", method=method)
            assert not bool(_row(reached, target).mask.any()), method
    finally:
        trace.cleanup()


def test_output_role_spelling_names_the_projective_target() -> None:
    """``target=`` accepts the IO role field status reports, not only the op label."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 1, 8, 8), capture=_ARMED)
    try:
        source = next(item for item in trace._layer_list if item.is_input)
        output = trace.output_ops[0]
        role = str(output.io_role or "")
        assert role and role != output.label
        by_op = source.projective_field.at((4, 4), target=output)
        by_role = source.projective_field.at((4, 4), target=role)
        assert (by_role.io_role, by_role.axes) == (by_op.io_role, by_op.axes)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Paths that merge: an unconstrained axis on one route, causal depth on both.
# ---------------------------------------------------------------------------


class _ScorePair(nn.Module):
    """Attention scores whose two operands read the same input."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x) @ self.k(x).transpose(-1, -2)


def test_two_routes_into_one_op_keep_the_unconstrained_axis() -> None:
    """One route reaches every key, the other every query, so the hull is the whole map.

    At base the per-axis union dropped the unconstrained route and answered the
    single cell where the two constrained axes met, which the gradient contains
    but does not cover.
    """

    torch.manual_seed(0)
    trace = tl.trace(_ScorePair(), torch.randn(1, _SEQ, 4).requires_grad_(True), capture=_ARMED)
    try:
        source = next(item for item in trace._layer_list if item.is_input)
        scores = trace.output_ops[0]
        reached = source.projective_field.of([(0, 3, j) for j in range(4)], within=scores.label)
        mask = _row(reached, scores).mask
        assert bool(mask.all()), _cells(mask)
        _all_pass(cross_validate(trace, ops=[scores.label]))
    finally:
        trace.cleanup()


class _TwoCausalBlocks(nn.Module):
    """Two causal attention hops in sequence, the smallest transformer depth."""

    def __init__(self) -> None:
        super().__init__()
        self.first = _CausalAttention(fused=False, width=4)
        self.second = _CausalAttention(fused=False, width=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def test_causal_narrowing_survives_a_second_attention_hop() -> None:
    """Position 3 still reaches only positions 3 and later after two causal hops.

    At base the second hop widened to every position: the route that carried the
    affected keys left the query axis unconstrained, and the masked softmax did
    not narrow it back to the rows those keys can reach.
    """

    torch.manual_seed(0)
    trace = tl.trace(
        _TwoCausalBlocks(), torch.randn(1, _SEQ, 4).requires_grad_(True), capture=_ARMED
    )
    try:
        source = next(item for item in trace._layer_list if item.is_input)
        out = trace.output_ops[0]
        reached = source.projective_field.of([(0, 3, j) for j in range(4)], within=out.label)
        assert _positions(_row(reached, out).mask) == [3, 4]
        _all_pass(cross_validate(trace, ops=[out.label]))
    finally:
        trace.cleanup()
