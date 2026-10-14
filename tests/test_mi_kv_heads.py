"""Per-query-head patching under grouped-query attention (the k/v interceptor).

A tiny random-init Llama (4 query heads over 2 KV groups, eager attention, no
download) is the subject. The oracle is an independent hand-written
reference: it swaps ``transformers``' own ``repeat_kv`` for a version that
splices the clean copy into one query head, so it never touches TorchLens's
hooks, selectors or traces. Grid cells must equal it exactly.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("matplotlib")

import transformers.models.llama.modeling_llama as llama_modeling  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import torchlens as tl  # noqa: E402
import torchlens.mechinterp as mi  # noqa: E402
import torchlens.tviz as tviz  # noqa: E402
from torchlens.errors._base import TorchLensWarning  # noqa: E402
from torchlens.mechinterp import MechInterpError  # noqa: E402
from torchlens.mechinterp._kv_heads import _patch_query_heads_once  # noqa: E402
from torchlens.semantic.patching import _trace_kwargs  # noqa: E402

CLEAN = torch.tensor([[1, 5, 9, 3, 7, 2]])
CORRUPTED = torch.tensor([[1, 6, 9, 3, 8, 2]])
LAYERS = ("model.layers.0.self_attn", "model.layers.1.self_attn")
ANSWER, OTHER = 3, 4


def _llama(attn: str = "eager", n_kv: int = 2) -> LlamaForCausalLM:
    """A seeded two-layer Llama with 4 query heads (no download)."""

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=n_kv,
        max_position_embeddings=32,
        attn_implementation=attn,
    )
    return LlamaForCausalLM(config).eval()


def _from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Last-position logit difference between two fixed tokens."""

    return logits[0, -1, ANSWER] - logits[0, -1, OTHER]


def logit_diff(trace: Any) -> torch.Tensor:
    """The same metric read from a trace's captured logits."""

    head = next(
        module
        for module in trace.modules
        if module.facets is not None and "logits" in list(module.facets.keys())
    )
    return _from_logits(head.facets["logits"].value)


def _trace_metric(model: LlamaForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    """Metric of one plain capture."""

    log = tl.trace(model, ids, **_trace_kwargs(None))
    try:
        return logit_diff(log).detach()
    finally:
        log.cleanup()


@contextmanager
def _repeat_kv_swapped(replacement: Any) -> Iterator[None]:
    """Swap ``transformers``' module-level ``repeat_kv`` for the block."""

    original = llama_modeling.repeat_kv
    llama_modeling.repeat_kv = replacement
    try:
        yield
    finally:
        llama_modeling.repeat_kv = original


def _reference_cell(
    model: LlamaForCausalLM, *, site: str, layer: int, heads: tuple[int, ...]
) -> torch.Tensor:
    """Hand-written oracle: splice clean copies of ``heads`` at one repeat_kv call."""

    original = llama_modeling.repeat_kv
    clean_copies: list[torch.Tensor] = []

    def record(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
        out = original(hidden, n_rep)
        clean_copies.append(out.detach().clone())
        return out

    with torch.no_grad(), _repeat_kv_swapped(record):
        model(CLEAN)
    # eager_attention_forward expands the key, then the value, once per layer.
    target = 2 * layer + (1 if site == "v" else 0)
    calls = itertools.count()

    def splice(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
        out = original(hidden, n_rep)
        if next(calls) != target:
            return out
        patched = out.clone()
        patched[:, list(heads)] = clean_copies[target][:, list(heads)]
        return patched

    with torch.no_grad(), _repeat_kv_swapped(splice):
        logits = model(CORRUPTED).logits
    return _from_logits(logits).to(torch.float32)


@pytest.fixture(scope="module")
def model() -> LlamaForCausalLM:
    """The shared eager GQA model."""

    return _llama()


@pytest.fixture(scope="module")
def value_grid(model: LlamaForCausalLM) -> mi.PatchGrid:
    """The value-site query-head grid, computed once for the module."""

    return mi.patch_query_heads_grid(model, CLEAN, CORRUPTED, logit_diff, site="v")


@pytest.mark.parametrize("site", ["v", "k"])
def test_every_cell_equals_the_repeat_kv_reference(
    model: LlamaForCausalLM, value_grid: mi.PatchGrid, site: str
) -> None:
    """Each [layer, query head] cell equals the hand-written repeat_kv splice exactly."""

    grid = (
        value_grid
        if site == "v"
        else mi.patch_query_heads_grid(model, CLEAN, CORRUPTED, logit_diff, site="k")
    )
    assert grid.axes == ("layer", "head")
    assert grid.coordinates == {"layer": LAYERS, "head": (0, 1, 2, 3)}
    for row, col in itertools.product(range(2), range(4)):
        expected = _reference_cell(model, site=site, layer=row, heads=(col,))
        assert torch.equal(grid.values[row, col], expected), (site, row, col)
    corrupted = _trace_metric(model, CORRUPTED).to(torch.float32)
    assert not torch.equal(grid.values, corrupted.expand_as(grid.values))


def test_group_heads_together_equal_the_kv_group_cell(
    model: LlamaForCausalLM, value_grid: mi.PatchGrid
) -> None:
    """All query heads of group g together reproduce the kv-group cell; one alone does not."""

    group_grid = mi.grid(model, CLEAN, CORRUPTED, logit_diff, sites="v", axes=("layer", "head"))
    assert group_grid.coordinates["head"] == (0, 1)
    for row, address in enumerate(LAYERS):
        for group in (0, 1):
            members = (2 * group, 2 * group + 1)
            together = _patch_query_heads_once(
                model, CLEAN, CORRUPTED, logit_diff, layer=address, heads=members
            )
            assert together == float(group_grid.values[row, group]), (address, group)
            assert together == float(_reference_cell(model, site="v", layer=row, heads=members))
            for head in members:
                assert float(value_grid.values[row, head]) != together, (address, head)


def test_one_fire_per_cell_and_clean_into_clean_moves_nothing(
    model: LlamaForCausalLM, value_grid: mi.PatchGrid
) -> None:
    """Every cell fires once; patching clean into clean leaves the metric exactly unmoved."""

    receipts = value_grid.receipts
    assert receipts["n_cells"] == 8
    assert receipts["fires"] == 8
    assert receipts["identical_fires"] < 8
    assert receipts["effect_site"] == (
        "expanded value copy read by one query head (op-level interceptor)"
    )
    assert receipts["n_q_heads"] == 4
    assert receipts["n_kv_heads"] == 2
    assert set(receipts["expansion_ops"]) == set(LAYERS)

    with pytest.warns(TorchLensWarning, match="IDENTICAL"):
        control = mi.patch_query_heads_grid(model, CLEAN, CLEAN, logit_diff)
    clean = _trace_metric(model, CLEAN).to(torch.float32)
    assert control.receipts["fires"] == 8
    assert control.receipts["identical_fires"] == 8
    assert torch.equal(control.values - clean, torch.zeros_like(control.values))


def test_sdpa_gqa_refuses_kv_expansion_unavailable() -> None:
    """SDPA with enable_gqa expands inside the kernel: no query head's copy exists."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.patch_query_heads_grid(_llama("sdpa"), CLEAN, CORRUPTED, logit_diff)
    fields = excinfo.value.fields
    assert fields["code"] == "mi_kv_expansion_unavailable"
    assert 'attn_implementation="eager"' in fields["remedy"]
    assert '"torchlens" attention backend' in fields["remedy"]


def test_mha_refuses_kv_heads_not_grouped() -> None:
    """Equal query and KV head counts share nothing; the remedy is the per-head grid."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.patch_query_heads_grid(_llama(n_kv=4), CLEAN, CORRUPTED, logit_diff)
    assert excinfo.value.fields["code"] == "mi_kv_heads_not_grouped"
    assert "patch_heads_grid" in excinfo.value.fields["remedy"]


def test_two_expansions_refuse_kv_expansion_ambiguous(model: LlamaForCausalLM) -> None:
    """Two separate expansion ops of one tensor: patching one would miss the other."""

    original = llama_modeling.repeat_kv

    def twice(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
        return (original(hidden, n_rep) + original(hidden, n_rep)) / 2

    with _repeat_kv_swapped(twice), pytest.raises(MechInterpError) as excinfo:
        mi.patch_query_heads_grid(model, CLEAN, CORRUPTED, logit_diff)
    assert excinfo.value.fields["code"] == "mi_kv_expansion_ambiguous"
    assert len(excinfo.value.fields["candidates"]) == 2


def test_interceptor_receipt_discloses_effect_source_and_annotates(
    model: LlamaForCausalLM, value_grid: mi.PatchGrid, tmp_path: Path
) -> None:
    """An interceptor grid is an ordinary query-head receipt that names its op."""

    clean = float(_trace_metric(model, CLEAN))
    corrupted = float(_trace_metric(model, CORRUPTED))
    joint = mi.grid(model, CLEAN, CORRUPTED, logit_diff, sites="v", axes=("layer",))
    receipt = tviz.receipt_from_patch_grid(
        value_grid,
        layer=LAYERS[0],
        negative_control=0.0,
        positive_control=clean - corrupted,
        joint_effect=float(joint.values[0]),
        intervention="clean value copy patched into one query head",
    )
    assert receipt.unit == "head"
    assert receipt.heads == (0, 1, 2, 3)
    assert receipt.effect_source == (
        "expanded value copy read by one query head (op-level interceptor)"
    )
    artifact = tviz.render_receipt_grid(receipt, tmp_path / "query_heads.png")
    assert (
        "pattern source: captured; effect source: expanded value copy read by one query "
        "head (op-level interceptor)"
    ) in artifact.disclosure_lines
    annotation = tviz.Annotation.from_receipt(receipt)
    assert annotation.heads == (0, 1, 2, 3)
    assert annotation.values == receipt.effects

    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.receipt_from_patch_grid(
            value_grid,
            layer=LAYERS[0],
            negative_control=0.0,
            positive_control=clean - corrupted,
            kv_groups=tviz.GqaInfo(n_query_heads=4, n_kv_heads=2),
        )
    assert excinfo.value.fields["code"] == "tv_receipt_invalid"


def test_real_gqa_value_grid_builds_a_kv_group_receipt(
    model: LlamaForCausalLM, tmp_path: Path
) -> None:
    """A real GQA model's value-site grid becomes a kv-group receipt naming each group."""

    log = tl.trace(model, CLEAN, **_trace_kwargs(None))
    facets = log.modules[LAYERS[0]].facets
    gqa = tviz.GqaInfo(n_query_heads=facets.get("n_q_heads"), n_kv_heads=facets.get("n_kv_heads"))
    log.cleanup()
    group_grid = mi.grid(model, CLEAN, CORRUPTED, logit_diff, sites="v", axes=("layer", "head"))
    joint = mi.grid(model, CLEAN, CORRUPTED, logit_diff, sites="v", axes=("layer",))
    receipt = tviz.receipt_from_patch_grid(
        group_grid,
        layer=LAYERS[1],
        negative_control=0.0,
        positive_control=float(_trace_metric(model, CLEAN) - _trace_metric(model, CORRUPTED)),
        joint_effect=float(joint.values[1]),
        intervention="kv group value patched",
        kv_groups=gqa,
    )
    assert receipt.unit == "kv group"
    assert receipt.heads == (0, 1)
    assert receipt.effects == tuple(float(value) for value in group_grid.values[1])
    assert receipt.effect_source == "capture route"
    assert receipt.sharing_lines() == (
        "kv group 1 of 2, shared by query heads 0-1",
        "kv group 2 of 2, shared by query heads 2-3",
    )
    artifact = tviz.render_receipt_grid(receipt, tmp_path / "kv_groups.png")
    assert any("no cell is a per-query-head effect" in line for line in artifact.disclosure_lines)
