"""F08 real-model pins, derived by the ONE committed script at zero network.

Summary memo section 5 (the canonical pin table): the constants below are
the memo's build-time pins; tests re-derive them through
``tools/derive_summary_pins.py`` so the tests and the script share one
identity-partition basis. HF architectures are config-built (gpt2-124M IS
``GPT2Config()``'s architecture) and torchvision models load
``weights=None`` -- counts, ties, FLOPs, and row counts are architecture
facts independent of weight values.

Cost tier: real-architecture captures run seconds each; the file is
``heavy`` (5-20 s budget) with the gpt2 flagship marked ``slow``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

_TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from derive_summary_pins import derive_pins  # noqa: E402


@pytest.mark.slow
def test_gpt2_flagship_pins() -> None:
    """gpt2: declared 124,439,808 (tie named), true fwd FLOPs at T=16,
    default = FULL folded tree with nothing hidden (memo section 5)."""

    pytest.importorskip("transformers")
    pins = derive_pins("gpt2")
    assert pins["params_declared"] == 124_439_808
    assert pins["params_per_path"] == 163_037_184
    assert pins["tied_groups"] == [["transformer.wte.weight", "lm_head.weight"]]
    # The identity-partition constant: label-sweep 3,974,578,192 + the
    # 147,456 residue the sweep missed (memo section 4, FABLE r4 V1d).
    assert pins["flops_forward_fma2"] == 3_974_725_648
    assert pins["default_rung"] == "tree"
    assert pins["default_depth"] == max(pins["tree_rows_by_depth"])  # FULL tree
    assert pins["default_rows"] == 19
    assert pins["default_folds"] >= 1  # h.0..11 folds


def test_vgg_budget_anchors() -> None:
    """vgg16 hybrid 42 rows / vgg19 hybrid 48 = the budget, exactly."""

    pytest.importorskip("torchvision")
    vgg16 = derive_pins("vgg16")
    assert vgg16["hybrid_rows"] == 42
    assert vgg16["compute_ops"] == 40  # incl. the orphan torch.flatten
    assert vgg16["default_rung"] == "hybrid"
    vgg19 = derive_pins("vgg19")
    assert vgg19["hybrid_rows"] == 48
    assert vgg19["compute_ops"] == 46
    assert vgg19["default_rung"] == "hybrid"
    assert vgg19["default_rows"] == 48


def test_resnet50_ladder_pins() -> None:
    """resnet50: 25,557,032 params; ladder 10/18/78/86; default d2 = 18."""

    pytest.importorskip("torchvision")
    pins = derive_pins("resnet50")
    assert pins["params_declared"] == 25_557_032
    assert pins["tree_rows_by_depth"] == {1: 10, 2: 18, 3: 78, 4: 86}
    assert pins["default_rung"] == "tree"
    assert pins["default_depth"] == 2
    assert pins["default_rows"] == 18


@pytest.mark.slow
def test_densenet121_floor_pin() -> None:
    """densenet121: the gate minimum default; >= min(8, deepest) floor."""

    pytest.importorskip("torchvision")
    pins = derive_pins("densenet121")
    assert pins["default_rows"] >= 8
    assert pins["default_rung"] in ("tree", "elided")


@pytest.mark.slow
def test_inception_v3_declared_executed_split() -> None:
    """inception_v3 eval: 27,161,264 declared / 23,834,568 executed /
    3,326,696 never ran (AuxLogits.*) -- A3's canonical case."""

    pytest.importorskip("torchvision")
    pins = derive_pins("inception_v3")
    assert pins["params_declared"] == 27_161_264
    assert pins["params_executed"] == 23_834_568
    assert pins["params_unexecuted"] == 3_326_696


def test_bert_base_exact_macs() -> None:
    """bert-base at 12 tokens: exact MACs (never the killed flops//2).

    The memo's 1,019,805,696 is the 73 fused-bias LINEAR ops' MAC sum
    (12 layers x 6 linears + pooler); whole-forward MACs additionally
    count the attention score/value matmuls -- exactly
    12 tokens x 12 layers x 2 matmuls x 12 tokens x 768 = 2,654,208.
    Both terms are closed-form; their sum is the partition total.
    """

    pytest.importorskip("transformers")
    pins = derive_pins("bert-base")
    linear_macs = 1_019_805_696
    attention_matmul_macs = 12 * 12 * 2 * 12 * 768
    assert pins["macs_forward"] == linear_macs + attention_matmul_macs == 1_022_459_904
    assert pins["flops_forward_fma2"] > 2 * pins["macs_forward"]  # bias/norm terms exist
