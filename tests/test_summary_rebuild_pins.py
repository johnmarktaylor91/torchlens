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


@pytest.mark.slow
def test_inception_v3_declared_executed_split() -> None:
    """inception_v3 eval: 27,161,264 declared / 23,834,568 executed /
    3,326,696 never ran (AuxLogits.*) -- A3's canonical case."""

    pytest.importorskip("torchvision")
    pins = derive_pins("inception_v3")
    assert pins["params_declared"] == 27_161_264
    assert pins["params_executed"] == 23_834_568
    assert pins["params_unexecuted"] == 3_326_696
