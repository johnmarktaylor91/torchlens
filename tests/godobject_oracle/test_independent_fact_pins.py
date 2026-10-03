"""Hand-pinned independent facts for the godobject-oracle golden suite.

b9-opus R75-2: the godobject goldens are regenerated THROUGH torchlens, so
they detect CHANGE, not correctness. These pins assert facts about the
committed golden FILES that are derived from first principles — the corpus
model's own source (``test_viz_identity.VizCNN``:
``head(relu(conv(x)).mean(dim=(2, 3)))``) and plain-torch parameter
arithmetic — never from a regeneration run.

Updating an expected value below requires arguing from the MODEL, not from
observed output.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

_GOLDENS = Path(__file__).with_name("goldens")

#: Hand-derived from VizCNN.forward: input -> conv -> relu -> mean -> linear
#: -> output, a single unbranched chain.
_EXPECTED_CHAIN = [
    ("input_1", "conv2d_1_1"),
    ("conv2d_1_1", "relu_1_2"),
    ("relu_1_2", "mean_1_3"),
    ("mean_1_3", "linear_1_4"),
    ("linear_1_4", "output_1"),
]


def test_viz_cnn_rolled_golden_matches_first_principles() -> None:
    """The committed rolled DOT golden pins the hand-derived VizCNN facts."""

    source = (_GOLDENS / "viz_viz_cnn_rolled.gv").read_text(encoding="utf-8")
    for src, dst in _EXPECTED_CHAIN:
        assert f"{src} -> {dst}" in source, f"missing hand-derived edge {src} -> {dst}"
    # Parameter count from plain torch on the same constructors VizCNN uses:
    # Conv2d(1, 2, 3) weight 2*1*3*3=18 + bias 2, Linear(2, 3) weight 6 +
    # bias 3 -> 29 total.
    plain = nn.ModuleDict(
        {
            "conv": nn.Conv2d(1, 2, kernel_size=3, padding=1),
            "head": nn.Linear(2, 3),
        }
    )
    expected_params = sum(parameter.numel() for parameter in plain.parameters())
    assert expected_params == 29
    assert f"{expected_params} params" in source


def test_viz_cnn_unrolled_golden_matches_first_principles() -> None:
    """The unrolled DOT golden pins the same single-pass chain.

    VizCNN calls every module exactly once, so the unrolled graph is the same
    unbranched chain as the rolled one.
    """

    source = (_GOLDENS / "viz_viz_cnn_unrolled.gv").read_text(encoding="utf-8")
    for src, dst in _EXPECTED_CHAIN:
        assert f"{src}pass1 -> {dst}pass1" in source, (
            f"missing hand-derived edge {src}pass1 -> {dst}pass1"
        )


def test_viz_cnn_input_shape_fact_matches_plain_torch() -> None:
    """The rolled golden's conv output shape follows from plain torch.

    The oracle feeds a ``(1, 1, 4, 4)`` input; ``Conv2d(1, 2, 3, padding=1)``
    preserves H and W and emits 2 channels, so the conv node must carry
    ``(1, 2, 4, 4)`` — computed here with plain torch, never with torchlens.
    """

    with torch.no_grad():
        conv_out = nn.Conv2d(1, 2, kernel_size=3, padding=1)(torch.zeros(1, 1, 4, 4))
    assert tuple(conv_out.shape) == (1, 2, 4, 4)
    source = (_GOLDENS / "viz_viz_cnn_rolled.gv").read_text(encoding="utf-8")
    assert "(1, 2, 4, 4)" in source
