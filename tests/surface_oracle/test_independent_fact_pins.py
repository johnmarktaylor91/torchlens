"""Hand-pinned independent facts for the surface-oracle golden suite.

b9-opus R75-2: the surface oracle regenerates its expected snapshots THROUGH
torchlens, so it detects CHANGE, not correctness. These pins assert facts about
the committed golden FILE that are derived from the corpus model's own source
(``capture_oracle._models.PlainCNN``: ``relu(conv(x))`` with a single child
module registered under the attribute name ``conv``) — never from a
regeneration run.

Updating an expected value below requires arguing from the MODEL, not from
observed output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GOLDEN = Path(__file__).with_name("goldens") / "plain_cnn.json"


def _golden() -> dict[str, Any]:
    """Load the committed plain_cnn surface golden.

    Returns
    -------
    dict[str, Any]
        Decoded golden payload, read from the FILE (never regenerated).
    """

    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_plain_cnn_surface_golden_layers_match_first_principles() -> None:
    """Every layer-bearing stage pins the hand-derived four-node topology.

    ``PlainCNN.forward`` is exactly ``torch.relu(self.conv(x))``: one conv op,
    one relu op, plus the input and output boundary nodes — nothing else.
    """

    golden = _golden()
    expected_labels = ["conv2d_1_1", "input_1", "output_1", "relu_1_2"]
    layer_stages = [stage for stage, payload in golden.items() if "layers" in payload]
    assert layer_stages, "golden has no layer-bearing stages"
    for stage in layer_stages:
        layers = golden[stage]["layers"]
        assert sorted(layers) == expected_labels, f"stage {stage!r} layer set drifted"
        conv = layers["conv2d_1_1"]
        relu = layers["relu_1_2"]
        # Chain and module address, from the model definition alone.
        assert conv["children"] == ["relu_1_2"], f"stage {stage!r}"
        assert conv["address"] == "conv", f"stage {stage!r}"
        assert relu["parents"] == ["conv2d_1_1"], f"stage {stage!r}"
        assert relu["children"] == ["output_1"], f"stage {stage!r}"


def test_plain_cnn_surface_golden_params_match_first_principles() -> None:
    """The conv layer's recorded params are exactly conv.weight and conv.bias.

    ``nn.Conv2d`` with default ``bias=True`` owns exactly two parameters,
    named ``weight`` and ``bias``; the module is registered as ``conv``.
    """

    golden = _golden()
    layer_stages = [stage for stage, payload in golden.items() if "layers" in payload]
    assert layer_stages
    for stage in layer_stages:
        conv = golden[stage]["layers"]["conv2d_1_1"]
        labels = [entry["label"] for entry in conv["_param_logs"]]
        assert labels == ["conv.weight", "conv.bias"], f"stage {stage!r}"
