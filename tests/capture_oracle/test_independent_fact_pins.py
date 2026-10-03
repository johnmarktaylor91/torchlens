"""Hand-pinned independent facts for the capture-oracle golden suite.

b9-opus R75-2: golden suites regenerate expected output THROUGH the code under
test, so they detect CHANGE, not correctness. These pins assert facts about the
committed golden FILE that are derived from first principles — the model's own
source (``_models.PlainCNN``: ``relu(conv(x))`` with ``Conv2d(1, 2, 3,
padding=1)``) and plain torch — never from a regeneration run. If capture or
the characterizer's shared adapter starts lying, the goldens re-pin around it;
these do not.

Updating an expected value below requires arguing from the MODEL, not from
observed output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ._models import PlainCNN, build_model_case

_GOLDEN = Path(__file__).with_name("goldens") / "plain_cnn__exhaustive.json"


def _golden_record() -> dict[str, Any]:
    """Load the committed plain_cnn exhaustive golden record.

    Returns
    -------
    dict[str, Any]
        The golden's ``record`` payload, read from the FILE (never regenerated).
    """

    return json.loads(_GOLDEN.read_text(encoding="utf-8"))["record"]


def test_plain_cnn_golden_events_match_first_principles() -> None:
    """The golden's ground-truth events pin the hand-derived op sequence.

    ``PlainCNN.forward`` is exactly ``torch.relu(self.conv(x))``: two compute
    ops, conv2d then relu, with the relu consuming the conv output — written
    down here from the model source, not from any capture run.
    """

    events = _golden_record()["ground_truth"]["events"]
    assert [event["identity"]["func_name"] for event in events] == ["conv2d", "relu"]
    conv_event, relu_event = events
    # The conv consumes only the model input; the relu only the conv output.
    assert len(conv_event["identity"]["parent_labels_raw"]) == 1
    assert relu_event["identity"]["parent_labels_raw"] == [conv_event["identity"]["label_raw"]]
    # Conv2d(1, 2, kernel_size=3, padding=1) preserves H and W and emits 2
    # channels; the expected shape comes from plain torch on the corpus input.
    model, model_input = build_model_case("plain_cnn")
    with torch.no_grad():
        expected_shape = list(model.conv(model_input).shape)
    assert conv_event["identity"]["shape"] == expected_shape
    assert relu_event["identity"]["shape"] == expected_shape
    # The conv op consumed the module's two parameters (weight + bias),
    # counted on the plain nn.Module itself; the relu consumed none. The
    # golden's population projection flattens each consumed ParamRef to
    # ``params[i].*`` paths, so the distinct indexes count the parameters.
    expected_param_count = len(list(PlainCNN().conv.parameters()))
    assert expected_param_count == 2
    conv_param_indexes = {
        path.split("].", 1)[0] for path in conv_event["population"] if path.startswith("params[")
    }
    assert conv_param_indexes == {f"params[{i}" for i in range(expected_param_count)}
    assert relu_event["population"].get("params") == "defaulted_empty"


def test_plain_cnn_golden_final_ops_match_first_principles() -> None:
    """The golden's final topology pins the hand-derived four-node chain."""

    final_ops = _golden_record()["ground_truth"]["final_ops"]
    by_label = {row["label"]: row for row in final_ops}
    assert sorted(by_label) == ["conv2d_1_1", "input_1", "output_1", "relu_1_2"]
    assert by_label["input_1"]["children"] == ["conv2d_1_1"]
    assert by_label["conv2d_1_1"]["parents"] == ["input_1"]
    assert by_label["conv2d_1_1"]["children"] == ["relu_1_2"]
    assert by_label["relu_1_2"]["parents"] == ["conv2d_1_1"]
    assert by_label["relu_1_2"]["children"] == ["output_1"]
    assert by_label["output_1"]["parents"] == ["relu_1_2"]
    assert by_label["conv2d_1_1"]["func_name"] == "conv2d"
    assert by_label["relu_1_2"]["func_name"] == "relu"
