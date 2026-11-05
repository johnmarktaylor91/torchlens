"""``decode_output`` on an intervened trace names the intervened answer.

A fork copies the parent's captured decoded rows; after ``do()`` propagates
an edit to the output those rows would still name the clean class. Replay
and rerun re-decode from the retained output.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _Config:
    def __init__(self, labels: dict[int, str]) -> None:
        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)
        self.model_type = "tiny"
        self.architectures = ["TinyForSequenceClassification"]


class _Classifier(nn.Module):
    """Logits are the input passed through a linear map: an edit upstream moves the answer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3))
        self.config = _Config({0: "zero", 1: "one", 2: "two"})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(x))


def _top_label(trace: tl.Trace) -> str:
    rows = trace.decode_output(k=1)["rows"]
    return str(rows[0]["label"])


@pytest.fixture
def clean() -> tl.Trace:
    model = _Classifier().eval()
    x = torch.tensor([[0.1, 0.2, 3.0]])
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def test_clean_rows_name_the_clean_class(clean: tl.Trace) -> None:
    assert _top_label(clean) == "two"


def test_replayed_fork_decodes_its_own_output(clean: tl.Trace) -> None:
    fork = clean.fork()
    fork.do(fork[tl.func("relu")], tl.zero_ablate())
    assert fork.state.value == "replay_propagated"
    assert torch.equal(fork["fc"].out, torch.zeros(1, 3))
    rows = fork.decode_output()["rows"]
    assert all(row["prob"] == pytest.approx(1 / 3) for row in rows)
    assert _top_label(clean) == "two"


def test_rerun_fork_decodes_its_own_output(clean: tl.Trace) -> None:
    model = _Classifier().eval()
    plain = tl.trace(model, torch.tensor([[0.1, 0.2, 3.0]]))
    fork = plain.fork()
    fork.do(
        fork[tl.func("relu")], tl.scale(0.0), model=model, inputs=torch.tensor([[0.1, 0.2, 3.0]])
    )
    assert fork.state.value == "rerun_propagated"
    rows = fork.decode_output()["rows"]
    assert all(row["prob"] == pytest.approx(1 / 3) for row in rows)


def test_decode_output_tracks_a_patch_that_changes_the_class(clean: tl.Trace) -> None:
    donor = tl.trace(
        _Classifier().eval(),
        torch.tensor([[5.0, 0.0, 0.0]]),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    assert _top_label(donor) == "zero"
    fork = clean.fork()
    fork.do(fork[tl.func("relu")], tl.patch_from(donor))
    assert _top_label(fork) == "zero"
    assert _top_label(clean) == "two"
