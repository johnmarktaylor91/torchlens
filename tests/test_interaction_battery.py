"""Interaction battery: common pairs of features, exercised together on one tiny model.

Each test composes two features and checks the result against a closed-form oracle (the
tiny model's weights are fixed, so every expected value is computed by hand). The
interaction matrix (``tests/_interaction_matrix.py``) cites these tests as the evidence
for its ``composes`` cells; ``tests/test_interaction_matrix_totality.py`` checks that every
cited test exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class LinearReluLinear(nn.Module):
    """Two fixed linear sites around a relu, so every edit has a closed-form result."""

    def __init__(self) -> None:
        """Initialize deterministic weights: identity then a row sum plus one."""

        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(4))
            self.encoder.bias.fill_(1.0)
            self.head.weight.fill_(1.0)
            self.head.bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder, relu and head."""

        return self.head(torch.relu(self.encoder(x)))


X = torch.tensor([[1.0, -3.0, 0.5, 2.0]])
X2 = torch.tensor([[2.0, 0.0, -5.0, 1.0]])


def _relu_of(x: torch.Tensor) -> torch.Tensor:
    """Closed-form relu activation of the tiny model."""

    return torch.relu(x + 1.0)


def _output_of(relu: torch.Tensor) -> torch.Tensor:
    """Closed-form head output for a relu activation."""

    return relu.sum(dim=-1, keepdim=True) + 1.0


def _relu_op(log: Any) -> Any:
    """Return the relu op of a trace."""

    return next(op for op in log.layer_list if op.type == "relu")


ZERO_RELU = tl.when(tl.func("relu"), tl.zero_ablate())


def test_save_x_intervene_saves_the_edited_value() -> None:
    """A site that is both saved and edited saves the edited activation."""

    log = tl.trace(LinearReluLinear(), X, save=tl.func("relu"), intervene=ZERO_RELU)
    relu = _relu_op(log)
    assert torch.equal(relu.out, torch.zeros(1, 4))
    assert relu.is_edited


def test_save_x_rerun_refills_the_selected_sites() -> None:
    """``run(model, x)`` re-arms the save selector on the new input."""

    model = LinearReluLinear()
    log = tl.trace(model, X, save=tl.func("relu"))
    log.run(model, X2)
    assert torch.equal(_relu_op(log).out, _relu_of(X2))


def test_intervene_x_rerun_applies_the_edit_once() -> None:
    """A staged edit is applied again on rerun, exactly once, and the spec does not grow."""

    model = LinearReluLinear()
    log = tl.trace(model, X, save=tl.func("relu"), intervene=ZERO_RELU)
    log.run(model, X2)
    relu = _relu_op(log)
    assert torch.equal(relu.out, torch.zeros(1, 4))
    assert len(relu.interventions) == 1


def test_fork_do_x_index_edits_the_fork_only() -> None:
    """``fork().do`` on an indexed op edits the fork and leaves the parent intact."""

    log = tl.trace(LinearReluLinear(), X, capture=CaptureOptions(replayable=True))
    label = _relu_op(log).label
    fork = log.fork()
    fork.do(fork[label], tl.zero_ablate())
    assert torch.equal(fork[label].out, torch.zeros(1, 4))
    assert torch.equal(log[label].out, _relu_of(X))


def test_record_x_intervene_returns_the_edited_output() -> None:
    """``tl.record`` with ``intervene=`` returns the edited output and records the site."""

    output, recording = tl.record(
        LinearReluLinear(), X, save=tl.func("relu"), intervene=ZERO_RELU, return_output=True
    )
    assert torch.equal(output, torch.ones(1, 1))
    assert len(recording.records) == 1


def test_bind_x_intervene_matches_the_traced_edit() -> None:
    """The capture-free executor and a full trace apply the same edit."""

    model = LinearReluLinear()
    with torch.no_grad():
        bound = ZERO_RELU.bind(model)(X)
    traced = tl.trace(model, X, save=["head"], intervene=ZERO_RELU)
    assert torch.equal(bound, traced["head"].out)
    assert torch.equal(bound, torch.ones(1, 1))


def test_record_x_save_matches_the_trace() -> None:
    """A sparse recording materialized to a trace holds the same saved activation."""

    model = LinearReluLinear()
    recording = tl.record(model, X, save=tl.func("relu"))
    traced = tl.trace(model, X, save=tl.func("relu"))
    assert torch.equal(_relu_op(recording.to_trace()).out, _relu_op(traced).out)


def test_extract_x_save_reads_the_same_module_outputs() -> None:
    """``tl.extract`` and a list ``save=`` resolve module names to the same outputs."""

    model = LinearReluLinear()
    stages = tl.extract(model, X, ["encoder", "head"])
    log = tl.trace(model, X, save=["encoder", "head"])
    assert torch.equal(_as_tensor(stages["encoder"]), log["encoder"].out)
    assert torch.equal(log["head"].out, _output_of(_relu_of(X)))


def _as_tensor(value: Any) -> torch.Tensor:
    """Unwrap an extraction value into a tensor."""

    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def test_save_x_storage_reads_back_the_saved_values(tmp_path: Path) -> None:
    """Saved activations streamed to disk read back equal to the in-memory trace."""

    model = LinearReluLinear()
    disk = tl.trace(model, X, save=tl.func("relu"), storage=tl.to_disk(tmp_path / "run.tlspec"))
    assert torch.equal(_relu_op(disk).out, _relu_of(X))
