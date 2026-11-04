"""``transformed_output`` never describes a forward the trace no longer holds.

``output_transform`` runs on the model's return value. A rerun re-runs the
model and re-applies it; a replay recomputes ops from captured values and has
no model return to apply it to. After a replay rewrites an output op's values,
the inherited transform result describes the parent's forward, so reading it
refuses typed, the same way the fork's ``trace.output`` does. A replay that
leaves every output op's values unchanged keeps the inherited result.
``copy.deepcopy`` and pickle keep the refusal; a portable save stores no
transformed output for such a trace, so the loaded trace reads ``None``.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _TinyClassifier(nn.Module):
    """Two linear layers; zero-ablating ``hidden`` leaves only the head bias."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 3, bias=False)
        self.head = nn.Linear(3, 3)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
            self.head.weight.copy_(torch.eye(3) * 2.0)
            self.head.bias.copy_(torch.tensor([0.0, 0.5, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(torch.relu(self.hidden(x)))


def _times_ten(output: torch.Tensor) -> torch.Tensor:
    return output * 10.0


def _capture(**capture: Any) -> tuple[nn.Module, torch.Tensor, Any]:
    model = _TinyClassifier().eval()
    x = torch.ones(1, 2)
    options = tl.options.CaptureOptions(output_transform=_times_ten, **capture)
    trace = tl.trace(model, x, capture=options)
    return model, x, trace


def _fork_do(trace: Any, model: nn.Module, x: torch.Tensor, edit: Any, engine: str) -> Any:
    options = tl.options.InterventionOptions(engine=engine)
    return trace.fork().do(tl.module("hidden"), edit, model=model, inputs=x, intervention=options)


def test_replay_that_changes_an_output_op_refuses_transformed_output() -> None:
    """The parent's transform result is not served as the edited fork's."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.zero_ablate(), "replay")

    assert not torch.allclose(edited[out].out, trace[out].out)
    with pytest.raises(PayloadUnavailableError) as caught:
        _ = edited.transformed_output
    assert caught.value.fields["code"] == "output_not_stored"
    # The parent keeps its own, still-true transform result.
    assert torch.allclose(trace.transformed_output, trace[out].out * 10.0)


def test_replay_that_changes_no_output_op_keeps_transformed_output() -> None:
    """An edit whose replay leaves the outputs unchanged keeps the inherited result."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.scale(1.0), "replay")

    assert torch.equal(edited[out].out, trace[out].out)
    assert torch.equal(edited.transformed_output, trace.transformed_output)


def test_rerun_recomputes_transformed_output_from_the_edited_forward() -> None:
    """The rerun engine re-applies ``output_transform`` to the edited return value."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.zero_ablate(), "rerun")

    assert not torch.allclose(edited[out].out, trace[out].out)
    assert torch.allclose(edited.transformed_output, edited[out].out * 10.0)
    assert torch.allclose(trace.transformed_output, trace[out].out * 10.0)


def _stale_fork(**capture: Any) -> tuple[Any, Any]:
    model, x, trace = _capture(**capture)
    return trace, _fork_do(trace, model, x, tl.zero_ablate(), "replay")


def _assert_refuses(trace: Any) -> None:
    with pytest.raises(PayloadUnavailableError) as caught:
        _ = trace.transformed_output
    assert caught.value.fields["code"] == "output_not_stored"
    assert "replay rewrote the model output" in str(caught.value)
    assert "rerun" in str(caught.value)


def test_refusal_names_the_replay_and_the_rerun_remedy() -> None:
    """The refusal says why (a replay rewrite) and how to recompute (the rerun engine)."""

    _assert_refuses(_stale_fork()[1])


def test_deepcopy_of_a_stale_fork_still_refuses() -> None:
    """``copy.deepcopy`` keeps the stale marker: the copy refuses too."""

    _assert_refuses(copy.deepcopy(_stale_fork()[1]))


def test_pickle_round_trip_of_a_stale_fork_still_refuses() -> None:
    """The marker pickles by reference, so an unpickled stale fork refuses."""

    _assert_refuses(pickle.loads(pickle.dumps(_stale_fork()[1])))


@pytest.mark.parametrize("save_policy", ["small", True])
def test_saved_stale_fork_loads_with_no_transformed_output(
    tmp_path: Path, save_policy: str | bool
) -> None:
    """A portable save writes no transformed output for a stale fork (``None`` on load).

    ``save_transformed_output=True`` saves the value whole, so only the save
    path's own stale check keeps the marker out of the bundle.
    """

    parent, edited = _stale_fork(save_transformed_output=save_policy)
    bundle = tmp_path / "stale_fork.tlspec"
    edited.save(str(bundle))
    assert tl.load(str(bundle)).transformed_output is None
    _assert_refuses(edited)
    assert parent.transformed_output is not None


def test_unchanged_output_fork_keeps_its_value_through_copy_pickle_and_save(
    tmp_path: Path,
) -> None:
    """A replay that changed no output op keeps the parent's value everywhere it travels."""

    model, x, trace = _capture()
    edited = _fork_do(trace, model, x, tl.scale(1.0), "replay")
    expected = trace.transformed_output
    assert torch.equal(copy.deepcopy(edited).transformed_output, expected)
    assert torch.equal(pickle.loads(pickle.dumps(edited)).transformed_output, expected)
    bundle = tmp_path / "unchanged_fork.tlspec"
    edited.save(str(bundle))
    assert torch.equal(tl.load(str(bundle)).transformed_output, expected)


def test_aborted_selection_batch_restores_the_transform_result() -> None:
    """A batch that rewrites the output then refuses rolls the stored result back too."""

    capture = tl.options.CaptureOptions(output_transform=_times_ten, replayable=True)
    trace = tl.trace(_TinyClassifier().eval(), torch.ones(1, 2), capture=capture)
    fork = trace.fork()
    expected = fork.transformed_output
    head_op = fork[fork.output_layers[0]].parents[0]

    def _wrong_shape(tensor: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        return tensor[:1]

    pairs = [
        (fork.layers[head_op].__selection__(), tl.scale(2.0)),
        (tl.param("head.weight"), _wrong_shape),
    ]
    with pytest.raises(Exception) as caught:
        fork.do(pairs)
    assert "code" in getattr(caught.value, "fields", {}), "a typed refusal"
    assert torch.equal(fork.transformed_output, expected)


def test_fork_of_a_stale_fork_still_refuses() -> None:
    """Forking a stale fork carries the refusal; forking the parent does not."""

    parent, edited = _stale_fork()
    _assert_refuses(edited.fork())
    assert torch.equal(parent.fork().transformed_output, parent.transformed_output)


def test_cleanup_of_a_stale_fork_succeeds() -> None:
    """Releasing a stale fork's fields does not trip over the refusing getter."""

    _parent, edited = _stale_fork()
    edited.cleanup()
    assert "transformed_output" not in edited.__dict__


def test_fork_output_refusal_names_the_fork_not_a_disk_load() -> None:
    """``trace.output`` on a replay fork says it is a fork; an unpickled copy keeps the disk text."""

    _parent, edited = _stale_fork()
    with pytest.raises(PayloadUnavailableError) as caught:
        _ = edited.output
    message = str(caught.value)
    assert caught.value.fields["code"] == "output_not_stored"
    assert "fork" in message
    assert "replay" in message
    assert "loaded from disk" not in message
    assert "rerun" in message
    with pytest.raises(PayloadUnavailableError) as unpickled:
        _ = pickle.loads(pickle.dumps(edited)).output
    assert unpickled.value.fields["code"] == "output_not_stored"
    assert "loaded from disk (or unpickled)" in str(unpickled.value)
