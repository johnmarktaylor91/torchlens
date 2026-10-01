"""Regression coverage for fast-pass output population."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _SelectiveSaveModel(nn.Module):
    """Small stable graph with an explicit output node."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear, relu, and passthrough module."""

        return self.identity(torch.relu(self.linear(x)))


def _saved_tensor_by_label(trace: tl.Trace, labels: list[str]) -> dict[str, torch.Tensor]:
    """Return saved output tensors for the selected labels."""

    saved: dict[str, torch.Tensor] = {}
    for label in labels:
        layer = trace[label]
        assert layer.has_saved_activation
        assert layer.out is not None
        saved[label] = layer.out.detach().clone()
    return saved


def test_fast_pass_selective_save_matches_fresh_trace_outputs() -> None:
    """Fast selective save keeps postprocessed outputs identical to a fresh trace."""

    torch.manual_seed(7)
    model = _SelectiveSaveModel()
    initial_input = torch.randn(2, 3)
    replacement_input = torch.randn(2, 3)

    fast_trace = tl.trace(model, initial_input, capture=tl.options.CaptureOptions(random_seed=11))
    fresh_trace = None
    try:
        relu_label = next(layer.layer_label for layer in fast_trace if layer.layer_type == "relu")
        selected_labels = [relu_label, fast_trace.output_layers[0]]

        fast_trace.save_new_outs(
            model,
            replacement_input,
            layers_to_save=selected_labels,
            random_seed=11,
        )
        fresh_trace = tl.trace(
            model,
            replacement_input,
            capture=tl.options.CaptureOptions(layers_to_save="all", random_seed=11),
        )

        fast_saved = _saved_tensor_by_label(fast_trace, selected_labels)
        fresh_saved = _saved_tensor_by_label(fresh_trace, selected_labels)
        assert fast_saved.keys() == fresh_saved.keys()
        for label, fast_out in fast_saved.items():
            torch.testing.assert_close(fast_out, fresh_saved[label])
    finally:
        fast_trace.cleanup()
        if fresh_trace is not None:
            fresh_trace.cleanup()


class _RawHookReplacedModel(nn.Module):
    """Model whose submodule output is replaced by a raw forward hook."""

    def __init__(self) -> None:
        """Initialize a linear layer feeding a relu."""

        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear then relu."""

        return torch.relu(self.linear(x))


def test_record_with_raw_replacement_hook_merges_intervention_lane() -> None:
    """tl.record survives a raw forward hook replacement and journals the edit.

    Regression: the recorder's per-pass fold crashed with
    ``KeyError('intervention_events')`` because the lane declared
    ``append_restamp`` but had no registered appender, so any raw
    ``register_forward_hook`` returning a new object killed ``tl.record()``
    (and could replace the user's real exception during failed-forward
    recovery).
    """

    model = _RawHookReplacedModel()
    injected = torch.ones(1, 2) * 7
    handle = model.linear.register_forward_hook(lambda _m, _a, _o: injected)
    try:
        recording = tl.record(model, torch.randn(1, 2), save=tl.func("relu"))
    finally:
        handle.remove()

    events = recording.event_stream
    assert events is not None
    edited = [event.label_raw for event in events.intervention_events]
    assert len(edited) >= 1
    # Sanctioned-merge chain of custody: the per-pass genuine edit re-binds
    # to the recorder journal (run token + re-stamped target seq), so the
    # merged record still names a real op event in THIS journal.
    target_ids = {(event.label_raw, event.seq) for event in events.op_events}
    for edit in events.intervention_events:
        assert edit.run_token == events.run_nonce
        assert (edit.label_raw, edit.target_seq) in target_ids
    all_seqs = [
        event.seq
        for lane in (
            events.op_events,
            events.module_prep_events,
            events.module_enter_events,
            events.module_exit_events,
            events.pre_hook_events,
            events.intervention_events,
        )
        for event in lane
    ]
    assert len(all_seqs) == len(set(all_seqs))
    assert max(all_seqs) <= events.event_seq
