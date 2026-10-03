"""Replay engine: buffer-state threading and in-place ownership.

Regression net for the 2026-08-19 buffer-replay finding: the replay engine
executed in-place funcs on tensors it did not own (captured record outs and
committed overlay tensors, resolved BY IDENTITY), and buffer version records
were excluded from overlay propagation so buffer state never threaded through
a replayed cone. Symptom on a 3-pass ``self.b.add_(1.0)`` loop with captured
buffer versions 0,1,2,3: replacing pass 1 with 7s read back ``:1=8, :2=2,
:3=3, :4=3`` -- the origin readback mutated by its in-place child, the later
records corrupted IN CAPTURE (and, through copy-on-write forks, in the SOURCE
trace), and nothing chained.

Contract under test:

- A value edit on a written-buffer version THREADS: every later ``add_``
  re-executes from the propagated state and every later buffer version record
  commits the recomputed value.
- Replay never mutates capture truth: the source trace's saved tensors are
  bit-identical after a fork's ``do()``, and the user's edit tensor is never
  written through.
- An in-place op inside the cone cannot corrupt the committed readback of its
  parent (the origin keeps the substituted value, not value+1).
- A buffer version that cannot be provably threaded keeps its captured value
  and discloses the gap with ``BufferThreadGapWarning``.

Test-design rules (do not weaken): ground truth is computed BY HAND from the
model's own weights; fidelity assertions are PER PASS; the loop cell uses
``bias=True`` so zero is not a fixed point.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import BufferThreadGapWarning

ATOL = 1e-6


class BufLoop(nn.Module):
    """3-pass loop that mutates a registered buffer in place each pass."""

    def __init__(self) -> None:
        """Build the shared cell and the mutable buffer."""

        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell 3 times, bumping the buffer before each pass."""

        h = x
        for _ in range(3):
            self.b.add_(1.0)
            h = torch.relu(self.cell(h) + self.b)
        return h


def _traced_buf_loop() -> tuple[BufLoop, torch.Tensor, tl.Trace]:
    """Return a seeded model, input, and intervention-ready exhaustive trace."""

    torch.manual_seed(0)
    model = BufLoop()
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True),
    )
    return model, x, log


BUF_KEYS = ("buffer_1:1", "buffer_1:2", "buffer_1:3", "buffer_1:4")


def test_inplace_buffer_edit_threads_through_every_pass() -> None:
    """Editing buffer pass 1 propagates through every later in-place write."""

    model, x, log = _traced_buf_loop()
    assert not torch.equal(model.cell.bias.detach(), torch.zeros(4)), "bias confound guard"
    for key, expected in zip(BUF_KEYS, (0.0, 1.0, 2.0, 3.0)):
        assert torch.equal(log.layer_dict_all_keys[key].out, torch.full((4,), expected))

    fork = log.fork()
    edit = torch.full((4,), 7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BufferThreadGapWarning)
        fork.do(
            tl.units("buffer_1:1", [(i,) for i in range(4)]).resolve(fork),
            tl.replace_with(edit),
        )

    # The origin keeps the substituted value (honest readback, no +1 from the
    # downstream in-place add), and each later version is the threaded state.
    for key, expected in zip(BUF_KEYS, (7.0, 8.0, 9.0, 10.0)):
        got = fork.layer_dict_all_keys[key].out
        assert torch.equal(got, torch.full((4,), expected)), (key, got)

    # Downstream per-pass fidelity against a hand-computed forward with b0=7.
    with torch.no_grad():
        b = torch.full((4,), 7.0)
        h = x
        for pass_num in range(1, 4):
            b = b + 1.0
            h = torch.relu(model.cell(h) + b)
            got = fork.layer_dict_all_keys[f"relu_1_4:{pass_num}"].out
            assert torch.allclose(got, h, atol=ATOL), f"relu_1_4:{pass_num}"

    assert torch.equal(edit, torch.full((4,), 7.0)), "user edit tensor was mutated"


def test_replay_leaves_source_trace_capture_untouched() -> None:
    """A fork's replay never writes through shared payloads to the source."""

    _model, _x, log = _traced_buf_loop()
    snapshot = {
        key: record.out.clone()
        for key, record in log.layer_dict_all_keys.items()
        if isinstance(key, str) and isinstance(record.out, torch.Tensor)
    }

    fork = log.fork()
    fork.do(
        tl.units("buffer_1:1", [(i,) for i in range(4)]).resolve(fork),
        tl.replace_with(torch.full((4,), 7.0)),
    )

    for key, saved in snapshot.items():
        assert torch.equal(log.layer_dict_all_keys[key].out, saved), (
            f"source trace record {key!r} mutated by a fork's replay"
        )


class InplaceAct(nn.Module):
    """Linear -> ReLU -> in-place add on the activation -> scale."""

    def __init__(self) -> None:
        """Build the linear stage."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pipeline with one in-place op on an activation."""

        h = torch.relu(self.lin(x))
        h.add_(1.0)
        return h * 2


def test_inplace_activation_op_does_not_corrupt_origin_readback() -> None:
    """A non-buffer in-place op in the cone cannot mutate its parent's commit."""

    torch.manual_seed(1)
    model = InplaceAct()
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True),
    )
    fork = log.fork()
    edit = torch.full((2, 4), 5.0)
    fork.do(
        tl.units("relu_1_2:1", [(i, j) for i in range(2) for j in range(4)]).resolve(fork),
        tl.replace_with(edit),
    )

    relu_out = fork.layer_dict_all_keys["relu_1_2:1"].out
    assert torch.equal(relu_out, torch.full((2, 4), 5.0)), (
        "origin readback corrupted by downstream in-place add_"
    )
    add_out = fork.layer_dict_all_keys["add_1_3:1"].out
    assert torch.equal(add_out, torch.full((2, 4), 6.0))
    final = fork.layer_dict_all_keys["mul_1_4:1"].out
    assert torch.equal(final, torch.full((2, 4), 12.0))


@pytest.mark.smoke
def test_unprovable_buffer_thread_warns_and_keeps_captured_value() -> None:
    """A write kind outside the provable set keeps capture and discloses."""

    _model, _x, log = _traced_buf_loop()
    fork = log.fork()
    # Simulate a write the engine cannot certify (e.g. a fused mutator whose
    # op output is not the post-write buffer state).
    fork.layer_dict_all_keys["buffer_1:2"]._internal_set("buffer_write_kind", "fused")

    with pytest.warns(BufferThreadGapWarning, match="buffer_1:2"):
        fork.do(
            tl.units("buffer_1:1", [(i,) for i in range(4)]).resolve(fork),
            tl.replace_with(torch.full((4,), 7.0)),
        )

    # The uncertified version keeps its captured value; later versions thread
    # from the stale-but-honest state (add_ re-executes from 1s -> 2s -> 3s).
    assert torch.equal(fork.layer_dict_all_keys["buffer_1:1"].out, torch.full((4,), 7.0))
    assert torch.equal(fork.layer_dict_all_keys["buffer_1:2"].out, torch.full((4,), 1.0))
