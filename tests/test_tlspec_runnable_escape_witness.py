"""Round-9 escape-witness honesty for sparse runnable execution.

Round-8 closed the verbatim scalar bake by correlating a baked ``LiteralAtom`` to
an escaped scalar by exact VALUE EQUALITY. That heuristic is fundamentally too
weak: ANY Python arithmetic on the escaped scalar, a multi-element ``.tolist()``
sequence bake, a dual-use escape whose op also feeds the graph, a non-bool scalar
steering pure-Python control flow, or a module buffer truth-test all evade it and
replay a STALE baked literal / taken branch while still reporting
``path_faithfulness=VERIFIED``.

The redesign keys the witness on the tensor->host ESCAPE EVENT (and on unbound
state), not on value-correlating a baked literal:

* The dispatch census records the SOURCE op of every ``aten._local_scalar_dense``
  escape (``.item()`` / ``int()`` / ``float()`` / ``__index__`` / ``bool()``); the
  descriptor witnesses that source slot with its capture-time byte digest.
* A census-invisible ``.tolist()`` / ``.numpy()`` sequence bake is caught by an
  internal-sink value-equality net.
* A registered buffer/param read only through an untraced host path (unbound state)
  is witnessed by its capture-time byte digest.

At run time a source slot / unbound state value that recomputes different bytes
(a CHANGED input or CHANGED staged state) reports ``UNVERIFIABLE`` +
``NOT_APPLICABLE``; the ORIGINAL/unchanged input+state still reports ``VERIFIED``
(+``ATTESTED`` where eligible). A model with no escape and no unbound state -- and a
genuinely dead scalar sink -- is UNCHANGED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Attack models (each one's captured-input run stays VERIFIED and correct;
# only the changed input / changed state must refuse to bless a stale result).
# --------------------------------------------------------------------------- #
class TransformedScalar(nn.Module):
    """H1: Python arithmetic on an ``.item()`` scalar defeats value-equality."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean().item()
        return x * (s * 2.0 + 1.0)


class LoopAccumScalar(nn.Module):
    """H1 variant: a Python loop accumulates ``.item()`` escapes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for i in range(2):
            total = total + x[i].item()
        return x * total


class TolistSequenceBake(nn.Module):
    """H2: a multi-element ``.tolist()`` escape baked as a sequence literal."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = x.sum(dim=0).tolist()
        return x + torch.tensor(vals)


class DualUseScalar(nn.Module):
    """H3: the escaped op ALSO feeds the graph (dual-use verbatim bake)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean()
        y = x * s
        return y + s.item()


class IntBranchSelect(nn.Module):
    """H4: a non-bool ``int(argmax())`` escape steers pure-Python control flow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = int(x.argmax().item())
        if idx == 0:
            return x * 2.0
        return x + 7.0


class BufferBranch(nn.Module):
    """H5: a registered bool buffer truth-test drives untraced control flow."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1 if self.gate else x - 1


def _save(model: nn.Module, capture_input: torch.Tensor, path: Path, **save_kwargs) -> Path:
    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def _assert_not_blessed(report) -> None:
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert report.poisoned


# --------------------------------------------------------------------------- #
# H1-H4: input-derived escapes. Changed input -> UNVERIFIABLE; original -> VERIFIED.
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "capture_x", "changed_x"),
    [
        (TransformedScalar, torch.tensor([2.0, 2.0]), torch.tensor([10.0, 10.0])),
        (LoopAccumScalar, torch.tensor([2.0, 3.0]), torch.tensor([5.0, 7.0])),
        (
            TolistSequenceBake,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 5.0], [5.0, 5.0]]),
        ),
        (DualUseScalar, torch.tensor([2.0, 2.0]), torch.tensor([10.0, 10.0])),
        (IntBranchSelect, torch.tensor([9.0, 1.0]), torch.tensor([1.0, 9.0])),
    ],
    ids=["transformed_scalar", "loop_accum", "tolist_seq", "dual_use", "int_branch"],
)
def test_changed_input_escape_is_unverifiable(
    model_cls: type[nn.Module],
    capture_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    path = _save(model_cls(), capture_x, tmp_path / "m.tlspec", include_activations=True)

    result = tl.load(path).run(inputs=changed_x)

    _assert_not_blessed(result.report)
    # The recorded taken path / baked literal is stale for the changed input, so the
    # sparse output differs from a true live forward -- and the report refuses to
    # bless it.
    live = model_cls()(changed_x)
    assert not torch.allclose(result.output, live)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "capture_x"),
    [
        (TransformedScalar, torch.tensor([2.0, 2.0])),
        (LoopAccumScalar, torch.tensor([2.0, 3.0])),
        (TolistSequenceBake, torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        (DualUseScalar, torch.tensor([2.0, 2.0])),
        (IntBranchSelect, torch.tensor([9.0, 1.0])),
    ],
    ids=["transformed_scalar", "loop_accum", "tolist_seq", "dual_use", "int_branch"],
)
def test_original_input_escape_still_verified_attested(
    model_cls: type[nn.Module],
    capture_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    path = _save(model_cls(), capture_x, tmp_path / "m.tlspec", include_activations=True)

    result = tl.load(path).run(inputs=capture_x.clone())

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert not result.report.poisoned
    assert torch.allclose(result.output, model_cls()(capture_x.clone()))


# --------------------------------------------------------------------------- #
# H5: state-derived escape. Changed staged buffer -> UNVERIFIABLE; original -> VERIFIED.
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_buffer_branch_changed_state_is_unverifiable(tmp_path: Path) -> None:
    x = torch.tensor([4.0])
    path = _save(BufferBranch(), x, tmp_path / "b.tlspec", include_weights=True)

    loaded = tl.load(path)
    loaded.load_state_dict({"gate": torch.tensor(False)})
    result = loaded.run(inputs=x)

    _assert_not_blessed(result.report)
    # Captured gate=True replays x+1; a true live forward with gate=False is x-1.
    assert torch.allclose(result.output, x + 1)
    assert not torch.allclose(result.output, x - 1)


@pytest.mark.smoke
def test_buffer_branch_original_state_still_verified(tmp_path: Path) -> None:
    x = torch.tensor([4.0])
    path = _save(BufferBranch(), x, tmp_path / "b.tlspec", include_weights=True)

    # Embedded capture state (gate=True).
    embedded = tl.load(path).run(inputs=x)
    assert embedded.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not embedded.report.poisoned
    assert torch.allclose(embedded.output, x + 1)

    # Re-staging the identical capture-equivalent state stays VERIFIED.
    loaded = tl.load(path)
    loaded.load_state_dict({"gate": torch.tensor(True)})
    restaged = loaded.run(inputs=x)
    assert restaged.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not restaged.report.poisoned


# --------------------------------------------------------------------------- #
# No over-triggering: a dead scalar sink and an escape-free model are UNCHANGED.
# --------------------------------------------------------------------------- #
class DeadSinkModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = (x * 3.0).sum()  # scalar sink whose value never escapes to the host
        return x + 1.0


def test_dead_sink_and_escape_free_models_unchanged(tmp_path: Path) -> None:
    from torchlens._io.runnable import build_sparse_run_descriptor

    trace = tl.trace(DeadSinkModel(), torch.tensor([1.0, 2.0]), capture=_CAPTURE)
    descriptor = build_sparse_run_descriptor(trace)
    # No tensor->host escape occurred, so no source-slot witness is emitted.
    assert not any(
        witness.kind.value == "tensor_derived_scalar_literal"
        for witness in descriptor.control_witnesses
    )

    path = _save(DeadSinkModel(), torch.tensor([1.0, 2.0]), tmp_path / "d.tlspec")
    result = tl.load(path).run(inputs=torch.tensor([7.0, 8.0]))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, DeadSinkModel()(torch.tensor([7.0, 8.0])))


def test_escape_source_recorded_by_census(tmp_path: Path) -> None:
    # The dispatch census records the source op of a scalar escape even when the
    # escaped value is transformed before use (value-equality would find nothing).
    from torchlens.backends.torch.completeness_witness import host_escape_source_labels

    trace = tl.trace(TransformedScalar(), torch.tensor([2.0, 2.0]), capture=_CAPTURE)
    assert host_escape_source_labels(trace), "census must record the .item() escape source"
