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


# --------------------------------------------------------------------------- #
# R10-H1: a direct MODEL-INPUT escape must be witnessed, not skipped by the
# is_input/is_output boundary filter. The input is exactly what changes on a
# changed-input run, so an input-sourced escape is the MOST important to witness.
# --------------------------------------------------------------------------- #
class RawInputScalarEscape(nn.Module):
    """R10-H1: ``float(x)`` reads the RAW input tensor on the host."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float(x)


class RawInputBoolBranch(nn.Module):
    """R10-H1: ``if x:`` truth-tests the RAW input on the host."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x:
            return x * 2.0
        return x + 7.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "capture_x", "changed_x"),
    [
        (RawInputScalarEscape, torch.tensor(3.0), torch.tensor(5.0)),
        (RawInputBoolBranch, torch.tensor(1.0), torch.tensor(0.0)),
    ],
    ids=["raw_input_scalar", "raw_input_bool"],
)
def test_r10_input_escape_changed_is_unverifiable(
    model_cls: type[nn.Module],
    capture_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    path = _save(model_cls(), capture_x, tmp_path / "i.tlspec")
    result = tl.load(path).run(inputs=changed_x.clone(), on_divergence="return_diverged")
    _assert_not_blessed(result.report)
    # The recorded taken path / baked literal is stale for the changed input.
    assert not torch.allclose(result.output, model_cls()(changed_x.clone()))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "capture_x"),
    [
        (RawInputScalarEscape, torch.tensor(3.0)),
        (RawInputBoolBranch, torch.tensor(1.0)),
    ],
    ids=["raw_input_scalar", "raw_input_bool"],
)
def test_r10_input_escape_original_still_verified(
    model_cls: type[nn.Module],
    capture_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    path = _save(model_cls(), capture_x, tmp_path / "i.tlspec")
    result = tl.load(path).run(inputs=capture_x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, model_cls()(capture_x.clone()))


# --------------------------------------------------------------------------- #
# R10-C1: a `.data` (unlabelled-alias) escape source cannot be attributed to any
# source slot, so it cannot be digest-witnessed. The producer must fail honest
# (INCOMPLETE -> UNVERIFIABLE), never leave completeness COMPLETE -> false VERIFIED.
# --------------------------------------------------------------------------- #
class DataAliasScalarEscape(nn.Module):
    """R10-C1: ``.data.item()`` reads an UNLABELLED alias -> census attributes nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x.mean().data.item() * 3.0 + 1.0)


def test_r10_data_alias_escape_is_unverifiable(tmp_path: Path) -> None:
    # An unattributable escape is fail-closed on BOTH the original and the changed
    # input: without a source slot the escaped value cannot be proven valid, so the
    # honest ceiling is UNVERIFIABLE (never a false VERIFIED on the changed input).
    from torchlens.backends.torch.completeness_witness import host_escape_has_unattributable

    trace = tl.trace(DataAliasScalarEscape(), torch.tensor([2.0, 2.0]), capture=_CAPTURE)
    assert host_escape_has_unattributable(trace), "census must flag the unlabelled .data escape"

    path = _save(DataAliasScalarEscape(), torch.tensor([2.0, 2.0]), tmp_path / "c.tlspec")
    changed = tl.load(path).run(inputs=torch.tensor([10.0, 10.0]), on_divergence="return_diverged")
    _assert_not_blessed(changed.report)
    assert not torch.allclose(changed.output, DataAliasScalarEscape()(torch.tensor([10.0, 10.0])))
    original = tl.load(path).run(inputs=torch.tensor([2.0, 2.0]), on_divergence="return_diverged")
    assert original.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


# --------------------------------------------------------------------------- #
# R10-H2: an orphan-PRUNED, param-rooted host escape chain (reaching neither input
# nor output) has no witnessable slot. Its census raw label never resolves to a
# final op, so the producer must fail honest instead of silently dropping it (which
# left completeness COMPLETE -> false VERIFIED even for a BOUND param the unbound
# net exempts). Both runs are UNVERIFIABLE; the changed one must never be VERIFIED.
# --------------------------------------------------------------------------- #
class PrunedParamEscape(nn.Module):
    """R10-H2: ``float((w + 1).sum())`` on a dual-use (bound) param, host-only + pruned."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.w  # w is BOUND (feeds a traced call) -> unbound net exempts it
        s = float((self.w + 1.0).sum())  # param-rooted host escape, orphan-pruned
        return y + s


def test_r10_pruned_param_escape_changed_state_is_unverifiable(tmp_path: Path) -> None:
    x = torch.tensor([4.0])
    path = _save(PrunedParamEscape(), x, tmp_path / "p.tlspec", include_weights=True)
    loaded = tl.load(path)
    loaded.load_state_dict({"w": torch.tensor([7.0])})
    result = loaded.run(inputs=x.clone(), on_divergence="return_diverged")
    _assert_not_blessed(result.report)
    # Captured s=3.0 replays 4*7 + 3 = 31; a true live forward with w=7 is 4*7 + 8 = 36.
    assert abs(result.output.item() - 31.0) < 1e-5
    assert abs(result.output.item() - 36.0) > 1e-5


# --------------------------------------------------------------------------- #
# R10-H3: the escape source is mutated IN PLACE after the escape read it. The
# witness digests the source at a mutation-consistent snapshot (its production
# point) at BOTH save and run, so the ORIGINAL input stays VERIFIED (no spurious
# downgrade) while a changed input is still caught as UNVERIFIABLE.
# --------------------------------------------------------------------------- #
class InPlaceMutatedEscapeSource(nn.Module):
    """R10-H3: ``float(y)`` reads y == 6.0, then ``y.add_(1.0)`` mutates y to 7.0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2.0
        s = float(y)  # escape reads the pre-mutation value (6.0)
        y.add_(1.0)  # source mutated in place after the escape
        return y * s


def test_r10_inplace_mutated_source_original_still_verified(tmp_path: Path) -> None:
    x = torch.tensor(3.0)
    path = _save(InPlaceMutatedEscapeSource(), x, tmp_path / "m3.tlspec")
    result = tl.load(path).run(inputs=x.clone())
    # The original input is value-correct (7 * 6 = 42) and must NOT be spuriously
    # downgraded by digesting the post-mutation live slot instead of the snapshot.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert abs(result.output.item() - 42.0) < 1e-5


def test_r10_inplace_mutated_source_changed_is_still_unverifiable(tmp_path: Path) -> None:
    x = torch.tensor(3.0)
    path = _save(InPlaceMutatedEscapeSource(), x, tmp_path / "m3.tlspec")
    result = tl.load(path).run(inputs=torch.tensor(5.0), on_divergence="return_diverged")
    # Snapshot-based staleness still fires: capture s=6.0, changed production is 10.0.
    _assert_not_blessed(result.report)
    live = InPlaceMutatedEscapeSource()(torch.tensor(5.0))
    assert not torch.allclose(result.output, live)
