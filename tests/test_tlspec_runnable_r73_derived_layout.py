"""Input-derived intermediate-layout escape ceiling (r73 F1).

Round-72 (hon1) confirmed a false VERIFIED on the residual line: a model branching on an
INTERMEDIATE tensor's contiguity (``(x * 2).is_contiguous(memory_format=channels_last)``)
reported VERIFIED when replayed on a runtime input with a DIFFERENT memory format
(channels_last twin: identical shape+dtype+values), while oracle-1 (a fresh instance on
that input) takes the OTHER arm. The input contract pins shape+dtype+device+strided-ness
but not strides; the r27/r31 metadata net witnesses layout reads only on the input LEAF
(or a ``.data``/``.detach()`` alias), so a read on a fresh-storage activation derived from
the input recorded nothing -- yet memory format PROPAGATES through elementwise ops.

The r73 closure attributes such a read by TRACED VALUE ANCESTRY (``OpEvent.input_ancestors``)
and records a synthetic ``derived_layout_read`` fact -- carrying the rooting leaf's
capture-time stride tuple -- inside the existing totalized ``model_input_metadata`` envelope
of each rooting input site. At run time the executor compares the RAW runtime leaf's strides
and, on ANY difference, ceilings the run UNVERIFIABLE (a run-time ceiling, never DIVERGED:
a changed input layout does not PROVE the derived read flipped -- an intervening ``reshape``
can canonicalize it -- so the honest verdict is "cannot verify", per the r35 three-state
rule).

Zero-collateral pins (each was a hard requirement of the fix):

* honest channels_last models run on same-layout inputs stay VERIFIED;
* layout-oblivious models stay VERIFIED even on changed-layout inputs (no contamination
  from TorchLens's own internal ``is_contiguous(memory_format=...)`` payload-copy reads);
* the STATE-rooted twin (contract residual (3), ``(self.w * 2).is_contiguous()``) records
  nothing here and must NOT couple to input-layout changes;
* the DIRECT-leaf spelling keeps its stricter r27/r31 semantics (witnessed leaf fact ->
  PathDivergenceError on a changed-layout input).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


def _save(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact with embedded weights."""

    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(path, level="runnable", include_weights=True)
    return path


class IntermediateLayoutBranch(nn.Module):
    """Branch on an INTERMEDIATE's channels_last contiguity (the r72 hon1 F1 repro)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(4, 4, 1)
        self.b = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route through ``a`` iff the elementwise intermediate is channels_last."""

        y = x * 2.0  # fresh storage; memory format propagates from x
        if y.is_contiguous(memory_format=torch.channels_last):
            return self.a(y)
        return self.b(y)


class DeepIntermediateLayoutBranch(nn.Module):
    """Same escape two traced ops away from the input (ancestry must propagate)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently by the second-order intermediate's default contiguity."""

        z = (x * 2.0) + 1.0
        return z + 10.0 if z.is_contiguous() else z - 10.0


class LeafLayoutBranch(nn.Module):
    """Branch DIRECTLY on the input leaf's channels_last flag (r27/r31 control)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(4, 4, 1)
        self.b = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route through ``a`` iff the input leaf itself is channels_last."""

        if x.is_contiguous(memory_format=torch.channels_last):
            return self.a(x)
        return self.b(x)


class PlainConv(nn.Module):
    """Layout-oblivious model: no Python-level layout read anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve an elementwise intermediate; never reads layout."""

        return self.c(x * 2.0)


class StateIntermediateLayoutBranch(nn.Module):
    """Residual-(3) crossing control: layout read on a STATE-rooted intermediate."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a param-derived intermediate's layout; never on the input's."""

        z = self.w * 2.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


def _nchw() -> torch.Tensor:
    """Return a fixed contiguous NCHW probe input."""

    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)


@pytest.mark.smoke
def test_r73_intermediate_layout_branch_ceilings_on_changed_layout(tmp_path: Path) -> None:
    """The F1 repro must ceiling UNVERIFIABLE on a channels_last twin (was VERIFIED)."""

    x = _nchw()
    path = _save(IntermediateLayoutBranch().eval(), x, tmp_path / "f1.tlspec")

    result = tl.load(path).run(inputs=x.clone().to(memory_format=torch.channels_last))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r73_deep_intermediate_layout_read_ceilings(tmp_path: Path) -> None:
    """Ancestry attribution reaches reads two traced ops away from the input."""

    x = _nchw()
    path = _save(DeepIntermediateLayoutBranch(), x, tmp_path / "deep.tlspec")

    # Same shape+dtype+values, different strides (transposed twin materialized
    # through a transposed clone so the branch's comparison basis changes).
    twin = x.clone().permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    result = tl.load(path).run(inputs=twin)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r73_same_layout_run_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: the SAME-layout replay of a layout-reading model stays VERIFIED."""

    x = _nchw()
    path = _save(IntermediateLayoutBranch().eval(), x, tmp_path / "same.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r73_honest_channels_last_model_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: channels_last capture + channels_last run stays VERIFIED."""

    x_cl = _nchw().to(memory_format=torch.channels_last)
    path = _save(IntermediateLayoutBranch().eval(), x_cl, tmp_path / "cl.tlspec")

    run_input = _nchw().clone().to(memory_format=torch.channels_last)
    result = tl.load(path).run(inputs=run_input)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r73_layout_oblivious_model_never_triggers(tmp_path: Path) -> None:
    """Zero over-trigger: a model with NO layout read stays VERIFIED on a changed-layout input.

    This is also the contamination tripwire: TorchLens's own payload-copy machinery
    calls ``is_contiguous(memory_format=...)`` on every saved activation; those internal
    reads run under ``pause_logging`` and must never record a ``derived_layout_read``
    fact.
    """

    x = _nchw()
    path = _save(PlainConv().eval(), x, tmp_path / "plain.tlspec")

    result = tl.load(path).run(inputs=x.clone().to(memory_format=torch.channels_last))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r73_state_rooted_read_does_not_couple_to_input_layout(tmp_path: Path) -> None:
    """Residual (3) stays untouched: a STATE-rooted intermediate read records no input fact.

    ``(self.w * 2)`` has no input ancestor, so a changed-layout INPUT must not ceiling
    the run -- the state-layout twin remains the accepted contract residual (3), and
    taint-coupling it to input layout would be exactly the over-refusal the residual's
    zero-collateral pledge forbids.
    """

    x = _nchw()
    path = _save(StateIntermediateLayoutBranch().eval(), x, tmp_path / "state.tlspec")

    result = tl.load(path).run(inputs=x.clone().to(memory_format=torch.channels_last))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r73_direct_leaf_read_still_diverges(tmp_path: Path) -> None:
    """The DIRECT-leaf spelling keeps its stricter witnessed-fact semantics (DIVERGED)."""

    x = _nchw()
    path = _save(LeafLayoutBranch().eval(), x, tmp_path / "leaf.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=x.clone().to(memory_format=torch.channels_last))


class TwoInputOneRooted(nn.Module):
    """Two-input model whose ONLY layout read roots at the first input."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Branch on an ``a``-derived intermediate's layout; ``b`` stays layout-oblivious."""

        y = a * 2.0
        z = b + 1.0
        if y.is_contiguous(memory_format=torch.channels_last):
            return z + y
        return z - y


@pytest.mark.smoke
def test_r73_fact_scopes_to_the_rooting_input_only(tmp_path: Path) -> None:
    """Per-ancestor precision: only the ROOTING input's layout change ceilings the run.

    Ancestry attribution records the fact on ``arg:0`` alone, so a changed-layout
    NON-rooting input must stay VERIFIED (no coarse any-input coupling), while the
    rooting input's channels_last twin ceilings UNVERIFIABLE.
    """

    torch.manual_seed(0)
    a = torch.randn(2, 4, 8, 8)
    b = torch.randn(2, 4, 8, 8)
    trace = tl.trace(
        TwoInputOneRooted(),
        (a, b),
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    path = tmp_path / "two.tlspec"
    trace.save(path, level="runnable", include_weights=True)

    reads_by_position = {
        site.position: tuple(
            name for tensor_site in site.tensor_sites for name in tensor_site.metadata_reads
        )
        for site in tl.load(path).__dict__["_runnable_descriptor"].input_boundary
    }
    assert "derived_layout_read" in reads_by_position["arg:0"]
    assert "derived_layout_read" not in reads_by_position["arg:1"]

    non_rooting = tl.load(path).run(
        inputs=(a.clone(), b.clone().to(memory_format=torch.channels_last))
    )
    assert non_rooting.report.path_faithfulness is PathFaithfulness.VERIFIED

    rooting = tl.load(path).run(inputs=(a.clone().to(memory_format=torch.channels_last), b.clone()))
    assert rooting.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert rooting.report.poisoned


@pytest.mark.smoke
def test_r73_derived_fact_rides_the_metadata_envelope(tmp_path: Path) -> None:
    """The fact is declared on the boundary record AND carried by the site envelope.

    The r71 obligation model makes the pair strip-coherent: deleting the envelope
    breaks totality, deleting only the fact breaks the envelope/owner fact-name
    equality -- both refuse at parse. A layout-oblivious capture declares neither.
    """

    x = _nchw()
    path = _save(IntermediateLayoutBranch().eval(), x, tmp_path / "envelope.tlspec")
    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    reads = [
        name
        for site in descriptor.input_boundary
        for tensor_site in site.tensor_sites
        for name in tensor_site.metadata_reads
    ]
    assert "derived_layout_read" in reads

    plain_path = _save(PlainConv().eval(), x, tmp_path / "envelope_plain.tlspec")
    plain_descriptor = tl.load(plain_path).__dict__["_runnable_descriptor"]
    plain_reads = [
        name
        for site in plain_descriptor.input_boundary
        for tensor_site in site.tensor_sites
        for name in tensor_site.metadata_reads
    ]
    assert "derived_layout_read" not in plain_reads
