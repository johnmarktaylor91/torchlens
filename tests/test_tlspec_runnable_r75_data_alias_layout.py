"""``.data``-alias intermediate-layout fail-open closure (r75 F1).

Round-74 (hon1 + free-roam, cross-lab) confirmed the r73 intermediate-layout net FAIL-OPENED
on unlabeled receivers: the C-level ``.data`` getter destroyed labels and ``_base``, so
``(x * 2).data.is_contiguous(channels_last)`` recorded nothing and a channels-last twin
replayed the captured arm as a false VERIFIED. The r75 origin/storage ladder below remains
the defense-in-depth attribution path. TorchLens now additionally captures the getter's real
``aten.detach`` dispatch as the canonical replayable ``Tensor.detach`` op, preserving direct
graph ancestry for ordinary captures without admitting the unsafe descriptor itself.

The r75 closure makes the layout-trio fall-through FAIL-CLOSED BY CONSTRUCTION -- no rung
records nothing silently -- with graduated precision:

* OWN LABEL, accepted only when the traced ancestry is INTACT (no op in the transitive parent
  chain carries ``unattributed_tensor_args`` -- the exact break marker a consumed unlabeled
  alias leaves behind);
* DISPATCH-ORIGIN LEDGER leaf origins (r37 mechanism A) for unlabeled aliases and
  ancestry-tainted receivers -- value-DAG-true through the break, so every ``.data`` spelling
  resolves PRECISELY to its rooting input (same fact, same ceiling as the labeled r73 path);
* STORAGE IDENTITY against live captured producers (r31 leaf / r63 state precedent) when the
  ledger never saw the alias mechanism;
* otherwise the ``_INPUT_METADATA_VIEW_READ`` completeness downgrade (UNVERIFIABLE), the same
  presence-only fail-close the sibling ``_HOST_ESCAPE_UNATTRIBUTABLE_*`` nets apply to this
  receiver class.

Zero-collateral pins: precise resolution keeps same-layout and original-input runs VERIFIED
for every ``.data`` spelling; layout-oblivious models stay VERIFIED on twins; STATE-rooted
reads (including the ``self.w.data`` alias spelling) record nothing (contract residual (3),
STATE side only); ``.detach()`` keeps its witnessed/ceiled behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


def _capture(model: nn.Module, x: torch.Tensor) -> "tl.Trace":
    """Capture one runnable-ready trace."""

    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )


def _save(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact with embedded weights."""

    _capture(model, x).save(path, level="runnable", include_weights=True)
    return path


def _nchw() -> torch.Tensor:
    """Return a fixed contiguous NCHW probe input."""

    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)


def _twin(x: torch.Tensor) -> torch.Tensor:
    """Return the same-shape+dtype+values channels_last twin."""

    return x.clone().to(memory_format=torch.channels_last)


class DataDirectBranch(nn.Module):
    """Layout read routed through the ``.data`` alias of a HELD intermediate (r74 F1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the alias's channels_last contiguity."""

        y = x * 2.0
        if y.data.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class DataDeadProducerBranch(nn.Module):
    """Same read where the producing temporary is DEAD by read time (ledger-only rung)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a one-expression ``.data`` alias's contiguity."""

        if (x * 2).data.is_contiguous(memory_format=torch.channels_last):
            return x + 10.0
        return x - 10.0


class DataStrideBranch(nn.Module):
    """``.stride()`` spelling of the same escape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the alias's innermost stride."""

        y = x * 2.0
        if y.data.stride()[-1] == 1:
            return y + 10.0
        return y - 10.0


class DataTransitiveGetitemBranch(nn.Module):
    """TRANSITIVE laundering: the read receiver is a LOGGED op downstream of ``.data``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the layout of a slice of the alias."""

        if (x * 2).data[0].is_contiguous(memory_format=torch.channels_last):
            return x + 10.0
        return x - 10.0


class DataOrphanedMulBranch(nn.Module):
    """ANCESTRY-ORPHANED-EMPTY: labeled receiver whose ancestry broke at the alias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the layout of ``y.data * 1.0`` (empty ``input_ancestors``)."""

        y = x * 2.0
        z = y.data * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class DetachControlBranch(nn.Module):
    """Control: ``.detach()`` is a LOGGED op and was already witnessed/ceiled by r73."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the detached intermediate's contiguity."""

        y = x * 2.0
        if y.detach().is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class PlainConv(nn.Module):
    """Layout-oblivious model: no Python-level layout read anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve an elementwise intermediate; never reads layout."""

        return self.c(x * 2.0)


class StateDataRootedBranch(nn.Module):
    """Residual-(3) crossing control: layout read on a STATE ``.data``-alias intermediate."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a param-``.data``-derived intermediate's layout; never the input's."""

        z = self.w.data * 2.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


# Pre-capture NON-module tensor: no label, no ledger entry, no captured storage, and no
# state address (a module ATTRIBUTE tensor is address-stamped and owned by the r63 state
# nets -- its non-canonical layout read already REFUSES at save; a free global like this
# one is the genuinely unresolvable receiver the fail-closed floor must catch).
_HIDDEN_GLOBAL = torch.randn(2, 4, 8, 8).to(memory_format=torch.channels_last)


class HiddenGlobalLayoutBranch(nn.Module):
    """Fail-closed floor: layout read on a PRE-CAPTURE non-module tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the unlabeled, unresolvable global tensor's layout."""

        if _HIDDEN_GLOBAL.is_contiguous(memory_format=torch.channels_last):
            return x + 10.0
        return x - 10.0


class MixedCatDirectConsumption(nn.Module):
    """A logged op DIRECTLY consuming the ``.data`` alias alongside a labeled parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the layout of ``cat([y, y.data])`` (mixed labeled/unlabeled args)."""

        y = x * 2.0
        w = torch.cat([y, y.data], dim=1)
        if w.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls",
    [
        DataDirectBranch,
        DataDeadProducerBranch,
        DataStrideBranch,
        DataTransitiveGetitemBranch,
        DataOrphanedMulBranch,
    ],
    ids=["data_direct", "data_dead_producer", "data_stride", "data_transitive", "data_orphaned"],
)
def test_r75_data_alias_spellings_ceiling_on_changed_layout(
    tmp_path: Path, model_cls: type[nn.Module]
) -> None:
    """Every r74 ``.data`` spelling must ceiling UNVERIFIABLE on the layout twin (was VERIFIED)."""

    x = _nchw()
    path = _save(model_cls().eval(), x, tmp_path / "alias.tlspec")

    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls",
    [DataDirectBranch, DataTransitiveGetitemBranch, DataOrphanedMulBranch],
    ids=["data_direct", "data_transitive", "data_orphaned"],
)
def test_r75_data_alias_spellings_stay_verified_on_same_layout(
    tmp_path: Path, model_cls: type[nn.Module]
) -> None:
    """Zero collateral: precise (not blanket) resolution keeps same-layout runs VERIFIED.

    A blanket completeness downgrade would ceiling even the original input; the ledger /
    ancestry resolution records the same per-input fact as the labeled r73 path, so an
    unchanged-layout replay still verifies.
    """

    x = _nchw()
    path = _save(model_cls().eval(), x, tmp_path / "same.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r75_detach_control_still_ceiled(tmp_path: Path) -> None:
    """The logged ``.detach()`` spelling keeps its r73 witnessed/ceiled behavior."""

    x = _nchw()
    path = _save(DetachControlBranch().eval(), x, tmp_path / "detach.tlspec")

    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r75_layout_oblivious_model_still_never_triggers(tmp_path: Path) -> None:
    """Zero over-trigger: a model with NO layout read stays VERIFIED on a twin."""

    x = _nchw()
    path = _save(PlainConv().eval(), x, tmp_path / "plain.tlspec")

    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r75_state_data_rooted_read_stays_residual_three(tmp_path: Path) -> None:
    """Residual (3) untouched THROUGH the alias: ``self.w.data`` roots at state, records nothing.

    The ledger resolves the state alias to a pure ``state:`` origin -- a POSITIVE
    state-rooted signal, not an ancestry orphan -- so a changed-layout INPUT must not
    ceiling the run (the state-layout twin remains the accepted contract residual (3)).
    """

    x = _nchw()
    path = _save(StateDataRootedBranch().eval(), x, tmp_path / "statedata.tlspec")

    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r75_unresolvable_receiver_fails_closed(tmp_path: Path) -> None:
    """The fail-closed floor: an unresolvable unlabeled receiver downgrades completeness.

    A pre-capture NON-module tensor has no label, no ledger entry, no captured storage,
    and no state address: no positive rung resolves it, so the read must ceiling the
    artifact UNVERIFIABLE on EVERY run (r73 silently recorded nothing here -- a hidden
    Python-state layout read could steer the branch while the replay claimed VERIFIED).
    """

    x = _nchw()
    path = _save(HiddenGlobalLayoutBranch().eval(), x, tmp_path / "hidden.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r75_direct_alias_consumption_is_replayable(tmp_path: Path) -> None:
    """A logged op directly consuming ``.data`` retains complete replay provenance.

    ``torch.cat([y, y.data])`` now receives both the original producer and the canonical
    detach op as graph parents. The sparse recipe can therefore rebuild the call without
    broadening callable trust to the unsafe ``data`` descriptor.
    """

    x = _nchw()
    path = _save(MixedCatDirectConsumption().eval(), x, tmp_path / "mixed.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r75_storage_identity_rung_resolves_when_ledger_missing(tmp_path: Path) -> None:
    """Storage-rung precision pin (r31/r63 precedent): live captured-storage identity.

    White-box: evict the receiver's dispatch-origin ledger entry mid-forward so the ledger
    rung misses; the alias must STILL resolve -- by storage identity against the live held
    producer -- to a PRECISE per-input fact (declared on the boundary record), keeping the
    same-layout run VERIFIED (a blanket fail-close would ceiling it) while the twin
    ceilings UNVERIFIABLE.
    """

    from torchlens import _state
    from torchlens.backends.torch import completeness_witness as cw

    class LedgerEvictedDataBranch(nn.Module):
        """Reads layout on a ``.data`` alias whose ledger entry was surgically evicted."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Branch on the evicted alias's contiguity."""

            y = x * 2.0
            d = y.data
            registry = cw._DISPATCH_TENSOR_ORIGINS.get(_state._active_trace)
            if registry is not None:
                registry._entries.pop(id(d), None)
            if d.is_contiguous(memory_format=torch.channels_last):
                return y + 10.0
            return y - 10.0

    x = _nchw()
    path = _save(LedgerEvictedDataBranch().eval(), x, tmp_path / "storage.tlspec")

    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    reads = [
        name
        for site in descriptor.input_boundary
        for tensor_site in site.tensor_sites
        for name in tensor_site.metadata_reads
    ]
    assert "derived_layout_read" in reads

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert twin.report.poisoned
