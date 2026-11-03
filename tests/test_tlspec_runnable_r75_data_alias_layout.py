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

import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import NumericAttestationStatus, PathFaithfulness
from torchlens._state import pause_logging
from torchlens.options import CaptureOptions


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture one runnable-ready trace."""

    return tl.trace(
        model,
        x,
        capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
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


# A tensor the capture never sees: no label, no ledger entry, no captured storage, and
# no state address (a module ATTRIBUTE tensor is address-stamped and owned by the r63
# state nets -- its non-canonical layout read already REFUSES at save; a free global is
# now a recorded ``constant`` source, so the forward copies this one with logging paused
# to plant the genuinely unresolvable receiver the fail-closed floor must catch).
_HIDDEN_GLOBAL = torch.randn(2, 4, 8, 8).to(memory_format=torch.channels_last)


class HiddenGlobalLayoutBranch(nn.Module):
    """Fail-closed floor: layout read on a tensor the capture never saw."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the unlabeled, unresolvable global tensor's layout."""

        with pause_logging():
            hidden = _HIDDEN_GLOBAL.clone()
        if hidden.is_contiguous(memory_format=torch.channels_last):
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


# Kept: sole branch coverage of backends/torch/_completeness_metadata.py; a sole mutation kill; the catch of its regression fault.
def test_r75_unresolvable_receiver_fails_closed(tmp_path: Path) -> None:
    """The fail-closed floor: an unresolvable unlabeled receiver downgrades completeness.

    A tensor the capture never saw has no label, no ledger entry, no captured storage,
    and no state address: no positive rung resolves it, so the read must ceiling the
    artifact UNVERIFIABLE on EVERY run (r73 silently recorded nothing here -- a hidden
    Python-state layout read could steer the branch while the replay claimed VERIFIED).
    """

    x = _nchw()
    path = _save(HiddenGlobalLayoutBranch().eval(), x, tmp_path / "hidden.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.is_poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# Kept: sole branch coverage of backends/torch/_completeness_metadata.py; a sole mutation kill.
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
    assert not result.report.is_poisoned
