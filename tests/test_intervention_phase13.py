"""Phase 13 discoverability tests."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class M(nn.Module):
    """Small model with a stable relu site.

    Kept inline deliberately (NOT migrated to ``example_models``): the
    auto-naming tests below derive trace names from this class's name
    (``m_1``), so the class identity is part of the test contract.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run relu and add one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor plus one.
        """

        return torch.relu(x) + 1


class BertForSequenceClassification(nn.Module):
    """HuggingFace-style class name for auto-naming tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Original tensor.
        """

        return x + 0


def _capture(*, name: str | None = None) -> tl.Trace:
    """Capture an intervention-ready test log.

    Parameters
    ----------
    name:
        Optional explicit log name.

    Returns
    -------
    tl.Trace
        Captured log.
    """

    return tl.trace(
        M(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True, name=name),
    )


def test_auto_naming_explicit_name_and_reset_counter() -> None:
    """Automatic naming increments and explicit names are preserved."""

    tl.io.reset_naming_counter()
    log1 = _capture()
    log2 = _capture()
    log3 = _capture(name="custom_name")

    assert log1.name == "m_1"
    assert log2.name == "m_2"
    assert log3.name == "custom_name"

    tl.io.reset_naming_counter("m")
    log4 = _capture()
    assert log4.name == "m_1"


def test_summary_includes_phase13_sections() -> None:
    """The discoverability fields live on provenance() (summary memo 3.6)."""

    log = _capture(name="phase13_summary")
    summary = log.provenance._summary()

    assert "TorchLens Discoverability Summary" in summary
    assert "state:" in summary
    assert "direct_write_dirty:" in summary
    assert "graph_shape_hash:" in summary
    assert "portability:" in summary
    assert "RNG and helper notes:" in summary


def test_last_run_records_returns_tuple_snapshot() -> None:
    """Fire records from the most recent propagated run are returned as a tuple."""

    log = _capture()
    log.do(
        tl.func("relu"),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(confirm_mutation=True),
    )

    records = log.last_run_records()

    assert isinstance(records, tuple)
    assert len(records) >= 1
    assert records[-1].engine == "replay"
