"""Phase 13 discoverability tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.resolver import SiteTable


class M(nn.Module):
    """Small model with a stable relu site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run relu and add one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted relu output.
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


def _capture(*, name: str | None = None) -> tl.ModelLog:
    """Capture an intervention-ready test log.

    Parameters
    ----------
    name:
        Optional explicit log name.

    Returns
    -------
    tl.ModelLog
        Captured log.
    """

    return tl.log_forward_pass(
        M(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        name=name,
    )


def test_auto_naming_explicit_name_and_reset_counter() -> None:
    """Automatic naming increments and explicit names are preserved."""

    tl.reset_naming_counter()
    log1 = _capture()
    log2 = _capture()
    log3 = _capture(name="custom_name")

    assert log1.name == "m_1"
    assert log2.name == "m_2"
    assert log3.name == "custom_name"

    tl.reset_naming_counter("m")
    log4 = _capture()
    assert log4.name == "m_1"


def test_huggingface_suffix_is_stripped_for_auto_name() -> None:
    """Common HuggingFace suffixes are stripped before lowercasing."""

    tl.reset_naming_counter()
    log = tl.log_forward_pass(
        BertForSequenceClassification(),
        torch.randn(1, 2),
        vis_opt="none",
        intervention_ready=True,
    )

    assert log.name == "bert_1"


def test_list_logs_returns_tuple_snapshot() -> None:
    """The process registry exposes an immutable tuple snapshot."""

    log = _capture()
    logs = tl.list_logs()

    assert isinstance(logs, tuple)
    assert log in logs


def test_summary_includes_phase13_sections() -> None:
    """ModelLog.summary includes discoverability fields."""

    log = _capture(name="phase13_summary")
    summary = log.summary()

    assert "TorchLens Discoverability Summary" in summary
    assert "run_state:" in summary
    assert "direct_write_dirty:" in summary
    assert "graph_shape_hash:" in summary
    assert "portability:" in summary
    assert "RNG and helper notes:" in summary


def test_last_run_records_returns_tuple_snapshot() -> None:
    """Fire records from the most recent propagated run are returned as a tuple."""

    log = _capture()
    log.attach_hooks(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    log.replay()

    records = log.last_run_records()

    assert isinstance(records, tuple)
    assert len(records) >= 1
    assert records[-1].engine == "replay"


def test_find_sites_repr_empty_one_and_many() -> None:
    """SiteTable repr is compact for empty, single, and multi-site tables."""

    log = _capture()

    empty = SiteTable(())
    one = log.find_sites(tl.func("relu"))
    many = SiteTable(tuple(log.layer_list + log.layer_list))

    assert repr(empty) == "SiteTable(0 sites)"
    assert repr(one).startswith("SiteTable(1 site: relu")
    assert repr(many).startswith("SiteTable(")
    assert "..." in repr(many)


def test_bundle_default_names_derive_from_log_names_with_collision_suffix() -> None:
    """Implicit Bundle member names use log.name and suffix collisions."""

    log1 = _capture(name="same")
    log2 = _capture(name="same")

    bundle = tl.bundle([log1, log2])

    assert bundle.names == ["same", "same_2"]
    assert log1.name == "same"
    assert log2.name == "same"


def test_loaded_log_preserves_name_without_incrementing_counter(tmp_path: Path) -> None:
    """Loading preserves saved names and does not consume fresh auto-name counters."""

    tl.reset_naming_counter()
    log = _capture(name="saved_name")
    path = tmp_path / "saved.tl"
    tl.save(log, path, overwrite=True)

    loaded = tl.load(path)
    fresh = _capture()

    assert loaded.name == "saved_name"
    assert fresh.name == "m_1"
