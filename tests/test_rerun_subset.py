"""Tests for rerun preserving selective save scope."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class ThreeOpModel(nn.Module):
    """Small model with several selectable operation sites."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a deterministic three-op computation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        y = x + 1
        z = torch.relu(y)
        return z * 3


def _saved_op_labels(trace: tl.Trace) -> list[str]:
    """Return labels for Ops with saved activations.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[str]
        Saved Op labels in execution order.
    """

    return [op.label for op in trace.layer_list if op.has_saved_activation]


def test_rerun_preserves_static_layers_to_save_subset() -> None:
    """Rerun does not save every Op after a static selective capture."""

    x = torch.randn(2, 3)
    log = tl.trace(
        ThreeOpModel(),
        x,
        capture=CaptureOptions(layers_to_save=["relu"], save_args=True),
    )
    saved_before = _saved_op_labels(log)
    num_ops_before = len(log.layer_list)

    log.run(ThreeOpModel(), x + 1)

    assert saved_before
    assert _saved_op_labels(log) == saved_before
    assert len(saved_before) < num_ops_before


def test_rerun_preserves_predicate_save_subset() -> None:
    """Rerun keeps predicate-selected saves scoped to matching Ops."""

    x = torch.randn(2, 3)
    log = tl.trace(ThreeOpModel(), x, save=tl.func("relu"))
    saved_before = _saved_op_labels(log)
    num_ops_before = len(log.layer_list)

    log.run(ThreeOpModel(), x + 1)

    assert saved_before
    assert _saved_op_labels(log) == saved_before
    assert len(saved_before) < num_ops_before


def test_rerun_save_scope_reads_lookback_payload_policy_directly() -> None:
    """``_rerun_save_scope`` must not silently default a renamed options field.

    R47-11: the lookback payload policy used to be read with
    ``getattr(..., "metadata_only")``, so a ``RecordingOptions`` field rename
    would silently fall back instead of raising. Direct attribute access makes
    drift raise ``AttributeError``.
    """

    from types import SimpleNamespace

    import pytest

    from torchlens.intervention.rerun import _rerun_save_scope

    drifted_options = SimpleNamespace(keep_op=lambda ctx: True, lookback=2)
    drifted_log = SimpleNamespace(_predicate_save_options=drifted_options)

    with pytest.raises(AttributeError, match="lookback_payload_policy"):
        _rerun_save_scope(drifted_log)  # type: ignore[arg-type]

    intact_options = SimpleNamespace(
        keep_op=lambda ctx: True, lookback=2, lookback_payload_policy="detached_raw"
    )
    intact_log = SimpleNamespace(_predicate_save_options=intact_options)

    layers_to_save, predicate, lookback, payload_policy = _rerun_save_scope(
        intact_log  # type: ignore[arg-type]
    )
    assert layers_to_save == "all"
    assert predicate is intact_options.keep_op
    assert lookback == 2
    assert payload_policy == "detached_raw"
