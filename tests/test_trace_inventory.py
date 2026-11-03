"""Pins for the trace inventory surfaces: ``sites_table`` and ``bill_of_materials``.

Both are read-only rollups of already-captured facts (DOCUMENTED-UNSTABLE
spellings). ``sites_table`` follows the L1 consumer matrix: a legacy keyless
artifact refuses ``site_key_unavailable``, never a silently empty table.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

pd = pytest.importorskip("pandas")


class LoopMlp(nn.Module):
    """Recurrent MLP: the shared linear reuses one site across passes."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer twice.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output after two recurrent passes.
        """

        for _ in range(2):
            x = torch.relu(self.fc(x))
        return x


def test_sites_table_groups_ops_by_site_key() -> None:
    """One row per distinct site key, in first-occurrence execution order."""

    torch.manual_seed(0)
    trace = tl.trace(LoopMlp(), torch.randn(1, 4))

    frame = trace.sites()

    assert list(frame.columns) == [
        "site_key",
        "module_site",
        "layer_type",
        "output_slot",
        "call_ordinal",
        "n_ops",
        "labels",
        "layer_labels",
        "passes",
        "shapes",
    ]
    assert len(frame) == len(set(frame["site_key"]))
    assert sum(frame["n_ops"]) == len(trace._layer_list)
    # The reused-module linear shares ONE site across both passes (per-call-
    # instance ordinal restart), so its row aggregates both pass labels.
    linear = frame.loc[frame["layer_type"] == "linear"].iloc[0]
    assert linear["module_site"] == "fc"
    assert linear["n_ops"] == 2
    assert linear["passes"] == (1, 2)
    assert linear["labels"] == ("linear_1_1:1", "linear_1_1:2")
    # First-occurrence execution order: input site first.
    assert frame.iloc[0]["layer_type"] == "input"


def test_sites_table_refuses_keyless_legacy_artifacts() -> None:
    """Legacy-artifact shape: no op carries a key -> typed refusal."""

    torch.manual_seed(1)
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    for op in trace._layer_list:
        op.site_key = None

    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.sites()
    assert excinfo.value.fields["code"] == "site_key_unavailable"
