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

    frame = trace.sites_table()

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
    assert sum(frame["n_ops"]) == len(list(trace.ops))
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
    for op in trace.ops:
        op.site_key = None

    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.sites_table()
    assert excinfo.value.fields["code"] == "site_key_unavailable"


def test_bill_of_materials_reports_captured_facts() -> None:
    """Every figure matches the fields the trace already carries."""

    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.BatchNorm1d(3))
    trace = tl.trace(model, torch.randn(2, 4), save=tl.func("relu"))

    bom = trace.bill_of_materials()

    assert set(bom) == {
        "capture",
        "graph",
        "parameters",
        "buffers",
        "activations",
        "backward",
        "annotations",
        # A09 sumfam D20: logged values are inventory (read-back surface).
        "logged_values",
    }
    assert bom["capture"]["backend"] == "torch"
    assert bom["capture"]["outcome_status"] == "complete"
    assert bom["capture"]["model_class_name"] == "Sequential"
    assert bom["graph"]["num_ops"] == len(list(trace.ops))
    assert bom["graph"]["num_layers"] == trace.num_layers
    assert bom["graph"]["num_modules"] == trace.num_modules
    assert bom["parameters"]["num_params"] == trace.num_params
    assert bom["parameters"]["num_param_tensors"] == len(trace.params)
    assert bom["buffers"]["num_buffer_tensors"] == len(trace.buffers)
    saved = [op for op in trace.ops if op.has_saved_activation]
    # Payload-scope law (C02; sumfam D8): BOM's byte figures are
    # retained_now; the at_capture facts print BESIDE them.
    assert bom["activations"]["payload_scope"] == "retained_now"
    assert bom["activations"]["num_saved_at_capture"] == len(saved)
    assert bom["activations"]["at_capture_memory"] == sum(int(op.activation_memory) for op in saved)
    # A live full-save trace still holds everything it captured.
    assert bom["activations"]["retained_now_memory"] == bom["activations"]["at_capture_memory"]
    assert bom["backward"]["num_grad_fn_records"] == 0
    assert bom["annotations"] == ()


def test_bill_of_materials_is_read_only_and_repeatable() -> None:
    """Building the inventory twice yields equal sections and mutates nothing."""

    torch.manual_seed(3)
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))

    first = trace.bill_of_materials()
    second = trace.bill_of_materials()

    assert first == second
