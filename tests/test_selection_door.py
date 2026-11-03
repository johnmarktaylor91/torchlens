"""Behaviour of ``trace[selector]``, ``tl.Selection`` members, ``Accessor.count_by`` and ``tl.in_module``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidArgumentError


class _TwoRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


@pytest.fixture
def trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(_TwoRelu(), torch.randn(3, 4))


def test_set_selector_returns_every_match_as_a_selection(trace: tl.Trace) -> None:
    selection = trace[tl.func("relu")]
    assert isinstance(selection, tl.Selection)
    assert len(selection) == 2
    assert selection.kind == "ACT"
    assert selection.selector is not None
    elements = sum(row.num_selected for row in selection)
    assert elements == 3 * 4 + 3 * 2
    assert repr(selection) == f"Selection[ACT](2 sites, {elements} elements)"
    assert all(isinstance(row, tl.types.Site) for row in selection)
    assert all(row.num_selected == row.mask.numel() for row in selection)


def test_set_selector_is_never_capped(trace: tl.Trace) -> None:
    everything = trace[tl.where(lambda op: True)]
    assert len(everything) == len(trace._layer_list)


@pytest.mark.smoke
def test_set_selector_with_no_match_is_an_empty_selection(trace: tl.Trace) -> None:
    selection = trace[tl.func("conv2d")]
    assert isinstance(selection, tl.Selection)
    assert len(selection) == 0
    assert selection.is_empty


def test_public_surface_has_no_site_table_or_resolve_sites() -> None:
    assert not hasattr(tl.Trace, "resolve_sites")
    assert "resolve_sites" not in dir(tl.validation)
    assert "SiteTable" not in dir(tl.intervention)
    for name in ("resolve_sites", "SiteTable"):
        with pytest.raises(AttributeError) as info:
            getattr(tl, name)
        assert "trace[selector]" in str(info.value) or "Selection" in str(info.value)


def test_count_by_refuses_no_key_and_unknown_key(trace: tl.Trace) -> None:
    with pytest.raises(InvalidArgumentError) as info:
        trace.layers.count_by()
    assert info.value.fields["code"] == "accessor_count_by_key_required"
    with pytest.raises(InvalidArgumentError) as info:
        trace.layers.count_by("colour")
    assert info.value.fields["code"] == "accessor_count_by_key_unknown"
    assert "'module'" in str(info.value) and "'kind'" in str(info.value)


def test_count_by_reads_any_scalar_field_of_other_accessors(trace: tl.Trace) -> None:
    by_class = trace.modules.count_by("class_name")
    assert by_class.get("Linear") == 2
    assert sum(trace.params.count_by("dtype").values()) == len(trace.params)


def test_site_record_returns_the_op_param_or_edge_it_names(trace: tl.Trace) -> None:
    """``Site.record`` hands back the record a row names (walk surface.5 / RC-45)."""

    act = trace[tl.func("relu")]
    for row in act:
        layer_label, pass_index = row.key
        op = row.record
        assert op is trace.ops[op.label]
        assert (op.layer_label, op.pass_number or 1) == (layer_label, pass_index)
    param_rows = trace[tl.param("fc1.weight")]
    assert len(param_rows) == 1
    assert param_rows[0].record is trace.params["fc1.weight"]


def test_site_record_on_edge_rows_is_the_edge_use_record() -> None:
    """EDGE rows return the EdgeUseRecord whose address they carry, never a synthesized one."""

    torch.manual_seed(0)
    replayable = tl.trace(
        _TwoRelu(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    relu_op = next(op for op in replayable.ops if op.func_name == "relu")
    use = relu_op.edge_uses[0]
    rows = replayable[use.__selection__()]
    assert rows.kind == "EDGE" and len(rows) == 1
    assert rows[0].record is use


def test_site_record_refuses_typed_once_the_trace_is_released() -> None:
    """A row outliving its trace refuses ``selection_trace_released`` instead of crashing."""

    import gc

    from torchlens.errors import SelectionError

    torch.manual_seed(0)
    trace = tl.trace(_TwoRelu(), torch.randn(3, 4))
    row = trace[tl.func("relu")][0]
    assert row.record is not None
    del trace
    gc.collect()
    with pytest.raises(SelectionError) as info:
        _ = row.record
    assert info.value.fields["code"] == "selection_trace_released"
