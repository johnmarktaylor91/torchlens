"""Behaviour of ``trace[selector]``, ``tl.Selection`` members, ``Accessor.count_by`` and ``tl.in_module``."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidArgumentError
from torchlens.intervention.errors import AmbiguousSiteError, SiteResolutionError
from torchlens.intervention.types import TargetSpec

pytestmark = pytest.mark.smoke


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
    assert repr(selection) == "Selection[ACT](2 sites, 30 elements)"
    assert all(isinstance(row, tl.types.Site) for row in selection)
    assert all(row.num_selected == row.mask.numel() for row in selection)


def test_set_selector_is_never_capped(trace: tl.Trace) -> None:
    everything = trace[tl.where(lambda op: True)]
    assert len(everything) == len(trace.layer_list)


def test_set_selector_with_no_match_is_an_empty_selection(trace: tl.Trace) -> None:
    selection = trace[tl.func("conv2d")]
    assert isinstance(selection, tl.Selection)
    assert len(selection) == 0
    assert selection.is_empty


def test_selection_labels_slice_and_rows(trace: tl.Trace) -> None:
    selection = trace[tl.func("relu")]
    assert len(selection.labels) == 2
    for label, row in zip(selection.labels, selection, strict=True):
        record = trace[label]
        assert (record.layer_label, record.pass_index) == row.key
    head = selection[:1]
    assert isinstance(head, tl.Selection)
    assert len(head) == 1
    assert head.labels == selection.labels[:1]
    assert head.selector is selection.selector


def test_selection_to_pandas_has_one_row_per_site(trace: tl.Trace) -> None:
    pd = pytest.importorskip("pandas")
    frame = trace[tl.func("relu")].to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "label",
        "kind",
        "key",
        "num_selected",
        "numel",
        "shape",
        "relation",
        "source",
    ]
    assert len(frame) == 2


def test_tl_selector_resolves_through_the_same_door(trace: tl.Trace) -> None:
    selector = tl.func("relu") | tl.func("linear")
    selection = trace[selector]
    assert isinstance(selection, tl.Selection)
    assert len(selection) == 4
    assert not hasattr(tl.Selector, "resolve")


def test_exact_label_returns_the_one_record(trace: tl.Trace) -> None:
    label = trace[tl.func("relu")].labels[0]
    record = trace[tl.label(label)]
    assert record is trace[label]


def test_one_thing_spelling_refuses_several_matches(trace: tl.Trace) -> None:
    spec = TargetSpec(selector_kind="func", selector_value="relu")
    with pytest.raises(AmbiguousSiteError) as info:
        trace[spec]
    assert info.value.fields["code"] == "site_spelling_not_unique"
    assert info.value.fields["matched"] == 2
    one = TargetSpec(selector_kind="func", selector_value="relu")
    assert trace[TargetSpec(selector_kind="label", selector_value=trace[tl.func("relu")].labels[0])]
    assert one.selector_kind == "func"


def test_one_thing_spelling_with_no_match_still_refuses(trace: tl.Trace) -> None:
    with pytest.raises(SiteResolutionError):
        trace[tl.label("does_not_exist")]


def test_public_surface_has_no_site_table_or_resolve_sites() -> None:
    assert not hasattr(tl.Trace, "resolve_sites")
    assert "resolve_sites" not in dir(tl.validation)
    assert "SiteTable" not in dir(tl.intervention)
    for name in ("resolve_sites", "SiteTable"):
        with pytest.raises(AttributeError) as info:
            getattr(tl, name)
        assert "trace[selector]" in str(info.value) or "Selection" in str(info.value)


def test_count_by_one_and_two_keys(trace: tl.Trace) -> None:
    by_kind = trace.layers.count_by("kind")
    assert by_kind == {"linear": 2, "relu": 2}
    by_module = trace.layers.count_by("module")
    assert sum(by_module.values()) == len(trace.layers)
    both = trace.layers.count_by("module", "kind")
    assert all(isinstance(key, tuple) and len(key) == 2 for key in both)
    assert sum(both.values()) == len(trace.layers)
    assert trace.ops.count_by("kind") == {"linear": 2, "relu": 2}


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


def test_layer_accessor_has_no_by_methods(trace: tl.Trace) -> None:
    for name in ("by_operator", "by_module", "by_module_and_operator"):
        assert not hasattr(trace.layers, name)


def test_in_module_has_one_signature(trace: tl.Trace) -> None:
    assert list(inspect.signature(tl.in_module).parameters) == ["address"]
    selection = trace[tl.in_module("fc1")]
    assert len(selection) == 1
    assert trace[selection.labels[0]].module == "fc1"
    with pytest.raises(TypeError):
        tl.in_module(trace[selection.labels[0]], "fc1")  # type: ignore[call-arg]
