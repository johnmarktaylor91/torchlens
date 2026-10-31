"""SEL batch-2 rows: refusals whose remedies run, the appliance door, and loud facet fan-out.

- Bug 6: the documented ``tl.trace(save=[...])`` list form runs, and every remedy the
  ``save=`` door emits is executed here.
- U27.2: ``activations_by_address`` on an unmatched address refuses typed and lists the
  addresses that exist.
- forks.8: ``trace.using(name)`` returns a registered appliance; unknown names refuse typed
  with the registered names; outside packages register through the same function.
- forks.9: an unscoped ``tl.head(i)`` says how many sites it writes, and the number equals
  the facet slices the run actually edits.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import errors
from torchlens.accessors import _appliances


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer1(x))


@pytest.fixture
def net() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Net().eval(), torch.randn(2, 4)


def _saved_labels(trace: Any) -> list[str]:
    return [op.label for op in trace.layer_list if op.has_saved_activation]


_SAVE_SPELLING = re.compile(r"save=(\[[^\]]*\]|tl\.\w+\([^)]*\)|'[^']*')")


def test_save_list_form_runs_and_matches_layers_to_save(net) -> None:
    model, x = net
    by_save = tl.trace(model, x, save=["layer1", "fc"])
    by_capture = tl.trace(
        model, x, capture=tl.options.CaptureOptions(layers_to_save=["layer1", "fc"])
    )
    assert _saved_labels(by_save) == _saved_labels(by_capture)
    torch.testing.assert_close(by_save["fc"].out, model(x))
    assert _saved_labels(tl.trace(model, x, save="all")) == _saved_labels(tl.trace(model, x))


def test_save_type_refusal_remedies_all_run(net) -> None:
    model, x = net
    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        tl.trace(model, x, save=3)
    assert exc_info.value.fields["code"] == "save_predicate_type_invalid"
    spellings = _SAVE_SPELLING.findall(exc_info.value.fields["remedy"])
    assert len(spellings) == 3
    for spelling in spellings:
        trace = eval(f"tl.trace(model, x, save={spelling})", {"tl": tl, "model": model, "x": x})
        assert _saved_labels(trace), spelling


def test_save_selection_conflict_remedies_run(net) -> None:
    model, x = net
    capture = tl.options.CaptureOptions(layers_to_save=["fc"])
    with pytest.raises(errors.ArgumentConflictError) as exc_info:
        tl.trace(model, x, save=["layer1"], capture=capture)
    assert exc_info.value.fields["code"] == "save_selection_conflict"
    calls = re.findall(r"tl\.trace\([^`]*?\)(?= without| and|$)", exc_info.value.fields["remedy"])
    assert len(calls) == 2
    namespace = {"tl": tl, "model": model, "x": x, "CaptureOptions": tl.options.CaptureOptions}
    for call in calls:
        assert _saved_labels(eval(call, namespace)), call


def test_activations_by_address_lists_existing_addresses(net) -> None:
    model, x = net
    trace = tl.trace(model, x)
    assert [op.label for op in trace.activations_by_address("fc")] == [trace["fc"].label]
    with pytest.raises(errors.InvalidArgumentError) as exc_info:
        trace.activations_by_address("fc2")
    fields = exc_info.value.fields
    assert fields["code"] == "activation_address_unmatched"
    assert {"fc", "layer1.0", "layer1.1"} <= set(fields["known_addresses"])
    assert "'fc'" in str(exc_info.value)
    remedy_call = re.search(r"activations_by_address\('[^']*'\)", fields["remedy"])
    assert remedy_call is not None
    assert eval(f"trace.{remedy_call.group(0)}", {"trace": trace})


@pytest.fixture
def empty_registry() -> Any:
    """Isolate the appliance registry: unregister everything a test added."""

    before = set(_appliances._APPLIANCES.list_ids())
    hidden = {name: _appliances._APPLIANCES.get(name) for name in before}
    for name in before:
        _appliances._APPLIANCES.unregister(name)
    yield _appliances._APPLIANCES
    for name in set(_appliances._APPLIANCES.list_ids()):
        _appliances._APPLIANCES.unregister(name)
    for name, entry in hidden.items():
        tl.accessors.register_appliance(name, entry.factory, summary=entry.summary)


class _OutsideAppliance:
    """Stands in for an appliance shipped by another package."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace

    def op_count(self) -> int:
        return len(self.trace.layer_list)


def test_using_returns_registered_appliances(net, empty_registry) -> None:
    model, x = net
    trace = tl.trace(model, x)
    tl.accessors.register_appliance("outside", _OutsideAppliance, summary="op counter")
    tl.accessors.register_appliance("lazy", "builtins:id")
    assert trace.using("outside").op_count() == len(trace.layer_list)
    assert trace.using("outside").trace is trace
    assert trace.using("lazy") == id(trace)
    assert tl.accessors.registered_appliances() == {"lazy": "", "outside": "op counter"}


def test_using_unknown_name_refuses_with_registered_names(net, empty_registry) -> None:
    model, x = net
    trace = tl.trace(model, x)
    with pytest.raises(errors.InvalidArgumentError) as empty_info:
        trace.using("mi")
    assert empty_info.value.fields["code"] == "appliance_unknown"
    assert "register_appliance(name, factory)" in empty_info.value.fields["remedy"]
    tl.accessors.register_appliance("outside", _OutsideAppliance)
    with pytest.raises(errors.InvalidArgumentError) as exc_info:
        trace.using("mi")
    assert exc_info.value.fields["registered"] == ("outside",)
    remedy_call = re.search(r"trace\.using\('[^']*'\)", exc_info.value.fields["remedy"])
    assert remedy_call is not None
    assert isinstance(eval(remedy_call.group(0), {"trace": trace}), _OutsideAppliance)


def test_register_appliance_refusals(empty_registry) -> None:
    tl.accessors.register_appliance("outside", _OutsideAppliance)
    with pytest.raises(errors.ArgumentConflictError) as taken:
        tl.accessors.register_appliance("outside", _OutsideAppliance)
    assert taken.value.fields["code"] == "appliance_name_taken"
    tl.accessors.register_appliance("outside", _OutsideAppliance, replace=True)
    for name, factory, error_type in [
        (3, _OutsideAppliance, errors.ArgumentTypeError),
        ("9lives", _OutsideAppliance, errors.InvalidArgumentError),
        ("ok", 3, errors.ArgumentTypeError),
        ("ok", "no_colon_path", errors.InvalidArgumentError),
    ]:
        with pytest.raises(error_type) as exc_info:
            tl.accessors.register_appliance(name, factory)  # type: ignore[arg-type]
        assert exc_info.value.fields["code"] == "appliance_registration_invalid"


def _gpt2() -> tuple[nn.Module, torch.Tensor]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=12, n_head=12, n_embd=48, vocab_size=64, n_positions=16
    )
    return transformers.GPT2LMHeadModel(config).eval(), torch.randint(0, 64, (1, 6))


def _head_slices(trace: Any, head: int) -> list[torch.Tensor]:
    slices = []
    for layer in range(12):
        view = trace.modules[f"transformer.h.{layer}.attn"].facets.head(head)
        slices.extend(view[name] for name in ("q", "k", "v"))
    return [torch.as_tensor(item.value if hasattr(item, "value") else item) for item in slices]


@pytest.mark.heavy
def test_unscoped_head_reports_the_sites_it_writes() -> None:
    model, ids = _gpt2()
    capture = tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)
    clean = tl.trace(model, ids, capture=capture)
    edited = clean.fork("ablated")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edited.attach_hooks(tl.head(5), tl.zero_ablate())
    reports = [
        w.message
        for w in caught
        if getattr(w.message, "fields", {}).get("code") == "facet_selector_fanout"
    ]
    assert len(reports) == 1
    assert "tl.head(5)" in str(reports[0])
    reported = reports[0].fields["site_count"]
    assert f"writes {reported} sites" in str(reports[0])
    edited.run(model, ids)
    written = sum(
        bool(torch.count_nonzero(after) == 0 and torch.count_nonzero(before) > 0)
        for before, after in zip(_head_slices(clean, 5), _head_slices(edited, 5), strict=True)
    )
    assert reported == written == 36
    scoped = clean.fork("scoped")
    with warnings.catch_warnings(record=True) as caught_scoped:
        warnings.simplefilter("always")
        scoped.attach_hooks(tl.head(5, "q").in_module("transformer.h.0.attn"), tl.zero_ablate())
    assert not [
        w
        for w in caught_scoped
        if getattr(w.message, "fields", {}).get("code") == "facet_selector_fanout"
    ]


class _UnregisteredAttention(nn.Module):
    """An attention block whose class no facet recipe knows."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)
        self.v = nn.Linear(8, 8)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.q(h) @ self.k(h).transpose(-1, -2), dim=-1)
        return weights @ self.v(h)


class _AttnHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _UnregisteredAttention()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.attn(h)


def test_facet_doors_refuse_typed_on_an_unregistered_attention_class() -> None:
    """U36.1: both facet doors refuse with MissingFacetError, never a bare KeyError."""

    from torchlens.errors import MissingFacetError

    trace = tl.trace(_AttnHost().eval(), torch.randn(1, 4, 8))
    for view in (trace.modules["attn"].facets, trace.module_calls["attn:1"].facets):
        with pytest.raises(MissingFacetError) as exc_info:
            view["q"]
        assert type(exc_info.value) is not KeyError
        assert "_UnregisteredAttention" in str(exc_info.value)
        assert "torchlens.semantic.register" in str(exc_info.value)
        assert view.get("q") is None
        assert not hasattr(view, "q")
