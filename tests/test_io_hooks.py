"""Public input and output detector registration (``tl.preprocessing``, ``tl.postprocessing``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.postprocessing
import torchlens.preprocessing
from torchlens._autoroute import input as input_registry, output as output_registry
from torchlens._autoroute._builtin_output import output_decoders_cache_key
from torchlens.postprocessing import ResolvedPostprocessing


class _CodesModel(nn.Module):
    """Takes integer character codes."""

    def forward(self, codes: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
        """Return the codes as floats, scaled (doubled by default)."""

        return codes.float() * scale


class _ThreeWay(nn.Module):
    """Three-class toy classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return three logits per batch item."""

        return self.fc(x)


class _OtherThreeWay(_ThreeWay):
    """Same shape, another class name."""


def _codes_detector(model: nn.Module, payload: object, **kwargs: Any) -> Any:
    """Trace a ``_CodesModel`` called on a string as its character codes."""

    if not isinstance(model, _CodesModel) or not isinstance(payload, str):
        return None
    return tl.trace(model, torch.tensor([[ord(ch) for ch in payload]]), **kwargs)


def _labels_detector(name: str, labels: dict[int, str], class_name: str = "_ThreeWay") -> Any:
    """Build an output detector labelling ``class_name`` models with ``labels``."""

    def detector(outputs: object, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        if meta.get("model_class_name") != class_name:
            return None
        if meta.get("style") not in (None, name):
            return None
        return ResolvedPostprocessing(
            source="project",
            identifier=name,
            verified=True,
            config={"num_labels": len(labels), "id2label": labels},
            description=f"{name} label bank",
            style="classification",
            selected_output_head=meta.get("head"),
        )

    return detector


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Restore both registries whatever a test registered."""

    with input_registry.snapshot(), output_registry.snapshot():
        yield


def test_namespaces_export_the_registry_doors() -> None:
    """Each processing namespace exports its registry's register and unregister."""

    assert tl.preprocessing.register == input_registry.register
    assert tl.preprocessing.unregister == input_registry.unregister
    assert {"register", "unregister"} <= set(tl.preprocessing.__all__)
    assert tl.postprocessing.register == output_registry.register
    assert tl.postprocessing.unregister == output_registry.unregister


def test_input_register_unregister_round_trip() -> None:
    """Register lists the detector, a duplicate refuses naming the door, unregister removes it."""

    before = [detector.name for detector in input_registry.list()]
    returned = tl.preprocessing.register(name="rt_codes", priority=50)(_codes_detector)
    assert returned is _codes_detector
    assert "rt_codes" in [detector.name for detector in input_registry.list()]
    with pytest.raises(ValueError, match=r"tl\.preprocessing\.unregister\('rt_codes'\)"):
        tl.preprocessing.register(name="rt_codes")(_codes_detector)
    tl.preprocessing.unregister("rt_codes")
    assert [detector.name for detector in input_registry.list()] == before
    with pytest.raises(KeyError, match="no detector named 'rt_codes'"):
        tl.preprocessing.unregister("rt_codes")


def test_output_register_unregister_round_trip() -> None:
    """Register lists the detector, a duplicate refuses naming the door, unregister removes it."""

    before = [detector.name for detector in output_registry.list()]
    detector = _labels_detector("rt_labels", {0: "a", 1: "b", 2: "c"})
    assert tl.postprocessing.register(name="rt_labels")(detector) is detector
    assert "rt_labels" in [entry.name for entry in output_registry.list()]
    with pytest.raises(ValueError, match=r"tl\.postprocessing\.unregister\('rt_labels'\)"):
        tl.postprocessing.register(name="rt_labels")(detector)
    tl.postprocessing.unregister("rt_labels")
    assert [entry.name for entry in output_registry.list()] == before
    with pytest.raises(KeyError, match="no detector named 'rt_labels'"):
        tl.postprocessing.unregister("rt_labels")


def test_input_detector_applies_to_next_trace_only() -> None:
    """Traces before registration and after unregister never reach the detector."""

    calls: list[object] = []

    def recording(model: nn.Module, payload: object, **kwargs: Any) -> Any:
        calls.append(payload)
        return _codes_detector(model, payload, **kwargs)

    model = _CodesModel()
    before = tl.trace(model, torch.tensor([[1, 2]]))
    assert calls == []
    tl.preprocessing.register(name="codes_next", priority=5)(recording)
    after = tl.trace(model, "hi")

    assert calls[0] == "hi"
    assert torch.equal(after.output_ops[0].out, torch.tensor([[208.0, 210.0]]))
    assert torch.equal(before.output_ops[0].out, torch.tensor([[2.0, 4.0]]))

    tl.preprocessing.unregister("codes_next")
    seen = len(calls)
    tl.trace(model, torch.tensor([[3]]))
    assert len(calls) == seen


def test_input_detector_declines_and_forwards_options() -> None:
    """``None`` declines; a claimed call keeps the caller's capture options."""

    calls: list[object] = []

    def recording(model: nn.Module, payload: object, **kwargs: Any) -> Any:
        calls.append(payload)
        return _codes_detector(model, payload, **kwargs)

    tl.preprocessing.register(name="codes_recording", priority=5)(recording)
    tl.trace(_ThreeWay(), torch.randn(1, 4))
    assert len(calls) == 1  # offered the call, declined it

    trace = tl.trace(_CodesModel(), "ab", input_kwargs={"scale": 3.0})
    assert calls[1] == "ab"
    assert torch.equal(trace.output_ops[0].out, torch.tensor([[291.0, 294.0]]))


def test_explicit_input_transform_skips_input_detectors() -> None:
    """An explicit ``input_transform`` bypasses every registered input detector."""

    calls: list[object] = []

    def recording(model: nn.Module, payload: object, **kwargs: Any) -> None:
        calls.append(payload)
        return None

    tl.preprocessing.register(name="codes_skipped", priority=5)(recording)
    capture = tl.options.CaptureOptions(
        input_transform=lambda raw: torch.tensor([[ord(ch) for ch in raw]])
    )
    trace = tl.trace(_CodesModel(), "a", capture=capture)
    assert calls == []
    assert torch.equal(trace.output_ops[0].out, torch.tensor([[194.0]]))


def test_output_decoder_applies_to_next_trace_only() -> None:
    """A trace made before registration keeps no decode; the next trace is labelled."""

    torch.manual_seed(0)
    model = _ThreeWay()
    x = torch.randn(2, 4)
    before = tl.trace(model, x)
    tl.postprocessing.register(name="three_labels", priority=5)(
        _labels_detector("three_labels", {0: "x", 1: "y", 2: "z"})
    )
    after = tl.trace(model, x)

    rows = after.decode_output()["rows"]
    assert {row["label"] for row in rows} <= {"x", "y", "z"}
    assert [row["batch_item"] for row in after.decode_output(k=1)["rows"]] == [0, 1]
    assert after.provenance.output_postprocessor.identifier == "three_labels"
    assert before.provenance.output_postprocessor is None
    # A read-time decode consults the registry as it is now; the record the
    # earlier capture stored stays unchanged.
    before.decode_output()
    assert before.provenance.output_postprocessor is None

    tl.postprocessing.unregister("three_labels")
    assert tl.trace(model, x).provenance.output_postprocessor is None


def test_output_decoder_filters_by_model_class_and_decodes_on_demand() -> None:
    """``meta["model_class_name"]`` filters; ``style=name`` decodes on demand."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)
    tl.postprocessing.register(name="mapped", priority=5)(
        _labels_detector("mapped", {0: "zero", 1: "one", 2: "two"})
    )
    other = tl.trace(_OtherThreeWay(), x)
    assert other.provenance.output_postprocessor is None

    trace = tl.trace(_ThreeWay(), x)
    top = trace.decode_output(k=1)["rows"][0]
    assert top["label"] == ["zero", "one", "two"][top["class_index"]]
    on_demand = trace.decode_output(k=1, style="mapped")["rows"][0]
    assert on_demand["label"] == top["label"]


def test_output_decoder_none_declines() -> None:
    """A detector returning ``None`` leaves the capture undecoded."""

    tl.postprocessing.register(name="declines")(lambda outputs, meta: None)
    trace = tl.trace(_ThreeWay(), torch.randn(1, 4))
    assert trace.provenance.output_postprocessor is None


def test_output_registry_changes_capture_cache_key() -> None:
    """Registering or re-registering a detector changes the capture-cache fragment."""

    base = output_decoders_cache_key()
    tl.postprocessing.register(name="cache_labels")(lambda outputs, meta: None)
    registered = output_decoders_cache_key()
    tl.postprocessing.unregister("cache_labels")
    tl.postprocessing.register(name="cache_labels")(lambda outputs, meta: None)
    assert len({base, registered, output_decoders_cache_key()}) == 3
