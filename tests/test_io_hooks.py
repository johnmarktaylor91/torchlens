"""Public input preprocessor and output decoder registration (``tl.inputs``, ``tl.outputs``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._autoroute import input as input_registry, output as output_registry
from torchlens._autoroute._builtin_output import output_decoders_cache_key
from torchlens.errors import PayloadUnavailableError


class _CodesModel(nn.Module):
    """Takes integer character codes."""

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Return the codes as floats, doubled."""

        return codes.float() * 2


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


def _codes(model: nn.Module, raw: object) -> torch.Tensor | None:
    """Encode a string as a batch of one row of character codes."""

    if not isinstance(raw, str):
        return None
    return torch.tensor([[ord(ch) for ch in raw]])


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Restore both registries whatever a test registered."""

    with input_registry.snapshot(), output_registry.snapshot():
        yield


def test_namespaces_are_lazy_public_modules() -> None:
    """``tl.inputs`` and ``tl.outputs`` resolve to their modules with two public names each."""

    import torchlens.inputs
    import torchlens.outputs

    assert tl.inputs is torchlens.inputs
    assert tl.outputs is torchlens.outputs
    assert tl.inputs.__all__ == ["register", "unregister"]
    assert tl.outputs.__all__ == ["register", "unregister"]


def test_input_register_unregister_round_trip() -> None:
    """Register lists the preprocessor, a duplicate refuses, unregister removes it."""

    before = [detector.name for detector in input_registry.list()]
    returned = tl.inputs.register(name="rt_codes", priority=50)(_codes)
    assert returned is _codes
    assert "rt_codes" in [detector.name for detector in input_registry.list()]
    with pytest.raises(ValueError, match=r"tl\.inputs\.unregister\('rt_codes'\)"):
        tl.inputs.register(name="rt_codes")(_codes)
    tl.inputs.unregister("rt_codes")
    assert [detector.name for detector in input_registry.list()] == before
    with pytest.raises(KeyError, match="no detector named 'rt_codes'"):
        tl.inputs.unregister("rt_codes")


def test_output_register_unregister_round_trip() -> None:
    """Register lists the decoder, a duplicate refuses, unregister removes it."""

    before = [detector.name for detector in output_registry.list()]

    def labels(outputs: object) -> list[str]:
        return ["a", "b", "c"]

    assert tl.outputs.register(name="rt_labels")(labels) is labels
    assert "rt_labels" in [detector.name for detector in output_registry.list()]
    with pytest.raises(ValueError, match=r"tl\.outputs\.unregister\('rt_labels'\)"):
        tl.outputs.register(name="rt_labels")(labels)
    tl.outputs.unregister("rt_labels")
    assert [detector.name for detector in output_registry.list()] == before
    with pytest.raises(KeyError, match="no detector named 'rt_labels'"):
        tl.outputs.unregister("rt_labels")


def test_input_preprocessor_applies_to_next_trace_only() -> None:
    """A trace made before registration is unchanged; the next trace uses the preprocessor."""

    model = _CodesModel()
    before = tl.trace(model, torch.tensor([[1, 2]]))
    tl.inputs.register(name="codes_next", class_name="_CodesModel")(_codes)
    after = tl.trace(model, "hi")

    assert after.input == "hi"
    record = after.provenance.input_preprocessor
    assert record.identifier == "codes_next"
    assert record.source == "registered_input_preprocessor"
    assert torch.equal(after.output_ops[0].out, torch.tensor([[208.0, 210.0]]))
    assert before.provenance.input_preprocessor is None

    tl.inputs.unregister("codes_next")
    plain = tl.trace(model, torch.tensor([[3]]))
    assert plain.provenance.input_preprocessor is None


def test_input_preprocessor_declines_and_filters_by_class() -> None:
    """``None`` declines, and ``class_name=`` keeps other models on the plain path."""

    calls: list[object] = []

    def recording(model: nn.Module, raw: object) -> torch.Tensor | None:
        calls.append(raw)
        return _codes(model, raw)

    tl.inputs.register(name="codes_filtered", class_name="NotThisModel")(recording)
    trace = tl.trace(_CodesModel(), torch.tensor([[1]]))
    assert calls == []
    assert trace.provenance.input_preprocessor is None

    tl.inputs.unregister("codes_filtered")
    tl.inputs.register(name="codes_declines", class_name="_Codes*")(recording)
    tl.trace(_CodesModel(), torch.tensor([[1]]))
    assert len(calls) == 1


def test_input_preprocessor_runs_once_per_trace() -> None:
    """The route's prepared value is reused by the capture's input transform."""

    calls: list[object] = []

    def counting(model: nn.Module, raw: object) -> torch.Tensor | None:
        calls.append(raw)
        return _codes(model, raw)

    tl.inputs.register(name="codes_once")(counting)
    tl.trace(_CodesModel(), "ab")
    assert calls == ["ab"]


def test_output_decoder_applies_to_next_trace_only() -> None:
    """A trace made before registration keeps no decode; the next trace is labelled."""

    torch.manual_seed(0)
    model = _ThreeWay()
    x = torch.randn(2, 4)
    before = tl.trace(model, x)
    tl.outputs.register(name="three_labels", class_name="_ThreeWay")(
        lambda outputs: ["x", "y", "z"]
    )
    after = tl.trace(model, x)

    rows = after.decode_output()["rows"]
    assert {row["label"] for row in rows} <= {"x", "y", "z"}
    assert [row["batch_item"] for row in after.decode_output(k=1)["rows"]] == [0, 1]
    assert after.provenance.output_postprocessor.identifier == "three_labels"
    assert after.provenance.output_postprocessor.source == "registered_output_decoder"
    assert before.provenance.output_postprocessor is None
    with pytest.raises(PayloadUnavailableError):
        before.decode_output()


def test_output_decoder_mapping_class_filter_and_style() -> None:
    """Mapping labels decode; ``class_name=`` filters; ``style=name`` decodes on demand."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)
    tl.outputs.register(name="mapped", class_name="_ThreeWay")(
        lambda outputs: {0: "zero", 1: "one", 2: "two"}
    )
    other = tl.trace(_OtherThreeWay(), x)
    assert other.provenance.output_postprocessor is None

    model = _ThreeWay()
    trace = tl.trace(model, x)
    top = trace.decode_output(k=1)["rows"][0]
    assert top["label"] == ["zero", "one", "two"][top["class_index"]]
    on_demand = trace.decode_output(k=1, style="mapped")["rows"][0]
    assert on_demand["label"] == top["label"]


def test_output_decoder_none_declines() -> None:
    """A decoder returning ``None`` leaves the capture undecoded."""

    tl.outputs.register(name="declines")(lambda outputs: None)
    trace = tl.trace(_ThreeWay(), torch.randn(1, 4))
    assert trace.provenance.output_postprocessor is None


def test_output_registry_changes_capture_cache_key() -> None:
    """Registering or re-registering a decoder changes the capture-cache fragment."""

    base = output_decoders_cache_key()
    tl.outputs.register(name="cache_labels")(lambda outputs: ["a"])
    registered = output_decoders_cache_key()
    tl.outputs.unregister("cache_labels")
    tl.outputs.register(name="cache_labels")(lambda outputs: ["b"])
    assert len({base, registered, output_decoders_cache_key()}) == 3
