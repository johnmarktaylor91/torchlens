"""``tl.record`` dispatch through ``capability_implementations["fastlog"]``.

A backend that binds a record producer records through the neutral event
interface; a False flag, an unbound flag or a binding that is neither the torch
recorder nor a producer refuses typed; torch-path options and producer contract
breaches refuse instead of being ignored.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from _parity_b_record_fake import FAKE_BACKEND, FakeProducer, fake_backend
from torch import nn

import torchlens as tl
from torchlens.errors import BackendUnsupportedError, RecorderStateError
from torchlens.recording import CaptureSpec, RecordContext
from torchlens.recording._backend_recorder import BackendRecorder

pytestmark = [pytest.mark.backend_parity]


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def _ops(ctx: RecordContext) -> bool:
    return ctx.kind == "op"


def _register_variant(**changes: Any) -> None:
    spec = tl.backends.spec(FAKE_BACKEND)
    tl.backends.register(dataclasses.replace(spec, **changes), replace=True)


def test_fastlog_false_refuses_with_the_capability_code() -> None:
    with fake_backend(FakeProducer()):
        spec = tl.backends.spec(FAKE_BACKEND)
        implementations = dict(spec.capability_implementations)
        implementations.pop("fastlog")
        _register_variant(
            capabilities=dataclasses.replace(spec.capabilities, fastlog=False),
            capability_implementations=implementations,
        )
        with pytest.raises(BackendUnsupportedError, match="capabilities.fastlog=False") as info:
            tl.record(_model(), torch.ones(1, 4), save=_ops, backend=FAKE_BACKEND)
    assert info.value.fields["code"] == "fastlog_unsupported"


def test_binding_that_is_not_a_producer_refuses() -> None:
    with fake_backend(FakeProducer()):
        implementations = dict(tl.backends.spec(FAKE_BACKEND).capability_implementations)
        implementations["fastlog"] = lambda: object()
        _register_variant(capability_implementations=implementations)
        with pytest.raises(BackendUnsupportedError, match="no non-torch dispatch path") as info:
            tl.record(_model(), torch.ones(1, 4), save=_ops, backend=FAKE_BACKEND)
    assert info.value.fields["code"] == "fastlog_unsupported"


_TORCH_PATH_OPTIONS = {
    "intervene": ({"intervene": lambda ctx: None}, "interventions_unsupported"),
    "save_grads": (
        {"capture": tl.options.CaptureOptions(save_grads=True)},
        "backward_capture_unsupported",
    ),
    "seed": ({"capture": tl.options.CaptureOptions(seed=0)}, "rng_replay_unsupported"),
    "activation_transform": (
        {"capture": tl.options.CaptureOptions(activation_transform=torch.abs)},
        "trace_option_unsupported",
    ),
    "generate_kwargs": (
        {"generate_kwargs": {"max_new_tokens": 1}},
        "record_generation_unsupported",
    ),
    "postprocess": (
        {"recording": tl.options.RecordingOptions(postprocess="module_path_strings")},
        "trace_option_unsupported",
    ),
}


@pytest.mark.parametrize("option", sorted(_TORCH_PATH_OPTIONS))
def test_torch_path_options_refuse_instead_of_being_ignored(option: str) -> None:
    kwargs, code = _TORCH_PATH_OPTIONS[option]
    with fake_backend(FakeProducer()) as producer:
        with pytest.raises(BackendUnsupportedError) as info:
            tl.record(_model(), torch.ones(1, 4), save=_ops, backend=FAKE_BACKEND, **kwargs)
    assert info.value.fields["code"] == code
    assert producer.passes == 0


def test_supported_options_admit_an_option_the_producer_honours() -> None:
    producer = FakeProducer()
    producer.supported_options = frozenset({"seed"})
    with fake_backend(producer):
        recording = tl.record(
            _model(),
            torch.ones(1, 4),
            save=_ops,
            capture=tl.options.CaptureOptions(seed=0),
            backend=FAKE_BACKEND,
        )
    assert recording.records


@pytest.mark.parametrize(
    "spec",
    [
        CaptureSpec(save_out=True, save_metadata=True, save_grad=True),
        CaptureSpec(save_out=True, save_metadata=True, dtype=torch.float16),
    ],
    ids=["save_grad", "dtype"],
)
def test_capture_spec_fields_without_a_codec_step_refuse(spec: CaptureSpec) -> None:
    def save(ctx: RecordContext) -> CaptureSpec | bool:
        return spec if ctx.kind == "op" else False

    with fake_backend(FakeProducer()):
        with pytest.raises(BackendUnsupportedError) as info:
            tl.record(_model(), torch.ones(1, 4), save=save, backend=FAKE_BACKEND)
    assert info.value.fields["code"] == "trace_option_unsupported"


class _ContractBreaker(FakeProducer):
    """Producer that breaks one rule of the event interface."""

    def __init__(self, breach: str) -> None:
        super().__init__()
        self.breach = breach

    def run_pass(self, prepared: Any, input_args: Any, input_kwargs: Any, sink: Any) -> Any:
        from torchlens.recording._backend_events import BackendOpEvent

        event = BackendOpEvent(raw_label="relu_1_1", layer_type="relu", type_index=1, raw_index=1)
        decision = sink.emit(event)
        if self.breach == "undelivered":
            return None
        sink.deliver(decision.request, torch.ones(1))
        if self.breach == "double_delivery":
            sink.deliver(decision.request, torch.ones(1))
        if self.breach == "emit_after_stop":
            sink.emit(dataclasses.replace(event, raw_label="relu_1_2", raw_index=2))
        return None


@pytest.mark.parametrize("breach", ["undelivered", "double_delivery", "emit_after_stop"])
def test_producer_contract_breaches_refuse(breach: str) -> None:
    stop_after = (lambda ctx: True) if breach == "emit_after_stop" else None
    with pytest.raises(RecorderStateError):
        with BackendRecorder(
            _model(), producer=_ContractBreaker(breach), save=_ops, stop_after=stop_after
        ) as recorder:
            recorder(torch.ones(1, 4))


def test_event_kinds_outside_the_interface_refuse() -> None:
    from torchlens.recording._backend_events import BackendOpEvent

    with pytest.raises(RecorderStateError, match="module_stack"):
        BackendOpEvent(
            raw_label="x", layer_type="relu", type_index=1, raw_index=1, kind="module_enter"
        )
    with pytest.raises(RecorderStateError):
        BackendOpEvent(raw_label="", layer_type="relu", type_index=1, raw_index=1)
