"""Typed capture-entry refusals from the honesty lane, each with a running remedy.

- ``emit_nvtx=True`` on a host without CUDA refuses typed
  (``emit_nvtx_unavailable``), as ``torch.autograd.profiler.emit_nvtx`` does,
  instead of silently swallowing every range push.
- ``on_nonfinite`` must be one of its four modes (``on_nonfinite_invalid``);
  the Recorder's compiled-model refusal is typed (``compiled_model_unsupported``)
  and states the condition it checks.
- Both compiled refusals name ``tl.bridge.depyf``.
- A non-Mapping ``input_kwargs`` (the two-positional spelling) refuses
  before backend dispatch, so non-torch backends get the same refusal.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _op_markers
from torchlens.errors import ArgumentTypeError, InvalidArgumentError
from torchlens.recording import Recorder


class _Tiny(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups_capture.py.
def test_emit_nvtx_refuses_without_cuda_and_the_remedy_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse typed on a CUDA-less host; dropping the option captures."""

    monkeypatch.setattr(_op_markers, "_nvtx_host_available", lambda: False)
    x = torch.randn(2, 4)
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(_Tiny(), x, capture=tl.options.CaptureOptions(emit_nvtx=True))
    assert info.value.fields["code"] == "emit_nvtx_unavailable"
    assert info.value.fields["argument"] == "emit_nvtx"
    assert "drop emit_nvtx=True" in info.value.fields["remedy"]

    trace = tl.trace(_Tiny(), x)
    assert trace.options.emit_nvtx is False and len(trace._layer_list) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_emit_nvtx_refusal_tracks_the_real_host() -> None:
    """On this host the probe is torch's own CUDA availability."""

    assert _op_markers._nvtx_host_available() is torch.cuda.is_available()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _options_validation.py, planted fault catches.
@pytest.mark.parametrize("value", [0.0])
def test_on_nonfinite_must_be_a_known_mode(value: object) -> None:
    """A value outside the closed vocabulary (bools included) refuses typed."""

    with pytest.raises(InvalidArgumentError) as info:
        tl.options.CaptureOptions(on_nonfinite=value)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "on_nonfinite_invalid"
    assert info.value.fields["argument"] == "on_nonfinite"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _training_validation.py, recording/_recorder.py.
def test_recorder_compiled_refusal_is_typed_and_accurate() -> None:
    """Typed, no false backward_ready claim, names depyf; remedy runs."""

    model = _Tiny().eval()
    compiled = torch.compile(model)
    with pytest.raises(InvalidArgumentError) as info:
        Recorder(compiled)
    exc = info.value
    assert exc.fields["code"] == "compiled_model_unsupported"
    assert "backward_ready" not in str(exc)
    assert "tl.bridge.depyf.dump" in exc.fields["remedy"]
    assert "compiled._orig_mod" in exc.fields["remedy"]

    recorder = Recorder(compiled._orig_mod, save=tl.func("relu"))
    assert recorder.model is model
    with pytest.warns(UserWarning, match="compiled model detected"):
        trace = tl.trace(compiled, torch.randn(2, 4))
    assert len(trace._layer_list) > 0


def test_compiled_callable_refusal_names_depyf_and_the_remedy_runs() -> None:
    """The plain compiled-callable refusal points at tl.bridge.depyf."""

    model = _Tiny().eval()
    compiled_fn = torch.compile(lambda t: model(t))
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(compiled_fn, torch.randn(2, 4))
    assert info.value.fields["code"] == "compiled_callable_unsupported"
    assert "tl.bridge.depyf.dump" in info.value.fields["remedy"]
    assert callable(tl.bridge.depyf.dump)

    assert len(tl.trace(model, torch.randn(2, 4))._layer_list) > 0


def test_jax_two_positional_remedy_runs() -> None:
    """JAX leg: the refusal, then the tuple spelling the remedy names, both run."""

    jnp = pytest.importorskip("jax.numpy")
    x = jnp.ones(3)
    with pytest.raises(ArgumentTypeError) as info:
        tl.trace(lambda p, q: p + q, x, input_kwargs=x, backend="jax")  # type: ignore[arg-type]
    assert info.value.fields["code"] == "input_kwargs_type_invalid"
    trace = tl.trace(lambda p, q: p + q, (x, x), backend="jax")
    assert len(trace._layer_list) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_outputs.py, campaign mutant kills.
def test_graphmodule_attribution_remedy_names_the_source_module() -> None:
    """An fx GraphModule root gets the export branch of the remedy."""

    import weakref
    from types import SimpleNamespace

    from torchlens.backends.torch.backend import output_attribution_remedy

    model = _Tiny().eval()
    graph_module = torch.fx.symbolic_trace(model)
    as_graph = SimpleNamespace(_source_model_ref=weakref.ref(graph_module))
    as_eager = SimpleNamespace(_source_model_ref=weakref.ref(model))
    assert "trace the source nn.Module before export" in output_attribution_remedy(as_graph)
    assert "bind/import torch functions" in output_attribution_remedy(as_eager)
