"""Typed capture-entry refusals from the honesty lane, each with a running remedy.

- U35.3: ``emit_nvtx=True`` on a host without CUDA refuses typed
  (``emit_nvtx_unavailable``), as ``torch.autograd.profiler.emit_nvtx`` does,
  instead of silently swallowing every range push.
- U31.4: ``raise_on_nan`` must be a real bool (``raise_on_nan_type_invalid``);
  the Recorder's compiled-model refusal is typed (``compiled_model_unsupported``)
  and states the condition it checks.
- U53.3: both compiled refusals name ``tl.bridge.depyf``.
- U11.2: a non-Mapping ``input_kwargs`` (the two-positional spelling) refuses
  before backend dispatch, so non-torch backends get the same refusal.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _op_markers
from torchlens.errors import ArgumentTypeError, InvalidArgumentError
from torchlens.fastlog import Recorder


class _Tiny(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class _Pair(nn.Module):
    """Two positional inputs."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def test_emit_nvtx_refuses_without_cuda_and_the_remedy_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U35.3: refuse typed on a CUDA-less host; dropping the option captures."""

    monkeypatch.setattr(_op_markers, "_nvtx_host_available", lambda: False)
    x = torch.randn(2, 4)
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(_Tiny(), x, capture=tl.options.CaptureOptions(emit_nvtx=True))
    assert info.value.fields["code"] == "emit_nvtx_unavailable"
    assert info.value.fields["argument"] == "emit_nvtx"
    assert "drop emit_nvtx=True" in info.value.fields["remedy"]

    trace = tl.trace(_Tiny(), x)
    assert trace.options.emit_nvtx is False and len(trace.layer_list) > 0


def test_emit_nvtx_refusal_tracks_the_real_host() -> None:
    """On this host the probe is torch's own CUDA availability."""

    assert _op_markers._nvtx_host_available() is torch.cuda.is_available()


@pytest.mark.parametrize("value", [0.0, 1, "yes", None])
def test_raise_on_nan_must_be_a_real_bool(value: object) -> None:
    """U31.4: a non-bool (falsy ones used to disable the check) refuses typed."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.options.CaptureOptions(raise_on_nan=value)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "raise_on_nan_type_invalid"
    assert info.value.fields["received_type"] == type(value).__name__


def test_raise_on_nan_remedy_runs() -> None:
    """The remedy spelling, a real bool, captures and keeps the tripwire armed."""

    class _NaN(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.log(x - 10.0)

    with pytest.raises(Exception, match=r"(?i)nan|inf|non-finite"):
        tl.trace(_NaN(), torch.zeros(2), capture=tl.options.CaptureOptions(raise_on_nan=True))
    clean = tl.trace(
        _Tiny(), torch.randn(2, 4), capture=tl.options.CaptureOptions(raise_on_nan=False)
    )
    assert len(clean.layer_list) > 0


def test_recorder_compiled_refusal_is_typed_and_accurate() -> None:
    """U31.4 + U53.3: typed, no false backward_ready claim, names depyf; remedy runs."""

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
    assert len(trace.layer_list) > 0


def test_compiled_callable_refusal_names_depyf_and_the_remedy_runs() -> None:
    """U53.3: the plain compiled-callable refusal points at tl.bridge.depyf."""

    model = _Tiny().eval()
    compiled_fn = torch.compile(lambda t: model(t))
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(compiled_fn, torch.randn(2, 4))
    assert info.value.fields["code"] == "compiled_callable_unsupported"
    assert "tl.bridge.depyf.dump" in info.value.fields["remedy"]
    assert callable(tl.bridge.depyf.dump)

    assert len(tl.trace(model, torch.randn(2, 4)).layer_list) > 0


def test_two_positional_inputs_refuse_before_dispatch_and_the_remedy_runs() -> None:
    """U11.2: the torch path refuses typed; the tuple spelling captures."""

    a, b = torch.randn(3), torch.randn(3)
    with pytest.raises(ArgumentTypeError) as info:
        tl.trace(_Pair(), a, b)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "input_kwargs_type_invalid"
    trace = tl.trace(_Pair(), (a, b))
    assert torch.equal(trace.output_ops[0].out, a + b)


def test_two_positional_inputs_refuse_for_an_explicit_non_torch_backend() -> None:
    """U11.2: the refusal fires before backend resolution, even with jax absent."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.trace(lambda p, q: p + q, torch.randn(3), torch.randn(3), backend="jax")  # type: ignore[arg-type]
    assert info.value.fields["code"] == "input_kwargs_type_invalid"


def test_jax_two_positional_remedy_runs() -> None:
    """U11.2 JAX leg: the refusal, then the tuple spelling the remedy names, both run."""

    jnp = pytest.importorskip("jax.numpy")
    x = jnp.ones(3)
    with pytest.raises(ArgumentTypeError) as info:
        tl.trace(lambda p, q: p + q, x, x, backend="jax")  # type: ignore[arg-type]
    assert info.value.fields["code"] == "input_kwargs_type_invalid"
    trace = tl.trace(lambda p, q: p + q, (x, x), backend="jax")
    assert len(trace.layer_list) > 0


def test_exported_program_refusal_names_the_source_module_and_the_remedy_runs() -> None:
    """U35.2: an ExportedProgram refuses with the trace-the-source remedy, which works."""

    model, x = _Tiny().eval(), torch.randn(2, 4)
    program = torch.export.export(model, (x,))
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(program, x)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "model_type_unsupported"
    assert "trace the source nn.Module before export" in info.value.fields["remedy"]

    assert len(tl.trace(model, x).layer_list) > 0


def test_graphmodule_attribution_remedy_names_the_source_module() -> None:
    """U35.2: an fx GraphModule root gets the export branch of the remedy."""

    import weakref
    from types import SimpleNamespace

    from torchlens.backends.torch.backend import output_attribution_remedy

    model = _Tiny().eval()
    graph_module = torch.fx.symbolic_trace(model)
    as_graph = SimpleNamespace(_source_model_ref=weakref.ref(graph_module))
    as_eager = SimpleNamespace(_source_model_ref=weakref.ref(model))
    assert "trace the source nn.Module before export" in output_attribution_remedy(as_graph)
    assert "bind/import torch functions" in output_attribution_remedy(as_eager)
