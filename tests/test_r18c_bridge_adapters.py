"""Regression tests for r18c bridge/compat adapter hardening.

Covers the depyf prepare_debug contract, tensor_layers explicit-site validation
(A3-14), the extractor/ILG .model unwrap gate (A3-15), dialz analyzer-class
instantiation (A3-16), the lovely non-mutating fallback (A3-17), the repeng /
steering fail-loud default guard (A3-21), the profiler blank-label guard
(A3-01 partial), null traceEvents handling (LOW-9), the execution_trace schema
docstring (A3-18), and the push_to_hub artifact-format tag (LOW-4 / LOW-S1).

The adapters are exercised with stubbed optional dependencies and fake logs so
the suite needs none of the optional extras installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch


def _module(name: str, **attrs: Any) -> types.ModuleType:
    """Return a throwaway module object carrying ``attrs``."""

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeLayer:
    """Minimal layer-pass record with a tensor ``out``."""

    def __init__(self, label: str, out: Any = None, is_input: bool = False) -> None:
        self.layer_label = label
        self.out = torch.zeros(3) if out is None else out
        self.is_input = is_input
        # Neuro eligibility gate (F22, memo item 9): saved-payload evidence and
        # shape ride the record, never an ``out`` read.
        self.has_saved_activation = isinstance(self.out, torch.Tensor)
        self.shape = tuple(self.out.shape) if isinstance(self.out, torch.Tensor) else None


class _FakeLog:
    """Minimal ``Trace``-shaped object exposing ``layer_list``."""

    def __init__(self, layers: list[Any]) -> None:
        self.layer_list = layers


# --------------------------------------------------------------------------- #
# depyf: dump compiles under prepare_debug and returns the files it wrote
# --------------------------------------------------------------------------- #
def _fake_prepare_debug(files: list[str], calls: list[Any]) -> Any:
    """Return a fake ``depyf.prepare_debug`` that writes ``files`` on exit."""

    import contextlib
    from pathlib import Path

    @contextlib.contextmanager
    def prepare_debug(dump_src_dir: str, **kwargs: Any) -> Any:
        calls.append((dump_src_dir, kwargs))
        yield
        for name in files:
            (Path(dump_src_dir) / name).write_text("# dumped\n")

    return prepare_debug


def test_depyf_dump_compiles_under_prepare_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """dump runs torch.compile(model)(x) inside prepare_debug and lists new files."""

    from torchlens.bridge import depyf

    calls: list[Any] = []
    compiled_inputs: list[Any] = []
    (tmp_path / "old.py").write_text("# pre-existing\n")
    prepare = _fake_prepare_debug(["__compiled_fn_1.Forward_graph.0.py", "full_code.py"], calls)
    monkeypatch.setitem(sys.modules, "depyf", _module("depyf", prepare_debug=prepare))
    monkeypatch.setattr(torch, "compile", lambda model: lambda *a: compiled_inputs.append(a))

    written = depyf.dump("model", (1, 2), tmp_path, log_bytecode=True)

    assert [p.name for p in written] == ["__compiled_fn_1.Forward_graph.0.py", "full_code.py"]
    assert calls == [(str(tmp_path), {"log_bytecode": True})]
    assert compiled_inputs == [(1, 2)]


def test_depyf_dump_requires_path() -> None:
    """The output directory is required; the bridge never picks one."""

    from torchlens.bridge import depyf

    with pytest.raises(TypeError):
        depyf.dump("model", "x")  # type: ignore[call-arg]


def test_depyf_dump_refuses_when_nothing_was_dumped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A cached compile that dumps nothing is refused, never an empty list."""

    from torchlens.bridge import depyf

    prepare = _fake_prepare_debug([], [])
    monkeypatch.setitem(sys.modules, "depyf", _module("depyf", prepare_debug=prepare))
    monkeypatch.setattr(torch, "compile", lambda model: lambda *a: None)
    with pytest.raises(RuntimeError, match="torch._dynamo.reset"):
        depyf.dump("model", "x", tmp_path)


def test_depyf_inbody_error_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A failure inside the compiled run surfaces unchanged."""

    from torchlens.bridge import depyf

    def _boom(*args: Any) -> None:
        raise TypeError("internal compile failure")

    prepare = _fake_prepare_debug(["x.py"], [])
    monkeypatch.setitem(sys.modules, "depyf", _module("depyf", prepare_debug=prepare))
    monkeypatch.setattr(torch, "compile", lambda model: _boom)
    with pytest.raises(TypeError, match="internal compile failure"):
        depyf.dump("model", "x", tmp_path)


# --------------------------------------------------------------------------- #
# A3-14 tensor_layers: explicit sites enforce the tensor-out invariant
# --------------------------------------------------------------------------- #
def test_tensor_layers_explicit_nontensor_raises() -> None:
    """An explicit site without a tensor out is rejected, not passed downstream."""

    from torchlens.bridge._utils import tensor_layers

    class NonTensorSite:
        layer_label = "metadata_only"
        out = "not-a-tensor"

    with pytest.raises(ValueError, match="does not have a saved tensor out"):
        tensor_layers(object(), sites=[NonTensorSite()])


def test_tensor_layers_explicit_tensor_accepted() -> None:
    """An explicit site carrying a tensor out is returned unchanged."""

    from torchlens.bridge._utils import tensor_layers

    site = _FakeLayer("ok")
    result = tensor_layers(object(), sites=[site])
    assert result == [site]


# --------------------------------------------------------------------------- #
# A3-15 from_torchextractor / from_ilg: gate the .model unwrap
# --------------------------------------------------------------------------- #
def _child() -> torch.nn.Module:
    layer = torch.nn.Linear(2, 2)
    return layer


def test_from_torchextractor_keeps_plain_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain model with a .model child + explicit layers is not unwrapped."""

    import torchlens.compat as compat

    monkeypatch.setitem(sys.modules, "torchextractor", _module("torchextractor"))

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _child()

        def forward(self, x: Any) -> Any:
            return self.model(x)

    wrapper = Wrapper()
    extractor = compat.from_torchextractor(wrapper, layers=["head"])
    assert extractor.model is wrapper
    assert extractor.model is not wrapper.model


def test_from_torchextractor_unwraps_genuine_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine extractor-like wrapper (has .model AND layers) is unwrapped."""

    import torchlens.compat as compat

    monkeypatch.setitem(sys.modules, "torchextractor", _module("torchextractor"))

    class ExtractorLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _child()
            self.layers = ["fc"]

        def forward(self, x: Any) -> Any:
            return self.model(x)

    wrapper = ExtractorLike()
    extractor = compat.from_torchextractor(wrapper)
    assert extractor.model is wrapper.model
    assert extractor.layers == ["fc"]


def test_from_ilg_keeps_plain_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_ilg does not unwrap a plain wrapper lacking return_layers."""

    import torchlens.compat as compat

    monkeypatch.setitem(sys.modules, "torchvision", _module("torchvision"))

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _child()

        def forward(self, x: Any) -> Any:
            return self.model(x)

    wrapper = Wrapper()
    extractor = compat.from_ilg(wrapper, return_layers={"layer1": "feat1"})
    assert extractor.model is wrapper


# --------------------------------------------------------------------------- #
# A3-16 dialz: analyzer classes are instantiated before use
# --------------------------------------------------------------------------- #
def test_dialz_instantiates_analyzer_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dialz Analyzer class is instantiated; self is not the outs list."""

    from torchlens.bridge import dialz

    class Analyzer:
        def analyze(self, outs: list[Any], labels: list[str] | None = None) -> dict[str, Any]:
            return {
                "self_is_analyzer": isinstance(self, Analyzer),
                "outs_is_list": isinstance(outs, list),
                "n": len(outs),
                "labels": labels,
            }

    monkeypatch.setitem(sys.modules, "dialz", _module("dialz", Analyzer=Analyzer))
    log = _FakeLog([_FakeLayer("a"), _FakeLayer("b")])
    payload = dialz.analyze(log)
    result = payload["result"]
    assert result["self_is_analyzer"] is True
    assert result["outs_is_list"] is True
    assert result["n"] == 2
    assert result["labels"] == ["a", "b"]


def test_dialz_module_level_function_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module-level analyze function is still called directly."""

    from torchlens.bridge import dialz

    def analyze(outs: list[Any], *, labels: list[str]) -> dict[str, Any]:
        return {"count": len(outs)}

    monkeypatch.setitem(sys.modules, "dialz", _module("dialz", analyze=analyze))
    payload = dialz.analyze(_FakeLog([_FakeLayer("a")]))
    assert payload["result"]["count"] == 1


# --------------------------------------------------------------------------- #
# A3-17 compat.lovely: non-mutating fallback
# --------------------------------------------------------------------------- #
def test_lovely_fallback_does_not_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lovely_tensors without lovely() raises instead of patching global repr."""

    import importlib

    patched = {"called": False}

    def monkey_patch() -> None:
        patched["called"] = True

    monkeypatch.setitem(
        sys.modules, "lovely_tensors", _module("lovely_tensors", monkey_patch=monkey_patch)
    )
    lovely_mod = importlib.import_module("torchlens.compat.lovely")
    with pytest.raises(RuntimeError, match="does not expose a lovely"):
        lovely_mod.lovely(torch.zeros(2), color=False)
    assert patched["called"] is False


def test_lovely_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formatter path forwards caller args/kwargs (no silent drop)."""

    import importlib

    seen: dict[str, Any] = {}

    def lovely(tensor: Any, *args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "formatted"

    monkeypatch.setitem(sys.modules, "lovely_tensors", _module("lovely_tensors", lovely=lovely))
    lovely_mod = importlib.import_module("torchlens.compat.lovely")
    assert lovely_mod.lovely(torch.zeros(2), color=False) == "formatted"
    assert seen == {"color": False}


# --------------------------------------------------------------------------- #
# A3-21 repeng / steering: fail loud when the default cannot take saved outs
# --------------------------------------------------------------------------- #
def test_repeng_default_factory_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-shaped ControlVector.train default is refused with a clear error."""

    from torchlens.bridge import repeng

    class ControlVector:
        @classmethod
        def train(cls, model: Any, tokenizer: Any, dataset: Any, **kw: Any) -> str:
            return "trained"

    monkeypatch.setitem(sys.modules, "repeng", _module("repeng", ControlVector=ControlVector))
    log = _FakeLog([])
    with pytest.raises(RuntimeError, match="explicit vector_factory"):
        repeng.control_vector(log, _FakeLayer("p"), _FakeLayer("n"))


def test_repeng_explicit_factory_bypasses_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit factory is called directly, never gated by the default guard."""

    from torchlens.bridge import repeng

    class ControlVector:
        @classmethod
        def train(cls, model: Any, tokenizer: Any, dataset: Any, **kw: Any) -> str:
            return "trained"

    monkeypatch.setitem(sys.modules, "repeng", _module("repeng", ControlVector=ControlVector))
    payload = repeng.control_vector(
        _FakeLog([]),
        _FakeLayer("p"),
        _FakeLayer("n"),
        vector_factory=lambda positive, negative, **kw: "explicit",
    )
    assert payload["control_vector"] == "explicit"


def test_steering_default_trainer_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-shaped train_steering_vector default is refused with a clear error."""

    from torchlens.bridge import steering_vectors

    def train_steering_vector(model: Any, tokenizer: Any, training_samples: Any, **kw: Any) -> str:
        return "trained"

    monkeypatch.setitem(
        sys.modules,
        "steering_vectors",
        _module("steering_vectors", train_steering_vector=train_steering_vector),
    )
    with pytest.raises(RuntimeError, match="explicit trainer"):
        steering_vectors.vector(_FakeLog([]), _FakeLayer("p"), _FakeLayer("n"))


def test_steering_compatible_default_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A default trainer that accepts (positive, negative) is used as-is."""

    from torchlens.bridge import steering_vectors

    def train_steering_vector(positive: Any, negative: Any, *, normalize: bool = False) -> dict:
        return {"normalize": normalize}

    monkeypatch.setitem(
        sys.modules,
        "steering_vectors",
        _module("steering_vectors", train_steering_vector=train_steering_vector),
    )
    payload = steering_vectors.vector(
        _FakeLog([]), _FakeLayer("p"), _FakeLayer("n"), normalize=True
    )
    assert payload["vector"] == {"normalize": True}


# --------------------------------------------------------------------------- #
# A3-01 (partial) profiler: blank label/func must not match every event
# --------------------------------------------------------------------------- #
def test_profiler_blank_label_matches_nothing() -> None:
    """A layer missing layer_label/func_name absorbs no events."""

    from torchlens.bridge import profiler

    class Blank:
        raw_index = 1

    trace = {
        "traceEvents": [
            {"name": "conv2d_kernel_1", "dur": 5.0},
            {"name": "totally_unrelated_gc", "dur": 13.0},
        ]
    }
    row = profiler.join(_FakeLog([Blank()]), trace)["ops"][0]
    assert row["kineto_event_count"] == 0
    assert row["kineto_duration_us"] == 0.0


def test_profiler_named_label_still_matches() -> None:
    """A non-blank label still matches its events (join contract unchanged)."""

    from torchlens.bridge import profiler

    class Layer:
        def __init__(self, label: str, func: str, idx: int) -> None:
            self.layer_label = label
            self.func_name = func
            self.raw_index = idx

    trace = {"traceEvents": [{"name": "my_conv_kernel", "dur": 5.0}]}
    row = profiler.join(_FakeLog([Layer("conv", "conv2d", 1)]), trace)["ops"][0]
    assert row["kineto_event_count"] == 1
    assert row["kineto_duration_us"] == 5.0


# --------------------------------------------------------------------------- #
# LOW-9 profiler: null traceEvents/events tolerated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trace", [{"traceEvents": None}, {"events": None}, {}])
def test_profiler_null_events(trace: dict[str, Any]) -> None:
    """A payload with null/missing events yields no ops instead of a TypeError."""

    from torchlens.bridge import profiler

    assert profiler.join(_FakeLog([]), trace)["ops"] == []


# --------------------------------------------------------------------------- #
# A3-18 profiler docstring honesty
# --------------------------------------------------------------------------- #
def test_execution_trace_docstring_is_honest() -> None:
    """The docstring no longer claims ExecutionTraceObserver compatibility."""

    from torchlens.bridge import profiler

    doc = profiler.execution_trace.__doc__ or ""
    assert "ExecutionTraceObserver-compatible" not in doc
    assert "torchlens.execution_trace.v1" in doc


# --------------------------------------------------------------------------- #
# LOW-4 / LOW-S1 huggingface: artifact format tag + extension-aware name
# --------------------------------------------------------------------------- #
def test_push_to_hub_pickle_format_tag() -> None:
    """A picklable artifact reports the pickle format and keeps the .pkl name."""

    from torchlens.bridge import huggingface

    result = huggingface.push_to_hub({"trace": "data"}, "example/repo", dry_run=True)
    assert result["format"] == "pickle"
    assert result["path_in_repo"] == "torchlens_artifact.pkl"


def test_push_to_hub_tarball_format_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle fallback reports tar.gz and switches the default name extension."""

    from torchlens.bridge import huggingface

    class Unpicklable:
        def __reduce__(self) -> Any:
            raise TypeError("cannot pickle")

    def fake_saver(_obj: Any) -> Any:
        def _save(path: Any, *, level: str, overwrite: bool) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "manifest.json").write_text("{}", encoding="utf-8")

        return _save

    monkeypatch.setattr(huggingface, "_resolve_bundle_saver", fake_saver)
    result = huggingface.push_to_hub(Unpicklable(), "example/repo", dry_run=True)
    assert result["format"] == "tar.gz"
    assert result["path_in_repo"] == "torchlens_artifact.tar.gz"
