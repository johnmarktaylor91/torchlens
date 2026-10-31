"""Regression tests for r18c bridge/compat adapter hardening.

Covers the depyf prepare_debug contract, tensor_layers explicit-site validation
(A3-14), the extractor/ILG .model unwrap gate (A3-15), the lovely
non-mutating fallback (A3-17), the profiler blank-label guard (A3-01 partial), null traceEvents handling (LOW-9), and the push_to_hub artifact-format tag (LOW-4 / LOW-S1).

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
        self._layer_list = layers


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
    with pytest.raises(RuntimeError, match=r"torch\.compiler\.reset\(\)"):
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


def test_profiler_label_range_matches_by_exact_equality() -> None:
    """A record_function range matches the layer whose label it EQUALS, never a substring."""

    from torchlens.bridge import profiler

    class Layer:
        def __init__(self, label: str, func: str, idx: int) -> None:
            self.layer_label = label
            self.func_name = func
            self.raw_index = idx

    trace = {
        "traceEvents": [
            {"name": "conv2d_1_1", "dur": 5.0},
            {"name": "conv2d_1_1 cpu", "dur": 7.0},
            {"name": "my_conv_kernel", "dur": 11.0},
        ]
    }
    joined = profiler.join(_FakeLog([Layer("conv2d_1_1", "conv2d", 1)]), trace)
    row = joined["ops"][0]
    assert row["kineto_event_count"] == 1
    assert row["kineto_duration_us"] == 5.0
    assert joined["unmatched_event_counts"] == {"conv2d_1_1 cpu": 1, "my_conv_kernel": 1}


def test_profiler_repeated_op_type_matches_kth_event_to_kth_layer() -> None:
    """Two conv layers share two aten::conv2d events one each, in execution order."""

    from torchlens.bridge import profiler

    class Layer:
        def __init__(self, label: str, func: str, idx: int) -> None:
            self.layer_label = label
            self.func_name = func
            self.raw_index = idx

    layers = [Layer("conv2d_1_1", "conv2d", 1), Layer("conv2d_2_3", "conv2d", 3)]
    events = [
        {"name": "aten::conv2d", "ph": "X", "ts": 0.0, "dur": 10.0, "tid": 1},
        {"name": "aten::convolution", "ph": "X", "ts": 1.0, "dur": 8.0, "tid": 1},
        {"name": "aten::conv2d", "ph": "X", "ts": 20.0, "dur": 30.0, "tid": 1},
    ]
    joined = profiler.join(_FakeLog(layers), {"traceEvents": events})
    assert [row["kineto_duration_us"] for row in joined["ops"]] == [10.0, 30.0]
    assert [row["kineto_event_count"] for row in joined["ops"]] == [1, 1]
    assert joined["unmatched_event_counts"] == {"aten::convolution": 1}

    # Two repeated forwards: 4 events over 2 layers, event k to layer k mod 2.
    twice = events + [
        {"name": "aten::conv2d", "ph": "X", "ts": 100.0, "dur": 1.0, "tid": 1},
        {"name": "aten::conv2d", "ph": "X", "ts": 120.0, "dur": 3.0, "tid": 1},
    ]
    joined = profiler.join(_FakeLog(layers), {"traceEvents": twice})
    assert [row["kineto_duration_us"] for row in joined["ops"]] == [11.0, 33.0]


def test_profiler_count_mismatch_is_disclosed_not_guessed() -> None:
    """Three events over two layers of one type stay unmatched, with a warning."""

    from torchlens.bridge import profiler

    class Layer:
        def __init__(self, label: str, func: str, idx: int) -> None:
            self.layer_label = label
            self.func_name = func
            self.raw_index = idx

    layers = [Layer("relu_1_2", "relu", 2), Layer("relu_2_4", "relu", 4)]
    events = [{"name": "aten::relu", "ph": "X", "ts": float(t), "dur": 1.0} for t in (0, 5, 9)]
    with pytest.warns(UserWarning, match="relu") as record:
        joined = profiler.join(_FakeLog(layers), {"traceEvents": events})
    codes = [getattr(w.message, "fields", {}).get("code") for w in record]
    assert "profiler_join_op_types_unmatched" in codes
    assert all(row["kineto_event_count"] == 0 for row in joined["ops"])
    assert joined["mismatched_op_types"] == {"relu": {"events": 3, "layers": 2}}
    assert joined["unmatched_event_counts"] == {"aten::relu": 3}


class _ProfLayer:
    def __init__(self, label: str, func: str, idx: int) -> None:
        self.layer_label = label
        self.func_name = func
        self.raw_index = idx


def _x(name: str, ts: float, dur: float) -> dict[str, Any]:
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "tid": 1}


def test_profiler_aten_event_nested_in_other_aten_op_is_internal() -> None:
    """An aten::matmul inside aten::linear is the linear's internal call, not a matmul layer's."""

    from torchlens.bridge import profiler

    layers = [_ProfLayer("linear_1_1", "linear", 1), _ProfLayer("matmul_1_2", "matmul", 2)]
    forward = [
        _x("aten::linear", 0.0, 10.0),
        _x("aten::matmul", 1.0, 8.0),
        _x("aten::matmul", 20.0, 5.0),
    ]
    # Two forwards: the old same-name filter saw 4 matmuls over 1 layer as "4 forwards".
    events = forward + [{**e, "ts": e["ts"] + 100.0} for e in forward]
    joined = profiler.join(_FakeLog(layers), {"traceEvents": events})
    assert [row["kineto_duration_us"] for row in joined["ops"]] == [20.0, 10.0]
    assert [e["ts"] for e in joined["ops"][1]["kineto_events"]] == [20.0, 120.0]
    assert joined["unmatched_event_counts"] == {"aten::matmul": 2}
    assert joined["forwards"] == 2
    assert joined["mismatched_op_types"] == {}


def test_profiler_inconsistent_forward_count_is_disclosed() -> None:
    """An op type whose count implies a different forward count stays unmatched."""

    from torchlens.bridge import profiler

    layers = [
        _ProfLayer("conv2d_1_1", "conv2d", 1),
        _ProfLayer("relu_1_2", "relu", 2),
        _ProfLayer("matmul_1_3", "matmul", 3),
    ]
    events = [
        _x("aten::conv2d", 0.0, 1.0),
        _x("aten::relu", 2.0, 1.0),
        _x("aten::matmul", 4.0, 1.0),
        _x("aten::conv2d", 10.0, 1.0),
        _x("aten::relu", 12.0, 1.0),
        _x("aten::matmul", 14.0, 1.0),
        _x("aten::matmul", 16.0, 1.0),
        _x("aten::matmul", 18.0, 1.0),
    ]
    with pytest.warns(UserWarning, match="matmul"):
        joined = profiler.join(_FakeLog(layers), {"traceEvents": events})
    assert [row["kineto_event_count"] for row in joined["ops"]] == [2, 2, 0]
    assert joined["forwards"] == 2
    assert joined["mismatched_op_types"] == {
        "matmul": {"events": 4, "layers": 1, "forwards": 4, "expected_forwards": 2}
    }


def test_profiler_dunder_and_plain_spellings_pool_in_execution_order() -> None:
    """``add`` and ``__add__`` layers share aten::add events in execution order."""

    from torchlens.bridge import profiler

    layers = [
        _ProfLayer("add_1_1", "add", 1),
        _ProfLayer("__add___1_2", "__add__", 2),
        _ProfLayer("add_2_3", "add", 3),
    ]
    events = [_x("aten::add", float(t), float(d)) for t, d in ((0, 1), (5, 2), (9, 3))]
    joined = profiler.join(_FakeLog(layers), {"traceEvents": events})
    assert [row["kineto_duration_us"] for row in joined["ops"]] == [1.0, 2.0, 3.0]
    assert joined["mismatched_op_types"] == {}


def test_module_site_unresolved_and_ambiguous_refuse_typed() -> None:
    """A site no module returns, or one two sibling modules return, refuses with its code."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.bridge._utils import module_for_site

    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Identity())
    model.add_module("b", torch.nn.Identity())

    class Site:
        out = torch.zeros(1)
        layer_label = "add_1_1"
        output_of_module_calls: tuple[str, ...] = ()

    class SiteLog:
        layer_list: list[Any] = []

        def _source_model_ref(self) -> torch.nn.Module:
            return model

    with pytest.raises(InvalidArgumentError) as info:
        module_for_site(SiteLog(), Site(), bridge="gradcam")
    assert info.value.fields["code"] == "bridge_module_site_unresolved"
    assert info.value.fields["remedy"]

    Site.output_of_module_calls = ("a:1", "b:1")
    with pytest.raises(InvalidArgumentError) as info:
        module_for_site(SiteLog(), Site(), bridge="gradcam")
    assert info.value.fields["code"] == "bridge_module_site_ambiguous"
    assert info.value.fields["remedy"]


def test_module_call_count_prefers_the_module_record_and_never_swallows_errors() -> None:
    """The record's ``num_calls`` wins; only a missing record falls back to counting calls."""

    from torchlens.bridge._utils import _module_num_calls

    layers = [types.SimpleNamespace(output_of_module_calls=("a:1", "a:2", "b:1"))]
    # Duck-typed log with no module records: distinct calls of the address are counted.
    assert _module_num_calls(_FakeLog(layers), "a") == 2
    # A module mapping without the address falls back the same way.
    with_records = types.SimpleNamespace(
        layer_list=layers, modules={"b": types.SimpleNamespace(num_calls=5)}
    )
    assert _module_num_calls(with_records, "a") == 2
    assert _module_num_calls(with_records, "b") == 5

    class BrokenModules:
        def __contains__(self, key: object) -> bool:
            raise RuntimeError("module table corrupted")

    broken = types.SimpleNamespace(layer_list=layers, modules=BrokenModules())
    with pytest.raises(RuntimeError, match="module table corrupted"):
        _module_num_calls(broken, "a")


# --------------------------------------------------------------------------- #
# LOW-9 profiler: null traceEvents/events tolerated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trace", [{"traceEvents": None}, {"events": None}, {}])
def test_profiler_null_events(trace: dict[str, Any]) -> None:
    """A payload with null/missing events yields no ops instead of a TypeError."""

    from torchlens.bridge import profiler

    assert profiler.join(_FakeLog([]), trace)["ops"] == []


# --------------------------------------------------------------------------- #
# LOW-4 / LOW-S1 huggingface: artifact format tag + extension-aware name
# --------------------------------------------------------------------------- #
def test_push_to_hub_refuses_objects_without_a_tlspec_path() -> None:
    """A plain object is refused typed; push_to_hub never falls back to pickle."""

    from torchlens._errors import ArgumentTypeError
    from torchlens.bridge import huggingface_hub

    with pytest.raises(ArgumentTypeError) as info:
        huggingface_hub.push_to_hub({"trace": "data"}, "example/repo", dry_run=True)
    assert info.value.fields["code"] == "hub_push_artifact_unsupported"


def test_push_to_hub_tarball_format_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle upload reports tar.gz under the default ``.tar.gz`` name."""

    from torchlens.bridge import huggingface_hub

    class Unpicklable:
        def __reduce__(self) -> Any:
            raise TypeError("cannot pickle")

    def fake_saver(_obj: Any) -> Any:
        def _save(path: Any, *, level: str, overwrite: bool) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "manifest.json").write_text("{}", encoding="utf-8")

        return _save

    monkeypatch.setattr(huggingface_hub, "_resolve_bundle_saver", fake_saver)
    result = huggingface_hub.push_to_hub(Unpicklable(), "example/repo", dry_run=True)
    assert result["format"] == "tar.gz"
    assert result["path_in_repo"] == "torchlens_artifact.tar.gz"
