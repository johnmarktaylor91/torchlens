"""JMT's 3 Nov 2026 names-section rulings (walk 2026-11-03-decisions, names.2 to names.12).

One regression test per ruling this lane applied, each failing at the merge-train
base: names.2 (tinygrad ``capture_trace(save=)``), names.5 (the public logit lens on
``trace.using("mechinterp")``), names.6 (``rank_changes(by="module")``), names.8
(``Recorder(recording=RecordingOptions(...))``), names.10 (``save_transformed_output=True``
without ``output_transform`` refuses before capture) and names.12
(``Trace.summary(flop_convention=)`` kept). names.13, names.14 and names.9 live in
``tests/test_rulings_names_1103_intervention.py``.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentTypeError, InvalidArgumentError
from torchlens._semantic.logit_lens import LogitLensError
from torchlens.options import CaptureOptions
from torchlens.recording import Recorder
from torchlens.recording.options import RecordingOptions


class _Blocks(nn.Module):
    """Two named blocks so module addresses differ from op labels."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def _blocks_trace() -> tuple[nn.Module, tl.Trace]:
    torch.manual_seed(0)
    model = _Blocks()
    return model, tl.trace(model, torch.randn(2, 4))


# --- names.2: tinygrad capture_trace(save=) ---------------------------------------------


def test_names2_tinygrad_capture_trace_takes_save_not_layers_to_save() -> None:
    """names.2 "(a) Rename to save=": the tinygrad backend's keyword matches tl.trace(save=)."""

    tinygrad = pytest.importorskip("tinygrad")
    from torchlens.backends.tinygrad import TinygradBackend

    def block(x: Any) -> Any:
        return (x + 1.0).sum()

    x = tinygrad.Tensor([1.0, -2.0, 3.0])
    assert "save" in inspect.signature(TinygradBackend.capture_trace).parameters
    assert "layers_to_save" not in inspect.signature(TinygradBackend.capture_trace).parameters
    trace = TinygradBackend().capture_trace(block, (x,), save="all")
    assert len(trace.ops) > 0
    with pytest.raises(Exception, match="layers_to_save"):
        TinygradBackend().capture_trace(block, (x,), layers_to_save="all")


# --- names.5: the public logit lens ------------------------------------------------------


def test_names5_mechinterp_appliance_registered_under_its_ruled_name() -> None:
    """names.5 "(d)": the kit appliance is ``"mechinterp"``; the interim ``"mi"`` is gone."""

    names = tl.appliances.names()
    assert "mechinterp" in names
    assert "mi" not in names
    assert "logit_lens" in names["mechinterp"]
    _, trace = _blocks_trace()
    with pytest.raises(InvalidArgumentError) as info:
        trace.using("mi")
    assert info.value.fields["code"] == "appliance_unknown"
    appliance = trace.using("mechinterp")
    assert callable(appliance.logit_lens)
    assert "logit_lens" in repr(appliance)


def test_names5_logit_lens_forwards_to_the_private_lens(monkeypatch: pytest.MonkeyPatch) -> None:
    """``trace.using("mechinterp").logit_lens(modules=...)`` is the private lens on this trace."""

    import torchlens.mechinterp._appliance as kit_module

    _, trace = _blocks_trace()
    seen: dict[str, Any] = {}

    def fake(trace_arg: Any, **kwargs: Any) -> str:
        seen["trace"] = trace_arg
        seen.update(kwargs)
        return "lens"

    monkeypatch.setattr(kit_module, "_logit_lens", fake)
    assert trace.using("mechinterp").logit_lens(modules=["dec"], verify=False) == "lens"
    assert seen["trace"] is trace
    assert seen["layers"] == ["dec"]
    assert seen["verify"] is False
    assert seen["facet"] == "resid_post"


def test_names5_logit_lens_keeps_the_fail_closed_refusal() -> None:
    """A trace with no head facets still refuses (the validation refusal is unchanged)."""

    _, trace = _blocks_trace()
    with pytest.raises(LogitLensError):
        trace.using("mechinterp").logit_lens()


# --- names.6: rank_changes(by="module") --------------------------------------------------


def _ablated_pair() -> tuple[tl.Trace, tl.Bundle]:
    # do() on this base still needs replayable=True (names.1 lands with replay-templates).
    trace = tl.trace(_Blocks(), torch.randn(2, 4), replayable=True)
    fork = trace.fork()
    fork.do(tl.label(str(trace.modules["dec.0"].calls[0].output_ops[0])), tl.zero_ablate())
    return trace, tl.Bundle({"base": trace, "edited": fork}, baseline="base")


def test_names6_rank_changes_by_module_names_modules() -> None:
    """names.6 "(a) Yes, by="module"": rows are module addresses, one per module."""

    trace, pair = _ablated_pair()
    site_rows = pair.rank_changes(k=64)
    module_rows = pair.rank_changes(k=64, by="module")
    module_keys = [key for key, _ in module_rows]
    assert module_keys == sorted(set(module_keys), key=module_keys.index)
    assert set(module_keys) <= {str(m.address) for m in trace.modules} | {k for k, _ in site_rows}
    assert "dec.0" in module_keys
    scores = [score for _, score in module_rows]
    assert scores == sorted(scores, reverse=True)
    # The module's score is its most changed site's score.
    dec_sites = [score for key, score in site_rows if key.startswith(("dec.0", "dec"))]
    assert dict(module_rows)["dec.0"] == pytest.approx(max(dec_sites))
    # The default is unchanged: one row per site.
    assert {key for key, _ in pair.rank_changes(k=64, by="site")} == {key for key, _ in site_rows}


def test_names6_rank_changes_by_refuses_other_values() -> None:
    _, pair = _ablated_pair()
    with pytest.raises(InvalidArgumentError) as info:
        pair.rank_changes(by="layer")
    assert info.value.fields["code"] == "rank_changes_by_invalid"
    assert "module" in info.value.fields["remedy"]


# --- names.8: Recorder(recording=RecordingOptions(...)) -----------------------------------


def _op_labels(recording: Any) -> list[str]:
    return [record.ctx.label for record in recording.records if record.ctx.kind == "op"]


def test_names8_recorder_takes_recording_options_like_tl_record() -> None:
    """names.8 "(a) Yes, recording=RecordingOptions(...)": the eight flat keywords are gone."""

    model = _Blocks()
    x = torch.randn(2, 4)
    with pytest.raises(TypeError, match="default_op"):
        Recorder(model, default_op=True)
    one_shot = tl.record(model, x, default_op=True, history_size=4)
    with Recorder(model, recording=RecordingOptions(default_op=True, history_size=4)) as recorder:
        recorder(x)
    assert _op_labels(recorder.recording) == _op_labels(one_shot)
    assert recorder.options.history_size == 4


def test_names8_recorder_refuses_recording_fields_it_cannot_read() -> None:
    """``keep_op``, ``intervene`` and ``postprocess`` on the bundle refuse, never a silent no-op."""

    model = _Blocks()
    for field, value in (("postprocess", "module_path_strings"), ("keep_op", tl.func("relu"))):
        with pytest.raises(ArgumentTypeError) as info:
            Recorder(model, recording=RecordingOptions(**{field: value}))
        assert info.value.fields["code"] == "record_capture_option_unsupported"
        assert field in str(info.value)
        assert info.value.fields["argument"] == "recording"


# --- names.10: save_transformed_output=True without output_transform refuses -------------


class _Counting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.lin(x)


def test_names10_save_transformed_output_true_without_transform_refuses_before_capture() -> None:
    """names.10 "(b) Refuse instead": the self-contradicting setting never reaches the forward."""

    with pytest.raises(InvalidArgumentError) as info:
        CaptureOptions(save_transformed_output=True)
    assert info.value.fields["code"] == "save_transformed_output_without_output_transform"
    assert info.value.fields["argument"] == "save_transformed_output"
    remedy = info.value.fields["remedy"]
    assert "output_transform" in remedy and "drop save_transformed_output" in remedy
    model = _Counting()
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(save_transformed_output=True))
    assert info.value.fields["code"] == "save_transformed_output_without_output_transform"
    assert model.calls == 0


def test_names10_other_save_transformed_output_settings_still_capture() -> None:
    """The default ``"small"`` and ``True`` with a transform are untouched."""

    model = _Counting()
    fired: list[int] = []
    model.lin.register_forward_hook(lambda *_: fired.append(1))
    x = torch.randn(2, 4)
    assert CaptureOptions().save_transformed_output == "small"
    with_transform = CaptureOptions(save_transformed_output=True, output_transform=lambda o: o * 2)
    trace = tl.trace(model, x, capture=with_transform)
    assert fired == [1]
    assert trace.options.save_transformed_output is True


# --- names.12: Trace.summary(flop_convention=) kept -------------------------------------


def test_names12_summary_keeps_flop_convention() -> None:
    """names.12 "(a) Keep flop_convention=": the keyword stays on ``Trace.summary``."""

    assert "flop_convention" in inspect.signature(tl.Trace.summary).parameters
    _, trace = _blocks_trace()
    assert trace.summary(flop_convention="fma1") is not None


# --- names.7: trace["<module>"].facets is the module's facet view ----------------------


def test_names7_plain_door_module_facets_are_the_modules_facets() -> None:
    """names.7 "(a) Yes": ``trace["dec.0"].facets`` is the block's view, not its output op's.

    The plain door returns the Module for a module address (readme.8), so ``.facets``
    reads the same lazy view as ``trace.modules["dec.0"].facets``.
    """

    _, trace = _blocks_trace()
    module = trace.modules["dec.0"]
    assert trace["dec.0"] is module
    assert trace["dec.0"].facets is module.facets
    assert list(trace["dec.0"].facets.keys()) == list(module.facets.keys())
