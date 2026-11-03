"""``trace.do`` on a default capture replays through the recorded argument templates.

Every eager ``tl.trace`` capture records per-op argument templates, which is all
saved-graph replay reads, so ``trace.do(site, edit)`` and
``trace.fork().do(site, tl.patch_from(other))`` work without ``replayable=True``
and produce the same tensors as the replay on a ``replayable=True`` capture.
``replayable=True`` keeps its name and now means "runnable save ready" only.
A trace that keeps no templates (loaded from a file, structure-only) still refuses.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl

pytestmark = pytest.mark.filterwarnings("ignore::torchlens.errors.InterventionWarning")

_REPLAYABLE = tl.options.CaptureOptions(replayable=True)


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class _Transformer(torch.nn.Module):
    def __init__(self, vocab: int, width: int, n_layer: int) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(vocab, width)
        self.h = torch.nn.ModuleList([_Block(width) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(width)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(ids)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class _Gpt2Shaped(torch.nn.Module):
    """``transformer.h.0`` to ``transformer.h.5`` then ``lm_head``, like GPT-2 small."""

    def __init__(self, vocab: int = 16, width: int = 8) -> None:
        super().__init__()
        self.transformer = _Transformer(vocab, width, n_layer=6)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(ids))


def _output(trace: tl.Trace) -> torch.Tensor:
    out = trace[trace.output_layers[0]].out
    assert isinstance(out, torch.Tensor)
    return out


def _assert_do_matches_replayable_replay(
    model: torch.nn.Module, x: torch.Tensor, site, edit
) -> None:
    """``do(site, edit)`` on a default capture equals the replay on a replayable capture."""

    default = tl.trace(model, x)
    ready = tl.trace(model, x, capture=_REPLAYABLE)
    assert default.is_replayable is False
    assert ready.is_replayable is True
    assert default.options.save_arg_templates is True

    edited_default = default.fork().do(site, edit)
    edited_ready = ready.fork().do(site, edit)
    assert edited_default.last_run["engine"] == "replay"
    assert edited_ready.last_run["engine"] == "replay"
    assert torch.equal(_output(edited_default), _output(edited_ready))
    # The edit took effect (the two forks differ from their roots alike).
    assert not torch.equal(_output(edited_default), _output(default))
    assert torch.equal(_output(default), _output(ready))


def _assert_patch_from_matches_replayable_replay(
    model: torch.nn.Module, x_a: torch.Tensor, x_b: torch.Tensor, site
) -> None:
    """``fork().do(site, tl.patch_from(other))`` on default captures equals the replayable pair."""

    a_default, b_default = tl.trace(model, x_a), tl.trace(model, x_b)
    a_ready = tl.trace(model, x_a, capture=_REPLAYABLE)
    b_ready = tl.trace(model, x_b, capture=_REPLAYABLE)

    patched_default = b_default.fork().do(site, tl.patch_from(a_default))
    patched_ready = b_ready.fork().do(site, tl.patch_from(a_ready))
    assert patched_default is not b_default
    assert patched_default.last_run["engine"] == "replay"
    assert torch.equal(_output(patched_default), _output(patched_ready))
    assert not torch.equal(_output(patched_default), _output(b_default))


def test_mlp_do_on_default_capture_equals_replayable_replay() -> None:
    torch.manual_seed(0)
    model = _Mlp().eval()
    x = torch.randn(4, 6)
    _assert_do_matches_replayable_replay(model, x, tl.module("fc2"), tl.zero_ablate())
    _assert_patch_from_matches_replayable_replay(model, x, torch.randn(4, 6), tl.module("fc2"))


def test_gpt2_shaped_do_on_default_capture_equals_replayable_replay() -> None:
    torch.manual_seed(0)
    model = _Gpt2Shaped().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    site = tl.func("gelu") & tl.in_module("transformer.h.5")
    _assert_do_matches_replayable_replay(model, ids, site, tl.zero_ablate())
    _assert_patch_from_matches_replayable_replay(
        model, ids, torch.tensor([[1, 2, 3, 6, 5]]), tl.module("transformer.h.5")
    )


@pytest.mark.heavy
def test_resnet18_do_on_default_capture_equals_replayable_replay() -> None:
    torchvision_models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = torchvision_models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    _assert_do_matches_replayable_replay(model, x, tl.module("layer3"), tl.zero_ablate())
    _assert_patch_from_matches_replayable_replay(
        model, x, torch.randn(1, 3, 64, 64), tl.module("layer3")
    )


@pytest.mark.heavy
def test_real_gpt2_tiny_patch_from_on_default_captures_equals_replayable_replay() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=6, n_head=2, use_cache=False
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    france = torch.tensor([[3, 7, 1, 9, 4, 3]])
    italy = torch.tensor([[3, 7, 1, 9, 8, 3]])
    _assert_patch_from_matches_replayable_replay(model, france, italy, tl.module("transformer.h.5"))
    _assert_do_matches_replayable_replay(
        model, italy, tl.func("gelu") & tl.in_module("transformer.h.5"), tl.zero_ablate()
    )


def test_readme_first_divergence_on_default_capture_pair() -> None:
    """The README sequence: patch block 5, then ask where Italy and the patched run first differ."""

    torch.manual_seed(0)
    model = _Gpt2Shaped().eval()
    france = tl.trace(model, torch.tensor([[1, 2, 3, 4, 5]]))
    italy = tl.trace(model, torch.tensor([[1, 2, 3, 6, 5]]))
    patched = italy.fork().do(tl.module("transformer.h.5"), tl.patch_from(france))

    row = tl.debug.first_divergence(italy, patched)
    assert row is not None
    assert row["allclose"] is False
    # The first differing op is inside block 5, the patch site, never earlier.
    block_labels = {
        label.split(":")[0] for label in italy.find_sites(tl.in_module("transformer.h.5")).labels()
    }
    assert str(row["op"]).split(":")[0] in block_labels
    assert tl.debug.first_divergence(italy, italy.fork()) is None


def test_saved_and_loaded_default_trace_refuses_replay_naming_the_loaded_file(tmp_path) -> None:
    """A save keeps no callables or templates, so a loaded trace refuses and says so."""

    torch.manual_seed(0)
    model = _Mlp().eval()
    trace = tl.trace(model, torch.randn(2, 6))
    bundle = tmp_path / "default.tlspec"
    trace.save(str(bundle))
    loaded = tl.load(str(bundle))

    with pytest.raises(tl.errors.ReplayPreconditionError) as info:
        loaded.fork().do(tl.module("fc2"), tl.zero_ablate())
    message = str(info.value)
    assert "loaded from a saved file" in message
    assert "tl.trace(model, inputs)" in message
    assert "replayable=True" not in message


def test_saved_default_trace_persists_no_argument_templates(tmp_path) -> None:
    """Templates are session-only: a default save carries none, so its size is unchanged."""

    torch.manual_seed(0)
    model = _Mlp().eval()
    trace = tl.trace(model, torch.randn(2, 6))
    assert any(op.args_template is not None for op in trace.ops)
    bundle = tmp_path / "default.tlspec"
    trace.save(str(bundle))
    loaded = tl.load(str(bundle))
    assert all(op.args_template is None for op in loaded.ops)
    assert all(op.kwargs_template is None for op in loaded.ops)


def test_structure_only_capture_records_no_templates_and_refuses_replay() -> None:
    """A structure-only capture has no values to replay; it stays as before."""

    model = _Mlp().eval()
    trace = tl.trace(
        model, torch.randn(2, 6), capture=tl.options.CaptureOptions(structure_only=True)
    )
    assert trace.is_replayable is False
    assert trace.options.save_arg_templates is False
    assert all(op.args_template is None for op in trace.ops)
    with pytest.raises(tl.errors.ReplayPreconditionError) as info:
        trace.fork().do(tl.module("fc2"), tl.zero_ablate())
    message = str(info.value)
    assert "structure_only" in message
    assert "tl.trace(model, inputs)" in message
    assert "replayable=True" not in message


def test_runnable_save_on_default_capture_still_names_replayable(tmp_path) -> None:
    """``level="runnable"`` is the one door that still needs ``replayable=True``, and says so."""

    torch.manual_seed(0)
    trace = tl.trace(_Mlp().eval(), torch.randn(2, 6))
    with pytest.raises(tl.errors.RunnablePreflightError) as info:
        trace.save(str(tmp_path / "runnable.tlspec"), level="runnable")
    message = str(info.value)
    assert "callable_ref_missing" in message
    assert "replayable=True" in message
