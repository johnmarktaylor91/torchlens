"""Recurrence grouping: a module anchors passes only when it is called again.

Two ops inside ONE call of a module (the attention and MLP residual adds of a
decoder layer, or ``x.float().float()``) are two source sites of that forward.
They used to be fused into one two-pass layer because every op inside any
module counted as anchored; that broke module ``input_ops`` reads and made the
replay engine refuse a tiny Hugging Face Gemma-2 capture. A module called more
than once (one ``nn.ReLU`` reused, one block applied twice) still anchors.
"""

from __future__ import annotations

import warnings
from collections import Counter

import pytest
import torch
from torch import nn

import torchlens as tl


def _layer_passes(trace: tl.Trace) -> dict[str, int]:
    return {str(op.layer_label): int(op.num_passes) for op in trace}


def _pass_profile(trace: tl.Trace) -> Counter:
    """Multiset of (function, pass count) over layers, independent of labels."""

    layers: dict[str, tuple[str, int]] = {}
    for op in trace:
        if op.func_name == "none":
            continue
        layers[str(op.layer_label)] = (str(op.func_name), int(op.num_passes))
    return Counter(layers.values())


class _ResidualPair(nn.Module):
    """Two residual adds in one forward, around two different linears."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.a(x)
        return h + self.b(h)


class _TiedLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(3):
            h = torch.tanh(self.a(h))
        return h


class _BareLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.a(x)
        for _ in range(3):
            h = h + torch.sin(h)
        return h


class _BareChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x).double().double().float()


class _Wrapped(nn.Module):
    """The same body run once inside a submodule."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _AppliedTwice(nn.Module):
    """One residual block called twice: its ops are genuine two-pass layers."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _ResidualPair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.block(x))


def test_residual_adds_in_one_module_call_are_distinct_layers() -> None:
    torch.manual_seed(0)
    traced = tl.trace(_Wrapped(_ResidualPair()), torch.randn(2, 4))
    adds = {label: n for label, n in _layer_passes(traced).items() if label.startswith("add")}
    assert sorted(adds.values()) == [1, 1], adds


@pytest.mark.parametrize("body", [_ResidualPair, _TiedLoop, _BareLoop, _BareChain])
def test_a_single_call_submodule_groups_like_the_root_forward(body: type[nn.Module]) -> None:
    """Moving a body into a submodule called once never changes its pass counts."""

    torch.manual_seed(0)
    root = body()
    wrapped = _Wrapped(body())
    wrapped.body.load_state_dict(root.state_dict())
    x = torch.randn(2, 4)
    assert _pass_profile(tl.trace(wrapped, x)) == _pass_profile(tl.trace(root, x))


def test_a_block_called_twice_still_groups_across_calls() -> None:
    """A repeated module call keeps anchoring: its ops stay multi-pass layers."""

    torch.manual_seed(0)
    profile = _pass_profile(tl.trace(_AppliedTwice(), torch.randn(2, 4)))
    assert profile[("linear", 2)] == 2, profile
    adds = {n for func, n in profile if func == "__add__"}
    assert adds and min(adds) > 1, profile


class TestTinyGemma2:
    """F4's tiny random Gemma-2: no fused residual adds, module reads and replay work."""

    @pytest.fixture(scope="class")
    def setup(self):
        pytest.importorskip("transformers")
        from tests import _ag3_fixtures as fx

        model = fx.build_model(torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.trace(
                model,
                torch.tensor([fx.TOKENS]),
                capture=tl.options.CaptureOptions(intervention_ready=True),
            )
        return fx, model, trace

    def test_no_layer_inside_one_decoder_call_is_multi_pass(self, setup) -> None:
        _, _, trace = setup
        multi = sorted({str(op.layer_label) for op in trace if int(op.num_passes) > 1})
        assert multi == [], multi

    def test_module_input_ops_name_a_single_pass_site(self, setup) -> None:
        fx, _, trace = setup
        for layer in range(fx.N_LAYERS):
            (label,) = list(trace.modules[fx.READ.format(layer)].input_ops)
            assert int(trace[str(label)].num_passes) == 1
            assert fx.read_label(trace, layer) in (str(label), f"{label}:1")

    def test_replay_engine_holds_every_norm_and_pattern(self, setup) -> None:
        """Holding every rsqrt and softmax at its recorded value replays exactly."""

        _, _, trace = setup
        held = [str(op.label) for op in trace if op.func_name in ("rsqrt", "softmax")]
        assert held
        lm = str(trace.modules["lm_head"].output_ops[0])
        fork = trace.fork()
        fork.do(
            {label: tl.patch_from(trace) for label in held},
            intervention=tl.options.InterventionOptions(engine="replay"),
        )
        assert torch.equal(fork[lm].out, trace[lm].out)
