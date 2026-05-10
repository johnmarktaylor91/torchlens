"""Fixture models for module-containment snapshot equality tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch
from torch import nn

FixtureResult: TypeAlias = tuple[nn.Module, Any, str]
HookedFixtureResult: TypeAlias = tuple[nn.Module, Any, str, torch.utils.hooks.RemovableHandle]
FixtureBuilder: TypeAlias = Callable[[], FixtureResult | HookedFixtureResult]


def _seed() -> None:
    """Reset torch RNG state for deterministic fixture construction."""

    torch.manual_seed(42)


def tiny_mlp() -> FixtureResult:
    """Return a tiny MLP in a sequential container."""

    _seed()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    return model, torch.randn(2, 8), "tiny_mlp"


class _TinyResnetBlock(nn.Module):
    """Small residual block split into two sequential halves."""

    def __init__(self) -> None:
        """Initialize residual block submodules."""

        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.second = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
        )
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual block forward pass."""

        y = self.first(x)
        y = self.second(y)
        return self.out_relu(y + x)


def tiny_resnet_block() -> FixtureResult:
    """Return a tiny residual Conv/BN block."""

    _seed()
    return _TinyResnetBlock(), torch.randn(1, 4, 8, 8), "tiny_resnet_block"


class _RecurrentLstmCell(nn.Module):
    """Loop over one shared LSTMCell three times."""

    def __init__(self) -> None:
        """Initialize recurrent cell state projection."""

        super().__init__()
        self.cell = nn.LSTMCell(8, 16)
        self.readout = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent LSTMCell steps."""

        h = x.new_zeros(x.shape[0], 16)
        c = x.new_zeros(x.shape[0], 16)
        for _ in range(3):
            h, c = self.cell(x, (h, c))
        return self.readout(h)


def recurrent_lstm_cell() -> FixtureResult:
    """Return an LSTMCell recurrence fixture."""

    _seed()
    return _RecurrentLstmCell(), torch.randn(2, 8), "recurrent_lstm_cell"


class _BranchingUnet(nn.Module):
    """Two-level U-Net style model with ModuleList skip connections."""

    def __init__(self) -> None:
        """Initialize encoder and decoder convolution lists."""

        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Conv2d(4, 8, kernel_size=3, padding=1),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.Conv2d(12, 4, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run U-Net style forward pass with additive and concat skips."""

        skips = []
        y = x
        for encoder in self.encoders:
            y = torch.relu(encoder(y))
            skips.append(y)
        y = torch.cat([y, skips[-2]], dim=1)
        y = torch.relu(self.decoders[0](y))
        y = torch.cat([y, skips[-3]], dim=1)
        y = torch.relu(self.decoders[1](y))
        y = torch.cat([y, x], dim=1)
        return self.decoders[2](y)


def branching_unet() -> FixtureResult:
    """Return a small U-Net style branching fixture."""

    _seed()
    return _BranchingUnet(), torch.randn(1, 4, 16, 16), "branching_unet"


class _ConditionalModule(nn.Module):
    """Module with tensor-predicate branch over two different arms."""

    def __init__(self) -> None:
        """Initialize both conditional arms."""

        super().__init__()
        self.positive = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        self.negative = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one of two submodule arms based on input mean."""

        if x.mean() > 0:
            return self.positive(x)
        return self.negative(x)


def conditional_module_positive() -> FixtureResult:
    """Return conditional fixture input that exercises the positive arm."""

    _seed()
    return _ConditionalModule(), torch.randn(2, 8) + 10, "conditional_module_positive"


def conditional_module_negative() -> FixtureResult:
    """Return conditional fixture input that exercises the negative arm."""

    _seed()
    return _ConditionalModule(), torch.randn(2, 8) - 10, "conditional_module_negative"


class _InternalFactoryModule(nn.Module):
    """Linear module fed by an internally created zeros_like tensor."""

    def __init__(self) -> None:
        """Initialize projection submodule."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create an internal tensor before projection."""

        z = torch.zeros_like(x)
        return self.proj(x + z)


def internal_factory_module() -> FixtureResult:
    """Return internal factory tensor fixture."""

    _seed()
    return _InternalFactoryModule(), torch.randn(2, 4), "internal_factory_module"


class _MultiheadAttentionDemo(nn.Module):
    """Small MultiheadAttention wrapper."""

    def __init__(self) -> None:
        """Initialize attention module."""

        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention and return the attention output."""

        out, _weights = self.attn(x, x, x, need_weights=True)
        return out


def multihead_attention_demo() -> FixtureResult:
    """Return MultiheadAttention fixture."""

    _seed()
    return _MultiheadAttentionDemo(), torch.randn(4, 2, 8), "multihead_attention_demo"


def lazy_linear_demo() -> FixtureResult:
    """Return LazyLinear materialization fixture."""

    _seed()
    model = nn.Sequential(nn.LazyLinear(16), nn.ReLU())
    input_args = torch.randn(2, 8)
    model(input_args)
    return model, input_args, "lazy_linear_demo"


class _MixedContainers(nn.Module):
    """Model combining Sequential, ModuleList, ModuleDict, and raw ops."""

    def __init__(self) -> None:
        """Initialize mixed container submodules."""

        super().__init__()
        self.seq = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.ReLU()])
        self.choices = nn.ModuleDict({"proj": nn.Linear(8, 4), "gate": nn.Sigmoid()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through mixed containers with raw torch ops between calls."""

        y = self.seq(x)
        y = torch.sin(y)
        y = self.layers[0](y)
        y = self.layers[1](y)
        y = y + torch.zeros_like(y)
        y = self.choices["gate"](y)
        return self.choices["proj"](y)


def mixed_containers() -> FixtureResult:
    """Return mixed-container fixture."""

    _seed()
    return _MixedContainers(), torch.randn(2, 8), "mixed_containers"


class _FactoryThenSibling(nn.Module):
    """Factory tensor followed by sibling submodule call."""

    def __init__(self) -> None:
        """Initialize projection submodule."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create zeros_like tensor and feed sibling projection."""

        z = torch.zeros_like(x)
        return self.proj(z + x)


def factory_then_submodule() -> FixtureResult:
    """Return factory-then-submodule fixture."""

    _seed()
    return _FactoryThenSibling(), torch.randn(2, 4), "factory_then_submodule"


class _InplaceThenSibling(nn.Module):
    """In-place ReLU followed by sibling Linear."""

    def __init__(self) -> None:
        """Initialize in-place and sibling submodules."""

        super().__init__()
        self.a = nn.ReLU(inplace=True)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run in-place activation before sibling projection."""

        y = x + 1
        y = self.a(y)
        return self.b(y)


def inplace_nested_sibling() -> FixtureResult:
    """Return in-place nested sibling fixture."""

    _seed()
    return _InplaceThenSibling(), torch.randn(2, 4), "inplace_nested_sibling"


class _TupleDictReturn(nn.Module):
    """Inner module returning a dict containing a tensor tuple."""

    def __init__(self) -> None:
        """Initialize tuple-return sibling modules."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return both sibling outputs in a dict."""

        a = self.left(x)
        b = self.right(x)
        return {"pair": (a, b)}


class _TupleDictOuter(nn.Module):
    """Outer wrapper that reduces tuple/dict output to one tensor."""

    def __init__(self) -> None:
        """Initialize inner tuple/dict module."""

        super().__init__()
        self.inner = _TupleDictReturn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single tensor derived from inner tuple/dict output."""

        pair = self.inner(x)["pair"]
        return pair[0] + pair[1]


def tuple_dict_return() -> FixtureResult:
    """Return tuple/dict-return fixture wrapped to one tensor output."""

    _seed()
    return _TupleDictOuter(), torch.randn(2, 4), "tuple_dict_return"


class _HookReplacementModel(nn.Module):
    """ReLU fixture with a downstream projection for hook replacement tests."""

    def __init__(self) -> None:
        """Initialize ReLU and downstream projection."""

        super().__init__()
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run hooked ReLU then downstream projection."""

        return self.proj(self.relu(x))


def raw_hook_replacement_logged() -> HookedFixtureResult:
    """Return fixture whose user hook replacement is logged by TorchLens."""

    _seed()
    model = _HookReplacementModel()
    handle = model.relu.register_forward_hook(lambda _mod, _inp, output: output * 0.5)
    return model, torch.randn(2, 4), "raw_hook_replacement_logged", handle


def raw_hook_replacement_synthetic() -> HookedFixtureResult:
    """Return fixture whose user hook replacement is a fresh tensor."""

    _seed()
    model = _HookReplacementModel()
    replacement = torch.ones(2, 4)
    handle = model.relu.register_forward_hook(lambda _mod, _inp, _output: replacement)
    return model, torch.randn(2, 4), "raw_hook_replacement_synthetic", handle


class _SharedTwoAddrs(nn.Module):
    """Same Linear instance registered at two addresses."""

    def __init__(self) -> None:
        """Initialize shared and alias addresses."""

        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.alias = self.shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same module instance through both addresses."""

        return self.shared(x) + self.alias(x)


def shared_instance_two_addresses() -> FixtureResult:
    """Return shared-instance two-address fixture."""

    _seed()
    return _SharedTwoAddrs(), torch.randn(2, 4), "shared_instance_two_addresses"


class _DynamicBufferChild(nn.Module):
    """Child module that consumes a dynamic buffer."""

    def __init__(self) -> None:
        """Initialize child projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, dyn: torch.Tensor) -> torch.Tensor:
        """Add dynamic buffer before projection."""

        return self.proj(x + dyn)


class _DynamicBufferModule(nn.Module):
    """Model registering a buffer inside forward on first call."""

    def __init__(self) -> None:
        """Initialize dynamic-buffer child module."""

        super().__init__()
        self.child = _DynamicBufferChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Register a buffer once and use it in a child module."""

        if "dyn" not in self._buffers:
            self.register_buffer("dyn", torch.randn(4))
        return self.child(x, self.dyn)


def dynamic_buffer_module() -> FixtureResult:
    """Return dynamic-buffer fixture."""

    _seed()
    return _DynamicBufferModule(), torch.randn(2, 4), "dynamic_buffer_module"


ALL_FIXTURES: tuple[FixtureBuilder, ...] = (
    tiny_mlp,
    tiny_resnet_block,
    recurrent_lstm_cell,
    branching_unet,
    conditional_module_positive,
    conditional_module_negative,
    internal_factory_module,
    multihead_attention_demo,
    lazy_linear_demo,
    mixed_containers,
    factory_then_submodule,
    inplace_nested_sibling,
    tuple_dict_return,
    raw_hook_replacement_logged,
    raw_hook_replacement_synthetic,
    shared_instance_two_addresses,
    dynamic_buffer_module,
)
