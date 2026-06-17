"""Shared tiny model zoo for TorchLens human-facing audit notebooks.

Each public factory in ZOO returns ``(model, example_input)`` -- a CPU-ready
pair that can be passed directly to ``tl.trace(model, x)``.  All models are
intentionally tiny so every notebook runs in seconds.

Usage::

    from _models import ZOO
    model, x = ZOO["tiny_mlp"]()
    trace = tl.trace(model, x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Models copied from notebooks/_demo_models.py
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """MLP with stable layer labels for intervention and bundle demos."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.relu(self.in_proj(x)))


class TinyBranchCNN(nn.Module):
    """Tiny conv net with one tensor-driven if/else branch and dual linear heads."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.up_head = nn.Linear(4, 3)
        self.down_head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            y = self.up_head(F.relu(self.conv(x)).mean(dim=(2, 3)))
        else:
            y = self.down_head(F.relu(self.conv(x)).mean(dim=(2, 3)).neg())
        return y


class _LoopInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.fc2(x)
            x = F.relu(x)
        return x


class _DemoInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loop_module = _LoopInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        x = F.relu(x)
        x = self.loop_module(x)
        return x


class DemoModel(nn.Module):
    """Kitchen-sink: cos op, buffer add, conditional branch, loop submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.register_buffer("example_buffer", torch.tensor([1, 2, 3, 4]).float())
        self.inner_module = _DemoInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cos(x)
        x = self.fc1(x) + self.example_buffer
        if x.mean() > 0:
            x = self.inner_module(x)
        x = x + torch.rand(x.shape)
        return x


# ---------------------------------------------------------------------------
# Models copied from tests/test_backward_visualization.py
# ---------------------------------------------------------------------------


class LinearReluModel(nn.Module):
    """Small model with module and functional ops; good for backward demos."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x)).sum()


class ViewModel(nn.Module):
    """Small model with a view op; exercises reshape in backward path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        return y.view(2, 3, 2).sum()


# ---------------------------------------------------------------------------
# Models copied from tests/test_render_bugs.py
# ---------------------------------------------------------------------------


class LargeChainRenderModel(nn.Module):
    """Deep linear chain -- exercises large-graph PDF bbox and layout."""

    def __init__(self, width: int = 4, depth: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


# ---------------------------------------------------------------------------
# Models copied from tests/test_conditional_rendering.py
# ---------------------------------------------------------------------------


class SimpleIfElseModel(nn.Module):
    """Minimal if/else branch -- relu or sigmoid depending on mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            return torch.relu(x)
        else:
            return torch.sigmoid(x)


class ElifLadderModel(nn.Module):
    """Four-arm if/elif/elif/else ladder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() < -0.5:
            return torch.relu(x)
        elif x.mean() < 0.0:
            return torch.sigmoid(x)
        elif x.mean() < 0.5:
            return torch.tanh(x)
        else:
            return torch.square(x)


# ---------------------------------------------------------------------------
# Models copied from tests/test_multi_output_modules.py
# ---------------------------------------------------------------------------


class LSTMModel(nn.Module):
    """Small LSTM model exposing all three outputs (output, h_n, c_n)."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(5, 10)
        self.label = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[1]
        h_0 = torch.zeros(1, batch_size, 10)
        c_0 = torch.zeros(1, batch_size, 10)
        _output, (h_n, _c_n) = self.lstm(x, (h_0, c_0))
        return self.label(h_n[-1])


# ---------------------------------------------------------------------------
# Models copied from tests/test_container_visualization.py
# ---------------------------------------------------------------------------


class DictOutputModel(nn.Module):
    """Return a two-leaf dict container output."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"a": x + 1, "b": x + 2}


class TupleOutputModel(nn.Module):
    """Return a four-element tuple output."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(x + i for i in range(4))


class _PairModule(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"left": x + 1, "right": x + 2}


class MidGraphContainerModel(nn.Module):
    """Container mid-graph: submodule returns dict, parent consumes leaves."""

    def __init__(self) -> None:
        super().__init__()
        self.pair = _PairModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = self.pair(x)
        return pair["left"] * pair["right"]


# ---------------------------------------------------------------------------
# Models copied from tests/test_edge_multiplicity_render.py
# ---------------------------------------------------------------------------


class AddTwice(nn.Module):
    """Input feeds both slots of add -- exercises edge-multiplicity rendering."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class CatTwice(nn.Module):
    """Input appears twice inside cat -- exercises sequence-arg multiplicity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x], dim=0)


# ---------------------------------------------------------------------------
# Models copied from tests/test_loop_module_rolling.py
# ---------------------------------------------------------------------------


class ReusedReluLoop(nn.Module):
    """Single ReLU module called 4x in a loop -- exercises rolling/self-loop labels."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(4):
            x = self.relu(x)
        return x


class RNNCellLoop(nn.Module):
    """RNN-cell-style loop with hidden-state recurrence -- exercises back-edge midpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(4):
            h = self.cell(h)
        return h


class ParallelFanout(nn.Module):
    """Shared projection stacked 4x -- exercises sibling ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.proj(x) for _ in range(4)])


# ---------------------------------------------------------------------------
# Extra: small BatchNorm model for buffer coverage
# ---------------------------------------------------------------------------


class BatchNormModel(nn.Module):
    """Simple BatchNorm model -- exposes running_mean/running_var buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


# ---------------------------------------------------------------------------
# ZOO registry
# ---------------------------------------------------------------------------

ZOO: dict[str, "callable[[], tuple[nn.Module, torch.Tensor]]"] = {
    "tiny_mlp": lambda: (TinyMLP(), torch.randn(2, 8)),
    # exercises: linear chain, relu, module accessors, intervention
    "tiny_branch_cnn": lambda: (TinyBranchCNN(), torch.ones(1, 1, 8, 8)),
    # exercises: conv2d, conditional branch, dual heads
    "demo_model": lambda: (DemoModel(), torch.randn(4).float()),
    # exercises: cos, buffer add, conditional, loop submodule
    "linear_relu": lambda: (LinearReluModel(), torch.randn(1, 3, requires_grad=True)),
    # exercises: small backward-friendly model, sum output, gradients
    "view_model": lambda: (ViewModel(), torch.randn(2, 6, requires_grad=True)),
    # exercises: view/reshape in backward path (batch=2 so view(2,3,2) matches 12 elements)
    "large_chain": lambda: (LargeChainRenderModel(width=4, depth=24), torch.randn(1, 4)),
    # exercises: large-graph bbox, deep chain rendering
    "simple_if_else": lambda: (SimpleIfElseModel(), torch.ones(2, 3)),
    # exercises: if/else conditional branch (THEN arm; mean > 0)
    "elif_ladder": lambda: (ElifLadderModel(), torch.full((2, 3), 0.3)),
    # exercises: four-arm if/elif/elif/else conditional rendering
    "lstm": lambda: (LSTMModel(), torch.randn(3, 2, 5)),
    # exercises: multi-output LSTM, tuple output, hidden state
    "dict_output": lambda: (DictOutputModel(), torch.randn(2)),
    # exercises: dict container output, show_containers modes
    "tuple_output": lambda: (TupleOutputModel(), torch.randn(2)),
    # exercises: homogeneous tuple container output
    "mid_graph_container": lambda: (MidGraphContainerModel(), torch.randn(2)),
    # exercises: container mid-graph, submodule dict consumed by parent
    "add_twice": lambda: (AddTwice(), torch.randn(3)),
    # exercises: edge multiplicity (both slots of add from same tensor)
    "cat_twice": lambda: (CatTwice(), torch.randn(3)),
    # exercises: sequence-arg edge multiplicity in cat
    "reused_relu_loop": lambda: (ReusedReluLoop(), torch.randn(4)),
    # exercises: rolled self-loop, single-module loop, label merge
    "rnn_cell_loop": lambda: (RNNCellLoop(), torch.randn(4)),
    # exercises: back-edge midpoints, hidden-state recurrence
    "parallel_fanout": lambda: (ParallelFanout(), torch.randn(4)),
    # exercises: sibling ordering (order_siblings), shared proj stacked 4x
    "batch_norm": lambda: (BatchNormModel(), torch.randn(4, 8)),
    # exercises: BatchNorm buffers (running_mean, running_var, num_batches_tracked)
}


if __name__ == "__main__":
    import sys

    print(f"Smoke-testing {len(ZOO)} models...\n")
    try:
        import torchlens as tl
    except ImportError:
        print("ERROR: torchlens not importable -- install in editable mode first")
        sys.exit(1)

    passed = []
    failed = []
    for name, factory in ZOO.items():
        try:
            m, x = factory()
            tl.trace(m, x)
            print(f"  OK   {name}")
            passed.append(name)
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failed.append(name)

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)
