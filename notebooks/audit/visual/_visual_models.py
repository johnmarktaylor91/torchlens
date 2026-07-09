"""Extra models for the visual audit pack.

These extend the shared ``notebooks/audit/_models.py`` ZOO with architectures
needed to exercise the full TorchLens visual language: genuinely recurrent
models (cell-based loops that roll, fused RNN kernels that do not), deep
same-class stacks for ellipsis/repeat-fold demos, branching topologies,
a small transformer, multi-input/multi-output models, and degenerate cases.

Each factory returns ``(model, example_input_or_tuple)`` ready for
``tl.trace(model, *inputs)``.  All custom models are tiny; torchvision models
are constructed with ``weights=None`` (random init -- structure is all the
visual pack needs) and traced with small inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Recurrent models (Section D: loop rolling)
# ---------------------------------------------------------------------------


class RNNCellSeq(nn.Module):
    """nn.RNNCell applied over a sequence in a Python loop.

    Every timestep reuses the same cell parameters, so rolled mode should
    merge the per-step ops into one set of nodes with "(xT)" labels and a
    hidden-state back-edge.
    """

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.RNNCell(6, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[1], 8)
        for t in range(self.steps):
            h = self.cell(x[t], h)
        return self.head(h)


class LSTMCellSeq(nn.Module):
    """nn.LSTMCell loop: TWO recurrent state tensors (h and c) -> two back-edges."""

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.LSTMCell(6, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[1], 8)
        c = torch.zeros(x.shape[1], 8)
        for t in range(self.steps):
            h, c = self.cell(x[t], (h, c))
        return self.head(h)


class GRUCellSeq(nn.Module):
    """nn.GRUCell loop over a sequence."""

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.GRUCell(6, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[1], 8)
        for t in range(self.steps):
            h = self.cell(x[t], h)
        return h


class FusedLSTM(nn.Module):
    """nn.LSTM (fused cuDNN-style kernel): the whole sequence is ONE op call.

    Contrast page: fused kernels do NOT expose per-timestep internals, so
    there is nothing to roll -- the graph shows a single lstm op.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(6, 8, num_layers=1)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[-1])


class WeightTiedLoop(nn.Module):
    """Custom weight-tied refinement loop with parameterizable step count.

    The same two Linears are applied T times; rolled mode should show one
    block with "(xT)".
    """

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.mix = nn.Linear(8, 8)
        self.gate = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            x = x + torch.tanh(self.mix(x)) * torch.sigmoid(self.gate(x))
        return x


class BranchingLoop(nn.Module):
    """Weight-tied loop whose body BRANCHES on a tensor value each iteration.

    Iterations may take different arms, so the rolled graph must reconcile
    loop rolling with per-pass control flow.
    """

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.shrink = nn.Linear(8, 8)
        self.grow = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            if x.mean() > 0:
                x = torch.relu(self.shrink(x)) - 0.5
            else:
                x = torch.tanh(self.grow(x)) + 0.25
        return x


# ---------------------------------------------------------------------------
# Stacks and branching topologies (Sections E and J)
# ---------------------------------------------------------------------------


class SmallResBlock(nn.Module):
    """Conv-BN-ReLU residual block; stacking N of these gives N same-class,
    DIFFERENT-parameter instances (the "+N more" ellipsis case, and the BN
    buffers exercise collapsed-box remainder labels)."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)) + x)


class BlockStack(nn.Module):
    """Stem + 8 SmallResBlocks + head: minimal repeat-fold / ellipsis demo."""

    def __init__(self, depth: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 3, padding=1)
        self.blocks = nn.Sequential(*[SmallResBlock(4) for _ in range(depth)])
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(self.stem(x))
        return self.head(y.mean(dim=(2, 3)))


class MixedBuffers(nn.Module):
    """BatchNorm (noise buffers) plus a MEANINGFUL registered buffer.

    Built so show_buffer_layers' three modes all differ: 'never' hides both,
    'meaningful' shows only the offset buffer (BN running stats are classified
    as bookkeeping noise), 'always' shows everything.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.register_buffer("offset", torch.linspace(0, 1, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x)) + self.offset


class DictInput(nn.Module):
    """Consumes a dict input: {'a': tensor, 'b': tensor}."""

    def forward(self, payload: dict) -> torch.Tensor:
        return payload["a"] + payload["b"]


class NestedContainers(nn.Module):
    """Return nested dict-and-tuple output containers with live tensor leaves.

    The visual pack uses this model for container modes because a flat output
    can accidentally make several container presentations indistinguishable.
    """

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        """Build a nested output structure from three related tensor leaves.

        Parameters
        ----------
        x:
            Input activation.

        Returns
        -------
        dict[str, object]
            A dict containing a homogeneous tuple and a nested summary dict.
        """
        left = x + 1
        right = x * 2
        merged = left + right
        return {"branches": (left, right, merged), "summary": {"mean": merged.mean()}}


class AltBlockA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(x))


class AltBlockB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))


class InterleavedStack(nn.Module):
    """A/B/A/B/A/B alternating block classes.

    Run folding of interleaved same-class runs can draw edges that LOOK like
    a cycle (known rendering artifact); this model is the repro."""

    def __init__(self, pairs: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(pairs):
            layers += [AltBlockA(), AltBlockB()]
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class InceptionSprig(nn.Module):
    """One inception-style block: 4 parallel branches concatenated."""

    def __init__(self, cin: int, cout_each: int = 2) -> None:
        super().__init__()
        self.b1 = nn.Conv2d(cin, cout_each, 1)
        self.b3 = nn.Sequential(
            nn.Conv2d(cin, cout_each, 1), nn.Conv2d(cout_each, cout_each, 3, padding=1)
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(cin, cout_each, 1), nn.Conv2d(cout_each, cout_each, 5, padding=2)
        )
        self.bp = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(cin, cout_each, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class MiniInception(nn.Module):
    """Stem + two inception blocks: branching / merge topology at small scale."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 3, padding=1)
        self.inc1 = InceptionSprig(4)
        self.inc2 = InceptionSprig(8)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inc2(self.inc1(F.relu(self.stem(x))))
        return self.head(y.mean(dim=(2, 3)))


class TinyTransformer(nn.Module):
    """ONE TransformerEncoder layer on a short sequence.

    Small enough that every node label stays readable at page scale, which is
    what the attention-pattern and node_mode='attention' pages need.
    """

    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MiniTransformer(nn.Module):
    """Two TransformerEncoder layers: attention internals, LayerNorms, MHA."""

    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1))


# ---------------------------------------------------------------------------
# Multi-input / multi-output (Section J)
# ---------------------------------------------------------------------------


class MultiInMultiOut(nn.Module):
    """Two tensor inputs, two tensor outputs (tuple)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc_a = nn.Linear(6, 8)
        self.enc_b = nn.Linear(4, 8)
        self.head_sum = nn.Linear(8, 3)
        self.head_diff = nn.Linear(8, 3)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ya = torch.relu(self.enc_a(a))
        yb = torch.relu(self.enc_b(b))
        return self.head_sum(ya + yb), self.head_diff(ya - yb)


# ---------------------------------------------------------------------------
# Degenerate cases (Section K)
# ---------------------------------------------------------------------------


class SingleOp(nn.Module):
    """Forward is exactly one op; the smallest possible graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class NoSubmodules(nn.Module):
    """Functional ops only -- no child modules, no module boxes at all."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ x.T).sum(dim=1)


class ScalarOut(nn.Module):
    """Output is a 0-dim scalar tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).sum()


class NaNMidway(nn.Module):
    """Produces NaN in the MIDDLE of the graph (sqrt of a negative shift).

    Built for the ``node_overlay="nan"`` debugging demo: the overlay should
    flag the sqrt op and everything downstream of it, not the clean prefix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.fc1(x))
        y = torch.sqrt(y - 10.0)  # negative operand -> NaN from here on
        return self.fc2(y)


class SmallConv(nn.Module):
    """Small conv model on an RGB image input, for raw input/output rendering."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(F.relu(self.conv1(x))))


class ParamlessDeep(nn.Module):
    """A chain of ops with NO parameters anywhere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(5):
            x = torch.relu(x) * 0.9 + 0.05
        return x


# ---------------------------------------------------------------------------
# torchvision at scale (random init; structure only)
# ---------------------------------------------------------------------------


def _resnet18() -> tuple[nn.Module, torch.Tensor]:
    from torchvision.models import resnet18

    m = resnet18(weights=None).eval()
    return m, torch.randn(1, 3, 64, 64)


def _resnet50() -> tuple[nn.Module, torch.Tensor]:
    from torchvision.models import resnet50

    m = resnet50(weights=None).eval()
    return m, torch.randn(1, 3, 64, 64)


def _mobilenet_v2() -> tuple[nn.Module, torch.Tensor]:
    from torchvision.models import mobilenet_v2

    m = mobilenet_v2(weights=None).eval()
    return m, torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VZOO: dict[str, object] = {
    # recurrence
    "rnn_cell_seq": lambda: (RNNCellSeq(steps=4), torch.randn(4, 1, 6)),
    "lstm_cell_seq": lambda: (LSTMCellSeq(steps=4), torch.randn(4, 1, 6)),
    "gru_cell_seq": lambda: (GRUCellSeq(steps=4), torch.randn(4, 1, 6)),
    "fused_lstm": lambda: (FusedLSTM(), torch.randn(4, 1, 6)),
    "weight_tied_loop_2": lambda: (WeightTiedLoop(steps=2), torch.randn(1, 8)),
    "weight_tied_loop_5": lambda: (WeightTiedLoop(steps=5), torch.randn(1, 8)),
    "weight_tied_loop_8": lambda: (WeightTiedLoop(steps=8), torch.randn(1, 8)),
    "branching_loop": lambda: (BranchingLoop(steps=4), torch.randn(1, 8)),
    # stacks / topology
    # eval(): train-mode BN scatters orphan buffer-update ops across every
    # page; the one deliberate demo of that lives in Section A (a4_buffers).
    "block_stack": lambda: (BlockStack(depth=8).eval(), torch.randn(1, 1, 8, 8)),
    "interleaved_stack": lambda: (InterleavedStack(pairs=4), torch.randn(1, 8)),
    "mini_inception": lambda: (MiniInception(), torch.randn(1, 1, 8, 8)),
    "mini_transformer": lambda: (MiniTransformer(), torch.randn(1, 6, 16)),
    "tiny_transformer": lambda: (TinyTransformer(), torch.randn(1, 4, 16)),
    # multi-io
    "multi_in_multi_out": lambda: (MultiInMultiOut(), (torch.randn(2, 6), torch.randn(2, 4))),
    # degenerate
    "single_op": lambda: (SingleOp(), torch.randn(2, 3)),
    "no_submodules": lambda: (NoSubmodules(), torch.randn(3, 3)),
    "scalar_out": lambda: (ScalarOut(), torch.randn(2, 4)),
    "paramless_deep": lambda: (ParamlessDeep(), torch.randn(2, 4)),
    "nan_midway": lambda: (NaNMidway(), torch.randn(2, 4)),
    "small_conv": lambda: (SmallConv(), torch.rand(2, 3, 64, 64)),
    "mixed_buffers": lambda: (MixedBuffers(), torch.randn(4, 8)),
    "dict_input": lambda: (DictInput(), {"a": torch.ones(2), "b": torch.ones(2) * 2}),
    "nested_containers": lambda: (NestedContainers(), torch.ones(2)),
    # torchvision
    "resnet18": _resnet18,
    "resnet50": _resnet50,
    "mobilenet_v2": _mobilenet_v2,
}
