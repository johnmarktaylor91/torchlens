"""Model axis by mechanism: small synthetic models, one per mechanism capture must handle.

Each row names one mechanism in its ``reason``: state carried across calls,
module sharing and tying, hooks installed before capture, compiled and
checkpointed children, in-place and aliasing behaviour, nested containers,
lazy materialization, ``torch.func`` transforms, custom autograd, keyword-only
inputs, mixed dtypes, sparse tensors, the attention fast path, non-persistent
buffers and generator-driven forwards. Data-dependent control flow lives in
``control_flow.py``; hooks that modify or raise belong to the hooks rows.

Every model is built locally (no downloads). Rows carry cost 1 to 3, so their
pre-merge cells stay cheap; the wider dtype variants run in the full matrix.
"""

from __future__ import annotations

import math
from collections import namedtuple
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from torch.utils import checkpoint

from ..generator import Inputs, ModelSpec

WIDTH = 8
BATCH = 3
STEPS = 5
VOCAB = 11
F32 = ("f32",)
F32_F64 = ("f32", "f64")
FLOAT_DTYPES = ("f32", "f16", "bf16", "f64")


def _vector(_model: nn.Module) -> Inputs:
    return ((torch.randn(BATCH, WIDTH),), {})


def _sequence(_model: nn.Module) -> Inputs:
    return ((torch.randn(BATCH, STEPS, WIDTH),), {})


class KVCacheAttention(nn.Module):
    """Single-head attention decoding one step at a time over a rolling KV cache.

    The cache is a plain attribute kept across top-level calls (a fixed window,
    so the shapes reach a steady state after the first call).
    """

    WINDOW = 4

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def _step(self, x_t: torch.Tensor) -> torch.Tensor:
        k, v = self.k(x_t), self.v(x_t)
        if self.cache_k is not None and self.cache_v is not None:
            k = torch.cat([self.cache_k, k], dim=1)[:, -self.WINDOW :]
            v = torch.cat([self.cache_v, v], dim=1)[:, -self.WINDOW :]
        self.cache_k, self.cache_v = k.detach(), v.detach()
        scores = self.q(x_t) @ k.transpose(1, 2) / math.sqrt(WIDTH)
        return self.out(torch.softmax(scores, dim=-1) @ v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._step(x[:, t : t + 1]) for t in range(x.shape[1])], dim=1)


class GRUCarriedState(nn.Module):
    """GRU whose hidden state lives in a buffer updated in place on every call."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.register_buffer("hidden", torch.zeros(1, BATCH, WIDTH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hidden = self.gru(x, self.hidden)
        self.hidden.copy_(hidden.detach())
        return out


class LSTMCarriedState(nn.Module):
    """LSTM whose ``(h, c)`` pair is a plain attribute carried across calls."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(WIDTH, WIDTH, batch_first=True)
        self.state: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.lstm(x, self.state)
        self.state = (h.detach(), c.detach())
        return out


class SharedTwice(nn.Module):
    """One submodule called twice in the same forward."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(torch.relu(self.shared(x)))


class TiedEmbedding(nn.Module):
    """Embedding and output projection share one weight parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.mix(self.embed(tokens))))


def _tokens(_model: nn.Module) -> Inputs:
    return ((torch.randint(0, VOCAB, (BATCH, STEPS)),), {})


class PreHooked(nn.Module):
    """Observer forward hook and pre-hook installed on a child before capture."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(WIDTH, WIDTH)
        self.second = nn.Linear(WIDTH, WIDTH)
        self.seen: list[tuple[int, ...]] = []
        self.first.register_forward_pre_hook(self._before)
        self.first.register_forward_hook(self._after)

    def _before(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        self.seen = [*self.seen[-3:], tuple(args[0].shape)]

    def _after(self, _module: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> None:
        self.seen = [*self.seen[-3:], tuple(out.shape)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


class CompiledChild(nn.Module):
    """A ``torch.compile``d child (eager backend, no toolchain) inside an eager parent."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.block = torch.compile(
            nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU()), backend="eager", dynamic=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(torch.relu(self.pre(x))) + x


class Checkpointed(nn.Module):
    """A ``torch.utils.checkpoint`` segment in the middle of the forward."""

    def __init__(self, reentrant: bool) -> None:
        super().__init__()
        self.reentrant = reentrant
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.block = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.pre(x))
        return checkpoint.checkpoint(self.block, h, use_reentrant=self.reentrant) + h


class InPlaceOnInput(nn.Module):
    """In-place arithmetic applied to the caller's input tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.mul_(0.5).add_(1.0)
        return self.linear(x)


class TrainModeBatchNorm(nn.Module):
    """BatchNorm kept in train mode, so forward mutates its running-stat buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
        self.norm = nn.BatchNorm1d(WIDTH)

    def train(self, mode: bool = True) -> TrainModeBatchNorm:
        super().train(mode)
        self.norm.train(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class AliasedOutputs(nn.Module):
    """Returns one tensor twice plus a view of it."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.linear(x))
        return h, h, h.t()


Pair = namedtuple("Pair", ["left", "right"])


class NestedContainers(nn.Module):
    """Takes a dict holding a list and a namedtuple; returns the same nesting."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(WIDTH, WIDTH)
        self.right = nn.Linear(WIDTH, WIDTH)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        pair = batch["pair"]
        parts = [torch.relu(self.left(item)) for item in batch["items"]]
        return {
            "sum": parts[0] + parts[1],
            "parts": parts,
            "pair": Pair(self.left(pair.left), self.right(pair.right)),
        }


def _nested(_model: nn.Module) -> Inputs:
    items = [torch.randn(BATCH, WIDTH), torch.randn(BATCH, WIDTH)]
    pair = Pair(torch.randn(BATCH, WIDTH), torch.randn(BATCH, WIDTH))
    return (({"items": items, "pair": pair},), {})


class LazyFirstCall(nn.Module):
    """``nn.LazyLinear`` recreated every forward, so the captured call is its first call."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy: nn.Module = nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(4)
        return self.lazy(torch.relu(self.pre(x)))


class VmapInside(nn.Module):
    """``torch.func.vmap`` over a per-row function inside forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def _row(self, row: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(row)) * row.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(self._row)(x)


class GradInside(nn.Module):
    """``torch.func.grad`` of an inner loss taken inside forward."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(WIDTH, WIDTH) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def loss(weight: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x @ weight).pow(2).sum()

        return x @ torch.func.grad(loss)(self.weight)


class ScaledSquare(torch.autograd.Function):
    """Custom autograd function: ``2 * x**2`` with a hand-written backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return 2.0 * x * x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad * 4.0 * x


class CustomAutograd(nn.Module):
    """A custom ``autograd.Function`` between two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(WIDTH, WIDTH)
        self.second = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(ScaledSquare.apply(self.first(x)))


class KeywordOnly(nn.Module):
    """Forward whose second and third inputs are keyword-only tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(
        self, x: torch.Tensor, *, scale: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.linear(x) * scale
        return y if mask is None else torch.where(mask, y, torch.zeros_like(y))


def _keyword_only(_model: nn.Module) -> Inputs:
    x = torch.randn(BATCH, WIDTH)
    return ((x,), {"scale": torch.rand(WIDTH) + 0.5, "mask": torch.randn(BATCH, WIDTH) > 0})


class MixedDtype(nn.Module):
    """fp16 parameters fed by an fp32 input cast to half inside forward, then cast back."""

    def __init__(self) -> None:
        super().__init__()
        self.half_linear = nn.Linear(WIDTH, WIDTH).half()
        self.float_linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.half_linear(x.half())
        return self.float_linear(h.float()) + x


class SparseMatmul(nn.Module):
    """A sparse COO adjacency buffer multiplied into dense features."""

    def __init__(self) -> None:
        super().__init__()
        dense = torch.eye(BATCH) + torch.diag(torch.ones(BATCH - 1), 1)
        self.register_buffer("adjacency", dense.to_sparse())
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.adjacency, self.linear(x))


class FastPathAttention(nn.Module):
    """``nn.MultiheadAttention`` set up for its inference fast path (eval, frozen, batch-first)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.attention.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(x, x, x, need_weights=False)
        return out


class NonPersistentBuffer(nn.Module):
    """A non-persistent buffer read and updated in place by forward (outside the state dict)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
        self.register_buffer("running", torch.zeros(WIDTH), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        self.running.mul_(0.5).add_(h.detach().mean(0))
        return h - self.running


class _SharedBranch(nn.Module):
    """One of two models built around the same block instance."""

    def __init__(self, block: nn.Module, act: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x))


class SharedAcrossModels(nn.Module):
    """Two separately built models that hold one block instance, run side by side."""

    def __init__(self) -> None:
        super().__init__()
        block = nn.Linear(WIDTH, WIDTH)
        self.model_a = _SharedBranch(block, nn.ReLU())
        self.model_b = _SharedBranch(block, nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model_a(x) + self.model_b(x)


class GeneratorForward(nn.Module):
    """Forward driven by a generator method that yields per stage and returns early."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(4))

    def _run(self, x: torch.Tensor) -> Iterator[torch.Tensor]:
        for index, stage in enumerate(self.stages):
            x = torch.relu(stage(x))
            yield x
            if index == 2:
                return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = list(self._run(x))
        return sum((out.mean(-1, keepdim=True) for out in outs), outs[-1])


def _row(
    row_id: str,
    build: Callable[[], nn.Module],
    reason: str,
    *,
    example_input: Callable[[nn.Module], Inputs] = _vector,
    dtypes: tuple[str, ...] = F32_F64,
    cost: int = 1,
    memory_budget_mib: int = 512,
) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=build,
        example_input=example_input,
        reason=reason,
        dtypes=dtypes,
        cost=cost,
        memory_budget_mib=memory_budget_mib,
        tags=frozenset({"mechanism"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _row(
        "mech_kv_cache",
        KVCacheAttention,
        "attention decoding over a KV cache kept as a plain attribute across calls",
        example_input=_sequence,
    ),
    _row(
        "mech_gru_state",
        GRUCarriedState,
        "GRU hidden state carried across calls in a buffer updated in place",
        example_input=_sequence,
    ),
    _row(
        "mech_lstm_state",
        LSTMCarriedState,
        "LSTM (h, c) carried across calls as a plain attribute",
        example_input=_sequence,
    ),
    _row(
        "mech_shared_twice",
        SharedTwice,
        "one submodule called twice in one forward",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_tied_params",
        TiedEmbedding,
        "embedding and output head tied to one parameter",
        example_input=_tokens,
    ),
    _row(
        "mech_prehooks",
        PreHooked,
        "observer forward hook and pre-hook installed before capture",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_compiled_child",
        CompiledChild,
        "torch.compile child inside an eager parent (a boundary; eager source logged)",
        dtypes=F32,
        cost=3,
        memory_budget_mib=1024,
    ),
    _row(
        "mech_checkpoint_reentrant",
        lambda: Checkpointed(reentrant=True),
        "reentrant torch.utils.checkpoint segment",
        cost=2,
    ),
    _row(
        "mech_checkpoint_nonreentrant",
        lambda: Checkpointed(reentrant=False),
        "non-reentrant torch.utils.checkpoint segment",
        cost=2,
    ),
    _row(
        "mech_inplace_input",
        InPlaceOnInput,
        "in-place ops applied to the caller's input tensor",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_bn_train",
        TrainModeBatchNorm,
        "BatchNorm in train mode mutating its running-stat buffers in forward",
    ),
    _row(
        "mech_aliased_outputs",
        AliasedOutputs,
        "one tensor returned twice plus a view of it",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_nested_io",
        NestedContainers,
        "dict input holding a list and a namedtuple; nested dict, list, namedtuple output",
        example_input=_nested,
        dtypes=F32,
    ),
    _row(
        "mech_lazy_first_call",
        LazyFirstCall,
        "nn.LazyLinear materialized on its first call inside the captured forward",
        dtypes=F32,
    ),
    _row(
        "mech_func_vmap",
        VmapInside,
        "torch.func.vmap called inside forward",
        cost=2,
    ),
    _row(
        "mech_func_grad",
        GradInside,
        "torch.func.grad called inside forward",
        cost=2,
    ),
    _row(
        "mech_custom_autograd",
        CustomAutograd,
        "custom torch.autograd.Function with its own backward",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_kwonly_inputs",
        KeywordOnly,
        "forward with keyword-only tensor inputs",
        example_input=_keyword_only,
    ),
    _row(
        "mech_mixed_dtype",
        MixedDtype,
        "fp16 parameters with an fp32 input cast to half and back inside forward",
        dtypes=F32,
    ),
    _row(
        "mech_sparse_mm",
        SparseMatmul,
        "sparse COO buffer through torch.sparse.mm",
    ),
    _row(
        "mech_mha_fastpath",
        FastPathAttention,
        "nn.MultiheadAttention eligible for its inference fast path",
        example_input=_sequence,
        cost=2,
    ),
    _row(
        "mech_nonpersistent_buffer",
        NonPersistentBuffer,
        "non-persistent buffer updated in place in forward (outside the state dict)",
    ),
    _row(
        "mech_shared_across_models",
        SharedAcrossModels,
        "one block instance owned by two separately built models",
        dtypes=FLOAT_DTYPES,
    ),
    _row(
        "mech_generator_forward",
        GeneratorForward,
        "forward consuming a generator that yields per stage and returns early",
        dtypes=FLOAT_DTYPES,
    ),
)
