# SOURCE: vendored from Ber666/ToolkenGPT @ main (commit 5a5d4121bad0fd198f43d76320ddead091edf350)
# https://raw.githubusercontent.com/Ber666/ToolkenGPT/main/llama/model.py
#
# Hao et al. 2023 (NeurIPS 2023 oral) "ToolkenGPT: Augmenting Frozen Language Models
# with Massive Tools via Tool Embeddings" -- a LLaMA-architecture decoder
# (`RMSNorm`, rotary positional embeddings `apply_rotary_emb`, causal self-attention
# `Attention` with a KV cache, SwiGLU `FeedForward`, pre-norm `TransformerBlock`
# stack in `Transformer`) wrapped by `FunctionLM`, which freezes the base LLaMA and
# adds a single new architectural component: `func_embed`, an `nn.Linear` "toolken"
# head projecting the final hidden state to per-function logits. At each decoding
# step `FunctionLM.generate` concatenates the frozen LM's token logits
# (`self.model.output(h)`) with the toolken head's function logits
# (`self.func_embed(h)`) into one joint distribution over (vocabulary union
# tool-call tokens) -- this is the paper's actual mechanism, not a paraphrase.
#
# `RMSNorm`, `precompute_freqs_cis`, `reshape_for_broadcast`, `apply_rotary_emb`,
# `Attention`, `FeedForward`, `TransformerBlock`, `Transformer`, `FunctionLM` are
# copied with unchanged forward-pass bodies from `llama/model.py`. No architectural
# code was rewritten; only these mechanical, import-isolation changes were made:
#   - Upstream's `Attention`/`FeedForward`/`Transformer` use fairscale's
#     `ColumnParallelLinear`/`RowParallelLinear`/`ParallelEmbedding` for
#     tensor-model-parallel sharding across multiple GPUs (`fs_init.get_model_
#     parallel_world_size()`), each constructed with `init_method=lambda x: x`
#     (i.e. no special init -- they behave as plain, unsharded `nn.Linear`/
#     `nn.Embedding` whenever `world_size == 1`, which is exactly the case here:
#     this build runs single-process with no `torch.distributed`/fairscale process
#     group). `fairscale` is not an installed base library, so those three classes
#     are replaced 1:1 with plain `nn.Linear(..., bias=False)` /
#     `nn.Embedding(...)` -- same shapes, same computation
#     (`y = x @ W.T` / embedding lookup), just without the sharding machinery that
#     is a no-op at `world_size=1`. `self.n_local_heads` (`= n_heads //
#     world_size`) becomes `self.n_heads` directly.
#   - The float16+CUDA-only KV cache (`torch.zeros(...).cuda()` inside
#     `Attention.__init__`, and `torch.set_default_tensor_type(torch.cuda.HalfTensor)`
#     in the real `load()` entrypoint) is generalized to build the cache on
#     `x.device`/`x.dtype` at first `forward()` call, so this runs on CPU/float32 for
#     validation while exercising the identical caching logic.
#   - `FunctionLM.__init__` hardcodes `.to("cuda")` for `func_embed` and the real
#     `load()`/`generate()` entrypoints assume a CUDA + fairscale distributed
#     environment (checkpoint sharding, `torch.distributed.init_process_group`,
#     tokenizer files). Those loading/generation entrypoints are training/inference
#     driver code, not the model architecture; `build_toolkengpt()` below
#     constructs `Transformer` + `FunctionLM` directly (skipping the checkpoint
#     loader).
#   - Upstream `FunctionLM` has no `forward` method (only `get_loss`/`generate`,
#     both driver entrypoints that do tokenizer-driven text handling around the
#     same tensor computation). The tensor computation inside `get_loss` --
#     `self.model(input_ids, 0)` -> `self.model.output(h)` (frozen LM token
#     logits) concatenated with `self.func_embed(h)` (toolken logits) -> cross
#     entropy -- is renamed to `forward` here, verbatim, purely so the module is
#     directly callable as `model(input_ids, labels)`; no computation was added,
#     removed, or reordered.

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # NOTE: upstream `self.n_local_heads = args.n_heads //
        # fs_init.get_model_parallel_world_size()`; single-process here so
        # world_size == 1 and n_local_heads == n_heads.
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # NOTE: upstream uses fairscale ColumnParallelLinear/RowParallelLinear with
        # init_method=lambda x: x (no special init); at world_size=1 these reduce to
        # plain nn.Linear(bias=False).
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.cache_k = None
        self.cache_v = None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.cache_k is None:
            self.cache_k = torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_local_heads, self.head_dim),
                device=xq.device,
                dtype=xq.dtype,
            )
            self.cache_v = torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_local_heads, self.head_dim),
                device=xq.device,
                dtype=xq.dtype,
            )

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # NOTE: plain nn.Linear replacing fairscale Column/RowParallelLinear (see
        # module header) -- identical computation at world_size=1.
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # NOTE: plain nn.Embedding replacing fairscale ParallelEmbedding (see module
        # header) -- identical lookup at world_size=1.
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # (bsz, partial_seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), h


class FunctionLM(nn.Module):
    def __init__(self, base_model, func_dict, load_path=None, inference_mode="func_embedding"):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.func_dict = func_dict
        self.func_list = {v: k for k, v in func_dict.items()}
        # NOTE: `.to("cuda")` dropped so this runs on CPU for validation; the
        # architectural component (a plain nn.Linear "toolken" head) is unchanged.
        self.func_embed = nn.Linear(base_model.params.dim, len(func_dict), bias=False)
        if load_path is not None and load_path != "None":  # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = {"weight": embedding}

            # truncate the embedding if necessary
            if embedding["weight"].shape[0] > len(func_dict):
                embedding["weight"] = embedding["weight"][: len(func_dict)]

            self.func_embed.load_state_dict(embedding)

        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.logits_bias = 0

    def set_bias(self, logits_bias):
        self.logits_bias = logits_bias

    def forward(self, input_ids, labels):
        # NOTE: renamed from upstream `get_loss` (see module header) -- this is the
        # architecture's actual joint token+toolken forward/loss path (tokenizer-driven
        # label construction from raw text is upstream data-prep, not architecture;
        # here `input_ids`/`labels` are supplied directly as already-tokenized tensors).
        with torch.no_grad():
            last_logits, h = self.model(input_ids, 0)  # h: (bsz, seqlen, dim)
            token_logits = self.model.output(h)  # (bsz, seqlen, vocab_size)

        func_logits = self.func_embed(h.float())  # (bsz, seqlen, len(func_list))

        concat_logits = torch.cat(
            [token_logits, func_logits], dim=-1
        )  # (bsz, seqlen, vocab_size + len(func_list))
        loss = F.cross_entropy(
            concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100
        )
        return loss, concat_logits


def _build_func_dict(n_funcs=4):
    return {f"<func_{i}>": i for i in range(n_funcs)}


def build_toolkengpt():
    args = ModelArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        multiple_of=8,
        max_batch_size=2,
        max_seq_len=32,
    )
    base_model = Transformer(args)
    func_dict = _build_func_dict()
    return FunctionLM(base_model, func_dict)


def example_input_toolkengpt():
    batch_size = 1
    seq_len = 10
    vocab_size = 128
    n_funcs = 4

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size + n_funcs, (batch_size, seq_len))
    return (input_ids, labels)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ToolkenGPT", "build_toolkengpt", "example_input_toolkengpt", 2023, "vendored"),
]
