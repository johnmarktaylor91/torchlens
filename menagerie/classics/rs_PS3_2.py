# SOURCE: vendored from lucidrains/simplicial-attention @ ce34f62fc3b5dfc612d3bc874d8e9e68c1e18e93
# https://raw.githubusercontent.com/lucidrains/simplicial-attention/main/simplicial_attention/simplicial_attention.py
# https://raw.githubusercontent.com/lucidrains/simplicial-attention/main/simplicial_attention/simplicial_mha.py
#
# Clift, Doryn, Murfet 2019 "Logic and the 2-Simplicial Transformer" (arXiv:1909.00668)
# introduced the theory; Roy et al. 2025 "Fast and Simplex: Attention with Higher-Order
# Simplicial Interactions" (arXiv:2507.02754) made 2-simplicial attention practical with
# fused Triton kernels. This module vendors the naive (pure PyTorch, non-Triton) reference
# path from lucidrains/simplicial-attention: `HigherOrderAttention` / `TwoSimplicialMHA`
# generalize dot-product query-key attention to a TRILINEAR query/key1/key2 similarity
# tensor (`naive_two_simplicial_attend`, with an optional `signed_determinant` similarity
# per the paper's Eq. 8), softmaxed jointly over both key axes and contracted against two
# value sets -- a genuinely higher-order attention primitive, not a masked/kernel
# approximation of ordinary attention.
#
# The fused Triton kernel (triton_two_simplicial_attention.py) and the extra
# hyper-connections / x-mlps-pytorch package deps used elsewhere in the repo are NOT
# vendored (not needed): `HigherOrderAttention`'s default `attend=naive_two_simplicial_attend`
# code path used here needs only torch + einops + opt_einsum (opt_einsum ships as a
# transitive dependency in this environment already; no extra install was performed).
"""2-simplicial attention: trilinear query/key1/key2 higher-order attention primitive."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from einops import einsum, pack, rearrange, unpack
from einops.layers.torch import Rearrange
from opt_einsum import contract
from torch import Tensor, cat, stack, tensor
from torch.nn import Identity, Linear, Module, ModuleList, RMSNorm

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from simplicial_attention/simplicial_attention.py ---
def divisible_by(num, den):
    return (num % den) == 0


def join(arr, delimiter=", "):
    return delimiter.join(arr)


# rotary


def apply_rotation(
    t: Tensor,
    rot: Tensor,  # Float[3, 3]
):
    device, dim = t.device, t.shape[-1]
    dim = dim // 3 * 3

    t, t_rest = t[..., :dim], t[..., dim:]
    t = rearrange(t, "... (d r) -> ... d r", r=3)
    t = t @ rot
    t = rearrange(t, "... d r -> ... (d r)")

    return cat((t, t_rest), dim=-1)


# signed determinant


def signed_determinant(q, k1, k2):
    device, dim = q.device, q.shape[-1]
    dim = dim // 3 * 3

    q, q_rest = q[..., :dim], q[..., dim:]
    k1, k1_rest = k1[..., :dim], k1[..., dim:]
    k2, k2_rest = k2[..., :dim], k2[..., dim:]

    has_rest = q_rest.numel() > 0

    # following eq 8.
    # they use this in place of dot product for similarity in attention
    # for rotating in positions and keeping invariance
    # i don't know if all this effort really adds anything, but it is a fun exercise

    k1 = rearrange(k1, "... (d r) -> ... d r", r=3)
    k2 = rearrange(k2, "... (d r) -> ... d r", r=3)

    index1 = tensor([2, 0, 1], device=device)
    index2 = tensor([1, 2, 0], device=device)

    lq = q
    rq = q
    lk1 = torch.index_select(k1, dim=-1, index=index2)
    rk1 = torch.index_select(k1, dim=-1, index=index1)
    lk2 = torch.index_select(k2, dim=-1, index=index1)
    rk2 = torch.index_select(k2, dim=-1, index=index2)

    lk1, rk1, lk2, rk2 = (rearrange(t, "... d r -> ... (d r)") for t in (lk1, rk1, lk2, rk2))

    if has_rest:
        lq = cat((lq, q_rest), dim=-1)
        lk1 = cat((lk1, k1_rest), dim=-1)
        lk2 = cat((lk2, k2_rest), dim=-1)

    lhs = einsum(lq, lk1, lk2, "b h ... i d, b h j d, b h k d -> b h ... i j k")

    rhs = einsum(rq, rk1, rk2, "b h ... i d, b h j d, b h k d -> b h ... i j k")

    return lhs - rhs


# 2-simplicial attention


def naive_two_simplicial_attend(
    q: Tensor,  # b h i d
    k: tuple[Tensor, Tensor],  # (b h j d,  b h k d)
    v: tuple[Tensor, Tensor],  # (b h j dv, b h k dv)
    causal=False,
    use_signed_determinant=False,
):  # b h i dv
    assert len(k) == len(v) == 2

    k1, k2 = k
    v1, v2 = v

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], k1.shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, "b (h g) i d -> b h g i d", g=groups)

    # variables

    scale = dim**-0.5

    q = q * scale

    if use_signed_determinant:
        sim = signed_determinant(q, k1, k2)
    else:
        sim = contract("... g i d, ... j d, ... k d -> ... g i j k", q, k1, k2)

    if causal:
        i, j = sim.shape[-2:]
        assert i == j

        causal_mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
        causal_mask = causal_mask[..., :, None] | causal_mask[..., None, :]
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    packed_sim, packed_shape = pack((sim,), "b h g i *")

    packed_attn = packed_sim.softmax(dim=-1)

    (attn,) = unpack(packed_attn, packed_shape, "b h g i *")

    out = contract("... g i j k, ... j d, ... k d -> ... g i d", attn, v1, v2)

    return rearrange(out, "b h g ... -> b (h g) ...")


# n-th order attention, for good measure


def nth_order_attend(
    q: Tensor,  # b h i d
    keys: tuple[Tensor, ...],  # tuple[b h jkl... d]
    values: tuple[Tensor, ...],  # tuple[b h jkl... dv]
    causal=False,
):  # b h i dv
    assert len(keys) == len(values)
    n = len(keys)

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], keys[0].shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, "b (h g) i d -> b h g i d", g=groups)

    scale = q.shape[-1] ** -0.5

    q = q * scale

    # construct equations

    start_index = ord("j")

    ord_indices = list(range(start_index, start_index + n))

    similarity_lfs_eq = join([f"... {chr(i)} d" for i in ord_indices], ", ")

    similarity_rhs_eq = join([chr(i) for i in ord_indices], " ")

    similarity_ein_equation = f"... g i d, {similarity_lfs_eq} -> ... g i {similarity_rhs_eq}"

    aggregate_ein_equation = f"... g i {similarity_rhs_eq}, {similarity_lfs_eq} -> ... g i d"

    # nth order attention

    sim = contract(similarity_ein_equation, q, *keys)

    # maybe causal

    if causal:
        seq_len = sim.shape[-1]
        one_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).triu(1)

        causal_mask = one_mask

        for _ in range(n - 1):
            one_mask = one_mask[..., None, :]
            causal_mask = causal_mask[..., :, None] | one_mask

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # attention

    packed_sim, packed_shape = pack((sim,), "b h g i *")

    packed_attn = packed_sim.softmax(dim=-1)

    (attn,) = unpack(packed_attn, packed_shape, "b h g i *")

    # aggregate out

    out = contract(aggregate_ein_equation, attn, *values)

    return rearrange(out, "b h g ... -> b (h g) ...")


# --- vendored from simplicial_attention/simplicial_mha.py ---
def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class HigherOrderAttention(Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,  # query/key head dimension
        dim_head_values=None,  # value head dimension, defaults to `dim_head`
        heads=8,  # query heads
        key_value_heads=None,  # key/value heads, default to query heads `heads`
        number_key_value_sets=2,  # 2 for 2-simplicial, but can go higher. the century is young
        qk_rmsnorm=True,  # qk rmsnorm, used in a number of models without issues now. helps with stability
        prenorm=False,  # pre rmsnorm for pre-norm transformer pattern
        postnorm=False,  # post rmsnorm, proven out in alphagenome for even more stability (sandwich norm from some old paper i will find later)
        attend: Callable | None = None,
        head_first_dim=True,
    ):
        super().__init__()

        # variables

        self.causal = causal
        self.scale = dim_head**-0.5

        key_value_heads = default(key_value_heads, heads)

        assert divisible_by(heads, key_value_heads)
        self.query_head_groups = heads // key_value_heads

        dim_head_values = default(dim_head_values, dim_head)

        kv_sets = number_key_value_sets

        # maybe pre norm or post norm

        self.prenorm = RMSNorm(dim) if prenorm else Identity()

        self.postnorm = RMSNorm(dim) if postnorm else Identity()

        # maybe qk rmsnorm

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norms = ModuleList([RMSNorm(dim_head) for _ in range(kv_sets)])
            self.v_norms = ModuleList([RMSNorm(dim_head) for _ in range(kv_sets)])

        # to queries and sets of keys / values

        self.split_dims = (
            heads * dim_head,  # queries
            kv_sets * key_value_heads * dim_head,  # keys
            kv_sets * key_value_heads * dim_head_values,  # values
        )

        split_heads_eq = "b n (h d) -> b h n d" if head_first_dim else "b n (h d) -> b n h d"
        merge_heads_eq = "b h n d -> b n (h d)" if head_first_dim else "b n h d -> b n (h d)"

        self.split_q_heads = Rearrange(split_heads_eq, h=heads)
        self.split_kv_heads = Rearrange(split_heads_eq, h=key_value_heads)

        self.kv_sets = kv_sets
        self.to_qkv = Linear(dim, sum(self.split_dims), bias=False)

        # attention function

        self.use_nth_order_attend = kv_sets > 2
        assert not (causal and self.use_nth_order_attend)

        if not exists(attend):
            attend = (
                naive_two_simplicial_attend if not self.use_nth_order_attend else nth_order_attend
            )

        if causal:
            attend = partial(attend, causal=causal)

        self.attend = attend

        # combine heads out

        self.merge_heads = Rearrange(merge_heads_eq)
        self.combine_heads = Linear(heads * dim_head_values, dim, bias=False)

    def forward(self, tokens):
        tokens = self.prenorm(tokens)

        q, k, v = self.to_qkv(tokens).split(self.split_dims, dim=-1)

        queries = self.split_q_heads(q)
        keys = self.split_kv_heads(k).chunk(self.kv_sets, dim=-1)
        values = self.split_kv_heads(v).chunk(self.kv_sets, dim=-1)

        # maybe qk rmsnorm

        if self.qk_rmsnorm:
            queries = self.q_norm(queries)
            keys = tuple(norm(t) for norm, t in zip(self.k_norms, keys, strict=False))
            values = tuple(norm(t) for norm, t in zip(self.v_norms, values, strict=False))

        # higher order attention

        out = self.attend(queries, keys, values)

        # merge heads and combine with linear out

        out = self.merge_heads(out)
        out = self.combine_heads(out)

        return self.postnorm(out)


# 2-simplicial mha


class TwoSimplicialMHA(HigherOrderAttention):
    def __init__(self, *args, **kwargs):
        assert "number_key_value_sets" not in kwargs

        super().__init__(*args, number_key_value_sets=2, **kwargs)


def build_two_simplicial_mha():
    torch.manual_seed(0)
    return TwoSimplicialMHA(dim=32, dim_head=8, heads=2)


def example_input_two_simplicial_mha():
    torch.manual_seed(0)
    return (torch.randn(1, 6, 32),)


MENAGERIE_ENTRIES = [
    (
        "2-Simplicial Attention (TwoSimplicialMHA)",
        "build_two_simplicial_mha",
        "example_input_two_simplicial_mha",
        2025,
        "vendored",
    ),
]
