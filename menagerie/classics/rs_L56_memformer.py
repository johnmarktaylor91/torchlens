# SOURCE: vendored from lucidrains/memformer @ c8f9a97f8b3fc7a90c21f5d3ae911499bf13e50a
# https://raw.githubusercontent.com/lucidrains/memformer/c8f9a97f8b3fc7a90c21f5d3ae911499bf13e50a/memformer/memformer.py
# https://raw.githubusercontent.com/lucidrains/memformer/c8f9a97f8b3fc7a90c21f5d3ae911499bf13e50a/memformer/autoregressive_wrapper.py
#
# lucidrains' implementation of Wu et al. 2020 "Memformer: A Memory-Augmented
# Transformer for Sequence Modeling" -- a Transformer encoder-decoder with a
# fixed-size bank of learned "memory slots" (`Memformer.memory_slots`) that are
# cross-attended into the encoder at every layer and then updated after each
# forward pass via write-attention (`mem_updater`, an `Attention` module) fused
# through a `nn.GRUCell` gate and a residual feedforward ("memory slot attention"
# + "forget/update gate" from the paper). This is a genuine architectural
# contribution (relative-position-biased self/cross attention blocks +
# GRU-gated external memory read/write cycle), so it is vendored verbatim, not
# constructed from an unmodified base-library class.
#
# `memformer/memformer.py` and `memformer/autoregressive_wrapper.py` are the
# complete, unmodified model-definition files; they import only `torch`,
# `torch.nn.functional`, `torch.nn.utils.rnn.pad_sequence`, and `einops`
# (`rearrange`, `repeat`), all base libs already installed. No architectural
# changes were made; only mechanical fixes for import isolation:
#   - `from memformer.autoregressive_wrapper import AutoregressiveWrapper` ->
#     both files' contents are concatenated into this single module so the
#     package-relative import isn't needed.

import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# memformer/autoregressive_wrapper.py (verbatim)
# ---------------------------------------------------------------------------


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        **kwargs,
    ):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop("input_mask", None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            input_mask = input_mask[:, -self.max_seq_len :]

            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss=False, **kwargs):
        pad = partial(pad_sequence, batch_first=True, padding_value=self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]

            # help auto-solve an area of confusion around input masks in auto-regressive
            # if user supplies a mask that is only off by one from the source sequence, resolve it for them
            mask = kwargs.pop("src_mask", None)
            if mask is not None and mask.shape[1] == x.shape[1]:
                mask = mask[:, :-1]
                kwargs.update(src_mask=mask)
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        out = self.net(xi, **kwargs)

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
        return loss


# ---------------------------------------------------------------------------
# memformer/memformer.py (verbatim)
# ---------------------------------------------------------------------------

# constants

Results = namedtuple("Results", ["enc_out", "mem", "dec_out"])
EncOnlyResults = namedtuple("EncOnlyResults", ["enc_out", "mem"])

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key, None), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def group_by_key_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# positional embedding


class RelativePositionBias(nn.Module):
    def __init__(self, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen, klen):
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos, causal=self.causal, num_buckets=self.num_buckets
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> () h i j")


# main classes


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, causal=False, rel_pos_emb=False):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by number of heads"
        dim_head = dim // heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        context=None,
        pos_emb=None,
        mask=None,
        query_mask=None,
        kv_mask=None,
        attend_self=False,
    ):
        b, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device

        if attend_self:
            kv_input = torch.cat((x, context), dim=1)
        else:
            kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, *kv))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if exists(pos_emb):
            pos_emb_bias = pos_emb(*dots.shape[-2:])
            dots += pos_emb_bias

        mask_value = max_neg_value(dots)

        if self.causal:
            causal_mask = torch.ones((n, n), device=device).triu_(1).bool()
            dots.masked_fill_(causal_mask, mask_value)
            del causal_mask

        if any(map(exists, (query_mask, kv_mask))):
            query_mask = default(query_mask, lambda: torch.ones((b, n), device=device).bool())

            if exists(context):
                kv_mask = default(
                    kv_mask, lambda: torch.ones((b, context.shape[1]), device=device).bool()
                )
            else:
                kv_mask = default(kv_mask, query_mask)

            query_mask = rearrange(query_mask, "b i -> b () i ()")
            kv_mask = rearrange(kv_mask, "b j -> b () () j")
            seq_mask = query_mask * kv_mask
            dots.masked_fill_(~seq_mask, mask_value)
            del seq_mask

        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads=8):
        super().__init__()
        self.rel_pos_emb = RelativePositionBias(heads=heads)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, rel_pos_emb=True))),
                        Residual(PreNorm(dim, Attention(dim, heads=heads))),
                        Residual(PreNorm(dim, FeedForward(dim))),
                    ]
                )
            )

    def forward(self, x, context=None, src_mask=None):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, pos_emb=self.rel_pos_emb, query_mask=src_mask)
            x = cross_attn(x, context=context)
            x = ff(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads=8):
        super().__init__()
        self.rel_pos_emb = RelativePositionBias(heads=heads, causal=True)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(dim, Attention(dim, heads=heads, causal=True, rel_pos_emb=True))
                        ),
                        Residual(PreNorm(dim, Attention(dim, heads=heads))),
                        Residual(PreNorm(dim, FeedForward(dim))),
                    ]
                )
            )

    def forward(self, x, context=None, src_mask=None, tgt_mask=None):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, pos_emb=self.rel_pos_emb, query_mask=src_mask)
            x = cross_attn(x, context=context, query_mask=src_mask, kv_mask=tgt_mask)
            x = ff(x)
        return x


class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads=8, return_logits=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len
        self.layer_blocks = layer_blocks
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device  # noqa: F841 (verbatim upstream; kept unused for fidelity)
        x = self.token_emb(x)
        x = self.layer_blocks(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)


class Memformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_memory_slots,
        num_mem_updates=1,
        encoder_only=False,
        mem_update_attn_heads=8,
        **kwargs,
    ):
        super().__init__()
        enc_kwargs, kwargs = group_by_key_prefix_and_trim("enc_", kwargs)
        dec_kwargs, kwargs = group_by_key_prefix_and_trim("dec_", kwargs)
        assert "dim" not in enc_kwargs and "dim" not in dec_kwargs, (
            "dimension of either encoder or decoder must be set with `dim` keyword"
        )
        enc_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], enc_kwargs)
        dec_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], dec_kwargs)

        self.encoder = TransformerWrapper(
            dim=dim,
            layer_blocks=Encoder(dim=dim, **enc_kwargs),
            return_logits=False,
            **enc_transformer_kwargs,
        )

        self.decoder = (
            TransformerWrapper(
                dim=dim,
                layer_blocks=Decoder(dim=dim, **dec_kwargs),
                return_logits=True,
                **dec_transformer_kwargs,
            )
            if not encoder_only
            else None
        )

        if exists(self.decoder):
            self.decoder = AutoregressiveWrapper(self.decoder)

        self.num_mem = num_memory_slots
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, dim))

        self.num_mem_updates = num_mem_updates
        self.mem_updater = Attention(dim, heads=mem_update_attn_heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mem_ff = Residual(PreNorm(dim, FeedForward(dim)))

    def get_initial_mem(self, batch_size):
        return repeat(self.memory_slots, "n d -> b n d", b=batch_size)

    def forward(self, src, tgt=None, mems=None, src_mask=None, tgt_mask=None):
        b, n, num_mem, device = *src.shape, self.num_mem, src.device
        mems = default(mems, lambda: self.get_initial_mem(b))

        enc = self.encoder(src, context=mems, src_mask=src_mask)

        if exists(self.decoder) and exists(tgt):
            dec_out = self.decoder(
                tgt, context=enc, src_mask=tgt_mask, tgt_mask=src_mask, return_loss=True
            )
        else:
            dec_out = torch.tensor(0.0, requires_grad=True, device=device)

        # update memory with attention
        mem_mask = torch.eye(num_mem, num_mem, device=device).bool()
        mem_mask = repeat(mem_mask, "i j -> b i j", b=b)
        mem_mask = F.pad(mem_mask, (0, n), value=True)

        if exists(src_mask):
            src_mask = rearrange(src_mask, "b j -> b () j")
            mem_enc_mask = F.pad(src_mask, (num_mem, 0), value=True)
            mem_mask &= mem_enc_mask

        for _ in range(self.num_mem_updates):
            prev_mems = mems
            updated_mems = self.mem_updater(mems, enc, mask=mem_mask, attend_self=True)

            next_mems = self.gru(
                rearrange(updated_mems, "b n d -> (b n) d"),
                rearrange(prev_mems, "b n d -> (b n) d"),
            )

            mems = rearrange(next_mems, "(b n) d -> b n d", b=b)
            mems = self.mem_ff(mems)

        if not exists(self.decoder):
            return EncOnlyResults(enc, mems)

        return Results(enc, mems, dec_out)


def build_memformer():
    return Memformer(
        dim=32,
        num_memory_slots=4,
        num_mem_updates=2,
        enc_num_tokens=100,
        enc_max_seq_len=16,
        enc_depth=2,
        enc_heads=2,
        dec_num_tokens=100,
        dec_max_seq_len=16,
        dec_depth=2,
        dec_heads=2,
        mem_update_attn_heads=2,
    )


def example_input_memformer():
    batch = 2
    seq_len = 8
    src = torch.randint(0, 100, (batch, seq_len))
    tgt = torch.randint(0, 100, (batch, seq_len))
    return (src, tgt)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Memformer", "build_memformer", "example_input_memformer", 2020, "vendored"),
]
