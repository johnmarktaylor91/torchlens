# FAITHFUL PORT of https://github.com/mit-han-lab/lite-transformer @ master (original
# framework: fairseq, a large custom seq2seq framework not installed / not reasonably
# installable alongside base torch here)
#
# Lite Transformer (Wu, Liu, Lin, Lin, Han, "Lite Transformer with Long-Short Range
# Attention", ICLR 2020): a fairseq encoder-decoder Transformer whose self-attention
# sub-layer is replaced (per layer, optionally) with a `MultiBranch` "Long-Short Range
# Attention" (LSRA) block that splits the embedding dimension across parallel branches --
# some running lightweight/dynamic depthwise convolution (short-range, local context) and
# some running standard multi-head attention (long-range, global context) -- then
# concatenates the branch outputs back into the full embedding width. This module
# transcribes the actual repo code 1:1 (not a paper paraphrase):
#   fairseq/models/transformer_multibranch_v2.py  (`TransformerMultibranchModel`,
#     `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`,
#     `TransformerDecoderLayer`, `base_architecture`)
#   fairseq/modules/multibranch.py                (`MultiBranch`)
#   fairseq/modules/lightweight_convolution.py     (`LightweightConv1dTBC`)
#   fairseq/modules/dynamic_convolution.py         (`DynamicConv1dTBC`)
#   fairseq/modules/unfold.py                      (`unfold1d`)
#   fairseq/modules/multihead_attention.py         (`MultiheadAttention`)
#   fairseq/modules/sinusoidal_positional_embedding.py (`SinusoidalPositionalEmbedding`)
#
# fairseq itself (argparse-driven `--encoder-branch-type`/`--decoder-branch-type` CLI
# config, `register_model`/`register_model_architecture` decorators, `FairseqEncoder`/
# `FairseqIncrementalDecoder` base classes, incremental decoding state dicts, ONNX-export
# branches, ``ninja``/CUDA `dynamicconv_layer`/`lightconv_layer` fused kernels, adaptive
# softmax, ``ratio_xavier_uniform_`` big-model init) is not installed and is far too large a
# framework to vendor for one model, so the architecture is transcribed here as
# self-contained torch: the same `MultiBranch` long-short-range-attention layer, the same
# lightweight/dynamic depthwise-conv branch math (`_forward_expanded`, the "matrix trick"
# banded-matrix formulation used for non-incremental training/inference, which is what a
# single forward pass with no `incremental_state` takes), the same encoder/decoder pre/post
# layernorm placement, and the same sinusoidal positional embeddings. CUDA fused-kernel
# fallback branches, incremental (autoregressive one-step) decoding, adaptive softmax output,
# and ONNX-export bookkeeping are inference/serving/training-optimization concerns absent
# from the core forward architecture and are not ported; `weight_softmax=True` (the repo's
# published default, `base_architecture`) is kept.
#
# Repo: https://github.com/mit-han-lab/lite-transformer @ master

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """Unfold T x B x C to T x B x C x K. Faithful port of fairseq/modules/unfold.py."""
    if kernel_size > 1:
        t, b, c = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((t, b, c, kernel_size), (b * c, c, 1, b * c))
    else:
        x = x.unsqueeze(3)
    return x


class LightweightConv1dTBC(nn.Module):
    """Lightweight (depthwise, shared-per-head) convolution over T x B x C input.

    Faithful port of fairseq/modules/lightweight_convolution.py `LightweightConv1dTBC`
    (the pure-torch fallback path used whenever the fused CUDA `LightconvLayer` kernel is
    unavailable, which is the architecturally-faithful reference implementation).
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding_l=None,
        num_heads=1,
        weight_dropout=0.0,
        weight_softmax=False,
        bias=False,
        with_linear=False,
        out_dim=None,
    ):
        super().__init__()
        self.embed_dim = input_size
        out_dim = input_size if out_dim is None else out_dim
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.linear1 = self.linear2 = None
        if with_linear:
            self.linear1 = linear(input_size, input_size)
            self.linear2 = linear(input_size, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, unfold=False):
        """x: T x B x C -> T x B x C (or T x B x out_dim if with_linear)."""
        if self.linear1 is not None:
            x = self.linear1(x)

        if unfold:
            output = self._forward_unfolded(x)
        else:
            output = self._forward_expanded(x)

        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)

        if self.linear2 is not None:
            output = self.linear2(output)
        return output

    def _forward_unfolded(self, x):
        t, b, c = x.size()
        k, h = self.kernel_size, self.num_heads
        r = c // h
        weight = self.weight.view(h, k)
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
        x_unfold = x_unfold.view(t * b * h, r, k)
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1).type_as(weight)
        weight = weight.view(1, h, k).expand(t * b, h, k).contiguous().view(t * b * h, k, 1)
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        output = torch.bmm(x_unfold, weight)
        output = output.view(t, b, c)
        return output

    def _forward_expanded(self, x):
        """Turn the convolution filters into band matrices and do matrix multiplication.

        Faster for short sequences; this is the path used by a plain (non-incremental)
        forward pass, matching what the model actually runs during a single trace/inference
        call.
        """
        t, b, c = x.size()
        k, h = self.kernel_size, self.num_heads
        r = c // h
        weight = self.weight.view(h, k)
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1).type_as(weight)
        weight = weight.view(1, h, k).expand(t * b, h, k).contiguous()
        weight = weight.view(t, b * h, k).transpose(0, 1)

        x = x.view(t, b * h, r).transpose(0, 1)
        p = self.padding_l
        if k > t and p == k - 1:
            weight = weight.narrow(2, k - t, t)
            k, p = t, t - 1
        weight_expanded = weight.new_zeros(b * h, t, t + k - 1, requires_grad=False)
        weight_expanded.as_strided((b * h, t, k), (t * (t + k - 1), t + k, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, p, t)
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(t, b, c)
        return output


class DynamicConv1dTBC(nn.Module):
    """Dynamic (input-dependent) lightweight convolution over T x B x C input.

    Faithful port of fairseq/modules/dynamic_convolution.py `DynamicConv1dTBC` (pure-torch
    fallback path, the architecturally-faithful reference implementation).
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding_l=None,
        num_heads=1,
        weight_dropout=0.0,
        weight_softmax=False,
        bias=False,
        conv_bias=False,
        query_size=None,
        with_linear=False,
        glu=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax

        self.weight_linear = linear(self.query_size, num_heads * kernel_size, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None

        self.linear1 = self.linear2 = None
        self.act = None
        if with_linear:
            if glu:
                self.linear1 = linear(input_size, input_size * 2)
                self.act = nn.GLU()
            else:
                self.linear1 = linear(input_size, input_size)
            self.linear2 = linear(input_size, input_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.0)

    def forward(self, x, query=None, unfold=None):
        if self.linear1 is not None:
            x = self.linear1(x)
            if self.act is not None:
                x = self.act(x)
        unfold = x.size(0) > 512 if unfold is None else unfold
        if query is None:
            query = x
        if unfold:
            output = self._forward_unfolded(x, query)
        else:
            output = self._forward_expanded(x, query)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        if self.linear2 is not None:
            output = self.linear2(output)
        return output

    def _forward_unfolded(self, x, query):
        t, b, c = x.size()
        k, h = self.kernel_size, self.num_heads
        r = c // h
        weight = self.weight_linear(query).view(t * b * h, -1)

        padding_l = self.padding_l
        if k > t and padding_l == k - 1:
            weight = weight.narrow(1, k - t, t)
            k, padding_l = t, t - 1
        x_unfold = unfold1d(x, k, padding_l, 0)
        x_unfold = x_unfold.view(t * b * h, r, k)

        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, k)
        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        output = torch.bmm(x_unfold, weight.unsqueeze(2))
        output = output.view(t, b, c)
        return output

    def _forward_expanded(self, x, query):
        t, b, c = x.size()
        k, h = self.kernel_size, self.num_heads
        r = c // h
        weight = self.weight_linear(query).view(t * b * h, -1)

        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)
        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        weight = weight.narrow(1, 0, k).contiguous()
        weight = weight.view(t, b * h, k).transpose(0, 1)

        x = x.view(t, b * h, r).transpose(0, 1)
        p = self.padding_l
        if k > t and p == k - 1:
            weight = weight.narrow(2, k - t, t)
            k, p = t, t - 1
        weight_expanded = weight.new_zeros(b * h, t, t + k - 1, requires_grad=False)
        weight_expanded.as_strided((b * h, t, k), (t * (t + k - 1), t + k, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, p, t)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(t, b, c)
        return output


class MultiheadAttention(nn.Module):
    """Standard multi-head attention, faithful port of
    fairseq/modules/multihead_attention.py (separate q/k/v/out projections, T x B x C
    convention). Incremental-decoding cache and ONNX-export branches are dropped (not part
    of the forward architecture for a single non-incremental pass).
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """query/key/value: T x B x C."""
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn


class MultiBranch(nn.Module):
    """Long-Short Range Attention (LSRA): parallel branches over embedding-dim slices,
    concatenated back to full width. Faithful port of fairseq/modules/multibranch.py.
    """

    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        tgt_len, bsz, embed_size = query.size()
        assert sum(self.embed_dim_list) == embed_size
        out = []
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            q = query[..., start : start + embed_dim]
            if key is not None:
                k = key[..., start : start + embed_dim]
                v = value[..., start : start + embed_dim]
            start += embed_dim

            if isinstance(branch, MultiheadAttention):
                x = branch(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                mask = key_padding_mask
                if mask is not None:
                    q = q.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
                x = branch(q.contiguous())
            out.append(x)

        return torch.cat(out, dim=-1)


class SinusoidalPositionalEmbedding(nn.Module):
    """Faithful port of fairseq/modules/sinusoidal_positional_embedding.py (non-incremental
    forward path; padding-aware position indices via `make_positions`).
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx):
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(self, tokens):
        """tokens: B x T (LongTensor) -> B x T x embedding_dim."""
        bsz, seq_len = tokens.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(tokens, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


def linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    with torch.no_grad():
        m.weight[padding_idx].fill_(0)
    return m


def make_branch_layer(layer_type, out_dim, num_heads, kernel_size, args, is_decoder):
    """type:kernel:dim:head -> a lightweight / dynamic / attn branch layer, matching
    `TransformerEncoderLayer.get_layer` / `TransformerDecoderLayer.get_layer`.
    """
    if is_decoder:
        padding_l = kernel_size - 1
    else:
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else (kernel_size - 1) // 2

    if "lightweight" in layer_type:
        return LightweightConv1dTBC(
            out_dim,
            kernel_size,
            padding_l=padding_l,
            num_heads=num_heads,
            weight_softmax=args["weight_softmax"],
            weight_dropout=args["weight_dropout"],
            with_linear=args["conv_linear"],
        )
    elif "dynamic" in layer_type:
        return DynamicConv1dTBC(
            out_dim,
            kernel_size,
            padding_l=padding_l,
            num_heads=num_heads,
            weight_softmax=args["weight_softmax"],
            weight_dropout=args["weight_dropout"],
            with_linear=args["conv_linear"],
            glu=args["glu"],
        )
    elif "attn" in layer_type:
        return MultiheadAttention(out_dim, num_heads, dropout=args["attention_dropout"])
    else:
        raise NotImplementedError(layer_type)


def build_self_attn(branch_type, embed_dim, kernel_size, args, is_decoder):
    """branch_type is either None (plain MultiheadAttention over the full embed_dim) or a
    list of "type:kernel:dim:head" strings (LSRA MultiBranch), matching
    `--encoder-branch-type` / `--decoder-branch-type`.
    """
    if branch_type is None:
        return MultiheadAttention(
            embed_dim, args["attention_heads"], dropout=args["attention_dropout"]
        )

    layers = []
    embed_dims = []
    for spec in branch_type:
        parts = spec.split(":")
        k = parts[1]
        k = kernel_size if k == "default" else int(k)
        embed_dims.append(int(parts[2]))
        heads = int(parts[3])
        layers.append(make_branch_layer(spec, embed_dims[-1], heads, k, args, is_decoder))
    assert sum(embed_dims) == embed_dim
    return MultiBranch(layers, embed_dims)


class TransformerEncoderLayer(nn.Module):
    """Faithful port of `TransformerEncoderLayer` (pre/post layernorm placement per
    `encoder_normalize_before`, LSRA `MultiBranch` self-attn, position-wise FFN).
    """

    def __init__(self, args, index):
        super().__init__()
        embed_dim = args["encoder_embed_dim"]
        self.embed_dim = embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = args["dropout"]
        self.activation_fn = F.relu
        self.activation_dropout = args["activation_dropout"]
        self.normalize_before = args["encoder_normalize_before"]

        kernel_size = args["encoder_kernel_size_list"][index]
        self.self_attn = build_self_attn(
            args["encoder_branch_type"], embed_dim, kernel_size, args, is_decoder=False
        )

        self.fc1 = linear(embed_dim, args["encoder_ffn_embed_dim"])
        self.fc2 = linear(args["encoder_ffn_embed_dim"], embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before != after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x


class TransformerDecoderLayer(nn.Module):
    """Faithful port of `TransformerDecoderLayer` (LSRA self-attn + encoder-decoder cross
    attention + position-wise FFN).
    """

    def __init__(self, args, index):
        super().__init__()
        embed_dim = args["decoder_embed_dim"]
        self.embed_dim = embed_dim
        self.dropout = args["dropout"]
        self.activation_fn = F.relu
        self.activation_dropout = args["activation_dropout"]
        self.normalize_before = args["decoder_normalize_before"]

        kernel_size = args["decoder_kernel_size_list"][index]
        self.self_attn = build_self_attn(
            args["decoder_branch_type"], embed_dim, kernel_size, args, is_decoder=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.encoder_attn = MultiheadAttention(
            embed_dim, args["attention_heads"], dropout=args["attention_dropout"]
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = linear(embed_dim, args["decoder_ffn_embed_dim"])
        self.fc2 = linear(args["decoder_ffn_embed_dim"], embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before != after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x

    def forward(self, x, encoder_out, encoder_padding_mask, self_attn_mask=None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.self_attn(query=x, key=x, value=x, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        x = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x


class LiteTransformerEncoder(nn.Module):
    """Faithful port of `TransformerEncoder`."""

    def __init__(self, args, padding_idx):
        super().__init__()
        embed_dim = args["encoder_embed_dim"]
        self.padding_idx = padding_idx
        self.dropout = args["dropout"]
        self.embed_tokens = embedding(args["src_vocab_size"], embed_dim, padding_idx)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, padding_idx, init_size=args["max_source_positions"] + padding_idx + 1
        )
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args, i) for i in range(args["encoder_layers"])]
        )
        self.layer_norm = nn.LayerNorm(embed_dim) if args["encoder_normalize_before"] else None

    def forward(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # T x B x C

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, encoder_padding_mask


class LiteTransformerDecoder(nn.Module):
    """Faithful port of `TransformerDecoder` (non-incremental / teacher-forcing forward)."""

    def __init__(self, args, padding_idx):
        super().__init__()
        embed_dim = args["decoder_embed_dim"]
        self.padding_idx = padding_idx
        self.dropout = args["dropout"]
        self.embed_tokens = embedding(args["tgt_vocab_size"], embed_dim, padding_idx)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, padding_idx, init_size=args["max_target_positions"] + padding_idx + 1
        )
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(args, i) for i in range(args["decoder_layers"])]
        )
        self.layer_norm = nn.LayerNorm(embed_dim) if args["decoder_normalize_before"] else None
        self.share_input_output_embed = args["share_decoder_input_output_embed"]
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(args["tgt_vocab_size"], embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim**-0.5)

    def buffered_future_mask(self, x):
        dim = x.size(0)
        mask = torch.triu(torch.full((dim, dim), float("-inf"), device=x.device, dtype=x.dtype), 1)
        return mask

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask):
        positions = self.embed_positions(prev_output_tokens)
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # T x B x C

        self_attn_mask = self.buffered_future_mask(x)
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_padding_mask, self_attn_mask=self_attn_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)  # B x T x C
        if self.share_input_output_embed:
            return F.linear(x, self.embed_tokens.weight)
        return F.linear(x, self.embed_out)


class LiteTransformerModel(nn.Module):
    """Faithful port of `TransformerMultibranchModel` (encoder + decoder wrapper)."""

    def __init__(self, args):
        super().__init__()
        padding_idx = 1  # fairseq dictionaries reserve 1 for <pad> by convention
        self.encoder = LiteTransformerEncoder(args, padding_idx)
        self.decoder = LiteTransformerDecoder(args, padding_idx)

    def forward(self, src_tokens, prev_output_tokens):
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        return self.decoder(prev_output_tokens, encoder_out, encoder_padding_mask)


def _lite_transformer_iwslt_args():
    """Mirrors `transformer_iwslt_de_en` + `base_architecture` defaults, but with the LSRA
    `encoder-branch-type` / `decoder-branch-type` actually populated (published IWSLT
    configs, e.g. configs/iwslt14.de-en/attention/multibranch_v2/embed160.yml, set
    3 lightweight-conv branches + 1 attention branch per layer) and shrunk to a tiny size
    for a fast trace; the LSRA branch structure and pre/post-norm placement are unchanged.
    """
    encoder_embed_dim = 32
    decoder_embed_dim = 32
    branch_type = [
        "lightweight:default:8:2",
        "lightweight:default:8:2",
        "dynamic:default:8:2",
        "attn:default:8:2",
    ]
    return {
        "src_vocab_size": 64,
        "tgt_vocab_size": 64,
        "max_source_positions": 32,
        "max_target_positions": 32,
        "encoder_embed_dim": encoder_embed_dim,
        "encoder_ffn_embed_dim": 64,
        "encoder_layers": 2,
        "attention_heads": 4,
        "encoder_normalize_before": False,
        "decoder_embed_dim": decoder_embed_dim,
        "decoder_ffn_embed_dim": 64,
        "decoder_layers": 2,
        "decoder_normalize_before": False,
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "weight_dropout": 0.0,
        "weight_softmax": True,
        "conv_linear": False,
        "glu": True,
        "share_decoder_input_output_embed": True,
        "encoder_branch_type": branch_type,
        "decoder_branch_type": branch_type,
        "encoder_kernel_size_list": [3, 7],
        "decoder_kernel_size_list": [3, 7],
    }


def build_lite_transformer():
    return LiteTransformerModel(_lite_transformer_iwslt_args())


def example_input_lite_transformer():
    batch, src_len, tgt_len, vocab = 2, 6, 5, 64
    src_tokens = torch.randint(2, vocab, (batch, src_len))
    prev_output_tokens = torch.randint(2, vocab, (batch, tgt_len))
    return (src_tokens, prev_output_tokens)


MENAGERIE_ENTRIES = [
    (
        "Lite Transformer (LSRA long-short range attention)",
        build_lite_transformer,
        example_input_lite_transformer,
        2020,
        MENAGERIE_ZOO,
    ),
]
