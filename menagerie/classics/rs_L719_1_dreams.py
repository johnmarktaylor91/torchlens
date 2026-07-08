# SOURCE: vendored from pluskal-lab/DreaMS @ main (dreams/models/dreams/dreams.py,
#         dreams/models/dreams/layers.py, dreams/models/layers/fourier_features.py,
#         dreams/models/layers/feed_forward.py)
#
# DreaMS (Bushuiev et al., Nature Biotechnology 2025, "Emergence of molecular structures from
# repository-scale self-supervised learning on tandem mass spectra") is a self-supervised
# Transformer for tandem (MS/MS) mass spectra. Each spectrum peak (m/z, intensity) is lifted to
# a token via a small feed-forward "peak" embedding, optionally concatenated with learned
# Fourier features of the m/z value, then processed by a stack of custom pre-norm Transformer
# encoder layers (a from-scratch multi-head self-attention + feed-forward block with its own
# fused QKV projection and ScaleNorm/LayerNorm option) to produce per-peak embeddings, with the
# first ("precursor") token used as the whole-spectrum embedding. Vendored verbatim
# (architecture-relevant classes/functions only; the encoder internals in
# dreams/models/dreams/layers.py -- MultiheadAttention, FeedForward, ScaleNorm,
# TransformerEncoder -- and the two feature-embedding layers in
# dreams/models/layers/{fourier_features,feed_forward}.py). Non-architectural changes: the
# real `DreaMS` class is a `pytorch_lightning.LightningModule` configured from a large
# argparse `Namespace` and additionally carries training/validation-loop methods (`step`,
# `training_step`, `configure_optimizers`, NIST20/MoNA retrieval-validation callbacks, etc.)
# that read training-only files from disk and are not part of the forward-pass architecture.
# This vendored copy keeps only `__init__` (layer construction, using the paper's default
# hyperparameters as plain kwargs instead of an argparse Namespace) and `forward` (the encoder
# forward pass), matching the "get embeddings from the last Transformer encoder layer" path
# used at inference/embedding time in the real repo's `get_embeddings()` helper. The
# `graphormer_mz_diffs` / `d_mz_token` / `vanilla_transformer` branches are kept intact but
# constructed with the model's default-off settings (Fourier-feature path, as used in the
# released pretrained checkpoints).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# --- dreams/models/layers/fourier_features.py (verbatim) ---
class FourierFeatures(nn.Module):
    def __init__(
        self, strategy, x_min, x_max, trainable=True, funcs="both", sigma=10, num_freqs=512
    ):
        assert strategy in {"random", "voronov_et_al", "lin_float_int"}
        assert funcs in {"both", "sin", "cos"}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == "random":
            self.b = torch.randn(num_freqs) * sigma
        if self.strategy == "voronov_et_al":
            self.b = torch.tensor(
                [
                    1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2)))
                    for i in range(1, num_freqs)
                ],
            )
        elif self.strategy == "lin_float_int":
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)]
                + [1 / (1 * i) for i in range(2, ceil(x_max), 1)],
            )
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter("Fourier frequencies", self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == "both":
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == "cos":
            x = torch.cos(x)
        elif self.funcs == "sin":
            x = torch.sin(x)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != "both" else 2 * self.b.shape[1]


# --- dreams/models/layers/feed_forward.py (verbatim, `hidden_dim` restricted to int here since
#     the "interpolated" branch pulls in `dreams.utils.misc.interpolate_interval`, a non-
#     architectural training-utility helper not vendored) ---
class PeakFeedForward(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        depth=None,
        act_last=True,
        act=nn.ReLU,
        bias=True,
        dropout=0,
    ):
        super().__init__()

        assert isinstance(hidden_dim, int) and depth is not None
        hidden_dim = [hidden_dim] * depth

        self.ff = nn.ModuleList([])
        for l in range(depth):  # noqa: E741 (verbatim upstream variable name)
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)


# --- dreams/models/dreams/layers.py (verbatim) ---
class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module
    I learned a lot from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
    """

    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = args.att_dropout
        self.use_transformer_bias = not args.no_transformer_bias
        self.attn_mech = args.attn_mech
        self.d_graphormer_params = args.d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError("Required: d_model % n_heads == 0.")

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim**-0.5

        # Parameters for linear projections of queries, keys, values and output
        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        # initializing
        # If we do Xavier normal initialization, std = sqrt(2/(2D))
        # but it's too big and causes unstability in PostNorm
        # so we use the smaller std of feedforward module, i.e. sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.0)

        if self.attn_mech == "additive_v":
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        bs, n, d = q.size()

        def _split_heads(tensor):
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        if self.attn_mech == "dot-product":
            att_weights = torch.einsum("bhnd,bhdm->bhnm", q, k.transpose(-2, -1))
        elif self.attn_mech == "additive_v" or self.attn_mech == "additive_fixed":
            att_weights = q.unsqueeze(-2) - k.unsqueeze(-3)
            if self.attn_mech == "additive_v":
                att_weights = att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = (
            output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        )
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)


class FeedForward(nn.Module):
    """FeedForward"""

    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        self.use_transformer_bias = not args.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    """Self-attention Transformer Encoder"""

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.residual_dropout = args.residual_dropout
        self.n_layers = args.n_layers
        self.pre_norm = args.pre_norm
        self._gradient_checkpointing = False

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.d_model**0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def _layer_forward(self, i, x, src_mask, graphormer_dists):
        pre_norm = self.pre_norm
        post_norm = not pre_norm
        att = self.atts[i]
        ff = self.ffs[i]
        att_scale = self.scales[2 * i]
        ff_scale = self.scales[2 * i + 1]

        residual = x
        x = att_scale(x) if pre_norm else x
        x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = att_scale(x) if post_norm else x

        residual = x
        x = ff_scale(x) if pre_norm else x
        x = ff(x)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = ff_scale(x) if post_norm else x
        return x

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            x = self._layer_forward(i, x, src_mask, graphormer_dists)

        x = self.scales[-1](x) if self.pre_norm else x
        return x


class _DreaMSArgs:
    """Plain container standing in for the real repo's argparse Namespace (constructor-only
    hyperparameters; the real training script builds this via argparse)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# --- dreams/models/dreams/dreams.py (architecture-relevant subset: __init__ + forward) ---
class DreaMS(nn.Module):
    def __init__(self, args: _DreaMSArgs):
        super().__init__()

        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.train_objective = args.train_objective
        self.charge_feature = args.charge_feature
        self.d_fourier = args.d_fourier
        self.d_peak = args.d_peak
        self.d_mz_token = args.d_mz_token
        self.d_model = sum(d for d in [self.d_fourier, self.d_peak, self.d_mz_token] if d)
        args.d_model = self.d_model
        self.hot_mz_bin_size = args.hot_mz_bin_size
        self.vanilla_transformer = args.vanilla_transformer
        self.fourier_strategy = args.fourier_strategy
        self.mask_val = args.mask_val
        self.max_mz = args.max_mz
        if self.charge_feature is None:
            self.charge_feature = False
        if args.graphormer_mz_diffs and args.graphormer_parametrized:
            args.d_graphormer_params = args.d_fourier if args.d_fourier else 1
        else:
            args.d_graphormer_params = 0
        self.graphormer_mz_diffs = args.graphormer_mz_diffs

        token_dim = 2
        if args.charge_feature:
            token_dim += 1

        # Fourier features encoding (for m/z's only)
        if self.d_fourier:
            self.fourier_enc = FourierFeatures(
                strategy=args.fourier_strategy,
                num_freqs=args.fourier_num_freqs,
                x_min=args.fourier_min_freq,
                x_max=args.max_mz,
                trainable=args.fourier_trainable,
            )

            self.ff_fourier = PeakFeedForward(
                in_dim=self.fourier_enc.num_features(),
                out_dim=args.d_fourier,
                dropout=args.dropout,
                depth=args.ff_fourier_depth,
                hidden_dim=args.ff_fourier_d,
                bias=not args.no_ffs_bias,
            )
        elif self.d_mz_token:
            self.mz_tokenizer = nn.Embedding(
                num_embeddings=1
                + num_hot_classes(max_val=args.max_mz, bin_size=args.hot_mz_bin_size),
                embedding_dim=self.d_mz_token,
                padding_idx=0,
            )
            self.ff_mz_token = PeakFeedForward(
                in_dim=self.d_mz_token,
                hidden_dim=self.d_mz_token,
                out_dim=self.d_mz_token,
                depth=2,
                dropout=args.dropout,
            )

        # Input position-wise feed forward: (bs, peaks_n, token_dim) -> (bs, peaks_n, d_peak)
        self.ff_peak = PeakFeedForward(
            in_dim=token_dim,
            hidden_dim=args.d_peak,
            out_dim=args.d_peak,
            depth=args.ff_peak_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias,
        )

        # Stack of the Transformer encoder layers (i.e. BERT)
        if args.vanilla_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                dim_feedforward=self.d_model * 4,
                nhead=self.n_heads,
                activation="gelu",
                dropout=args.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.n_layers
            )
        else:
            self.transformer_encoder = TransformerEncoder(args)

    def forward(self, spec, charge=None):
        """Returns embeddings from the last Transformer encoder layer."""

        # Generate padding mask
        padding_mask = spec[:, :, 0] == 0

        # Append charge to each token
        if self.charge_feature:
            if charge is None:
                raise ValueError
            charge_features = ~padding_mask * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)

        # Lift peaks to d_peak (m/z's are normalized)
        peak_embs = self.ff_peak(self.__normalize_spec(spec))

        # Concatenate with fourier features
        if self.d_fourier:
            fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
            spec = torch.cat([peak_embs, fourier_features], dim=-1)
        elif self.d_mz_token:
            tokenized_mzs = self.mz_tokenizer(
                to_classes(
                    spec[..., [0]],
                    max_val=self.max_mz,
                    bin_size=self.hot_mz_bin_size,
                    special_vals=[self.mask_val],
                ).squeeze()
            )
            tokenized_mzs = self.ff_mz_token(tokenized_mzs)
            spec = torch.cat([peak_embs, tokenized_mzs], dim=-1)
        else:
            spec = peak_embs

        graphormer_dists = None
        if self.graphormer_mz_diffs:
            if self.d_fourier:
                graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            else:
                graphormer_dists = spec[..., 0].unsqueeze(2) - spec[..., 0].unsqueeze(1)
                graphormer_dists = graphormer_dists.unsqueeze(-1)

        # Transformer encoder blocks
        if self.vanilla_transformer:
            spec = self.transformer_encoder(spec, src_key_padding_mask=padding_mask)
        else:
            spec = self.transformer_encoder(spec, padding_mask, graphormer_dists)

        return spec

    def __normalize_spec(self, spec):
        return spec / torch.tensor([self.max_mz, 1.0], device=spec.device, dtype=spec.dtype)


# --- dreams/utils/spectra.py (verbatim helper, only the piece `to_classes` needs) ---
def num_hot_classes(max_val: float, bin_size: float) -> int:
    num_classes = max_val / bin_size
    assert num_classes == int(num_classes)
    return int(num_classes)


def to_classes(vals, max_val: float, bin_size: float, special_vals=()):
    special_masks = [vals == v for v in special_vals]
    n_classes = num_hot_classes(max_val, bin_size)
    classes = torch.round(vals / bin_size).long()
    classes = classes.clamp(max=n_classes - 1)
    for i, m in enumerate(special_masks):
        classes[m] = n_classes + i
    return classes


def build_dreams():
    torch.manual_seed(0)
    args = _DreaMSArgs(
        n_layers=2,
        n_heads=2,
        train_objective="mask_mz",
        charge_feature=False,
        d_fourier=16,
        d_peak=16,
        d_mz_token=0,
        max_mz=1000.0,
        hot_mz_bin_size=0.05,
        vanilla_transformer=False,
        fourier_strategy="lin_float_int",
        fourier_num_freqs=16,
        fourier_min_freq=0.001,
        fourier_trainable=True,
        dropout=0.0,
        ff_fourier_depth=1,
        ff_fourier_d=16,
        no_ffs_bias=False,
        ff_peak_depth=1,
        mask_val=-1.0,
        graphormer_mz_diffs=False,
        graphormer_parametrized=False,
        att_dropout=0.0,
        no_transformer_bias=False,
        attn_mech="dot-product",
        residual_dropout=0.0,
        pre_norm=True,
        scnorm=False,
        ff_dropout=0.0,
    )
    return DreaMS(args)


def example_input_dreams():
    # `spec` is a batch of peak lists: (batch, n_peaks, 2) with columns (m/z, intensity);
    # rows of all-zero m/z are the right-padding the real DreaMS dataloader emits for variable
    # peak counts (detected via `spec[:, :, 0] == 0` inside forward to build the padding mask).
    torch.manual_seed(0)
    batch, n_peaks = 2, 8
    mz = torch.rand(batch, n_peaks, 1) * 500.0 + 1.0
    intensity = torch.rand(batch, n_peaks, 1)
    spec = torch.cat([mz, intensity], dim=-1)
    spec[:, -2:, :] = 0.0  # trailing padding peaks
    return (spec,)


MENAGERIE_ENTRIES = [
    ("DreaMS", "build_dreams", "example_input_dreams", 2025, "vendored-pytorch"),
]
