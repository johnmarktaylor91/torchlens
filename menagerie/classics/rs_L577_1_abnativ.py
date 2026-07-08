# SOURCE: vendored from oxpig/AbNatiV (pip package "abnativ" 2.0.8) @ PyPI release 2.0.8
#   abnativ/model/abnativ.py    (PositionalEncoding, MHAEncoderBlock, Encoder, Decoder, AbNatiV_Model)
#   abnativ/model/vq.py         (CosineSimCodebook, VectorQuantize -- adapted from
#                                 lucidrains/vector-quantize-pytorch, MIT licenced, credited in-file)
#   abnativ/model/utils.py      (CNN1d padding helpers + VQ helper functions)
"""AbNatiV: VQ-VAE antibody/nanobody "nativeness" scorer (Sormannilab, Ramon &
Sormanni, Nature Machine Intelligence 2023; pip package ``abnativ`` 2.0.8).

The real architecture is a 1D-CNN + multi-head-self-attention encoder/decoder
around a cosine-similarity vector-quantized (VQ-VAE) latent bottleneck, trained
BERT-style (masked-residue reconstruction) on AHo-aligned antibody Fv one-hot
sequences (length 149, 21-letter alphabet incl. gap). This is the real,
unmodified encoder/decoder/VQ architecture from ``abnativ.model.abnativ`` and
``abnativ.model.vq``, vendored verbatim.

The published ``AbNatiV_Model`` is a ``pytorch_lightning.LightningModule``
whose ``__init__``/``forward`` (this file, verbatim) implement the actual
network; everything below ``training_step`` in the real class (PL training
hooks: ``training_step``, ``validation_step``, ``on_validation_epoch_end``,
etc.) is training/analysis-loop-only boilerplate that imports plotting/ANARCI
helpers (``matplotlib``, ``scikit-learn``, ``Bio``) unrelated to the forward
computational graph -- it is dropped here (not part of the architecture) so
this module has zero non-base-lib import-time dependencies. The base class is
swapped from ``pl.LightningModule`` to plain ``nn.Module`` for the same
reason (LightningModule IS an nn.Module; this changes no forward-pass
semantics). Hyperparameters below are the real released ``hparams.yml``
defaults (``d_embedding=768, kernel=8, stride=8, num_heads=8,
num_mha_layers=3, length_seq=149, alphabet_size=21, num_embeddings=512,
embedding_dim_code_book=64``), shrunk only for a tiny-sized trace instance.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.amp import autocast

# ---------------------------------------------------------------------------
# abnativ/model/utils.py (verbatim: helpers used by Encoder/Decoder + VQ)
# ---------------------------------------------------------------------------


def l_out_cnn1d(L_in: int, K: int, S: int, P: int, D: int = 1) -> float:
    """Formula to find the L_out dimension of an input (dim=L_in) in cnn_1d."""
    return (L_in + 2 * P - D * (K - 1) - 1) / S + 1


def find_optimal_cnn1d_padding(L_in: int, K, S: int) -> Tuple[int, int]:
    """Find the minimal padding giving the kernel size K and stride S
    for a CNN1D without losing any piece of information."""
    P = 0
    L_out = l_out_cnn1d(L_in, K, S, P)

    assert L_in >= K, "Kernel size higher than input dimension, the conv1d will not work"

    while not L_out.is_integer() and 2 * P <= S:
        L_out = l_out_cnn1d(L_in, K, S, P)
        P += 1

    if 2 * P >= S:
        P -= 1
    return math.floor(L_out), P


def l_out_cnn1d_transpose(L_in: int, K: int, S: int, P: int, D: int = 1) -> int:
    """Formula to find the L_out dimension of an input (dim=L_in) in cnn_1d."""
    return (L_in - 1) * S - 2 * P + D * (K - 1) + 1


def find_out_padding_cnn1d_transpose(L_obj: int, L_in: int, K: int, S: int, P: int) -> int:
    """Find the minimal output padding giving the kernel size K and stride S
    to add after a CNN1D transpose layer to reach L_obj (objective)."""
    L_out = l_out_cnn1d_transpose(L_in, K, S, P)
    assert L_obj >= L_out, (
        "Make sure the padding is correct, the ouput of the CNN1D transpose is larger than expeceted"
    )
    return L_obj - L_out


# From the enhancing VQ (https://github.com/lucidrains/vector-quantize-pytorch)
# Copyright (c) 2020 Phil Wang (MIT Licenced)


def _vq_exists(val):
    return val is not None


def vq_default(val, d):
    return val if _vq_exists(val) else d


def vq_noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def _vq_log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -_vq_log(-_vq_log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    use_cosine_sim=False,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=vq_noop,
):
    from einops import repeat as _repeat

    num_codebooks, dim, dtype, device = (  # noqa: F841 (device unused; kept verbatim from upstream)
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, _repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    from einops import repeat as _repeat

    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = _repeat(indices, "h b n -> h b n d", d=dim)
    embeds = _repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# ---------------------------------------------------------------------------
# abnativ/model/vq.py (verbatim: CosineSimCodebook, VectorQuantize)
# ---------------------------------------------------------------------------


class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=3,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0.0,
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        # Distributed sync paths from the original are intentionally left
        # unreachable (use_ddp defaults False, matching non-distributed use).
        self.sample_fn = batched_sample_vectors
        self.kmeans_all_reduce_fn = vq_noop
        self.all_reduce_fn = vq_noop

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, "... -> 1 ..."), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, "1 ... -> ...")

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast("cuda", enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, "h ... d -> h (...) d")
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        dist = einsum("h n d, h c d -> h n c", flatten, embed)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, "... -> ... 1")
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, "... -> ... 1"), embed, embed_normalized
            )

            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind))

        return quantize, embed_ind


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        kmeans_init=True,
        kmeans_iters=10,
        sync_kmeans=True,
        threshold_ema_dead_code=3,
        commitment_weight=1.0,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        sample_codebook_temp=0.0,
        sync_codebook=False,
    ):
        super().__init__()
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = vq_default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = CosineSimCodebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp,
        )

        self.codebook_size = codebook_size

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    def forward(self, x):
        shape, device, heads, is_multiheaded, codebook_size = (  # noqa: F841 (unused; kept verbatim from upstream)
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            self.codebook_size,
        )

        x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        detached_inputs = x.detach()
        loss = F.mse_loss(quantize, detached_inputs, reduction="none")
        loss_pbe = torch.mean(loss, dim=(1, 2))  # (batch_size)

        if self.commitment_weight > 0:
            detached_quantize = quantize.detach()
            commit_loss = F.mse_loss(detached_quantize, x, reduction="none")

            loss_pbe = loss_pbe + torch.mean(
                commit_loss * self.commitment_weight, dim=(1, 2)
            )  # (batch_size)

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        quantize_latent = quantize.detach().clone()
        quantize = self.project_out(quantize)

        avg_probs = torch.mean(
            F.one_hot(embed_ind, self.codebook_size)
            .type(torch.float32)
            .view((-1, self.codebook_size)),
            0,
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantize_projected_in": x,  # (batch_size, l_r, codebook_dim)
            "quantize_latent": quantize_latent,  # (batch_size, l_r, codebook_dim)
            "quantize_projected_out": quantize,  # (batch_size, l_r, dim)
            "loss_vq_commit_pbe": loss_pbe,  # (batch_size)
            "perplexity": perplexity,  # (batch_size)
            "encoding_indices": embed_ind,  # (batch_size, l_r)
        }


# ---------------------------------------------------------------------------
# abnativ/model/abnativ.py (verbatim: PositionalEncoding, MHAEncoderBlock,
# Encoder, Decoder, AbNatiV_Model.__init__/forward + loss helpers)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        """x: Tensor, shape [batch_size, input_seq_len, d_embedding]"""
        x = x + self.pe[: x.size(1)]
        return x


class MHAEncoderBlock(nn.Module):
    def __init__(self, d_embedding, num_heads, d_ff, dropout):
        super().__init__()

        self.self_MHA = torch.nn.MultiheadAttention(d_embedding, num_heads, batch_first=True)

        self.MLperceptron = nn.Sequential(
            nn.Linear(d_embedding, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_embedding),
        )

        self.layernorm1 = nn.LayerNorm(d_embedding, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_embedding, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: Tensor, shape [batch_size, input_seq_len, d_embedding]"""
        attn_output, attn_output_weights = self.self_MHA(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        linear_output = self.MLperceptron(x)
        x = x + self.dropout(linear_output)
        x = self.layernorm2(x)

        return x, attn_output_weights


class Encoder(nn.Module):
    def __init__(
        self,
        d_embedding,
        kernel,
        stride,
        num_heads,
        num_mha_layers,
        d_ff,
        length_seq,
        alphabet_size,
        dropout=0,
    ):
        super().__init__()

        self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
        self.cnn_embedding = nn.Sequential(
            Rearrange("b l r -> b r l"),
            nn.Conv1d(
                alphabet_size, d_embedding, kernel_size=kernel, stride=stride, padding=self.padding
            ),
            Rearrange("b r l -> b l r"),
        )

        self.en_pos_encoding = PositionalEncoding(d_embedding, max_len=self.l_red)
        self.en_dropout = nn.Dropout(dropout)

        self.en_MHA_blocks = nn.ModuleList(
            [MHAEncoderBlock(d_embedding, num_heads, d_ff, dropout) for _ in range(num_mha_layers)]
        )

    def forward(self, x) -> torch.Tensor:
        """x: Tensor, shape [batch_size, input_seq_len, alphabet_size]"""
        h = self.cnn_embedding(x)  # (batch_size, l_red, d_embedding)

        h = self.en_pos_encoding(h)
        h = self.en_dropout(h)

        for i, _l in enumerate(self.en_MHA_blocks):
            h, attn_enc_weights = self.en_MHA_blocks[i](h)  # (batch_size, l_red, d_embedding)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        d_embedding,
        kernel,
        stride,
        num_heads,
        num_mha_layers,
        d_ff,
        length_seq,
        alphabet_size,
        dropout=0,
    ):
        super().__init__()

        self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
        self.de_pos_encoding = PositionalEncoding(d_embedding, max_len=self.l_red)
        self.de_dropout = nn.Dropout(dropout)

        self.de_MHA_blocks = nn.ModuleList(
            [MHAEncoderBlock(d_embedding, num_heads, d_ff, dropout) for _ in range(num_mha_layers)]
        )

        self.dense_to_alphabet = nn.Linear(d_embedding, alphabet_size)
        self.dense_reconstruction = nn.Linear(
            alphabet_size * self.l_red, length_seq * alphabet_size
        )

        self.out_pad = find_out_padding_cnn1d_transpose(
            L_obj=length_seq, L_in=self.l_red, K=kernel, S=stride, P=self.padding
        )
        self.cnn_reconstruction = nn.Sequential(
            Rearrange("b l r -> b r l"),
            nn.ConvTranspose1d(
                d_embedding,
                alphabet_size,
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                output_padding=self.out_pad,
            ),
            Rearrange("b r l -> b l r"),
        )

    def forward(self, q) -> torch.Tensor:
        """q: Tensor, shape [batch_size, l_red, d_embedding]"""
        z = self.de_pos_encoding(q)
        z = self.de_dropout(z)

        for i, _l in enumerate(self.de_MHA_blocks):
            z, attn_dec_weights = self.de_MHA_blocks[i](z)  # (batch_size, l_red, d_embedding)

        z = self.cnn_reconstruction(z)  # (batch_size, input_seq_len, alphabet_size)
        z_recon = F.softmax(z, dim=-1)

        return z_recon


class AbNatiV_Model(nn.Module):
    """Real AbNatiV1 VQ-VAE (``abnativ.model.abnativ.AbNatiV_Model``), with
    the PL training/analysis-loop methods dropped (see module docstring) and
    the base class swapped from ``pl.LightningModule`` to ``nn.Module``
    (both are ``nn.Module`` subclasses; forward-pass math is unchanged)."""

    def __init__(self, hparams: dict):
        super().__init__()

        self.run_name = hparams["run_name"]

        self.encoder = Encoder(
            hparams["d_embedding"],
            hparams["kernel"],
            hparams["stride"],
            hparams["num_heads"],
            hparams["num_mha_layers"],
            hparams["d_ff"],
            hparams["length_seq"],
            hparams["alphabet_size"],
            dropout=hparams["drop"],
        )

        self.decoder = Decoder(
            hparams["d_embedding"],
            hparams["kernel"],
            hparams["stride"],
            hparams["num_heads"],
            hparams["num_mha_layers"],
            hparams["d_ff"],
            hparams["length_seq"],
            hparams["alphabet_size"],
            dropout=hparams["drop"],
        )

        self.vqvae = VectorQuantize(
            dim=hparams["d_embedding"],
            codebook_size=hparams["num_embeddings"],
            codebook_dim=hparams["embedding_dim_code_book"],
            decay=hparams["decay"],
            kmeans_init=True,
            commitment_weight=hparams["commitment_cost"],
        )

        self.learning_rate = hparams["learning_rate"]

        self.loss = hparams.get("loss", "mse")
        self.gamma = hparams.get("gamma", 1)
        self.lambda_ = hparams.get("lambda", 1)

        self.batch_size = hparams["batch_size"]

    def forward(self, data) -> dict:
        inputs = data[:][0][:][:]
        m_inputs = data[:][1][:][:]

        x = self.encoder(m_inputs)
        vq_outputs = self.vqvae(x)
        x_recon = self.decoder(vq_outputs["quantize_projected_out"])

        recon_error_pposi = self.calculate_recon_error_pposi(inputs, x_recon)
        recon_error_pbe = torch.mean(recon_error_pposi, dim=1)
        lambda_vq_output = self.lambda_ * vq_outputs["loss_vq_commit_pbe"]
        loss_pbe = torch.add(recon_error_pbe, lambda_vq_output)

        return {
            "inputs": inputs,  # (batch_size, input_seq_len, alphabet_size)
            "x_recon": x_recon,  # (batch_size, input_seq_len, alphabet_size)
            "recon_error_pposi": recon_error_pposi,  # (batch_size, input_seq_len)
            "recon_error_pbe": recon_error_pbe,  # (batch_size)
            "loss_pbe": loss_pbe,  # (batch_size)
            **vq_outputs,
        }

    def calculate_MSE_recon_error_pposi(
        self, inputs, x_recon, conservation_index=False, focal=False, alpha=False
    ):
        recon_error_pres_pposi = F.mse_loss(x_recon, inputs, reduction="none")
        recon_error_pposi = torch.mean(recon_error_pres_pposi, dim=-1)
        return recon_error_pposi

    def calculate_CE_recon_error_pposi(
        self, inputs, x_recon, conservation_index=False, focal=False
    ):
        x_recon = x_recon.reshape(x_recon.shape[0], x_recon.shape[2], x_recon.shape[1])
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[1])
        recon_error_pposi = F.cross_entropy(x_recon, inputs, reduction="none")
        return recon_error_pposi

    def calculate_recon_error_pposi(self, inputs, x_recon):
        if self.loss == "mse":
            recon_error_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon)
        elif self.loss == "ce":
            recon_error_pposi = self.calculate_CE_recon_error_pposi(inputs, x_recon)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss}")

        return recon_error_pposi


# ---------------------------------------------------------------------------
# Tiny-scale staging wrapper (torchlens capture needs a single forward()
# call with a concrete example input; the real forward() takes a
# ``data[:][0]``/``data[:][1]`` masked-vs-unmasked one-hot pair produced by
# the BERT-style masking dataloader. We supply that pair directly as a
# 2-tuple, matching real usage, and expose only the reconstruction tensor.)
# ---------------------------------------------------------------------------

# Real released hparams.yml defaults (oxpig/AbNatiV `abnativ_v1` run),
# shrunk only in width/depth for a tiny trace instance; length_seq=149 and
# alphabet_size=21 are architecture-defining (AHo-aligned Fv one-hot) and
# kept faithful to the real config.
_LENGTH_SEQ = 149
_ALPHABET_SIZE = 21


class AbNatiVTraceWrapper(nn.Module):
    """Wraps AbNatiV_Model.forward with a fixed example input so TorchLens
    can capture a plain single-tensor-in forward pass."""

    def __init__(self, model: AbNatiV_Model):
        super().__init__()
        self.model = model

    def forward(self, onehot_seq: torch.Tensor) -> torch.Tensor:
        # Real dataloader (data_loader_masking_bert_onehot_fasta) yields
        # (unmasked_onehot, masked_onehot) pairs; here the same tensor
        # stands in for both (no residues masked) to keep a single concrete
        # tensor input for tracing.
        data = (onehot_seq, onehot_seq)
        out = self.model(data)
        return out["x_recon"]


def build_abnativ() -> AbNatiVTraceWrapper:
    hparams = {
        "run_name": "abnativ_v1_tiny",
        "alphabet_size": _ALPHABET_SIZE,
        "batch_size": 2,
        "commitment_cost": 2,
        "d_embedding": 16,  # real default 768, shrunk for tracing
        "d_ff": 8,  # real default 128, shrunk for tracing
        "decay": 0.90,
        "drop": 0,
        "embedding_dim_code_book": 4,  # real default 64, shrunk for tracing
        "kernel": 8,
        "learning_rate": 4.0e-05,
        "length_seq": _LENGTH_SEQ,
        "num_embeddings": 16,  # real default 512, shrunk for tracing
        "num_heads": 2,  # real default 8, shrunk for tracing (must divide d_embedding)
        "num_mha_layers": 1,  # real default 3, shrunk for tracing
        "stride": 8,
        "loss": "mse",
    }
    model = AbNatiV_Model(hparams)
    model.eval()
    return AbNatiVTraceWrapper(model)


def example_input_abnativ() -> torch.Tensor:
    onehot = F.one_hot(
        torch.randint(0, _ALPHABET_SIZE, (2, _LENGTH_SEQ)), num_classes=_ALPHABET_SIZE
    )
    return onehot.float()


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "AbNatiV",
        "build_abnativ",
        "example_input_abnativ",
        2023,
        "vendored-pytorch",
    ),
]
