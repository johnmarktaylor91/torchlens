# SOURCE: vendored from https://github.com/batistagroup/DirectMultiStep @ master
#
# DirectMultiStep (Batista group, "Beam Search for Automated Synthesis Planning with
# Transformer Networks" / DirectMultiStep, JCIM 2025) -- a single-pass sequence-to-sequence
# Transformer that maps a target-molecule + starting-materials product string directly to a
# multi-step synthesis route string (no iterative retrosynthesis search). Vendored verbatim
# (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/batistagroup/DirectMultiStep/master/src/directmultistep/model/architecture.py
#   https://raw.githubusercontent.com/batistagroup/DirectMultiStep/master/src/directmultistep/model/components/encoder.py
#   https://raw.githubusercontent.com/batistagroup/DirectMultiStep/master/src/directmultistep/model/components/decoder.py
#   https://raw.githubusercontent.com/batistagroup/DirectMultiStep/master/src/directmultistep/model/components/attention.py
#   https://raw.githubusercontent.com/batistagroup/DirectMultiStep/master/src/directmultistep/model/components/moe.py
#
# What is kept: Seq2Seq (the top-level encoder/decoder wrapper + src masking),
# MultiHeadAttentionLayer (real scaled_dot_product_attention-based MHA used by every
# encoder/decoder sub-layer), PositionwiseFeedforwardLayer, NoisyTopkRouter + Expert +
# SparseMoE (the real noisy top-k Mixture-of-Experts feedforward block), MoEEncoderLayer /
# MoEEncoder and MoEDecoderLayer / MoEDecoder (the token+positional+step embeddings, the
# per-layer self-attention/cross-attention/MoE-FFN stack) -- every mechanism in the real
# trainable network, transcribed unmodified. This staging build exercises the
# Mixture-of-Experts encoder+decoder variant (`MoEEncoder`/`MoEDecoder`), the architecturally
# richer of the two configurations the repo ships (plain `Encoder`/`Decoder` share the same
# attention/embedding code minus routing).
#
# What is dropped (infra plumbing, not part of the forward-pass computation graph):
# `ModelFactory`/`Seq2SeqConfig` YAML-preset loading, `torch.compile` wrapping, and
# checkpoint I/O -- this staging module constructs the real `Seq2Seq(MoEEncoder, MoEDecoder)`
# directly with a tiny hidden size instead of the published presets (flash_10M / deep_40M /
# explorer_xl_50M / wide_40M), matching the exact constructor signature the repo's own
# `ModelFactory.create_model` uses.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

Tensor = torch.Tensor

activation_dict = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}


# ---------------------------------------------------------------------------
# from model/components/attention.py (verbatim)
# ---------------------------------------------------------------------------
class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.query = nn.Linear(hid_dim, hid_dim, bias=attn_bias)
        self.key = nn.Linear(hid_dim, hid_dim, bias=attn_bias)
        self.value = nn.Linear(hid_dim, hid_dim, bias=attn_bias)

        self.projection = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_BLD: Tensor,
        key_BMD: Tensor,
        value_BMD: Tensor,
        mask_B11M: Tensor | None = None,
    ) -> Tensor:
        B, L, _ = query_BLD.shape
        Q_BLD = self.query(query_BLD)
        K_BMD = self.key(key_BMD)
        V_BMD = self.value(value_BMD)
        Q_BHLD = Q_BLD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K_BHMD = K_BMD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V_BHMD = V_BMD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if mask_B11M is not None:
            mask_BHLM = mask_B11M.expand(B, self.n_heads, L, -1)
            is_causal = False
        else:
            mask_BHLM = None
            is_causal = True

        attn_output_BHLD = nn.functional.scaled_dot_product_attention(
            query=Q_BHLD,
            key=K_BHMD,
            value=V_BHMD,
            attn_mask=mask_BHLM,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output_BLD = attn_output_BHLD.permute(0, 2, 1, 3).contiguous().view(B, L, self.hid_dim)
        output_BLD = cast(Tensor, self.projection(attn_output_BLD))
        return output_BLD


# ---------------------------------------------------------------------------
# from model/components/moe.py (verbatim)
# ---------------------------------------------------------------------------
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        ff_mult: int,
        ff_activation: nn.Module,
        dropout: float,
    ):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, ff_mult * hid_dim)
        self.activ = ff_activation
        self.fc_2 = nn.Linear(hid_dim * ff_mult, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BLD: Tensor) -> Tensor:
        x_BLF = self.dropout(self.activ(self.fc_1(x_BLD)))
        x_BLD = self.fc_2(x_BLF)
        return x_BLD


class NoisyTopkRouter(nn.Module):
    def __init__(self, hid_dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(hid_dim, n_experts)
        self.noise_linear = nn.Linear(hid_dim, n_experts)

    def forward(self, x_BLD: Tensor) -> tuple[Tensor, Tensor]:
        logits_BLE = self.topkroute_linear(x_BLD)
        noise_logits_BLE = self.noise_linear(x_BLD)
        noise_BLE = torch.randn_like(logits_BLE) * F.softplus(noise_logits_BLE)
        noisy_logits_BLE = logits_BLE + noise_BLE

        top_k_logits_BLE, indices_BLK = noisy_logits_BLE.topk(self.top_k, dim=-1)
        zeros_BLE = torch.full_like(noisy_logits_BLE, float("-inf"))
        sparse_logits_BLE = zeros_BLE.scatter(-1, indices_BLK, top_k_logits_BLE)
        router_output_BLE = F.softmax(sparse_logits_BLE, dim=-1)
        return router_output_BLE, indices_BLK


class Expert(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim, ff_mult * hid_dim),
            activation_dict[ff_activation],
            nn.Linear(ff_mult * hid_dim, hid_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_BLD: Tensor) -> Tensor:
        return self.net(x_BLD)  # type: ignore


class SparseMoE(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        capacity_factor: float,
    ):
        super().__init__()
        self.router = NoisyTopkRouter(hid_dim, n_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(hid_dim, ff_mult, ff_activation, dropout) for _ in range(n_experts)]
        )
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def forward(self, x_BLD: Tensor) -> Tensor:
        B, L, _ = x_BLD.shape
        gating_output_BLE, indices_BLK = self.router(x_BLD)
        final_output_BLD = torch.zeros_like(x_BLD)

        flat_x_FD = x_BLD.view(-1, x_BLD.size(-1))
        flat_gating_output_FE = gating_output_BLE.view(-1, gating_output_BLE.size(-1))
        n_tkns = B * L * self.top_k
        capacity = int((n_tkns / self.n_experts) * self.capacity_factor)

        updates_FD = torch.zeros_like(flat_x_FD)
        for i, expert in enumerate(self.experts):
            expert_mask_BL = (indices_BLK == i).any(dim=-1)
            flat_mask_F = expert_mask_BL.view(-1)
            selected_idxs_F = torch.nonzero(flat_mask_F).squeeze(-1)
            limited_idxs_F = (
                selected_idxs_F[:capacity]
                if selected_idxs_F.numel() > capacity
                else selected_idxs_F
            )

            if limited_idxs_F.numel() > 0:
                expert_input_SD = flat_x_FD[limited_idxs_F]
                expert_output_SD = expert(expert_input_SD)

                gating_scores_S1 = flat_gating_output_FE[limited_idxs_F, i].unsqueeze(1)
                weighted_output_SD = expert_output_SD * gating_scores_S1

                updates_FD.index_add_(0, limited_idxs_F, weighted_output_SD)

        final_output_BLD += updates_FD.view(B, L, -1)

        return final_output_BLD


# ---------------------------------------------------------------------------
# from model/components/encoder.py (verbatim; MoE variant)
# ---------------------------------------------------------------------------
class MoEEncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
        capacity_factor: float,
    ):
        super().__init__()

        self.attn_ln = nn.LayerNorm(hid_dim)
        self.ff_ln = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.mlp = SparseMoE(
            hid_dim=hid_dim,
            n_experts=n_experts,
            top_k=top_k,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            dropout=dropout,
            capacity_factor=capacity_factor,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_BCD: Tensor, src_mask_B11C: Tensor) -> Tensor:
        attn_output_BCD = self.attention(input_BCD, input_BCD, input_BCD, src_mask_B11C)
        src_BCD = self.attn_ln(input_BCD + self.dropout(attn_output_BCD))
        ff_out_BCD = self.mlp(src_BCD)
        final_out_BLD = self.ff_ln(src_BCD + self.dropout(ff_out_BCD))
        return cast(Tensor, final_out_BLD)


class MoEEncoder(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
        context_window: int,
        initiate_steps: bool,
        include_steps: bool,
        capacity_factor: float,
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(context_window, hid_dim)
        if initiate_steps:
            self.step_embedding = nn.Embedding(1, hid_dim)
        self.include_steps = include_steps

        self.layers = nn.ModuleList(
            [
                MoEEncoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    n_experts=n_experts,
                    top_k=top_k,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    capacity_factor=capacity_factor,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, src_BC: Tensor, src_mask_B11C: Tensor, steps_B1: Tensor) -> Tensor:
        B, C = src_BC.shape
        tok_emb_BCD = self.tok_embedding(src_BC) * self.scale.to(src_BC)
        pos_BC = torch.arange(0, C).unsqueeze(0).repeat(B, 1).to(src_BC)
        pos_emb_BCD = self.pos_embedding(pos_BC)
        comb_BCD = tok_emb_BCD + pos_emb_BCD
        if self.include_steps:
            step_BC = torch.zeros(C).unsqueeze(0).repeat(B, 1).long().to(src_BC)
            step_emb_BCD = self.step_embedding(step_BC) * steps_B1.view(-1, 1, 1)
            comb_BCD += step_emb_BCD
        src_BCD = self.dropout(comb_BCD)
        for layer in self.layers:
            src_BCD = layer(src_BCD, src_mask_B11C)
        return cast(Tensor, src_BCD)


# ---------------------------------------------------------------------------
# from model/components/decoder.py (verbatim; MoE variant)
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
    ) -> None:
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(hid_dim)
        self.enc_attn_ln = nn.LayerNorm(hid_dim)
        self.ff_ln = nn.LayerNorm(hid_dim)
        self.self_attn = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.encoder_attn = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.mlp: nn.Module = PositionwiseFeedforwardLayer(
            hid_dim=hid_dim,
            ff_mult=ff_mult,
            ff_activation=activation_dict[ff_activation],
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg_BLD: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Tensor,
    ) -> Tensor:
        self_attn_BLD = self.self_attn(trg_BLD, trg_BLD, trg_BLD, trg_mask_B1LL)
        trg_BLD = self.self_attn_ln(trg_BLD + self.dropout(self_attn_BLD))
        enc_attn_BLD = self.encoder_attn(trg_BLD, enc_src_BCD, enc_src_BCD, src_mask_B11C)
        trg_BLD = self.enc_attn_ln(trg_BLD + self.dropout(enc_attn_BLD))
        ff_out_BLD = self.mlp(trg_BLD)
        trg_BLD = self.ff_ln(trg_BLD + self.dropout(ff_out_BLD))
        return trg_BLD


class MoEDecoderLayer(DecoderLayer):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
        n_experts: int,
        top_k: int,
        capacity_factor: float,
    ) -> None:
        super().__init__(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
        )
        self.mlp = SparseMoE(
            hid_dim=hid_dim,
            n_experts=n_experts,
            top_k=top_k,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            dropout=dropout,
            capacity_factor=capacity_factor,
        )


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        context_window: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.tok_embedding = nn.Embedding(vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(context_window, hid_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, vocab_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(
        self,
        trg_BL: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Tensor | None = None,
    ) -> Tensor:
        B, L = trg_BL.shape
        pos_BL = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(trg_BL)
        tok_emb_BLD = self.tok_embedding(trg_BL) * self.scale.to(trg_BL)
        pos_emb_BLD = self.pos_embedding(pos_BL)
        trg_BLD = self.dropout(tok_emb_BLD + pos_emb_BLD)
        for layer in self.layers:
            trg_BLD = layer(trg_BLD, enc_src_BCD, src_mask_B11C, trg_mask_B1LL)
        output_BLV = self.fc_out(trg_BLD)
        return cast(Tensor, output_BLV)


class MoEDecoder(Decoder):
    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        context_window: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
        n_experts: int,
        top_k: int,
        capacity_factor: float,
    ):
        super().__init__(
            vocab_dim=vocab_dim,
            hid_dim=hid_dim,
            context_window=context_window,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
        )
        self.layers = nn.ModuleList(
            [
                MoEDecoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                    n_experts=n_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                )
                for _ in range(n_layers)
            ]
        )


# ---------------------------------------------------------------------------
# from model/architecture.py (verbatim)
# ---------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_pad_idx: int,
        trg_pad_idx: int,
    ):
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src_BC: Tensor) -> Tensor:
        src_mask_B11C = (src_BC != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask_B11C

    def forward(self, src_BC: Tensor, trg_BL: Tensor, steps_B1: Tensor) -> Tensor:
        src_mask_B11C = self.make_src_mask(src_BC.long())

        enc_src_BCD = self.encoder(src_BC.long(), src_mask_B11C, steps_B1)
        trg_mask = None  # this will trigger is_causal=True
        output_BLV = self.decoder(trg_BL, enc_src_BCD, src_mask_B11C, trg_mask_B1LL=trg_mask)
        return cast(Tensor, output_BLV)


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture) -- mirrors
# ModelFactory.create_model's MoE-encoder/MoE-decoder branch at tiny size
# ---------------------------------------------------------------------------
def build_directmultistep():
    vocab_dim = 53
    pad_idx = 52
    encoder = MoEEncoder(
        vocab_dim=vocab_dim,
        hid_dim=32,
        context_window=48,
        n_layers=2,
        n_heads=4,
        n_experts=4,
        top_k=2,
        ff_mult=2,
        ff_activation="gelu",
        dropout=0.0,
        attn_bias=False,
        initiate_steps=True,
        include_steps=True,
        capacity_factor=1.0,
    )
    decoder = MoEDecoder(
        vocab_dim=vocab_dim,
        hid_dim=32,
        context_window=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        attn_bias=False,
        ff_mult=2,
        ff_activation="gelu",
        n_experts=4,
        top_k=2,
        capacity_factor=1.0,
    )
    return Seq2Seq(encoder=encoder, decoder=decoder, src_pad_idx=pad_idx, trg_pad_idx=pad_idx)


def example_input_directmultistep():
    generator = torch.Generator().manual_seed(0)
    src_BC = torch.randint(0, 51, (2, 24), generator=generator)
    trg_BL = torch.randint(0, 51, (2, 16), generator=generator)
    steps_B1 = torch.randint(1, 5, (2, 1), generator=generator).float()
    return (src_BC, trg_BL, steps_B1)


MENAGERIE_ENTRIES = [
    (
        "DirectMultiStep",
        "build_directmultistep",
        "example_input_directmultistep",
        2025,
        "vendored-pytorch",
    ),
]
