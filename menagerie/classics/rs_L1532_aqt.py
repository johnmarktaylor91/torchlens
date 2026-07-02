# SOURCE: vendored from machine-perception-robotics-group/Action-Q-Transformer @ main
# https://github.com/machine-perception-robotics-group/Action-Q-Transformer
# Files: model_act_q_transformer.py (AQT, NoisyLinear), transformer.py
# (TransformerEncoder/TransformerDecoder/TransformerEncoderLayer/TransformerDecoderLayer,
# itself a copy of Facebook's DETR transformer that the repo vendors in-tree).
#
# AQT (Action Q-Transformer, Itaya, Hirakawa, Yamashita, Fujiyoshi & Sugiura,
# IEEE Access 2025 / arXiv:2306.13879). A Rainbow-DQN-style distributional
# Q-network in which the usual dueling value/advantage MLP heads are replaced
# by a Transformer encoder-decoder: a small CNN feature extractor produces a
# 7x7 feature map, a Transformer encoder attends over the flattened spatial
# patches (with learned row/col positional embeddings) to produce the value
# stream, and a Transformer decoder cross-attends using per-action "action
# query" embeddings (one-hot action id -> Linear) to produce the advantage
# stream for each discrete action -- giving per-action attention maps for
# explainability. Value/advantage heads are C51-style NoisyLinear distributional
# heads combined via the standard dueling formula. No architecture was altered;
# only this staging glue (build_/example_input_/MENAGERIE_ENTRIES) was added.

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# transformer.py (DETR-style transformer encoder/decoder, vendored as-is)
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# model_act_q_transformer.py (AQT, NoisyLinear, vendored as-is)
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Factorised NoisyLinear layer with bias"""

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class AQT(nn.Module):
    """model architecture: AQT (patch size: 7*7)"""

    def __init__(self, args, action_space, head_dim=32, num_encoder_layers=1, num_decoder_layers=1):
        super(AQT, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        hidden_dim = self.action_space * head_dim

        # feature extractor
        self.convs = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.encoder_output_size = 7 * 7 * hidden_dim

        # transformer decoder
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )
        self.decoder_output_size = hidden_dim

        # query branch
        self.act_list = torch.zeros(self.action_space, self.action_space, device=args.device)
        for i in range(self.action_space):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.action_space, hidden_dim)

        # positional encodings
        self.row_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))

        # value branch
        self.fc_h_v = NoisyLinear(
            self.encoder_output_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(
            self.decoder_output_size, args.hidden_size, std_init=args.noisy_std
        )

        # advantage branch
        self.fc_h_a = NoisyLinear(
            self.decoder_output_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_z_a = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset_noise(self):
        for name, module in self.named_children():
            if "fc" in name:
                module.reset_noise()

    def a_reset_noise(self):
        for name, module in self.named_children():
            if ("fc" in name) and ("_a" in name):
                module.reset_noise()

    def forward(self, x, log=False):
        x = self.convs(x)

        # transformer encoder
        bs, c, h, w = x.shape
        pos = (
            torch.cat(
                [
                    self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
                    self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        pos = pos.expand(pos.shape[0], bs, pos.shape[2])
        src = x.flatten(2).permute(2, 0, 1)
        memory = self.transformer_encoder(pos + 0.1 * src)

        # value head
        v = self.fc_z_v(
            F.relu(self.fc_h_v(memory.permute(1, 2, 0).reshape(-1, self.encoder_output_size)))
        )

        # action queries
        query_embed = []
        for action in self.act_list:
            action_query = self.action_encoder(action).unsqueeze(0)
            query_embed.append(action_query)
        query_embed = torch.cat(query_embed, dim=0)

        # transformer decoder
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        hs = self.transformer_decoder(tgt, memory)[0]  # 6,batch,192
        hs = hs.permute(1, 0, 2)  # batch,6,192

        # advantage head
        adv = self.fc_z_a(F.relu(self.fc_h_a(hs)))  # batch,6,51

        # Combine streams
        v, adv = (
            v.view(-1, 1, self.atoms),
            adv.view(-1, self.action_space, self.atoms),
        )  # batch,1,51 / batch,6,51
        q = v + adv - adv.mean(1, keepdim=True)
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q, v, adv


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


class _AQTArgs:
    """Minimal stand-in for main.py's argparse Namespace -- only the fields
    AQT.__init__ reads (atoms, history_length, device, hidden_size, noisy_std),
    at the repo's own argparse defaults except a shrunk `hidden_size`."""

    def __init__(self):
        self.atoms = 51
        self.history_length = 4
        self.device = torch.device("cpu")
        self.hidden_size = 16
        self.noisy_std = 0.1


def build_aqt():
    torch.manual_seed(0)
    args = _AQTArgs()
    model = AQT(args, action_space=6, head_dim=8, num_encoder_layers=1, num_decoder_layers=1)
    model.eval()
    return model


def example_input_aqt():
    torch.manual_seed(0)
    # A single stack of `history_length` 84x84 Atari frames, as passed to
    # AQT.forward (matches the repo's canonical Atari preprocessing size).
    return torch.rand(1, 4, 84, 84)


MENAGERIE_ENTRIES = [
    (
        "AQT (Action Q-Transformer, distributional Q with encoder-decoder attention)",
        "build_aqt",
        "example_input_aqt",
        2023,
        MENAGERIE_ZOO,
    ),
]
