# SOURCE: vendored from https://github.com/lssiair/TUTR @ main
# (model.py + transformer_encoder.py + transformer_decoder.py)
"""TUTR: Trajectory Unified TRansformer (ICCV 2023, lssiair/TUTR). Unifies multimodal
motion-mode classification and social-interaction-conditioned trajectory regression in a
single encoder-decoder transformer. Vendored verbatim from the official repo's
``TrajectoryModel``/``Encoder``/``Decoder`` classes; only the two hardcoded
``.cuda()`` index-tensor constructions in ``forward`` are made device-agnostic
(``.to(ped_obs.device)`` instead of an unconditional ``.cuda()``) since the original
repo assumes a CUDA-only training/eval environment -- this is a portability fix to the
call site, not an architectural change (every layer/module is unmodified).
"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ----------------------------- transformer_encoder.py -----------------------------


class Encoder(nn.Module):
    def __init__(
        self, embed_size, num_layers, heads, forward_expansion, dropout=0.1, islinear=True
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    embed_size, heads, forward_expansion, dropout, islinear=islinear
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x


class EncoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, head, forward_expansion, dropout, islinear=True):
        super(EncoderTransformerBlock, self).__init__()

        self.attn = EncoderMultihHeadAttention(embed_size, head, islinear=islinear)
        self.norm1 = EncoderLayerNorm(embed_size)
        self.norm2 = EncoderLayerNorm(embed_size)
        self.feed_forward = EncoderFeedForwardLayer(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        logits = self.attn(query, key, value, mask)
        x = self.dropout(self.norm1(logits + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class EncoderFeedForwardLayer(nn.Module):
    def __init__(self, d_model, forward_expansion):
        super(EncoderFeedForwardLayer, self).__init__()
        self.w1 = nn.Linear(d_model, d_model * forward_expansion)
        self.w2 = nn.Linear(d_model * forward_expansion, d_model)

    def forward(self, x):
        return self.w2((F.relu(self.w1(x))))


class EncoderLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(EncoderLayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class EncoderMultihHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, islinear=True):
        super(EncoderMultihHeadAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.w_key = (
            nn.Linear(d_model, d_model)
            if islinear
            else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        )
        self.w_query = (
            nn.Linear(d_model, d_model)
            if islinear
            else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        )
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.atten = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query = self.w_query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.atten = encoder_attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc_out(x)


def encoder_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, value), scores


# ----------------------------- transformer_decoder.py -----------------------------


class Decoder(nn.Module):
    def __init__(
        self, embed_size, num_layers, heads, forward_expansion, dropout=0.1, islinear=True
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    embed_size, heads, forward_expansion, dropout, islinear=islinear
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        # query [B K embed_size]
        # key [B N embed_size]
        # mask [B K N]

        for layer in self.layers:
            x = layer(q, k, k, mask)

        return x


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, head, forward_expansion, dropout, islinear=True):
        super(DecoderTransformerBlock, self).__init__()

        self.attn = DecoderMultihHeadAttention(embed_size, head, islinear=islinear)
        self.norm1 = DecoderLayerNorm(embed_size)
        self.norm2 = DecoderLayerNorm(embed_size)
        self.feed_forward = DecoderFeedForwardLayer(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        # query [B 1 embed_size]
        # key [B N embed_size]
        # value [B N embed_size]
        # mask [B 1 N]

        logits = self.attn(query, key, value, mask)  # [B K embed_size]
        x = self.dropout(self.norm1(logits + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class DecoderFeedForwardLayer(nn.Module):
    def __init__(self, d_model, forward_expansion):
        super(DecoderFeedForwardLayer, self).__init__()
        self.w1 = nn.Linear(d_model, d_model * forward_expansion)
        self.w2 = nn.Linear(d_model * forward_expansion, d_model)

    def forward(self, x):
        return self.w2((F.relu(self.w1(x))))


class DecoderLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(DecoderLayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class DecoderMultihHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, islinear=True):
        super(DecoderMultihHeadAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.w_key = (
            nn.Linear(d_model, d_model)
            if islinear
            else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        )
        self.w_query = (
            nn.Linear(d_model, d_model)
            if islinear
            else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        )
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.atten = None

    def forward(self, query, key, value, mask=None):
        # query [B K embed_size]
        # key [B N embed_size]
        # value [B N embed_size]
        # mask [B K N]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(
                1, self.h, 1, 1
            )  # [B h K N] adding the dimension of head

        batch_size = query.size(0)
        query = (
            self.w_query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        )  # [B h K d_k]
        key = self.w_key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [B h N d_k]
        value = (
            self.w_value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        )  # [B h N d_k]

        x, self.atten = decoder_attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [B K d_model]

        return self.fc_out(x)


def decoder_attention(query, key, value, mask=None, dropout=None):
    # query [B h K d_k]
    # key [B h N d_k]
    # value [B h N d_k]
    # mask [B h K N]

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B h K N]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)  # [B h K N]

    if dropout is not None:
        scores = dropout(scores)

    logits = torch.matmul(scores, value)  # [B h K d_k]

    return logits, scores


# ----------------------------- model.py -----------------------------


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        in_size,
        obs_len,
        pred_len,
        embed_size,
        enc_num_layers,
        int_num_layers_list,
        heads,
        forward_expansion,
    ):
        super(TrajectoryModel, self).__init__()

        self.embedding = nn.Linear(in_size * (obs_len + pred_len), embed_size)

        self.mode_encoder = Encoder(
            embed_size, enc_num_layers, heads, forward_expansion, islinear=True
        )
        self.cls_head = nn.Linear(embed_size, 1)

        self.nei_embedding = nn.Linear(in_size * obs_len, embed_size)
        self.social_decoder = Decoder(
            embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False
        )
        self.reg_head = nn.Linear(embed_size, in_size * pred_len)

    def spatial_interaction(self, ped, neis, mask):
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]

        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]

        return int_feat  # [B K embed_size]

    def forward(
        self, ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False, num_k=20
    ):
        # ped_obs [B obs_len 2]
        # nei_obs [B N obs_len 2]
        # motion_modes [K pred_len 2]
        # closest_mode_indices [B]

        ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)  # [B K obs_len 2]
        motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)

        ped_seq = torch.cat(
            (ped_obs, motion_modes), dim=-2
        )  # [B K seq_len 2] seq_len = obs_len + pred_len
        ped_seq = ped_seq.reshape(ped_seq.shape[0], ped_seq.shape[1], -1)  # [B K seq_len*2]
        ped_embedding = self.embedding(ped_seq)  # [B K embed_size]

        ped_feat = self.mode_encoder(ped_embedding)  # [B K embed_size]
        scores = self.cls_head(ped_feat).squeeze()  # [B K]

        if not test:
            # NOTE: original repo hardcodes `.cuda()` here (CUDA-only training loop);
            # made device-agnostic via `.to(ped_obs.device)` -- call-site portability
            # fix only, not an architecture change.
            index1 = torch.LongTensor(range(closest_mode_indices.shape[0])).to(
                ped_obs.device
            )  # [B]
            index2 = closest_mode_indices
            closest_feat = ped_feat[index1, index2].unsqueeze(1)  # [B 1 embed_size]

            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)  # [B 1 embed_size]
            pred_traj = self.reg_head(int_feat.squeeze())  # [B pred_len*2]

            return pred_traj, scores

        if test:
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [B num_k]
            top_k_indices = top_k_indices.flatten()  # [B*num_k]
            # NOTE: original repo hardcodes `.cuda()` here too; same device-agnostic fix.
            index1 = torch.LongTensor(range(ped_feat.shape[0])).to(ped_obs.device)  # [B]
            index1 = index1.unsqueeze(1).repeat(1, num_k).flatten()  # [B*num_k]
            index2 = top_k_indices  # [B*num_k]
            top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
            top_k_feat = top_k_feat.reshape(ped_feat.shape[0], num_k, -1)  # [B num_k embed_size]

            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)  # [B num_k embed_size]
            pred_trajs = self.reg_head(int_feats)  # [B num_k pred_size*2]

            return pred_trajs, scores


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo)
# ---------------------------------------------------------------------------


def build_tutr():
    # Real hyperparameters from train.py's TrajectoryModel(...) construction
    # (embed_size from config/sdd.py's model_hidden_dim=64).
    return TrajectoryModel(
        in_size=2,
        obs_len=8,
        pred_len=12,
        embed_size=64,
        enc_num_layers=2,
        int_num_layers_list=[1, 1],
        heads=4,
        forward_expansion=2,
    )


def example_input_tutr():
    batch = 2
    obs_len = 8
    pred_len = 12
    n_agents = 5  # max number of neighboring agents in the scene
    n_modes = 6  # K motion modes (n_clusters in the real config is 100; kept small here)
    num_k = 3  # top-k modes retained at test time (must be <= n_modes)

    ped_obs = torch.randn(batch, obs_len, 2)
    neis_obs = torch.randn(batch, n_agents, obs_len, 2)
    motion_modes = torch.randn(n_modes, pred_len, 2)
    mask = torch.ones(batch, n_agents, n_agents)
    closest_mode_indices = None

    return (ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, True, num_k)


MENAGERIE_ENTRIES = [
    ("tutr", build_tutr, example_input_tutr, 2023, "SOURCE_AVAILABLE"),
]
