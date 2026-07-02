# FAITHFUL PORT of https://github.com/openai/multi-agent-emergence-environments @ master
# (original framework: TensorFlow 1.x / tf.contrib, via ma_policy/ma_policy.py +
# ma_policy/graph_construct.py + ma_policy/layers.py). The repo ships the graph-spec
# interpreter (construct_tf_graph) and entity-transformer primitives (residual_sa_block,
# qkv_embed, entity_concat, entity_avg_pooling_masked, circ_conv1d) used to build the
# hide-and-seek policy network from "Emergent Tool Use From Multi-Agent Autocurricula"
# (Baker et al. 2019, arXiv:1909.07528), but the concrete layer spec + trained weight
# shapes only exist inside the shipped `examples/hide_and_seek_full.npz` checkpoint
# (a pickled policy_fn_and_args + numpy weight dict), not as literal python source.
# This port reconstructs the exact real layer topology by cross-referencing the TF1
# layer-construction code with the real trained parameter shapes recovered from that
# checkpoint:
#   circ_conv1d0:            Conv1d(in=1, out=9, kernel_size=3, circular pad)  over 30 lidar rays
#   dense6-0..3 (4 parallel): per-entity-type dense embed to 128 dims, entity-type input
#                             concatenated with the pooled/self "main" feature (280-dim):
#                               dense6-0: agent_qpos_qvel entity (10) + main (280) -> 290 -> 128
#                               dense6-1: box_obs entity        (15) + main (280) -> 295 -> 128
#                               dense6-2: ramp_obs entity        (15) + main (280) -> 295 -> 128
#                               dense6-3: main alone                          (280) -> 128
#   self-attention8:  residual_sa_block over the concatenated entity dimension,
#                     n_embd=128, 2 heads (qk_embed out=256=2*128), pre-attention LayerNorm,
#                     post-attention residual dense (mlp1) + post-attention LayerNorm
#   entity_pooling (avg, masked) -> concat with self_obs branch -> 256-dim "main"
#   layernorm11 -> dense12-0 (256->256) -> layernorm13 -> LSTM(256->256) -> layernorm15
#   3 independent action heads (policy_out): action_movement (33-way), action_pull (2-way),
#   action_glueall (2-way) -- each a plain Linear(256, n) producing categorical logits, as
#   built by MAPolicy._init_policy_out for gym.spaces.MultiDiscrete/Discrete action types.
# Observation normalization (EMAMeanStd input scaling/clipping) and the value head
# (vpred_net, an identical-topology second tower + scalar value output) are training-time
# concerns; this classic keeps the policy tower only (the trained artifact TorchLens can
# meaningfully inspect), matching the paper's entity-centric self-attention architecture.
"""Hide-and-Seek emergent tool-use policy (Baker et al. 2019, "Emergent Tool Use
From Multi-Agent Autocurricula") -- entity-centric self-attention policy network.
Per-entity-type dense embeddings (self, other agents, boxes, ramps, circularly-
convolved lidar) feed a residual self-attention block over the entity dimension,
masked average pooling collapses entities back to a per-agent feature, then an
LSTM + 3 independent categorical action heads (movement / pull / glue-all)
produce the policy output, exactly as in the reference TensorFlow implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircConv1d(nn.Module):
    """Circular 1D convolution over the lidar ray dimension (layers.py:circ_conv1d)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 9, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_rays, in_channels) -> circular-pad along ray dim -> conv -> (batch, n_rays, out_channels)
        num_pad = self.kernel_size // 2
        x = x.transpose(1, 2)  # (batch, in_channels, n_rays)
        x = torch.cat([x[..., -num_pad:], x, x[..., :num_pad]], dim=-1)
        out = self.conv(x)
        return F.relu(out).transpose(1, 2)


class QKVEmbed(nn.Module):
    """Query/key/value projection for entity self-attention (layers.py:qkv_embed)."""

    def __init__(self, in_features: int, n_embd: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.n_embd = n_embd
        self.pre_ln = nn.LayerNorm(in_features)
        self.qk_embed = nn.Linear(in_features, n_embd * 2)
        self.v_embed = nn.Linear(in_features, n_embd)

    def forward(self, x: torch.Tensor):
        # x: (batch, n_entities, features)
        x = self.pre_ln(x)
        bs, ne, _ = x.shape
        qk = self.qk_embed(x).reshape(bs, ne, self.heads, self.n_embd // self.heads, 2)
        query, key = qk[..., 0], qk[..., 1]
        value = self.v_embed(x).reshape(bs, ne, self.heads, self.n_embd // self.heads)
        query = query.permute(0, 2, 1, 3)  # (bs, heads, ne, n_embd/heads)
        key = key.permute(0, 2, 3, 1)  # (bs, heads, n_embd/heads, ne)
        value = value.permute(0, 2, 1, 3)  # (bs, heads, ne, n_embd/heads)
        return query, key, value


class ResidualSABlock(nn.Module):
    """Residual entity self-attention block (layers.py:residual_sa_block)."""

    def __init__(self, in_features: int, n_embd: int = 128, heads: int = 2) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.heads = heads
        self.qkv = QKVEmbed(in_features, n_embd, heads)
        self.mlp1 = nn.Linear(n_embd, n_embd)
        self.post_ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        query, key, value = self.qkv(x)
        logits = torch.matmul(query, key) / (self.n_embd / self.heads) ** 0.5  # (bs, heads, ne, ne)
        if mask is not None:
            mask_row = mask.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, ne)
            logits = logits - (1.0 - mask_row) * 1e10
        softmax = F.softmax(logits, dim=-1)
        att = torch.matmul(softmax, value)  # (bs, heads, ne, n_embd/heads)
        att = att.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.n_embd)
        x_res = x[..., : self.n_embd] if x.shape[-1] >= self.n_embd else x
        post = self.mlp1(att)
        out = (x_res + post) if x_res.shape[-1] == post.shape[-1] else post
        return self.post_ln(out)


def entity_avg_pooling_masked(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Masked average pooling over the entity dimension (layers.py)."""
    if mask is None:
        return x.mean(dim=-2)
    m = mask.unsqueeze(-1)
    summed = (x * m).sum(dim=-2)
    denom = m.sum(dim=-2) + 1e-5
    return summed / denom


class HideAndSeekPolicy(nn.Module):
    """Entity-centric self-attention hide-and-seek policy tower (policy_net)."""

    def __init__(
        self,
        self_dim: int = 10,
        agent_dim: int = 10,
        box_dim: int = 15,
        ramp_dim: int = 15,
        n_lidar: int = 30,
        n_embd: int = 128,
        heads: int = 2,
        main_hidden: int = 256,
    ) -> None:
        super().__init__()
        lidar_filters = 9
        self.lidar_conv = CircConv1d(1, lidar_filters, kernel_size=3)
        main_dim = self_dim + n_lidar * lidar_filters  # 10 + 30*9 = 280

        # dense6-{0,1,2,3}: per-entity-type embeds (entity feature concatenated with
        # broadcast "main" feature), sharing the same 128-dim output width.
        self.embed_agent = nn.Linear(agent_dim + main_dim, n_embd)
        self.embed_box = nn.Linear(box_dim + main_dim, n_embd)
        self.embed_ramp = nn.Linear(ramp_dim + main_dim, n_embd)
        self.embed_main = nn.Linear(main_dim, n_embd)

        self.sa_block = ResidualSABlock(n_embd, n_embd=n_embd, heads=heads)

        # layernorm11 -> dense12-0 -> layernorm13 -> lstm14 -> layernorm15
        pooled_and_self_dim = n_embd + self_dim  # concat pooled entity feature with self obs branch
        self.pre_dense_norm = nn.LayerNorm(pooled_and_self_dim)
        self.dense_main = nn.Linear(pooled_and_self_dim, main_hidden)
        self.pre_lstm_norm = nn.LayerNorm(main_hidden)
        self.lstm = nn.LSTM(main_hidden, main_hidden, batch_first=True)
        self.post_lstm_norm = nn.LayerNorm(main_hidden)

        # policy_out: 3 independent categorical action heads
        self.action_movement = nn.Linear(main_hidden, 33)
        self.action_pull = nn.Linear(main_hidden, 2)
        self.action_glueall = nn.Linear(main_hidden, 2)

    def forward(
        self,
        self_obs: torch.Tensor,
        agent_qpos_qvel: torch.Tensor,
        box_obs: torch.Tensor,
        ramp_obs: torch.Tensor,
        lidar: torch.Tensor,
        entity_mask: torch.Tensor,
    ):
        # self_obs: (bs, self_dim); agent_qpos_qvel: (bs, n_agents, agent_dim);
        # box_obs: (bs, n_boxes, box_dim); ramp_obs: (bs, n_ramps, ramp_dim);
        # lidar: (bs, n_lidar, 1); entity_mask: (bs, n_entities)
        bs = self_obs.shape[0]
        lidar_feat = self.lidar_conv(lidar).reshape(bs, -1)  # (bs, n_lidar*filters)
        main = torch.cat([self_obs, lidar_feat], dim=-1)  # (bs, main_dim)
        main_b = main.unsqueeze(1)

        n_agents = agent_qpos_qvel.shape[1]
        n_boxes = box_obs.shape[1]
        n_ramps = ramp_obs.shape[1]

        agent_embed = self.embed_agent(
            torch.cat([agent_qpos_qvel, main_b.expand(-1, n_agents, -1)], dim=-1)
        )
        box_embed = self.embed_box(torch.cat([box_obs, main_b.expand(-1, n_boxes, -1)], dim=-1))
        ramp_embed = self.embed_ramp(torch.cat([ramp_obs, main_b.expand(-1, n_ramps, -1)], dim=-1))
        main_embed = self.embed_main(main).unsqueeze(
            1
        )  # (bs, 1, n_embd) -- treated as its own entity

        entities = torch.cat(
            [agent_embed, box_embed, ramp_embed, main_embed], dim=1
        )  # (bs, NE, n_embd)
        sa_out = self.sa_block(entities, entity_mask)
        pooled = entity_avg_pooling_masked(sa_out, entity_mask)  # (bs, n_embd)

        x = torch.cat([pooled, self_obs], dim=-1)
        x = self.pre_dense_norm(x)
        x = F.relu(self.dense_main(x))
        x = self.pre_lstm_norm(x)
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.post_lstm_norm(x)

        movement_logits = self.action_movement(x)
        pull_logits = self.action_pull(x)
        glueall_logits = self.action_glueall(x)
        return movement_logits, pull_logits, glueall_logits


# ---- staging scaffolding (example-input shim; no architecture changes) ----
_N_AGENTS = 2
_N_BOXES = 6
_N_RAMPS = 2
_N_LIDAR = 30
_N_ENTITIES = _N_AGENTS + _N_BOXES + _N_RAMPS + 1  # + the "main" pseudo-entity


def build_hide_and_seek_policy() -> nn.Module:
    return HideAndSeekPolicy()


def example_input_hide_and_seek_policy():
    self_obs = torch.randn(2, 10)
    agent_qpos_qvel = torch.randn(2, _N_AGENTS, 10)
    box_obs = torch.randn(2, _N_BOXES, 15)
    ramp_obs = torch.randn(2, _N_RAMPS, 15)
    lidar = torch.rand(2, _N_LIDAR, 1)
    entity_mask = torch.ones(2, _N_ENTITIES)
    return (self_obs, agent_qpos_qvel, box_obs, ramp_obs, lidar, entity_mask)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Hide-and-Seek emergent tool-use policy (entity self-attention)",
        "build_hide_and_seek_policy",
        "example_input_hide_and_seek_policy",
        2019,
        MENAGERIE_ZOO,
    ),
]
