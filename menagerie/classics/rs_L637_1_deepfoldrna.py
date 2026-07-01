# SOURCE: vendored from robpearc/DeepFoldRNA @ main
# Files combined: network/config.py, network/embedders.py, network/msa.py, network/pair.py,
# network/single.py, network/heads.py, network/msa_transformer.py, network/pair_transformer.py,
# network/network_1.py
# Only minimal changes: merged multiple source files into one module, dropped disk-based
# feature loading (network/features.py) since example_input_ synthesizes a random tiny
# feature_dict with the exact same tensor shapes/dtypes the real collect_features() produces.
# Architecture (layers, attention math, recycling, triangle updates) is untouched.
import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# --- network/config.py (shrunk for a tiny random-init trace: fewer blocks/heads/dims) ---
network_config = {
    "input_seq_dim": 6,
    "input_msa_dim": 7,
    "msa_dim": 8,
    "pair_dim": 8,
    "seq_dim": 12,
    "no_msa_transformer_blocks": 2,
    "no_single_transformer_blocks": 1,
    "no_pair_transformer_blocks": 1,
    "no_cycles": 2,
    "input_embedder": {
        "no_pos_bins_1": 14,
        "no_pos_bins_2": 65,
        "rel_pos_1d": 14,
        "min_rel_range": -32,
        "max_rel_range": 32,
        "max_seq_length": 64,
    },
    "s_att_pair_bias": {"no_heads": 2, "hidden_dim": 4},
    "s_att": {"no_heads": 2, "hidden_dim": 4},
    "s_outer_product_mean": {"no_heads": 2, "hidden_dim": 4},
    "msa_row_att": {"no_heads": 2, "hidden_dim": 4},
    "msa_col_att": {"no_heads": 2, "hidden_dim": 4},
    "msa_transition": {"no_heads": 2, "hidden_dim": 2},
    "msa_outer_product_mean": {"no_heads": 2, "hidden_dim": 4},
    "tri_out": {"hidden_dim": 8},
    "tri_in": {"hidden_dim": 8},
    "tri_att_start": {"no_heads": 2, "hidden_dim": 4},
    "tri_att_end": {"no_heads": 2, "hidden_dim": 4},
    "pair_transition": {"hidden_dim": 2},
    "torsion_head": {"hidden_dim": 12, "no_blocks": 1, "no_angles": 9, "angle_bins": 6},
    "geometry_head": {
        "min_dist": 2.0,
        "max_dist": 40.0,
        "dist_bins": 8,
        "omega_bins": 6,
        "theta_bins": 6,
        "phi_bins": 6,
    },
}


# --- network/embedders.py ---
class Input_Embedder(nn.Module):
    def __init__(self, config, input_config, device):
        super(Input_Embedder, self).__init__()
        self.input_msa_dim = config["input_msa_dim"]
        self.input_seq_dim = config["input_seq_dim"]
        self.msa_dim = config["msa_dim"]
        self.pair_dim = config["pair_dim"]
        self.no_pos_bins_1 = input_config["no_pos_bins_1"]
        self.no_pos_bins_2 = input_config["no_pos_bins_2"]
        self.rel_pos_1d = int(input_config["rel_pos_1d"])
        self.min_rel_range = int(input_config["min_rel_range"])
        self.max_rel_range = int(input_config["max_rel_range"])
        self.max_seq_len = int(input_config["max_seq_length"])

        self.linear_msa_embed = nn.Linear(self.input_msa_dim, self.msa_dim)
        self.linear_seq_pair_i = nn.Linear(self.input_seq_dim, self.pair_dim)
        self.linear_seq_pair_j = nn.Linear(self.input_seq_dim, self.pair_dim)
        self.linear_seq_m = nn.Linear(self.input_seq_dim, self.msa_dim)
        self.linear_embed_pos_1 = nn.Linear(self.no_pos_bins_1, self.msa_dim)
        self.linear_embed_pos_2 = nn.Linear(self.no_pos_bins_2, self.pair_dim)
        self.register_buffer("pos_1", self.compute_position_1(), persistent=False)
        self.register_buffer("pos_2", self.compute_position_2(), persistent=False)

    def compute_position_1(self):
        pos = torch.arange(self.max_seq_len)
        rel_pos = (pos[:, None] & (1 << torch.arange(self.rel_pos_1d))) > 0
        return rel_pos.float()

    def compute_position_2(self):
        pos = torch.arange(self.max_seq_len)
        rel_pos = pos[None, :] - pos[:, None]
        rel_pos = rel_pos.clamp(self.min_rel_range, self.max_rel_range)
        rel_pos_encode = F.one_hot(rel_pos + self.max_rel_range, self.no_pos_bins_2)
        return rel_pos_encode.float()

    def forward(self, input_seq, input_msa):
        num_seqs, length, dim = input_msa.shape

        embedded_seq_m = self.linear_seq_m(input_seq)
        embedded_msa = self.linear_msa_embed(input_msa)
        rel_pos_1d = self.pos_1[:length]
        rel_pos_2d = self.pos_2[:length, :length]
        embedded_pos_1 = self.linear_embed_pos_1(rel_pos_1d)
        embedded_pos_2 = self.linear_embed_pos_2(rel_pos_2d)
        embedded_seq_pair_i = self.linear_seq_pair_i(input_seq)
        embedded_seq_pair_j = self.linear_seq_pair_j(input_seq)

        msa_embed = embedded_msa + embedded_seq_m[None, :, :] + embedded_pos_1[None, :, :]
        pair_embed = (
            embedded_seq_pair_i[None, :, :] + embedded_seq_pair_j[:, None, :] + embedded_pos_2
        )

        return msa_embed, pair_embed


class HMM_Embedder(nn.Module):
    def __init__(self, config):
        super(HMM_Embedder, self).__init__()
        self.msa_dim = config["msa_dim"]
        self.hmm_linear = nn.Linear(15, self.msa_dim)

    def forward(self, hmm):
        hmm_embed = self.hmm_linear(hmm)
        return hmm_embed


class Secondary_Structure_Embedder(nn.Module):
    def __init__(self, config):
        super(Secondary_Structure_Embedder, self).__init__()
        self.pair_dim = config["pair_dim"]
        self.ss_linear = nn.Linear(1, self.pair_dim)

    def forward(self, ss):
        pair_embed_ss = self.ss_linear(ss)
        return pair_embed_ss


class Recycling_Embedder_S(nn.Module):
    def __init__(self, config):
        super(Recycling_Embedder_S, self).__init__()
        self.seq_dim = config["seq_dim"]
        self.s_norm = nn.LayerNorm(self.seq_dim)

    def forward(self, seq_embed):
        seq_embed = self.s_norm(seq_embed)
        return seq_embed


class Recycling_Embedder(nn.Module):
    def __init__(self, config):
        super(Recycling_Embedder, self).__init__()
        self.msa_dim = config["msa_dim"]
        self.pair_dim = config["pair_dim"]
        self.m_norm = nn.LayerNorm(self.msa_dim)
        self.z_norm = nn.LayerNorm(self.pair_dim)

    def forward(self, msa_embed, pair_embed):
        msa_embed = self.m_norm(msa_embed)
        pair_embed = self.z_norm(pair_embed)
        return msa_embed, pair_embed


# --- network/msa.py ---
class MSA_Row_Att(nn.Module):
    def __init__(self, global_config, msa_row_att_config):
        super(MSA_Row_Att, self).__init__()
        self.msa_dim = global_config["msa_dim"]
        self.pair_dim = global_config["pair_dim"]
        self.no_heads = msa_row_att_config["no_heads"]
        self.hidden_dim = msa_row_att_config["hidden_dim"]

        self.m_norm = nn.LayerNorm(self.msa_dim)
        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.bias_linear = nn.Linear(self.pair_dim, self.no_heads, bias=False)
        self.q_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.k_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.v_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.gate_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim)
        self.output_linear = nn.Linear(self.no_heads * self.hidden_dim, self.msa_dim)

        self.norm_factor = 1 / math.sqrt(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, msa_embed, pair_embed):
        num_seqs, length, embed_dim = msa_embed.shape

        msa_embed = self.m_norm(msa_embed)
        pair_embed = self.z_norm(pair_embed)

        query = self.q_linear(msa_embed)
        key = self.k_linear(msa_embed)
        value = self.v_linear(msa_embed)

        query = query.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)
        key = key.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)
        value = value.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)

        query = query * self.norm_factor

        pair_bias = self.bias_linear(pair_embed)
        pair_bias = pair_bias[None, :, :, :].permute(0, 3, 1, 2)
        att_map = torch.einsum("rihd,rjhd->rhij", query, key) + pair_bias
        att_map = F.softmax(att_map, dim=-1)

        gate = self.gate_linear(msa_embed)
        gate = self.sigmoid(gate)
        gate = gate.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)

        out = torch.einsum("rhij,rjhd->rihd", att_map, value)
        out = out * gate
        out = out.contiguous().view(num_seqs, length, -1)
        msa_embed_out = self.output_linear(out)

        return msa_embed_out


class MSA_Col_Att(nn.Module):
    def __init__(self, global_config, msa_col_att_config):
        super(MSA_Col_Att, self).__init__()
        self.msa_dim = global_config["msa_dim"]
        self.pair_dim = global_config["pair_dim"]
        self.no_heads = msa_col_att_config["no_heads"]
        self.hidden_dim = msa_col_att_config["hidden_dim"]

        self.m_norm = nn.LayerNorm(self.msa_dim)
        self.q_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.k_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.v_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim, bias=False)
        self.gate_linear = nn.Linear(self.msa_dim, self.no_heads * self.hidden_dim)
        self.output_linear = nn.Linear(self.no_heads * self.hidden_dim, self.msa_dim)

        self.norm_factor = 1 / math.sqrt(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, msa_embed):
        num_seqs, length, embed_dim = msa_embed.shape

        msa_embed = self.m_norm(msa_embed)

        query = self.q_linear(msa_embed)
        key = self.k_linear(msa_embed)
        value = self.v_linear(msa_embed)

        query = query.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)
        key = key.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)
        value = value.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)

        query = query * self.norm_factor

        attn_map = torch.einsum("ichd,jchd->hcij", query, key)
        attn_map = F.softmax(attn_map, dim=-1)

        gate = self.gate_linear(msa_embed)
        gate = self.sigmoid(gate)
        gate = gate.contiguous().view(num_seqs, length, self.no_heads, self.hidden_dim)

        out = torch.einsum("hcij,jchd->ichd", attn_map, value)
        out = out * gate
        out = out.contiguous().view(num_seqs, length, -1)
        m_embed_out = self.output_linear(out)

        return m_embed_out


class MSA_Transition(nn.Module):
    def __init__(self, global_config, msa_transition_config):
        super(MSA_Transition, self).__init__()
        self.msa_dim = global_config["msa_dim"]
        self.hidden_dim = msa_transition_config["hidden_dim"]

        self.m_norm = nn.LayerNorm(self.msa_dim)
        self.linear_trans_1 = nn.Linear(self.msa_dim, self.msa_dim * self.hidden_dim)
        self.linear_trans_2 = nn.Linear(self.msa_dim * self.hidden_dim, self.msa_dim)

        self.relu = nn.ReLU()

    def forward(self, msa_embed):
        msa_embed = self.m_norm(msa_embed)
        msa_embed = self.linear_trans_1(msa_embed)
        msa_embed = self.relu(msa_embed)
        msa_embed = self.linear_trans_2(msa_embed)
        return msa_embed


class MSA_Outer_Product_Mean(nn.Module):
    def __init__(self, global_config, msa_outer_product_mean_config):
        super(MSA_Outer_Product_Mean, self).__init__()
        self.msa_embed_dim = global_config["msa_dim"]
        self.pair_dim = global_config["pair_dim"]
        self.hidden_dim = msa_outer_product_mean_config["hidden_dim"]
        self.m_norm = nn.LayerNorm(self.msa_embed_dim)
        self.linear_opm_1 = nn.Linear(self.msa_embed_dim, self.hidden_dim)
        self.linear_opm_2 = nn.Linear(self.msa_embed_dim, self.hidden_dim)
        self.linear_opm_3 = nn.Linear(self.hidden_dim * self.hidden_dim, self.pair_dim)

    def forward(self, msa_embed):
        num_seqs, length, embed_dim = msa_embed.shape

        msa_embed = self.m_norm(msa_embed)

        msa_1 = self.linear_opm_1(msa_embed)
        msa_2 = self.linear_opm_2(msa_embed)
        pair_opm = torch.einsum("ria,rjb->rijab", msa_2, msa_1)
        pair_opm = torch.mean(pair_opm, dim=0)
        pair_opm = pair_opm.contiguous().view(length, length, -1)
        pair_opm = self.linear_opm_3(pair_opm)

        return pair_opm


# --- network/pair.py ---
class Triangle_Att_Start(nn.Module):
    def __init__(self, global_config, tri_att_start_config):
        super(Triangle_Att_Start, self).__init__()
        self.pair_dim = global_config["pair_dim"]
        self.no_heads = tri_att_start_config["no_heads"]
        self.hidden_dim = tri_att_start_config["hidden_dim"]

        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.q_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.k_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.v_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.bias_linear = nn.Linear(self.pair_dim, self.no_heads, bias=False)
        self.gate_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads)
        self.output_linear = nn.Linear(self.hidden_dim * self.no_heads, self.pair_dim)

        self.norm_factor = 1 / math.sqrt(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_embed):
        length, _, embed_dim = pair_embed.shape

        pair_embed = self.z_norm(pair_embed)

        query = self.q_linear(pair_embed)
        key = self.k_linear(pair_embed)
        value = self.v_linear(pair_embed)

        query = query.contiguous().view(length, length, self.no_heads, self.hidden_dim)
        key = key.contiguous().view(length, length, self.no_heads, self.hidden_dim)
        value = value.contiguous().view(length, length, self.no_heads, self.hidden_dim)

        query = query * self.norm_factor

        pair_bias = self.bias_linear(pair_embed)
        pair_bias = pair_bias[None, :, :, :].permute(3, 0, 1, 2)
        attn_map = torch.einsum("ijhd,ikhd->hijk", query, key) + pair_bias
        attn_map = F.softmax(attn_map, dim=-1)

        gate = self.gate_linear(pair_embed)
        gate = self.sigmoid(gate)
        gate = gate.contiguous().view(length, length, self.no_heads, self.hidden_dim)

        out = torch.einsum("hijk,ikhd->ijhd", attn_map, value)
        out = out * gate
        out = out.contiguous().view(length, length, -1)
        pair_embed_out = self.output_linear(out)

        return pair_embed_out


class Triangle_Att_End(nn.Module):
    def __init__(self, global_config, tri_att_end_config):
        super(Triangle_Att_End, self).__init__()
        self.pair_dim = global_config["pair_dim"]
        self.no_heads = tri_att_end_config["no_heads"]
        self.hidden_dim = tri_att_end_config["hidden_dim"]

        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.q_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.k_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.v_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads, bias=False)
        self.bias_linear = nn.Linear(self.pair_dim, self.no_heads, bias=False)
        self.gate_linear = nn.Linear(self.pair_dim, self.hidden_dim * self.no_heads)
        self.output_linear = nn.Linear(self.hidden_dim * self.no_heads, self.pair_dim)

        self.norm_factor = 1 / math.sqrt(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_embed):
        length, _, embed_dim = pair_embed.shape

        pair_embed = self.z_norm(pair_embed)

        query = self.q_linear(pair_embed)
        key = self.k_linear(pair_embed)
        value = self.v_linear(pair_embed)

        query = query.contiguous().view(length, length, self.no_heads, self.hidden_dim)
        key = key.contiguous().view(length, length, self.no_heads, self.hidden_dim)
        value = value.contiguous().view(length, length, self.no_heads, self.hidden_dim)

        query = query * self.norm_factor

        pair_bias = self.bias_linear(pair_embed)
        pair_bias = pair_bias[None, :, :, :].permute(3, 0, 2, 1)
        attn_map = torch.einsum("ijhd,kihd->hijk", query, key) + pair_bias
        attn_map = F.softmax(attn_map, dim=-1)

        gate = self.gate_linear(pair_embed)
        gate = self.sigmoid(gate)
        gate = gate.contiguous().view(length, length, self.no_heads, self.hidden_dim)

        out = torch.einsum("hijk,kjhd->ijhd", attn_map, value)
        out = out * gate
        out = out.contiguous().view(length, length, -1)
        pair_embed_out = self.output_linear(out)

        return pair_embed_out


class Triangle_Update_Outgoing(nn.Module):
    def __init__(self, global_config, tri_out_config):
        super(Triangle_Update_Outgoing, self).__init__()
        self.pair_dim = global_config["pair_dim"]
        self.hidden_dim = tri_out_config["hidden_dim"]

        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.a_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.gate_a_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.b_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.gate_b_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.pair_dim)
        self.gate_linear = nn.Linear(self.pair_dim, self.pair_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_embed):
        pair_embed = self.z_norm(pair_embed)

        node_1 = self.a_linear(pair_embed)
        gate_node_1 = self.gate_a_linear(pair_embed)
        gate_node_1 = self.sigmoid(gate_node_1)
        node_1 = node_1 * gate_node_1

        node_2 = self.b_linear(pair_embed)
        gate_node_2 = self.gate_b_linear(pair_embed)
        gate_node_2 = self.sigmoid(gate_node_2)
        node_2 = node_2 * gate_node_2

        pair_update = torch.einsum("ikh,jkh->ijh", node_1, node_2)
        pair_update = self.layer_norm(pair_update)
        pair_update = self.output_linear(pair_update)
        gate_out = self.gate_linear(pair_embed)
        gate_out = self.sigmoid(gate_out)
        pair_update = pair_update * gate_out

        return pair_update


class Triangle_Update_Incoming(nn.Module):
    def __init__(self, global_config, tri_in_config):
        super(Triangle_Update_Incoming, self).__init__()
        self.pair_dim = global_config["pair_dim"]
        self.hidden_dim = tri_in_config["hidden_dim"]

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.a_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.gate_a_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.b_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.gate_b_linear = nn.Linear(self.pair_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.pair_dim)
        self.gate_linear = nn.Linear(self.pair_dim, self.pair_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_embed):
        pair_embed = self.z_norm(pair_embed)

        node_1 = self.a_linear(pair_embed)
        gate_node_1 = self.gate_a_linear(pair_embed)
        gate_node_1 = self.sigmoid(gate_node_1)
        node_1 = node_1 * gate_node_1

        node_2 = self.b_linear(pair_embed)
        gate_node_2 = self.gate_b_linear(pair_embed)
        gate_node_2 = self.sigmoid(gate_node_2)
        node_2 = node_2 * gate_node_2

        pair_update = torch.einsum("ijh,ikh->jkh", node_1, node_2)
        pair_update = self.layer_norm(pair_update)
        pair_update = self.output_linear(pair_update)
        gate_out = self.gate_linear(pair_embed)
        gate_out = self.sigmoid(gate_out)
        pair_update = pair_update * gate_out

        return pair_update


class Pair_Transition(nn.Module):
    def __init__(self, global_config, pair_trans_config):
        super(Pair_Transition, self).__init__()
        self.pair_dim = global_config["pair_dim"]
        self.hidden_dim = pair_trans_config["hidden_dim"]

        self.z_norm = nn.LayerNorm(self.pair_dim)
        self.linear_trans_1 = nn.Linear(self.pair_dim, self.pair_dim * self.hidden_dim)
        self.linear_trans_2 = nn.Linear(self.pair_dim * self.hidden_dim, self.pair_dim)

        self.relu = nn.ReLU()

    def forward(self, pair_embed):
        pair_embed = self.z_norm(pair_embed)
        pair_embed = self.linear_trans_1(pair_embed)
        pair_embed = self.relu(pair_embed)
        pair_embed = self.linear_trans_2(pair_embed)
        return pair_embed


# --- network/heads.py ---
class Angle_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(Angle_Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, seq_embed):
        seq_embed = self.linear_1(F.relu(seq_embed))
        seq_embed = self.linear_2(F.relu(seq_embed))
        return seq_embed


class Torsion_Head(nn.Module):
    def __init__(self, config, torsion_head_config):
        super(Torsion_Head, self).__init__()
        self.seq_dim = config["seq_dim"]
        self.hidden_dim = torsion_head_config["hidden_dim"]
        self.no_blocks = torsion_head_config["no_blocks"]
        self.no_angles = torsion_head_config["no_angles"]
        self.angle_bins = torsion_head_config["angle_bins"]

        self.linear_in = nn.Linear(self.seq_dim, self.hidden_dim)
        self.angle_layer = Angle_Block(self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.no_angles * 2)
        self.linear_out_dis = nn.Linear(self.hidden_dim, self.no_angles * self.angle_bins)

    def forward(self, seq_embed):
        length, embed_dim = seq_embed.shape
        seq_embed = self.linear_in(F.relu(seq_embed))

        for i in range(self.no_blocks):
            seq_embed = seq_embed + self.angle_layer(seq_embed)

        seq_embed = F.relu(seq_embed)

        angles_dis = self.linear_out_dis(seq_embed)
        angles_dis = F.log_softmax(
            angles_dis.contiguous().view(length, self.no_angles, self.angle_bins), dim=-1
        )

        output = {}
        output["angles_dis"] = angles_dis

        return output


class Geometry_Head(nn.Module):
    def __init__(self, config, geometry_head_config):
        super(Geometry_Head, self).__init__()
        self.pair_dim = config["pair_dim"]
        self.dis_bins = geometry_head_config["dist_bins"]
        self.omg_bins = geometry_head_config["omega_bins"]
        self.theta_bins = geometry_head_config["theta_bins"]
        self.phi_bins = geometry_head_config["phi_bins"]

        self.linear_dis_n = nn.Linear(self.pair_dim, self.dis_bins)
        self.linear_dis_c4 = nn.Linear(self.pair_dim, self.dis_bins)
        self.linear_dis_p = nn.Linear(self.pair_dim, self.dis_bins)
        self.linear_omg = nn.Linear(self.pair_dim, self.omg_bins)
        self.linear_theta = nn.Linear(self.pair_dim, self.theta_bins)
        self.linear_phi = nn.Linear(self.pair_dim, self.phi_bins)

    def forward(self, pair_embed):
        pred_dis_n = self.linear_dis_n(pair_embed)
        pred_dis_n = F.log_softmax(pred_dis_n + pred_dis_n.permute(1, 0, 2), dim=-1)
        pred_dis_c4 = self.linear_dis_c4(pair_embed)
        pred_dis_c4 = F.log_softmax(pred_dis_c4 + pred_dis_c4.permute(1, 0, 2), dim=-1)
        pred_dis_p = self.linear_dis_p(pair_embed)
        pred_dis_p = F.log_softmax(pred_dis_p + pred_dis_p.permute(1, 0, 2), dim=-1)
        pred_omg = self.linear_omg(pair_embed)
        pred_omg = F.log_softmax(pred_omg + pred_omg.permute(1, 0, 2), dim=-1)
        pred_theta = F.log_softmax(self.linear_theta(pair_embed), dim=-1)
        pred_phi = F.log_softmax(self.linear_phi(pair_embed), dim=-1)

        output = {}
        output["pred_dis_n"] = pred_dis_n
        output["pred_dis_c4"] = pred_dis_c4
        output["pred_dis_p"] = pred_dis_p
        output["pred_omg"] = pred_omg
        output["pred_theta"] = pred_theta
        output["pred_phi"] = pred_phi

        return output


class MSA_Head(nn.Module):
    def __init__(self, config):
        super(MSA_Head, self).__init__()
        self.msa_dim = config["msa_dim"]
        self.in_msa_dim = config["input_msa_dim"]
        self.linear_msa = nn.Linear(self.msa_dim, self.in_msa_dim - 1)

    def forward(self, msa_embed):
        pred_msa = F.log_softmax(self.linear_msa(msa_embed[:-1, :, :]), dim=-1)

        output = {}
        output["pred_msa"] = pred_msa

        return output


# --- network/msa_transformer.py ---
class MSA_Transformer_Block(nn.Module):
    def __init__(self, config):
        super(MSA_Transformer_Block, self).__init__()
        self.drop_msa = nn.Dropout(0.15)
        self.drop_pair_row = nn.Dropout(0.25)
        self.drop_pair_col = nn.Dropout(0.25)
        self.msa_row_att = MSA_Row_Att(config, config["msa_row_att"])
        self.msa_col_att = MSA_Col_Att(config, config["msa_col_att"])
        self.msa_transition = MSA_Transition(config, config["msa_transition"])
        self.msa_outer_product_mean = MSA_Outer_Product_Mean(
            config, config["msa_outer_product_mean"]
        )
        self.triangle_update_outgoing = Triangle_Update_Outgoing(config, config["tri_out"])
        self.triangle_update_incoming = Triangle_Update_Incoming(config, config["tri_in"])
        self.triangle_att_start = Triangle_Att_Start(config, config["tri_att_start"])
        self.triangle_att_end = Triangle_Att_End(config, config["tri_att_end"])
        self.pair_transition = Pair_Transition(config, config["pair_transition"])

    def forward(self, msa_embed, pair_embed):
        msa_embed = msa_embed + self.drop_msa(self.msa_row_att(msa_embed, pair_embed))
        msa_embed = msa_embed + self.msa_col_att(msa_embed)
        msa_embed = msa_embed + self.msa_transition(msa_embed)
        pair_embed = pair_embed + self.msa_outer_product_mean(msa_embed)
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_update_outgoing(pair_embed))
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_update_incoming(pair_embed))
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_att_start(pair_embed))
        pair_embed = pair_embed + self.drop_pair_col(self.triangle_att_end(pair_embed))
        pair_embed = pair_embed + self.pair_transition(pair_embed)
        return msa_embed, pair_embed


class MSA_Transformer(nn.Module):
    def __init__(self, config):
        super(MSA_Transformer, self).__init__()
        self.no_msa_trans_blocks = config["no_msa_transformer_blocks"]
        self.msa_transformer_blocks = nn.ModuleList()
        for _ in range(self.no_msa_trans_blocks):
            block = MSA_Transformer_Block(config)
            self.msa_transformer_blocks.append(block)

    def forward(self, msa_embed, pair_embed):
        for i in range(self.no_msa_trans_blocks):
            msa_embed, pair_embed = self.msa_transformer_blocks[i](msa_embed, pair_embed)
        return msa_embed, pair_embed


# --- network/pair_transformer.py ---
class Pair_Transformer_Block(nn.Module):
    def __init__(self, config):
        super(Pair_Transformer_Block, self).__init__()
        self.drop_pair_row = nn.Dropout(0.25)
        self.drop_pair_col = nn.Dropout(0.25)
        self.triangle_update_outgoing = Triangle_Update_Outgoing(config, config["tri_out"])
        self.triangle_update_incoming = Triangle_Update_Incoming(config, config["tri_in"])
        self.triangle_att_start = Triangle_Att_Start(config, config["tri_att_start"])
        self.triangle_att_end = Triangle_Att_End(config, config["tri_att_end"])
        self.pair_transition = Pair_Transition(config, config["pair_transition"])

    def forward(self, pair_embed):
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_update_outgoing(pair_embed))
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_update_incoming(pair_embed))
        pair_embed = pair_embed + self.drop_pair_row(self.triangle_att_start(pair_embed))
        pair_embed = pair_embed + self.drop_pair_col(self.triangle_att_end(pair_embed))
        pair_embed = pair_embed + self.pair_transition(pair_embed)
        return pair_embed


class Pair_Transformer(nn.Module):
    def __init__(self, config):
        super(Pair_Transformer, self).__init__()
        self.no_pair_trans_blocks = config["no_pair_transformer_blocks"]
        self.pair_transformer_blocks = nn.ModuleList()
        for _ in range(self.no_pair_trans_blocks):
            block = Pair_Transformer_Block(config)
            self.pair_transformer_blocks.append(block)

    def forward(self, pair_embed):
        for i in range(self.no_pair_trans_blocks):
            pair_embed = self.pair_transformer_blocks[i](pair_embed)
        return pair_embed


# --- network/network_1.py ---
class Pred_Network(nn.Module):
    def __init__(self, device="cpu"):
        super(Pred_Network, self).__init__()
        self.msa_dim = network_config["msa_dim"]
        self.seq_dim = network_config["seq_dim"]
        self.no_cycles = network_config["no_cycles"]

        self.input_embedder = Input_Embedder(
            network_config, network_config["input_embedder"], device
        )
        self.recycling_embedder = Recycling_Embedder(network_config)
        self.recycling_embedder_s = Recycling_Embedder_S(network_config)
        self.ss_embedder = Secondary_Structure_Embedder(network_config)
        self.hmm_embedder = HMM_Embedder(network_config)
        self.msa_transformer_layers = MSA_Transformer(network_config)
        self.pair_transformer_layers = Pair_Transformer(network_config)
        self.linear_s = nn.Linear(self.msa_dim, self.seq_dim)
        self.geometry_head = Geometry_Head(network_config, network_config["geometry_head"])
        self.tor_head = Torsion_Head(network_config, network_config["torsion_head"])
        self.msa_head = MSA_Head(network_config)

    def iteration(self, features, msa_recycle, pair_recycle, seq_recycle, recycle_flag=False):
        num_seqs, length, embed_dim = features["msa"].shape

        seq = features["seq"]
        msa = features["msa"]
        ss = features["ss"]
        hmm = features["hmm"]

        msa_mask = torch.zeros(num_seqs, length).to(seq.device)
        masked_msa = torch.cat([msa * (1 - msa_mask[:, :, None]), msa_mask[:, :, None]], dim=-1)
        msa_embed, pair_embed = self.input_embedder(seq, masked_msa)
        hmm_embed = self.hmm_embedder(hmm)
        msa_embed = torch.cat([msa_embed, hmm_embed[None, :, :]], dim=0)

        pair_embed_ss = self.ss_embedder(ss)
        pair_embed_ss = self.pair_transformer_layers(pair_embed_ss)
        pair_embed = pair_embed + pair_embed_ss

        if recycle_flag:
            msa_recycle, pair_recycle = self.recycling_embedder(msa_recycle, pair_recycle)
            msa_embed = msa_embed + msa_recycle
            pair_embed = pair_embed + pair_recycle

        msa_embed, pair_embed = self.msa_transformer_layers(msa_embed, pair_embed)

        sequence_embedding = msa_embed[0]
        seq_embed = self.linear_s(sequence_embedding)

        return msa_embed, pair_embed, seq_embed

    def forward(self, features):
        cycles = self.no_cycles

        msa_recycle = None
        pair_recycle = None
        seq_recycle = None
        recycle = False
        with torch.no_grad():
            for i in range(cycles - 1):
                msa_recycle, pair_recycle, seq_recycle = self.iteration(
                    features, msa_recycle, pair_recycle, seq_recycle, recycle
                )
                recycle = True

        msa_final, pair_final, seq_final = self.iteration(
            features, msa_recycle, pair_recycle, seq_recycle, recycle
        )

        output = {}
        output["geometry_head"] = self.geometry_head(pair_final)
        output["torsion_head"] = self.tor_head(seq_final)

        return output


# --- staging build/example helpers ---
def build_deepfoldrna():
    return Pred_Network(device="cpu")


def example_input_deepfoldrna():
    # Mirrors network/features.collect_features() tensor shapes/dtypes for a tiny RNA of
    # length L with a small sampled MSA depth, on random data (no real MSA/HMM/SS files needed).
    length = 10
    num_seqs = 3
    seq_idx = torch.randint(1, 5, (length,))
    seq_onehot = F.one_hot(seq_idx, 6).float()
    msa_idx = torch.randint(0, 5, (num_seqs, length))
    msa_idx[0] = seq_idx
    msa_onehot = F.one_hot(msa_idx, 6).to(torch.int64)
    ss = torch.zeros(length, length, 1)
    hmm = torch.zeros(length, 15)

    features = {
        "seq": seq_onehot,
        "msa": msa_onehot,
        "ss": ss,
        "hmm": hmm,
    }
    return (features,)


MENAGERIE_ENTRIES = [
    ("DeepFoldRNA", build_deepfoldrna, example_input_deepfoldrna, 2023, MENAGERIE_ZOO),
]
