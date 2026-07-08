# FAITHFUL PORT of https://github.com/Tsinghua-MARS-Lab/DenseTNT @ a0e3b8a51aecf9f9046db4fb72e2793684c96e69
# (original framework: PyTorch; src/modeling/vectornet.py + decoder.py + lib.py)
"""DenseTNT (ICCV 2021): dense goal-heatmap trajectory prediction over vectorized HD-map + agent history.

The official repo's model classes (`VectorNet`, `Decoder`, and the shared submodules in
`modeling/lib.py`: `GlobalGraph`, `CrossAttention`, `GlobalGraphRes`, `PointSubGraph`, `MLP`,
`LayerNorm`) are architecturally self-contained pure-torch code. They are transcribed here
FAITHFULLY layer-for-layer from the real source.

What differs from a direct vendor (why this is a PORT, not a vendored file):
  - The official forward pass is written for VARIABLE-length, per-sample Python lists/dicts
    (`mapping: List[Dict]`, ragged `polyline_spans`, per-example loops) driven by a persistent
    global `args` config singleton (`utils.Args`), and its "dense goal" candidates
    (`mapping[i]['goals_2D']`) are produced by a separate HD-map rasterization / dataset
    preprocessing step (`utils.get_neighbour_points`, cython NMS in `utils_cython`) that lives
    outside the neural network. None of that is torch code; it cannot be "vendored" without a
    full dataset/cython environment.
  - This port batches the same per-example computation into fixed-shape tensors (single example,
    fixed element/lane/goal counts) and takes the dense-goal candidate set as a plain (N, 2)
    input tensor instead of running the offline map-rasterization pipeline, so the module is
    directly torch-traceable. The learned architecture -- sub-graph polyline encoder, global
    self-attention graph, laneGCN-style agent<->lane cross-attention fusion, and the goal-scoring
    decoder (`point_sub_graph` + `goal_scoring` branch, DenseTNT's actual default config) -- is
    reproduced exactly as in the official layers.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# modeling/lib.py -- shared submodules (transcribed faithfully)
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Layer normalization (DenseTNT's own re-implementation, matches lib.py::LayerNorm)."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class GlobalGraph(nn.Module):
    """Multi-head self-attention ("global graph" in the paper's terminology)."""

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = (
            hidden_size // num_attention_heads
            if attention_head_size is None
            else attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = F.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(
        self,
        hidden_size,
        attention_head_size=None,
        num_attention_heads=1,
        key_hidden_size=None,
        query_hidden_size=None,
    ):
        super().__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class GlobalGraphRes(nn.Module):
    """Two-head global graph whose outputs are concatenated (`enhance_global_graph` branch)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = torch.cat(
            [
                self.global_graph(hidden_states, attention_mask),
                self.global_graph2(hidden_states, attention_mask),
            ],
            dim=-1,
        )
        return hidden_states


class PointSubGraph(nn.Module):
    """Encode 2D goal candidates conditioned on the target agent feature (`point_sub_graph`)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                MLP(2, hidden_size // 2),
                MLP(hidden_size, hidden_size // 2),
                MLP(hidden_size, hidden_size),
            ]
        )

    def forward(self, hidden_states, agent):
        predict_agent_num, point_num = hidden_states.shape[0], hidden_states.shape[1]
        hidden_size = self.hidden_size
        agent = (
            agent[:, : hidden_size // 2]
            .unsqueeze(1)
            .expand([predict_agent_num, point_num, hidden_size // 2])
        )
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(torch.cat([hidden_states, agent], dim=-1))
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super().__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# modeling/vectornet.py -- polyline sub-graph + global graph encoder
# ---------------------------------------------------------------------------


class NewSubGraph(nn.Module):
    """Polyline sub-graph: encodes each vectorized polyline (agent history or map lane segment)
    into a single feature via stacked self-attention layers, matching `vectornet.py::NewSubGraph`.
    """

    def __init__(self, hidden_size, depth=3):
        super().__init__()
        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList(
            [GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)]
        )
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: (batch, max_vector_num, hidden_size) padded polyline-vector features.
        attention_mask: (batch, max_vector_num, max_vector_num) 1=attend, 0=pad.
        Returns pooled polyline feature (batch, hidden_size).
        """
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)
        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)
        return torch.max(hidden_states, dim=1)[0]


class VectorNet(nn.Module):
    """VectorNet encoder with the laneGCN agent<->lane fusion DenseTNT enables by default.

    Ports `vectornet.py::VectorNet.forward_encode_sub_graph` + the global-graph stage of
    `vectornet.py::VectorNet.forward`, specialized to a single batch element with fixed
    element counts (agents + lanes) so the whole pipeline is torch-traceable.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.global_graph = GlobalGraphRes(hidden_size)  # enhance_global_graph (paper default)

        # laneGCN fusion (paper default 'laneGCN' other_param)
        self.laneGCN_A2L = CrossAttention(hidden_size)

    def forward(self, agent_polylines, agent_mask, lane_polylines, lane_mask):
        """
        agent_polylines: (num_agents, max_pts, hidden_size) padded per-agent history vectors.
        agent_mask: (num_agents, max_pts, max_pts) self-attention mask per agent polyline.
        lane_polylines: (num_lanes, max_pts, hidden_size) padded per-lane vectorized segments.
        lane_mask: (num_lanes, max_pts, max_pts) self-attention mask per lane polyline.
        Returns: element_states (1, num_agents + num_lanes, hidden_size) -- global-graph output,
                 with element index 0 == the target (ego) agent, matching the official ordering.
        """
        agent_states = self.point_level_sub_graph(agent_polylines, agent_mask)
        lane_states = self.point_level_sub_graph(lane_polylines, lane_mask)

        # laneGCN: fuse realtime agent context into lane nodes (one fusion layer, as in repo).
        num_agents = agent_states.shape[0]
        lanes = lane_states + self.laneGCN_A2L(
            lane_states.unsqueeze(0),
            torch.cat([lane_states, agent_states[0:1]], dim=0).unsqueeze(0),
        ).squeeze(0)

        element_states = torch.cat([agent_states, lanes], dim=0).unsqueeze(0)
        _ = num_agents  # kept for readability / parity with source's local variable
        global_mask = torch.ones(
            1, element_states.shape[1], element_states.shape[1], device=element_states.device
        )
        hidden_states = self.global_graph(element_states, global_mask)
        return hidden_states


# ---------------------------------------------------------------------------
# modeling/decoder.py -- dense goal-scoring decoder ('goal_scoring' + 'point_sub_graph')
# ---------------------------------------------------------------------------


class GoalScoringDecoder(nn.Module):
    """Ports `Decoder`'s default inference branch: encode 2D goal candidates with
    `PointSubGraph`, cross-attend to the encoded scene elements, and score each candidate
    (`Decoder.get_scores`, `point_sub_graph` branch -- DenseTNT's actual released config).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
        self.goals_2D_cross_attention = CrossAttention(hidden_size)
        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)

    def forward(self, goals_2D, element_states, target_agent_feature):
        """
        goals_2D: (num_goals, 2) dense candidate goal coordinates (from HD-map rasterization
            upstream of the network in the official pipeline; supplied here as a plain tensor).
        element_states: (1, num_elements, hidden_size) VectorNet global-graph output.
        target_agent_feature: (1, hidden_size) target agent's own encoded feature
            (== element_states[:, 0, :] in the official code).
        Returns: log-softmax scores over goal candidates, shape (num_goals,).
        """
        goals_2D_hidden = self.goals_2D_point_sub_graph(
            goals_2D.unsqueeze(0), target_agent_feature
        ).squeeze(0)
        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), element_states
        ).squeeze(0)

        expanded_agent = target_agent_feature.expand(goals_2D_hidden.shape[0], -1)
        scores = self.goals_2D_decoder(
            torch.cat([expanded_agent, goals_2D_hidden, goals_2D_hidden_attention], dim=-1)
        )
        scores = scores.squeeze(-1)
        scores = F.log_softmax(scores, dim=-1)
        return scores


class DenseTNT(nn.Module):
    """Full DenseTNT: VectorNet scene encoder + dense goal-scoring decoder."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.vectornet = VectorNet(hidden_size)
        self.decoder = GoalScoringDecoder(hidden_size)

    def forward(self, agent_polylines, agent_mask, lane_polylines, lane_mask, goals_2D):
        element_states = self.vectornet(agent_polylines, agent_mask, lane_polylines, lane_mask)
        target_agent_feature = element_states[:, 0, :]
        scores = self.decoder(goals_2D, element_states, target_agent_feature)
        return scores


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------

_HIDDEN = 128
_MAX_PTS = 9  # DenseTNT default sub_graph vector points per polyline
_NUM_AGENTS = 5
_NUM_LANES = 10
_NUM_GOALS = 64


def build_densetnt():
    return DenseTNT(hidden_size=_HIDDEN)


def example_input_densetnt():
    agent_polylines = torch.randn(_NUM_AGENTS, _MAX_PTS, _HIDDEN)
    agent_mask = torch.ones(_NUM_AGENTS, _MAX_PTS, _MAX_PTS)
    lane_polylines = torch.randn(_NUM_LANES, _MAX_PTS, _HIDDEN)
    lane_mask = torch.ones(_NUM_LANES, _MAX_PTS, _MAX_PTS)
    goals_2D = torch.randn(_NUM_GOALS, 2)
    return (agent_polylines, agent_mask, lane_polylines, lane_mask, goals_2D)


MENAGERIE_ENTRIES = [
    ("DenseTNT", "build_densetnt", "example_input_densetnt", 2021, "ported-pytorch"),
]
