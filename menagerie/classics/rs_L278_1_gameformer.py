# SOURCE: vendored from https://github.com/MCZhi/GameFormer @ main (model/GameFormer.py, model/modules.py)
# GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction
# and Planning for Autonomous Driving. ICCV 2023 Oral.
"""GameFormer: level-k game-theoretic interaction decoder.

Vendored verbatim from MCZhi/GameFormer `model/GameFormer.py` + `model/modules.py`.
Architecture is unmodified; only this header/build/example wrapper were added for
menagerie staging.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# model/modules.py
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super(PositionalEncoding, self).__init__()
        d_model = 256
        dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)


class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]
        type = self.type_emb(inputs[:, -1, 8].int())
        output = output + type

        return output


class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[..., 6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int())
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        interpolating = self.interpolating(inputs[..., 14].int())
        stop_sign = self.stop_sign(inputs[..., 15].int())

        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat(
            [self_line, left_line, right_line, speed_limit, lane_attr], dim=-1
        )

        # process
        output = self.position_encode(self.pointnet(lane_embedding))

        return output


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256)
        )

    def forward(self, inputs):
        output = self.point_net(inputs)

        return output


class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256))
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs, theta, v, size], dim=-1)  # (x, y, heading, vx, vy, w, l, h)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        type = self.type_emb(current_states[:, :, None, 8].int())
        output = torch.max(trajs, dim=-2).values
        output = output + type

        return output


class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len * 4)
        )
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(
            B, M, self._future_len, 4
        )  # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return res, score


class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len):
        super(InitialDecoder, self).__init__()
        dim = 256
        self._modalities = modalities
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors + 1, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer("modal", torch.arange(modalities).long())
        self.register_buffer("agent", torch.arange(neighbors + 1).long())

    def forward(self, id, current_state, encoding, mask):
        # get query
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[id])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :]
        query = encoding[:, None, id] + multi_modal_agent_query

        # decode trajectories
        query_content = self.query_encoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(query_content)

        # post process
        predictions[..., :2] += current_state[:, None, None, :2]

        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len):
        super(InteractionDecoder, self).__init__()
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
        B, N, M, T, _ = actors.shape

        # encoding the trajectories from the last level
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)

        # encoding the interaction using self-attention transformer
        interaction = self.interaction_encoder(futures, mask[:, :N])

        # append the interaction encoding to the context encoding
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask[:, id] = True  # mask the agent future itself from last level

        # decoding the trajectories from the current level
        query = last_content + multi_futures[:, id]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        trajectories, scores = self.decoder(query_content)

        # post process
        trajectories[..., :2] += current_states[:, id, None, None, :2]

        return query_content, trajectories, scores


# ---------------------------------------------------------------------------
# model/GameFormer.py
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6):
        super(Encoder, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation=F.gelu,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            attention_layer, layers, enable_nested_tensor=False
        )

    def segment_map(self, map, map_encoding):
        stride = 10
        B, N_e, N_p, D = map_encoding.shape

        # segment map
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, stride))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        # segment mask
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(
            B, N_e, N_p // stride, N_p // (N_p // stride)
        )
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, inputs):
        # agent encoding
        ego = inputs["ego_state"]
        neighbors = inputs["neighbors_state"]
        actors = torch.cat([inputs["ego_state"].unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # map encoding
        map_lanes = inputs["map_lanes"]
        map_crosswalks = inputs["map_crosswalks"]
        encoded_map_lanes = self.lane_encoder(map_lanes)
        encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)

        # attention fusion
        encodings = []
        masks = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, "Too many neighbors to predict"

        for i in range(N):
            lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
            crosswalks, crosswalks_mask = self.segment_map(
                map_crosswalks[:, i], encoded_map_crosswalks[:, i]
            )
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            masks.append(mask)
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {"actors": actors, "encodings": encodings, "masks": masks}

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3):
        super(Decoder, self).__init__()
        self._levels = levels
        self._neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len)
        self.interaction_stage = nn.ModuleList(
            [InteractionDecoder(future_encoder, future_len) for _ in range(levels)]
        )

    def forward(self, encoder_inputs):
        decoder_outputs = {}
        N = self._neighbors + 1
        assert encoder_inputs["actors"].shape[1] >= N, "Too many neighbors to predict"

        current_states = encoder_inputs["actors"][:, :, -1]
        encodings, masks = encoder_inputs["encodings"], encoder_inputs["masks"]

        # level 0
        results = [
            self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i])
            for i in range(N)
        ]
        last_content = torch.stack([result[0] for result in results], dim=1)
        last_level = torch.stack([result[1] for result in results], dim=1)
        last_scores = torch.stack([result[2] for result in results], dim=1)
        decoder_outputs["level_0_interactions"] = last_level
        decoder_outputs["level_0_scores"] = last_scores

        # level k reasoning
        for k in range(1, self._levels + 1):
            interaction_decoder = self.interaction_stage[k - 1]
            results = [
                interaction_decoder(
                    i,
                    current_states[:, :N],
                    last_level,
                    last_scores,
                    last_content[:, i],
                    encodings[:, i],
                    masks[:, i],
                )
                for i in range(N)
            ]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            decoder_outputs[f"level_{k}_interactions"] = last_level
            decoder_outputs[f"level_{k}_scores"] = last_scores

        return decoder_outputs


class GameFormer(nn.Module):
    def __init__(
        self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4
    ):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(neighbors_to_predict, encoder_layers)
        self.decoder = Decoder(modalities, future_len, neighbors_to_predict, decoder_levels)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)

        return outputs


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------

_HIST_LEN = 11  # matches the official Waymo interaction_prediction data_process.py hist_len
_FUTURE_LEN = 8  # reduced from the paper's 80 to keep the tiny build fast
_NUM_NEIGHBORS = 3  # reduced from the paper's 32
_NEIGHBORS_TO_PREDICT = 1
_MODALITIES = 3
_ENCODER_LAYERS = 1
_DECODER_LEVELS = 1


def build_gameformer():
    return GameFormer(
        modalities=_MODALITIES,
        neighbors_to_predict=_NEIGHBORS_TO_PREDICT,
        future_len=_FUTURE_LEN,
        encoder_layers=_ENCODER_LAYERS,
        decoder_levels=_DECODER_LEVELS,
    )


def _make_agent_state(batch, n, hist_len):
    # AgentEncoder consumes [:8] via LSTM and the object-type index at [8] (0-3, padding_idx=0).
    # A fully-zero final timestep is treated as "missing" (actors_mask), so keep type != 0.
    state = torch.randn(batch, n, hist_len, 9)
    state[..., 8] = torch.randint(1, 4, (batch, n, hist_len)).float()
    return state


def _make_map_lanes(batch, n_agents):
    # LaneEncoder consumes columns: [0:3]=self_line, [3:6]=left_line, [6:9]=right_line,
    # [9]=speed_limit, [10]=self_type (0-3), [11]=left_type (0-10), [12]=right_type (0-10),
    # [13]=traffic_light_type (0-8), [14]=interpolating (0-1), [15]=stop_sign (0-1).
    # n_points MUST equal PositionalEncoding's max_len=100 (hardcoded in LaneEncoder,
    # added without slicing), and must be divisible by the segment_map stride=10.
    n_lanes, n_points = 2, 100  # n_lanes reduced from the paper's 6; n_points fixed at 100
    shape = (batch, n_agents, n_lanes, n_points, 16)
    map_lanes = torch.zeros(shape)
    map_lanes[..., :10] = torch.randn(batch, n_agents, n_lanes, n_points, 10)
    map_lanes[..., 10] = torch.randint(0, 4, shape[:-1]).float()
    map_lanes[..., 11] = torch.randint(0, 11, shape[:-1]).float()
    map_lanes[..., 12] = torch.randint(0, 11, shape[:-1]).float()
    map_lanes[..., 13] = torch.randint(0, 9, shape[:-1]).float()
    map_lanes[..., 14] = torch.randint(0, 2, shape[:-1]).float()
    map_lanes[..., 15] = torch.randint(0, 2, shape[:-1]).float()
    return map_lanes


def example_input_gameformer():
    batch = 1
    n_agents = _NEIGHBORS_TO_PREDICT + 1
    n_crosswalks, n_cw_points = 2, 10  # reduced from the paper's (4, 100)

    ego_state = _make_agent_state(batch, 1, _HIST_LEN).squeeze(1)
    neighbors_state = _make_agent_state(batch, _NUM_NEIGHBORS, _HIST_LEN)
    map_lanes = _make_map_lanes(batch, n_agents)
    map_crosswalks = torch.randn(batch, n_agents, n_crosswalks, n_cw_points, 3)

    inputs = {
        "ego_state": ego_state,
        "neighbors_state": neighbors_state,
        "map_lanes": map_lanes,
        "map_crosswalks": map_crosswalks,
    }
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("GameFormer", "build_gameformer", "example_input_gameformer", 2023, "vendored-pytorch"),
]
