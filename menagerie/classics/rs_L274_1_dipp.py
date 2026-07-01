# SOURCE: vendored from https://github.com/MCZhi/DIPP @ 3f9d821c3a08f14412691bbe042bb8a46df6499f
# (model/predictor.py) -- Differentiable Integrated Prediction and Planning (DIPP), TNNLS 2023.
"""DIPP: Predictor network.

Vendored verbatim from MCZhi/DIPP `model/predictor.py`. Architecture is unmodified;
only this header/build/example wrapper were added for menagerie staging.
"""

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]

        return output


# Local context encoders
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
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(
            nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU()
        )

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
        stop_point = self.stop_point(inputs[..., 14].int())
        interpolating = self.interpolating(inputs[..., 15].int())
        stop_sign = self.stop_sign(inputs[..., 16].int())

        lane_attr = (
            self_type
            + left_type
            + right_type
            + traffic_light
            + stop_point
            + interpolating
            + stop_sign
        )
        lane_embedding = torch.cat(
            [self_line, left_line, right_line, speed_limit, lane_attr], dim=-1
        )

        # process
        output = self.pointnet(lane_embedding)

        return output


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

    def forward(self, inputs):
        output = self.point_net(inputs)

        return output


# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)
        self.transformer = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
        )

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)

        return output


class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)]
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output


# Transformer-based encoders
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, activation="relu", batch_first=True
        )
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, mask=None):
        output = self.interaction_net(inputs, src_key_padding_mask=mask)

        return output


class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        self.crosswalk_attention = CrossTransformer()
        self.map_attention = MultiModalTransformer()

    def forward(self, actor, lanes, crosswalks, mask):
        query = actor.unsqueeze(1)
        lanes_actor = [
            self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])
        ]
        crosswalks_actor = [
            self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i])
            for i in range(crosswalks.shape[1])
        ]
        map_actor = torch.cat(lanes_actor + crosswalks_actor, dim=1)
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

        return map_actor, output


# Decoders
class AgentDecoder(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps
        self.decode = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps * 3)
        )

    def transform(self, prediction, current_state):
        x = current_state[:, 0]
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x
        new_y = y.unsqueeze(1) + delta_y
        new_theta = theta.unsqueeze(1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj

    def forward(self, agent_map, agent_agent, current_state):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
        decoded = self.decode(feature).view(-1, 3, 10, self._future_steps, 3)
        trajs = torch.stack(
            [
                self.transform(decoded[:, i, j], current_state[:, j])
                for i in range(3)
                for j in range(10)
            ],
            dim=1,
        )
        trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))

        return trajs


class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.control = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps * 2)
        )
        self.cost = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1)
        )
        self.register_buffer("scale", torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer("constraint", torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        actions = self.control(feature).view(-1, 3, self._future_steps, 2)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat(
            [self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1
        )

        return actions, cost_function_weights


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.decode = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def forward(self, map_feature, agent_agent, agent_map):
        # pooling
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, 3, 1), agent_map.detach()], dim=-1)
        scores = self.decode(feature).squeeze(-1)

        return scores


# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps):
        super(Predictor, self).__init__()
        self._future_steps = future_steps

        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()

        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # decode layers
        self.plan = AVDecoder(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)
        self.score = Score()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        # actors
        ego_actor = self.vehicle_net(ego)
        vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1)
        pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1)
        cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
        neighbor_actors = torch.where(
            neighbors[:, :, -1, -1].unsqueeze(2) == 2, pedestrians, vehicles
        )
        neighbor_actors = torch.where(
            neighbors[:, :, -1, -1].unsqueeze(2) == 3, cyclists, neighbor_actors
        )
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[
            :, :, -1, -1
        ]

        # maps
        lane_feature = self.lane_net(map_lanes)
        crosswalk_feature = self.crosswalk_net(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False  # prevent nan

        # actor to actor
        agent_agent = self.agent_agent(actors, actor_mask)

        # map to actor
        map_feature = []
        agent_map = []
        for i in range(actors.shape[1]):
            output = self.agent_map(
                agent_agent[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i]
            )
            map_feature.append(output[0])
            agent_map.append(output[1])

        map_feature = torch.stack(map_feature, dim=1)
        agent_map = torch.stack(agent_map, dim=2)

        # plan + prediction
        plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, 0])
        predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        scores = self.score(map_feature, agent_agent, agent_map)

        return plans, predictions, scores, cost_function_weights


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------

_HIST_LEN = 20
_FUTURE_LEN = 50
_NUM_NEIGHBORS = 10


def build_dipp():
    return Predictor(future_steps=_FUTURE_LEN)


def _make_map_lanes(batch):
    # LaneEncoder consumes columns: [0:3]=self_line, [3:6]=left_line, [6:9]=right_line,
    # [9]=speed_limit, [10]=self_type (0-3), [11]=left_type (0-10), [12]=right_type (0-10),
    # [13]=traffic_light_type (0-8), [14]=stop_point (0-1), [15]=interpolating (0-1),
    # [16]=stop_sign (0-1). Categorical columns must stay within their nn.Embedding bounds.
    shape = (batch, _NUM_NEIGHBORS + 1, 6, 100, 17)
    map_lanes = torch.zeros(shape)
    map_lanes[..., :10] = torch.randn(batch, _NUM_NEIGHBORS + 1, 6, 100, 10)
    map_lanes[..., 10] = torch.randint(0, 4, shape[:-1]).float()
    map_lanes[..., 11] = torch.randint(0, 11, shape[:-1]).float()
    map_lanes[..., 12] = torch.randint(0, 11, shape[:-1]).float()
    map_lanes[..., 13] = torch.randint(0, 9, shape[:-1]).float()
    map_lanes[..., 14] = torch.randint(0, 2, shape[:-1]).float()
    map_lanes[..., 15] = torch.randint(0, 2, shape[:-1]).float()
    map_lanes[..., 16] = torch.randint(0, 2, shape[:-1]).float()
    return map_lanes


def example_input_dipp():
    batch = 1
    ego = torch.randn(batch, _HIST_LEN, 8)
    neighbors = torch.randn(batch, _NUM_NEIGHBORS, _HIST_LEN, 9)
    map_lanes = _make_map_lanes(batch)
    map_crosswalks = torch.randn(batch, _NUM_NEIGHBORS + 1, 4, 100, 3)
    return (ego, neighbors, map_lanes, map_crosswalks)


MENAGERIE_ENTRIES = [
    ("DIPP", "build_dipp", "example_input_dipp", 2023, "vendored-pytorch"),
]
