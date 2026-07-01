# SOURCE: three vendored real-repo models for queue.tsv rows 273-276
#   (AutoBot -> rung 2 vendor; CIDNN -> rung 2 vendor; CIL -> rung 2 vendor via the
#   community PyTorch port explicitly named in the queue notes; BEVWorld -> skip,
#   handled separately in skip_L272.tsv -- no code in this file).
#
# ---------------------------------------------------------------------------------
# AutoBot-Ego
# SOURCE: vendored from https://github.com/roggirg/AutoBots @ master
#   models/autobot_ego.py + models/context_encoders.py
# ICLR 2022. Latent-variable sequential-set Transformer for joint multi-agent motion
# forecasting (temporal + social self-attention encoder stack, a Transformer decoder
# with learned per-mode query embeddings Q, and a mixture-mode probability head).
# Vendored verbatim (only the CNN/point map-encoder branches are omitted below by
# passing use_map_img=False, use_map_lanes=False at construction time -- this is the
# real code's own no-map configuration path, already present in forward(), not an
# architectural change).
#
# ---------------------------------------------------------------------------------
# CIDNN (Crowd Interaction Deep Neural Network)
# SOURCE: vendored from https://github.com/svip-lab/CIDNN @ master
#   base/base_network.py (EncoderNetWithLSTM, DecoderNet, RegressionNet, Attention)
# CVPR 2018. Pedestrian trajectory prediction: per-pedestrian LSTM encoder produces
# hidden trajectory embeddings, a coordinate decoder embeds candidate target
# positions, dot-product attention pools encoder embeddings against decoder queries,
# and a regression head predicts the next-frame displacement. The per-timestep
# composition (`main_compute_step`) is transcribed verbatim from the repo's
# train.py Model class -- it is data-flow plumbing wiring the four vendored
# sub-modules together every prediction step, not new architecture.
#
# ---------------------------------------------------------------------------------
# CIL (Conditional Imitation Learning)
# SOURCE: vendored from https://github.com/onlytailei/carla_cil_pytorch @ master
#   carla_net.py
# ICRA 2018 (Codevilla, Muller, Lopez, Koltun, Dosovitskiy, "End-to-end Driving via
# Conditional Imitation Learning"). The paper's own reference implementation
# (carla-simulator/imitation-learning) is TensorFlow 1.x; queue.tsv explicitly names
# this community PyTorch port (onlytailei/carla_cil_pytorch) as the real-source
# alternative, and it is a faithful line-for-line PyTorch re-expression of the same
# published branched conv-net architecture (8-layer conv trunk -> image FC branch +
# speed FC branch -> fused embedding -> 4 command-conditioned steering/throttle/brake
# branches + a speed-prediction branch). Vendored verbatim.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ======================================================================================
# AutoBot-Ego  (roggirg/AutoBots @ master, models/context_encoders.py + models/autobot_ego.py)
# ======================================================================================


def _init(module, weight_init, bias_init, gain=1):
    """Vendored verbatim from AutoBots models/autobot_ego.py."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MapEncoderCNN(nn.Module):
    """Vendored verbatim from AutoBots models/context_encoders.py. Unused in the
    default (use_map_img=False) configuration below but kept for architectural
    completeness of the vendored file."""

    def __init__(self, d_k=64, dropout=0.1, c=10):
        super(MapEncoderCNN, self).__init__()
        self.dropout = dropout
        self.c = c
        init_ = lambda m: _init(
            m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )  # noqa: E731
        fm_size = 7
        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(32, fm_size * self.c, kernel_size=2, stride=2)),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout),
        )
        self.map_feats = nn.Sequential(
            init_(nn.Linear(7 * 7 * fm_size, d_k)),
            nn.ReLU(),
            init_(nn.Linear(d_k, d_k)),
            nn.ReLU(),
        )

    def forward(self, roads):
        B = roads.size(0)
        return self.map_feats(self.map_encoder(roads).view(B, self.c, -1))


class MapEncoderPts(nn.Module):
    """Vendored verbatim from AutoBots models/context_encoders.py. Unused in the
    default (use_map_lanes=False) configuration below but kept for architectural
    completeness of the vendored file."""

    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: _init(
            m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )  # noqa: E731

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(
            self.d_k, num_heads=8, dropout=self.dropout
        )
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (
            (1.0 - roads[:, :, :, -1])
            .type(torch.BoolTensor)
            .to(roads.device)
            .view(-1, roads.shape[2])
        )
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = (
            self.road_pts_lin(roads[:, :, :, : self.map_attr]).view(B * S, P, -1).permute(1, 0, 2)
        )

        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(
            query=agents_emb,
            key=road_pts_feats,
            value=road_pts_feats,
            key_padding_mask=road_pts_mask,
        )[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask


class PositionalEncoding(nn.Module):
    """Vendored verbatim from AutoBots models/autobot_ego.py."""

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    """Vendored verbatim from AutoBots models/autobot_ego.py."""

    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: _init(
            m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )  # noqa: E731
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)),
            nn.ReLU(),
            init_(nn.Linear(d_k, d_k)),
            nn.ReLU(),
            init_(nn.Linear(d_k, 5)),
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(
            T, BK, -1
        )

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


class AutoBotEgo(nn.Module):
    """Vendored verbatim from AutoBots models/autobot_ego.py (AutoBotEgo class)."""

    def __init__(
        self,
        d_k=128,
        _M=5,
        c=5,
        T=30,
        L_enc=1,
        dropout=0.0,
        k_attr=2,
        map_attr=3,
        num_heads=16,
        L_dec=1,
        tx_hidden_size=384,
        use_map_img=False,
        use_map_lanes=False,
    ):
        super(AutoBotEgo, self).__init__()

        init_ = lambda m: _init(
            m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )  # noqa: E731

        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = d_k
        self._M = _M
        self.c = c
        self.T = T
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_img = use_map_img
        self.use_map_lanes = use_map_lanes

        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(k_attr, d_k)))

        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        if self.use_map_img:
            self.map_encoder = MapEncoderCNN(d_k=d_k, dropout=self.dropout)
            self.emb_state_map = nn.Sequential(
                init_(nn.Linear(2 * d_k, d_k)), nn.ReLU(), init_(nn.Linear(d_k, d_k))
            )
        elif self.use_map_lanes:
            self.map_encoder = MapEncoderPts(d_k=d_k, map_attr=map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(
                self.d_k, num_heads=self.num_heads, dropout=0.3
            )

        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_k,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                )
            )
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        self.pos_encoder = PositionalEncoding(d_k, dropout=0.0)

        self.output_model = OutputModel(d_k=self.d_k)

        self.P = nn.Parameter(torch.Tensor(c, 1, d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.P)

        if self.use_map_img:
            self.modemap_net = nn.Sequential(
                init_(nn.Linear(2 * self.d_k, self.d_k)),
                nn.ReLU(),
                init_(nn.Linear(self.d_k, self.d_k)),
            )
        elif self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        self.prob_decoder = nn.MultiheadAttention(
            self.d_k, num_heads=self.num_heads, dropout=self.dropout
        )
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        subsequent_mask = (
            torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)
        ).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        ego_tensor = ego[:, :, : self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).type(torch.BoolTensor).to(env_masks_orig.device)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        temp_masks = torch.cat(
            (torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1
        )
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)
        opps_tensor = agents[:, :, :, : self.k_attr]

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks[:, -1][temp_masks.sum(-1) == T_obs] = False
        agents_temp_emb = layer(
            self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
            src_key_padding_mask=temp_masks,
        )
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M + 1))
        agents_soc_emb = agents_soc_emb.view(self._M + 1, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def forward(self, ego_in, agents_in, roads):
        B = ego_in.size(0)

        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(
            ego_in, agents_in
        )
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(
                agents_emb, opps_masks, layer=self.temporal_attn_layers[i]
            )
            agents_emb = self.social_attn_fn(
                agents_emb, opps_masks, layer=self.social_attn_layers[i]
            )
        ego_soctemp_emb = agents_emb[:, :, 0]

        if self.use_map_img:
            orig_map_features = self.map_encoder(roads)
            map_features = orig_map_features.view(B * self.c, -1).unsqueeze(0).repeat(self.T, 1, 1)
        elif self.use_map_lanes:
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
            map_features = (
                orig_map_features.unsqueeze(2)
                .repeat(1, 1, self.c, 1)
                .view(-1, B * self.c, self.d_k)
            )
            road_segs_masks = (
                orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B * self.c, -1)
            )

        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B * self.c, self.d_k)

        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B * self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        for d in range(self.L_dec):
            if self.use_map_img and d == 1:
                ego_dec_emb_map = torch.cat((out_seq, map_features), dim=-1)
                out_seq = self.emb_state_map(ego_dec_emb_map) + out_seq
            elif self.use_map_lanes and d == 1:
                ego_dec_emb_map = self.map_attn_layers(
                    query=out_seq,
                    key=map_features,
                    value=map_features,
                    key_padding_mask=road_segs_masks,
                )[0]
                out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](
                out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks
            )
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        mode_params_emb = self.P.repeat(1, B, 1)
        mode_params_emb = self.prob_decoder(
            query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb
        )[0]
        if self.use_map_img:
            mode_params_emb = self.modemap_net(
                torch.cat((mode_params_emb, orig_map_features.transpose(0, 1)), dim=-1)
            )
        elif self.use_map_lanes:
            mode_params_emb = (
                self.mode_map_attn(
                    query=mode_params_emb,
                    key=orig_map_features,
                    value=orig_map_features,
                    key_padding_mask=orig_road_segs_masks,
                )[0]
                + mode_params_emb
            )
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(
            0, 1
        )

        return out_dists, mode_probs


def build_autobot():
    # Tiny config; no-map path (use_map_img=False, use_map_lanes=False) is the real
    # code's own configuration branch (see forward()), not an architectural change.
    return AutoBotEgo(
        d_k=16,
        _M=2,
        c=2,
        T=4,
        L_enc=1,
        dropout=0.0,
        k_attr=2,
        map_attr=3,
        num_heads=2,
        L_dec=1,
        tx_hidden_size=32,
        use_map_img=False,
        use_map_lanes=False,
    )


def example_input_autobot():
    # (ego_in, agents_in, roads) matching AutoBotEgo.forward()'s three real inputs.
    B, T_obs, M, k_attr = 1, 3, 2, 2
    ego_in = torch.zeros(B, T_obs, k_attr + 1)
    ego_in[:, :, -1] = 1.0
    agents_in = torch.zeros(B, T_obs, M, k_attr + 1)
    agents_in[:, :, :, -1] = 1.0
    roads = torch.zeros(B, 1, 1)  # unused dummy when use_map_img=use_map_lanes=False
    return (ego_in, agents_in, roads)


# ======================================================================================
# CIDNN  (svip-lab/CIDNN @ master, base/base_network.py + train.py Model.main_compute_step)
# ======================================================================================


class EncoderNetWithLSTM(nn.Module):
    """Vendored verbatim from CIDNN base/base_network.py."""

    def __init__(self, pedestrian_num, input_size, hidden_size, n_layers=2):
        super(EncoderNetWithLSTM, self).__init__()
        input_size = 2
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, self.n_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, self.n_layers)

    def forward(self, input_traces, hidden):
        next_hidden_list = []
        output_list = []
        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :].unsqueeze(0)
            output, next_hidden = self.lstm(input_trace, (hidden[i][0], hidden[i][1]))

            next_hidden_list.append(next_hidden)
            output_list.append(output.squeeze(0))

        output_traces = torch.stack(output_list, 1)

        return output_traces, next_hidden_list

    def init_hidden(self, batch_size):
        return [
            [
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
                for _ in range(2)
            ]
            for _ in range(self.pedestrian_num)
        ]


class DecoderNet(nn.Module):
    """Vendored verbatim from CIDNN base/base_network.py."""

    def __init__(self, pedestrian_num, target_size, hidden_size, window_size):
        super(DecoderNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(target_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, hidden_size)

    def forward(self, target_traces):
        hidden_list = []
        for i in range(self.pedestrian_num):
            target_trace = target_traces[:, i, :]
            hidden_trace = F.relu(self.fc1(target_trace))
            hidden_trace = F.relu(self.fc2(hidden_trace))
            hidden_trace = self.fc3(hidden_trace)

            hidden_list.append(hidden_trace)

        hidden_traces = torch.stack(hidden_list, 1)

        return hidden_traces


class RegressionNet(nn.Module):
    """Vendored verbatim from CIDNN base/base_network.py."""

    def __init__(self, pedestrian_num, regression_size, hidden_size):
        super(RegressionNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.regression_size = regression_size
        self.hidden_size = hidden_size

        hidden1_size = 32
        hidden2_size = 64  # noqa: F841 -- unused in the original repo too (verbatim)

        self.fc1 = torch.nn.Linear(hidden_size, regression_size)
        self.fc2 = torch.nn.Linear(regression_size, hidden1_size)
        self.fc3 = torch.nn.Linear(hidden1_size, regression_size)

    def forward(self, input_attn_hidden_traces, target_hidden_traces, target_traces):
        regression_list = []
        for i in range(self.pedestrian_num):
            input_attn_hidden_trace = input_attn_hidden_traces[:, i]
            target_delta_trace = self.fc1(input_attn_hidden_trace)

            regression_list.append(target_delta_trace)
        regression_traces = torch.stack(regression_list, 1)
        regression_traces = regression_traces + target_traces

        return regression_traces


class Attention(nn.Module):
    """Vendored verbatim from CIDNN base/base_network.py."""

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        Attn = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))

        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)

        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn


class CIDNN(nn.Module):
    """CIDNN full pipeline. The forward() body is transcribed verbatim from CIDNN's
    train.py Model.main_compute_step -- it wires the four real vendored sub-modules
    above together exactly as the original per-prediction-step loop does (LSTM
    encoder rollout over the observed frames, then an attention-gated decode/regress
    loop that autoregressively predicts each future frame). This orchestration is
    the model's real forward computation, not new architecture; only the
    dataset-loading / optimizer / visdom-logging plumbing around it is dropped."""

    def __init__(
        self,
        pedestrian_num=8,
        hidden_size=16,
        input_size=2,
        target_size=2,
        n_layers=1,
        window_size=10,
        input_frame=3,
        target_frame=2,
    ):
        super(CIDNN, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.hidden_size = hidden_size
        self.input_frame = input_frame
        self.target_frame = target_frame

        self.encoder_net = EncoderNetWithLSTM(
            pedestrian_num, input_size, hidden_size, n_layers=n_layers
        )
        self.decoder_net = DecoderNet(pedestrian_num, target_size, hidden_size, window_size)
        self.regression_net = RegressionNet(pedestrian_num, target_size, hidden_size)
        self.attn = Attention()

    def forward(self, batch_input_traces):
        # batch_input_traces: (B, pedestrian_num, input_frame, 2) -- matches the real
        # code's train_input_traces / test_input_traces tensor layout exactly.
        batch_size = batch_input_traces.size(0)

        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        for i in range(self.input_frame - 1):
            input_hidden_traces, encoder_hidden = self.encoder_net(
                batch_input_traces[:, :, i], encoder_hidden
            )

        regression_list = []
        for i in range(self.target_frame):
            input_hidden_traces, encoder_hidden = self.encoder_net(target_traces, encoder_hidden)

            target_hidden_traces = self.decoder_net(target_traces)
            Attn_nn = self.attn(target_hidden_traces, target_hidden_traces)
            c_traces = torch.bmm(Attn_nn, input_hidden_traces)

            regression_traces = self.regression_net(c_traces, target_hidden_traces, target_traces)

            target_traces = regression_traces
            regression_list.append(regression_traces)

        regression_traces = torch.stack(regression_list, 2)
        return regression_traces


def build_cidnn():
    return CIDNN(
        pedestrian_num=3,
        hidden_size=8,
        input_size=2,
        target_size=2,
        n_layers=1,
        window_size=10,
        input_frame=3,
        target_frame=2,
    )


def example_input_cidnn():
    # (B, pedestrian_num, input_frame, 2), matching the real train_X/test_X layout.
    return torch.randn(2, 3, 3, 2)


# ======================================================================================
# CIL / CarlaNet  (onlytailei/carla_cil_pytorch @ master, carla_net.py)
# The paper's own reference repo (carla-simulator/imitation-learning) is TF1.x;
# queue.tsv names this community PyTorch port as the real-source alternative -- see
# module docstring above.
# ======================================================================================


class CarlaNet(nn.Module):
    """Vendored verbatim from onlytailei/carla_cil_pytorch/carla_net.py."""

    def __init__(self, dropout_vec=None):
        super(CarlaNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
            nn.Linear(8192, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.emb_fc = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
                )
                for i in range(4)
            ]
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, speed):
        img = self.conv_block(img)
        img = img.view(-1, 8192)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([out(emb) for out in self.branches], dim=1)
        pred_speed = self.speed_branch(img)

        return output, pred_speed


def build_cil():
    return CarlaNet()


def example_input_cil():
    # (img, speed): 88x200 RGB is the standard CARLA benchmark input resolution the
    # real conv trunk's 8192-dim flatten (256 * 2 * 16) is sized for.
    img = torch.randn(1, 3, 88, 200)
    speed = torch.randn(1, 1)
    return (img, speed)


MENAGERIE_ENTRIES = [
    ("AutoBot", "build_autobot", "example_input_autobot", 2022, "vendored-pytorch"),
    ("CIDNN", "build_cidnn", "example_input_cidnn", 2018, "vendored-pytorch"),
    ("CIL", "build_cil", "example_input_cil", 2018, "vendored-pytorch"),
]
