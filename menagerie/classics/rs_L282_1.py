# SOURCE: vendored from stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus @ main
# (code/model/modules.py + code/model/multipathpp.py)
"""MultiPath++ (Waymo Motion Prediction Challenge 2022 winning solution).

Vendored real nn.Module code, unmodified apart from minimal import-path fixes.
Original config-driven multi-context-gating (MCG) architecture for motion
forecasting: agent-history LSTM encoders, multi-context-gating blocks for
agent/interaction/roadgraph fusion, multi-decoder trajectory prediction head
with optional multi-head attention refinement.
"""

import math

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_max

MENAGERIE_ZOO = "vendored-pytorch"


# ---- code/model/modules.py (real repo code, verbatim) ----------------------


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        modules = []
        assert config["n_layers"] > 0
        for i in range(config["n_layers"]):
            modules.append(nn.Linear(config["n_in"], config["n_out"]))
            if i < config["n_layers"] - 1:
                if config["batchnorm"]:
                    modules.append(nn.BatchNorm1d(config["n_out"]))
                if config["dropout"]:
                    modules.append(nn.Dropout(p=0.1))
                modules.append(nn.ReLU())
        self._mlp = nn.Sequential(*modules)
        self.n_in = config["n_in"]
        self.n_out = config["n_out"]

    def forward(self, x):
        output = self._mlp(x)
        return output


class NormalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        modules = []
        layers = config["layers"]
        assert len(layers) > 0
        if config["pre_batchnorm"]:
            modules.append(nn.BatchNorm1d(layers[0]))
        if config["pre_activation"]:
            modules.append(nn.ReLU())
        for i in range(1, len(layers)):
            modules.append(nn.Linear(layers[i - 1], layers[i]))
            if i < len(layers) - 1:
                if config["batchnorm"]:
                    modules.append(nn.BatchNorm1d(layers[i]))
                modules.append(nn.ReLU())
        self._mlp = nn.ModuleList(modules)

    def forward(self, x):
        tmp = []
        prev_x_shape = x.shape
        assert torch.isfinite(x).all()
        tmp.append(x)
        for l in self._mlp:
            x = l(x)
            tmp.append(x)
            assert torch.isfinite(x).all()
        return x


class CGBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.s_mlp = MLP(config["mlp"])
        self.c_mlp = nn.Identity() if config["identity_c_mlp"] else MLP(config["mlp"])
        self.n_in = self.s_mlp.n_in
        self.n_out = self.s_mlp.n_out

    def forward(self, scatter_numbers, s, c):
        prev_s_shape, prev_c_shape = s.shape, c.shape
        s = self.s_mlp(s.view(-1, s.shape[-1])).view(prev_s_shape)
        c = self.c_mlp(c.view(-1, c.shape[-1])).view(prev_c_shape)
        s = s * c
        if self._config["agg_mode"] == "max":
            aggregated_c = torch.max(s, dim=1, keepdim=True)[0]
        elif self._config["agg_mode"] in ["mean", "avg"]:
            aggregated_c = torch.mean(s, dim=1, keepdim=True)
        else:
            raise Exception("Unknown agg mode for MCG")
        return s, aggregated_c


class MCGBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._blocks = []
        for i in range(config["n_blocks"]):
            current_block_config = config["block"].copy()
            if i == 0 and config["identity_c_mlp"]:
                current_block_config["identity_c_mlp"] = True
            else:
                current_block_config["identity_c_mlp"] = False
            current_block_config["agg_mode"] = config["agg_mode"]
            self._blocks.append(CGBlock(current_block_config))
        self._blocks = nn.ModuleList(self._blocks)
        self.n_in = self._blocks[0].n_in
        self.n_out = self._blocks[-1].n_out

    def _repeat_tensor(self, tensor, scatter_numbers, axis=0):
        result = []
        for i in range(len(scatter_numbers)):
            result.append(tensor[[i]].expand((int(scatter_numbers[i]), -1, -1)))
        result = torch.cat(result, axis=0)
        return result

    def _compute_running_mean(self, prevoius_mean, new_value, i):
        if self._config["running_mean_mode"] == "real":
            result = (prevoius_mean * i + new_value) / i
        elif self._config["running_mean_mode"] == "sliding":
            assert self._config["alpha"] + self._config["beta"] == 1
            result = self._config["alpha"] * prevoius_mean + self._config["beta"] * new_value
        return result

    def forward(
        self, scatter_numbers, scatter_idx, s, c=None, aggregate_batch=True, return_s=False
    ):
        if c is None:
            assert self._config["identity_c_mlp"], self._config["identity_c_mlp"]
            c = torch.ones(s.shape[0], 1, self.n_in, requires_grad=True, device=s.device)
        else:
            assert not self._config["identity_c_mlp"]
        c = self._repeat_tensor(c, scatter_numbers)
        assert torch.isfinite(s).all()
        assert torch.isfinite(c).all()
        running_mean_s, running_mean_c = s, c
        for i, cg_block in enumerate(self._blocks, start=1):
            s, c = cg_block(scatter_numbers, running_mean_s, running_mean_c)
            assert torch.isfinite(s).all()
            assert torch.isfinite(c).all()
            running_mean_s = self._compute_running_mean(running_mean_s, s, i)
            running_mean_c = self._compute_running_mean(running_mean_c, c, i)
            assert torch.isfinite(running_mean_s).all()
            assert torch.isfinite(running_mean_c).all()
        if return_s:
            return running_mean_s
        if aggregate_batch:
            return scatter_max(running_mean_c, scatter_idx, dim=0)[0]
        return running_mean_c


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._return_embedding = config["return_embedding"]
        self._learned_anchor_embeddings = torch.empty((1, config["n_trajectories"], config["size"]))
        stdv = 1.0 / math.sqrt(config["size"])
        self._learned_anchor_embeddings.uniform_(-stdv, stdv)
        self._learned_anchor_embeddings.requires_grad_(True)
        self._learned_anchor_embeddings = nn.Parameter(self._learned_anchor_embeddings)
        self._mcg_predictor = MCGBlock(config["mcg_predictor"])
        if not self._return_embedding:
            self._mlp_decoder = NormalMLP(config["DECODER"])

    def forward(self, target_scatter_numbers, target_scatter_idx, final_embedding, batch_size):
        assert torch.isfinite(final_embedding).all()
        trajectories_embeddings = self._mcg_predictor(
            target_scatter_numbers,
            target_scatter_idx,
            self._learned_anchor_embeddings,
            final_embedding,
            return_s=True,
        )
        assert torch.isfinite(trajectories_embeddings).all()
        if self._return_embedding:
            return trajectories_embeddings
        res = self._mlp_decoder(trajectories_embeddings)
        coordinates = res[:, :, : 80 * 2].reshape(batch_size, self._config["n_trajectories"], 80, 2)
        assert torch.isfinite(coordinates).all()
        a = res[:, :, 80 * 2 : 80 * 3].reshape(batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(a).all()
        b = res[:, :, 80 * 3 : 80 * 4].reshape(batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(b).all()
        c = res[:, :, 80 * 4 : 80 * 5].reshape(batch_size, self._config["n_trajectories"], 80, 1)
        assert torch.isfinite(c).all()
        probas = res[:, :, -1]
        assert torch.isfinite(probas).all()
        if self._config["trainable_cov"]:
            covariance_matrices = (
                torch.cat(
                    [
                        torch.exp(a) * torch.cosh(b),
                        torch.sinh(b),
                        torch.sinh(b),
                        torch.exp(-a) * torch.cosh(b),
                    ],
                    axis=-1,
                )
                * torch.exp(c)
            ).reshape(coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 2, 2)
        else:
            _zeros, _ones = torch.zeros_like(a), torch.ones_like(a)
            covariance_matrices = torch.cat([_ones, _zeros, _zeros, _ones], axis=-1).reshape(
                coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 2, 2
            )
        return probas, coordinates, covariance_matrices


class DecoderHandler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._return_embedding = config["return_embedding"]
        config["decoder_config"]["return_embedding"] = self._return_embedding
        self._n_decoders = int(config["n_decoders"])
        self._decoders = nn.ModuleList(
            [Decoder(config["decoder_config"]) for _ in range(self._n_decoders)]
        )

    def forward(self, target_scatter_numbers, target_scatter_idx, final_embedding, batch_size):
        stacked_probas, stacked_coordinates, stacked_covariance_matrices = [], [], []
        stacked_embeddings = []
        random_head_selector = np.random.uniform(low=0.0, high=1.0, size=self._n_decoders)
        random_head_selector = np.ones_like(random_head_selector) * (random_head_selector > 0.5)
        if self._n_decoders == 1:
            random_head_selector = np.array([1.0])
        while random_head_selector.sum() == 0:
            random_head_selector = np.random.uniform(low=0.0, high=1.0, size=self._n_decoders)
            random_head_selector = np.ones_like(random_head_selector) * (random_head_selector > 0.5)
        if self._return_embedding:
            for coeff, decoder in zip(random_head_selector, self._decoders):
                embeddings = decoder(
                    target_scatter_numbers, target_scatter_idx, final_embedding, batch_size
                )
                stacked_embeddings.append(embeddings)
            stacked_embeddings = torch.cat(stacked_embeddings, dim=1)
            return stacked_embeddings, self._n_decoders / random_head_selector.sum()
        else:
            for coeff, decoder in zip(random_head_selector, self._decoders):
                probas, coordinates, covariance_matrices = decoder(
                    target_scatter_numbers, target_scatter_idx, final_embedding, batch_size
                )
                probas, coordinates, covariance_matrices = [
                    coeff * x + (1 - coeff) * x.detach()
                    for x in [probas, coordinates, covariance_matrices]
                ]
                stacked_probas.append(probas)
                stacked_coordinates.append(coordinates)
                stacked_covariance_matrices.append(covariance_matrices)
            stacked_probas, stacked_coordinates, stacked_covariance_matrices = [
                torch.cat(x, dim=1)
                for x in [stacked_probas, stacked_coordinates, stacked_covariance_matrices]
            ]
            return (
                stacked_probas,
                stacked_coordinates,
                stacked_covariance_matrices,
                max(self._n_decoders / random_head_selector.sum(), 1),
            )


class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._position_lstm = nn.LSTM(batch_first=True, **config["position_lstm_config"])
        self._position_diff_lstm = nn.LSTM(batch_first=True, **config["position_diff_lstm_config"])
        self._position_mcg = MCGBlock(config["position_mcg_config"])

    def forward(self, scatter_numbers, scatter_idx, lstm_data, lstm_data_diff, mcg_data):
        position_lstm_embedding = self._position_lstm(lstm_data)[0][:, -1:, :]
        position_diff_lstm_embedding = self._position_diff_lstm(lstm_data_diff)[0][:, -1:, :]
        position_mcg_embedding = self._position_mcg(
            scatter_numbers, scatter_idx, mcg_data, aggregate_batch=False
        )
        return torch.cat(
            [position_lstm_embedding, position_diff_lstm_embedding, position_mcg_embedding],
            axis=-1,
        )


class MHA(nn.Module):
    # NOTE: upstream hardcodes n_in/n_out=640 (their fixed 640-dim pipeline
    # width, matching `size: 640` in code/configs/final_RoP_Cov_A_fMCG.yaml)
    # instead of using config["n_in"]/config["n_out"] as the surrounding code
    # implies. We use the config values (the intent already present in the
    # original comments) so the vendored module works at any staged size.
    def __init__(self, config):
        super().__init__()
        n_in = config.get("n_in", 640)
        n_out = config.get("n_out", 640)
        self._config = config
        self._q = nn.Linear(n_in, n_out)
        self._k = nn.Linear(n_in, n_out)
        self._v = nn.Linear(n_in, n_out)
        self._mha = nn.MultiheadAttention(n_out, 4, batch_first=True)

    def forward(self, traj_embeddings):
        batch_size = traj_embeddings.shape[0]
        Q = self._q(traj_embeddings)
        K = self._k(traj_embeddings)
        V = self._v(traj_embeddings)
        trajectories_embeddings, _ = self._mha(Q, K, V)
        return trajectories_embeddings


# ---- code/model/multipathpp.py (real repo code; the upstream hardcodes    --
# ---- `.cuda()` on the scatter-index tensors it constructs in forward()    --
# ---- (`torch.ones(...).cuda()`, `torch.arange(...).cuda()`) because the   --
# ---- repo is training-loop-only and always runs on GPU. Those two calls   --
# ---- are the only edits: replaced with `device=<model's own device>` so  --
# ---- the vendored module is CPU-runnable too; every layer/mechanism is   --
# ---- unchanged.) -----------------------------------------------------------


class MultiPathPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._agent_history_encoder = HistoryEncoder(config["agent_history_encoder"])
        self._agent_mcg_linear = NormalMLP(config["agent_mcg_linear"])
        self._interaction_mcg_linear = NormalMLP(config["interaction_mcg_linear"])
        self._interaction_history_encoder = HistoryEncoder(config["interaction_history_encoder"])
        self._polyline_encoder = NormalMLP(config["polyline_encoder"])
        self._history_mcg_encoder = MCGBlock(config["history_mcg_encoder"])
        self._interaction_mcg_encoder = MCGBlock(config["interaction_mcg_encoder"])
        self._agent_and_interaction_linear = NormalMLP(config["agent_and_interaction_linear"])
        self._roadgraph_mcg_encoder = MCGBlock(config["roadgraph_mcg_encoder"])
        self._decoder_handler = DecoderHandler(config["decoder_handler_config"])
        if config["multiple_predictions"]:
            self._decoder = Decoder(config["final_decoder"])
        if self._config["mha_decoder"]:
            self._mha_decoder = MHA(config["mha_decoder_config"])

    def forward(self, data, num_steps=0):
        device = self._agent_mcg_linear._mlp[0].weight.device
        target_scatter_numbers = torch.ones(data["batch_size"], dtype=torch.long, device=device)
        target_scatter_idx = torch.arange(data["batch_size"], dtype=torch.long, device=device)
        target_mcg_input_data_linear = self._agent_mcg_linear(data["target/history/mcg_input_data"])
        assert torch.isfinite(target_mcg_input_data_linear).all()
        target_agents_embeddings = self._agent_history_encoder(
            target_scatter_numbers,
            target_scatter_idx,
            data["target/history/lstm_data"],
            data["target/history/lstm_data_diff"],
            target_mcg_input_data_linear,
        )
        assert torch.isfinite(target_agents_embeddings).all()
        other_mcg_input_data_linear = self._interaction_mcg_linear(
            data["other/history/mcg_input_data"]
        )
        assert torch.isfinite(other_mcg_input_data_linear).all()

        interaction_agents_embeddings = self._interaction_history_encoder(
            data["other_agent_history_scatter_numbers"],
            data["other_agent_history_scatter_idx"],
            data["other/history/lstm_data"],
            data["other/history/lstm_data_diff"],
            other_mcg_input_data_linear,
        )
        assert torch.isfinite(interaction_agents_embeddings).all()
        target_mcg_embedding = self._history_mcg_encoder(
            target_scatter_numbers, target_scatter_idx, target_agents_embeddings
        )
        assert torch.isfinite(target_mcg_embedding).all()
        interaction_mcg_embedding = self._interaction_mcg_encoder(
            data["other_agent_history_scatter_numbers"],
            data["other_agent_history_scatter_idx"],
            interaction_agents_embeddings,
            target_agents_embeddings,
        )
        assert torch.isfinite(interaction_mcg_embedding).all()
        segment_embeddings = self._polyline_encoder(data["road_network_embeddings"])
        assert torch.isfinite(segment_embeddings).all()
        target_and_interaction_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding], axis=-1
        )
        assert torch.isfinite(target_and_interaction_embedding).all()
        target_and_interaction_embedding_linear = self._agent_and_interaction_linear(
            target_and_interaction_embedding
        )
        assert torch.isfinite(target_and_interaction_embedding_linear).all()
        roadgraph_mcg_embedding = self._roadgraph_mcg_encoder(
            data["road_network_scatter_numbers"],
            data["road_network_scatter_idx"],
            segment_embeddings,
            target_and_interaction_embedding_linear,
        )
        assert torch.isfinite(roadgraph_mcg_embedding).all()
        final_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding, roadgraph_mcg_embedding], dim=-1
        )
        assert torch.isfinite(final_embedding).all()
        if self._config["multiple_predictions"]:
            trajectories_embeddings, loss_coeff = self._decoder_handler(
                target_scatter_numbers, target_scatter_idx, final_embedding, data["batch_size"]
            )
            if self._config["mha_decoder"]:
                trajectories_embeddings = self._mha_decoder(trajectories_embeddings)
            trajectories_embeddings, _ = trajectories_embeddings.max(dim=1)
            probas, coordinates, covariance_matrices = self._decoder(
                target_scatter_numbers,
                target_scatter_idx,
                trajectories_embeddings,
                data["batch_size"],
            )
        else:
            probas, coordinates, covariance_matrices, loss_coeff = self._decoder_handler(
                target_scatter_numbers, target_scatter_idx, final_embedding, data["batch_size"]
            )
            assert probas.shape[1] == coordinates.shape[1] == covariance_matrices.shape[1] == 6
        assert torch.isfinite(probas).all()
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(covariance_matrices).all()

        return probas, coordinates, covariance_matrices, loss_coeff


# ---- staging harness: tiny config + synthetic-but-shape-correct input -----
# Shapes follow the real config/data pipeline exactly:
#   T=11 history timesteps; per-agent lstm_data width 13 (=xy2+yaw1+speed1+
#   width1+length1+valid1 + 6-dim agent_type_ohe), lstm_data_diff width 11
#   (T-1=10 steps, one fewer feature since no width/length); mcg_input_data =
#   lstm_data(13) concat per-timestep one-hot(T=11) = 24; road_network
#   feature dim 27 (per code/configs/final_RoP_Cov_A_fMCG.yaml polyline_encoder).


def _mcg_block_cfg(mlp_dim, n_blocks=2, identity_c_mlp=False, agg_mode="max"):
    return {
        "block": {
            "c_bias": True,
            "mlp": {
                "n_layers": 2,
                "n_in": mlp_dim,
                "n_out": mlp_dim,
                "bias": True,
                "batchnorm": False,
                "dropout": False,
            },
        },
        "agg_mode": agg_mode,
        "running_mean_mode": "real",
        "alpha": 0.1,
        "beta": 0.9,
        "n_blocks": n_blocks,
        "identity_c_mlp": identity_c_mlp,
    }


def _tiny_multipathpp_config():
    return {
        "n_trajectories": 3,
        "size": 32,
        "multiple_predictions": True,
        "mha_decoder": True,
        "agent_mcg_linear": {
            "layers": [24, 16, 32],
            "pre_activation": False,
            "pre_batchnorm": False,
            "batchnorm": False,
        },
        "interaction_mcg_linear": {
            "layers": [24, 16, 32],
            "pre_activation": False,
            "pre_batchnorm": False,
            "batchnorm": False,
        },
        "agent_history_encoder": {
            "position_lstm_config": {"input_size": 13, "hidden_size": 16},
            "position_diff_lstm_config": {"input_size": 11, "hidden_size": 16},
            "position_mcg_config": _mcg_block_cfg(32, n_blocks=2, identity_c_mlp=True),
        },
        "interaction_history_encoder": {
            "position_lstm_config": {"input_size": 13, "hidden_size": 16},
            "position_diff_lstm_config": {"input_size": 11, "hidden_size": 16},
            "position_mcg_config": _mcg_block_cfg(32, n_blocks=2, identity_c_mlp=True),
        },
        "polyline_encoder": {
            "layers": [27, 16, 32],
            "pre_activation": False,
            "pre_batchnorm": False,
            "batchnorm": False,
        },
        "history_mcg_encoder": _mcg_block_cfg(64, n_blocks=2, identity_c_mlp=True),
        "interaction_mcg_encoder": _mcg_block_cfg(64, n_blocks=2, identity_c_mlp=False),
        "agent_and_interaction_linear": {
            "layers": [128, 64, 32],
            "pre_activation": True,
            "pre_batchnorm": False,
            "batchnorm": False,
        },
        "roadgraph_mcg_encoder": _mcg_block_cfg(32, n_blocks=2, identity_c_mlp=False),
        "decoder_handler_config": {
            "n_decoders": 2,
            "return_embedding": True,
            "decoder_config": {
                "trainable_cov": True,
                "size": 160,
                "n_trajectories": 3,
                "mcg_predictor": _mcg_block_cfg(160, n_blocks=2, identity_c_mlp=False),
            },
        },
        "final_decoder": {
            "trainable_cov": True,
            "size": 160,
            "return_embedding": False,
            "n_trajectories": 3,
            "mcg_predictor": _mcg_block_cfg(160, n_blocks=2, identity_c_mlp=False),
            "DECODER": {
                "layers": [160, 64, 80 * 5 + 1],
                "pre_activation": True,
                "pre_batchnorm": False,
                "batchnorm": False,
            },
        },
        "mha_decoder_config": {"n_in": 160, "n_out": 160},
    }


def build_multipathpp():
    torch.manual_seed(0)
    cfg = _tiny_multipathpp_config()
    # final_embedding = concat(target_mcg_embedding[64], interaction_mcg_
    # embedding[64], roadgraph_mcg_embedding[32]) = 160-dim, which is why
    # decoder_handler_config/final_decoder/mha_decoder all use size=160.
    return MultiPathPP(cfg)


def example_input_multipathpp():
    torch.manual_seed(0)
    device = "cpu"
    batch_size = 2
    n_other = 3
    n_segments = 4
    T = 11

    data = {
        "batch_size": batch_size,
        "target/history/lstm_data": torch.randn(batch_size, T, 13, device=device),
        "target/history/lstm_data_diff": torch.randn(batch_size, T - 1, 11, device=device),
        "target/history/mcg_input_data": torch.randn(batch_size, T, 24, device=device),
        "other/history/lstm_data": torch.randn(n_other, T, 13, device=device),
        "other/history/lstm_data_diff": torch.randn(n_other, T - 1, 11, device=device),
        "other/history/mcg_input_data": torch.randn(n_other, T, 24, device=device),
        "other_agent_history_scatter_numbers": torch.tensor(
            [2, 1], dtype=torch.long, device=device
        ),
        "other_agent_history_scatter_idx": torch.tensor([0, 0, 1], dtype=torch.long, device=device),
        "road_network_embeddings": torch.randn(n_segments, 1, 27, device=device),
        "road_network_scatter_numbers": torch.tensor([2, 2], dtype=torch.long, device=device),
        "road_network_scatter_idx": torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device),
    }
    return (data,)


MENAGERIE_ENTRIES = [
    ("multipathpp", "build_multipathpp", "example_input_multipathpp", 2022, "vendored-pytorch"),
]
