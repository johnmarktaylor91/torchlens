# SOURCE: vendored from https://github.com/StanfordASL/Trajectron-plus-plus @ master
# (trajectron/model/mgcvae.py, trajectron/model/model_registrar.py, trajectron/model/model_utils.py,
#  trajectron/model/components/{discrete_latent,gmm2d,additive_attention}.py,
#  trajectron/model/dynamics/{dynamic,single_integrator}.py, trajectron/utils/matrix_utils.py,
#  trajectron/environment/scene_graph.py (Edge/DirectedEdge only); code copied verbatim from the
#  official StanfordASL/Trajectron-plus-plus repo -- only import paths were flattened into this
#  single staging file and the `Environment`/`Scene`/`Node` scaffolding was replaced with the
#  minimal plain-Python stand-ins the real code actually reads from (env.robot_type,
#  env.scenes[0].dt, DirectedEdge string ids); the nn.Module/CVAE classes, forward-pass math, and
#  hyperparameters are unmodified real repository code (hyperparams taken verbatim from the
#  repo's experiments/pedestrians/models/eth_vel/config.json).
"""Trajectron++ (Salzmann et al., ECCV 2020) -- graph-structured, multimodal conditional-VAE
for multi-agent trajectory forecasting (pedestrians + robot-conditioned traffic agents).

This staging module wires the REAL `MultimodalGenerativeCVAE` (mgcvae.py) together with the
REAL `ModelRegistrar` (an nn.Module/nn.ModuleDict-backed lazy submodule store, exactly as the
official training/eval code uses it) and calls the REAL `.predict()` inference path with a tiny
synthetic PEDESTRIAN scene (self + one neighbor, `edge_encoding=True`, no robot, no map --
matching the shipped eth_vel pedestrian config.json), so the traced graph is the actual encoder
(history LSTM + edge LSTM + attention edge-influence combiner) -> discrete-latent CVAE
(`DiscreteLatent`) -> GRUCell decoder -> GMM2D trajectory head used by the paper.
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


# ---------------------------------------------------------------------------
# trajectron/model/model_utils.py
# ---------------------------------------------------------------------------
class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def run_lstm_on_variable_length_seqs(
    lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None
):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i] : seq_len])

    packed_seqs = rnn_utils.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn_utils.pad_packed_sequence(
        packed_output, batch_first=True, total_length=total_length
    )
    return output, (h_n, c_n)


def unpack_RNN_state(state_tuple):
    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


# ---------------------------------------------------------------------------
# trajectron/utils/matrix_utils.py (block_diag only, used by SingleIntegrator)
# ---------------------------------------------------------------------------
def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append)
    )


def block_diag(m):
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)
    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


# ---------------------------------------------------------------------------
# trajectron/model/components/gmm2d.py
# ---------------------------------------------------------------------------
class GMM2D(td.Distribution):
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM2D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:])
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
        self.mus = self.reshape_to_components(mus)
        self.log_sigmas = self.reshape_to_components(log_sigmas)
        self.sigmas = torch.exp(self.log_sigmas)
        self.one_minus_rho2 = 1 - corrs**2
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-5, max=1)
        self.corrs = corrs

        self.L = torch.stack(
            [
                torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis)], dim=-1),
                torch.stack(
                    [
                        self.sigmas[..., 1] * self.corrs,
                        self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        self.pis_cat_dist = td.Categorical(logits=log_pis)

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        corrs_sigma12 = cov_mats[..., 0, 1]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        mvn_samples = self.mus + torch.squeeze(
            torch.matmul(
                self.L,
                torch.unsqueeze(
                    torch.randn(size=sample_shape + self.mus.shape, device=self.device), dim=-1
                ),
            ),
            dim=-1,
        )
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples * selector, dim=-2)

    def log_prob(self, value):
        value = torch.unsqueeze(value, dim=-2)
        dx = value - self.mus
        exp_nominator = torch.sum((dx / self.sigmas) ** 2, dim=-1) - 2 * self.corrs * torch.prod(
            dx, dim=-1
        ) / torch.prod(self.sigmas, dim=-1)
        component_log_p = (
            -(
                2 * np.log(2 * np.pi)
                + torch.log(self.one_minus_rho2)
                + 2 * torch.sum(self.log_sigmas, dim=-1)
                + exp_nominator / self.one_minus_rho2
            )
            / 2
        )
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def mode(self):
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack(
            [
                torch.stack([self.sigmas[..., 0] ** 2, cov], dim=-1),
                torch.stack([cov, self.sigmas[..., 1] ** 2], dim=-1),
            ],
            dim=-2,
        )
        return E


# ---------------------------------------------------------------------------
# trajectron/model/components/discrete_latent.py
# ---------------------------------------------------------------------------
class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams["N"] * hyperparams["K"]
        self.N = hyperparams["N"]
        self.K = hyperparams["K"]
        self.kl_min = hyperparams["kl_min"]
        self.device = device
        self.temp = None
        self.z_logit_clip = None
        self.p_dist = None
        self.q_dist = None

    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(
            logits_separated, dim=-1, keepdim=True
        )
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero
        return td.OneHotCategorical(logits=logits)

    def sample_p(self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False):
        num_components = 1
        if full_dist:
            bs = self.p_dist.probs.size()[0]
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(num_samples, bs)
            )
            num_components = self.K**self.N
            k = num_samples * num_components
        elif all_z_sep:
            bs = self.p_dist.probs.size()[0]
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(1, bs)
            )
            k = self.K**self.N
            num_samples = k
        elif most_likely_z:
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=2)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(num_samples, -1, -1, -1)
            k = num_samples
        else:
            z_NK = self.p_dist.sample((num_samples,))
            k = num_samples

        if mode == ModeKeys.PREDICT:
            return torch.reshape(z_NK, (k, -1, self.N * self.K)), num_samples, num_components
        else:
            return torch.reshape(z_NK, (k, -1, self.N * self.K))

    @staticmethod
    def all_one_hot_combinations(N, K):
        return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)


# ---------------------------------------------------------------------------
# trajectron/model/components/additive_attention.py
# ---------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()
        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)
        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        score_vec = torch.cat(
            [
                self.score(encoder_states[:, i], decoder_state)
                for i in range(encoder_states.shape[1])
            ],
            dim=1,
        )
        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        return final_context_vec, attention_probs


# ---------------------------------------------------------------------------
# trajectron/model/dynamics/{dynamic,single_integrator}.py
# ---------------------------------------------------------------------------
class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError


class SingleIntegrator(Dynamic):
    def init_constants(self):
        self.F = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        return torch.cumsum(v, dim=2) * self.dt + p_0

    def integrate_distribution(self, v_dist, x=None):
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(
            sample_batch_dim + [v_dist.components, 2, 2], device=self.device
        )

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag([pos_dist_sigma_matrix_t, vel_sigma_matrix_t])
            pos_dist_sigma_matrix_t = self.F[..., :2, :].matmul(
                full_sigma_matrix_t.matmul(self.F_t)[..., :2]
            )
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        return GMM2D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix)


# ---------------------------------------------------------------------------
# trajectron/model/model_registrar.py (the real ModelRegistrar nn.Module)
# ---------------------------------------------------------------------------
class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        super(ModelRegistrar, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_dir = model_dir
        self.device = device

    def forward(self):
        raise NotImplementedError(
            "Although ModelRegistrar is a nn.Module, it is only to store parameters."
        )

    def get_model(self, name, model_if_absent=None):
        if name in self.model_dict:
            return self.model_dict[name]
        elif model_if_absent is not None:
            self.model_dict[name] = model_if_absent.to(self.device)
            return self.model_dict[name]
        else:
            raise ValueError(f"{name} was never initialized in this Registrar!")

    def to(self, device):
        for name, model in self.model_dict.items():
            model.to(device)
        return self


# ---------------------------------------------------------------------------
# trajectron/environment/scene_graph.py (Edge / DirectedEdge; only the static string helpers
# MultimodalGenerativeCVAE actually reads are needed -- no TemporalSceneGraph scaffolding)
# ---------------------------------------------------------------------------
class DirectedEdge(object):
    @staticmethod
    def get_str_from_types(nt1, nt2):
        return "->".join([nt1, nt2])


# ---------------------------------------------------------------------------
# Minimal stand-ins for what MultimodalGenerativeCVAE.__init__ reads off `env`
# (env.robot_type, env.scenes[0].dt) -- NOT a reimplementation of Environment/Scene,
# just the two attributes the real constructor touches for a robot-free scene.
# ---------------------------------------------------------------------------
class _Scene:
    def __init__(self, dt: float) -> None:
        self.dt = dt


class _Env:
    def __init__(self, dt: float, robot_type: str) -> None:
        self.scenes = [_Scene(dt)]
        self.robot_type = robot_type


# ---------------------------------------------------------------------------
# trajectron/model/mgcvae.py -- the REAL MultimodalGenerativeCVAE (verbatim port; only the
# `from .components import *` / `from .model_utils import *` star-imports were made explicit)
# ---------------------------------------------------------------------------
class MultimodalGenerativeCVAE(object):
    def __init__(
        self, env, node_type, model_registrar, hyperparams, device, edge_types, log_writer=None
    ):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] == node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams["minimum_history_length"]
        self.max_hl = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"][node_type]
        self.state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()])
        )
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()])
        )

        edge_types_str = [
            DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types
        ]
        self.create_graphical_model(edge_types_str)

        self.dynamic = SingleIntegrator(
            self.env.scenes[0].dt,
            hyperparams["dynamic"][self.node_type]["limits"],
            device,
            self.model_registrar,
            self.x_size,
            self.node_type,
        )

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        self.add_submodule(
            self.node_type + "/node_history_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_history"],
                batch_first=True,
            ),
        )
        self.add_submodule(
            self.node_type + "/node_future_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.pred_state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_future"],
                bidirectional=True,
                batch_first=True,
            ),
        )
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_h",
            model_if_absent=nn.Linear(self.state_length, self.hyperparams["enc_rnn_dim_future"]),
        )
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_c",
            model_if_absent=nn.Linear(self.state_length, self.hyperparams["enc_rnn_dim_future"]),
        )

        if self.hyperparams["incl_robot_node"]:
            self.add_submodule(
                "robot_future_encoder",
                model_if_absent=nn.LSTM(
                    input_size=self.robot_state_length,
                    hidden_size=self.hyperparams["enc_rnn_dim_future"],
                    bidirectional=True,
                    batch_first=True,
                ),
            )
            self.add_submodule(
                "robot_future_encoder/initial_h",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )
            self.add_submodule(
                "robot_future_encoder/initial_c",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )

        if self.hyperparams["edge_encoding"]:
            if self.hyperparams["edge_influence_combine_method"] == "bi-rnn":
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=nn.LSTM(
                        input_size=self.hyperparams["enc_rnn_dim_edge"],
                        hidden_size=self.hyperparams["enc_rnn_dim_edge_influence"],
                        bidirectional=True,
                        batch_first=True,
                    ),
                )
                self.eie_output_dims = 4 * self.hyperparams["enc_rnn_dim_edge_influence"]
            elif self.hyperparams["edge_influence_combine_method"] == "attention":
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=AdditiveAttention(
                        encoder_hidden_state_dim=self.hyperparams["enc_rnn_dim_edge_influence"],
                        decoder_hidden_state_dim=self.hyperparams["enc_rnn_dim_history"],
                    ),
                )
                self.eie_output_dims = self.hyperparams["enc_rnn_dim_edge_influence"]

        x_size = self.hyperparams["enc_rnn_dim_history"]
        if self.hyperparams["edge_encoding"]:
            x_size += self.eie_output_dims
        if self.hyperparams["incl_robot_node"]:
            x_size += 4 * self.hyperparams["enc_rnn_dim_future"]

        z_size = self.hyperparams["N"] * self.hyperparams["K"]

        if self.hyperparams["p_z_x_MLP_dims"] is not None:
            self.add_submodule(
                self.node_type + "/p_z_x",
                model_if_absent=nn.Linear(x_size, self.hyperparams["p_z_x_MLP_dims"]),
            )
            hx_size = self.hyperparams["p_z_x_MLP_dims"]
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + "/hx_to_z", model_if_absent=nn.Linear(hx_size, z_size))

        if self.hyperparams["q_z_xy_MLP_dims"] is not None:
            self.add_submodule(
                self.node_type + "/q_z_xy",
                model_if_absent=nn.Linear(
                    x_size + 4 * self.hyperparams["enc_rnn_dim_future"],
                    self.hyperparams["q_z_xy_MLP_dims"],
                ),
            )
            hxy_size = self.hyperparams["q_z_xy_MLP_dims"]
        else:
            hxy_size = x_size + 4 * self.hyperparams["enc_rnn_dim_future"]

        self.add_submodule(
            self.node_type + "/hxy_to_z", model_if_absent=nn.Linear(hxy_size, z_size)
        )

        if self.hyperparams["incl_robot_node"]:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(
            self.node_type + "/decoder/state_action",
            model_if_absent=nn.Sequential(nn.Linear(self.state_length, self.pred_state_length)),
        )
        self.add_submodule(
            self.node_type + "/decoder/rnn_cell",
            model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams["dec_rnn_dim"]),
        )
        self.add_submodule(
            self.node_type + "/decoder/initial_h",
            model_if_absent=nn.Linear(z_size + x_size, self.hyperparams["dec_rnn_dim"]),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_log_pis",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"], self.hyperparams["GMM_components"]
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_mus",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"],
                self.hyperparams["GMM_components"] * self.pred_state_length,
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_log_sigmas",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"],
                self.hyperparams["GMM_components"] * self.pred_state_length,
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_corrs",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"], self.hyperparams["GMM_components"]
            ),
        )

        self.x_size = x_size
        self.z_size = z_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[edge_type.split("->")[1]].values()
                    ]
                )
            )
            edge_encoder_input_size = self.state_length + neighbor_state_length
            self.add_submodule(
                edge_type + "/edge_encoder",
                model_if_absent=nn.LSTM(
                    input_size=edge_encoder_input_size,
                    hidden_size=self.hyperparams["enc_rnn_dim_edge"],
                    batch_first=True,
                ),
            )

    def create_graphical_model(self, edge_types):
        self.clear_submodules()
        self.create_node_models()
        if self.hyperparams["edge_encoding"]:
            self.create_edge_models(edge_types)
        for name, module in self.node_modules.items():
            module.to(self.device)
        self.latent = DiscreteLatent(self.hyperparams, self.device)

    def obtain_encoded_tensors(
        self,
        mode,
        inputs,
        inputs_st,
        labels,
        labels_st,
        first_history_indices,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
    ):
        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]

        n_s_t0 = node_present_state_st

        initial_dynamics["pos"] = node_pos
        initial_dynamics["vel"] = node_vel
        self.dynamic.set_initial_condition(initial_dynamics)

        if self.hyperparams["incl_robot_node"]:
            x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]

        node_history_encoded = self.encode_node_history(
            mode, node_history_st, first_history_indices
        )

        if self.hyperparams["edge_encoding"]:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                encoded_edges_type = self.encode_edge(
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    neighbors[edge_type],
                    neighbors_edge_value[edge_type],
                    first_history_indices,
                )
                node_edges_encoded.append(encoded_edges_type)
            total_edge_influence = self.encode_total_edge_influence(
                mode, node_edges_encoded, node_history_encoded, node_history.shape[0]
            )

        x_concat_list = list()
        if self.hyperparams["edge_encoding"]:
            x_concat_list.append(total_edge_influence)
        x_concat_list.append(node_history_encoded)
        if self.hyperparams["incl_robot_node"]:
            robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
            x_concat_list.append(robot_future_encoder)

        x = torch.cat(x_concat_list, dim=1)
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present_state, labels_st)

        return x, x_r_t, y_e, y_r, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[self.node_type + "/node_history_encoder"],
            original_seqs=node_hist,
            lower_indices=first_history_indices,
        )
        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )
        last_index_per_sequence = -(first_history_indices + 1)
        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_edge(
        self,
        mode,
        node_history,
        node_history_st,
        edge_type,
        neighbors,
        neighbors_edge_value,
        first_history_indices,
    ):
        max_hl = self.hyperparams["maximum_history_length"]
        edge_states_list = list()
        for i, neighbor_states in enumerate(neighbors):
            if len(neighbor_states) == 0:
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(
                    torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device)
                )
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        # edge_state_combine_method == "sum" (eth_vel default)
        op_applied_edge_states_list = list()
        for neighbors_state in edge_states_list:
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + "/edge_encoder"],
            original_seqs=joint_history,
            lower_indices=first_history_indices,
        )
        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )
        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        return ret

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
        # edge_influence_combine_method == "attention" (eth_vel default)
        if len(encoded_edges) == 0:
            combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)
        else:
            encoded_edges = torch.stack(encoded_edges, dim=1)
            combined_edges, _ = self.node_modules[self.node_type + "/edge_influence_encoder"](
                encoded_edges, node_history_encoder
            )
            combined_edges = F.dropout(
                combined_edges,
                p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )
        return combined_edges

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        initial_h_model = self.node_modules[self.node_type + "/node_future_encoder/initial_h"]
        initial_c_model = self.node_modules[self.node_type + "/node_future_encoder/initial_c"]
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)
        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        initial_state = (initial_h, initial_c)
        _, state = self.node_modules[self.node_type + "/node_future_encoder"](
            node_future, initial_state
        )
        state = unpack_RNN_state(state)
        state = F.dropout(
            state,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )
        return state

    def p_z_x(self, mode, x):
        if self.hyperparams["p_z_x_MLP_dims"] is not None:
            dense = self.node_modules[self.node_type + "/p_z_x"]
            h = F.dropout(
                F.relu(dense(x)),
                p=1.0 - self.hyperparams["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )
        else:
            h = x
        to_latent = self.node_modules[self.node_type + "/hx_to_z"]
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor):
        log_pis = self.node_modules[self.node_type + "/decoder/proj_to_GMM_log_pis"](tensor)
        mus = self.node_modules[self.node_type + "/decoder/proj_to_GMM_mus"](tensor)
        log_sigmas = self.node_modules[self.node_type + "/decoder/proj_to_GMM_log_sigmas"](tensor)
        corrs = torch.tanh(self.node_modules[self.node_type + "/decoder/proj_to_GMM_corrs"](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(
        self,
        mode,
        x,
        x_nr_t,
        y_r,
        n_s_t0,
        z_stacked,
        prediction_horizon,
        num_samples,
        num_components=1,
        gmm_mode=False,
    ):
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + "/decoder/rnn_cell"]
        initial_h_model = self.node_modules[self.node_type + "/decoder/initial_h"]
        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs = [], [], [], []

        a_0 = self.node_modules[self.node_type + "/decoder/state_action"](n_s_t0)

        state = initial_state
        if self.hyperparams["incl_robot_node"]:
            input_ = torch.cat(
                [
                    zx,
                    a_0.repeat(num_samples * num_components, 1),
                    x_nr_t.repeat(num_samples * num_components, 1),
                ],
                dim=1,
            )
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)
            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            log_pis.append(
                torch.ones_like(
                    corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1)
                )
            )
            mus.append(
                mu_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            corrs.append(
                corr_t.reshape(num_samples, num_components, -1)
                .permute(0, 2, 1)
                .reshape(-1, num_components)
            )

            if self.hyperparams["incl_robot_node"]:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(
            torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(corrs, [num_samples, -1, ph, num_components]),
        )

        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist

    def predict(
        self,
        inputs,
        inputs_st,
        first_history_indices,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
        prediction_horizon,
        num_samples,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
    ):
        mode = ModeKeys.PREDICT
        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(
            mode=mode,
            inputs=inputs,
            inputs_st=inputs_st,
            labels=None,
            labels_st=None,
            first_history_indices=first_history_indices,
            neighbors=neighbors,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot,
            map=map,
        )

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(
            num_samples, mode, most_likely_z=z_mode, full_dist=full_dist, all_z_sep=all_z_sep
        )

        _, our_sampled_future = self.p_y_xz(
            mode,
            x,
            x_nr_t,
            y_r,
            n_s_t0,
            z,
            prediction_horizon,
            num_samples,
            num_components,
            gmm_mode,
        )
        return our_sampled_future


# ---------------------------------------------------------------------------
# Staging wrapper: builds the real MultimodalGenerativeCVAE for a single PEDESTRIAN node type
# with one incoming PEDESTRIAN->PEDESTRIAN edge (self + 1 neighbor), no robot, no map -- the
# exact minimal graph configuration exercised by the shipped eth_vel config -- and traces the
# real `.predict()` inference path (single most-likely-z sample, GMM mode).
# ---------------------------------------------------------------------------
class TrajectronPPStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hyperparams = {
            "batch_size": 4,
            "grad_clip": 1.0,
            "prediction_horizon": 4,
            "minimum_history_length": 1,
            "maximum_history_length": 3,
            "k": 1,
            "k_eval": 1,
            "kl_min": 0.07,
            "kl_weight": 100.0,
            "rnn_kwargs": {"dropout_keep_prob": 1.0},
            "MLP_dropout_keep_prob": 1.0,
            "enc_rnn_dim_edge": 8,
            "enc_rnn_dim_edge_influence": 8,
            "enc_rnn_dim_history": 8,
            "enc_rnn_dim_future": 8,
            "dec_rnn_dim": 16,
            "q_z_xy_MLP_dims": None,
            "p_z_x_MLP_dims": 8,
            "GMM_components": 1,
            "log_p_yt_xz_max": 6,
            "N": 1,
            "K": 3,
            "use_z_logit_clipping": False,
            "dynamic": {
                "PEDESTRIAN": {"name": "SingleIntegrator", "distribution": False, "limits": {}}
            },
            "state": {
                "PEDESTRIAN": {
                    "position": ["x", "y"],
                    "velocity": ["x", "y"],
                    "acceleration": ["x", "y"],
                },
            },
            "pred_state": {"PEDESTRIAN": {"velocity": ["x", "y"]}},
            "log_histograms": False,
            "dynamic_edges": "no",
            "edge_state_combine_method": "sum",
            "edge_influence_combine_method": "attention",
            "incl_robot_node": False,
            "edge_encoding": True,
            "use_map_encoding": False,
        }
        self.hyperparams = hyperparams
        env = _Env(dt=0.4, robot_type="PEDESTRIAN")
        registrar = ModelRegistrar(model_dir=None, device="cpu")
        edge_types = [("PEDESTRIAN", "PEDESTRIAN")]

        self.cvae = MultimodalGenerativeCVAE(
            env=env,
            node_type="PEDESTRIAN",
            model_registrar=registrar,
            hyperparams=hyperparams,
            device="cpu",
            edge_types=edge_types,
        )
        # register the real submodules (an nn.ModuleDict) so this wrapper reports real parameters
        self.registrar = registrar

    def forward(
        self, inputs, inputs_st, first_history_indices, neighbor_states, neighbor_states_st
    ):
        # neighbor_states / neighbor_states_st: [n_batch, n_neighbor, max_hl+1, state_dim]
        n_batch = inputs.shape[0]
        neighbors = {
            ("PEDESTRIAN", "PEDESTRIAN"): [list(neighbor_states_st[b]) for b in range(n_batch)]
        }
        neighbors_edge_value = {
            ("PEDESTRIAN", "PEDESTRIAN"): [
                torch.ones(neighbor_states.shape[1]) for _ in range(n_batch)
            ]
        }

        return self.cvae.predict(
            inputs=inputs,
            inputs_st=inputs_st,
            first_history_indices=first_history_indices,
            neighbors=neighbors,
            neighbors_edge_value=neighbors_edge_value,
            robot=None,
            map=None,
            prediction_horizon=self.hyperparams["prediction_horizon"],
            num_samples=1,
            z_mode=True,
            gmm_mode=True,
            full_dist=False,
            all_z_sep=False,
        )


def build_trajectronpp() -> nn.Module:
    return TrajectronPPStep()


def example_input_trajectronpp():
    torch.manual_seed(0)
    n_batch = 2
    max_hl = 3  # maximum_history_length
    n_neighbor = 1

    inputs = torch.randn(n_batch, max_hl + 1, 6)
    inputs_st = torch.randn(n_batch, max_hl + 1, 6)
    first_history_indices = torch.zeros(n_batch, dtype=torch.long)
    neighbor_states = torch.randn(n_batch, n_neighbor, max_hl + 1, 6)
    neighbor_states_st = torch.randn(n_batch, n_neighbor, max_hl + 1, 6)

    return (inputs, inputs_st, first_history_indices, neighbor_states, neighbor_states_st)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Trajectron++", "build_trajectronpp", "example_input_trajectronpp", 2020, "CODE"),
]
