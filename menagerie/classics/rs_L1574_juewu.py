# SOURCE: vendored from https://github.com/tencent-ailab/hok_env @ master
#   (aiarena/1v1/common/config.py: DimConfig/Config; aiarena/1v1/common/algorithm_torch.py:
#    Algorithm/MLP/make_fc_layer; aiarena/1v1/actor/model.py: the `lstm_time_steps=1` inference
#    override applied by get_model_class())
#
# JueWu (Wei Fu et al. / Tencent AI Lab, "Honor of Kings" 1v1 MOBA RL agent; the JueWu-SL /
# actor-critic policy described in arxiv 1912.09729 was not itself released, as noted by the
# menagerie queue). Tencent DID release, in this repo's `aiarena` competition harness, the
# actual official 1v1 baseline policy/value network used to compete against JueWu-family agents
# in the Honor-of-Kings AI Arena -- the real per-entity (hero/soldier/organ) feature-MLP +
# max-pool aggregation + LSTM + multi-label action-head + target-attention architecture that
# defines this whole model family. This is that real network, vendored verbatim (PyTorch
# backend; `Config.backend` also supports an equivalent TensorFlow path in the same repo, not
# used here). Trimmed: `Algorithm.compute_loss` (training-loss-only, not part of the forward
# architecture) and `Algorithm.format_data` (a thin numpy/tensor reshape helper for the RL data
# pipeline) are omitted; this file constructs `[feature_vec, legal_action, lstm_initial_state]`
# directly and calls the real `forward(data_list, inference=True)` branch, matching exactly what
# `format_data(..., inference=True)` would hand it. `MLP`/`make_fc_layer` are the real generic
# building blocks (also used, unmodified in the original, by the 3v3 variant of this codebase).

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

MENAGERIE_ZOO = "vendored-pytorch"


# ---- aiarena/1v1/common/config.py (vendored verbatim) ----
class DimConfig:
    # main camp soldier
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    # enemy camp soldier
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    # main camp organ
    DIM_OF_ORGAN_1_2 = [18, 18]
    # enemy camp organ
    DIM_OF_ORGAN_3_4 = [18, 18]
    # main camp hero
    DIM_OF_HERO_FRD = [235]
    # enemy camp hero
    DIM_OF_HERO_EMY = [235]
    # public hero info
    DIM_OF_HERO_MAIN = [14]  # main_hero_vec

    DIM_OF_GLOBAL_INFO = [25]


class Config:
    backend = os.getenv("AIARENA_BACKEND", "pytorch")
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32
    data_shapes = [
        [12944],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [128],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]


# ---- aiarena/1v1/common/algorithm_torch.py (vendored verbatim, minus compute_loss/format_data) ----
def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer"""
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    """A simple multi-layer perceptron"""

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: type = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_10[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1_2[0])
        self.hero_main_feature_dim = int(DimConfig.DIM_OF_HERO_MAIN[0])
        self.global_feature_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))

        self.all_hero_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD))
            + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
            + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
        )
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_11_20)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)) + int(
            np.sum(DimConfig.DIM_OF_ORGAN_3_4)
        )

        """ hero_main module"""
        fc_hero_main_dim_list = [self.hero_main_feature_dim, 64, 32, 16]
        self.hero_main_mlp = MLP(fc_hero_main_dim_list, "hero_main_mlp")

        """ hero_share module"""
        fc_hero_dim_list = [self.single_hero_feature_dim, 512, 256, 128]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = nn.Sequential(
            OrderedDict(
                [("hero_frd_fc", make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]))]
            )
        )
        self.hero_emy_fc = nn.Sequential(
            OrderedDict(
                [("hero_emy_fc", make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]))]
            )
        )

        """ soldier_share module"""
        fc_soldier_dim_list = [self.single_soldier_feature_dim, 64, 64, 32]
        self.soldier_mlp = MLP(fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_frd_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )
        self.soldier_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_emy_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )

        """ organ_share module"""
        fc_organ_dim_list = [self.single_organ_feature_dim, 64, 64, 32]
        self.organ_mlp = MLP(fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = nn.Sequential(
            OrderedDict(
                [("organ_frd_fc", make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]))]
            )
        )
        self.organ_emy_fc = nn.Sequential(
            OrderedDict(
                [("organ_emy_fc", make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]))]
            )
        )

        """public concat"""
        concat_dim = (
            fc_hero_main_dim_list[-1]
            + 2 * fc_hero_dim_list[-1]
            + 2 * fc_soldier_dim_list[-1]
            + 2 * fc_organ_dim_list[-1]
            + self.global_feature_dim
        )
        fc_concat_dim_list = [concat_dim, 512]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)

        """public lstm"""
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        """output label"""
        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.lstm_unit_size, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        """output value"""
        self.value_mlp = MLP([self.lstm_unit_size, 64, 1], "hero_value_mlp")

        self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        if not inference:
            _, data_list = data_list

        feature_vec, legal_action, lstm_initial_state = data_list

        result_list = []

        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.all_organ_feature_dim,
                self.global_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(
            [
                int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
                int(np.sum(DimConfig.DIM_OF_HERO_MAIN)),
            ],
            dim=1,
        )
        soldier_vec_list = feature_vec_split_list[1].split(
            [
                int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)),
                int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20)),
            ],
            dim=1,
        )
        organ_vec_list = feature_vec_split_list[2].split(
            [
                int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)),
                int(np.sum(DimConfig.DIM_OF_ORGAN_3_4)),
            ],
            dim=1,
        )
        global_info_list = feature_vec_split_list[3]

        _soldier_1_10 = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_10, dim=1)
        _soldier_11_20 = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_11_20, dim=1)

        _organ_1_2 = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1_2, dim=1)
        _organ_3_4 = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_3_4, dim=1)
        _hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        _hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        _hero_main = hero_vec_list[2].split(DimConfig.DIM_OF_HERO_MAIN, dim=1)
        _global_info = global_info_list

        tar_embed_list = []

        # hero_main
        for index in range(len(_hero_main)):
            main_hero = self.hero_main_mlp(_hero_main[index])
        hero_main_result = main_hero

        hero_emy_result_list = []
        for index in range(len(_hero_emy)):
            hero_emy_mlp_out = self.hero_mlp(_hero_emy[index])
            hero_emy_fc_out = self.hero_emy_fc(hero_emy_mlp_out)
            _, split_1 = hero_emy_fc_out.split([96, 32], dim=1)
            tar_embed_list.append(split_1)
            hero_emy_result_list.append(hero_emy_fc_out)

        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)
        reshape_hero_emy = hero_emy_concat_result.reshape(-1, 1, 1, 128)
        pool_hero_emy, _ = reshape_hero_emy.max(dim=2)
        output_dim = int(np.prod(pool_hero_emy.shape[1:]))
        reshape_pool_hero_emy = pool_hero_emy.reshape(-1, output_dim)

        hero_frd_result_list = []
        for index in range(len(_hero_frd)):
            hero_frd_mlp_out = self.hero_mlp(_hero_frd[index])
            hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
            _, split_1 = hero_frd_fc_out.split([96, 32], dim=1)
            tar_embed_list.append(split_1)
            hero_frd_result_list.append(hero_frd_fc_out)

        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)
        reshape_hero_frd = hero_frd_concat_result.reshape(-1, 1, 1, 128)
        pool_hero_frd, _ = reshape_hero_frd.max(dim=2)
        output_dim = int(np.prod(pool_hero_frd.shape[1:]))
        reshape_pool_hero_frd = pool_hero_frd.reshape(-1, output_dim)

        soldier_frd_result_list = []
        for index in range(len(_soldier_1_10)):
            soldier_frd_mlp_out = self.soldier_mlp(_soldier_1_10[index])
            soldier_frd_fc_out = self.soldier_frd_fc(soldier_frd_mlp_out)
            soldier_frd_result_list.append(soldier_frd_fc_out)

        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1)
        reshape_frd_soldier = soldier_frd_concat_result.reshape(-1, 1, 4, 32)
        pool_frd_soldier, _ = reshape_frd_soldier.max(dim=2)
        output_dim = int(np.prod(pool_frd_soldier.shape[1:]))
        reshape_pool_frd_soldier = pool_frd_soldier.reshape(-1, output_dim)

        soldier_emy_result_list = []
        for index in range(len(_soldier_11_20)):
            soldier_emy_mlp_out = self.soldier_mlp(_soldier_11_20[index])
            soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
            soldier_emy_result_list.append(soldier_emy_fc_out)
            tar_embed_list.append(soldier_emy_fc_out)

        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1)
        reshape_emy_soldier = soldier_emy_concat_result.reshape(-1, 1, 4, 32)
        pool_emy_soldier, _ = reshape_emy_soldier.max(dim=2)
        output_dim = int(np.prod(pool_emy_soldier.shape[1:]))
        reshape_pool_emy_soldier = pool_emy_soldier.reshape(-1, output_dim)

        organ_frd_result_list = []
        for index in range(len(_organ_1_2)):
            organ_frd_mlp_out = self.organ_mlp(_organ_1_2[index])
            organ_frd_fc_out = self.organ_frd_fc(organ_frd_mlp_out)
            organ_frd_result_list.append(organ_frd_fc_out)

        organ_1_concat_result = torch.cat(organ_frd_result_list, dim=1)
        reshape_frd_organ = organ_1_concat_result.reshape(-1, 1, 2, 32)
        pool_frd_organ, _ = reshape_frd_organ.max(dim=2)
        output_dim = int(np.prod(pool_frd_organ.shape[1:]))
        reshape_pool_frd_organ = pool_frd_organ.reshape(-1, output_dim)

        organ_emy_result_list = []
        for index in range(len(_organ_3_4)):
            organ_emy_mlp_out = self.organ_mlp(_organ_3_4[index])
            organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
            organ_emy_result_list.append(organ_emy_fc_out)

        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)
        reshape_emy_organ = organ_emy_concat_result.reshape(-1, 1, 2, 32)
        pool_emy_organ, _ = reshape_emy_organ.max(dim=2)
        output_dim = int(np.prod(pool_emy_organ.shape[1:]))
        reshape_pool_emy_organ = pool_emy_organ.reshape(-1, output_dim)
        tar_embed_list.append(reshape_pool_emy_organ)

        tar_embed_0 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, tar_embed_0)

        concat_result = torch.cat(
            [
                reshape_pool_frd_soldier,
                reshape_pool_emy_soldier,
                reshape_pool_frd_organ,
                reshape_pool_emy_organ,
                hero_main_result,
                reshape_pool_hero_frd,
                reshape_pool_hero_emy,
                _global_info,
            ],
            dim=1,
        )

        fc_public_result = self.concat_mlp(concat_result)
        reshape_fc_public_result = fc_public_result.reshape(-1, self.lstm_time_steps, 512)

        lstm_initial_state_in = [
            lstm_initial_state[0].unsqueeze(0),
            lstm_initial_state[1].unsqueeze(0),
        ]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state_in)

        lstm_outputs = torch.cat(
            [lstm_outputs[:, idx, :] for idx in range(lstm_outputs.size(1))], dim=1
        )
        self.lstm_cell_output = state[1]
        self.lstm_hidden_output = state[0]
        reshape_lstm_outputs_result = lstm_outputs.reshape(-1, self.lstm_unit_size)

        for label_index, label_dim in enumerate(self.label_size_list[:-1]):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](
                reshape_lstm_outputs_result
            )
            result_list.append(label_mlp_out)

        lstm_tar_embed_result = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result)

        tar_embedding = torch.stack(tar_embed_list, dim=1)

        ulti_tar_embedding = self.target_embed_mlp(tar_embedding)
        reshape_label_result = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)

        label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)
        target_output_dim = int(np.prod(label_result.shape[1:]))

        reshape_label_result = label_result.reshape(-1, target_output_dim)
        result_list.append(reshape_label_result)

        value_result = self.value_mlp(reshape_lstm_outputs_result)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list


class JueWuModel(Algorithm):
    """aiarena/1v1/actor/model.py:get_model_class() applies exactly this override
    (lstm_time_steps=1) when constructing the deployable inference-time singleton Model."""

    def __init__(self):
        super().__init__()
        self.lstm_time_steps = 1


def build_juewu():
    return JueWuModel()


def example_input_juewu():
    n = 2
    feature_vec = torch.randn(n, 725)
    legal_action = torch.ones(n, 84)
    lstm_hidden = torch.zeros(n, 512)
    lstm_cell = torch.zeros(n, 512)
    data_list = [feature_vec, legal_action, (lstm_hidden, lstm_cell)]
    return (data_list,)


class _InferenceWrapper(nn.Module):
    """Thin wrapper so tracing calls the real forward's inference=True branch (a tensor-list
    output) instead of the training branch, matching how aiarena/1v1/actor/model.py's deployed
    Model is actually invoked at inference time."""

    def __init__(self):
        super().__init__()
        self.model = build_juewu()

    def forward(self, data_list):
        return self.model(data_list, inference=True)


def build_juewu_traceable():
    return _InferenceWrapper()


MENAGERIE_ENTRIES = [
    (
        "JueWu (1v1 aiarena baseline policy net)",
        build_juewu_traceable,
        example_input_juewu,
        2019,
        "REAL",
    ),
]
