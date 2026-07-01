# SOURCE: vendored from https://github.com/ChuhuaW/SGNet.pytorch @ main
# (lib/models/SGNet.py, lib/models/feature_extractor.py)
# SGNet: Stepwise Goal-Driven Networks for Trajectory Prediction. RA-L/ICRA 2022.
"""SGNet: stepwise goal-driven trajectory-prediction network (ETH/UCY variant).

Vendored verbatim from ChuhuaW/SGNet.pytorch `lib/models/SGNet.py` (the base,
non-CVAE `SGNet` class) plus its `ETHUCYFeatureExtractor` dependency from
`lib/models/feature_extractor.py`. Architecture is unmodified; only this
header/build/example wrapper and a minimal `_Args` config shim (replacing the
repo's argparse-based config object, which is data/usage, not architecture)
were added for menagerie staging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# lib/models/feature_extractor.py (ETHUCYFeatureExtractor only)
# ---------------------------------------------------------------------------
class ETHUCYFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size), nn.ReLU())

    def forward(self, inputs):
        box_input = inputs
        embedded_box_input = self.embed(box_input)
        return embedded_box_input


def build_feature_extractor(args):
    # Repo dispatches on args.dataset across PIE/JAAD/ETH-UCY extractors; we only
    # vendor the ETH/UCY branch (2D-trajectory forecasting, no image features).
    return ETHUCYFeatureExtractor(args)


# ---------------------------------------------------------------------------
# lib/models/SGNet.py (SGNet base class, ETH/UCY forward branch)
# ---------------------------------------------------------------------------
class SGNet(nn.Module):
    def __init__(self, args):
        super(SGNet, self).__init__()

        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        if self.dataset in ["JAAD", "PIE"]:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim), nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        elif self.dataset in ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim))

        self.enc_goal_attn = nn.Sequential(
            nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True)
        )
        self.dec_goal_attn = nn.Sequential(
            nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True)
        )

        self.enc_to_goal_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4), nn.ReLU(inplace=True)
        )
        self.enc_to_dec_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True)
        )

        self.goal_hidden_to_input = nn.Sequential(
            nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True)
        )
        self.dec_hidden_to_input = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True)
        )
        self.goal_hidden_to_traj = nn.Sequential(
            nn.Linear(self.hidden_size // 4, self.hidden_size), nn.ReLU(inplace=True)
        )
        self.goal_to_enc = nn.Sequential(
            nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True)
        )
        self.goal_to_dec = nn.Sequential(
            nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True)
        )
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)

        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size // 4, self.hidden_size // 4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size // 4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # regress goal traj for loss
            goal_traj[:, dec_step, :] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list], dim=1)
        enc_attn = self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim=1).unsqueeze(1)
        goal_for_enc = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec):
        # initial trajectory tensor
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(
                dec_hidden.size(0), self.dec_steps, self.hidden_size // 4
            )
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:], dim=1)
            goal_dec_input[:, dec_step:, :] = goal_dec_input_temp
            dec_attn = self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim=1).unsqueeze(1)
            goal_dec_input = torch.bmm(dec_attn, goal_dec_input).squeeze(1)

            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input, dec_dec_input), dim=-1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            dec_traj[:, dec_step, :] = self.regressor(dec_hidden)
        return dec_traj

    def encoder(self, traj_input, flow_input=None, start_index=0):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(
            traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim
        )
        all_dec_traj = traj_input.new_zeros(
            traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim
        )
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size // 4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(
                self.enc_drop(torch.cat((traj_input[:, enc_step, :], goal_for_enc), 1)),
                traj_enc_hidden,
            )
            if self.dataset in ["JAAD", "PIE", "ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
                enc_hidden = traj_enc_hidden
            # generate hidden states for goal and decoder
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)

            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            dec_traj = self.decoder(dec_hidden, goal_for_dec)

            # output
            all_goal_traj[:, enc_step, :, :] = goal_traj
            all_dec_traj[:, enc_step, :, :] = dec_traj

        return all_goal_traj, all_dec_traj

    def forward(self, inputs, start_index=0):
        if self.dataset in ["JAAD", "PIE"]:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_dec_traj = self.encoder(traj_input)
            return all_goal_traj, all_dec_traj
        elif self.dataset in ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]:
            traj_input_temp = self.feature_extractor(inputs[:, start_index:, :])
            traj_input = traj_input_temp.new_zeros(
                (inputs.size(0), inputs.size(1), traj_input_temp.size(-1))
            )
            traj_input[:, start_index:, :] = traj_input_temp
            all_goal_traj, all_dec_traj = self.encoder(traj_input, None, start_index)
            return all_goal_traj, all_dec_traj


# ---------------------------------------------------------------------------
# menagerie staging wrapper
# ---------------------------------------------------------------------------
class _Args:
    """Minimal stand-in for the repo's argparse Namespace (config, not architecture).

    Tiny sizes for fast tracing; ETH/UCY branch (pred_dim=2, input_dim=6).
    """

    def __init__(self):
        self.dataset = "ETH"
        self.hidden_size = 32
        self.enc_steps = 4
        self.dec_steps = 3
        self.dropout = 0.0
        self.input_dim = 6


def build_sgnet():
    return SGNet(_Args())


def example_input_sgnet():
    args = _Args()
    batch = 2
    # inputs: (batch, enc_steps, input_dim=6) raw trajectory features (x, y, vx, vy, ...)
    inputs = torch.randn(batch, args.enc_steps, args.input_dim)
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("SGNet", "build_sgnet", "example_input_sgnet", 2022, "vendored-pytorch"),
]
