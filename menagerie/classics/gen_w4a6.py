"""Menagerie batch w4a6: pedestrian/vehicle trajectory & driving-policy nets.

Sources checked (reference only; no cloning, no pip installs):
  - CIDNN (cand_00372): Xu, Piao & Gao, CVPR 2018, "Encoding Crowd
    Interaction with Deep Neural Network for Pedestrian Trajectory
    Prediction". Paper https://arxiv.org/abs/1806.06216 (catalog notes cite
    1805.06880), official source https://github.com/svip-lab/CIDNN. Each
    pedestrian's observed trajectory is encoded independently by a shared
    per-agent LSTM. A pairwise "motion gate" / spatial-affinity module then
    computes an attention-like weight between every pair of agents from their
    current *relative displacement* (a small MLP over the 2D offset, i.e. a
    location-based, not content-based, affinity -- CIDNN's namesake
    "encoding crowd interaction" mechanism), and each agent's hidden state is
    updated to a weighted sum of all agents' encoder states using that
    displacement-conditioned affinity matrix. The pooled, interaction-aware
    hidden state is then decoded by a second per-agent LSTM into future
    displacement predictions. This module reproduces exactly that shared
    encoder -> displacement-conditioned pairwise affinity pooling -> decoder
    structure.
  - CIL (cand_00373): Codevilla, Muller, Lopez, Koltun & Dosovitskiy,
    ICRA 2018, "End-to-end Driving via Conditional Imitation Learning".
    Paper https://arxiv.org/abs/1710.02410, official source
    https://github.com/carla-simulator/imitation-learning (agent/network.py,
    TensorFlow); widely-cited PyTorch community port
    https://github.com/onlytailei/carla_cil_pytorch mirrors the same graph.
    An 8-conv-layer image branch and a small measurement branch (ego speed)
    are each embedded to a feature vector and concatenated ("joint" branch).
    The fused feature is then fed into *K parallel command-conditioned
    branches* (one small FC head per discrete high-level navigation command
    -- follow lane / turn left / turn right / go straight), each regressing
    the 3 continuous control outputs (steer, throttle, brake); at run time
    only the branch selected by the current navigation command is used. This
    module reproduces the namesake "conditional" branching: a shared
    perception trunk feeding N parallel per-command control heads, gated by
    a one-hot command input.
  - CILRS (cand_00374): Codevilla, Santana, Lopez & Gaidon, ICCV 2019,
    "Exploring the Limitations of Behavior Cloning for Autonomous Driving".
    Paper https://arxiv.org/abs/1904.08980, official source
    https://github.com/felipecode/coiltraine (network/models/CoILICRA.py).
    CILRS extends CIL with (1) a ResNet image backbone (in place of the
    shallow conv stack) whose pooled features feed the same speed branch +
    command-conditioned action-branch structure as CIL, and (2) an added
    *auxiliary speed-prediction head* that regresses ego speed directly from
    the image features alone (no measurement input), used as an auxiliary
    training signal to discourage the network from keying off the
    ground-truth speed input for control. This module reproduces the
    ResNet-stem perception trunk + command-conditioned action branches +
    image-only speed-regression head, matching CILRS's namesake extension
    over CIL ("+ ResNet + Speed").
  - Convolutional Social Pooling / CS-LSTM (cand_00376 and cand_00379 are
    the SAME paper/repo -- Deo & Trivedi, CVPRW 2018, "Convolutional Social
    Pooling for Vehicle Trajectory Prediction", https://arxiv.org/abs/1805.06771,
    official source https://github.com/nachiket92/conv-social-pooling; the
    catalog TSV marks cand_00379 POTENTIAL_DEDUP with cand_00376). Built
    once here as CSLSTM and registered under BOTH catalog names, matching
    the fact they are literally the same architecture: each nearby vehicle's
    track is embedded by a shared LSTM encoder; the encoders' final hidden
    states are scattered into their relative grid cells of a spatial
    occupancy tensor (the ego vehicle's neighborhood, discretized into a
    grid), and a small 2D conv + pool stack ("convolutional social pooling")
    aggregates this spatial tensor into a fixed-size social-context vector
    -- the paper's efficient conv replacement for the social-pooling FC
    layers of Social LSTM. The social-context vector, concatenated with the
    ego encoder state, drives (a) a maneuver classification head (lateral x
    longitudinal maneuver classes) and (b) an LSTM decoder that outputs
    Gaussian-mixture trajectory parameters, conditioned on maneuver.
  - CoverNet (cand_00378): Phan-Minh, Grigore, Boulton, Beijbom & Wolff,
    CVPR 2020, "CoverNet: Multimodal Behavior Prediction using Trajectory
    Sets". Paper https://arxiv.org/abs/1911.10298, official source
    https://github.com/nutonomy/nuscenes-devkit
    (python-sdk/nuscenes/prediction/models/covernet.py). CoverNet reframes
    trajectory prediction as *classification over a fixed, pre-computed
    trajectory set* rather than direct regression: a CNN backbone encodes a
    rasterized bird's-eye-view scene image, its pooled features are
    concatenated with the agent's current state vector, and an MLP head
    outputs a softmax distribution (logits) over K anchor trajectories drawn
    from an epsilon-covering trajectory codebook. This module reproduces
    exactly that raster-CNN + state-vector fusion -> logits-over-fixed-
    trajectory-set structure, with the trajectory codebook stored as a fixed
    (non-trainable) buffer of shape (K, T, 2), matching CoverNet's namesake
    "cover" of the space of plausible future trajectories.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CIDNN -- displacement-conditioned pairwise affinity pooling for
# crowd interaction trajectory prediction
# ============================================================


class CIDNN(nn.Module):
    """CIDNN: LSTM encoder + displacement-gated pairwise interaction pooling.

    Every agent's observed track is embedded independently by a shared LSTM
    encoder. A small MLP ("motion gate") maps each pair of agents' current
    relative displacement to a scalar affinity; softmax-normalizing over
    neighbors gives a spatial-affinity matrix that pools all agents' encoder
    states into an interaction-aware hidden state per agent. A shared LSTM
    decoder then unrolls that pooled state into future displacements.
    """

    def __init__(
        self,
        hidden: int = 16,
        n_future: int = 4,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_future = n_future
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        # Motion gate: maps a pairwise relative-displacement vector (2D) to
        # an unnormalized affinity score -- CIDNN's location-based (not
        # content-based) crowd-interaction attention.
        self.motion_gate = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.pool_proj = nn.Linear(hidden, hidden)
        self.decoder_cell = nn.LSTMCell(input_size=2, hidden_size=hidden)
        self.out_proj = nn.Linear(hidden, 2)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        """Predict future displacements for every agent in a scene.

        Parameters
        ----------
        tracks : torch.Tensor
            Observed 2D positions, shape ``(n_agents, T_obs, 2)``.

        Returns
        -------
        torch.Tensor
            Predicted future displacements, shape
            ``(n_agents, n_future, 2)``.
        """
        n_agents, t_obs, _ = tracks.shape
        enc_out, (h_n, c_n) = self.encoder(tracks)  # (n_agents, T_obs, hidden)
        h_last = h_n[-1]  # (n_agents, hidden) -- per-agent encoder summary

        cur_pos = tracks[:, -1, :]  # (n_agents, 2)
        # Pairwise relative displacement -> motion-gated affinity matrix.
        rel = cur_pos.unsqueeze(1) - cur_pos.unsqueeze(0)  # (n_agents, n_agents, 2)
        affinity_logits = self.motion_gate(rel).squeeze(-1)  # (n_agents, n_agents)
        eye_mask = torch.eye(n_agents, device=tracks.device, dtype=torch.bool)
        affinity_logits = affinity_logits.masked_fill(eye_mask, float("-inf"))
        affinity = F.softmax(affinity_logits, dim=-1)  # (n_agents, n_agents)

        pooled = affinity @ h_last  # (n_agents, hidden) interaction-aware pooling
        state = torch.tanh(self.pool_proj(pooled)) + h_last  # residual fuse w/ self state
        cell = c_n[-1]

        preds = []
        dec_in = tracks[:, -1, :] - tracks[:, -2, :]  # last observed displacement
        h_t, c_t = state, cell
        for _ in range(self.n_future):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            dec_in = self.out_proj(h_t)
            preds.append(dec_in)
        return torch.stack(preds, dim=1)


def build_cidnn() -> nn.Module:
    """Build a small CIDNN crowd-interaction trajectory predictor."""
    return CIDNN(hidden=16, n_future=4).eval()


def example_input_cidnn() -> torch.Tensor:
    """Observed 2D tracks for ``6`` agents over ``8`` timesteps: (6, 8, 2)."""
    return torch.randn(6, 8, 2)


# ============================================================
# CIL -- command-conditioned branched imitation-learning policy
# ============================================================


class CIL(nn.Module):
    """CIL: shallow-conv image branch + speed branch -> command-gated heads.

    An image branch and a scalar-speed branch are each embedded and
    concatenated into a joint feature. ``n_commands`` parallel FC "action"
    branches each regress (steer, throttle, brake); at inference the
    branch selected by a one-hot navigation command is used -- CIL's
    namesake conditional-imitation-learning mechanism.
    """

    def __init__(self, n_commands: int = 4, feat_dim: int = 32) -> None:
        super().__init__()
        self.n_commands = n_commands
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.speed_branch = nn.Sequential(
            nn.Linear(1, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.joint = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
        )
        # N parallel command-conditioned control heads.
        self.action_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_dim // 2, 3),  # steer, throttle, brake
                )
                for _ in range(n_commands)
            ]
        )

    def forward(
        self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """Predict control outputs gated by a one-hot navigation command.

        Parameters
        ----------
        image : torch.Tensor
            RGB frame, shape ``(B, 3, H, W)``.
        speed : torch.Tensor
            Ego speed, shape ``(B, 1)``.
        command : torch.Tensor
            One-hot navigation command, shape ``(B, n_commands)``.

        Returns
        -------
        torch.Tensor
            Selected control outputs ``(steer, throttle, brake)``, shape
            ``(B, 3)``.
        """
        img_feat = self.image_branch(image)
        spd_feat = self.speed_branch(speed)
        joint_feat = self.joint(torch.cat([img_feat, spd_feat], dim=-1))

        branch_outs = torch.stack(
            [branch(joint_feat) for branch in self.action_branches], dim=1
        )  # (B, n_commands, 3)
        selected = (branch_outs * command.unsqueeze(-1)).sum(dim=1)  # gated select
        return selected


def build_cil() -> nn.Module:
    """Build a small CIL command-conditioned imitation-learning policy."""
    return CIL(n_commands=4, feat_dim=32).eval()


def example_input_cil() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB frame ``(2,3,64,64)``, speed ``(2,1)``, one-hot command ``(2,4)``."""
    image = torch.randn(2, 3, 64, 64)
    speed = torch.rand(2, 1)
    command = F.one_hot(torch.tensor([0, 2]), num_classes=4).float()
    return image, speed, command


# ============================================================
# CILRS -- CIL + ResNet backbone + auxiliary speed-prediction head
# ============================================================


class _ResBlock(nn.Module):
    """Minimal pre-activation residual block for the CILRS ResNet stem."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(x + out, inplace=True)


class CILRS(nn.Module):
    """CILRS: ResNet perception trunk + CIL action branches + speed head.

    Extends CIL with a small ResNet-style image backbone in place of the
    shallow conv stack, and adds an auxiliary head that regresses ego speed
    directly from the image features alone (no measurement input) --
    CILRS's namesake "+ ResNet + Speed" extension over CIL, used during
    training to discourage the policy from ignoring the visual scene.
    """

    def __init__(self, n_commands: int = 4, feat_dim: int = 32, stem_ch: int = 16) -> None:
        super().__init__()
        self.n_commands = n_commands
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(stem_ch),
            nn.Conv2d(stem_ch, stem_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch * 2),
            nn.ReLU(inplace=True),
            _ResBlock(stem_ch * 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_proj = nn.Linear(stem_ch * 2, feat_dim)

        self.speed_branch = nn.Sequential(
            nn.Linear(1, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.joint = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.action_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_dim // 2, 3),
                )
                for _ in range(n_commands)
            ]
        )
        # Auxiliary speed-prediction head: image features only.
        self.speed_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(
        self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict control outputs and an auxiliary speed estimate.

        Parameters
        ----------
        image : torch.Tensor
            RGB frame, shape ``(B, 3, H, W)``.
        speed : torch.Tensor
            Ego speed, shape ``(B, 1)``.
        command : torch.Tensor
            One-hot navigation command, shape ``(B, n_commands)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected control outputs ``(B, 3)`` and predicted speed ``(B, 1)``.
        """
        feat_map = self.res_blocks(self.stem(image))
        img_feat = self.img_proj(self.pool(feat_map).flatten(1))

        pred_speed = self.speed_head(img_feat)

        spd_feat = self.speed_branch(speed)
        joint_feat = self.joint(torch.cat([img_feat, spd_feat], dim=-1))
        branch_outs = torch.stack([branch(joint_feat) for branch in self.action_branches], dim=1)
        selected = (branch_outs * command.unsqueeze(-1)).sum(dim=1)
        return selected, pred_speed


def build_cilrs() -> nn.Module:
    """Build a small CILRS ResNet-backbone command-conditioned policy."""
    return CILRS(n_commands=4, feat_dim=32, stem_ch=16).eval()


def example_input_cilrs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB frame ``(2,3,64,64)``, speed ``(2,1)``, one-hot command ``(2,4)``."""
    image = torch.randn(2, 3, 64, 64)
    speed = torch.rand(2, 1)
    command = F.one_hot(torch.tensor([1, 3]), num_classes=4).float()
    return image, speed, command


# ============================================================
# CS-LSTM / Convolutional Social Pooling -- occupancy-grid conv pooling
# for maneuver-conditioned multi-modal vehicle trajectory prediction
# ============================================================


class CSLSTM(nn.Module):
    """CS-LSTM: LSTM track encoders scattered onto a grid + conv social pooling.

    Every nearby vehicle's observed track (including the ego vehicle) is
    embedded by a shared LSTM encoder. Each neighbor's final hidden state is
    scattered into its relative cell of a spatial occupancy grid around the
    ego vehicle; a small 2D conv + pool stack ("convolutional social
    pooling") aggregates that grid into a fixed-size social-context vector,
    replacing the fully-connected social pooling of Social-LSTM with a
    translation-equivariant conv. The social vector, concatenated with the
    ego encoder state, drives a maneuver-classification head (lateral x
    longitudinal classes) and an LSTM decoder that predicts, for every
    future timestep, the parameters of a bivariate Gaussian over position.
    """

    def __init__(
        self,
        hidden: int = 16,
        grid_h: int = 5,
        grid_w: int = 5,
        n_future: int = 5,
        n_lat: int = 3,
        n_lon: int = 2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_future = n_future
        self.n_lat = n_lat
        self.n_lon = n_lon

        self.track_encoder = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        self.social_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.maneuver_head = nn.Linear(hidden * 2, n_lat + n_lon)
        self.decoder_cell = nn.LSTMCell(input_size=2, hidden_size=hidden)
        self.dec_init = nn.Linear(hidden * 2 + n_lat + n_lon, hidden)
        # Bivariate Gaussian params per future step: mu_x, mu_y, sigma_x, sigma_y, rho.
        self.gauss_head = nn.Linear(hidden, 5)

    def forward(
        self, neighbor_tracks: torch.Tensor, grid_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict a maneuver class and a multi-modal future trajectory.

        Parameters
        ----------
        neighbor_tracks : torch.Tensor
            Observed 2D tracks of the ego vehicle (index 0) and its
            neighbors, shape ``(n_agents, T_obs, 2)``.
        grid_idx : torch.Tensor
            Integer (row, col) grid cell for every agent relative to the
            ego vehicle, shape ``(n_agents, 2)``, with ``grid_idx[0]``
            being the ego vehicle's own cell.

        Returns
        -------
        dict[str, torch.Tensor]
            ``maneuver_logits`` ``(n_lat + n_lon,)`` and ``gauss_params``
            ``(n_future, 5)``.
        """
        n_agents = neighbor_tracks.shape[0]
        _, (h_n, _) = self.track_encoder(neighbor_tracks)  # (1, n_agents, hidden)
        h_last = h_n[-1]  # (n_agents, hidden)

        # Scatter neighbor encoder states onto the spatial occupancy grid.
        grid = h_last.new_zeros(1, self.hidden, self.grid_h, self.grid_w)
        rows = grid_idx[:, 0].clamp(0, self.grid_h - 1)
        cols = grid_idx[:, 1].clamp(0, self.grid_w - 1)
        for i in range(n_agents):
            grid[0, :, rows[i], cols[i]] = h_last[i]

        social_ctx = self.social_conv(grid).flatten(1)  # (1, hidden)
        ego_state = h_last[0:1]  # (1, hidden)
        fused = torch.cat([ego_state, social_ctx], dim=-1)  # (1, 2*hidden)

        maneuver_logits = self.maneuver_head(fused).squeeze(0)  # (n_lat+n_lon,)

        dec_h = torch.tanh(self.dec_init(torch.cat([fused, maneuver_logits.unsqueeze(0)], dim=-1)))
        dec_c = torch.zeros_like(dec_h)
        dec_in = neighbor_tracks[0, -1, :] - neighbor_tracks[0, -2, :]
        dec_in = dec_in.unsqueeze(0)

        outputs = []
        h_t, c_t = dec_h, dec_c
        for _ in range(self.n_future):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            gauss = self.gauss_head(h_t)  # (1, 5)
            outputs.append(gauss.squeeze(0))
            dec_in = gauss[:, :2]  # feed predicted mean position forward
        gauss_params = torch.stack(outputs, dim=0)  # (n_future, 5)

        return {"maneuver_logits": maneuver_logits, "gauss_params": gauss_params}


def build_cslstm() -> nn.Module:
    """Build a small CS-LSTM convolutional-social-pooling trajectory predictor."""
    return CSLSTM(hidden=16, grid_h=5, grid_w=5, n_future=5, n_lat=3, n_lon=2).eval()


def example_input_cslstm() -> tuple[torch.Tensor, torch.Tensor]:
    """Ego + 5 neighbor tracks ``(6, 8, 2)`` and their grid indices ``(6, 2)``."""
    tracks = torch.randn(6, 8, 2)
    grid_idx = torch.tensor([[2, 2], [1, 2], [3, 2], [2, 1], [2, 3], [0, 4]], dtype=torch.long)
    return tracks, grid_idx


# ============================================================
# CoverNet -- classification over a fixed trajectory codebook
# ============================================================


class CoverNet(nn.Module):
    """CoverNet: raster-scene CNN + state vector -> logits over a fixed
    trajectory set.

    Reframes multimodal trajectory prediction as classification: a CNN
    encodes a rasterized bird's-eye-view scene, its pooled features are
    fused with the agent's current-state vector, and an MLP outputs a
    softmax distribution over K anchor trajectories drawn from a
    pre-computed epsilon-covering trajectory codebook (stored here as a
    fixed, non-trainable buffer) -- CoverNet's namesake reframing of
    trajectory regression as trajectory-set classification.
    """

    def __init__(
        self,
        n_trajectories: int = 12,
        n_future: int = 6,
        state_dim: int = 3,
        feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + state_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, n_trajectories),
        )
        # Fixed epsilon-covering trajectory set (non-trainable): the "cover"
        # of plausible future trajectories that CoverNet classifies over.
        self.register_buffer("trajectory_set", torch.randn(n_trajectories, n_future, 2) * 5.0)

    def forward(self, raster: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Score a fixed trajectory codebook given a rasterized scene + state.

        Parameters
        ----------
        raster : torch.Tensor
            Rasterized BEV scene image, shape ``(B, 3, H, W)``.
        state : torch.Tensor
            Agent current-state vector (e.g. velocity, accel, heading rate),
            shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Logits over the fixed trajectory set, shape
            ``(B, n_trajectories)``.
        """
        img_feat = self.backbone(raster)
        fused = torch.cat([img_feat, state], dim=-1)
        return self.head(fused)


def build_covernet() -> nn.Module:
    """Build a small CoverNet trajectory-set classifier."""
    return CoverNet(n_trajectories=12, n_future=6, state_dim=3, feat_dim=32).eval()


def example_input_covernet() -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterized BEV scene ``(2,3,64,64)`` and agent state ``(2,3)``."""
    raster = torch.randn(2, 3, 64, 64)
    state = torch.randn(2, 3)
    return raster, state


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("CIDNN", "build_cidnn", "example_input_cidnn", "2018", "VIS"),
    ("CIL", "build_cil", "example_input_cil", "2018", "RL"),
    ("CILRS", "build_cilrs", "example_input_cilrs", "2019", "RL"),
    ("Convolutional Social Pooling", "build_cslstm", "example_input_cslstm", "2018", "VIS"),
    ("CS-LSTM", "build_cslstm", "example_input_cslstm", "2018", "VIS"),
    ("CoverNet", "build_covernet", "example_input_covernet", "2020", "VIS"),
]
