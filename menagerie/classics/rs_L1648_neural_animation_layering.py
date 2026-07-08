# FAITHFUL REIMPLEMENTATION from "Neural Animation Layering for Synthesizing
# Martial Arts Movements" (Starke, Zhao, Zinno, Komura; ACM TOG / SIGGRAPH
# 2021, https://doi.org/10.1145/3450626.3459881) -- no public code. The
# sebastianstarke/AI4Animation repo (queue's cited source) ships only the
# paper PDF + Unity media assets for SIGGRAPH_2021 (Media/SIGGRAPH_2021/);
# the martial-arts motion generator training/inference code was never open
# sourced.
#
# The paper (Sec. 4 "Motion Generator", Fig. 2) specifies the architecture
# precisely: a mixture-of-experts motion generator composed of (1) a Gating
# Network that maps a low/high-frequency joint-velocity signal to blending
# weights B over a pool of K=8 experts, and (2) a Pose Prediction Network
# whose per-layer weights are formed by blending the K experts' weights with
# B, then applied to the control-series input to predict the next pose. Both
# subnetworks have 2 hidden layers with ELU activation; gating hidden size
# 128, pose-predictor hidden size 512, 8 expert weight sets, dropout 0.3 on
# the gating network (Sec. 7 "Network Training").
#
# This is the same mixture-of-experts family the same lead author (Starke)
# used in MANN (SIGGRAPH 2018) and later open-sourced verbatim in
# AI4Animation/SIGGRAPH_2022/Unity/.../MixtureOfExpertsNetwork.cs -- that
# real, published inference code (gating: 2 ELU hidden layers + softmax ->
# blending weights; expert blending: per-sample weighted sum of K experts'
# per-layer {weight, bias} tensors; main network: blended-weight 2-hidden-
# layer ELU MLP) is used here to fix the exact MoE composition mechanics
# faithfully, with the SIGGRAPH-2021 paper's own stated dims (gating
# hidden=128, pose-predictor hidden=512, 8 experts, dropout=0.3, ELU
# activation) driving every architectural hyperparameter.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Gating network: 2 ELU hidden layers -> softmax blending weights over
    `num_experts`, per the paper's "gating network... hidden layer size in
    the gating network is set to 128... dropout rate is set to 0.3" and the
    same-author MixtureOfExpertsNetwork.cs inference (GW0/GW1/GW2 3-layer
    ELU/ELU/SoftMax gating stack)."""

    def __init__(self, input_size, num_experts, hidden_size=128, dropout=0.3):
        super().__init__()
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(x)
        h = F.elu(self.fc0(h))
        h = self.dropout(h)
        h = F.elu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return F.softmax(h, dim=-1)


class PosePredictionNetwork(nn.Module):
    """Pose predictor: per-layer weights are blended across `num_experts`
    expert weight tensors using the gating network's blending weights, then
    applied as a 2-hidden-layer ELU MLP, per the paper's "pose predictor
    network is constructed by blending the weights of a fixed number of
    structurally identical networks, called experts, according to a set of
    learned blending weights" (Sec. 4) with hidden size 512 and 8 experts,
    matching the expert-blend mechanics of MixtureOfExpertsNetwork.cs
    (Matrix.BlendAll over per-expert {weight, bias} banks)."""

    def __init__(self, input_size, output_size, num_experts, hidden_size=512):
        super().__init__()
        self.num_experts = num_experts
        self.w0 = nn.Parameter(
            torch.empty(num_experts, hidden_size, input_size).uniform_(-0.1, 0.1)
        )
        self.b0 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.w1 = nn.Parameter(
            torch.empty(num_experts, hidden_size, hidden_size).uniform_(-0.1, 0.1)
        )
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.w2 = nn.Parameter(
            torch.empty(num_experts, output_size, hidden_size).uniform_(-0.1, 0.1)
        )
        self.b2 = nn.Parameter(torch.zeros(num_experts, output_size))

    def forward(self, x, blending_weights):
        # blending_weights: (batch, num_experts) -> per-sample blended
        # {weight, bias} tensors for each of the 3 layers.
        bw0 = torch.einsum("be,eho->bho", blending_weights, self.w0)
        bb0 = torch.einsum("be,eh->bh", blending_weights, self.b0)
        bw1 = torch.einsum("be,eho->bho", blending_weights, self.w1)
        bb1 = torch.einsum("be,eh->bh", blending_weights, self.b1)
        bw2 = torch.einsum("be,eho->bho", blending_weights, self.w2)
        bb2 = torch.einsum("be,eh->bh", blending_weights, self.b2)

        h = torch.einsum("bho,bo->bh", bw0, x) + bb0
        h = F.elu(h)
        h = torch.einsum("bho,bo->bh", bw1, h) + bb1
        h = F.elu(h)
        y = torch.einsum("bho,bo->bh", bw2, h) + bb2
        return y


class NeuralAnimationLayeringMotionGenerator(nn.Module):
    """Full motion-generator network from Fig. 2: the current control series
    (concatenated key-joint trajectories, dim `control_size`) drives both the
    gating network (whose softmax output blends the K pose-predictor
    experts) and the pose-predictor network itself, producing the next-frame
    full-body pose (dim `pose_size`)."""

    def __init__(
        self,
        control_size=253,
        pose_size=200,
        num_experts=8,
        gating_hidden=128,
        pose_hidden=512,
        dropout=0.3,
    ):
        super().__init__()
        self.gating = GatingNetwork(
            control_size, num_experts, hidden_size=gating_hidden, dropout=dropout
        )
        self.pose_predictor = PosePredictionNetwork(
            control_size, pose_size, num_experts, hidden_size=pose_hidden
        )

    def forward(self, control_series):
        blending_weights = self.gating(control_series)
        pose = self.pose_predictor(control_series, blending_weights)
        return pose


# --- staging harness: build + example input ---------------------------------


def build_neural_animation_layering():
    # tiny sizes for a fast trace; control_size/pose_size are toy stand-ins
    # for the paper's actual control-series / full-body-pose feature dims
    # (Sec. 4 gives C_i as N=11 key-joint trajectories over a 1s window plus
    # the root trajectory; pose output includes joint positions/rotations/
    # velocities, finger transforms, and contact states). num_experts=8,
    # gating_hidden=128, pose_hidden=512 match the paper's stated dims
    # exactly; here we shrink pose_hidden for a fast trace since the
    # architecture (not the width) is what is being captured.
    return NeuralAnimationLayeringMotionGenerator(
        control_size=64,
        pose_size=48,
        num_experts=8,
        gating_hidden=32,
        pose_hidden=64,
        dropout=0.0,
    )


def example_input_neural_animation_layering():
    batch_size = 4
    control_size = 64
    return torch.randn(batch_size, control_size)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Neural Animation Layering",
        build_neural_animation_layering,
        example_input_neural_animation_layering,
        2021,
        MENAGERIE_ZOO,
    ),
]
