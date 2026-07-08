# SOURCE: vendored from https://github.com/yjb767868009/NSM_pytorch @ master
#
# PyTorch reimplementation of the Neural State Machine (NSM) character-animation
# controller (Starke, Zheng, Komura, Saito, SIGGRAPH 2019). The original
# authors ship only a Unity/C# runtime (sebastianstarke/AI4Animation); this
# repo is the community PyTorch training reimplementation. Vendored verbatim
# (only import paths flattened for standalone use) from model/network/Encoder.py,
# model/network/Expert.py, and model/utils/activation_layer.py -- the real
# gating-network + motion-network mixture-of-experts architecture: per-channel
# Encoder MLPs project (trajectory/goal/interaction/gating) inputs into a
# shared "status" embedding, a gating Expert produces softmax blending weights
# over expert-network components, and a second (blended-weight) Expert MLP
# consumes the concatenated status + blend weights to regress the next pose.
import numpy as np
import torch
import torch.nn as nn

activation_layer_list = {
    "elu": nn.ELU(),
    "softmax": nn.Softmax(dim=1),
    "None": None,
}


def activation_layer(s):
    return activation_layer_list.get(s)


class Encoder(nn.Module):
    """Vendored from model/network/Encoder.py -- per-channel status encoder."""

    def __init__(self, encoder_dims, encoder_activations, encoder_dropout):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.encoder_dropout = encoder_dropout
        self.layer_nums = len(encoder_dims) - 1

        self.layer1 = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            activation_layer(encoder_activations[0]),
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            activation_layer(encoder_activations[1]),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Expert(nn.Module):
    """Vendored from model/network/Expert.py -- weight-blended expert MLP.

    Used both as the gating network (softmax blend weights over expert
    components) and as the motion network (blended-weight regression of the
    output pose), matching the real Model.train()/test() call pattern.
    """

    def __init__(self, expert_nums, expert_dims, expert_activations, expert_dropout):
        super().__init__()
        self.expert_dims = expert_dims
        self.expert_activations = expert_activations
        self.expert_dropout = expert_dropout
        self.expert_nums = expert_nums
        self.layer_nums = len(expert_dims) - 1

        self.W = nn.ParameterList()
        self.B = nn.ParameterList()
        self.D = []
        self.A = []
        for i in range(self.layer_nums):
            w = self.init_weight((self.expert_nums, self.expert_dims[i + 1], self.expert_dims[i]))
            self.W.append(nn.Parameter(w))
            b = torch.zeros(self.expert_nums, self.expert_dims[i + 1], 1)
            self.B.append(nn.Parameter(b))
            self.D.append(nn.Dropout(p=expert_dropout))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, weight_blend, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            x = x.unsqueeze(-1)
            weight = self.get_wb(self.W[i], weight_blend)
            t = torch.bmm(weight, x)
            bias = self.get_wb(self.B[i], weight_blend)
            x = torch.add(t, bias)
            x = x.squeeze(-1)
            if self.A[i]:
                x = self.A[i](x)
        return x

    def init_weight(self, shape):
        a = np.sqrt(6.0 / np.prod(shape[-2:]))
        w = np.asarray(np.random.uniform(low=-a, high=a, size=shape), dtype=np.float32)
        return torch.Tensor(w)

    def get_wb(self, x, weight_blend):
        batch_nums = weight_blend.size()[0]
        c = weight_blend.unsqueeze(-1).unsqueeze(-1)
        x_size = x.size()
        x = x.unsqueeze(0).expand(batch_nums, x_size[0], x_size[1], x_size[2])
        x = c * x
        return x.sum(dim=1)


class NeuralStateMachine(nn.Module):
    """Wraps the real per-frame call pattern from model/model.py Model.train()/test():

    status = cat([encoder_i(x_segment_i) for i in encoders], dim=1)
    weight_blend = gating_expert(weight_blend_init, x_gating_segment)
    output = motion_expert(weight_blend, status)
    """

    def __init__(
        self,
        encoder_dims,
        encoder_activations,
        encoder_dropout,
        expert_components,
        expert_dims,
        expert_activations,
        expert_dropout,
        segmentation,
    ):
        super().__init__()
        self.segmentation = segmentation
        self.encoder_nums = len(encoder_dims)
        self.encoders = nn.ModuleList(
            [
                Encoder(encoder_dims[i], encoder_activations[i], encoder_dropout)
                for i in range(self.encoder_nums)
            ]
        )
        self.expert_nums = len(expert_components)
        self.experts = nn.ModuleList(
            [
                Expert(expert_components[i], expert_dims[i], expert_activations[i], expert_dropout)
                for i in range(self.expert_nums)
            ]
        )

    def forward(self, x):
        batch_nums = x.size()[0]
        weight_blend_first = torch.ones(batch_nums, 1, dtype=x.dtype, device=x.device)

        status_outputs = []
        for i, encoder in enumerate(self.encoders):
            status_output = encoder(x[:, self.segmentation[i] : self.segmentation[i + 1]])
            status_outputs.append(status_output)
        status = torch.cat(tuple(status_outputs), 1)

        # Gating Network
        expert_first = self.experts[0]
        weight_blend = expert_first(
            weight_blend_first, x[:, self.segmentation[-2] : self.segmentation[-1]]
        )

        # Motion Network
        expert_last = self.experts[-1]
        output = expert_last(weight_blend, status)
        return output


# --- staging harness: build + example input ---------------------------------


def build_neural_state_machine():
    # Shrunk from the real config.py sizes (419/156/2034/2048-dim encoders,
    # 512-wide hidden layers, 10 gating experts) to a tiny architecturally
    # faithful config: same encoder/expert/gating structure, small dims.
    encoder_dims = [[8, 16, 16], [6, 8, 8], [10, 16, 16], [12, 16, 16]]
    encoder_activations = [["elu", "elu"]] * 4
    segmentation = [0, 8, 14, 24, 36, 40]
    expert_components = [1, 4]
    gating_in_dim = segmentation[-1] - segmentation[-2]
    status_dim = sum(d[-1] for d in encoder_dims)
    expert_dims = [
        [gating_in_dim, 16, 16, 4],
        [status_dim, 16, 16, 20],
    ]
    expert_activations = [["elu", "elu", "softmax"], ["elu", "elu", None]]
    return NeuralStateMachine(
        encoder_dims=encoder_dims,
        encoder_activations=encoder_activations,
        encoder_dropout=0.0,
        expert_components=expert_components,
        expert_dims=expert_dims,
        expert_activations=expert_activations,
        expert_dropout=0.0,
        segmentation=segmentation,
    ).eval()


def example_input_neural_state_machine():
    return (torch.randn(2, 40),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "NeuralStateMachine",
        build_neural_state_machine,
        example_input_neural_state_machine,
        2019,
        MENAGERIE_ZOO,
    ),
]
