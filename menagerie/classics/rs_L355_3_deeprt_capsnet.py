# SOURCE: vendored from horsepurve/DeepRTplus @ master
# https://raw.githubusercontent.com/horsepurve/DeepRTplus/master/capsule_network_emb_cpu.py
# https://raw.githubusercontent.com/horsepurve/DeepRTplus/master/config.py
#
# Guan, Chu, Fan, Chen, Xiao, Fu, He, Yan, Xie 2020 (Anal. Chem.) "DeepRT(+): ultra-precise
# peptide retention time prediction by capsule network with embedding" -- itself an embedding
# + 1D "sequence-mode" adaptation of Sabour, Frosst, Hinton 2017 "Dynamic Routing Between
# Capsules" (the header comment in the real file credits the base `CapsuleLayer` routing
# implementation to Kenta Iwasaki @ Gram.AI, with the RT-specific `CapsuleNet` wiring
# ("CapsRT") added by the DeepRT author). A learned amino-acid embedding feeds two stacked
# Conv2d+BatchNorm+ReLU layers, then a "primary capsule" layer (parallel Conv2d capsules with
# squash nonlinearity), then a "digit capsule" layer using dynamic routing-by-agreement
# (`CapsuleLayer` with `num_route_nodes != -1`), and the capsule-pose-vector norm is read out
# directly as the (1-class) retention-time-related score in DeepRT's 1D/`param_1D_rt` mode
# (`x = self.digit_capsules(x).squeeze()[:, None, :]`, no MNIST-style reconstruction decoder
# needed on this branch).
#
# `softmax`, `CapsuleLayer`, and `CapsuleNet` are copied verbatim from
# `capsule_network_emb_cpu.py` (the CPU-only variant of `capsule_network_emb.py`; identical
# model code, `CUDA = False` hardcoded and no `.cuda()` calls). `config.py`'s `conv1_kernel`,
# `conv2_kernel` (both 10) and `max_length` (50) are the real repo defaults for the shipped
# `mod` (HeLa, modification-aware) dataset pipeline (`pipeline_mod_cpu.sh`). The real
# `NUM_ROUTING_ITERATIONS`, `NUM_CLASSES=1`, `param_1D_rt` dict (`dim=1`,
# `conv1_kernel=(len(dictionary), conv1_kernel)`, `pri_caps_kernel=(1, conv2_kernel)`,
# `stride=1`) are copied verbatim; only `len(dictionary)` (the data-file-derived amino-acid
# vocabulary size -- built at runtime from the training TSV's observed sequence alphabet by
# `RTdata_emb.Dictionary`, not part of the architecture) is inlined as a fixed
# representative size (25: the pad symbol + 20 standard amino acids + 4 modification-digit
# codes used by the shipped `mod.txt`/`mod_train_2.txt` datasets, per the README's modified-AA
# encoding table). The unused MNIST/2D branches (`param_2D`, `augmentation`, `CapsuleLoss`,
# and the reconstruction-decoder code path gated by `2 == param['dim']`) are exercised only
# when `param['dim'] == 2`, which this module never selects (`param = param_1D_rt` always,
# matching the real script's runtime configuration) -- they are retained verbatim in the class
# body (no architectural pruning) but structurally dead on this build.

import torch
import torch.nn.functional as F
from torch import nn

CNN_EMB = True
NUM_ROUTING_ITERATIONS = 1
CUDA = False

# Inlined from config.py (real DeepRTplus/mod defaults):
conv1_kernel = 10
conv2_kernel = 10
max_length = 50

# Inlined from RTdata_emb.Dictionary: real vocab size is data-file-derived
# (pad symbol + observed amino-acid alphabet of the training TSV, e.g. mod_train_2.txt).
# 25 = pad + 20 standard AAs + 4 modification-digit codes ('1'..'4') per the README's
# encoding table -- a representative fixed size, not an architectural choice.
DICT_SIZE = 25

NUM_CLASSES = 1

param_1D_rt = {
    "data": "rt",
    "dim": 1,
    "conv1_kernel": (DICT_SIZE, conv1_kernel),
    "pri_caps_kernel": (1, conv2_kernel),
    "stride": 1,
    "digit_caps_nodes": 32 * 1 * (max_length - conv1_kernel * 2 + 2 - conv2_kernel + 1),
    "NUM_CLASSES": NUM_CLASSES,
}

param = param_1D_rt


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(
        transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=1
    )
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(
        self,
        num_capsules,
        num_route_nodes,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=None,
        num_iterations=NUM_ROUTING_ITERATIONS,
    ):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
            )
        else:
            self.capsules = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=0,
                    )
                    for _ in range(num_capsules)
                ]
            )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = torch.zeros(*priors.size())

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, conv1_kernel, conv2_kernel):
        super(CapsuleNet, self).__init__()
        EMB_SIZE = 0
        if CNN_EMB:
            EMB_SIZE = 20
            self.emb = nn.Embedding(DICT_SIZE, EMB_SIZE)
        else:
            EMB_SIZE = DICT_SIZE
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=256, kernel_size=(EMB_SIZE, conv1_kernel), stride=1
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, conv1_kernel), stride=1
        )
        self.bn2 = nn.BatchNorm2d(256)

        self.primary_capsules = CapsuleLayer(
            num_capsules=8,
            num_route_nodes=-1,
            in_channels=256,
            out_channels=32,
            kernel_size=(1, conv2_kernel),
            stride=param["stride"],
        )

        self.digit_capsules = CapsuleLayer(
            num_capsules=param["NUM_CLASSES"],
            num_route_nodes=32 * 1 * (max_length - conv1_kernel * 2 + 2 - conv2_kernel + 1),
            in_channels=8,
            out_channels=16,
        )

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        if CNN_EMB:
            x = self.emb(x)  # [batch, len] -> [batch, len, dict]
            x = x.transpose(dim0=1, dim1=2)  # -> [batch, dict, len]
            x = x[:, None, :, :]  # -> [batch, 1, dict, len]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.primary_capsules(x)
        if 2 == param["dim"]:
            x = self.digit_capsules(x).squeeze().transpose(0, 1)
        if 1 == param["dim"]:
            x = self.digit_capsules(x).squeeze()[:, None, :]

        classes = (x**2).sum(dim=-1) ** 0.5
        if 2 == param["dim"]:
            classes = F.softmax(classes)

        if y is None:
            if 2 == param["dim"]:
                _, max_length_indices = classes.max(dim=1)
                y = torch.sparse.torch.eye(NUM_CLASSES).index_select(
                    dim=0, index=max_length_indices.data
                )

        if 2 == param["dim"]:
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            return classes, reconstructions
        if 1 == param["dim"]:
            return classes, x


MENAGERIE_ZOO = "vendored-pytorch"


def build_deeprt_capsnet():
    model = CapsuleNet(conv1_kernel, conv2_kernel)
    model.eval()
    return model


def example_input_deeprt_capsnet():
    # batch > 1: the real forward()'s `.squeeze()` calls (faithfully preserved above)
    # collapse a size-1 batch dim along with the routing dims, so batch=1 breaks the
    # indexing exactly as in the upstream code -- use batch=2 to keep dims intact.
    batch = 2
    seq_ids = torch.randint(0, DICT_SIZE, (batch, max_length))
    return (seq_ids,)


MENAGERIE_ENTRIES = [
    (
        "DeepRT CapsuleNet (CapsRT)",
        build_deeprt_capsnet,
        example_input_deeprt_capsnet,
        2020,
        MENAGERIE_ZOO,
    ),
]
