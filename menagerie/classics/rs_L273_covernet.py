# SOURCE: vendored from nutonomy/nuscenes-devkit @ master
# Files combined:
#   python-sdk/nuscenes/prediction/models/covernet.py (CoverNet)
#   python-sdk/nuscenes/prediction/models/backbone.py (ResNetBackbone,
#       trim_network_at_index, calculate_backbone_feature_dim -- the
#       backbone helpers CoverNet's constructor consumes)
#
# CoverNet (Phan-Minh et al., CVPR 2020) is a fixed-trajectory-set ("lattice") classifier:
# a CNN backbone (ResNet with the FC head trimmed off) processes the rasterized scene image,
# its pooled features are concatenated with a 3-dim agent-state vector, and a small stack of
# ReLU-activated linear layers scores each trajectory in a pre-computed lattice. The vendored
# code is the exact upstream nuscenes-devkit module; the `nuscenes-devkit` pip package itself
# is not installed (needs_env territory), but this specific file only imports torch/torchvision,
# so it is vendored directly rather than gated behind the full devkit dependency.
#
# Import-only fix applied (no architectural change): `from nuscenes.prediction.models.backbone
# import calculate_backbone_feature_dim` resolved by inlining backbone.py's
# trim_network_at_index/calculate_backbone_feature_dim/ResNetBackbone from the same repo path
# into this file. CoverNet's body is otherwise byte-for-byte the original.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from typing import List, Tuple

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# Number of entries in Agent State Vector
ASV_DIM = 3


# ---------------------------------------------------------------------------
# backbone.py
# ---------------------------------------------------------------------------
def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])


def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """Helper to calculate the shape of the fully-connected regression layer."""
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]


RESNET_VERSION_TO_MODEL = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
    """

    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super().__init__()

        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(
                f"Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}"
                f". Received {version}."
            )

        self.backbone = trim_network_at_index(RESNET_VERSION_TO_MODEL[version](), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)


# ---------------------------------------------------------------------------
# covernet.py
# ---------------------------------------------------------------------------
class CoverNet(nn.Module):
    """Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf"""

    def __init__(
        self,
        backbone: nn.Module,
        num_modes: int,
        n_hidden_layers: List[int] = None,
        input_shape: Tuple[int, int, int] = (3, 500, 500),
    ):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone
        :param num_modes: Number of modes in the lattice
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(
                f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}"
            )

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + ASV_DIM] + n_hidden_layers + [num_modes]

        linear_layers = [
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])
        ]

        self.head = nn.ModuleList(linear_layers)
        self.relu = nn.ReLU()

    def forward(self, image_tensor: torch.Tensor, agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)

        for linear in self.head:
            logits = self.relu(linear(logits))

        return logits


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"

_INPUT_SHAPE = (3, 64, 64)  # shrunk from the repo default (3, 500, 500) for fast tracing


def build_covernet():
    backbone = ResNetBackbone("resnet18")
    return CoverNet(backbone, num_modes=8, n_hidden_layers=[64], input_shape=_INPUT_SHAPE)


def example_input_covernet():
    image_tensor = torch.rand(1, *_INPUT_SHAPE)
    agent_state_vector = torch.rand(1, ASV_DIM)
    return (image_tensor, agent_state_vector)


MENAGERIE_ENTRIES = [
    ("CoverNet", "build_covernet", "example_input_covernet", 2020, MENAGERIE_ZOO),
]
