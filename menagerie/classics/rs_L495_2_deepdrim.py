# FAITHFUL PORT of jiaxchen2-c/DeepDRIM @ master (original framework: TensorFlow/Keras)
# (DeepDRIM.py: direct_model1_squarematrix.get_single_image_model /
#  get_pair_image_model / construct_model)
#
# DeepDRIM: supervised CNN for gene-regulatory-network (TF-gene edge) inference from
# scRNA-seq co-expression images (2021). The real Keras model takes a stack of n images
# per pair: the "primary" TF-gene co-expression image goes through a standalone 3-block
# Conv2D tower (get_single_image_model); each of the n-1 "neighbor" co-expression images
# (giving the transitive-interaction context) goes through a SECOND, separately-weighted
# but internally identical 3-block Conv2D tower (get_pair_image_model) that is SHARED
# (same nn.Module / same weights) across all neighbor images, matching the Keras code's
# reuse of one `pair_image_model` Model instance across the `input_img_multi_list` loop.
# Each conv block is Conv2d(k=3,pad=same)->ReLU ->Conv2d(k=3,valid)->ReLU->MaxPool2d(2)->
# Dropout(0.25), channel progression 32->64->128, exactly as in the Keras
# get_single_image_model/get_pair_image_model definitions. The single-image embedding and
# the concatenation of all neighbor-image embeddings are concatenated, then pushed through
# Dense(512)->Dropout(0.5)->Dense(128)->Dropout(0.5)->Dense(1)->sigmoid (construct_model's
# `combined` stack), matching the num_classes==2 branch used throughout the repo's actual
# CLI entry points (binary_crossentropy + sigmoid). No TensorFlow/Keras dependency is
# required to run this port; every conv/pool/dropout/dense stage is transcribed 1:1 from
# the real Keras functional-API code.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class _ImageTower(nn.Module):
    """Port of get_single_image_model / get_pair_image_model: both towers share the exact
    same 3-block Conv2D architecture in the original code, differing only in which weights
    are used (primary image vs. shared neighbor-image tower)."""

    def __init__(self, in_channels=1, image_size=32, embed_dim=512):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        flat_dim = self._infer_flat_dim(in_channels, image_size)
        self.fc = nn.Linear(flat_dim, embed_dim)

    def _infer_flat_dim(self, in_channels, image_size):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, image_size, image_size)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class DeepDRIM(nn.Module):
    """Port of direct_model1_squarematrix.construct_model. `n_neighbors` is the number of
    neighbor co-expression images per TF-gene pair (x_train.shape[1] - 1 in the Keras code)."""

    def __init__(self, n_neighbors=4, image_size=32, embed_dim=512):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.single_tower = _ImageTower(in_channels=1, image_size=image_size, embed_dim=embed_dim)
        self.pair_tower = _ImageTower(
            in_channels=1, image_size=image_size, embed_dim=embed_dim
        )  # shared weights across neighbors

        combined_dim = embed_dim * (1 + n_neighbors)
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_drop1 = nn.Dropout(0.5)
        self.combined_fc2 = nn.Linear(512, 128)
        self.combined_drop2 = nn.Dropout(0.5)
        self.out = nn.Linear(128, 1)

    def forward(self, images):
        # images: (batch, 1 + n_neighbors, H, W) -- images[:, 0] is the primary TF-gene
        # co-expression image, images[:, 1:] are the neighbor context images.
        primary = images[:, 0:1, :, :]
        single_out = self.single_tower(primary)

        pair_outs = []
        for i in range(self.n_neighbors):
            neighbor = images[:, i + 1 : i + 2, :, :]
            pair_outs.append(self.pair_tower(neighbor))
        merged = torch.cat(pair_outs, dim=1)

        combined = torch.cat([single_out, merged], dim=1)
        combined = nn.functional.dropout(combined, p=0.5, training=self.training)
        combined = torch.relu(self.combined_fc1(combined))
        combined = self.combined_drop1(combined)
        combined = torch.relu(self.combined_fc2(combined))
        combined = self.combined_drop2(combined)
        out = torch.sigmoid(self.out(combined))
        return out


def build_deepdrim():
    return DeepDRIM(n_neighbors=4, image_size=32, embed_dim=64)


def example_input_deepdrim():
    torch.manual_seed(0)
    batch = 2
    n_images = 1 + 4  # primary + 4 neighbors
    return (torch.randn(batch, n_images, 32, 32),)


MENAGERIE_ENTRIES = [
    ("DeepDRIM", build_deepdrim, example_input_deepdrim, 2021, "REIMPLEMENT"),
]
