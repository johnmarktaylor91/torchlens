# SOURCE: vendored from MilesZhao/PGCGM @ main (model.py)
# Code copied verbatim from the real repo (no architecture modified). Only the unused
# torch.autograd import for calc_grad_penalty (WGAN-GP training helper, not part of the
# generator/discriminator architecture) was dropped since it is not needed for tracing.
# Real inference input shapes are taken directly from the repo's own create_cif.py sampling
# script: sp_inputs = spinfo.symm_op_collection[sp_id] with shape (batch, 192, 4, 4)
# (192 = max symmetry ops padded, 4x4 affine matrices); ele_inputs = atom_embedding[ele_ids]
# transposed to (batch, ele_vec_dim, element_crystal) with ele_vec_dim=23 (README default,
# data/elements_features.npy) and element_crystal=3 (README default --element_crystal); z is
# standard Gaussian noise of shape (batch, latent_dim=128) (README default --latent_dim).
# The repo ships no training script (inference-only release) so Discriminator's real "crystal"
# spatial dims are not concretely specified anywhere in the repo; Generator is traced since its
# forward-call shapes are fully and concretely specified by create_cif.py.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.crystal_block = nn.Sequential(
            nn.Conv2d(3, 16, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 96, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 128, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 192, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(192, 256, 2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.sp_block = nn.Sequential(
            nn.Conv2d(192, 64, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.dense_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, crystal, symm_mat):
        x1 = self.crystal_block(crystal)
        x2 = self.sp_block(symm_mat)
        x = torch.cat((x1, x2), 1)
        x = self.dense_block(x)

        return x


class Generator(nn.Module):
    def __init__(self, ele_vec_dim=23, noise_dim=128):
        super().__init__()
        self.sp_block = nn.Sequential(
            nn.Conv2d(192, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.ele_block = nn.Sequential(
            nn.Conv1d(ele_vec_dim, 64, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.noise_block = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.coords_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, (2, 2), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, (2, 2), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        self.length_block = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Tanh(),
        )

    def forward(self, sp_inputs, ele_inputs, z):
        sp_embedding = self.sp_block(sp_inputs)
        ele_embedding = self.ele_block(ele_inputs)
        z_embedding = self.noise_block(z)

        x1 = torch.cat((z_embedding, ele_embedding), 1)
        x2 = torch.cat((z_embedding, sp_embedding), 1)

        coords = self.coords_block1(x1.view(-1, 512, 1, 1))
        length = self.length_block(x2)

        return coords, length


def build_pgcgm_generator():
    return Generator(ele_vec_dim=23, noise_dim=128)


def example_input_pgcgm_generator():
    torch.manual_seed(0)
    batch = 4
    sp_inputs = torch.randn(batch, 192, 4, 4)
    ele_inputs = torch.randn(batch, 23, 3)
    z = torch.randn(batch, 128)
    return (sp_inputs, ele_inputs, z)


MENAGERIE_ENTRIES = [
    (
        "PGCGM_Generator",
        "build_pgcgm_generator",
        "example_input_pgcgm_generator",
        "2023",
        "vendored-pytorch",
    ),
]
