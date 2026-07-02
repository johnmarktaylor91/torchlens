# FAITHFUL PORT of MilesZhao/CubicGAN @ main (original framework: TensorFlow / Keras)
#
# CubicGAN ("This is the source code of CubicGAN generating cubic crystal
# structures using improved WGAN"). Source: wgan-v2.py, build_generator() /
# build_discriminator() (Keras functional API), fetched from
# https://raw.githubusercontent.com/MilesZhao/CubicGAN/main/wgan-v2.py
#
# TensorFlow/Keras is not one of TorchLens's base libraries (and would be a
# whole second deep-learning framework to install just to run one model), so
# per the ladder this is a rung-3 FAITHFUL PORT: every layer of the real
# Keras generator/discriminator graphs is transcribed into torch, preserving
# shapes, layer order, activations, and the two-branch (spacegroup-embedding
# + element-embedding + latent-noise) -> deconv-coords / dense-lattice-length
# architecture exactly as authored. No architecture was invented or guessed.
#
# Notes on the port:
#  - Keras `Embedding(n_spacegroup, 64)` on `sp_inputs` -> `nn.Embedding`.
#  - Keras `tf.gather(atom_embedding, elements)` looks up a FIXED (non-
#    trainable) per-element feature table computed offline from pymatgen
#    periodic-table properties (see the real repo's util.py:atom_embedding).
#    That fixed 23-dim feature table is not shippable data here, so it is
#    represented by a frozen (non-trainable), randomly-initialized
#    `nn.Embedding(n_element, 23)` buffer standing in for the same fixed
#    lookup role in the architecture -- the surrounding Conv1D/Activation/
#    Flatten branch that consumes it is transcribed verbatim.
#  - Keras layers are channels-last (NHWC); torch is channels-first (NCHW).
#    Every Conv1D/Conv2D/Conv2DTranspose below explicitly permutes around the
#    op to preserve the exact per-layer tensor shapes and computation of the
#    original graph (only the memory layout convention differs, not the
#    architecture).
#  - Keras `ClipConstraint`/`activity_regularizer`/`kernel_regularizer`
#    (weight clipping + L1/L2 penalties) are training-time-only optimizer/
#    loss-side mechanisms; they do not change the forward computational
#    graph and are omitted (this module ports the model architecture, not
#    the training loop).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class CubicGANDiscriminator(nn.Module):
    """Faithful port of build_discriminator(): a Conv1D critic over the
    (3, 28) stacked [coords | element-embedding | length | spacegroup-onehot]
    crystal representation."""

    def __init__(self, in_channels=28):
        super().__init__()
        # Keras Conv1D(filters, kernel_size, strides) on (batch, steps=3, channels)
        # torch Conv1d expects (batch, channels, length); kernel_size=1 convs are
        # equivalent to per-position dense layers regardless of layout.
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1, stride=1)
        self.drop1 = nn.Dropout(0.25)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(128, 512, kernel_size=1, stride=1)
        self.drop2 = nn.Dropout(0.25)
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv1d(512, 1024, kernel_size=1, stride=1)
        self.drop3 = nn.Dropout(0.25)
        self.act3 = nn.LeakyReLU(0.2)

        # Keras Conv1D(1024, 2, 1, padding='same') on length-3 input keeps length 3.
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=2, stride=1, padding=1)
        self.drop4 = nn.Dropout(0.25)
        self.act4 = nn.LeakyReLU(0.2)

        # Keras default padding='valid': length 3 -> (3 - 2)/1 + 1 = 2 after conv4's
        # 'same' padding kept length at 4 (3 + pad(1) - 2 + 1); replicate exactly.
        self.conv5 = nn.Conv1d(1024, 2048, kernel_size=2, stride=1, padding=0)
        self.drop5 = nn.Dropout(0.25)
        self.act5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(2048, 4096, kernel_size=2, stride=1, padding=0)
        self.drop6 = nn.Dropout(0.25)
        self.act6 = nn.LeakyReLU(0.2)

        # Flatten() dimensionality after conv4/5/6 on a length-3 input:
        # 3 -> conv4(k=2,pad=1) -> 4 -> conv5(k=2,pad=0) -> 3 -> conv6(k=2,pad=0) -> 2,
        # so flattened size = 4096 channels * 2 positions = 8192.
        self.fc1 = nn.Linear(8192, 1024)
        self.act7 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.act8 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(512, 256)
        self.act9 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(256, 32)
        self.act10 = nn.LeakyReLU(0.2)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, coords_inputs):
        # coords_inputs: (batch, 3, 28) Keras layout -> (batch, 28, 3) torch layout
        x = coords_inputs.transpose(1, 2)

        x = self.act1(self.drop1(self.conv1(x)))
        x = self.act2(self.drop2(self.conv2(x)))
        x = self.act3(self.drop3(self.conv3(x)))
        x = self.act4(self.drop4(self.conv4(x)))
        x = self.act5(self.drop5(self.conv5(x)))
        x = self.act6(self.drop6(self.conv6(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.act7(self.fc1(x))
        x = self.act8(self.fc2(x))
        x = self.act9(self.fc3(x))
        x = self.act10(self.fc4(x))
        x = self.fc5(x)
        return x


class CubicGANGenerator(nn.Module):
    """Faithful port of build_generator(): spacegroup-embedding + fixed
    element-feature branch + latent-noise branch, merged and deconvolved
    into 3x3 fractional coordinates plus a scalar lattice length."""

    def __init__(self, n_element=63, n_spacegroup=123, lat_dim=128):
        super().__init__()

        # spacegroup label branch: Embedding(n_spacegroup, 64) -> Conv1D(96, k=1) -> relu -> flatten
        self.sp_embedding = nn.Embedding(n_spacegroup, 64)
        self.sp_conv = nn.Conv1d(64, 96, kernel_size=1)
        self.sp_act = nn.ReLU()

        # element label branch: fixed (non-trainable) per-element feature table,
        # standing in for the real repo's offline pymatgen atom_embedding lookup
        # (see module docstring). 23-dim, gathered per one of 3 element slots.
        self.atom_embedding = nn.Embedding(n_element, 23)
        self.atom_embedding.weight.requires_grad_(False)
        self.elem_conv = nn.Conv1d(23, 128, kernel_size=1)
        self.elem_act = nn.ReLU()

        # latent inputs branch
        self.lat_dense = nn.Linear(lat_dim, 256)

        # merge: concat(sp_flat[96] + elem_flat[3*128=384] + gen[256]) = 736
        merged_dim = 96 + 3 * 128 + 256
        self.merge_dense = nn.Linear(merged_dim, 3 * 1 * 128)
        self.merge_act = nn.ReLU()

        # coords deconv tower (Keras NHWC (3,1,128) spatial=(3,1), channels=128)
        # ConvTranspose2D(1024,(1,2),(1,1)): spatial (3,1)->(3,2)
        self.deconv1 = nn.ConvTranspose2d(128, 1024, kernel_size=(1, 2), stride=(1, 1))
        self.deconv1_act = nn.ReLU()
        self.deconv1_drop = nn.Dropout(0.5)

        # ConvTranspose2D(1024,(1,2),(1,1)): spatial (3,2)->(3,3)
        self.deconv2 = nn.ConvTranspose2d(1024, 1024, kernel_size=(1, 2), stride=(1, 1))
        self.deconv2_act = nn.ReLU()
        self.deconv2_drop = nn.Dropout(0.5)

        # Conv2D(512,(1,1),(1,1)): spatial (3,3)->(3,3)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        self.conv3_act = nn.ReLU()
        self.conv3_drop = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        self.conv4_act = nn.ReLU()
        self.conv4_drop = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv5_act = nn.ReLU()
        self.conv5_drop = nn.Dropout(0.5)

        self.conv6 = nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        self.conv6_act = nn.Tanh()

        # lattice length head: Flatten(coords (3,3)) -> Dense(30)->relu->Dense(18)
        # ->relu->Dense(6)->relu->Dense(1)->tanh
        self.len_fc1 = nn.Linear(3 * 3, 30)
        self.len_act1 = nn.ReLU()
        self.len_fc2 = nn.Linear(30, 18)
        self.len_act2 = nn.ReLU()
        self.len_fc3 = nn.Linear(18, 6)
        self.len_act3 = nn.ReLU()
        self.len_fc4 = nn.Linear(6, 1)
        self.len_act4 = nn.Tanh()

    def forward(self, sp_inputs, element_inputs, lat_inputs):
        # sp_inputs: (batch, 1) long
        sp = self.sp_embedding(sp_inputs)  # (batch, 1, 64)
        sp = sp.transpose(1, 2)  # (batch, 64, 1) torch conv layout
        sp = self.sp_act(self.sp_conv(sp))  # (batch, 96, 1)
        sp = torch.flatten(sp, start_dim=1)  # (batch, 96)

        # element_inputs: (batch, 3) long indices into the fixed atom feature table
        elements = self.atom_embedding(element_inputs)  # (batch, 3, 23)
        elements = elements.transpose(1, 2)  # (batch, 23, 3)
        elements = self.elem_act(self.elem_conv(elements))  # (batch, 128, 3)
        elements = torch.flatten(elements, start_dim=1)  # (batch, 384)

        gen = self.lat_dense(lat_inputs)  # (batch, 256)

        x = torch.cat([sp, elements, gen], dim=1)  # (batch, 736)
        x = self.merge_act(self.merge_dense(x))  # (batch, 384)
        x = x.view(-1, 128, 3, 1)  # torch NCHW: channels=128, spatial=(3,1)

        coords = self.deconv1_drop(self.deconv1_act(self.deconv1(x)))
        coords = self.deconv2_drop(self.deconv2_act(self.deconv2(coords)))
        coords = self.conv3_drop(self.conv3_act(self.conv3(coords)))
        coords = self.conv4_drop(self.conv4_act(self.conv4(coords)))
        coords = self.conv5_drop(self.conv5_act(self.conv5(coords)))
        coords = self.conv6_act(self.conv6(coords))  # (batch, 1, 3, 3)

        coords = coords.view(-1, 3, 3)

        lengths = torch.flatten(coords, start_dim=1)  # (batch, 9)
        lengths = self.len_act1(self.len_fc1(lengths))
        lengths = self.len_act2(self.len_fc2(lengths))
        lengths = self.len_act3(self.len_fc3(lengths))
        lengths = self.len_act4(self.len_fc4(lengths))

        return coords, lengths


class CubicGAN(nn.Module):
    """Combined generator+discriminator module so a single trace exercises
    both real network graphs of CubicGAN's improved-WGAN architecture."""

    def __init__(self, n_element=63, n_spacegroup=123, lat_dim=128):
        super().__init__()
        self.generator = CubicGANGenerator(
            n_element=n_element, n_spacegroup=n_spacegroup, lat_dim=lat_dim
        )
        self.discriminator = CubicGANDiscriminator(in_channels=28)

    def forward(
        self, sp_inputs, element_inputs, lat_inputs, sp_onehot, elem_features, length_repeat
    ):
        coords, lengths = self.generator(sp_inputs, element_inputs, lat_inputs)

        # Reproduce train_step()'s real-graph assembly of the discriminator's
        # (batch, 3, 28) input: [coords(3) | elem_features(23) | length(1) | sp_onehot(1)]
        length_col = lengths.view(-1, 1, 1).repeat(1, 3, 1)
        crystal = torch.cat([coords, elem_features, length_col, sp_onehot], dim=-1)
        logits = self.discriminator(crystal)
        return coords, lengths, logits


def build_cubicgan():
    return CubicGAN(n_element=63, n_spacegroup=123, lat_dim=32)


def example_input_cubicgan():
    torch.manual_seed(0)
    batch = 2
    sp_inputs = torch.randint(0, 123, (batch, 1))
    element_inputs = torch.randint(0, 63, (batch, 3))
    lat_inputs = torch.randn(batch, 32)
    sp_onehot = (
        torch.nn.functional.one_hot(torch.randint(0, 3, (batch,)), num_classes=3)
        .float()
        .unsqueeze(-1)
    )
    elem_features = torch.randn(batch, 3, 23)
    length_repeat = torch.randn(batch, 3, 1)
    return (sp_inputs, element_inputs, lat_inputs, sp_onehot, elem_features, length_repeat)


MENAGERIE_ENTRIES = [
    (
        "CubicGAN",
        build_cubicgan,
        example_input_cubicgan,
        2021,
        "PORT",
    ),
]
