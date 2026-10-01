# SOURCE: vendored from frankligy/DeepImmuno @ df42ac5b6bddfe531268335e2dcb496559cd488b
# https://raw.githubusercontent.com/frankligy/DeepImmuno/df42ac5b6bddfe531268335e2dcb496559cd488b/extension/deepimmuno-gan-train.py
#
# Li & Mansour et al. (Cell Reports Methods 2021) "Deep learning-based
# prediction of immunogenic epitopes" -- the main deepimmuno-cnn classifier
# in the repo root is TensorFlow/Keras, but the DeepImmuno-GAN peptide
# generator (paper's "epitope generation" extension, `extension/
# deepimmuno-gan-train.py`) is real, unmodified PyTorch: a WGAN-GP
# (Wasserstein GAN with gradient penalty, "feedback GAN"-style) over
# one-hot-encoded peptide sequences. `Generator` maps a 128-dim latent noise
# vector through a linear projection + 5 residual 1D-conv blocks (`ResBlock`)
# to per-position amino-acid logits, then a Gumbel-softmax to produce a
# differentiable one-hot peptide sequence. `Discriminator` mirrors the
# architecture (1D conv + 5 ResBlocks + linear) to score real vs. generated
# peptides. This ResBlock/Generator/Discriminator design is the paper's
# architectural contribution (adapted from the improved-WGAN-GAN literature
# for the peptide domain), so it is vendored rather than constructed from a
# base-library class.
#
# `ResBlock`, `Generator`, and `Discriminator` are the real, unmodified
# classes from the file above (layer composition and forward-pass control
# flow are byte-for-byte the original). Only mechanical staging edits:
#   - Dropped the training-loop-only imports (`matplotlib.pyplot`, `argparse`,
#     `os`) and the entire training/data-pipeline code (`real_dataset_class`,
#     `discriminator_train`, `generator_train`, `calculate_gradient_penalty`,
#     `train()`, the peptide/HLA preprocessing helpers, and the
#     `if __name__ == '__main__':` CLI block) -- none of that is part of the
#     `nn.Module` architecture itself.
#   - Added `build_deepimmuno_gan()`/`example_input_deepimmuno_gan()` staging
#     entry points at the original module's default sizes
#     (hidden=128, seq_len=10, n_chars=21) with batch_size=2 (the
#     `Generator.forward`/`Discriminator.forward` reshape logic is
#     batch-size-parametric via `self.batch_size`, matching the original).

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, hidden):  # hidden means the number of filters
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),  # in_place = True
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, input):  # input [N, hidden, seq_len]
        output = self.res_block(input)
        return input + 0.3 * output  # [N, hidden, seq_len]  doesn't change anything


class Generator(nn.Module):
    def __init__(self, hidden, seq_len, n_chars, batch_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, kernel_size=1)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self, noise):  # noise [batch,128]
        output = self.fc1(noise)  # [batch,hidden*seq_len]
        output = output.view(-1, self.hidden, self.seq_len)  # [batch,hidden,seq_len]
        output = self.block(output)  # [batch,hidden,seq_len]
        output = self.conv1(output)  # [batch,n_chars,seq_len]
        """
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.

        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as
        being created form stracth
        """
        output = output.transpose(1, 2)  # [batch,seq_len,n_chars]
        output = output.contiguous()
        output = output.view(self.batch_size * self.seq_len, self.n_chars)
        output = F.gumbel_softmax(
            output, tau=0.75, hard=False
        )  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size, self.seq_len, self.n_chars)  # [batch,seq_len,n_chars]
        return output


class Discriminator(nn.Module):
    def __init__(self, hidden, n_chars, seq_len):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(n_chars, hidden, 1)
        self.fc = nn.Linear(seq_len * hidden, 1)
        self.hidden = hidden
        self.n_chars = n_chars
        self.seq_len = seq_len

    def forward(self, input):  # input [N,seq_len,n_chars]
        output = input.transpose(1, 2)  # input [N, n_chars, seq_len]
        output = output.contiguous()
        output = self.conv1(output)  # [N,hidden,seq_len]
        output = self.block(output)  # [N, hidden, seq_len]
        output = output.view(-1, self.seq_len * self.hidden)  # [N, hidden*seq_len]
        output = self.fc(output)  # [N,1]
        return output


def build_deepimmuno_gan():
    # Original default sizes: hidden=128, seq_len=10 (peptide length), n_chars=21
    # (20 amino acids + gap token), batch_size=64. Use batch_size=2 here for a
    # small trace; Generator/Discriminator reshape logic is batch-parametric.
    return Generator(hidden=128, seq_len=10, n_chars=21, batch_size=2)


def example_input_deepimmuno_gan():
    return torch.randn(2, 128)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepImmuno-GAN", "build_deepimmuno_gan", "example_input_deepimmuno_gan", 2021, "vendored"),
]
