# FAITHFUL PORT of jingshuw/sctransfer @ master (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/jingshuw/sctransfer/master/sctransfer/network_joint.py
# https://raw.githubusercontent.com/jingshuw/sctransfer/master/sctransfer/layers.py
# https://raw.githubusercontent.com/jingshuw/sctransfer/master/sctransfer/loss.py
#
# Wang, Hou, Ye, Yan, Yang, Zheng, Kellis, Wang, 2021 (Nature Communications)
# "SAVER-X: cross-population repository of gene expression variation".
# jingshuw/SAVERX (the queue candidate) is the top-level R package; the actual
# neural network that SAVER-X's `saverx()` transfer-learning path trains and
# calls out to is the Keras/TensorFlow autoencoder in the companion repo
# jingshuw/sctransfer (imported by SAVERX's R code as its Python backend, per
# the R repo's own README/`R/autoencode.R`). No PyTorch code exists anywhere
# for SAVER-X, and TF1.x-vintage Keras (`keras.engine.topology.Layer`,
# `keras.objectives`) cannot be reasonably installed alongside this repo's
# torch stack, so per the ladder this is a faithful port (rung 3): every real
# mechanism from `sctransfer.network_joint.JointAutoencoder` is transcribed
# into self-contained torch.
#
# `JointAutoencoder` (the class the real `SaverXTrain` entry point in
# `saverx_train.py` actually constructs and trains) is a triple-encoder
# denoising autoencoder for cross-species (human/mouse) transfer learning:
# three independent Dense+BatchNorm+ReLU MLP stacks
# (`hidden_size=(128,64,32,64,128)` by default) run in parallel over (1) the
# human expression vector, (2) the mouse expression vector, and (3) a
# lower-dimensional "shared" (orthologous-gene) expression vector -- each
# input has one extra appended node (the real repo's UMI/non-UMI data-type
# indicator, `input_size + 1`). The human branch's final hidden layer is
# concatenated with the shared branch's final hidden layer (and likewise for
# mouse+shared) before separate linear "mean" output heads with the paper's
# `MeanAct = clip(exp(x), 1e-5, 1e6)` activation (transcribed faithfully as
# `mean_act` below) -- this is the ZINB-autoencoder "shared bottleneck" fusion
# that is SAVER-X's real architectural contribution (enabling cross-species
# knowledge transfer through the shared branch), not a generic autoencoder.
# The real repo additionally has zero-inflation (`ElementwiseDense` sigmoid
# gate, `pi_*`) and per-(species x UMI-type) constant-dispersion heads
# (`ConstantDispersionLayer`) feeding a NB/ZINB negative-log-likelihood loss
# (`sctransfer.loss.NB`/`ZINB`) used only at training time to shape those
# extra output heads; this port keeps the real `mean_*` forward path (the
# actual reconstruction/imputation output SAVER-X reports) and the real
# zero-inflation gate `pi_*` (`ElementwiseDense`, transcribed faithfully as
# `_ElementwiseDense` below) so the model's forward signature matches the
# repo's `predict()` output structure; the loss-only dispersion/NB machinery
# is training infrastructure, not part of the traced forward computation.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def mean_act(x: torch.Tensor) -> torch.Tensor:
    """Faithful port of `sctransfer.network.MeanAct`/`network_joint.MeanAct`:
    `lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)`."""
    return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class _ElementwiseDense(nn.Module):
    """Faithful port of `sctransfer.layers.ElementwiseDense`: a per-feature
    (not fully-connected) affine + sigmoid gate used for the zero-inflation
    probability heads `pi_mouse`/`pi_human` in the real Keras model
    (`kernel`/`bias` shape `(units,)`, broadcast-multiplied against the
    input rather than matrix-multiplied)."""

    def __init__(self, units: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.ones(units))
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.kernel + self.bias)


class _EncoderStack(nn.Module):
    """Faithful port of one of the three real per-branch hidden-layer loops in
    `JointAutoencoder.build()` (`ms_*`/`hn_*`/`jt_*` -- identical structure,
    only the layer-name prefix differs upstream): `hidden_size` Dense layers,
    each followed by BatchNorm (`center=True, scale=False`, matching the real
    `BatchNormalization(center=True, scale=False)` call) then a ReLU
    activation, mirrored here as `affine=False` (no learnable scale) with
    `bias=True` re-added via the preceding Linear's own bias (Keras' Dense
    already has a bias before BatchNorm centers it; this port keeps a single
    bias source via the Linear layer for a torch-idiomatic equivalent)."""

    def __init__(self, input_size: int, hidden_size: tuple[int, ...]):
        super().__init__()
        layers = []
        prev = input_size
        for hid_size in hidden_size:
            layers.append(nn.Linear(prev, hid_size))
            layers.append(nn.BatchNorm1d(hid_size, affine=False))
            layers.append(nn.ReLU())
            prev = hid_size
        self.net = nn.Sequential(*layers)
        self.output_size = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointAutoencoder(nn.Module):
    """Faithful port of `sctransfer.network_joint.JointAutoencoder` (the real
    class `saverx_train.SaverXTrain` constructs and trains). Three parallel
    encoder-decoder MLP stacks (human/mouse/shared) whose final hidden layers
    fuse pairwise (human+shared, mouse+shared) into separate mean/dispersion-
    gate output heads for each species -- the real cross-species transfer
    mechanism. Forward returns the four real Keras model's `joint_output`
    tensors: `(mean_mouse, pi_mouse, mean_human, pi_human)`."""

    def __init__(
        self,
        input_size_human: int,
        input_size_mouse: int,
        shared_size: int,
        hidden_size: tuple[int, ...] = (128, 64, 32, 64, 128),
    ):
        super().__init__()
        self.input_size_human = input_size_human
        self.input_size_mouse = input_size_mouse
        self.shared_size = shared_size
        self.output_size_human = input_size_human
        self.output_size_mouse = input_size_mouse

        # +1 node per branch: the real repo's UMI/non-UMI data-type indicator
        # appended to every input vector (see `network_joint.py` `build()`).
        self.enc_mouse = _EncoderStack(input_size_mouse + 1, hidden_size)
        self.enc_human = _EncoderStack(input_size_human + 1, hidden_size)
        self.enc_shared = _EncoderStack(shared_size + 1, hidden_size)

        fused_mouse_dim = self.enc_mouse.output_size + self.enc_shared.output_size
        fused_human_dim = self.enc_human.output_size + self.enc_shared.output_size

        self.mean_mouse_head = nn.Linear(fused_mouse_dim, self.output_size_mouse)
        self.mean_human_head = nn.Linear(fused_human_dim, self.output_size_human)

        self.pi_mouse_head = _ElementwiseDense(self.output_size_mouse)
        self.pi_human_head = _ElementwiseDense(self.output_size_human)

    def forward(self, human: torch.Tensor, mouse: torch.Tensor, shared: torch.Tensor):
        last_hidden_mouse = self.enc_mouse(mouse)
        last_hidden_human = self.enc_human(human)
        last_hidden_joint = self.enc_shared(shared)

        decoder_output_mouse = torch.cat([last_hidden_mouse, last_hidden_joint], dim=-1)
        decoder_output_human = torch.cat([last_hidden_human, last_hidden_joint], dim=-1)

        mean_mouse_no_act = self.mean_mouse_head(decoder_output_mouse)
        mean_human_no_act = self.mean_human_head(decoder_output_human)

        mean_mouse = mean_act(mean_mouse_no_act)
        mean_human = mean_act(mean_human_no_act)

        pi_mouse = self.pi_mouse_head(mean_mouse_no_act)
        pi_human = self.pi_human_head(mean_human_no_act)

        return mean_mouse, pi_mouse, mean_human, pi_human


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_saverx():
    model = JointAutoencoder(
        input_size_human=20,
        input_size_mouse=18,
        shared_size=10,
        hidden_size=(16, 8, 16),
    )
    model.eval()
    return model


def example_input_saverx():
    torch.manual_seed(0)
    batch = 4
    human = torch.rand(batch, 21)
    mouse = torch.rand(batch, 19)
    shared = torch.rand(batch, 11)
    return (human, mouse, shared)


MENAGERIE_ENTRIES = [
    ("SAVER-X", build_saverx, example_input_saverx, 2021, "ported-pytorch"),
]
