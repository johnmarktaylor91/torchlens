# FAITHFUL PORT of WGLab/NanoCaller @ master (original framework: TensorFlow / Keras)
# https://github.com/WGLab/NanoCaller/blob/master/nanocaller_src/model_architect.py
#
# NanoCaller's SNP_model (a `tf.keras.Model` subclass) has no PyTorch
# implementation in the repo -- it is TF1/Keras-graph code with weights
# distributed as TF checkpoint files (model-*.data-00000-of-00001), so
# rung-2 vendoring is not possible in a torch-only environment. This is a
# mechanism-faithful transcription of `SNP_model.__init__`/`.call` into
# torch: every branch (three parallel Conv2d "stems" over the pileup image,
# concatenated -> two more Conv2d layers -> flatten -> shared FC trunk ->
# four per-base 2-way heads fused with a reference one-hot side input ->
# a genotype head fused from the base heads) is reproduced with the same
# layer shapes/strides/paddings and the same SELU activations. TF's
# NHWC channel-last conv layout is translated to torch's NCHW; TF
# `padding='same'` stride-1 convs are translated to the equivalent
# symmetric explicit padding; `tf.concat(..., axis=-1/-2)` -> `torch.cat`
# on the matching torch dim.
#
# Input pileup shape, taken from the repo's own null-input sanity check in
# snpCaller.py (`null_x=np.zeros(5*41*5).reshape(1,5,41,5)`, TF NHWC) and
# `generate_SNP_pileups.py` (`nbr_size=20` -> width 2*20+1=41): here as
# torch NCHW that is (batch, C=5, H=5, W=41). The four base-reference
# side-inputs (`A_ref`/`G_ref`/`T_ref`/`C_ref`) are each a single-column
# float tensor per repo call site (`batch_ref[:, k][:, np.newaxis]`).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class SNPModel(nn.Module):
    """Faithful torch port of NanoCaller's `model_architect.SNP_model`."""

    def __init__(self):
        super(SNPModel, self).__init__()
        # conv1_1: kernel [1,5], stride [1,1], padding='same' -> pad (0, 2)
        self.conv1_1 = nn.Conv2d(5, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        # conv1_2: kernel [5,1], stride [1,1], padding='same' -> pad (2, 0)
        self.conv1_2 = nn.Conv2d(5, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        # conv1_3: kernel [5,5], stride [1,1], padding='same' -> pad (2, 2)
        self.conv1_3 = nn.Conv2d(5, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        # conv2/conv3: 'valid' padding (no padding) in the source.
        self.conv2 = nn.Conv2d(48, 32, kernel_size=(2, 3), stride=(1, 2), padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2), padding=0)

        # Flattened feature size for a (5, 41)-wide input (see header note):
        # after conv2 -> (4, 20), after conv3 -> (3, 9), 64 * 3 * 9 = 1728.
        self.fc1 = nn.Linear(1728, 48)
        self.dropout = nn.Dropout(0.5)

        self.fa = nn.Linear(48, 16)

        self.A = nn.Linear(16 + 1, 2)
        self.G = nn.Linear(16 + 1, 2)
        self.T = nn.Linear(16 + 1, 2)
        self.C = nn.Linear(16 + 1, 2)

        self.fc2 = nn.Linear(48, 16)
        self.fc3 = nn.Linear(16 + 2 * 4, 8)
        self.GT = nn.Linear(8, 2)

    def forward(self, x, A_ref, G_ref, T_ref, C_ref):
        c1_1 = F.selu(self.conv1_1(x))
        c1_2 = F.selu(self.conv1_2(x))
        c1_3 = F.selu(self.conv1_3(x))
        merge_conv_0 = torch.cat([c1_1, c1_2, c1_3], dim=1)  # channel dim in NCHW

        c2 = F.selu(self.conv2(merge_conv_0))
        c3 = F.selu(self.conv3(c2))

        flat_nn = c3.reshape(c3.shape[0], -1)

        fc1 = F.selu(self.fc1(flat_nn))
        drop_out_fc1 = self.dropout(fc1)

        fa = F.selu(self.fa(drop_out_fc1))

        out_A = F.softmax(self.A(torch.cat([fa, A_ref], dim=1)), dim=-1)
        out_G = F.softmax(self.G(torch.cat([fa, G_ref], dim=1)), dim=-1)
        out_T = F.softmax(self.T(torch.cat([fa, T_ref], dim=1)), dim=-1)
        out_C = F.softmax(self.C(torch.cat([fa, C_ref], dim=1)), dim=-1)

        fc2 = F.selu(self.fc2(drop_out_fc1))
        fc3 = F.selu(self.fc3(torch.cat([fc2, out_A, out_G, out_T, out_C], dim=1)))
        out_GT = F.softmax(self.GT(fc3), dim=-1)

        return out_A, out_G, out_T, out_C, out_GT


def build_nanocaller_snp_model():
    return SNPModel()


def example_input_nanocaller_snp_model():
    batch = 4
    x = torch.randn(batch, 5, 5, 41)
    A_ref = torch.randn(batch, 1)
    G_ref = torch.randn(batch, 1)
    T_ref = torch.randn(batch, 1)
    C_ref = torch.randn(batch, 1)
    return (x, A_ref, G_ref, T_ref, C_ref)


MENAGERIE_ENTRIES = [
    (
        "NanoCaller-SNP",
        "build_nanocaller_snp_model",
        "example_input_nanocaller_snp_model",
        2020,
        "ported-pytorch",
    ),
]
