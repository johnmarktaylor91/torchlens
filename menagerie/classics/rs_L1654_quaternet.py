# SOURCE: vendored from facebookresearch/QuaterNet @ main
# https://github.com/facebookresearch/QuaterNet -- "QuaterNet: A Quaternion-based
# Recurrent Model for Human Motion" (Pavllo, Grangier, Auli, BMVC 2018). The core
# architectural contribution is a 2-layer GRU that predicts per-joint quaternion
# rotations (with an optional velocity-modeling quaternion-multiplication block and
# a learned initial hidden state), used autoregressively for short-term and
# long-term human motion prediction. `QuaterNet` (common/quaternet.py) and `qmul`
# (common/quaternion.py) are transcribed verbatim; only the `PoseNetwork` training
# wrapper (numpy dataset iteration, Adam optimizer loop) was dropped since it is a
# downstream training harness, not part of the QuaterNet architecture itself.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from common/quaternion.py ----
def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


# ---- verbatim from common/quaternet.py ----
class QuaterNet(nn.Module):
    def __init__(self, num_joints, num_outputs=0, num_controls=0, model_velocities=False):
        """
        Construct a QuaterNet neural network.
        Arguments:
         -- num_joints: number of skeleton joints.
         -- num_outputs: extra inputs/outputs (e.g. translations), in addition to joint rotations.
         -- num_controls: extra input-only features.
         -- model_velocities: add a quaternion multiplication block on the RNN output to force
                              the network to model velocities instead of absolute rotations.
        """
        super().__init__()

        self.num_joints = num_joints
        self.num_outputs = num_outputs
        self.num_controls = num_controls

        if num_controls > 0:
            fc1_size = 30
            fc2_size = 30
            self.fc1 = nn.Linear(num_controls, fc1_size)
            self.fc2 = nn.Linear(fc1_size, fc2_size)
            self.relu = nn.LeakyReLU(0.05, inplace=True)
        else:
            fc2_size = 0

        h_size = 1000
        self.rnn = nn.GRU(
            input_size=num_joints * 4 + num_outputs + fc2_size,
            hidden_size=h_size,
            num_layers=2,
            batch_first=True,
        )
        self.h0 = nn.Parameter(
            torch.zeros(self.rnn.num_layers, 1, h_size).normal_(std=0.01), requires_grad=True
        )

        self.fc = nn.Linear(h_size, num_joints * 4 + num_outputs)
        self.model_velocities = model_velocities

    def forward(self, x, h=None, return_prenorm=False, return_all=False):
        """
        Run a forward pass of this model.
        Arguments:
         -- x: input tensor of shape (N, L, J*4 + O + C), where N is the batch size, L is the sequence length,
               J is the number of joints, O is the number of outputs, and C is the number of controls.
               Features must be provided in the order J, O, C.
         -- h: hidden state. If None, it defaults to the learned initial state.
         -- return_prenorm: if True, return the quaternions prior to normalization.
         -- return_all: if True, return all L frames, otherwise return only the last frame. If only the latter
                        is wanted (e.g. when conditioning the model with an initialization sequence), this
                        argument should be left to False as it avoids unnecessary computation.
        """
        assert len(x.shape) == 3
        assert x.shape[-1] == self.num_joints * 4 + self.num_outputs + self.num_controls

        x_orig = x
        if self.num_controls > 0:
            controls = x[:, :, self.num_joints * 4 + self.num_outputs :]
            controls = self.relu(self.fc1(controls))
            controls = self.relu(self.fc2(controls))
            x = torch.cat((x[:, :, : self.num_joints * 4 + self.num_outputs], controls), dim=2)

        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, h = self.rnn(x, h)
        if return_all:
            x = self.fc(x)
        else:
            x = self.fc(x[:, -1:])
            x_orig = x_orig[:, -1:]

        pre_normalized = x[:, :, : self.num_joints * 4].contiguous()
        normalized = pre_normalized.view(-1, 4)
        if self.model_velocities:
            normalized = qmul(
                normalized, x_orig[:, :, : self.num_joints * 4].contiguous().view(-1, 4)
            )
        normalized = F.normalize(normalized, dim=1).view(pre_normalized.shape)

        if self.num_outputs > 0:
            x = torch.cat((normalized, x[:, :, self.num_joints * 4 :]), dim=2)
        else:
            x = normalized

        if return_prenorm:
            return x, h, pre_normalized
        else:
            return x, h


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_quaternet():
    torch.manual_seed(0)
    model = QuaterNet(num_joints=20, num_outputs=4, num_controls=0, model_velocities=True)
    model.eval()
    return model


def example_input_quaternet():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 8
    num_joints = 20
    num_outputs = 4
    x = torch.randn(batch_size, seq_len, num_joints * 4 + num_outputs)
    return (x,)


MENAGERIE_ENTRIES = [
    ("QuaterNet", build_quaternet, example_input_quaternet, 2018, "vendored-pytorch"),
]
