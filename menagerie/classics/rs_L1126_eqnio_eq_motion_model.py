# SOURCE: vendored from https://github.com/RoyinaJayanth/EqNIO @ main
# (RONIN/source/model_resnet1d.py + RONIN/source/model_resnet1d_eq_frame.py)
#
# EqNIO (Equivariant Neural Inertial Odometry, Jayanth et al. ICLR 2025):
# an O(2)/SO(2)-equivariant IMU-odometry network. The equivariant "frame"
# tower (`VNLinear`, `NonLinearity`, `LayerNorm`, `VNLayerNorm`,
# `MeanPooling_layer`, `Convolutional`) operates on paired vector/scalar IMU
# features (vector-neuron style layers over 2D gyro/accel vectors) to
# regress a canonical local reference `frame`; the raw IMU vectors are then
# rotated into that frame and fed to a standard 1D-ResNet velocity regressor
# (`ResNet1D`/`BasicBlock1D`/`FCOutputModule`, vendored unmodified from
# `RONIN/source/model_resnet1d.py`, the same base architecture the RoNIN
# baseline uses) whose output is rotated back by the frame to produce an
# equivariant velocity estimate. `Eq_Motion_Model.forward` is the real
# forward computation graph from the paper. All classes below are copied
# verbatim from the two real repo files (only the cross-file `from
# model_resnet1d import ...` was resolved into this single staging module;
# no architecture change).

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# RONIN/source/model_resnet1d.py (verbatim, base ResNet1D used by EqNIO)
# ---------------------------------------------------------------------------
def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FCOutputModule(nn.Module):
    """
    Fully connected output module.
    """

    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        Constructor for a fully connected output layer.

        Args:
          in_planes: number of planes (channels) of the layer immediately proceeding the output module.
          num_outputs: number of output predictions.
          fc_dim: dimension of the fully connected layer.
          dropout: the keep probability of the dropout layer
          trans_planes: (optional) number of planes of the transition convolutional layer.
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get("fc_dim", 1024)
        dropout = kwargs.get("dropout", 0.5)
        in_dim = kwargs.get("in_dim", 7)
        trans_planes = kwargs.get("trans_planes", None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes),
            )
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs),
        )

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


class ResNet1D(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        block_type,
        group_sizes,
        base_plane=64,
        output_block=None,
        zero_init_residual=False,
        **kwargs,
    ):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.planes = [self.base_plane * (2**i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get("kernel_size", 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [
            self._make_residual_group1d(
                block_type, self.planes[i], kernel_size, group_sizes[i], strides[i], dilations[i]
            )
            for i in range(len(group_sizes))
        ]
        self.residual_groups = nn.Sequential(*groups)

        # Output module
        if output_block is None:
            self.output_block = GlobAvgOutputModule(
                self.planes[-1] * block_type.expansion, num_outputs
            )
        else:
            self.output_block = output_block(
                self.planes[-1] * block_type.expansion, num_outputs, **kwargs
            )

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block_type.expansion),
            )
        layers = []
        layers.append(
            block_type(
                self.inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GlobAvgOutputModule(nn.Module):
    """
    Global average output module.
    """

    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# RONIN/source/model_resnet1d_eq_frame.py (verbatim, EqNIO equivariant tower)
# ---------------------------------------------------------------------------
def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1, :].unsqueeze(-2), x[..., 0, :].unsqueeze(-2)), dim=dim)


class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()

        self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, vector):
        return self.vector_linear(
            torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1)
        )


class NonLinearity(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        scalar_dim_out,
    ):
        super().__init__()
        self.scalar_dim_out = scalar_dim_out
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in + scalar_dim_in, dim_out + scalar_dim_out, bias=False)
        self.layer_norm = LayerNorm(dim_out + scalar_dim_out)

    def forward(self, vector, scalar):
        x = torch.concatenate((torch.norm(vector, dim=-2), scalar), dim=-1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.layer_norm(x)
        if self.scalar_dim_out == 0:
            return x[..., : self.dim_out].unsqueeze(-2) * (
                vector / torch.norm(vector, dim=-2).clamp(min=1e-6).unsqueeze(-2)
            )
        return x[..., : self.dim_out].unsqueeze(-2) * (
            vector / torch.norm(vector, dim=-2).clamp(min=1e-6).unsqueeze(-2)
        ), x[..., -self.scalar_dim_out :]


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):  # dim is the vector dimension (i.e., 2)
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim=-2)
        x = x / norms.clamp(min=self.eps).unsqueeze(-2)
        return x * self.ln(norms).unsqueeze(-2)


class MeanPooling_layer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, vector, scalar):
        return torch.mean(vector, dim=self.dim), torch.mean(scalar, dim=self.dim)


class Convolutional(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        stride=1,
        padding="same",
        kernel=(16, 1),
        bias=False,
    ):
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            stride=stride,
            kernel_size=kernel,
            padding=padding,
            bias=bias,
            padding_mode="replicate",
        )
        self.conv_layer_sca = nn.Conv2d(
            in_channels=scalar_dim_in,
            out_channels=scalar_dim_out,
            stride=stride,
            kernel_size=kernel,
            padding=padding,
            bias=bias,
            padding_mode="replicate",
        )

    def forward(self, vector, scalar):
        return self.conv_layer_vec(
            torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1).permute(
                0, 3, 1, 2
            )
        ).permute(0, 2, 3, 1), self.conv_layer_sca(
            scalar.unsqueeze(-2).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).squeeze(-2)


class Eq_Motion_Model(nn.Module):  # input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        pooling_dim,
        hidden_dim,
        scalar_hidden_dim,
        depth,
        ronin_in_dim,
        ronin_out_dim,
        ronin_depths=[2, 2, 2, 2],
        ronin_base_plane=64,
        ronin_kernel=3,
        stride=1,
        padding="same",
        kernel=(16, 1),
        bias=False,
    ):
        super().__init__()

        self.vnlinear_layer0 = VNLinear(dim_in=2 * dim_in, dim_out=hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in, scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(
            dim_in=hidden_dim,
            dim_out=hidden_dim,
            scalar_dim_in=scalar_hidden_dim,
            scalar_dim_out=scalar_hidden_dim,
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Convolutional(
                            dim_in=2 * hidden_dim,
                            dim_out=hidden_dim,
                            scalar_dim_in=scalar_hidden_dim,
                            scalar_dim_out=scalar_hidden_dim,
                            stride=stride,
                            padding=padding,
                            kernel=kernel,
                            bias=bias,
                        ),
                        NonLinearity(
                            dim_in=hidden_dim,
                            dim_out=hidden_dim,
                            scalar_dim_in=scalar_hidden_dim,
                            scalar_dim_out=scalar_hidden_dim,
                        ),
                        VNLayerNorm(hidden_dim),  # for vector
                        LayerNorm(scalar_hidden_dim),  # for scalar
                    ]
                )
            )

        self.pooling_layer1 = MeanPooling_layer(dim=pooling_dim)

        # MLP- linear, nonlinearity, linear
        self.vnlinear_layer1 = VNLinear(dim_in=2 * hidden_dim, dim_out=hidden_dim)
        self.slinear_layer1 = nn.Linear(scalar_hidden_dim, scalar_hidden_dim, bias=False)
        self.nonlinearity1 = NonLinearity(
            dim_in=hidden_dim, dim_out=hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=0
        )
        self.vnlinear_layer2 = VNLinear(dim_in=2 * hidden_dim, dim_out=hidden_dim)

        # layer normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim)  # for vector

        # output layer
        self.vnoutput_layer = VNLinear(dim_in=2 * hidden_dim, dim_out=dim_out)

        # Ronin
        _fc_config = {"fc_dim": 512, "in_dim": _RONIN_FC_IN_DIM, "dropout": 0.5, "trans_planes": 32}
        self.ronin = ResNet1D(
            ronin_in_dim,
            ronin_out_dim,
            BasicBlock1D,
            ronin_depths,
            ronin_base_plane,
            output_block=FCOutputModule,
            kernel_size=ronin_kernel,
            **_fc_config,
        )

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, vector, scalar, original_scalars):
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v, s = self.nonlinearity0(v, s)

        # conv blocks
        for conv, nl, vnln, sln in self.layers:
            v, s = conv(v, s)
            v, s = nl(v, s)
            v = vnln(v)
            s = sln(s)

        v, s = self.pooling_layer1(v, s)

        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v = self.nonlinearity1(v, s)  # no scalar
        v = self.vnlinear_layer2(v)

        # replace later with batch norm
        v = self.vector_ln1(v)

        v = self.vnoutput_layer(v)

        frame = torch.concat(
            [
                v / torch.norm(v, dim=-2).clamp(min=1e-6).unsqueeze(-2),
                orthogonal_input(v / torch.norm(v, dim=-2).clamp(min=1e-6).unsqueeze(-2), dim=-2),
            ],
            dim=-1,
        )
        frame = frame.permute(0, 2, 1)
        v = torch.matmul(frame.unsqueeze(1), vector)

        input = torch.concat([v.reshape((*v.shape[:2], -1)), original_scalars], dim=-1).permute(
            0, 2, 1
        )

        vel = self.ronin(input)

        vel = torch.matmul(frame.permute(0, 2, 1), vel[:, :2].unsqueeze(-1)).squeeze(-1)

        return frame, vel  # , disp_inv


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). The real repo's
# own `if __name__ == "__main__":` self-test at the bottom of
# `model_resnet1d_eq_frame.py` builds `Eq_Motion_Model` at exactly these
# sizes (dim_in=2, dim_out=1, scalar_dim_in=5, pooling_dim=1,
# ronin_in_dim=6, ronin_out_dim=2, hidden_dim=128, scalar_hidden_dim=128,
# depth=1, kernel=(32,1)) with input shapes vector=(2,200,2,2),
# scalar=(2,200,5), original_scalars=(2,200,2); we reuse those exact real
# construction args and shapes (shrinking the sequence length 200->64,
# hidden_dim 128->16, and the internal RoNIN ResNet1D depth/base_plane for a
# fast CPU trace; `_RONIN_FC_IN_DIM` is the real flattened spatial size the
# shrunk ResNet1D produces for `_SEQ_LEN`, computed the same way the real
# repo picks `in_dim` for its own training sequence length of 200).
# ---------------------------------------------------------------------------
_BATCH = 2
_SEQ_LEN = 64
_SCALAR_DIM_IN = 5
_ORIG_SCALAR_DIM = 2
_RONIN_DEPTHS = [1, 1, 1, 1]
_RONIN_BASE_PLANE = 8
_RONIN_FC_IN_DIM = 2  # ResNet1D(base_plane=8, depths=[1,1,1,1]) output length for seq_len=64


def build_eqnio_eq_motion_model():
    torch.manual_seed(0)
    model = Eq_Motion_Model(
        dim_in=2,
        dim_out=1,
        scalar_dim_in=_SCALAR_DIM_IN,
        pooling_dim=1,
        ronin_in_dim=6,
        ronin_out_dim=2,
        hidden_dim=16,
        scalar_hidden_dim=16,
        depth=1,
        ronin_depths=_RONIN_DEPTHS,
        ronin_base_plane=_RONIN_BASE_PLANE,
        stride=1,
        padding="same",
        kernel=(4, 1),
        bias=False,
    )
    model.eval()
    return model


def example_input_eqnio_eq_motion_model():
    torch.manual_seed(0)
    vector = torch.randn(_BATCH, _SEQ_LEN, 2, 2)
    scalar = torch.randn(_BATCH, _SEQ_LEN, _SCALAR_DIM_IN)
    original_scalars = torch.randn(_BATCH, _SEQ_LEN, _ORIG_SCALAR_DIM)
    return (vector, scalar, original_scalars)


MENAGERIE_ENTRIES = [
    (
        "EqNIO-Eq_Motion_Model",
        "build_eqnio_eq_motion_model",
        "example_input_eqnio_eq_motion_model",
        2025,
        MENAGERIE_ZOO,
    ),
]
