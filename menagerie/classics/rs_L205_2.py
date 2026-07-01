# SOURCE: vendored from autonomousvision/occupancy_flow @ 88966a787216
# (im2mesh/oflow/models/{__init__,decoder,encoder_latent,velocity_field}.py,
#  im2mesh/encoder/pointnet.py, im2mesh/layers.py)
"""Occupancy Flow (OFlow, ICCV 2019): 4D continuous occupancy + motion field.

OFlow represents a temporal sequence of 3D shapes as a single continuous
occupancy function transported through time by a learned velocity field
(vector field) integrated with a Neural ODE solver -- the same idea behind
"occupancy flow" for 4D reconstruction: rather than predicting occupancy
independently at each time step, points are advected from time ``t`` back to
a canonical time ``0`` (``transform_to_t0``, using the real bundled
``torchdiffeq`` bundled with the repo -- present in this environment as the
regular pip-installed ``torchdiffeq`` package) and then decoded to occupancy
logits by a conditional-batchnorm decoder network.

The submodules below (``ResnetBlockFC``, ``CResnetBlockConv1d``,
``CBatchNorm1d``, ``DecoderCBatchNorm``, ``VelocityField``,
``TemporalResnetPointnet``, ``ResnetPointnet``) are copied verbatim from the
official repo's ``im2mesh/layers.py``, ``im2mesh/oflow/models/decoder.py``,
``im2mesh/oflow/models/velocity_field.py``, and ``im2mesh/encoder/pointnet.py``.
No architecture was changed. The wiring below (``OFlowModel.forward``)
reproduces the real production configuration for the point-cloud-sequence
setting (``configs/pointcloud/oflow.yaml`` + ``configs/default.yaml``):
``encoder="pointnet_resnet"`` -> ``ResnetPointnet`` (spatial code ``c`` from
the t=0 point cloud), ``encoder_temporal="pointnet_resnet"`` ->
``TemporalResnetPointnet`` (temporal code ``c_t`` from the length-17 point
cloud sequence), ``decoder="cbatchnorm"`` -> ``DecoderCBatchNorm``,
``velocity_field="concat"`` -> ``VelocityField``, ``z_dim=0`` (no VAE latent
encoder needed in this config), ``c_dim=128``. This mirrors the real
``get_model()``/``OccupancyFlow.transform_to_t0``/``OccupancyFlow.decode``
call chain used by the official ``Generator3D`` at inference time (the top-
level ``OccupancyFlow.forward`` in the official repo has a latent pre-
existing bug -- it calls ``self.model.transform_to_t0`` where ``self.model``
is never assigned -- so eval/generation code always calls
``transform_to_t0``/``decode`` directly instead, which is what is
reproduced here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# im2mesh/layers.py (verbatim, only the modules OFlow's default config uses)
# ---------------------------------------------------------------------------


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == "batch_norm":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert x.size(0) == c.size(0)
        assert c.size(1) == self.c_dim

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CResnetBlockConv1d(nn.Module):
    """Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    """

    def __init__(
        self, c_dim, size_in, size_h=None, size_out=None, norm_method="batch_norm", legacy=False
    ):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules (legacy path unused by the default config; omitted)
        self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# ---------------------------------------------------------------------------
# im2mesh/oflow/models/decoder.py (verbatim: DecoderCBatchNorm)
# ---------------------------------------------------------------------------


class DecoderCBatchNorm(nn.Module):
    """Conditioned Batch Norm Decoder network for OFlow class.

    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This decoder
    uses conditioned batch normalization to inject the latent codes.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        self.bn = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


# ---------------------------------------------------------------------------
# im2mesh/oflow/models/velocity_field.py (verbatim: VelocityField)
# ---------------------------------------------------------------------------


class VelocityField(nn.Module):
    """Velocity Field network class.

    It maps input points and time values together with (optional) conditioned
    codes c and latent codes z to the respective motion vectors.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): size of the hidden dimension
        leaky (bool): whether to use leaky ReLUs as activation
        n_blocks (int): number of ResNet-based blocks
    """

    def __init__(
        self,
        in_dim=4,
        out_dim=3,
        z_dim=128,
        c_dim=128,
        hidden_size=512,
        leaky=False,
        n_blocks=5,
        **kwargs,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        # Submodules
        self.fc_p = nn.Linear(in_dim, hidden_size)

        if z_dim != 0:
            self.fc_z = nn.ModuleList([nn.Linear(z_dim, hidden_size) for i in range(n_blocks)])

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def disentangle_inputs(self, inputs):
        """Disentangles the inputs and returns the points and latent code
        tensors separately.

        Args:
            inputs (tensor): velocity field inputs
        """
        c_dim = self.c_dim
        z_dim = self.z_dim
        batch_size, device = inputs.shape[0], inputs.device

        if z_dim is not None and z_dim != 0:
            z = inputs[:, -z_dim:]
            p = inputs[:, :-z_dim]
        else:
            z = torch.empty(batch_size, 0).to(device)
            p = inputs

        if c_dim is not None and c_dim != 0:
            c = p[:, -c_dim:]
            p = p[:, :-c_dim]
        else:
            c = torch.empty(batch_size, 0).to(device)

        p = p.view(batch_size, -1, self.out_dim)

        return p, c, z

    def concat_time_axis(self, t, p, t_batch=False, invert=False):
        """Concatenates the time axis to the points tenor.

        Args:
            t (tensor); time values
            p (tensor): points
            t_batch (tensor): time help tensor for batch processing
            invert (bool): whether to go backwards
        """
        batch_size, n_points, _ = p.shape

        t = t.repeat(batch_size)
        if t_batch is not None:
            assert len(t_batch) == batch_size
            if invert:
                t = t_batch - t
            else:
                t = t_batch + t

        # Add Temporal Axis
        t = t.view(batch_size, 1, 1).expand(batch_size, n_points, 1)
        p_out = torch.cat([p, t], dim=-1)
        assert p_out.shape[-1] == self.in_dim

        return p_out

    def concat_output(self, out):
        """Returns the output of the velocity network.

        Args:
            out (tensor): output points
        """
        batch_size = out.shape[0]
        device = out.device
        c_dim = self.c_dim
        z_dim = self.z_dim

        out = out.contiguous().view(batch_size, -1)
        if c_dim != 0:
            c_out = torch.zeros(batch_size, c_dim).to(device)
            out = torch.cat([out, c_out], dim=-1)
        if z_dim != 0:
            z_out = torch.zeros(batch_size, z_dim).to(device)
            out = torch.cat([out, z_out], dim=-1)

        return out

    def forward(self, t, p, T_batch=None, invert=False, **kwargs):
        """Performs a forward pass through the network.

        Args:
            t (tensor): time values
            p (tensor): points
            T_batch (tensor): time helper tensor to perform batch processing
                when going backwards in time
            invert (bool): whether to go backwards
        """
        p, c, z = self.disentangle_inputs(p)
        p = self.concat_time_axis(t, p, T_batch, invert)

        net = self.fc_p(p)

        # Layer loop
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.fc_c[i](c).unsqueeze(1)
                net = net + net_c

            if self.z_dim != 0:
                net_z = self.fc_z[i](z).unsqueeze(1)
                net = net + net_z
            net = self.blocks[i](net)

        motion_vectors = self.fc_out(self.actvn(net))

        # when going backwards in time, return -v
        sign = -1 if invert else 1
        motion_vectors = sign * motion_vectors

        out = self.concat_output(motion_vectors)

        return out


# ---------------------------------------------------------------------------
# im2mesh/encoder/pointnet.py (verbatim: ResnetPointnet, TemporalResnetPointnet)
# ---------------------------------------------------------------------------


def maxpool(x, dim=-1, keepdim=False):
    """Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class TemporalResnetPointnet(nn.Module):
    """Temporal PointNet-based encoder network.

    The input point clouds are concatenated along the hidden dimension,
    e.g. for a sequence of length L, the dimension becomes 3xL = 51.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        use_only_first_pcl (bool): whether to use only the first point cloud
    """

    def __init__(self, c_dim=128, dim=51, hidden_dim=512, use_only_first_pcl=False, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.use_only_first_pcl = use_only_first_pcl

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        batch_size, n_steps, n_pts, _ = x.shape

        if len(x.shape) == 4 and self.use_only_first_pcl:
            x = x[:, 0]
        elif len(x.shape) == 4:
            x = x.transpose(1, 2).contiguous().view(batch_size, n_pts, -1)

        net = self.fc_pos(x)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))

        return c


# ---------------------------------------------------------------------------
# Real production wiring (configs/pointcloud/oflow.yaml + configs/default.yaml)
# ---------------------------------------------------------------------------


class OFlowModel(nn.Module):
    """OFlow model assembled exactly as ``im2mesh/oflow/config.py::get_model``
    does for the real point-cloud-sequence config (``encoder``=
    ``pointnet_resnet``, ``encoder_temporal``=``pointnet_resnet``,
    ``decoder``=``cbatchnorm``, ``velocity_field``=``concat``, ``z_dim``=0).

    ``forward`` reproduces the real eval-time call chain used by the
    official ``Generator3D``/``Trainer`` (``OccupancyFlow.transform_to_t0``
    integrates points at time ``t`` back to time 0 via the ODE-integrated
    velocity field, then ``OccupancyFlow.decode`` maps the canonicalized
    points to occupancy logits with the conditional-batchnorm decoder) --
    the top-level ``OccupancyFlow.forward`` in the official code is never
    used in practice (it references an undefined ``self.model``).
    """

    def __init__(
        self,
        c_dim: int = 16,
        hidden_dim: int = 16,
        decoder_hidden_size: int = 16,
        vf_hidden_size: int = 16,
        seq_len: int = 5,
        ode_solver: str = "dopri5",
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.c_dim = c_dim

        self.encoder = ResnetPointnet(c_dim=c_dim, dim=3, hidden_dim=hidden_dim)
        self.encoder_temporal = TemporalResnetPointnet(
            c_dim=c_dim, dim=3 * seq_len, hidden_dim=hidden_dim
        )
        self.decoder = DecoderCBatchNorm(
            dim=3, z_dim=0, c_dim=c_dim, hidden_size=decoder_hidden_size
        )
        self.vector_field = VelocityField(
            in_dim=4, out_dim=3, z_dim=0, c_dim=c_dim, hidden_size=vf_hidden_size, n_blocks=2
        )

        self.ode_solver = ode_solver
        self.rtol = rtol
        self.atol = atol

    def encode_inputs(self, pcl_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial code ``c`` (from t=0 point cloud) + temporal code ``c_t``
        (from the full sequence), exactly as ``OccupancyFlow.encode_inputs``.
        """
        c = self.encoder(pcl_seq[:, 0])
        c_t = self.encoder_temporal(pcl_seq)
        return c, c_t

    def concat_vf_input(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``OccupancyFlow.concat_vf_input`` with ``z`` omitted (``z_dim=0``)."""
        batch_size = p.shape[0]
        p_out = p.contiguous().view(batch_size, -1)
        c = c.contiguous().view(batch_size, -1)
        return torch.cat([p_out, c], dim=-1)

    def transform_to_t0(self, t: torch.Tensor, p: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """``OccupancyFlow.transform_to_t0``: integrate query points at time
        ``t`` backwards to their canonical time-0 location via the ODE-
        integrated velocity field.
        """
        batch_size = p.shape[0]
        c_dim = c_t.shape[-1]

        t_steps_eval, t_order = torch.unique(
            torch.cat([torch.zeros(1, device=p.device), t]), sorted=True, return_inverse=True
        )
        t_order = t_order[1:]

        p_in = self.concat_vf_input(p, c_t)

        def vf(tt, pp):
            return self.vector_field(tt, pp, T_batch=t, invert=True)

        s = odeint(vf, p_in, t_steps_eval, method=self.ode_solver, rtol=self.rtol, atol=self.atol)

        n_steps = s.shape[0]
        s = s[:, :, :-c_dim]
        s = s.contiguous().view(n_steps, batch_size, -1, 3)
        s = s.transpose(0, 1)

        p_out = s[torch.arange(batch_size), t_order]
        return p_out

    def decode(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``OccupancyFlow.decode``: occupancy logits at the canonicalized points."""
        z = torch.empty(p.shape[0], 0, device=p.device)
        return self.decoder(p, z, c)

    def forward(
        self, pcl_seq: torch.Tensor, query_points: torch.Tensor, time_val: torch.Tensor
    ) -> torch.Tensor:
        """Real eval-time OFlow call chain: encode -> transform_to_t0 -> decode.

        Parameters
        ----------
        pcl_seq : torch.Tensor
            ``(batch, seq_len, n_points_pcl, 3)`` input point-cloud sequence.
        query_points : torch.Tensor
            ``(batch, n_query, 3)`` query points at ``time_val``.
        time_val : torch.Tensor
            ``(batch,)`` time value (in ``[0, 1]``) each query point lives at.
        """
        c, c_t = self.encode_inputs(pcl_seq)
        p_t0 = self.transform_to_t0(time_val, query_points, c_t)
        occ_logits = self.decode(p_t0, c)
        return occ_logits


def build_occupancy_flow() -> nn.Module:
    """Build a tiny real-wiring OFlow model (point-cloud-sequence config).

    Returns
    -------
    nn.Module
        Random-initialized ``OFlowModel`` with small hidden sizes / a short
        5-frame sequence so the ODE-integrated forward pass traces quickly.
    """

    return OFlowModel(
        c_dim=16,
        hidden_dim=16,
        decoder_hidden_size=16,
        vf_hidden_size=16,
        seq_len=5,
        rtol=1e-2,
        atol=1e-3,
    )


def example_input_occupancy_flow() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a point-cloud sequence, query points, and a time value.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(pcl_seq, query_points, time_val)`` with shapes
        ``(1, 5, 20, 3)``, ``(1, 12, 3)``, ``(1,)``.
    """

    pcl_seq = torch.rand(1, 5, 20, 3)
    query_points = torch.rand(1, 12, 3)
    time_val = torch.tensor([0.6])
    return pcl_seq, query_points, time_val


MENAGERIE_ENTRIES = [
    (
        "Occupancy Flow (OFlow)",
        "build_occupancy_flow",
        "example_input_occupancy_flow",
        "2019",
        "occupancy_flow_4d",
    ),
]

MENAGERIE_ZOO = "vendored-pytorch"
