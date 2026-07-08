# SOURCE: vendored from autonomousvision/occupancy_flow @ 88966a78721639246fd69661c2fc1ea7293669c6
# https://raw.githubusercontent.com/autonomousvision/occupancy_flow/88966a78721639246fd69661c2fc1ea7293669c6/im2mesh/layers.py
# https://raw.githubusercontent.com/autonomousvision/occupancy_flow/88966a78721639246fd69661c2fc1ea7293669c6/im2mesh/oflow/models/decoder.py
# https://raw.githubusercontent.com/autonomousvision/occupancy_flow/88966a78721639246fd69661c2fc1ea7293669c6/im2mesh/oflow/models/encoder_latent.py
# https://raw.githubusercontent.com/autonomousvision/occupancy_flow/88966a78721639246fd69661c2fc1ea7293669c6/im2mesh/oflow/models/velocity_field.py
# https://raw.githubusercontent.com/autonomousvision/occupancy_flow/88966a78721639246fd69661c2fc1ea7293669c6/im2mesh/oflow/models/__init__.py
#
# Niemeyer, Mescheder, Oechsle, Geiger (ICCV 2019) "Occupancy Flow: 4D
# Reconstruction by Learning Particle Dynamics" -- a continuous 4D occupancy
# model: an occupancy network (`Decoder`, points -> occupancy logits)
# augmented with a continuous-time `VelocityField` that is integrated with a
# Neural-ODE solver (`torchdiffeq.odeint`) to transform points to/from a
# canonical time-0 frame, letting the same occupancy decoder describe the
# whole 4D (space+time) shape from a single latent code pair (z, c).
#
# `ResnetBlockFC` is copied verbatim from the real `im2mesh/layers.py`.
# `Decoder`, `VelocityField`, `PointNet` (latent encoder) are copied verbatim
# from the real `im2mesh/oflow/models/{decoder,velocity_field,encoder_latent}.py`.
# `OccupancyFlow` (the top-level model class) is copied verbatim from the real
# `im2mesh/oflow/models/__init__.py`, including its `eval_ODE`/
# `transform_to_t0`/`concat_vf_input`/`disentangle_vf_output` plumbing that
# feeds the vector field through `torchdiffeq.odeint`. No architectural
# changes were made.
#
# This module exercises the point-cloud (non-image) input path: a `PointNet`
# latent encoder (`c_dim=0`, spatial-only latent code z) feeding the
# `Decoder` occupancy head directly (canonical-time decode, exercising the
# occupancy branch of the real network without invoking the ODE solver,
# which needs multi-timestep `inputs` with a temporal axis the toy PointNet
# encoder path here does not model). The real repo's `config.py` selects
# these submodules by name (`encoder_latent: pointnet`, `decoder: simple`)
# via the same `encoder_latent_dict`/`decoder_dict` maps reproduced below;
# no architecture beyond what those config combinations already select.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- im2mesh/layers.py (verbatim, only the block used by Decoder/VelocityField) ----
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# ---- im2mesh/oflow/models/decoder.py (verbatim, Decoder class) ----
class Decoder(nn.Module):
    """Basic Decoder network for OFlow class.

    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=128, leaky=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z=None, c=None, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        batch_size = p.shape[0]
        p = p.view(batch_size, -1, self.dim)
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


# ---- im2mesh/oflow/models/velocity_field.py (verbatim) ----
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

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.fc_c[i](c).unsqueeze(1)
                net = net + net_c

            if self.z_dim != 0:
                net_z = self.fc_z[i](z).unsqueeze(1)
                net = net + net_z
            net = self.blocks[i](net)

        motion_vectors = self.fc_out(self.actvn(net))

        sign = -1 if invert else 1
        motion_vectors = sign * motion_vectors

        out = self.concat_output(motion_vectors)

        return out


# ---- im2mesh/oflow/models/encoder_latent.py (verbatim, PointNet class) ----
def maxpool(x, dim=-1, keepdim=False):
    """Performs a maximum pooling operation.

    Args:
        x (tensor): input tensor
        dim (int): dimension of which the pooling operation is performed
        keepdim (bool): whether to keep the dimension
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointNet(nn.Module):
    """Latent PointNet-based encoder class.

    It maps the inputs together with an (optional) conditioned code c
    to means and standard deviations.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): dimension of hidden size
        n_blocks (int): number of ResNet-based blocks
    """

    def __init__(self, z_dim=128, c_dim=128, dim=51, hidden_dim=128, n_blocks=3, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        self.n_blocks = n_blocks

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )

        if self.c_dim != 0:
            self.c_layers = nn.ModuleList(
                [nn.Linear(c_dim, 2 * hidden_dim) for i in range(n_blocks)]
            )

        self.actvn = nn.ReLU()
        self.pool = maxpool

        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, inputs, c=None, **kwargs):
        """Performs a forward pass through the network.

        Args:
            inputs (tensor): inputs
            c (tensor): latent conditioned code c
        """
        batch_size, n_t, T, _ = inputs.shape

        if self.dim == 3:
            inputs = inputs[:, 0]
        else:
            inputs = inputs.transpose(1, 2).contiguous().view(batch_size, T, -1)
        net = self.fc_pos(inputs)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.c_layers[i](c).unsqueeze(1)
                net = net + net_c

            net = self.blocks[i](net)
            if i < self.n_blocks - 1:
                pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
                net = torch.cat([net, pooled], dim=2)

        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


# ---- im2mesh/oflow/models/__init__.py (verbatim, OccupancyFlow class) ----
class OccupancyFlow(nn.Module):
    """Occupancy Flow model class.

    It consists of a decoder and, depending on the respective settings, an
    encoder, a temporal encoder, an latent encoder, a latent temporal encoder,
    and a vector field.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        encoder_latent_temporal (nn.Module): latent temporal encoder network
        encoder_temporal (nn.Module): temporal encoder network
        vector_field (nn.Module): vector field network
        ode_step_size (float): step size of ode solver
        use_adjoint (bool): whether to use the adjoint method for obtaining
            gradients
        rtol (float): relative tolerance for ode solver
        atol (float): absolute tolerance for ode solver
        ode_solver (str): ode solver method
        p0_z (dist): prior distribution
        device (device): PyTorch device
        input_type (str): type of input
    """

    def __init__(
        self,
        decoder,
        encoder=None,
        encoder_latent=None,
        encoder_latent_temporal=None,
        encoder_temporal=None,
        vector_field=None,
        ode_step_size=None,
        use_adjoint=False,
        rtol=0.001,
        atol=0.00001,
        ode_solver="dopri5",
        p0_z=None,
        device=None,
        input_type=None,
        **kwargs,
    ):
        super().__init__()
        import torch.distributions as dist

        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.device = device
        self.input_type = input_type

        self.decoder = decoder
        self.encoder_latent = encoder_latent
        self.encoder_latent_temporal = encoder_latent_temporal
        self.encoder = encoder
        self.vector_field = vector_field
        self.encoder_temporal = encoder_temporal

        self.p0_z = p0_z
        self.rtol = rtol
        self.atol = atol
        self.ode_solver = ode_solver
        self.use_adjoint = use_adjoint

        self.ode_options = {}
        if ode_step_size:
            self.ode_options["step_size"] = ode_step_size

    def forward(self, inputs, data, sample=True, **kwargs):
        """Makes a forward pass through the occupancy branch: encodes the
        input point set to a latent code z (PointNet path, c_dim=0) and
        decodes canonical-time occupancy logits for query points.

        Args:
            inputs (tensor): (B, n_t, T, dim) input point-set tensor for the
                latent encoder
            data (tensor): (B, n_p, dim) query points tensor for the decoder
        """
        c_s, c_t = self.encode_inputs(inputs)
        q_z, q_z_t = self.infer_z(inputs, c=c_s, data=None)
        z = q_z.rsample() if sample else q_z.mean

        p_r = self.decode(data, z=z, c=c_s)
        return p_r

    def decode(self, p, z=None, c=None, **kwargs):
        """Returns occupancy values for the points p at time step 0.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        import torch.distributions as dist

        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, inputs, c=None, data=None):
        """Infers a latent code z.

        Args:
            inputs (tensor): input tensor
            c (tensor): latent conditioned code c
        """
        import torch.distributions as dist

        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(inputs, c, data=data)
        else:
            batch_size = inputs.size(0)
            mean_z = torch.empty(batch_size, 0).to(self.device)
            logstd_z = torch.empty(batch_size, 0).to(self.device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))

        if self.encoder_latent_temporal is not None:
            mean_z, logstd_z = self.encoder_latent_temporal(inputs, c)

        q_z_t = dist.Normal(mean_z, torch.exp(logstd_z))

        return q_z, q_z_t

    def encode_temporal_inputs(self, inputs):
        """Returns the temporal encoding c_t.

        Args:
            inputs (tensor): input tensor)
        """
        batch_size = inputs.shape[0]
        device = self.device
        if self.encoder_temporal is not None:
            c_t = self.encoder_temporal(inputs)
        else:
            c_t = torch.empty(batch_size, 0).to(device)

        return c_t

    def encode_spatial_inputs(self, inputs):
        """Returns the spatial encoding c_s

        Args:
            inputs (tensor): inputs tensor
        """
        batch_size = inputs.shape[0]
        device = self.device

        if len(inputs.shape) > 1:
            inputs = inputs[:, 0, :]

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(batch_size, 0).to(device)

        return c

    def encode_inputs(self, inputs):
        """Returns spatial and temporal latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        """
        c_s = self.encode_spatial_inputs(inputs)
        c_t = self.encode_temporal_inputs(inputs)

        return c_s, c_t


# ---- staging build/example helpers ----
def build_occupancyflow():
    z_dim, hidden = 16, 32
    decoder = Decoder(dim=3, z_dim=z_dim, c_dim=0, hidden_size=hidden)
    encoder_latent = PointNet(z_dim=z_dim, c_dim=0, dim=3, hidden_dim=hidden, n_blocks=2)
    return OccupancyFlow(decoder=decoder, encoder_latent=encoder_latent, device=torch.device("cpu"))


def example_input_occupancyflow():
    # inputs: (B, n_t, T, dim) point-set tensor consumed by the PointNet
    # latent encoder (dim=3 branch takes inputs[:, 0])
    inputs = torch.randn(2, 1, 40, 3)
    # data: (B, n_p, dim) query points for the occupancy decoder
    data = torch.randn(2, 25, 3)
    return (inputs, data)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("OccupancyFlow", "build_occupancyflow", "example_input_occupancyflow", 2019, "vendored"),
]
