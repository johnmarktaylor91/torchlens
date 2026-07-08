# FAITHFUL PORT of Aalto-QuML/ClimODE @ e729d23e8799ce0e075699e76d60227d848d8d0c (original framework: PyTorch)
#
# Source files vendored/transcribed verbatim (architecture only, no changes beyond
# minimal import surgery):
#   - model_utils.py -> boundarypad, ResidualBlock, Self_attn_conv
#   - model_function.py -> Climate_ResNet_2D, Climate_encoder_free_uncertain (the
#     "global" ClimODE model constructed in train_global.py)
#
# ClimODE (ICLR 2024 oral, https://github.com/Aalto-QuML/ClimODE) is a
# physics-informed Neural ODE for gridded weather/climate forecasting: an advection
# PDE (continuity equation) is integrated with `torchdiffeq.odeint`, where the
# velocity field and an uncertainty (noise) head are parameterized by 2D CNN
# ResNets with circular/reflect boundary padding (spherical-grid aware) and a
# self-attention correction term. The real repo depends on `utils.py` (xarray,
# torchcubicspline, properscoring, matplotlib -- pure data loading / training
# utilities, NOT part of the model architecture) which is not installed here and
# is not needed to construct or run the model itself; `model_utils.py`'s
# `from utils import *` is unused by the vendored classes below, so it is dropped.
#
# The real training script (train_global.py) constructs the model as:
#   Climate_encoder_free_uncertain(len(paths_to_data), 2, out_types=len(paths_to_data),
#                                   method=args.solver, use_att=True, use_err=True,
#                                   use_pos=False)
# and drives it via `model.update_param([past_sample, const_channels_info, lat_map,
# lon_map])` followed by `model(t, data)`. The staging harness below reproduces
# that exact call sequence with tiny synthetic ERA5-shaped tensors (5 channels,
# 8x8 grid) instead of real reanalysis data.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint

MENAGERIE_ZOO = "ported-pytorch"


class boundarypad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.pad(F.pad(input, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        x_mod = F.pad(F.pad(x, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        # Second convolution layer
        h = F.pad(F.pad(h, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class Self_attn_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Self_attn_conv, self).__init__()
        self.query = self._conv(in_channels, in_channels // 8, stride=1)
        self.key = self.key_conv(in_channels, in_channels // 8, stride=2)
        self.value = self.key_conv(in_channels, out_channels, stride=2)
        self.post_map = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0)
        )
        self.out_ch = out_channels

    def _conv(self, n_in, n_out, stride):
        return nn.Sequential(
            boundarypad(),
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            boundarypad(),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            boundarypad(),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=stride, padding=0),
        )

    def key_conv(self, n_in, n_out, stride):
        return nn.Sequential(
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=1, padding=0),
        )

    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = (
            self.query(x).flatten(-2, -1),
            self.key(x).flatten(-2, -1),
            self.value(x).flatten(-2, -1),
        )
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o


class Climate_ResNet_2D(nn.Module):
    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(
                    self.make_layer(self.block, num_channels, hidden_size[idx], layers[idx])
                )
            else:
                layers_cnn.append(
                    self.make_layer(self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx])
                )

        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self, block, in_channels, out_channels, reps):
        layers = []
        layers.append(block(in_channels, out_channels))
        self.inplanes = out_channels
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for layer in self.layer_cnn:
            dx_final = layer(dx_final)

        return dx_final


class Climate_encoder_free_uncertain(nn.Module):
    def __init__(self, num_channels, const_channels, out_types, method, use_att, use_err, use_pos):
        super().__init__()
        self.layers = [5, 3, 2]
        self.hidden = [128, 64, 2 * out_types]
        input_channels = 30 + out_types * int(use_pos) + 34 * (1 - int(use_pos))
        self.vel_f = Climate_ResNet_2D(input_channels, self.layers, self.hidden)

        if use_att:
            self.vel_att = Self_attn_conv(input_channels, 10)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels

        self.out_ch = out_types
        self.past_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.method = method
        err_in = 9 + out_types * int(use_pos) + 34 * (1 - int(use_pos))
        if use_err:
            self.noise_net = Climate_ResNet_2D(err_in, [3, 2, 2], [128, 64, 2 * out_types])
        if use_pos:
            self.pos_enc = Climate_ResNet_2D(4, [2, 1, 1], [32, 16, out_types])
        self.att = use_att
        self.err = use_err
        self.pos = use_pos
        self.pos_feat = 0
        self.lsm = 0
        self.oro = 0

    def update_param(self, params):
        self.past_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]

    def pde(self, t, vs):
        ds = vs[:, -self.out_ch :, :, :].view(-1, self.out_ch, vs.shape[2], vs.shape[3]).float()
        v = (
            vs[:, : 2 * self.out_ch, :, :]
            .view(-1, 2 * self.out_ch, vs.shape[2], vs.shape[3])
            .float()
        )
        t_emb = ((t * 100) % 24).view(1, 1, 1, 1).expand(ds.shape[0], 1, ds.shape[2], ds.shape[3])
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)

        day_emb = torch.cat([sin_t_emb, cos_t_emb], dim=1)
        seas_emb = torch.cat([sin_seas_emb, cos_seas_emb], dim=1)

        ds_grad_x = torch.gradient(ds, dim=3)[0]
        ds_grad_y = torch.gradient(ds, dim=2)[0]
        nabla_u = torch.cat([ds_grad_x, ds_grad_y], dim=1)

        if self.pos:
            comb_rep = torch.cat(
                [t_emb / 24, day_emb, seas_emb, nabla_u, v, ds, self.pos_feat], dim=1
            )
        else:
            cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(self.new_lat_map)
            cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(self.new_lon_map)
            t_cyc_emb = torch.cat([day_emb, seas_emb], dim=1)
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )
            pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_feats)
            comb_rep = torch.cat(
                [
                    t_emb / 24,
                    day_emb,
                    seas_emb,
                    nabla_u,
                    v,
                    ds,
                    self.new_lat_map,
                    self.new_lon_map,
                    self.lsm,
                    self.oro,
                    pos_feats,
                    pos_time_ft,
                ],
                dim=1,
            )

        if self.att:
            dv = self.vel_f(comb_rep) + self.gamma * self.vel_att(comb_rep)
        else:
            dv = self.vel_f(comb_rep)
        v_x = v[:, : self.out_ch, :, :].view(-1, self.out_ch, vs.shape[2], vs.shape[3]).float()
        v_y = v[:, -self.out_ch :, :, :].view(-1, self.out_ch, vs.shape[2], vs.shape[3]).float()

        adv1 = v_x * ds_grad_x + v_y * ds_grad_y
        adv2 = ds * (torch.gradient(v_x, dim=3)[0] + torch.gradient(v_y, dim=2)[0])

        ds = adv1 + adv2

        dvs = torch.cat([dv, ds], 1)
        return dvs

    def get_time_pos_embedding(self, time_feats, pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:, idx].unsqueeze(dim=1) * pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out, tf], dim=1)

        return final_out

    def noise_net_contrib(self, t, pos_enc, s_final, noise_net, H, W):
        t_emb = (t % 24).view(-1, 1, 1, 1, 1)
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(
            len(s_final), s_final.shape[1], 1, H, W
        )

        pos_enc = pos_enc.expand(len(s_final), s_final.shape[1], -1, H, W).flatten(
            start_dim=0, end_dim=1
        )
        t_cyc_emb = torch.cat([sin_t_emb, cos_t_emb, sin_seas_emb, cos_seas_emb], dim=2).flatten(
            start_dim=0, end_dim=1
        )

        pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_enc[:, 2:-2])

        comb_rep = torch.cat(
            [t_cyc_emb, s_final.flatten(start_dim=0, end_dim=1), pos_enc, pos_time_ft], dim=1
        )

        final_out = noise_net(comb_rep).view(len(t), -1, 2 * self.out_ch, H, W)

        mean = s_final + final_out[:, :, : self.out_ch]
        std = nn.Softplus()(final_out[:, :, self.out_ch :])

        return mean, std

    def forward(self, T, data, atol=0.1, rtol=0.1):
        H, W = self.past_samples.shape[2], self.past_samples.shape[3]
        final_data = torch.cat([self.past_samples, data.float().view(-1, self.out_ch, H, W)], 1)
        init_time = T[0].item() * 6
        final_time = T[-1].item() * 6
        steps_val = final_time - init_time

        if self.pos:
            lat_map = self.lat_map.unsqueeze(dim=0) * torch.pi / 180
            lon_map = self.lon_map.unsqueeze(dim=0) * torch.pi / 180
            pos_rep = torch.cat(
                [lat_map.unsqueeze(dim=0), lon_map.unsqueeze(dim=0), self.const_info], dim=1
            )
            self.pos_feat = self.pos_enc(pos_rep).expand(
                data.shape[0], -1, data.shape[3], data.shape[4]
            )
            final_pos_enc = self.pos_feat

        else:
            self.oro, self.lsm = self.const_info[0, 0], self.const_info[0, 1]
            self.lsm = self.lsm.unsqueeze(dim=0).expand(
                data.shape[0], -1, data.shape[3], data.shape[4]
            )
            self.oro = (
                F.normalize(self.const_info[0, 0])
                .unsqueeze(dim=0)
                .expand(data.shape[0], -1, data.shape[3], data.shape[4])
            )
            self.new_lat_map = (
                self.lat_map.expand(data.shape[0], 1, data.shape[3], data.shape[4]) * torch.pi / 180
            )
            self.new_lon_map = (
                self.lon_map.expand(data.shape[0], 1, data.shape[3], data.shape[4]) * torch.pi / 180
            )
            cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(self.new_lat_map)
            cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(self.new_lon_map)
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )
            final_pos_enc = torch.cat(
                [self.new_lat_map, self.new_lon_map, pos_feats, self.lsm, self.oro], dim=1
            )

        new_time_steps = torch.linspace(init_time, final_time, steps=int(steps_val) + 1).to(
            data.device
        )
        t = 0.01 * new_time_steps.float().to(data.device).flatten().float()
        pde_rhs = lambda t, vs: self.pde(t, vs)  # noqa: E731 -- matches upstream ODE-rhs closure idiom
        final_result = odeint(pde_rhs, final_data, t, method=self.method, atol=atol, rtol=rtol)
        s_final = final_result[:, :, -self.out_ch :, :, :].view(len(t), -1, self.out_ch, H, W)

        if self.err:
            mean, std = self.noise_net_contrib(
                T, final_pos_enc, s_final[0 : len(s_final) : 6], self.noise_net, H, W
            )

        else:
            s_final = s_final[0 : len(s_final) : 6]

        return mean, std, s_final[0 : len(s_final) : 6]


class ClimODEGlobalStaging(nn.Module):
    """Staging wrapper: constructs the real Climate_encoder_free_uncertain exactly as
    train_global.py does (5 ERA5 channels, use_att=True, use_err=True, use_pos=False,
    euler solver) and drives update_param() + forward() with tiny synthetic
    ERA5-shaped tensors in place of real reanalysis grids/constants."""

    def __init__(self, H=32, W=32, num_years=1, out_types=5):
        super().__init__()
        self.H = H
        self.W = W
        self.out_types = out_types
        self.model = Climate_encoder_free_uncertain(
            out_types,
            2,
            out_types=out_types,
            method="euler",
            use_att=True,
            use_err=True,
            use_pos=False,
        )
        # Real repo constants: orography/lsm from ERA5 constants file, shape (1,2,H,W)
        self.register_buffer("const_channels_info", torch.randn(1, 2, H, W))
        # Real repo: lat2d/lon2d grids in degrees, shape (H,W)
        self.register_buffer("lat_map", torch.linspace(-90, 90, H).view(H, 1).expand(H, W).clone())
        self.register_buffer("lon_map", torch.linspace(0, 360, W).view(1, W).expand(H, W).clone())
        # Real repo: fitted advection velocity, shape (num_years, 2*out_types, H, W)
        self.register_buffer("past_sample", torch.randn(num_years, 2 * out_types, H, W))

    def forward(self, data, t):
        self.model.update_param(
            [self.past_sample, self.const_channels_info, self.lat_map, self.lon_map]
        )
        mean, std, s_final = self.model(t, data)
        return mean


def build_climode_global():
    return ClimODEGlobalStaging(H=32, W=32, num_years=1, out_types=5)


def example_input_climode_global():
    # data: (num_years, 1, out_types, H, W) matching train_global.py's
    # data.view(num_years, 1, len(paths_to_data)*(scale+1), H, W)
    # H=W=32 (real repo default grid is 32x64) -- needed so the stride-2 convs
    # inside Self_attn_conv's key/value branches don't collapse the spatial dims
    # below the 3x3 kernel size.
    data = torch.randn(1, 1, 5, 32, 32)
    # t: batch of two timesteps (paper uses batch_size timesteps per window);
    # kept tiny (init/final differ by 1 six-hour step) so the euler odeint
    # integration stays cheap to trace.
    t = torch.tensor([0.0, 1.0])
    return data, t


MENAGERIE_ENTRIES = [
    (
        "ClimODE-global",
        "build_climode_global",
        "example_input_climode_global",
        2024,
        "ported-pytorch",
    ),
]
