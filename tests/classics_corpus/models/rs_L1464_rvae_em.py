# SOURCE: vendored from Audio-WestlakeU/RVAE-EM @ 3398d13ca903
# https://raw.githubusercontent.com/Audio-WestlakeU/RVAE-EM/3398d13ca903/model/RVAE.py
# https://raw.githubusercontent.com/Audio-WestlakeU/RVAE-EM/3398d13ca903/model/module.py
#
# RVAE-EM (Pengyu Wang et al., "RVAE-EM: Generative Speech Dereverberation Based on
# Recurrent Variational Auto-Encoder and Convolutive Transfer Function", ICASSP 2024)
# combines a recurrent variational autoencoder (RVAE) speech-spectrum prior with a
# classical EM-based convolutive-transfer-function dereverberation step. The RVAE
# (`model/RVAE.py::RVAE`) is the trainable generative network -- the EM step
# (`model/my_EM.py`) is a signal-processing routine, not an nn.Module, and is not part
# of the vendored network here (the RVAE decoder/encoder IS the neural architecture).
# The encoder is a causal-conv front end (`net_pre_conv_enc` -> `ResBlock2D` ->
# `net_post_conv_enc`) feeding a backward (flipped) GRU over the per-frame log-spectral
# feature (`gru_x_enc`), fused per-timestep with an autoregressive latent-state GRU
# (`gru_z_enc`) that reads the PREVIOUS sampled z_t, then two parallel MLP heads produce
# the Gaussian posterior mean/logvar per timestep (matching a recurrent/DKS-style VAE
# posterior, sampled via reparameterization at every step inside a Python for-loop over
# `seq_len`). The decoder runs a forward GRU over z (`gru_x_dec`), reshapes into a
# 2-channel "image", and passes it through a mirrored conv/ResBlock2D/conv stack to
# reconstruct the (exponentiated) log-spectrum.
#
# Vendored verbatim from `model/RVAE.py` (class `RVAE`) and its `model/module.py`
# helpers (`build_conv2d`, `ResBlock2D`, `build_GRU`, `build_MLP`, `reparametrization`,
# `init_weights`, `activation_func`) -- every layer, its constructor args, and every
# permute/reshape/concat in `encoder`/`decoder`/`forward` is unchanged.
#
# Config values transcribed from the repo's real training config `config/config_S.json`
# ("model" block): dim_x=512, dim_z=32, gru_dim_x_enc=512, gru_dim_z_enc=256,
# gru_dim_x_dec=512, pre_conv_enc=[1,64,1,1,0], pre_conv_dec=[2,64,1,1,0],
# resblock_enc/dec=[[64,64,3,1,1],[64,64,3,1,1]], post_conv_enc/dec=[64,1,1,1,0],
# num_resblock=8, num_GRU_layer_enc/dec=1, dense_zmean_zlogvar=[256,256,32],
# dense_activation_type="tanh", dropout_p=0.2, batch_norm=false. All conv/GRU/dense
# dimensions below are shrunk uniformly (dim_x, dim_z, GRU hidden sizes, num_resblock,
# ResBlock channel widths) purely for a fast trace; the architecture graph
# (encoder/decoder topology, per-timestep autoregressive loop, reparameterization) is
# unchanged.
#
# Only base-lib deps used: torch, torch.nn, torch.nn.functional.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def activation_func(type: str):
    if type.upper() == "RELU":
        nn_activation = nn.ReLU(inplace=False)
    elif type.upper() == "TANH":
        nn_activation = nn.Tanh()
    elif type.upper() == "LEAKYRELU":
        nn_activation = nn.LeakyReLU(0.1, inplace=False)
    else:
        raise ValueError("Unavailable activation type")
    return nn_activation


def build_MLP(dim_in: int, dense: list, activation_type: str, dropout_p: float = 0):
    nn_activation = activation_func(activation_type)
    dic_layers = OrderedDict()
    if len(dense) == 0:
        dic_layers["Identity"] = nn.Identity()
        dim_y = dim_in
    else:
        for n in range(len(dense)):
            if n == 0:
                dic_layers["Linear" + str(n)] = nn.Linear(dim_in, dense[n])
            else:
                dic_layers["Linear" + str(n)] = nn.Linear(dense[n - 1], dense[n])
            if n != len(dense) - 1:
                dic_layers["activation" + str(n)] = nn_activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=dropout_p)
        dim_y = dense[-1]

    return nn.Sequential(dic_layers), dim_y


def build_conv2d(config: list, name: str, norm: bool = False):
    dic_layers = OrderedDict()
    [ch_in, ch_out, kernel_size, stride, padding] = config
    dic_layers[name] = nn.Conv2d(
        ch_in,
        ch_out,
        (kernel_size, kernel_size),
        (stride, stride),
        (padding, padding),
    )
    if norm:
        dic_layers[name + "_bn"] = torch.nn.BatchNorm2d(
            num_features=ch_out, affine=True, track_running_stats=True
        )
    return nn.Sequential(dic_layers)


class ResBlock2D(torch.nn.Module):
    def __init__(self, config: list, norm: bool = False, nblock: int = 1, dropout_p: float = 0):
        super(ResBlock2D, self).__init__()
        self.c = config
        self.convs = nn.ModuleList([])
        self.n_layers = len(config)
        self.norm = norm
        for iblock in range(nblock):
            for n in range(self.n_layers):
                [ch_in, ch_out, kernel_size, stride, padding] = config[n]
                self.convs.append(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        (kernel_size, kernel_size),
                        (stride, stride),
                        (padding, padding),
                    )
                )
        self.convs.apply(init_weights)
        if norm:
            self.bn = nn.ModuleList([])
            for iblock in range(nblock):
                [_, ch, _, _, _] = config[0]
                self.bn.append(
                    torch.nn.BatchNorm2d(num_features=ch, affine=True, track_running_stats=True)
                )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        i = 0
        x_res = x
        for conv in self.convs:
            iblock = i // self.n_layers
            i = i + 1
            xt = F.leaky_relu(x, 0.1, inplace=False)
            x = conv(xt)
            x = self.dropout(x)

            if i % self.n_layers == 0:
                if self.norm:
                    bn = self.bn[iblock]
                    x = bn(x)
                x = x + x_res
                x_res = x
        return x


def build_GRU(dim_in: int, dim_hidden: int, num_layers: int, bidir: bool):
    gru = nn.GRU(
        input_size=dim_in,
        hidden_size=dim_hidden,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidir,
    )
    gru.flatten_parameters()
    return gru


def reparametrization(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch.addcmul(mean, eps, std)


class RVAE(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        gru_dim_x_enc: int,
        gru_dim_z_enc: int,
        gru_dim_x_dec: int,
        pre_conv_enc: list,
        pre_conv_dec: list,
        resblock_enc: list,
        resblock_dec: list,
        post_conv_enc: list,
        post_conv_dec: list,
        num_resblock: int,
        num_GRU_layer_enc: int,
        num_GRU_layer_dec: int,
        dense_zmean_zlogvar: list,
        dense_activation_type: str,
        dropout_p: float,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.net_pre_conv_enc = build_conv2d(pre_conv_enc, "Conv_pre_enc", False)
        self.net_conv_enc = ResBlock2D(resblock_enc, batch_norm, num_resblock, dropout_p)
        self.net_post_conv_enc = build_conv2d(post_conv_enc, "Conv_post_enc", False)
        self.gru_x_enc = build_GRU(dim_x, gru_dim_x_enc, num_GRU_layer_enc, True)
        self.gru_z_enc = build_GRU(dim_z, gru_dim_z_enc, num_GRU_layer_enc, False)
        dim_h_enc = gru_dim_x_enc * 2 + gru_dim_z_enc
        assert dense_zmean_zlogvar[-1] == dim_z
        self.mlp_zmean_enc, _ = build_MLP(
            dim_h_enc, dense_zmean_zlogvar, dense_activation_type, dropout_p
        )
        self.mlp_zlogvar_enc, _ = build_MLP(
            dim_h_enc, dense_zmean_zlogvar, dense_activation_type, dropout_p
        )
        self.gru_x_dec = build_GRU(dim_z, gru_dim_x_dec, num_GRU_layer_dec, True)
        self.net_pre_conv_dec = build_conv2d(pre_conv_dec, "Conv_pre_dec", False)
        self.net_conv_dec = ResBlock2D(resblock_dec, batch_norm, num_resblock, dropout_p)
        self.net_post_conv_dec = build_conv2d(post_conv_dec, "Conv_post_dec", False)
        self.num_GRU_layer_enc = num_GRU_layer_enc
        self.num_GRU_layer_dec = num_GRU_layer_dec
        self.dim_hz_enc = gru_dim_z_enc
        self.dim_z = dim_z

    def encoder(self, x):
        device = x.device
        x = (x + 1e-8).log()
        bs, seq_len, dim_feature = x.shape

        z = torch.zeros([bs, seq_len, self.dim_z]).to(device)
        z_t = torch.zeros([bs, self.dim_z]).to(device)
        h_hz_t_enc = torch.zeros([self.num_GRU_layer_enc, bs, self.dim_hz_enc]).to(device)

        zmean = torch.zeros([bs, seq_len, self.dim_z]).to(device)
        zlogvar = torch.zeros([bs, seq_len, self.dim_z]).to(device)

        x = x.unsqueeze(1)
        x_temp = self.net_pre_conv_enc(x)
        x_temp = self.net_conv_enc(x_temp)
        x_temp = self.net_post_conv_enc(x_temp)
        hx_in_enc = x_temp.squeeze(1)
        hx_enc, _ = self.gru_x_enc(torch.flip(hx_in_enc, [1]))
        hx_enc = torch.flip(hx_enc, [1])

        for t in range(seq_len):
            hz_t_in_enc = z_t.unsqueeze(1)
            hz_t_enc, h_hz_t_enc = self.gru_z_enc(hz_t_in_enc, h_hz_t_enc)
            hz_t_enc = hz_t_enc.squeeze(1)
            h_t_enc = torch.cat([hx_enc[:, t, :], hz_t_enc], -1)
            zmean_t = self.mlp_zmean_enc(h_t_enc)
            zlogvar_t = self.mlp_zlogvar_enc(h_t_enc)
            z_t = reparametrization(zmean_t, zlogvar_t)
            z[:, t, :] = z_t
            zmean[:, t, :] = zmean_t
            zlogvar[:, t, :] = zlogvar_t

        return z, zmean, zlogvar

    def decoder(self, z):
        [bs, seq_len, _] = z.shape
        h_dec, _ = self.gru_x_dec(z)
        h_dec = h_dec.reshape([bs, seq_len, 2, -1])
        h_dec = h_dec.permute(0, 2, 1, 3)

        h_dec = self.net_pre_conv_dec(h_dec)
        h_dec = self.net_conv_dec(h_dec)
        h_dec = self.net_post_conv_dec(h_dec)
        logx = h_dec.squeeze(1)
        x = logx.exp()
        return x

    def forward(self, x):
        z, zmean, zlogvar = self.encoder(x)
        x_reconstruct = self.decoder(z)
        return x_reconstruct, zmean, zlogvar, z


def build_rvae_em():
    torch.manual_seed(0)
    # Shrunk from config/config_S.json's "model" block (dim_x=512, dim_z=32,
    # gru_dim_x_enc=512, gru_dim_z_enc=256, gru_dim_x_dec=512, num_resblock=8,
    # channel width 64) for a fast trace; architecture/topology unchanged.
    model = RVAE(
        dim_x=16,
        dim_z=4,
        gru_dim_x_enc=8,
        gru_dim_z_enc=6,
        gru_dim_x_dec=8,
        pre_conv_enc=[1, 4, 1, 1, 0],
        pre_conv_dec=[2, 4, 1, 1, 0],
        resblock_enc=[[4, 4, 3, 1, 1], [4, 4, 3, 1, 1]],
        resblock_dec=[[4, 4, 3, 1, 1], [4, 4, 3, 1, 1]],
        post_conv_enc=[4, 1, 1, 1, 0],
        post_conv_dec=[4, 1, 1, 1, 0],
        num_resblock=1,
        num_GRU_layer_enc=1,
        num_GRU_layer_dec=1,
        dense_zmean_zlogvar=[6, 4],
        dense_activation_type="tanh",
        dropout_p=0.2,
        batch_norm=False,
    )
    model.eval()
    return model


def example_input_rvae_em():
    torch.manual_seed(0)
    # x: (batch, seq_len, dim_x) magnitude spectrogram (must be > 0 since encoder()
    # takes log(x + eps)).
    return torch.rand(2, 5, 16) + 0.1


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RVAE-EM", "build_rvae_em", "example_input_rvae_em", 2024, MENAGERIE_ZOO),
]
