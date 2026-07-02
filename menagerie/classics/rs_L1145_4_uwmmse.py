# SOURCE: vendored from lsky96/gcnwmmse @ main
# https://github.com/lsky96/gcnwmmse/blob/main/comm/reference_wmmse_unrolls.py
# https://github.com/lsky96/gcnwmmse/blob/main/comm/network.py
# https://github.com/lsky96/gcnwmmse/blob/main/comm/mathutil.py
# "Unfolding WMMSE using Graph Neural Networks for Efficient Power
# Allocation" (Chowdhury, Verma, Rao, Segarra & Ribeiro 2021, the UWMMSE
# algorithm). `GCNLayer` / `GCN` (from `network.py`, a graph-shift-operator
# convolution stack used to predict the per-user MMSE step-size
# hyperparameters `a`,`b`) and `UWMMSELayer` (from
# `reference_wmmse_unrolls.py`, the u/w/v unfolded-WMMSE update step with
# the two GCNs replacing the closed-form step sizes) are transcribed
# VERBATIM from the real repo. `mmchain` (from `mathutil.py`) is copied
# unchanged; the rest of `mathutil.py` (complex-space helpers, plotting)
# is unused by these classes and dropped, along with the outer `UWMMSE`
# class's scenario-dict/dataset plumbing (`scenario_extraction`,
# `vectorize`/`devectorize`, complex-valued channel bookkeeping) which is
# simulation-pipeline glue, not architecture. `Model` here re-implements
# only the real `UWMMSE.__init__`'s `nn.ModuleList` layer-stacking loop
# and a straightforward `forward` over the vendored `UWMMSELayer`, driven
# directly by the real per-user power-control tensors
# (`channel_mat`, `user_noise_pow`, `bss_pow`, `v_in`) that
# `UWMMSELayer.forward` itself consumes, instead of the dataset-specific
# scenario dict.
import torch
import torch.nn as nn
import torch.nn.functional as F


def mmchain(*args, **kwargs):
    """
    Calculates the chain of matrix products of two or more matrix batches
    :param args: two or more matrix batches
    :return: matrix product
    """
    if len(args) >= 2:
        return torch.matmul(args[0], mmchain(*args[1:]), **kwargs)
    else:
        return args[0]


class GCNLayer(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_features_out,
        num_operators,
        num_space="real",
        activation="relu",
        weightinit="normal",  # "pos"
        biasinit=0,
        device=torch.device("cpu"),
    ):
        super(GCNLayer, self).__init__()

        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.num_operators = num_operators
        self.activation = activation
        self.num_space = num_space
        self.weightinit = weightinit
        self.biasinit = biasinit
        self.device = device
        dtype = torch.float64

        if num_space == "real":
            if self.weightinit == "normal":
                self.filter_taps = nn.Parameter(
                    torch.randn(
                        self.num_operators,
                        self.num_features_in,
                        self.num_features_out,
                        device=self.device,
                        dtype=dtype,
                    )
                    / torch.sqrt(
                        torch.tensor(
                            self.num_features_in + self.num_features_out, device=self.device
                        )
                    )
                )
            elif self.weightinit == "pos":
                self.filter_taps = nn.Parameter(
                    torch.rand(
                        self.num_operators,
                        self.num_features_in,
                        self.num_features_out,
                        device=self.device,
                        dtype=dtype,
                    )
                    / torch.sqrt(
                        torch.tensor(
                            self.num_features_in + self.num_features_out, device=self.device
                        )
                    )
                )
            elif self.weightinit == "glorot":
                self.filter_taps = nn.Parameter(
                    (
                        2
                        * torch.rand(
                            self.num_operators,
                            self.num_features_in,
                            self.num_features_out,
                            device=self.device,
                            dtype=dtype,
                        )
                        - 1
                    )
                    * torch.sqrt(
                        torch.tensor(6, device=self.device)
                        / (self.num_features_in + self.num_features_out)
                    )
                )
            else:
                raise ValueError
            self.bias = nn.Parameter(
                torch.ones(self.num_features_out, device=self.device, dtype=dtype) * self.biasinit
            )
        else:
            raise ValueError

    def forward(self, x, gso):
        """
        :param x: (*batch_size, N, feat_in)
        :param gso: (*batch_size, num_operators, N, N), can omit batch_size dimensions
        :return:
        """
        N = x.size()[-2]
        batch_size = list(x.size())[:-2]
        x_temp = torch.reshape(x, (*batch_size, 1, N, self.num_features_in))
        x_out = mmchain(gso, x_temp, self.filter_taps).sum(dim=-3) + self.bias

        if self.activation is None:
            pass
        elif self.activation == "relu":
            x_out = F.relu(x_out)
        elif self.activation == "sigmoid":
            x_out = torch.sigmoid(x_out)
        else:
            raise ValueError

        return x_out


class GCN(nn.Module):
    def __init__(
        self,
        num_features,  # list of length num_layers + 1, (in_feat, .., out_feat)
        num_layers,
        num_operators,  # number of shift matrices (for example individual exponents of it etc.)
        activations,  # list of None, relu, sigmoid
        num_space="real",
        recomb=False,  # linear recombination of features of last layer, feat_out will then be 1
        weightinit="normal",  # "normal", "pos"
        biasinit=0,
        device=torch.device("cpu"),
    ):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_operators = num_operators
        self.activations = activations
        self.num_space = num_space
        self.recomb = recomb
        self.weightinit = weightinit
        self.biasinit = biasinit
        self.device = device

        self.layers = []
        for i_layer in range(num_layers):
            self.layers.append(
                GCNLayer(
                    self.num_features[i_layer],
                    self.num_features[i_layer + 1],
                    self.num_operators,
                    num_space=self.num_space,
                    activation=activations[i_layer],
                    weightinit=self.weightinit,
                    biasinit=self.biasinit,
                    device=self.device,
                )
            )
        self.layers = nn.ModuleList(self.layers)
        self.recomb_mat = None

    def forward(self, x, gso):
        """
        :param x: (*batch_size, N, feat_in)
        :param gso: (*batch_size, num_operators, N, N), can omit batch_size dimensions
        :return: (*batch_size, N, feat_out)
        """
        x_out = x
        for i_layer in range(self.num_layers):
            x_out = self.layers[i_layer](x_out, gso)

        if self.recomb_mat is not None:
            x_out = torch.matmul(x_out, self.recomb_mat)

        return x_out


class UWMMSELayer(nn.Module):
    def __init__(self, num_features, num_gcn_layers, device=torch.device("cpu")):
        super(UWMMSELayer, self).__init__()
        self.num_features = num_features
        self.num_gcn_layers = num_gcn_layers
        self.device = device

        self.gcn_features = [1] + [self.num_features] * (self.num_gcn_layers - 1) + [1]
        self.gcn_act = ["relu"] * (self.num_gcn_layers - 1) + ["sigmoid"]
        self.gcn_a = GCN(
            self.gcn_features,
            self.num_gcn_layers,
            2,
            self.gcn_act,
            weightinit="glorot",
            biasinit=0.1,
        )
        self.gcn_b = GCN(
            self.gcn_features,
            self.num_gcn_layers,
            2,
            self.gcn_act,
            weightinit="glorot",
            biasinit=0.1,
        )

    def forward(self, channel_mat, user_noise_pow, bss_pow, v_in):
        eps = 1e-12

        def u_step(channel_mat, user_noise_pow, v):
            u_tilde = torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v
            cov = torch.matmul(channel_mat.square(), v.square()).sum(
                dim=-1, keepdim=True
            ) + user_noise_pow.unsqueeze(-1)
            u = u_tilde / cov
            return u

        def w_step(channel_mat, v, u):
            channel_mat_diag = torch.diag_embed(
                torch.diagonal(channel_mat, dim1=-2, dim2=-1), dim1=-2, dim2=-1
            )
            gso_ops = torch.stack((channel_mat_diag, channel_mat), dim=-3)
            ones = torch.ones_like(v)

            a = self.gcn_a(ones, gso_ops)
            b = self.gcn_b(ones, gso_ops)

            error = 1 - u * torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v

            w = a / error + b

            return w

        def v_step(channel_mat, bss_pow, u, w):
            v_tilde = u * torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * w
            ul_cov = torch.square(u) * w
            ul_cov = torch.matmul(torch.square(channel_mat).transpose(-2, -1), ul_cov).sum(
                dim=-1, keepdim=True
            )
            v = v_tilde / (ul_cov + eps)
            v = torch.clamp(v, min=0)
            v_max = torch.sqrt(bss_pow).unsqueeze(-1)  # same size as v
            overshoot = v - v_max > 0
            v[overshoot] = v_max[overshoot]
            return v

        u_out = u_step(channel_mat, user_noise_pow, v_in)
        w_out = w_step(channel_mat, v_in, u_out)
        v_out = v_step(channel_mat, bss_pow, u_out, w_out)

        return v_out, u_out, w_out


class Model(nn.Module):
    """Stacks `num_layers` UWMMSELayer unfolded-WMMSE iterations, matching
    the real `UWMMSE.__init__`'s `nn.ModuleList` construction."""

    def __init__(self, num_layers=3, num_features=4, num_gcn_layers=2):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(UWMMSELayer(num_features, num_gcn_layers))
        self.layers = nn.ModuleList(layers)

    def forward(self, channel_mat, user_noise_pow, bss_pow, v_in):
        v = v_in
        for i_layer in range(self.num_layers):
            v, u, w = self.layers[i_layer](channel_mat, user_noise_pow, bss_pow, v)
        return v, u, w


MENAGERIE_ZOO = "vendored-pytorch"


def build_uwmmse():
    torch.manual_seed(0)
    model = Model(num_layers=3, num_features=4, num_gcn_layers=2)
    model.eval()
    return model


def example_input_uwmmse():
    torch.manual_seed(0)
    dtype = torch.float64
    num_users = 5
    batch_size = 2
    channel_mat = torch.rand(batch_size, num_users, num_users, dtype=dtype) + 0.1
    user_noise_pow = torch.rand(batch_size, num_users, dtype=dtype) + 0.01
    bss_pow = torch.rand(batch_size, num_users, dtype=dtype) + 1.0
    v_in = torch.rand(batch_size, num_users, 1, dtype=dtype) + 0.1
    return (channel_mat, user_noise_pow, bss_pow, v_in)


MENAGERIE_ENTRIES = [
    ("UWMMSE", "build_uwmmse", "example_input_uwmmse", 2021, MENAGERIE_ZOO),
]
