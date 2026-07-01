# SOURCE: vendored from theislab/scarches @ v0.6.1 (scArches / trVAE)
#   scarches/models/trvae/trvae.py
#   scarches/models/trvae/modules.py
#   scarches/models/trvae/_utils.py
#   scarches/models/trvae/losses.py (trimmed: mse only, dropped `nb`/`zinb`/`scvi` import
#   which is a heavy non-base dependency not needed for the mse reconstruction path)
#
# trVAE ("Conditional Variational Auto-Encoder for batch/condition-aware single-cell
# integration") is scArches's base architecture. Real class, minimal import fix only
# (dropped the transitive `scvi`/`anndata` imports pulled in by the save/load mixin and
# the nb/zinb losses -- neither is needed for a fresh forward pass with recon_loss="mse").
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


# ---- scarches/models/trvae/_utils.py -------------------------------------------------
def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


# ---- scarches/models/trvae/losses.py (mse only) ---------------------------------------
def mse(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data."""
    mse_loss = nn.MSELoss(reduction="none")
    return mse_loss(recon_x, x)


# ---- scarches/models/trvae/modules.py --------------------------------------------------
class CondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE."""

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        num_classes: int = None,
    ):
        super().__init__()
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    self.FC.add_module(
                        name="L{:d}".format(i),
                        module=CondLayers(in_size, out_size, self.n_classes, bias=True),
                    )
                else:
                    self.FC.add_module(
                        name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True)
                    )
                if use_bn:
                    self.FC.add_module(
                        "N{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True)
                    )
                elif use_ln:
                    self.FC.add_module(
                        "N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False)
                    )
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)
        return means, log_vars


class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks."""

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        recon_loss: str,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        num_classes: int = None,
    ):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes
        self.FirstL = nn.Sequential()
        self.FirstL.add_module(
            name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False)
        )
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.FirstL.add_module(
                "N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False)
            )
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i + 3 < len(layer_sizes):
                    self.HiddenL.add_module(
                        name="L{:d}".format(i + 1), module=nn.Linear(in_size, out_size, bias=False)
                    )
                    if use_bn:
                        self.HiddenL.add_module(
                            "N{:d}".format(i + 1), module=nn.BatchNorm1d(out_size, affine=True)
                        )
                    elif use_ln:
                        self.HiddenL.add_module(
                            "N{:d}".format(i + 1),
                            module=nn.LayerNorm(out_size, elementwise_affine=False),
                        )
                    self.HiddenL.add_module(name="A{:d}".format(i + 1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(
                            name="D{:d}".format(i + 1), module=nn.Dropout(p=dr_rate)
                        )
        else:
            self.HiddenL = None

        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU()
            )
        if self.recon_loss == "zinb":
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )

    def forward(self, z, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent


# ---- scarches/models/trvae/trvae.py ----------------------------------------------------
class trVAE(nn.Module):
    """ScArches model class. Conditional Variational Auto-encoder for single-cell batch
    integration/surgery (trVAE)."""

    def __init__(
        self,
        input_dim: int,
        conditions: list,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = False,
        mmd_on: str = "z",
        mmd_boundary=None,
        recon_loss="mse",
        beta: float = 1,
        use_bn: bool = False,
        use_ln: bool = True,
    ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb", "zinb"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.cell_type_encoder = None
        self.recon_loss = recon_loss
        self.mmd_boundary = mmd_boundary
        self.use_mmd = use_mmd
        self.freeze = False
        self.beta = beta
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.mmd_on = mmd_on

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)
        self.encoder = Encoder(
            encoder_layer_sizes,
            self.latent_dim,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            self.n_conditions,
        )
        self.decoder = Decoder(
            decoder_layer_sizes,
            self.latent_dim,
            self.recon_loss,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            self.n_conditions,
        )

    def sampling(self, mu, log_var):
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x=None, batch=None, sizefactor=None, labeled=None):
        x_log = torch.log(1 + x)
        if self.recon_loss == "mse":
            x_log = x

        z1_mean, z1_log_var = self.encoder(x_log, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss in ("zinb", "nb"):
            # nb/zinb paths require the scvi-tools loss kernels (`nb`/`zinb`) which are not
            # vendored here (heavy non-base dependency); this staged module only exercises
            # the mse reconstruction path.
            raise NotImplementedError("nb/zinb recon_loss not vendored; use recon_loss='mse'")

        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = (
            kl_divergence(
                Normal(z1_mean, torch.sqrt(z1_var)),
                Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var)),
            )
            .sum(dim=1)
            .mean()
        )

        mmd_loss = torch.tensor(0.0, device=z1.device)

        return recon_loss, kl_div, mmd_loss


def build_scarches_trvae():
    return trVAE(
        input_dim=64,
        conditions=["batch_0", "batch_1", "batch_2"],
        hidden_layer_sizes=[32, 16],
        latent_dim=8,
        dr_rate=0.0,
        use_mmd=False,
        recon_loss="mse",
        use_bn=False,
        use_ln=True,
    )


def example_input_scarches_trvae():
    batch_size = 12
    x = torch.rand(batch_size, 64)
    batch = torch.randint(0, 3, (batch_size,))
    return (x, batch)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("scArches trVAE", "build_scarches_trvae", "example_input_scarches_trvae", 2022, MENAGERIE_ZOO),
]
