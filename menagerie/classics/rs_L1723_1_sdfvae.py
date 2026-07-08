# SOURCE: vendored from dlagul/SDFVAE @ main
#
# File: sdfvae/model.py -> ConvUnit, ConvUnitTranspose, LinearUnit, SDFVAE
# (architecture untouched; only the constructor call and example input are new)
#
# SDFVAE (Static and Dynamic Factorized VAE) is a sequential VAE for KPI/time-series
# anomaly detection: it factorizes the latent code into a per-sequence static factor
# `s` (encoded by a bidirectional LSTM over CNN-embedded per-timestep frames) and a
# per-timestep dynamic factor `d` (an LSTMCell-driven time-dependent latent chain,
# with a matching LSTMCell prior), then reconstructs each frame with a CNN-transpose
# decoder conditioned on [d_t, h_t, s]. Repo ships only this native-PyTorch
# implementation.
#
# Repo: https://github.com/dlagul/SDFVAE

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# sdfvae/model.py
# ---------------------------------------------------------------------------
class ConvUnit(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel, stride=1, padding=0, nonlinearity=nn.LeakyReLU(0.2)
    ):
        super(ConvUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity
        )

    def forward(self, x):
        return self.model(x)


class ConvUnitTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        out_padding=0,
        nonlinearity=nn.LeakyReLU(0.2),
    ):
        super(ConvUnitTranspose, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding), nonlinearity
        )

    def forward(self, x):
        return self.model(x)


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class SDFVAE(nn.Module):
    def __init__(
        self,
        s_dim=8,
        d_dim=10,
        conv_dim=100,
        hidden_dim=40,
        T=20,
        w=36,
        n=24,
        enc_dec="CNN",
        nonlinearity=None,
        device=torch.device("cuda:0"),
    ):
        super(SDFVAE, self).__init__()

        self.s_dim = s_dim
        self.d_dim = d_dim
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.w = w
        self.n = n
        self.enc_dec = enc_dec
        self.device = device
        self.nonlinearity = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.dec_init_dim = self.s_dim + self.d_dim + self.hidden_dim

        self.d_lstm_prior = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.d_mean_prior = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_logvar_prior = nn.Linear(self.hidden_dim, self.d_dim)
        self.phi_d_prior = nn.Sequential(nn.Linear(self.d_dim, self.hidden_dim), self.nonlinearity)

        self.enc_d_prior = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Bidirectional LSTM with option bidirectional=True
        self.s_lstm = nn.LSTM(
            self.conv_dim, self.hidden_dim, 1, batch_first=True, bidirectional=True
        )

        self.s_mean = LinearUnit(self.hidden_dim * 2, self.s_dim)
        self.s_logvar = LinearUnit(self.hidden_dim * 2, self.s_dim)

        self.phi_conv = nn.Sequential(
            nn.Linear(self.conv_dim, self.hidden_dim),
            self.nonlinearity,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
        )

        self.phi_d = nn.Sequential(nn.Linear(self.d_dim, self.hidden_dim), self.nonlinearity)

        self.enc_d = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
        )

        self.d_mean = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_logvar = nn.Linear(self.hidden_dim, self.d_dim)

        self.d_rnn = nn.LSTMCell(2 * self.hidden_dim, self.hidden_dim, bias=True)

        # set up the kernel_size, stride, padding of CNN with respect to different n and w
        if self.enc_dec == "CNN":
            if self.n == 16 or self.n == 24:
                k0_0, s0_0, p0_0 = 2, 2, 0
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 38:
                k0_0, s0_0, p0_0 = 2, 2, 1
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(((s0_0 * p0_0) + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 48:
                k0_0, s0_0, p0_0 = 4, 4, 0
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            else:
                raise ValueError(
                    "Invalid Kpi numbers: {}, choose from the candidate set [16,24,38,48].".format(
                        self.n
                    )
                )

            if self.w == 36:
                k1_0, s1_0, p1_0 = 3, 3, 0
                k1_1, s1_1, p1_1 = 2, 2, 0
                k1_2, s1_2, p1_2 = 2, 2, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 144:
                k1_0, s1_0, p1_0 = 4, 4, 0
                k1_1, s1_1, p1_1 = 4, 4, 0
                k1_2, s1_2, p1_2 = 3, 3, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 288:
                k1_0, s1_0, p1_0 = 8, 8, 0
                k1_1, s1_1, p1_1 = 4, 4, 0
                k1_2, s1_2, p1_2 = 3, 3, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            else:
                raise ValueError(
                    "Invalid window size: {}, choose from the set [36,144,288]".format(self.w)
                )

            self.krl = [[k0_0, k1_0], [k0_1, k1_1], [k0_2, k1_2]]
            self.srd = [[s0_0, s1_0], [s0_1, s1_1], [s0_2, s1_2]]
            self.pd = [[p0_0, p1_0], [p0_1, p1_1], [p0_2, p1_2]]
            self.cd = [64, sd_0, sd_1]

            self.conv = nn.Sequential(
                ConvUnit(
                    1,
                    8,
                    kernel=(self.krl[0][0], self.krl[0][1]),
                    stride=(self.srd[0][0], self.srd[0][1]),
                    padding=(self.pd[0][0], self.pd[0][1]),
                ),
                ConvUnit(
                    8,
                    32,
                    kernel=(self.krl[1][0], self.krl[1][1]),
                    stride=(self.srd[1][0], self.srd[1][1]),
                    padding=(self.pd[1][0], self.pd[1][1]),
                ),
                ConvUnit(
                    32,
                    64,
                    kernel=(self.krl[2][0], self.krl[2][1]),
                    stride=(self.srd[2][0], self.srd[2][1]),
                    padding=(self.pd[2][0], self.pd[2][1]),
                ),
            )

            self.conv_fc = nn.Sequential(
                LinearUnit(self.cd[0] * self.cd[1] * self.cd[2], self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim),
            )

            self.deconv_fc_mu = nn.Sequential(
                LinearUnit(self.dec_init_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1] * self.cd[2]),
            )
            self.deconv_mu = nn.Sequential(
                ConvUnitTranspose(
                    64,
                    32,
                    kernel=(self.krl[2][0], self.krl[2][1]),
                    stride=(self.srd[2][0], self.srd[2][1]),
                    padding=(self.pd[2][0], self.pd[2][1]),
                ),
                ConvUnitTranspose(
                    32,
                    8,
                    kernel=(self.krl[1][0], self.krl[1][1]),
                    stride=(self.srd[1][0], self.srd[1][1]),
                    padding=(self.pd[1][0], self.pd[1][1]),
                ),
                ConvUnitTranspose(
                    8,
                    1,
                    kernel=(self.krl[0][0], self.krl[0][1]),
                    stride=(self.srd[0][0], self.srd[0][1]),
                    padding=(self.pd[0][0], self.pd[0][1]),
                    nonlinearity=nn.Tanh(),
                ),
            )
            self.deconv_fc_logsigma = nn.Sequential(
                LinearUnit(self.dec_init_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1] * self.cd[2]),
            )
            self.deconv_logsigma = nn.Sequential(
                ConvUnitTranspose(
                    64,
                    32,
                    kernel=(self.krl[2][0], self.krl[2][1]),
                    stride=(self.srd[2][0], self.srd[2][1]),
                    padding=(self.pd[2][0], self.pd[2][1]),
                ),
                ConvUnitTranspose(
                    32,
                    8,
                    kernel=(self.krl[1][0], self.krl[1][1]),
                    stride=(self.srd[1][0], self.srd[1][1]),
                    padding=(self.pd[1][0], self.pd[1][1]),
                ),
                ConvUnitTranspose(
                    8,
                    1,
                    kernel=(self.krl[0][0], self.krl[0][1]),
                    stride=(self.srd[0][0], self.srd[0][1]),
                    padding=(self.pd[0][0], self.pd[0][1]),
                    nonlinearity=nn.Tanh(),
                ),
            )

        else:
            raise ValueError("Unknown encoder and decoder: {}".format(self.enc_dec))

        for m in self.modules():
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.Linear)
            ):
                nn.init.kaiming_normal_(m.weight)

    # Time-dependent prior distribution of dynamic latent variables
    def sample_d_lstmcell(self, batch_size, random_sampling=True):
        d_out = None
        d_means = None
        d_logvars = None

        d_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        # Here we assume that p(d_0) = N(0,I), thus d_mean_0 = 0, d_logvar_0 = 0 due to log1 = 0
        d_mean_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        d_logvar_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(self.T):
            enc_d_t = self.enc_d_prior(h_t)
            d_mean_t = self.d_mean_prior(enc_d_t)
            d_logvar_t = self.d_logvar_prior(enc_d_t)
            d_t = self.reparameterize(d_mean_t, d_logvar_t, random_sampling)
            phi_d_t = self.phi_d_prior(d_t)
            h_t, c_t = self.d_lstm_prior(phi_d_t, (h_t, c_t))
            if d_out is None:
                d_out = d_t.unsqueeze(1)
                d_means = d_mean_t.unsqueeze(1)
                d_logvars = d_logvar_t.unsqueeze(1)
            else:
                d_out = torch.cat((d_out, d_t.unsqueeze(1)), dim=1)
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1)
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1)
        return d_means, d_logvars, d_out

    def encoder_x(self, x):
        if self.enc_dec == "CNN":
            x = x.view(-1, 1, self.n, self.w)
            x = self.conv(x)
            x = x.view(-1, self.cd[0] * self.cd[1] * self.cd[2])
            x = self.conv_fc(x)
            x = x.view(-1, self.T, self.conv_dim)
        else:
            raise ValueError("Unknown encoder and decoder: {}".format(self.enc_dec))
        return x

    def decoder_mu(self, sdh):
        if self.enc_dec == "CNN":
            x = self.deconv_fc_mu(sdh)
            x = x.view(-1, self.cd[0], self.cd[1], self.cd[2])
            x = self.deconv_mu(x)
            x = x.view(-1, self.T, 1, self.n, self.w)
        else:
            raise ValueError("Unknown encoder and decoder: {}".format(self.enc_dec))
        return x

    def decoder_logsigma(self, sdh):
        if self.enc_dec == "CNN":
            x = self.deconv_fc_logsigma(sdh)
            x = x.view(-1, self.cd[0], self.cd[1], self.cd[2])
            x = self.deconv_logsigma(x)
            x = x.view(-1, self.T, 1, self.n, self.w)
        else:
            raise ValueError("Unknown encoder and decoder: {}".format(self.enc_dec))
        return x

    # See https://arxiv.org/pdf/1606.05908.pdf, Page 11, Equation (10) for details.
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_s(self, x):
        lstm_out, _ = self.s_lstm(x)
        backward = lstm_out[:, 0, self.hidden_dim : self.hidden_dim * 2]
        frontal = lstm_out[:, 1, 0 : self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.s_mean(lstm_out)
        logvar = self.s_logvar(lstm_out)
        s = self.reparameterize(mean, logvar, self.training)
        return mean, logvar, s

    def encode_d(self, batch_size, x):
        d_out = None
        d_means = None
        d_logvars = None
        h_out = None

        d_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        d_mean_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        d_logvar_t = torch.zeros(batch_size, self.d_dim, device=self.device)

        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for t in range(self.T):
            phi_conv_t = self.phi_conv(x[:, t, :])
            enc_d_t = self.enc_d(torch.cat([phi_conv_t, h_t], 1))
            d_mean_t = self.d_mean(enc_d_t)
            d_logvar_t = self.d_logvar(enc_d_t)
            d_t = self.reparameterize(d_mean_t, d_logvar_t, self.training)
            phi_d_t = self.phi_d(d_t)
            if d_out is None:
                d_out = d_t.unsqueeze(1)
                d_means = d_mean_t.unsqueeze(1)
                d_logvars = d_logvar_t.unsqueeze(1)
                h_out = h_t.unsqueeze(1)
            else:
                d_out = torch.cat((d_out, d_t.unsqueeze(1)), dim=1)
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1)
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1)
                h_out = torch.cat((h_out, h_t.unsqueeze(1)), dim=1)
            # We update h_t here, because we want to get the tupples like (h_0, d_1), (h_1, d_2), i.e., (h_(t-1), d_t)
            h_t, c_t = self.d_rnn(torch.cat([phi_conv_t, phi_d_t], 1), (h_t, c_t))
        return d_means, d_logvars, d_out, h_out

    def forward(self, x):
        x = x.float()
        d_mean_prior, d_logvar_prior, _ = self.sample_d_lstmcell(
            x.size(0), random_sampling=self.training
        )
        x_hat = self.encoder_x(x)
        d_mean, d_logvar, d, h = self.encode_d(x.size(0), x_hat)
        s_mean, s_logvar, s = self.encode_s(x_hat)
        s_expand = s.unsqueeze(1).expand(-1, self.T, self.s_dim)
        # We concat [d_t, h_(t-1), s] and feed it to decoder network
        ds = torch.cat((d, s_expand), dim=2)
        dsh = torch.cat((ds, h), dim=2)
        recon_x_mu = self.decoder_mu(dsh)
        recon_x_logsigma = self.decoder_logsigma(dsh)
        return (
            s_mean,
            s_logvar,
            s,
            d_mean,
            d_logvar,
            d,
            d_mean_prior,
            d_logvar_prior,
            recon_x_mu,
            recon_x_logsigma,
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
# n=38, T=20, w=36 mirror the repo's Machine-1-1/Machine-1-5 SMD training scripts
# (sdfvae_scripts.txt); device pinned to CPU for the random-init trace.
def build_sdfvae():
    return SDFVAE(n=38, T=20, w=36, device=torch.device("cpu"))


def example_input_sdfvae():
    return torch.randn(2, 20, 38, 36)


MENAGERIE_ENTRIES = [
    (
        "SDFVAE",
        build_sdfvae,
        example_input_sdfvae,
        2021,
        "VENDOR",
    ),
]
