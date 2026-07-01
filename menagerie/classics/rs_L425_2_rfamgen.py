# SOURCE: vendored from Shunsuke-1994/rfamgen @ main
# https://raw.githubusercontent.com/Shunsuke-1994/rfamgen/main/src/models/CMVAE.py
# https://raw.githubusercontent.com/Shunsuke-1994/rfamgen/main/src/encoder.py
# https://raw.githubusercontent.com/Shunsuke-1994/rfamgen/main/src/decoder.py
#
# Sumi, S., Hamada, M. & Saito, H. (2024, Nature Communications) "Deep generative
# design of RNA family sequences" -- RfamGen's `CovarianceModelVAE` (CM-VAE) is a
# convolutional VAE over a covariance-model (Rfam CM) grammar encoding of an RNA
# alignment: three parallel Conv1d encoder stacks (`tr_encode`/`s_encode`/
# `p_encode`, one per grammar-rule-transition/state/pairwise-emission channel
# group) feed a shared FC bottleneck producing `mu`/`logvar`; the reparameterized
# latent `z` is decoded by three matching ConvTranspose1d stacks
# (`tr_decode`/`s_decode`/`p_decode`) that reconstruct per-channel-group logits.
# `CovarianceModelEncoder`/`CovarianceModelDecoder`/`CovarianceModelVAE` are
# copied verbatim from the real `src/encoder.py`/`src/decoder.py`/
# `src/models/CMVAE.py` (only the `sample()` reparameterization trick's
# `Normal(...).sample()` device placement is unchanged from source); no
# architectural changes were made. The `MyDataset` (h5py-backed) class and the
# CLI-facing `build_from_config` staticmethod (which reads a YAML config via the
# repo's own `util.load_config`) are NOT vendored here since they require the
# repo's own data/`util`/`preprocess` modules that are unrelated to the model
# architecture; instead `build_rfamgen_cmvae()` below constructs
# `CovarianceModelVAE` directly with small tr/s/p sequence lengths, mirroring
# what `build_from_config` does at trace-relevant granularity.

import torch
from torch import nn
from torch.distributions import Normal


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class View(nn.Module):
    def __init__(self, dim1, dim2):
        super(View, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.dim1, self.dim2)


class CovarianceModelEncoder(nn.Module):
    """
    Convolutional encoder for CM-VAE.
    Applies a series of one-dimensional convolutions to a batch
    of tr/s/p encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(
        self,
        tr_len,
        s_len,
        p_len,
        hidden_size=435,
        z_dim=46,
        stride=1,
        conv_params={"ker1": 5, "ch1": 5, "ker2": 5, "ch2": 5, "ker3": 7, "ch3": 8},
    ):
        """
        shape: (n_seq, n_rules)
        """
        super(CovarianceModelEncoder, self).__init__()

        self.ker1, self.ch1 = conv_params["ker1"], conv_params["ch1"]
        self.ker2, self.ch2 = conv_params["ker2"], conv_params["ch2"]
        self.ker3, self.ch3 = conv_params["ker3"], conv_params["ch3"]

        self.bn = True
        self.stride = stride
        self.n_fc = 0

        # calculation of unit number after 3 convs.
        # see: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def get_lastlen(len_sequence, dilation=1, padding=0):
            len_sequence = int(
                (
                    (len_sequence + 2 * (self.ker1 // 2) - dilation * (self.ker1 - 1) - 1)
                    / self.stride
                )
                + 1
            )
            len_sequence = int(
                ((len_sequence + 2 * padding - dilation * (self.ker2 - 1) - 1) / 1) + 1
            )
            len_sequence = int(
                ((len_sequence + 2 * padding - dilation * (self.ker3 - 1) - 1) / 1) + 1
            )
            return len_sequence

        self.mu = nn.Linear(hidden_size, z_dim)
        self.logvar = nn.Linear(hidden_size, z_dim)

        self.tr_encode = nn.Sequential(
            nn.Conv1d(
                56, self.ch1, kernel_size=self.ker1, padding=self.ker1 // 2, stride=self.stride
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding=0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding=0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten(),
        )

        self.s_encode = nn.Sequential(
            nn.Conv1d(
                4, self.ch1, kernel_size=self.ker1, padding=self.ker1 // 2, stride=self.stride
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding=0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding=0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten(),
        )

        self.p_encode = nn.Sequential(
            nn.Conv1d(
                16, self.ch1, kernel_size=self.ker1, padding=self.ker1 // 2, stride=self.stride
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding=0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding=0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten(),
        )

        self.fcn = nn.Sequential(
            nn.Linear(
                (get_lastlen(tr_len) + get_lastlen(s_len) + get_lastlen(p_len)) * self.ch3,
                hidden_size,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        tr, s, p = x
        h_tr = self.tr_encode(tr)
        h_s = self.s_encode(s)
        h_p = self.p_encode(p)
        h = torch.cat((h_tr, h_s, h_p), dim=1)
        h = self.fcn(h)  # (some, hidden_dim) x (batch, some) -> (batch, hidden_dim)
        return self.mu(h), self.logvar(h)


class CovarianceModelDecoder(nn.Module):
    """
    Convolutional encoder for CM-VAE(split type).
    Applies a series of one-dimensional convolutions to a batch
    of tr/s/p encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(
        self,
        tr_len,
        s_len,
        p_len,
        hidden_size=435,
        z_dim=46,
        stride=1,
        conv_params={"ker1": 5, "ch1": 5, "ker2": 5, "ch2": 5, "ker3": 7, "ch3": 8},
    ):
        """
        shape: (n_seq, n_rules)
        """
        super(CovarianceModelDecoder, self).__init__()

        self.ker1, self.ch1 = conv_params["ker1"], conv_params["ch1"]
        self.ker2, self.ch2 = conv_params["ker2"], conv_params["ch2"]
        self.ker3, self.ch3 = conv_params["ker3"], conv_params["ch3"]
        self.bn = True
        self.stride = stride
        self.n_fc = 0

        # calculation of unit number after 3 convs.
        # see: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def get_padding_param(tr_len, s_len, p_len):
            def outpads(leng):
                # calc on encoder side
                conv1_in = leng
                conv2_in = int(((conv1_in + 2 * (self.ker1 // 2) - self.ker1) / self.stride) + 1)
                conv3_in = int(((conv2_in + 2 * 0 - self.ker2) / 1) + 1)
                conv3_out = int(((conv3_in + 2 * 0 - self.ker3) / 1) + 1)

                deconv3_out = int((conv3_out - 1) * 1 - 2 * 0 + (self.ker3 - 1) + 1)
                deconv2_out = int((deconv3_out - 1) * 1 - 2 * 0 + (self.ker2 - 1) + 1)
                deconv1_out = int(
                    (deconv2_out - 1) * self.stride - 2 * (self.ker1 // 2) + (self.ker1 - 1) + 1
                )

                outpad3 = conv3_in - deconv3_out
                outpad2 = conv2_in - deconv2_out
                outpad1 = conv1_in - deconv1_out
                return conv3_out, (outpad1, outpad2, outpad3)

            tr_conv3out, tr_outpads = outpads(tr_len)
            s_conv3out, s_outpads = outpads(s_len)
            p_conv3out, p_outpads = outpads(p_len)
            return (tr_conv3out, s_conv3out, p_conv3out), tr_outpads, s_outpads, p_outpads

        self.leng, tr_outpads, s_outpads, p_outpads = get_padding_param(tr_len, s_len, p_len)
        self.fcn_x2 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.ch3 * sum(self.leng)),
            View(self.ch3, sum(self.leng)),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
        )

        outpad1, outpad2, outpad3 = tr_outpads
        self.tr_decode = nn.Sequential(
            nn.ConvTranspose1d(
                self.ch3,
                self.ch2,
                kernel_size=self.ker3,
                padding=0,
                stride=1,
                output_padding=outpad3,
            ),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch2,
                self.ch1,
                kernel_size=self.ker2,
                padding=0,
                stride=1,
                output_padding=outpad2,
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch1,
                56,
                kernel_size=self.ker1,
                padding=self.ker1 // 2,
                stride=self.stride,
                output_padding=outpad1,
            ),
            nn.BatchNorm1d(56),
            nn.ReLU(),
        )

        outpad1, outpad2, outpad3 = s_outpads
        self.s_decode = nn.Sequential(
            nn.ConvTranspose1d(
                self.ch3,
                self.ch2,
                kernel_size=self.ker3,
                padding=0,
                stride=1,
                output_padding=outpad3,
            ),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch2,
                self.ch1,
                kernel_size=self.ker2,
                padding=0,
                stride=1,
                output_padding=outpad2,
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch1,
                4,
                kernel_size=self.ker1,
                padding=self.ker1 // 2,
                stride=self.stride,
                output_padding=outpad1,
            ),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        outpad1, outpad2, outpad3 = p_outpads
        self.p_decode = nn.Sequential(
            nn.ConvTranspose1d(
                self.ch3,
                self.ch2,
                kernel_size=self.ker3,
                padding=0,
                stride=1,
                output_padding=outpad3,
            ),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch2,
                self.ch1,
                kernel_size=self.ker2,
                padding=0,
                stride=1,
                output_padding=outpad2,
            ),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.ch1,
                16,
                kernel_size=self.ker1,
                padding=self.ker1 // 2,
                stride=self.stride,
                output_padding=outpad1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

    def forward(self, z):
        """Encode x into a mean and variance of a Normal"""
        h = self.fcn_x2(z)
        h_tr, h_s, h_p = (
            h[:, :, : self.leng[0]],
            h[:, :, self.leng[0] : -self.leng[2]],
            h[:, :, -self.leng[2] :],
        )
        return self.tr_decode(h_tr), self.s_decode(h_s), self.p_decode(h_p)


class CovarianceModelVAE(nn.Module):
    """
    CM-VAE. Encode and Decode CM or alignment on CM.
    """

    def __init__(
        self,
        hidden_encoder_size,
        z_dim,
        hidden_decoder_size,
        tr_len,
        s_len,
        p_len,
        stride=1,
        conv_params={"ker1": 5, "ch1": 5, "ker2": 5, "ch2": 5, "ker3": 7, "ch3": 8},
    ):
        super(CovarianceModelVAE, self).__init__()
        self.tr_len = tr_len
        self.s_len = s_len
        self.p_len = p_len
        self.z_dim = z_dim
        self.hidden_encoder_size = hidden_encoder_size
        self.hidden_decoder_size = hidden_decoder_size
        self.stride = stride
        self.conv_params = conv_params

        # Real source pins this to CUDA-if-available; pinned to CPU here so the
        # staging module traces deterministically in a CPU-only env (device
        # selection is orthogonal to the architecture).
        self.device = torch.device("cpu")
        self.encoder = CovarianceModelEncoder(
            self.tr_len,
            self.s_len,
            self.p_len,
            self.hidden_encoder_size,
            self.z_dim,
            stride=self.stride,
            conv_params=conv_params,
        ).to(self.device)
        self.decoder = CovarianceModelDecoder(
            self.tr_len,
            self.s_len,
            self.p_len,
            self.hidden_decoder_size,
            self.z_dim,
            stride=self.stride,
            conv_params=conv_params,
        ).to(self.device)

    def sample(self, mu, logvar):
        """Reparametrized sample from a N(mu, sigma) distribution
        input: (mu, logvar)
        """
        sigma = (0.5 * logvar).exp()
        normal = Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        eps = normal.sample().to(self.device)
        z = mu + eps * sigma
        return z

    def kl(self, mu, logvar):
        """KL divergence between two normal distributions"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        logits = self.decoder(z)
        return logits


def build_rfamgen_cmvae():
    # Real defaults from RfamGen's shipped RF00234 config.yaml: hidden=435,
    # z_dim=46, stride=1, conv_params={ker1:5,ch1:5,ker2:5,ch2:5,ker3:7,ch3:8}.
    # tr/s/p sequence lengths are shrunk from the real ~hundreds-of-positions
    # alignment widths to small placeholders (tr_len=40, s_len=24, p_len=24)
    # for a fast trace; the get_lastlen()/get_padding_param() shape-inference
    # helpers (copied verbatim from source) make the encoder/decoder self-
    # consistent at any length.
    model = CovarianceModelVAE(
        hidden_encoder_size=32,
        z_dim=8,
        hidden_decoder_size=32,
        tr_len=40,
        s_len=24,
        p_len=24,
        stride=1,
        conv_params={"ker1": 5, "ch1": 5, "ker2": 5, "ch2": 5, "ker3": 7, "ch3": 8},
    )
    model.eval()
    return model


def example_input_rfamgen_cmvae():
    # (tr, s, p) one-hot-ish channel-grouped encodings of a CM-grammar rule
    # sequence, each shaped (batch, channels, length) exactly as
    # `MyDataset.__getitem__` yields after its transpose(-2, -1): tr has 56
    # channels (transition-rule one-hot), s has 4 channels (single-emission
    # nucleotide one-hot), p has 16 channels (pairwise-emission dinucleotide
    # one-hot).
    batch = 2
    tr = torch.rand(batch, 56, 40)
    s = torch.rand(batch, 4, 24)
    p = torch.rand(batch, 16, 24)
    return ((tr, s, p),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RfamGen-CMVAE", "build_rfamgen_cmvae", "example_input_rfamgen_cmvae", 2024, "vendored"),
]
