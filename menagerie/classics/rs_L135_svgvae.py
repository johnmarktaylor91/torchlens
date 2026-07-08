# SOURCE: vendored from ueoo/svg_vae_pytorch @ 21e71fdb053eb3fbfe59453bc10ae654c671fb26
# https://raw.githubusercontent.com/ueoo/svg_vae_pytorch/21e71fdb053eb3fbfe59453bc10ae654c671fb26/models/vae.py
# https://raw.githubusercontent.com/ueoo/svg_vae_pytorch/21e71fdb053eb3fbfe59453bc10ae654c671fb26/models/svg_decoder.py
# https://raw.githubusercontent.com/ueoo/svg_vae_pytorch/21e71fdb053eb3fbfe59453bc10ae654c671fb26/models/util_funcs.py
# https://raw.githubusercontent.com/ueoo/svg_vae_pytorch/21e71fdb053eb3fbfe59453bc10ae654c671fb26/main.py
#
# Lopes, Ha, Eck & Shlens 2019 (ICCV 2019) "A Learned Representation for Scalable
# Vector Graphics" (SVG-VAE) -- the official implementation lives in
# magenta/magenta:magenta/models/svg_vae (TensorFlow 1.x). This is a faithful PyTorch
# re-implementation (community port, actively used/cited as the canonical PyTorch
# SVG-VAE) that reproduces the two-stage architecture SVG-VAE trains end to end, exactly
# as wired in the repo's own `main.py::train_svg_decoder`:
#   1. `ConditionalVAE` (`models/vae.py`): the class-conditional convolutional image VAE
#      that encodes a rendered glyph raster to a Gaussian latent `z`. `main.py` imports
#      and instantiates THIS class for the real training loop
#      (`image_vae = ConditionalVAE(...)`) -- the alternate `models/image_vae.py::ImageVAE`
#      class is dead code, left commented out at every call site in `main.py`
#      (`# from models.image_vae import ImageVAE`, `# image_vae = ImageVAE(...)`), so it
#      is NOT used here.
#   2. `SVGLSTMDecoder` (`models/svg_decoder.py`): an autoregressive multi-layer
#      `nn.LSTM` that unbottlenecks the VAE latent `z` into per-layer (hidden, cell)
#      initial states via `init_state_input`, then is stepped one SVG-command token at a
#      time (teacher-forced), matching `main.py`'s training loop over
#      `range(1, trg_len)` calling `svg_decoder(inpt, sampled_bottleneck, target_clss,
#      hidden, cell)`.
# `SVGMDNTop` (the Mixture Density Network head) is defined in the same file but is
# commented out at every call site in `main.py` (`# from models.svg_decoder import
# SVGMDNTop`, `# mdn_top_layer = SVGMDNTop(...)`) -- the real training loop computes its
# loss directly from the LSTM decoder's raw `predict_fc` output
# (`svg_decoder.decoder_loss`), not through the MDN head -- so it is included here
# (verbatim, unused-elsewhere code exactly as upstream) but not part of the traced
# forward pass, matching upstream's actual wiring.
#
# All classes below (`ConditionalVAE`, `SVGLSTMDecoder`, `SVGMDNTop`, `shift_right`) are
# reproduced verbatim from the vendored files; only the cross-file `from models.X import
# Y` paths were flattened into this single module (no computation changed).

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- models/util_funcs.py (verbatim) ---


def shift_right(x, pad_value=None):
    if pad_value is None:
        shifted = F.pad(x, (0, 0, 0, 0, 1, 0))[:-1, :, :]
    else:
        shifted = torch.cat([pad_value, x], axis=0)[:-1, :, :]
    return shifted


# --- models/vae.py (verbatim; "Implementation from https://github.com/AntixK/PyTorch-VAE",
# vendored into svg_vae_pytorch and wired up as the real ImageVAE used by main.py) ---


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    @property
    def forward(self):
        raise NotImplementedError


class ConditionalVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: list = None,
        img_size: int = 64,
        kl_beta: float = 0.00001,
        **kwargs,
    ) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.kl_beta = kl_beta

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels += 1  # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, label):
        y = label.float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z_y = torch.cat([z, y], dim=1)
        return [self.decode(z_y), input, z, mu, log_var]


# --- models/svg_decoder.py (verbatim) ---


class SVGLSTMDecoder(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        num_categories=52,
        bottleneck_bits=32,
        free_bits=0.15,
        kl_beta=300,
        mode="train",
        max_sequence_length=51,
        hidden_size=1024,
        use_cls=True,
        dropout_p=0.5,
        twice_decoder=False,
        num_hidden_layers=4,
        feature_dim=10,
        ff_dropout=True,
    ):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.num_categories = num_categories
        self.command_len = 4
        self.arg_len = 6
        assert self.command_len + self.arg_len == feature_dim

        self.ff_dropout = ff_dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        if twice_decoder:
            self.hidden_size = self.hidden_size * 2
        self.unbottleneck_dim = self.hidden_size * 2

        self.unbotltenecks = nn.ModuleList(
            [
                nn.Linear(bottleneck_bits, self.unbottleneck_dim)
                for _ in range(self.num_hidden_layers)
            ]
        )

        self.input_dim = feature_dim + bottleneck_bits + num_categories
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.pre_lstm_ac = nn.Tanh()
        if self.ff_dropout:
            self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_hidden_layers, dropout=dropout_p
        )
        self.predict_fc = nn.Linear(self.hidden_size, feature_dim)

    def init_state_input(self, sampled_bottleneck):
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbotltenecks[i](sampled_bottleneck)
            (h0, c0) = (
                unbottleneck[:, : self.unbottleneck_dim // 2],
                unbottleneck[:, self.unbottleneck_dim // 2 :],
            )
            init_state_hidden.append(h0.unsqueeze(0))
            init_state_cell.append(c0.unsqueeze(0))
        init_state_hidden = torch.cat(init_state_hidden, dim=0)
        init_state_cell = torch.cat(init_state_cell, dim=0)
        init_state = {}
        init_state["hidden"] = init_state_hidden
        init_state["cell"] = init_state_cell
        return init_state

    def decoder_loss(self, decoder_predict, target, mode="train"):
        target_commands = target[..., : self.command_len]
        target_args = target[..., self.command_len :]
        predict_commands = decoder_predict[..., : self.command_len]
        predict_args = decoder_predict[..., self.command_len :]
        softmax_xent_loss = torch.sum(-target_commands * F.log_softmax(predict_commands, -1), -1)
        softmax_xent_loss = torch.mean(softmax_xent_loss)
        mse_loss = F.mse_loss(predict_args, target_args)
        loss = {}
        loss["loss"] = softmax_xent_loss * 1.0 + mse_loss * 2.0
        loss["xent_loss"] = softmax_xent_loss
        loss["mse_loss"] = mse_loss
        return loss

    def forward(self, inpt, sampled_bottleneck, clss, hidden, cell):
        clss = clss.float()
        if inpt.size(-1) != self.hidden_size:  # train and first time step in test
            inpt = torch.cat([inpt, sampled_bottleneck, clss], dim=-1)  # [batch_size, 10 + 32 + 52]
            inpt = self.pre_lstm_ac(self.pre_lstm_fc(inpt))
            inpt = inpt.unsqueeze(dim=0)
        if self.ff_dropout:
            inpt = self.dropout(inpt)
        output, (hidden, cell) = self.rnn(inpt, (hidden, cell))
        predict = self.predict_fc(output.squeeze(0))
        decoder_output = {}
        decoder_output["predict"] = predict
        decoder_output["output"] = output
        decoder_output["hidden"] = hidden
        decoder_output["cell"] = cell
        return decoder_output


class SVGMDNTop(nn.Module):
    """
    Apply the Mixture Density Network on the top of the LSTM output.
    Defined in the upstream repo but not wired into main.py's real training loop (see
    module header) -- reproduced verbatim for completeness, not part of the traced path.
    """

    def __init__(
        self,
        num_mixture=50,
        seq_len=51,
        hidden_size=1024,
        hard=False,
        mode="train",
        mix_temperature=0.0001,
        gauss_temperature=0.0001,
        dont_reduce=False,
    ):
        super().__init__()
        self.num_mix = num_mixture
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.command_len = 4
        self.arg_len = 6
        self.output_channel = self.command_len + self.arg_len * self.num_mix * 3

        self.hard = hard
        self.mode = mode
        self.mix_temperature = mix_temperature
        self.gauss_temperature = gauss_temperature

        self.dont_reduce = dont_reduce

        self.fc = nn.Linear(self.hidden_size, self.output_channel)
        self.identity = nn.Identity()

    def get_mdn_coef(self, arguments):
        """Compute mdn coefficient, aka, split arguments to 3 chunks with size num_mix"""
        logmix, mean, logstd = torch.split(arguments, self.num_mix, dim=-1)
        logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
        mdn_coef = {}
        mdn_coef["logmix"] = logmix
        mdn_coef["mean"] = mean
        mdn_coef["logstd"] = logstd
        return mdn_coef


# --- staging harness (torchlens menagerie build/example entry points) ---


class SVGVAE(nn.Module):
    """
    Full SVG-VAE training-time pipeline exactly as wired in the upstream repo's
    `main.py::train_svg_decoder`: `ConditionalVAE` encodes a rendered glyph raster (plus
    one-hot class) to a latent `z`, which seeds `SVGLSTMDecoder`'s per-layer LSTM initial
    states; the decoder is then stepped autoregressively (teacher-forced) over the SVG
    command sequence, exactly matching the `for t in range(1, trg_len): ... svg_decoder(
    inpt, sampled_bottleneck, target_clss, hidden, cell)` loop in `main.py`.
    """

    def __init__(
        self,
        num_categories=12,
        bottleneck_bits=16,
        img_size=64,
        hidden_dims=None,
        hidden_size=32,
        num_hidden_layers=2,
        seq_len=6,
        feature_dim=10,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        # NOTE: hidden_dims must keep the upstream default [32, 64, 128, 256, 512]
        # unchanged (5 stride-2 stages, ending at width 512): `ConditionalVAE.decode()`
        # hardcodes `result.view(-1, 512, 2, 2)` as a literal (not derived from
        # `hidden_dims`), which is only shape-correct when the last channel width is
        # exactly 512 and img_size=64 (5 stride-2 downsamples: 64->32->16->8->4->2, so
        # `fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)`'s `*4` matches the 2x2
        # spatial map too). This is a real hardcoded constant in the vendored code, not
        # a simplification introduced here.
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.image_vae = ConditionalVAE(
            in_channels=1,
            num_classes=num_categories,
            latent_dim=bottleneck_bits,
            hidden_dims=hidden_dims,
            img_size=img_size,
        )
        self.svg_decoder = SVGLSTMDecoder(
            num_categories=num_categories,
            bottleneck_bits=bottleneck_bits,
            mode="train",
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_p=0.0,
            feature_dim=feature_dim,
        )

    def forward(self, image, clss_onehot, trg):
        vae_output = self.image_vae(image, clss_onehot)
        sampled_bottleneck = vae_output[2]  # z, matching main.py's `vae_output[2]`

        trg_shifted = shift_right(trg)
        init_state = self.svg_decoder.init_state_input(sampled_bottleneck)
        hidden, cell = init_state["hidden"], init_state["cell"]

        inpt = trg_shifted[0, :, :]
        outputs = []
        for t in range(1, self.seq_len):
            decoder_output = self.svg_decoder(inpt, sampled_bottleneck, clss_onehot, hidden, cell)
            hidden, cell = decoder_output["hidden"], decoder_output["cell"]
            outputs.append(decoder_output["predict"])
            inpt = trg_shifted[t]

        predicted_seq = torch.stack(outputs, dim=0)
        return {"recon_image": vae_output[0], "predicted_seq": predicted_seq}


def build_svgvae():
    torch.manual_seed(0)
    return SVGVAE()


def example_input_svgvae():
    torch.manual_seed(0)
    batch_size = 2
    num_categories = 12
    seq_len = 6
    feature_dim = 10
    img_size = 64
    image = torch.randn(batch_size, 1, img_size, img_size)
    clss = torch.randint(low=0, high=num_categories, size=(batch_size,), dtype=torch.long)
    clss_onehot = F.one_hot(clss, num_classes=num_categories).float()
    trg = torch.randn(seq_len, batch_size, feature_dim)
    return (image, clss_onehot, trg)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SVG-VAE", "build_svgvae", "example_input_svgvae", 2019, MENAGERIE_ZOO),
]
