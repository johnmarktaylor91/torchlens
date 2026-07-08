# SOURCE: vendored from https://github.com/MishaLaskin/curl @ master (8416d6e3869e)
# (encoder.py PixelEncoder + curl_sac.py Critic/CURL classes, minimal changes:
#  merged into one file, imports fixed, added a thin CURLModel forward wrapper
#  that calls the real encode()/compute_logits() methods for a single-tensor-pair
#  trace)
"""CURL (Contrastive Unsupervised Representations for RL, Laskin et al. 2020):
a convolutional PixelEncoder shared by the actor/critic, plus a CURL contrastive
head that computes a bilinear similarity matrix (InfoNCE-style logits) between
an anchor and positive (augmented) observation's encodings. The actor/critic MLP
trunks are standard SAC MLPs (same family already covered elsewhere); CURL's
distinguishing architecture is the encoder + contrastive bilinear-logits head
captured here."""

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class CURL(nn.Module):
    """CURL contrastive head."""

    def __init__(
        self, obs_shape, z_dim, batch_size, encoder, encoder_target, output_type="continuous"
    ):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = encoder
        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class CURLModel(nn.Module):
    """Thin wrapper composing the real CURL encode() + compute_logits() calls
    into a single forward() so TorchLens can trace the full CURL contrastive
    pipeline (anchor encoder, momentum/target encoder, bilinear logits) from
    one (obs_anchor, obs_pos) tensor pair."""

    def __init__(
        self, obs_shape, feature_dim=50, z_dim=50, batch_size=32, num_layers=4, num_filters=32
    ):
        super().__init__()
        encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits=True)
        encoder_target = PixelEncoder(
            obs_shape, feature_dim, num_layers, num_filters, output_logits=True
        )
        encoder_target.load_state_dict(encoder.state_dict())
        self.curl = CURL(
            obs_shape, z_dim, batch_size, encoder, encoder_target, output_type="continuous"
        )

    def forward(self, obs_anchor, obs_pos):
        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos, ema=True)
        logits = self.curl.compute_logits(z_a, z_pos)
        return logits


def build_curl():
    return CURLModel(
        obs_shape=(3, 64, 64), feature_dim=50, z_dim=50, batch_size=8, num_layers=4, num_filters=16
    )


def example_input_curl():
    return (torch.rand(8, 3, 64, 64) * 255.0, torch.rand(8, 3, 64, 64) * 255.0)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "CURL (Contrastive Unsupervised Representations for RL)",
        build_curl,
        example_input_curl,
        2020,
        MENAGERIE_ZOO,
    ),
]
