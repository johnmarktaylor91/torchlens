# SOURCE: vendored from electronicarts/character-motion-vaes @ main (vae_motion/models.py)
# https://github.com/electronicarts/character-motion-vaes -- "Character Controllers
# Using Motion VAEs" (Ling, Zinno, Cheng, van de Panne, SIGGRAPH 2020). The repo's
# core architectural contribution is a conditional VAE that autoregressively
# generates the next character pose given a latent sample and a history of prior
# poses, using a Mixture-of-Experts (MoE) decoder whose per-layer weights are a
# softmax-gated blend of `num_experts` independent weight tensors (batched via
# `torch.baddbmm`). `Encoder`/`Decoder`/`MixedDecoder`/`PoseMixtureVAE` are
# transcribed verbatim from vae_motion/models.py; only the RL controller classes
# (`PoseVAEController`, `PoseVAEPolicy`, which additionally import
# `common.controller.init`/`DiagGaussian`) were dropped since they are a downstream
# PPO policy, not part of the motion VAE architecture itself.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from vae_motion/models.py ----
class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)


class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for weight, bias, activation in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class PoseMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_motionvae():
    torch.manual_seed(0)
    frame_size = 16
    latent_size = 8
    num_condition_frames = 1
    num_future_predictions = 1
    num_experts = 4
    normalization = {"mode": "none"}
    model = PoseMixtureVAE(
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    )
    model.eval()
    return model


def example_input_motionvae():
    torch.manual_seed(0)
    batch_size = 2
    frame_size = 16
    num_condition_frames = 1
    num_future_predictions = 1
    x = torch.randn(batch_size, frame_size * num_future_predictions)
    c = torch.randn(batch_size, frame_size * num_condition_frames)
    return (x, c)


MENAGERIE_ENTRIES = [
    ("MotionVAE", build_motionvae, example_input_motionvae, 2020, "vendored-pytorch"),
]
