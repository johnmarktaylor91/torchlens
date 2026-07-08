# FAITHFUL PORT of https://github.com/kimbring2/dmlab_ctf @ main (original framework: TensorFlow 2 / Keras)
#
# Community reimplementation of DeepMind's "Capture the Flag: the emergence
# of complex cooperative agents" (Jaderberg et al. 2019) For-The-Win (FTW)
# agent -- kimbring2/dmlab_ctf, network.py / CaptureTheFlag_A2C_LSTM.py. The
# repo itself: (a) is TensorFlow/Keras, not PyTorch; (b) requires the
# DeepMind Lab game engine (deepmind_lab, Bazel-built C++ extension) to run
# at all, so it cannot be installed/vendored/run in this env; (c) explicitly
# states in its own README "The scale of the network will be a little
# smaller than the original paper" -- it is already a simplified restaging,
# not a claim of exact DeepMind-paper fidelity. We faithfully transcribe its
# actual architecture (network.py's ActorCritic: CNN+VAE visual encoder ->
# LSTM core -> actor/critic heads, matching the real Keras layer graph
# 1:1) into self-contained torch, preserving every layer and the forward
# dataflow (encode -> concat(mean,logvar) -> reshape -> LSTM -> flatten ->
# concat with inventory features -> actor/critic heads, plus the VAE
# reconstruction branch used for the auxiliary CVAE loss).
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEEncoder(nn.Module):
    """Port of network.py CVAE.encoder (128x128x3 variant)."""

    def __init__(self, latent_dim, in_hw=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, in_hw, in_hw)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            self.flat_dim = dummy.numel()
        self.fc = nn.Linear(self.flat_dim, latent_dim + latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class CVAEDecoder(nn.Module):
    """Port of network.py CVAE.decoder (128x128x3 variant)."""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 16 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(
            16, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(x.shape[0], 16, 16, 16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class CVAE(nn.Module):
    """Port of network.py CVAE: conditional visual encoder used by the agent."""

    def __init__(self, latent_dim, in_hw=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(latent_dim, in_hw=in_hw)
        self.decoder = CVAEDecoder(latent_dim)

    def encode(self, x):
        out = self.encoder(x)
        mean, logvar = torch.split(out, self.latent_dim, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(torch.tensor(2.0 * 3.141592653589793))
    return torch.sum(
        -0.5 * ((sample - mean) ** 2.0 * torch.exp(-logvar) + logvar + log2pi), dim=raxis
    )


class FTWActorCritic(nn.Module):
    """Port of network.py ActorCritic: combined actor-critic network.

    Real Keras forward (network.py `call`):
      mean, logvar = CVAE.encode(screen_obs)
      cvae_output = concat(mean, logvar) -> reshape(16, 32) -> LSTM(256) w/ initial (memory,carry)
      X_input_screen = flatten(lstm_out) -> Dense(num_hidden, relu)
      X_input_inv = Dense(64, relu)(screen_inv)
      x = Dense(num_hidden, relu)(concat(X_input_screen, X_input_inv))
      actor(x), critic(x), final_memory_state, final_carry_state, cvae_loss
    """

    def __init__(
        self, num_actions, num_hidden_units, latent_dim=256, in_hw=128, lstm_hidden=256, inv_dim=3
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_hidden_units = num_hidden_units
        self.latent_dim = latent_dim
        self.lstm_hidden = lstm_hidden

        self.cvae = CVAE(latent_dim, in_hw=in_hw)

        # cvae_output = concat(mean, logvar) has width 2*latent_dim; reshaped to (16, width/16)
        self.lstm_seq_len = 16
        self.lstm_in_dim = (2 * latent_dim) // self.lstm_seq_len
        self.lstm = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=lstm_hidden, batch_first=True)

        self.common_1 = nn.Linear(self.lstm_seq_len * lstm_hidden, num_hidden_units)
        self.common_2 = nn.Linear(inv_dim, 64)
        self.common_3 = nn.Linear(num_hidden_units + 64, num_hidden_units)

        self.actor = nn.Linear(num_hidden_units, num_actions)
        self.critic = nn.Linear(num_hidden_units, 1)

    def forward(self, screen_obs, screen_inv, memory_state, carry_state):
        batch_size = screen_obs.shape[0]

        mean, logvar = self.cvae.encode(screen_obs)
        cvae_output = torch.cat((mean, logvar), dim=1)
        cvae_output_reshaped = cvae_output.reshape(batch_size, self.lstm_seq_len, self.lstm_in_dim)

        h0 = memory_state.unsqueeze(0)
        c0 = carry_state.unsqueeze(0)
        lstm_output, (final_memory_state, final_carry_state) = self.lstm(
            cvae_output_reshaped, (h0, c0)
        )

        x_input_screen = lstm_output.reshape(batch_size, -1)
        x_input_screen = F.relu(self.common_1(x_input_screen))

        x_input_inv = F.relu(self.common_2(screen_inv))

        x_input = torch.cat([x_input_screen, x_input_inv], dim=1)
        x = F.relu(self.common_3(x_input))

        z = self.cvae.reparameterize(mean, logvar)
        x_logit = self.cvae.decode(z)
        cross_ent = F.binary_cross_entropy_with_logits(x_logit, screen_obs, reduction="none")

        logpx_z = -torch.sum(cross_ent, dim=[1, 2, 3])
        logpz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
        logqz_x = log_normal_pdf(z, mean, logvar)
        cvae_loss = logpx_z + logpz - logqz_x

        return (
            self.actor(x),
            self.critic(x),
            final_memory_state.squeeze(0),
            final_carry_state.squeeze(0),
            cvae_loss,
        )


# --- staging harness: build + example input ---------------------------------


def build_ftw_actor_critic():
    # NOTE: the decoder architecture (network.py CVAE.decoder) always upsamples via three fixed
    # stride-2 ConvTranspose2d layers from a 16x16 base, i.e. it always targets 128x128 regardless
    # of latent_dim -- matching the real code's CVAE (encoder/decoder pair is only valid together
    # at in_hw=128). We keep in_hw=128 for architectural fidelity and shrink latent_dim/hidden
    # sizes instead to keep the traced example lightweight.
    return FTWActorCritic(
        num_actions=7, num_hidden_units=32, latent_dim=16, in_hw=128, lstm_hidden=8, inv_dim=3
    ).eval()


def example_input_ftw_actor_critic():
    batch = 1
    in_hw = 128
    screen_obs = torch.rand(batch, 3, in_hw, in_hw)
    screen_inv = torch.rand(batch, 3)
    memory_state = torch.zeros(batch, 8)
    carry_state = torch.zeros(batch, 8)
    return (screen_obs, screen_inv, memory_state, carry_state)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "FTW-CTF-ActorCritic",
        build_ftw_actor_critic,
        example_input_ftw_actor_critic,
        2023,
        MENAGERIE_ZOO,
    ),
]
