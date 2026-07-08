# FAITHFUL PORT of https://github.com/kimbring2/dmlab_ctf @ main (original framework: TensorFlow/Keras)
# Ported from CaptureTheFlag_A2C_LSTM.py: OurModel (screen-CVAE encoder + LSTM
# + actor/critic heads for a DeepMind-Lab Capture-the-Flag agent). Every layer
# and control-flow branch in the original `OurModel.call` is reproduced 1:1;
# only the TF/Keras layer calls are swapped for their torch equivalents
# (Conv2D->Conv2d with matching 'same'/'valid' padding, LSTM(return_state)
# ->LSTMCell stepped over the (fixed) 16-step reshaped CVAE-latent sequence,
# Dense->Linear). The auxiliary CVAE reconstruction loss (cvae_loss) computed
# inside forward() in the original is preserved and returned unchanged since
# it participates in the real model's own training objective.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 7
SCREEN_SIZE = (3, 64, 64)  # (C, H, W); original screen_size = (64, 64, 3)
INV_SIZE = 3
LATENT_DIM = 256
LSTM_HIDDEN = 128


def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = math.log(2.0 * math.pi)
    return torch.sum(
        -0.5 * ((sample - mean) ** 2.0 * torch.exp(-logvar) + logvar + log2pi), dim=raxis
    )


class CVAE(nn.Module):
    """Port of the original `CVAE` (screen encoder/decoder)."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # encoder: InputLayer(64,64,3) -> Conv2D(16,3,s2,relu) -> Conv2D(32,3,s2,relu) -> Flatten -> Dense(2*latent)
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # 64 -> 31 -> 15 spatial (valid padding, matches Keras default 'valid')
        self._enc_spatial = 15
        self.enc_fc = nn.Linear(32 * self._enc_spatial * self._enc_spatial, latent_dim + latent_dim)

        # decoder: Dense(16*16*16, relu) -> Reshape(16,16,16) -> ConvT(32,3,s2,'same',relu)
        # -> ConvT(16,3,s2,'same',relu) -> ConvT(3,3,s1,'same')
        self.dec_fc = nn.Linear(latent_dim, 16 * 16 * 16)
        self.dec_convt1 = nn.ConvTranspose2d(
            16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_convt2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_convt3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.reshape(h.shape[0], -1)
        out = self.enc_fc(h)
        mean, logvar = torch.split(out, self.latent_dim, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        h = F.relu(self.dec_fc(z))
        h = h.reshape(h.shape[0], 16, 16, 16)
        h = F.relu(self.dec_convt1(h))
        h = F.relu(self.dec_convt2(h))
        logits = self.dec_convt3(h)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


class OurModel(nn.Module):
    """Port of the original `OurModel` actor-critic (CVAE encoder + LSTM +
    actor/critic heads), preserving the exact forward computation of
    `OurModel.call` including the auxiliary CVAE ELBO term."""

    def __init__(self, action_space):
        super().__init__()
        self.common_1 = nn.Linear(16 * LSTM_HIDDEN, 512)
        self.common_2 = nn.Linear(INV_SIZE, 64)
        self.common_3 = nn.Linear(512 + 64, 512)

        self.cvae = CVAE(LATENT_DIM)
        # Original: LSTM(128, return_sequences=True, return_state=True) fed a
        # (batch, 16, 32) sequence (cvae_output reshaped from (batch, 512)).
        self.lstm = nn.LSTM(input_size=32, hidden_size=LSTM_HIDDEN, batch_first=True)

        self.actor = nn.Linear(512, action_space)
        self.critic = nn.Linear(512, 1)

    def forward(self, input_screen, input_inv, memory_state, carry_state):
        batch_size = input_screen.shape[0]

        mean, logvar = self.cvae.encode(input_screen)
        cvae_output = torch.cat((mean, logvar), dim=1)  # (batch, 512)
        cvae_output_reshaped = cvae_output.reshape(batch_size, 16, 32)

        initial_state = (memory_state.unsqueeze(0), carry_state.unsqueeze(0))
        lstm_output, (final_memory_state, final_carry_state) = self.lstm(
            cvae_output_reshaped, initial_state
        )

        x_input_screen = lstm_output.reshape(batch_size, -1)
        x_input_screen = F.relu(self.common_1(x_input_screen))

        x_input_inv = F.relu(self.common_2(input_inv))

        x_input = torch.cat([x_input_screen, x_input_inv], dim=1)
        x = F.relu(self.common_3(x_input))

        z = self.cvae.reparameterize(mean, logvar)
        x_logit = self.cvae.decode(z)
        cross_ent = F.binary_cross_entropy_with_logits(x_logit, input_screen, reduction="none")

        logpx_z = -torch.sum(cross_ent, dim=[1, 2, 3])
        logpz = _log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
        logqz_x = _log_normal_pdf(z, mean, logvar)
        cvae_loss = logpx_z + logpz - logqz_x

        action_logit = self.actor(x)
        value = self.critic(x)

        return (
            action_logit,
            value,
            final_memory_state.squeeze(0),
            final_carry_state.squeeze(0),
            cvae_loss,
        )


def build_ctf_a2c_lstm():
    return OurModel(action_space=NUM_ACTIONS)


def example_input_ctf_a2c_lstm():
    batch = 2
    input_screen = torch.rand(batch, *SCREEN_SIZE)
    input_inv = torch.rand(batch, INV_SIZE)
    memory_state = torch.zeros(batch, LSTM_HIDDEN)
    carry_state = torch.zeros(batch, LSTM_HIDDEN)
    return (input_screen, input_inv, memory_state, carry_state)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "CTF A2C-LSTM Agent (DeepMind Lab)",
        build_ctf_a2c_lstm,
        example_input_ctf_a2c_lstm,
        2021,
        MENAGERIE_ZOO,
    ),
]
