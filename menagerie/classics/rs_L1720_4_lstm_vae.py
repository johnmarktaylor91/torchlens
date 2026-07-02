# FAITHFUL PORT of TimyadNyda/Variational-Lstm-Autoencoder @ master (original
# framework: TensorFlow 1.x, static-graph placeholders/tf.Session)
#
# Ported from: LstmVAE/model.py, class LSTM_Var_Autoencoder
#
# LSTM-VAE anomaly detector: a variational autoencoder whose encoder/decoder are LSTM
# cells over a time-series window, used for time-series anomaly detection via
# reconstruction + KL-regularized latent space (the "standard architecture" this queue
# entry is filed under; multiple community ports of this same TF1 design exist, this
# port transcribes the original TimyadNyda implementation).
#
# Original TF1 static graph (transcribed layer-for-layer):
#   encoder: single tf.nn.rnn_cell.LSTMCell(intermediate_dim), tf.nn.dynamic_rnn
#            -> take the LAST timestep's hidden output
#   z_mean  = Dense(outputs[:, -1, :])       # weights['z_mean'], biases['z_mean_b']
#   z_sigma = softplus(Dense(outputs[:, -1, :]))  # weights['log_sigma'], biases['z_std_b']
#   z       = z_mean + exp(0.5 * z_sigma) * eps               # gauss_sampling
#   repeated_z = RepeatVector(timesteps)(z)                    # broadcast z across time
#   decoder: MultiRNNCell([LSTMCell(intermediate_dim), LSTMCell(n_dim)]) over repeated_z
#            -> x_reconstr_mean (per-timestep reconstruction, dim = n_dim)
#
# The `stateful` cross-batch-state-carrying variant (tf.nn.rnn_cell state Variables +
# manual reset/update ops) is training/serving infrastructure, not part of the
# differentiable architecture, so this port implements the (default) stateless path
# exactly as `LSTM_Var_Autoencoder.__init__(stateful=False)` builds it -- every layer
# and the "outputs[:, -1, :] used for BOTH z_mean and z_sigma" quirk from the original
# code is preserved faithfully (upstream computes both projections from the same
# encoder's last hidden state, not a shared pre-projection layer).

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class LSTMVarAutoencoder(nn.Module):
    """Faithful torch port of LstmVAE.model.LSTM_Var_Autoencoder (stateful=False path).

    encoder: single-layer LSTM -> last hidden state
    latent: z_mean, z_sigma (softplus) linear heads off the same encoder output ->
            reparameterized gaussian sample z
    decoder: z repeated across the input's timestep count, run through a 2-layer
             stacked LSTM (intermediate_dim -> n_dim) to reconstruct the sequence.
    """

    def __init__(self, n_dim, intermediate_dim, z_dim):
        super().__init__()
        self.n_dim = n_dim
        self.intermediate_dim = intermediate_dim
        self.z_dim = z_dim

        # encoder: tf.nn.rnn_cell.LSTMCell(intermediate_dim) run via dynamic_rnn
        self.encoder_lstm = nn.LSTM(
            input_size=n_dim, hidden_size=intermediate_dim, batch_first=True
        )

        # weights['z_mean'] / biases['z_mean_b'], weights['log_sigma'] / biases['b_log_sigma']
        self.z_mean_layer = nn.Linear(intermediate_dim, z_dim)
        self.z_sigma_layer = nn.Linear(intermediate_dim, z_dim)

        # decoder: MultiRNNCell([LSTMCell(intermediate_dim), LSTMCell(n_dim)])
        self.decoder_lstm1 = nn.LSTM(
            input_size=z_dim, hidden_size=intermediate_dim, batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=intermediate_dim, hidden_size=n_dim, batch_first=True
        )

    def forward(self, x):
        # x: (batch, timesteps, n_dim)
        timesteps = x.shape[1]

        encoder_outputs, _ = self.encoder_lstm(x.float())
        last_hidden = encoder_outputs[:, -1, :]  # outputs[:, -1, :]

        z_mean = self.z_mean_layer(last_hidden)
        z_sigma = torch.nn.functional.softplus(self.z_sigma_layer(last_hidden))

        # gauss_sampling: z = mean + exp(0.5 * sigma) * eps
        eps = torch.randn_like(z_sigma)
        z = z_mean + torch.exp(0.5 * z_sigma) * eps

        # RepeatVector(timesteps)(z): broadcast the latent code across every timestep
        repeated_z = z.unsqueeze(1).repeat(1, timesteps, 1)

        decoded, _ = self.decoder_lstm1(repeated_z)
        x_reconstr_mean, _ = self.decoder_lstm2(decoded)

        return x_reconstr_mean


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_N_DIM = 4
_INTERMEDIATE_DIM = 12
_Z_DIM = 6
_TIMESTEPS = 15
_BATCH = 2


def build_lstm_vae():
    return LSTMVarAutoencoder(n_dim=_N_DIM, intermediate_dim=_INTERMEDIATE_DIM, z_dim=_Z_DIM)


def example_input_lstm_vae():
    return torch.randn(_BATCH, _TIMESTEPS, _N_DIM)


MENAGERIE_ENTRIES = [
    (
        "LSTM-VAE anomaly detector",
        build_lstm_vae,
        example_input_lstm_vae,
        2019,
        "PORT",
    ),
]
