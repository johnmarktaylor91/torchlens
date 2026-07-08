# FAITHFUL REIMPLEMENTATION from arXiv:2012.03861 (no public code) -- A/B sonnet
"""Hierarchical Supervised LSTM Autoencoder for fault detection and diagnosis
on the Tennessee-Eastman Process (TEP), arXiv:2012.03861.

Distinctive mechanisms implemented faithfully:
1. Each hierarchical level is a "Supervised DRAE-NN" -- a supervised deep
   recurrent (LSTM) AUTOENCODER: an LSTM encoder compresses the sensor
   window into a latent code z (Sec. "Architecture"/Eq. 2-5), an LSTM
   decoder reconstructs the input window from z (Eq. 6), and a dense
   softmax classification head hangs off the SAME latent z (Eq. 7) --
   i.e. reconstruction and classification are jointly supervised from one
   shared bottleneck, not two independent networks.
2. TWO hierarchical levels with paper-specified, DIFFERENT layer widths and
   depths (Sec. 5): Level 1 is a single-layer 182-unit encoder / 116-unit
   decoder / 18-way softmax that isolates "non-incipient" faults from a
   {normal + incipient-fault} group; Level 2 is a STACKED 2-layer encoder
   (284 units -> 100 units) / 278-unit decoder / 4-way softmax that further
   disambiguates {normal, fault 3, fault 9, fault 15} -- the three
   "incipient" faults whose response resembles normal operation (Sec. 2,
   Table 1; Sec. 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_NUM_SENSORS = 52  # Table 2: 52 measured process variables.
_SEQ_LEN = 150  # Sec. 5 / Fig. 10: 150-timestep window found optimal.

_L1_ENCODER_UNITS = (182,)  # Sec. 5: "182 encoder LSTM units".
_L1_DECODER_UNITS = 116  # Sec. 5: "116 LSTM units for processing of the output".
_L1_NUM_CLASSES = 18  # Sec. 5: level-1 softmax has 18 units.

_L2_ENCODER_UNITS = (284, 100)  # Sec. 5: "284...first hidden layer, second layer...100".
_L2_DECODER_UNITS = 278  # Sec. 5: "followed by 278 LSTM units".
_L2_NUM_CLASSES = 4  # Sec. 5: normal + 3 incipient faults (3, 9, 15).


class LSTMSupervisedAutoencoder(nn.Module):
    """One hierarchical level's "Supervised DRAE-NN" (LSTM-SAE).

    Encoder: a (possibly stacked) LSTM whose final hidden state is the
    latent code z (Eq. 2-5). Decoder: an LSTM that reconstructs the input
    window from z, unrolled for `seq_len` steps (Eq. 6). Classifier: a
    dense + softmax head applied directly to z (Eq. 7) -- the "supervised"
    half of the autoencoder.
    """

    def __init__(
        self,
        num_sensors: int,
        encoder_units: tuple[int, ...],
        decoder_units: int,
        num_classes: int,
        seq_len: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = encoder_units[-1]

        encoder_layers = []
        in_dim = num_sensors
        for hidden in encoder_units:
            encoder_layers.append(nn.LSTM(in_dim, hidden, batch_first=True))
            in_dim = hidden
        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Decoder LSTM: latent z seeds the initial hidden/cell state; the
        # per-step input is the previous reconstructed sample (teacher-free
        # autoregressive decoding), a standard seq2seq-autoencoder choice
        # since the paper does not specify the decoder's per-step input.
        self.decoder_cell = nn.LSTMCell(num_sensors, decoder_units)
        self.decoder_h0 = nn.Linear(self.latent_dim, decoder_units)
        self.decoder_c0 = nn.Linear(self.latent_dim, decoder_units)
        self.decoder_out = nn.Linear(decoder_units, num_sensors)

        # Classification head off the shared bottleneck z (Eq. 7).
        self.classifier = nn.Linear(self.latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, num_sensors)
        h = x
        final_hidden = None
        for lstm in self.encoder_layers:
            h, (final_hidden, _) = lstm(h)
        z = final_hidden.squeeze(0)  # (batch, latent_dim), Eq. 5.

        # Autoregressive LSTM decoder reconstructing the input window (Eq. 6).
        batch = x.shape[0]
        dec_h = self.decoder_h0(z)
        dec_c = self.decoder_c0(z)
        dec_in = torch.zeros(batch, x.shape[-1], device=x.device, dtype=x.dtype)
        recon_steps = []
        for _ in range(self.seq_len):
            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            dec_in = self.decoder_out(dec_h)
            recon_steps.append(dec_in)
        reconstruction = torch.stack(recon_steps, dim=1)  # (batch, seq_len, num_sensors)

        class_logits = self.classifier(z)  # Eq. 7 (pre-softmax logits).
        return reconstruction, class_logits


class HierarchicalDRNNFDD(nn.Module):
    """Two-level hierarchical fault detection/diagnosis network (Sec. 4).

    Level 1: coarse detector separating "non-incipient" faults from the
    {normal + incipient} group (18-way softmax). Level 2: fine diagnoser
    focused on the 3 hardest-to-observe incipient faults 3, 9, 15 plus
    normal (4-way softmax). At deployment the paper applies Level 2 only to
    samples Level 1 routes to the {normal + incipient} group (Sec. 4); this
    module runs BOTH levels' full forward passes on every input (a batched,
    always-differentiable architecture snapshot) and returns both levels'
    (reconstruction, logits) outputs rather than performing paper's
    data-dependent hard routing -- see ASSUMPTIONS in the spec card.
    """

    def __init__(
        self,
        num_sensors: int = _NUM_SENSORS,
        seq_len: int = _SEQ_LEN,
    ):
        super().__init__()
        self.level1 = LSTMSupervisedAutoencoder(
            num_sensors=num_sensors,
            encoder_units=_L1_ENCODER_UNITS,
            decoder_units=_L1_DECODER_UNITS,
            num_classes=_L1_NUM_CLASSES,
            seq_len=seq_len,
        )
        self.level2 = LSTMSupervisedAutoencoder(
            num_sensors=num_sensors,
            encoder_units=_L2_ENCODER_UNITS,
            decoder_units=_L2_DECODER_UNITS,
            num_classes=_L2_NUM_CLASSES,
            seq_len=seq_len,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, num_sensors)
        recon1, logits1 = self.level1(x)
        recon2, logits2 = self.level2(x)
        return recon1, logits1, recon2, logits2


def build_hier_drnn_fdd() -> HierarchicalDRNNFDD:
    return HierarchicalDRNNFDD(num_sensors=_NUM_SENSORS, seq_len=_SEQ_LEN)


def example_input_hier_drnn_fdd() -> torch.Tensor:
    return torch.randn(2, _SEQ_LEN, _NUM_SENSORS)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("hierarchical_drnn_fdd", "build_hier_drnn_fdd", "example_input_hier_drnn_fdd", 2020, "REIMPL"),
]
