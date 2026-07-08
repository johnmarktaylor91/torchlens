# FAITHFUL REIMPLEMENTATION from Visin, Kastner, Cho, Matteucci, Courville & Bengio,
# "ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks"
# (arXiv:1505.00393, 2015) -- Section 3.1's patch-sweep bidirectional-RNN mechanism,
# transcribed with Gated Recurrent Units substituted for the paper's original
# tanh-RNN cells, applied to a 2D time-frequency spectrogram instead of an image.
#
# Queue candidate "2D-Gated Recurrent Unit for Spectrograms" notes: "no single
# originating canonical paper identified ... straightforward 2D GRU cell
# extension". Repo/paper search (gh code search, web search) turned up no
# dedicated repo or single paper defining a named "2D-GRU spectrogram"
# architecture; ReNet is the well-documented, widely cited generic mechanism for
# "an RNN that sweeps both axes of a 2D grid" that this candidate's oneline
# describes ("GRU operating on both time and frequency axes of a 2D spectrogram
# simultaneously") -- the same two-pass composition used by every
# frequency-then-time (or time-then-frequency) recurrent spectrogram model in the
# speech-enhancement literature (e.g. F-LSTM/T-LSTM hybrids). No usable code
# exists to vendor/port for the spectrogram-specific variant, so this transcribes
# ReNet's exact two-stage sweep (vertical sweep over patch columns, producing a
# composite feature map, followed by horizontal sweep over that composite's
# patch rows) with GRU replacing the original RNN cell, per rung 4.
#
# ReNet mechanism (arXiv:1505.00393 Sec. 3.1): the input is split into
# non-overlapping patches; a bidirectional RNN sweeps over the patch column axis
# (direction 1) shared across all columns, producing a composite feature map;
# a second bidirectional RNN then sweeps over the patch row axis (direction 2)
# of that composite map, shared across all rows. This module applies that
# composition with the frequency axis as ReNet's "vertical" axis and the time
# axis as ReNet's "horizontal" axis, and stacks the resulting ReNet-GRU layers
# with global average pooling and a linear classifier head, matching ReNet's own
# classification-network template (Sec. 4).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class ReNetGRULayer(nn.Module):
    """One ReNet sweep layer (Visin et al. 2015) with GRU cells, replacing one
    conv+pool layer. Two-stage bidirectional sweep: first vertically along the
    frequency-patch axis (weights shared across time-patch columns), then
    horizontally along the time-patch axis of the vertical sweep's composite
    output (weights shared across frequency-patch rows)."""

    def __init__(self, in_channels: int, patch_f: int, patch_t: int, hidden_size: int):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.hidden_size = hidden_size
        patch_dim = in_channels * patch_f * patch_t
        # Vertical sweep: bidirectional GRU over the frequency-patch axis.
        self.gru_freq = nn.GRU(patch_dim, hidden_size, batch_first=True, bidirectional=True)
        # Horizontal sweep: bidirectional GRU over the time-patch axis of the
        # vertical sweep's composite feature.
        self.gru_time = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, Freq, Time)
        n, c, f, t = x.shape
        pf, pt = self.patch_f, self.patch_t
        f_p, t_p = f // pf, t // pt

        # Patchify into non-overlapping (pf, pt) patches, flattened to vectors,
        # per ReNet Sec. 3.1's patch-extraction step.
        x = x[:, :, : f_p * pf, : t_p * pt]
        x = x.reshape(n, c, f_p, pf, t_p, pt)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(n, f_p, t_p, c * pf * pt)

        # Vertical sweep over the frequency-patch axis, shared across all
        # time-patch columns (ReNet: "the same network ... is applied to all
        # the patches of a column").
        x_cols = x.permute(0, 2, 1, 3).reshape(n * t_p, f_p, -1)
        v_out, _ = self.gru_freq(x_cols)  # (N*t_p, f_p, 2*hidden)
        v_out = v_out.reshape(n, t_p, f_p, 2 * self.hidden_size).permute(0, 2, 1, 3)
        # v_out: (N, f_p, t_p, 2*hidden)

        # Horizontal sweep over the time-patch axis of the composite feature
        # map, shared across all frequency-patch rows.
        h_rows = v_out.reshape(n * f_p, t_p, 2 * self.hidden_size)
        h_out, _ = self.gru_time(h_rows)  # (N*f_p, t_p, 2*hidden)
        h_out = h_out.reshape(n, f_p, t_p, 2 * self.hidden_size)

        return h_out.permute(0, 3, 1, 2)  # (N, 2*hidden, f_p, t_p)


class TwoDGRUSpectrogramNet(nn.Module):
    """Stack of ReNet-style 2D-GRU sweep layers over a spectrogram, followed by
    global average pooling and a linear classifier head -- ReNet's own
    classification-network template (Visin et al. 2015, Sec. 4), with GRU cells
    substituted for the original tanh-RNN per the "2D-Gated Recurrent Unit for
    Spectrograms" candidate."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 8,
        n_classes: int = 10,
        n_layers: int = 2,
        patch_f: int = 2,
        patch_t: int = 2,
    ):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(n_layers):
            layers.append(ReNetGRULayer(c, patch_f, patch_t, hidden_size))
            c = 2 * hidden_size
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_2dgru_spectrogram():
    return TwoDGRUSpectrogramNet(
        in_channels=1, hidden_size=8, n_classes=10, n_layers=2, patch_f=2, patch_t=2
    )


def example_input_2dgru_spectrogram():
    # (N, C, Freq, Time) spectrogram tile, tiny for fast tracing.
    return (torch.rand(1, 1, 16, 16),)


MENAGERIE_ENTRIES = [
    (
        "2D-Gated Recurrent Unit for Spectrograms",
        "build_2dgru_spectrogram",
        "example_input_2dgru_spectrogram",
        2015,
        "reimpl-pytorch",
    ),
]
