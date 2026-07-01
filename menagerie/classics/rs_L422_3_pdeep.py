# FAITHFUL PORT of pFindStudio/pDeep @ master (original framework: TensorFlow 1.x
# tf.contrib.rnn, via pDeep2/model/lstm_tf.py and pDeep3/pDeep/model_tf.py -- both
# generations are TF1/TF2 `tf.contrib`/Keras-RNN checkpoints; no PyTorch release
# of pDeep exists anywhere in pFindStudio/pDeep, pFindStudio/pDeep3, or on PyPI).
#
# pDeep (Zhou et al. 2017, Analytical Chemistry "pDeep: Predicting MS/MS Spectra
# of Peptides with Deep Learning") predicts b/y fragment-ion intensities for a
# peptide from its sequence, charge, instrument, and collision energy (NCE).
# Ported faithfully from `IonLSTM.BuildModel` in `pDeep2/model/lstm_tf.py`
# (`MultiLayer` branch, the one actually used by `pDeep2/predict.py` /
# `pDeep2/train.py`; the alternate `MultiLayer_with_Attention` branch is dead
# code, commented out at the call site):
#   1. concatenate per-residue AA/mod feature vector with the (broadcast)
#      precursor-charge scalar -> `ConcatToRNN`
#   2. `nlayers` (default 2) stacked bidirectional LSTM layers, each
#      concatenating the fw/bw outputs (`tf.concat(x, axis=2)`) and then
#      re-concatenating the charge (and instrument/NCE side-features on layer
#      0) back onto the stack output for the next layer -- `StackBiLSTM` ->
#      `MultiLayer` -> `ConcatFeatures`
#   3. an "instrument_nce" side-branch: charge+instrument+NCE features run
#      through a per-timestep linear projection (`LinearTrans`, ported as a
#      shared `nn.Linear` applied over the time axis, equivalent to the
#      original's `tf.scan` of a single weight matrix) to a 3-dim vector,
#      concatenated onto every BiLSTM layer's input alongside charge
#   4. output layer: one more bidirectional LSTM (`OutputRNN`) sized to
#      `output_size = len(ion_types) * max_ion_charge` per direction, with the
#      forward and backward direction outputs *summed* (`tf.add(outputs[0],
#      outputs[1])`), not concatenated -- this is the fragment-ion intensity
#      prediction (b/y ions x charges 1-2, per residue position)
#
# torch's `nn.LSTM(bidirectional=True)` returns fw/bw concatenated on the last
# axis (matching TF's `tf.concat(x, axis=2)` for `StackBiLSTM`/`MultiLayer`);
# the final `OutputRNN` fw/bw halves are split and summed explicitly to match
# `tf.add(outputs[0], outputs[1])`. `tf.contrib.rnn.LSTMCell(activation=tanh)`
# is torch's default `nn.LSTM` cell activation (both tanh gates), so no
# custom cell is needed. Dropout/attention/transfer-learning-only branches
# (`rnn_kp`, `output_kp`, `AttentionCellWrapper`, `BuildTransferModel`) are
# training-time-only and are omitted from this pure-inference port, matching
# `pdeep.LoadModel(...).Predict(...)` (inference mode, keep_prob=1).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class InstrumentNCEBranch(nn.Module):
    """Port of `LinearTrans`: concatenates the one-hot instrument feature with
    the scalar NCE, then applies one shared linear projection per timestep
    (the TF version scans a single weight matrix over the time axis with
    `tf.scan`, which is exactly a per-timestep shared linear layer)."""

    def __init__(self, max_instrument_num: int, out_size: int = 3):
        super().__init__()
        self.proj = nn.Linear(max_instrument_num + 1, out_size, bias=False)

    def forward(self, instrument: torch.Tensor, nce: torch.Tensor) -> torch.Tensor:
        ins = torch.cat((instrument, nce), dim=2)
        return self.proj(ins)


class PDeepIonLSTM(nn.Module):
    """Faithful port of pDeep2's `IonLSTM.BuildModel` (`MultiLayer` branch)."""

    def __init__(
        self,
        input_size: int = 82,
        output_size: int = 4,
        layer_size: int = 32,
        nlayers: int = 2,
        max_instrument_num: int = 8,
        enable_instrument_and_nce: bool = True,
    ):
        super().__init__()
        self.layer_size = layer_size
        self.nlayers = nlayers
        self.enable_instrument_and_nce = enable_instrument_and_nce

        if enable_instrument_and_nce:
            self.instrument_branch = InstrumentNCEBranch(max_instrument_num, out_size=3)
            side_size = 1 + 3  # charge + instrument/nce projection
        else:
            self.instrument_branch = None
            side_size = 1  # charge only

        # StackBiLSTM -> MultiLayer: `nlayers` BiLSTM layers, each re-fed the
        # concatenated side features (`ConcatFeatures`) alongside its own
        # bidirectional output before the next layer.
        self.bilstms = nn.ModuleList()
        in_size = input_size + side_size
        for _ in range(nlayers):
            self.bilstms.append(nn.LSTM(in_size, layer_size, batch_first=True, bidirectional=True))
            in_size = 2 * layer_size + side_size

        # OutputRNN: final bidirectional LSTM sized to `output_size`, fw/bw
        # summed (not concatenated) -- `tf.add(outputs[0], outputs[1])`.
        self.output_rnn = nn.LSTM(in_size, output_size, batch_first=True, bidirectional=True)

    def forward(
        self,
        x: torch.Tensor,
        charge: torch.Tensor,
        instrument: torch.Tensor,
        nce: torch.Tensor,
    ) -> torch.Tensor:
        # ConcatToRNN(x, ch): broadcast charge across the time axis.
        # charge is (batch, 1); reshape to (batch, 1, 1) then expand over time.
        ch = charge.view(charge.shape[0], 1, 1).expand(-1, x.shape[1], -1)

        if self.enable_instrument_and_nce:
            ins_feat = self.instrument_branch(instrument, nce)
            side = torch.cat((ch, ins_feat), dim=2)
        else:
            side = ch

        h = torch.cat((x, side), dim=2)
        for bilstm in self.bilstms:
            h, _ = bilstm(h)
            h = torch.cat((h, side), dim=2)

        out, _ = self.output_rnn(h)
        fwd, bwd = out[..., : out.shape[-1] // 2], out[..., out.shape[-1] // 2 :]
        return fwd + bwd


def build_pdeep():
    model = PDeepIonLSTM(
        input_size=82,
        output_size=4,
        layer_size=32,
        nlayers=2,
        max_instrument_num=8,
        enable_instrument_and_nce=True,
    )
    model.eval()
    return model


def example_input_pdeep():
    batch, time_step = 1, 12
    x = torch.randn(batch, time_step, 82)
    charge = torch.randn(batch, 1)
    instrument = torch.zeros(batch, time_step, 8)
    instrument[..., 0] = 1.0
    nce = torch.full((batch, time_step, 1), 0.3)
    return (x, charge, instrument, nce)


MENAGERIE_ENTRIES = [
    ("pDeep", "build_pdeep", "example_input_pdeep", 2017, "ported-pytorch"),
]
