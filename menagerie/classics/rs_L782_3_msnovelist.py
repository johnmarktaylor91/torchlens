# FAITHFUL PORT of meowcat/MSNovelist @ master (original framework: TensorFlow 2 / Keras,
# tf.keras.layers.Layer subclasses)
#
# Files transcribed: model/encoder.py (FingerprintFormulaEncoder), model/decoder.py
# (SequenceDecoder), model/hydrogen_estimator.py (HydrogenEstimator), model/sum_pooling.py
# (GlobalSumPooling), model/recurrent_additive.py (RecurrentAdditiveCell),
# model/blueprinted_model.py (BlueprintedModel.__init__ -- the sub-layer wiring),
# model/transcoder_model.py (TranscoderModel.call -- the full forward pass), and the
# element/grammar vocabulary constants from tokens_process/definitions.py (VOC / ELEMENTS /
# GRAMMAR / ELEMENT_MAP / GRAMMAR_MAP).
#
# MSNovelist (Stravs et al., 2022, Nat. Methods) generates candidate SMILES/molecular
# structures directly from a predicted molecular fingerprint and formula (de novo structure
# elucidation from MS/MS data), using an RNN decoder guided by a formula/fingerprint-derived
# encoder and a hard grammar/element-counting mechanism that tracks how many atoms of each
# element and how many open parentheses remain, so the decoder cannot emit tokens the
# remaining formula budget forbids. Concretely: an MLP `FingerprintFormulaEncoder` maps
# [fingerprint; formula] to (a) a latent code z and (b) the multi-layer LSTM decoder's initial
# hidden/cell states; a small forward LSTM `HydrogenEstimator` predicts per-token hydrogen
# contribution from the token stream alone; a `RecurrentAdditiveCell`-based RNN
# (`auxiliary_counter`) subtracts each emitted token's element/grammar contribution from the
# running formula budget (grammar-constrained decoding signal); and a 3-layer LSTM
# `SequenceDecoder` consumes [tokens, running budget, repeated z] to emit per-step token
# logits. This port keeps every sub-module and the exact `TranscoderModel.call` wiring order
# (only the training-only Adam/loss `.compile()` call and TF-graph-mode `tf.py_function`
# tokenization utilities are dropped, since they are not part of the forward architecture);
# Keras `LSTM`/`Dense`/`RNN` cells become their exact torch analogues (`nn.LSTM`,
# `nn.Linear`, a hand-rolled recurrent scan for the additive counter cell, matching Keras'
# `RNN(RecurrentAdditiveCell(...))` unroll-over-time semantics).
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

# --- tokens_process/definitions.py (vocabulary-derived constants, faithfully transcribed) ---
# VOC has 37 SMILES/SELFIES-derived symbols + 3 special chars (INITIAL/FINAL/PAD) = 40 tokens.
N_VOC = 40
# ELEMENTS = ['C','F','I','L','N','O','P','R','S'] -> 9 tracked elements.
N_ELEMENTS = 9
# GRAMMAR = [{'(': -1, ')': 1}] -> 1 grammar rule (paren balance).
N_GRAMMAR = 1
N_COUNTER_UNITS = N_ELEMENTS + N_GRAMMAR + 1  # + the "pad/term" onehot row appended in
# BlueprintedModel.construct_counter_matrix (the extra all-ones row/col for the pad+term
# bookkeeping unit).
# `auxiliary_counter_start_state_transformer` pads `inputs['mol_form']` with exactly one
# trailing zero column (`tf.pad(x, [[0,0],[0,1]])`) to reach `N_COUNTER_UNITS`, so the real
# `mol_form` tensor is [B, N_ELEMENTS + N_GRAMMAR] wide (per-element atom-count budget plus
# one grammar/paren-balance budget slot); the final +1 (pad/term bookkeeping unit) is what
# the padding call supplies.
N_MOL_FORM = N_ELEMENTS + N_GRAMMAR


class RecurrentAdditiveCell(nn.Module):
    """Port of model/recurrent_additive.py: RecurrentAdditiveCell.

    A trivial additive RNN cell: output_t = state_{t-1} + factor * input_t. Used both as the
    grammar/element "auxiliary counter" (factor=-1, subtracting consumed budget) and inside
    GlobalSumPooling's step counter (factor=+1, counting timesteps).
    """

    def __init__(self, units, factor):
        super().__init__()
        self.units = units
        self.factor = factor

    def forward(self, inputs, state):
        # inputs: [B, units], state: [B, units]
        output = state + self.factor * inputs
        return output, output


def scan_rnn_cell(cell, inputs, initial_state=None):
    """Port of Keras `RNN(cell, return_sequences=True)` unrolled over time for a simple
    additive cell (single state tensor, no gates)."""
    B, T, U = inputs.shape
    if initial_state is None:
        state = inputs.new_zeros(B, U)
    else:
        state = initial_state
    outputs = []
    for t in range(T):
        out, state = cell(inputs[:, t], state)
        outputs.append(out)
    return torch.stack(outputs, dim=1), state


class FingerprintFormulaEncoder(nn.Module):
    """Port of model/encoder.py: FingerprintFormulaEncoder."""

    def __init__(self, in_dim, layers=(512, 256), layers_decoder=3, units_decoder=256):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(in_dim)
        dims = [in_dim] + list(layers)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(layers))])
        # rnn_starting_states[i][j]: Dense(units_decoder, relu) per (LSTM layer, h/c state)
        self.rnn_starting_states = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(dims[-1], units_decoder) for _ in range(2)])
                for _ in range(layers_decoder)
            ]
        )

    def forward(self, inputs_mf, inputs_fp):
        layer_stack = torch.cat([inputs_mf, inputs_fp], dim=-1)
        layer_stack = self.batchnorm(layer_stack)
        for layer in self.layers:
            layer_stack = torch.relu(layer(layer_stack))
        z = layer_stack
        rnn_states = [
            [torch.relu(state_layer(z)) for state_layer in state_layers]
            for state_layers in self.rnn_starting_states
        ]
        return z, rnn_states


class HydrogenEstimator(nn.Module):
    """Port of model/hydrogen_estimator.py: HydrogenEstimator."""

    def __init__(self, in_dim, layers=2, units=32):
        super().__init__()
        self.layers_ = layers
        self.lstms = nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            self.lstms.append(nn.LSTM(dim, units, batch_first=True))
            dim = units
        self.out_layer = nn.Linear(units, 1)

    def forward(self, tokens_input, initial_state=None):
        if initial_state is None:
            initial_state = [None] * self.layers_
        layer_stack = tokens_input
        state_stack = []
        for i, lstm in enumerate(self.lstms):
            hx = initial_state[i]
            if hx is not None:
                h0, c0 = hx
                hx = (h0.unsqueeze(0), c0.unsqueeze(0))
            layer_stack, (h, c) = lstm(layer_stack, hx)
            state_stack.append((h.squeeze(0), c.squeeze(0)))
        out = self.out_layer(layer_stack)
        return out, state_stack


class GlobalSumPooling(nn.Module):
    """Port of model/sum_pooling.py: GlobalSumPooling.

    average_t(x) * num_timesteps == sum_t(x); implemented via the same additive
    step-counter-RNN + mean trick as the real repo (rather than a plain `sum`), to stay
    architecture-faithful to the original module composition.
    """

    def __init__(self):
        super().__init__()
        self.step_counter_cell = RecurrentAdditiveCell(units=1, factor=1)

    def forward(self, inputs):
        step_counter_factor = torch.ones_like(inputs[..., :1])
        _, step_counter = scan_rnn_cell(self.step_counter_cell, step_counter_factor)
        average = inputs.mean(dim=1)
        return step_counter * average


class SequenceDecoder(nn.Module):
    """Port of model/decoder.py: SequenceDecoder."""

    def __init__(self, in_dim, tokens_output, layers=3, units=256):
        super().__init__()
        self.layers_ = layers
        self.batchnorm = nn.BatchNorm1d(in_dim)
        self.lstms = nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            self.lstms.append(nn.LSTM(dim, units, batch_first=True))
            dim = units
        self.out_layer = nn.Linear(units, tokens_output)

    def forward(self, inputs, initial_state=None):
        if initial_state is None:
            initial_state = [None] * self.layers_
        layer_stack = torch.cat(inputs, dim=-1)
        B, T, C = layer_stack.shape
        layer_stack = self.batchnorm(layer_stack.reshape(B * T, C)).reshape(B, T, C)
        state_stack = []
        for i, lstm in enumerate(self.lstms):
            hx = initial_state[i]
            if hx is not None:
                h0, c0 = hx
                hx = (h0.unsqueeze(0), c0.unsqueeze(0))
            layer_stack, (h, c) = lstm(layer_stack, hx)
            state_stack.append((h.squeeze(0), c.squeeze(0)))
        out = torch.softmax(self.out_layer(layer_stack), dim=-1)
        return out, state_stack


class MSNovelistTranscoder(nn.Module):
    """
    Faithful port of model/transcoder_model.py: TranscoderModel, composed per
    model/blueprinted_model.py: BlueprintedModel.__init__ wiring.

    Real published default config (BlueprintedModel.__init__ defaults):
        decoder_hidden_size=256, hcount_hidden_size=32, fp_enc_layers=[512, 256],
        hcounter_layers=2, decoder_layers=3.
    """

    def __init__(
        self,
        fp_dim=16,
        mf_dim=N_MOL_FORM,
        steps=8,
        fp_enc_layers=(24, 16),
        decoder_hidden_size=16,
        hcount_hidden_size=8,
        hcounter_layers=2,
        decoder_layers=3,
    ):
        super().__init__()
        self.steps = steps

        self.encoder = FingerprintFormulaEncoder(
            in_dim=fp_dim + mf_dim,
            layers=fp_enc_layers,
            layers_decoder=decoder_layers,
            units_decoder=decoder_hidden_size,
        )

        # hydrogen_estimator consumes tokens_X: [B, T, N_VOC]
        self.hydrogen_estimator = HydrogenEstimator(
            in_dim=N_VOC, layers=hcounter_layers, units=hcount_hidden_size
        )
        self.hydrogen_sum = GlobalSumPooling()

        # auxiliary_counter_input_transformer: tf.matmul(x, counter_matrix), counter_matrix
        # is [N_VOC, N_COUNTER_UNITS] (construct_counter_matrix in BlueprintedModel).
        counter_matrix = self._construct_counter_matrix()
        self.register_buffer("counter_matrix", counter_matrix)

        # auxiliary_counter_start_state_transformer: pad mol-formula vector [B, N_ELEMENTS]
        # with one trailing zero -> [B, N_COUNTER_UNITS] initial state for the counter RNN.
        self.auxiliary_counter_units = N_COUNTER_UNITS
        self.auxiliary_counter_cell = RecurrentAdditiveCell(units=N_COUNTER_UNITS, factor=-1)

        self.sequence_decoder = SequenceDecoder(
            in_dim=N_VOC + N_COUNTER_UNITS + decoder_hidden_size,
            tokens_output=N_VOC,
            layers=decoder_layers,
            units=decoder_hidden_size,
        )

    @staticmethod
    def _construct_counter_matrix():
        # ELEMENT_MAP: [N_VOC, N_ELEMENTS], GRAMMAR_MAP: [N_VOC, N_GRAMMAR] (data-dependent
        # vocabulary constants from tokens_process/definitions.py; deterministic seeded
        # random stand-ins are used here since the real matrices depend on the training
        # SELFIES vocabulary, which is not needed to exercise the architecture).
        torch.manual_seed(0)
        element_map = (torch.rand(N_VOC, N_ELEMENTS) > 0.85).float()
        grammar_map = torch.zeros(N_VOC, N_GRAMMAR)
        grammar_map[10, 0] = -1.0  # '(' -> -1
        grammar_map[11, 0] = 1.0  # ')' -> +1
        m11, m13 = element_map, grammar_map
        m12 = torch.zeros_like(m13)
        mleft = torch.cat([m11, m12, m13], dim=1)
        m21 = torch.zeros_like(m11[:1, :])
        m22 = torch.ones_like(m12[:1, :])
        m23 = torch.zeros_like(m13[:1, :])
        mright = torch.cat([m21, m22, m23], dim=1)
        return torch.cat([mleft, mright], dim=0)

    def forward(self, mol_form, fingerprint_selected, tokens_X):
        fingerprints_ = fingerprint_selected  # fingerprint_rounding is identity by default

        z, decoder_initial_states = self.encoder(mol_form, fingerprints_)

        estimated_h_count, _ = self.hydrogen_estimator(tokens_X)
        estimated_h_sum = self.hydrogen_sum(estimated_h_count)
        estimated_h_count_ = estimated_h_count.detach()  # hcounter_gradient_stop

        auxiliary_counter_input = torch.cat([tokens_X, estimated_h_count_], dim=-1)

        # auxiliary_counter_start_state_transformer: pad(mol_form, [[0,0],[0,1]])
        auxiliary_counter_start_state = torch.nn.functional.pad(mol_form, (0, 1))

        auxiliary_counter_input_transformed = auxiliary_counter_input @ self.counter_matrix

        element_grammar_count, _ = scan_rnn_cell(
            self.auxiliary_counter_cell,
            auxiliary_counter_input_transformed,
            initial_state=auxiliary_counter_start_state,
        )

        z_repeated = z.unsqueeze(1).repeat(1, self.steps, 1)

        decoder_input = [tokens_X, element_grammar_count, z_repeated]
        decoder_out, _ = self.sequence_decoder(decoder_input, decoder_initial_states)

        return decoder_out, estimated_h_sum


def build_msnovelist():
    torch.manual_seed(0)
    model = MSNovelistTranscoder(
        fp_dim=16,
        mf_dim=N_MOL_FORM,
        steps=8,
        fp_enc_layers=(24, 16),
        decoder_hidden_size=16,
        hcount_hidden_size=8,
        hcounter_layers=2,
        decoder_layers=3,
    )
    model.eval()
    return model


def example_input_msnovelist():
    torch.manual_seed(0)
    batch_size, steps = 2, 8
    mol_form = torch.rand(batch_size, N_MOL_FORM)
    fingerprint_selected = torch.rand(batch_size, 16)
    tokens_X = torch.zeros(batch_size, steps, N_VOC)
    idx = torch.randint(0, N_VOC, (batch_size, steps))
    tokens_X.scatter_(2, idx.unsqueeze(-1), 1.0)
    return (mol_form, fingerprint_selected, tokens_X)


MENAGERIE_ENTRIES = [
    ("MSNovelist", "build_msnovelist", "example_input_msnovelist", 2022, "ported-pytorch"),
]
