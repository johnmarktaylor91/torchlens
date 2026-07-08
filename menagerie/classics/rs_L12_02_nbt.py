# FAITHFUL PORT of nmrksic/neural-belief-tracker @ master (code/models.py:
# define_CNN_model, model_definition) (original framework: TensorFlow 1.x /
# Python 2, tf.compat.v1 graph-mode with placeholders + tf.Variable)
#
# Neural Belief Tracker (Mrksic et al., ACL 2017, "Neural Belief Tracker:
# Data-Driven Dialogue State Tracking"). The original repo is Python 2 +
# TensorFlow 1.x graph-mode code (`print "..."` statements, `cPickle`,
# `ConfigParser`, `tf.placeholder`/`tf.Variable`/`tf.nn.conv2d` etc.) and
# cannot run in a base-env torch install; this module transcribes
# `model_definition()` faithfully, mechanism-for-mechanism, into a
# self-contained `nn.Module`:
#
#   1) `define_CNN_model`: an n-gram CNN utterance encoder over pretrained
#      word vectors -- 3 parallel conv2d branches (filter widths 1, 2, 3,
#      `num_filters` channels each) + ReLU + full-length max-pool, summed
#      (not concatenated -- the original literally accumulates
#      `hidden_representation += ...` across the three filter widths, which
#      requires equal `num_filters` per branch; ported as `NBTUtteranceCNN`).
#   2) Candidate-value transform: `candidates_transform = sigmoid(W_values @
#      w_candidates + b_candidates)`, one row per ontology value.
#   3) Per-value interaction + joint hidden layer + presoftmax collapse
#      (`w_joint_hidden_layer`/`w_joint_presoftmax`), producing one score per
#      candidate value.
#   4) System-request network (`sysreq_*`): dot-products the slot vector
#      against `system_act_slots`, gates the utterance representation, and
#      runs an independent hidden-layer + presoftmax read-out PER VALUE
#      (the original re-instantiates fresh `sysreq_w_softmax`/
#      `sysreq_b_softmax` weights inside the per-value loop -- transcribed
#      here as one `nn.Linear` per value in `NBTSysreqNet`, not weight-tied).
#   5) System-confirm network (`confirm_*`): dot-product of slot-match AND
#      value-match indicators, gating the utterance representation through a
#      single shared hidden-layer + presoftmax (tied across values, matching
#      the original where `confirm_w1_hidden_layer`/`confirm_w1_softmax` are
#      declared once outside the per-value loop).
#   6) NONE-class padding + `y_presoftmax = y_presoftmax + sysconf + sysreq`
#      (all three padded with a zero column for the NONE label).
#   7) Belief-state update: the non-value-specific branch (`else:` under
#      `learn_belief_state_update`, `value_specific_decoder=False`, the
#      configuration actually used in `nbt.py`'s calls to
#      `model_definition`) blends the current turn against `y_past_state`
#      via `W_memory = a_memory * I + b_memory * (1 - I)` and
#      `W_current = a_current * I + b_current * (1 - I)`
#      (diag/off-diag scalars broadcast over the label-count x label-count
#      matrix), then a final softmax.
#
# Only `use_softmax=True`, `use_delex_features=False`,
# `value_specific_decoder=False`, `learn_belief_state_update=True` is ported
# (the actual configuration `nbt.py` trains with); the dead
# `value_specific_decoder` branch is gated `and False` in the original and is
# not ported. Loss/optimizer (`tf.train.AdamOptimizer`) is training-only and
# not part of the traceable forward architecture.

import torch
import torch.nn as nn


class NBTUtteranceCNN(nn.Module):
    """Ported from define_CNN_model(): 3 parallel conv2d+ReLU+max-pool
    branches (filter widths 1/2/3) over word-vector-dimension-wide input,
    summed elementwise (matches the original's `+=` accumulation, so
    all branches must share `num_filters` output channels)."""

    def __init__(self, num_filters, vector_dimension, longest_utterance_length):
        super().__init__()
        self.num_filters = num_filters
        self.longest_utterance_length = longest_utterance_length
        self.filter_sizes = (1, 2, 3)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, kernel_size=(fs, vector_dimension))
                for fs in self.filter_sizes
            ]
        )

    def forward(self, utterance_representations_full):
        # utterance_representations_full: [batch, longest_utterance_length, vector_dimension]
        x = utterance_representations_full.unsqueeze(1)  # [batch, 1, L, D] (NCHW, channel=1)
        hidden_representation = torch.zeros(
            x.shape[0], self.num_filters, device=x.device, dtype=x.dtype
        )
        for conv in self.convs:
            h = torch.relu(conv(x))  # [batch, num_filters, L - fs + 1, 1]
            pooled = torch.amax(h, dim=2)  # full-length max-pool -> [batch, num_filters, 1]
            hidden_representation = hidden_representation + pooled.squeeze(-1)
        return hidden_representation


class NBTSysreqNet(nn.Module):
    """Ported from the '=== NETWORK FOR SYSTEM REQUESTS ===' block: one
    independent hidden-layer + presoftmax readout PER candidate value (the
    original instantiates fresh sysreq_w_softmax/b_softmax per value_idx
    inside the loop, so these are not weight-tied across values)."""

    def __init__(self, vector_dimension, hidden_units, label_count):
        super().__init__()
        self.hidden = nn.Linear(vector_dimension, hidden_units)
        self.presoftmax = nn.ModuleList([nn.Linear(hidden_units, 1) for _ in range(label_count)])

    def forward(self, decision):
        contributions = []
        for presoftmax in self.presoftmax:
            hidden = torch.sigmoid(self.hidden(decision))
            contributions.append(presoftmax(hidden))
        return torch.cat(contributions, dim=1)


class NBTConfirmNet(nn.Module):
    """Ported from the '=== NETWORK FOR CONFIRMATIONS ===' block: single
    shared hidden-layer + presoftmax readout, tied across all candidate
    values (confirm_w1_hidden_layer/confirm_w1_softmax declared once, outside
    the per-value loop, in the original)."""

    def __init__(self, vector_dimension, hidden_units):
        super().__init__()
        self.hidden = nn.Linear(vector_dimension, hidden_units)
        self.presoftmax = nn.Linear(hidden_units, 1)

    def forward(self, decision):
        hidden = torch.sigmoid(self.hidden(decision))
        return self.presoftmax(hidden)


class NeuralBeliefTracker(nn.Module):
    """Faithful port of model_definition() (use_softmax=True,
    use_delex_features=False, value_specific_decoder=False,
    learn_belief_state_update=True -- the configuration nbt.py trains)."""

    def __init__(
        self,
        vector_dimension,
        label_count,
        hidden_units_1=100,
        longest_utterance_length=40,
        num_filters=300,
    ):
        super().__init__()
        self.vector_dimension = vector_dimension
        self.label_count = label_count
        self.label_size = label_count + 1  # +1 for NONE
        self.hidden_units_1 = hidden_units_1
        self.longest_utterance_length = longest_utterance_length
        self.num_filters = num_filters
        # The original ties hidden_utterance_size = num_filters and later
        # multiplies h_utterance_representation elementwise against
        # candidates_transform (vector_dimension-shaped); this requires
        # num_filters == vector_dimension, matching the repo's actual runs
        # (both 300).
        assert num_filters == vector_dimension

        self.cnn = NBTUtteranceCNN(num_filters, vector_dimension, longest_utterance_length)

        # candidates_transform: sigmoid(W_values @ w_candidates + b_candidates)
        self.w_candidates = nn.Linear(vector_dimension, vector_dimension)

        # joint hidden layer + presoftmax collapse (per-value interaction)
        self.w_joint_hidden_layer = nn.Linear(vector_dimension, hidden_units_1)
        self.w_joint_presoftmax = nn.Linear(hidden_units_1, 1)

        self.sysreq_net = NBTSysreqNet(vector_dimension, hidden_units_1, label_count)
        self.confirm_net = NBTConfirmNet(vector_dimension, hidden_units_1)

        # belief-state update parameters (non-value-specific branch):
        # W_memory = a_memory * I + b_memory * (1 - I); W_current likewise.
        self.a_memory = nn.Parameter(torch.randn(1))
        self.b_memory = nn.Parameter(torch.randn(1))
        self.a_current = nn.Parameter(torch.randn(1))
        self.b_current = nn.Parameter(torch.randn(1))

    def forward(
        self,
        utterance_representations_full,
        system_act_slots,
        system_act_confirm_slots,
        system_act_confirm_values,
        y_past_state,
        slot_vectors,
        value_vectors,
    ):
        # slot_vectors: [label_size, vector_dimension]; value_vectors: [label_count, vector_dimension]
        h_utterance_representation = self.cnn(utterance_representations_full)  # [B, num_filters]

        candidates_transform = torch.sigmoid(self.w_candidates(value_vectors))  # [label_count, D]

        # interaction of utterance with each candidate value
        list_of_value_contributions = [
            h_utterance_representation * candidates_transform[value_idx, :]
            for value_idx in range(self.label_count)
        ]
        # [label_count, B, D] -> [B, label_count, D] -> [B * label_count, D]
        stacked = torch.stack(list_of_value_contributions, dim=0).transpose(0, 1)
        interaction = stacked.reshape(-1, self.vector_dimension)

        hidden_layer_joint = torch.sigmoid(self.w_joint_hidden_layer(interaction))
        hidden_layer_joint = hidden_layer_joint.reshape(-1, self.hidden_units_1)
        y_presoftmax = self.w_joint_presoftmax(hidden_layer_joint).reshape(-1, self.label_count)

        # === system requests ===
        system_act_candidate_interaction = slot_vectors[0, :] * system_act_slots
        dot_product_sysreq = system_act_candidate_interaction.mean(dim=1, keepdim=True)
        decision_sysreq = dot_product_sysreq * h_utterance_representation
        sysreq = self.sysreq_net(decision_sysreq)

        # === system confirmations ===
        slot_match = (slot_vectors[0, :] * system_act_confirm_slots).mean(dim=1)
        value_match = (value_vectors[0, :] * system_act_confirm_values).mean(dim=1)
        dot_product_confirm = slot_match * value_match
        dot_product_confirm = (dot_product_confirm == 1.0).to(dot_product_confirm.dtype)
        decision_confirm = dot_product_confirm.unsqueeze(1) * h_utterance_representation
        confirm_scalar = self.confirm_net(decision_confirm)
        sysconf = confirm_scalar.repeat(1, self.label_count)

        batch_size = y_presoftmax.shape[0]
        zeros_col = torch.zeros(batch_size, 1, dtype=y_presoftmax.dtype, device=y_presoftmax.device)
        y_presoftmax = torch.cat([y_presoftmax, zeros_col], dim=1)
        sysreq = torch.cat([sysreq, zeros_col], dim=1)
        sysconf = torch.cat([sysconf, zeros_col], dim=1)
        y_presoftmax = y_presoftmax + sysconf + sysreq

        # belief-state update (non-value-specific branch)
        eye = torch.eye(self.label_size, dtype=y_presoftmax.dtype, device=y_presoftmax.device)
        w_memory = self.a_memory * eye + self.b_memory * (1 - eye)
        w_current = self.a_current * eye + self.b_current * (1 - eye)
        y_combine = y_past_state @ w_memory + y_presoftmax @ w_current
        y = torch.softmax(y_combine, dim=-1)
        return y, y_combine


MENAGERIE_ZOO = "ported-pytorch"

_VECTOR_DIM = 16
_LABEL_COUNT = 5
_HIDDEN_UNITS = 24
_UTT_LEN = 8
# NOTE: the original ties num_filters == vector_dimension implicitly: the CNN
# output (hidden_utterance_size = num_filters) is later multiplied elementwise
# against candidates_transform, which is vector_dimension-shaped
# (`tf.multiply(h_utterance_representation, candidates_transform[value_idx,
# :])`); the repo's actual configs run both at 300. Keep them equal here.
_NUM_FILTERS = _VECTOR_DIM


def build_nbt():
    model = NeuralBeliefTracker(
        vector_dimension=_VECTOR_DIM,
        label_count=_LABEL_COUNT,
        hidden_units_1=_HIDDEN_UNITS,
        longest_utterance_length=_UTT_LEN,
        num_filters=_NUM_FILTERS,
    )
    model.eval()
    return model


def example_input_nbt():
    batch = 2
    utterance_representations_full = torch.randn(batch, _UTT_LEN, _VECTOR_DIM)
    system_act_slots = torch.randn(batch, _VECTOR_DIM)
    system_act_confirm_slots = torch.randn(batch, _VECTOR_DIM)
    system_act_confirm_values = torch.randn(batch, _VECTOR_DIM)
    y_past_state = torch.softmax(torch.randn(batch, _LABEL_COUNT + 1), dim=-1)
    slot_vectors = torch.randn(_LABEL_COUNT + 2, _VECTOR_DIM)
    value_vectors = torch.randn(_LABEL_COUNT, _VECTOR_DIM)
    return (
        utterance_representations_full,
        system_act_slots,
        system_act_confirm_slots,
        system_act_confirm_values,
        y_past_state,
        slot_vectors,
        value_vectors,
    )


MENAGERIE_ENTRIES = [
    ("Neural Belief Tracker (NBT)", build_nbt, example_input_nbt, 2017, "ported-pytorch"),
]
