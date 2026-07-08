# FAITHFUL PORT of linyuwangPHD/RNA-Secondary-Structure-Database @ master
# (DMfold_Program/PU_Part/SecondaryModel.py, original framework: TensorFlow 1.x)
#
# DMfold (Wang, Liu, Fan & Yao, Frontiers in Genetics 2019) predicts RNA secondary
# structure per-base: a 3-layer bidirectional LSTM sequence encoder (`PaperModel` in
# the real repo, `HIDDEN_SIZE=300`, `NUM_LAYERS=3`) reads the length-300-padded 8-dim
# per-base input, and a 3-layer fully-connected decoder (2*HIDDEN_SIZE -> HIDDEN_SIZE ->
# HIDDEN_SIZE/2 -> N_CLASSES=7, each hidden layer ReLU + dropout) maps each timestep's
# concatenated forward/backward LSTM state to one of 7 per-base structural classes.
# The real repo (`DMfold_Program/PU_Part/SecondaryModel.py`) is TF1.x graph-mode code
# (`tf.placeholder`, `tf.get_variable`, `tf.nn.rnn_cell.BasicLSTMCell` +
# `tf.nn.static_bidirectional_rnn`, manual `tf.clip_by_global_norm` training loop) --
# TF1.x is not installed and is not reasonably compatible with the torch/timm/
# transformers stack already in this env, so the architecture is faithfully
# transcribed into torch: `tf.nn.rnn_cell.MultiRNNCell` of 3 `BasicLSTMCell`s run in
# both directions (`static_bidirectional_rnn`) is the same computation as
# `nn.LSTM(hidden_size, num_layers=3, bidirectional=True)`; the two 300x300 and one
# 300x150 dense layers with ReLU + dropout between (`weightone/weighttwo/weightthree`)
# are transcribed as `nn.Linear` + `nn.ReLU` + `nn.Dropout`. Training-only elements
# (dropout-keep-prob toggling by `is_training`, gradient-norm clipping, the exponential
# learning-rate decay, and the sequence-length masking used only for the loss) are
# dropped since they have no effect on the forward computation graph.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

HIDDEN_SIZE = 300  # The number of LSTM hidden nodes (real repo constant).
NUM_LAYERS = 3  # The number of stacked LSTM layers (real repo constant).
NUM_INPUT = 8  # The number of input features per base-position step (real repo constant).
N_CLASSES = 7  # The number of per-base structural classes predicted (real repo constant).


class DMfoldSecondaryModel(nn.Module):
    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_input=NUM_INPUT,
        n_classes=N_CLASSES,
        dropout_keep_prob=0.9,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # tf.nn.rnn_cell.MultiRNNCell([...NUM_LAYERS BasicLSTMCells...]) run via
        # tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, ...) == nn.LSTM(...,
        # num_layers=NUM_LAYERS, bidirectional=True); inter-layer dropout matches the
        # real repo's DropoutWrapper(output_keep_prob=LSTM_KEEP_PROB) between LSTM layers.
        self.bilstm = nn.LSTM(
            input_size=num_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(1.0 - dropout_keep_prob) if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )

        # Decoder: weightone (2H,H) -> weighttwo (H,H/2) -> weightthree (H/2,N_CLASSES),
        # each followed by ReLU then dropout (FULL_LAYER_DROPOUT), exactly as the real
        # repo's `layer1`/`layer2`/`logits` computation.
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=1.0 - dropout_keep_prob)

    def forward(self, x):
        # x: [num_steps, batch, num_input] (real repo: input_data is [batch, num_steps,
        # num_input] then transposed to [num_steps, batch, num_input] before splitting
        # into a per-timestep list for static_bidirectional_rnn -- nn.LSTM's default
        # batch_first=False layout is the direct torch equivalent of that transposed form).
        outputs, _ = self.bilstm(x)  # [num_steps, batch, 2*hidden_size]
        num_steps, batch, _ = outputs.shape
        outputs = outputs.reshape(-1, 2 * self.hidden_size)  # [num_steps*batch, 2*hidden]

        layer1 = self.dropout(self.relu(self.fc1(outputs)))
        layer2 = self.dropout(self.relu(self.fc2(layer1)))
        logits = self.relu(self.fc3(layer2))  # real repo applies relu to the final logits too

        return logits.reshape(num_steps, batch, -1)


def build_dmfold():
    # Real repo's PaperModel(is_training, batch_size, num_steps, n_input, n_classes) uses
    # HIDDEN_SIZE=300 across NUM_LAYERS=3 stacked bidirectional LSTM layers -- kept as-is
    # since these are the real published architecture hyperparameters, not a knob a
    # caller varies per-run.
    return DMfoldSecondaryModel()


def example_input_dmfold():
    torch.manual_seed(0)
    # [num_steps, batch, NUM_INPUT] -- real repo pads/truncates every RNA sequence to
    # TRAIN_NUM_STEP=300 steps; a small num_steps is used here purely to keep the tiny
    # random-init trace fast (architecture and per-step feature width are unchanged).
    num_steps, batch = 5, 2
    return (torch.randn(num_steps, batch, NUM_INPUT),)


MENAGERIE_ENTRIES = [
    ("DMfold", "build_dmfold", "example_input_dmfold", 2019, MENAGERIE_ZOO),
]
