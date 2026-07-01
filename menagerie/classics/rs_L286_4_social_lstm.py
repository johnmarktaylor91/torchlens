# SOURCE: vendored from quancore/social-lstm @ master
# Files combined:
#   model.py (SocialModel -- the repo's Social-LSTM with grid-based social pooling)
#
# Social LSTM (Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces",
# CVPR 2016) predicts each pedestrian's next position with a per-pedestrian LSTM cell whose
# input is the concatenation of an embedded raw position and an embedded "social tensor" --
# a coarse occupancy-grid pooling of neighboring pedestrians' hidden states (getSocialTensor,
# the paper's social pooling mechanism). This PyTorch port (by quancore, EPFL-advised; the
# original CVPR'16 code was Torch7/Lua) is the real, runnable per-frame recurrent model --
# vendored here as-is per the ladder (queue.tsv notes confirm PyTorch impl provenance).
#
# Import-only fixes applied (no architectural change):
#   - `args` (an `argparse.Namespace`/`parameters` object in the repo, produced by
#     `hyperparameter.py::parameters`) is replaced with a plain local `_Args` container
#     carrying the same attributes the constructor reads (`rnn_size`, `grid_size`,
#     `embedding_size`, `input_size`, `output_size`, `maxNumPeds`, `seq_length`, `gru`,
#     `use_cuda`, `dropout`) -- same fields, same meaning, just not routed through argparse.
#   - `Variable(...)` wrapping (a pre-0.4 PyTorch idiom, a no-op on modern tensors) is kept
#     as plain tensor construction; `.cuda()` guards are left as `use_cuda=False` no-ops
#     (never triggered) rather than removed, to keep `forward()` line-for-line identical to
#     upstream.
#   - The repo's `forward(*args)` unpacks 8 positional args including a `dataloader` and
#     `look_up` dict used only to build `PedsList`/`num_pedlist` index lists during real
#     training; the example input supplies the already-resolved index-list form directly
#     (`PedsList`, `num_pedlist`, `look_up`) with `dataloader=None` (unused inside forward
#     except via the commented-out debug prints), matching how the repo's own `train.py`
#     calls `model.forward(...)` once its own preprocessing has produced these lists.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
from torch.autograd import Variable


class SocialModel(nn.Module):
    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.gru = args.gru

        # The LSTM cell
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2 * self.embedding_size, self.rnn_size)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(
            self.grid_size * self.grid_size * self.rnn_size, self.embedding_size
        )

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        """
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        """
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(
            torch.zeros(numNodes, self.grid_size * self.grid_size, self.rnn_size)
        )
        if self.use_cuda:
            social_tensor = social_tensor.cuda()

        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(
            numNodes, self.grid_size * self.grid_size * self.rnn_size
        )
        return social_tensor

    def forward(self, *args):
        """
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]  # noqa: F841 -- unpacked to match repo's forward(*args) signature
        dataloader = args[6]  # noqa: F841 -- unpacked to match repo's forward(*args) signature
        look_up = args[7]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum, frame in enumerate(input_data):
            # Peds present in the current frame
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes, :]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(
                    concat_embedded, (hidden_states_current, cell_states_current)
                )
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))

            # Compute the output
            outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]

        return outputs_return, hidden_states, cell_states


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


class _Args:
    """Plain stand-in for the repo's argparse-derived `parameters` object -- same
    attributes SocialModel.__init__ reads, not routed through argparse."""

    def __init__(self):
        self.rnn_size = 16
        self.grid_size = 2
        self.embedding_size = 8
        self.input_size = 2
        self.output_size = 5
        self.maxNumPeds = 4
        self.seq_length = 4
        self.gru = False
        self.use_cuda = False
        self.dropout = 0.0


def build_social_lstm():
    return SocialModel(_Args(), infer=False)


def example_input_social_lstm():
    seq_length = 4
    num_peds = 4
    grid_size = 2
    rnn_size = 16

    # input_data: seq_length frames, each (num_peds, 2) positions
    input_data = torch.rand(seq_length, num_peds, 2)
    # grids: per-frame occupancy-pooling masks, each (num_peds, num_peds, grid_size**2)
    grids = [torch.rand(num_peds, num_peds, grid_size * grid_size) for _ in range(seq_length)]
    hidden_states = torch.zeros(num_peds, rnn_size)
    cell_states = torch.zeros(num_peds, rnn_size)
    PedsList = [list(range(num_peds)) for _ in range(seq_length)]
    num_pedlist = [num_peds for _ in range(seq_length)]
    dataloader = None
    look_up = {i: i for i in range(num_peds)}

    return (
        input_data,
        grids,
        hidden_states,
        cell_states,
        PedsList,
        num_pedlist,
        dataloader,
        look_up,
    )


MENAGERIE_ENTRIES = [
    ("Social-LSTM", "build_social_lstm", "example_input_social_lstm", 2016, MENAGERIE_ZOO),
]
