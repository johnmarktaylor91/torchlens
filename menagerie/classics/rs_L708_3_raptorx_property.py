# FAITHFUL PORT of Indicator/RaptorX-SS8 @ master (original framework: C++, MPI; no Python)
# Ported from: src/bCNF_mpi_tp.h (bCNF_Model, SEQUENCE classes) and src/bCNF_mpi_tp.cpp
# (SEQUENCE::ComputeGates, SEQUENCE::ComputeVi, SEQUENCE::ComputeScore, SEQUENCE::ComputeForward /
# ComputeBackward). This is the "Conditional Neural Field" (CNF) model of Wang, Zhao & Xu,
# "Protein 8-class secondary structure prediction using conditional neural fields", Proteins 2011
# -- RaptorX-Property's RaptorX-SS8 predictor. The repo has no PyTorch/Theano/Python model code at
# all (raw C++ + MPI, SVN-imported), so this is a faithful architectural transcription (not
# vendored code) of the model actually implemented in bCNF_mpi_tp.{h,cpp}, not a from-scratch
# guess from the paper abstract:
#   - ComputeGates / GetGateOutput -> a sliding-window (window_size positions) linear layer over
#     the per-position dense features (PSSM-like "ps" features + position-independent "pi"
#     features) followed by a sigmoid ("Gate(sum) = 1/(1+exp(-sum))"): implemented here as a
#     Conv1d(kernel_size=window_size, padding=same) + Sigmoid, matching the exact windowed-linear
#     -> sigmoid computation in GetGateOutput.
#   - ComputeVi (SEQUENCE::ComputeVi) -> per-state emission scores are a LINEAR combination of the
#     gate outputs at each position: arrVi[state,pos] = sum_k weights[state,k]*gates[k,pos]. This
#     is a Linear(num_gates -> num_states) applied per position -- implemented as Conv1d(kernel=1).
#   - ComputeScore / ComputeForward / ComputeBackward -> a linear-chain CRF over `num_states`
#     structured labels (bilabel begin/end encoding doubles num_label -> num_states = num_label*2)
#     with a learned transition matrix, scored by the standard forward algorithm (log-partition).
# Sizes shrunk from the paper's reported best config (window_size=13, num_gates=20) to a tiny
# menagerie-scale trace while preserving the 3-stage gate->emission->CRF topology.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class CNFGateLayer(nn.Module):
    """Port of SEQUENCE::ComputeGates / bCNF_Model::GetGateOutput: for every sequence position,
    a windowed (window_size wide, zero-padded at the ends -- matching the reference C++'s
    zero-feature boundary handling) linear combination of the dense per-position features,
    passed through a sigmoid ("neural gate"). One Conv1d channel per gate."""

    def __init__(self, dim_dense, num_gates, window_size):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_dense, num_gates, kernel_size=window_size, padding=window_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, dim_dense, seq_len)
        return self.sigmoid(self.conv(x))


class CNFEmissionLayer(nn.Module):
    """Port of SEQUENCE::ComputeVi: per-position emission score for every structured state is a
    linear combination of that position's gate outputs (bCNF_Model::LocalWeights matrix)."""

    def __init__(self, num_gates, num_states):
        super().__init__()
        self.linear = nn.Conv1d(num_gates, num_states, kernel_size=1)

    def forward(self, gates):
        # gates: (batch, num_gates, seq_len) -> (batch, num_states, seq_len)
        return self.linear(gates)


class LinearChainCRFHead(nn.Module):
    """Port of SEQUENCE::ComputeScore + ComputeForward/ComputeBackward: a linear-chain CRF over
    `num_states` structured labels. Holds a learned (num_states+1, num_states) transition matrix
    (row 0 = DUMMY start-state transitions, exactly as bCNF_Model's `weights[(1+leftState)*
    num_label + currLabel]` indexing) and returns the log-partition function (forward algorithm)
    -- the score RaptorX-SS8 maximizes during CRF training and normalizes for Viterbi decoding."""

    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        # transitions[0] = start -> state scores (DUMMY), transitions[1:] = state -> state.
        self.transitions = nn.Parameter(torch.zeros(num_states + 1, num_states))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions):
        # emissions: (batch, num_states, seq_len)
        batch, num_states, seq_len = emissions.shape
        emissions = emissions.transpose(1, 2)  # (batch, seq_len, num_states)

        # alpha_0 = start-transition + emission_0  (DUMMY -> state row)
        alpha = self.transitions[0].unsqueeze(0) + emissions[:, 0, :]
        for t in range(1, seq_len):
            # broadcast: (batch, leftState, 1) + (num_states, currState) + emission_t
            trans = self.transitions[1:].unsqueeze(0)  # (1, num_states, num_states)
            scores = alpha.unsqueeze(2) + trans  # (batch, leftState, currState)
            alpha = torch.logsumexp(scores, dim=1) + emissions[:, t, :]

        log_partition = torch.logsumexp(alpha, dim=1)  # (batch,)
        return log_partition


class RaptorXProperty(nn.Module):
    """Faithful port of RaptorX-SS8's Conditional Neural Field model (Wang, Zhao & Xu 2011):
    windowed-sigmoid gate layer -> per-position linear emission head -> linear-chain CRF
    log-partition, over the 8-class (bilabel-doubled) protein secondary-structure state space."""

    def __init__(self, dim_dense, num_gates, window_size, num_label):
        super().__init__()
        num_states = num_label * 2  # bilabel begin/end encoding (bCNF_mpi_tp.cpp: num_label*=2)
        self.gate = CNFGateLayer(dim_dense, num_gates, window_size)
        self.emission = CNFEmissionLayer(num_gates, num_states)
        self.crf = LinearChainCRFHead(num_states)

    def forward(self, x):
        # x: (batch, dim_dense, seq_len) dense per-position features (PSSM + position-independent)
        gates = self.gate(x)
        emissions = self.emission(gates)
        log_partition = self.crf(emissions)
        return log_partition


def build_raptorx_property():
    # Reference best config: window_size=13, num_gates=20, num_label=8 (SS8) -> num_states=16.
    # Shrunk here: window_size=5, num_gates=6, num_label=8 (kept, defines the SS8 output space).
    return RaptorXProperty(dim_dense=10, num_gates=6, window_size=5, num_label=8)


def example_input_raptorx_property():
    torch.manual_seed(0)
    # batch=1, dim_dense=10 (PSSM-like + position-independent features), seq_len=24 residues.
    return (torch.randn(1, 10, 24),)


MENAGERIE_ENTRIES = [
    (
        "RaptorX-Property",
        "build_raptorx_property",
        "example_input_raptorx_property",
        2011,
        "ported-pytorch",
    ),
]
