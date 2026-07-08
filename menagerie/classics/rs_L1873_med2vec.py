# FAITHFUL PORT of https://github.com/mp2893/med2vec @ master (med2vec.py) (original framework: Theano)
#
# Med2Vec (Choi et al., KDD 2016 "Multi-layer Representation Learning for Medical
# Concepts") learns visit-level and code-level embeddings from EHR visit sequences.
# The original `med2vec.py` is Theano + Python 2 (`print` statements, `iteritems`,
# `xrange`) and cannot run in this environment. Its `init_params`/`build_model`
# define the ENTIRE forward architecture:
#
#   emb    = relu(x @ W_emb + b_emb)                       # multi-hot codes -> code embedding
#   [emb   = concat(emb, demographics)]                     # optional demographic features
#   visit  = relu(emb @ W_hidden + b_hidden)                # visit representation
#   result = softmax(visit @ W_output + b_output)           # next/prev-visit code prediction
#
# This is transcribed verbatim (same three affine+nonlinearity stages, same
# ordering, same optional demographic-concat branch) as a torch nn.Module. The
# Theano code trains one forward pass per padded visit-matrix batch (shape
# [n_visits, numXcodes]) with a skip-gram-style code-embedding loss added on top
# at train time (`emb_cost` via the iVector/jVector code-cooccurrence pairs) --
# that auxiliary loss term operates on `W_emb` directly and is not part of the
# per-visit forward computation graph reproduced here (forward inference is the
# visit_representation -> next-visit-code-distribution path used for downstream
# tasks in the paper/README, e.g. "Heart Failure Prediction via med2vec").

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class Med2Vec(nn.Module):
    """
    Faithful port of med2vec.py::init_params + med2vec.py::build_model.

    :param num_x_codes: size of the multi-hot input code vocabulary (numXcodes).
    :param num_y_codes: size of the (optionally grouped) output code vocabulary
        (numYcodes); if 0, the model predicts over the same numXcodes vocabulary
        (matches the original `if numYcodes > 0 ... else ...` branch).
    :param emb_dim_size: code embedding dimension (embDimSize / --cr_size).
    :param hidden_dim_size: visit representation dimension (hiddenDimSize / --vr_size).
    :param demo_size: size of the demographic feature vector (demoSize); 0 disables
        the demographic-concat branch, matching the original `demoSize > 0` checks.
    """

    def __init__(
        self,
        num_x_codes: int,
        num_y_codes: int = 0,
        emb_dim_size: int = 200,
        hidden_dim_size: int = 200,
        demo_size: int = 0,
    ):
        super().__init__()
        self.num_x_codes = num_x_codes
        self.num_y_codes = num_y_codes
        self.demo_size = demo_size

        # params['W_emb'], params['b_emb']
        self.emb = nn.Linear(num_x_codes, emb_dim_size)
        self.relu_emb = nn.ReLU()

        # params['W_hidden'], params['b_hidden']  (input width grows by demoSize
        # when demographic features are concatenated onto the code embedding)
        self.hidden = nn.Linear(emb_dim_size + demo_size, hidden_dim_size)
        self.relu_hidden = nn.ReLU()

        # params['W_output'], params['b_output']
        out_size = num_y_codes if num_y_codes > 0 else num_x_codes
        self.output = nn.Linear(hidden_dim_size, out_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, d: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: [n_visits, num_x_codes] multi-hot visit code matrix.
        :param d: [n_visits, demo_size] demographic feature matrix, required iff
            demo_size > 0 (mirrors the original `if options['demoSize'] > 0`
            branch in `build_model`).
        """
        emb = self.relu_emb(self.emb(x))
        if self.demo_size > 0:
            assert d is not None, "demo tensor required when demo_size > 0"
            emb = torch.cat((emb, d), dim=1)
        visit = self.relu_hidden(self.hidden(emb))
        results = self.softmax(self.output(visit))
        return results


def build_med2vec():
    # Tiny config in the spirit of the paper's Heart-Failure-prediction setup
    # (numXcodes ~thousands of ICD codes in the original paper; shrunk here).
    return Med2Vec(num_x_codes=64, num_y_codes=0, emb_dim_size=16, hidden_dim_size=16, demo_size=0)


def example_input_med2vec():
    # A batch of multi-hot visit vectors (n_visits=8, num_x_codes=64).
    x = (torch.rand(8, 64) > 0.9).float()
    return (x,)


MENAGERIE_ENTRIES = [
    ("Med2Vec", "build_med2vec", "example_input_med2vec", 2016, "ported-pytorch"),
]
