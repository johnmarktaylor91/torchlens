# FAITHFUL PORT of https://github.com/jilljenn/ktm @ master (original framework: numpy/autograd
# + libFM C++ binary via pywFM)
#
# Knowledge Tracing Machines (Vie & Kashima, AAAI 2019) score a sparse one-hot-encoded
# feature vector (user, item, skill, wins, fails, ...) with a second-order Factorization
# Machine: y = mu + x.w + 0.5 * sum_k( (x.V_k)^2 - x^2.V_k^2 ). The repo's real runnable
# implementations are (a) `ofm.py`'s OFM class -- pure numpy/autograd, hand-rolled SGD,
# no nn.Module -- and (b) `fm.py`, which does not implement the FM math in Python at all
# but instead shells out to the `pywFM` wrapper around the external compiled `libFM`
# binary (`os.environ['LIBFM_PATH']`); neither is a torch nn.Module we can vendor
# directly or run in this environment. This port transcribes OFM.predict's exact
# formula (see `ofm.py`, function `predict`) into an nn.Module: dense one-hot encoding
# handled the same way (`X_fm = X`, `X2_fm = X`, matching the repo's toggled-off
# sparse-squaring shortcut), factor matrix `V` and linear weights `w` as nn.Parameter,
# global bias `mu` as nn.Parameter, sigmoid link -- no architectural invention beyond
# translating numpy ops to torch ops one-for-one.

import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """Direct torch port of ktm/ofm.py's OFM.predict formula."""

    def __init__(self, n_features, d=5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.rand(n_features))
        self.V = nn.Parameter(torch.rand(n_features, d))

    def forward(self, X):
        # X: (batch, n_features) dense one-hot / multi-hot encoded feature matrix,
        # exactly as ktm/encode.py produces (converted to dense here; OFM.predict
        # itself operates on X.toarray() / X_fm = X when the sparse-squaring shortcut
        # is disabled, as in the repo's default code path).
        V2 = self.V**2
        X2 = X**2

        linear_term = X @ self.w
        interaction_term = 0.5 * ((X @ self.V) ** 2 - (X2 @ V2)).sum(dim=1)

        y_pred = self.mu + linear_term + interaction_term
        return torch.sigmoid(y_pred)


_N_FEATURES = 40
_D = 5


def build_ktm():
    model = FactorizationMachine(n_features=_N_FEATURES, d=_D)
    model.eval()
    return model


def example_input_ktm():
    batch = 8
    # One-hot/multi-hot sparse feature encoding of (user, item, skill, wins, fails),
    # densified -- matches the shape produced by ktm/encode.py's scipy.sparse output
    # once converted to a dense tensor for the FM forward pass.
    X = torch.zeros(batch, _N_FEATURES)
    idx = torch.randint(0, _N_FEATURES, (batch, 4))
    X.scatter_(1, idx, 1.0)
    return X


MENAGERIE_ENTRIES = [
    ("KTM", build_ktm, example_input_ktm, 2019, "MENAGERIE_ZOO"),
]

MENAGERIE_ZOO = "ported-pytorch"
