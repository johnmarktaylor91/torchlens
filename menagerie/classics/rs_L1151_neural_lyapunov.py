# SOURCE: vendored from YaChienChang/Neural-Lyapunov-Control @ master
# https://github.com/YaChienChang/Neural-Lyapunov-Control/blob/master/Inverted%20_Pendulum.ipynb
# (identical `Net` class also appears in
#  "Path_Following (kinematic bicycle model).ipynb")
# "Neural Lyapunov Control" (Chang, Roohi, Gao, NeurIPS 2019). The `Net` class is the
# real `torch.nn.Module` from the paper's official notebooks: a 2-layer tanh MLP that
# outputs a learned Lyapunov-function candidate `V(x)`, plus a linear control-policy
# head `u(x)` initialized from an LQR solution. The notebook's training loop couples
# this network to `dreal` (an SMT solver, used only for formal falsification of the
# candidate Lyapunov function -- NOT part of the network's forward pass) and is not
# vendored here since it needs a non-base package; the `Net` module itself needs only
# `torch` and is copied verbatim.
"""SOURCE: vendored Neural-Lyapunov-Control `Net` module (Chang et al., NeurIPS 2019)."""

import torch


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, lqr):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)
        self.control = torch.nn.Linear(n_input, 1, bias=False)
        self.control.weight = torch.nn.Parameter(lqr)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        u = self.control(x)
        return out, u


_N_INPUT = 2
_N_HIDDEN = 6
_N_OUTPUT = 1


def build_neural_lyapunov() -> torch.nn.Module:
    # LQR solution for the inverted-pendulum demo, from the real notebook.
    lqr = torch.tensor([[-23.58639732, -5.31421063]])
    model = Net(_N_INPUT, _N_HIDDEN, _N_OUTPUT, lqr)
    model.eval()
    return model


def example_input_neural_lyapunov() -> torch.Tensor:
    torch.manual_seed(10)
    x = torch.Tensor(8, _N_INPUT).uniform_(-6, 6)
    return x


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Neural Lyapunov Control",
        build_neural_lyapunov,
        example_input_neural_lyapunov,
        2019,
        MENAGERIE_ZOO,
    ),
]
