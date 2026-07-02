# SOURCE: vendored from AmanPriyanshu/Deep-Belief-Networks-in-PyTorch @ main
#
# Files combined below (imports/paths adjusted only, architecture untouched):
#   RBM.py -> RBM (restricted Boltzmann machine, contrastive-divergence trainer)
#   DBN.py -> DBN (stacked RBMs; `initialize_model()` assembles the greedily
#                  pretrained RBM stack into the real feed-forward torch.nn.Sequential
#                  that TorchLens traces)
#
# A Deep Belief Network is a stack of Restricted Boltzmann Machines pretrained
# layer-by-layer via contrastive divergence; DBN.initialize_model() is the repo's
# own code path that converts the trained RBM stack (W/vb/hb per layer) into the
# equivalent feed-forward torch.nn.Sequential(Linear, Sigmoid, Linear, Sigmoid, ...)
# used at inference time -- that Sequential IS the traceable model this file exposes.
# We skip the (irrelevant-to-tracing) contrastive-divergence training loop and instead
# populate `layer_parameters` with correctly-shaped random tensors before calling the
# real `initialize_model()`, exactly mirroring what the repo does after `train_DBN()`.
#
# Repo: https://github.com/AmanPriyanshu/Deep-Belief-Networks-in-PyTorch

import numpy as np
import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# RBM.py (only the pieces `DBN` depends on: weight/bias shapes + sampling API)
# ---------------------------------------------------------------------------
class RBM:
    def __init__(
        self,
        n_visible,
        n_hidden,
        lr=0.001,
        epochs=5,
        mode="bernoulli",
        batch_size=32,
        k=3,
        optimizer="adam",
        gpu=False,
        savefile=None,
        early_stopping_patience=5,
    ):
        self.mode = mode  # bernoulli or gaussian RBM
        self.n_hidden = n_hidden  # Number of hidden nodes
        self.n_visible = n_visible  # Number of visible nodes
        self.lr = lr  # Learning rate for the CD algorithm
        self.epochs = epochs  # Number of iterations to run the algorithm for
        self.batch_size = batch_size
        self.k = k
        self.optimizer = optimizer
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-07
        self.m = [0, 0, 0]
        self.v = [0, 0, 0]
        self.m_batches = {0: [], 1: [], 2: []}
        self.v_batches = {0: [], 1: [], 2: []}
        self.savefile = savefile
        self.early_stopping_patience = early_stopping_patience
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.progress = []

        if torch.cuda.is_available() and gpu:
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        # Initialize weights and biases
        std = 4 * np.sqrt(6.0 / (self.n_visible + self.n_hidden))
        self.W = torch.normal(mean=0, std=std, size=(self.n_hidden, self.n_visible))
        self.vb = torch.zeros(size=(1, self.n_visible), dtype=torch.float32)
        self.hb = torch.zeros(size=(1, self.n_hidden), dtype=torch.float32)

        self.W = self.W.to(self.device)
        self.vb = self.vb.to(self.device)
        self.hb = self.hb.to(self.device)


# ---------------------------------------------------------------------------
# DBN.py
# ---------------------------------------------------------------------------
class DBN:
    def __init__(self, input_size, layers, mode="bernoulli", gpu=False, k=5, savefile=None):
        self.layers = layers
        self.input_size = input_size
        self.layer_parameters = [{"W": None, "hb": None, "vb": None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile

    def sample_v(self, y, W, vb):
        wy = torch.mm(y, W)
        activation = wy + vb
        p_v_given_h = torch.sigmoid(activation)
        if self.mode == "bernoulli":
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        else:
            return p_v_given_h, torch.add(
                p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape)
            )

    def sample_h(self, x, W, hb):
        wx = torch.mm(x, W.t())
        activation = wx + hb
        p_h_given_v = torch.sigmoid(activation)
        if self.mode == "bernoulli":
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        else:
            return p_h_given_v, torch.add(
                p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape)
            )

    def generate_input_for_layer(self, index, x):
        if index > 0:
            x_gen = []
            for _ in range(self.k):
                x_dash = x.clone()
                for i in range(index):
                    _, x_dash = self.sample_h(
                        x_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["hb"]
                    )
                x_gen.append(x_dash)

            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
        else:
            x_dash = x.clone()
        return x_dash

    def train_DBN(self, x):
        for index, layer in enumerate(self.layers):
            if index == 0:
                vn = self.input_size
            else:
                vn = self.layers[index - 1]
            hn = self.layers[index]

            rbm = RBM(
                vn,
                hn,
                epochs=100,
                mode="bernoulli",
                lr=0.0005,
                k=10,
                batch_size=128,
                gpu=True,
                optimizer="adam",
                early_stopping_patience=10,
            )
            x_dash = self.generate_input_for_layer(index, x)
            rbm.train(x_dash)
            self.layer_parameters[index]["W"] = rbm.W.cpu()
            self.layer_parameters[index]["hb"] = rbm.hb.cpu()
            self.layer_parameters[index]["vb"] = rbm.vb.cpu()
        if self.savefile is not None:
            torch.save(self.layer_parameters, self.savefile)

    def reconstructor(self, x):
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(len(self.layer_parameters)):
                _, x_dash = self.sample_h(
                    x_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["hb"]
                )
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(len(self.layer_parameters)):
                i = len(self.layer_parameters) - 1 - i
                _, y_dash = self.sample_v(
                    y_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["vb"]
                )
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)

        return y_dash, x_dash

    def initialize_model(self):
        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer["W"].shape[1], layer["W"].shape[0]))
            if index < len(self.layer_parameters) - 1:
                modules.append(torch.nn.Sigmoid())
        model = torch.nn.Sequential(*modules)

        for layer_no, layer in enumerate(model):
            if layer_no // 2 == len(self.layer_parameters) - 1:
                break
            if layer_no % 2 == 0:
                model[layer_no].weight = torch.nn.Parameter(
                    self.layer_parameters[layer_no // 2]["W"]
                )
                model[layer_no].bias = torch.nn.Parameter(
                    self.layer_parameters[layer_no // 2]["hb"]
                )

        return model


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_INPUT_SIZE = 10
_LAYERS = [7, 5, 2]


def build_dbn():
    dbn = DBN(_INPUT_SIZE, _LAYERS)
    # Populate layer_parameters with correctly-shaped random tensors (standing in
    # for the contrastive-divergence-trained weights `train_DBN` would produce);
    # this reproduces exactly what `initialize_model()` consumes post-training.
    vn = _INPUT_SIZE
    for index, hn in enumerate(_LAYERS):
        dbn.layer_parameters[index]["W"] = torch.randn(hn, vn) * 0.1
        dbn.layer_parameters[index]["hb"] = torch.zeros(1, hn)
        dbn.layer_parameters[index]["vb"] = torch.zeros(1, vn)
        vn = hn
    return dbn.initialize_model()


def example_input_dbn():
    return torch.randn(4, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    (
        "Deep Belief Network",
        build_dbn,
        example_input_dbn,
        2006,
        "VENDOR",
    ),
]
