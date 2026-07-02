# SOURCE: vendored from https://github.com/YichenTang97/DBN_Autoencoder_Classifier @ main
# (RBM.py :: RBM, GBRBM  +  modules.py :: DBN, CDBN)
#
# DBN-TEP: a Deep Belief Network built from stacked Restricted Boltzmann
# Machines (a Gaussian-Bernoulli RBM for the real-valued visible layer, then
# Bernoulli RBMs for subsequent hidden layers), pretrained greedily via
# contrastive divergence, then unrolled into a classifier (CDBN) by attaching
# a fine-tuned encoder to a linear class-output layer. The companion repo
# (mv-per Tennessee Eastman Process benchmark data) applies this DBN
# classifier to Tennessee Eastman Process (TEP) fault classification -- this
# is that repo's DBN-classifier construction.
#
# The RBM/GBRBM/DBN/CDBN classes are vendored verbatim from RBM.py/modules.py
# (only the sklearn-based DBNAC.py convenience wrapper -- unused for tracing --
# was dropped; every nn.Module layer/forward is untouched). Since CDBN.forward
# is the traceable inference path (encoder -> linear classifier head), we
# build the CDBN directly from a DBN's pretrained-in-shape encoder, mirroring
# what `DBNClassifier.fit()` does after RBM contrastive-divergence pretraining
# and autoencoder fine-tuning (both training-only, non-architectural).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# RBM.py
# ---------------------------------------------------------------------------
class RBM(nn.Module):
    """A pytorch implementation of the Bernoulli Restricted Boltzmann Machine (RBM)."""

    def __init__(
        self,
        n_visible,
        n_hidden,
        lr=1e-5,
        epochs=10,
        batch_size=30,
        k=3,
        use_gpu=True,
        verbose=True,
    ):
        super(RBM, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.use_gpu = use_gpu
        self.verbose = verbose

        if torch.cuda.is_available() and use_gpu:
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device_ = torch.device(dev)

        # Initialise weights and biases
        std = 4 * (6.0 / (self.n_visible + self.n_hidden)) ** 0.5
        self.W = torch.normal(mean=0, std=std, size=(self.n_visible, self.n_hidden))
        self.vb = torch.zeros(self.n_visible)
        self.hb = torch.zeros(self.n_hidden)

        self.W = self.W.to(self.device_)
        self.vb = self.vb.to(self.device_)
        self.hb = self.hb.to(self.device_)

    def v_to_h(self, v):
        h = torch.matmul(v, self.W)
        h = torch.add(h, self.hb)
        h = torch.sigmoid(h)
        return h, torch.bernoulli(h)

    def h_to_v(self, h):
        v = torch.matmul(h, self.W.t())
        v = torch.add(v, self.vb)
        v = torch.sigmoid(v)
        return v, torch.bernoulli(v)

    def forward(self, X):
        return self.v_to_h(X)


class GBRBM(RBM):
    """A Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM).

    Visible layer can assume real values, while hidden layer assumes binary values only.
    """

    def h_to_v(self, h):
        v = torch.matmul(h, self.W.t())
        v = torch.add(v, self.vb)
        return v, v + torch.normal(mean=0, std=1, size=v.shape).to(self.device_)


# ---------------------------------------------------------------------------
# modules.py
# ---------------------------------------------------------------------------
class DBN(nn.Module):
    """Deep Belief Network (DBN): stack of an input GBRBM layer with multiple hidden RBM layers."""

    def __init__(
        self,
        n_visible,
        n_hiddens,
        lr=1e-5,
        epochs=100,
        batch_size=50,
        k=3,
        use_gpu=True,
        verbose=True,
    ):
        super(DBN, self).__init__()

        self.n_layers = len(n_hiddens)
        self.n_visible = n_visible
        self.n_hiddens = n_hiddens
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k

        self.rbm_layers_ = []
        for i in range(self.n_layers):
            if i == 0:
                n_in = n_visible
                rbm = GBRBM(
                    n_in,
                    n_hiddens[0],
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    k=k,
                    use_gpu=use_gpu,
                    verbose=verbose,
                )
            else:
                n_in = n_hiddens[i - 1]
                rbm = RBM(
                    n_in,
                    n_hiddens[i],
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    k=k,
                    use_gpu=use_gpu,
                    verbose=verbose,
                )
            self.rbm_layers_.append(rbm)

    def forward(self, X):
        h = torch.as_tensor(X, dtype=torch.float)
        for rbm in self.rbm_layers_:
            h = h.view((h.shape[0], -1))
            p_h, h = rbm.v_to_h(h)
        return p_h, h


class CDBN(nn.Module):
    """A classifier module constructed from a (fine-tuned) DBN encoder."""

    def __init__(
        self,
        encoder,
        encode_size,
        n_class=5,
        loss="CrossEntropyLoss",
        optimizer="Adam",
        lr=0.01,
        epochs=50,
        batch_size=50,
        loss_kwargs={},
        optimizer_kwargs=dict(),
        verbose=True,
    ):
        super(CDBN, self).__init__()

        self.encoder = encoder
        self.encode_size = encode_size
        self.n_class = n_class
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_kwargs = loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.output_layer_ = nn.Linear(self.encode_size, n_class)

    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float)
        return self.output_layer_(self.encoder(X))


def _build_encoder(n_visible, n_hiddens):
    """Mirrors AEDBN.construct_autoencoder()'s encoder-half construction: a
    Linear+Sigmoid stack per DBN layer, sized from the DBN's RBM stack shapes
    (weights themselves are just random-init here, standing in for the
    contrastive-divergence-pretrained + autoencoder-fine-tuned weights the
    real training pipeline would produce before `to_clf()` is called)."""
    modules = []
    n_in = n_visible
    for n_hidden in n_hiddens:
        modules.append(nn.Linear(n_in, n_hidden))
        modules.append(nn.Sigmoid())
        n_in = n_hidden
    return nn.Sequential(*modules)


# Tennessee Eastman Process benchmark: 52 process variables, small hidden
# stack, 21 fault classes (+ normal) -- matching the repo's TEP application.
_N_VISIBLE = 52
_N_HIDDENS = [32, 16]
_N_CLASS = 22


def build_dbn_tep():
    encoder = _build_encoder(_N_VISIBLE, _N_HIDDENS)
    return CDBN(encoder, _N_HIDDENS[-1], n_class=_N_CLASS)


def example_input_dbn_tep():
    return torch.randn(8, _N_VISIBLE)


MENAGERIE_ENTRIES = [
    (
        "DBN-TEP",
        build_dbn_tep,
        example_input_dbn_tep,
        2023,
        MENAGERIE_ZOO,
    ),
]
