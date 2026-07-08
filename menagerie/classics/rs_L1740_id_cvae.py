# SOURCE: vendored from imoken1122/Intrusion-Detection-CVAE @ master
#
# Vendors the real `CVAE` nn.Module from model.py: a label-conditioned variational
# autoencoder for network-intrusion detection (trained/evaluated on the NSL-KDD dataset per
# the repo's dataset/ + cvea.py training script), i.e. a Conditional VAE (CVAE) whose decoder
# is conditioned on a one-hot attack-class label concatenated into the hidden state.
#
# No fixes needed -- the class runs as-is; only the training/data-loading glue (cvea.py's
# pandas/CSV loading, Adadelta training loop, matplotlib plotting) was left out since it is
# harness code, not the model definition itself, per the module contract.
#
# Repo: https://github.com/imoken1122/Intrusion-Detection-CVAE @ master
# File: model.py

import torch as th
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(116, 500)
        self.hidden = nn.Linear(500, 500)
        self.mu = nn.Linear(500, 25)
        self.sigma = nn.Linear(500, 25)

        self.fc2 = nn.Linear(25, 495)
        self.fc3 = nn.Linear(500, 116)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def encoder(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.hidden(h))
        h = self.hidden(h)
        return self.mu(h), self.sigma(h)

    def revize_parameter(self, mu, logsigma):
        sigma = th.exp(0.5 * logsigma)
        eps = th.randn(sigma.size())
        return sigma.mul(eps) + mu

    def decoder(self, z, oh_label):
        h = self.relu(self.fc2(z))

        h = th.cat((h, oh_label), dim=1)
        h = self.fc3(self.relu(self.hidden(h)))
        return self.sigmoid(h)

    def forward(self, x, label):
        mu, sigma = self.encoder(x)
        z = self.revize_parameter(mu, sigma)
        output = self.decoder(z, label)
        return output, mu, sigma


def build_id_cvae():
    return CVAE()


def example_input_id_cvae():
    # x: 116-dim NSL-KDD one-hot/scaled feature vector (fc1 in_features=116);
    # label: 5-way one-hot attack-class label (fc3 concatenates 495 + 5 = 500 == hidden width).
    batch = 4
    x = th.rand(batch, 116)
    label = F.one_hot(th.randint(0, 5, (batch,)), num_classes=5).float()
    return (x, label)


MENAGERIE_ENTRIES = [
    (
        "ID-CVAE (Intrusion-Detection Conditional VAE)",
        build_id_cvae,
        example_input_id_cvae,
        2018,
        MENAGERIE_ZOO,
    ),
]
