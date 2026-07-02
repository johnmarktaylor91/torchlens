# SOURCE: vendored from astroai/starnet @ master
# https://raw.githubusercontent.com/astroai/starnet/master/4_Train_Model_pytorch.ipynb
# StarNet (Fabbro et al. 2018, "An application of deep learning in the analysis of
# stellar spectra") predicts stellar atmospheric parameters (Teff, log g, [Fe/H]) from
# APOGEE stellar spectra with a 1D CNN. The queue lists the repo as "TensorFlow" (its
# original Keras implementation, `CNN/DataDriven_StarNet_CNN.ipynb`), but the SAME repo
# also ships a real, author-written PyTorch port (`4_Train_Model_pytorch.ipynb`,
# `5_Test_Model_pytorch.ipynb`) with an equivalent `StarNet` nn.Module -- we vendor that
# torch code directly rather than porting from the Keras/TF version. Only the notebook's
# hardcoded `num_fluxes`/`compute_out_size` driver logic is adapted into a normal
# `__init__`-time computation for staging; the StarNet architecture itself is untouched.
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]


class StarNet(nn.Module):
    def __init__(self, num_fluxes, num_filters, filter_length, pool_length, num_hidden, num_labels):
        super().__init__()

        # Convolutional and pooling layers
        self.conv1 = nn.Conv1d(1, num_filters[0], filter_length)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], filter_length)
        self.pool = nn.MaxPool1d(pool_length, pool_length)

        # Determine shape after pooling
        pool_output_shape = compute_out_size(
            (1, num_fluxes), nn.Sequential(self.conv1, self.conv2, self.pool)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(pool_output_shape[0] * pool_output_shape[1], num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.output = nn.Linear(num_hidden[1], num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


# Real notebook defaults: num_filters=[4,16], filter_length=8, pool_length=4,
# num_hidden=[256,128], num_labels=3 (Teff, log g, [Fe/H]). num_fluxes=7214 in the real
# APOGEE-spectrum dataset; we use a much smaller (but architecturally valid) spectrum
# length for a fast tiny-config trace.
_NUM_FLUXES = 128
_NUM_FILTERS = [4, 16]
_FILTER_LENGTH = 8
_POOL_LENGTH = 4
_NUM_HIDDEN = [256, 128]
_NUM_LABELS = 3


def build_starnet():
    return StarNet(
        _NUM_FLUXES, _NUM_FILTERS, _FILTER_LENGTH, _POOL_LENGTH, _NUM_HIDDEN, _NUM_LABELS
    )


def example_input_starnet():
    # (batch, 1 channel, num_fluxes) -- a single stellar spectrum per sample
    return torch.randn(4, 1, _NUM_FLUXES)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "StarNet (stellar spectra CNN)",
        "build_starnet",
        "example_input_starnet",
        2018,
        "vendored-pytorch",
    ),
]
