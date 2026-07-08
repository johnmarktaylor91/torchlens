# SOURCE: vendored from XifengGuo/CapsNet-Pytorch @ master
# (capsulelayers.py + capsulenet.py CapsuleNet class, real Hinton dynamic-routing
# capsule layers, applied here as "Capsule-Fault" per queue notes: CapsNet adapted
# for bearing-fault / vibration-signal classification using the standard CapsNet
# PyTorch base repo -- https://github.com/XifengGuo/CapsNet-Pytorch)
#
# Only functional change from the original: replaced hardcoded `.cuda()` calls
# with device-agnostic tensor construction so the model runs on CPU (the
# repo assumes a CUDA-only training loop). No architectural change.

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large
    vector to near 1 and small vector to 0.
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has
    `in_num` inputs, each is a scalar, the output of the neuron from the former
    layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size =
    [None, in_num_caps, in_dim_caps] and output size =
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of capsules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(
            0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps)
        )

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = torch.zeros(
            x.size(0), self.out_num_caps, self.in_num_caps, device=x.device, dtype=x.dtype
        )

        assert self.routings > 0, "The 'routings' should be > 0."
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        outputs = self.conv2d(x)
        # .reshape() instead of the original .view(): TorchLens's capture wrappers
        # can leave the conv output non-view-compatible; .reshape() is the standard,
        # behavior-identical PyTorch-recommended substitute and is not an
        # architectural change.
        outputs = outputs.reshape(x.size(0), -1, self.dim_caps)
        return squash(outputs)


class CapsuleNet(nn.Module):
    """
    A Capsule Network, applied here to bearing-fault / vibration-signal spectrogram
    classification ("Capsule-Fault"): 2D time-frequency representations of vibration
    signals (e.g. CWRU bearing dataset) fed through the standard Hinton dynamic-routing
    CapsNet architecture in place of the original MNIST digit images.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes).
        - Output: ((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(
            in_num_caps=32 * 6 * 6,
            in_dim_caps=8,
            out_num_caps=classes,
            out_dim_caps=16,
            routings=routings,
        )

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16 * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if (
            y is None
        ):  # during testing/tracing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = torch.zeros(length.size(), device=x.device, dtype=x.dtype).scatter_(
                1, index.view(-1, 1), 1.0
            )
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


# --- menagerie staging entrypoints -----------------------------------------------

MENAGERIE_ZOO = "vendored-pytorch"


def build_capsule_fault():
    # Sized to match the original 28x28 MNIST spatial layout so the fixed
    # 32*6*6 primarycaps fan-in in DenseCapsule stays valid; here the single
    # input channel stands in for a vibration-signal time-frequency (e.g.
    # CWT/STFT) map rather than a digit image, per the CapsNet-for-bearing-fault
    # adaptation described in the queue notes.
    return CapsuleNet(input_size=[1, 28, 28], classes=10, routings=3)


def example_input_capsule_fault():
    return torch.rand(2, 1, 28, 28)


MENAGERIE_ENTRIES = [
    ("Capsule-Fault", build_capsule_fault, example_input_capsule_fault, 2017, MENAGERIE_ZOO),
]
