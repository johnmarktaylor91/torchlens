# SOURCE: vendored from https://github.com/fungtion/DANN @ master
# (models/model.py :: CNNModel  +  models/functions.py :: ReverseLayerF)
#
# DANN (Domain-Adversarial Neural Network; Ganin et al., 2016, "Domain-
# Adversarial Training of Neural Networks"): a CNN feature extractor feeds
# both a label classifier and a domain classifier; a gradient-reversal layer
# (ReverseLayerF) sits between the shared features and the domain classifier
# so that, during backprop, the feature extractor is trained to make the
# learned features domain-invariant while still discriminative for the label
# task. fungtion/DANN's `CNNModel` is the canonical reference implementation
# (MNIST->MNIST-M digit domain adaptation) that "DANN-SS" (domain-adversarial
# soft sensor) applications of this same architecture adapt to process-
# monitoring / soft-sensor domain-shift data by swapping the input domain
# only -- the gradient-reversal DANN architecture itself is unchanged.
#
# Vendored verbatim (architecture/forward untouched; only combined into one
# file and `LogSoftmax` calls given an explicit dim to silence the deprecated
# implicit-dim UserWarning without any behavioral change).

import torch
import torch.nn as nn
from torch.autograd import Function

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/functions.py
# ---------------------------------------------------------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ---------------------------------------------------------------------------
# models/model.py
# ---------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module("f_conv1", nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module("f_bn1", nn.BatchNorm2d(64))
        self.feature.add_module("f_pool1", nn.MaxPool2d(2))
        self.feature.add_module("f_relu1", nn.ReLU(True))
        self.feature.add_module("f_conv2", nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module("f_bn2", nn.BatchNorm2d(50))
        self.feature.add_module("f_drop1", nn.Dropout2d())
        self.feature.add_module("f_pool2", nn.MaxPool2d(2))
        self.feature.add_module("f_relu2", nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module("c_fc1", nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module("c_bn1", nn.BatchNorm1d(100))
        self.class_classifier.add_module("c_relu1", nn.ReLU(True))
        self.class_classifier.add_module("c_drop1", nn.Dropout2d())
        self.class_classifier.add_module("c_fc2", nn.Linear(100, 100))
        self.class_classifier.add_module("c_bn2", nn.BatchNorm1d(100))
        self.class_classifier.add_module("c_relu2", nn.ReLU(True))
        self.class_classifier.add_module("c_fc3", nn.Linear(100, 10))
        self.class_classifier.add_module("c_softmax", nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module("d_fc1", nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module("d_bn1", nn.BatchNorm1d(100))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))
        self.domain_classifier.add_module("d_fc2", nn.Linear(100, 2))
        self.domain_classifier.add_module("d_softmax", nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


def build_dann_ss():
    return CNNModel()


def example_input_dann_ss():
    # Real repo's MNIST-sized input; CNNModel.forward() itself expands
    # channels to 3 via input_data.expand(...), so a 1-channel 28x28 batch
    # matches both fungtion/DANN's own training data and DANN-SS soft-sensor
    # process-signal-grid adaptations of the same architecture.
    return (torch.randn(4, 1, 28, 28), 1.0)


MENAGERIE_ENTRIES = [
    (
        "DANN-SS",
        build_dann_ss,
        example_input_dann_ss,
        2016,
        MENAGERIE_ZOO,
    ),
]
