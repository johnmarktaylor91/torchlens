# SOURCE: vendored from FunctionLab/ExPecto @ master (chromatin.py)
#
# Beluga: the DeepSEA-derived chromatin-effect CNN used by ExPecto (Zhou et al. 2018,
# "Deep learning sequence-based ab initio prediction of variant effects on expression
# and disease risk"). Beluga takes a one-hot-encoded 4x2000bp DNA window (as a
# (N, 4, 1, 2000) 2D tensor with a singleton "height" axis) and predicts 2002 chromatin
# marks via a stack of Conv2d+ReLU+Dropout+MaxPool blocks followed by two Linear heads.
# Copied verbatim from the real repo's Beluga/LambdaBase/Lambda classes (aside from
# stripping the CLI/argparse wrapper and the pretrained-weights load, which are not part
# of the architecture).
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 320, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(
                    Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
                    nn.Linear(67840, 2003),
                ),
                nn.ReLU(),
                nn.Sequential(
                    Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
                    nn.Linear(2003, 2002),
                ),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def build_beluga():
    return Beluga()


def example_input_beluga():
    # Real input: one-hot DNA (N, 4, 1, 2000); use a smaller batch for tracing speed.
    return torch.randn(2, 4, 1, 2000)


MENAGERIE_ENTRIES = [
    ("Beluga", build_beluga, example_input_beluga, 2018, "SOURCE_AVAILABLE"),
]
