# SOURCE: vendored from FunctionLab/ExPecto @ master (chromatin.py)
#
# ExPecto (Zhou et al., Nature Genetics 2018) predicts tissue-specific
# transcriptional effects of genetic variants by feeding a chromatin-effect
# representation (produced by the "Beluga" deep CNN, itself a re-trained
# extension of the DeepSEA architecture predicting 2002 chromatin features
# from a 2000bp one-hot DNA window) into downstream gradient-boosted trees.
# This vendors the real `LambdaBase` / `Lambda` / `Beluga` classes verbatim
# from chromatin.py: a 3-stage Conv2d(kernel=(1,8)) stack (each stage:
# conv-relu-conv-relu[-dropout-maxpool]) over a one-hot (4, 1, 2000) DNA
# window, flattened through two `Lambda`-wrapped reshape ops into a
# Linear(67840, 2003) -> ReLU -> Linear(2003, 2002) -> Sigmoid head. Only the
# module-level CLI/argparse/genome-loading/`model.load_state_dict(...)`
# script glue (which needs a local hg19.fa + pretrained .pth checkpoint) is
# dropped; the Beluga forward-pass architecture is unchanged.
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


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Real usage: a one-hot (batch, 4, 1, 2000) DNA window (2000bp, forward +
# reverse-complement strand encoded across the 4 channels). The
# Linear(67840, 2003) head bakes in the exact 2000bp window length via the
# conv/pool stack's flattened output size (empirically verified: 4 -> 320 ->
# 480 -> 640 channels over 3 conv2d(kernel=(1,8)) stages with two
# MaxPool2d((1,4)) reductions yields flatten dim 640*106==67840 for L=2000),
# so this architecture is not "tiny-input-shrinkable" like most CNNs -- the
# real fixed input size is used as-is.
_SEQ_LEN = 2000


def build_expecto_beluga():
    return Beluga()


def example_input_expecto_beluga():
    return torch.randn(1, 4, 1, _SEQ_LEN)


MENAGERIE_ENTRIES = [
    (
        "ExPecto-Beluga",
        build_expecto_beluga,
        example_input_expecto_beluga,
        2018,
        "SOURCE_AVAILABLE",
    ),
]
