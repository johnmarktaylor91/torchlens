# SOURCE: vendored from rsinghlab/Hi-CY @ bcc65d7 (src/methods/HiCNN.py)
#
# HiCNN: a single deep convolutional neural network for enhancing Hi-C data resolution
# (Liu & Wang, Genes 2019, "HiCNN: A Very Deep Convolutional Neural Network to Better
# Enhance the Resolution of Hi-C Data"; original release at http://dna.cs.miami.edu/HiCNN/).
# A 13x13 entry conv projects a 40x40 low-resolution Hi-C submatrix down to a single channel,
# a 3x3 conv expands to 128 channels, then a residual tower applies a shared pair of 3x3 convs
# (`conv4R`) 25 times with additive skip connections back to the post-expansion features
# (giving HiCNN its very-deep character), before a final 3x3 conv + residual-add back to the
# post-entry features produces the 28x28 super-resolved output. Copied verbatim from the real
# repo's HiCNN class (the module-level `hicnn`/`upscale` helpers, which load pretrained weights
# from disk and run inference, are orchestration code, not part of the architecture, and are not
# vendored here).
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class HiCNN(nn.Module):
    def __init__(self):
        super(HiCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 13)
        self.conv2 = nn.Conv2d(8, 1, 1)
        self.conv3 = nn.Conv2d(1, 128, 3, padding=1, bias=False)
        self.conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(128, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        residual = x
        x2 = self.conv3(x)
        out = x2
        for _ in range(25):
            out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
            out = torch.add(out, x2)

        out = self.conv5(self.relu(out))
        out = torch.add(out, residual)
        return out


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
def build_hicnn():
    model = HiCNN()
    model.eval()
    return model


def example_input_hicnn():
    # (batch, 1, 40, 40) low-resolution Hi-C submatrix -> (batch, 1, 28, 28) super-resolved out.
    return torch.randn(2, 1, 40, 40)


MENAGERIE_ENTRIES = [
    (
        "HiCNN",
        build_hicnn,
        example_input_hicnn,
        2019,
        MENAGERIE_ZOO,
    ),
]
