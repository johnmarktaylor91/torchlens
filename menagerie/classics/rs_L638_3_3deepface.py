# SOURCE: vendored from DeepRank/deeprank @ master
# File: paper_pretrained_models/3DeepFace/arch_001_02.py (`cnn_class`) -- the real, published
# pretrained-model architecture (ships with best_model.pt weights in the same directory).
# Only minimal changes: no architectural changes; `input_shape` is fixed to a representative
# 5-channel 10x10x10 voxel grid for a tiny random-init trace (see example_input below).
# 3D CNN scorer for protein-protein docking-pose classification/ranking. The real caller
# (test.py) builds a `deeprank.learn.DataGenerator` that computes atomic + PSSM + BSA +
# residue-density features over a 3D grid (grid_info: number_of_points=[10,10,10],
# resolution=[3.,3.,3.]) around a docked protein-protein interface, then constructs
# `NeuralNet(database, cnn3d_class, pretrained_model='best_model.pt')`. The channel count of
# `input_shape[0]` is determined at runtime by which features are computed
# (deeprank.features.{AtomicFeature,FullPSSM,PSSM_IC,BSA,ResidueDensity}); 5 is a
# representative channel count for that feature configuration.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class cnn_class(nn.Module):
    def __init__(self, input_shape):
        super(cnn_class, self).__init__()

        self.convlayer_000 = nn.Conv3d(input_shape[0], 80, kernel_size=2)
        self.convlayer_001 = nn.MaxPool3d((2, 2, 2))
        self.convlayer_002 = nn.Conv3d(80, 120, kernel_size=2)
        self.convlayer_003 = nn.MaxPool3d((2, 2, 2))

        size = self._get_conv_output(input_shape)

        self.fclayer_000 = nn.Linear(size, 120)
        self.fclayer_001 = nn.Linear(120, 2)

    def _get_conv_output(self, shape):
        inp = torch.rand(1, *shape)
        out = self._forward_features(inp)
        return out.data.view(1, -1).size(1)

    def _forward_features(self, x):
        x = F.relu(self.convlayer_000(x))
        x = self.convlayer_001(x)
        x = F.relu(self.convlayer_002(x))
        x = self.convlayer_003(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fclayer_000(x))
        x = self.fclayer_001(x)
        return x


def build_3deepface():
    # 5 channels representative of the paper's atomic-density + PSSM feature stack;
    # real channel count is data-dependent on the feature set computed at grid-mapping time.
    return cnn_class((5, 10, 10, 10))


def example_input_3deepface():
    torch.manual_seed(0)
    # [batch, 5 feature channels, 10, 10, 10] voxel grid, matching grid_info
    # number_of_points=[10,10,10] from the real test.py verbatim.
    return (torch.randn(1, 5, 10, 10, 10),)


MENAGERIE_ENTRIES = [
    ("3DeepFace", build_3deepface, example_input_3deepface, 2019, MENAGERIE_ZOO),
]
