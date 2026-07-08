# SOURCE: vendored from kundajelab/ChromDragoNN @ master
# (model_zoo/stage1/resnet.py + model_zoo/stage2/complex.py)
"""
ChromDragoNN: two-tower CNN for predicting cell-type-specific chromatin
accessibility (Basset-ResNet DNA-sequence tower + RNA-seq gene-expression
tower fused with the frozen stage-1 conv features). Vendored verbatim from
the real ChromDragoNN model_zoo files; only the CLI-arg plumbing (argparse /
sys.path hacks / the original file's __main__ block) is replaced with a
plain namespace object carrying the same default hyperparameters used by
`utils/fetch_global_args.py::stage1_global_argparser` /
`stage2_global_argparser`, so the real `Net` classes can be built directly.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Stage 1: model_zoo/stage1/resnet.py (Basset-ResNet sequence tower)
# ---------------------------------------------------------------------------
class L1Block(nn.Module):
    def __init__(self):
        super(L1Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.layer = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2
        )

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L2Block(nn.Module):
    def __init__(self):
        super(L2Block, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2
        )

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L3Block(nn.Module):
    def __init__(self):
        super(L3Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(200)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)

        self.layer = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.conv3,
            self.bn3,
        )

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L4Block(nn.Module):
    def __init__(self):
        super(L4Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn2 = nn.BatchNorm2d(200)
        self.layer = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2
        )

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class Stage1Net(nn.Module):
    """`Net` from model_zoo/stage1/resnet.py (renamed to avoid clashing
    with Stage2Net below; the real class name in the repo is also `Net`)."""

    def __init__(self, args):
        super(Stage1Net, self).__init__()

        self.dropout = args.dropout
        self.num_cell_types = (
            args.num_total_cell_types - len(args.validation_list) - len(args.test_list)
        )

        self.conv1 = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.prelayer = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(*[L1Block() for _ in range(args.blocks[0])])
        self.layer2 = nn.Sequential(*[L2Block() for _ in range(args.blocks[1])])
        self.layer3 = nn.Sequential(*[L3Block() for _ in range(args.blocks[2])])
        self.layer4 = nn.Sequential(*[L4Block() for _ in range(args.blocks[3])])

        self.c1to2 = nn.Conv2d(64, 128, (3, 1), stride=(1, 1), padding=(1, 0))
        self.b1to2 = nn.BatchNorm2d(128)
        self.l1tol2 = nn.Sequential(self.c1to2, self.b1to2, nn.ReLU(inplace=True))

        self.c2to3 = nn.Conv2d(128, 200, (1, 1), padding=(3, 0))
        self.b2to3 = nn.BatchNorm2d(200)
        self.l2tol3 = nn.Sequential(self.c2to3, self.b2to3, nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, self.num_cell_types)
        self.flayer = self.final_layer()

    def final_layer(self):
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)  # batch_size x 4 x 1000 x 1 [4 channels]

        out = self.prelayer(s)
        out = self.layer1(out)
        out = self.layer2(self.l1tol2(out))
        out = self.maxpool1(out)
        out = self.layer3(self.l2tol3(out))
        out = self.maxpool2(out)
        out = self.layer4(out)
        out = self.flayer(out)
        out = self.maxpool3(out)
        out = out.view(-1, 4200)
        conv_out = out
        out = F.dropout(F.relu(self.bn4(self.fc1(out))), p=self.dropout, training=self.training)
        out = F.dropout(F.relu(self.bn5(self.fc2(out))), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out, conv_out


# ---------------------------------------------------------------------------
# Stage 2: model_zoo/stage2/complex.py (fusion tower: sequence conv features
# + RNA-seq gene-expression features -> binary accessibility prediction)
# ---------------------------------------------------------------------------
class Stage2Net(nn.Module):
    """`Net` from model_zoo/stage2/complex.py."""

    def __init__(self, BASSET_NUM_CELL_TYPES, basset_model, args):
        super(Stage2Net, self).__init__()
        self.basset_model = basset_model
        if args.freeze_pretrained_model:
            for param in self.basset_model.parameters():
                param.requires_grad = False
            self.basset_model.eval()

        self.layer1 = nn.Linear(4200 + args.num_genes, 1000)
        self.bn1 = nn.BatchNorm1d(1000)

        if args.with_mean:
            self.layer2 = nn.Linear(1000 + BASSET_NUM_CELL_TYPES + 2, 100)
        else:
            self.layer2 = nn.Linear(1000 + BASSET_NUM_CELL_TYPES + 1, 100)

        self.bn2 = nn.BatchNorm1d(100)
        self.layer3 = nn.Linear(100, 2)
        self.args = args

    def forward(self, s, g, m=None):
        if self.args.freeze_pretrained_model:
            self.basset_model.eval()
        basset_out, conv_out = self.basset_model(
            s
        )  # batch_size x BASSET_NUM_CELL_TYPES, batch_size x 4200
        basset_out_mean = torch.mean(torch.sigmoid(basset_out), 1, True)  # batch_size x 1
        conv_gene = torch.cat([conv_out, g], dim=-1)
        out = F.dropout(
            F.relu(self.bn1(self.layer1(conv_gene))), p=self.args.dropout, training=self.training
        )

        if self.args.with_mean:
            out = torch.cat([out, basset_out, basset_out_mean, m.view(-1, 1)], dim=-1)
        else:
            out = torch.cat([out, basset_out, basset_out_mean], dim=-1)

        out = F.dropout(
            F.relu(self.bn2(self.layer2(out))), p=self.args.dropout, training=self.training
        )
        out = self.layer3(out)
        return F.log_softmax(out, dim=-1)


class ChromDragoNN(nn.Module):
    """End-to-end wrapper chaining the real stage-1 sequence tower into the
    real stage-2 fusion tower, matching how the two stages are composed at
    inference time in the original repo (stage-1 frozen, feeding stage-2)."""

    def __init__(self, num_cell_types=8, num_genes=32, with_mean=False):
        super().__init__()
        stage1_args = types.SimpleNamespace(
            dropout=0.3,
            num_total_cell_types=num_cell_types,
            validation_list=[],
            test_list=[],
            blocks=[1, 1, 1, 1],
        )
        stage2_args = types.SimpleNamespace(
            dropout=0.3,
            freeze_pretrained_model=1,
            num_genes=num_genes,
            with_mean=1 if with_mean else 0,
        )
        self.basset_model = Stage1Net(stage1_args)
        self.net = Stage2Net(num_cell_types, self.basset_model, stage2_args)
        self.with_mean = with_mean

    def forward(self, seq, genes, mean_acc=None):
        if self.with_mean:
            return self.net(seq, genes, mean_acc)
        return self.net(seq, genes)


def build_chromdragonn():
    return ChromDragoNN(num_cell_types=8, num_genes=32, with_mean=False)


def example_input_chromdragonn():
    seq = torch.randn(2, 1000, 4)  # one-hot-ish DNA sequence window (batch, length, 4 bases)
    genes = torch.randn(2, 32)  # RNA-seq gene-expression feature vector
    return (seq, genes)


MENAGERIE_ENTRIES = [
    ("ChromDragoNN", build_chromdragonn, example_input_chromdragonn, 2019, "vendored-pytorch"),
]
