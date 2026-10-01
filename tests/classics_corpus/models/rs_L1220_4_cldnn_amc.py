# SOURCE: vendored from Singingkettle/ChangShuoRadioRecognition @ main
# https://raw.githubusercontent.com/Singingkettle/ChangShuoRadioRecognition/main/csrr/models/backbones/cldnn.py
#
# CLDNN for Automatic Modulation Classification (AMC): a CNN+LSTM+DNN backbone that
# classifies raw I/Q radio frames -- stacked Conv2d feature extraction over the
# (1, 2, L) I/Q frame, fed into an LSTM, then an MLP classifier head. The official
# ChangShuoRadioRecognition (CSRR) framework provides two real variants, CLDNNL and
# CLDNNW (named after the first author of the two papers they reference); both
# forward()/__init__() bodies below are copied verbatim from csrr/models/backbones/cldnn.py.
# The only change: CSRR wraps every backbone in `BaseBackbone(BaseModule,
# metaclass=ABCMeta)` from the mmengine registry framework (`mmengine.model.BaseModule`)
# plus an `@BACKBONES.register_module()` decorator -- neither contributes any
# architecture, they only add mmengine's config-driven weight-init/registry
# bookkeeping. Both classes are rebased onto plain `nn.Module` here (dropping the
# unused `init_cfg`/registry plumbing) with the Conv2d/Dropout/ReLU/LSTM/Linear
# layers, `forward()` control flow, and classifier head structure otherwise
# unmodified.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class CLDNNL(nn.Module):
    """`CLDNNL <https://ieeexplore.ieee.org/abstract/document/8335483>`_ backbone
    Actually, the details of neural network structure is not provided in the paper.
    To deal with that, we have referred to the code OF CLDNN2 in the AMR-Benchmark.
    Basically, there are two versions of CLDNN. In order to identify them, we add an
    extra letter after CLDNN, which is borrowed from the first name of the first author.
    The input for CNN1 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1):
        super(CLDNNL, self).__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=(2, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 80, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(80, 80, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(input_size=self.frame_length - 8, hidden_size=50, batch_first=True)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.reshape(x, [-1, 80, self.frame_length - 8])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)


class CLDNNW(nn.Module):
    """`CLDNNW <https://ieeexplore.ieee.org/abstract/document/7920754>`_ backbone
    Actually, the details of neural network structure is not provided in the paper.
    To deal with that, we have referred to the code of CLDNN in the AMR-Benchmark.
    Basically, there are two versions of CLDNN. In order to identify them, we add an
    extra letter after CLDNN, which is borrowed from the first name of the first author.
    The input for CNN1 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1):
        super(CLDNNW, self).__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        # Compared to AMR-Benchmark, we remove the Padding layer.
        # Basically, the padding layer is mainly used to keep the size same before and after (such as) conv.
        # However, in this CLDNN, the padding layer cannot keep the same, and it introduces some useless information '0'
        # As a result, we remove the padding layers.
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(
            input_size=(self.frame_length * 2 - 7 * 4) * 2, hidden_size=50, batch_first=True
        )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.concatenate((x1, x2), dim=3)
        x = torch.reshape(x, [-1, 50, (self.frame_length * 2 - 7 * 4) * 2])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)


def build_cldnn_amc():
    # Real constructor args from the repo's cldnnw configs (e.g.
    # configs/cldnnw/cldnnw_iq-deepsig-201610A.py): frame_length=128,
    # num_classes=11 (RadioML 2016.10A has 11 modulation classes).
    return CLDNNW(frame_length=128, num_classes=11)


def example_input_cldnn_amc():
    # Raw I/Q frame, shape (batch, 1, 2, frame_length) matching the module's
    # "1*2*L frame" convention documented in the source docstring.
    torch.manual_seed(0)
    return (torch.randn(2, 1, 2, 128),)


MENAGERIE_ENTRIES = [
    (
        "CLDNN (Automatic Modulation Classification)",
        "build_cldnn_amc",
        "example_input_cldnn_amc",
        2018,
        "vendored-pytorch",
    ),
]
