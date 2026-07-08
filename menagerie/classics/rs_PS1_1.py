# SOURCE: vendored from Nablax/ACLnet-Pytorch @ master
# (https://github.com/Nablax/ACLnet-Pytorch/blob/master/models_pt.py,
#  https://github.com/Nablax/ACLnet-Pytorch/blob/master/ConvLSTM.py)
#
# The queue's listed repo (wenguanwang/DHF1K) is the *dataset* repo for Wang et al.'s
# original ACLNet paper and contains no PyTorch model code (only README/PDFs/MATLAB
# scripts/xlsx). The queue's other lead, DaveKentucky/VideoSaliency, is a DIFFERENT
# model (an S3D-encoder video saliency net inspired by ViNet/TASED-Net) whose own
# README explicitly cites this repo (Nablax/ACLnet-Pytorch) as "a more modern version
# of Wang's et al. ACLNet" -- i.e. this is the real, runnable ACLNet PyTorch source.
#
# Vendored verbatim except: (1) `from config import *` replaced by the single
# `num_frames = 5` constant it needed (config.py is otherwise Windows training paths,
# not architecture); (2) hardcoded `.cuda()` calls on the zero-init LSTM gate biases
# and on the ConvLSTM submodule/its input removed so the real architecture runs on
# CPU (device placement only, no architectural change).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

num_frames = 5


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1])
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            torch.zeros(batch_size, hidden, shape[0], shape[1]),
            torch.zeros(batch_size, hidden, shape[0], shape[1]),
        )


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super().__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = "cell{}".format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = "cell{}".format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width)
                    )
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


class acl_net_pt(nn.Module):
    def __init__(self):
        super().__init__()
        self.dcn_vgg()
        self.acl_vgg()

    def forward(self, x):
        out = self.dcn_vgg_fwd(x)
        outs = self.acl_vgg_fwd(out)
        return outs

    def dcn_vgg(self):
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.padding_one_side = nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0)
        self.maxpool4 = nn.MaxPool2d((2, 2), stride=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))

    def dcn_vgg_fwd(self, x):
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = self.maxpool1(out)

        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.maxpool3(out)

        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.padding_one_side(out)
        out = self.maxpool4(out)

        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        return out

    def acl_vgg(self):
        self.maxpool_atn1_1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_1 = nn.Conv2d(512, 64, 1)
        self.conv_atn1_2 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.maxpool_atn1_2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_3 = nn.Conv2d(128, 64, 1)
        self.conv_atn1_4 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv_atn1_5 = nn.Conv2d(128, 1, 1)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=4)

        self.convLSTM = ConvLSTM(
            input_channels=512,
            hidden_channels=[256],
            kernel_size=3,
            step=num_frames,
            effective_step=[4],
        )
        self.conv_atn2_1 = nn.Conv2d(256, 1, 1)
        self.upsampling2_1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upsampling2_2 = nn.UpsamplingNearest2d(scale_factor=2)

    def acl_vgg_fwd(self, x):
        outs = x
        attention = self.maxpool_atn1_1(outs)
        attention = F.relu(self.conv_atn1_1(attention))
        attention = F.relu(self.conv_atn1_2(attention))
        attention = self.maxpool_atn1_2(attention)
        attention = F.relu(self.conv_atn1_3(attention))
        attention = F.relu(self.conv_atn1_4(attention))
        attention = torch.sigmoid(self.conv_atn1_5(attention))
        attention = self.upsampling1(attention)

        f_attention = attention.repeat(1, 512, 1, 1)
        m_outs = f_attention * outs
        outs = outs + m_outs

        outs = self.convLSTM(outs)[0][0]
        outs = torch.sigmoid(self.conv_atn2_1(outs))
        outs = self.upsampling2_1(outs)
        return outs


def build_aclnet():
    return acl_net_pt()


def example_input_aclnet():
    # single video frame, NCHW; H,W must be divisible by 32 for the attention
    # sub-branch's two extra 2x maxpools + 4x upsample to line back up with the
    # /8-resolution VGG feature map.
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("ACLNet", "build_aclnet", "example_input_aclnet", 2018, MENAGERIE_ZOO),
]
