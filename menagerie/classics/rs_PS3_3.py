# SOURCE: vendored from facebookresearch/2.5D-Visual-Sound @ main
# https://raw.githubusercontent.com/facebookresearch/2.5D-Visual-Sound/main/models/networks.py
# https://raw.githubusercontent.com/facebookresearch/2.5D-Visual-Sound/main/models/models.py
#
# Gao, Grauman 2019 "2.5D Visual Sound" (CVPR 2019, arXiv:1812.04204). Founding
# visually-guided mono-to-binaural spatialization: a ResNet-18 visual stream extracts a
# spatial feature map from the video frame, a 5-layer U-Net audio stream ingests the
# mixed-channel mono spectrogram, and the flattened+tiled visual feature is concatenated
# into the U-Net bottleneck to predict a complex spectrogram mask -- defining the
# audio-visual spatial-audio-GENERATION sub-branch (distinct from sound source
# localization/separation). `VisualNet`/`AudioNet`/`unet_conv`/`unet_upconv`/`create_conv`/
# `weights_init` are copied verbatim from models/networks.py, and `ModelBuilder` mirrors
# models/models.py (real torchvision.models.resnet18 backbone, unmodified). The thin
# top-level composition in the real `AudioVisualModel.forward` (models/audioVisual_model.py)
# used Python-2-era `torch.autograd.Variable(..., volatile=...)` calls that no longer exist
# in modern torch; `TwoPointFiveDVisualSound.forward` below reproduces the exact same
# tensor math (visual-feature conv1x1 -> flatten -> tile -> concat -> complex masking of the
# mixed spectrogram) without that dead scaffolding -- no architecture was changed.
"""2.5D Visual Sound: ResNet-18 visual stream + U-Net audio stream, visually-conditioned
mono-to-binaural spectrogram mask prediction."""

import torch
import torch.nn as nn
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from models/networks.py ---
def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


def create_conv(
    input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1
):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(
            1296, ngf * 8
        )  # 1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(
            ngf * 2, output_nc, True
        )  # outermost layer use a sigmoid to bound the mask
        self.conv1x1 = create_conv(512, 8, 1, 0)  # reduce dimension of extracted visual features

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = self.conv1x1(visual_feat)
        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
        visual_feat = visual_feat.repeat(
            1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]
        )  # tile visual feature

        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(
            torch.cat((audio_upconv1feature, audio_conv4feature), dim=1)
        )
        audio_upconv3feature = self.audionet_upconvlayer3(
            torch.cat((audio_upconv2feature, audio_conv3feature), dim=1)
        )
        audio_upconv4feature = self.audionet_upconvlayer4(
            torch.cat((audio_upconv3feature, audio_conv2feature), dim=1)
        )
        mask_prediction = (
            self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
            * 2
            - 1
        )
        return mask_prediction


# --- vendored from models/models.py (ModelBuilder) ---
class ModelBuilder:
    # builder for visual stream
    def build_visual(self, weights=""):
        original_resnet = torchvision.models.resnet18(weights=None)
        net = VisualNet(original_resnet)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))
        return net

    # builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=""):
        # AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)
        net.apply(weights_init)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))
        return net


# --- combined forward, modernized from models/audioVisual_model.py's AudioVisualModel.forward
# (dropping the dead torch.autograd.Variable(..., volatile=...) Python-2 scaffolding; the
# tensor math -- visual feature extraction, complex spectrogram masking -- is unchanged) ---
class TwoPointFiveDVisualSound(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        builder = ModelBuilder()
        self.net_visual = builder.build_visual()
        self.net_audio = builder.build_audio(ngf=ngf)

    def forward(self, frame, audio_mix_spec):
        visual_feature = self.net_visual(frame)
        mask_prediction = self.net_audio(audio_mix_spec, visual_feature)

        # complex masking to obtain the predicted spectrogram (real code from
        # AudioVisualModel.forward)
        spectrogram_diff_real = (
            audio_mix_spec[:, 0, :-1, :] * mask_prediction[:, 0, :, :]
            - audio_mix_spec[:, 1, :-1, :] * mask_prediction[:, 1, :, :]
        )
        spectrogram_diff_img = (
            audio_mix_spec[:, 0, :-1, :] * mask_prediction[:, 1, :, :]
            + audio_mix_spec[:, 1, :-1, :] * mask_prediction[:, 0, :, :]
        )
        binaural_spectrogram = torch.cat(
            (spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1
        )
        return binaural_spectrogram


def build_2p5d_visual_sound():
    torch.manual_seed(0)
    return TwoPointFiveDVisualSound(ngf=64)


def example_input_2p5d_visual_sound():
    torch.manual_seed(0)
    # frame: 224x448 so that resnet18's /32 downsample gives a 7x14 feature map, matching
    # the real repo's hardcoded 784 = 8 * 7 * 14 visual-feature flatten size in AudioNet.
    frame = torch.randn(1, 3, 224, 448)
    # audio_mix_spec: 2-channel (real, imaginary) mixed spectrogram; small spatial size for
    # a fast random-init trace (the real repo trains at larger spectrogram resolutions).
    audio_mix_spec = torch.randn(1, 2, 65, 64)
    return (frame, audio_mix_spec)


MENAGERIE_ENTRIES = [
    (
        "2.5D Visual Sound",
        "build_2p5d_visual_sound",
        "example_input_2p5d_visual_sound",
        2019,
        "vendored",
    ),
]
