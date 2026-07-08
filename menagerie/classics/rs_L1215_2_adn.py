# SOURCE: vendored from liaohaofu/adn @ master
# https://github.com/liaohaofu/adn/blob/master/adn/networks/adn.py
# https://github.com/liaohaofu/adn/blob/master/adn/networks/blocks.py
# ADN (Artifact Disentanglement Network, MICCAI 2019 + IEEE TMI 2019): dual encoders
# (low-quality/artifact-containing, high-quality/artifact-free) plus a shared artifact
# encoder disentangle a CT image into content and artifact codes, decoded back through
# a shared/duplicated decoder with side-feature fusion. Transcribed verbatim from the
# real `Encoder`/`Decoder`/`ADN` classes (adn.py) and their `ConvolutionBlock`/
# `ResidualBlock` building blocks (blocks.py); the `NLayerDiscriminator` (GAN head,
# itself adapted upstream from pytorch-CycleGAN-and-pix2pix) is included for
# completeness but not used by the traced entry point. Only change: the relative
# imports (`.blocks`, `..utils`) are inlined into this single file and the unused
# `print_model`/`FunctionModel` helper import is dropped (not referenced by any of
# the vendored classes).
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from copy import deepcopy

MENAGERIE_ZOO = "vendored-pytorch"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
    zero=nn.ZeroPad2d,
    reflect=nn.ReflectionPad2d,
    replicate=nn.ReplicationPad2d,
)

conv_dict = dict(
    conv2d=nn.Conv2d,
    deconv2d=nn.ConvTranspose2d,
)

norm_dict = dict(
    none=lambda x: lambda x: x,
    spectral=lambda x: lambda x: x,
    batch=nn.BatchNorm2d,
    instance=nn.InstanceNorm2d,
    layer=LayerNorm,
)

activ_dict = dict(
    none=lambda: lambda x: x,
    relu=lambda: nn.ReLU(inplace=True),
    lrelu=lambda: nn.LeakyReLU(0.2, inplace=True),
    prelu=lambda: nn.PReLU(),
    selu=lambda: nn.SELU(inplace=True),
    tanh=lambda: nn.Tanh(),
)


class ConvolutionBlock(nn.Module):
    def __init__(
        self, conv="conv2d", norm="instance", activ="relu", pad="reflect", padding=0, **conv_opts
    ):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts["out_channels"]
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral":
            self.conv = nn.utils.spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="instance", activ="relu", pad="reflect"):
        super(ResidualBlock, self).__init__()

        block = []
        block += [
            ConvolutionBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activ=activ,
                pad=pad,
            )
        ]
        block += [
            ConvolutionBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activ="none",
                pad=pad,
            )
        ]
        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x) + x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm="none", activ="relu"):
        super(FullyConnectedBlock, self).__init__()

        self.fc = nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral":
            self.fc = nn.utils.spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.fc(x)))


class Encoder(nn.Module):
    def __init__(
        self, input_ch, base_ch, num_down, num_residual, res_norm="instance", down_norm="instance"
    ):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch,
            out_channels=base_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            pad="reflect",
            norm=down_norm,
            activ="relu",
        )

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch,
                out_channels=output_ch * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                pad="reflect",
                norm=down_norm,
                activ="relu",
            )
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(
                self,
                "res{}".format(i),
                ResidualBlock(output_ch, pad="reflect", norm=res_norm, activ="relu"),
            )

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + [
            getattr(self, "res{}".format(i)) for i in range(num_residual)
        ]

    def forward(self, x):
        sides = []
        for layer in self.layers:
            x = layer(x)
            sides.append(x)
        return x, sides[::-1]


class Decoder(nn.Module):
    def __init__(
        self,
        output_ch,
        base_ch,
        num_up,
        num_residual,
        num_sides,
        res_norm="instance",
        up_norm="layer",
        fuse=False,
    ):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2**num_up
        input_chs = []

        for i in range(num_residual):
            setattr(
                self,
                "res{}".format(i),
                ResidualBlock(input_ch, pad="reflect", norm=res_norm, activ="lrelu"),
            )
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch,
                    out_channels=input_ch // 2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    pad="reflect",
                    norm=up_norm,
                    activ="lrelu",
                ),
            )
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch,
            out_channels=output_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            pad="reflect",
            norm="none",
            activ="tanh",
        )
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + [
            getattr(self, "conv{}".format(i)) for i in range(num_up + 1)
        ]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i), nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides=[]):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            x = self.layers[i](x)

        for i, j in enumerate(range(m - n, m)):
            x = self.fuse(x, sides[i], i)
            x = self.layers[j](x)
        return x


class ADN(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(
        self,
        input_ch=1,
        base_ch=64,
        num_down=2,
        num_residual=4,
        num_sides="all",
        res_norm="instance",
        down_norm="instance",
        up_norm="layer",
        fuse=True,
        shared_decoder=False,
    ):
        super(ADN, self).__init__()

        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_low = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_high = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_art = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.decoder = Decoder(
            input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse
        )
        self.decoder_art = self.decoder if shared_decoder else deepcopy(self.decoder)

    def forward1(self, x_low):
        _, sides = self.encoder_art(x_low)  # encode artifact
        self.saved = (x_low, sides)
        code, _ = self.encoder_low(x_low)  # encode low quality image
        y1 = self.decoder_art(code, sides[-self.n :])  # decode image with artifact (low quality)
        y2 = self.decoder(code)  # decode image without artifact (high quality)
        return y1, y2

    def forward2(self, x_low, x_high):
        if hasattr(self, "saved") and self.saved[0] is x_low:
            sides = self.saved[1]
        else:
            _, sides = self.encoder_art(x_low)  # encode artifact

        code, _ = self.encoder_high(x_high)  # encode high quality image
        y1 = self.decoder_art(code, sides[-self.n :])  # decode image with artifact (low quality)
        y2 = self.decoder(code)  # decode without artifact (high quality)
        return y1, y2

    def forward_lh(self, x_low):
        code, _ = self.encoder_low(x_low)  # encode low quality image
        y = self.decoder(code)
        return y

    def forward_hl(self, x_low, x_high):
        _, sides = self.encoder_art(x_low)  # encode artifact
        code, _ = self.encoder_high(x_high)  # encode high quality image
        y = self.decoder_art(code, sides[-self.n :])  # decode image with artifact (low quality)
        return y

    def forward(self, x_low):
        # menagerie entry point: mirrors forward1, the network's primary
        # disentangle-and-reconstruct pass (single low-quality image in).
        return self.forward1(x_low)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) is str:
            norm_layer = {
                "layer": nn.LayerNorm,
                "instance": nn.InstanceNorm2d,
                "batch": nn.BatchNorm2d,
                "none": None,
            }[norm_layer]

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += (
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                ]
                + ([norm_layer(ndf * nf_mult)] if norm_layer else [])
                + [nn.LeakyReLU(0.2, True)]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += (
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                )
            ]
            + ([norm_layer(ndf * nf_mult)] if norm_layer else [])
            + [nn.LeakyReLU(0.2, True)]
        )

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def build_adn():
    # Tiny menagerie-scale config: source defaults (base_ch=64, num_down=2,
    # num_residual=4) shrunk for fast tracing; input_ch=1 matches the source's
    # single-channel CT slice convention.
    return ADN(
        input_ch=1,
        base_ch=8,
        num_down=2,
        num_residual=2,
        num_sides="all",
        fuse=True,
        shared_decoder=False,
    )


def example_input_adn():
    # Spatial dims must be divisible by 2**num_down (=4 here).
    torch.manual_seed(0)
    return (torch.randn(1, 1, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "ADN",
        "build_adn",
        "example_input_adn",
        2019,
        "vendored-pytorch",
    ),
]
