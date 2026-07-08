# SOURCE: vendored from Huage001/AdaAttN @ main (models/networks.py)
from __future__ import annotations

import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class AdaAttN(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[: self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat**2) - mean**2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):
    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(
            in_planes=in_planes, key_planes=key_planes + 512 if shallow_layer else key_planes
        )
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(
        self,
        content4_1,
        style4_1,
        content5_1,
        style5_1,
        content4_1_key,
        style4_1_key,
        content5_1_key,
        style5_1_key,
        seed=None,
    ):
        return self.merge_conv(
            self.merge_conv_pad(
                self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
                + self.upsample5_1(
                    self.attn_adain_5_1(
                        content5_1, style5_1, content5_1_key, style5_1_key, seed=seed
                    )
                )
            )
        )


class Decoder(nn.Module):
    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs


def build_adaattn() -> AdaAttN:
    """Build a tiny AdaAttN module for TorchLens tracing."""
    model = AdaAttN(in_planes=8, max_sample=64, key_planes=4)
    model.eval()
    return model


def example_input_adaattn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return content/style features and corresponding attention keys."""
    content = torch.randn(1, 8, 4, 4)
    style = torch.randn(1, 8, 4, 4)
    content_key = torch.randn(1, 4, 4, 4)
    style_key = torch.randn(1, 4, 4, 4)
    return content, style, content_key, style_key


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [("AdaAttN", "build_adaattn", "example_input_adaattn", 2021, "CV1")]
