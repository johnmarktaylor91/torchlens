# SOURCE: vendored from HalbertCH/IEContraAST @ 50ee949f5302a7e4a3cae3226610c03462093c21
# https://raw.githubusercontent.com/HalbertCH/IEContraAST/50ee949f5302a7e4a3cae3226610c03462093c21/net.py
#
# Chen et al. 2021 (NeurIPS) "Artistic Style Transfer with Internal-external Learning
# and Contrastive Learning" -- a fixed VGG-19 encoder (`vgg`, truncated at relu5_1)
# feeding a two-stage `SANet` (style-attentional network: per-pixel content/style
# cross-attention via 1x1-conv Q/K/V + softmax + residual add) fused across relu4_1
# and relu5_1 features in `Transform` (`sanet4_1`/`sanet5_1` + nearest-upsample merge
# + a 3x3 merge conv), followed by a mirrored-VGG `decoder` back to RGB. This
# attention-based style/content transform (the "internal learning" AdaIN-successor
# SANet mechanism) plus the auxiliary `MultiDiscriminator` (multi-scale PatchGAN) and
# `proj_style`/`proj_content` contrastive-projection MLPs (the "external"
# contrastive-learning heads used only at train time to pull/push style and content
# embeddings) are the paper's architectural contribution over a plain
# encoder-AdaIN-decoder baseline.
#
# `Net` is the model exactly as defined upstream (constructor `(encoder, decoder,
# start_iter)`, unchanged). No architectural changes were made; only mechanical fixes
# for self-containment:
#   - The upstream `Net.__init__` unconditionally loads
#     `torch.load('transformer_iter_' + ...)` / `decoder_iter_...` checkpoint files
#     when `start_iter > 0`; `build_iecontraast()` below passes `start_iter=0` (a
#     config choice already supported by the original code, not a code change) so no
#     filesystem checkpoint is required.
#   - `Net.forward` computes GAN/contrastive/perceptual training losses and expects a
#     `batch_size` argument for its identity/contrastive splits; `example_input_*`
#     supplies an even `batch_size` (matching the paper's training setup) so the
#     `half = int(batch_size / 2)` contrastive-comparison indexing exercises the
#     real code path unmodified.
#   - `vgg`/`decoder` are the real truncated-VGG19 `nn.Sequential` definitions from
#     the source file; `Net.__init__` slices `encoder.children()` into
#     `enc_1`..`enc_5` exactly as upstream (`enc_layers[:4]`, `[4:11]`, `[11:18]`,
#     `[18:31]`, `[31:44]`), which requires the untruncated 44-layer `vgg` Sequential
#     (copied verbatim from the source, including the unused relu5-1..5-4 tail so the
#     slice indices line up with the original module).

import torch
import torch.nn as nn


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


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert feat.size()[0] == 3
    assert isinstance(feat, torch.FloatTensor)
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
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

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128),
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1),
                ),
            )

        self.downsample = nn.AvgPool2d(
            in_channels, stride=2, padding=[1, 1], count_include_pad=False
        )

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))  # noqa: E741
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)  # noqa: E741
        O = self.out_conv(O)  # noqa: E741
        O += content  # noqa: E741
        return O


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        # self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        self.upsample5_1 = nn.Upsample(
            size=(content4_1.size()[2], content4_1.size()[3]), mode="nearest"
        )
        return self.merge_conv(
            self.merge_conv_pad(
                self.sanet4_1(content4_1, style4_1)
                + self.upsample5_1(self.sanet5_1(content5_1, style5_1))
            )
        )


class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        # projection
        self.proj_style = projection_style
        self.proj_content = projection_content

        # transform
        self.transform = Transform(in_planes=512)
        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if start_iter > 0:
            self.transform.load_state_dict(
                torch.load("transformer_iter_" + str(start_iter) + ".pth")
            )
            self.decoder.load_state_dict(torch.load("decoder_iter_" + str(start_iter) + ".pth"))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4", "enc_5"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        if norm == False:  # noqa: E712
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        # loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(
            out, torch.tensor([index], dtype=torch.long, device=feat_q.device)
        )
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        # out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def forward(self, content, style, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(
            content_feats[3], style_feats[3], content_feats[4], style_feats[4]
        )
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(
            g_t_feats[3], content_feats[3], norm=True
        ) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(
            self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4])
        )
        Iss = self.decoder(
            self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4])
        )
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(
            Fss[0], style_feats[0]
        )
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(
                Fcc[i], content_feats[i]
            ) + self.calc_content_loss(Fss[i], style_feats[i])

        # Contrastive learning.
        half = int(batch_size / 2)
        style_up = self.style_feature_contrastive(g_t_feats[2][0:half])
        style_down = self.style_feature_contrastive(g_t_feats[2][half:])
        content_up = self.content_feature_contrastive(g_t_feats[3][0:half])
        content_down = self.content_feature_contrastive(g_t_feats[3][half:])

        style_contrastive_loss = 0
        for i in range(half):
            reference_style = style_up[i : i + 1]

            if i == 0:
                style_comparisons = torch.cat([style_down[0 : half - 1], style_up[1:]], 0)
            elif i == 1:
                style_comparisons = torch.cat([style_down[1:], style_up[0:1], style_up[2:]], 0)
            elif i == (half - 1):
                style_comparisons = torch.cat(
                    [style_down[half - 1 :], style_down[0 : half - 2], style_up[0 : half - 1]], 0
                )
            else:
                style_comparisons = torch.cat(
                    [style_down[i:], style_down[0 : i - 1], style_up[0:i], style_up[i + 1 :]], 0
                )

            style_contrastive_loss += self.compute_contrastive_loss(
                reference_style, style_comparisons, 0.2, 0
            )

        for i in range(half):
            reference_style = style_down[i : i + 1]

            if i == 0:
                style_comparisons = torch.cat([style_up[0:1], style_up[2:], style_down[1:]], 0)
            elif i == (half - 2):
                style_comparisons = torch.cat(
                    [
                        style_up[half - 2 : half - 1],
                        style_up[0 : half - 2],
                        style_down[0 : half - 2],
                        style_down[half - 1 :],
                    ],
                    0,
                )
            elif i == (half - 1):
                style_comparisons = torch.cat(
                    [style_up[half - 1 :], style_up[1 : half - 1], style_down[0 : half - 1]], 0
                )
            else:
                style_comparisons = torch.cat(
                    [
                        style_up[i : i + 1],
                        style_up[0:i],
                        style_up[i + 2 :],
                        style_down[0:i],
                        style_down[i + 1 :],
                    ],
                    0,
                )

            style_contrastive_loss += self.compute_contrastive_loss(
                reference_style, style_comparisons, 0.2, 0
            )

        content_contrastive_loss = 0
        for i in range(half):
            reference_content = content_up[i : i + 1]

            if i == 0:
                content_comparisons = torch.cat(
                    [content_down[half - 1 :], content_down[1 : half - 1], content_up[1:]], 0
                )
            elif i == 1:
                content_comparisons = torch.cat(
                    [content_down[0:1], content_down[2:], content_up[0:1], content_up[2:]], 0
                )
            elif i == (half - 1):
                content_comparisons = torch.cat(
                    [
                        content_down[half - 2 : half - 1],
                        content_down[0 : half - 2],
                        content_up[0 : half - 1],
                    ],
                    0,
                )
            else:
                content_comparisons = torch.cat(
                    [
                        content_down[i - 1 : i],
                        content_down[0 : i - 1],
                        content_down[i + 1 :],
                        content_up[0:i],
                        content_up[i + 1 :],
                    ],
                    0,
                )

            content_contrastive_loss += self.compute_contrastive_loss(
                reference_content, content_comparisons, 0.2, 0
            )

        for i in range(half):
            reference_content = content_down[i : i + 1]

            if i == 0:
                content_comparisons = torch.cat([content_up[1:], content_down[1:]], 0)
            elif i == (half - 2):
                content_comparisons = torch.cat(
                    [
                        content_up[half - 1 :],
                        content_up[0 : half - 2],
                        content_down[0 : half - 2],
                        content_down[half - 1 :],
                    ],
                    0,
                )
            elif i == (half - 1):
                content_comparisons = torch.cat(
                    [content_up[0 : half - 1], content_down[0 : half - 1]], 0
                )
            else:
                content_comparisons = torch.cat(
                    [
                        content_up[i + 1 : i + 2],
                        content_up[0:i],
                        content_up[i + 2 :],
                        content_down[0:i],
                        content_down[i + 1 :],
                    ],
                    0,
                )

            content_contrastive_loss += self.compute_contrastive_loss(
                reference_content, content_comparisons, 0.2, 0
            )

        return (
            g_t,
            loss_c,
            loss_s,
            l_identity1,
            l_identity2,
            content_contrastive_loss,
            style_contrastive_loss,
        )


def build_iecontraast():
    net = Net(vgg, decoder, start_iter=0)
    net.eval()
    return net


def example_input_iecontraast():
    # (content, style, batch_size); batch_size must be even for the contrastive split.
    batch_size = 4
    content = torch.randn(batch_size, 3, 64, 64)
    style = torch.randn(batch_size, 3, 64, 64)
    return (content, style, batch_size)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("IEContraAST", "build_iecontraast", "example_input_iecontraast", 2021, "vendored"),
]
