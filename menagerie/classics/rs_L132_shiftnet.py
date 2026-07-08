# SOURCE: vendored from Zhaoyi-Yan/Shift-Net_pytorch @ master
# (models/modules/shift_unet.py, models/modules/unet.py, models/modules/modules.py,
#  models/shift_net/InnerShiftTriple.py, models/shift_net/InnerShiftTripleFunction.py,
#  models/shift_net/InnerCos.py, models/shift_net/InnerCosFunction.py,
#  util/NonparametricShift.py, util/util.py)
"""Shift-Net: Image Inpainting via Deep Feature Rearrangement.

Yan, Zhaoyi, et al. ECCV 2018. Official PyTorch implementation confirmed. Encoder-decoder
U-Net image inpainter whose novel contribution is a differentiable "shift-connection" layer
(`InnerShiftTriple`, backed by a custom `torch.autograd.Function`) inserted into one U-Net
skip connection: at that resolution, for every masked-region feature location the layer finds
the most cosine-similar non-masked-region feature (via unfold-based patch matching) and copies
("shifts") it in, concatenating {decoder features, encoder features, shifted features} before
the next up-conv. A companion `InnerCos` layer adds a guidance loss target (identity in the
forward pass; the loss/backward hook is training-only) just before the shift layer.

This file vendors the real generator entry point `UnetGeneratorShiftTriple` (`which_model_netG
== 'unet_shift_triple'` in `models/networks.py:define_G`) plus every real module it composes:
`UnetSkipConnectionShiftBlock`, the plain `UnetSkipConnectionBlock` (used for the non-shift
skip levels), `spectral_norm` and the shift/cos autograd machinery. Only import paths were
flattened into one file (the original spreads these across `models/networks.py`,
`models/modules/{modules,unet,shift_unet}.py`, `models/shift_net/Inner{ShiftTriple,Cos}
{,Function}.py`, and `util/{util,NonparametricShift}.py`); the `InnerShiftTriple`/`InnerCos`
call sites are pinned to `device='cpu'` (the repo's own CPU branch, `_split_mask` is skipped)
so the layer traces on CPU exactly as the repo intends when no GPU is available. No
architecture was altered. The Face/Res/PatchSoft shift variants and the discriminator/losses
in the same source files are omitted (not part of the base Shift-Net generator).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/modules/modules.py (verbatim: spectral_norm helper used throughout)
# ---------------------------------------------------------------------------
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# ---------------------------------------------------------------------------
# util/NonparametricShift.py (verbatim: Batch_NonShift, the batched patch-matching
# helper used by InnerShiftTripleFunction)
# ---------------------------------------------------------------------------
class Batch_NonShift(object):
    def _extract_patches_from_flag(self, img, patch_size, stride, flag, value):
        input_windows = self._unfold(img, patch_size, stride)
        input_windows = self._filter(input_windows, flag, value)
        return self._norm(input_windows)

    # former: content, to be replaced.
    # latter: style, source pixels.
    def cosine_similarity(self, former, latter, patch_size, stride, flag, with_former=False):
        former_windows = self._unfold(former, patch_size, stride)
        former = self._filter(former_windows, flag, 1)

        latter_windows, i_2, i_3, i_1 = self._unfold(latter, patch_size, stride, with_indexes=True)
        latter = self._filter(latter_windows, flag, 0)

        num = torch.einsum("bik,bjk->bij", [former, latter])
        norm_latter = torch.einsum("bij,bij->bi", [latter, latter])
        norm_former = torch.einsum("bij,bij->bi", [former, former])
        den = torch.sqrt(torch.einsum("bi,bj->bij", [norm_former, norm_latter]))
        if not with_former:
            return num / den, latter_windows, i_2, i_3, i_1
        else:
            return num / den, latter_windows, former_windows, i_2, i_3, i_1

    # delete i_4, as i_4 is 1
    def _paste(self, input_windows, transition_matrix, i_2, i_3, i_1):
        bz = input_windows.size(0)
        input_windows = torch.bmm(transition_matrix, input_windows)
        input_windows = input_windows.view(bz, i_2, i_3, i_1)
        input_windows = input_windows.permute(0, 3, 1, 2)
        return input_windows

    def _unfold(self, img, patch_size, stride, with_indexes=False):
        n_dim = 4
        assert img.dim() == n_dim, "image must be of dimension 4."

        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(2, kH, dH).unfold(3, kW, dW)

        i_0, i_1, i_2, i_3, i_4, i_5 = input_windows.size()

        if with_indexes:
            input_windows = (
                input_windows.permute(0, 2, 3, 1, 4, 5).contiguous().view(i_0, i_2 * i_3, i_1)
            )
            return input_windows, i_2, i_3, i_1
        else:
            input_windows = (
                input_windows.permute(0, 2, 3, 1, 4, 5)
                .contiguous()
                .view(i_0, i_2 * i_3, i_1, i_4, i_5)
            )
            return input_windows

    def _filter(self, input_windows, flag, value):
        assert flag.dim() == 2, "flag should be batch version"
        input_window = input_windows[flag == value]
        bz = flag.size(0)
        return input_window.view(bz, input_window.size(0) // bz, -1)


# ---------------------------------------------------------------------------
# util/util.py (verbatim, just the two mask-shaping helpers InnerShiftTriple/InnerCos need)
# ---------------------------------------------------------------------------
def cal_feat_mask(inMask, nlayers):
    # inMask is tensor should be bz*1*256*256 float. Return: ByteTensor
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    inMask = inMask.float()
    ntimes = 2**nlayers
    inMask = F.interpolate(
        inMask, (inMask.size(2) // ntimes, inMask.size(3) // ntimes), mode="nearest"
    )
    inMask = inMask.detach().byte()
    return inMask


def cal_flag_given_mask_thred(mask, patch_size, stride, mask_thred):
    # flag size: bz*(h*w), it is only for patch_size=1 for now (Shift-Net's usage).
    assert mask.dim() == 4, "mask must be 4 dimensions"
    assert mask.size(1) == 1, "the size of the dim=1 must be 1"
    mask = mask.float()
    b = mask.size(0)
    mask = F.pad(
        mask, (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), "constant", 0
    )
    m = mask.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    m = m.contiguous().view(b, 1, -1, patch_size, patch_size)
    m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
    mm = m.ge(mask_thred / (1.0 * patch_size**2)).long()
    flag = mm.view(b, -1)
    return flag


# ---------------------------------------------------------------------------
# models/shift_net/InnerShiftTripleFunction.py (verbatim)
# ---------------------------------------------------------------------------
class InnerShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, shift_sz, stride, triple_w, flag, show_flow):
        InnerShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.show_flow = show_flow

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real

        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_().to(input)

        former_all = input.narrow(1, 0, c // 2)  # decoder feature
        latter_all = input.narrow(1, c // 2, c // 2)  # encoder feature

        ctx.flag = ctx.flag.to(input).long()

        bNonparm = Batch_NonShift()
        ctx.shift_offsets = []

        cosine, latter_windows, i_2, i_3, i_1 = bNonparm.cosine_similarity(
            former_all.clone(), latter_all.clone(), 1, stride, flag
        )

        _, indexes = torch.max(cosine, dim=2)

        mask_indexes = (flag == 1).nonzero(as_tuple=False)[:, 1].view(ctx.bz, -1)

        non_mask_indexes = (
            (flag == 0).nonzero(as_tuple=False)[:, 1].view(ctx.bz, -1).gather(1, indexes)
        )

        idx_b = torch.arange(ctx.bz).long().unsqueeze(1).expand(ctx.bz, mask_indexes.size(1))
        ctx.ind_lst[(idx_b, mask_indexes, non_mask_indexes)] = 1

        shift_masked_all = bNonparm._paste(latter_windows, ctx.ind_lst, i_2, i_3, i_1)

        if ctx.show_flow:
            raise AssertionError(
                "show_flow path intentionally unsupported (matches upstream note)."
            )

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    @staticmethod
    def get_flow_src():
        return InnerShiftTripleFunction.ctx.flow_srcs

    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        grad_former_all = grad_output[:, 0 : c // 3, :, :]
        grad_latter_all = grad_output[:, c // 3 : c * 2 // 3, :, :].clone()
        grad_shifted_all = grad_output[:, c * 2 // 3 : c, :, :].clone()

        W_mat_t = ind_lst.permute(0, 2, 1).contiguous()
        grad = grad_shifted_all.view(ctx.bz, c // 3, -1).permute(0, 2, 1)
        grad_shifted_weighted = torch.bmm(W_mat_t, grad)
        grad_shifted_weighted = (
            grad_shifted_weighted.permute(0, 2, 1).contiguous().view(ctx.bz, c // 3, ctx.h, ctx.w)
        )
        grad_latter_all = torch.add(grad_latter_all, grad_shifted_weighted.mul(ctx.triple_w))

        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# models/shift_net/InnerShiftTriple.py (verbatim, device forced to 'cpu' at
# construction time below since the menagerie build has no GPU dependency)
# ---------------------------------------------------------------------------
class InnerShiftTriple(nn.Module):
    def __init__(
        self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, layer_to_last=3, device="cpu"
    ):
        super(InnerShiftTriple, self).__init__()

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.layer_to_last = layer_to_last
        self.device = device
        self.show_flow = False
        self.flow_srcs = None

    def set_mask(self, mask_global):
        self.mask_all = cal_feat_mask(mask_global, self.layer_to_last)

    def _split_mask(self, cur_bsize):
        cur_device = torch.cuda.current_device()
        self.cur_mask = self.mask_all[
            cur_device * cur_bsize : (cur_device + 1) * cur_bsize, :, :, :
        ]

    def forward(self, input):
        self.bz, self.c, self.h, self.w = input.size()
        if self.device != "cpu":
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.flag = cal_flag_given_mask_thred(
            self.cur_mask, self.shift_sz, self.stride, self.mask_thred
        )
        final_out = InnerShiftTripleFunction.apply(
            input, self.shift_sz, self.stride, self.triple_weight, self.flag, self.show_flow
        )
        if self.show_flow:
            self.flow_srcs = InnerShiftTripleFunction.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + "(" + " ,triple_weight " + str(self.triple_weight) + ")"


# ---------------------------------------------------------------------------
# models/shift_net/InnerCosFunction.py (verbatim)
# ---------------------------------------------------------------------------
class InnerCosFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, criterion, strength, target, mask):
        ctx.c = input.size(1)
        ctx.strength = strength
        ctx.criterion = criterion
        if len(target.size()) == 0:  # For the first iteration.
            target = target.expand_as(input.narrow(1, ctx.c // 2, ctx.c // 2)).type_as(input)

        ctx.save_for_backward(input, target, mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            input, target, mask = ctx.saved_tensors
            former = input.narrow(1, 0, ctx.c // 2)
            former_in_mask = torch.mul(former, mask)
            if former_in_mask.size() != target.size():
                target = target.narrow(0, 0, 1).expand_as(former_in_mask).type_as(former_in_mask)

            former_in_mask_clone = former_in_mask.clone().detach().requires_grad_(True)
            ctx.loss = ctx.criterion(former_in_mask_clone, target) * ctx.strength
            ctx.loss.backward()

        grad_output[:, 0 : ctx.c // 2, :, :] += former_in_mask_clone.grad

        return grad_output, None, None, None, None


# ---------------------------------------------------------------------------
# models/shift_net/InnerCos.py (verbatim, device forced to 'cpu' below)
# ---------------------------------------------------------------------------
class InnerCos(nn.Module):
    def __init__(self, crit="MSE", strength=1, skip=0, layer_to_last=3, device="cpu"):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == "MSE" else torch.nn.L1Loss()
        self.strength = strength
        self.skip = skip
        self.layer_to_last = layer_to_last
        self.device = device
        self.target = torch.tensor(1.0)

    def set_mask(self, mask_global):
        mask_all = cal_feat_mask(mask_global, self.layer_to_last)
        self.mask_all = mask_all.float()

    def _split_mask(self, cur_bsize):
        cur_device = torch.cuda.current_device()
        self.cur_mask = self.mask_all[
            cur_device * cur_bsize : (cur_device + 1) * cur_bsize, :, :, :
        ]

    def forward(self, in_data):
        self.bz, self.c, _, _ = in_data.size()
        if self.device != "cpu":
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.cur_mask = self.cur_mask.to(in_data)
        if not self.skip:
            self.output = InnerCosFunction.apply(
                in_data, self.criterion, self.strength, self.target, self.cur_mask
            )
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).detach()
        else:
            self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = "True" if not self.skip else "False"
        return (
            self.__class__.__name__
            + "("
            + "skip: "
            + skip_str
            + "layer "
            + str(self.layer_to_last)
            + " to last"
            + " ,strength: "
            + str(self.strength)
            + ")"
        )


# ---------------------------------------------------------------------------
# models/modules/unet.py (verbatim: plain UnetSkipConnectionBlock used at the
# non-shift skip levels of UnetGeneratorShiftTriple)
# ---------------------------------------------------------------------------
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False,
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(
            nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode="bilinear")
            return torch.cat([x_latter, x], 1)


# ---------------------------------------------------------------------------
# models/modules/shift_unet.py (verbatim: the shift-connection generator itself)
# ---------------------------------------------------------------------------
class UnetGeneratorShiftTriple(nn.Module):
    # Defines the Unet generator.
    # |num_downs|: number of downsamplings in UNet. For example,
    # if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        opt,
        innerCos_list,
        shift_list,
        mask_global,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False,
    ):
        super(UnetGeneratorShiftTriple, self).__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            use_spectral_norm=use_spectral_norm,
        )
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
        )

        unet_shift_block = UnetSkipConnectionShiftBlock(
            ngf * 2,
            ngf * 4,
            opt,
            innerCos_list,
            shift_list,
            mask_global,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
            layer_to_last=3,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_shift_block,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
        )
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
        )

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class UnetSkipConnectionShiftBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        opt,
        innerCos_list,
        shift_list,
        mask_global,
        input_nc,
        submodule=None,
        shift_layer=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False,
        layer_to_last=3,
    ):
        super(UnetSkipConnectionShiftBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(
            nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        device = "cpu" if len(opt.gpu_ids) == 0 else "gpu"
        shift = InnerShiftTriple(
            opt.shift_sz,
            opt.stride,
            opt.mask_thred,
            opt.triple_weight,
            layer_to_last=layer_to_last,
            device=device,
        )

        shift.set_mask(mask_global)
        shift_list.append(shift)

        innerCos = InnerCos(
            strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last, device=device
        )
        innerCos.set_mask(mask_global)
        innerCos_list.append(innerCos)

        if outermost:
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = spectral_norm(
                nn.ConvTranspose2d(inner_nc * 3, outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            )
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu; innerCos placed before shift.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode="bilinear")
            return torch.cat([x_latter, x], 1)


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
class _ShiftNetOpt:
    """Minimal stand-in for the repo's argparse `opt` namespace (models/networks.py:define_G
    / models/modules/shift_unet.py:UnetSkipConnectionShiftBlock)."""

    def __init__(self):
        self.gpu_ids = []
        self.shift_sz = 1
        self.stride = 1
        self.mask_thred = 1
        self.triple_weight = 1
        self.strength = 1
        self.skip = 0


def build_shiftnet():
    opt = _ShiftNetOpt()
    # 64x64 input, num_downs=6: matches the repo's `unet_shift_triple` topology (5 fixed
    # levels + num_downs-5 extra bottleneck levels) at menagerie scale; the shift-connection
    # sits at the 3rd-to-last layer (layer_to_last=3), same as the repo default.
    mask_global = torch.zeros(1, 1, 64, 64)
    mask_global[:, :, 24:40, 24:40] = 1  # a centered square hole, like the repo's inpainting masks
    innerCos_list = []
    shift_list = []
    model = UnetGeneratorShiftTriple(
        input_nc=3,
        output_nc=3,
        num_downs=6,
        opt=opt,
        innerCos_list=innerCos_list,
        shift_list=shift_list,
        mask_global=mask_global,
        ngf=8,
        norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False,
    )
    model.eval()
    return model


def example_input_shiftnet():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("Shift-Net", build_shiftnet, example_input_shiftnet, 2018, MENAGERIE_ZOO),
]
