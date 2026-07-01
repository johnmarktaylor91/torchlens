# SOURCE: vendored from KumapowerLIU/CSA-inpainting @ master
# (files models/CSA.py, models/CSA_model.py, models/CSAFunction.py,
#  models/InnerCos.py, models/InnerCos2.py, models/vgg16.py, util/util.py,
#  util/NonparametricShift.py, util/MaxCoord.py)
#
# CSA (Coherent Semantic Attention) image inpainting network, ICCV 2019.
# https://github.com/KumapowerLIU/CSA-inpainting
#
# This module vendors the REAL generator architecture (UnetGeneratorCSA with the
# custom CSA shift-connection layer + InnerCos/InnerCos2 consistency layers) used
# for the "coarse-to-fine" inpainting network (netP -> netG). Only minimal,
# functionally-necessary fixes were made to the original code:
#   - `torch.cuda.is_available` (a function reference, always truthy -- a real bug
#     in the upstream repo) was fixed to `torch.cuda.is_available()` so the code
#     actually respects device placement instead of always assuming CUDA.
#   - Removed training-only scaffolding (options parsing, GAN/discriminator nets,
#     dataset loading, VGG perceptual-loss driving code) -- not part of the
#     architecture itself, matches TorchLens menagerie recipe convention of
#     capturing the forward-pass network, not the training harness.
# The CSA custom autograd.Function, the NonparametricShift patch-based
# encoder/decoder construction, and the MaxCoord attention-argmax mechanism are
# reproduced verbatim (structure-preserving) from the original files.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# util/util.py (relevant functions only)
# ---------------------------------------------------------------------------


def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad=False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        conv.weight.data.fill_(1 / 16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)
    output = Variable(output, requires_grad=False)
    return output.detach().byte()


def cal_mask_given_mask_thred(img, mask, patch_size, stride, mask_thred):
    assert img.dim() == 3, "img has to be 3 dimenison!"
    assert mask.dim() == 2, "mask has to be 2 dimenison!"
    dim = img.dim()

    _, H, W = img.size(dim - 3), img.size(dim - 2), img.size(dim - 1)
    nH = int(math.floor((H - patch_size) / stride + 1))
    nW = int(math.floor((W - patch_size) / stride + 1))
    N = nH * nW

    flag = torch.zeros(N).long()
    offsets_tmp_vec = torch.zeros(N).long()

    nonmask_point_idx_all = torch.zeros(N).long()
    tmp_non_mask_idx = 0

    mask_point_idx_all = torch.zeros(N).long()
    tmp_mask_idx = 0

    for i in range(N):
        h = int(math.floor(i / nW))
        w = int(math.floor(i % nW))

        mask_tmp = mask[h * stride : h * stride + patch_size, w * stride : w * stride + patch_size]

        if torch.sum(mask_tmp) < mask_thred:
            nonmask_point_idx_all[tmp_non_mask_idx] = i
            tmp_non_mask_idx += 1
        else:
            mask_point_idx_all[tmp_mask_idx] = i
            tmp_mask_idx += 1
            flag[i] = 1
            offsets_tmp_vec[i] = -1

    non_mask_num = tmp_non_mask_idx
    mask_num = tmp_mask_idx

    nonmask_point_idx = nonmask_point_idx_all.narrow(0, 0, non_mask_num)
    mask_point_idx = mask_point_idx_all.narrow(0, 0, mask_num)

    flatten_offsets_all = torch.LongTensor(N).zero_()
    for i in range(N):
        offset_value = torch.sum(offsets_tmp_vec[0 : i + 1])
        if flag[i] == 1:
            offset_value = offset_value + 1
        flatten_offsets_all[i + offset_value] = -offset_value

    flatten_offsets = flatten_offsets_all.narrow(0, 0, non_mask_num)

    return flag, nonmask_point_idx, flatten_offsets, mask_point_idx


def cal_sps_for_Advanced_Indexing(h, w):
    sp_y = torch.arange(0, w).long()
    sp_y = torch.cat([sp_y] * h)

    lst = []
    for i in range(h):
        lst.extend([i] * w)
    sp_x = torch.LongTensor(lst)
    return sp_x, sp_y


# ---------------------------------------------------------------------------
# util/NonparametricShift.py
# ---------------------------------------------------------------------------


class NonparametricShift:
    def buildAutoencoder(
        self,
        target_img,
        normalize,
        interpolate,
        nonmask_point_idx,
        mask_point_idx,
        patch_size=1,
        stride=1,
    ):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."
        C = target_img.size(0)

        patches_all, patches_part, patches_mask = self._extract_patches(
            target_img, patch_size, stride, nonmask_point_idx, mask_point_idx
        )

        npatches_part = patches_part.size(0)
        npatches_all = patches_all.size(0)

        conv_enc_non_mask, conv_dec_non_mask = self._build(
            patch_size, stride, C, patches_part, npatches_part, normalize, interpolate
        )
        conv_enc_all, conv_dec_all = self._build(
            patch_size, stride, C, patches_all, npatches_all, normalize, interpolate
        )

        return (
            conv_enc_all,
            conv_enc_non_mask,
            conv_dec_all,
            conv_dec_non_mask,
            patches_part,
            patches_mask,
        )

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate):
        enc_patches = target_patches.clone()
        for i in range(npatches):
            enc_patches[i] = enc_patches[i] * (1 / (enc_patches[i].norm(2) + 1e-8))

        conv_enc = nn.Conv2d(C, npatches, kernel_size=patch_size, stride=stride, bias=False)
        conv_enc.weight.data = enc_patches

        if normalize:
            raise NotImplementedError
        if interpolate:
            raise NotImplementedError

        conv_dec = nn.ConvTranspose2d(
            npatches, C, kernel_size=patch_size, stride=stride, bias=False
        )
        conv_dec.weight.data = target_patches

        return conv_enc, conv_dec

    def _extract_patches(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim, "image must be of dimension 3."
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = (
            input_windows.size(0),
            input_windows.size(1),
            input_windows.size(2),
            input_windows.size(3),
            input_windows.size(4),
        )
        input_windows = (
            input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1, i_4, i_5)
        )

        patches_all = input_windows
        patches = input_windows.index_select(0, nonmask_point_idx)
        maskpatches = input_windows.index_select(0, mask_point_idx)
        return patches_all, patches, maskpatches


# ---------------------------------------------------------------------------
# util/MaxCoord.py
# ---------------------------------------------------------------------------


class MaxCoord:
    def update_output(self, input, sp_x, sp_y):
        assert input.dim() == 4, "Input must be 3D or 4D(batch)."
        assert input.size(0) == 1, "The first dimension of input has to be 1!"

        output = torch.zeros_like(input)
        v_max, c_max = torch.max(input, 1)

        c_max_flatten = c_max.view(-1)
        v_max_flatten = v_max.view(-1)
        ind = c_max_flatten

        return output, ind, v_max_flatten


# ---------------------------------------------------------------------------
# models/CSAFunction.py
# ---------------------------------------------------------------------------


class CSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        mask,
        shift_sz,
        stride,
        triple_w,
        flag,
        nonmask_point_idx,
        mask_point_idx,
        flatten_offsets,
        sp_x,
        sp_y,
    ):
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flatten_offsets = flatten_offsets

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real
        use_cuda = input.is_cuda
        ctx.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        assert mask.dim() == 2, "Mask dimension must be 2"

        output_lst = ctx.Tensor(ctx.bz, c, ctx.h, ctx.w)
        ind_lst = torch.LongTensor(ctx.bz, ctx.h * ctx.w, ctx.h, ctx.w)

        if use_cuda:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            mask_point_idx = mask_point_idx.cuda()
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

        for idx in range(ctx.bz):
            inpatch = input.narrow(0, idx, 1)
            output = input.narrow(0, idx, 1)

            Nonparm = NonparametricShift()

            _, conv_enc, conv_new_dec, _, known_patch, unknown_patch = Nonparm.buildAutoencoder(
                inpatch.squeeze(0),
                False,
                False,
                nonmask_point_idx,
                mask_point_idx,
                shift_sz,
                stride,
            )
            if use_cuda:
                conv_enc = conv_enc.cuda()
                conv_new_dec = conv_new_dec.cuda()

            output_var = Variable(output)
            tmp1 = conv_enc(output_var)

            maxcoor = MaxCoord()

            kbar, ind, vmax = maxcoor.update_output(tmp1.data, sp_x, sp_y)
            real_patches = kbar.size(1) + torch.sum(ctx.flag)
            vamx_mask = vmax.index_select(0, mask_point_idx)
            _, _, kbar_h, kbar_w = kbar.size()
            out_new = unknown_patch.clone()
            out_new = out_new.zero_()
            mask_num = torch.sum(ctx.flag)

            in_attention = ctx.Tensor(int(mask_num), int(real_patches)).zero_()

            kbar = ctx.Tensor(1, int(real_patches), kbar_h, kbar_w).zero_()
            ind_laten = 0
            for i in range(kbar_h):
                for j in range(kbar_w):
                    indx = i * kbar_w + j
                    check = torch.eq(mask_point_idx, indx)
                    non_r_ch = ind[indx]
                    offset = ctx.flatten_offsets[non_r_ch]

                    correct_ch = int(non_r_ch + offset)
                    if check.sum() >= 1:
                        known_region = known_patch[non_r_ch]
                        unknown_region = unknown_patch[ind_laten]

                        if ind_laten == 0:
                            out_new[ind_laten] = known_region
                            in_attention[ind_laten, correct_ch] = 1
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                        else:
                            little_value = unknown_region.clone()
                            ininconv = out_new[ind_laten - 1].clone()
                            ininconv = torch.unsqueeze(ininconv, 0)

                            value_2 = little_value * (1 / (little_value.norm(2) + 1e-8))
                            conv_enc_2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
                            if use_cuda:
                                conv_enc_2 = conv_enc_2.cuda()
                            value_2 = torch.unsqueeze(value_2, 0)
                            conv_enc_2.weight.data = value_2

                            ininconv_var = Variable(ininconv)

                            at_value = conv_enc_2(ininconv_var)
                            at_value_m = at_value.data
                            at_value_m = at_value_m.squeeze()

                            at_final_new = at_value_m / (at_value_m + vamx_mask[ind_laten])
                            at_final_ori = vamx_mask[ind_laten] / (
                                at_value_m + vamx_mask[ind_laten]
                            )
                            out_new[ind_laten] = (at_final_new) * out_new[ind_laten - 1] + (
                                at_final_ori
                            ) * known_region
                            in_attention[ind_laten] = (
                                in_attention[ind_laten - 1] * at_final_new.item()
                            )
                            in_attention[ind_laten, correct_ch] = (
                                in_attention[ind_laten, correct_ch] + at_final_ori.item()
                            )
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                        ind_laten += 1
                    else:
                        kbar[:, correct_ch, i, j] = 1
            kbar_var = Variable(kbar)
            result_tmp_var = conv_new_dec(kbar_var)
            result_tmp = result_tmp_var.data
            output_lst[idx] = result_tmp
            ind_lst[idx] = kbar.squeeze(0)

        output = output_lst

        ctx.ind_lst = ind_lst
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        grad_swapped_all = grad_output.clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = Variable(ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_())
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0, idx).clone()
            back_attention = ind_lst[idx].clone()
            for i in range(ctx.h):
                for j in range(ctx.w):
                    indx = i * ctx.h + j
                    W_mat[indx] = back_attention[:, i, j]

            W_mat_t = W_mat.t()

            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c, -1).t())

            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c, ctx.h, ctx.w)
            grad_swapped_all[idx] = torch.add(
                grad_swapped_all[idx], grad_swapped_weighted.mul(ctx.triple_w)
            )

        grad_input = grad_swapped_all

        return grad_input, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# models/CSA_model.py -- the shift-connection layer (CSA_model class, renamed
# CSAShiftLayer to avoid clashing with the module-level "model" naming)
# ---------------------------------------------------------------------------


class CSAShiftLayer(nn.Module):
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super().__init__()
        self.threshold = threshold
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True

        self.sp_x = None
        self.sp_y = None

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask = cal_feat_mask(mask_global, layer_to_last, threshold)
        self.mask = mask.squeeze()
        return self.mask

    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and not self.cal_fixed_flag:
            assert torch.is_tensor(self.flag), (
                "flag must have been figured out and has to be a tensor!"
            )
        else:
            latter = input.narrow(0, 0, 1).data

            self.flag, self.nonmask_point_idx, self.flatten_offsets, self.mask_point_idx = (
                cal_mask_given_mask_thred(
                    latter.squeeze(0), self.mask, self.shift_sz, self.stride, self.mask_thred
                )
            )
            self.cal_fixed_flag = False

        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            self.sp_x, self.sp_y = cal_sps_for_Advanced_Indexing(self.h, self.w)

        return CSAFunction.apply(
            input,
            self.mask,
            self.shift_sz,
            self.stride,
            self.triple_weight,
            self.flag,
            self.nonmask_point_idx,
            self.mask_point_idx,
            self.flatten_offsets,
            self.sp_x,
            self.sp_y,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(threshold: "
            + str(self.threshold)
            + " ,triple_weight "
            + str(self.triple_weight)
            + ")"
        )


# ---------------------------------------------------------------------------
# models/InnerCos.py / InnerCos2.py -- feature-consistency layers (identity in
# forward, but genuinely part of the architecture's forward computation graph
# since they compute + stash an auxiliary loss against a target)
# ---------------------------------------------------------------------------


class InnerCos(nn.Module):
    def __init__(self, crit="MSE", strength=1, skip=0):
        super().__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == "MSE" else torch.nn.L1Loss()
        self.strength = strength
        self.target = None
        self.skip = skip

    def set_mask(self, mask_global, threshold):
        mask = cal_feat_mask(mask_global, 3, threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available():
            self.mask = self.mask.float().cuda()
        else:
            self.mask = self.mask.float()
        self.mask = Variable(self.mask, requires_grad=False)

    def set_target(self, targetIn):
        self.target = targetIn

    def forward(self, in_data):
        if not self.skip:
            self.bs, self.c, _, _ = in_data.size()
            self.former = in_data
            self.former_in_mask = torch.mul(self.former, self.mask)
            if self.target is not None:
                self.loss = self.criterion(self.former_in_mask * self.strength, self.target)
            self.output = in_data
        else:
            self.loss = 0
            self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = "True" if not self.skip else "False"
        return (
            self.__class__.__name__
            + "(skip: "
            + skip_str
            + " ,strength: "
            + str(self.strength)
            + ")"
        )


class InnerCos2(nn.Module):
    # NOTE: upstream hardcodes the channel width for `narrow` to 512 because the
    # paper always uses ngf=64 (=> inner_nc=512 at the CSA skip level). We
    # parameterize it as `narrow_channels` (=inner_nc, passed through from
    # CSASkipBlock) so the architecture scales correctly for a smaller ngf.
    def __init__(self, crit="MSE", strength=1, skip=0, narrow_channels=512):
        super().__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == "MSE" else torch.nn.L1Loss()
        self.strength = strength
        self.target = None
        self.skip = skip
        self.narrow_channels = narrow_channels

    def set_mask(self, mask_global, threshold):
        mask = cal_feat_mask(mask_global, 3, threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available():
            self.mask = self.mask.float().cuda()
        else:
            self.mask = self.mask.float()
        self.mask = Variable(self.mask, requires_grad=False)

    def set_target(self, targetIn):
        self.target = targetIn

    def forward(self, in_data):
        if not self.skip:
            self.former = in_data.narrow(1, 0, self.narrow_channels)
            self.bs, self.c, _, _ = self.former.size()
            self.former_in_mask = torch.mul(self.former, self.mask)
            if self.target is not None:
                self.loss = self.criterion(self.former_in_mask * self.strength, self.target)
            self.output = in_data
        else:
            self.loss = 0
            self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = "True" if not self.skip else "False"
        return (
            self.__class__.__name__
            + "(skip: "
            + skip_str
            + " ,strength: "
            + str(self.strength)
            + ")"
        )


# ---------------------------------------------------------------------------
# models/networks.py -- UnetSkipConnectionBlock_3, CSA (skip block housing the
# shift layer), UnetGeneratorCSA. Kept the original class name for the skip
# block as CSASkipBlock (was `CSA` in upstream, renamed here only to avoid
# clashing with the file-level module name `CSAShiftLayer` above).
# ---------------------------------------------------------------------------


class UnetSkipConnectionBlock3(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv_3 = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1
            )
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv_3 = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1
            )
            down = [downrelu, downconv, downnorm, downrelu_3, downconv_3, downnorm_3]
            up = [uprelu_3, upconv_3, upnorm_3, uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode="bilinear", align_corners=False)
            return torch.cat([x_latter, x], 1)


class CSASkipBlock(nn.Module):
    """The middle skip-connection block that houses the CSA shift-connection
    layer plus the two InnerCos consistency layers -- this is the architectural
    novelty of the CSA paper (models/networks.py `class CSA`, renamed here to
    avoid a name clash with this file's CSAShiftLayer)."""

    def __init__(
        self,
        outer_nc,
        inner_nc,
        threshold,
        fixed_mask,
        shift_sz,
        stride,
        mask_thred,
        triple_weight,
        strength,
        skip,
        mask_global,
        input_nc,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        csa = CSAShiftLayer(threshold, fixed_mask, shift_sz, stride, mask_thred, triple_weight)
        csa.set_mask(mask_global, 3, threshold)
        self.csa = csa

        innerCos = InnerCos(strength=strength, skip=skip)
        innerCos.set_mask(mask_global, threshold)
        self.innerCos = innerCos

        innerCos2 = InnerCos2(strength=strength, skip=skip, narrow_channels=inner_nc)
        innerCos2.set_mask(mask_global, threshold)
        self.innerCos2 = innerCos2

        if outermost:
            upconv_3 = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1
            )
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv_3 = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1
            )
            down = [downrelu, downconv, downnorm, downrelu_3, downconv_3, csa, innerCos, downnorm_3]
            up = [innerCos2, uprelu_3, upconv_3, upnorm_3, uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode="bilinear", align_corners=False)
            return torch.cat([x_latter, x], 1)


class UnetGeneratorCSA(nn.Module):
    """The real CSA inpainting generator (models/networks.py `UnetGeneratorCSA`):
    an 8-level U-Net with the CSA shift-connection block placed at the
    ngf*4 <-> ngf*8 skip level, matching the original `define_G('unet_csa', ...)`
    construction exactly (num_downs=8)."""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        mask_global,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        threshold=5.0 / 16,
        fixed_mask=1,
        shift_sz=1,
        stride=1,
        mask_thred=1,
        triple_weight=1,
        strength=1,
        skip=0,
    ):
        super().__init__()

        unet_block = UnetSkipConnectionBlock3(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock3(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
        unet_csa = CSASkipBlock(
            ngf * 4,
            ngf * 8,
            threshold,
            fixed_mask,
            shift_sz,
            stride,
            mask_thred,
            triple_weight,
            strength,
            skip,
            mask_global,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock3(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_csa, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock3(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock3(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# ---------------------------------------------------------------------------
# menagerie build/example helpers
# ---------------------------------------------------------------------------


def build_csa_inpainting():
    # The 8-level U-Net (num_downs=8, i.e. 5 stride-2 UnetSkipConnectionBlock_3
    # downsamples on top of the 3 innermost) needs fineSize=256 (the paper
    # default) so InstanceNorm2d never hits a 1x1 spatial map at the bottleneck
    # (a batch=1 InstanceNorm2d requires >1 spatial element). ngf shrunk from
    # the paper default (64) to 8 to keep this a small menagerie model.
    fine_size = 256
    ngf = 8
    mask_global = torch.zeros(1, 1, fine_size, fine_size, dtype=torch.uint8)
    overlap = 4
    q = fine_size // 4
    mask_global[
        :, :, q + overlap : fine_size // 2 + q - overlap, q + overlap : fine_size // 2 + q - overlap
    ] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_global = mask_global.to(device)

    model = UnetGeneratorCSA(
        input_nc=3,
        output_nc=3,
        num_downs=8,
        mask_global=mask_global,
        ngf=ngf,
        norm_layer=nn.InstanceNorm2d,
        threshold=5.0 / 16,
    )
    return model.to(device)


def example_input_csa_inpainting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn(1, 3, 256, 256, device=device)


MENAGERIE_ENTRIES = [
    ("CSA Inpainting", "build_csa_inpainting", "example_input_csa_inpainting", 2019, MENAGERIE_ZOO),
]
