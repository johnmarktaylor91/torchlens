# FAITHFUL REIMPLEMENTATION of the Active Appearance Model (AAM) face decoder
# used in Chang & Tsao, "The Code for Facial Identity in the Primate Brain"
# (Cell, 2017; PMC8088389), built on the classic AAM formulation of Cootes,
# Edwards & Taylor, "Active Appearance Models" (ECCV 1998) and Cootes, Edwards &
# Taylor, "Active Appearance Models" (IJCV/PAMI 2001). No dedicated code repo for
# either the classic AAM decoder or Chang & Tsao's specific face-space model was
# found (Chang & Tsao's STAR Methods describe the model but do not release a
# repo; the widely-known `menpo`/`menpofit` AAM toolkits are unmaintained,
# heavy-C-extension packages not in this environment's base libs).
#
# Chang & Tsao STAR Methods ("Generation of parameterized face stimuli", PMC
# 8088389) give the exact pipeline transcribed here: (1) hand-labeled facial
# landmarks on a face database were "smoothly morphed to a standard template
# (average shape of landmarks)" producing shape-free ("appearance") images; (2)
# PCA was run separately on the shape descriptors and the shape-free appearance
# descriptors, "retaining the first 25 PCs for shape and first 25 PCs for
# appearance" -- a 50-d face space; (3) a face is reconstructed "starting with
# the average face, first adding the appearance transform, and then applying
# the shape transform to the landmarks". That exact order (texture PCA in the
# canonical/mean-shape frame, THEN warp to the target landmark shape) is what
# `ActiveAppearanceModelDecoder.forward` below implements:
#   shape_landmarks    = mean_shape    + shape_basis      @ shape_params   (25-d)
#   canonical_texture  = mean_texture  + appearance_basis @ appearance_params (25-d)
#   output_image       = warp(canonical_texture, mean_shape -> shape_landmarks)
#
# The paper's own warp step ("smoothly morphed") is the classic AAM
# piecewise-affine warp over a Delaunay triangulation of the landmarks (Cootes
# et al. 1998/2001). This module instead uses a Thin-Plate-Spline (TPS) warp --
# a standard, widely used, differentiable realization of the same "smooth
# landmark-driven morph to/from a canonical template" operation (same role as
# Cootes' piecewise-affine warp, implemented here as a closed-form batched
# linear solve for a torch-only differentiable decoder rather than a
# triangulation + barycentric-coordinate rasterizer). Exact facial landmark
# count/positions are not published in Chang & Tsao's methods (they used
# hand-labeled points on the FEI database); 68 landmarks (the standard face
# landmark count used across the classic-AAM/face-alignment literature) are
# used here with a randomly initialized mean shape, matching this menagerie's
# random-init convention for the trainable AAM basis matrices themselves.
#
# MENAGERIE_ZOO = "reimpl-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


def _tps_fit(src_pts, dst_pts, reg=1e-6):
    """Closed-form batched Thin-Plate-Spline fit mapping src_pts -> dst_pts.

    src_pts, dst_pts: (B, K, 2). Returns TPS parameters (B, K+3, 2): the first
    K rows are the per-control-point warp weights, the last 3 are the affine
    part (bias, x-coeff, y-coeff).
    """
    b, k, _ = src_pts.shape
    diff = src_pts.unsqueeze(2) - src_pts.unsqueeze(1)  # (B,K,K,2)
    dist = diff.norm(dim=-1).clamp(min=1e-6)
    u = dist**2 * torch.log(dist)  # TPS radial basis kernel r^2 log(r)
    eye = torch.eye(k, device=src_pts.device, dtype=src_pts.dtype)
    ones = torch.ones(b, k, 1, device=src_pts.device, dtype=src_pts.dtype)
    p = torch.cat([ones, src_pts], dim=-1)  # (B,K,3)
    top = torch.cat([u + reg * eye, p], dim=-1)  # (B,K,K+3)
    zeros33 = torch.zeros(b, 3, 3, device=src_pts.device, dtype=src_pts.dtype)
    bottom = torch.cat([p.transpose(1, 2), zeros33], dim=-1)  # (B,3,K+3)
    lhs = torch.cat([top, bottom], dim=1)  # (B,K+3,K+3)
    zeros32 = torch.zeros(b, 3, 2, device=src_pts.device, dtype=src_pts.dtype)
    rhs = torch.cat([dst_pts, zeros32], dim=1)  # (B,K+3,2)
    return torch.linalg.solve(lhs, rhs)  # (B,K+3,2)


def _tps_eval(query_pts, src_pts, params):
    """Evaluate a fitted TPS warp at query_pts. query_pts: (B,N,2)."""
    diff = query_pts.unsqueeze(2) - src_pts.unsqueeze(1)  # (B,N,K,2)
    dist = diff.norm(dim=-1).clamp(min=1e-6)
    u = dist**2 * torch.log(dist)  # (B,N,K)
    ones = torch.ones(*query_pts.shape[:2], 1, device=query_pts.device, dtype=query_pts.dtype)
    p = torch.cat([ones, query_pts], dim=-1)  # (B,N,3)
    basis = torch.cat([u, p], dim=-1)  # (B,N,K+3)
    return torch.bmm(basis, params)  # (B,N,2)


class ActiveAppearanceModelDecoder(nn.Module):
    """AAM face-space decoder: linear shape PCA + linear texture PCA in a
    canonical mean-shape frame, composed via a landmark-driven warp (Cootes
    et al. 1998/2001; Chang & Tsao 2017 face-space usage)."""

    def __init__(self, n_landmarks=68, n_shape=25, n_appearance=25, channels=3, tex_size=48):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.channels = channels
        self.tex_h = tex_size
        self.tex_w = tex_size

        self.mean_shape = nn.Parameter(torch.randn(n_landmarks, 2) * 0.4)
        self.shape_basis = nn.Parameter(torch.randn(n_landmarks * 2, n_shape) * 0.02)

        self.mean_texture = nn.Parameter(torch.rand(channels, tex_size, tex_size))
        self.appearance_basis = nn.Parameter(
            torch.randn(channels * tex_size * tex_size, n_appearance) * 0.02
        )

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, tex_size),
            torch.linspace(-1, 1, tex_size),
            indexing="ij",
        )
        self.register_buffer("_output_grid_pts", torch.stack([xs, ys], dim=-1).reshape(1, -1, 2))

    def forward(self, shape_params, appearance_params):
        b = shape_params.shape[0]

        # Step 1: shape landmarks = mean_shape + shape PCA transform
        shape_disp = shape_params @ self.shape_basis.t()
        shape_landmarks = self.mean_shape.unsqueeze(0) + shape_disp.view(b, self.n_landmarks, 2)

        # Step 2: canonical (shape-free) texture = mean_texture + appearance PCA transform
        tex_disp = appearance_params @ self.appearance_basis.t()
        canonical_texture = self.mean_texture.unsqueeze(0) + tex_disp.view(
            b, self.channels, self.tex_h, self.tex_w
        )

        # Step 3: warp the canonical texture from the mean shape onto the
        # instance-specific landmarks (Cootes et al.'s "morph to template",
        # applied here in reverse/inverse-warp form for grid_sample).
        mean_shape_batch = self.mean_shape.unsqueeze(0).expand(b, -1, -1)
        tps_params = _tps_fit(shape_landmarks, mean_shape_batch)

        grid_pts = self._output_grid_pts.expand(b, -1, -1)
        sample_coords = _tps_eval(grid_pts, shape_landmarks, tps_params)
        sample_grid = sample_coords.view(b, self.tex_h, self.tex_w, 2)

        return F.grid_sample(
            canonical_texture,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


def build_aam_face_decoder():
    return ActiveAppearanceModelDecoder(
        n_landmarks=68, n_shape=25, n_appearance=25, channels=3, tex_size=48
    )


def example_input_aam_face_decoder():
    shape_params = torch.randn(2, 25) * 2.0
    appearance_params = torch.randn(2, 25) * 2.0
    return (shape_params, appearance_params)


MENAGERIE_ENTRIES = [
    (
        "Active Appearance Model (IT face-patch encoding)",
        "build_aam_face_decoder",
        "example_input_aam_face_decoder",
        2017,
        MENAGERIE_ZOO,
    ),
]
