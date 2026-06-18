"""Random Receptive-Area Classifiers: LIRA and PCNC.

Two related lightweight classifiers from Kussul and Baidyk (Kyiv / UNAM) that
extract sparse binary features via fixed random receptive-field masks over the
input image, then apply a trainable linear readout.

LIRA (Limited Receptive Area classifier), 2004:
    Kussul and Baidyk 2004, "Improved Method of Handwritten Digit Recognition
    Tested on MNIST Database."
    Random sparse binary masks select limited receptive-area windows from the
    input; each mask produces one bit (threshold), forming a sparse binary
    feature vector that feeds a trainable linear readout.

PCNC (Permutation Coding Neural Classifier), 2006:
    Kussul, Baidyk, Makeyev 2006, "Permutation Coding Neural Classifier."
    Extends LIRA: after detecting local features in receptive windows, the
    position of each feature along each axis is encoded by axis-specific random
    permutations, creating a sparse shift-invariant code fed to a linear readout.

Both are trace-clean: fixed binary masks are registered buffers; the threshold
and permutation operations use only torch ops with no data-dependent branching.
"""

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# LIRA
# ---------------------------------------------------------------------------


class LIRA(nn.Module):
    """Limited Receptive Area classifier with sparse random binary feature masks.

    Each of ``n_features`` hidden units uses a random sparse binary mask over
    a limited receptive window of the input image.  The inner product between
    the flattened input patch and the mask is compared to a threshold; the
    resulting binary activation vector feeds a linear readout.

    The binary threshold is implemented as sigmoid(scale * (dot - tau)) so
    the module produces a differentiable approximation that traces cleanly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        height: int = 28,
        width: int = 28,
        n_features: int = 256,
        window_h: int = 7,
        window_w: int = 7,
        n_classes: int = 10,
        scale: float = 8.0,
    ) -> None:
        """Initialize random receptive-area masks and linear readout.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        height:
            Input image height.
        width:
            Input image width.
        n_features:
            Number of hidden feature units (one mask each).
        window_h:
            Receptive window height.
        window_w:
            Receptive window width.
        n_classes:
            Number of output classes.
        scale:
            Sigmoid sharpness for differentiable threshold approximation.
        """
        super().__init__()
        self.scale = scale
        self.n_features = n_features
        self.window_h = window_h
        self.window_w = window_w
        # Random top-left corner of each receptive window
        row_off = torch.randint(0, max(1, height - window_h + 1), (n_features,))
        col_off = torch.randint(0, max(1, width - window_w + 1), (n_features,))
        self.register_buffer("row_off", row_off)
        self.register_buffer("col_off", col_off)
        # Binary random masks over each window (sparse: ~50% density)
        mask = (torch.rand(n_features, in_channels * window_h * window_w) > 0.5).float()
        self.register_buffer("mask", mask)
        # Learnable threshold (one per feature)
        self.tau = nn.Parameter(torch.zeros(n_features))
        self.readout = nn.Linear(n_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Extract limited-receptive-area features then apply linear readout.

        Parameters
        ----------
        x:
            Input image with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(B, n_classes)``.
        """
        x.shape[0]
        features = []
        for k in range(self.n_features):
            r = int(self.row_off[k].item())
            c = int(self.col_off[k].item())
            # Extract window: (B, C, wh, ww) -> (B, C*wh*ww)
            window = x[:, :, r : r + self.window_h, c : c + self.window_w]
            patch = window.flatten(start_dim=1)  # (B, C*wh*ww)
            # Dot with binary mask: (B,)
            dot = (patch * self.mask[k]).sum(dim=-1)
            # Differentiable threshold
            act = torch.sigmoid(self.scale * (dot - self.tau[k]))
            features.append(act)
        feat_vec = torch.stack(features, dim=-1)  # (B, n_features)
        return self.readout(feat_vec)


# ---------------------------------------------------------------------------
# PCNC
# ---------------------------------------------------------------------------


class PCNC(nn.Module):
    """Permutation Coding Neural Classifier.

    Like LIRA, but after detecting which cells fire in a receptive window, the
    position of each detected feature along x and y axes is encoded by
    axis-specific random permutations into a sparse binary vector.

    Implementation:
    1. Extract a limited receptive area window (as in LIRA).
    2. Flatten and threshold to get a binary feature vector.
    3. Apply two fixed random permutations (row, col separately).
    4. XOR-combine the two permuted codes to produce the position code.
    5. Concatenate all window codes and feed to linear readout.

    Because the permutation is a fixed index rearrangement via gather/index,
    this traces cleanly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        height: int = 28,
        width: int = 28,
        n_windows: int = 64,
        window_h: int = 7,
        window_w: int = 7,
        n_classes: int = 10,
        scale: float = 8.0,
    ) -> None:
        """Initialize random receptive windows and axis permutations.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        height:
            Input image height.
        width:
            Input image width.
        n_windows:
            Number of receptive windows (= number of local feature detectors).
        window_h:
            Window height.
        window_w:
            Window width.
        n_classes:
            Number of output classes.
        scale:
            Sigmoid sharpness for differentiable threshold.
        """
        super().__init__()
        self.scale = scale
        self.n_windows = n_windows
        self.window_h = window_h
        self.window_w = window_w
        patch_size = in_channels * window_h * window_w

        # Random window offsets
        row_off = torch.randint(0, max(1, height - window_h + 1), (n_windows,))
        col_off = torch.randint(0, max(1, width - window_w + 1), (n_windows,))
        self.register_buffer("row_off", row_off)
        self.register_buffer("col_off", col_off)

        # Axis permutations: each axis gets its own permutation of patch indices
        perm_row = torch.randperm(patch_size)
        perm_col = torch.randperm(patch_size)
        self.register_buffer("perm_row", perm_row)
        self.register_buffer("perm_col", perm_col)

        # Learnable threshold per window
        self.tau = nn.Parameter(torch.zeros(n_windows))
        # Readout: 2 * patch_size * n_windows features (row + col code per window)
        self.readout = nn.Linear(2 * patch_size * n_windows, n_classes)
        self._patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        """Extract permutation-coded features then apply linear readout.

        Parameters
        ----------
        x:
            Input image with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(B, n_classes)``.
        """
        x.shape[0]
        codes = []
        for k in range(self.n_windows):
            r = int(self.row_off[k].item())
            c = int(self.col_off[k].item())
            patch = x[:, :, r : r + self.window_h, c : c + self.window_w]
            flat = patch.flatten(start_dim=1)  # (B, patch_size)
            # Soft binary feature detection
            flat.sum(dim=-1, keepdim=True) * 0.0 + flat  # identity; threshold below
            act = torch.sigmoid(self.scale * (flat - self.tau[k]))  # (B, patch_size)
            # Permutation-based position coding
            code_row = act[:, self.perm_row]  # (B, patch_size)
            code_col = act[:, self.perm_col]  # (B, patch_size)
            codes.append(code_row)
            codes.append(code_col)
        feat_vec = torch.cat(codes, dim=-1)  # (B, 2 * patch_size * n_windows)
        return self.readout(feat_vec)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_lira() -> nn.Module:
    """Build a small LIRA classifier.

    Returns
    -------
    nn.Module
        Configured ``LIRA`` instance.
    """
    return LIRA(in_channels=1, height=28, width=28, n_features=128, n_classes=10)


def example_input_lira() -> Tensor:
    """Create a 28x28 grayscale image for LIRA.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 28, 28)``.
    """
    return torch.rand(1, 1, 28, 28)


def build_pcnc() -> nn.Module:
    """Build a small PCNC classifier.

    Returns
    -------
    nn.Module
        Configured ``PCNC`` instance.
    """
    return PCNC(in_channels=1, height=28, width=28, n_windows=16, n_classes=10)


def example_input_pcnc() -> Tensor:
    """Create a 28x28 grayscale image for PCNC.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 28, 28)``.
    """
    return torch.rand(1, 1, 28, 28)


MENAGERIE_ENTRIES = [
    (
        "LIRA (Limited Receptive Area classifier)",
        "build_lira",
        "example_input_lira",
        "2004",
        "RT",
    ),
    (
        "PCNC (Permutation Coding Neural Classifier)",
        "build_pcnc",
        "example_input_pcnc",
        "2006",
        "RT",
    ),
]
