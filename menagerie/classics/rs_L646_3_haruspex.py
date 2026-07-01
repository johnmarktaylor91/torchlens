# FAITHFUL PORT of thorn-lab/haruspex @ 009f0059abbb325f29369fa788cd985d067f81c9
# (source/hpx_unet_190116.py, `unet_model_fn`) (original framework: TensorFlow 1.x,
# tf.estimator API + tf.layers). Haruspex is a cryo-EM secondary-structure
# segmentation 3D U-Net (predicts sheet/helix/nucleic-acid/empty per voxel from a
# 40^3 density sub-volume). The upstream repo ships only a frozen TF1 checkpoint
# (model.ckpt-40000 + graph.pbtxt) and a `tf.estimator`-based training/inference
# script -- both TF1-only and not runnable/vendorable in this base env (no
# TensorFlow 1.x, no `tf.estimator`, deprecated since TF2). The architecture itself
# is a plain, fully-specified feedforward 3D U-Net with explicit conv/pool/crop/
# concat wiring (source/hpx_unet_190116.py:169-296, `unet_model_fn`'s inference
# path through `logits = tf.layers.conv3d(...)`), so it is faithfully transcribed
# here layer-for-layer into self-contained torch (every conv, its exact
# filters/kernel_size/padding-mode/activation, every pool, every crop offset, and
# every skip-connection concat, matching the upstream shape trace 40->38->36->
# pool->18->16->pool->8->8(same)->upconv->16->concat->14->12->upconv->24->crop
# conv1_2 to 24->concat->22->20|4, exactly as commented in the original source).
# Training-only code (loss, weighted cross-entropy, eval metrics, TF1 data
# augmentation/cropping pipeline) is intentionally dropped -- this ports the
# inference/forward architecture only, per the menagerie's model (not training
# pipeline) scope.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class HaruspexUNet(nn.Module):
    """3D U-Net for cryo-EM secondary-structure segmentation (sheet/helix/
    nucleic-acid/empty), ported layer-for-layer from Haruspex's TF1
    `unet_model_fn` (source/hpx_unet_190116.py).

    Input: [B, 1, 40, 40, 40] density sub-volume.
    Output: [B, 4, 20, 20, 20] per-voxel class logits.
    """

    def __init__(self):
        super().__init__()

        # ##### LEVEL 1
        # Conv1_1: 40^3|1 -> 38^3|32 (valid, k3, relu)
        self.conv1_1 = nn.Conv3d(1, 32, kernel_size=3, padding=0)
        # Conv1_2: 38^3|32 -> 36^3|64 (valid, k3, relu)
        self.conv1_2 = nn.Conv3d(32, 64, kernel_size=3, padding=0)
        # Pool1: 36^3|64 -> 18^3|64
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # ##### LEVEL 2
        # Conv2_1: 18^3|64 -> 16^3|128 (valid, k3, relu)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=0)
        # Pool2: 16^3|128 -> 8^3|128
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # ##### LEVEL 3
        # Conv3: 8^3|128 -> 8^3|256 (same, k3, relu)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        # UConv3 (upconv): 8^3|256 -> 16^3|256 (k2, s2, relu, no bias)
        self.uconv3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, bias=False)

        # ##### LEVEL 4 (upconv)
        # ccat4 = concat([conv2_1 (128ch), uconv3 (256ch)]) -> 16^3|384
        # Conv4_1: 16^3|384 -> 14^3|256 (valid, k3, relu)
        self.conv4_1 = nn.Conv3d(384, 256, kernel_size=3, padding=0)
        # Conv4_2: 14^3|256 -> 12^3|128 (valid, k3, relu)
        self.conv4_2 = nn.Conv3d(256, 128, kernel_size=3, padding=0)
        # UConv4 (upconv): 12^3|128 -> 24^3|128 (k2, s2, relu, no bias)
        self.uconv4 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, bias=False)

        # ##### LEVEL 5 (upconv)
        # crop conv1_2 (36^3|64) to central 24^3|64, concat with uconv4 (24^3|128)
        # -> ccat5: 24^3|192
        # Conv5_1: 24^3|192 -> 22^3|128 (valid, k3, relu)
        self.conv5_1 = nn.Conv3d(192, 128, kernel_size=3, padding=0)
        # logits: 22^3|128 -> 20^3|4 (valid, k3, no activation)
        self.logits_conv = nn.Conv3d(128, 4, kernel_size=3, padding=0)

    @staticmethod
    def _center_crop_3d(x, target_size):
        """Crops the spatial (D, H, W) dims of `x` to `target_size` centered,
        matching `tf.slice(conv1_2, [0, 6, 6, 6, 0], [-1, 24, 24, 24, 64])`."""
        _, _, d, h, w = x.shape
        off_d = (d - target_size) // 2
        off_h = (h - target_size) // 2
        off_w = (w - target_size) // 2
        return x[
            :,
            :,
            off_d : off_d + target_size,
            off_h : off_h + target_size,
            off_w : off_w + target_size,
        ]

    def forward(self, x):
        # x: [B, 1, 40, 40, 40]

        # LEVEL 1
        conv1_1 = F.relu(self.conv1_1(x))
        conv1_2 = F.relu(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)

        # LEVEL 2
        conv2_1 = F.relu(self.conv2_1(pool1))
        pool2 = self.pool2(conv2_1)

        # LEVEL 3
        conv3 = F.relu(self.conv3(pool2))
        uconv3 = F.relu(self.uconv3(conv3))

        # LEVEL 4 (upconv)
        ccat4 = torch.cat([conv2_1, uconv3], dim=1)
        conv4_1 = F.relu(self.conv4_1(ccat4))
        conv4_2 = F.relu(self.conv4_2(conv4_1))
        uconv4 = F.relu(self.uconv4(conv4_2))

        # LEVEL 5 (upconv)
        crop5 = self._center_crop_3d(conv1_2, 24)
        ccat5 = torch.cat([crop5, uconv4], dim=1)
        conv5_1 = F.relu(self.conv5_1(ccat5))
        logits = self.logits_conv(conv5_1)

        return logits


#########################################################################
# --- staging harness ---
#########################################################################


def build_haruspex():
    torch.manual_seed(0)
    return HaruspexUNet()


def example_input_haruspex():
    torch.manual_seed(0)
    return torch.randn(1, 1, 40, 40, 40)


MENAGERIE_ENTRIES = [
    ("Haruspex", "build_haruspex", "example_input_haruspex", 2019, "ported-pytorch"),
]
