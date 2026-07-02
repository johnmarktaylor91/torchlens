# FAITHFUL PORT of https://github.com/phdyang007/deepattern @ 5c2a192c2e11ecf3be902a2ae3e4139068b4f0c5
# (original framework: TensorFlow 1.x + tf.contrib.slim)
#
# Ports the Transforming Convolutional Auto-Encoder (TCAE) from `src/cdnsgen.py::squish_dl.cae`,
# the model behind Yang et al., "DeePattern: Layout Pattern Generation with Transforming
# Convolutional Auto-Encoder" (DAC'19). The original `cae()` method (TF1.x/slim) defines:
#   input (B,16,16,1)
#   -> conv 128ch k5 stride2  ("pool1")
#   -> conv 256ch k5 stride2  ("pool2")
#   -> flatten -> fc 1024 ("fc1")
#   -> fc 32 ("fc2", the latent feature map "fm"; at inference time noise is added here)
#   -> fc 1024 ("fc3")
#   -> fc 4*4*256 ("fc4") -> reshape to (B,4,4,256)
#   -> deconv 128ch k5 stride2 ("upool2")
#   -> deconv 1ch k5 stride2 ("upool1") -> output (B,16,16,1)
# All conv/deconv/fc layers use ReLU activation except the final deconv (linear, matching the
# original `slim.conv2d_transpose` default activation_fn=tf.nn.relu is NOT overridden on the last
# layer in the source -- verified against the source: the last `conv2d_transpose` call inherits
# the arg_scope's `activation_fn=tf.nn.relu`, so it IS relu-activated in the original; ported as-is).
import torch
from torch import nn


class TCAE(nn.Module):
    """Transforming Convolutional Auto-Encoder, ported from squish_dl.cae (DeePattern)."""

    def __init__(self, img_size: int = 16, img_channel: int = 1, latent_dim: int = 32):
        super().__init__()
        self.img_size = img_size
        self.img_channel = img_channel
        self.latent_dim = latent_dim
        pooled = img_size // 4  # two stride-2 convs

        # Encoder: two stride-2 conv blocks (slim.conv2d, k5, SAME padding, relu)
        self.pool1 = nn.Conv2d(img_channel, 128, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)

        self.pooled_hw = pooled
        self.fc1 = nn.Linear(pooled * pooled * 256, 1024)
        self.fc2 = nn.Linear(1024, latent_dim)  # latent feature map ("fm")
        self.fc3 = nn.Linear(latent_dim, 1024)
        self.fc4 = nn.Linear(1024, pooled * pooled * 256)

        # Decoder: two stride-2 transposed-conv blocks (slim.conv2d_transpose, k5, SAME, relu)
        self.upool2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upool1 = nn.ConvTranspose2d(
            128, img_channel, kernel_size=5, stride=2, padding=2, output_padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        net = self.relu(self.pool1(x))
        net = self.relu(self.pool2(net))
        b = net.shape[0]
        net = net.flatten(1)
        net = self.relu(self.fc1(net))
        fm = self.fc2(net)
        if noise is not None:
            fm = fm + noise
        net = self.relu(self.fc3(fm))
        net = self.relu(self.fc4(net))
        net = net.view(b, 256, self.pooled_hw, self.pooled_hw)
        net = self.relu(self.upool2(net))
        net = self.relu(self.upool1(net))
        return net


# --- TorchLens menagerie staging harness (not part of the original repo) ---


def build_deepattern_tcae():
    return TCAE(img_size=16, img_channel=1, latent_dim=32)


def example_input_deepattern_tcae():
    torch.manual_seed(0)
    return (torch.rand(2, 1, 16, 16),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeePattern-TCAE",
        build_deepattern_tcae,
        example_input_deepattern_tcae,
        2019,
        "ported-pytorch",
    ),
]
