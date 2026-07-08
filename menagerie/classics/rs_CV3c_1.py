# SOURCE: vendored from Charlie0215/AWNet-Attentive-Wavelet-Network-for-Image-ISP @ master
# (models/model_3channel.py, models/modules_3channel.py, models/utils.py)

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from CV3c_awnet_vendor.model_3channel import AWNet


def build_awnet_attention_wavelet_network_for_isp() -> AWNet:
    """
    Build a tiny three-channel AWNet model.

    Returns
    -------
    AWNet
        Vendored AWNet with reduced residual block counts for traceability.
    """
    model = AWNet(in_channels=3, out_channels=3, block=[1, 1, 1, 1, 1])
    model.eval()
    return model


def example_input_awnet_attention_wavelet_network_for_isp() -> torch.Tensor:
    """
    Create an example RGB image input for AWNet.

    Returns
    -------
    torch.Tensor
        Random image tensor shaped ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "AWNet (Attention Wavelet Network for ISP)",
        build_awnet_attention_wavelet_network_for_isp,
        example_input_awnet_attention_wavelet_network_for_isp,
        2020,
        "awnet_attention_wavelet_network_for_isp",
    )
]
