"""Input-attribution methods for TorchLens."""

from torchlens.attribution import onebackward
from torchlens.attribution._core import (
    AttributionError,
    AttributionResult,
    input_x_grad,
    integrated_gradients,
    saliency,
    smoothgrad,
)
from torchlens.attribution._gradient_shap import gradient_shap
from torchlens.attribution._guided import deconvolution, guided_backprop
from torchlens.attribution._layer import (
    grad_cam,
    layer_attribution,
    layer_conductance,
    layer_integrated_gradients,
)
from torchlens.attribution._metrics import MetricResult, infidelity, sensitivity
from torchlens.attribution._noise_tunnel import noise_tunnel
from torchlens.attribution._occlusion import occlusion, occlusion_map
from torchlens.attribution._stash import SiteStash
from torchlens.attribution._text import TokenAttributionPayload, TokenAttributionResult, text
from torchlens.attribution._viz import overlay
from torchlens.attribution.onebackward import (
    DEFAULT,
    ReadError,
    ReadRow,
    ReadTable,
    load_read_table,
    read,
    seed,
)

__all__ = [
    "AttributionError",
    "AttributionResult",
    "DEFAULT",
    "ReadError",
    "ReadRow",
    "ReadTable",
    "deconvolution",
    "grad_cam",
    "gradient_shap",
    "guided_backprop",
    "input_x_grad",
    "integrated_gradients",
    "layer_attribution",
    "layer_conductance",
    "infidelity",
    "layer_integrated_gradients",
    "load_read_table",
    "MetricResult",
    "noise_tunnel",
    "occlusion",
    "occlusion_map",
    "onebackward",
    "overlay",
    "read",
    "saliency",
    "seed",
    "sensitivity",
    "SiteStash",
    "smoothgrad",
    "text",
    "TokenAttributionPayload",
    "TokenAttributionResult",
]
