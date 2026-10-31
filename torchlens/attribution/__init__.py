"""Input-attribution methods for TorchLens."""

from torchlens.attribution import onebackward
from torchlens.attribution._cam import grad_cam
from torchlens.attribution._causal_audit import (
    CausalAudit,
    CausalAuditSet,
    CausalAuditUnit,
    causal_audit,
)
from torchlens.attribution._core import (
    Attribution,
    AttributionError,
    input_x_grad,
    integrated_gradients,
    saliency,
    smoothgrad,
)
from torchlens.attribution._curves import (
    CurveStep,
    PerturbationCurve,
    deletion_curve,
    insertion_curve,
)
from torchlens.attribution._gradient_shap import gradient_shap
from torchlens.attribution._guided import deconvolution, guided_backprop
from torchlens.attribution._layer import (
    layer_attribution,
    layer_conductance,
    layer_integrated_gradients,
)
from torchlens.attribution._metrics import AttributionMetric, infidelity, sensitivity
from torchlens.attribution._noise_tunnel import noise_tunnel
from torchlens.attribution._occlusion import occlusion, occlusion_map
from torchlens.attribution._randomization import (
    RandomizationStep,
    RandomizationTest,
    randomization_test,
)
from torchlens.attribution._road import road_curve
from torchlens.attribution._site import SiteTarget
from torchlens.attribution._stash import SiteStash
from torchlens.attribution._text import TokenAttribution, TokenAttributionPayload, text
from torchlens.attribution._viz import overlay
from torchlens.attribution.graph import AttributionGraph, GraphNode, build_graph
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
    "AttributionGraph",
    "Attribution",
    "CausalAudit",
    "CausalAuditSet",
    "CausalAuditUnit",
    "causal_audit",
    "PerturbationCurve",
    "CurveStep",
    "DEFAULT",
    "ReadError",
    "ReadRow",
    "ReadTable",
    "build_graph",
    "deconvolution",
    "deletion_curve",
    "grad_cam",
    "gradient_shap",
    "GraphNode",
    "guided_backprop",
    "input_x_grad",
    "insertion_curve",
    "integrated_gradients",
    "layer_attribution",
    "layer_conductance",
    "infidelity",
    "layer_integrated_gradients",
    "load_read_table",
    "AttributionMetric",
    "noise_tunnel",
    "occlusion",
    "occlusion_map",
    "onebackward",
    "overlay",
    "randomization_test",
    "RandomizationTest",
    "RandomizationStep",
    "read",
    "road_curve",
    "saliency",
    "seed",
    "sensitivity",
    "SiteStash",
    "SiteTarget",
    "smoothgrad",
    "text",
    "TokenAttributionPayload",
    "TokenAttribution",
]
