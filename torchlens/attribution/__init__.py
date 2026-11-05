"""Input-attribution methods for TorchLens."""

from typing import TYPE_CHECKING, Any

from torchlens._private_submodules import (
    refuse_private_submodule as _refuse_private_submodule,
)
from torchlens.attribution._cam import grad_cam
from torchlens.attribution._causal_audit import (
    CausalAudit,
    CausalAuditSet,
    CausalAuditUnit,
    causal_audit,
)
from torchlens.attribution._core import (
    Attribution,
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
from torchlens.attribution._graph import (
    AttributionGraph,
    GraphNode,
    build_graph,
    upload_to_neuronpedia,
)
from torchlens.attribution._guided import deconvolution, guided_backprop
from torchlens.attribution._layer import (
    layer_attribution,
    layer_conductance,
    layer_integrated_gradients,
)
from torchlens.attribution._metrics import AttributionMetric, infidelity, sensitivity
from torchlens.attribution._noise_tunnel import noise_tunnel
from torchlens.attribution._occlusion import occlusion, occlusion_map
from torchlens.attribution._onebackward import (
    ReadRow,
    ReadTable,
    load_read_table,
    read,
    seed,
)
from torchlens.attribution._randomization import (
    RandomizationStep,
    RandomizationTest,
    randomization_test,
)
from torchlens.attribution._road import road_curve
from torchlens.attribution._site import SiteTarget
from torchlens.attribution._text import TokenAttribution, TokenAttributionPayload, text

__all__ = [
    "AttributionGraph",
    "Attribution",
    "CausalAudit",
    "CausalAuditSet",
    "CausalAuditUnit",
    "causal_audit",
    "PerturbationCurve",
    "CurveStep",
    "ReadRow",
    "ReadTable",
    "build_graph",
    "upload_to_neuronpedia",
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
    "randomization_test",
    "RandomizationTest",
    "RandomizationStep",
    "read",
    "road_curve",
    "saliency",
    "seed",
    "sensitivity",
    "SiteTarget",
    "smoothgrad",
    "text",
    "TokenAttributionPayload",
    "TokenAttribution",
]


# Runtime only: under TYPE_CHECKING a module ``__getattr__`` would make every
# name on this module type-check as Any.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Refuse a submodule path that 3.0 made private, typed; miss plainly otherwise.

        Parameters
        ----------
        name:
            Attribute normal lookup did not find.

        Raises
        ------
        AttributeError
            Always; ``FacadeTeachingError`` (``facade_redirect``) for a private
            submodule's old public spelling.
        """

        _refuse_private_submodule(__name__, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
