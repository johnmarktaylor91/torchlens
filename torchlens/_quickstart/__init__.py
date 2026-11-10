"""TorchLens quickstart: input resolution, input provenance and the drawing entry point.

The quickstart package resolves the model input for ``tl.summary``, ``tl.draw``
and ``tl.trace`` in one of three ways:

- a real input: TorchLens guesses nothing;
- ``input_shape=``: your shape with random values, disclosed;
- nothing: an inferred shape with random values, disclosed, or an error that says why.

Every random input is disclosed in a provenance record that survives save and
load on the ``Trace.provenance.input_preprocessor`` field, and analyses that need
real data refuse to run on random values.

Names in this private package may change; the public entry points are the
functions themselves.
"""

from __future__ import annotations

from ..viz._drawing import Drawing
from ._gate import (
    internal_read,
    is_gold,
    require_gold,
    warn_on_raw_read,
)
from ._grammar import InputSpec, parse_input_size, synthesize
from ._plan import InputPlan, normalize_gold_args, resolve_rung
from ._primitive import ExecutionReceipt
from ._provenance import (
    InputProvenance,
    TensorFacts,
    provenance_from_record,
    provenance_to_record,
    trace_input_provenance,
)
from ._render import draw
from ._resolve import ResolvedInputs, attach_provenance, resolve_inputs

__tl_layer__ = "FACADE"

#: Typed root-surface registration fragment (feature packages submit registrations in their own
#: packages; the root facade collects them). Keys are requested ``tl.<name>`` spellings; values are
#: ``(home_module, attribute)`` rows in ``_LAZY_ATTRS`` form.
ROOT_SURFACE_REGISTRATION: dict[str, tuple[str, str]] = {
    "draw": ("torchlens._user_funcs", "draw"),
    "Drawing": ("torchlens.viz", "Drawing"),
}

__all__ = [
    "ExecutionReceipt",
    "InputPlan",
    "InputProvenance",
    "InputSpec",
    "Drawing",
    "ResolvedInputs",
    "TensorFacts",
    "attach_provenance",
    "internal_read",
    "is_gold",
    "normalize_gold_args",
    "parse_input_size",
    "provenance_from_record",
    "provenance_to_record",
    "draw",
    "require_gold",
    "resolve_inputs",
    "resolve_rung",
    "synthesize",
    "trace_input_provenance",
    "warn_on_raw_read",
]
