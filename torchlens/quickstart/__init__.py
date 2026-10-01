"""TorchLens quickstart: one input ladder, one provenance record, one render door.

The quickstart package (megasprint lane F17; quickstart panel memo of record
2026-08-26) owns the three-rung input ladder shared by ``tl.summary``,
``tl.render``, and ``tl.trace``:

- rung 1 (gold): a real input -- TorchLens guesses nothing;
- rung 2: ``input_size=`` -- your shape, synthesized values, disclosed;
- rung 3: nothing -- inferred shape, synthesized values, disclosed or a teach.

Every synthesized input is disclosed in a provenance record that survives
save/load on the existing ``Trace.input_preprocessor`` KEEP field, and the
capability gate refuses derived-semantics claims on synthesized values.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint; the
stable spellings are the verbs themselves.
"""

from __future__ import annotations

from ._gate import (
    SynthesizedValueReadWarning,
    internal_read,
    is_gold,
    require_gold,
    warn_on_raw_read,
)
from ._grammar import InputSpec, parse_input_size, synthesize
from ._plan import InputPlan, normalize_gold_args, resolve_rung
from ._primitive import ExecutionReceipt, capture_concrete, state_dict_hash
from ._provenance import (
    InputProvenance,
    TensorFacts,
    provenance_from_record,
    provenance_to_record,
    trace_input_provenance,
)
from ._render import RenderResult, render
from ._resolve import ResolvedInputs, attach_provenance, resolve_inputs

__tl_layer__ = "FACADE"

#: Typed root-surface registration fragment for lane F35 (merge law s5:
#: feature lanes submit registrations in their own packages; F35 lands them
#: on the root facade). Keys are requested ``tl.<name>`` spellings; values
#: are ``(home_module, attribute)`` rows in ``_LAZY_ATTRS`` form.
ROOT_SURFACE_REGISTRATION: dict[str, tuple[str, str]] = {
    "render": ("torchlens.user_funcs", "render"),
    "RenderResult": ("torchlens.quickstart", "RenderResult"),
}

__all__ = [
    "ExecutionReceipt",
    "InputPlan",
    "InputProvenance",
    "InputSpec",
    "RenderResult",
    "ResolvedInputs",
    "SynthesizedValueReadWarning",
    "TensorFacts",
    "attach_provenance",
    "capture_concrete",
    "internal_read",
    "is_gold",
    "normalize_gold_args",
    "parse_input_size",
    "provenance_from_record",
    "provenance_to_record",
    "render",
    "require_gold",
    "resolve_inputs",
    "resolve_rung",
    "state_dict_hash",
    "synthesize",
    "trace_input_provenance",
    "warn_on_raw_read",
]
