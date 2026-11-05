"""Import-free probe for ``torch.nn.attention.bias.CausalBias`` (stdlib only).

``torchlens._utils._torch_compat`` evaluates ``HAS_ATTENTION_CAUSAL_BIAS`` from
this probe at import time; the flag, its snapshot row and its consumers stay
there. The probe lives in its own leaf so the compat module stays inside its
line cap.
"""

from __future__ import annotations

import importlib.util
import sys


def probe_attention_causal_bias() -> bool:
    """Return whether ``torch.nn.attention.bias.CausalBias`` is discoverable.

    NEVER executes ``torch.nn.attention.bias`` (r45/r49 lazy-import belt): that
    module's top level calls ``torch._dynamo.allow_in_graph(...)``, dragging the
    whole ``_dynamo``/``_inductor`` tree into every ``import torchlens``. When
    the USER has already imported the module, the class is inspected exactly;
    otherwise ``find_spec`` proves existence without execution (the parent
    ``torch.nn.attention`` package body is dynamo-free) and the exact
    ``__torch_function__`` site check is deferred to causal-bias shim install,
    which resolves only through ``sys.modules``.

    Returns
    -------
    bool
        ``True`` when the module is already imported and the class defines its
        own ``__torch_function__`` (the identity-dispatch site the causal-bias
        identity shim normalizes), or when the module exists but has not been
        imported yet (the shim installer re-reads ``sys.modules`` at each wrap).
        Absent on torch builds predating the ``torch.nn.attention`` namespace.

    Notes
    -----
    This probe NEVER imports ``torch.nn.attention.bias``: on torch 2.13 its
    module body reaches ``torch._dynamo``, whose import tree also drags in
    ``torch.distributed.fsdp`` -- breaking the W21 cold-start guarantee that a
    plain eager capture pays neither import. ``find_spec`` imports only the
    parent ``torch.nn.attention`` package (verified dynamo-free), and a process
    that never imported the module cannot hold a ``CausalBias`` instance, so
    the deferred read is exact, not heuristic.
    """

    module = sys.modules.get("torch.nn.attention.bias")
    if module is not None:
        causal_bias = getattr(module, "CausalBias", None)
        return causal_bias is not None and "__torch_function__" in vars(causal_bias)
    try:
        return importlib.util.find_spec("torch.nn.attention.bias") is not None
    except (ImportError, AttributeError, ValueError):
        return False
