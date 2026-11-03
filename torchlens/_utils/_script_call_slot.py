"""Probe for the TorchScript call boundary: can ``ScriptFunction.__call__`` be replaced.

``backends/torch/_script_boundary.py`` records a call to a ``torch.jit.ScriptFunction``
made from an eager forward as one boundary op by replacing ``__call__`` on the pybind
type. That works only when the type is a heap type that owns its ``__call__`` (true of
every pybind11 class). The ``HAS_SCRIPT_FUNCTION_CALL_HOOK`` flag and its
capability-snapshot row stay in ``torchlens._utils._torch_compat``, which imports this leaf
at probe time so its eager import closure does not grow (the sibling of
``_type_sequence_slot`` and ``_type_new_slot``).
"""

from __future__ import annotations

import torch

_HEAPTYPE_FLAG = 1 << 9
"""CPython ``Py_TPFLAGS_HEAPTYPE``: the type's slots may be reassigned from Python."""


def probe_script_function_call_hook() -> bool:
    """Return whether calls to ``torch.jit.ScriptFunction`` can be intercepted.

    Returns
    -------
    bool
        True when ``torch.jit.ScriptFunction`` is a heap type with its own callable
        ``__call__``; False degrades the TorchScript boundary to today's census gap.
    """

    script_function = getattr(getattr(torch, "jit", None), "ScriptFunction", None)
    if not isinstance(script_function, type):
        return False
    if not callable(script_function.__dict__.get("__call__")):
        return False
    return bool(script_function.__flags__ & _HEAPTYPE_FLAG)
