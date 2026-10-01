"""Re-export shim: ``TraceState`` moved to :mod:`torchlens._vocab.trace_state`.

Ratified V2 relocation (architecture memo 3.3, C01 item 6). Every historical
spelling and the pickle-visible identity stay unchanged; new lower-layer
consumers import from the vocabulary home.
"""

from ._vocab.trace_state import TraceState

__tl_layer__ = "FACADE"
__all__ = ["TraceState"]
