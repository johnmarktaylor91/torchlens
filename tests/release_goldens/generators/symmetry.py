# ruff: noqa -- harvest-time provenance script, committed as run (lint header added at import)
from torchlens.data_classes._state_adapter import state_new, state_restore
from torchlens.data_classes.op import Op

print("Op has __dict__?", "__dict__" in dir(Op), "| __slots__ present:", hasattr(Op, "__slots__"))
o = state_new(Op)
try:
    state_restore(o, {"a_field_a_future_release_added": 1})
    print("TOLERANT: unknown pickled field accepted")
except AttributeError as e:
    print("INTOLERANT: unknown pickled field ->", type(e).__name__, e)
print(
    "dropped_edge_tensor_args in main Op slots:",
    "dropped_edge_tensor_args" in getattr(Op, "__slots__", ()),
)
