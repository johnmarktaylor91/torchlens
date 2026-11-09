"""tinygrad's op spelling list (see :mod:`torchlens.backends._op_spelling`).

tinygrad records each data-flow UOp under the tinygrad API call that made it
when the observer saw one, else the UOp's op name lowercased (``cmplt``,
``reduce``; see ``_uop_graph._op_func_name``). A matmul is reshape, expand,
mul and reduce, and a relu is cmplt and where: the list names the UOps, never
the decomposition. An unlisted name falls back to its own name with family
``other``. Used only for candidate preference and drawing labels; never for
pairing.
"""

from __future__ import annotations

_SAME = [
    "add",
    "sub",
    "mul",
    "neg",
    "where",
    "exp2",
    "log2",
    "sin",
    "sqrt",
    "reciprocal",
    "pow",
    "reshape",
    "permute",
    "expand",
    "pad",
    "flip",
    "detach",
    "contiguous",
    "xor",
    "or",
    "and",
    "trunc",
]

#: tinygrad op name to the canonical torch spelling.
SPELLINGS: dict[str, str] = {name: name for name in _SAME} | {
    # Ops.MAX is the elementwise binary maximum.
    "max": "maximum",
    "cmplt": "lt",
    "cmpne": "ne",
    "cmpeq": "eq",
    "recip": "reciprocal",
    "shrink": "getitem",
    "cast": "to",
    "idiv": "div",
    "fdiv": "div",
    "mod": "remainder",
    "copy": "clone",
}

#: tinygrad op names with no canonical torch spelling, and their families.
UNMAPPED: dict[str, str] = {
    "const": "source",
    "reduce": "reduce",
    "reduce_axis": "reduce",
    "bitcast": "cast",
    "mulacc": "arith",
    "shl": "arith",
    "shr": "arith",
    "threefry": "other",
    "assign": "identity",
}
