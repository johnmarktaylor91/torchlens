"""Hand-pinned independent facts for the ``tests/golden/`` file suite.

b9-opus R75-2: each ``tests/golden/*.json`` file is regenerated THROUGH
torchlens, so its consuming test detects CHANGE, not correctness. These pins
read the committed golden FILES as plain JSON and assert facts derived from
first principles — the generator models' own source and the documented
``<func>_<type_index>_<op_index>`` label convention — never from a
regeneration run.

Generator models (facts hand-derived from their forward source):

- ``selector_semantics_matrix.json``: ``TinyConvNet`` in
  ``test_selector_semantics_matrix.py`` — features Sequential(Conv2d, ReLU,
  Conv2d), then ``torch.relu``, ``x + 1``, ``flatten``, Linear head. Op order:
  conv2d, relu, conv2d, relu, add, flatten, linear.
- ``rank_render_ir_semantics.json``: ``_ModuleDictQuoteKeyModel`` in
  ``test_render_dotid_cert10.py`` — one Sequential(Linear, ReLU, Linear).
- ``viz_render_identity_oracle.json``: ``OracleCNN`` in
  ``test_viz_render_identity_oracle.py`` — ``pool(relu(conv(x)))``.

Updating an expected value below requires arguing from the MODEL, not from
observed output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GOLDEN_DIR = Path(__file__).parent / "golden"

#: TinyConvNet's full op-label universe, written down from its forward source
#: and the documented label convention: two convs (type indexes 1, 2), two
#: relus, then add, flatten, linear at op indexes 5-7, plus the boundary nodes.
_TINY_CONV_UNIVERSE = [
    "add_1_5",
    "conv2d_1_1",
    "conv2d_2_3",
    "flatten_1_6",
    "input_1",
    "linear_1_7",
    "output_1",
    "relu_1_2",
    "relu_2_4",
]


def _load(name: str) -> Any:
    """Read one committed golden file as plain JSON.

    Parameters
    ----------
    name:
        Golden file name inside ``tests/golden/``.

    Returns
    -------
    Any
        Decoded JSON payload.
    """

    return json.loads((_GOLDEN_DIR / name).read_text(encoding="utf-8"))


def test_selector_matrix_golden_rows_match_hand_computed_truth_table() -> None:
    """Hand-computed selector truth-table rows hold in the committed golden.

    Each expected match set is derived from TinyConvNet's forward source: the
    model contains exactly two relu calls, one ``+`` (add) call, and the nine
    ops of ``_TINY_CONV_UNIVERSE`` — so ``func("relu")``, the add selectors,
    the raw-substring universe row, and ``not func("relu")`` follow from
    first principles.
    """

    golden = _load("selector_semantics_matrix.json")
    assert golden["capture/conv/func_relu"] == ["relu_1_2", "relu_2_4"]
    assert golden["capture/conv/func_add_dunder"] == ["add_1_5"]
    assert golden["capture/conv/func_add_type"] == ["add_1_5"]
    assert golden["capture/conv/contains_raw_sub"] == _TINY_CONV_UNIVERSE
    expected_not_relu = [label for label in _TINY_CONV_UNIVERSE if not label.startswith("relu")]
    assert golden["capture/conv/not_func"] == expected_not_relu


def test_rank_render_golden_edges_match_hand_derived_chain() -> None:
    """The rank-render golden's edge set is the hand-derived five-node chain.

    ``_ModuleDictQuoteKeyModel`` runs one Sequential(Linear, ReLU, Linear), so
    the unrolled single-pass graph is input -> linear -> relu -> linear ->
    output with pass-1 suffixes and nothing else.
    """

    golden = _load("rank_render_ir_semantics.json")
    edges = {(edge["source"], edge["target"]) for edge in golden["edges"]}
    assert edges == {
        ("input_1pass1", "linear_1_1pass1"),
        ("linear_1_1pass1", "relu_1_2pass1"),
        ("relu_1_2pass1", "linear_2_3pass1"),
        ("linear_2_3pass1", "output_1pass1"),
    }
    node_names = sorted(node["name"] for node in golden["nodes"])
    assert node_names == [
        "input_1pass1",
        "linear_1_1pass1",
        "linear_2_3pass1",
        "output_1pass1",
        "relu_1_2pass1",
    ]


def test_viz_identity_golden_cnn_case_matches_hand_derived_chain() -> None:
    """The viz-identity CNN case pins OracleCNN's hand-derived structure.

    ``OracleCNN.forward`` is exactly ``pool(relu(conv(x)))``: three compute
    ops in that order between the input and output boundary nodes.
    """

    golden = _load("viz_render_identity_oracle.json")
    structural = golden["record"]["cases"]["cnn_rolled_profiling"]["structural"]
    edges = [(edge["src_name"], edge["dst_name"]) for edge in structural["edges"]]
    assert edges == [
        ("input_1", "conv2d_1_1"),
        ("conv2d_1_1", "relu_1_2"),
        ("relu_1_2", "maxpool2d_1_3"),
        ("maxpool2d_1_3", "output_1"),
    ]
    node_names = {node["name"] for node in structural["nodes"]}
    assert node_names == {"input_1", "conv2d_1_1", "relu_1_2", "maxpool2d_1_3", "output_1"}
