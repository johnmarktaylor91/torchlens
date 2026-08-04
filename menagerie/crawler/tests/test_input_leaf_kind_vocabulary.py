"""``input_contract`` leaf ``kind`` is closed because the executor's vocabulary is.

``worker._materialize_declarative_call`` materializes exactly two leaf kinds:
``tensor``, allocated from ``shape``/``dtype``/``distribution``, and
``constructed``, built through the declarative constructor grammar. Every other
value is refused there and nowhere earlier.

The schema, however, published ``kind`` as an open ``nonempty_string``. So a
proposal could name a perfectly descriptive kind, validate, pass the independent
checker, be admitted, and only then die in the worker -- after the single
authoring visit this catalog gives a model. The rung-3 census spent ``m7362``
(``timm`` ``mobileone_s3``, an entirely ordinary R1 image model) exactly that way,
four times across its train and eval forwards, on the sole defect that its image
leaf said ``kind: "standard-image-tensor"`` instead of ``kind: "tensor"``. The
refusal text -- "contains a non-tensor leaf" -- described the leaf as something it
was not.

These tests pin the closure with the REAL archived contract that failed, in the
failing direction:

- the authored value still refuses at execution, and now names itself;
- the SAME contract with the one word corrected materializes the real tensor;
- the schema now refuses the authored value up front, where a refusal is cheap;
- the ``constructed`` arm, which the census's DGL contract used correctly, keeps
  working -- the closure is a vocabulary fix, not a narrowing of expressiveness.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION_V3, RunMode
from menagerie.crawler.schema import get_validator
from menagerie.crawler.tests.conftest import make_author_proposal
from menagerie.crawler.worker import WorkerRequest, _materialize_declarative_call

# The exact leaf ``m7362`` published, and the exact leaf it needed to publish.
_AUTHORED_LEAF: dict[str, Any] = {
    "path": "args[0]",
    "kind": "standard-image-tensor",
    "semantic_role": (
        "Batch of RGB images to classify, in NCHW layout; values are normalized "
        "image intensities."
    ),
    "shape": [1, 3, 224, 224],
    "dtype": "float32",
    "device_policy": "cpu",
    "distribution": "normal",
    "constraints": ["batch dimension is free and set to 1"],
    "source_evidence_ids": ["ev-timm-forward"],
}

_CONSTRUCTED_LEAF: dict[str, Any] = {
    "path": "args[0]",
    "kind": "constructed",
    "semantic_role": "homogeneous DGLGraph supplying the connectivity",
    "shape": [6, 6],
    "dtype": "int64",
    "device_policy": "cpu",
    "distribution": "constructor",
    "constraints": ["must be homogeneous"],
    "source_evidence_ids": ["ev-layer-ctor"],
    "constructor": {"module": "dgl", "symbol": "graph", "kwargs": {"data": [[0, 1]]}},
}


def _contract(leaf: dict[str, Any]) -> dict[str, Any]:
    """Wrap one leaf in a minimal complete declarative input contract.

    Parameters
    ----------
    leaf:
        Single positional ``input_leaf``.

    Returns
    -------
    dict[str, Any]
        Contract shaped as ``_materialize_declarative_call`` consumes it.
    """

    return {"args": [deepcopy(leaf)], "kwargs": [], "non_tensor_values": []}


def _request(contract: dict[str, Any], tmp_path: Path) -> WorkerRequest:
    """Build one worker request around a declarative contract.

    Parameters
    ----------
    contract:
        Input contract under test.
    tmp_path:
        Per-test scratch root.

    Returns
    -------
    WorkerRequest
        Request sufficient for declarative input materialization.
    """

    return WorkerRequest(
        stable_id="m7362",
        recipe={"distribution": "timm", "version": "1.0.28"},
        modality="image",
        input_spec=None,
        scratch_root=tmp_path,
        receipt_path=tmp_path / "receipt.json",
        work_id="work-m7362",
        input_contract=contract,
        device="cpu",
        framework="torch",
        mode=RunMode.EVAL,
    )


def test_the_authored_kind_still_refuses_and_now_names_itself(tmp_path: Path) -> None:
    """The executor's vocabulary is unchanged; only the diagnosis improves."""

    with pytest.raises(TypeError) as excinfo:
        _materialize_declarative_call(_request(_contract(_AUTHORED_LEAF), tmp_path))
    message = str(excinfo.value)
    assert "'args[0]'" in message
    assert "'standard-image-tensor'" in message
    assert "'tensor' or 'constructed'" in message
    # The old text asserted a falsehood about an ordinary image tensor.
    assert "contains a non-tensor leaf" not in message


def test_the_same_contract_with_the_one_word_corrected_materializes(tmp_path: Path) -> None:
    """The census model was one label away from a real forward, and nothing else."""

    torch = pytest.importorskip("torch")
    corrected = {**_AUTHORED_LEAF, "kind": "tensor"}
    args, kwargs, input_kind, _asset, _note = _materialize_declarative_call(
        _request(_contract(corrected), tmp_path)
    )
    assert kwargs == {}
    materialized = args[0]
    assert isinstance(materialized, torch.Tensor)
    assert tuple(materialized.shape) == (1, 3, 224, 224)
    assert materialized.dtype is torch.float32
    assert input_kind == "standard-image"


def test_the_schema_now_refuses_the_authored_kind_before_execution() -> None:
    """The refusal moves to proposal validation, where it costs no permanent record."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    baseline = make_author_proposal()
    assert list(validator.iter_errors(baseline)) == []
    assert baseline["proposed_facts"]["input_contract"]["args"][0]["kind"] == "tensor"

    refused = deepcopy(baseline)
    refused["proposed_facts"]["input_contract"]["args"][0]["kind"] = _AUTHORED_LEAF["kind"]
    messages = [error.message for error in validator.iter_errors(refused)]
    assert messages == ["'standard-image-tensor' is not one of ['tensor', 'constructed']"]


def test_the_closed_kinds_both_validate() -> None:
    """The closure names exactly the executor's two kinds, and refuses nothing else."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    proposal = make_author_proposal()
    contract = proposal["proposed_facts"]["input_contract"]
    contract["args"] = [deepcopy(_CONSTRUCTED_LEAF)]
    assert list(validator.iter_errors(proposal)) == []


def test_the_constructed_arm_is_untouched(tmp_path: Path) -> None:
    """Closing the vocabulary narrows labels, never the grammar's expressiveness."""

    pytest.importorskip("dgl")
    args, _kwargs, input_kind, _asset, _note = _materialize_declarative_call(
        _request(_contract(_CONSTRUCTED_LEAF), tmp_path)
    )
    assert input_kind == "standard-constructed-input"
    assert args
