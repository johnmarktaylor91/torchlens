"""The native contrastive direction builder and its library converters.

``torchlens.intervention.direction_from_pairs`` on a tiny seeded model, checked
against oracles computed without TorchLens: forward hooks read the rows, and
each method's math is re-derived independently (an eigendecomposition for
``"pca"``, an L-BFGS solve of the same objective for ``"logistic"``). The
converters are checked against stand-in library modules here; the real
packages run in ``tests/test_bridge_real_steering_vectors.py`` and
``tests/test_bridge_real_repeng.py``.
"""

from __future__ import annotations

import hashlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import Direction, _directions, direction_from_pairs
from torchlens.selection_subspace import BasisProvenance

_HIDDEN = 8
_SITE = "layers.1"


class _TinyLM(nn.Module):
    """Embedding, two tanh blocks and a head; the mask zeroes padded rows."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(13, _HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN) for _ in range(2)])
        self.head = nn.Linear(_HIDDEN, 13)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.emb(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask[..., None]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.head(hidden)


def _ids(rows: int, seed: int) -> torch.Tensor:
    """Return seeded ``[rows, 5]`` token ids."""

    return torch.randint(0, 13, (rows, 5), generator=torch.Generator().manual_seed(seed))


def _trace(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor | None = None) -> Any:
    """Trace with every layer saved, passing the mask as an input when given."""

    capture = tl.options.CaptureOptions(layers_to_save="all")
    if mask is None:
        return tl.trace(model, ids, capture=capture)
    kwargs = {"input_ids": ids, "attention_mask": mask}
    return tl.trace(model, (), input_kwargs=kwargs, capture=capture)


def _hooked_rows(
    model: nn.Module, ids: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Read ``layers[1]``'s output at one position per row with a forward hook."""

    seen: list[torch.Tensor] = []
    handle = model.layers[1].register_forward_hook(lambda _m, _a, out: seen.append(out.detach()))
    try:
        with torch.no_grad():
            model(ids, mask)
    finally:
        handle.remove()
    return seen[0][torch.arange(ids.shape[0]), positions]


@pytest.fixture(scope="module")
def stack() -> Iterator[dict[str, Any]]:
    """A seeded tiny model with traced positive and negative batches."""

    torch.manual_seed(0)
    model = _TinyLM().eval()
    pos_ids, neg_ids = _ids(6, 1), _ids(6, 2)
    stack = {
        "model": model,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "pos": _trace(model, pos_ids),
        "neg": _trace(model, neg_ids),
    }
    last = torch.full((6,), 4)
    stack["rows"] = (
        _hooked_rows(model, pos_ids, last),
        _hooked_rows(model, neg_ids, last),
    )
    try:
        yield stack
    finally:
        stack["pos"].cleanup()
        stack["neg"].cleanup()


def _logistic_oracle(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Solve the logistic objective by L-BFGS, independently of the Newton solver."""

    rows = torch.cat([positive - (positive + negative) / 2, negative - (positive + negative) / 2])
    rows = rows.to(torch.float64)
    labels = torch.cat([torch.ones(positive.shape[0]), -torch.ones(negative.shape[0])]).double()
    weight = torch.zeros(rows.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [weight], max_iter=2000, tolerance_grad=1e-14, tolerance_change=1e-16, history_size=50
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = 0.5 * weight @ weight + torch.nn.functional.softplus(-labels * (rows @ weight)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    found = weight.detach()
    return found / found.norm()


def test_mean_diff_is_the_mean_row_difference(stack: dict[str, Any]) -> None:
    """``mean_diff`` is ``(pos - neg).mean(0)`` of the hooked last-token rows, bit for bit."""

    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE)
    positive, negative = stack["rows"]
    assert isinstance(direction, Direction)
    assert direction.vector.dtype == torch.float32
    assert torch.equal(direction.vector, (positive - negative).mean(dim=0))
    assert direction.n_pairs == 6 and direction.site == _SITE


def test_pca_is_the_top_eigenvector_of_the_differences(stack: dict[str, Any]) -> None:
    """``pca`` is the top eigenvector of D^T D, unit norm, signed toward the positives."""

    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE, "pca")
    positive, negative = stack["rows"]
    deltas = (positive - negative).double()
    _values, vectors = torch.linalg.eigh(deltas.T @ deltas)
    oracle = vectors[:, -1]
    oracle = oracle if float((deltas @ oracle).mean()) > 0 else -oracle
    torch.testing.assert_close(direction.vector.double(), oracle, atol=1e-5, rtol=0)
    assert float((deltas @ direction.vector.double()).mean()) > 0


@pytest.mark.parametrize("pairs", [3, 12], ids=["fewer_rows_than_features", "more_rows"])
def test_logistic_matches_an_independent_solve(pairs: int) -> None:
    """``logistic`` matches L-BFGS on the same objective, on both Newton solve paths."""

    torch.manual_seed(0)
    model = _TinyLM().eval()
    pos_ids, neg_ids = _ids(pairs, 3), _ids(pairs, 4)
    pos, neg = _trace(model, pos_ids), _trace(model, neg_ids)
    try:
        direction = direction_from_pairs(pos, neg, _SITE, "logistic")
    finally:
        pos.cleanup()
        neg.cleanup()
    last = torch.full((pairs,), 4)
    oracle = _logistic_oracle(
        _hooked_rows(model, pos_ids, last), _hooked_rows(model, neg_ids, last)
    )
    torch.testing.assert_close(direction.vector.double(), oracle, atol=1e-5, rtol=0)


def test_provenance_reuses_the_basis_record(stack: dict[str, Any]) -> None:
    """The record is a BasisProvenance whose digest re-derives from the vector."""

    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE, "pca")
    record = direction.provenance
    canonical = direction.vector.double().reshape(1, -1)
    digest = hashlib.sha256()
    digest.update(repr(tuple(canonical.shape)).encode())
    digest.update(canonical.numpy().tobytes())
    assert isinstance(record, BasisProvenance)
    assert record.sha256 == digest.hexdigest()
    assert (record.method, record.n_vectors, record.space_dim, record.dim) == ("pca", 1, 8, -1)
    assert f"site={_SITE!r}" in record.origin and "n_pairs=6" in record.origin
    assert "pairs_sha256=" in record.origin
    assert "sha256=" in repr(direction) and "tensor" not in repr(direction)


def test_sequence_of_traces_equals_one_batched_trace(stack: dict[str, Any]) -> None:
    """Traces of single prompts concatenate to the batched trace's direction."""

    model = stack["model"]
    singles = [_trace(model, stack["pos_ids"][i : i + 1]) for i in range(6)]
    try:
        split = direction_from_pairs(singles, stack["neg"], _SITE, "pca")
    finally:
        for trace in singles:
            trace.cleanup()
    batched = direction_from_pairs(stack["pos"], stack["neg"], _SITE, "pca")
    torch.testing.assert_close(split.vector, batched.vector, atol=1e-6, rtol=0)


def test_padded_rows_read_the_last_real_token(stack: dict[str, Any]) -> None:
    """With a saved mask, ``-1`` reads each right-padded prompt's last real token."""

    model = stack["model"]
    mask = torch.ones(6, 5, dtype=torch.long)
    mask[0, 3:] = 0
    mask[2, 2:] = 0
    pos, neg = _trace(model, stack["pos_ids"], mask), _trace(model, stack["neg_ids"], mask)
    try:
        direction = direction_from_pairs(pos, neg, _SITE)
    finally:
        pos.cleanup()
        neg.cleanup()
    last = mask.sum(dim=1) - 1
    rows = [_hooked_rows(model, ids, last, mask) for ids in (stack["pos_ids"], stack["neg_ids"])]
    assert torch.equal(direction.vector, (rows[0] - rows[1]).mean(dim=0))


def test_steer_edit_adds_the_scaled_vector(stack: dict[str, Any]) -> None:
    """``direction.steer(m)`` placed at the site adds ``m * vector`` to its output."""

    model = stack["model"]
    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE, "pca")
    with torch.no_grad():
        steered = tl.when(tl.module(_SITE), direction.steer(3.0)).bind(model)(stack["pos_ids"])
    handle = model.layers[1].register_forward_hook(lambda _m, _a, out: out + 3.0 * direction.vector)
    try:
        with torch.no_grad():
            oracle = model(stack["pos_ids"])
    finally:
        handle.remove()
    assert torch.equal(steered, oracle)


def _fake_module(name: str, **attrs: Any) -> types.ModuleType:
    """Build a stand-in library module."""

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_to_steering_vectors_wraps_the_vector(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converter keys the direction's own vector by the site's layer index."""

    class SteeringVector:
        def __init__(self, layer_activations: dict[int, torch.Tensor], layer_type: str) -> None:
            self.layer_activations = layer_activations
            self.layer_type = layer_type

    monkeypatch.setitem(
        sys.modules,
        "steering_vectors",
        _fake_module("steering_vectors", SteeringVector=SteeringVector),
    )
    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE)
    result = direction.to_steering_vectors()
    assert list(result.layer_activations) == [1] and result.layer_type == "decoder_block"
    assert torch.equal(result.layer_activations[1], direction.vector)
    assert result.layer_activations[1] is not direction.vector


def test_to_repeng_wraps_the_vector(stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The repeng converter carries float32 values and an explicit model type."""

    class ControlVector:
        def __init__(self, model_type: str, directions: dict[int, Any]) -> None:
            self.model_type = model_type
            self.directions = directions

    monkeypatch.setitem(sys.modules, "repeng", _fake_module("repeng", ControlVector=ControlVector))
    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE)
    result = direction.to_repeng(model_type="tiny")
    assert result.model_type == "tiny" and list(result.directions) == [1]
    assert result.directions[1].dtype.name == "float32"
    assert torch.equal(torch.from_numpy(result.directions[1]), direction.vector)
    assert _code(direction.to_repeng) == "direction_model_type_unknown"


def _code(call: Any) -> str:
    """Return the refusal code a call raises."""

    with pytest.raises(InvalidArgumentError) as info:
        call()
    fields = info.value.fields
    assert fields["remedy"]
    assert str(info.value).rstrip().endswith(f"{fields['remedy']}.")
    return str(fields["code"])


def test_builder_refusals(stack: dict[str, Any]) -> None:
    """Each builder refusal carries its code and a remedy."""

    pos, neg = stack["pos"], stack["neg"]
    assert _code(lambda: direction_from_pairs(pos, neg, _SITE, "svd")) == "direction_method_unknown"
    assert _code(lambda: direction_from_pairs([], neg, _SITE)) == "direction_pairs_empty"
    assert _code(lambda: direction_from_pairs(pos, pos, _SITE)) == "direction_pairs_identical"
    assert (
        _code(lambda: direction_from_pairs(pos, neg, _SITE, read_token_index=[1, 2]))
        == "direction_read_token_count"
    )
    short = _trace(stack["model"], stack["neg_ids"][:4])
    try:
        assert _code(lambda: direction_from_pairs(pos, short, _SITE)) == "direction_pairs_mismatch"
    finally:
        short.cleanup()


def test_reader_refusals(stack: dict[str, Any]) -> None:
    """Rank, mask and unsaved-site refusals carry their codes."""

    flat = torch.zeros(3, _HIDDEN)
    assert _code(lambda: _directions._read_rows(flat, -1, None, "positive")) == (
        "direction_read_token_rank"
    )
    indices = torch.full((3,), -1)
    assert (
        _code(
            lambda: _directions._adjust_for_padding(indices, torch.ones(3, 4), (3, 5), "positive")
        )
        == "direction_mask_shape"
    )
    empty = torch.ones(3, 5)
    empty[1] = 0
    assert (
        _code(lambda: _directions._adjust_for_padding(indices, empty, (3, 5), "positive"))
        == "direction_mask_empty_row"
    )
    unsaved = tl.trace(
        stack["model"],
        stack["pos_ids"],
        capture=tl.options.CaptureOptions(layers_to_save=["head"]),
    )
    try:
        assert _code(lambda: direction_from_pairs(unsaved, stack["neg"], _SITE)) == (
            "direction_site_unsaved"
        )
    finally:
        unsaved.cleanup()
    wide = (torch.zeros(2, 4), torch.zeros(2, 5))
    assert _code(lambda: _directions._check_pairs(*wide)) == "direction_pairs_mismatch"


def test_logistic_refuses_when_unconverged(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of Newton steps refuses instead of returning a partial solve."""

    monkeypatch.setattr(_directions, "_LOGISTIC_MAX_ITER", 1)
    assert (
        _code(lambda: direction_from_pairs(stack["pos"], stack["neg"], _SITE, "logistic"))
        == "direction_logistic_unconverged"
    )


def test_library_layer_refusals(stack: dict[str, Any]) -> None:
    """A site without an index needs layer=; a layer that disagrees is refused."""

    direction = direction_from_pairs(stack["pos"], stack["neg"], _SITE)
    assert direction.library_layer() == 1 and direction.library_layer(1) == 1
    assert _code(lambda: direction.library_layer(0)) == "direction_layer_mismatch"
    assert _code(lambda: direction.library_layer(True)) == "direction_layer_unknown"
    head = direction_from_pairs(stack["pos"], stack["neg"], "head")
    assert _code(head.library_layer) == "direction_layer_unknown"
    assert head.library_layer(1) == 1
