"""Case registry for the behaviour net: classics corpus entries plus toy models.

A case is one row: a name, a model factory, an inputs factory, tags the
features consult (``inplace``, ``views``, ``branching``, ``recurrent``,
``dtype``, ``stochastic``, ``child_module``, ``float_output``, ...) and an
estimated cost in seconds. Add a case by adding a row (a toy row below, or a
corpus manifest entry); never a file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch import nn

from tests import example_models
from tests.classics_corpus._loader import CorpusEntry, build_entry, corpus_entries

BUILD_SEED = 0
INPUT_SEED = 1

#: Corpus coverage features that mean the forward draws random numbers.
_RANDOM_FUNCS = frozenset(
    {
        "func:rand",
        "func:randn",
        "func:rand_like",
        "func:randn_like",
        "func:randint",
        "func:randperm",
        "func:bernoulli",
        "func:multinomial",
        "func:normal",
        "func:poisson",
        "func:dropout",
    }
)
_INPLACE_MARKERS = ("_",)


def _always_draws(op: Any) -> bool:
    """Whether an ATen overload draws random numbers on every call.

    Torch's ``nondeterministic_seeded`` tag also covers ops whose draw is gated
    by a ``train``/``training`` flag (the dropout family, ``lstm``, ``gru``,
    ``scaled_dot_product_attention``); those are deterministic on the eval-mode
    models the net runs. A sampler that always draws takes an explicit
    ``Generator?`` argument and has no such gate.
    """

    if torch.Tag.nondeterministic_seeded not in getattr(op, "tags", ()):
        return False
    arguments = getattr(getattr(op, "_schema", None), "arguments", ())
    has_generator = any(a.name == "generator" or "Generator" in str(a.type) for a in arguments)
    gated = any(a.name in {"train", "training"} for a in arguments)
    return has_generator and not gated


def _draws_random_numbers(feature: str) -> bool:
    """Whether a ``func:<name>`` coverage feature names an op that draws random numbers.

    The hand list above is the floor; beyond it the test is torch's own
    ``nondeterministic_seeded`` tag on an ATen sampler that always draws
    (``exponential_``, ``uniform_``, ``random_``, ``cauchy_``, ..., so a
    Gumbel-softmax forward counts as stochastic without a name being spelled
    here), judged by ``_always_draws``.
    """

    if feature in _RANDOM_FUNCS:
        return True
    if not feature.startswith("func:"):
        return False
    name = feature[len("func:") :]
    candidates = (name, name[:-1]) if name.endswith("_") and not name.startswith("_") else (name,)
    for candidate in candidates:
        packet = getattr(torch.ops.aten, candidate, None)
        overloads = getattr(packet, "overloads", None)
        if not callable(overloads):
            continue
        if any(_always_draws(getattr(packet, overload_name)) for overload_name in overloads()):
            return True
    return False


_VIEW_FUNCS = frozenset(
    {
        "func:view",
        "func:reshape",
        "func:permute",
        "func:transpose",
        "func:t",
        "func:expand",
        "func:squeeze",
        "func:unsqueeze",
        "func:flatten",
        "func:__getitem__",
        "func:narrow",
        "func:select",
    }
)


@dataclass(frozen=True)
class Case:
    """One model in the net.

    Attributes
    ----------
    name:
        Unique case name, used in test ids.
    build:
        Zero-argument factory returning ``(model, inputs)`` with the model in
        eval mode and the inputs in the shape ``tl.trace`` takes positionally.
    tags:
        Coverage tags the features consult through ``has``.
    seconds:
        Estimated warm cost of one plain validate in seconds.
    source:
        ``"corpus"`` or ``"toy"``.
    """

    name: str
    build: Callable[[], tuple[nn.Module, Any]]
    tags: frozenset[str]
    seconds: float
    source: str

    def has(self, tag: str) -> bool:
        """Return whether the case carries ``tag``."""

        return tag in self.tags


def _corpus_tags(entry: CorpusEntry) -> frozenset[str]:
    """Derive the net's tags from a corpus entry's census features."""

    features = set(entry.features)
    tags: set[str] = {"corpus"}
    if "trace:is_recurrent" in features:
        tags.add("recurrent")
    if "trace:is_branching" in features:
        tags.add("branching")
    if "trace:has_conditional_branching" in features:
        tags.add("conditional")
    if any(f.startswith("func:") and f.endswith(_INPLACE_MARKERS) for f in features):
        tags.add("inplace")
    if features & _VIEW_FUNCS:
        tags.add("views")
    if any(f.startswith("dtype:") and f not in {"dtype:float32"} for f in features):
        tags.add("dtype")
    if any(_draws_random_numbers(feature) for feature in features):
        tags.add("stochastic")
    if any(f.startswith("mod:") for f in features):
        tags.add("child_module")
    if "dtype:float32" in features or "dtype:float64" in features:
        tags.add("float_output")
    tags.add("tier:" + entry.tier)
    return frozenset(tags)


def _corpus_case(entry: CorpusEntry) -> Case:
    """Wrap one corpus entry as a case."""

    def build() -> tuple[nn.Module, Any]:
        return build_entry(entry)

    row = _manifest_row(entry.id)
    seconds = float(row.get("warm_validate_seconds") or row.get("census_seconds") or 5.0)
    return Case(entry.id, build, _corpus_tags(entry), seconds, "corpus")


@cache
def _manifest_rows() -> dict[str, dict[str, Any]]:
    """Return the manifest rows by entry id."""

    from tests.classics_corpus._loader import load_manifest

    return {row["id"]: row for row in load_manifest()["entries"]}


def _manifest_row(entry_id: str) -> dict[str, Any]:
    """Return one manifest row."""

    return _manifest_rows()[entry_id]


def _toy(
    name: str,
    model_cls: type[nn.Module],
    inputs: Callable[[], Any],
    *tags: str,
    seconds: float = 0.3,
) -> Case:
    """Build a toy case row from ``tests/example_models.py``."""

    def build() -> tuple[nn.Module, Any]:
        torch.manual_seed(BUILD_SEED)
        model = model_cls()
        model.eval()
        torch.manual_seed(INPUT_SEED)
        return model, inputs()

    return Case(f"toy:{name}", build, frozenset({"toy", "float_output", *tags}), seconds, "toy")


def _img() -> torch.Tensor:
    return torch.rand(2, 3, 8, 8)


def _img224() -> torch.Tensor:
    return torch.rand(1, 3, 224, 224)


def _vec() -> torch.Tensor:
    return torch.rand(5, 5)


def _four() -> torch.Tensor:
    return torch.rand(4, 4)


def _three_imgs() -> list[torch.Tensor]:
    return [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]


#: Toy rows: the hand-built shapes `tests/example_models.py` keeps. Each row names
#: the model, its inputs and the tags the features consult.
TOY_CASES: tuple[Case, ...] = (
    _toy("simple_ff", example_models.SimpleFF, _img),
    _toy("inplace_funcs", example_models.InPlaceFuncs, _img, "inplace", "child_module"),
    _toy("batchnorm", example_models.BatchNormModel, _img, "inplace", "child_module"),
    _toy("concat", example_models.ConcatTensors, _img),
    _toy("split", example_models.SplitTensor, _img224, "views"),
    _toy("multi_inputs", example_models.MultiInputs, _three_imgs, "containers"),
    _toy("list_input", example_models.ListInput, lambda: [_three_imgs()], "containers"),
    _toy(
        "dict_input",
        example_models.DictInput,
        lambda: [{"x": _img(), "y": _img(), "z": _img()}],
        "containers",
    ),
    _toy("multi_outputs", example_models.MultiOutputs, _img, "containers"),
    _toy("dict_output", example_models.DictOutput, _img, "containers"),
    _toy("nested_output", example_models.NestedOutput, _img, "containers"),
    _toy("buffer", example_models.BufferModel, lambda: torch.rand(12, 12), "stochastic"),
    _toy("simple_branching", example_models.SimpleBranching, _img, "branching"),
    _toy(
        "conditional_branching",
        example_models.ConditionalBranching,
        _img,
        "branching",
        "conditional",
    ),
    _toy("repeated_module", example_models.RepeatedModule, _img, "recurrent", "child_module"),
    _toy("nested_modules", example_models.NestedModules, _img, "recurrent", "child_module"),
    _toy("loop_no_param", example_models.SimpleLoopNoParam, _img, "recurrent"),
    _toy(
        "recurrent_params",
        example_models.RecurrentParamsSimple,
        _vec,
        "recurrent",
        "child_module",
    ),
    _toy("lstm", example_models.LSTMModel, _vec, "recurrent", "child_module"),
    _toy("rnn", example_models.RNNModel, _vec, "recurrent", "child_module"),
    _toy(
        "view_mutation_transpose",
        example_models.ViewMutationTranspose,
        _vec,
        "views",
        "inplace",
        "child_module",
    ),
    _toy("mixed_dtype", example_models.MixedDtypeModel, _four, "dtype", "views"),
    _toy("type_cast_chain", example_models.TypeCastChainModel, _four, "dtype", "child_module"),
    _toy("getitem_tracking", example_models.GetItemTracking, _vec, "views"),
)


@cache
def corpus_cases() -> tuple[Case, ...]:
    """Return every classics corpus entry as a case, in manifest order."""

    return tuple(_corpus_case(entry) for entry in corpus_entries())


@cache
def smoke_cases() -> tuple[Case, ...]:
    """Return the cheap corpus cases: the manifest's smoke tier, the 13 cheapest measured."""

    return tuple(case for case in corpus_cases() if case.has("tier:smoke"))


@cache
def all_cases() -> tuple[Case, ...]:
    """Return every case: corpus entries then toys."""

    return (*corpus_cases(), *TOY_CASES)


def case_by_name(name: str) -> Case:
    """Return the case called ``name``."""

    for case in all_cases():
        if case.name == name:
            return case
    raise KeyError(name)
