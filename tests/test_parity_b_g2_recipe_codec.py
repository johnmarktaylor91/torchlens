"""Replay recipe section: save, load and bind against the NumPy fake adapter (no framework).

Covers the section written at ``executable_with_callables``, the bind on the first
validation-status read, and every honest refusal: an op without a recipe names
the op as unverified; a framework version mismatch, an unreadable section, a
recipe outside the allowlist and a backend mismatch report unavailable; a trace
loaded without the section keeps the old behaviour.
"""

from __future__ import annotations

import gc
import json
import types
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support import replay_fake_adapter as fake
from torchlens._io import _replay_recipe_adapter as binder, _replay_recipe_codec as codec
from torchlens._io.format3.payloads import PayloadStore
from torchlens._validation_status import is_known_validation_reason

_VERSION = "fake-9.9"


class _LoadedTrace:
    """Stand-in for a loaded trace: backend, load marker and a capture slot."""

    def __init__(self, backend: str = "jax") -> None:
        self._provenance_backend = backend
        self._loaded_from_bundle = True
        self.payload_load_status = None
        self.captures: dict[str, Any] = {}
        self.params: dict[str, Any] = {}


def _fake_trace() -> fake.FakeTrace:
    ops = [
        fake.FakeOp("x", "input", (), ("x",)),
        fake.FakeOp("y", "mul", ("x", "x"), ("y",)),
        fake.FakeOp("z", "scale", ("y",), ("z",), literals=((("factor",), 2.5),)),
        fake.FakeOp("s", "split2", ("z",), ("s", "s_1"), attrs={"index": 1}),
        fake.FakeOp("o", "opaque", ("s",), ("o",)),
    ]
    trace = fake.build_trace(ops[:4], {"x": np.arange(3.0)})
    trace.ops["o"] = ops[4]
    trace._layer_list = list(trace.ops.values())  # type: ignore[attr-defined]
    return trace


def _bind_capturing(trace: Any, recipes: Any) -> None:
    trace.captures = dict(recipes)


def _adapter(**overrides: Any) -> types.SimpleNamespace:
    functions = {name: getattr(fake, name) for name in ("can_replay", "recipe", "bind_recipes")}
    functions["bind_recipes"] = _bind_capturing
    functions.update(overrides)
    return types.SimpleNamespace(ALLOWLIST=fake.ALLOWLIST, **functions)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """The fake adapter served for ``jax``, with a pinned framework version."""

    module = _adapter()
    monkeypatch.setattr(binder, "get_adapter", lambda name: module)
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: _VERSION)
    return module


def _section(adapter: Any, trace: Any | None = None) -> dict[str, Any]:
    document = binder.build_section(trace or _fake_trace(), adapter, "jax")
    # The fake's recipes say backend "fake"; the section is saved for "jax".
    for row in document["recipes"]:
        row["backend"] = "jax"
    return document


def _load(tmp_path: Path, document: Any, backend: str = "jax") -> _LoadedTrace:
    text = document if isinstance(document, str) else json.dumps(document)
    (tmp_path / codec.REPLAY_RECIPES_FILE).write_text(text, encoding="utf-8")
    loaded = _LoadedTrace(backend)
    codec.attach_replay_recipes(loaded, tmp_path, {"files": {codec.REPLAY_RECIPES_FILE: {}}})
    return loaded


def test_section_records_recipes_and_not_replayable_ops(adapter: Any) -> None:
    document = _section(adapter)
    assert document["format"] == codec.SECTION_FORMAT
    assert document["version"] == codec.SECTION_VERSION
    assert document["framework_version"] == _VERSION
    slots = [tuple(row["output_slots"]) for row in document["recipes"]]
    assert slots == [("x",), ("y",), ("z",), ("s", "s_1")]
    assert document["missing"] == []
    assert [row["op_label"] for row in document["not_replayable"]] == ["o"]


def test_load_binds_recipes_once_with_repeated_parents_and_multi_output(
    tmp_path: Path, adapter: Any
) -> None:
    loaded = _load(tmp_path, _section(adapter))
    assert binder.loaded_replay_recipe_status(loaded) is None
    assert set(loaded.captures) == {"x", "y", "z", "s"}
    assert loaded.captures["y"].parent_labels == ("x", "x")
    assert loaded.captures["s"].output_slots == ("s", "s_1")
    assert loaded.captures["z"].literals == ((("factor",), 2.5),)
    loaded.captures = {}
    assert binder.loaded_replay_recipe_status(loaded) is None
    assert loaded.captures == {}, "a bound section is never bound twice"


def test_op_without_recipe_reports_unverified_by_name(tmp_path: Path, adapter: Any) -> None:
    def recipe(trace: Any, op: Any) -> Any:
        return None if op.label == "z" else fake.recipe(trace, op)

    document = _section(_adapter(recipe=recipe))
    loaded = _load(tmp_path, document)
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None
    assert (status.state, status.is_available, status.source) == ("unverified", False, "loaded")
    assert "z" in status.message
    assert [row["op_label"] for row in status.decisions] == ["z"]
    assert status.unverified_reason_counts == {"no_replayable_function": 1}
    assert is_known_validation_reason(status.reason)
    assert loaded.captures == {}, "nothing is bound while an op lacks a recipe"


def test_raising_adapter_recipe_is_recorded_and_named(tmp_path: Path, adapter: Any) -> None:
    def recipe(trace: Any, op: Any) -> Any:
        if op.label == "y":
            raise RuntimeError("cannot describe y")
        return fake.recipe(trace, op)

    loaded = _load(tmp_path, _section(_adapter(recipe=recipe)))
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unverified"
    assert "cannot describe y" in status.decisions[0]["detail"]


def test_recipe_of_a_session_refused_op_is_kept_for_the_validator(
    tmp_path: Path, adapter: Any
) -> None:
    def recipe(trace: Any, op: Any) -> Any:
        if op.label == "o":
            return fake.OpRecipe(
                backend="fake",
                op_identity="opaque",
                inputs=(fake.InputOccurrence("s", (0,)),),
                output_slots=("o",),
            )
        return fake.recipe(trace, op)

    document = _section(_adapter(recipe=recipe))
    assert document["not_replayable"] == []
    loaded = _load(tmp_path, document)
    assert binder.loaded_replay_recipe_status(loaded) is None
    assert loaded.captures["o"].op_identity == "opaque"


def test_raising_recipe_of_a_refused_op_stays_not_replayable(adapter: Any) -> None:
    def recipe(trace: Any, op: Any) -> Any:
        if op.label == "o":
            raise RuntimeError("no capture")
        return fake.recipe(trace, op)

    document = _section(_adapter(recipe=recipe))
    assert document["missing"] == []
    assert document["not_replayable"] == [{"op_label": "o", "detail": "opaque region"}]


def test_framework_version_mismatch_is_unavailable(
    tmp_path: Path, adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _load(tmp_path, _section(adapter))
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: "fake-10.0")
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None
    assert (status.state, status.is_available) == ("unavailable", False)
    assert _VERSION in status.message and "fake-10.0" in status.message
    assert is_known_validation_reason(status.reason)
    assert loaded.captures == {}


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.update(version=2), "section version"),
        (lambda d: d.update(format="other"), "format"),
        (lambda d: d.update(recipes={}), "recipes"),
        (lambda d: d["recipes"][1].update(op_identity="os_system"), "(op y)"),
        (lambda d: d["recipes"][1].update(format_version=99), "format_version"),
    ],
)
def test_corrupt_section_is_unavailable_naming_cause(
    tmp_path: Path, adapter: Any, mutate: Any, needle: str
) -> None:
    document = _section(adapter)
    mutate(document)
    loaded = _load(tmp_path, document)
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unavailable"
    assert needle in status.message
    assert loaded.captures == {}


def test_unparsable_section_is_unavailable(tmp_path: Path, adapter: Any) -> None:
    loaded = _load(tmp_path, "{not json")
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unavailable"
    assert "unreadable" in status.message


def test_section_for_another_backend_is_unavailable(tmp_path: Path, adapter: Any) -> None:
    loaded = _load(tmp_path, _section(adapter), backend="tf")
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unavailable"
    assert "'jax'" in status.message


def test_raising_bind_is_unavailable_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bind(trace: Any, recipes: Any) -> None:
        raise KeyError("payload of y")

    module = _adapter(bind_recipes=bind)
    monkeypatch.setattr(binder, "get_adapter", lambda name: module)
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: _VERSION)
    loaded = _load(tmp_path, _section(module))
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unavailable"
    assert "KeyError" in status.message


def test_unlisted_section_file_is_ignored(tmp_path: Path, adapter: Any) -> None:
    (tmp_path / codec.REPLAY_RECIPES_FILE).write_text(json.dumps(_section(adapter)))
    loaded = _LoadedTrace()
    codec.attach_replay_recipes(loaded, tmp_path, {"files": {}})
    assert codec.loaded_section(loaded) is None
    assert binder.loaded_replay_recipe_status(loaded) is None


def test_section_released_with_trace(tmp_path: Path, adapter: Any) -> None:
    loaded = _load(tmp_path, _section(adapter))
    ref = weakref.ref(loaded)
    before = len(codec._LOADED_SECTIONS)
    del loaded
    gc.collect()
    assert ref() is None
    assert len(codec._LOADED_SECTIONS) == before - 1


def test_write_only_at_replay_level_for_adapter_backends(tmp_path: Path, adapter: Any) -> None:
    trace = _fake_trace()
    assert not binder.write_replay_recipes(tmp_path, trace, backend="jax", save_level="portable")
    assert not binder.write_replay_recipes(
        tmp_path, trace, backend="torch", save_level="executable_with_callables"
    )
    assert not (tmp_path / codec.REPLAY_RECIPES_FILE).exists()
    assert binder.write_replay_recipes(
        tmp_path, trace, backend="jax", save_level="executable_with_callables"
    )
    written = json.loads((tmp_path / codec.REPLAY_RECIPES_FILE).read_text())
    assert len(written["recipes"]) == 4


def test_loaded_trace_resaves_its_section_unchanged(tmp_path: Path, adapter: Any) -> None:
    document = _section(adapter)
    loaded = _load(tmp_path, document)
    out = tmp_path / "resave"
    out.mkdir()
    assert binder.write_replay_recipes(
        out, loaded, backend="jax", save_level="executable_with_callables"
    )
    assert json.loads((out / codec.REPLAY_RECIPES_FILE).read_text()) == document


_PARAMS = {
    "layer.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
    "layer.bias": np.array([0.5, -1.5], dtype=np.float64),
    "scale": np.array(3, dtype=np.int32),
}


def _bind_with_params(trace: Any, recipes: Any, *, param_payloads: Any) -> None:
    trace.captures = dict(recipes)
    trace.params = dict(param_payloads)


def _write_params_section(root: Path, monkeypatch: pytest.MonkeyPatch, values: Any) -> Any:
    module = _adapter(live_param_payloads=lambda trace: values, bind_recipes=_bind_with_params)
    monkeypatch.setattr(binder, "get_adapter", lambda name: module)
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: _VERSION)
    store = PayloadStore(root)
    assert binder.write_replay_recipes(
        root, _fake_trace(), backend="jax", save_level="executable_with_callables", store=store
    )
    store.close()
    document = json.loads((root / codec.REPLAY_RECIPES_FILE).read_text())
    for row in document["recipes"]:
        row["backend"] = "jax"
    (root / codec.REPLAY_RECIPES_FILE).write_text(json.dumps(document))
    return document


def _attach(root: Path) -> _LoadedTrace:
    loaded = _LoadedTrace()
    codec.attach_replay_recipes(loaded, root, {"files": {codec.REPLAY_RECIPES_FILE: {}}})
    return loaded


def test_parameter_values_round_trip_into_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _write_params_section(tmp_path, monkeypatch, _PARAMS)
    assert set(document["param_payloads"]) == set(_PARAMS)
    assert "param_payload_problem" not in document
    text = (tmp_path / codec.REPLAY_RECIPES_FILE).read_text()
    assert "0.5" not in text, "parameter values live in payload blobs, not in the section"
    loaded = _attach(tmp_path)
    assert binder.loaded_replay_recipe_status(loaded) is None
    assert set(loaded.params) == set(_PARAMS)
    for address, value in _PARAMS.items():
        got = loaded.params[address]
        assert got.dtype == value.dtype and got.shape == value.shape
        np.testing.assert_array_equal(got, value)


def test_loaded_trace_resaves_its_parameter_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    _write_params_section(first, monkeypatch, _PARAMS)
    loaded = _attach(first)
    loaded._loaded_from_bundle = True
    second = tmp_path / "second"
    second.mkdir()
    store = PayloadStore(second)
    assert binder.write_replay_recipes(
        second, loaded, backend="jax", save_level="executable_with_callables", store=store
    )
    store.close()
    again = _attach(second)
    assert binder.loaded_replay_recipe_status(again) is None
    for address, value in _PARAMS.items():
        np.testing.assert_array_equal(again.params[address], value)


def test_raising_parameter_report_is_recorded_and_nothing_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def live(trace: Any) -> Any:
        raise RuntimeError("no parameter capture")

    module = _adapter(live_param_payloads=live)
    monkeypatch.setattr(binder, "get_adapter", lambda name: module)
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: _VERSION)
    assert binder.write_replay_recipes(
        tmp_path, _fake_trace(), backend="jax", save_level="executable_with_callables"
    )
    document = json.loads((tmp_path / codec.REPLAY_RECIPES_FILE).read_text())
    assert "param_payloads" not in document
    assert "no parameter capture" in document["param_payload_problem"]


def test_parameter_values_without_a_store_are_recorded_as_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _adapter(live_param_payloads=lambda trace: _PARAMS)
    monkeypatch.setattr(binder, "get_adapter", lambda name: module)
    monkeypatch.setattr(binder, "installed_framework_version", lambda name: _VERSION)
    assert binder.write_replay_recipes(
        tmp_path, _fake_trace(), backend="jax", save_level="executable_with_callables"
    )
    document = json.loads((tmp_path / codec.REPLAY_RECIPES_FILE).read_text())
    assert "payload store" in document["param_payload_problem"]


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda refs: refs["layer.weight"].update(dtype="<f8"), "saved as float64"),
        (lambda refs: refs["layer.weight"].update(key="missing"), "parameter values"),
        (lambda refs: refs.update(scale="not a reference"), "no blob reference"),
    ],
)
def test_corrupt_parameter_reference_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, needle: str
) -> None:
    document = _write_params_section(tmp_path, monkeypatch, _PARAMS)
    mutate(document["param_payloads"])
    (tmp_path / codec.REPLAY_RECIPES_FILE).write_text(json.dumps(document))
    loaded = _attach(tmp_path)
    status = binder.loaded_replay_recipe_status(loaded)
    assert status is not None and status.state == "unavailable"
    assert needle in status.message
    assert loaded.captures == {}


def test_param_payloads_field_must_be_an_object(tmp_path: Path, adapter: Any) -> None:
    document = _section(adapter)
    document["param_payloads"] = []
    status = binder.loaded_replay_recipe_status(_load(tmp_path, document))
    assert status is not None and status.state == "unavailable"
    assert "param_payloads" in status.message


def test_bind_gets_no_parameter_keyword_without_saved_values(tmp_path: Path, adapter: Any) -> None:
    # The two-argument _bind_capturing would raise TypeError on a keyword.
    loaded = _load(tmp_path, _section(adapter))
    assert binder.loaded_replay_recipe_status(loaded) is None
    assert set(loaded.captures) == {"x", "y", "z", "s"}


def test_installed_framework_version_reads_metadata_only() -> None:
    assert binder.installed_framework_version("torch") == ""
    assert isinstance(binder.installed_framework_version("jax"), str)
