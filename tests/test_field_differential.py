"""Proof that the field differential sees what it must and hides only the fixed mask.

``tools/field_differential.py`` is the instrument that shows a refactor left
every public field value as it was. An instrument is trusted only after it
has been shown to report faults, so this file plants three kinds of fault (a
changed value, a changed alias, a missing field) and checks that each is
reported at the field it was planted in, and that a clean tree compared with
itself reports nothing. The mask list is pinned here too: it was fixed once,
and a new entry must be a visible change to this file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "field_differential.py"

sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib import mask as mask_module  # noqa: E402
from field_diff_lib.canonical import Canonicalizer, tensor_hash  # noqa: E402
from field_diff_lib.compare import diff, field_key, name_key  # noqa: E402
from field_diff_lib.faults import FAULTS, PLANT_CASE  # noqa: E402
from field_diff_lib.mask import Mask  # noqa: E402

#: Entries of the fixed mask, and the digest of their lines (see ``_mask_lines``).
APPROVED_MASK_ENTRIES = 72
APPROVED_MASK_DIGEST = "b99af388732d79a2"


def _mask_lines() -> list[str]:
    """Return one line per mask entry, text and not-read entries included."""

    lines = [f"name {rule.field} {rule.rule}" for rule in mask_module.MASK]
    lines += [f"text {rule.pattern} {rule.replacement}" for rule in mask_module.TEXT_RULES]
    read_on = ",".join(mask_module.GEOMETRY_READ_MODELS)
    lines += [f"not-read {name} except {read_on}" for name in mask_module.NOT_READ]
    tags = ";".join(
        f"{name}:" + ",".join(f"{head}={base}" for head, base in sorted(names.items()))
        for name, names in sorted(mask_module.TYPE_TAG_ONLY.items())
    )
    lines.append(f"type-tag-only {tags}")
    classes = ",".join(mask_module.RELEASE_CLASSES)
    lines.append(f"release-order {mask_module.RELEASE_ORDER} steps/outcome {classes}")
    lines += [
        f"release-later {name} steps " + ",".join(sorted(steps))
        for name, steps in mask_module.RELEASE_LATER_CLASSES.items()
    ]
    counts = ";".join(f"{name}:{bucket}" for name, bucket in sorted(mask_module.COUNT_ONLY.items()))
    lines.append(f"count-only {counts}")
    return sorted(lines)


def test_mask_is_the_approved_list() -> None:
    """The mask is fixed: adding or changing an entry must change this pin on purpose."""

    lines = _mask_lines()
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    assert (len(lines), digest) == (APPROVED_MASK_ENTRIES, APPROVED_MASK_DIGEST), "\n".join(lines)


def test_every_mask_entry_states_its_rule_and_reason() -> None:
    """Each entry names a known rule and why the value differs between two runs of one commit."""

    names = [rule.field for rule in mask_module.MASK]
    assert len(names) == len(set(names))
    for rule in mask_module.MASK:
        assert rule.rule in mask_module.RULES
        assert len(rule.reason) > 10
    for text_rule in mask_module.TEXT_RULES:
        assert len(text_rule.reason) > 10
    assert len(mask_module.TYPE_TAG_ONLY_REASON) > 10
    assert len(mask_module.RELEASE_ORDER_REASON) > 10
    assert len(mask_module.RELEASE_LATER_REASON) > 10
    assert len(mask_module.COUNT_ONLY_REASON) > 10


def test_diff_reports_value_type_length_and_missing_key() -> None:
    """The comparison reports each kind of difference with its path."""

    base = {"a": 1, "b": [1, 2], "c": {"x": "s"}, "d": 1}
    head = {"a": 2, "b": [1, 2, 3], "c": {"x": None}, "e": 1}
    found = {path: (left, right) for path, left, right in diff(base, head)}
    assert found == {
        "/a": (1, 2),
        "/b/<len>": (2, 3),
        "/c/x": ("s", None),
        "/d": (1, "<absent>"),
        "/e": ("<absent>", 1),
    }
    assert diff(base, base) == []


def test_diff_keeps_mapping_order_and_names_entries_by_key() -> None:
    """A reordered mapping is a difference; an equal key is named in the path."""

    base = {"__dict__": "dict", "items": [["k", 1], ["m", 2]]}
    changed = {"__dict__": "dict", "items": [["k", 1], ["m", 3]]}
    swapped = {"__dict__": "dict", "items": [["m", 2], ["k", 1]]}
    assert [path for path, _, _ in diff(base, changed)] == ["/items[1]/m"]
    assert len(diff(base, swapped)) == 4
    path = "/traces/trace/ops/relu_1_2:1/args_template/fields/items[0]/total_s/__float__"
    assert field_key(path) == "ops/args_template"
    assert name_key(path) == "ops/args_template :: total_s"


def test_clock_rule_hides_only_the_reading() -> None:
    """A clock field keeps presence, counts and shape, so a reading that turns to ``None`` shows."""

    mask = Mask()
    assert mask.apply("func_duration", {"__float__": "0.25"}) == "<float>"
    assert mask.apply("func_duration", {"__float__": "Duration('36.2 us')"}) == "<float>"
    assert mask.apply("func_duration", None) is None
    raised = {"__raises__": "AttributeError", "code": None, "message": "gone"}
    assert mask.apply("func_duration", raised) == raised
    timings = {"__dict__": "dict", "items": [["relu", {"__float__": "0.1"}], ["count", 3]]}
    assert mask.apply("grad_fn_fire_timings", timings)["items"] == [
        ["relu", "<float>"],
        ["count", 3],
    ]
    assert mask.apply("backward_durations", [{"__float__": "1.0"}]) == ["<float>"]


def test_reading_rule_hides_every_number_and_keeps_presence() -> None:
    """Backward ``peak_memory`` hides integers and floats; ``None`` and shape still compare."""

    mask = Mask()
    assert mask.apply("peak_memory", 196608) == "<reading>"
    assert mask.apply("peak_memory", 0) == "<reading>"
    assert mask.apply("peak_memory", {"__float__": "1.5"}) == "<reading>"
    assert mask.apply("peak_memory", None) is None
    assert mask.apply("peak_memory", [1, None]) == ["<reading>", None]
    assert mask.apply("forward_peak_memory", 196608) == 196608


def test_type_tag_only_reads_the_subclass_as_its_base() -> None:
    """The four row-key containers compare as their base type; items and other tags still show."""

    mask = Mask()
    seq = {"__seq__": "OpSequence", "items": [{"__ref__": "Op", "label": "relu_1"}]}
    assert mask.apply("_layer_list", seq) == [{"__ref__": "Op", "label": "relu_1"}]
    other = {"__seq__": "SomethingElse", "items": []}
    assert mask.apply("_layer_list", other) == other
    assert mask.apply("_other_list", seq) == seq
    labels = {"__dict__": "OpOrderedLabelMap", "items": [["relu_1", 1]]}
    assert mask.apply("_layer_dict_all_keys", labels)["__dict__"] == "OrderedDict"
    assert mask.apply("_layer_dict_all_keys", labels)["items"] == [["relu_1", 1]]
    plain = {"__dict__": "OpLabelMap", "items": []}
    assert mask.apply("_layer_dict_main_keys", plain)["__dict__"] == "dict"
    layers = {"__dict__": "LayerStore", "items": []}
    assert mask.apply("_layer_logs", layers)["__dict__"] == "OrderedDict"
    assert mask.apply("_layer_logs", {"__dict__": "dict", "items": []})["__dict__"] == "dict"
    loaded = {"__dict__": "LayerDictStore", "items": [["relu_1", 1]]}
    assert mask.apply("_layer_logs", loaded) == {"__dict__": "dict", "items": [["relu_1", 1]]}
    assert mask.apply("_layer_dict_main_keys", loaded) == loaded
    nested = {"__dict__": "dict", "items": [["_layer_logs", layers]]}
    assert mask.apply("trace", nested)["items"][0][1]["__dict__"] == "OrderedDict"
    assert Mask(enabled=False).apply("_layer_list", seq) == seq


def _release(**classes: list[str]) -> dict[str, Any]:
    """Return a canonical release outcome with the given classes (others empty)."""

    items = [[key, classes.get(key, [])] for key in mask_module.RELEASE_CLASSES]
    return {"__dict__": "dict", "items": [*items, ["watched_by_address", []]]}


def _release_log(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a one-step sequence-8 log."""

    return [{"step": "fork_held", "outcome": outcome, "warnings": []}]


def test_release_order_accepts_only_earlier_release() -> None:
    """A record freed earlier is accepted; a move toward the collector still fails."""

    name = mask_module.RELEASE_ORDER
    base = _release(
        freed_when_trace_dropped=["op"],
        freed_when_kept_dropped=["layer"],
        freed_only_by_collector=["param"],
    )
    earlier = _release(freed_when_trace_dropped=["layer", "op", "param"])
    accepted = mask_module.accept_release_moves(name, _release_log(base), _release_log(earlier))
    assert diff(_release_log(base), accepted) == []

    toward = _release(freed_when_kept_dropped=["layer", "op"], freed_only_by_collector=["param"])
    kept = mask_module.accept_release_moves(name, _release_log(base), _release_log(toward))
    assert diff(_release_log(base), kept) != []

    collector = _release(
        freed_when_trace_dropped=["op"], freed_only_by_collector=["layer", "param"]
    )
    assert not mask_module.release_moved_earlier(base, collector)
    never = _release(freed_when_trace_dropped=["op", "layer"], never_freed=["param"])
    assert not mask_module.release_moved_earlier(base, never)
    extra = _release(freed_when_trace_dropped=["layer", "op", "param", "buffer"])
    assert not mask_module.release_moved_earlier(base, extra)
    fewer = _release(freed_when_trace_dropped=["layer", "op"])
    assert not mask_module.release_moved_earlier(base, fewer)
    address = {**earlier, "items": [*earlier["items"][:-1], ["watched_by_address", ["op"]]]}
    assert not mask_module.release_moved_earlier(base, address)
    dumped = mask_module.accept_release_moves(
        name, {"steps": _release_log(base)}, {"steps": _release_log(earlier)}
    )
    assert diff({"steps": _release_log(base)}, dumped) == []
    dumped_toward = mask_module.accept_release_moves(
        name, {"steps": _release_log(base)}, {"steps": _release_log(toward)}
    )
    assert diff({"steps": _release_log(base)}, dumped_toward) != []
    other = mask_module.accept_release_moves(
        "sequence-5_forks", _release_log(base), _release_log(earlier)
    )
    assert diff(_release_log(base), other) != []


def test_release_later_is_accepted_only_in_a_step_that_keeps_the_record() -> None:
    """A held record may keep its trace: trace-dropped to kept-dropped, in its own steps only."""

    name = mask_module.RELEASE_ORDER
    base = _release(freed_when_trace_dropped=["op", "trace"], freed_when_kept_dropped=["param"])
    later = _release(freed_when_kept_dropped=["op", "param", "trace"])
    for step in ("params_and_buffers_held", "modules_held", "facets_read", "every_door_held"):
        assert mask_module.release_moved_earlier(base, later, step)
        log_base = [{"step": step, "outcome": base, "warnings": []}]
        log_head = [{"step": step, "outcome": later, "warnings": []}]
        assert diff(log_base, mask_module.accept_release_moves(name, log_base, log_head)) == []
    assert not mask_module.release_moved_earlier(base, later)
    assert not mask_module.release_moved_earlier(base, later, "fork_held")
    kept = mask_module.accept_release_moves(name, _release_log(base), _release_log(later))
    assert diff(_release_log(base), kept) != []
    collector = _release(freed_when_kept_dropped=["op", "param"], freed_only_by_collector=["trace"])
    assert not mask_module.release_moved_earlier(base, collector, "params_and_buffers_held")
    never_base = _release(freed_when_trace_dropped=["op"], never_freed=["trace"])
    never_head = _release(freed_when_kept_dropped=["op", "trace"])
    assert not mask_module.release_moved_earlier(never_base, never_head, "modules_held")


def test_param_tuple_reads_as_the_plain_tuple_tag() -> None:
    """``_param_logs`` held as a ``ParamTuple`` reads as the tuple it replaces, elsewhere not."""

    mask = Mask()
    held = {"__seq__": "ParamTuple", "items": [{"__ref__": "param:fc.weight"}]}
    assert mask.apply("_param_logs", held) == {"__tuple__": held["items"]}
    assert mask.apply("parent_params", held) == held
    state = {"__dict__": "StateDict", "items": [["1", 2]]}
    assert mask.apply("_grad_fn_logs", state)["__dict__"] == "OrderedDict"
    assert mask.apply("_backward_pass_logs", state)["__dict__"] == "OrderedDict"


def test_identity_rule_keeps_presence_and_shape() -> None:
    """An address is hidden; whether it is there, and how many there are, is compared."""

    mask = Mask()
    assert mask.apply("grad_fn_object_id", 140351636512656) == "<id>"
    assert mask.apply("grad_fn_object_id", None) is None
    assert mask.apply("next_grad_fn_ids", [140351636512999, 7]) == ["<id>", "<id>"]
    assert mask.apply("backend_handle_id", "140350999223472") == "<id>"
    assert mask.apply("backend_handle_id", "not-a-number") == "not-a-number"


def test_id_keys_rule_masks_only_the_keys() -> None:
    """A mapping keyed by address hides its keys and compares every value under them."""

    record = {"__dataclass__": "GradFn", "fields": {"name": "AddBackward0", "index": 3}}
    value = {"__dict__": "dict", "items": [[140351636512656, record], ["label", 7]]}
    assert Mask().apply("_grad_fn_logs", value) == {
        "__dict__": "dict",
        "items": [["<id>", record], ["label", 7]],
    }
    assert Mask(enabled=False).apply("_grad_fn_logs", value) == value


def test_address_rule_leaves_small_numbers_alone() -> None:
    """Only address-sized integers are hidden inside a mixed value."""

    value = {"__tuple__": [140351813906736, 18, 271, "name", 140351813906736]}
    assert Mask().apply("_code_context_cache", value) == {
        "__tuple__": ["<id>", 18, 271, "name", "<id>"]
    }


def test_barcode_rule_keeps_equality_between_fields() -> None:
    """A random identifier is numbered by first appearance, in one table per trace."""

    mask = Mask()
    assert mask.apply("handle", "hook-6a53") == "<barcode#0>"
    assert mask.apply("handles", {"__tuple__": ["hook-2fd3", "hook-6a53"]}) == {
        "__tuple__": ["<barcode#1>", "<barcode#0>"]
    }
    assert Mask().apply("handle", "hook-2fd3") == "<barcode#0>"


def test_named_and_text_rules_reach_inside_values() -> None:
    """Named entries are masked inside mappings and dataclasses; durations inside reprs."""

    mask = Mask()
    value = {
        "__dataclass__": "Event",
        "fields": {
            "backend_handle_id": "140350999223472",
            "label": "relu_1_2",
            "timings": {"__dict__": "dict", "items": [["total_s", {"__float__": "0.1"}]]},
            "cost": {"__object__": "TraceCost", "repr": "TraceCost of m: 489 us compute, 192 B"},
        },
    }
    masked = mask.apply("event_stream", value)["fields"]
    assert masked["backend_handle_id"] == "<id>"
    assert masked["label"] == "relu_1_2"
    assert masked["timings"]["items"] == [["total_s", "<float>"]]
    assert masked["cost"]["repr"] == "TraceCost of m: <duration> compute, 192 B"
    share = "captured  <duration> and its logging share"
    assert mask.text("captured  19.0 ms (18.5 ms of it logging)") == share
    assert mask.text("captured  19.0 ms, almost all logging") == share


def test_disabled_mask_changes_nothing() -> None:
    """With the mask off, every value passes through as it is."""

    value = {"__dict__": "dict", "items": [["total_s", {"__float__": "0.1"}]]}
    assert Mask(enabled=False).apply("func_duration", value) == value


def test_tensor_form_separates_content_identity_and_storage() -> None:
    """Equal content, the same object and shared storage are three different facts."""

    canon = Canonicalizer([], frozenset())
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = base[1]
    clone = base.clone()
    first, same, shared, copied = (canon.tensor(t) for t in (base, base, view, clone))
    assert first["object"] == same["object"] != copied["object"]
    assert first["sha256"] == copied["sha256"]
    assert first["storage"] == shared["storage"] != copied["storage"]
    assert (shared["storage_offset"], shared["strides"]) == (3, [1])
    assert first["requires_grad"] is False and first["has_grad_fn"] is False
    tracked = (base.clone().requires_grad_() * 2).sum()
    assert canon.tensor(tracked)["has_grad_fn"] is True


def test_tensor_hash_reads_sparse_content() -> None:
    """Sparse tensors hash their indices and values, so a changed entry shows."""

    indices = torch.tensor([[0, 1], [1, 0]])
    one = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 2.0]), (2, 2))
    two = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 3.0]), (2, 2))
    assert tensor_hash(one) == tensor_hash(one.clone())
    assert tensor_hash(one) != tensor_hash(two)


def test_canonical_strings_drop_addresses_and_roots() -> None:
    """Paths under a known root and hex addresses do not reach a dump."""

    canon = Canonicalizer([("/somewhere/checkout", "<repo>")], frozenset())
    text = "<function f at 0x7fdc140db250> in /somewhere/checkout/torchlens/x.py"
    assert canon.string(text) == "<function f at <addr>> in <repo>/torchlens/x.py"
    assert canon.value(0.1) == {"__float__": "0.1"}
    assert canon.value((1, "a")) == {"__tuple__": [1, "a"]}


def _dump(out: Path, *extra: str) -> dict[str, Any]:
    """Dump the fault case of the current tree in a fresh process and load it."""

    command = [sys.executable, str(TOOL), "dump", "--out", str(out), "--what", "fields"]
    command += ["--cases", PLANT_CASE, *extra]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    loaded: dict[str, Any] = json.loads((out / f"case-{PLANT_CASE}.json").read_text())
    return loaded


@pytest.fixture(scope="module")
def clean_dump(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Return the clean dump of the fault case, taken once."""

    return _dump(tmp_path_factory.mktemp("clean"))


@pytest.mark.heavy
def test_same_tree_twice_has_no_differences(clean_dump: dict[str, Any], tmp_path: Path) -> None:
    """Two processes of one tree differ nowhere outside the mask."""

    assert "case_raises" not in clean_dump
    assert diff(clean_dump, _dump(tmp_path)) == []


#: The field each planted fault must be reported in (family/field, as the report names it).
EXPECTED_FAULT_FIELDS = {
    "value": "ops/out",
    "meta": "ops/func_name",
    "alias": "ops/out",
    "missing": "ops/func_name",
    "absent": "ops/func_name",
}


@pytest.mark.heavy
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_fault_is_reported(fault: str, clean_dump: dict[str, Any], tmp_path: Path) -> None:
    """Each planted fault is reported, in the field it was planted in."""

    found = diff(clean_dump, _dump(tmp_path, "--plant", fault))
    reported = sorted({field_key(path) for path, _, _ in found})
    assert reported, f"planted fault {fault!r} was not reported"
    assert EXPECTED_FAULT_FIELDS[fault] in reported, reported[:40]


@pytest.mark.heavy
def test_two_tree_command_exit_status(tmp_path: Path) -> None:
    """``selfcheck`` exits 0 on a clean tree; with a planted fault it exits 0 only if reported."""

    base = [sys.executable, str(TOOL), "selfcheck", "--what", "fields", "--cases", PLANT_CASE]
    clean = subprocess.run(
        [*base, "--work", str(tmp_path / "clean")],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert clean.returncode == 0, clean.stdout[-2000:] + clean.stderr[-2000:]
    assert "field differential: 0 differences" in clean.stdout
    planted = subprocess.run(
        [*base, "--work", str(tmp_path / "planted"), "--plant", "value"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert planted.returncode == 0, planted.stdout[-2000:] + planted.stderr[-2000:]
    assert "planted fault 'value': reported" in planted.stdout
