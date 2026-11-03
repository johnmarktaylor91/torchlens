"""Io-verb fills for the behaviour net: persistence promises and new toy rows.

The features that persist or export a capture (``save``, ``storage``,
``extract``, ``export``, ``validate``) get an ``after`` hook that checks the
feature's whole promise through the public surface:

* ``save``: the selected ops and nothing else carry payloads, live and loaded;
  every save level round-trips to the same trace or refuses typed; a re-save
  of the loaded trace is byte-identical to the first save; a lazy load reads
  only the payloads asked for; an artifact stamped by torchlens 2.x refuses
  with the one typed error; the bundle loads in another process to the same
  trace.
* ``storage``: every streamed payload equals the plain capture's tensor (no
  op receives another op's tensor); the stream bundle loads back equal; it
  re-saves portably; it loads lazily; concurrent readers agree; a second
  capture into the same path refuses typed.
* ``extract``: several layers at once, by address and by op label, with user
  labels, detached, equal to the plain capture; an unknown layer refuses typed.
* ``export``: the frame describes the ops (labels, shapes, dtypes) and the
  loaded trace's frame equals the live one; the csv, json and safetensors
  exports round-trip.
* ``validate``: a true saved-scope verdict leaves no failure record.

``features.py`` and ``cases.py`` each import one name from here under their
``# wave-3 fills`` marker: ``FEATURES`` (the registry's rows with the claimed
rows re-pointed) and ``TOY_CASES`` (the registry's toys plus these). Both are
computed on first import through the module ``__getattr__`` from the
registry's *current* value, so the three fill lines chain in any order.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArtifactTooOldError, TorchLensError

if TYPE_CHECKING:
    from tests.behaviour.cases import Case
    from tests.behaviour.features import Feature
    from tests.behaviour.harness import Cell

SEED = 0
PORTABLE = "portable"
#: The other public save levels; each round-trips or refuses typed.
OTHER_LEVELS = ("audit", "executable_with_callables", "runnable")
#: A manifest stamp below the oldest format this release reads.
STAMP_2X = 1


def _not_applicable(reason: str) -> Exception:
    from tests.behaviour.features import NotApplicable

    return NotApplicable(reason)


def _leaf_difference(want: Any, got: Any) -> str | None:
    from tests.behaviour.harness import first_leaf_difference

    return first_leaf_difference([want], [got])


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Return every file under ``root`` by relative path, with its bytes."""

    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _identity(op: Any) -> tuple[Any, ...]:
    """The op facts the loaded trace must reproduce."""

    return (
        op.label,
        op.func_name,
        tuple(op.shape) if op.shape is not None else None,
        str(op.dtype),
        tuple(op.parents),
        tuple(op.children),
        op.pass_number,
    )


def _expect_same_trace(cell: Cell, want: Any, got: Any, what: str) -> None:
    """Expect ``got`` to be ``want`` through the public surface (no payloads)."""

    cell.expect(
        [op.label for op in want.ops] == [op.label for op in got.ops],
        f"{what}: op labels differ",
    )
    cell.expect(
        tl.fingerprint.trace(want) == tl.fingerprint.trace(got),
        f"{what}: structural hash differs",
    )
    for a, b in zip(want.ops, got.ops, strict=True):
        cell.expect(_identity(a) == _identity(b), f"{what}: op {a.label}: {_identity(b)}")
    cell.expect(
        set(want.modules.keys()) == set(got.modules.keys()),
        f"{what}: module addresses differ",
    )
    cell.expect(
        {op.label for op in want.saved_ops} == {op.label for op in got.saved_ops},
        f"{what}: saved op sets differ",
    )


def _expect_same_payloads(cell: Cell, want: Any, got: Any, what: str) -> None:
    """Expect every saved payload of ``want`` to be present and equal on ``got``."""

    for op in want.saved_ops:
        other = got[op.label]
        diff = _leaf_difference(op.out, other.out)
        cell.expect(diff is None, f"{what}: payload of {op.label}: {diff}")


def _saved_bundle(cell: Cell) -> Path:
    """The portable bundle the harness saved for this cell."""

    path = cell.tmp / "cell.tlspec"
    if cell.loaded is None or not path.is_dir():
        raise _not_applicable("the cell has no loaded bundle to check")
    return path


# --- save ---------------------------------------------------------------------


def _prepare_save(cell: Cell) -> None:
    cell.trace_kwargs["save"] = tl.func(cell.facts.common_func)


def _after_save(cell: Cell) -> None:
    _check_selection(cell)
    bundle = _saved_bundle(cell)
    _check_other_levels(cell)
    _check_resave_bytes(cell, bundle)
    _check_lazy_load(cell, bundle)
    _check_2x_refusal(cell, bundle)
    _check_process_boundary(cell, bundle)


def _check_selection(cell: Cell) -> None:
    """The selected ops carry payloads and nothing else does, live and loaded."""

    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        want = {op.label for op in trace.ops if op.func_name == cell.facts.common_func}
        got = {op.label for op in trace.saved_ops}
        allowed = {op.label for op in trace.output_ops}
        allowed |= {op.label for op in trace.ops if op.is_buffer_source}
        allowed |= {label for label in got if label.startswith("buffer_")}
        missing, extra = sorted(want - got), sorted(got - want - allowed)
        cell.expect(not missing, f"{name}: selected ops not saved: {missing[:8]}")
        cell.expect(not extra, f"{name}: unselected ops saved: {extra[:8]}")
        without = sorted(op.label for op in trace.saved_ops if op.out is None)
        cell.expect(not without, f"{name}: saved ops without a payload: {without[:8]}")
        with_payload = sorted(
            op.label for op in trace.ops if op.label not in got and op.out is not None
        )
        cell.expect(not with_payload, f"{name}: unsaved ops carrying a payload: {with_payload[:8]}")


def _check_other_levels(cell: Cell) -> None:
    """Every other save level round-trips to the same trace, or refuses typed."""

    live = cell.require_trace()
    for level in OTHER_LEVELS:
        path = cell.tmp / f"level-{level}.tlspec"
        try:
            live.save(path, level=level)
        except TorchLensError:
            continue  # a typed refusal is the level declining this capture
        loaded = tl.load(path)
        _expect_same_trace(cell, live, loaded, f"level {level}")
        manifest = tl.io.inspect_tlspec(path)
        cell.expect(
            manifest.get("save_level") == level,
            f"level {level}: manifest says save_level={manifest.get('save_level')!r}",
        )
        if level == "executable_with_callables":
            _expect_same_payloads(cell, live, loaded, f"level {level}")
        shutil.rmtree(path, ignore_errors=True)


#: The one file a re-save may rewrite: the loader stamps how grouping was
#: settled (``settlement_note``), so the pickled metadata of a loaded trace
#: differs from a fresh capture's by design. Blobs and the manifest never move.
RESAVE_MAY_DIFFER = frozenset({"metadata.pkl"})


def _check_resave_bytes(cell: Cell, bundle: Path) -> None:
    """Saving the loaded trace again writes the same files, blobs and manifest."""

    again = cell.tmp / "resave.tlspec"
    cell.loaded.save(again, level=PORTABLE)
    first, second = _tree_bytes(bundle), _tree_bytes(again)
    files = sorted(set(first) ^ set(second))
    cell.expect(not files, f"re-save changed the file set: {files[:8]}")
    changed = sorted(
        name for name in first if first[name] != second[name] and name not in RESAVE_MAY_DIFFER
    )
    cell.expect(not changed, f"re-save changed the bytes of {changed[:8]}")
    reloaded = tl.load(again)
    _expect_same_trace(cell, cell.loaded, reloaded, "re-save")
    _expect_same_payloads(cell, cell.loaded, reloaded, "re-save")
    shutil.rmtree(again, ignore_errors=True)


def _check_lazy_load(cell: Cell, bundle: Path) -> None:
    """A lazy load defers payload reads; an eager load has already read them."""

    live = cell.require_trace()
    if not live.saved_ops:
        return
    copy = cell.tmp / "lazy.tlspec"
    shutil.copytree(bundle, copy)
    lazy = tl.load(copy, lazy=True)
    cell.expect(
        getattr(lazy, "payload_load_status", None) == "loaded_lazy",
        f"lazy load reports {getattr(lazy, 'payload_load_status', None)!r}",
    )
    _expect_same_trace(cell, live, lazy, "lazy load")
    first = lazy.saved_ops[0]
    diff = _leaf_difference(live[first.label].out, first.materialize_out())
    cell.expect(diff is None, f"lazy materialize of {first.label}: {diff}")
    eager = tl.load(copy)
    shutil.rmtree(copy / "blobs", ignore_errors=True)
    _expect_same_payloads(cell, live, eager, "eager load after the blobs were removed")
    if len(lazy.saved_ops) > 1:
        other = lazy.saved_ops[1]
        try:
            other.materialize_out()
        except TorchLensError:
            pass
        else:
            cell.expect(False, f"lazy {other.label} materialized without its blob on disk")
    shutil.rmtree(copy, ignore_errors=True)


def _check_2x_refusal(cell: Cell, bundle: Path) -> None:
    """An artifact stamped below the format floor refuses with the one typed error."""

    old = cell.tmp / "stamped-2x.tlspec"
    shutil.copytree(bundle, old)
    manifest_path = old / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = STAMP_2X
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        tl.load(old)
    except ArtifactTooOldError as exc:
        cell.expect(
            exc.fields.get("code") == "artifact_version_below_floor",
            f"2.x refusal code {exc.fields.get('code')!r}",
        )
        cell.expect("saved by torchlens 2.x" in str(exc), f"2.x refusal message: {exc}")
        cell.expect(
            exc.fields.get("observed") == f"tlspec_version={STAMP_2X}",
            f"2.x refusal observed {exc.fields.get('observed')!r}",
        )
    except Exception as exc:  # noqa: BLE001 - any other error is the finding
        cell.expect(False, f"2.x artifact raised {type(exc).__name__} instead: {str(exc)[:160]}")
    else:
        cell.expect(False, "an artifact stamped tlspec_version=1 loaded")
    finally:
        shutil.rmtree(old, ignore_errors=True)


_CROSSED_PROCESS: set[str] = set()
_SUBPROCESS = (
    "import json, sys, torchlens as tl\n"
    "trace = tl.load(sys.argv[1])\n"
    "print(json.dumps({'labels': [op.label for op in trace.ops], "
    "'fingerprint': tl.fingerprint.trace(trace), "
    "'saved': sorted(op.label for op in trace.saved_ops)}))\n"
)


def _check_process_boundary(cell: Cell, bundle: Path) -> None:
    """The bundle loads in a fresh interpreter to the same trace (once per process)."""

    if _CROSSED_PROCESS:
        return
    _CROSSED_PROCESS.add(cell.case.name)
    live = cell.require_trace()
    run = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(bundle)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    cell.expect(run.returncode == 0, f"load in another process failed: {run.stderr[-400:]}")
    seen = json.loads(run.stdout.strip().splitlines()[-1])
    cell.expect(seen["labels"] == [op.label for op in live.ops], "labels differ across processes")
    cell.expect(
        seen["fingerprint"] == tl.fingerprint.trace(live),
        "structural hash differs across processes",
    )
    cell.expect(
        seen["saved"] == sorted(op.label for op in live.saved_ops),
        "saved op set differs across processes",
    )


# --- storage ------------------------------------------------------------------


def _halted(cell: Cell) -> bool:
    """Whether the cell's other features stop the capture early (fewer ops by design)."""

    return any(feature.name == "halt" for feature in cell.features)


def _stochastic(cell: Cell, trace: Any) -> bool:
    """Whether another forward may differ: the case tag, or a random-drawing op in the trace."""

    from tests.behaviour.cases import _draws_random_numbers

    return cell.case.has("stochastic") or any(
        _draws_random_numbers(f"func:{op.func_name}") for op in trace.ops
    )


def _frame_of(value: Any) -> Any:
    """Shape and dtype of a tensor payload, or the value's type."""

    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), value.dtype)
    return type(value).__name__


def _is_output(op: Any) -> bool:
    return bool(getattr(op, "is_output", False))


#: Features whose keyword is a predicate; such captures materialize payloads on ``out`` reads.
PREDICATE_FEATURES = frozenset({"save", "halt", "intervene"})


def _predicate_capture(cell: Cell) -> bool:
    return any(feature.name in PREDICATE_FEATURES for feature in cell.features)


def _stream_path(cell: Cell) -> Path:
    return cell.tmp / "stream.tlspec"


def _prepare_storage(cell: Cell) -> None:
    cell.trace_kwargs["storage"] = tl.to_disk(str(_stream_path(cell)))


def _after_storage(cell: Cell) -> None:
    trace = cell.require_trace()
    plain = cell.facts.plain
    if not _halted(cell):
        cell.expect(
            len(trace.ops) == len(cell.facts.labels), "streamed capture changed the op count"
        )
    _check_streamed_payloads(cell, trace, plain)
    path = _stream_path(cell)
    cell.expect(path.is_dir(), "no stream bundle on disk")
    reloaded = tl.load(path)
    _expect_same_trace(cell, trace, reloaded, "stream bundle load")
    _expect_streamed_payloads(cell, trace, plain, reloaded, "stream bundle load")
    lazy = tl.load(path, lazy=True)
    _expect_same_trace(cell, trace, lazy, "stream bundle lazy load")
    _check_stream_resave(cell, trace, plain)
    _check_concurrent_reads(cell, path, tl.fingerprint.trace(trace))
    _check_path_exists_refusal(cell, path)
    _check_chunked_capture(cell, plain)


def _check_chunked_capture(cell: Cell, plain: Any) -> None:
    """A chunked forward (``chunk_size=``) gives the plain capture's ops and output."""

    stimuli = _batched_input(cell)
    if stimuli is None or _stochastic(cell, plain):
        return
    rows = int(stimuli.shape[0])
    size = (rows + 1) // 2
    pieces = _row_wise_pieces(cell, stimuli, size)
    if pieces is None:
        return  # the model's forward is not row-wise over the batch; chunking is not defined
    torch.manual_seed(SEED)
    chunked = tl.trace(
        cell.model, stimuli.clone(), capture=tl.options.CaptureOptions(chunk_size=size)
    )
    same_labels = [op.label for op in chunked.ops] == list(cell.facts.labels)
    if not same_labels and cell.case.has("inplace"):
        return  # state carried in mutated buffers changes the graph between forwards
    cell.expect(same_labels, "a chunked capture changed the op labels")
    got = list(cell.leaves(chunked.reconstruct_output()))
    if len(got) != 1 or not isinstance(got[0], torch.Tensor):
        return
    cell.expect_same_leaves([torch.cat(pieces, dim=0)], [got[0].detach()], "chunked capture output")


def _row_wise_pieces(cell: Cell, stimuli: torch.Tensor, size: int) -> list[torch.Tensor] | None:
    """Plain per-chunk outputs when the model is row-wise over the batch, else ``None``."""

    rows = int(stimuli.shape[0])
    torch.manual_seed(SEED)
    try:
        pieces = [
            cell.model(stimuli[start : start + size].clone()) for start in range(0, rows, size)
        ]
    except Exception:  # noqa: BLE001 - the model's own forward needs the whole batch
        return None
    if not all(isinstance(piece, torch.Tensor) for piece in pieces):
        return None
    first = pieces[0]
    if any(piece.dim() != first.dim() or piece.shape[1:] != first.shape[1:] for piece in pieces):
        return None  # the output's non-batch shape depends on the batch size
    return [piece.detach() for piece in pieces]


def _check_streamed_payloads(cell: Cell, trace: Any, plain: Any) -> None:
    """Every streamed payload is the plain capture's tensor for that op."""

    # Streamed payloads live on disk: every saved op carries an ``out_ref``. The output tail is
    # always retained (``torchlens/_user_funcs.py``, "always-retained output tail"): the caller
    # holds that tensor anyway. A partner that changes forward_match (intervene) makes the
    # forward_match oracle call ``trace.run``, which refuses typed on a streamed trace
    # (``trace_option_unsupported``, IO-8), so the payloads stay the bundle's here too.
    unstreamed = sorted(
        op.label for op in trace.saved_ops if op.out_ref is None and not _is_output(op)
    )
    cell.expect(not unstreamed, f"saved ops without a blob on disk: {unstreamed[:8]}")
    # Reading ``op.out`` on a capture with predicate keywords (save=, stop_after=, intervene=)
    # materializes the blob by design (the lazy-payload honesty gate in ``Op``), so the pinned
    # ``None`` is asserted only on plain streamed captures.
    if not _predicate_capture(cell):
        retained = sorted(
            op.label for op in trace.saved_ops if op.out is not None and not _is_output(op)
        )
        cell.expect(not retained, f"streamed payloads kept in memory: {retained[:8]}")
    # A stochastic forward draws other numbers, and a partner that changes forward_match
    # (intervene) changes the values downstream of its site: only the payload's frame can
    # match the plain capture's then (``_expect_streamed_payloads`` makes the same call).
    frame_only = _stochastic(cell, trace) or cell.changes("forward_match")
    for op in trace.saved_ops:
        if _halted(cell) and _is_output(op):
            continue  # a halted capture's output is the halt frontier, not the model output
        reference = plain[op.label].out
        if reference is None:
            continue
        if frame_only:
            got = op.materialize_out()
            cell.expect(
                _frame_of(got) == _frame_of(reference),
                f"streamed payload of {op.label}: {_frame_of(got)} vs {_frame_of(reference)}",
            )
            continue
        diff = _leaf_difference(reference, op.materialize_out())
        cell.expect(diff is None, f"streamed payload of {op.label}: {diff}")


def _expect_streamed_payloads(cell: Cell, trace: Any, plain: Any, got: Any, what: str) -> None:
    """Expect ``got`` to carry the plain payload of every op the streamed capture saved.

    The streamed capture, not the plain one, defines the saved set: a ``save=`` partner
    narrows it, and a ``stop_after=`` partner's output op is the halt frontier.
    """

    # The streamed capture is its own reference when the plain capture cannot be: a stochastic
    # forward draws other numbers, and a partner that changes forward_match (intervene) changes
    # the values downstream of its site.
    own_reference = _stochastic(cell, trace) or cell.changes("forward_match")
    for op in trace.saved_ops:
        if _halted(cell) and _is_output(op):
            continue
        reference = plain[op.label].out
        if reference is None:
            continue
        if own_reference:
            reference = trace[op.label].materialize_out()
        diff = _leaf_difference(reference, got[op.label].out)
        cell.expect(diff is None, f"{what}: payload of {op.label}: {diff}")


def _check_stream_resave(cell: Cell, trace: Any, plain: Any) -> None:
    """A streamed trace saves portably and the portable copy holds the same payloads."""

    path = cell.tmp / "stream-portable.tlspec"
    trace.save(path, level=PORTABLE)
    loaded = tl.load(path)
    _expect_same_trace(cell, trace, loaded, "portable save of a streamed trace")
    _expect_streamed_payloads(cell, trace, plain, loaded, "portable save of a streamed trace")
    shutil.rmtree(path, ignore_errors=True)


def _check_concurrent_reads(cell: Cell, path: Path, fingerprint: str) -> None:
    """Two threads loading the same bundle at once both get the trace."""

    results: list[Any] = [None, None]

    def read(index: int) -> None:
        try:
            results[index] = tl.fingerprint.trace(tl.load(path))
        except Exception as exc:  # noqa: BLE001 - reported through expect below
            results[index] = f"{type(exc).__name__}: {str(exc)[:160]}"

    threads = [threading.Thread(target=read, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cell.expect(results == [fingerprint, fingerprint], f"concurrent loads disagree: {results}")


def _check_path_exists_refusal(cell: Cell, path: Path) -> None:
    """A second streamed capture into an existing bundle path refuses typed."""

    try:
        torch.manual_seed(SEED)
        tl.trace(cell.model, cell.fresh_inputs(), storage=tl.to_disk(str(path)))
    except TorchLensError:
        return
    except Exception as exc:  # noqa: BLE001 - an untyped error is the finding
        cell.expect(False, f"existing-path capture raised {type(exc).__name__}: {str(exc)[:160]}")
    else:
        cell.expect(False, "a second capture streamed into an existing bundle path")


# --- extract ------------------------------------------------------------------


def _no_stochastic(cell: Cell) -> None:
    if cell.case.has("stochastic"):
        raise _not_applicable("the forward draws random numbers; a second run cannot match")


def _after_extract(cell: Cell) -> None:
    _no_stochastic(cell)
    if cell.case.has("containers"):
        raise _not_applicable("extract takes one tensor input")
    address = cell.facts.single_call_module
    if address is None:
        raise _not_applicable("no child module called exactly once")
    cell.require_trace()
    plain = cell.facts.plain
    label = cell.facts.labels[len(cell.facts.labels) // 2]
    torch.manual_seed(SEED)
    features = tl.extract(cell.model, cell.fresh_inputs(), [address, label])
    cell.expect(set(features) == {address, label}, f"extract keys {sorted(features)}")
    for key in (address, label):
        diff = _leaf_difference(plain[key].out, features[key])
        cell.expect(diff is None, f"extract({key!r}): {diff}")
        detached = not getattr(features[key], "requires_grad", False)
        cell.expect(detached, f"extract({key!r}) is not detached")
    torch.manual_seed(SEED)
    named = tl.extract(cell.model, cell.fresh_inputs(), {"first": address, "second": label})
    cell.expect(set(named) == {"first", "second"}, f"extract mapping keys {sorted(named)}")
    diff = _leaf_difference(plain[address].out, named["first"])
    cell.expect(diff is None, f"extract mapping 'first': {diff}")
    _check_dataset_export(cell, address)
    _check_unknown_layer_refusal(cell)


def _batched_input(cell: Cell) -> torch.Tensor | None:
    """The case's input when it is one tensor with two or more rows along dim 0."""

    inputs = cell.inputs
    if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
        inputs = inputs[0]
    if isinstance(inputs, torch.Tensor) and inputs.dim() >= 1 and inputs.shape[0] >= 2:
        return inputs
    return None


def _check_dataset_export(cell: Cell, address: str) -> None:
    """The dataset extractor's rows for one batch equal the plain capture's payload."""

    from torchlens.dataset_extraction import load_extraction

    stimuli = _batched_input(cell)
    reference = cell.facts.plain[address].out
    if stimuli is None or not isinstance(reference, torch.Tensor):
        return
    if reference.dim() == 0 or reference.shape[0] != stimuli.shape[0]:
        return  # the extractor concatenates rows; a non-batched payload is not its contract
    path = cell.tmp / "dataset"
    torch.manual_seed(SEED)
    tl.extract_dataset(
        cell.model,
        stimuli.clone(),
        [address],
        batch_size=int(stimuli.shape[0]),
        path=path,
        progress=False,
    )
    loaded = load_extraction(path)
    cell.expect(address in loaded.activations, f"extraction lacks {address!r}")
    diff = _leaf_difference(reference, loaded.activations[address])
    cell.expect(diff is None, f"extract_dataset({address!r}): {diff}")
    shutil.rmtree(path, ignore_errors=True)


def _check_unknown_layer_refusal(cell: Cell) -> None:
    """A layer lookup that matches nothing refuses typed, never crashes."""

    try:
        torch.manual_seed(SEED)
        tl.extract(cell.model, cell.fresh_inputs(), ["no_such_layer_anywhere"])
    except TorchLensError:
        return
    except Exception as exc:  # noqa: BLE001 - an untyped error is the finding
        cell.expect(False, f"extract of an unknown layer raised {type(exc).__name__}: {exc}")
    cell.expect(False, "extract of an unknown layer returned instead of refusing")


# --- export -------------------------------------------------------------------


def _after_export(cell: Cell) -> None:
    live = cell.require_trace()
    frame = live.to_pandas()
    cell.expect(len(frame) == len(live), f"to_pandas has {len(frame)} rows for {len(live)} ops")
    # One row per op row (``Trace._layer_list``): the ops plus the input and output boundary rows.
    cell.expect(
        sorted(frame["label"]) == sorted(entry.label for entry in live._layer_list),
        "to_pandas rows do not name the trace's op rows",
    )
    if cell.loaded is not None:
        loaded = cell.loaded
        if [op.label for op in live.ops] != [op.label for op in loaded.ops]:
            # A rerun feature replaced the graph since the save (a plain forward between
            # capture and rerun filled a first-call cache, so the rerun lost those ops and
            # warned ``ControlFlowDivergenceWarning``): the frame promise is about an
            # artifact of THIS graph, so save and load the live trace again.
            live.save(cell.tmp / "export.tlspec", level="portable")
            loaded = tl.load(cell.tmp / "export.tlspec")
        other = loaded.to_pandas()
        absent = [column for column in FRAME_IDENTITY if column not in frame.columns]
        cell.expect(not absent, f"to_pandas lacks the columns {absent}")
        mine = frame.set_index("label").sort_index()
        theirs = other.set_index("label").sort_index()
        cell.expect(
            list(mine.index) == list(theirs.index), "the loaded trace's frame names other rows"
        )
        columns = [column for column in FRAME_IDENTITY if column != "label"]
        differing = [column for column in columns if not mine[column].equals(theirs[column])]
        cell.expect(
            not differing, f"the loaded trace's frame differs from the live one in {differing}"
        )
    _check_tabular_exports(cell, live, frame)
    _check_safetensors_export(cell, live)


#: The frame columns the equality oracle promises the loaded trace reproduces.
FRAME_IDENTITY = (
    "label",
    "func_name",
    "shape",
    "dtype",
    "pass_number",
    "num_passes",
    "parents",
    "children",
    "has_saved_activation",
)


def _check_tabular_exports(cell: Cell, live: Any, frame: Any) -> None:
    """csv and json exports write files holding every op."""

    import pandas as pd

    csv_path = tl.export.csv(live, cell.tmp / "export.csv")
    back = pd.read_csv(csv_path, comment="#")  # the honesty preamble rides # comment lines
    cell.expect(len(back) == len(frame), f"csv round-trip has {len(back)} rows")
    json_path = tl.export.json(live, cell.tmp / "export.json")
    text = Path(json_path).read_text(encoding="utf-8")
    missing = [op.label for op in live.ops if op.label not in text][:8]
    cell.expect(not missing, f"json export lacks ops {missing}")


def _check_safetensors_export(cell: Cell, live: Any) -> None:
    """The safetensors export reads back to the live payloads."""

    if not live.saved_ops:
        return
    path = tl.export.safetensors(live, cell.tmp / "export.safetensors")
    read = tl.io.read_safetensors(path)
    for op in live.saved_ops:
        if op.out is None or not isinstance(op.out, torch.Tensor):
            continue
        payloads = read.tensors.get(op.label, {})
        cell.expect("out" in payloads, f"safetensors export lacks the out of {op.label}")
        if "out" in payloads:
            diff = _leaf_difference(op.out, payloads["out"])
            cell.expect(diff is None, f"safetensors {op.label}: {diff}")


# --- validate -----------------------------------------------------------------


def _after_validate(cell: Cell) -> None:
    from torchlens.validation import last_validation_failure

    verdict = tl.validate(cell.model, cell.fresh_inputs(), scope="saved", seed=SEED)
    cell.expect(bool(verdict), cell.validation_failure_summary())
    cell.expect(
        last_validation_failure() is None,
        "a true saved-scope verdict left a failure record",
    )


# --- the claimed rows -----------------------------------------------------------

_AFTER = {
    "save": _after_save,
    "storage": _after_storage,
    "extract": _after_extract,
    "export": _after_export,
    "validate": _after_validate,
}
_PREPARE = {"save": _prepare_save, "storage": _prepare_storage}


def _filled_features() -> tuple[Feature, ...]:
    from tests.behaviour import features

    rows = []
    for row in features.FEATURES:
        changes: dict[str, Any] = {}
        if row.name in _AFTER:
            changes["after"] = _AFTER[row.name]
        if row.name in _PREPARE:
            changes["prepare"] = _PREPARE[row.name]
        rows.append(dataclasses.replace(row, **changes) if changes else row)
    return tuple(rows)


# --- toy rows ---------------------------------------------------------------------


class MutatingBuffers(nn.Module):
    """Adds to a buffer and to a parameter (under no_grad) during every forward."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("running", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count.add_(1.0)
        self.running.mul_(0.5).add_(x.mean(dim=0) * 0.5)
        with torch.no_grad():
            self.proj.bias.add_(0.01)
        return self.proj(x) * self.running + self.count


class AliasedOutputs(nn.Module):
    """Returns the same tensor object in two output slots."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = torch.relu(self.proj(x))
        return hidden, [x, hidden]


class EmptyAndScalar(nn.Module):
    """Produces a zero-size payload, a 0-d payload and a strided view."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.proj(x)
        empty = hidden[:0]
        scalar = hidden.sum()
        strided = hidden.t()[::2]
        return empty, scalar, strided


class ComplexSpectrum(nn.Module):
    """Carries a complex payload through the graph and returns a real one."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(self.proj(x), dim=-1)
        return spectrum.abs() + spectrum.angle()


class BigPayload(nn.Module):
    """One payload above a mebibyte, for the streaming and blob paths."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = torch.relu(self.conv(x))
        return wide.mean(dim=1)


def _vec4() -> torch.Tensor:
    return torch.rand(3, 4)


def _vec8() -> torch.Tensor:
    return torch.rand(2, 8)


def _plane() -> torch.Tensor:
    return torch.rand(1, 1, 192, 192)


def _filled_toys() -> tuple[Case, ...]:
    from tests.behaviour import cases

    toy = cases._toy
    return (
        *cases.TOY_CASES,
        toy("mutating_buffers", MutatingBuffers, _vec4, "inplace", "child_module", "stochastic"),
        toy("aliased_outputs", AliasedOutputs, _vec4, "containers", "child_module"),
        toy("empty_and_scalar", EmptyAndScalar, _vec4, "containers", "views", "child_module"),
        toy("complex_spectrum", ComplexSpectrum, _vec8, "dtype", "child_module"),
        toy("big_payload", BigPayload, _plane, "child_module", seconds=1.0),
    )


def __getattr__(name: str) -> Any:
    """Compute ``FEATURES`` and ``TOY_CASES`` when a registry imports them."""

    if name == "FEATURES":
        return _filled_features()
    if name == "TOY_CASES":
        return _filled_toys()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
