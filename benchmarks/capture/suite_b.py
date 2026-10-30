"""Suite B: what the capture overhead buys.

Each row is one TorchLens operation with a ``keeps`` statement saying exactly what it
leaves the user holding, beside two reference rows (a plain forward and module hooks
retaining every module output). A hook row is not a full-trace substitute: it keeps
module outputs only, with no operation graph, no functional operations inside
``forward``, no replay check and no file, and its ``keeps`` text says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.capture.arms import Subject, first_tensor


@dataclass(frozen=True)
class Row:
    """One Suite B row."""

    name: str
    keeps: str
    uses_torchlens: bool = True
    needs_graphviz: bool = False


ROWS = {
    row.name: row
    for row in (
        Row("raw_forward", "nothing: the model output only", uses_torchlens=False),
        Row(
            "hooks_all_modules",
            "every module's output tensor; no operation graph, no functional operations "
            "inside forward, no replay check, no file (not a full-trace substitute)",
            uses_torchlens=False,
        ),
        Row(
            "raw_forward_wrapped",
            "nothing: a plain forward after TorchLens has installed its torch wrappers",
        ),
        Row(
            "full_trace_default",
            "every operation's metadata and graph edges, every activation (default retention)",
        ),
        Row(
            "full_trace_declared",
            "every operation's metadata and graph edges, activations of one declared module only",
        ),
        Row("record", "events and payloads for one declared module; no graph postprocessing"),
        Row(
            "validate",
            "a forward-replay verdict with metadata invariants (pass, fail or unavailable)",
        ),
        Row("save_load", "a portable trace file on disk, and the trace loaded back from it"),
        Row(
            "backward_capture",
            "the forward graph plus gradients at one declared module after log_backward",
        ),
        Row("draw", "a rendered graph file of the whole trace", needs_graphviz=True),
        Row(
            "bundle_compare",
            "a two-member bundle (clean and ablated traces) and its shared and divergent "
            "operation labels",
        ),
    )
}


def _options(**kw: Any) -> Any:
    from torchlens.options import CaptureOptions

    return CaptureOptions(unwrap_when_done=False, **kw)


def _hooks_all(subject: Subject) -> Callable[[], Any]:
    model = subject.model
    names = [name for name, _m in model.named_modules() if name]

    def run() -> dict[str, torch.Tensor]:
        store: dict[str, torch.Tensor] = {}

        def make(name: str) -> Callable[..., None]:
            def hook(_m: Any, _i: Any, out: Any) -> None:
                try:
                    store[name] = first_tensor(out).detach()
                except TypeError:
                    return

            return hook

        handles = [model.get_submodule(n).register_forward_hook(make(n)) for n in names]
        try:
            with torch.no_grad():
                model(subject.x)
        finally:
            for handle in handles:
                handle.remove()
        return store

    return run


def build_row(
    name: str, subject: Subject, workdir: Path
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Return the timed callable for one row and facts measured during setup.

    Parameters
    ----------
    name:
        Row name from :data:`ROWS`.
    subject:
        Prepared subject.
    workdir:
        Scratch directory for files the row writes.

    Returns
    -------
    tuple[Callable[[], Any], dict[str, Any]]
        The operation to time and setup facts (operation count of a default trace).
    """

    model, x, site = subject.model, subject.x, subject.site
    if name == "raw_forward":
        return (lambda: _no_grad(model, x)), {}
    if name == "hooks_all_modules":
        return _hooks_all(subject), {}
    import torchlens as tl

    probe = tl.trace(model, x, capture=_options())
    facts = {"num_ops": len(list(probe.ops))}
    if name == "raw_forward_wrapped":
        return (lambda: _no_grad(model, x)), facts
    if name == "full_trace_default":
        return (lambda: tl.trace(model, x, capture=_options())), facts
    if name == "full_trace_declared":
        return (lambda: tl.trace(model, x, save=tl.module(site), capture=_options())), facts
    if name == "record":
        return (lambda: _no_grad_record(tl, model, x, site)), facts
    if name == "validate":
        return (lambda: tl.validate(model, x, scope="forward", random_seed=0)), facts
    if name == "save_load":
        path = workdir / "trace.tlspec"

        def save_load() -> Any:
            tl.save(probe, path, level="portable")
            return tl.load(path)

        return save_load, facts
    if name == "backward_capture":
        return _backward(tl, subject), facts
    if name == "draw":
        out = str(workdir / "graph")
        return (
            lambda: probe.draw(vis_outpath=out, vis_save_only=True, vis_fileformat="svg")
        ), facts
    if name == "bundle_compare":
        return _bundle(tl, subject), facts
    raise ValueError(name)


def _no_grad(model: Any, x: Any) -> Any:
    with torch.no_grad():
        return model(x)


def _no_grad_record(tl: Any, model: Any, x: Any, site: str) -> Any:
    with torch.no_grad():
        return tl.record(model, x, save=tl.module(site), return_output=True)


def _backward(tl: Any, subject: Subject) -> Callable[[], Any]:
    site, final = subject.site, subject.final_site

    def run() -> Any:
        trace = tl.trace(
            subject.model,
            subject.x,
            save=tl.module(site) | tl.module(final),
            capture=_options(save_grads=tl.module(site)),
        )
        out = first_tensor(trace.find_sites(tl.module(final)).first().out)
        trace.log_backward(out.sum())
        subject.model.zero_grad(set_to_none=True)
        return trace

    return run


def _bundle(tl: Any, subject: Subject) -> Callable[[], Any]:
    model, x, site = subject.model, subject.x, subject.edit_site

    def run() -> Any:
        clean = tl.trace(model, x, capture=_options())
        ablated = tl.trace(
            model, x, intervene=tl.when(tl.module(site), tl.zero_ablate()), capture=_options()
        )
        bundle = tl.bundle({"clean": clean, "ablated": ablated}, baseline="clean")
        return bundle.shared_op_labels, bundle.divergent_op_labels

    return run
