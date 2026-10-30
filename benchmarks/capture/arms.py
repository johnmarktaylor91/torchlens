"""Suite A arms: tools that each produce the same output for a matched contract.

A contract (a Suite A case) fixes the output every arm must produce on a subject:

* ``single_module_output``: the output of one module;
* ``matched_retained``: the outputs of a fixed set of modules;
* ``replacement_edit``: the model output with one module's output replaced by zeros;
* ``repeated_steering``: the model outputs for eight added multiples of one direction;
* ``same_site_gradient``: the gradient of the summed model output at one module output;
* ``ordered_dataset_features``: one module's outputs over a dataset, in stimulus order.

The ``hooks`` arm is the reference; every other arm's result is compared with it.
The ``raw`` arm runs the forward passes a contract needs with nothing attached, so it
is the floor, not a contract. An arm that cannot produce a contract raises
:class:`Unsupported` with the reason, and the cell stays in the table.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

CASES = (
    "single_module_output",
    "matched_retained",
    "replacement_edit",
    "repeated_steering",
    "same_site_gradient",
    "ordered_dataset_features",
)
STEER_MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DATASET_SIZE = 16
DATASET_BATCH = 4


class Unsupported(RuntimeError):
    """The arm cannot produce this contract; the message says why."""


@dataclass
class Subject:
    """A model with its input, dataset and sites, shared by every arm."""

    name: str
    model: nn.Module
    x: Any
    meta: dict[str, Any]
    dataset: torch.Tensor
    site_shape: tuple[int, ...] = ()
    edit_shape: tuple[int, ...] = ()
    direction: torch.Tensor | None = None

    @property
    def site(self) -> str:
        return str(self.meta["site"])

    @property
    def edit_site(self) -> str:
        return str(self.meta.get("edit_site", self.meta["site"]))

    @property
    def sites(self) -> list[str]:
        return list(self.meta["sites"])

    @property
    def final_site(self) -> str:
        return str(self.meta["final_site"])

    def batches(self) -> Iterator[torch.Tensor]:
        for start in range(0, self.dataset.shape[0], DATASET_BATCH):
            yield self.dataset[start : start + DATASET_BATCH]


def first_tensor(value: Any) -> torch.Tensor:
    """Return ``value`` if it is a tensor, else its first tensor element."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return first_tensor(value[0])
    out = getattr(value, "last_hidden_state", None)
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"no tensor in {type(value).__name__}")


def _map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    return value


def make_dataset(x: Any, n: int = DATASET_SIZE) -> torch.Tensor:
    """Return ``n`` seeded stimuli shaped like one row of ``x``, stacked."""

    gen = torch.Generator().manual_seed(7)
    if x.dtype in (torch.int64, torch.int32):
        high = int(x.max()) + 1
        return torch.randint(0, high, (n, *x.shape[1:]), generator=gen, dtype=x.dtype)
    return torch.randn((n, *x.shape[1:]), generator=gen, dtype=x.dtype)


# ----------------------------------------------------------------------------- hooks
@contextlib.contextmanager
def _forward_hooks(model: nn.Module, hooks: dict[str, Callable[..., Any]]) -> Iterator[None]:
    handles = [model.get_submodule(s).register_forward_hook(h) for s, h in hooks.items()]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _grab(store: dict[str, Any], site: str) -> Callable[..., None]:
    def hook(_m: nn.Module, _i: Any, out: Any) -> None:
        store[site] = first_tensor(out).detach()

    return hook


def _edit(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[..., Any]:
    def hook(_m: nn.Module, _i: Any, out: Any) -> Any:
        return _map_tensors(out, fn)

    return hook


class HooksArm:
    """Plain ``register_forward_hook`` code: the reference for every contract."""

    name = "hooks"

    def __init__(self, subject: Subject) -> None:
        self.s = subject

    def read(self, x: Any, sites: list[str]) -> list[torch.Tensor]:
        store: dict[str, Any] = {}
        with _forward_hooks(self.s.model, {s: _grab(store, s) for s in sites}), torch.no_grad():
            self.s.model(x)
        return [store[s] for s in sites]

    def edited(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        with _forward_hooks(self.s.model, {self.s.edit_site: _edit(fn)}), torch.no_grad():
            return first_tensor(self.s.model(self.s.x))

    def run(self, case: str) -> Any:
        s = self.s
        if case == "single_module_output":
            return self.read(s.x, [s.site])[0]
        if case == "matched_retained":
            return self.read(s.x, s.sites)
        if case == "replacement_edit":
            return self.edited(torch.zeros_like)
        if case == "repeated_steering":
            return [self.edited(_adder(s, m)) for m in STEER_MAGNITUDES]
        if case == "same_site_gradient":
            return self._gradient()
        if case == "ordered_dataset_features":
            return torch.cat([self.read(b, [s.site])[0] for b in s.batches()])
        raise ValueError(case)

    def _gradient(self) -> torch.Tensor:
        held: dict[str, torch.Tensor] = {}

        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            tensor = first_tensor(out)
            tensor.retain_grad()
            held["act"] = tensor

        with _forward_hooks(self.s.model, {self.s.site: hook}):
            first_tensor(self.s.model(self.s.x)).sum().backward()
        self.s.model.zero_grad(set_to_none=True)
        grad = held["act"].grad
        assert grad is not None
        return grad.detach()


def _adder(s: Subject, magnitude: float) -> Callable[[torch.Tensor], torch.Tensor]:
    direction = s.direction

    def add(t: torch.Tensor) -> torch.Tensor:
        if direction is None or t.shape != direction.shape:
            return t
        return t + direction * magnitude

    return add


class RawArm:
    """The forward (and backward) passes a contract needs, with nothing attached."""

    name = "raw"

    def __init__(self, subject: Subject) -> None:
        self.s = subject

    def run(self, case: str) -> None:
        model, x = self.s.model, self.s.x
        if case == "same_site_gradient":
            first_tensor(model(x)).sum().backward()
            model.zero_grad(set_to_none=True)
            return None
        with torch.no_grad():
            if case == "repeated_steering":
                for _ in STEER_MAGNITUDES:
                    model(x)
            elif case == "ordered_dataset_features":
                for batch in self.s.batches():
                    model(batch)
            else:
                model(x)
        return None


# ----------------------------------------------------------------------------- torchlens
def _tl() -> Any:
    import torchlens as tl

    return tl


def _sel(tl: Any, sites: list[str]) -> Any:
    sel = tl.module(sites[0])
    for site in sites[1:]:
        sel = sel | tl.module(site)
    return sel


def _pick(cands: list[Any], shape: tuple[int, ...]) -> torch.Tensor:
    tensors = []
    for cand in cands:
        with contextlib.suppress(TypeError):
            tensors.append(first_tensor(cand))
    for tensor in tensors:
        if tuple(tensor.shape) == shape:
            return tensor
    if tensors:
        return tensors[-1]
    raise RuntimeError("no tensor output at the site")


class _TorchLensBase:
    def __init__(self, subject: Subject) -> None:
        self.s = subject
        self.tl = _tl()
        from torchlens.options import CaptureOptions

        self.CaptureOptions = CaptureOptions

    def _shape(self, site: str) -> tuple[int, ...]:
        if site == self.s.site:
            return self.s.site_shape
        return tuple(self.s.meta.get("shapes", {}).get(site, ()))

    def _site_out(self, trace: Any, site: str) -> torch.Tensor:
        ops = trace.find_sites(self.tl.module(site))
        return _pick([op.out for op in ops], self._shape(site))


class TLExtractArm(_TorchLensBase):
    """``tl.extract``: named module outputs from one call."""

    name = "tl_extract"

    def run(self, case: str) -> Any:
        tl, s = self.tl, self.s
        if case in ("replacement_edit", "repeated_steering", "same_site_gradient"):
            raise Unsupported(
                "tl.extract reads outputs; edits and gradients use other entry points"
            )
        if case == "single_module_output":
            with torch.no_grad():
                return first_tensor(tl.extract(s.model, s.x, {"f": s.site})["f"])
        if case == "matched_retained":
            with torch.no_grad():
                got = tl.extract(s.model, s.x, {site: site for site in s.sites})
            return [first_tensor(got[site]) for site in s.sites]
        with torch.no_grad():
            return torch.cat(
                [first_tensor(tl.extract(s.model, b, {"f": s.site})["f"]) for b in s.batches()]
            )


class TLTraceSparseArm(_TorchLensBase):
    """``tl.trace`` with a ``save=`` selector: the full graph, only the named payloads."""

    name = "tl_trace_sparse"

    def _opts(self) -> Any:
        return self.CaptureOptions(inference_only=True, unwrap_when_done=False)

    def _trace(self, x: Any, sites: list[str], **kw: Any) -> Any:
        return self.tl.trace(self.s.model, x, save=_sel(self.tl, sites), capture=self._opts(), **kw)

    def _edited(self, edit: Any) -> torch.Tensor:
        tl, s = self.tl, self.s
        spec = tl.when(tl.module(s.edit_site), edit)
        trace = self._trace(s.x, [s.final_site], intervene=spec)
        return first_tensor(trace.find_sites(tl.module(s.final_site)).first().out)

    def run(self, case: str) -> Any:
        tl, s = self.tl, self.s
        if case == "single_module_output":
            return self._site_out(self._trace(s.x, [s.site]), s.site)
        if case == "matched_retained":
            trace = self._trace(s.x, s.sites)
            return [self._site_out(trace, site) for site in s.sites]
        if case == "replacement_edit":
            return self._edited(tl.zero_ablate())
        if case == "repeated_steering":
            assert s.direction is not None
            return [self._edited(tl.add(s.direction * m)) for m in STEER_MAGNITUDES]
        if case == "same_site_gradient":
            sel = _sel(tl, [s.site, s.final_site])
            trace = tl.trace(
                s.model,
                s.x,
                save=sel,
                capture=self.CaptureOptions(save_grads=tl.module(s.site), unwrap_when_done=False),
            )
            out = first_tensor(trace.find_sites(tl.module(s.final_site)).first().out)
            trace.log_backward(out.sum())
            s.model.zero_grad(set_to_none=True)
            ops = list(trace.find_sites(tl.module(s.site)))
            act = _pick([op.out for op in ops], s.site_shape)
            op = next(op for op in ops if first_tensor(op.out) is act)
            return first_tensor(op.grad).detach()
        return torch.cat([self._site_out(self._trace(b, [s.site]), s.site) for b in s.batches()])


class TLRecordArm(_TorchLensBase):
    """``tl.record``: the sparse event recorder, no graph postprocessing."""

    name = "tl_record"

    def _get(self, rec: Any, site: str) -> torch.Tensor:
        records = rec[1] if isinstance(rec, tuple) else rec
        cands = []
        for record in records:
            ctx = record.ctx
            calls = tuple(getattr(ctx, "output_of_module_calls", ()) or ())
            if ctx.address == site or any(str(c).split(":")[0] == site for c in calls):
                payload = (
                    record.ram_payload if record.ram_payload is not None else record.disk_payload
                )
                if payload is not None:
                    cands.append(payload)
        if not cands:
            raise RuntimeError(f"record: no payload at {site}")
        return _pick(cands, self._shape(site))

    def _record(self, x: Any, sites: list[str], **kw: Any) -> Any:
        with torch.no_grad():
            return self.tl.record(
                self.s.model, x, save=_sel(self.tl, sites), return_output=True, **kw
            )

    def _edited(self, edit: Any) -> torch.Tensor:
        tl, s = self.tl, self.s
        out, _rec = self._record(
            s.x, [s.edit_site], intervene=tl.when(tl.module(s.edit_site), edit)
        )
        return first_tensor(out)

    def run(self, case: str) -> Any:
        tl, s = self.tl, self.s
        if case == "single_module_output":
            return self._get(self._record(s.x, [s.site]), s.site)
        if case == "matched_retained":
            rec = self._record(s.x, s.sites)
            return [self._get(rec, site) for site in s.sites]
        if case == "replacement_edit":
            return self._edited(tl.zero_ablate())
        if case == "repeated_steering":
            assert s.direction is not None
            return [self._edited(tl.add(s.direction * m)) for m in STEER_MAGNITUDES]
        if case == "same_site_gradient":
            raise Unsupported("this comparison uses tl.record without backward capture")
        return torch.cat([self._get(self._record(b, [s.site]), s.site) for b in s.batches()])


class TLBindArm(_TorchLensBase):
    """A bound live edit: ``tl.when(...).bind(model)``, no recording."""

    name = "tl_bind"

    def run(self, case: str) -> Any:
        tl, s = self.tl, self.s
        if case == "replacement_edit":
            bound = tl.when(tl.module(s.edit_site), tl.zero_ablate()).bind(s.model)
            with torch.no_grad():
                return first_tensor(bound(s.x))
        if case == "repeated_steering":
            assert s.direction is not None
            outs = []
            for m in STEER_MAGNITUDES:
                bound = tl.when(tl.module(s.edit_site), tl.add(s.direction * m)).bind(s.model)
                with torch.no_grad():
                    outs.append(first_tensor(bound(s.x)))
            return outs
        raise Unsupported("a bound edit records nothing; it serves edit contracts only")


class TLExtractDatasetArm(_TorchLensBase):
    """``tl.extract_dataset``: ordered features over a stimulus set."""

    name = "tl_extract_dataset"

    def run(self, case: str) -> Any:
        if case != "ordered_dataset_features":
            raise Unsupported("dataset extraction serves the dataset contract only")
        s = self.s
        with torch.no_grad():
            got = self.tl.extract_dataset(
                s.model, s.dataset, [s.site], batch_size=DATASET_BATCH, progress=False
            )
        return first_tensor(next(iter(got.values())))


# ----------------------------------------------------------------------------- peers
def _require(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - an absent peer is a visible unsupported cell
        raise Unsupported(f"{module} not importable: {type(exc).__name__}") from exc


class NNsightArm:
    """nnsight's ``NNsight`` wrapper on the unmodified model."""

    name = "nnsight"

    def __init__(self, subject: Subject) -> None:
        nnsight = _require("nnsight")
        self.s = subject
        self.m = nnsight.NNsight(subject.model)

    def _env(self, site: str) -> Any:
        env = self.m
        for part in site.split("."):
            env = env[int(part)] if part.isdigit() else getattr(env, part)
        return env

    def _read(self, x: Any, sites: list[str]) -> list[torch.Tensor]:
        saved = {}
        with torch.no_grad(), self.m.trace(x):
            for site in sites:
                saved[site] = self._env(site).output.save()
        return [first_tensor(saved[site]).detach() for site in sites]

    def _edited(self, fn: Callable[[Any], Any]) -> torch.Tensor:
        with torch.no_grad(), self.m.trace(self.s.x):
            env = self._env(self.s.edit_site)
            env.output = fn(env.output)
            out = self.m.output.save()
        return first_tensor(out)

    def run(self, case: str) -> Any:
        s = self.s
        if case == "single_module_output":
            return self._read(s.x, [s.site])[0]
        if case == "matched_retained":
            return self._read(s.x, s.sites)
        if case == "replacement_edit":
            return self._edited(lambda o: _map_tensors(o, torch.zeros_like))
        if case == "repeated_steering":
            return [
                self._edited(lambda o, m=m: _map_tensors(o, _adder(s, m))) for m in STEER_MAGNITUDES
            ]
        if case == "same_site_gradient":
            with self.m.trace(s.x):
                act = first_tensor(self._env(s.site).output)
                grad = act.grad.save()
                self.m.output.sum().backward()
            s.model.zero_grad(set_to_none=True)
            if grad is None:
                raise Unsupported("the .grad proxy resolved to None in this nnsight version")
            return first_tensor(grad).detach()
        return torch.cat([self._read(b, [s.site])[0] for b in s.batches()])


class TorchvisionFxArm:
    """torchvision ``create_feature_extractor``: a symbolically traced copy of the model."""

    name = "tv_fx"

    def __init__(self, subject: Subject) -> None:
        _require("torchvision")
        from torchvision.models.feature_extraction import create_feature_extractor

        self.s = subject
        if not subject.meta.get("fx_traceable", True):
            raise Unsupported("symbolic tracing refuses this model (data-dependent control flow)")
        try:
            self.single = create_feature_extractor(subject.model, return_nodes={subject.site: "f"})
            self.many = create_feature_extractor(
                subject.model, return_nodes={site: site for site in subject.sites}
            )
        except Exception as exc:  # noqa: BLE001 - the refusal is the finding
            raise Unsupported(
                f"symbolic tracing failed: {type(exc).__name__}: {str(exc)[:160]}"
            ) from exc

    def run(self, case: str) -> Any:
        s = self.s
        if case not in ("single_module_output", "matched_retained", "ordered_dataset_features"):
            raise Unsupported(
                "a feature extractor reads outputs; it has no edit or gradient contract"
            )
        with torch.no_grad():
            if case == "single_module_output":
                return first_tensor(self.single(s.x)["f"])
            if case == "matched_retained":
                got = self.many(s.x)
                return [first_tensor(got[site]) for site in s.sites]
            return torch.cat([first_tensor(self.single(b)["f"]) for b in s.batches()])


class ThingsvisionArm:
    """thingsvision's extractor on its own task: ordered dataset features."""

    name = "thingsvision"

    def __init__(self, subject: Subject) -> None:
        tv = _require("thingsvision")
        self.s = subject
        self.extractor = tv.get_extractor_from_model(
            model=subject.model, device=str(subject.x.device), backend="pt"
        )

    def run(self, case: str) -> Any:
        if case != "ordered_dataset_features":
            raise Unsupported("thingsvision measures only its extraction task")
        s = self.s
        feats = self.extractor.extract_features(
            batches=list(s.batches()), module_name=s.site, flatten_acts=False, output_type="tensor"
        )
        return torch.as_tensor(feats)


class CaptumArm:
    """Captum ``LayerGradientXActivation`` without the activation factor: the layer gradient."""

    name = "captum"

    def __init__(self, subject: Subject) -> None:
        attr = _require("captum.attr")
        self.s = subject
        layer = subject.model.get_submodule(subject.site)

        def forward(inp: torch.Tensor) -> torch.Tensor:
            return first_tensor(subject.model(inp)).flatten(1).sum(1)

        self.method = attr.LayerGradientXActivation(forward, layer, multiply_by_inputs=False)

    def run(self, case: str) -> Any:
        if case != "same_site_gradient":
            raise Unsupported("Captum is measured only on its matching attribution contract")
        got = self.method.attribute(self.s.x)
        self.s.model.zero_grad(set_to_none=True)
        return first_tensor(got).detach()


class TransformerLensArm:
    """TransformerLens runs its own reimplementation of supported transformer families."""

    name = "transformer_lens"

    def __init__(self, subject: Subject) -> None:
        raise Unsupported(
            "TransformerLens reimplements its supported decoder families; this subject is not one "
            "(decoders are measured in the mech-interp workloads)"
        )


ARMS: dict[str, type] = {
    "raw": RawArm,
    "hooks": HooksArm,
    "tl_extract": TLExtractArm,
    "tl_trace_sparse": TLTraceSparseArm,
    "tl_record": TLRecordArm,
    "tl_bind": TLBindArm,
    "tl_extract_dataset": TLExtractDatasetArm,
    "nnsight": NNsightArm,
    "transformer_lens": TransformerLensArm,
    "tv_fx": TorchvisionFxArm,
    "thingsvision": ThingsvisionArm,
    "captum": CaptumArm,
}


def prepare_subject(
    name: str, model: nn.Module, x: Any, meta: dict[str, Any], device: str = "cpu"
) -> Subject:
    """Move a subject to ``device`` and fill its shapes and steering direction (untimed)."""

    model = model.to(device)
    dataset = make_dataset(x).to(device)
    x = x.to(device)
    subject = Subject(name, model, x, dict(meta), dataset)
    sites = list(dict.fromkeys([subject.site, subject.edit_site, *subject.sites]))
    outs = HooksArm(subject).read(x, sites)
    shapes = {site: tuple(out.shape) for site, out in zip(sites, outs, strict=True)}
    subject.meta["shapes"] = {k: list(v) for k, v in shapes.items()}
    subject.site_shape = shapes[subject.site]
    subject.edit_shape = shapes[subject.edit_site]
    if outs[sites.index(subject.edit_site)].is_floating_point():
        gen = torch.Generator().manual_seed(11)
        direction = torch.randn(subject.edit_shape, generator=gen)
        subject.direction = (direction / direction.norm()).to(device)
    return subject


def compare(got: Any, ref: Any) -> dict[str, Any]:
    """Compare an arm's result with the hooks reference.

    Returns
    -------
    dict[str, Any]
        ``max_abs_diff``, ``bit_identical`` and ``matched`` (within float32 noise), or a
        ``shape_mismatch`` description.
    """

    got_list = got if isinstance(got, list) else [got]
    ref_list = ref if isinstance(ref, list) else [ref]
    if len(got_list) != len(ref_list):
        return {"matched": False, "shape_mismatch": f"{len(got_list)} vs {len(ref_list)} tensors"}
    worst = 0.0
    identical = True
    for g, r in zip(got_list, ref_list, strict=True):
        g = first_tensor(g).detach().reshape(-1).float()
        r = first_tensor(r).detach().reshape(-1).float()
        if g.shape != r.shape:
            return {"matched": False, "shape_mismatch": f"{list(g.shape)} vs {list(r.shape)}"}
        diff = float((g - r).abs().max()) if g.numel() else 0.0
        worst = max(worst, diff)
        identical = identical and bool(torch.equal(g, r))
        if not torch.allclose(g, r, rtol=1e-5, atol=1e-6):
            return {"matched": False, "max_abs_diff": worst, "bit_identical": False}
    return {"matched": True, "max_abs_diff": worst, "bit_identical": identical}
