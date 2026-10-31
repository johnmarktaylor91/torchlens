"""SAE latents as a facet family, and the logged splice that puts them in a trace.

``register(name, sae, site=...)`` makes the latents of ``sae`` a facet family
named ``name`` at ``site``: ``tl.facet(name).latent(k)`` addresses latent
``k`` in every verb a ``tl.head`` selector serves, and
``trace[site].facets[name]`` (or ``f"{name}[{k}]"``) reads it. A read is the
encoder activation. A write edits the latents and adds the decoder-space
change ``decode(z') - decode(z)`` to the site, so the SAE's reconstruction
error (the site minus ``decode(z)``) is carried unchanged; for an affine
decoder that change is ``delta * W_dec[k]``. Nothing here needs an SAE
package: the SAE is duck-typed (``encode`` and ``decode``).

Every spelling in this module is DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from ._utils import module_for_site, resolve_one_site

_DOCS = "docs/recipes/pretrained_saes.md"
_INVALID_NAME_CHARS = frozenset("[],")


def require_codec(sae: Any, door: str = "SAE splice") -> None:
    """Refuse SAE objects without an encode/decode pair.

    Raises
    ------
    TypeError
        If ``sae`` does not expose callable ``encode`` and ``decode``.
    """

    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise TypeError(
            f"{door} requires an SAE exposing encode(...) and decode(...); "
            f"got {type(sae).__name__!r}. Any object with that pair works -- "
            "an SAE Lens SAE, or a hand-built module."
        )


def require_live_model(model_or_log: Any) -> None:
    """Refuse ``splice(..., log_latents=True)`` from a Trace (there are no inputs to rerun).

    Raises
    ------
    InvalidArgumentError
        ``sae_splice_logged_needs_live_model``.
    """

    if not isinstance(model_or_log, nn.Module):
        raise InvalidArgumentError(
            "splice(..., log_latents=True) captures the live model again with the SAE inside "
            "it, so it needs the model and its inputs; it got a Trace",
            code="sae_splice_logged_needs_live_model",
            remedy="call splice(model, inputs, site=..., sae=..., log_latents=True) with the "
            f"live model; see {_DOCS}",
        )


def _latent_width(sae: Any) -> int | None:
    """Return the SAE's latent count when it is declared (``W_dec`` rows or ``cfg.d_sae``)."""

    w_dec = getattr(sae, "W_dec", None)
    if isinstance(w_dec, torch.Tensor) and w_dec.ndim == 2:
        return int(w_dec.shape[0])
    d_sae = getattr(getattr(sae, "cfg", None), "d_sae", None)
    return int(d_sae) if isinstance(d_sae, int) and not isinstance(d_sae, bool) else None


def _index_key(index: int | tuple[int, ...]) -> int | list[int]:
    """Return the last-axis key for a latent index or tuple of indices."""

    return list(index) if isinstance(index, tuple) else index


@dataclass(frozen=True, eq=False)
class _LatentCodec:
    """Read and write chosen latents of one SAE at its home activation.

    Parameters
    ----------
    name:
        Registered family name (for display).
    sae:
        Duck-typed SAE.
    index:
        Latent index, tuple of indices, or ``None`` for every latent.
    """

    name: str
    sae: Any
    index: int | tuple[int, ...] | None

    def read(self, value: torch.Tensor) -> torch.Tensor:
        """Return the encoder activation of the chosen latents of ``value``."""

        latents = self._encode(value)
        return latents if self.index is None else latents[..., _index_key(self.index)]

    def scatter_update(
        self, home_out: torch.Tensor, edited: torch.Tensor, *, mode: str = "replace"
    ) -> torch.Tensor:
        """Return ``home_out`` plus the decoder-space change of the edited latents.

        Parameters
        ----------
        home_out:
            Site activation before the edit.
        edited:
            New value of the chosen latents (``mode="replace"``) or the
            amount added to them (``mode="add"``).
        mode:
            ``"replace"`` or ``"add"``.

        Returns
        -------
        torch.Tensor
            ``home_out + decode(z') - decode(z)``: the reconstruction error
            ``home_out - decode(z)`` is unchanged.
        """

        if mode not in {"replace", "add"}:
            raise ValueError("Facet scatter_update mode must be 'replace' or 'add'.")
        latents = self._encode(home_out)
        changed = latents.clone()
        if self.index is None:
            changed = changed + edited if mode == "add" else edited.to(latents.dtype)
        elif mode == "add":
            changed[..., _index_key(self.index)] += edited.to(latents.dtype)
        else:
            changed[..., _index_key(self.index)] = edited.to(latents.dtype)
        delta = self._decode(changed) - self._decode(latents)
        return home_out + delta.to(device=home_out.device, dtype=home_out.dtype)

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``sae.encode(value)``, refusing a non-tensor result."""

        latents = self.sae.encode(value)
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                f"SAE encode(...) must return a torch.Tensor; got {type(latents).__name__!r}"
            )
        return latents

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return ``sae.decode(latents)``, refusing a non-tensor result."""

        reconstruction = self.sae.decode(latents)
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError(
                f"SAE decode(...) must return a torch.Tensor; got {type(reconstruction).__name__!r}"
            )
        return reconstruction

    def __repr__(self) -> str:
        """Return the facet spelling these latents answer to."""

        return f"sae_latents({self.name!r}, {self.index!r})"


@dataclass(eq=False)
class _SAEFamily:
    """One registered SAE: its name, codec and the site its latents live at.

    Parameters
    ----------
    name:
        Registered name, the facet name its latents answer to.
    sae:
        Duck-typed SAE.
    site:
        Site spelling resolved on each trace: a string names one op (it
        must match exactly one); a selector may fan out over several.
    """

    name: str
    sae: Any
    site: Any
    width: int | None = None
    _homes: dict[int, tuple[Any, frozenset[str] | InvalidArgumentError]] = field(
        default_factory=dict, repr=False
    )

    def facet_spec(self, record: Any, index: int | tuple[int, ...] | None) -> Any:
        """Return the latents facet on ``record``, or ``None`` when it is not the home.

        Raises
        ------
        InvalidArgumentError
            ``sae_latent_site_ambiguous`` when a string site does not name one
            op of a single-pass layer of the record's trace,
            ``sae_latent_site_unmatched`` when a selector site matches none,
            ``sae_latent_index_unknown`` when ``index`` is outside the SAE's
            latents.
        """

        homes = self._home_labels(record)
        if isinstance(homes, InvalidArgumentError):
            raise homes
        if homes is None or _record_label(record) not in homes:
            return None
        if index is not None:
            self._check_range(index, record)
        return self._spec(record, index)

    def listed_spec(self, record: Any) -> Any:
        """Return the all-latents facet on ``record`` for listings; never raises."""

        homes = self._home_labels(record)
        if not isinstance(homes, frozenset) or _record_label(record) not in homes:
            return None
        return self._spec(record, None)

    def selector_name(self, index: Any) -> str:
        """Return the facet name of latents ``index``, refusing an index the SAE lacks."""

        indices = _as_indices(self.name, index)
        if self.width is not None:
            self._check_width(indices, self.width)
        joined = ",".join(str(item) for item in indices)
        # A one-latent list keeps its latent axis: "res[3,]" reads shape (..., 1).
        trailing = "," if isinstance(index, Sequence) and len(indices) == 1 else ""
        return f"{self.name}[{joined}{trailing}]"

    def _spec(self, record: Any, index: int | tuple[int, ...] | None) -> Any:
        """Build the codec facet spec anchored to ``record``."""

        from ..semantic.facets import FacetSpec, TransformPrimitive

        codec = _LatentCodec(self.name, self.sae, index)
        primitive = TransformPrimitive("codec", (codec,), capability_class="codec")
        home = FacetSpec.from_home(record, function_id=f"sae:{self.name}")
        return home._append(primitive)

    def _check_range(self, index: int | tuple[int, ...], record: Any) -> None:
        """Refuse latents outside the SAE's width (read from the home when undeclared)."""

        width = self.width
        if width is None:
            out = getattr(record, "out", None)
            if not isinstance(out, torch.Tensor):
                return
            with torch.no_grad():
                width = int(_LatentCodec(self.name, self.sae, None).read(out).shape[-1])
        self._check_width(index if isinstance(index, tuple) else (index,), width)

    def _check_width(self, indices: tuple[int, ...], width: int) -> None:
        """Refuse any index outside ``range(width)``."""

        outside = [item for item in indices if item >= width]
        if outside:
            raise InvalidArgumentError(
                f"tl.facet({self.name!r}).latent(...) asks for latent {outside[0]}, but the SAE "
                f"registered as {self.name!r} has {width} latents (0 to {width - 1})",
                code="sae_latent_index_unknown",
                remedy=f"name a latent from 0 to {width - 1}, e.g. "
                f"tl.facet({self.name!r}).latent({width - 1}); see {_DOCS}",
            )

    def _home_labels(self, record: Any) -> frozenset[str] | InvalidArgumentError | None:
        """Return the op labels of this family's site on the record's trace (cached)."""

        trace = getattr(record, "trace", None) or getattr(record, "source_trace", None)
        if trace is None or not hasattr(trace, "_resolve_sites"):
            return None
        cached = self._homes.get(id(trace))
        if cached is not None and cached[0]() is trace:
            return cached[1]
        homes = self._resolve_homes(trace)
        self._homes[id(trace)] = (weakref.ref(trace), homes)
        return homes

    def _resolve_homes(self, trace: Any) -> frozenset[str] | InvalidArgumentError:
        """Resolve the site on ``trace``: one op for a string, every match for a selector.

        A latent edit installs its hook on the home's layer, which fires on
        every pass of that layer; so a string site must name a layer that
        runs once, and a selector keeps one home per matched layer.
        """

        from ..intervention.errors import MultiMatchWarning, SiteResolutionError

        if not isinstance(self.site, str):
            # The edit's own fan-out warning reports the count; resolving the
            # homes for a listing must not warn as well.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", MultiMatchWarning)
                    matched = trace._resolve_sites(self.site, max_fanout=len(trace.layer_list) or 1)
            except (SiteResolutionError, InvalidArgumentError, KeyError, ValueError) as exc:
                return InvalidArgumentError(
                    f"the SAE registered as {self.name!r} sits at {self.site!r}, which matches "
                    f"no op of this trace ({exc})",
                    code="sae_latent_site_unmatched",
                    remedy="read and edit its latents on a trace of the model the SAE was "
                    f"registered for, or register it at a site this model has; see {_DOCS}",
                )
            first_pass: dict[Any, str] = {}
            for op in matched:
                first_pass.setdefault(getattr(op, "layer_label", None), str(op.label))
            return frozenset(first_pass.values())
        try:
            op = resolve_one_site(trace, self.site)
        except (SiteResolutionError, InvalidArgumentError, KeyError, ValueError) as exc:
            return self._ambiguous(
                f"it does not name exactly one op of this trace ({exc})", f"{self.site}:1"
            )
        if getattr(op, "layer_label", None) is None:
            return self._ambiguous(
                f"it names a {type(op).__name__} record, not one op (the module runs more "
                "than once)",
                _first_op_label(op, self.site),
            )
        passes = sum(
            1
            for record in trace.layer_list
            if getattr(record, "layer_label", None) == getattr(op, "layer_label", None)
        )
        if passes > 1:
            return self._ambiguous(
                f"it names {op.label!r}, one pass of a layer that runs {passes} times, and a "
                "latent edit there would fire on every pass",
                str(op.label),
            )
        return frozenset({str(op.label)})

    def _ambiguous(self, why: str, one_pass: str) -> InvalidArgumentError:
        """Build the coded refusal for a string site that is not one single-pass op.

        The remedy edits the latents of one pass through the splice door, whose
        hook targets that pass alone; ``one_pass`` names the pass and rides in
        the error's ``one_pass_site`` field.
        """

        return InvalidArgumentError(
            f"the SAE registered as {self.name!r} sits at {self.site!r}, but {why}",
            code="sae_latent_site_ambiguous",
            remedy="edit the latents of one pass with tl.bridge.sae.splice(model, inputs, "
            f"site={one_pass!r}, sae=sae, latents_edit=fn); see {_DOCS}",
            one_pass_site=one_pass,
        )


def _first_op_label(record: Any, site: str) -> str:
    """Return the label of a module record's first op, else ``site`` with pass ``1``."""

    for op in getattr(record, "ops", None) or ():
        return str(getattr(op, "label", op))
    return f"{site}:1"


def _record_label(record: Any) -> Any:
    """Return the pass-qualified label of an op record (``None`` for a module record)."""

    return getattr(record, "label", None) if hasattr(record, "layer_label") else None


def _as_indices(name: str, index: Any) -> tuple[int, ...]:
    """Return ``index`` as a tuple of latent indices, refusing other types."""

    def _is_index(value: Any) -> bool:
        """Return whether ``value`` is a non-negative ``int`` (``bool`` excluded)."""
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    if _is_index(index):
        return (int(index),)
    if (
        isinstance(index, Sequence)
        and not isinstance(index, str)
        and len(index) > 0
        and all(_is_index(item) for item in index)
    ):
        return tuple(int(item) for item in index)
    raise InvalidArgumentError(
        f"tl.facet({name!r}).latent({index!r}): a latent is a non-negative int, or a "
        "non-empty list of them",
        code="sae_latent_index_invalid",
        remedy=f"pass an int or a list of ints, e.g. tl.facet({name!r}).latent(0); see {_DOCS}",
    )


def register(name: str, sae: Any, *, site: Any) -> None:
    """Make the latents of ``sae`` a facet family called ``name`` at ``site``.

    After ``tl.bridge.sae.register("res", sae, site="blocks.4")``,
    ``tl.facet("res").latent(3442)`` addresses latent 3442 wherever a
    ``tl.head`` selector works (``Trace.set``, ``attach_hooks``, ``do``,
    ``fork().do``), and ``trace["blocks.4"].facets["res[3442]"]`` reads it.
    Registration is process-wide and seen by traces captured before it.

    Parameters
    ----------
    name:
        Family name: no brackets, commas or whitespace.
    sae:
        Duck-typed SAE exposing ``encode`` and ``decode``.
    site:
        Where the SAE reads: a layer label or module address (must name one
        op of each trace), or a selector (every match is a home).

    Raises
    ------
    TypeError
        If ``sae`` lacks an encode/decode pair.
    InvalidArgumentError
        ``sae_family_name_invalid`` for a malformed name,
        ``sae_family_name_taken`` when the name is registered already.
    """

    from ..semantic import _families

    require_codec(sae, "tl.bridge.sae.register")
    if (
        not isinstance(name, str)
        or not name
        or any(char in _INVALID_NAME_CHARS or char.isspace() for char in name)
    ):
        raise InvalidArgumentError(
            f"tl.bridge.sae.register({name!r}, ...): an SAE name is a non-empty string "
            "without brackets, commas or whitespace",
            code="sae_family_name_invalid",
            remedy=f"name it plainly, e.g. tl.bridge.sae.register('res', sae, site=...); see {_DOCS}",
        )
    if name in _families._FACET_FAMILIES:
        raise InvalidArgumentError(
            f"an SAE is already registered as {name!r}",
            code="sae_family_name_taken",
            remedy=f"tl.bridge.sae.unregister({name!r}) first, or register under a new name",
        )
    _families._FACET_FAMILIES[name] = _SAEFamily(
        name=name, sae=sae, site=site, width=_latent_width(sae)
    )


def unregister(name: str) -> None:
    """Remove the SAE registered as ``name``.

    Raises
    ------
    InvalidArgumentError
        ``sae_family_unknown`` when nothing is registered under ``name``.
    """

    from ..semantic import _families

    if name not in _families._FACET_FAMILIES:
        registered = sorted(_families._FACET_FAMILIES)
        raise InvalidArgumentError(
            f"no SAE is registered as {name!r}; registered: {registered if registered else 'none'}",
            code="sae_family_unknown",
            remedy="unregister a registered name, or nothing",
        )
    del _families._FACET_FAMILIES[name]


def logged_splice_trace(
    model: nn.Module, inputs: Any, clean: Any, site_label: str, reconstruction: Any
) -> tuple[Any, str]:
    """Capture ``model`` with the SAE spliced in as ordinary logged ops.

    The module whose output is the site gets a temporary forward hook that
    encodes that output, applies any latent edit and decodes, under capture,
    so the encoder ops, the latents and the decoder ops are trace nodes
    between the site and its consumers. For the capture the reconstruction
    module (holding the SAE) is a temporary child of ``model``, so the SAE's
    parameters are known parameter sources rather than tensors from outside
    the model; the child itself never runs as a module call and is removed
    after. ``tl.release_model`` runs before and after, so the capture
    prepares the tree with the child and the next one without it.

    Parameters
    ----------
    model:
        Live model.
    inputs:
        Its inputs.
    clean:
        Clean trace, used to find the module whose output is the site.
    site_label:
        Resolved site label.
    reconstruction:
        The splice's reconstruction module (its SAE, latent edit and kept
        leading positions); the hook runs the same arithmetic as its forward.

    Returns
    -------
    tuple[Any, str]
        The logged trace and the label of its latents node.

    Raises
    ------
    InvalidArgumentError
        ``bridge_module_site_*`` when no single module call outputs the site
        (the hook needs a module); ``sae_splice_logged_output_not_tensor``
        when that module returns a non-tensor.
    """

    import torchlens as tl

    from .._utils.tensor_utils import get_tensor_label

    module = module_for_site(clean, site_label, bridge="splice(..., log_latents=True)")
    raw: dict[str, str | None] = {}

    def _keep(latents: torch.Tensor) -> None:
        """Record the raw label of the latents tensor the hook produced."""
        raw["latents"] = get_tensor_label(latents)

    def _hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> torch.Tensor:
        """Replace the site module's output with the SAE reconstruction plus its error."""
        if not isinstance(output, torch.Tensor):
            raise InvalidArgumentError(
                f"splice(..., log_latents=True) at {site_label!r}: the module whose output is "
                f"the site returns a {type(output).__name__}, not a tensor",
                code="sae_splice_logged_output_not_tensor",
                remedy="splice without log_latents=True, or at a site a module returns as a "
                f"tensor; see {_DOCS}",
            )
        from .sae import _encode_decode

        latents, _edited, replacement = _encode_decode(
            reconstruction._sae, output, reconstruction._latents_edit, reconstruction._start
        )
        _keep(latents)
        return replacement

    child = _free_child_name(model)
    # Releasing the model makes the next capture prepare its module tree from
    # scratch, so the temporary child (and any submodules the SAE calls) carry
    # TorchLens metadata; releasing again after removing it leaves the model
    # as a plain module whose next capture prepares the tree it really has.
    tl.release_model(model)
    model.add_module(child, reconstruction)
    handle = module.register_forward_hook(_hook)
    try:
        logged = tl.trace(model, inputs, capture=tl.options.CaptureOptions(intervention_ready=True))
    finally:
        handle.remove()
        del model._modules[child]
        tl.release_model(model)
    return logged, str(_record_label(logged[raw["latents"]]))


def _free_child_name(model: nn.Module) -> str:
    """Return a child-module name ``model`` does not use yet."""

    name, counter = "sae_splice", 0
    while hasattr(model, name):
        counter += 1
        name = f"sae_splice_{counter}"
    return name
