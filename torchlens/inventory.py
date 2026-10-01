"""Site inventory: the audience's "what can I extract?" answer (tvscope B6).

Two honest rungs. The NO-INPUT rung lists every registered module with its
address and class -- free, but shapes are unknown and it cannot see
functional (non-module) operations. The FORWARD rung runs ONE disclosed-cost
metadata-only real forward and reports every executed site: shapes, dtypes,
pass indices, and functional-vs-module origin. Sites on untaken
data-dependent branches are absent from a one-forward inventory, and the
result says so.

Structured rows, never formatted strings (the deferred-CLI plumbing rule);
the copy-paste-safe selector is the FIRST column, chosen so it round-trips
through ``tl.extract`` keyed by the requested string (the B6 round-trip
invariant), or the resolution helper raises a typed, useful ambiguity whose
remedy text shows a working spelling.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint; import as
``import torchlens.inventory``.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from torchlens._errors import _actionable_message, _ActionableErrorMixin
from torchlens.errors._base import ConfigurationError

if TYPE_CHECKING:
    from torch import nn

__tl_layer__ = "L6"


class SiteInventoryError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Typed selector-resolution refusal from the site inventory (B6).

    Codes: ``site_selector_unknown`` (nothing matched; remedy names the
    nearest working spellings) and ``site_selector_ambiguous`` (several
    sites matched; remedy lists every working spelling).
    """

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize a typed inventory refusal.

        Parameters
        ----------
        problem:
            What failed to resolve, naming the needle.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action, always showing a WORKING spelling.
        **context:
            Structured diagnostic context (candidate lists).
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class SiteRow:
    """One extractable site.

    Attributes
    ----------
    selector:
        Copy-paste-safe extraction selector (FIRST column): a qualified
        module address when it names exactly one executed site, else the
        exact (pass-qualified where multi-pass) layer label. Round-trips
        through ``tl.extract`` keyed by this string.
    kind:
        ``"module"`` (registered module row) or ``"op"`` (executed
        operation row from the forward rung).
    module_address:
        Qualified module address (``None`` for module-less functional ops).
    layer_label:
        Executed layer label (``None`` on the no-input rung).
    module_type:
        Module class name (``None`` for functional ops).
    origin:
        ``"module_output"`` / ``"function"`` / ``None`` (no-input rung).
    shape:
        Output shape tuple (``None`` on the no-input rung).
    dtype:
        Output dtype string (``None`` on the no-input rung).
    pass_index:
        1-based pass for recurrent sites (``None`` on the no-input rung).
    num_passes:
        Total passes of the owning layer (``None`` on the no-input rung).
    training:
        The owning module's ``training`` flag (reported, never mutated;
        ``None`` for module-less ops).
    """

    selector: str
    kind: str
    module_address: str | None
    layer_label: str | None
    module_type: str | None
    origin: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    pass_index: int | None
    num_passes: int | None
    training: bool | None

    def to_json(self) -> dict[str, Any]:
        """Serialize this row to a JSON-portable dict."""

        payload = {
            "selector": self.selector,
            "kind": self.kind,
            "module_address": self.module_address,
            "layer_label": self.layer_label,
            "module_type": self.module_type,
            "origin": self.origin,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "pass_index": self.pass_index,
            "num_passes": self.num_passes,
            "training": self.training,
        }
        return payload


@dataclass(frozen=True)
class SiteInventory:
    """The structured site inventory (tvscope B6).

    Attributes
    ----------
    rows:
        Site rows: registration order (no-input rung) or execution order
        (forward rung).
    rung:
        ``"modules"`` (no input; free) or ``"forward"`` (one disclosed-cost
        real forward).
    disclosures:
        Honest limits of this inventory (untaken branches absent, cost,
        hypothesis shapes).
    shapes_are_hypotheses:
        True when the backing capture was ``structure_only``: every shape is
        a HYPOTHESIS, not an observed value.
    """

    rows: tuple[SiteRow, ...]
    rung: str
    disclosures: tuple[str, ...]
    shapes_are_hypotheses: bool = False

    def __len__(self) -> int:
        """Number of site rows."""

        return len(self.rows)

    def __iter__(self) -> Any:
        """Iterate site rows."""

        return iter(self.rows)

    def to_json(self) -> dict[str, Any]:
        """Serialize the inventory (rows + disclosures) to JSON-portable form."""

        return {
            "schema": "tl_site_inventory_v1",
            "rung": self.rung,
            "shapes_are_hypotheses": self.shapes_are_hypotheses,
            "disclosures": list(self.disclosures),
            "rows": [row.to_json() for row in self.rows],
        }

    def resolve(self, needle: str) -> SiteRow:
        """Resolve a user spelling to exactly one site row (B6 invariant).

        Exact selector, module address, or layer label matches win. A bare
        leaf name (the audience's first question: ``"visual_projection"``)
        resolves when exactly one address ends with it; anything else raises
        typed with WORKING spellings in the remedy.

        Parameters
        ----------
        needle:
            Selector, address, label, or bare leaf-name spelling.

        Returns
        -------
        SiteRow
            The one matching row.

        Raises
        ------
        SiteInventoryError
            ``site_selector_ambiguous`` when several sites match;
            ``site_selector_unknown`` when none does.
        """

        exact = [
            row
            for row in self.rows
            if needle in (row.selector, row.module_address, row.layer_label)
        ]
        if len(exact) == 1:
            return exact[0]
        if not exact:
            exact = [
                row
                for row in self.rows
                if row.module_address is not None
                and row.module_address.rsplit(".", 1)[-1] == needle
            ]
            if len(exact) == 1:
                return exact[0]
        if exact:
            spellings = sorted({row.selector for row in exact})
            raise SiteInventoryError(
                f"site spelling {needle!r} matches {len(exact)} sites.",
                code="site_selector_ambiguous",
                remedy=(
                    "pick one working selector: "
                    + ", ".join(repr(s) for s in spellings[:8])
                    + ("..." if len(spellings) > 8 else "")
                ),
                needle=needle,
                candidates=spellings,
            )
        near = [
            row.selector
            for row in self.rows
            if needle.lower() in row.selector.lower()
            or (row.layer_label or "").lower().startswith(needle.lower())
        ]
        raise SiteInventoryError(
            f"site spelling {needle!r} matches no inventoried site.",
            code="site_selector_unknown",
            remedy=(
                ("nearest working selectors: " + ", ".join(repr(s) for s in near[:8]))
                if near
                else "list the inventory rows (inventory.rows) and copy a selector"
            ),
            needle=needle,
            candidates=near[:20],
        )


def _module_rung(model: nn.Module) -> SiteInventory:
    """Build the no-input module listing (rung one)."""

    rows = [
        SiteRow(
            selector=address,
            kind="module",
            module_address=address,
            layer_label=None,
            module_type=type(module).__name__,
            origin=None,
            shape=None,
            dtype=None,
            pass_index=None,
            num_passes=None,
            training=bool(module.training),
        )
        for address, module in model.named_modules()
        if address
    ]
    return SiteInventory(
        rows=tuple(rows),
        rung="modules",
        disclosures=(
            "no-input listing: shapes, dtypes, and functional (non-module) "
            "operations are unknown until a forward runs -- call "
            "list_sites(model, x) for one disclosed-cost real forward",
            "module training flags are reported as found, never mutated",
        ),
    )


def _op_selector(trace: Any, layer: Any, address_counts: Counter[str]) -> str:
    """Pick the copy-paste-safe selector for one executed layer.

    The qualified module address (the INNERMOST owning module --
    ``output_of_modules`` is innermost-first) wins ONLY when the engine's
    own matcher certifies it: it names exactly one executed site in this
    inventory AND ``trace[address]`` resolves to this very layer (a
    multi-output container module's address is engine-ambiguous even when
    the inventory holds one row for it). Otherwise the exact layer label,
    pass-qualified when the owning layer is multi-pass. This certification
    IS the B6 round-trip invariant, enforced at emission time.

    Parameters
    ----------
    trace:
        The backing trace (the certification oracle).
    layer:
        Executed layer record.
    address_counts:
        How many executed layers each module address owns as output.

    Returns
    -------
    str
        The selector string.
    """

    outputs_of = tuple(getattr(layer, "output_of_modules", ()) or ())
    address = outputs_of[0] if outputs_of else None
    if address and address_counts[address] == 1:
        # engine-ambiguous or unresolvable addresses fall to the label
        with contextlib.suppress(Exception):
            resolved = trace[str(address)]
            if str(getattr(resolved, "layer_label", "")) == str(layer.layer_label):
                return str(address)
    if int(getattr(layer, "num_passes", 1) or 1) > 1:
        return str(layer.label)
    return str(layer.layer_label)


def sites_of_trace(trace: Any) -> SiteInventory:
    """Build the forward-rung inventory from an existing trace (B6).

    Parameters
    ----------
    trace:
        Any TorchLens ``Trace`` (a metadata-only capture suffices; a
        ``structure_only`` capture serves shapes labeled as hypotheses).

    Returns
    -------
    SiteInventory
        One row per executed non-input/output layer, execution order.
    """

    address_counts: Counter[str] = Counter()
    layers = [
        layer
        for layer in trace.layer_list
        if getattr(layer, "layer_type", None) not in {"input", "output", "buffer"}
    ]
    for layer in layers:
        outputs_of = tuple(getattr(layer, "output_of_modules", ()) or ())
        if outputs_of:
            address_counts[outputs_of[0]] += 1
    rows: list[SiteRow] = []
    for layer in layers:
        outputs_of = tuple(getattr(layer, "output_of_modules", ()) or ())
        address = outputs_of[0] if outputs_of else None
        shape = getattr(layer, "shape", None)
        module_record = _module_record_of(trace, address)
        rows.append(
            SiteRow(
                selector=_op_selector(trace, layer, address_counts),
                kind="op",
                module_address=str(address) if address else None,
                layer_label=str(layer.layer_label),
                module_type=(str(module_record.class_name) if module_record is not None else None),
                origin="module_output" if address else "function",
                shape=tuple(int(d) for d in shape) if shape is not None else None,
                dtype=str(getattr(layer, "dtype", None)),
                pass_index=int(getattr(layer, "pass_index", 1) or 1),
                num_passes=int(getattr(layer, "num_passes", 1) or 1),
                training=(
                    bool(module_record.training)
                    if module_record is not None and module_record.training is not None
                    else None
                ),
            )
        )
    structure_only = bool(getattr(trace, "structure_only", False))
    disclosures = [
        "one real forward: sites on untaken data-dependent branches are absent from this inventory",
        "module training flags are reported as found, never mutated",
    ]
    if structure_only:
        disclosures.append(
            "structure_only capture: every shape is a HYPOTHESIS, not an observed value"
        )
    return SiteInventory(
        rows=tuple(rows),
        rung="forward",
        disclosures=tuple(disclosures),
        shapes_are_hypotheses=structure_only,
    )


def _module_record_of(trace: Any, address: str | None) -> Any:
    """Look up a module record by address, tolerating misses."""

    if not address:
        return None
    record = None
    with contextlib.suppress(Exception):
        record = trace.modules[str(address)]
    return record


def list_sites(
    model: nn.Module,
    x: Any = None,
    *,
    input_kwargs: dict[str, Any] | None = None,
    capture: Any = None,
) -> SiteInventory:
    """Inventory a model's extractable sites (tvscope B6; two honest rungs).

    Parameters
    ----------
    model:
        PyTorch model to inventory.
    x:
        Optional input. Omitted: the free no-input module listing (shapes
        unknown). Supplied: ONE disclosed-cost metadata-only real forward
        (no activations retained) reporting shapes, dtypes, pass indices,
        and functional-vs-module origin.
    input_kwargs:
        Optional keyword inputs for the forward rung (kwargs-only models).
    capture:
        Optional ``CaptureOptions`` override for the forward rung, used
        verbatim (e.g. ``structure_only=True``; shapes are then labeled
        hypotheses). Default is a metadata-only capture retaining no
        activation payloads.

    Returns
    -------
    SiteInventory
        Structured rows with the copy-paste-safe selector first.
    """

    if x is None and input_kwargs is None:
        return _module_rung(model)
    import torchlens as tl

    options = capture if capture is not None else tl.options.CaptureOptions(layers_to_save=None)
    trace = tl.trace(model, x if x is not None else (), input_kwargs, capture=options)
    return sites_of_trace(trace)


__all__ = [
    "SiteInventory",
    "SiteInventoryError",
    "SiteRow",
    "list_sites",
    "sites_of_trace",
]
