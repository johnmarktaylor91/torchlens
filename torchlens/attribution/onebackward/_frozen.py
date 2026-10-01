"""``frozen=`` linearization policy (M(reads) item 3, decisions D5/D6).

Three-state contract: omitted / ``DEFAULT`` freezes every trustworthy
executed transformer MLP-output facet home (the published attribution-graph
semantics, per the digest's unqualified ruling) for EVERY target kind;
``frozen=None`` is the explicit total derivative with no TorchLens-added
stops; an explicit ACT Selection freezes exactly that, REPLACING (never
unioning with) the default.

Mechanism: temporary slot-targeted ``Node.register_prehook`` hooks on
registry nodes -- on the registry substrate there is no live tensor to hook,
so prehooks are forced, not preferred (the payload-route alternative
measurably installs hooks that never fire: a silent no-op freeze, the worst
failure mode in the feature). Hook bodies are batch-compatible arithmetic
(multiply only; no detach/clone/item/views under vmap); every handle is
removed in ``finally``, with restoration measured bit-exact in the tests.

Documented semantics, verbatim (D6): "prehook at this node, post-freeze for
downstream frozen sites" -- a frozen site's OWN row is NOT its gradient from
an otherwise-unfrozen graph. ``frozen_applied`` is cone-dependent: assert
``<=`` requested and non-empty, never an exact count.

The frozen-default disclosure (FORK-1) ships as ONE detector with BOTH
branches wired behind ``FROZEN_DEFAULT_DISCLOSURE_BRANCH``: branch W (the
panel recommendation and the shipped default) warns once per process and
proceeds with MLP-out stops; branch R refuses typed until the user chooses.
Both branches share the one message and both are tested green.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from ...errors import TorchLensWarning
from ...utils._torch_compat import HAS_NODE_PREHOOK
from ._accessor import ReadEdgeIndex, SiteEdge
from ._errors import ReadError
from ._facet_bridge import FrozenResolution, resolve_frozen_policy
from ._targets import resolve_site_edge

__all__ = [
    "DEFAULT",
    "FrozenPlan",
    "resolve_frozen",
    "install_freeze_hooks",
    "FROZEN_DEFAULT_DISCLOSURE_BRANCH",
]


class _DefaultFrozenSentinel:
    """Sentinel type for the explicit default-policy spelling."""

    def __repr__(self) -> str:
        return "torchlens.attribution.onebackward.DEFAULT"


DEFAULT = _DefaultFrozenSentinel()
"""Explicit spelling of the default frozen policy (placeholder name)."""

# FORK-1 branch constant: "warn" (branch W, panel recommendation 3-0, the
# digest ruling's reading) or "refuse" (branch R, fully designed and kept
# buildable). One word flips the shipped behavior; both branches are tested.
FROZEN_DEFAULT_DISCLOSURE_BRANCH: str = "warn"

_DISCLOSURE_WARNED: set[str] = set()

_DISCLOSURE_MESSAGE = (
    "frozen= was omitted on an attention-bearing model: the default "
    "linearization freezes MLP-output sites (the published attribution-graph "
    "direct-edge semantics), which deliberately changes transformer saliency "
    "numbers relative to Captum-family tools. frozen=None = total "
    "derivative, the Captum-familiar number; frozen=tl DEFAULT / omitted = "
    "published direct-edge linearization. Remedy: pass frozen= explicitly "
    "to silence this disclosure"
)


@dataclass(frozen=True)
class FrozenPlan:
    """The resolved freeze plan one read executes under.

    Attributes
    ----------
    policy:
        ``'mlp_out'`` (default policy), ``'none'``, or ``'explicit'``.
    sites:
        label -> :class:`SiteEdge` freeze sites.
    masks:
        label -> freeze mask (``True`` = frozen element) or ``None`` for a
        whole-slot freeze.
    digest:
        Stable digest of (policy, sorted site labels) for provenance and
        row stamping.
    alias_disclosures:
        label -> sibling labels sharing the frozen ``(node, slot)``:
        freezing one alias freezes the whole group and says so (D11).
    disclosures:
        Policy-level disclosure strings (for provenance warnings).
    """

    policy: str
    sites: dict[str, SiteEdge]
    masks: dict[str, torch.Tensor | None]
    digest: str
    alias_disclosures: dict[str, tuple[str, ...]]
    disclosures: tuple[str, ...]


def _plan_digest(policy: str, labels: tuple[str, ...]) -> str:
    """Return the stable digest for a frozen plan."""

    hasher = hashlib.sha256()
    hasher.update(policy.encode("utf-8"))
    for label in sorted(labels):
        hasher.update(b"\x00")
        hasher.update(label.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _alias_disclosures(
    index: ReadEdgeIndex, sites: dict[str, SiteEdge]
) -> dict[str, tuple[str, ...]]:
    """Map each freeze site to the sibling labels its freeze also affects."""

    disclosures: dict[str, tuple[str, ...]] = {}
    for label, edge in sites.items():
        members = index.alias_groups.get(edge.alias_key)
        if members:
            siblings = tuple(member for member in members if member != label)
            if siblings:
                disclosures[label] = siblings
    return disclosures


def _emit_default_disclosure(method: str) -> tuple[str, ...]:
    """Run the FORK-1 detector's disclosure arm; returns warning codes emitted.

    Branch W warns exactly once per process; branch R refuses typed with the
    same message. The detector itself (attention-bearing AND gradient-bearing
    method AND ``frozen=`` omitted AND resolved set non-empty) is evaluated
    by the caller; this helper only executes the chosen branch.
    """

    if FROZEN_DEFAULT_DISCLOSURE_BRANCH == "refuse":
        raise ReadError(
            _DISCLOSURE_MESSAGE + ". Remedy: pass frozen=None or an explicit "
            "frozen= Selection (nothing was computed)",
            code="read_frozen_choice_required",
            method=method,
        )
    if "frozen_default" not in _DISCLOSURE_WARNED:
        _DISCLOSURE_WARNED.add("frozen_default")
        warnings.warn(
            TorchLensWarning(
                _DISCLOSURE_MESSAGE,
                code="read_frozen_default_linearization",
            ),
            stacklevel=4,
        )
    return ("read_frozen_default_linearization",)


def _resolve_explicit(
    trace: Any, index: ReadEdgeIndex, frozen: Any
) -> tuple[dict[str, SiteEdge], dict[str, torch.Tensor | None]]:
    """Resolve an explicit ``frozen=`` request to sites and masks.

    Accepts a site label, an Op/Layer, an iterable of those, or anything
    selection-shaped (resolved against the trace; ACT kind only; element
    masks freeze exactly the masked elements of the slot).
    """

    from ...selection import ResolvedSelection, Selection

    sites: dict[str, SiteEdge] = {}
    masks: dict[str, torch.Tensor | None] = {}
    selection = None
    if isinstance(frozen, ResolvedSelection):
        selection = frozen
    elif isinstance(frozen, Selection) or hasattr(frozen, "__selection__"):
        lifted = frozen if isinstance(frozen, Selection) else frozen.__selection__()
        if getattr(lifted, "kind", "ACT") != "ACT":
            raise ReadError(
                f"frozen= must be an ACT selection; got kind "
                f"{getattr(lifted, 'kind', None)!r}. Remedy: select "
                "activation sites",
                code="read_frozen_selection_invalid",
                kind=getattr(lifted, "kind", None),
            )
        selection = lifted.resolve(trace)
    if selection is not None:
        if selection.kind != "ACT":
            raise ReadError(
                f"frozen= must be an ACT selection; got kind "
                f"{selection.kind!r}. Remedy: select activation sites",
                code="read_frozen_selection_invalid",
                kind=selection.kind,
            )
        for entry in selection:
            site_key = entry.site_key
            if len(site_key) != 2:
                raise ReadError(
                    f"frozen= selection entry {site_key!r} is not an ACT "
                    "site. Remedy: select activation sites",
                    code="read_frozen_selection_invalid",
                    site_key=list(site_key),
                )
            label = f"{site_key[0]}:{site_key[1]}"
            edge = index.edges.get(label)
            if edge is None:
                raise ReadError(
                    f"frozen= names site {label!r}, which has no usable "
                    "autograd (node, slot) on this trace. Remedy: freeze "
                    "differentiable op sites",
                    code="read_frozen_selection_invalid",
                    label=label,
                )
            sites[label] = edge
            mask = entry.mask
            masks[label] = None if bool(mask.all()) else mask
        return sites, masks
    requests = frozen if isinstance(frozen, (list, tuple, set, frozenset)) else [frozen]
    for request in requests:
        edge = resolve_site_edge(index, request, role="frozen")
        sites[edge.label] = edge
        masks[edge.label] = None
    return sites, masks


def resolve_frozen(
    trace: Any,
    index: ReadEdgeIndex,
    frozen: Any,
    *,
    method: str,
) -> tuple[FrozenPlan, tuple[str, ...]]:
    """Resolve the three-state ``frozen=`` contract to a :class:`FrozenPlan`.

    Parameters
    ----------
    trace:
        The resolution trace.
    index:
        Its addressable-edge index.
    frozen:
        Omitted (``DEFAULT`` sentinel), ``None`` (total derivative), or an
        explicit ACT selection / site spec / iterable of site specs.
    method:
        The read method; the disclosure detector only fires for
        gradient-bearing methods.

    Returns
    -------
    tuple[FrozenPlan, tuple[str, ...]]
        The plan and any emitted warning codes.
    """

    warning_codes: tuple[str, ...] = ()
    if frozen is None:
        return (
            FrozenPlan(
                policy="none",
                sites={},
                masks={},
                digest=_plan_digest("none", ()),
                alias_disclosures={},
                disclosures=(),
            ),
            warning_codes,
        )
    if isinstance(frozen, _DefaultFrozenSentinel):
        resolution: FrozenResolution = resolve_frozen_policy(trace, index, "mlp_out")
        gradient_bearing = method in ("activation_x_grad", "grad")
        if resolution.attention_bearing and gradient_bearing and resolution.sites:
            warning_codes = _emit_default_disclosure(method)
        sites = dict(resolution.sites)
        return (
            FrozenPlan(
                policy="mlp_out",
                sites=sites,
                masks=dict.fromkeys(sites),
                digest=_plan_digest("mlp_out", tuple(sites)),
                alias_disclosures=_alias_disclosures(index, sites),
                disclosures=resolution.disclosures,
            ),
            warning_codes,
        )
    sites, masks = _resolve_explicit(trace, index, frozen)
    return (
        FrozenPlan(
            policy="explicit",
            sites=sites,
            masks=masks,
            digest=_plan_digest("explicit", tuple(sites)),
            alias_disclosures=_alias_disclosures(index, sites),
            disclosures=(),
        ),
        warning_codes,
    )


class FreezeHooks:
    """Installed freeze prehooks with fired-tracking and exact removal.

    Use as a context manager around the engine call; ``fired_labels`` is
    valid after the block. Removal runs in ``finally`` and restores
    gradients bit-exact (pinned by the acceptance tests).
    """

    def __init__(self, plan: FrozenPlan) -> None:
        """Prepare (but do not install) hooks for a plan."""

        self._plan = plan
        self._handles: list[Any] = []
        self.fired_labels: set[str] = set()

    def __enter__(self) -> FreezeHooks:
        """Install one slot-targeted prehook per unique freeze ``(node, slot)``."""

        if self._plan.sites and not HAS_NODE_PREHOOK:
            raise ReadError(
                "frozen= needs torch autograd Node.register_prehook, which "
                "this torch build does not provide. Remedy: upgrade torch "
                "or pass frozen=None",
                code="onebackward_torch_unsupported",
            )
        installed: set[tuple[int, int]] = set()
        for label, edge in self._plan.sites.items():
            if edge.alias_key in installed:
                continue
            installed.add(edge.alias_key)
            self._handles.append(
                edge.node.register_prehook(
                    self._make_prehook(label, edge.slot, self._plan.masks.get(label))
                )
            )
        return self

    def _make_prehook(self, label: str, slot: int, mask: torch.Tensor | None) -> Any:
        """Build one batch-compatible slot-targeted freeze prehook.

        Multiplication only: under ``is_grads_batched`` the incoming grad is
        a functorch BatchedTensor, where detach/clone/item/view escape the
        vmap level; ``g * keep`` stays inside it.
        """

        keep = None
        if mask is not None:
            keep = (~mask.bool()).to(torch.float32)

        def _prehook(grad_outputs: tuple[Any, ...]) -> tuple[Any, ...]:
            """Zero/mask the frozen slot's incoming gradient, batch-safely."""

            self.fired_labels.add(label)
            adjusted = list(grad_outputs)
            grad = adjusted[slot] if slot < len(adjusted) else None
            if grad is not None:
                if keep is None:
                    adjusted[slot] = grad * 0
                else:
                    adjusted[slot] = grad * keep.to(dtype=grad.dtype, device=grad.device)
            return tuple(adjusted)

        return _prehook

    def __exit__(self, *exc_info: Any) -> None:
        """Remove every installed handle, even on failure."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def fired_for_plan(self) -> set[str]:
        """Freeze labels whose hook fired, expanded across alias groups."""

        fired = set(self.fired_labels)
        for label, siblings in self._plan.alias_disclosures.items():
            if label in fired:
                fired.update(sibling for sibling in siblings if sibling in self._plan.sites)
        return fired


def install_freeze_hooks(plan: FrozenPlan) -> FreezeHooks:
    """Return the context manager installing a plan's freeze prehooks."""

    return FreezeHooks(plan)
