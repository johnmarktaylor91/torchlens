"""Site pinning, the ``blocks`` preset, and the structural site resolver.

Site identity is the structural ``site_key``, never the layer label and not the
module address alone (LIT-panel memo D4): padding shifts every block LABEL on
every architecture measured (0/6, 0/12 survivors), while ``site_key`` survived
padding, ragged batches, and eager-vs-sdpa byte-identically. Each exposed site
pins ``(field_name, module_address, site_key, pass_index)`` at construction;
per-request traces re-resolve through a per-trace index over concrete ``Op``
records matching ``(site_key, pass index)`` and refuse on zero or multiple
matches (memo D5 -- deliberately NOT ``align_to()``, which rejects exactly the
padding-drifted traces this bridge must accept). All discovery and reads are
PER-OP (memo D6): ``Layer`` aggregates raise or silently return empty on
multi-pass boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import _refusals

_MAX_CANDIDATES_NAMED = 12
_PASS_SUFFIX = re.compile(r"^(?P<address>.+):(?P<pass_index>\d+)$")
_TRAILING_INT = re.compile(r"^(?P<stem>.*?)(?P<index>\d+)$")


@dataclass(frozen=True)
class SiteSpec:
    """One pinned LIT embedding site (LIT-panel memo D4 pin quad).

    Parameters
    ----------
    field_name:
        Visible LIT output-spec field name (placeholder spelling, [UI-SPRINT]).
    module_address:
        Module-tree address whose output op this site reads.
    site_key:
        TorchLens structural site key pinned on the unpadded probe trace.
    pass_index:
        1-based pass index of the pinned op.
    """

    field_name: str
    module_address: str
    site_key: str
    pass_index: int


def module_output_ops(trace: Any) -> list[Any]:
    """Return every op that is a module output, per-op (memo D6).

    Parameters
    ----------
    trace:
        A finished TorchLens ``Trace``.

    Returns
    -------
    list[Any]
        Ops (one per pass) in execution order carrying ``output_of_modules``.
    """

    return [op for op in trace.layer_list if getattr(op, "output_of_modules", ())]


def _ops_for_address(trace: Any, address: str) -> list[Any]:
    """Return the module-output ops of one address in execution order.

    Parameters
    ----------
    trace:
        A finished TorchLens ``Trace``.
    address:
        Module-tree address.

    Returns
    -------
    list[Any]
        Matching ops, one entry per pass.
    """

    return [op for op in module_output_ops(trace) if address in op.output_of_modules]


def _module_addresses(trace: Any) -> tuple[str, ...]:
    """Return a bounded sample of module addresses for teaching messages.

    Parameters
    ----------
    trace:
        A finished TorchLens ``Trace``.

    Returns
    -------
    tuple[str, ...]
        Up to ``_MAX_CANDIDATES_NAMED`` addresses that produced output ops.
    """

    seen: list[str] = []
    for op in module_output_ops(trace):
        for address in op.output_of_modules:
            if address not in seen:
                seen.append(address)
    return tuple(seen[:_MAX_CANDIDATES_NAMED])


def discover_block_stack(trace: Any) -> tuple[str, ...]:
    """Discover the largest homogeneous repeated-module stack (memo D7).

    The preset is built from module-call metadata, never op-name matching: the
    op-name-substring heuristic it replaces selected interleaved sublayer norms
    on the encoders and, on GPT-2, twelve fields of which ZERO were block
    outputs. A stack is a group of sibling modules sharing one class whose
    address components end in consecutive integers (``transformer.h.0`` ...,
    ResNet's ``layer1`` ... ``layer4``).

    Parameters
    ----------
    trace:
        The unpadded probe trace.

    Returns
    -------
    tuple[str, ...]
        The stack's module addresses in index order.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Code ``lit_blocks_preset_unavailable`` when no stack exists or two
        stacks tie for largest.
    """

    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for module in trace.modules:
        address = str(module.address)
        parent, _, leaf = address.rpartition(".")
        match = _TRAILING_INT.match(leaf)
        if match is None:
            continue
        key = (f"{parent}.{match.group('stem')}", str(module.class_name))
        groups.setdefault(key, []).append((int(match.group("index")), address))

    stacks = {k: sorted(v) for k, v in groups.items() if _is_consecutive(sorted(v))}
    if not stacks:
        _refuse_no_stack(trace, "no repeated same-class sibling modules found")
    best_size = max(len(v) for v in stacks.values())
    winners = [k for k, v in stacks.items() if len(v) == best_size]
    if len(winners) > 1:
        # Size ties are real on small models (a 2-block GPT-2 ties with each
        # block's ln_1/ln_2 pair); the OUTERMOST stack is the block stack, so
        # the shallowest address depth breaks the tie structurally. A residual
        # same-depth tie stays a refusal -- the preset never guesses.
        best_depth = min(key[0].count(".") for key in winners)
        winners = [key for key in winners if key[0].count(".") == best_depth]
    if len(winners) > 1:
        tied = tuple(sorted(f"{k[0]}* ({k[1]})" for k in winners))
        _refusals.refuse_blocks_preset_unavailable(
            f"{len(winners)} stacks tie at {best_size} members and equal depth", tied
        )
    return tuple(address for _, address in stacks[winners[0]])


def _is_consecutive(members: list[tuple[int, str]]) -> bool:
    """Return whether a sorted candidate group forms one consecutive stack.

    Parameters
    ----------
    members:
        ``(index, address)`` pairs sorted by index.

    Returns
    -------
    bool
        True for two or more members with consecutive indexes.
    """

    if len(members) < 2:
        return False
    indexes = [index for index, _ in members]
    return indexes == list(range(indexes[0], indexes[0] + len(indexes)))


def _refuse_no_stack(trace: Any, detail: str) -> None:
    """Refuse preset discovery, naming candidate addresses.

    Parameters
    ----------
    trace:
        The probe trace whose addresses seed the teaching message.
    detail:
        Why discovery failed.
    """

    _refusals.refuse_blocks_preset_unavailable(detail, _module_addresses(trace))


def pin_blocks(trace: Any) -> list[SiteSpec]:
    """Pin the ``blocks`` preset sites on the probe trace.

    The preset takes the LAST pass of each block's module-output op, as tested
    preset semantics (memo D8: on GPT-2 the last-pass residual add equals HF
    ``output_hidden_states`` bit-exactly for blocks 0-10; block 11 is
    pre-``ln_f`` -- documented). Field names are ``tl_block_<i>`` placeholders
    pending the naming sprint ([UI-SPRINT] item 2).

    Parameters
    ----------
    trace:
        The unpadded probe trace.

    Returns
    -------
    list[SiteSpec]
        One pinned site per block, in stack order.
    """

    specs: list[SiteSpec] = []
    for index, address in enumerate(discover_block_stack(trace)):
        ops = _ops_for_address(trace, address)
        if not ops:
            _refuse_no_stack(trace, f"stack member {address!r} produced no output op")
        last = ops[-1]
        specs.append(
            SiteSpec(
                field_name=f"tl_block_{index}",
                module_address=address,
                site_key=str(last.site_key),
                pass_index=int(last.pass_index),
            )
        )
    return specs


def pin_explicit(trace: Any, sites: tuple[str, ...]) -> list[SiteSpec]:
    """Pin explicit user sites on the probe trace (memo D7/D8).

    Each entry is a module address, optionally pass-qualified as
    ``"address:N"``. An unqualified address whose module fires more than once
    per forward refuses (explicit sites carry their pass in the key or refuse);
    field names are ``tl_<address with dots as underscores>`` placeholders.

    Parameters
    ----------
    trace:
        The unpadded probe trace.
    sites:
        Explicit site spellings.

    Returns
    -------
    list[SiteSpec]
        One pinned site per entry, in request order.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Codes ``lit_site_unresolvable`` / ``lit_site_ambiguous``.
    """

    specs: list[SiteSpec] = []
    for raw in sites:
        address, wanted_pass = _split_pass(str(raw))
        ops = _ops_for_address(trace, address)
        if not ops:
            _refusals.refuse_site_unresolvable(str(raw), _module_addresses(trace))
        chosen = _select_pass(str(raw), address, ops, wanted_pass)
        suffix = f"_pass{wanted_pass}" if wanted_pass is not None else ""
        specs.append(
            SiteSpec(
                field_name="tl_" + address.replace(".", "_") + suffix,
                module_address=address,
                site_key=str(chosen.site_key),
                pass_index=int(chosen.pass_index),
            )
        )
    return specs


def _split_pass(raw: str) -> tuple[str, int | None]:
    """Split an optional ``:N`` pass qualifier off a site spelling.

    Parameters
    ----------
    raw:
        The user's site spelling.

    Returns
    -------
    tuple[str, int | None]
        ``(module_address, pass_index_or_None)``.
    """

    match = _PASS_SUFFIX.match(raw)
    if match is None:
        return raw, None
    return match.group("address"), int(match.group("pass_index"))


def _select_pass(raw: str, address: str, ops: list[Any], wanted: int | None) -> Any:
    """Select the requested pass among a site's module-output ops.

    Parameters
    ----------
    raw:
        The original site spelling (for teaching messages).
    address:
        The module address.
    ops:
        Module-output ops for the address, one per pass.
    wanted:
        Explicit 1-based pass index, or None for unqualified.

    Returns
    -------
    Any
        The selected op record.
    """

    passes = tuple(int(op.pass_index) for op in ops)
    if wanted is None:
        if len(ops) > 1:
            _refusals.refuse_site_ambiguous(raw, passes)
        return ops[0]
    for op in ops:
        if int(op.pass_index) == wanted:
            return op
    _refusals.refuse_site_unresolvable(raw, tuple(f"{address}:{p}" for p in passes))


def resolve_pinned(trace: Any, specs: list[SiteSpec]) -> dict[str, Any]:
    """Re-resolve pinned sites on a fresh trace via the structural index.

    Builds one index over concrete per-op records keyed by
    ``(site_key, pass_index)`` and demands exactly one match per pinned site
    (memo D5). Zero matches refuse as drift, naming both the address and the
    key; multiple matches refuse as ambiguity (bare ``site_key`` is unique only
    per call instance -- torchlens invariant I-S2).

    Parameters
    ----------
    trace:
        The predict-time trace.
    specs:
        The construction-time pin quads.

    Returns
    -------
    dict[str, Any]
        ``field_name -> op record`` for every pinned site.

    Raises
    ------
    torchlens._errors.RecordBindingError
        Code ``lit_site_key_drift`` on zero or multiple structural matches.
    """

    index: dict[tuple[str, int], list[Any]] = {}
    for op in trace.layer_list:
        key = (str(op.site_key), int(op.pass_index))
        index.setdefault(key, []).append(op)

    resolved: dict[str, Any] = {}
    for spec in specs:
        matches = index.get((spec.site_key, spec.pass_index), [])
        if len(matches) != 1:
            _refusals.refuse_site_key_drift(spec.field_name, spec.module_address, spec.site_key)
        resolved[spec.field_name] = matches[0]
    return resolved
