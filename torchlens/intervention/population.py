"""The public population noun for stochastic/population edits (F02, edits memo D2).

One provenance-stamped population constructor (:func:`reference`; the NAME is a
placeholder pending naming-session ratification -- every spelling here is
DOCUMENTED-UNSTABLE). The noun is the ONE home for the origin contract, the
content digest, per-member datums, privacy rules, and the tensor-returning
reducers -- the design borrows the shipped ``tl.subspace`` contract verbatim:
"a direction without a recorded origin is an unreproducible result" applies to
populations word for word (edits memo D2), so ``origin=`` is REQUIRED and
non-empty (memo DIS-3: explicit required in v1; auto-derivation is a recorded
ergonomic follow-up).

Population laws implemented here:

- D2: one public noun, ``origin=`` required.
- D3: reductions are not draws -- :meth:`Reference.mean` / :meth:`Reference.std`
  return TENSORS, never invent a seed, and accumulate reduced-precision
  members in float32.
- D10: populations are PREPARED at construction; hooks only index, cast,
  move, and combine (no loader iteration or model run at fire time).
- D11: external tensor populations carry a mandatory sha256 content digest
  paid once at preparation; trace-backed populations record an ADDRESS
  identity by default (``digest_kind="address"``) with ``digest_population=True``
  opting into the value hash.
- D13: persisted records store digests and counts; ephemeral refusal messages
  may print agreement keys and class sizes (they teach without persisting).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .._errors import InvalidArgumentError

__all__ = [
    "OneDatum",
    "PerRowDatums",
    "Reference",
    "reference",
]


@dataclass(frozen=True)
class OneDatum:
    """One agreement datum describing the WHOLE subject event.

    The internal matching carrier (edits memo section 9: the public wrapper
    names are a [UI-SPRINT] handoff; the internal distinction ships now).
    """

    value: Any


@dataclass(frozen=True)
class PerRowDatums:
    """Per-row agreement datums for the subject batch (one datum per row)."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        """Normalize the values container to a tuple."""

        object.__setattr__(self, "values", tuple(self.values))


def _tensor_content_digest(members: Sequence[torch.Tensor]) -> str:
    """Compute the mandatory sha256 content digest over tensor members (D11).

    Parameters
    ----------
    members:
        Prepared member tensors, in member order.

    Returns
    -------
    str
        Hex sha256 digest over member bytes, shapes, and dtypes.
    """

    hasher = hashlib.sha256()
    for member in members:
        hasher.update(str(tuple(member.shape)).encode())
        hasher.update(str(member.dtype).encode())
        hasher.update(
            member.detach().to("cpu").contiguous().numpy().tobytes()
            if member.dtype not in (torch.bfloat16,)
            else member.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
        )
    return hasher.hexdigest()


def _trace_address_digest(members: Sequence[Any]) -> str:
    """Compute the ADDRESS identity digest for trace-backed members (D11).

    The address -- trace fingerprints in member order -- identifies the
    population without hashing payload bytes; ``digest_population=True`` opts
    into the value hash at first per-site resolution instead.
    """

    hasher = hashlib.sha256()
    for member in members:
        identity = (
            str(getattr(member, "trace_label", "") or ""),
            str(getattr(member, "model_class_qualname", "") or ""),
            str(getattr(member, "random_seed", "") or ""),
            str(getattr(member, "num_operations", "") or ""),
        )
        hasher.update("|".join(identity).encode())
        hasher.update(str(id(member)).encode())
    return hasher.hexdigest()


@dataclass(frozen=True)
class Reference:
    """A prepared, provenance-stamped donor population (the population noun).

    Parameters
    ----------
    members:
        Prepared member payloads: tensors (``kind="tensor"``) or captured
        traces (``kind="trace"``; a member's donor value resolves at the
        firing site, pass-qualified, at fire time -- an index/read, never a
        model run).
    datums:
        Optional per-member agreement datums (``None`` when absent). Length
        always equals ``len(members)`` when present.
    origin:
        REQUIRED non-empty human provenance string (whence and why).
    kind:
        Closed member-kind vocabulary: ``"tensor"`` or ``"trace"``.
    content_digest:
        sha256 identity: member bytes for tensor members, the address digest
        for trace members (see ``digest_kind``).
    digest_kind:
        ``"content"`` (value hash) or ``"address"`` (identity facts only).
    """

    members: tuple[Any, ...]
    datums: tuple[Any, ...] | None
    origin: str
    kind: Literal["tensor", "trace"]
    content_digest: str
    digest_kind: Literal["content", "address"]
    _member_shape: tuple[int, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate the prepared population invariants."""

        if not self.members:
            raise InvalidArgumentError(
                "a population must carry at least one member",
                code="population_empty",
                remedy="pass a non-empty tensor stack, member sequence, or trace sequence",
                argument="source",
            )
        if self.datums is not None and len(self.datums) != len(self.members):
            raise InvalidArgumentError(
                f"population data= carries {len(self.datums)} datums for "
                f"{len(self.members)} members; every member needs exactly one datum",
                code="population_datum_count_mismatch",
                remedy="pass one datum per member, aligned with member order",
                argument="data",
                member_count=len(self.members),
                datum_count=len(self.datums),
            )

    def __len__(self) -> int:
        """Return the member count."""

        return len(self.members)

    @property
    def population_identity(self) -> str:
        """Stable identity string for seed derivation and audit records.

        Digest-first (privacy rule D13: identity travels, payloads never);
        the origin string participates so two same-bytes populations with
        different declared provenance stay distinguishable in the audit.
        """

        basis = f"{self.kind}|{self.digest_kind}|{self.content_digest}|{self.origin}"
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # agreement machinery (shared by the sampling plans and the reducers)
    # ------------------------------------------------------------------
    def eligible_indices(
        self,
        agree_on: Callable[[Any], Any] | None,
        matching_value: Any,
    ) -> tuple[int, ...]:
        """Return member indices whose agreement key matches one subject key.

        Parameters
        ----------
        agree_on:
            Datum -> hashable key callable (``None`` means every member is
            eligible).
        matching_value:
            The subject's datum for this event or row (already unwrapped).

        Returns
        -------
        tuple[int, ...]
            Eligible member indices in member order.

        Raises
        ------
        InvalidArgumentError
            ``population_agreement_empty`` when no member matches -- the
            EPHEMERAL message names the requested key and the available keys
            with their class sizes (privacy rule D13: refusals teach without
            persisting; persisted records carry digests and counts only).
        """

        if agree_on is None:
            return tuple(range(len(self.members)))
        if self.datums is None:
            raise InvalidArgumentError(
                "agreement-conditioned sampling needs per-member datums, but this "
                "population was built without data=",
                code="population_agreement_empty",
                remedy="rebuild the population with data=<one datum per member>",
                argument="agree_on",
            )
        subject_key = agree_on(matching_value)
        classes: dict[Any, list[int]] = {}
        for index, datum in enumerate(self.datums):
            classes.setdefault(agree_on(datum), []).append(index)
        eligible = classes.get(subject_key)
        if not eligible:
            available = ", ".join(f"{key!r} (n={len(rows)})" for key, rows in classes.items())
            raise InvalidArgumentError(
                f"no population member agrees with the subject key {subject_key!r}; "
                f"available agreement classes: {available}",
                code="population_agreement_empty",
                remedy="pick a matching= datum whose agreement key has members, or "
                "widen agree_on so a class matches",
                argument="matching",
                requested_key=repr(subject_key),
                class_sizes={repr(key): len(rows) for key, rows in classes.items()},
            )
        return tuple(eligible)

    # ------------------------------------------------------------------
    # reducers (D3: reductions are not draws)
    # ------------------------------------------------------------------
    def _tensor_members(self, verb: str) -> tuple[torch.Tensor, ...]:
        """Return tensor members or refuse for trace-backed populations."""

        if self.kind != "tensor":
            raise InvalidArgumentError(
                f"{verb} needs a tensor-membered population; this population's "
                "members are captured traces (their donor values resolve "
                "per-site at fire time and have no single tensor stack)",
                code="population_reduce_unsupported",
                remedy="build the population from site tensors, e.g. "
                "reference([t.out for t in traces at one site], origin=...)",
                argument="ref",
            )
        return self.members

    def _reduce(
        self,
        verb: str,
        reducer: Callable[[torch.Tensor], torch.Tensor],
        *,
        group_by: Callable[[Any], Any] | None = None,
        matching: Any = None,
        min_members: int = 1,
    ) -> torch.Tensor:
        """Shared deterministic reduction over the (eligible) member stack."""

        members = self._tensor_members(verb)
        indices: tuple[int, ...]
        if group_by is not None or matching is not None:
            if group_by is None or matching is None:
                raise InvalidArgumentError(
                    f"{verb} takes group_by= and matching= together: the key "
                    "function and the subject datum are two halves of one "
                    "agreement condition",
                    code="population_matching_invalid",
                    remedy="pass both group_by= and matching=, or neither",
                    argument="matching" if matching is None else "group_by",
                )
            matching_value = matching.value if isinstance(matching, OneDatum) else matching
            if isinstance(matching_value, PerRowDatums):
                raise InvalidArgumentError(
                    f"{verb} reduces to ONE tensor, so it takes ONE subject datum; "
                    "per-row datums describe a batch, not an event",
                    code="population_matching_invalid",
                    remedy="pass matching=OneDatum(value) (or the bare value)",
                    argument="matching",
                )
            indices = self.eligible_indices(group_by, matching_value)
        else:
            indices = tuple(range(len(members)))
        if len(indices) < min_members:
            raise InvalidArgumentError(
                f"{verb} needs at least {min_members} eligible member(s); {len(indices)} matched",
                code="population_reduce_underdetermined",
                remedy="widen the agreement condition or add members",
                argument="ref",
                eligible_count=len(indices),
            )
        stack = torch.stack([members[index].detach() for index in indices], dim=0)
        original_dtype = stack.dtype
        if original_dtype in (torch.float16, torch.bfloat16):
            # A11 reducer contract: reduced-precision members accumulate in
            # float32, then cast back to the member dtype.
            result = reducer(stack.to(torch.float32)).to(original_dtype)
        else:
            result = reducer(stack)
        return result

    def mean(
        self,
        over: Any = None,
        *,
        group_by: Callable[[Any], Any] | None = None,
        matching: Any = None,
    ) -> torch.Tensor:
        """Return the deterministic member mean (a TENSOR, never a draw; D3).

        Parameters
        ----------
        over:
            ``None`` for the elementwise across-member mean, or explicit
            member-shape axes (int or tuple of ints) to reduce further with
            ``keepdim=True``. Named axis tokens resolve through the axis
            ladder and refuse ``axis_semantics_unknown`` until axis-role
            evidence exists (never guessed from rank; edits memo D25).
        group_by:
            Optional agreement key callable over member datums.
        matching:
            Subject datum selecting the eligible agreement class.

        Returns
        -------
        torch.Tensor
            Member-shaped mean (further reduced axes kept as size 1).
        """

        from .stochastic import _resolve_over_axes

        def _mean(stack: torch.Tensor) -> torch.Tensor:
            """Reduce the eligible member stack to its mean."""

            reduced = stack.mean(dim=0)
            if over is None:
                return reduced
            axes = _resolve_over_axes(over, reduced.ndim, verb="Reference.mean")
            if axes == "all":
                return reduced.mean()
            return reduced.mean(dim=axes, keepdim=True)

        return self._reduce("Reference.mean", _mean, group_by=group_by, matching=matching)

    def std(
        self,
        over: Any = None,
        *,
        group_by: Callable[[Any], Any] | None = None,
        matching: Any = None,
    ) -> torch.Tensor:
        """Return the deterministic member std (a TENSOR, never a draw; D3).

        Parameters mirror :meth:`mean`; at least two eligible members are
        required (a single-member standard deviation is undefined under the
        default correction and would silently read as certainty).
        """

        from .stochastic import _resolve_over_axes

        def _std(stack: torch.Tensor) -> torch.Tensor:
            """Reduce the eligible member stack to its std."""

            reduced = stack.std(dim=0)
            if over is None:
                return reduced
            axes = _resolve_over_axes(over, reduced.ndim, verb="Reference.std")
            if axes == "all":
                return reduced.mean()
            return reduced.mean(dim=axes, keepdim=True)

        return self._reduce(
            "Reference.std", _std, group_by=group_by, matching=matching, min_members=2
        )


def _prepare_tensor_members(source: Any) -> tuple[torch.Tensor, ...] | None:
    """Normalize tensor-shaped sources to a member tuple (or ``None``).

    A bare tensor's members are its axis-0 slices BY THE NOUN'S DECLARED
    CONTRACT (a construction-time documented rule, like ``torch.stack`` --
    never a runtime rank guess); a sequence of tensors is taken member by
    member.
    """

    if isinstance(source, torch.Tensor):
        return tuple(source[index] for index in range(source.shape[0])) if source.ndim > 0 else None
    if (
        isinstance(source, Sequence)
        and source
        and all(isinstance(item, torch.Tensor) for item in source)
    ):
        return tuple(source)
    return None


def _looks_like_trace(value: Any) -> bool:
    """Whether a member is a captured-trace-like donor (duck-typed)."""

    return hasattr(value, "layer_dict_all_keys") and hasattr(value, "ops")


def reference(
    source: Any,
    *,
    origin: str,
    data: Sequence[Any] | None = None,
    digest_population: bool = False,
) -> Reference:
    """Build the provenance-stamped donor population (the ONE public noun).

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification
    (``tl.reference`` vs ``tl.population`` is a recorded [UI-SPRINT] fork).

    Parameters
    ----------
    source:
        The population payload: a tensor (members = axis-0 slices, the noun's
        documented stacking contract), a sequence of same-shape tensors, or a
        sequence of captured Traces (donor values resolve per-site at fire
        time, pass-qualified).
    origin:
        REQUIRED non-empty provenance string -- whence and why (the
        ``tl.subspace`` precedent, edits memo D2/DIS-3).
    data:
        Optional per-member agreement datums, aligned with member order.
    digest_population:
        For trace-backed populations, opt into value hashing (measured cost:
        hashing one gpt2 batch-32 residual is ~310 ms). The default records
        the ADDRESS identity, which already identifies the draw completely
        together with the realized permutation (edits memo D11).

    Returns
    -------
    Reference
        The prepared population.

    Raises
    ------
    InvalidArgumentError
        ``population_origin_required`` on a missing/empty origin;
        ``population_heterogeneous`` when tensor members disagree in shape or
        dtype; ``population_empty`` / ``population_datum_count_mismatch`` per
        the dataclass invariants.
    """

    if not isinstance(origin, str) or not origin.strip():
        raise InvalidArgumentError(
            "a population without a recorded origin is an unreproducible result; "
            "origin= is required and non-empty",
            code="population_origin_required",
            remedy='describe the provenance, e.g. origin="clean run, templated prompts"',
            argument="origin",
        )
    datums = tuple(data) if data is not None else None

    tensor_members = _prepare_tensor_members(source)
    if tensor_members is not None:
        if not tensor_members:
            raise InvalidArgumentError(
                "a population must carry at least one member",
                code="population_empty",
                remedy="pass a tensor with a non-empty leading axis or a non-empty sequence",
                argument="source",
            )
        first = tensor_members[0]
        for index, member in enumerate(tensor_members):
            if tuple(member.shape) != tuple(first.shape) or member.dtype != first.dtype:
                raise InvalidArgumentError(
                    f"population member {index} has shape {tuple(member.shape)!r} / "
                    f"dtype {member.dtype}, but member 0 has {tuple(first.shape)!r} / "
                    f"{first.dtype}; a donor population is one homogeneous stack",
                    code="population_heterogeneous",
                    remedy="pad or split the members so every member shares one "
                    "shape and dtype (build one population per geometry)",
                    argument="source",
                )
        prepared = tuple(member.detach() for member in tensor_members)
        return Reference(
            members=prepared,
            datums=datums,
            origin=origin,
            kind="tensor",
            content_digest=_tensor_content_digest(prepared),
            digest_kind="content",
            _member_shape=tuple(first.shape),
        )

    if isinstance(source, Sequence) and source and all(_looks_like_trace(item) for item in source):
        members = tuple(source)
        digest_kind: Literal["content", "address"] = "address"
        digest = _trace_address_digest(members)
        if digest_population:
            # Value-hash opt-in: fold each member's retained site labels in;
            # per-site payload hashing stays lazy (paid at first resolution).
            hasher = hashlib.sha256(digest.encode())
            for member in members:
                hasher.update(",".join(sorted(getattr(member, "layer_labels", ()) or ())).encode())
            digest = hasher.hexdigest()
            digest_kind = "content"
        return Reference(
            members=members,
            datums=datums,
            origin=origin,
            kind="trace",
            content_digest=digest,
            digest_kind=digest_kind,
        )

    if _looks_like_trace(source):
        return reference([source], origin=origin, data=data, digest_population=digest_population)

    raise InvalidArgumentError(
        f"unsupported population source {type(source).__name__}; a population is "
        "a tensor stack, a sequence of same-shape tensors, or a sequence of "
        "captured Traces",
        code="population_source_invalid",
        remedy="pass a stacked tensor, a list of member tensors, or a list of Traces",
        argument="source",
    )
