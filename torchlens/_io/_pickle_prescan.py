"""Pre-scan for the restricted bundle unpickler: the plan the C machine enforces.

The loader runs untrusted ``metadata.pkl`` bytes through the C ``pickle.Unpickler``
for speed. The C machine exposes only ``find_class`` to a subclass, so the opcode-level
checks the pure-Python machine used to enforce on BUILD / REDUCE / NEWOBJ / NEWOBJ_EX /
INST / OBJ are established first, by a dry run of the same C machine over recording
probes, before any real object is built:

* every global reference is resolved through the caller's ``find_class`` policy
  (refusals fire here) and handed to the dry run as a probe type that only records;
* every REDUCE on a probe records a site, in execution order; the real run hands that
  callable back wrapped in a :class:`GuardedCallable` that runs the allocation belt on
  the real arguments at call time;
* every NEWOBJ / NEWOBJ_EX / INST / OBJ on a probe runs the allocation belt now, on its
  literal arguments, with the same refusal text the pure-Python handlers used;
* every BUILD is classified from the node it lands on: an instance of a constructed
  class is checked here, a REDUCE result is flagged so its guard checks the real result
  type before BUILD can reach it;
* stream corruption surfaces from the dry run as the C machine's own failures and is
  normalised by the unpickler's boundary, as the pure-Python boundary did;
* persistent ids and out-of-band buffers fail in the dry run as they fail in the real
  run; extension codes and over-declared frames are refused by the reader pass below.

A stream shape the dry run cannot classify (a constructed object called, a construction
with arguments built at run time, a global or a REDUCE result used both as a callable
and as a value or class) raises :class:`StreamNeedsInterpreter`; the unpickler then runs
the stream through the pure-Python machine, which enforces the same checks per opcode.
No TorchLens writer produces such a stream, so the fast path is the normal path and the
interpreter is the exact fallback, never a weaker one.

A tight byte reader (:func:`scan_opcodes`) walks the opcodes once before either machine
runs: it refuses a memo PUT index above the opcode count (the C machine's memo table is
sized from that index, so an absurd one is an allocation amplifier the pure-Python dict
memo never had) and counts opcodes against the ceiling on streams large enough to reach it.
"""

from __future__ import annotations

import io
import pickle
import pickletools
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GuardedCallable",
    "OpcodeCeilingExceeded",
    "PrescanPlan",
    "ReduceSite",
    "StreamNeedsInterpreter",
    "declared_protocol",
    "scan_opcodes",
    "prescan_stream",
]

#: Refusal-text prefix shared with the pure-Python machine's load boundary.
_CORRUPT_PREFIX = "corrupt or malformed pickle stream: "
_CORRUPT = _CORRUPT_PREFIX
#: Refusal texts shared with the pure-Python machine's EXT and FRAME handlers.
_EXT_REFUSAL = "extension codes are not used in bundle metadata"
_FRAME_REFUSAL = "FRAME declares more bytes than remain"
#: Importer of an admitted ``(module, name)`` pair: the base machine's ``find_class``.
Importer = Callable[[str, str], Any]
#: Largest literal argument tree checked statically for a construction; bigger argument
#: trees are not what a constructor call looks like and take the interpreter.
_MAX_STATIC_NODES = 256

ConstructionRefusal = Callable[[str, Any, Any, Any], "str | None"]
BuildRefusal = Callable[[Any], "str | None"]
BuildTypeRefusal = Callable[[type], "str | None"]


class StreamNeedsInterpreter(Exception):
    """The static walk cannot classify the stream; the pure-Python machine must run it."""


class OpcodeCeilingExceeded(pickle.UnpicklingError):
    """The stream holds more opcodes than the caller's allocation ceiling allows."""


class _StopDryRun(Exception):
    """The policy refused a global: the dry run ends and the plan carries the refusal."""


@dataclass(eq=False)
class ReduceSite:
    """What the C phase must enforce at one REDUCE opcode, in stream order."""

    index: int
    func_pair: tuple[str, str] | None
    derived_from: int
    build_target: bool = False
    result_callable: bool = False


@dataclass(eq=False)
class PrescanPlan:
    """Outcome of a successful pre-scan, consumed by the C phase."""

    opcode_count: int
    resolved: dict[tuple[str, str], Any]
    wrap_pairs: frozenset[tuple[str, str]]
    reduce_sites: list[ReduceSite]
    construction_refusal: ConstructionRefusal
    build_refusal: BuildRefusal
    cursor: int = 0
    global_lookups: int = field(default=0)
    #: A global the policy refused, with the refusal: the real run re-raises it when it
    #: reaches that GLOBAL, so the refusal fires in stream order (after any earlier
    #: REDUCE failure), exactly where the pure-Python machine raises it.
    refused: tuple[tuple[str, str], BaseException] | None = None

    def take_reduce_site(self) -> ReduceSite:
        """Return the next REDUCE site; refuse when the stream calls more than planned."""

        if self.cursor >= len(self.reduce_sites):
            raise pickle.UnpicklingError(
                _CORRUPT + "a guarded callable was invoked more often than the pre-scan planned."
            )
        site = self.reduce_sites[self.cursor]
        self.cursor += 1
        return site


class GuardedCallable:
    """Call-time belt around a callable the C machine will invoke for a REDUCE.

    The C ``pickle.Unpickler`` runs REDUCE in C, so the belt the pure-Python
    ``load_reduce`` applied before the call moves into the callable itself: the guard
    takes the next planned REDUCE site, verifies it is the planned callable, runs the
    construction belt on the real argument tuple, calls through, and then applies what
    the pre-scan asked of the result: the BUILD refusal when a later BUILD targets it,
    or a nested guard when a later REDUCE calls it.
    """

    __slots__ = ("_origin", "_pair", "_plan", "_real")

    def __init__(
        self,
        real: Any,
        plan: PrescanPlan,
        *,
        pair: tuple[str, str] | None = None,
        origin: int = -1,
    ) -> None:
        self._real = real
        self._plan = plan
        self._pair = pair
        self._origin = origin

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plan = self._plan
        site = plan.take_reduce_site()
        if self._pair is not None:
            planned = site.func_pair == self._pair
        else:
            planned = site.func_pair is None and site.derived_from == self._origin
        if not planned:
            raise _corrupt("a REDUCE invoked a callable the pre-scan did not plan for this site")
        reason = plan.construction_refusal("REDUCE", self._real, args, kwargs or None)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        result = self._real(*args, **kwargs)
        if site.build_target:
            build_reason = plan.build_refusal(result)
            if build_reason is not None:
                raise pickle.UnpicklingError(build_reason)
        if site.result_callable:
            return GuardedCallable(result, plan, origin=site.index)
        return result


def _corrupt(detail: str) -> pickle.UnpicklingError:
    """Return the corruption refusal for ``detail`` (the load boundary's wording)."""

    return pickle.UnpicklingError(_CORRUPT + detail)


_UNPACK_H = struct.Struct("<H").unpack_from
_UNPACK_I = struct.Struct("<i").unpack_from
_UNPACK_UI = struct.Struct("<I").unpack_from
_UNPACK_Q = struct.Struct("<Q").unpack_from
_UNPACK_D = struct.Struct(">d").unpack_from
#: Opcode byte -> (name, argument kind) from pickletools' own table.
_OPCODE_TABLE: dict[int, tuple[str, str | None]] = {
    op.code.encode("latin-1")[0]: (op.name, op.arg.name if op.arg is not None else None)
    for op in pickletools.opcodes
}
#: Length-prefixed argument kinds decoded inline (kind -> width of the length field).
_LENGTH_KINDS = {
    "bytes1": 1,
    "bytes4": 4,
    "bytes8": 8,
    "bytearray8": 8,
    "string1": 1,
    "string4": 4,
    "unicodestring1": 1,
    "unicodestring4": 4,
    "unicodestring8": 8,
}


def _slow_arg(data: bytes, pos: int, kind: str) -> tuple[Any, int]:
    """Decode a rare (protocol 0 / long) argument with pickletools' reader."""

    descriptor = getattr(pickletools, kind)
    handle = io.BytesIO(data)
    handle.seek(pos)
    return descriptor.reader(handle), handle.tell()


def _length_arg(data: bytes, pos: int, kind: str, end: int) -> tuple[Any, int]:
    """Decode a length-prefixed bytes / string argument starting at ``pos``."""

    width = _LENGTH_KINDS[kind]
    if pos + width > end:
        raise ValueError(f"truncated {kind} length")
    if width == 1:
        length = data[pos]
    elif width == 4:
        length = _UNPACK_UI(data, pos)[0]
    else:
        length = _UNPACK_Q(data, pos)[0]
    pos += width
    if pos + length > end:
        raise ValueError(f"truncated {kind} payload")
    raw = data[pos : pos + length]
    pos += length
    if kind.startswith("unicode"):
        return raw.decode("utf-8", "surrogatepass"), pos
    if kind.startswith("string"):
        return raw.decode("latin-1"), pos
    if kind == "bytearray8":
        return bytearray(raw), pos
    return raw, pos


def _iter_opcodes(data: bytes) -> Any:  # noqa: C901 -- one opcode-argument table
    """Yield ``(name, arg, pos)`` per opcode, as ``pickletools.genops`` would.

    A tight reader over the bytes: the binary-protocol arguments decode inline, GLOBAL
    and INST yield ``None`` (the walker reads their two lines itself) and the rare
    text-protocol arguments defer to pickletools. Truncation and unknown opcodes raise
    ``ValueError`` like ``genops``.
    """

    end = len(data)
    pos = 0
    table = _OPCODE_TABLE
    while True:
        if pos >= end:
            raise ValueError("pickle exhausted before seeing STOP")
        code = data[pos]
        entry = table.get(code)
        if entry is None:
            raise ValueError(f"at position {pos}, opcode {bytes([code])!r} unknown")
        name, kind = entry
        start = pos
        pos += 1
        arg: Any = None
        if kind is None:
            pass
        elif kind == "uint1":
            if pos >= end:
                raise ValueError("truncated uint1 argument")
            arg = data[pos]
            pos += 1
        elif kind in _LENGTH_KINDS:
            arg, pos = _length_arg(data, pos, kind, end)
        elif kind == "int4":
            arg = _UNPACK_I(data, pos)[0]
            pos += 4
        elif kind == "uint4":
            arg = _UNPACK_UI(data, pos)[0]
            pos += 4
        elif kind == "uint2":
            arg = _UNPACK_H(data, pos)[0]
            pos += 2
        elif kind == "float8":
            arg = _UNPACK_D(data, pos)[0]
            pos += 8
        elif kind == "uint8":
            arg = _UNPACK_Q(data, pos)[0]
            pos += 8
        elif kind == "stringnl_noescape_pair":
            first = data.find(b"\n", pos)
            second = data.find(b"\n", first + 1) if first >= 0 else -1
            if second < 0:
                raise ValueError("truncated GLOBAL/INST operand")
            pos = second + 1
        else:
            arg, pos = _slow_arg(data, pos, kind)
        yield name, arg, start
        if name == "STOP":
            return


class _Node:
    """A recorded construction in the dry run: what the C machine will build later."""

    __slots__ = ("args", "called", "cls_real", "items", "kwargs", "planner", "site", "state")

    def __init__(
        self,
        planner: _DryRunPlanner,
        *,
        site: int = -1,
        cls_real: Any = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.planner = planner
        self.site = site
        self.cls_real = cls_real
        self.args = args
        self.kwargs = kwargs
        self.called = False
        self.state: Any = None
        self.items: list[Any] = []

    def __setstate__(self, state: Any) -> None:
        self.state = state
        self.planner.on_build(self)

    def __call__(self, *args: Any, **kwargs: Any) -> _Node:
        return self.planner.on_derived_call(self, args, kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.items.append(key)
        self.items.append(value)

    def append(self, value: Any) -> None:
        """Record an APPEND / APPENDS member of the node."""

        self.items.append(value)

    def extend(self, values: Any) -> None:
        """Record the APPENDS members of the node."""

        self.items.extend(values)

    def add(self, value: Any) -> None:
        """Record an ADDITEMS member of the node."""

        self.items.append(value)


class _ProbeMeta(type):
    """Metaclass of the recording probes: ``probe(*args)`` is a REDUCE call."""

    _pair: tuple[str, str]
    _real: Any
    _planner: _DryRunPlanner

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls._planner.on_reduce(cls, args, kwargs)


class _ProbeBase(metaclass=_ProbeMeta):
    """Base of the recording probes: ``probe.__new__(probe, *args)`` is a construction."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls._planner.on_construct(cls, args, kwargs)


class _InterpreterImporter:
    """The pure-Python machine's ``find_class``, at the protocol the stream declared.

    The C machine's importer rebuilds a failed attribute lookup as its own
    ``AttributeError`` (no Python frame), where the pure-Python machine lets the module's
    own ``__getattr__`` failure through (an application frame). The load boundary
    classifies by frame locus, so the dry run resolves through the very code the
    pure-Python machine runs: same ``fix_imports`` mapping, same dotted walk at protocol
    4 and above, same exception identity.
    """

    __slots__ = ("fix_imports", "proto")

    def __init__(self, proto: int) -> None:
        self.proto = proto
        self.fix_imports = True

    def __call__(self, module: str, name: str) -> Any:
        """Import and resolve ``module.name`` as ``pickle._Unpickler.find_class`` does."""

        return pickle._Unpickler.find_class(self, module, name)  # type: ignore[arg-type]


def declared_protocol(data: bytes) -> int:
    """Return the protocol a leading PROTO opcode declares (0 when the stream has none)."""

    if data[:1] == pickle.PROTO and len(data) > 1:
        return data[1]
    return 0


class _DryRunUnpickler(pickle.Unpickler):
    """C machine whose globals are recording probes; it builds the skeleton, nothing real."""

    def __init__(self, data: bytes, planner: _DryRunPlanner) -> None:
        super().__init__(io.BytesIO(data))
        self._planner = planner
        self._importer = _InterpreterImporter(declared_protocol(data))

    def find_class(self, module: str, name: str) -> Any:
        """Hand the planner a probe, resolved through the interpreter's importer."""

        return self._planner.probe_for(module, name, self._importer)


class _DryRunPlanner:
    """Build the plan by running the stream through the C machine on recording probes.

    Every global resolves through the caller's policy (refusals fire here) and is handed
    to the dry run as a probe type; every REDUCE on a probe records a site in execution
    order, every NEWOBJ / NEWOBJ_EX / INST / OBJ on a probe runs the construction belt
    statically on its literal arguments, every BUILD records on its node, and a skeleton
    walk afterwards finds the probes stored as values and the nodes both called and
    stored. The dry run executes no admitted code: probes and nodes only record, so its
    allocation is the stream's own literals, the same class as any pickle parse.
    """

    def __init__(
        self,
        *,
        resolve: Callable[[str, str, Importer], Any],
        construction_refusal: ConstructionRefusal,
        build_refusal: BuildRefusal,
        build_type_refusal: BuildTypeRefusal,
    ) -> None:
        self.resolve = resolve
        self.construction_refusal = construction_refusal
        self.build_refusal = build_refusal
        self.build_type_refusal = build_type_refusal
        self.resolved: dict[tuple[str, str], Any] = {}
        self.probes: dict[tuple[str, str], type] = {}
        self.reduce_sites: list[ReduceSite] = []
        self.called_pairs: set[tuple[str, str]] = set()
        self.cls_pairs: set[tuple[str, str]] = set()
        self.refused: tuple[tuple[str, str], BaseException] | None = None

    def probe_for(self, module: str, name: str, import_global: Importer) -> type:
        """Return the recording probe for ``(module, name)``, resolving it once.

        ``import_global`` is the pure-Python machine's importer at the protocol the stream
        declared, so a dotted name attribute-walks exactly when that machine would
        (protocol 4 and above) and the policy judges the same object.
        """

        pair = (module, name)
        probe = self.probes.get(pair)
        if probe is None:
            try:
                real = self.resolve(module, name, import_global)
            except Exception as exc:  # noqa: BLE001 -- deferred, not handled: see below
                # A refusal (allowlist, belt, too-old module path) or an import failure
                # is the real run's to raise, at this GLOBAL, after every earlier opcode
                # ran for real; the load boundary classifies it there. The dry run ends.
                self.refused = (pair, exc)
                raise _StopDryRun from exc
            self.resolved[pair] = real
            probe = _ProbeMeta(
                "_Probe", (_ProbeBase,), {"_pair": pair, "_real": real, "_planner": self}
            )
            self.probes[pair] = probe
        return probe

    def on_reduce(self, probe: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Node:
        """Record a REDUCE of a probe's real callable; the C phase guards the real call."""

        pair = probe._pair
        site = ReduceSite(len(self.reduce_sites), pair, -1)
        self.reduce_sites.append(site)
        self.called_pairs.add(pair)
        return _Node(self, site=site.index, args=args, kwargs=kwargs or None)

    def on_derived_call(self, node: _Node, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Node:
        """Record a REDUCE whose callable is itself a REDUCE result."""

        if node.site < 0:
            raise StreamNeedsInterpreter("REDUCE on a constructed object")
        self.reduce_sites[node.site].result_callable = True
        node.called = True
        site = ReduceSite(len(self.reduce_sites), None, node.site)
        self.reduce_sites.append(site)
        return _Node(self, site=site.index, args=args, kwargs=kwargs or None)

    def on_construct(self, probe: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Node:
        """Run the construction belt statically on a NEWOBJ / NEWOBJ_EX / INST / OBJ."""

        pair = probe._pair
        self.cls_pairs.add(pair)
        budget = [_MAX_STATIC_NODES]
        try:
            literal_args = _literal(args, budget)
            literal_kwargs = _literal(kwargs, budget) if kwargs else None
        except _NotStatic as why:
            raise StreamNeedsInterpreter(
                f"construction with non-literal arguments: {why}"
            ) from None
        opcode = "NEWOBJ_EX" if kwargs else "NEWOBJ"
        reason = self.construction_refusal(opcode, probe._real, literal_args, literal_kwargs)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return _Node(self, cls_real=probe._real, args=args, kwargs=kwargs or None)

    def on_build(self, node: _Node) -> None:
        """Record a BUILD on a node; refuse it when the node's class refuses BUILD."""

        if node.site >= 0:
            self.reduce_sites[node.site].build_target = True
        elif isinstance(node.cls_real, type):
            reason = self.build_type_refusal(node.cls_real)
            if reason is not None:
                raise pickle.UnpicklingError(reason)

    def run(self, data: bytes) -> PrescanPlan:
        """Dry-run ``data`` and return the plan; hand the interpreter what it cannot finish.

        A policy refusal ends the dry run and rides in the plan (``refused``). Any other
        failure of the dry run (truncation, a stack fault, a probe the C machine cannot
        stand in for) is not a verdict: the pure-Python machine runs the stream and decides
        exactly as it always did.
        """

        try:
            root = _DryRunUnpickler(data, self).load()
        except StreamNeedsInterpreter:
            raise
        except _StopDryRun:
            return self._plan()
        except Exception as exc:  # noqa: BLE001 -- classified: not a verdict, a hand-off
            raise StreamNeedsInterpreter(f"dry run stopped: {type(exc).__name__}") from exc
        value_pairs = _walk_values(root)
        mixed = self.called_pairs & (value_pairs | self.cls_pairs)
        if mixed:
            raise StreamNeedsInterpreter(
                f"globals used both as callables and otherwise: {sorted(mixed)}"
            )
        return self._plan()

    def _plan(self) -> PrescanPlan:
        """Return the plan for what the dry run recorded (and any deferred refusal)."""

        return PrescanPlan(
            opcode_count=0,
            resolved=self.resolved,
            wrap_pairs=frozenset(self.called_pairs),
            reduce_sites=self.reduce_sites,
            construction_refusal=self.construction_refusal,
            build_refusal=self.build_refusal,
            refused=self.refused,
        )


class _NotStatic(Exception):
    """An argument tree holds a value only the real run knows."""


def _literal(value: Any, budget: list[int]) -> Any:
    """Return ``value`` with probes replaced by their real objects; refuse nodes."""

    budget[0] -= 1
    if budget[0] < 0:
        raise _NotStatic("argument tree too large for a static check")
    if isinstance(value, _Node):
        raise _NotStatic("argument built at run time")
    if isinstance(value, _ProbeMeta):
        return value._real
    if isinstance(value, (list, tuple)):
        members = [_literal(item, budget) for item in value]
        return tuple(members) if isinstance(value, tuple) else members
    if isinstance(value, dict):
        return {_literal(k, budget): _literal(v, budget) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        elements = {_literal(item, budget) for item in value}
        return elements if isinstance(value, set) else frozenset(elements)
    return value


def _walk_values(root: Any) -> set[tuple[str, str]]:
    """Return the probe pairs stored as values; refuse a node that is called and stored."""

    value_pairs: set[tuple[str, str]] = set()
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        value = stack.pop()
        kind = type(value)
        if kind is list or kind is tuple or kind is set or kind is frozenset:
            if id(value) in seen:
                continue
            seen.add(id(value))
            stack.extend(value)
        elif kind is dict:
            if id(value) in seen:
                continue
            seen.add(id(value))
            stack.extend(value.keys())
            stack.extend(value.values())
        elif kind is _Node:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.called:
                raise StreamNeedsInterpreter("a callable result is also stored as a value")
            stack.extend(value.args)
            if value.kwargs:
                stack.extend(value.kwargs.values())
            if value.state is not None:
                stack.append(value.state)
            stack.extend(value.items)
        elif kind is _ProbeMeta:
            value_pairs.add(value._pair)
    return value_pairs


def _next_opcode(ops: Any, enforcing_ceiling: bool) -> tuple[Any, Any, int]:
    """Return the next ``(opcode, arg, pos)``; a stream the reader cannot walk steps aside.

    Truncation, an unknown opcode or a bad operand is not this pass's verdict: the
    pure-Python machine runs such a stream and refuses it in opcode order (a blocked global
    before the damage is still reported as the blocked global). Under an opcode ceiling the
    walk must reach STOP to count, so there the unreadable stream is refused outright.
    """

    try:
        return next(ops)
    except StopIteration:
        raise _corrupt("pickle exhausted before seeing STOP") from None
    except (ValueError, EOFError, OverflowError, struct.error) as exc:
        if enforcing_ceiling:
            raise _corrupt(f"{type(exc).__name__}: {exc}") from exc
        raise StreamNeedsInterpreter(f"reader pass stopped: {exc}") from exc


def scan_opcodes(data: bytes, max_opcodes: int | None, *, source_sized: bool = True) -> int:
    """Walk the opcodes once: refuse an absurd memo index and enforce the opcode ceiling.

    The C machine sizes its memo table from the largest PUT index the stream names (a
    five-byte ``LONG_BINPUT`` can demand a table of billions of slots, which it fills
    before any check can run); the dict-backed pure-Python machine never did. A pickler
    assigns memo indices sequentially, so an index above the running opcode count is
    never honest and is refused here, before the dry run or the real run touch the
    stream. The same pass refuses the extension-code opcodes (EXT1 / EXT2 / EXT4, a
    process-wide registry lookup no writer emits) and a FRAME that declares more bytes
    than remain (the C machine sizes its read buffer from that length), and counts
    opcodes against ``max_opcodes`` when one is given. ``source_sized`` is whether the
    source the bytes came from could report its length: when it could not (a pipe, a
    socket) every FRAME is refused, the decision the pure-Python machine makes at its
    FRAME opcode, so both machines refuse the same streams.
    """

    count = 0
    end = len(data)
    mid_stream_proto = False
    ops = _iter_opcodes(data)
    while True:
        name, arg, pos = _next_opcode(ops, max_opcodes is not None)
        count += 1
        if name == "PROTO" and pos != 0:
            # The dry run resolves globals at the protocol the stream declared up front;
            # a later PROTO would change the walk rules under it, so the interpreter
            # (which tracks the protocol per opcode) runs that stream, once counted.
            mid_stream_proto = True
        if max_opcodes is not None and count > max_opcodes:
            raise OpcodeCeilingExceeded(_CORRUPT + "opcode count exceeds the allocation ceiling")
        if name in _PUT_OPCODES and (arg < 0 or arg > count):
            raise _corrupt("memo PUT index above the opcode count")
        if name in _EXT_OPCODES:
            # Extension codes resolve through the process-wide ``copyreg`` registry and
            # cache process-wide; refused before either machine can look one up.
            raise _corrupt(_EXT_REFUSAL)
        if name == "FRAME" and (not source_sized or arg > end - pos - _FRAME_HEADER_LEN):
            # The C machine grows its input buffer to the declared frame length before
            # reading it; a declaration beyond the bytes that remain is refused first,
            # and a source that could not report its length refuses every frame.
            raise _corrupt(_FRAME_REFUSAL)
        if name == "STOP":
            if mid_stream_proto:
                raise StreamNeedsInterpreter("the protocol changes after the first opcode")
            return count


_PUT_OPCODES = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_EXT_OPCODES = frozenset({"EXT1", "EXT2", "EXT4"})
#: One opcode byte plus the eight-byte little-endian length of a FRAME.
_FRAME_HEADER_LEN = 9


def prescan_stream(
    data: bytes,
    *,
    resolve: Callable[[str, str, Importer], Any],
    construction_refusal: ConstructionRefusal,
    build_refusal: BuildRefusal,
    build_type_refusal: BuildTypeRefusal,
) -> PrescanPlan:
    """Return the plan the C phase enforces for ``data``.

    Parameters
    ----------
    data:
        The complete pickle stream.
    resolve:
        The ``find_class`` policy, ``(module, name, import_global) -> object``; called
        once per distinct ``(module, name)`` with the dry-run machine's own importer.
    construction_refusal:
        ``(opcode, callable_or_class, args, kwargs) -> reason | None``: the allocation
        belt for a construction with literal arguments.
    build_refusal:
        ``(target_object) -> reason | None`` for a BUILD applied to a REDUCE result.
    build_type_refusal:
        ``(target_class) -> reason | None`` for a BUILD applied to an instance a class
        construction produced.

    Raises
    ------
    pickle.UnpicklingError
        A refusal (allowlist, belt, BUILD) or stream corruption.
    StreamNeedsInterpreter
        The stream must run through the pure-Python machine instead.
    """

    planner = _DryRunPlanner(
        resolve=resolve,
        construction_refusal=construction_refusal,
        build_refusal=build_refusal,
        build_type_refusal=build_type_refusal,
    )
    return planner.run(data)
