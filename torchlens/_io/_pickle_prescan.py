"""Static opcode pre-scan for the restricted bundle unpickler.

The loader runs untrusted ``metadata.pkl`` bytes through the C ``pickle.Unpickler``
for speed. The C machine exposes only ``find_class`` to a subclass, so the opcode-level
checks the pure-Python machine used to enforce on BUILD / REDUCE / NEWOBJ / NEWOBJ_EX /
INST / OBJ are established here first, on a symbolic walk of the opcode stream
(a tight opcode reader plus a stack model), before any object is built:

* every global reference is resolved through the caller's ``find_class`` policy
  (refusals fire here, before the C machine starts);
* every construction opcode whose class and arguments are statically known runs the
  allocation belt now, with the same refusal text the pure-Python handlers used;
* every construction whose callable arrives through ``find_class`` is marked so the C
  phase hands the callable back wrapped in a guard that runs the belt on the real
  arguments at call time (:class:`torchlens._io._safe_unpickle._GuardedCallable`);
* every BUILD target is classified: a constant or container can never be a tensor, a
  resolved global or a statically typed construction is checked here, and a REDUCE
  result is flagged so its guard checks the real result type before BUILD can reach it;
* stream corruption (stack underflow, missing memo keys, truncation, unknown opcodes)
  is refused as :class:`pickle.UnpicklingError`, the family the pure-Python boundary
  normalised to;
* persistent ids, extension codes and out-of-band buffers are refused, as both machines
  refuse them without the corresponding hooks.

A stream shape the static walk cannot classify (a callable of unknown provenance, a
constructor with non-literal arguments, a global used both as a value and as a
callable) raises :class:`StreamNeedsInterpreter`; the unpickler then runs the stream
through the pure-Python machine, which enforces the same checks per opcode. No
TorchLens writer produces such a stream, so the fast path is the normal path and the
interpreter is the exact fallback, never a weaker one.

Two checks here are deliberately stricter than the pure-Python machine was. A GLOBAL or
INST opcode whose module line contains a space is refused (no Python module is spelled
that way, and the two-line form must parse unambiguously), and an explicit memo PUT
index above the running opcode count is refused (a pickler assigns memo indices
sequentially; the C machine sizes its memo table from the index, so an absurd index is
an allocation amplifier the dict-backed pure-Python memo never had).
"""

from __future__ import annotations

import copyreg
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
    "count_opcodes",
    "prescan_stream",
]

#: Refusal-text prefix shared with the pure-Python machine's load boundary.
_CORRUPT = "corrupt or malformed pickle stream: "
#: Largest literal argument tree concretised for a static construction check; bigger
#: argument trees are not what a constructor call looks like and take the interpreter.
_MAX_STATIC_NODES = 256

ConstructionRefusal = Callable[[str, Any, Any, Any], "str | None"]
BuildRefusal = Callable[[Any], "str | None"]
BuildTypeRefusal = Callable[[type], "str | None"]


class StreamNeedsInterpreter(Exception):
    """The static walk cannot classify the stream; the pure-Python machine must run it."""


class OpcodeCeilingExceeded(pickle.UnpicklingError):
    """The stream holds more opcodes than the caller's allocation ceiling allows."""


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


class _Sym:
    """One symbolic stack value.

    Literals sit on the stack as themselves. ``kind`` is ``global`` (``pair`` names it,
    ``value`` is the resolved object), ``container`` (``value`` is the container kind
    name, ``items`` its member symbols; a dict stores alternating keys and values) or
    ``obj`` (``producer`` is the callable or class symbol, ``via`` the opcode name,
    ``site`` the REDUCE site index). ``roles`` records how the value is consumed.
    """

    __slots__ = ("items", "kind", "pair", "producer", "roles", "site", "value", "via")

    def __init__(
        self,
        kind: str,
        value: Any = None,
        *,
        pair: tuple[str, str] | None = None,
        items: list[Any] | None = None,
        producer: _Sym | None = None,
        via: str = "",
        site: int = -1,
    ) -> None:
        self.kind = kind
        self.value = value
        self.pair = pair
        self.items = items
        self.producer = producer
        self.via = via
        self.site = site
        self.roles: set[str] | None = None

    def tag(self, role: str) -> None:
        """Record one consumption role (``value``, ``func`` or ``cls``)."""

        if self.roles is None:
            self.roles = {role}
        else:
            self.roles.add(role)


class _NotStatic(Exception):
    """A symbol tree holds a value that is only known at run time."""


def _corrupt(detail: str) -> pickle.UnpicklingError:
    return pickle.UnpicklingError(_CORRUPT + detail)


def _tag(item: Any, role: str) -> None:
    """Record a consumption role on a symbol; literals carry none."""

    if isinstance(item, _Sym):
        item.tag(role)


def _is_container(item: Any, kind: str) -> bool:
    return isinstance(item, _Sym) and item.kind == "container" and item.value == kind


def _concrete(sym: Any, budget: list[int]) -> Any:
    """Return the literal Python value a symbol tree denotes, or raise ``_NotStatic``."""

    budget[0] -= 1
    if budget[0] < 0:
        raise _NotStatic("argument tree too large for a static check")
    if not isinstance(sym, _Sym):
        return sym
    if sym.kind == "global":
        return sym.value
    if sym.kind != "container":
        raise _NotStatic("argument built at run time")
    items = sym.items or []
    if sym.value == "dict":
        return {
            _concrete(items[index], budget): _concrete(items[index + 1], budget)
            for index in range(0, len(items), 2)
        }
    members = [_concrete(item, budget) for item in items]
    if sym.value == "tuple":
        return tuple(members)
    if sym.value == "list":
        return members
    try:
        return set(members) if sym.value == "set" else frozenset(members)
    except TypeError as exc:
        raise _NotStatic("unhashable literal member") from exc


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


class _Walker:
    """Symbolic stack machine over one pickle stream."""

    def __init__(
        self,
        data: bytes,
        *,
        resolve: Callable[[str, str], Any],
        construction_refusal: ConstructionRefusal,
        build_refusal: BuildRefusal,
        build_type_refusal: BuildTypeRefusal,
        max_opcodes: int | None,
    ) -> None:
        self.data = data
        self.resolve = resolve
        self.construction_refusal = construction_refusal
        self.build_refusal = build_refusal
        self.build_type_refusal = build_type_refusal
        self.max_opcodes = max_opcodes
        self.stack: list[Any] = []
        self.metastack: list[list[Any]] = []
        self.memo: dict[int, Any] = {}
        self.resolved: dict[tuple[str, str], Any] = {}
        self.global_syms: list[_Sym] = []
        self.reduce_sites: list[ReduceSite] = []
        self.reduce_funcs: list[_Sym] = []
        self.opcode_count = 0

    # -- stack primitives -------------------------------------------------------------

    def pop(self, opcode: str) -> Any:
        if not self.stack:
            raise _corrupt(f"stack underflow on {opcode}")
        return self.stack.pop()

    def top(self, opcode: str) -> Any:
        if not self.stack:
            raise _corrupt(f"stack underflow on {opcode}")
        return self.stack[-1]

    def pop_mark(self, opcode: str) -> list[Any]:
        if not self.metastack:
            raise _corrupt(f"{opcode} without a preceding MARK")
        items = self.stack
        self.stack = self.metastack.pop()
        return items

    def push_container(self, kind: str, items: list[Any]) -> None:
        for item in items:
            _tag(item, "value")
        self.stack.append(_Sym("container", kind, items=items))

    # -- globals ----------------------------------------------------------------------

    def read_global_lines(self, pos: int, opcode: str) -> tuple[str, str]:
        """Parse the two newline-terminated lines after a GLOBAL / INST opcode byte."""

        first_end = self.data.find(b"\n", pos + 1)
        second_end = self.data.find(b"\n", first_end + 1) if first_end >= 0 else -1
        if first_end < 0 or second_end < 0:
            raise _corrupt(f"truncated {opcode} operand")
        try:
            module = self.data[pos + 1 : first_end].decode("utf-8")
            name = self.data[first_end + 1 : second_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _corrupt(f"{opcode} operand is not valid UTF-8") from exc
        if " " in module:
            raise _corrupt(f"{opcode} module line contains a space")
        return module, name

    def resolve_global(self, module: str, name: str) -> _Sym:
        pair = (module, name)
        if pair in self.resolved:
            obj = self.resolved[pair]
        else:
            obj = self.resolve(module, name)
            self.resolved[pair] = obj
        sym = _Sym("global", obj, pair=pair)
        self.global_syms.append(sym)
        return sym

    # -- construction opcodes ---------------------------------------------------------

    def static_construction(
        self, opcode: str, cls_sym: Any, args_sym: Any, kwargs_sym: Any | None
    ) -> None:
        """Run the allocation belt now for a construction with literal arguments."""

        if not isinstance(cls_sym, _Sym) or cls_sym.kind != "global":
            raise StreamNeedsInterpreter(f"{opcode} on a class of unknown provenance")
        try:
            budget = [_MAX_STATIC_NODES]
            args = _concrete(args_sym, budget)
            kwargs = None if kwargs_sym is None else _concrete(kwargs_sym, budget)
        except _NotStatic as why:
            raise StreamNeedsInterpreter(f"{opcode} with non-literal arguments: {why}") from None
        reason = self.construction_refusal(opcode, cls_sym.value, args, kwargs)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        cls_sym.tag("cls")

    def op_reduce(self) -> None:
        args = self.pop("REDUCE")
        func = self.pop("REDUCE")
        if not _is_container(args, "tuple"):
            raise StreamNeedsInterpreter("REDUCE whose argument is not a literal tuple")
        args.tag("value")
        if not isinstance(func, _Sym):
            raise StreamNeedsInterpreter("REDUCE on a literal")
        if func.kind == "global":
            site = ReduceSite(len(self.reduce_sites), func.pair, -1)
        elif func.kind == "obj" and func.via == "REDUCE":
            site = ReduceSite(len(self.reduce_sites), None, func.site)
        else:
            raise StreamNeedsInterpreter("REDUCE on a callable of unknown provenance")
        func.tag("func")
        self.reduce_sites.append(site)
        self.reduce_funcs.append(func)
        self.stack.append(_Sym("obj", producer=func, via="REDUCE", site=site.index))

    def op_newobj(self, opcode: str) -> None:
        kwargs = None
        if opcode == "NEWOBJ_EX":
            kwargs = self.pop(opcode)
            if not _is_container(kwargs, "dict"):
                raise StreamNeedsInterpreter("NEWOBJ_EX whose keywords are not a literal dict")
            kwargs.tag("value")
        args = self.pop(opcode)
        cls = self.pop(opcode)
        if not _is_container(args, "tuple"):
            raise StreamNeedsInterpreter(f"{opcode} whose argument is not a literal tuple")
        args.tag("value")
        self.static_construction(opcode, cls, args, kwargs)
        self.stack.append(_Sym("obj", producer=cls, via=opcode))

    def op_inst(self, pos: int) -> None:
        module, name = self.read_global_lines(pos, "INST")
        items = self.pop_mark("INST")
        cls = self.resolve_global(module, name)
        self.static_construction("INST", cls, _Sym("container", "tuple", items=items), None)
        for item in items:
            _tag(item, "value")
        self.stack.append(_Sym("obj", producer=cls, via="INST"))

    def op_obj(self) -> None:
        items = self.pop_mark("OBJ")
        if not items:
            raise _corrupt("OBJ without a class on the stack")
        cls, args = items[0], items[1:]
        self.static_construction("OBJ", cls, _Sym("container", "tuple", items=args), None)
        for item in args:
            _tag(item, "value")
        self.stack.append(_Sym("obj", producer=cls, via="OBJ"))

    def op_build(self) -> None:
        state = self.pop("BUILD")
        target = self.top("BUILD")
        _tag(state, "value")
        if not isinstance(target, _Sym):
            return
        if target.kind == "global":
            reason = self.build_refusal(target.value)
            if reason is not None:
                raise pickle.UnpicklingError(reason)
        elif target.kind == "obj":
            if target.via == "REDUCE":
                self.reduce_sites[target.site].build_target = True
            else:
                producer = target.producer
                if producer is None or producer.kind != "global":
                    raise StreamNeedsInterpreter("BUILD on an object of unknown provenance")
                if isinstance(producer.value, type):
                    reason = self.build_type_refusal(producer.value)
                    if reason is not None:
                        raise pickle.UnpicklingError(reason)

    def op_stack_global(self) -> None:
        name = self.pop("STACK_GLOBAL")
        module = self.pop("STACK_GLOBAL")
        if isinstance(name, _Sym) or isinstance(module, _Sym):
            raise StreamNeedsInterpreter("STACK_GLOBAL with non-literal operands")
        if type(name) is not str or type(module) is not str:
            raise pickle.UnpicklingError("STACK_GLOBAL requires str")
        self.stack.append(self.resolve_global(module, name))

    # -- the walk ---------------------------------------------------------------------

    def step(self, name: str, arg: Any, pos: int) -> bool:
        """Apply one opcode; return ``True`` at STOP."""

        if name in _CONST_OPCODES:
            self.stack.append(arg)
            return False
        if name == "MEMOIZE":
            self.memo[len(self.memo)] = self.top(name)
            return False
        if name in _GET_OPCODES:
            sym = self.memo.get(arg)
            if sym is None:
                raise _corrupt(f"missing memo key {arg!r}")
            self.stack.append(sym)
            return False
        handler = _DISPATCH.get(name)
        if handler is None:  # pragma: no cover - every pickletools opcode is listed
            raise _corrupt(f"unsupported opcode {name}")
        return bool(handler(self, arg, pos))

    def op_put(self, arg: Any, pos: int) -> None:
        if arg < 0:
            raise _corrupt("negative PUT argument")
        if arg > self.opcode_count:
            raise _corrupt("memo PUT index above the opcode count")
        self.memo[arg] = self.top("PUT")

    def op_mark(self, arg: Any, pos: int) -> None:
        self.metastack.append(self.stack)
        self.stack = []

    def op_empty_list(self, arg: Any, pos: int) -> None:
        self.stack.append(_Sym("container", "list", items=[]))

    def op_empty_tuple(self, arg: Any, pos: int) -> None:
        self.stack.append(_Sym("container", "tuple", items=[]))

    def op_empty_dict(self, arg: Any, pos: int) -> None:
        self.stack.append(_Sym("container", "dict", items=[]))

    def op_empty_set(self, arg: Any, pos: int) -> None:
        self.stack.append(_Sym("container", "set", items=[]))

    def op_tuple_n(self, count: int) -> None:
        if len(self.stack) < count:
            raise _corrupt("stack underflow on TUPLE")
        items = self.stack[-count:]
        del self.stack[-count:]
        self.push_container("tuple", items)

    def op_tuple(self, arg: Any, pos: int) -> None:
        self.push_container("tuple", self.pop_mark("TUPLE"))

    def op_list(self, arg: Any, pos: int) -> None:
        self.push_container("list", self.pop_mark("LIST"))

    def op_dict(self, arg: Any, pos: int) -> None:
        items = self.pop_mark("DICT")
        if len(items) % 2:
            raise _corrupt("DICT with an odd number of items")
        self.push_container("dict", items)

    def op_frozenset(self, arg: Any, pos: int) -> None:
        self.push_container("frozenset", self.pop_mark("FROZENSET"))

    def op_append(self, arg: Any, pos: int) -> None:
        value = self.pop("APPEND")
        self.extend_container(self.top("APPEND"), [value], "list")

    def op_appends(self, arg: Any, pos: int) -> None:
        items = self.pop_mark("APPENDS")
        self.extend_container(self.top("APPENDS"), items, "list")

    def op_setitem(self, arg: Any, pos: int) -> None:
        value = self.pop("SETITEM")
        key = self.pop("SETITEM")
        self.extend_container(self.top("SETITEM"), [key, value], "dict")

    def op_setitems(self, arg: Any, pos: int) -> None:
        items = self.pop_mark("SETITEMS")
        if len(items) % 2:
            raise _corrupt("SETITEMS with an odd number of items")
        self.extend_container(self.top("SETITEMS"), items, "dict")

    def op_additems(self, arg: Any, pos: int) -> None:
        items = self.pop_mark("ADDITEMS")
        self.extend_container(self.top("ADDITEMS"), items, "set")

    def op_pop(self, arg: Any, pos: int) -> None:
        if self.stack:
            self.stack.pop()
        else:
            self.pop_mark("POP")

    def op_pop_mark(self, arg: Any, pos: int) -> None:
        self.pop_mark("POP_MARK")

    def op_dup(self, arg: Any, pos: int) -> None:
        self.stack.append(self.top("DUP"))

    def op_global(self, arg: Any, pos: int) -> None:
        module, global_name = self.read_global_lines(pos, "GLOBAL")
        self.stack.append(self.resolve_global(module, global_name))

    def op_persid(self, arg: Any, pos: int) -> None:
        raise pickle.UnpicklingError(
            "A load persistent id instruction was encountered, "
            "but no persistent_load function was specified."
        )

    def op_ext(self, arg: Any, pos: int) -> None:
        if arg in getattr(copyreg, "_inverted_registry", {}):
            raise StreamNeedsInterpreter("registered pickle extension code")
        raise _corrupt(f"unregistered extension code {arg!r}")

    def op_buffer(self, arg: Any, pos: int) -> None:
        raise pickle.UnpicklingError(
            "pickle stream refers to out-of-band data but no *buffers* argument was given"
        )

    def op_proto(self, arg: Any, pos: int) -> None:
        if arg > pickle.HIGHEST_PROTOCOL:
            raise _corrupt(f"unsupported pickle protocol: {arg}")

    def op_frame(self, arg: Any, pos: int) -> None:
        return None

    def op_stop(self, arg: Any, pos: int) -> bool:
        _tag(self.pop("STOP"), "value")
        return True

    def extend_container(self, target: Any, items: list[Any], kind: str) -> None:
        """Record items added to a container; other targets fail or act at run time."""

        for item in items:
            _tag(item, "value")
        if _is_container(target, kind) and target.items is not None:
            target.items.extend(items)

    def run(self) -> PrescanPlan:
        ops = _iter_opcodes(self.data)
        while True:
            name, arg, pos = _next_opcode(ops)
            self.opcode_count += 1
            if self.max_opcodes is not None and self.opcode_count > self.max_opcodes:
                raise OpcodeCeilingExceeded(
                    _CORRUPT + "opcode count exceeds the allocation ceiling"
                )
            if self.step(name, arg, pos):
                break
        return self.finish()

    def finish(self) -> PrescanPlan:
        """Decide which callables the C phase wraps; refuse mixed roles to the interpreter."""

        wrap_pairs: set[tuple[str, str]] = set()
        for site, func in zip(self.reduce_sites, self.reduce_funcs, strict=True):
            if func.roles != {"func"}:
                raise StreamNeedsInterpreter("a callable is also used as a value or a class")
            if func.kind == "global" and func.pair is not None:
                wrap_pairs.add(func.pair)
            else:
                self.reduce_sites[site.derived_from].result_callable = True
        for sym in self.global_syms:
            if sym.pair in wrap_pairs and sym.roles != {"func"}:
                raise StreamNeedsInterpreter("a global is used both as a callable and otherwise")
        return PrescanPlan(
            opcode_count=self.opcode_count,
            resolved=self.resolved,
            wrap_pairs=frozenset(wrap_pairs),
            reduce_sites=self.reduce_sites,
            construction_refusal=self.construction_refusal,
            build_refusal=self.build_refusal,
        )


_DISPATCH: dict[str, Callable[..., Any]] = {
    "PUT": _Walker.op_put,
    "BINPUT": _Walker.op_put,
    "LONG_BINPUT": _Walker.op_put,
    "MARK": _Walker.op_mark,
    "EMPTY_LIST": _Walker.op_empty_list,
    "EMPTY_TUPLE": _Walker.op_empty_tuple,
    "EMPTY_DICT": _Walker.op_empty_dict,
    "EMPTY_SET": _Walker.op_empty_set,
    "TUPLE1": lambda walker, arg, pos: walker.op_tuple_n(1),
    "TUPLE2": lambda walker, arg, pos: walker.op_tuple_n(2),
    "TUPLE3": lambda walker, arg, pos: walker.op_tuple_n(3),
    "TUPLE": _Walker.op_tuple,
    "LIST": _Walker.op_list,
    "DICT": _Walker.op_dict,
    "FROZENSET": _Walker.op_frozenset,
    "APPEND": _Walker.op_append,
    "APPENDS": _Walker.op_appends,
    "SETITEM": _Walker.op_setitem,
    "SETITEMS": _Walker.op_setitems,
    "ADDITEMS": _Walker.op_additems,
    "POP": _Walker.op_pop,
    "POP_MARK": _Walker.op_pop_mark,
    "DUP": _Walker.op_dup,
    "GLOBAL": _Walker.op_global,
    "STACK_GLOBAL": lambda walker, arg, pos: walker.op_stack_global(),
    "REDUCE": lambda walker, arg, pos: walker.op_reduce(),
    "NEWOBJ": lambda walker, arg, pos: walker.op_newobj("NEWOBJ"),
    "NEWOBJ_EX": lambda walker, arg, pos: walker.op_newobj("NEWOBJ_EX"),
    "INST": lambda walker, arg, pos: walker.op_inst(pos),
    "OBJ": lambda walker, arg, pos: walker.op_obj(),
    "BUILD": lambda walker, arg, pos: walker.op_build(),
    "PERSID": _Walker.op_persid,
    "BINPERSID": _Walker.op_persid,
    "EXT1": _Walker.op_ext,
    "EXT2": _Walker.op_ext,
    "EXT4": _Walker.op_ext,
    "NEXT_BUFFER": _Walker.op_buffer,
    "READONLY_BUFFER": _Walker.op_buffer,
    "PROTO": _Walker.op_proto,
    "FRAME": _Walker.op_frame,
    "STOP": _Walker.op_stop,
}
_CONST_OPCODES = frozenset(
    {
        "INT",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BINBYTES8",
        "BYTEARRAY8",
        "NONE",
        "NEWTRUE",
        "NEWFALSE",
        "UNICODE",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
        "FLOAT",
        "BINFLOAT",
    }
)
_GET_OPCODES = frozenset({"GET", "BINGET", "LONG_BINGET"})


def _next_opcode(ops: Any) -> tuple[Any, Any, int]:
    """Return the next ``(opcode, arg, pos)``; map reader failures to corruption."""

    try:
        return next(ops)
    except StopIteration:
        raise _corrupt("pickle exhausted before seeing STOP") from None
    except (ValueError, EOFError, OverflowError, struct.error) as exc:
        # The reader reports truncation, unknown opcodes and bad operands with these
        # builtins; the C machine would refuse the same bytes.
        raise _corrupt(f"{type(exc).__name__}: {exc}") from exc


def count_opcodes(data: bytes, max_opcodes: int) -> int:
    """Return the opcode count of ``data``, refusing past ``max_opcodes``."""

    count = 0
    ops = _iter_opcodes(data)
    while True:
        name, _arg, _pos = _next_opcode(ops)
        count += 1
        if count > max_opcodes:
            raise OpcodeCeilingExceeded(_CORRUPT + "opcode count exceeds the allocation ceiling")
        if name == "STOP":
            return count


def prescan_stream(
    data: bytes,
    *,
    resolve: Callable[[str, str], Any],
    construction_refusal: ConstructionRefusal,
    build_refusal: BuildRefusal,
    build_type_refusal: BuildTypeRefusal,
    max_opcodes: int | None = None,
) -> PrescanPlan:
    """Walk ``data`` symbolically and return the plan the C phase enforces.

    Parameters
    ----------
    data:
        The complete pickle stream.
    resolve:
        The ``find_class`` policy; called once per distinct ``(module, name)``.
    construction_refusal:
        ``(opcode, callable_or_class, args, kwargs) -> reason | None``: the allocation
        belt for a construction with literal arguments.
    build_refusal:
        ``(target_object) -> reason | None`` for a BUILD applied to a resolved global.
    build_type_refusal:
        ``(target_class) -> reason | None`` for a BUILD applied to an instance a class
        construction produced.
    max_opcodes:
        Opcode-count ceiling; exceeding it raises :class:`OpcodeCeilingExceeded`.

    Raises
    ------
    pickle.UnpicklingError
        A refusal (allowlist, belt, BUILD) or stream corruption.
    StreamNeedsInterpreter
        The stream must run through the pure-Python machine instead.
    """

    walker = _Walker(
        data,
        resolve=resolve,
        construction_refusal=construction_refusal,
        build_refusal=build_refusal,
        build_type_refusal=build_type_refusal,
        max_opcodes=max_opcodes,
    )
    return walker.run()
