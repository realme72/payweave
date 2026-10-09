"""Mapper sandbox: what a package's ``mappers.py`` may contain (gate G2's core).

Mappers must be pure functions: no I/O, no network, no clock, no dynamic code.
Two layers, so one mistake doesn't open everything:

1. **Static check** of the source (AST): an import allowlist, no dunder or
   private attribute access, no dangerous builtins, no classes/async/globals,
   and only definitions at module level.
2. **Restricted execution**: the module runs with a minimal ``__builtins__``
   and an ``__import__`` that only serves the allowlist.

This isn't an OS-level sandbox; the static check is what makes escape
routes (``().__class__.__subclasses__()``...) unreachable. Every package is
also reviewed and signed by a human before it can run (G8).
"""

from __future__ import annotations

import ast
import builtins
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

# Per module, the names a mapper may import (``None``: any public name). Only
# ``from x import y`` is allowed, so whole modules (``typing.get_type_hints``,
# ``types.FunctionType``: both can build or evaluate code) are never reachable.
ALLOWED_IMPORTS: Mapping[str, frozenset[str] | None] = MappingProxyType(
    {
        "__future__": frozenset({"annotations"}),
        "typing": frozenset({"Any", "Final", "cast"}),
        "collections.abc": frozenset({"Mapping", "Sequence", "Iterable"}),
        "datetime": frozenset({"datetime", "timedelta", "UTC"}),
        "payweave.packages.sdk": None,
    }
)
SAFE_BUILTINS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "range", "reversed", "round",
        "set", "sorted", "str", "sum", "tuple", "zip",
        "Exception", "KeyError", "IndexError", "TypeError", "ValueError",
        "True", "False", "None",
    }
)  # fmt: skip
_PY_BUILTINS = frozenset(dir(builtins))
_AVAILABLE = ", ".join(sorted(SAFE_BUILTINS - {"True", "False", "None"}))
# Names that could reach I/O, the clock, introspection or dynamic code.
FORBIDDEN_NAMES = frozenset(
    {
        "eval", "exec", "compile", "open", "input", "print", "breakpoint", "help",
        "__import__", "getattr", "setattr", "delattr", "hasattr", "globals", "locals",
        "vars", "dir", "type", "object", "super", "memoryview", "classmethod",
        "staticmethod", "property", "id", "exit", "quit",
    }
)  # fmt: skip
# Clock reads, and str.format/format_map: their format strings can reach
# attributes ("{0.__class__}") without any attribute node in the AST.
FORBIDDEN_ATTRIBUTES = frozenset({"now", "today", "utcnow", "format", "format_map"})
_FORBIDDEN_NODES: dict[type[ast.AST], str] = {
    ast.ClassDef: "classes",
    ast.AsyncFunctionDef: "async functions",
    ast.Await: "await",
    ast.AsyncFor: "async for",
    ast.AsyncWith: "async with",
    ast.With: "with",
    ast.While: "while loops",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.Yield: "generators",
    ast.YieldFrom: "generators",
    ast.Delete: "del",
}
_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.TypeAlias)
MAX_LINES = 1500


class MapperRejectedError(Exception):
    """The mapper source failed the static check. ``problems`` lists every finding."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("mappers.py rejected:\n- " + "\n- ".join(problems))
        self.problems = problems


def check_source(source: str, filename: str = "mappers.py") -> list[str]:  # noqa: PLR0912 - one branch per rule
    """Return every problem found (empty means the source passed)."""
    if source.count("\n") > MAX_LINES:
        return [f"longer than {MAX_LINES} lines"]
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"line {exc.lineno}: syntax error: {exc.msg}"]
    problems: list[str] = []

    def bad(node: ast.AST, message: str) -> None:
        problems.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    for i, stmt in enumerate(tree.body):
        is_docstring = (
            i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
        )
        if not (is_docstring or isinstance(stmt, _TOP_LEVEL)):
            bad(stmt, f"only imports, functions and constants allowed at module level, "
                      f"not {type(stmt).__name__}")  # fmt: skip

    # Names the module binds itself (so a local called `iter` isn't taken for the builtin).
    bound = (
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        | {
            a.arg
            for f in ast.walk(tree)
            if isinstance(f, (ast.FunctionDef, ast.Lambda))
            for a in (*f.args.posonlyargs, *f.args.args, *f.args.kwonlyargs)
        }
        | {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        | {
            a.asname or a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        }
        | {
            g.target.id
            for g in ast.walk(tree)
            if isinstance(g, ast.comprehension) and isinstance(g.target, ast.Name)
        }
    )

    for node in ast.walk(tree):
        for kind, label in _FORBIDDEN_NODES.items():
            if isinstance(node, kind):
                bad(node, f"{label} are not allowed")
        if isinstance(node, ast.Import):
            bad(node, "use 'from module import name'; whole-module imports are not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None or node.module not in ALLOWED_IMPORTS:
                bad(node, f"import from {node.module!r} is not allowed")
                continue
            allowed = ALLOWED_IMPORTS[node.module]
            for alias in node.names:
                if (
                    alias.name == "*"
                    or alias.name.startswith("_")
                    or (allowed is not None and alias.name not in allowed)
                ):
                    bad(node, f"import of {alias.name!r} from {node.module!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                bad(node, f"access to private attribute {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRIBUTES:
                bad(node, f"{node.attr!r} is not allowed (mappers can't read the clock)")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                bad(node, f"use of {node.id!r} is not allowed")
            elif (
                isinstance(node.ctx, ast.Load)
                and node.id in _PY_BUILTINS
                and node.id not in SAFE_BUILTINS
                and node.id not in bound
            ):
                # Would pass here and crash at the first call (NameError) in the sandbox.
                bad(
                    node, f"builtin {node.id!r} isn't available to mappers; available: {_AVAILABLE}"
                )
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.arg.startswith("__"):
                    bad(node, f"parameter name {arg.arg!r} is not allowed")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            bad(node, "decorators are not allowed")
    return problems


def _guarded_import(
    name: str,
    globals_: Mapping[str, object] | None = None,
    locals_: Mapping[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    allowed = ALLOWED_IMPORTS.get(name, frozenset())
    if level or name not in ALLOWED_IMPORTS or not fromlist:
        raise ImportError(f"import of {name!r} is not allowed in mappers")
    if allowed is not None and not set(fromlist) <= allowed:
        raise ImportError(f"import of {sorted(fromlist)} from {name!r} is not allowed")
    return importlib.import_module(name)


def _restricted_builtins() -> dict[str, object]:
    table: dict[str, object] = {
        n: getattr(builtins, n) for n in SAFE_BUILTINS if n not in {"True", "False", "None"}
    }
    table["__import__"] = _guarded_import
    table["__build_class__"] = None  # no classes even if the static check were bypassed
    return table


@dataclass(frozen=True, slots=True)
class LoadedMappers:
    functions: Mapping[str, Callable[..., Any]]


def load_mappers(source: str, *, module_name: str, filename: str = "mappers.py") -> LoadedMappers:
    """Statically check, then execute with restricted builtins. Returns public functions."""
    problems = check_source(source, filename)
    if problems:
        raise MapperRejectedError(problems)
    code = compile(source, filename, "exec")
    namespace: dict[str, Any] = {"__builtins__": _restricted_builtins(), "__name__": module_name}
    exec(code, namespace)  # noqa: S102 - statically checked above, restricted builtins
    functions = {
        name: value
        for name, value in namespace.items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == module_name
    }
    return LoadedMappers(MappingProxyType(functions))
