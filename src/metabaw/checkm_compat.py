"""Run native CheckM with safe NumPy-scalar statistics parsing.

The child entry point also runs under CheckM's own console-launcher Python.
Keep it standard-library-only: MetaBAW need not be installed in that runtime.
No installed CheckM files, global ast functions or numerical settings change.
"""
from __future__ import annotations

import ast
import copy
import importlib
import math
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys


FLOAT_SCALARS = frozenset({"float16", "float32", "float64"})
INTEGER_SCALARS = frozenset({"int8", "int16", "int32", "int64", "uint8", "uint16",
                             "uint32", "uint64", "intp", "uintp"})
BOOLEAN_SCALARS = frozenset({"bool", "bool_"})


class _ScalarLiterals(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if (isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"}
                and node.attr in {"True_", "False_"}):
            return ast.copy_location(ast.Constant(value=node.attr == "True_"), node)
        return node

    def visit_Call(self, node):
        function = node.func
        if not (isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in {"np", "numpy"}
                and len(node.args) == 1 and not node.keywords):
            raise ValueError("Unsupported function call in CheckM statistics")
        # Only read a literal argument. Never import/execute the named function,
        # evaluate expressions, or accept nested arbitrary calls.
        value = ast.literal_eval(node.args[0])
        if function.attr in FLOAT_SCALARS:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Invalid or non-finite NumPy floating-point statistic")
        elif function.attr in INTEGER_SCALARS:
            if type(value) is not int or (function.attr.startswith("u") and value < 0):
                raise ValueError("Invalid NumPy integer statistic")
        elif function.attr in BOOLEAN_SCALARS:
            if type(value) is not bool:
                raise ValueError("Invalid NumPy Boolean statistic")
        else:
            raise ValueError(f"Unsupported NumPy statistic type: {function.attr}")
        return ast.copy_location(ast.Constant(value=value), node)


def statistics_literal_eval(value):
    """Retain literal_eval semantics; also accept finite numeric NumPy reprs."""
    try:
        return ast.literal_eval(value)
    except ValueError:
        tree = (ast.parse(value.lstrip(" \t"), mode="eval") if isinstance(value, str)
                else copy.deepcopy(value))
        tree = _ScalarLiterals().visit(tree)
        return ast.literal_eval(tree)


class _StatisticsAst:
    """Replace only CheckM resultsParser's ast reference, not Python's ast."""
    literal_eval = staticmethod(statistics_literal_eval)

    def __getattr__(self, name):
        return getattr(ast, name)


def _run_native(executable: str, arguments: list[str]) -> None:
    # Preserve the native console script's import search directory and argv.
    # This patch exists only in the child and applies to all three upstream
    # statistics readers (ordinary, extended and marker-gene statistics).
    sys.path.insert(0, str(Path(executable).parent))
    parser = importlib.import_module("checkm.resultsParser")
    if getattr(parser, "ast", None) is not ast:
        raise ValueError("Unsupported CheckM statistics parser; expected its ast module reference")
    parser.ast = _StatisticsAst()
    print(f"[CHECKM COMPAT] executable={executable}; Python={sys.executable}; "
          "safe numeric NumPy-scalar statistics parsing enabled; "
          "native calculations and thresholds unchanged", flush=True)
    sys.argv = [executable, *arguments]
    runpy.run_path(executable, run_name="__main__")


def run_checkm(command: str, tmpdir: Path) -> None:
    """Invoke the resolved native CheckM interpreter without switching envs."""
    from .metadecoder_coverage import _python_launcher

    arguments = shlex.split(command)
    if not arguments or Path(arguments[0]).name != "checkm":
        raise ValueError("CheckM compatibility launcher requires a native checkm command")
    executable = shutil.which(arguments[0])
    if executable is None:
        raise ValueError("CheckM executable is not configured on PATH")
    try:
        launcher = _python_launcher(executable)
    except ValueError as exc:
        raise ValueError(f"Cannot identify CheckM's Python interpreter for {executable}; "
                         "reinstall CheckM using its standard Python console launcher") from exc
    if "--tmpdir" in arguments or any(arg.startswith("--tmpdir=") for arg in arguments):
        raise ValueError("CheckM tmpdir is managed by MetaBAW; do not override it in extra arguments")
    native = [*arguments[1:], "--tmpdir", str(tmpdir)]
    print("[CHECKM COMMAND] " + shlex.join([executable, *native]), flush=True)
    subprocess.run([*launcher, str(Path(__file__).resolve()), "--native-executable",
                    executable, "--", *native], check=True)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 3 or arguments[0] != "--native-executable" or arguments[2] != "--":
        raise ValueError("CheckM compatibility child requires --native-executable PATH -- ARGUMENTS")
    _run_native(arguments[1], arguments[3:])


if __name__ == "__main__":
    main()
