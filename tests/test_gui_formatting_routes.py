"""No GUI text is built from a number outside `peakfit.gui.formatting` (issue #211).

A static check over the GUI sources: format specifications with a precision or a
numeric presentation (``f"{x:.2f}"``, ``{x:g}``, ``{n:,}``), ``%`` and ``str.format``
templates and ``repr``. Integers interpolated as they are (``{n}``) and
width or alignment alone remain allowed.
"""

import ast
import re
from pathlib import Path

GUI = Path(__file__).resolve().parents[1] / "src/peakfit/gui"
FORMATTER = GUI / "formatting.py"
# A format spec that formats a number: precision, grouping or a numeric type.
_NUMERIC_SPEC = re.compile(r"\.\d|[,_]|[bcdeEfFgGnoxX%]$")
_PERCENT_TEMPLATE = re.compile(r"%[-+ #0]*\d*(\.\d+)?[diouxXeEfFgG]")
_FORMAT_TEMPLATE = re.compile(r"\{[^{}]*:[^{}]*(\.\d|[,_]|[eEfFgG%])[^{}]*\}")


def _spec_text(spec: ast.JoinedStr) -> str:
    return "".join(str(v.value) for v in spec.values if isinstance(v, ast.Constant))


def bypasses(source: str) -> list[tuple[int, str]]:
    """(line, reason) for each number turned into text without the formatter."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FormattedValue):
            if node.format_spec is not None:
                spec = _spec_text(node.format_spec)  # ty: ignore[invalid-argument-type]
                if _NUMERIC_SPEC.search(spec):
                    found.append((node.lineno, f"format spec {spec!r}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "repr":
                found.append((node.lineno, "repr()"))
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "format"
                and isinstance(func.value, ast.Constant)
                and isinstance(func.value.value, str)
                and _FORMAT_TEMPLATE.search(func.value.value)
            ):
                found.append((node.lineno, "str.format template"))
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and _PERCENT_TEMPLATE.search(node.left.value)
        ):
            found.append((node.lineno, "% template"))
    return sorted(set(found))


def test_the_check_finds_each_kind_of_bypass():
    source = (
        "a = f'{x:.2f}'\nb = f'{x:g}'\nc = f'{n:,}'\nd = f'{x:+.1e}'\ne = repr(x)\n"
        "f = '%.3f' % x\ng = '{:.1e}'.format(x)\nok = f'{n} {name:>8} {formatting.ppm(x)}'\n"
    )
    assert [line for line, _ in bypasses(source)] == [1, 2, 3, 4, 5, 6, 7]


def test_no_gui_label_formats_a_number_outside_the_formatter():
    found = [
        f"{path.relative_to(GUI)}:{line}: {reason}"
        for path in sorted(GUI.rglob("*.py"))
        if path != FORMATTER
        for line, reason in bypasses(path.read_text(encoding="utf-8"))
    ]
    assert not found, "Format these through peakfit.gui.formatting:\n" + "\n".join(found)
