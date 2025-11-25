"""Validate import structure follows architectural rules.

This script checks that imports respect the layered architecture:
- cli/ can only import from services/, ui/, and limited core/ (config)
- services/ can import from core/, infra/
- io/ cannot import from ui/
- core/ cannot import from ui/, cli/, services/
"""

import ast
import sys
from pathlib import Path

# Define forbidden import patterns (source_package, target_package)
FORBIDDEN_PATTERNS = [
    ("io", "ui"),  # io cannot import ui
    ("core", "ui"),  # core cannot import ui
    ("core", "cli"),  # core cannot import cli
    ("core", "services"),  # core cannot import services
    ("infra", "ui"),  # infra cannot import ui
    ("infra", "cli"),  # infra cannot import cli
]


def get_package_from_path(rel_path: Path) -> str:
    """Extract the top-level package name from a relative path."""
    parts = rel_path.parts
    return str(parts[0]) if parts else ""


def get_import_target(node: ast.Import | ast.ImportFrom) -> str | None:
    """Extract import target from AST node."""
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else None
    elif isinstance(node, ast.ImportFrom):
        return node.module
    return None


def extract_target_package(import_path: str) -> str | None:
    """Extract the target package from a full import path."""
    if not import_path or not import_path.startswith("peakfit."):
        return None
    # Remove "peakfit." prefix and get first package
    remainder = import_path.replace("peakfit.", "")
    return remainder.split(".")[0]


def validate_file(py_file: Path, src_dir: Path) -> list[str]:
    """Validate imports in a single file."""
    violations = []
    rel_path = py_file.relative_to(src_dir)
    source_pkg = get_package_from_path(rel_path)

    try:
        with open(py_file, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            target = get_import_target(node)
            if not target:
                continue

            target_pkg = extract_target_package(target)
            if not target_pkg:
                continue

            for src, tgt in FORBIDDEN_PATTERNS:
                if source_pkg == src and target_pkg == tgt:
                    line_no = getattr(node, "lineno", "?")
                    violations.append(
                        f"{py_file}:{line_no}: {source_pkg} -> {target_pkg} ({target})"
                    )

    return violations


def validate_imports(src_dir: Path) -> list[str]:
    """Validate all imports in source directory."""
    all_violations = []

    for py_file in src_dir.rglob("*.py"):
        violations = validate_file(py_file, src_dir)
        all_violations.extend(violations)

    return all_violations


def main() -> int:
    """Main entry point."""
    src_dir = Path("src/peakfit")

    if not src_dir.exists():
        # Try from project root
        src_dir = Path(__file__).parent.parent / "src" / "peakfit"

    if not src_dir.exists():
        print(f"Error: Could not find source directory: {src_dir}")
        return 1

    print(f"Validating imports in: {src_dir}")
    violations = validate_imports(src_dir)

    if violations:
        print(f"\n❌ Found {len(violations)} import violation(s):\n")
        for v in sorted(violations):
            print(f"  ✗ {v}")
        return 1
    else:
        print("\n✓ All imports follow architectural rules")
        return 0


if __name__ == "__main__":
    sys.exit(main())
