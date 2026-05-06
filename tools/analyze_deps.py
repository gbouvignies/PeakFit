#!/usr/bin/env python
"""Analyze PeakFit module dependencies and identify simplification targets."""

import ast
from collections import defaultdict
from pathlib import Path


def get_imports(filepath: Path) -> list[tuple[str, str]]:
    """Extract internal peakfit imports from a Python file."""
    with filepath.open(encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("peakfit."):
            imports.append((node.module, ", ".join(a.name for a in node.names)))
    return imports


def analyze_codebase():
    """Analyze the PeakFit codebase structure."""
    src_dir = Path("src/peakfit")

    # Collect module data
    module_data = {}
    layer_counts = defaultdict(lambda: {"files": 0, "lines": 0})

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        rel_path = py_file.relative_to(src_dir)
        module = str(rel_path).replace("/", ".").replace(".py", "")

        with py_file.open(encoding="utf-8") as f:
            content = f.read()
            lines = len(content.split("\n"))

        imports = get_imports(py_file)

        # Determine layer
        parts = module.split(".")
        layer = parts[0] if parts else "root"

        layer_counts[layer]["files"] += 1
        layer_counts[layer]["lines"] += lines

        module_data[module] = {
            "lines": lines,
            "imports": imports,
            "layer": layer,
            "path": str(py_file),
        }

    return module_data, layer_counts


def main():  # noqa: PLR0912, PLR0915
    module_data, layer_counts = analyze_codebase()

    print("=" * 70)
    print("PEAKFIT DEPENDENCY ANALYSIS")
    print("=" * 70)

    # Layer summary
    print("\n## LAYER SUMMARY")
    print("-" * 50)
    print(f"{'Layer':<20} {'Files':>8} {'Lines':>10}")
    print("-" * 50)
    for layer in ["core", "io", "services", "ui", "plotting", "cli"]:
        if layer in layer_counts:
            data = layer_counts[layer]
            print(f"{layer:<20} {data['files']:>8} {data['lines']:>10}")
    print("-" * 50)
    total_files = sum(d["files"] for d in layer_counts.values())
    total_lines = sum(d["lines"] for d in layer_counts.values())
    print(f"{'TOTAL':<20} {total_files:>8} {total_lines:>10}")

    # Top 15 largest modules
    print("\n## TOP 15 LARGEST MODULES")
    print("-" * 60)
    sorted_modules = sorted(module_data.items(), key=lambda x: -x[1]["lines"])
    for mod, data in sorted_modules[:15]:
        print(f"{data['lines']:5d} lines: {mod}")

    # Cross-layer imports
    print("\n## CROSS-LAYER DEPENDENCIES")
    print("-" * 60)

    layer_order = ["core", "io", "services", "ui", "plotting", "cli"]
    violations = []

    for mod, data in module_data.items():
        mod_layer = data["layer"]
        for imp_module, _ in data["imports"]:
            # Extract layer from import
            parts = imp_module.replace("peakfit.", "").split(".")
            if parts:
                imp_layer = parts[0]

                # Check for layer violations (lower importing higher)
                if mod_layer in layer_order and imp_layer in layer_order:
                    mod_idx = layer_order.index(mod_layer)
                    imp_idx = layer_order.index(imp_layer)
                    if mod_idx < imp_idx:
                        violations.append(f"  {mod_layer}/{mod} -> {imp_layer}")

    if violations:
        for v in sorted(set(violations))[:10]:
            print(v)
    else:
        print("  No layer violations detected")

    # Duplication analysis
    print("\n## POTENTIAL DUPLICATION PATTERNS")
    print("-" * 60)

    # Look for similar module patterns
    patterns = defaultdict(list)
    min_parts_for_suffix = 2
    min_modules_for_pattern = 3
    for mod in module_data:
        # Group by suffix
        parts = mod.split(".")
        if len(parts) >= min_parts_for_suffix:
            suffix = parts[-1]
            patterns[suffix].append(mod)

    for suffix, modules in sorted(patterns.items(), key=lambda x: -len(x[1])):
        if len(modules) >= min_modules_for_pattern and suffix not in [
            "__init__",
            "model",
            "kernel",
        ]:
            print(f"  '{suffix}' appears in {len(modules)} modules")

    # Service/Orchestrator pattern detection
    print("\n## SERVICE/ORCHESTRATOR PATTERN ANALYSIS")
    print("-" * 60)
    service_patterns = []
    for mod, data in module_data.items():
        mod_lower = mod.lower()
        if any(p in mod_lower for p in ["service", "orchestrator", "manager"]):
            service_patterns.append((mod, data["lines"]))

    for mod, lines in sorted(service_patterns, key=lambda x: -x[1]):
        print(f"  {lines:5d} lines: {mod}")


if __name__ == "__main__":
    main()
