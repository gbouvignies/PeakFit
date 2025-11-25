import ast
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "peakfit"


def iter_modules(root: Path):
    for path in root.rglob("*.py"):
        rel = path.relative_to(root.parent)
        module = str(rel).replace(os.sep, ".")
        if module.endswith("__init__.py"):
            module = module[:-12]
        else:
            module = module[:-3]
        yield module, path


def build_dependency_graph():
    module_edges: dict[str, set[str]] = defaultdict(set)
    package_edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for module, path in iter_modules(ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"Failed to parse {path}: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("peakfit") and name != module:
                        module_edges[module].add(name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("peakfit")
            ):
                module_edges[module].add(node.module)

    for src_module, targets in module_edges.items():
        src_parts = src_module.split(".")
        src_pkg = src_parts[1] if len(src_parts) > 1 else src_module
        for tgt in targets:
            tgt_parts = tgt.split(".")
            tgt_pkg = tgt_parts[1] if len(tgt_parts) > 1 else tgt
            if src_pkg == tgt_pkg:
                continue
            package_edges[src_pkg][tgt_pkg] += 1

    return module_edges, package_edges


def summarize(module_edges, package_edges, limit: int = 20):
    print("Package dependency counts:")
    for src_pkg, targets in sorted(package_edges.items()):
        for tgt_pkg, count in sorted(targets.items()):
            print(f"{src_pkg:15s} -> {tgt_pkg:15s} : {count}")

    print("\nTop modules importing many peers:")
    for module, targets in sorted(module_edges.items(), key=lambda kv: len(kv[1]), reverse=True)[
        :limit
    ]:
        print(f"{module:60s} {len(targets)}")


if __name__ == "__main__":
    module_edges, package_edges = build_dependency_graph()
    summarize(module_edges, package_edges)
