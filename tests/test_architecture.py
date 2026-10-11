"""Layering rules: models and the harness never depend on a task, and tasks never on each other."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "privacy_bench"
FORBIDDEN = {
    "core": ("tasks", "models", "report", "cli"),
    "models": ("tasks", "report", "cli"),
    "report": ("tasks", "models", "cli"),
}


def module_name(path: Path) -> list:
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
    return parts[:-1] if parts[-1] == "__init__" else parts


def imported_modules(path: Path):
    """Yield every imported module as a dotted path, with relative imports resolved."""
    module = module_name(path)
    package = module if path.name == "__init__.py" else module[:-1]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            target = base + (node.module.split(".") if node.module else [])
            yield target
            for alias in node.names:
                yield target + [alias.name]


def internal_dependencies():
    for path in sorted(PACKAGE.rglob("*.py")):
        source = module_name(path)
        for target in imported_modules(path):
            if target[:1] == ["privacy_bench"] and len(target) > 1:
                yield source, target


def test_layers_only_depend_downward():
    violations = []
    for source, target in internal_dependencies():
        layer = source[1] if len(source) > 1 else None
        if layer in FORBIDDEN and target[1] in FORBIDDEN[layer]:
            violations.append(f"{'.'.join(source)} imports {'.'.join(target)}")
    assert violations == []


def test_tasks_never_import_another_task():
    violations = []
    for source, target in internal_dependencies():
        if (len(source) > 2 and source[1] == "tasks" and len(target) > 2 and target[1] == "tasks"
                and source[2] != target[2] and source[2] != "__init__"):
            violations.append(f"{'.'.join(source)} imports {'.'.join(target)}")
    assert violations == []


def test_relative_imports_resolve_inside_the_package():
    assert any(target[:3] == ["privacy_bench", "core", "files"] for _, target in internal_dependencies())
