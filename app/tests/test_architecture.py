"""Import boundaries.

Every src module has a layer. A module may import only from lower layers (or from inside its own
package), so lower layers never import upward. A new module or package fails this test until it is
given a layer here, which is the point: every package must be placed deliberately. The domain
layer's stricter stdlib-only rule is in test_policies.py.
"""

import ast
from pathlib import Path

import src

ROOT = Path(src.__file__).parent

LAYERS = {
    "domain": 0,                                      # pure rules, standard library only
    "config": 1, "schemas": 1, "prompts": 1, "state": 1,
    "store": 2, "llm_client": 2, "tool_contracts": 2, "rag": 2, "telemetry": 2,
    "memory": 3, "intake": 3, "tools": 3, "grounding": 3, "reliability": 3, "guardrails": 3,  # framework-neutral services and tools
    "graph": 4, "guard_model": 4,
    "agent": 5,
    "agents": 6,                                      # the team, built on the single agent
    # A package needs a layer before it can exist:
    "mcp": 6, "workflows": 7, "api": 8,
}
# The tool, memory, store and retrieval layers stay framework-neutral so the MCP server can wrap them.
NO_LANGGRAPH = {"domain", "config", "schemas", "prompts", "store", "llm_client", "tool_contracts", "rag",
                "memory", "intake", "tools", "grounding"}


def _iter_modules():
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        yield path, parts


def _iter_imports(path: Path, parts: tuple[str, ...]):
    """(top-level src unit or None, absolute module name) for every import in the file."""
    package = list(parts[:-1])  # the module's package inside src
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
            target = [*base, *(node.module.split(".") if node.module else [])]
            yield (target[0] if target else None), ".".join(target)
        elif isinstance(node, ast.ImportFrom):
            yield (node.module.split(".")[1] if node.module.startswith("src.") else None), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield (alias.name.split(".")[1] if alias.name.startswith("src.") else None), alias.name


def test_every_module_has_a_layer():
    for _, parts in _iter_modules():
        unit = parts[0]
        if unit == "__init__":
            continue
        assert unit in LAYERS, f"src/{'/'.join(parts)}.py has no layer in tests/test_architecture.py"


def test_no_module_imports_from_a_higher_or_equal_layer():
    for path, parts in _iter_modules():
        unit = parts[0]
        if unit == "__init__":
            continue
        for target, name in _iter_imports(path, parts):
            if target is None or target == unit:
                continue  # third-party, standard library, or inside its own package
            assert LAYERS[target] < LAYERS[unit], (
                f"{'/'.join(parts)} (layer {LAYERS[unit]}) imports {name} (layer {LAYERS[target]})"
            )


def test_framework_neutral_layers_do_not_import_langgraph():
    for path, parts in _iter_modules():
        if parts[0] not in NO_LANGGRAPH:
            continue
        for target, name in _iter_imports(path, parts):
            assert not (target is None and name.split(".")[0] == "langgraph"), f"{'/'.join(parts)} imports {name}"


def test_the_checker_resolves_relative_imports():
    # Guards the guard: rag/index.py's "from ..config import" must resolve to config, ".chunker" to rag.
    found = {target for target, _ in _iter_imports(ROOT / "rag" / "index.py", ("rag", "index"))}
    assert {"config", "rag"} <= found
