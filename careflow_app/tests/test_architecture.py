"""Import boundaries (ADR-01, extended in M3 as the architect asked in feedback.md §8.2).

Every careflow module has a layer. A module may import only from lower layers (or from inside its own
package), so lower layers never import upward. A new module or package fails this test until it is
given a layer here, which is the point: M5's workflows/, M6's mcp/ and M8's api/ must be placed
deliberately. The domain layer's stricter stdlib-only rule is in test_policies.py.
"""

import ast
from pathlib import Path

import careflow

ROOT = Path(careflow.__file__).parent

LAYERS = {
    "domain": 0,                                      # pure rules, standard library only
    "config": 1, "schemas": 1, "prompts": 1, "state": 1,
    "store": 2, "llm_client": 2, "tool_contracts": 2, "rag": 2,
    "memory": 3, "intake": 3, "tools": 3, "grounding": 3,  # framework-neutral services and tools
    "graph": 4,
    "agent": 5,
    "agents": 6,                                      # M6: the team, built on the single agent
    # Reserved for later milestones (a layer is needed before the package can exist):
    "mcp": 6, "workflows": 7, "api": 8,
}
# The tool, memory, store and retrieval layers stay framework-neutral so M6's MCP server can wrap them.
NO_LANGGRAPH = {"domain", "config", "schemas", "prompts", "store", "llm_client", "tool_contracts", "rag",
                "memory", "intake", "tools", "grounding"}


def _modules():
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        yield path, parts


def _imports(path: Path, parts: tuple[str, ...]):
    """(top-level careflow unit or None, absolute module name) for every import in the file."""
    package = list(parts[:-1])  # the module's package inside careflow
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
            target = [*base, *(node.module.split(".") if node.module else [])]
            yield (target[0] if target else None), ".".join(target)
        elif isinstance(node, ast.ImportFrom):
            yield (node.module.split(".")[1] if node.module.startswith("careflow.") else None), node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield (a.name.split(".")[1] if a.name.startswith("careflow.") else None), a.name


def test_every_module_has_a_layer():
    for _, parts in _modules():
        unit = parts[0]
        if unit == "__init__":
            continue
        assert unit in LAYERS, f"careflow/{'/'.join(parts)}.py has no layer in tests/test_architecture.py"


def test_no_module_imports_from_a_higher_or_equal_layer():
    for path, parts in _modules():
        unit = parts[0]
        if unit == "__init__":
            continue
        for target, name in _imports(path, parts):
            if target is None or target == unit:
                continue  # third-party, standard library, or inside its own package
            assert LAYERS[target] < LAYERS[unit], (
                f"{'/'.join(parts)} (layer {LAYERS[unit]}) imports {name} (layer {LAYERS[target]})"
            )


def test_framework_neutral_layers_do_not_import_langgraph():
    for path, parts in _modules():
        if parts[0] not in NO_LANGGRAPH:
            continue
        for target, name in _imports(path, parts):
            assert not (target is None and name.split(".")[0] == "langgraph"), f"{'/'.join(parts)} imports {name}"


def test_the_checker_resolves_relative_imports():
    # Guards the guard: rag/index.py's "from ..config import" must resolve to config, ".chunker" to rag.
    found = {t for t, _ in _imports(ROOT / "rag" / "index.py", ("rag", "index"))}
    assert {"config", "rag"} <= found
