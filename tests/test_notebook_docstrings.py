"""Documentation coverage checks for the canonical working notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _contains_notebook_magic(source: str) -> bool:
    """Return whether a cell contains shell or notebook magic that Python AST cannot parse."""
    return any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())


def test_every_parseable_class_function_and_method_has_a_docstring():
    """Require documentation for every Python definition in the working notebook."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    missing: list[str] = []

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip() or _contains_notebook_magic(source):
            continue

        tree = ast.parse(source, filename=f"notebook-cell-{index}")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=True)
            if not docstring:
                missing.append(f"cell={cell.get('id')} line={node.lineno} name={node.name}")

    assert not missing, "Missing notebook docstrings:\n" + "\n".join(missing)
