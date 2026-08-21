"""Exhaustiveness checks for developer- and agent-facing multi-industry documentation."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import nbformat

from scripts.generate_multiindustry_contract_reference import (
    REFERENCE_PATH,
    generate_reference,
)


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _typed_dict(node: ast.ClassDef) -> bool:
    """Return whether one parsed class is a direct TypedDict contract."""
    return any(isinstance(base, ast.Name) and base.id == "TypedDict" for base in node.bases)


def test_every_multiindustry_definition_has_a_docstring() -> None:
    """All classes, methods, graph nodes, and helpers must remain discoverable to developers."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    missing: list[str] = []
    for cell in notebook.cells:
        cell_id = str(cell.get("id", ""))
        if not cell_id.startswith("multiindustry_") or cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{cell_id}:{node.name}")
    assert missing == []


def test_every_typed_dict_field_has_an_attributes_description() -> None:
    """Every state/contract field must explain ownership and meaning in its class docstring."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    missing: list[str] = []
    for cell in notebook.cells:
        cell_id = str(cell.get("id", ""))
        if not cell_id.startswith("multiindustry_") or cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _typed_dict(node):
                continue
            documented = {
                match.group(1)
                for line in (ast.get_docstring(node, clean=True) or "").splitlines()
                if (match := re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*):\s+.+", line))
            }
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    if statement.target.id not in documented:
                        missing.append(f"{cell_id}:{node.name}.{statement.target.id}")
    assert missing == []


def test_generated_reference_covers_every_field_and_method(tmp_path: Path) -> None:
    """The concise external index must be reproducible and exhaustive."""
    output = tmp_path / "contracts.md"
    counts = generate_reference(NOTEBOOK_PATH, output)
    rendered = output.read_text(encoding="utf-8")

    assert counts["fields"] >= 170
    assert counts["methods"] >= 200
    assert "`OrchestratorState` | `comparison_mode`" in rendered
    assert "`EvidenceRecord` | `evidence_id`" in rendered
    assert "`run_f15_validated_synthesis" in rendered
    assert REFERENCE_PATH.exists()

