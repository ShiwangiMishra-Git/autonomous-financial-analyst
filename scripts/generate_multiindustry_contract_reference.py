"""Generate an exhaustive developer/agent reference from canonical notebook docstrings.

The generator is read-only with respect to code cells. It extracts every multi-industry
``TypedDict`` field and every documented class/function signature, then writes one compact
Markdown reference. This makes the notebook contracts discoverable without duplicating runtime
implementation or private data.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Iterable

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
REFERENCE_PATH = PROJECT_ROOT / "docs" / "designs" / "multi-industry-state-contract-method-reference.md"


def _summary(docstring: str | None) -> str:
    """Return the first non-empty docstring line for a compact table description.

    Args:
        docstring: Raw AST docstring or ``None``.

    Returns:
        One-line Markdown-safe summary.

    Usage:
        Used for every class and function row in the generated reference.
    """
    line = next((line.strip() for line in (docstring or "").splitlines() if line.strip()), "")
    return line.replace("|", "\\|") or "Documented internal contract."


def _attributes(docstring: str | None) -> dict[str, str]:
    """Parse Google-style ``Attributes`` entries from one class docstring.

    Args:
        docstring: Raw class docstring.

    Returns:
        Field-name to one-line description mapping.

    Usage:
        Supplies human meaning alongside static TypedDict annotations.
    """
    output: dict[str, str] = {}
    active = False
    for line in (docstring or "").splitlines():
        stripped = line.strip()
        if stripped == "Attributes:":
            active = True
            continue
        if active and stripped and not line.startswith((" ", "\t")):
            break
        if active:
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*):\s*(.+)", stripped)
            if match:
                output[match.group(1)] = match.group(2)
    return output


def _is_typed_dict(node: ast.ClassDef) -> bool:
    """Return whether a class directly declares a ``TypedDict`` contract.

    Args:
        node: Parsed class definition.

    Returns:
        Boolean used to route classes into the state/contract section.
    """
    return any(
        (isinstance(base, ast.Name) and base.id == "TypedDict")
        or (isinstance(base, ast.Attribute) and base.attr == "TypedDict")
        for base in node.bases
    )


def _definitions(tree: ast.AST) -> Iterable[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield documented definitions in source order, including graph-local node functions.

    Args:
        tree: Parsed notebook cell syntax tree.

    Returns:
        Iterator of class, function, and async-function nodes.

    Usage:
        Includes nested LangGraph nodes because developers and agents need their contracts too.
    """
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    yield from sorted(nodes, key=lambda node: (node.lineno, node.col_offset, node.name))


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render one function's inputs without its implementation body.

    Args:
        node: Parsed function definition.

    Returns:
        Markdown-safe signature containing argument annotations/defaults.
    """
    rendered = ast.unparse(node.args).replace("|", "\\|")
    return f"`{node.name}({rendered})`"


def _return_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render the declared output type of one function.

    Args:
        node: Parsed function definition.

    Returns:
        Markdown code span, or ``unspecified`` for runtime/dynamic APIs.
    """
    return f"`{ast.unparse(node.returns)}`" if node.returns is not None else "unspecified"


def _usage(name: str, summary: str) -> str:
    """Derive a short developer/agent usage hint from a documented definition name.

    Args:
        name: Definition name.
        summary: Existing first-line docstring summary.

    Returns:
        Concise intended-use description.
    """
    if name.startswith("_"):
        return "Internal helper; use through its owning public boundary."
    if name.endswith("_tool") or name.endswith("_tool_node"):
        return "Guarded tool/node boundary; invoke with its declared schema."
    if name.startswith(("validate", "check")):
        return "Deterministic guard; call before the next workflow boundary."
    if name.startswith(("create", "build")):
        return "Factory/builder; call during graph or contract setup."
    if name.startswith(("run", "execute", "ask")):
        return "Execution entry point; inspect its structured terminal result."
    return summary


def generate_reference(
    notebook_path: Path = NOTEBOOK_PATH,
    output_path: Path = REFERENCE_PATH,
) -> dict[str, int]:
    """Generate the complete multi-industry state/contract/method reference.

    Args:
        notebook_path: Canonical working notebook to inspect.
        output_path: Markdown file to atomically replace.

    Returns:
        Counts of documented state fields, classes, and methods.

    Usage:
        Run after notebook integration; tests verify the generated reference remains exhaustive.
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    state_sections: list[str] = []
    method_sections: list[str] = []
    field_count = class_count = method_count = 0

    for cell in notebook.cells:
        cell_id = str(cell.get("id", ""))
        if not cell_id.startswith("multiindustry_") or cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        state_rows: list[str] = []
        method_rows: list[str] = []
        for node in _definitions(tree):
            docstring = ast.get_docstring(node, clean=True)
            if not docstring:
                raise ValueError(f"Undocumented definition {cell_id}:{node.name}")
            if isinstance(node, ast.ClassDef):
                class_count += 1
                if _is_typed_dict(node):
                    descriptions = _attributes(docstring)
                    for statement in node.body:
                        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                            continue
                        field = statement.target.id
                        annotation = ast.unparse(statement.annotation).replace("|", "\\|")
                        description = descriptions.get(field, "Typed field; see owning contract usage.")
                        state_rows.append(
                            f"| `{node.name}` | `{field}` | `{annotation}` | {description} |"
                        )
                        field_count += 1
                else:
                    method_rows.append(
                        f"| class `{node.name}` | — | — | {_summary(docstring)} | "
                        "Instantiate/use through the owning feature boundary. |"
                    )
            else:
                method_count += 1
                summary = _summary(docstring)
                method_rows.append(
                    f"| function | {_signature(node)} | {_return_type(node)} | {summary} | "
                    f"{_usage(node.name, summary)} |"
                )
        if state_rows:
            state_sections.extend([
                f"### `{cell_id}`", "",
                "| Contract | Field | Type | Meaning |",
                "|---|---|---|---|", *state_rows, "",
            ])
        if method_rows:
            method_sections.extend([
                f"### `{cell_id}`", "",
                "| Kind | Inputs | Output | Purpose | How to use |",
                "|---|---|---|---|---|", *method_rows, "",
            ])

    content = "\n".join([
        "# Multi-Industry State, Contract, and Method Reference",
        "",
        "**Generated from:** `Autonomous_financial_analyst_Learners_Notebook copy.ipynb`  ",
        "**Scope:** F00–F16 notebook-local multi-industry implementation",
        "",
        "This is the concise developer/agent contract index. Runtime types remain authoritative; ",
        "the descriptions explain ownership and intended use. Private evidence values and secrets ",
        "are deliberately excluded.",
        "",
        "## 1. State and structured-contract fields",
        "",
        *state_sections,
        "## 2. Classes and methods",
        "",
        "Inputs show the callable signature; Output shows the declared return annotation. Nested ",
        "LangGraph node functions are included because they are workflow contracts even though ",
        "developers normally call the compiled graph rather than those nodes directly.",
        "",
        *method_sections,
    ]).rstrip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output_path)
    return {"fields": field_count, "classes": class_count, "methods": method_count}


if __name__ == "__main__":
    print(generate_reference())

