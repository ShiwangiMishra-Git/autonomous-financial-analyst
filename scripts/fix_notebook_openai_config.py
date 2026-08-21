"""Idempotently normalize OpenAI credential mapping in the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def main() -> None:
    """Accept the documented API_KEY field without reading or exposing config.json values."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    candidates = [
        cell for cell in notebook.cells
        if cell.cell_type == "code"
        and 'os.environ["OPENAI_API_KEY"]' in cell.source
        and "config.get" in cell.source
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one notebook configuration cell, found {len(candidates)}")

    cell = candidates[0]
    old = 'os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")'
    new = (
        'os.environ["OPENAI_API_KEY"] = ('
        'config.get("OPENAI_API_KEY") or config.get("API_KEY") or ""'
        ')'
    )
    if old in cell.source:
        cell.source = cell.source.replace(old, new, 1)
        cell.execution_count = None
        cell.outputs = []
    elif new not in cell.source:
        raise ValueError("Notebook configuration cell has an unexpected API-key assignment")

    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: OpenAI API key mapping is compatible")


if __name__ == "__main__":
    main()
