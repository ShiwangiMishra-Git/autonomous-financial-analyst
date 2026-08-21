"""Idempotently add pypdf's AES backend to the working notebook install cell."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
DEPENDENCY_LINE = "  cryptography==46.0.7 \\\n"


def main() -> None:
    """Insert cryptography immediately after pypdf in the notebook pip cell."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    candidates = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "!pip install" in cell.source and "pypdf==6.2.0" in cell.source
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one pypdf installation cell, found {len(candidates)}")

    cell = candidates[0]
    if "cryptography==46.0.7" not in cell.source:
        cell.source = cell.source.replace(
            "  cryptography>=43.0.0 \\\n", "", 1,
        )
        marker = "  pypdf==6.2.0 \\\n"
        if marker not in cell.source:
            raise ValueError("Could not locate the pypdf dependency line")
        cell.source = cell.source.replace(marker, marker + DEPENDENCY_LINE, 1)
        cell.execution_count = None
        cell.outputs = []

    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: cryptography AES dependency is present")


if __name__ == "__main__":
    main()
