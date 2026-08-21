"""Run the corrected pharma scoring harness without any provider access."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-"
    "v92-harness-label-compatibility.ipynb"
)


def _cell_source(notebook, marker):
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if marker in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    return matches[0]


def test_active_renderer_passes_older_readability_harness():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {
        "route_pharma_query": lambda query, **kwargs: {},
        "display": lambda value: None,
        "Markdown": lambda value: value,
    }

    scoring_source = _cell_source(notebook, "Gated pharma scoring prototype (2026-08-18 v83)")
    exec(compile(scoring_source, str(NOTEBOOK), "exec"), namespace)

    renderer_source = _cell_source(notebook, "Pharma scoring citation/readability boundary (2026-08-18 v88)")
    tree = ast.parse(renderer_source)
    renderer = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "render_pharma_scores_v83")
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)

    harness_source = _cell_source(notebook, "Offline pharma scoring correctness and readability harness (v85)")
    exec(compile(harness_source, str(NOTEBOOK), "exec"), namespace)
    report = namespace["PHARMA_SCORING_HARNESS_REPORT_V85"]
    assert report["failed"] == 0
    assert report["passed"] == report["total"] == 15
    assert report["provider_calls"] == 0

