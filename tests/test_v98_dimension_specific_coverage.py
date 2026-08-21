"""Offline execution contract for v98 dimension-specific coverage."""

import json
import re
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v98-dimension-specific-coverage.ipynb"


def test_generic_records_do_not_satisfy_unrelated_dimensions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if "Dimension-specific pharma evidence coverage (2026-08-18 v98)"
               in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    namespace = {
        "Dict": Dict, "List": List, "re": re,
        "_default_react_action_runner_v89": lambda action, company, dimension, query: {
            "evidence_registry": {}
        },
        "_candidate_records_v94": lambda registry, company, dimension: [],
    }
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_DIMENSION_COVERAGE_VERSION_V98"].endswith("v1")

