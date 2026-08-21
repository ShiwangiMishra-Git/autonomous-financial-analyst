"""Offline execution contract for the v99 commercial extractor."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v99-commercial-concentration-extractor.ipynb"


def test_commercial_extractor_arithmetic_and_missing_policy():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if "Commercial concentration: vector retrieval + LLM extraction (v99)"
               in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    namespace = {"Dict": Dict, "deepcopy": deepcopy}
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    assert namespace["COMMERCIAL_CONCENTRATION_VERSION_V99"].endswith("v1")

