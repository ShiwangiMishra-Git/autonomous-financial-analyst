import json
from pathlib import Path
from typing import Dict

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v101-total-revenue-coverage.ipynb"

def test_revenues_label_requires_total_query_provenance():
    notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches=["".join(cell.get("source") or []) for cell in notebook["cells"]
             if "# --- 2.4bl Commercial total-revenue coverage correction (v101) ---"
             in "".join(cell.get("source") or [])]
    assert len(matches)==1
    namespace={"Dict":Dict}
    exec(compile(matches[0],str(NOTEBOOK),"exec"),namespace)
    assert namespace["COMMERCIAL_REVENUE_COVERAGE_VERSION_V101"].endswith("v3")
