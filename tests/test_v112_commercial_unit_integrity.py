import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v112-commercial-unit-integrity.ipynb"


def test_mixed_unit_merck_regression_is_explicitly_blocked():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bv Commercial unit and top-product integrity (v112) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert "31.641 billion must be returned as 31,641 millions" in source
    assert "COMMERCIAL_MIN_TOP3_COVERAGE_V112=.25" in source
    assert "top_three_product_revenue_coverage_or_unit_integrity" in source
    assert "pharma-commercial-extractor-v112-unit-integrity" in source


def test_v112_remains_free_of_live_demo_calls():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text="\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "Combined end-to-end pharma demo (live providers)" not in text
    assert "pharma-v108-demo-" not in text
