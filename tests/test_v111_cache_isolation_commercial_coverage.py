import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v111-cache-isolation-commercial-coverage.ipynb"


def test_web_cache_queries_are_company_scoped():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bu Company-isolated web cache and commercial coverage (v111) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert "_scoped_react_query_v98(company,dimension,query)" in source
    assert 'company_scoped_query_v111' in source
    assert "_v111_pfe!=_v111_mrk" in source


def test_incomplete_commercial_product_set_is_rejected():
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert "COMMERCIAL_MIN_LISTED_REVENUE_COVERAGE_V111=.25" in text
    assert "commercial_product_table_coverage" in text
    assert "pharma-commercial-extractor-v111-coverage" in text
