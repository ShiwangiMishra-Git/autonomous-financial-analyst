import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v104-patent-unit-normalization.ipynb"


def test_equivalent_billion_and_million_totals_reconcile():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bo Patent revenue unit normalization (v104) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    def semantic(company,facts,registry,commercial_reference=None):
        patent=float(facts["total_revenue"]["value"]); reference=float(commercial_reference["total_revenue"])
        missing=[] if abs(patent-reference)/reference<=.01 else ["commercial_total_reconciliation"]
        if facts["products"][0]["date_basis"]!="expected_loe": missing.append("explicit_material_loe_basis:Eliquis")
        return {"status":"N/A" if missing else "scored","score":None,"missing":missing,"facts":facts}
    namespace={"Dict":dict,"score_patent_resilience_v103":semantic}
    exec(compile(cells[0],str(NOTEBOOK),"exec"),namespace)
    assert namespace["_patent_scaled_value_v104"](62.579,"billions")==namespace["_patent_scaled_value_v104"](62579,"millions")
