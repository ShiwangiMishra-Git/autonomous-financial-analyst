import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v103-patent-semantic-validation.ipynb"


def _load_namespace():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bn Patent semantic validation (v103) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    def base(company,facts,registry):
        missing=[]
        products=facts.get("products") or []
        if len(products)<3: missing.append("top_three_product_revenues")
        if any(row.get("exclusivity_year") is None for row in products[:3]): missing.append("top_three_exclusivity_years")
        if missing: return {"status":"N/A","score":None,"missing":missing}
        return {"status":"scored","score":4.0,"missing":[],"total_revenue":facts["total_revenue"]["value"]}
    namespace={"Dict":dict,"score_patent_resilience_v102":base}
    exec(compile(cells[0],str(NOTEBOOK),"exec"),namespace)
    return namespace


def test_rejects_inconsistent_total_and_individual_patent_date():
    ns=_load_namespace()
    registry={"t":{"company":"PFE","source_role":"company_primary","content":"Revenues 62,579"},
              "r":{"company":"PFE","source_role":"company_primary","content":"Eliquis revenue 7,961"},
              "e":{"company":"PFE","source_role":"company_primary","content":"Eliquis patent expires 2036"}}
    facts={"period":"FY2025","total_revenue":{"value":51663,"metric_label":"segment revenue","source_ids":["t"]},
           "products":[{"name":"Eliquis","revenue":7961,"revenue_source_ids":["r"],
                        "exclusivity_year":2036,"date_basis":"individual_patent_expiry","exclusivity_source_ids":["e"]}]}
    result=ns["score_patent_resilience_v103"]("PFE",facts,registry,{"status":"scored","total_revenue":62579})
    assert result["status"]=="N/A"
    assert "commercial_total_reconciliation" in result["missing"]
    assert "consolidated_total_revenue_label" in result["missing"]


def test_accepts_reconciled_explicit_product_linked_loe():
    ns=_load_namespace()
    registry={"t":{"company":"PFE","source_role":"company_primary","content":"Total Revenues 62,579"}}
    products=[]
    for name,revenue,year in (("Alpha",10,2027),("Beta",8,2029),("Gamma",6,2031)):
        sid=name.lower(); registry[sid]={"company":"PFE","source_role":"company_primary","content":f"{name} expected LOE {year}"}
        products.append({"name":name,"revenue":revenue,"revenue_source_ids":[sid],"exclusivity_year":year,
                         "date_basis":"expected_loe","exclusivity_source_ids":[sid]})
    facts={"period":"FY2025","currency":"USD","scale":"millions",
           "total_revenue":{"value":62579,"metric_label":"Revenues","source_ids":["t"]},"products":products}
    result=ns["score_patent_resilience_v103"]("PFE",facts,registry,{"status":"scored","total_revenue":62579})
    assert result["status"]=="scored"
    assert result["commercial_total_revenue"]==62579
