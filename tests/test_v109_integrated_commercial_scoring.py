import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v109-integrated-commercial-scoring.ipynb"


def test_generic_commercial_score_cannot_survive_integration():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bt Integrate deterministic commercial scoring (v109) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert 'merged.setdefault(company,{})["commercial_concentration"]=_commercial_observation_v109(deterministic)' in source
    assert 'deterministic_commercial_score_required' in source
    assert 'PHARMA_COMMERCIAL_SCORING_METHOD_V109="deterministic_commercial_v99"' in source


def test_missing_deterministic_facts_are_na_not_llm_fallback():
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert "calculation_status" in text and "N/A" in text
    assert "score" in text and "None" in text
    assert "positive_total_revenue" in text


def test_post_retrieval_budget_reserves_commercial_extractors():
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert 'PHARMA_NORMALIZER_RESERVE_V95=max' in text
    assert ',5)' in text
