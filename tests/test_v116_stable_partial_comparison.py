import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-19-v116-final-partial-comparison.ipynb"


def test_v116_stable_policy_contract():
    assert NB.exists()
    nb=json.loads(NB.read_text(encoding="utf-8"))
    source="".join(nb["cells"][-1]["source"])
    assert 'PHARMA_PARTIAL_WEIGHTS_V116={"financial_strength":0.75,"independent_sentiment":0.25}' in source
    assert 'PHARMA_REACT_DIMENSIONS_V89=PHARMA_PARTIAL_DIMENSIONS_V116' in source
    assert "annual-report-backed financial strength (75%)" in source
    assert "no standalone R&D field is assumed" in source
    assert "route_pharma_query(" not in source


def test_v116_keeps_evidence_and_na_safeguards():
    source="".join(json.loads(NB.read_text(encoding="utf-8"))["cells"][-1]["source"])
    assert "company-primary source" in source
    assert "publisher-diversity" in source
    assert "Missing evidence is `N/A`, never zero" in source
    assert "both included dimensions are mandatory" in source
