import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build_v115_partial_financial_sentiment.py"
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v115-partial-financial-sentiment.ipynb"


def test_v115_notebook_and_policy_contract():
    assert BUILD.exists()
    assert NOTEBOOK.exists()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"][-2:])
    assert 'PHARMA_PARTIAL_WEIGHTS_V115={"financial_strength":0.60,"rd_funding_capacity":0.20,"independent_sentiment":0.20}' in source
    assert "PHARMA_REACT_DIMENSIONS_V89=PHARMA_PARTIAL_DIMENSIONS_V115" in source
    assert '"partial pharma comparison" in text' in source
    assert '"partial comparison" in text' in source
    assert "_extract_scoring_inputs_v109_base" in source
    assert "This is not a complete pharma-investment ranking" in source
    assert "FY2025 annual-report-backed R&D funding-capacity submetric" in source
    assert 'financial_observation.get("submetrics")' in source
    assert "all three included components are mandatory" in source
    assert "Missing evidence is `N/A`, never zero" in source


def test_v115_has_no_embedded_live_demo():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][-1]["source"])
    assert "route_pharma_query(" not in cell
    assert "provider_budget_remaining=100" not in cell
