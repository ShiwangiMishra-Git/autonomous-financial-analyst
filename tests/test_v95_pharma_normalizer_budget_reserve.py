"""Offline execution contract for the v95 provider-budget reserve."""

import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v95-normalizer-budget-reserve.ipynb"


def test_one_call_is_reserved_only_for_allowed_investment_requests():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if "Pharma scoring-normalizer budget reserve (2026-08-18 v95)"
               in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    source = matches[0]
    namespace = {
        "Dict": Dict,
        "_route_pharma_query_v84_non_scoring_base": lambda query, **kwargs: {},
        "_investment_scoring_request_v83": lambda query: "investment" in query.lower()
                                                    or "investments" in query.lower(),
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_NORMALIZER_RESERVE_V95"] == 1
