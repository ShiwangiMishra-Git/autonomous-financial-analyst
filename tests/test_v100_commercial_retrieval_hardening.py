"""Offline execution contract for v100 commercial retrieval."""

import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v100-commercial-retrieval-hardening.ipynb"


def test_split_evidence_and_deduplication_contracts():
    notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches=["".join(cell.get("source") or []) for cell in notebook["cells"]
             if "Commercial concentration retrieval hardening (v100)"
             in "".join(cell.get("source") or [])]
    assert len(matches)==1
    namespace={"Dict":Dict,"run_commercial_concentration_v99":lambda *a,**k:{},
               "extract_commercial_facts_v99":lambda *a,**k:{},
               "score_commercial_concentration_v99":lambda *a,**k:{}}
    exec(compile(matches[0],str(NOTEBOOK),"exec"),namespace)
    assert namespace["COMMERCIAL_RETRIEVAL_VERSION_V100"].endswith("v2")

