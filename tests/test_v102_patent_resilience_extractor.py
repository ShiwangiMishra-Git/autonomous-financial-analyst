import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v102-patent-resilience-extractor.ipynb"

def test_patent_exposure_and_missing_join_contracts():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches=["".join(c.get("source") or []) for c in nb["cells"] if "# --- 2.4bm Patent/exclusivity resilience extractor (v102) ---" in "".join(c.get("source") or [])]
    assert len(matches)==1
    namespace={"Dict":Dict,"deepcopy":deepcopy,
               "score_commercial_concentration_v99":lambda company,facts,registry:{}}
    exec(compile(matches[0],str(NOTEBOOK),"exec"),namespace)
    assert namespace["PATENT_RESILIENCE_VERSION_V102"].endswith("v1")
