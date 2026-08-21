import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v107-other-provider-caches.ipynb"


def test_v107_cache_policy_is_versioned_and_fail_closed():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4br Other pharma provider caches (v107) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert 'version="pharma-rag-v107-corpus-bound"' in source
    assert "BIOPHARMA_CORPUS_VERSION_V43" in source
    assert "_pharma_registry_fingerprint_v107" in source
    assert 'value.get("cache_complete") is True' in source
    assert '"cache_complete":_complete_scoring_inputs_v107' in source
    assert '"final_synthesis_cached":False' in source


def test_v107_ttls_match_freshness_policy():
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert "PHARMA_CLINICALTRIALS_TTL_V107=6*60*60" in text
    assert "PHARMA_EXTRACTION_TTL_V107=24*60*60" in text
    assert "stale_while_revalidate=False" in text
