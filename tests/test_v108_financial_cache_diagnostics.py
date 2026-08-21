import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v108-financial-cache-diagnostics.ipynb"


def test_v108_reports_both_cache_layers_without_adding_cache():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bs Shared financial cache diagnostics (v108) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert "peek_cached(get_financial_metrics" in source
    assert "peek_cached(_get_yf_info" in source
    assert "financial_metrics_disk_hit" in source
    assert "financial_metrics_miss_yahoo_info_disk_hit" in source
    assert "financial_metrics_miss_yahoo_provider_called" in source
    assert "@cached_call" not in source


def test_v108_preserves_existing_ttls():
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert "metrics_ttl_seconds" in text and "FINANCIAL_METRICS_TTL" in text
    assert "yahoo_info_ttl_seconds" in text and "STOCK_PRICE_TTL" in text
