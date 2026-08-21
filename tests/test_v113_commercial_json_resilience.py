import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v113-commercial-json-resilience.ipynb"


def test_parser_repairs_common_json_formatting_and_fails_closed():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bw Commercial JSON resilience (v113) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert "ast.literal_eval" in source
    assert "commercial_extraction_valid_json" in source
    assert "_raw_preview" in source
    assert "pharma-commercial-extractor-v113-json-resilient" in source


def test_v113_has_no_embedded_live_demo():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text="\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "Combined end-to-end pharma demo (live providers)" not in text
    assert "pharma-v108-demo-" not in text
