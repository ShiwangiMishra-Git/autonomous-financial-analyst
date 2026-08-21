import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v114-json-thousands-separator.ipynb"


def test_money_repair_is_field_scoped_and_cache_versioned():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bx Commercial JSON thousands-separator repair (v114) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    source=cells[0]
    assert '"(?:value|revenue)"' in source
    assert "65,011" in source and "31,641" in source
    assert "years" in source and "2025,2026" in source
    assert "pharma-commercial-extractor-v114-money-json" in source


def test_v114_has_no_live_demo_cell():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text="\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "pharma-v108-demo-" not in text
