"""Execute the complete deterministic Section 3 integration query cell."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def test_full_section3_cross_profile_query():
    """Run planning through isolated technology and biopharma worker results."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "query_private_database": lambda query: "legacy", "extract_ai_signals": lambda *a, **k: {},
        "score_companies": lambda *a, **k: {},
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for cell_id in (
            "multiindustry_state_contracts", "multiindustry_company_registry",
            "multiindustry_query_planner", "multiindustry_f03_smoke",
            "multiindustry_industry_profiles", "multiindustry_company_tasks",
            "multiindustry_evidence_adapters", "multiindustry_technology_profile",
            "multiindustry_biopharma_rag", "multiindustry_biopharma_signals",
            "multiindustry_company_worker", "multiindustry_f10_smoke",
            "multiindustry_parent_orchestrator", "multiindustry_f11_smoke",
            "multiindustry_fan_in_normalization", "multiindustry_f12_normalization_smoke",
            "multiindustry_comparison_mode_routing", "multiindustry_f12_routing_smoke",
            "multiindustry_section3_integration_test",
        ):
            exec(cells[cell_id], namespace)

    assert "Full Section 3 test query passed" in output.getvalue()
    assert "F11 smoke test passed" in output.getvalue()
    assert set(namespace["_section3_results"]) == {"MSFT", "PFE"}
    assert namespace["_section3_state"]["comparison_mode"] == "cross_profile"
    assert namespace["_section3_state"]["scoring_eligibility"]["eligible"] is False
