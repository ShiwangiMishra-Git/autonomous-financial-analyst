"""Tests for F16 canonical notebook integration and retained safe outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

from scripts.implement_multiindustry_f16 import (
    F16_DEMO_CODE,
    F16_INTERACTIVE_CODE,
    F16_LIVE_STATUS_CODE,
    F16_ONLINE_DEMO_CODE,
    F16_SETUP_CODE,
    execute_and_save_f16_outputs,
    integrate_f16_cells,
)


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _stream_text(cell) -> str:
    """Return concatenated stdout text from one executed notebook cell."""
    return "".join(
        output.get("text", "")
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_f16_cells_are_ordered_after_f15_and_match_sources():
    """Keep the canonical notebook synchronized without replacing earlier cells."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells = {cell.get("id"): cell for cell in notebook.cells}
    ids = [cell.get("id") for cell in notebook.cells]
    assert ids.index("multiindustry_f15_smoke") < ids.index("multiindustry_f16_intro")
    assert cells["multiindustry_f16_runner_setup"].source == F16_SETUP_CODE
    assert cells["multiindustry_f16_all_scenarios_demo"].source == F16_DEMO_CODE
    assert cells["multiindustry_f16_live_status"].source == F16_LIVE_STATUS_CODE
    assert cells["multiindustry_f16_online_demo"].source == F16_ONLINE_DEMO_CODE
    assert cells["multiindustry_f16_interactive_query"].source == F16_INTERACTIVE_CODE
    assert ids.index("multiindustry_f16_live_status") < ids.index("multiindustry_f16_online_demo")
    assert ids.index("multiindustry_f16_online_demo") < ids.index(
        "multiindustry_f16_interactive_query"
    )


def test_saved_demo_output_contains_all_ten_compact_summaries():
    """Verify the delivered notebook retains the executed all-scenario output."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells = {cell.get("id"): cell for cell in notebook.cells}
    demo_text = _stream_text(cells["multiindustry_f16_all_scenarios_demo"])
    summaries = json.loads(demo_text)
    assert len(summaries) == 10
    assert [item["scenario"] for item in summaries] == [
        "single_technology",
        "single_biopharma",
        "same_profile_technology",
        "same_profile_biopharma",
        "cross_profile",
        "alias_resolution",
        "unknown_company",
        "partial_rag_failure",
        "invalid_evidence_id",
        "modified_f13_score",
    ]
    assert all("answer" not in item and "errors" not in item for item in summaries)
    assert cells["multiindustry_f16_all_scenarios_demo"].execution_count is not None


def test_demo_can_execute_again_in_an_isolated_trace_directory(tmp_path):
    """Re-run only F16 cells and verify expected trace and live-gate outcomes."""
    temporary_notebook = tmp_path / "working-copy.ipynb"
    temporary_notebook.write_bytes(NOTEBOOK_PATH.read_bytes())
    integrate_f16_cells(temporary_notebook)
    outcome = execute_and_save_f16_outputs(
        temporary_notebook,
        trace_dir=tmp_path / ".research_runs",
    )
    summaries = outcome["summaries"]
    assert len(summaries) == 10
    assert len(list((tmp_path / ".research_runs").glob("*.json"))) == 9
    by_name = {item["scenario"]: item for item in summaries}
    assert by_name["unknown_company"]["final_status"] == "bounded_stop"
    assert by_name["invalid_evidence_id"]["final_status"] == "failed"
    assert by_name["modified_f13_score"]["final_status"] == "failed"
    assert outcome["live_summary"]["live_execution"] in {"skipped", "enabled"}
    assert outcome["online_summaries"] == []
    notebook = nbformat.read(temporary_notebook, as_version=4)
    online_cell = next(
        cell for cell in notebook.cells if cell.get("id") == "multiindustry_f16_online_demo"
    )
    online_output = json.loads(_stream_text(online_cell))
    assert online_output["online_demo"] == "skipped"
    assert isinstance(online_output["missing_environment_variable_names"], list)
    assert online_output["missing_contract_names"]
