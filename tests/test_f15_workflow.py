"""Integration tests for the reconciled bounded F14-to-F15 workflow."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE
from scripts.implement_multiindustry_f14 import F14_CODE
from scripts.implement_multiindustry_f15 import F15_SMOKE, F15_WORKFLOW_CODE
from scripts.implement_multiindustry_f15_evidence import F15_EVIDENCE_CODE
from scripts.implement_multiindustry_f15_traces import F15_TRACES_CODE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
TECH_PROFILE = "technology.ai.v1"
BIO_PROFILE = "healthcare.biopharma.v1"


def _namespace():
    """Execute stable contracts and the isolated F12/F14/F15 sources."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "get_industry_profile": lambda profile_id: {
            "profile_id": profile_id,
            "scoring_enabled": True,
            "rubric_id": "technology.ai.score.v1",
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(F12_ROUTING_CODE, namespace)
        exec(F14_CODE, namespace)
        exec(F15_EVIDENCE_CODE, namespace)
        exec(F15_TRACES_CODE, namespace)
        exec(F15_WORKFLOW_CODE, namespace)
    return namespace


def _company(ticker: str, profile_id: str):
    """Return one canonical resolved company used by workflow fixtures."""
    company_id = {"MSFT": "microsoft", "NVDA": "nvidia", "PFE": "pfizer"}[ticker]
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_id.title(),
        "profile_id": profile_id,
        "resolution_status": "resolved",
    }


def _result(run_id: str, ticker: str, profile_id: str):
    """Return one normalized result with current-run usable evidence."""
    company = _company(ticker, profile_id)
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [{
            "evidence_id": f"EV-{run_id}-{ticker}",
            "run_id": run_id,
            "company_id": company["company_id"],
            "ticker": ticker,
            "profile_id": profile_id,
            "evidence_type": "test_fixture",
            "source_name": "test",
            "source_uri": None,
            "status": "success",
        }],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


def _context(mode: str):
    """Build one valid synthesis context for each comparison mode."""
    run_id = f"workflow-{mode.replace('_', '-')}"
    entries = {
        "single": [("MSFT", TECH_PROFILE)],
        "same_profile": [("MSFT", TECH_PROFILE), ("NVDA", TECH_PROFILE)],
        "cross_profile": [("MSFT", TECH_PROFILE), ("PFE", BIO_PROFILE)],
    }[mode]
    results = {
        ticker: _result(run_id, ticker, profile_id)
        for ticker, profile_id in entries
    }
    eligible = mode == "same_profile"
    scores = (
        {
            "MSFT": {"total_score": 78.0, "rank": 1},
            "NVDA": {"total_score": 71.0, "rank": 2},
        }
        if eligible else None
    )
    return {
        "run_id": run_id,
        "original_query": f"Representative {mode} query.",
        "comparison_mode": mode,
        "normalized_results": results,
        "scoring_eligibility": {
            "eligible": eligible,
            "reason": "Eligible." if eligible else "Scoring not applicable.",
        },
        "scores": scores,
    }


class ScriptedSynthesisModel:
    """Return valid or citation-drifted F14 JSON from bounded payload messages."""

    def __init__(self, invalid_attempts: int = 0):
        """Configure how many initial drafts omit their inline citation."""
        self.invalid_attempts = invalid_attempts
        self.calls: list[list[object]] = []
        self.bind_tools_called = False

    def bind_tools(self, tools):
        """Fail if the F15 correction workflow attempts to expose any tools."""
        self.bind_tools_called = True
        raise AssertionError(f"F15 must not bind tools: {tools!r}")

    def invoke(self, messages):
        """Return one structured candidate based only on the serialized F14 payload."""
        self.calls.append(list(messages))
        payload = next(
            json.loads(message.content)
            for message in messages
            if '"available_evidence_ids"' in getattr(message, "content", "")
        )
        evidence_id = payload["available_evidence_ids"][0]
        valid = len(self.calls) > self.invalid_attempts
        return {
            "answer": (
                f"Grounded answer [{evidence_id}]."
                if valid else "Draft omitted its inline citation."
            ),
            "evidence_ids": [evidence_id],
            "limitations": payload["required_limitations"],
        }


@pytest.mark.parametrize("mode", ["single", "same_profile", "cross_profile"])
def test_representative_modes_validate_and_write_success_trace(mode, tmp_path):
    """Execute representative single, peer, and portfolio paths without live providers."""
    namespace = _namespace()
    model = ScriptedSynthesisModel()

    result = namespace["run_f15_validated_synthesis"](
        _context(mode), model, trace_dir=tmp_path,
    )

    assert result["final_status"] == "success"
    assert result["validation"]["valid"] is True
    assert result["attempts"] == 1
    assert result["correction_attempts"] == 0
    assert result["warnings"] == []
    assert model.bind_tools_called is False
    stored = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
    assert stored["comparison_mode"] == mode
    assert stored["final_status"] == "success"
    assert len(stored["validation_attempts"]) == 1


def test_failed_validation_returns_to_f14_with_feedback_once(tmp_path):
    """Correct citation/list drift using validation feedback and the same tool-free model."""
    namespace = _namespace()
    model = ScriptedSynthesisModel(invalid_attempts=1)

    result = namespace["run_f15_validated_synthesis"](
        _context("single"), model, trace_dir=tmp_path,
    )

    assert result["final_status"] == "success"
    assert result["attempts"] == 2
    assert result["correction_attempts"] == 1
    assert len(model.calls) == 2
    assert len(model.calls[1]) == 3
    feedback = json.loads(model.calls[1][-1].content)
    assert feedback["correction_attempt"] == 1
    assert any("exactly match" in error for error in feedback["deterministic_validation_errors"])
    stored = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
    assert [item["result"]["valid"] for item in stored["validation_attempts"]] == [
        False, True,
    ]


def test_retry_exhaustion_stops_after_two_corrections_and_warns(tmp_path):
    """Never loop beyond the initial draft plus the two permitted correction passes."""
    namespace = _namespace()
    model = ScriptedSynthesisModel(invalid_attempts=99)

    result = namespace["run_f15_validated_synthesis"](
        _context("cross_profile"), model, trace_dir=tmp_path,
    )

    assert result["final_status"] == "failed"
    assert result["attempts"] == 3
    assert result["correction_attempts"] == 2
    assert len(model.calls) == 3
    assert "Validation warning" in result["final_answer"]
    assert result["warnings"]
    stored = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
    assert stored["final_status"] == "failed"
    assert len(stored["validation_attempts"]) == 3
    assert all(not item["result"]["valid"] for item in stored["validation_attempts"])


@pytest.mark.parametrize("value", [-1, 3, True, 1.5])
def test_correction_budget_is_hard_capped(value, tmp_path):
    """Reject invalid or expanded retry budgets before model invocation."""
    namespace = _namespace()
    model = ScriptedSynthesisModel()
    with pytest.raises(ValueError, match="max_correction_attempts"):
        namespace["run_f15_validated_synthesis"](
            _context("single"),
            model,
            trace_dir=tmp_path,
            max_correction_attempts=value,
        )
    assert model.calls == []


def test_notebook_f15_cells_match_reconciled_sources_and_smoke_executes():
    """Keep notebook integration synchronized and run all three local example paths."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    assert cells["multiindustry_f15_evidence_validation"] == F15_EVIDENCE_CODE
    assert cells["multiindustry_f15_local_traces"] == F15_TRACES_CODE
    assert cells["multiindustry_f15_workflow"] == F15_WORKFLOW_CODE
    assert cells["multiindustry_f15_smoke"] == F15_SMOKE

    namespace = _namespace()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(F15_SMOKE, namespace)
    outputs = namespace["_f15_example_outputs"]
    assert set(outputs) == {"single", "same_profile", "cross_profile"}
    assert all(output["final_status"] == "success" for output in outputs.values())

