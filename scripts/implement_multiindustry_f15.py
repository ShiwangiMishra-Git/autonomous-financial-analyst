"""Reconcile and idempotently integrate the complete F15 notebook layer."""

from __future__ import annotations

from pathlib import Path

import nbformat

from scripts.implement_multiindustry_f15_evidence import (
    F15_EVIDENCE_CODE,
    F15_EVIDENCE_INTRO,
)
from scripts.implement_multiindustry_f15_traces import F15_TRACES_CODE, F15_TRACES_INTRO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f14_smoke"


F15_INTRO = """## Section 3.15: Deterministic Final Validation and Local Traces

F15 is the hard boundary between an F14 draft and a returned answer. It validates explicit
evidence provenance, immutable F13 score use, comparison-mode restrictions, and mandatory
limitations. A failed draft may return to the same tool-free F14 model for at most two correction
passes. Every attempt is recorded in one redacted local trace.

These checks are deliberately bounded. F15 performs no research, retrieval, normalization, or
score arithmetic, and citation provenance is not a semantic proof that every sentence is entailed
by its cited record.
"""


F15_WORKFLOW_CODE = r'''from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping, TypedDict

from langchain_core.messages import HumanMessage


class F15WorkflowResult(TypedDict):
    """Terminal result of bounded F14 synthesis, F15 validation, and trace persistence.

    Attributes:
        final_status: Success, failed, or interrupted workflow outcome.
        final_answer: Validated answer or visibly warned failed draft.
        synthesis: Last structured F14 candidate.
        validation: Deterministic evidence/score/mode/limitation verdict.
        attempts: Total synthesis attempts including the first draft.
        correction_attempts: Validation-driven correction count.
        warnings: Terminal warnings; empty on validated success.
        trace_path: Local redacted JSON trace path.
    """

    final_status: str
    final_answer: str
    synthesis: dict[str, Any]
    validation: ValidationResult
    attempts: int
    correction_attempts: int
    warnings: list[str]
    trace_path: str


F15_MAX_CORRECTION_ATTEMPTS = 2


def _f15_workflow_now(
    timestamp_provider: Callable[[], str | datetime] | None,
) -> str | datetime | None:
    """Return an injected timestamp or defer to the trace component's UTC clock."""
    return timestamp_provider() if timestamp_provider is not None else None


def _f15_failed_validation(error: Exception | str) -> ValidationResult:
    """Build a stable failed verdict when F14 cannot produce a structured candidate."""
    message = str(error).strip() or type(error).__name__
    return {
        "valid": False,
        "validated_evidence_ids": [],
        "inline_evidence_ids": [],
        "declared_evidence_ids": [],
        "evidence_valid": False,
        "score_fidelity_valid": False,
        "mode_restrictions_valid": False,
        "limitations_valid": False,
        "errors": [f"F14 synthesis failed: {message}"],
    }


class _F15CorrectionModel:
    """Append deterministic validation feedback to the next tool-free F14 invocation."""

    def __init__(
        self,
        model: Any,
        previous_synthesis: Mapping[str, Any] | None,
        validation_errors: list[str],
        correction_number: int,
    ):
        """Capture the underlying model and bounded correction context."""
        self.model = model
        self.previous_synthesis = deepcopy(dict(previous_synthesis or {}))
        self.validation_errors = list(validation_errors)
        self.correction_number = correction_number

    def invoke(self, messages: list[Any]) -> Any:
        """Invoke the model with the original F14 messages plus correction-only feedback."""
        feedback = {
            "correction_attempt": self.correction_number,
            "maximum_correction_attempts": F15_MAX_CORRECTION_ATTEMPTS,
            "previous_synthesis": self.previous_synthesis,
            "deterministic_validation_errors": self.validation_errors,
            "instructions": (
                "Correct only the structured F14 draft using the already supplied evidence and "
                "authoritative scores. Do not research, call tools, calculate scores, invent "
                "evidence IDs, or remove required limitations. Return JSON with answer, "
                "evidence_ids, and limitations."
            ),
        }
        return self.model.invoke([
            *messages,
            HumanMessage(content=json.dumps(feedback, indent=2, sort_keys=True, default=str)),
        ])


def _f15_set_trace_synthesis(
    trace: Mapping[str, Any],
    synthesis: Mapping[str, Any] | None,
) -> ResearchTrace:
    """Return a copied trace containing the latest redacted F14 candidate."""
    updated = deepcopy(dict(trace))
    updated["f14_synthesis"] = redact_trace_value(deepcopy(dict(synthesis or {})))
    return updated


def run_f15_validated_synthesis(
    context: SynthesisContext | Mapping[str, Any],
    injected_model: Any,
    *,
    trace_dir: str | Path = ".research_runs",
    retention_limit: int = F15_TRACE_DEFAULT_RETENTION,
    max_correction_attempts: int = F15_MAX_CORRECTION_ATTEMPTS,
    timestamp_provider: Callable[[], str | datetime] | None = None,
) -> F15WorkflowResult:
    """Run F14, enforce F15, persist every attempt, and return a bounded terminal result.

    The initial F14 call is followed by at most two correction calls. A validation failure never
    becomes a successful answer. When retries are exhausted, the last draft is returned only with
    an explicit warning and ``final_status='failed'``.
    """
    if (
        isinstance(max_correction_attempts, bool)
        or not isinstance(max_correction_attempts, int)
        or not 0 <= max_correction_attempts <= F15_MAX_CORRECTION_ATTEMPTS
    ):
        raise ValueError(
            f"max_correction_attempts must be between 0 and {F15_MAX_CORRECTION_ATTEMPTS}"
        )
    safe_context = _f14_validate_context(context)
    trace: ResearchTrace | None = None
    trace_path = ""
    last_synthesis: dict[str, Any] = {}
    last_validation = _f15_failed_validation("No synthesis attempt completed")
    feedback_errors: list[str] = []
    total_attempts = 1 + max_correction_attempts

    try:
        for attempt_number in range(1, total_attempts + 1):
            correction_number = attempt_number - 1
            attempt_model = (
                injected_model
                if correction_number == 0
                else _F15CorrectionModel(
                    injected_model,
                    last_synthesis,
                    feedback_errors,
                    correction_number,
                )
            )
            try:
                candidate = synthesize_answer(context, attempt_model)
                last_synthesis = deepcopy(dict(candidate))
                last_validation = validate_synthesis_result(
                    safe_context["run_id"],
                    safe_context["normalized_results"],
                    last_synthesis,
                    authoritative_scores=safe_context["scores"],
                    scoring_eligibility=safe_context["scoring_eligibility"],
                    required_limitations=safe_context["required_limitations"],
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                last_validation = _f15_failed_validation(exc)

            if trace is None:
                trace = create_research_trace(
                    run_id=safe_context["run_id"],
                    query=safe_context["original_query"],
                    comparison_mode=safe_context["comparison_mode"],
                    normalized_results=safe_context["normalized_results"],
                    f13_scores=safe_context["scores"],
                    f14_synthesis=last_synthesis,
                    timestamp=_f15_workflow_now(timestamp_provider),
                )
            else:
                trace = _f15_set_trace_synthesis(trace, last_synthesis)
            trace = record_validation_attempt(
                trace,
                last_validation,
                attempt_number=attempt_number,
                timestamp=_f15_workflow_now(timestamp_provider),
            )
            trace_path = write_research_trace(
                trace,
                trace_dir=trace_dir,
                retention_limit=retention_limit,
            )["path"]

            if last_validation["valid"]:
                trace = finalize_research_trace(
                    trace,
                    final_status="success",
                    timestamp=_f15_workflow_now(timestamp_provider),
                )
                trace_path = write_research_trace(
                    trace,
                    trace_dir=trace_dir,
                    retention_limit=retention_limit,
                )["path"]
                return {
                    "final_status": "success",
                    "final_answer": last_synthesis["answer"],
                    "synthesis": deepcopy(last_synthesis),
                    "validation": deepcopy(last_validation),
                    "attempts": attempt_number,
                    "correction_attempts": correction_number,
                    "warnings": [],
                    "trace_path": trace_path,
                }
            feedback_errors = list(last_validation["errors"])

        warning = "F15 validation failed after bounded correction attempts: " + "; ".join(
            last_validation["errors"]
        )
        trace = finalize_research_trace(
            trace,
            final_status="failed",
            terminal_error=warning,
            timestamp=_f15_workflow_now(timestamp_provider),
        )
        trace_path = write_research_trace(
            trace,
            trace_dir=trace_dir,
            retention_limit=retention_limit,
        )["path"]
        draft = last_synthesis.get("answer") or "No valid synthesis draft was produced."
        return {
            "final_status": "failed",
            "final_answer": f"{draft}\n\nValidation warning: {warning}",
            "synthesis": deepcopy(last_synthesis),
            "validation": deepcopy(last_validation),
            "attempts": total_attempts,
            "correction_attempts": max_correction_attempts,
            "warnings": [warning],
            "trace_path": trace_path,
        }
    except KeyboardInterrupt:
        if trace is None:
            trace = create_research_trace(
                run_id=safe_context["run_id"],
                query=safe_context["original_query"],
                comparison_mode=safe_context["comparison_mode"],
                normalized_results=safe_context["normalized_results"],
                f13_scores=safe_context["scores"],
                f14_synthesis=last_synthesis,
                timestamp=_f15_workflow_now(timestamp_provider),
            )
        trace = finalize_research_trace(
            trace,
            final_status="interrupted",
            terminal_error="Notebook synthesis was interrupted.",
            timestamp=_f15_workflow_now(timestamp_provider),
        )
        write_research_trace(
            trace,
            trace_dir=trace_dir,
            retention_limit=retention_limit,
        )
        raise


print("✅ F15 bounded validation, correction, and trace workflow defined")
'''


F15_SMOKE = r'''# F15 representative local examples; no provider, research, or score call.
import json as _f15_smoke_json
import tempfile as _f15_smoke_tempfile


class _F15SmokeModel:
    """Return grounded JSON and optionally force one citation-correction attempt."""

    def __init__(self, *, fail_first: bool = False):
        """Configure deterministic first-attempt behavior and call tracking."""
        self.fail_first = fail_first
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        """Read bounded F14 payload and return a deterministic structured candidate."""
        self.calls.append(list(messages))
        payload = next(
            _f15_smoke_json.loads(message.content)
            for message in messages
            if isinstance(message, HumanMessage)
            and '"available_evidence_ids"' in message.content
        )
        evidence_id = payload["available_evidence_ids"][0]
        omit_inline = self.fail_first and len(self.calls) == 1
        answer = (
            "Grounded representative answer."
            if omit_inline
            else f"Grounded representative answer [{evidence_id}]."
        )
        return {
            "answer": answer,
            "evidence_ids": [evidence_id],
            "limitations": payload["required_limitations"],
        }


def _f15_smoke_company(ticker: str, profile_id: str) -> dict[str, Any]:
    """Build one resolved canonical company for representative F15 examples."""
    company_id = {"MSFT": "microsoft", "NVDA": "nvidia", "PFE": "pfizer"}[ticker]
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_id.title(),
        "profile_id": profile_id,
        "resolution_status": "resolved",
    }


def _f15_smoke_result(run_id: str, ticker: str, profile_id: str) -> dict[str, Any]:
    """Build one normalized result with a canonical current-run evidence record."""
    company = _f15_smoke_company(ticker, profile_id)
    evidence_id = f"EV-{run_id}-{ticker}"
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [{
            "evidence_id": evidence_id,
            "run_id": run_id,
            "company_id": company["company_id"],
            "ticker": ticker,
            "profile_id": profile_id,
            "evidence_type": "representative_fixture",
            "source_name": "local-fixture",
            "status": "success",
        }],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


_f15_example_specs = {
    "single": [("MSFT", "technology.ai.v1")],
    "same_profile": [
        ("MSFT", "technology.ai.v1"),
        ("NVDA", "technology.ai.v1"),
    ],
    "cross_profile": [
        ("MSFT", "technology.ai.v1"),
        ("PFE", "healthcare.biopharma.v1"),
    ],
}
_f15_example_outputs: dict[str, F15WorkflowResult] = {}
with _f15_smoke_tempfile.TemporaryDirectory(prefix="f15-smoke-") as _f15_trace_dir:
    for _f15_mode, _f15_companies in _f15_example_specs.items():
        _f15_run_id = f"f15-{_f15_mode.replace('_', '-')}"
        _f15_results = {
            ticker: _f15_smoke_result(_f15_run_id, ticker, profile_id)
            for ticker, profile_id in _f15_companies
        }
        _f15_eligible = _f15_mode == "same_profile"
        _f15_scores = (
            {
                "MSFT": {"total_score": 78.0, "rank": 1},
                "NVDA": {"total_score": 71.0, "rank": 2},
            }
            if _f15_eligible else None
        )
        _f15_context = {
            "run_id": _f15_run_id,
            "original_query": f"Representative {_f15_mode} question.",
            "comparison_mode": _f15_mode,
            "normalized_results": _f15_results,
            "scoring_eligibility": {
                "eligible": _f15_eligible,
                "reason": "Eligible." if _f15_eligible else "Scoring not applicable.",
            },
            "scores": _f15_scores,
        }
        _f15_model = _F15SmokeModel(fail_first=_f15_mode == "single")
        _f15_output = run_f15_validated_synthesis(
            _f15_context,
            _f15_model,
            trace_dir=_f15_trace_dir,
        )
        assert _f15_output["final_status"] == "success"
        assert Path(_f15_output["trace_path"]).exists()
        _f15_example_outputs[_f15_mode] = _f15_output

assert _f15_example_outputs["single"]["attempts"] == 2
assert _f15_example_outputs["same_profile"]["attempts"] == 1
assert _f15_example_outputs["cross_profile"]["attempts"] == 1
print("✅ F15 representative single, same-profile, and cross-profile examples passed")
'''


CELL_SPECS = [
    ("multiindustry_f15_intro", "markdown", F15_INTRO),
    ("multiindustry_f15_evidence_intro", "markdown", F15_EVIDENCE_INTRO),
    ("multiindustry_f15_evidence_validation", "code", F15_EVIDENCE_CODE),
    ("multiindustry_f15_traces_intro", "markdown", F15_TRACES_INTRO),
    ("multiindustry_f15_local_traces", "code", F15_TRACES_CODE),
    ("multiindustry_f15_workflow", "code", F15_WORKFLOW_CODE),
    ("multiindustry_f15_smoke", "code", F15_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create one notebook cell with a stable identifier."""
    cell = (
        nbformat.v4.new_markdown_cell(source=source)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(source=source)
    )
    cell["id"] = cell_id
    return cell


def integrate_f15_cells(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh reconciled F15 cells after F14 without duplication."""
    notebook = nbformat.read(notebook_path, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    for cell_id, cell_type, source in CELL_SPECS:
        existing = cells_by_id.get(cell_id)
        if existing is not None:
            existing["cell_type"] = cell_type
            existing["source"] = source
            if cell_type == "code":
                existing["execution_count"] = None
                existing["outputs"] = []
    missing = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing:
        try:
            insertion_index = next(
                index for index, cell in enumerate(notebook.cells)
                if cell.get("id") == INSERT_AFTER_CELL_ID
            ) + 1
        except StopIteration as exc:
            raise ValueError(
                f"Notebook is missing insertion anchor {INSERT_AFTER_CELL_ID!r}"
            ) from exc
        notebook.cells[insertion_index:insertion_index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)
