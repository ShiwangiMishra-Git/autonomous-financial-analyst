"""Idempotently add the F14 grounded synthesis layer to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f13_smoke"


F14_INTRO = """## Section 3.14: Mode-Specific Grounded Synthesis

F14 is the answer-writing layer. It performs no research, normalization, or scoring arithmetic.
It receives F12-normalized current-run results plus an optional authoritative F13 score table and
selects one of three policies: `single`, `same_profile`, or `cross_profile`.

The model receives serialized evidence rather than research tools. Deterministic preparation and
result checks enforce the selected mode, current-run evidence IDs, immutable score-table use, and
required limitations. Cross-profile answers keep sector conclusions separate and cannot claim a
universal score; same-profile answers may explain but never recompute F13 scores.
"""


F14_CODE = r'''from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable, Mapping, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage


class SynthesisContext(TypedDict):
    """Validated inputs available to one mode-specific answer-writing call.

    Attributes:
        run_id: Current research run used to enforce evidence ownership.
        original_query: User question the answer must address.
        comparison_mode: F12-selected single/same-profile/cross-profile policy.
        normalized_results: Current-run F12 evidence and signals keyed by ticker.
        scoring_eligibility: Deterministic F12 scoring authorization.
        scores: Optional immutable authoritative F13 score table.
    """

    run_id: str
    original_query: str
    comparison_mode: ComparisonMode
    normalized_results: CompanyResultMap
    scoring_eligibility: ScoringEligibility
    scores: dict[str, Any] | None


class SynthesisResult(TypedDict):
    """Structured draft returned by the F14 synthesis boundary.

    Attributes:
        mode: Comparison mode actually used by the prompt policy.
        answer: Candidate grounded prose containing explicit evidence citations.
        evidence_ids: Declared current-run citations used by the answer.
        scores_used: Exact copied F13 scores for eligible same-profile synthesis.
        limitations: Required and model-added limitations disclosed to the user.
    """

    mode: ComparisonMode
    answer: str
    evidence_ids: list[str]
    scores_used: dict[str, Any]
    limitations: list[str]


F14_VALID_MODES = frozenset({"single", "same_profile", "cross_profile"})


def _f14_text(value: Any, field_name: str) -> str:
    """Return a stripped non-empty string or raise a deterministic validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _f14_unique_text(values: Any, field_name: str) -> list[str]:
    """Normalize a list of non-empty strings while preserving first-seen order."""
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    output: list[str] = []
    for value in values:
        text = _f14_text(value, field_name)
        if text not in output:
            output.append(text)
    return output


def _f14_required_limitations(
    results: Mapping[str, CompanyResearchResult],
    mode: ComparisonMode,
    eligibility: Mapping[str, Any],
    scores: Mapping[str, Any] | None,
) -> list[str]:
    """Derive limitations that the model is not permitted to omit."""
    limitations: list[str] = []
    for ticker, result in results.items():
        status = result.get("status")
        if status != "success":
            limitations.append(f"{ticker}: result status is {status}.")
        missing = result.get("missing_dimensions", [])
        if missing:
            limitations.append(f"{ticker}: missing dimensions: {', '.join(map(str, missing))}.")
        errors = result.get("errors", [])
        if errors:
            limitations.append(f"{ticker}: research limitations: {'; '.join(map(str, errors))}.")
    if mode == "single":
        limitations.append("Single-company analysis does not use a comparison score.")
    elif mode == "cross_profile":
        limitations.append("No universal numeric score was applied across industry profiles.")
    elif not eligibility.get("eligible"):
        reason = str(eligibility.get("reason") or "Scoring eligibility did not pass.")
        limitations.append(f"Numeric sector scoring was not applied: {reason}")
    elif not scores:
        limitations.append("Scoring was eligible, but no authoritative F13 score table was supplied.")
    return list(dict.fromkeys(limitations))


def _f14_validate_context(context: SynthesisContext | Mapping[str, Any]) -> dict[str, Any]:
    """Validate run, mode, evidence, and score boundaries and return a defensive copy."""
    if not isinstance(context, Mapping):
        raise ValueError("Synthesis context must be a mapping")
    run_id = _f14_text(context.get("run_id"), "run_id")
    query = _f14_text(context.get("original_query"), "original_query")
    mode = context.get("comparison_mode")
    if mode not in F14_VALID_MODES:
        raise ValueError(f"Unsupported synthesis mode: {mode!r}")
    results = context.get("normalized_results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError("normalized_results must be a non-empty ticker mapping")

    copied_results: dict[str, CompanyResearchResult] = deepcopy(dict(results))
    selected_mode = select_comparison_mode(copied_results)
    if selected_mode != mode:
        raise ValueError(
            f"Synthesis mode {mode!r} does not match normalized results mode {selected_mode!r}"
        )
    available_ids: list[str] = []
    for ticker, result in copied_results.items():
        if not isinstance(result, Mapping):
            raise ValueError(f"Result {ticker} must be a mapping")
        if result.get("run_id") != run_id:
            raise ValueError(f"Result {ticker} does not belong to current run {run_id}")
        company = result.get("company")
        if not isinstance(company, Mapping) or company.get("ticker") != ticker:
            raise ValueError(f"Result {ticker} has invalid canonical company identity")
        records = result.get("evidence")
        if not isinstance(records, list):
            raise ValueError(f"Result {ticker} evidence must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Result {ticker} contains malformed evidence")
            evidence_id = _f14_text(record.get("evidence_id"), f"{ticker}.evidence_id")
            if record.get("run_id") != run_id or record.get("ticker") != ticker:
                raise ValueError(f"Evidence {evidence_id} crossed the run or company boundary")
            if record.get("profile_id") != result.get("profile_id"):
                raise ValueError(f"Evidence {evidence_id} crossed the profile boundary")
            if record.get("status") == "success" and evidence_id not in available_ids:
                available_ids.append(evidence_id)

    eligibility = context.get("scoring_eligibility")
    if not isinstance(eligibility, Mapping):
        raise ValueError("scoring_eligibility must be a mapping")
    copied_eligibility = deepcopy(dict(eligibility))
    raw_scores = context.get("scores")
    if raw_scores is not None and not isinstance(raw_scores, Mapping):
        raise ValueError("scores must be a ticker mapping or None")
    copied_scores = deepcopy(dict(raw_scores or {}))
    if mode in {"single", "cross_profile"} and copied_scores:
        raise ValueError(f"{mode} synthesis cannot receive a sector comparison score")
    if copied_scores:
        if mode != "same_profile" or copied_eligibility.get("eligible") is not True:
            raise ValueError("Authoritative scores require eligible same_profile synthesis")
        if set(copied_scores) != set(copied_results):
            raise ValueError("Authoritative score table must cover the normalized company set")

    limitations = _f14_required_limitations(
        copied_results, mode, copied_eligibility, copied_scores,
    )
    return {
        "run_id": run_id,
        "original_query": query,
        "comparison_mode": mode,
        "normalized_results": copied_results,
        "scoring_eligibility": copied_eligibility,
        "scores": copied_scores,
        "available_evidence_ids": available_ids,
        "required_limitations": limitations,
    }


def build_single_prompt(context: Mapping[str, Any]) -> str:
    """Build the one-company synthesis policy without comparison-score authority."""
    return (
        "You are the single-company synthesis agent. Analyze exactly the supplied company using "
        "only current-run normalized evidence. Cover shared financial and profile-specific "
        "findings, cite supplied evidence IDs, and disclose every supplied limitation. Do not "
        "compare against absent companies and do not create or mention a comparison score. "
        "Return JSON with answer, evidence_ids, and limitations."
    )


def build_same_profile_prompt(context: Mapping[str, Any]) -> str:
    """Build the like-for-like sector synthesis policy with immutable optional scores."""
    return (
        "You are the same-profile synthesis agent. Compare the supplied companies like-for-like "
        "within their shared exact industry profile using only current-run normalized evidence. "
        "Cite supplied evidence IDs and disclose every supplied limitation. If authoritative F13 "
        "scores are supplied, explain them exactly as given: never recalculate, alter, reorder, "
        "or replace them. Return JSON with answer, evidence_ids, and limitations."
    )


def build_cross_profile_prompt(context: Mapping[str, Any]) -> str:
    """Build the qualitative portfolio policy without a universal sector score."""
    return (
        "You are the cross-profile portfolio synthesis agent. Compare only shared financial "
        "dimensions, keep each industry's profile-specific conclusions in separate sections, and "
        "interpret evidence in its sector context. Use only current-run normalized evidence, cite "
        "supplied evidence IDs, and disclose every supplied limitation. State explicitly that no "
        "universal numeric score was applied. Never apply one sector rubric to another profile. "
        "Return JSON with answer, evidence_ids, and limitations."
    )


def create_synthesizer(
    mode: ComparisonMode,
    profile: IndustryProfile | None = None,
) -> Callable[[Mapping[str, Any]], str]:
    """Select one prompt builder without binding any research or scoring tools."""
    builders: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "single": build_single_prompt,
        "same_profile": build_same_profile_prompt,
        "cross_profile": build_cross_profile_prompt,
    }
    if mode not in builders:
        raise ValueError(f"Unsupported synthesis mode: {mode!r}")
    if profile is not None and not isinstance(profile, Mapping):
        raise ValueError("profile must be an IndustryProfile mapping or None")
    return builders[mode]


def _f14_prompt_payload(context: Mapping[str, Any]) -> str:
    """Serialize the bounded synthesis inputs; omit score material outside same-profile mode."""
    payload = {
        "run_id": context["run_id"],
        "question": context["original_query"],
        "mode": context["comparison_mode"],
        "normalized_results": context["normalized_results"],
        "available_evidence_ids": context["available_evidence_ids"],
        "required_limitations": context["required_limitations"],
    }
    if context["comparison_mode"] == "same_profile" and context["scores"]:
        payload["authoritative_f13_scores"] = context["scores"]
        payload["scoring_eligibility"] = context["scoring_eligibility"]
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _f14_parse_response(response: Any) -> dict[str, Any]:
    """Parse a mapping or JSON message returned by an injected synthesis model."""
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Synthesis model must return valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Synthesis model response must be a JSON object")
    return deepcopy(dict(raw))


def synthesize_answer(
    context: SynthesisContext | Mapping[str, Any],
    injected_model: Any,
) -> SynthesisResult:
    """Generate and constrain one grounded mode-specific draft without research tools."""
    safe = _f14_validate_context(context)
    builder = create_synthesizer(safe["comparison_mode"])
    messages = [
        SystemMessage(content=builder(safe)),
        HumanMessage(content=_f14_prompt_payload(safe)),
    ]
    response = injected_model.invoke(messages)
    parsed = _f14_parse_response(response)
    answer = _f14_text(parsed.get("answer"), "answer")
    evidence_ids = _f14_unique_text(parsed.get("evidence_ids", []), "evidence_ids")
    invalid_ids = sorted(set(evidence_ids) - set(safe["available_evidence_ids"]))
    if invalid_ids:
        raise ValueError(f"Synthesis cited unavailable evidence IDs: {invalid_ids}")
    if safe["available_evidence_ids"] and not evidence_ids:
        raise ValueError("Synthesis must cite at least one supplied evidence ID")
    model_limitations = _f14_unique_text(parsed.get("limitations", []), "limitations")
    limitations = list(dict.fromkeys(safe["required_limitations"] + model_limitations))
    scores_used = deepcopy(safe["scores"]) if safe["comparison_mode"] == "same_profile" else {}
    return {
        "mode": safe["comparison_mode"],
        "answer": answer,
        "evidence_ids": evidence_ids,
        "scores_used": scores_used,
        "limitations": limitations,
    }


print("✅ F14 mode-specific grounded synthesis defined")
'''


F14_SMOKE = r'''# F14 local smoke uses F13 fixture data and an injected model; no provider call.
class _F14SmokeModel:
    """Return one deterministic grounded synthesis payload and retain received messages."""

    def __init__(self, evidence_id: str):
        """Store the evidence ID cited by the deterministic response."""
        self.evidence_id = evidence_id
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        """Record prompt messages and return a deliberately score-free model payload."""
        self.messages = list(messages)
        return {
            "answer": f"The comparison is grounded in [{self.evidence_id}].",
            "evidence_ids": [self.evidence_id],
            "limitations": [],
            "scores_used": {"tampered": True},
        }


_f14_evidence_id = _f13_smoke_results["MSFT"]["evidence"][0]["evidence_id"]
_f14_model = _F14SmokeModel(_f14_evidence_id)
_f14_context: SynthesisContext = {
    "run_id": "f13-smoke-run",
    "original_query": "Compare Microsoft and Google as technology investments.",
    "comparison_mode": "same_profile",
    "normalized_results": _f13_smoke_results,
    "scoring_eligibility": _f13_smoke_eligibility,
    "scores": _f13_scores_1,
}
_f14_result = synthesize_answer(_f14_context, _f14_model)
assert _f14_result["mode"] == "same_profile"
assert _f14_result["scores_used"] == _f13_scores_1
assert _f14_result["scores_used"] != {"tampered": True}
assert _f14_result["evidence_ids"] == [_f14_evidence_id]
assert len(_f14_model.messages) == 2

print("✅ F14 smoke passed: same-profile synthesis used immutable scores and supplied evidence")
'''


CELL_SPECS = [
    ("multiindustry_f14_intro", "markdown", F14_INTRO),
    ("multiindustry_mode_specific_synthesis", "code", F14_CODE),
    ("multiindustry_f14_smoke", "code", F14_SMOKE),
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


def integrate_f14_cells(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh F14 immediately after F13, without duplicating cells."""
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
            raise ValueError(f"Notebook is missing insertion anchor {INSERT_AFTER_CELL_ID!r}") from exc
        notebook.cells[insertion_index:insertion_index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)
