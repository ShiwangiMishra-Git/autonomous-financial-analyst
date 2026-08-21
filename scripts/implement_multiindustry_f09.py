"""Idempotently add F09 biopharma signals and scoring gate to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f08_smoke"


F09_INTRO = """## Section 3.9: Biopharma Signal Extraction and Rubric Gate

F09 maps validated official-source evidence into five stable biopharma dimensions. The LLM may
interpret evidence into a structured draft, but deterministic normalization verifies profile,
company, level, score, and evidence IDs. Any ungrounded non-missing signal is downgraded to
`missing` rather than accepted.

F13 consumes these evidence-linked signals through the notebook-local
`healthcare.biopharma.score.v1` baseline. The score is a transparent research-strength measure,
not a Buy/Hold/Sell recommendation. Missing or ungrounded signals remain ineligible.
"""


F09_CODE = r'''from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


BIOPHARMA_SIGNAL_NAMES = [
    "clinical_pipeline",
    "regulatory_progress",
    "exclusivity_and_patents",
    "commercialization",
    "sector_risks",
]
BIOPHARMA_SIGNAL_LEVEL_SCORES = {
    "none": 0.0,
    "partial": 0.5,
    "full": 1.0,
    "missing": None,
}
BIOPHARMA_SCORING_ENABLED = True
BIOPHARMA_SCORING_RUBRIC_ID = "healthcare.biopharma.score.v1"

PHARMA_SIGNAL_RUBRIC: dict[str, dict[str, str]] = {
    "clinical_pipeline": {
        "none": "No material supported development pipeline evidence.",
        "partial": "Some supported clinical programs or progress, with meaningful gaps or concentration.",
        "full": "Broad, well-supported pipeline with material late-stage or diversified programs.",
        "missing": "Insufficient official-source evidence to classify the pipeline.",
    },
    "regulatory_progress": {
        "none": "No supported regulatory progress and material setbacks may dominate.",
        "partial": "Some submissions, designations, approvals, or mixed regulatory outcomes.",
        "full": "Multiple material, well-supported approvals or advanced regulatory milestones.",
        "missing": "Insufficient official-source regulatory evidence.",
    },
    "exclusivity_and_patents": {
        "none": "Material loss-of-exclusivity exposure without supported mitigation.",
        "partial": "Mixed patent protection, concentration, or partially supported mitigation.",
        "full": "Strong supported exclusivity position with credible lifecycle protection.",
        "missing": "Insufficient official-source patent or exclusivity evidence.",
    },
    "commercialization": {
        "none": "No supported commercialization strength for material products.",
        "partial": "Some supported launches or sales execution with meaningful uncertainty.",
        "full": "Strong supported launch execution, market access, or product growth.",
        "missing": "Insufficient official-source commercialization evidence.",
    },
    "sector_risks": {
        "none": "No material sector-specific risk is evidenced in the reviewed scope.",
        "partial": "Manageable or mixed clinical, regulatory, pricing, safety, or concentration risks.",
        "full": "Multiple material sector risks are clearly supported by official evidence.",
        "missing": "Insufficient official-source evidence to classify sector risks.",
    },
}


def validate_pharma_signal_rubric() -> list[str]:
    """Return deterministic coverage errors for the biopharma signal rubric."""
    errors: list[str] = []
    required_levels = set(BIOPHARMA_SIGNAL_LEVEL_SCORES)
    if set(PHARMA_SIGNAL_RUBRIC) != set(BIOPHARMA_SIGNAL_NAMES):
        errors.append("Rubric dimensions do not match BIOPHARMA_SIGNAL_NAMES")
    for dimension in BIOPHARMA_SIGNAL_NAMES:
        definitions = PHARMA_SIGNAL_RUBRIC.get(dimension, {})
        if set(definitions) != required_levels:
            errors.append(f"{dimension} does not define every signal level")
        if any(not str(description).strip() for description in definitions.values()):
            errors.append(f"{dimension} contains an empty rubric description")
    return errors


def _validated_biopharma_evidence_ids(
    company: ResolvedCompany,
    records: list[EvidenceRecord],
) -> list[str]:
    """Validate biopharma evidence identity and return successful evidence IDs."""
    evidence_ids: list[str] = []
    for record in records:
        if record["company_id"] != company["company_id"] or record["ticker"] != company["ticker"]:
            raise ValueError(f"Biopharma evidence identity mismatch for {company['ticker']}")
        if record["profile_id"] != BIOPHARMA_PROFILE_ID:
            raise ValueError("Biopharma signals cannot use technology evidence")
        if record["status"] == "success":
            evidence_ids.append(record["evidence_id"])
    return evidence_ids


def _pharma_extraction_prompt(
    companies: list[ResolvedCompany],
    evidence_by_company: dict[str, list[EvidenceRecord]],
) -> str:
    """Build a bounded JSON extraction prompt containing evidence IDs and official text."""
    evidence_payload = {}
    for company in companies:
        evidence_payload[company["ticker"]] = [
            {"evidence_id": record["evidence_id"], "value": record["value"]}
            for record in evidence_by_company.get(company["company_id"], [])
            if record["status"] == "success"
        ]
    return json.dumps({
        "rubric": PHARMA_SIGNAL_RUBRIC,
        "required_dimensions": BIOPHARMA_SIGNAL_NAMES,
        "companies_and_evidence": evidence_payload,
        "output_contract": {
            "TICKER": {
                "dimension": {
                    "level": "none|partial|full|missing",
                    "reason": "brief evidence-grounded explanation",
                    "evidence_ids": ["current-run evidence IDs"],
                }
            }
        },
    }, indent=2, default=str)


def _default_pharma_extractor_model() -> Any:
    """Create the deterministic-temperature model used for structured pharma interpretation."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE"),
    )


def extract_pharma_signals(
    companies: list[ResolvedCompany],
    evidence_by_company: dict[str, list[EvidenceRecord]],
    raw_signals: dict[str, dict[str, Any]] | None = None,
    model: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract and deterministically normalize five evidence-linked biopharma signals.

    Args:
        companies: Resolved biopharma companies.
        evidence_by_company: Current-run canonical evidence keyed by company ID.
        raw_signals: Optional structured draft for deterministic tests.
        model: Optional injected chat model when a draft is not supplied.

    Returns:
        Ticker-keyed signal mappings with level, score, reason, and evidence IDs.

    Raises:
        ValueError: If company or evidence identity crosses profile boundaries.
    """
    for company in companies:
        if company["profile_id"] != BIOPHARMA_PROFILE_ID:
            raise ValueError("Pharma extractor received a non-biopharma company")
    if raw_signals is None:
        extractor_model = model or _default_pharma_extractor_model()
        response = extractor_model.invoke([
            SystemMessage(content=(
                "Classify only from supplied official evidence. Return raw JSON matching the "
                "contract. Never invent evidence IDs or infer missing facts."
            )),
            HumanMessage(content=_pharma_extraction_prompt(companies, evidence_by_company)),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        raw_signals = json.loads(cleaned)

    normalized: dict[str, dict[str, Any]] = {}
    for company in companies:
        records = evidence_by_company.get(company["company_id"], [])
        available_ids = _validated_biopharma_evidence_ids(company, records)
        available_set = set(available_ids)
        company_raw = raw_signals.get(company["ticker"], {})
        normalized[company["ticker"]] = {}
        for dimension in BIOPHARMA_SIGNAL_NAMES:
            raw = company_raw.get(dimension)
            if not isinstance(raw, dict):
                normalized[company["ticker"]][dimension] = {
                    "level": "missing", "score": None,
                    "reason": PHARMA_SIGNAL_RUBRIC[dimension]["missing"], "evidence_ids": [],
                }
                continue
            level = str(raw.get("level", "missing")).casefold()
            if level not in BIOPHARMA_SIGNAL_LEVEL_SCORES:
                level = "missing"
            requested_ids = raw.get("evidence_ids", [])
            evidence_ids = [item for item in requested_ids if item in available_set]
            if level != "missing" and not evidence_ids:
                normalized[company["ticker"]][dimension] = {
                    "level": "missing", "score": None,
                    "reason": "Signal rejected because it lacks valid current-run evidence IDs.",
                    "evidence_ids": [],
                }
                continue
            normalized[company["ticker"]][dimension] = {
                "level": level,
                "score": BIOPHARMA_SIGNAL_LEVEL_SCORES[level],
                "reason": str(raw.get("reason") or PHARMA_SIGNAL_RUBRIC[dimension][level]),
                "evidence_ids": evidence_ids,
            }
    return normalized


def check_biopharma_scoring_gate() -> ScoringEligibility:
    """Return the configuration gate for the validated notebook-local baseline rubric."""
    return {
        "eligible": BIOPHARMA_SCORING_ENABLED,
        "rubric_id": BIOPHARMA_SCORING_RUBRIC_ID,
        "reason": (
            "The notebook-local biopharma research-strength rubric has fixed weights, strict "
            "missing-data rules, deterministic bands, and calibration fixtures. Per-run F12 "
            "eligibility must still pass before scoring."
        ),
        "excluded_companies": [],
        "missing_requirements": {},
    }


_pharma_rubric_errors = validate_pharma_signal_rubric()
if _pharma_rubric_errors:
    raise ValueError("Invalid PHARMA_SIGNAL_RUBRIC: " + "; ".join(_pharma_rubric_errors))

print("✅ F09 biopharma signals defined; notebook-local scoring contract is available")
'''


F09_SMOKE = r'''# F09 local smoke test with injected evidence and structured output.
_f09_company = resolve_company_mention("Pfizer")
_f09_plan: QueryPlan = {
    "query_type": "analyze", "company_mentions": ["Pfizer"],
    "requested_dimensions": ["pipeline"], "risk_profile": "balanced",
    "scoring_requested": False, "freshness_required": False, "time_horizon": None,
}
_f09_task = build_company_tasks(_f09_plan, [_f09_company], "f09-smoke-run")[0]
_f09_evidence = query_biopharma_rag_evidence(
    _f09_task, "pipeline", {
        "status": "success", "ticker": "PFE", "data": [{
            "data": "Pfizer reports multiple clinical programs.", "ticker": "PFE",
            "document_name": "PFE.pdf", "page": 3,
        }],
    },
)
_f09_evidence_id = _f09_evidence[0]["evidence_id"]
_f09_raw = {"PFE": {
    name: {"level": "partial", "reason": "Supported.", "evidence_ids": [_f09_evidence_id]}
    for name in BIOPHARMA_SIGNAL_NAMES
}}
_f09_signals = extract_pharma_signals(
    [_f09_company], {"pfizer": _f09_evidence}, raw_signals=_f09_raw,
)
assert set(_f09_signals["PFE"]) == set(BIOPHARMA_SIGNAL_NAMES)
assert all(signal["evidence_ids"] == [_f09_evidence_id] for signal in _f09_signals["PFE"].values())
assert check_biopharma_scoring_gate()["eligible"] is True
assert check_biopharma_scoring_gate()["rubric_id"] == "healthcare.biopharma.score.v1"

print("✅ F09 smoke test passed: evidence-linked pharma signals and rubric configuration")
'''


CELL_SPECS = [
    ("multiindustry_f09_intro", "markdown", F09_INTRO),
    ("multiindustry_biopharma_signals", "code", F09_CODE),
    ("multiindustry_f09_smoke", "code", F09_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable identifier."""
    cell = nbformat.v4.new_markdown_cell(source=source) if cell_type == "markdown" else nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh F09 cells in the working notebook."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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
        index = next(i for i, cell in enumerate(notebook.cells) if cell.get("id") == INSERT_AFTER_CELL_ID) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F09 cells are present")


if __name__ == "__main__":
    main()
