"""Idempotently add the F03 structured query planner to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f02_smoke"


F03_INTRO = """## Section 3.3: Structured Free-Text Query Planner

The planner converts the user's free-text question into a bounded `QueryPlan`. It may interpret
intent and company mentions, but it cannot establish canonical company identity, select industry
profiles, choose tools, or decide scoring eligibility. Those decisions remain deterministic.

The live notebook uses the configured chat model with structured output. Tests inject a fake
structured model, so planner behavior can be verified without network calls or API keys.
"""


F03_CODE = r'''from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


QUERY_TYPES = {"fact", "analyze", "compare", "rank"}
RISK_PROFILES = {"conservative", "balanced", "growth"}


QUERY_PLANNER_PROMPT = """You are a bounded query planner for a local multi-industry financial
research notebook.

Convert the user's question into the supplied QueryPlan schema.

Rules:
1. query_type must be exactly one of: fact, analyze, compare, rank.
2. Return one company_mentions item per company. Never combine multiple companies into one item.
3. Preserve names or tickers from the question. Do not invent a ticker or company.
4. requested_dimensions must contain short snake_case research dimensions requested by the user.
5. risk_profile must be conservative, balanced, or growth. Use balanced when unspecified.
6. scoring_requested is true only when the user explicitly asks to rank, score, recommend an order,
   or choose a best investment. query_type=rank always requires scoring_requested=true.
7. freshness_required is true for current, latest, recent, today, price, news, or market questions.
8. If the question is a follow-up using words such as they/them/their/both and contains no company
   names, use only the provided remembered company identifiers.
9. You only interpret the request. Do not choose tools, industry profiles, or scoring eligibility.
"""


class QueryPlanningError(ValueError):
    """Raised when structured planner output violates the deterministic QueryPlan contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid query plan: " + "; ".join(errors))


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise QueryPlanningError([f"{field_name} must be a list of strings"])

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise QueryPlanningError([f"{field_name} must contain only strings"])
        cleaned = " ".join(item.split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(cleaned)
    return normalized


def _normalize_dimension(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"_+", "_", normalized)


def normalize_query_plan(raw_plan: Any) -> QueryPlan:
    """Normalize structured output without silently accepting invalid types or enum values."""
    if hasattr(raw_plan, "model_dump"):
        raw_plan = raw_plan.model_dump()
    if not isinstance(raw_plan, dict):
        raise QueryPlanningError(["planner output must be a dictionary-like object"])

    query_type = str(raw_plan.get("query_type", "analyze")).strip().casefold()
    risk_profile = str(raw_plan.get("risk_profile", "balanced")).strip().casefold()
    company_mentions = _normalize_string_list(
        raw_plan.get("company_mentions", []),
        "company_mentions",
    )
    raw_dimensions = _normalize_string_list(
        raw_plan.get("requested_dimensions", []),
        "requested_dimensions",
    )
    requested_dimensions = [
        normalized
        for item in raw_dimensions
        if (normalized := _normalize_dimension(item))
    ]

    scoring_requested = raw_plan.get("scoring_requested", query_type == "rank")
    freshness_required = raw_plan.get("freshness_required", False)
    time_horizon = raw_plan.get("time_horizon")
    if isinstance(time_horizon, str):
        time_horizon = " ".join(time_horizon.split()) or None

    # Ranking always routes through a later deterministic eligibility check.
    if query_type == "rank":
        scoring_requested = True

    plan: QueryPlan = {
        "query_type": query_type,
        "company_mentions": company_mentions,
        "requested_dimensions": requested_dimensions,
        "risk_profile": risk_profile,
        "scoring_requested": scoring_requested,
        "freshness_required": freshness_required,
        "time_horizon": time_horizon,
    }
    return plan


def validate_query_plan(plan: QueryPlan) -> list[str]:
    """Return every deterministic contract violation instead of accepting partial output."""
    errors: list[str] = []

    if plan.get("query_type") not in QUERY_TYPES:
        errors.append(f"query_type must be one of {sorted(QUERY_TYPES)}")
    if plan.get("risk_profile") not in RISK_PROFILES:
        errors.append(f"risk_profile must be one of {sorted(RISK_PROFILES)}")

    for field_name in ("company_mentions", "requested_dimensions"):
        value = plan.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{field_name} must be a list of strings")

    for field_name in ("scoring_requested", "freshness_required"):
        if not isinstance(plan.get(field_name), bool):
            errors.append(f"{field_name} must be a boolean")

    if plan.get("time_horizon") is not None and not isinstance(plan.get("time_horizon"), str):
        errors.append("time_horizon must be a string or None")

    if plan.get("query_type") == "rank" and plan.get("scoring_requested") is not True:
        errors.append("rank queries must set scoring_requested=true")

    return errors


def _query_uses_followup_reference(query: str) -> bool:
    return bool(
        re.search(
            r"\b(they|them|their|theirs|both|those|these|same companies|previous companies)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _conversation_excerpt(messages: Sequence[BaseMessage], limit: int = 6) -> str:
    lines: list[str] = []
    for message in list(messages)[-limit:]:
        if not isinstance(message, (HumanMessage, AIMessage)):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(none)"


def _default_query_planner_model():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE"),
    )


def plan_query(
    query: str,
    conversation_context: Sequence[BaseMessage] = (),
    remembered_company_ids: Sequence[str] = (),
    model: Any | None = None,
) -> QueryPlan:
    """Use structured LLM output, then normalize and deterministically validate the result."""
    if not isinstance(query, str) or not query.strip():
        raise QueryPlanningError(["query must be a non-empty string"])

    planner_model = model or _default_query_planner_model()
    structured_model = planner_model.with_structured_output(
        QueryPlan,
        method="function_calling",
    )
    request_payload = {
        "query": query.strip(),
        "remembered_company_ids": list(remembered_company_ids),
        "recent_conversation": _conversation_excerpt(conversation_context),
    }
    raw_plan = structured_model.invoke(
        [
            SystemMessage(content=QUERY_PLANNER_PROMPT),
            HumanMessage(content=json.dumps(request_payload, indent=2)),
        ]
    )
    plan = normalize_query_plan(raw_plan)

    if (
        not plan["company_mentions"]
        and remembered_company_ids
        and _query_uses_followup_reference(query)
    ):
        plan["company_mentions"] = _normalize_string_list(
            list(remembered_company_ids),
            "remembered_company_ids",
        )

    errors = validate_query_plan(plan)
    if errors:
        raise QueryPlanningError(errors)
    return plan


print("✅ F03 structured query planner defined")
'''


F03_SMOKE = r'''# F03 local smoke test with an injected fake model: no network or API keys required.
class _F03FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, method="function_calling"):
        self.schema = schema
        self.method = method
        return self

    def invoke(self, messages):
        assert self.schema is QueryPlan
        assert self.method == "function_calling"
        assert len(messages) == 2
        return self.response


_f03_model = _F03FakeStructuredModel({
    "query_type": "COMPARE",
    "company_mentions": ["Microsoft", "Pfizer", "microsoft"],
    "requested_dimensions": ["Financial Strength", "Long-term Innovation"],
    "risk_profile": "BALANCED",
    "scoring_requested": False,
    "freshness_required": True,
    "time_horizon": " long term ",
})
_f03_plan = plan_query(
    "Compare Microsoft and Pfizer on financial strength and long-term innovation",
    model=_f03_model,
)

assert _f03_plan["query_type"] == "compare"
assert _f03_plan["company_mentions"] == ["Microsoft", "Pfizer"]
assert _f03_plan["requested_dimensions"] == [
    "financial_strength",
    "long_term_innovation",
]
assert _f03_plan["time_horizon"] == "long term"

print("✅ F03 smoke test passed: structured output normalized and validated")
'''


CELL_SPECS = [
    ("multiindustry_f03_intro", "markdown", F03_INTRO),
    ("multiindustry_query_planner", "code", F03_CODE),
    ("multiindustry_f03_smoke", "code", F03_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source=source)
    else:
        cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
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

    missing_specs = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing_specs:
        insertion_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[insertion_index:insertion_index] = [
            _new_cell(*spec) for spec in missing_specs
        ]

    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F03 cells are present")


if __name__ == "__main__":
    main()
