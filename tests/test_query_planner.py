"""Offline tests for the F03 structured free-text query planner."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _planner_namespace():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}

    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(cells["multiindustry_company_registry"], namespace)
        exec(cells["multiindustry_query_planner"], namespace)
    return namespace


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.schema = None
        self.method = None
        self.invocations = []

    def with_structured_output(self, schema, method="function_calling"):
        self.schema = schema
        self.method = method
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


def _response(**overrides):
    response = {
        "query_type": "analyze",
        "company_mentions": ["Microsoft"],
        "requested_dimensions": ["financial health"],
        "risk_profile": "balanced",
        "scoring_requested": False,
        "freshness_required": True,
        "time_horizon": None,
    }
    response.update(overrides)
    return response


def test_single_company_plan_is_normalized_and_uses_structured_output():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(query_type="ANALYZE", risk_profile="BALANCED"))

    plan = namespace["plan_query"]("Analyze Microsoft", model=fake)

    assert plan["query_type"] == "analyze"
    assert plan["company_mentions"] == ["Microsoft"]
    assert plan["requested_dimensions"] == ["financial_health"]
    assert fake.schema is namespace["QueryPlan"]
    assert fake.method == "function_calling"
    assert len(fake.invocations) == 1


def test_same_industry_multi_company_plan_preserves_one_item_per_company():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="compare",
        company_mentions=["Microsoft", "Google", "NVIDIA"],
        requested_dimensions=["AI strategy", "financial strength"],
    ))

    plan = namespace["plan_query"]("Compare Microsoft, Google, and NVIDIA", model=fake)

    assert plan["company_mentions"] == ["Microsoft", "Google", "NVIDIA"]
    assert plan["requested_dimensions"] == ["ai_strategy", "financial_strength"]


def test_cross_industry_plan_preserves_all_company_mentions():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="compare",
        company_mentions=["Microsoft", "Pfizer", "Merck"],
        requested_dimensions=["long-term innovation"],
    ))

    plan = namespace["plan_query"](
        "Compare Microsoft, Pfizer, and Merck on long-term innovation",
        model=fake,
    )

    assert plan["company_mentions"] == ["Microsoft", "Pfizer", "Merck"]
    assert plan["requested_dimensions"] == ["long_term_innovation"]


def test_rank_query_forces_scoring_requested_true():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="rank",
        company_mentions=["MSFT", "PFE"],
        scoring_requested=False,
    ))

    plan = namespace["plan_query"]("Rank MSFT and PFE", model=fake)

    assert plan["query_type"] == "rank"
    assert plan["scoring_requested"] is True


def test_followup_uses_remembered_companies_when_model_returns_none():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="compare",
        company_mentions=[],
        requested_dimensions=["debt"],
    ))

    plan = namespace["plan_query"](
        "Compare their debt",
        conversation_context=[
            HumanMessage(content="Compare Microsoft and Pfizer"),
            AIMessage(content="Previous answer"),
        ],
        remembered_company_ids=["microsoft", "pfizer"],
        model=fake,
    )

    assert plan["company_mentions"] == ["microsoft", "pfizer"]


def test_non_followup_empty_company_list_is_preserved_for_later_clarification():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="analyze",
        company_mentions=[],
        requested_dimensions=["market outlook"],
    ))

    plan = namespace["plan_query"]("What is the market outlook?", model=fake)

    assert plan["company_mentions"] == []


def test_duplicate_mentions_are_removed_case_insensitively():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(
        query_type="compare",
        company_mentions=["Microsoft", "microsoft", "MSFT", "MSFT"],
    ))

    plan = namespace["plan_query"]("Compare Microsoft and MSFT", model=fake)

    assert plan["company_mentions"] == ["Microsoft", "MSFT"]


def test_invalid_risk_profile_is_rejected():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(risk_profile="extreme"))

    with pytest.raises(namespace["QueryPlanningError"], match="risk_profile"):
        namespace["plan_query"]("Analyze Microsoft", model=fake)


def test_invalid_boolean_type_is_rejected_without_truthy_coercion():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response(freshness_required="yes"))

    with pytest.raises(namespace["QueryPlanningError"], match="freshness_required"):
        namespace["plan_query"]("Analyze Microsoft", model=fake)


def test_empty_query_is_rejected_before_model_invocation():
    namespace = _planner_namespace()
    fake = FakeStructuredModel(_response())

    with pytest.raises(namespace["QueryPlanningError"], match="non-empty"):
        namespace["plan_query"]("   ", model=fake)
    assert fake.invocations == []


def test_conversation_excerpt_excludes_tool_messages():
    namespace = _planner_namespace()
    excerpt = namespace["_conversation_excerpt"]([
        HumanMessage(content="Question"),
        ToolMessage(content="secret tool payload", tool_call_id="call-1"),
        AIMessage(content="Answer"),
    ])

    assert "user: Question" in excerpt
    assert "assistant: Answer" in excerpt
    assert "secret tool payload" not in excerpt


def test_planner_output_must_be_dictionary_like():
    namespace = _planner_namespace()
    fake = FakeStructuredModel("not a structured plan")

    with pytest.raises(namespace["QueryPlanningError"], match="dictionary-like"):
        namespace["plan_query"]("Analyze Microsoft", model=fake)
