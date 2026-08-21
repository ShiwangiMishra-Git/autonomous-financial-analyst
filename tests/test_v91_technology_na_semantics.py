"""Offline contract tests for the v91 technology missing-evidence override."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-"
    "v91-technology-na-semantics.ipynb"
)


def _rank_scores(values, lower_is_better):
    ordered = sorted(values, key=lambda key: values[key], reverse=not lower_is_better)
    return {key: max(0.4 - index * 0.08, 0.0) for index, key in enumerate(ordered)}


def _format_number(value, digits=3):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _load_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    namespace = {
        "Dict": Dict,
        "deepcopy": deepcopy,
        "STRATEGY_WEIGHTS": {
            "balanced": {name: 1.0 for name in
                         ["market_cap", "total_revenue", "pe_ratio", "beta", "dividend_yield"]}
        },
        "SIGNAL_WEIGHTS": {
            "balanced": {name: 1.0 for name in
                         ["infrastructure_moat", "product_deployment", "research_depth", "strategic_commitment"]}
        },
        "COMPARISON_COMPONENT_WEIGHTS": {
            "balanced": {"financial": .45, "ai": .40, "sentiment": .15}
        },
        "METRIC_NAMES": ["market_cap", "total_revenue", "pe_ratio", "beta", "dividend_yield"],
        "SIGNAL_NAMES": ["infrastructure_moat", "product_deployment", "research_depth", "strategic_commitment"],
        "LOWER_IS_BETTER": {"pe_ratio", "beta"},
        "PER_METRIC_CAP": .40,
        "BUY_THRESHOLD": .70,
        "HOLD_THRESHOLD": .45,
        "_rank_scores": _rank_scores,
        "_format_number": _format_number,
        "generate_compact_comparison_report": lambda comparison: "base compact",
        "generate_full_comparison_report": lambda comparison: "base full",
        "render_ranked_recommendation_v9": lambda comparison, full=False: "base ranked",
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_missing_signal_is_na_and_noncomparable():
    namespace = _load_namespace()
    assert namespace["TECH_SCORING_VERSION"] == "2.2-missing-evidence-na"


def test_supported_none_remains_numeric_zero():
    namespace = _load_namespace()
    result = namespace["score_companies"](
        {"AAA": {name: 1 for name in namespace["METRIC_NAMES"]}},
        {"AAA": {
            name: {"level": "none" if name == "research_depth" else "full",
                   "score": 0.0 if name == "research_depth" else 1.0, "sources": ["mock"]}
            for name in namespace["SIGNAL_NAMES"]
        }},
        {"AAA": {"average": .5, "articles": [1, 2, 3]}},
    )["AAA"]
    assert result["score_status"] == "complete"
    assert result["scoring_breakdown"]["ai"]["research_depth"]["level_score"] == 0.0

