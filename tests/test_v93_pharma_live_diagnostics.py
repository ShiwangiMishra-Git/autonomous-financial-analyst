"""Offline execution contract for the v93 diagnostic cell."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-"
    "v93-pharma-live-diagnostics.ipynb"
)

DIMENSIONS = {
    "financial_strength": .25,
    "pipeline_diversification": .25,
    "clinical_maturity": .15,
    "regulatory_position": .10,
    "commercial_concentration": .10,
    "patent_exclusivity_risk": .10,
    "independent_sentiment": .05,
}


def _scoring(companies, observations, registry):
    return {
        "companies": companies,
        "company_scores": {
            company: {
                "dimensions": {
                    dimension: {
                        "status": "N/A", "score": None,
                        "source_ids": [], "missing": ["offline_missing"],
                    }
                    for dimension in DIMENSIONS
                }
            }
            for company in companies
        },
    }


def test_diagnostic_cell_executes_without_provider_tools():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    namespace = {
        "PHARMA_WEIGHTS_V83": DIMENSIONS,
        "PHARMA_DIMENSION_LABELS_V83": {
            key: key.replace("_", " ").title() for key in DIMENSIONS
        },
        "PHARMA_ALLOWED_ROLES_V83": {
            "financial_strength": {"company_primary"},
            "pipeline_diversification": {"company_primary", "trial_registry_primary", "regulator_primary"},
            "clinical_maturity": {"trial_registry_primary", "peer_reviewed_science", "regulator_primary", "company_primary"},
            "regulatory_position": {"regulator_primary"},
            "commercial_concentration": {"company_primary"},
            "patent_exclusivity_risk": {"company_primary", "patent_registry_primary", "regulator_primary"},
            "independent_sentiment": {"independent_editorial"},
        },
        "score_pharma_comparison_v83": _scoring,
        "display": lambda value: None,
        "Markdown": lambda value: value,
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_LIVE_DIAGNOSTIC_VERSION_V93"].endswith("v1")

