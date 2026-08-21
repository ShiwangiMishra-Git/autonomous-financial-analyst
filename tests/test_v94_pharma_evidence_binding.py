"""Offline execution contract for the v94 binding-hardening cell."""

import json
import re
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v94-pharma-evidence-binding.ipynb"


def test_v94_balanced_candidates_and_azn_routing():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    dimensions = {
        "financial_strength": .25, "pipeline_diversification": .25,
        "clinical_maturity": .15, "regulatory_position": .10,
        "commercial_concentration": .10, "patent_exclusivity_risk": .10,
        "independent_sentiment": .05,
    }
    allowed = {
        "financial_strength": {"company_primary"},
        "pipeline_diversification": {"company_primary", "trial_registry_primary", "regulator_primary"},
        "clinical_maturity": {"trial_registry_primary", "peer_reviewed_science", "regulator_primary", "company_primary"},
        "regulatory_position": {"regulator_primary"},
        "commercial_concentration": {"company_primary"},
        "patent_exclusivity_risk": {"company_primary", "patent_registry_primary", "regulator_primary"},
        "independent_sentiment": {"independent_editorial"},
    }
    namespace = {
        "re": re, "Dict": Dict, "List": List,
        "PHARMA_WEIGHTS_V83": dimensions,
        "PHARMA_ALLOWED_ROLES_V83": allowed,
        "PHARMA_MAX_ATTEMPTS_V42": 3,
        "_pharma_source_role_v42": lambda url, ticker="": "independent_editorial",
        "_pharma_query_specs_v42": lambda ticker, query, dimension: [],
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_BINDING_VERSION_V94"] == "pharma.scoring-candidates.v2"

