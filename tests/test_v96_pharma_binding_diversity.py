"""Offline execution contract for v96 role/publisher diversity."""

import json
import re
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v96-pharma-binding-diversity.ipynb"


def test_v96_diverse_binding_contracts_execute():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if "Pharma role/publisher-diverse score binding (2026-08-18 v96)"
               in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    source = matches[0]
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
        "extract_pharma_scoring_inputs_v84": lambda query, companies, registry, model=None: {},
        "diagnose_pharma_result_v93": lambda result: {"rows": []},
        "_normalized_score_v83": lambda value: None if value is None else float(value),
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_BINDING_DIVERSITY_VERSION_V96"] == "pharma.binding-diversity.v1"
