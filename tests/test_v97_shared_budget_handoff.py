"""Offline execution contract for v97 shared-budget accounting."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v97-shared-budget-handoff.ipynb"


class Control:
    def __init__(self, permission, deadline_monotonic, provider_budget_remaining):
        self.permission = permission
        self.deadline_monotonic = deadline_monotonic
        self.provider_budget_remaining = provider_budget_remaining


def test_reserved_call_survives_child_controller_handoff():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(cell.get("source") or []) for cell in notebook["cells"]
               if "Shared pharma provider-budget handoff (2026-08-18 v97)"
               in "".join(cell.get("source") or [])]
    assert len(matches) == 1
    namespace = {
        "Dict": Dict, "Optional": Optional, "deepcopy": deepcopy,
        "ProviderExecutionControl": Control,
    }
    exec(compile(matches[0], str(NOTEBOOK), "exec"), namespace)
    assert namespace["PHARMA_BUDGET_HANDOFF_VERSION_V97"].endswith("v1")

