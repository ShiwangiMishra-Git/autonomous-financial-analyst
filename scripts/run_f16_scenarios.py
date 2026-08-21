"""Run compact F16 offline scenarios and gate optional live smoke execution.

The default path is entirely deterministic: it uses local F12-normalized fixtures, an injected
fake F14 model, and the frozen F15 validation/trace workflow.  The module performs no work at
import time.  A provider-facing live adapter is invoked only when the caller explicitly opts in
*and* the documented environment variables are present.
"""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import dataclass
import importlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

# Support both ``python -m scripts.run_f16_scenarios`` and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE
from scripts.implement_multiindustry_f14 import F14_CODE
from scripts.implement_multiindustry_f15 import F15_WORKFLOW_CODE
from scripts.implement_multiindustry_f15_evidence import F15_EVIDENCE_CODE
from scripts.implement_multiindustry_f15_traces import F15_TRACES_CODE


WORKING_NOTEBOOK = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
DEFAULT_TRACE_DIR = PROJECT_ROOT / ".research_runs"
LIVE_OPT_IN_ENV = "F16_ENABLE_LIVE_TESTS"
LIVE_ADAPTER_ENV = "F16_LIVE_ADAPTER"
LIVE_REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_API_BASE", "TAVILY_API_KEY")
TECH_PROFILE = "technology.ai.v1"
BIOPHARMA_PROFILE = "healthcare.biopharma.v1"


@dataclass(frozen=True)
class ScenarioSpec:
    """Safe, bounded description of one F16 primary scenario."""

    name: str
    query: str
    mode: str
    companies: tuple[tuple[str, str, str], ...]
    scores: Mapping[str, Any] | None = None
    fixture_kind: str = "workflow"


PRIMARY_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="single_technology",
        query="Analyze Microsoft as a technology company.",
        mode="single",
        companies=(("microsoft", "MSFT", TECH_PROFILE),),
    ),
    ScenarioSpec(
        name="single_biopharma",
        query="Analyze Pfizer as a biopharma company.",
        mode="single",
        companies=(("pfizer", "PFE", BIOPHARMA_PROFILE),),
    ),
    ScenarioSpec(
        name="same_profile_technology",
        query="Compare Microsoft and Nvidia within the technology profile.",
        mode="same_profile",
        companies=(
            ("microsoft", "MSFT", TECH_PROFILE),
            ("nvidia", "NVDA", TECH_PROFILE),
        ),
        scores={
            "MSFT": {"total_score": 78.0, "rank": 1},
            "NVDA": {"total_score": 72.0, "rank": 2},
        },
    ),
    ScenarioSpec(
        name="cross_profile",
        query="Compare Microsoft and Pfizer as a cross-industry portfolio decision.",
        mode="cross_profile",
        companies=(
            ("microsoft", "MSFT", TECH_PROFILE),
            ("pfizer", "PFE", BIOPHARMA_PROFILE),
        ),
    ),
)


ALL_OFFLINE_SCENARIOS: tuple[ScenarioSpec, ...] = (
    PRIMARY_SCENARIOS[0],
    PRIMARY_SCENARIOS[1],
    PRIMARY_SCENARIOS[2],
    ScenarioSpec(
        name="same_profile_biopharma",
        query="Compare Pfizer and Merck within the biopharma profile.",
        mode="same_profile",
        companies=(
            ("pfizer", "PFE", BIOPHARMA_PROFILE),
            ("merck", "MRK", BIOPHARMA_PROFILE),
        ),
        scores={
            "PFE": {"total_score": 73.0, "rank": 1},
            "MRK": {"total_score": 69.0, "rank": 2},
        },
    ),
    PRIMARY_SCENARIOS[3],
    ScenarioSpec(
        name="alias_resolution",
        query="Analyze ASTRA ZENECA as a biopharma company.",
        mode="single",
        companies=(("astrazeneca", "AZN", BIOPHARMA_PROFILE),),
        fixture_kind="alias",
    ),
    ScenarioSpec(
        name="unknown_company",
        query="Analyze Example Unknown Therapeutics.",
        mode="bounded_stop",
        companies=(),
        fixture_kind="unknown_company",
    ),
    ScenarioSpec(
        name="partial_rag_failure",
        query="Analyze Pfizer when private RAG is temporarily unavailable.",
        mode="single",
        companies=(("pfizer", "PFE", BIOPHARMA_PROFILE),),
        fixture_kind="partial_rag_failure",
    ),
    ScenarioSpec(
        name="invalid_evidence_id",
        query="Validate a Microsoft draft with an unavailable evidence ID.",
        mode="single",
        companies=(("microsoft", "MSFT", TECH_PROFILE),),
        fixture_kind="invalid_evidence_id",
    ),
    ScenarioSpec(
        name="modified_f13_score",
        query="Validate a technology comparison with a modified F13 score.",
        mode="same_profile",
        companies=(
            ("microsoft", "MSFT", TECH_PROFILE),
            ("nvidia", "NVDA", TECH_PROFILE),
        ),
        scores={
            "MSFT": {"total_score": 78.0, "rank": 1},
            "NVDA": {"total_score": 72.0, "rank": 2},
        },
        fixture_kind="modified_f13_score",
    ),
)


class OfflineSynthesisModel:
    """Return a deterministic F14 payload using only IDs and limitations in the prompt."""

    def __init__(self) -> None:
        """Initialize call tracking used by scenario and notebook demonstrations."""
        self.calls: list[list[Any]] = []

    def bind_tools(self, tools: Any) -> None:
        """Fail closed if the offline synthesis boundary attempts to expose tools."""
        raise AssertionError(f"F14/F15 synthesis must not bind tools: {type(tools).__name__}")

    def invoke(self, messages: Sequence[Any]) -> dict[str, Any]:
        """Build one grounded response from the bounded serialized F14 context."""
        self.calls.append(list(messages))
        payload = next(
            json.loads(message.content)
            for message in messages
            if '"available_evidence_ids"' in getattr(message, "content", "")
        )
        evidence_ids = list(payload["available_evidence_ids"])
        citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
        answer = f"Deterministic {payload['mode']} scenario result {citations}.".strip()
        return {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "limitations": list(payload["required_limitations"]),
        }


def _notebook_state_source(notebook_path: str | Path = WORKING_NOTEBOOK) -> str:
    """Read only the frozen state-contract cell from the canonical working notebook."""
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("id") == "multiindustry_state_contracts":
            return "".join(cell.get("source", []))
    raise RuntimeError("Canonical notebook is missing multiindustry_state_contracts")


def build_offline_f15_api(
    notebook_path: str | Path = WORKING_NOTEBOOK,
) -> dict[str, Any]:
    """Load frozen F12-F15 contracts into an isolated namespace without provider calls."""
    namespace: dict[str, Any] = {
        "get_industry_profile": lambda profile_id: {
            "profile_id": profile_id,
            "scoring_enabled": True,
            "rubric_id": (
                "technology.ai.score.v1"
                if profile_id == TECH_PROFILE
                else "healthcare.biopharma.score.v1"
            ),
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_notebook_state_source(notebook_path), namespace)
        exec(F12_ROUTING_CODE, namespace)
        exec(F14_CODE, namespace)
        exec(F15_EVIDENCE_CODE, namespace)
        exec(F15_TRACES_CODE, namespace)
        exec(F15_WORKFLOW_CODE, namespace)
    return namespace


def _company_result(
    run_id: str,
    company_id: str,
    ticker: str,
    profile_id: str,
) -> dict[str, Any]:
    """Create one canonical normalized result with bounded synthetic provenance."""
    evidence_id = f"EV-{run_id}-{ticker}"
    company = {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_id.replace("_", " ").title(),
        "profile_id": profile_id,
        "resolution_status": "resolved",
    }
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [{
            "evidence_id": evidence_id,
            "run_id": run_id,
            "company_id": company_id,
            "ticker": ticker,
            "profile_id": profile_id,
            "evidence_type": "f16_offline_fixture",
            "source_name": "local-deterministic-fixture",
            "source_uri": None,
            "status": "success",
        }],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


def build_scenario_context(spec: ScenarioSpec) -> dict[str, Any]:
    """Convert a scenario specification into the frozen F14 ``SynthesisContext`` shape."""
    run_id = f"f16-{spec.name.replace('_', '-')}"
    results = {
        ticker: _company_result(run_id, company_id, ticker, profile_id)
        for company_id, ticker, profile_id in spec.companies
    }
    if not results:
        raise ValueError(f"Scenario {spec.name} stops before synthesis and has no F14 context")
    if spec.fixture_kind == "partial_rag_failure":
        ticker = next(iter(results))
        result = results[ticker]
        result["status"] = "partial"
        result["missing_dimensions"] = ["private_rag"]
        result["errors"] = ["Private RAG retrieval failed; usable market evidence retained."]
        result["evidence"].append({
            "evidence_id": f"EV-{run_id}-{ticker}-RAG",
            "run_id": run_id,
            "company_id": result["company"]["company_id"],
            "ticker": ticker,
            "profile_id": result["profile_id"],
            "evidence_type": "private_rag",
            "source_name": "local-deterministic-fixture",
            "source_uri": None,
            "status": "failed",
            "error": "retrieval_unavailable",
        })
    eligible = spec.mode == "same_profile" and spec.fixture_kind != "partial_rag_failure"
    return {
        "run_id": run_id,
        "original_query": spec.query,
        "comparison_mode": spec.mode,
        "normalized_results": results,
        "scoring_eligibility": {
            "eligible": eligible,
            "reason": (
                "Complete same-profile comparison with a versioned rubric."
                if eligible
                else "Numeric sector comparison is not applicable to this mode."
            ),
        },
        "scores": dict(spec.scores) if spec.scores is not None else None,
    }


def compact_scenario_summary(
    scenario_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return safe status metadata without answer prose, evidence bodies, or credentials."""
    validation = result.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    synthesis = result.get("synthesis")
    synthesis = synthesis if isinstance(synthesis, Mapping) else {}
    trace_path = result.get("trace_path")
    return {
        "scenario": scenario_name,
        "mode": synthesis.get("mode"),
        "final_status": result.get("final_status"),
        "validation_valid": validation.get("valid") is True,
        "validated_evidence_count": len(validation.get("validated_evidence_ids", [])),
        "validation_error_count": len(validation.get("errors", [])),
        "attempts": result.get("attempts"),
        "correction_attempts": result.get("correction_attempts"),
        "trace_file": Path(str(trace_path)).name if trace_path else None,
    }


def run_offline_scenarios(
    *,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    scenarios: Iterable[ScenarioSpec] = PRIMARY_SCENARIOS,
    workflow: Callable[..., Mapping[str, Any]] | None = None,
    model_factory: Callable[[], Any] = OfflineSynthesisModel,
    notebook_path: str | Path = WORKING_NOTEBOOK,
) -> list[dict[str, Any]]:
    """Execute the primary scenarios locally and return compact deterministic summaries."""
    selected = tuple(scenarios)
    if not selected:
        raise ValueError("At least one scenario is required")
    api = build_offline_f15_api(notebook_path)
    workflow_fn = workflow or api["run_f15_validated_synthesis"]
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        if spec.fixture_kind == "unknown_company":
            summaries.append({
                "scenario": spec.name,
                "mode": None,
                "final_status": "bounded_stop",
                "validation_valid": False,
                "validated_evidence_count": 0,
                "validation_error_count": 1,
                "attempts": 0,
                "correction_attempts": 0,
                "trace_file": None,
            })
            continue
        context = build_scenario_context(spec)
        if spec.fixture_kind in {"invalid_evidence_id", "modified_f13_score"}:
            result = _run_post_synthesis_validation_probe(
                api,
                spec,
                context,
                model_factory(),
                trace_dir=trace_dir,
            )
        else:
            result = workflow_fn(context, model_factory(), trace_dir=trace_dir)
        summaries.append(compact_scenario_summary(spec.name, result))
    return summaries


def _run_post_synthesis_validation_probe(
    api: Mapping[str, Any],
    spec: ScenarioSpec,
    context: Mapping[str, Any],
    model: Any,
    *,
    trace_dir: str | Path,
) -> dict[str, Any]:
    """Mutate a valid F14 artifact and demonstrate deterministic F15 rejection."""
    synthesis = deepcopy(dict(api["synthesize_answer"](context, model)))
    if spec.fixture_kind == "invalid_evidence_id":
        invalid_id = f"EV-{context['run_id']}-NOT-IN-CATALOG"
        synthesis["answer"] = f"Deliberately invalid citation [{invalid_id}]."
        synthesis["evidence_ids"] = [invalid_id]
    elif spec.fixture_kind == "modified_f13_score":
        first_ticker = next(iter(synthesis["scores_used"]))
        synthesis["scores_used"][first_ticker]["total_score"] += 1.0
    else:
        raise ValueError(f"Unsupported validation probe: {spec.fixture_kind}")

    safe = api["_f14_validate_context"](context)
    validation = api["validate_synthesis_result"](
        safe["run_id"],
        safe["normalized_results"],
        synthesis,
        authoritative_scores=safe["scores"],
        scoring_eligibility=safe["scoring_eligibility"],
        required_limitations=safe["required_limitations"],
    )
    trace = api["create_research_trace"](
        run_id=safe["run_id"],
        query=safe["original_query"],
        comparison_mode=safe["comparison_mode"],
        normalized_results=safe["normalized_results"],
        f13_scores=safe["scores"],
        f14_synthesis=synthesis,
    )
    trace = api["record_validation_attempt"](
        trace,
        validation,
        attempt_number=1,
    )
    trace = api["finalize_research_trace"](
        trace,
        final_status="failed",
        terminal_error="Deterministic F15 validation probe was rejected.",
    )
    trace_path = api["write_research_trace"](trace, trace_dir=trace_dir)["path"]
    return {
        "final_status": "failed",
        "final_answer": "Validation probe rejected; no answer returned.",
        "synthesis": synthesis,
        "validation": validation,
        "attempts": 1,
        "correction_attempts": 0,
        "warnings": ["Deterministic validation rejected the mutated artifact."],
        "trace_path": trace_path,
    }


def run_all_offline_scenarios(
    *,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    notebook_path: str | Path = WORKING_NOTEBOOK,
) -> list[dict[str, Any]]:
    """Run all ten reusable F16 scenarios for notebook demonstrations and local review."""
    return run_offline_scenarios(
        trace_dir=trace_dir,
        scenarios=ALL_OFFLINE_SCENARIOS,
        notebook_path=notebook_path,
    )


def live_configuration_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report live readiness using variable names only; never return secret values."""
    environment = os.environ if environ is None else environ
    enabled = str(environment.get(LIVE_OPT_IN_ENV, "")).strip().casefold() in {
        "1", "true", "yes",
    }
    missing = [name for name in LIVE_REQUIRED_ENV if not environment.get(name)]
    return {
        "opted_in": enabled,
        "configured": not missing,
        "missing_variables": missing,
        "required_variables": list(LIVE_REQUIRED_ENV),
    }


def run_live_scenarios(
    live_executor: Callable[[ScenarioSpec], Mapping[str, Any]],
    *,
    scenarios: Iterable[ScenarioSpec] = PRIMARY_SCENARIOS,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Invoke an injected live adapter only after explicit opt-in and configuration checks."""
    status = live_configuration_status(environ)
    if not status["opted_in"]:
        raise RuntimeError(f"Live scenarios require explicit {LIVE_OPT_IN_ENV}=1 opt-in")
    if not status["configured"]:
        missing = ", ".join(status["missing_variables"])
        raise RuntimeError(f"Live scenarios are not configured; missing variables: {missing}")
    if not callable(live_executor):
        raise TypeError("live_executor must be callable")
    summaries: list[dict[str, Any]] = []
    for spec in tuple(scenarios):
        result = live_executor(spec)
        if not isinstance(result, Mapping):
            raise TypeError(f"Live executor returned a non-mapping for {spec.name}")
        summaries.append(compact_scenario_summary(spec.name, result))
    return summaries


def _load_live_adapter(reference: str) -> Callable[[ScenarioSpec], Mapping[str, Any]]:
    """Load an explicitly named ``module:function`` live adapter after safety gates pass."""
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Live adapter must use module:function syntax")
    adapter = getattr(importlib.import_module(module_name), function_name)
    if not callable(adapter):
        raise TypeError("Configured live adapter is not callable")
    return adapter


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an offline-by-default F16 command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir", type=Path, default=DEFAULT_TRACE_DIR,
        help="Directory for bounded F15 JSON traces (offline mode).",
    )
    parser.add_argument(
        "--live-adapter",
        help=(
            "Optional module:function adapter. It is loaded only with explicit "
            f"{LIVE_OPT_IN_ENV}=1 and complete environment configuration."
        ),
    )
    parser.add_argument(
        "--all-offline",
        action="store_true",
        help="Run all ten deterministic F16 scenarios instead of the four primary scenarios.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline scenarios, or a separately gated injected live adapter, and print summaries."""
    args = _parse_args(argv)
    if args.live_adapter:
        status = live_configuration_status()
        if not status["opted_in"]:
            raise RuntimeError(f"Live scenarios require explicit {LIVE_OPT_IN_ENV}=1 opt-in")
        if not status["configured"]:
            missing = ", ".join(status["missing_variables"])
            raise RuntimeError(f"Live scenarios are not configured; missing variables: {missing}")
        summaries = run_live_scenarios(_load_live_adapter(args.live_adapter))
    else:
        summaries = (
            run_all_offline_scenarios(trace_dir=args.trace_dir)
            if args.all_offline
            else run_offline_scenarios(trace_dir=args.trace_dir)
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
