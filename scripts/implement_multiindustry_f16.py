"""Integrate F16 scenario demonstrations and persist their safe notebook outputs."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f15_smoke"


F16_INTRO = """## Section 3.16: End-to-End Offline Scenarios and Gated Online Demo

F16 demonstrates the implemented F1–F15 boundaries without credentials or provider calls. The
offline runner exposes ten compact scenarios: single technology, single biopharma, same-profile
technology, same-profile biopharma, cross-profile, alias resolution, unknown company, partial RAG
failure, invalid evidence ID, and modified F13 score.

The saved output intentionally contains only routing, validation, attempt, and trace-filename
metadata. It excludes answer prose, evidence bodies, validation-error text, secrets, endpoint
values, and full local paths. Provider-backed smoke testing remains explicitly opt-in through
`F16_ENABLE_LIVE_TESTS=1`. The online cell reuses the notebook's initialized graph, tools, local
RAG indexes, F12 routing, optional F13 scoring, and F15 validation path; it never makes live calls
automatically.
"""


F16_SETUP_CODE = r'''from pathlib import Path
import json

from scripts.run_f16_scenarios import (
    ALL_OFFLINE_SCENARIOS,
    LIVE_OPT_IN_ENV,
    PRIMARY_SCENARIOS,
    compact_scenario_summary,
    live_configuration_status,
    run_all_offline_scenarios,
)
from scripts.f16_live_adapter import (
    create_notebook_live_executor,
    notebook_live_readiness,
)
from scripts.f16_live_tools import bootstrap_live_source_tools


F16_DEMO_TRACE_DIR = Path(".research_runs")
f16_live_tool_bootstrap = bootstrap_live_source_tools(globals())
print(f"✅ F16 runner ready: {len(ALL_OFFLINE_SCENARIOS)} offline scenarios")
print(
    "✅ F16 live tools ready: "
    + ", ".join(f16_live_tool_bootstrap["installed_tool_names"])
)
'''


F16_DEMO_CODE = r'''# Execute all ten deterministic scenarios and retain compact safe output.
f16_demo_summaries = run_all_offline_scenarios(trace_dir=F16_DEMO_TRACE_DIR)
print(json.dumps(f16_demo_summaries, indent=2, sort_keys=True))
'''


F16_LIVE_STATUS_CODE = r'''# Presence-only online readiness check; never prints configuration values.
f16_live_status = live_configuration_status()
f16_notebook_live_status = notebook_live_readiness(globals())
f16_live_summary = {
    "live_execution": "enabled" if f16_live_status["opted_in"] else "skipped",
    "configured": f16_live_status["configured"],
    "missing_variable_names": f16_live_status["missing_variables"],
    "opt_in_variable": LIVE_OPT_IN_ENV,
    "notebook_contracts_ready": f16_notebook_live_status["contracts_ready"],
    "missing_contract_names": f16_notebook_live_status["missing_contract_names"],
    "missing_tool_names": f16_notebook_live_status["missing_tool_names"],
    "rag_ready": f16_notebook_live_status["rag_ready"],
}
print(json.dumps(f16_live_summary, indent=2, sort_keys=True))
'''


F16_ONLINE_INTRO = """### Optional online provider demonstration

Run this only after the notebook configuration, technology RAG, biopharma RAG, and F1–F15 cells
have completed. Set `F16_ENABLE_LIVE_TESTS=1` in the notebook environment, rerun the readiness
cell, and then run the cell below. The demo sends four representative free-text requests through
the real notebook graph. It prints company/mode progress, compact validation metadata, and each
successfully validated final answer. It does not print credentials, retrieved document bodies,
unvalidated drafts, or validation-error text.

An unavailable local RAG index is reported separately and may produce a bounded partial result.
Same-profile numeric scoring runs only when the free-text plan explicitly requests scoring and F12
authorizes complete canonical inputs; an ordinary qualitative comparison does not invent scores.
"""


F16_ONLINE_DEMO_CODE = r'''# Explicitly gated: skipped unless configuration and opt-in are present.
f16_notebook_live_status = notebook_live_readiness(globals())
if not f16_notebook_live_status["ready"]:
    f16_online_summaries = []
    print(json.dumps({
        "online_demo": "skipped",
        "reason": "Set F16_ENABLE_LIVE_TESTS=1, load provider configuration, and run prerequisite cells.",
        "missing_environment_variable_names": f16_notebook_live_status["missing_environment_variable_names"],
        "missing_contract_names": f16_notebook_live_status["missing_contract_names"],
        "missing_tool_names": f16_notebook_live_status["missing_tool_names"],
        "rag_ready": f16_notebook_live_status["rag_ready"],
    }, indent=2, sort_keys=True))
else:
    f16_live_executor = create_notebook_live_executor(
        globals(),
        trace_dir=F16_DEMO_TRACE_DIR,
        max_concurrency=2,
        worker_max_tool_rounds=4,
        progress=print,
    )
    f16_online_results = {}
    f16_online_summaries = []
    for f16_scenario in PRIMARY_SCENARIOS:
        f16_result = f16_live_executor(f16_scenario)
        f16_online_results[f16_scenario.name] = f16_result
        f16_summary = compact_scenario_summary(f16_scenario.name, f16_result)
        f16_online_summaries.append(f16_summary)
        print(json.dumps(f16_summary, indent=2, sort_keys=True))
        if f16_result.get("final_status") == "success" and (
            f16_result.get("validation") or {}
        ).get("valid") is True:
            print("Validated answer:\n" + f16_result["final_answer"] + "\n")
        else:
            print("No validated answer was returned for this scenario.\n")
'''


F16_INTERACTIVE_INTRO = """### Free-text online test

Edit only `USER_QUERY` in the next cell and run it. The coordinator extracts supported companies,
chooses `single`, `same_profile`, or `cross_profile`, fans out research, and returns an answer only
after F15 validation. The cell also shows whether authoritative F13 scores were used. Numeric
ranking is permitted only for complete same-profile requests with canonical scoring evidence. The
cell bootstraps its market, metric, news, sentiment, and Technology-RAG tools itself; it does not
depend on rerunning the original Part 1 tool cells.
"""


F16_INTERACTIVE_CODE = r'''# Edit this sentence, then run this cell.
import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from scripts.f16_live_adapter import create_notebook_live_executor, notebook_live_readiness
from scripts.f16_live_tools import bootstrap_live_source_tools


F16_DEMO_TRACE_DIR = Path(".research_runs")
# Running this explicitly labelled online cell is the user's opt-in to provider calls.
os.environ["F16_ENABLE_LIVE_TESTS"] = "1"
f16_live_tool_bootstrap = bootstrap_live_source_tools(globals())
USER_QUERY = "Compare Microsoft and Nvidia on financial strength and AI innovation."


def ask_financial_analyst(query: str):
    """Run one arbitrary free-text question through the online F1-F15 workflow."""
    readiness = notebook_live_readiness(globals())
    if not readiness["ready"]:
        print(json.dumps({
            "interactive_query": "skipped",
            "missing_environment_variable_names": readiness["missing_environment_variable_names"],
            "missing_contract_names": readiness["missing_contract_names"],
            "missing_tool_names": readiness["missing_tool_names"],
            "rag_ready": readiness["rag_ready"],
        }, indent=2, sort_keys=True))
        return None

    executor = create_notebook_live_executor(
        globals(),
        trace_dir=F16_DEMO_TRACE_DIR,
        max_concurrency=2,
        worker_max_tool_rounds=4,
        progress=print,
    )
    result = executor(query)
    synthesis = result.get("synthesis") or {}
    validation = result.get("validation") or {}
    print("\n--- ROUTING ---")
    print("Mode:", synthesis.get("mode") or "bounded_stop")
    print("Status:", result.get("final_status"))
    print("Validated:", validation.get("valid") is True)

    scores = synthesis.get("scores_used") or {}
    print("\n--- DETERMINISTIC RANKING ---")
    if scores:
        ranked = sorted(
            scores.items(),
            key=lambda item: float(item[1].get("total_score", float("-inf"))),
            reverse=True,
        )
        for rank, (ticker, values) in enumerate(ranked, start=1):
            print(
                f"{rank}. {ticker}: total_score={values.get('total_score')} "
                f"details={json.dumps(values, sort_keys=True)}"
            )
    else:
        print("No numeric ranking was authorized; this is a qualitative analysis.")

    print("\n--- ANSWER ---")
    if result.get("final_status") == "success" and validation.get("valid") is True:
        print(result["final_answer"])
    else:
        print("No validated answer was returned.")
    print("\nTrace:", Path(str(result.get("trace_path") or "")).name or "none")
    return result


f16_interactive_result = ask_financial_analyst(USER_QUERY)
'''


CELL_SPECS = [
    ("multiindustry_f16_intro", "markdown", F16_INTRO),
    ("multiindustry_f16_runner_setup", "code", F16_SETUP_CODE),
    ("multiindustry_f16_all_scenarios_demo", "code", F16_DEMO_CODE),
    ("multiindustry_f16_live_status", "code", F16_LIVE_STATUS_CODE),
    ("multiindustry_f16_online_intro", "markdown", F16_ONLINE_INTRO),
    ("multiindustry_f16_online_demo", "code", F16_ONLINE_DEMO_CODE),
    ("multiindustry_f16_interactive_intro", "markdown", F16_INTERACTIVE_INTRO),
    ("multiindustry_f16_interactive_query", "code", F16_INTERACTIVE_CODE),
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


def integrate_f16_cells(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh F16 cells after F15 without touching learner scaffolding."""
    notebook = nbformat.read(notebook_path, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    ordered_cells = []
    for cell_id, cell_type, source in CELL_SPECS:
        existing = cells_by_id.get(cell_id)
        if existing is None:
            existing = _new_cell(cell_id, cell_type, source)
        else:
            source_changed = existing.get("source") != source
            existing["cell_type"] = cell_type
            existing["source"] = source
            if cell_type == "code" and source_changed:
                existing["execution_count"] = None
                existing["outputs"] = []
        ordered_cells.append(existing)

    target_ids = {cell_id for cell_id, _, _ in CELL_SPECS}
    notebook.cells = [cell for cell in notebook.cells if cell.get("id") not in target_ids]
    try:
        insertion_index = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
    except StopIteration as exc:
        raise ValueError(
            f"Notebook is missing insertion anchor {INSERT_AFTER_CELL_ID!r}"
        ) from exc
    notebook.cells[insertion_index:insertion_index] = ordered_cells
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)


def execute_and_save_f16_outputs(
    notebook_path: Path = NOTEBOOK_PATH,
    *,
    trace_dir: Path | None = None,
) -> dict[str, object]:
    """Execute only the offline F16 demo/status cells and persist their stream outputs.

    Earlier assignment and implementation cells are not re-executed. The runner loads the frozen
    notebook contracts into an isolated namespace, and no live adapter is invoked.
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = {cell.get("id"): cell for cell in notebook.cells}
    required_ids = (
        "multiindustry_f16_runner_setup",
        "multiindustry_f16_all_scenarios_demo",
        "multiindustry_f16_live_status",
        "multiindustry_f16_online_demo",
    )
    missing = [cell_id for cell_id in required_ids if cell_id not in cells]
    if missing:
        raise ValueError(f"Notebook is missing F16 cells: {missing}")

    namespace: dict[str, object] = {}
    original_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        output_by_id: dict[str, str] = {}
        for cell_id in required_ids:
            if cell_id == "multiindustry_f16_all_scenarios_demo" and trace_dir is not None:
                namespace["F16_DEMO_TRACE_DIR"] = Path(trace_dir)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(cells[cell_id].source, namespace)
            output_by_id[cell_id] = stream.getvalue()
    finally:
        os.chdir(original_directory)

    existing_counts = [
        cell.execution_count
        for cell in notebook.cells
        if cell.cell_type == "code" and isinstance(cell.get("execution_count"), int)
    ]
    next_count = max(existing_counts, default=0) + 1
    for cell_id in required_ids:
        cell = cells[cell_id]
        cell.execution_count = next_count
        next_count += 1
        cell.outputs = [nbformat.v4.new_output(
            output_type="stream",
            name="stdout",
            text=output_by_id[cell_id],
        )]

    summaries = namespace.get("f16_demo_summaries")
    if not isinstance(summaries, list) or len(summaries) != 10:
        raise ValueError("F16 demo did not produce all ten scenario summaries")
    live_summary = namespace.get("f16_live_summary")
    if not isinstance(live_summary, dict):
        raise ValueError("F16 live readiness cell did not produce a summary")
    online_summaries = namespace.get("f16_online_summaries")
    if not isinstance(online_summaries, list):
        raise ValueError("F16 online demo cell did not produce a list")
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    return {
        "summaries": summaries,
        "live_summary": live_summary,
        "online_summaries": online_summaries,
        "trace_dir": str(namespace["F16_DEMO_TRACE_DIR"]),
    }
