import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v110-no-pharma-live-demos.ipynb"


def _source(cell):return "".join(cell.get("source") or [])


def _top_level_route_calls(source):
    safe_contract_markers=("Offline contracts:","Offline fixture contracts:",
        "confirmed mock mode only","Offline unified-boundary contract")
    if any(marker in source for marker in safe_contract_markers):return []
    try:tree=ast.parse(source)
    except SyntaxError:return []
    found=[]
    def visit(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):return
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) \
                and node.func.id in {"route_pharma_query","route_financial_query"}:
            found.append(node.func.id)
        for child in ast.iter_child_nodes(node):visit(child)
    for statement in tree.body:visit(statement)
    return found


def test_no_executable_top_level_route_calls_remain():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found=[]
    for index,cell in enumerate(nb["cells"]):
        if cell.get("cell_type")=="code":found.extend((index,name) for name in _top_level_route_calls(_source(cell)))
    assert found==[]


def test_known_pharma_live_demo_cells_removed_and_offline_harness_retained():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"));text="\n".join(_source(cell) for cell in nb["cells"])
    assert "Combined end-to-end pharma demo (live providers)" not in text
    assert "pharma-investment-demo-5" not in text
    assert "pharma-v108-demo-" not in text
    assert "Offline pharma scoring correctness and readability harness (v85)" in text
    metadata=nb["metadata"]["pharma_live_demo_sanitization"]
    assert metadata["top_level_route_calls_remaining"]==0
    assert metadata["offline_harnesses_preserved"] is True
