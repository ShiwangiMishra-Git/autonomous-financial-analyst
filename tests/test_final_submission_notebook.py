import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"Autonomous_financial_analyst_FINAL_SUBMISSION.ipynb"


def load():
    return json.loads(NB.read_text(encoding="utf-8"))


def all_source(nb):
    return "\n".join("".join(cell.get("source",[])) for cell in nb["cells"])


def test_submission_is_clean_and_tracks_both_stable_baselines():
    nb=load();meta=nb["metadata"]["submission_build"]
    assert len(nb["cells"])<230
    assert meta["technology_baseline"]=="v37-chat-ui-streaming"
    assert meta["pharma_baseline"]=="v116-final-partial-comparison"
    assert meta["live_calls_executed_during_build"] is False
    assert not any(output.get("output_type")=="error" for cell in nb["cells"] for output in cell.get("outputs",[]))


def test_submission_contains_final_router_controls_rag_and_policies():
    text=all_source(load())
    for required in (
        "2026-08-18-v37",
        "class ProviderExecutionControl",
        "query_private_database",
        "route_pharma_query",
        "2026-08-19-v116-final-partial-comparison",
        'PHARMA_PARTIAL_WEIGHTS_V116={"financial_strength":0.75,"independent_sentiment":0.25}',
        "_chat_panel",
    ):
        assert required in text


def test_submission_has_only_final_live_demo_and_ui_launch():
    nb=load()
    demo=[cell for cell in nb["cells"] if cell.get("metadata",{}).get("submission_section")=="demo-query"]
    ui=[cell for cell in nb["cells"] if cell.get("metadata",{}).get("submission_section")=="ui-demo" and cell.get("cell_type")=="code"]
    assert len(demo)==1 and len(ui)==1
    assert "route_financial_query(" in "".join(demo[0]["source"])
    assert "display(_chat_panel)" in "".join(ui[0]["source"])
    assert all(not cell.get("outputs") for cell in demo+ui)


def test_submission_preserves_graded_reflection_prompts_at_end():
    tail="\n".join("".join(cell.get("source",[])) for cell in load()["cells"][-6:])
    assert "Summary and Future Scope -【2 Marks】" in tail
    assert "A. Summary / Your Observations" in tail
    assert "B. Future scope" in tail
    assert "retrieval quantity does not guarantee answer quality" in tail
    assert "Expand the pharmaceutical workflow" in tail
    assert "1.\n2.\n3." not in tail


def test_no_unresolved_code_placeholders():
    nb=load()
    unresolved=[]
    for index,cell in enumerate(nb["cells"]):
        if cell.get("cell_type")=="code" and "Your Code Goes Here" in "".join(cell.get("source",[])):
            unresolved.append(index)
    assert unresolved==[]
