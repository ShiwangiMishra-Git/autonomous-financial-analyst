import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v105-qualitative-patent-policy.ipynb"


def test_patent_excluded_and_weights_renormalized():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bp Qualitative-only patent scoring policy (v105) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    weights={"financial_strength":.25,"pipeline_diversification":.25,"clinical_maturity":.15,
             "regulatory_position":.10,"commercial_concentration":.10,
             "patent_exclusivity_risk":.10,"independent_sentiment":.05}
    def validate(company,name,observation,registry):
        score=observation.get("score")
        return {"status":"scored" if score is not None else "N/A","score":score,"weight":weights[name],
                "reason":"","source_ids":["x"],"source_roles":[],"period":None,"submetrics":{},
                "missing":[] if score is not None else ["valid_0_to_5_score"]}
    namespace={"Dict":dict,"List":list,"deepcopy":__import__("copy").deepcopy,
               "PHARMA_WEIGHTS_V83":weights,"PHARMA_REACT_DIMENSIONS_V89":tuple(weights),
               "PHARMA_ALLOWED_ROLES_V83":{d:{"company_primary"} for d in weights},
               "PHARMA_DIMENSION_LABELS_V83":{d:d for d in weights},
               "PHARMA_SCORING_LABEL_V83":"Educational prototype",
               "_validate_dimension_v83":validate}
    exec(compile(cells[0],str(NOTEBOOK),"exec"),namespace)
    observations={d:{"score":5} for d in namespace["PHARMA_NUMERIC_DIMENSIONS_V105"]}
    result=namespace["score_pharma_comparison_v83"](["PFE"],{"PFE":observations},{})
    assert result["ranking_status"]=="available"
    assert result["company_scores"]["PFE"]["overall_score"]==100.0
    assert result["company_scores"]["PFE"]["dimensions"]["patent_exclusivity_risk"]["status"]=="qualitative_only"
    assert "patent_exclusivity_risk" not in namespace["PHARMA_REACT_DIMENSIONS_V89"]
    rendered=namespace["render_pharma_scores_v83"](result)
    assert "Qualitative only" in rendered and "normalized from 90% to 100%" in rendered


def test_missing_core_dimension_still_withholds_ranking():
    # Static contract protects against treating missing eligible evidence as zero.
    text=NOTEBOOK.read_text(encoding="utf-8")
    assert 'overall=None if missing else' in text
    assert 'missing_evidence_scored_as_zero' in text
