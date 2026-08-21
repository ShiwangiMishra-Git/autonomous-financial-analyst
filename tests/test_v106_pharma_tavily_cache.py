import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v106-pharma-tavily-cache.ipynb"


def _namespace():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=["".join(c.get("source") or []) for c in nb["cells"]
           if "# --- 2.4bq Shared Tavily cache and circuit breaker (v106) ---" in "".join(c.get("source") or [])]
    assert len(cells)==1
    def base_action(action,company,dimension,query): return {}
    def base_route(query,**kwargs): return {"react_trace":[]}
    namespace={"Dict":dict,"_default_react_action_runner_v89":base_action,"route_pharma_query":base_route,
       "PHARMA_REACT_WEB_POLICY_V89":{},"_collect_pharma_evidence_v43_base":lambda *a,**k:{},
       "_cached_tavily_call_v23":lambda *a,**k:{"results":[]}}
    exec(compile(cells[0],str(NOTEBOOK),"exec"),namespace)
    return namespace


def test_cache_hit_uses_one_live_call():
    ns=_namespace();ns["_PHARMA_TAVILY_RESPONSE_CACHE_V106"].clear();calls=[]
    def provider(query,topic,domains): calls.append(query);return {"results":[{"url":"https://fda.gov/a"}]}
    state={"call_budget":2,"live_calls":0,"cache_hits":0,"cache_misses":0,"circuit_open":False,
      "circuit_reason":None,"circuit_skips":0,"budget_skips":0,"dimension":"regulatory_status","events":[],"provider_fn":provider}
    token=ns["_PHARMA_TAVILY_TURN_V106"].set(state)
    try:
        ns["_pharma_tavily_search_v106"]({"query":"Pfizer FDA", "domain":"fda.gov"},1)
        ns["_pharma_tavily_search_v106"]({"query":"  pfizer   fda ","domain":"fda.gov"},2)
    finally: ns["_PHARMA_TAVILY_TURN_V106"].reset(token)
    assert len(calls)==1 and state["cache_hits"]==1 and state["live_calls"]==1


def test_permanent_failure_opens_circuit_and_prevents_second_call():
    ns=_namespace();ns["_PHARMA_TAVILY_RESPONSE_CACHE_V106"].clear();calls=[]
    class ForbiddenError(Exception): pass
    def provider(query,topic,domains): calls.append(query);raise ForbiddenError("usage limit")
    state={"call_budget":6,"live_calls":0,"cache_hits":0,"cache_misses":0,"circuit_open":False,
      "circuit_reason":None,"circuit_skips":0,"budget_skips":0,"dimension":"clinical_evidence","events":[],"provider_fn":provider}
    token=ns["_PHARMA_TAVILY_TURN_V106"].set(state)
    try:
        try: ns["_pharma_tavily_search_v106"]({"query":"first","domain":"fda.gov"},1)
        except ForbiddenError: pass
        try: ns["_pharma_tavily_search_v106"]({"query":"second","domain":"fda.gov"},2)
        except ns["ExternalProviderCircuitOpenV106"]: pass
    finally: ns["_PHARMA_TAVILY_TURN_V106"].reset(token)
    assert len(calls)==1 and state["live_calls"]==1 and state["circuit_open"] and state["circuit_skips"]==1


def test_tavily_budget_is_separate_and_fail_closed():
    ns=_namespace();ns["_PHARMA_TAVILY_RESPONSE_CACHE_V106"].clear()
    state={"call_budget":0,"live_calls":0,"cache_hits":0,"cache_misses":0,"circuit_open":False,
      "circuit_reason":None,"circuit_skips":0,"budget_skips":0,"dimension":"news_sentiment","events":[],"provider_fn":lambda *a:None}
    token=ns["_PHARMA_TAVILY_TURN_V106"].set(state)
    try:
        try: ns["_pharma_tavily_search_v106"]({"query":"news","domain":"reuters.com"},1)
        except ns["TavilyCallBudgetPermissionV106"]: pass
    finally: ns["_PHARMA_TAVILY_TURN_V106"].reset(token)
    assert state["live_calls"]==0 and state["budget_skips"]==1
