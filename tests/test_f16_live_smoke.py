"""Tests for the offline F16 runner and its optional live-execution safety boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_f16_scenarios import (
    ALL_OFFLINE_SCENARIOS,
    LIVE_ADAPTER_ENV,
    LIVE_OPT_IN_ENV,
    LIVE_REQUIRED_ENV,
    PRIMARY_SCENARIOS,
    _load_live_adapter,
    compact_scenario_summary,
    live_configuration_status,
    main,
    run_live_scenarios,
    run_all_offline_scenarios,
    run_offline_scenarios,
)


def test_four_primary_offline_scenarios_pass_and_write_bounded_traces(tmp_path):
    """Exercise four primary modes through frozen F14/F15 without credentials or network."""
    summaries = run_offline_scenarios(trace_dir=tmp_path)

    assert [summary["scenario"] for summary in summaries] == [
        "single_technology",
        "single_biopharma",
        "same_profile_technology",
        "cross_profile",
    ]
    assert [summary["mode"] for summary in summaries] == [
        "single", "single", "same_profile", "cross_profile",
    ]
    assert all(summary["final_status"] == "success" for summary in summaries)
    assert all(summary["validation_valid"] for summary in summaries)
    assert all(summary["validation_error_count"] == 0 for summary in summaries)
    assert all(summary["attempts"] == 1 for summary in summaries)
    assert all((tmp_path / summary["trace_file"]).is_file() for summary in summaries)

    stored = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("*.json")]
    assert len(stored) == 4
    assert {trace["final_status"] for trace in stored} == {"success"}
    assert all(trace["validation_attempts"][0]["result"]["valid"] for trace in stored)


def test_default_runner_does_not_inspect_credentials_or_call_live_adapter(tmp_path, monkeypatch):
    """Keep the normal CLI path offline even when live-looking environment values exist."""
    monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")
    for name in LIVE_REQUIRED_ENV:
        monkeypatch.setenv(name, "fixture-secret-that-must-not-print")

    assert main(["--trace-dir", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_all_ten_offline_scenarios_are_reusable_and_have_expected_boundaries(tmp_path):
    """Expose the complete catalog for notebook demos, including bounded rejection cases."""
    summaries = run_all_offline_scenarios(trace_dir=tmp_path)
    by_name = {summary["scenario"]: summary for summary in summaries}

    assert len(ALL_OFFLINE_SCENARIOS) == len(summaries) == len(by_name) == 10
    assert list(by_name) == [
        "single_technology",
        "single_biopharma",
        "same_profile_technology",
        "same_profile_biopharma",
        "cross_profile",
        "alias_resolution",
        "unknown_company",
        "partial_rag_failure",
        "invalid_evidence_id",
        "modified_f13_score",
    ]
    assert by_name["alias_resolution"]["final_status"] == "success"
    assert by_name["partial_rag_failure"]["final_status"] == "success"
    assert by_name["unknown_company"]["final_status"] == "bounded_stop"
    assert by_name["unknown_company"]["trace_file"] is None
    for name in ("invalid_evidence_id", "modified_f13_score"):
        assert by_name[name]["final_status"] == "failed"
        assert by_name[name]["validation_valid"] is False
        assert by_name[name]["validation_error_count"] >= 1
    assert len(list(tmp_path.glob("*.json"))) == 9


def test_live_configuration_reports_names_and_presence_without_values():
    """Expose only missing variable names, never configured secret or endpoint values."""
    environment = {
        LIVE_OPT_IN_ENV: "true",
        "OPENAI_API_KEY": "secret-key",
        "OPENAI_API_BASE": "https://private-proxy.example/v1",
    }

    status = live_configuration_status(environment)

    assert status == {
        "opted_in": True,
        "configured": False,
        "missing_variables": ["TAVILY_API_KEY"],
        "required_variables": list(LIVE_REQUIRED_ENV),
    }
    serialized = json.dumps(status)
    assert "secret-key" not in serialized
    assert "private-proxy" not in serialized


def test_live_executor_is_not_called_without_explicit_opt_in():
    """Fail before any injected provider adapter can run when opt-in is absent."""
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="explicit F16_ENABLE_LIVE_TESTS=1"):
        run_live_scenarios(
            lambda spec: calls.append(spec.name),
            environ={name: "configured" for name in LIVE_REQUIRED_ENV},
        )

    assert calls == []


def test_live_executor_is_not_called_when_configuration_is_incomplete():
    """Fail before provider execution while naming missing variables but not present values."""
    calls: list[str] = []
    environment = {LIVE_OPT_IN_ENV: "1", "OPENAI_API_KEY": "do-not-expose"}

    with pytest.raises(RuntimeError, match="OPENAI_API_BASE, TAVILY_API_KEY") as exc_info:
        run_live_scenarios(
            lambda spec: calls.append(spec.name),
            environ=environment,
        )

    assert "do-not-expose" not in str(exc_info.value)
    assert calls == []


def test_explicit_configured_live_boundary_returns_only_compact_summaries():
    """Pass safe specs to an opted-in adapter and discard answer/private provider payloads."""
    calls: list[str] = []
    environment = {LIVE_OPT_IN_ENV: "yes", **{name: "configured" for name in LIVE_REQUIRED_ENV}}

    def fake_live_executor(spec):
        """Return provider-shaped output containing values that summaries must omit."""
        calls.append(spec.name)
        return {
            "final_status": "success",
            "final_answer": "FULL PRIVATE DOCUMENT and secret-key",
            "synthesis": {"mode": spec.mode, "answer": "private prose"},
            "validation": {
                "valid": True,
                "validated_evidence_ids": ["EV-safe"],
                "errors": [],
            },
            "attempts": 1,
            "correction_attempts": 0,
            "trace_path": f"/private/location/{spec.name}.json",
        }

    summaries = run_live_scenarios(fake_live_executor, environ=environment)
    serialized = json.dumps(summaries)

    assert calls == [spec.name for spec in PRIMARY_SCENARIOS]
    assert all(summary["final_status"] == "success" for summary in summaries)
    assert "FULL PRIVATE DOCUMENT" not in serialized
    assert "secret-key" not in serialized
    assert "private prose" not in serialized
    assert "/private/location" not in serialized


def test_compact_summary_counts_errors_but_does_not_emit_error_text():
    """Keep diagnostics useful without printing potentially sensitive provider messages."""
    summary = compact_scenario_summary(
        "failed_live",
        {
            "final_status": "failed",
            "synthesis": {"mode": "single"},
            "validation": {
                "valid": False,
                "validated_evidence_ids": [],
                "errors": ["secret response body"],
            },
            "attempts": 3,
            "correction_attempts": 2,
            "trace_path": "/tmp/run.json",
        },
    )

    assert summary["validation_error_count"] == 1
    assert "secret response body" not in json.dumps(summary)
    assert summary["trace_file"] == "run.json"


@pytest.mark.skipif(
    not live_configuration_status()["opted_in"],
    reason=f"Optional live smoke requires explicit {LIVE_OPT_IN_ENV}=1",
)
def test_optional_live_environment_is_complete_before_external_adapter_use():
    """Run a real injected adapter only after opt-in, credentials, and adapter are configured."""
    status = live_configuration_status()
    if not status["configured"]:
        pytest.skip("Live smoke configuration is incomplete: " + ", ".join(status["missing_variables"]))
    adapter_reference = os.environ.get(LIVE_ADAPTER_ENV)
    if not adapter_reference:
        pytest.skip(f"Optional live smoke requires {LIVE_ADAPTER_ENV}=module:function")

    summaries = run_live_scenarios(_load_live_adapter(adapter_reference))

    assert len(summaries) == len(PRIMARY_SCENARIOS)
    assert all(summary["final_status"] == "success" for summary in summaries)
    assert all(summary["validation_valid"] for summary in summaries)
