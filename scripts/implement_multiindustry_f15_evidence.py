"""Provide the isolated F15 deterministic final-validation notebook source.

This module intentionally does not edit the working notebook.  The central integration step can
inject :data:`F15_EVIDENCE_CODE` after reconciling it with the local-trace implementation.
"""

from __future__ import annotations


F15_EVIDENCE_INTRO = """### F15.1: Deterministic Evidence and Output Validation

Before a synthesized draft may become the final answer, F15 verifies every explicit `[EV-*]`
citation against current-run normalized F12 evidence. It also checks immutable F13 score use,
comparison-mode boundaries, and mandatory structured limitations. These checks prove provenance
and contract fidelity; they do not claim that every prose statement is semantically supported.
"""


F15_EVIDENCE_CODE = r'''from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence, TypedDict


class ValidationResult(TypedDict):
    """Deterministic F15 verdict for one F14 synthesis result.

    Attributes:
        valid: Aggregate pass required before returning an answer.
        validated_evidence_ids: Current-run IDs that passed all ownership checks.
        inline_evidence_ids: IDs parsed from explicit ``[EV-*]`` answer citations.
        declared_evidence_ids: IDs listed by the structured synthesis result.
        evidence_valid: Existence, ownership, status, duplicate, and consistency verdict.
        score_fidelity_valid: Exact authoritative F13 score-use verdict.
        mode_restrictions_valid: Single/same/cross-profile boundary verdict.
        limitations_valid: Required limitation disclosure verdict.
        errors: Deterministic failure explanations used by bounded correction.
    """

    valid: bool
    validated_evidence_ids: list[str]
    inline_evidence_ids: list[str]
    declared_evidence_ids: list[str]
    evidence_valid: bool
    score_fidelity_valid: bool
    mode_restrictions_valid: bool
    limitations_valid: bool
    errors: list[str]


F15_VALID_MODES = frozenset({"single", "same_profile", "cross_profile"})
F15_USABLE_EVIDENCE_STATUSES = frozenset({"success"})
F15_INLINE_EVIDENCE_PATTERN = re.compile(
    r"\[((?i:EV)-[A-Za-z0-9][A-Za-z0-9._:-]*)\]"
)
F15_TICKER_SCORE_CLAIM_PATTERN = re.compile(
    r"\b(?P<ticker>[A-Z][A-Z0-9.]{0,7})\b[^.\n]{0,48}?"
    r"\b(?:total\s+)?score(?:d)?\s*(?:of|is|=|:)?\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
F15_NUMERIC_RANK_CLAIM_PATTERN = re.compile(
    r"\b(?P<ticker>[A-Z][A-Z0-9.]{0,7})\b[^.\n]{0,48}?"
    r"\brank(?:ed|ing)?\s*(?:at|as|is|=|:|#)?\s*"
    r"(?P<rank>\d+|first|second|third)\b",
    re.IGNORECASE,
)


def _f15_nonempty_text(value: Any, field_name: str) -> str | None:
    """Return stripped text or ``None``; validation callers add contextual errors."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _f15_unique_errors(errors: Sequence[str]) -> list[str]:
    """Remove repeated errors while preserving deterministic discovery order."""
    return list(dict.fromkeys(errors))


def extract_inline_evidence_ids(answer: Any) -> list[str]:
    """Extract ordered explicit ``[EV-*]`` citation tokens from answer prose.

    Only the citation syntax is inspected. This function does not infer whether nearby prose is
    entailed by the cited record and therefore makes no semantic-grounding claim.
    """
    if not isinstance(answer, str):
        return []
    return F15_INLINE_EVIDENCE_PATTERN.findall(answer)


def _f15_validate_prose_score_claims(
    answer: Any,
    mode: str | None,
    authoritative_scores: Mapping[str, Any],
) -> list[str]:
    """Validate only explicit machine-recognizable score and rank claims in prose.

    This deliberately narrow check does not infer recommendations, compare ordinary financial
    numbers, or attempt semantic entailment. It catches explicit ``TICKER score N`` and
    ``TICKER ranked N`` statements that can be compared with the authoritative F13 table.
    """
    if not isinstance(answer, str):
        return []
    errors: list[str] = []
    score_claims = list(F15_TICKER_SCORE_CLAIM_PATTERN.finditer(answer))
    rank_claims = list(F15_NUMERIC_RANK_CLAIM_PATTERN.finditer(answer))
    if mode in {"single", "cross_profile"}:
        if score_claims:
            errors.append(f"{mode} synthesis contains a prohibited numeric score claim")
        if rank_claims:
            errors.append(f"{mode} synthesis contains a prohibited numeric ranking claim")
        return errors
    if mode != "same_profile":
        return errors

    for match in score_claims:
        ticker = match.group("ticker").upper()
        score_record = authoritative_scores.get(ticker)
        if not isinstance(score_record, Mapping) or "total_score" not in score_record:
            errors.append(f"Prose score claim for {ticker} has no authoritative F13 total_score")
            continue
        try:
            claimed = float(match.group("value"))
            authoritative = float(score_record["total_score"])
        except (TypeError, ValueError):
            errors.append(f"Authoritative F13 total_score for {ticker} is not comparable")
            continue
        if claimed != authoritative:
            errors.append(
                f"Prose score claim for {ticker} ({claimed:g}) does not match "
                f"authoritative F13 total_score ({authoritative:g})"
            )
    for match in rank_claims:
        ticker = match.group("ticker").upper()
        score_record = authoritative_scores.get(ticker)
        rank_text = match.group("rank").casefold()
        rank_map = {"first": 1, "second": 2, "third": 3}
        claimed_rank = rank_map.get(rank_text, int(rank_text) if rank_text.isdigit() else None)
        if not isinstance(score_record, Mapping) or "rank" not in score_record:
            errors.append(f"Prose rank claim for {ticker} has no authoritative F13 rank")
            continue
        if claimed_rank != score_record["rank"]:
            errors.append(
                f"Prose rank claim for {ticker} ({claimed_rank}) does not match "
                f"authoritative F13 rank ({score_record['rank']})"
            )
    return errors


def _f15_required_limitations(
    normalized_results: Mapping[str, Any],
    mode: str | None,
    scoring_eligibility: Mapping[str, Any],
    authoritative_scores: Mapping[str, Any],
) -> list[str]:
    """Rebuild F14's deterministic mandatory limitations without trusting model output."""
    limitations: list[str] = []
    for ticker, result in normalized_results.items():
        if not isinstance(result, Mapping):
            continue
        status = result.get("status")
        if status != "success":
            limitations.append(f"{ticker}: result status is {status}.")
        missing = result.get("missing_dimensions", [])
        if isinstance(missing, list) and missing:
            limitations.append(
                f"{ticker}: missing dimensions: {', '.join(map(str, missing))}."
            )
        result_errors = result.get("errors", [])
        if isinstance(result_errors, list) and result_errors:
            limitations.append(
                f"{ticker}: research limitations: {'; '.join(map(str, result_errors))}."
            )
    if mode == "single":
        limitations.append("Single-company analysis does not use a comparison score.")
    elif mode == "cross_profile":
        limitations.append("No universal numeric score was applied across industry profiles.")
    elif mode == "same_profile" and scoring_eligibility.get("eligible") is not True:
        reason = str(
            scoring_eligibility.get("reason") or "Scoring eligibility did not pass."
        )
        limitations.append(f"Numeric sector scoring was not applied: {reason}")
    elif mode == "same_profile" and not authoritative_scores:
        limitations.append(
            "Scoring was eligible, but no authoritative F13 score table was supplied."
        )
    return list(dict.fromkeys(limitations))


def _f15_catalog_evidence(
    run_id: str | None,
    normalized_results: Any,
    errors: list[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, bool], str | None]:
    """Validate F12 evidence ownership and build an unambiguous ID catalog."""
    catalog: dict[str, Mapping[str, Any]] = {}
    usable: dict[str, bool] = {}
    duplicate_ids: set[str] = set()
    profile_ids: set[str] = set()

    if not isinstance(normalized_results, Mapping) or not normalized_results:
        errors.append("normalized_results must be a non-empty ticker mapping")
        return catalog, usable, None

    for raw_ticker, result in normalized_results.items():
        ticker = _f15_nonempty_text(raw_ticker, "result ticker")
        if ticker is None or not isinstance(result, Mapping):
            errors.append(f"Normalized result {raw_ticker!r} is malformed")
            continue
        result_run_id = _f15_nonempty_text(result.get("run_id"), f"{ticker}.run_id")
        profile_id = _f15_nonempty_text(result.get("profile_id"), f"{ticker}.profile_id")
        company = result.get("company")
        if result_run_id != run_id:
            errors.append(f"Result {ticker} does not belong to current run {run_id!r}")
        if profile_id is None:
            errors.append(f"Result {ticker} is missing profile_id")
        else:
            profile_ids.add(profile_id)
        if not isinstance(company, Mapping):
            errors.append(f"Result {ticker} is missing canonical company identity")
            company = {}
        company_ticker = _f15_nonempty_text(company.get("ticker"), f"{ticker}.company.ticker")
        company_id = _f15_nonempty_text(company.get("company_id"), f"{ticker}.company.company_id")
        company_profile = _f15_nonempty_text(
            company.get("profile_id"), f"{ticker}.company.profile_id"
        )
        if company_ticker != ticker:
            errors.append(f"Result {ticker} has the wrong company ticker ownership")
        if company_id is None:
            errors.append(f"Result {ticker} is missing company_id")
        if company_profile != profile_id:
            errors.append(f"Result {ticker} has inconsistent company/profile ownership")

        records = result.get("evidence")
        if not isinstance(records, list):
            errors.append(f"Result {ticker} evidence must be a list")
            continue
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                errors.append(f"Result {ticker} evidence[{index}] is malformed")
                continue
            evidence_id = _f15_nonempty_text(
                record.get("evidence_id"), f"{ticker}.evidence[{index}].evidence_id"
            )
            if evidence_id is None:
                errors.append(f"Result {ticker} evidence[{index}] is missing evidence_id")
                continue
            if evidence_id in catalog:
                duplicate_ids.add(evidence_id)
                usable[evidence_id] = False
                continue
            catalog[evidence_id] = record
            ownership_valid = True
            if record.get("run_id") != run_id or record.get("run_id") != result_run_id:
                errors.append(f"Evidence {evidence_id} has the wrong run_id ownership")
                ownership_valid = False
            if record.get("ticker") != ticker:
                errors.append(f"Evidence {evidence_id} has the wrong company ticker ownership")
                ownership_valid = False
            if record.get("company_id") != company_id:
                errors.append(f"Evidence {evidence_id} has the wrong company_id ownership")
                ownership_valid = False
            if record.get("profile_id") != profile_id:
                errors.append(f"Evidence {evidence_id} has the wrong profile ownership")
                ownership_valid = False
            status_valid = record.get("status") in F15_USABLE_EVIDENCE_STATUSES
            usable[evidence_id] = ownership_valid and status_valid

    for evidence_id in sorted(duplicate_ids):
        errors.append(f"Duplicate evidence_id {evidence_id} is ambiguous")
        usable[evidence_id] = False

    if len(normalized_results) == 1:
        expected_mode = "single"
    elif len(profile_ids) == 1 and profile_ids:
        expected_mode = "same_profile"
    else:
        expected_mode = "cross_profile"
    return catalog, usable, expected_mode


def validate_synthesis_result(
    run_id: Any,
    normalized_results: Any,
    synthesis_result: Any,
    *,
    authoritative_scores: Mapping[str, Any] | None = None,
    scoring_eligibility: Mapping[str, Any] | None = None,
    required_limitations: Sequence[str] | None = None,
) -> ValidationResult:
    """Validate one F14 result against current-run F12/F13 deterministic authority.

    The validator checks explicit citation provenance, exact score-table fidelity, routing-mode
    restrictions, and required structured limitations. It performs no research, arithmetic, or
    semantic proof of prose grounding.
    """
    errors: list[str] = []
    current_run_id = _f15_nonempty_text(run_id, "run_id")
    if current_run_id is None:
        errors.append("run_id must be a non-empty string")

    catalog, usable, expected_mode = _f15_catalog_evidence(
        current_run_id, normalized_results, errors
    )
    if not isinstance(synthesis_result, Mapping):
        errors.append("synthesis_result must be a mapping")
        synthesis_result = {}

    mode = synthesis_result.get("mode")
    mode_restrictions_valid = True
    if mode not in F15_VALID_MODES:
        errors.append(f"Unsupported synthesis mode: {mode!r}")
        mode_restrictions_valid = False
    elif expected_mode is not None and mode != expected_mode:
        errors.append(
            f"Synthesis mode {mode!r} does not match normalized result mode {expected_mode!r}"
        )
        mode_restrictions_valid = False

    answer = synthesis_result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("Synthesis answer must be a non-empty string")
    inline_ids = extract_inline_evidence_ids(answer)
    raw_declared = synthesis_result.get("evidence_ids")
    declared_ids: list[str] = []
    if not isinstance(raw_declared, list):
        errors.append("Synthesis evidence_ids must be a list")
    else:
        for index, value in enumerate(raw_declared):
            evidence_id = _f15_nonempty_text(value, f"evidence_ids[{index}]")
            if evidence_id is None:
                errors.append(f"Synthesis evidence_ids[{index}] must be a non-empty string")
            else:
                declared_ids.append(evidence_id)

    if len(inline_ids) != len(set(inline_ids)):
        errors.append("Inline citations contain duplicate evidence IDs")
    if len(declared_ids) != len(set(declared_ids)):
        errors.append("Synthesis evidence_ids contains duplicates")
    if inline_ids != declared_ids:
        errors.append(
            "Inline [EV-*] citations must exactly match SynthesisResult.evidence_ids in order"
        )
    if any(usable.values()) and not declared_ids:
        errors.append("Synthesis must cite current-run usable evidence when it is available")

    validated_ids: list[str] = []
    for evidence_id in declared_ids:
        record = catalog.get(evidence_id)
        if record is None:
            errors.append(f"Cited evidence ID {evidence_id} does not exist in current-run evidence")
            continue
        if record.get("status") not in F15_USABLE_EVIDENCE_STATUSES:
            errors.append(
                f"Cited evidence ID {evidence_id} has unusable status {record.get('status')!r}"
            )
            continue
        if not usable.get(evidence_id, False):
            errors.append(f"Cited evidence ID {evidence_id} failed ownership validation")
            continue
        if evidence_id not in validated_ids:
            validated_ids.append(evidence_id)

    evidence_error_markers = (
        "evidence", "citation", "run_id", "ownership", "company", "profile",
        "normalized result", "synthesis answer",
    )
    evidence_valid = not any(
        any(marker in error.casefold() for marker in evidence_error_markers)
        for error in errors
    )

    if authoritative_scores is None:
        authoritative: dict[str, Any] = {}
    elif isinstance(authoritative_scores, Mapping):
        authoritative = deepcopy(dict(authoritative_scores))
    else:
        authoritative = {}
        errors.append("authoritative_scores must be a mapping or None")
    raw_scores_used = synthesis_result.get("scores_used", {})
    scores_used = deepcopy(dict(raw_scores_used)) if isinstance(raw_scores_used, Mapping) else None
    score_errors_before = len(errors)
    if scores_used is None:
        errors.append("Synthesis scores_used must be a mapping")
    elif mode in {"single", "cross_profile"}:
        if scores_used:
            errors.append(f"{mode} synthesis cannot use comparison scores")
            mode_restrictions_valid = False
        if authoritative:
            errors.append(f"{mode} validation cannot receive authoritative comparison scores")
            mode_restrictions_valid = False
    elif mode == "same_profile":
        if authoritative and set(authoritative) != set(normalized_results or {}):
            errors.append("Authoritative F13 scores must cover the normalized company set")
        if scores_used != authoritative:
            errors.append("Synthesis scores_used does not exactly match authoritative F13 scores")
    prose_score_errors = _f15_validate_prose_score_claims(answer, mode, authoritative)
    if prose_score_errors:
        errors.extend(prose_score_errors)
        if mode in {"single", "cross_profile"}:
            mode_restrictions_valid = False
    score_fidelity_valid = len(errors) == score_errors_before

    if scoring_eligibility is None:
        eligibility: dict[str, Any] = {}
    elif isinstance(scoring_eligibility, Mapping):
        eligibility = dict(scoring_eligibility)
    else:
        eligibility = {}
        errors.append("scoring_eligibility must be a mapping or None")
    if mode == "same_profile" and authoritative and eligibility.get("eligible") is not True:
        errors.append("Authoritative F13 scores require eligible same-profile scoring context")
        mode_restrictions_valid = False
    limitation_errors_before = len(errors)
    raw_limitations = synthesis_result.get("limitations")
    limitations: list[str] = []
    if not isinstance(raw_limitations, list):
        errors.append("Synthesis limitations must be a list")
    else:
        for index, value in enumerate(raw_limitations):
            limitation = _f15_nonempty_text(value, f"limitations[{index}]")
            if limitation is None:
                errors.append(f"Synthesis limitations[{index}] must be a non-empty string")
            else:
                limitations.append(limitation)
    if len(limitations) != len(set(limitations)):
        errors.append("Synthesis limitations contains duplicates")

    mandatory = _f15_required_limitations(
        normalized_results if isinstance(normalized_results, Mapping) else {},
        mode if isinstance(mode, str) else None,
        eligibility,
        authoritative,
    )
    if required_limitations is not None:
        if isinstance(required_limitations, (str, bytes)):
            errors.append("required_limitations must be a sequence of strings")
        elif not isinstance(required_limitations, Sequence):
            errors.append("required_limitations must be a sequence of strings")
        else:
            for index, value in enumerate(required_limitations):
                limitation = _f15_nonempty_text(value, f"required_limitations[{index}]")
                if limitation is None:
                    errors.append(
                        f"required_limitations[{index}] must be a non-empty string"
                    )
                elif limitation not in mandatory:
                    mandatory.append(limitation)
    missing_limitations = [item for item in mandatory if item not in limitations]
    if missing_limitations:
        errors.append(f"Synthesis omitted required limitations: {missing_limitations}")
    limitations_valid = len(errors) == limitation_errors_before

    errors = _f15_unique_errors(errors)
    return {
        "valid": not errors,
        "validated_evidence_ids": validated_ids,
        "inline_evidence_ids": inline_ids,
        "declared_evidence_ids": declared_ids,
        "evidence_valid": evidence_valid,
        "score_fidelity_valid": score_fidelity_valid,
        "mode_restrictions_valid": mode_restrictions_valid,
        "limitations_valid": limitations_valid,
        "errors": errors,
    }


print("✅ F15 deterministic evidence and output validation defined")
'''
