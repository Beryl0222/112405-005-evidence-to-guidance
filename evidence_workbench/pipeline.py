"""端到端处理：case dict -> 可复核证据链报告 dict。

管线是纯函数、确定性的：同一份输入反复运行得到逐字节一致的 JSON。
"""

from __future__ import annotations

import copy
from typing import Any

from .audit import audit_wording
from .guidance import review_guidance
from .models import Claim
from .nlp import extract_claims_from_text
from .relations import (
    compute_statuses,
    extract_relations,
    group_trials,
    normalize_kind,
)

DISCLAIMER = "本结果基于虚构合成文本生成，仅用于流程演练，不构成任何真实医疗建议。"


def load_case_dict(payload: Any) -> dict[str, Any]:
    """兼容几种常见外层结构。"""
    if isinstance(payload, list):
        return {"case_id": "inline-list", "records": payload, "guidance": []}
    if not isinstance(payload, dict):
        raise ValueError("case 必须是 JSON 对象或记录数组")
    case = dict(payload)
    for key in ("records", "sources", "documents", "items"):
        if key in case and isinstance(case[key], list) and key != "records":
            case.setdefault("records", case[key])
    case.setdefault("records", [])
    for key in ("guidance", "recommendations", "advice", "published_guidance", "guidance_items", "建议"):
        if key in case and isinstance(case[key], list) and key != "guidance":
            case.setdefault("guidance", case[key])
    case.setdefault("guidance", [])
    return case


def _record_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "body", "abstract", "summary", "摘要", "正文", "内容"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _guidance_text(g: dict[str, Any]) -> str:
    for key in ("text", "recommendation", "statement", "content", "advice", "建议", "建议正文"):
        val = g.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _build_claims(record: dict[str, Any]) -> list[dict[str, Any]]:
    rid = record["id"]
    text = _record_text(record)
    raw_claims = extract_claims_from_text(text)

    # 合并输入中人工/上游结构化给出的 claims（保证机器抽取不完整时链路仍可复核）
    explicit = record.get("claims") if isinstance(record.get("claims"), list) else []
    merged: list[dict[str, Any]] = []
    for i, c in enumerate(raw_claims, start=1):
        claim = Claim(claim_id=f"{rid}#C{i}", record_id=rid, **{k: v for k, v in c.items()})
        merged.append(claim.to_dict())

    base_n = len(merged)
    for j, ec in enumerate(explicit, start=1):
        if not isinstance(ec, dict):
            continue
        sentence = ec.get("source_sentence") or text
        clause = ec.get("source_clause") or sentence
        claim = Claim(
            claim_id=ec.get("claim_id") or f"{rid}#X{j}",
            record_id=rid,
            claim_type=ec.get("claim_type", "result"),
            source_sentence=sentence,
            source_clause=clause,
            population=ec.get("population"),
            population_qualifiers=ec.get("population_qualifiers", []),
            intervention=ec.get("intervention"),
            comparator=ec.get("comparator"),
            comparator_status=ec.get("comparator_status", "reported" if ec.get("comparator") else "not_specified"),
            outcome=ec.get("outcome"),
            outcome_concept=ec.get("outcome_concept"),
            outcome_class=ec.get("outcome_class", "unknown"),
            direction=ec.get("direction", "unclear"),
            evidential_force=ec.get("evidential_force", "unknown"),
            direction_note=ec.get("direction_note"),
            effect_qualifiers=ec.get("effect_qualifiers", []),
            confidence=ec.get("confidence", "high"),
            extraction={"source": "explicit_input"},
        )
        d = claim.to_dict()
        # 校验原句确实来自原文
        d["source_verified"] = bool(sentence and sentence in text)
        merged.append(d)

    # NLP claim 也做一次原句校验（理论上必然通过）
    for d in merged[:base_n]:
        d["source_verified"] = d["source_sentence"] in text
    return merged


def process_case(case: dict[str, Any], *, status_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """处理一个 case。``status_overrides`` 可模拟季度更新中的状态变更。"""
    case = load_case_dict(case)
    records: list[dict[str, Any]] = []
    input_warnings: list[dict[str, Any]] = []

    seen_ids: set[str] = set()
    for raw in case["records"]:
        if not isinstance(raw, dict) or not raw.get("id"):
            input_warnings.append({"code": "record_missing_id", "record": str(raw)[:80]})
            continue
        rid = str(raw["id"])
        if rid in seen_ids:
            input_warnings.append({"code": "duplicate_record_id", "record_id": rid})
        seen_ids.add(rid)
        rec = copy.deepcopy(raw)
        rec["id"] = rid
        rec["text"] = _record_text(rec)
        rec["kind_normalized"] = normalize_kind(rec.get("kind") or rec.get("source"))
        records.append(rec)

    # 允许把 kind=guidance 的记录放在 records 里
    guidance: list[dict[str, Any]] = []
    for g in case.get("guidance", []):
        if not isinstance(g, dict):
            continue
        gid = g.get("id") or g.get("guidance_id") or g.get("recommendation_id")
        if not gid:
            continue
        g = dict(g)
        g["id"] = str(gid)
        g.setdefault("text", _guidance_text(g))
        guidance.append(g)
    guidance_ids = {g["id"] for g in guidance}
    for rec in list(records):
        if rec.get("kind_normalized") == "guidance" and rec["id"] not in guidance_ids:
            guidance.append({"id": rec["id"], "text": rec["text"]})

    # 1) 主张抽取 --------------------------------------------------------
    claims_by_record: dict[str, list[dict[str, Any]]] = {}
    all_claims: list[dict[str, Any]] = []
    for rec in records:
        cs = _build_claims(rec)
        for c in cs:
            c["record_kind"] = rec.get("kind_normalized")
        claims_by_record[rec["id"]] = cs
        all_claims.extend(cs)

    # 2) 关系 / 试验归并 / 状态 -----------------------------------------
    relations = extract_relations(records)
    trial_info = group_trials(records, relations)
    trials = trial_info["trials"]
    record_trial = trial_info["record_trial"]
    status_info = compute_statuses(records, relations, status_overrides)
    statuses = status_info["statuses"]

    records_view = [
        {
            "id": r["id"],
            "kind": r.get("kind"),
            "kind_normalized": r["kind_normalized"],
            "source": r.get("source"),
            "date": r.get("date"),
            "trial_id": record_trial.get(r["id"]),
            "status": statuses.get(r["id"], "active"),
            "text": r["text"],
        }
        for r in records
    ]

    # 3) 建议复核 --------------------------------------------------------
    guidance_reviews = [
        review_guidance(
            g,
            records=records,
            claims_by_record=claims_by_record,
            relations=relations,
            trials=trials,
            record_trial=record_trial,
            statuses=statuses,
        )
        for g in guidance
    ]

    # 4) 重审队列 --------------------------------------------------------
    priority_rank = {"critical": 0, "high": 1, "medium": 2, None: 3}
    queue = []
    for rev in guidance_reviews:
        if not rev["needs_review"]:
            continue
        queue.append(
            {
                "guidance_id": rev["guidance_id"],
                "suggested_action": rev["suggested_action"],
                "review_priority": rev["review_priority"],
                "triggers": [r["code"] for r in rev["reasons"]],
                "independent_study_count": rev["trace"]["independent_study_count"],
                "trial_ids": rev["trace"]["trial_ids"],
                "member_record_ids": rev["trace"]["member_record_ids"],
                "dossier_record_ids": rev["trace"].get("dossier_record_ids", []),
                "top_reasons_zh": [r["message_zh"] for r in rev["reasons"]],
            }
        )
    queue.sort(key=lambda q: (priority_rank.get(q["review_priority"], 3), q["guidance_id"]))

    # 5) 表述审计（建议正文 + 全部来源文本，含会议批注） -----------------
    wording_issues: list[dict[str, Any]] = []
    for g in guidance:
        wording_issues.extend(audit_wording(g.get("text", ""), source_id=g["id"]))
    for r in records:
        for issue in audit_wording(r["text"], source_id=r["id"]):
            wording_issues.append(issue)

    # 6) 汇总统计 --------------------------------------------------------
    action_counts = {"maintain": 0, "narrow": 0, "reopen": 0}
    for rev in guidance_reviews:
        action_counts[rev["suggested_action"]] += 1
    stats = {
        "record_count": len(records),
        "claim_count": len(all_claims),
        "trial_count": len(trials),
        "independent_study_count_total": len(trials),
        "guidance_count": len(guidance_reviews),
        "guidance_action_counts": action_counts,
        "re_review_queue_size": len(queue),
        "wording_issue_count": len(wording_issues),
        "unverified_source_sentences": [
            c["claim_id"] for c in all_claims if c.get("source_verified") is False
        ],
    }

    return {
        "schema_version": "1.0",
        "case_id": case.get("case_id", "unknown-case"),
        "disclaimer": DISCLAIMER,
        "records": records_view,
        "claims": all_claims,
        "trials": trials,
        "relations": relations,
        "status_events": status_info["events"],
        "guidance_reviews": guidance_reviews,
        "re_review_queue": queue,
        "wording_issues": wording_issues,
        "input_warnings": input_warnings,
        "stats": stats,
    }
