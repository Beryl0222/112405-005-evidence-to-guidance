"""证据状态变化后的影响分析与跨季度报告对比。

典型用法：
    baseline = process_case(case)
    result = review_after_change(case, {"AB-7": "retracted"})
    # result["newly_needs_review"] / impacted_guidance ...
"""

from __future__ import annotations

import copy
from typing import Any

from .pipeline import process_case
from .relations import extract_relations, group_trials


def impacted_guidance_ids(report: dict[str, Any], changed_record_ids: list[str]) -> list[str]:
    """给定发生状态变化的记录，利用报告中的追踪链找出受影响建议。

    变化集合会沿关系边与试验簇扩展：
    - AB-9（更正通知）变化 → 沿 corrects 边到 AB-7；
    - AB-7 变化 → 同试验 AB-8 与引用 AB-7 的 G-3 都会被命中。
    """
    changed = set(changed_record_ids)

    # 沿关系边双向扩展一层（通知 ↔ 被通知论文）
    for rel in report.get("relations", []):
        s, t = rel["source_id"], rel["target_id"]
        if s in changed or t in changed:
            changed.add(s)
            changed.add(t)

    # 同试验簇扩展
    for trial in report.get("trials", []):
        members = set(trial["member_record_ids"])
        if members & changed:
            changed |= members

    impacted: list[str] = []
    for rev in report.get("guidance_reviews", []):
        trace = rev.get("trace", {})
        trace_ids = set(trace.get("seed_record_ids", []))
        trace_ids |= set(trace.get("member_record_ids", []))
        trace_ids |= set(trace.get("dossier_record_ids", []))
        trace_ids |= set(trace.get("trial_ids", []))
        for p in trace.get("paths", []):
            trace_ids.add(p.get("from"))
            trace_ids.add(p.get("to"))
        trace_ids.discard(None)
        if trace_ids & changed:
            impacted.append(rev["guidance_id"])
    return sorted(set(impacted))


def review_after_change(
    case: dict[str, Any],
    status_changes: dict[str, str] | list[dict[str, Any]],
) -> dict[str, Any]:
    """应用状态变更并重跑管线，同时与基线对比。

    ``status_changes`` 支持 ``{"AB-7": "retracted"}`` 或
    ``[{"record_id": "AB-7", "status": "retracted"}, ...]``。
    """
    if isinstance(status_changes, list):
        overrides = {c["record_id"]: c["status"] for c in status_changes}
        changed_ids = [c["record_id"] for c in status_changes]
    else:
        overrides = dict(status_changes)
        changed_ids = list(overrides)

    baseline = process_case(case)
    updated = process_case(case, status_overrides=overrides)
    diff = diff_reports(baseline, updated)
    impacted = impacted_guidance_ids(baseline, changed_ids)
    return {
        "changed_record_ids": changed_ids,
        "impacted_guidance_ids": impacted,
        "newly_needs_review": diff["newly_needs_review"],
        "cleared_for_maintain": diff["cleared_for_maintain"],
        "decision_changes": diff["decision_changes"],
        "baseline_queue": baseline["re_review_queue"],
        "updated_queue": updated["re_review_queue"],
        "updated_report": updated,
    }


def diff_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """对比两期报告中建议处置状态的变化。"""
    old_rev = {r["guidance_id"]: r for r in old.get("guidance_reviews", [])}
    new_rev = {r["guidance_id"]: r for r in new.get("guidance_reviews", [])}

    newly_review: list[str] = []
    cleared: list[str] = []
    decision_changes: list[dict[str, Any]] = []

    for gid in sorted(set(old_rev) | set(new_rev)):
        o = old_rev.get(gid)
        n = new_rev.get(gid)
        od = o["suggested_action"] if o else None
        nd = n["suggested_action"] if n else None
        onr = o["needs_review"] if o else False
        nnr = n["needs_review"] if n else False
        if od != nd:
            decision_changes.append(
                {"guidance_id": gid, "old_action": od, "new_action": nd,
                 "old_needs_review": onr, "new_needs_review": nnr}
            )
        if nnr and not onr:
            newly_review.append(gid)
        if onr and not nnr:
            cleared.append(gid)

    return {
        "newly_needs_review": sorted(newly_review),
        "cleared_for_maintain": sorted(cleared),
        "decision_changes": decision_changes,
    }
