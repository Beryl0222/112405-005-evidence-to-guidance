"""把报告 dict 渲染成学科审阅者/起草人可读的中文 Markdown，并导出 CSV。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_ACTION_EMOJI = {"maintain": "🟢", "narrow": "🟡", "reopen": "🔴"}
_CHECK_ICON = {"pass": "✅", "warn": "⚠️", "block": "⛔", "info": "ℹ️"}

_DIRECTION_ZH = {
    "favorable": "有利",
    "unfavorable": "不利/有害",
    "no_significant_difference": "未发现差异（不确定，≠无效证明）",
    "no_effect_demonstrated": "有强证据表明无临床意义效应",
    "mixed": "混合",
    "not_assessed": "未测量（证据缺口）",
    "unclear": "不明确",
}
_OUTCOME_CLASS_ZH = {
    "patient_important": "市民/临床硬结局",
    "surrogate": "替代指标",
    "safety": "安全性",
    "symptom_or_function": "症状/功能",
    "unknown": "未分类",
}


def write_csv_exports(report: dict[str, Any], outdir: str | Path) -> list[Path]:
    """导出主张明细与建议复核两张 CSV（utf-8-sig，Excel 可直接打开）。"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    claims_path = out / "claims.csv"
    with claims_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["主张ID", "来源记录", "主张类型", "研究人群", "人群限定", "干预",
                    "比较条件", "结局", "结局概念", "结局类别", "方向", "证据含义",
                    "方向说明", "来源原句", "命题分句", "原句校验"])
        for c in report["claims"]:
            q = "；".join(f"{x.get('type')}:{x.get('raw') or x.get('condition') or x.get('min_age')}"
                          for x in c.get("population_qualifiers", []))
            w.writerow([
                c["claim_id"], c["record_id"], c.get("claim_type"),
                c.get("population") or "", q, c.get("intervention") or "",
                c.get("comparator") or "原文未说明", c.get("outcome") or "",
                c.get("outcome_concept") or "",
                _OUTCOME_CLASS_ZH.get(c.get("outcome_class"), c.get("outcome_class")),
                _DIRECTION_ZH.get(c.get("direction"), c.get("direction")),
                c.get("evidential_force", ""), c.get("direction_note") or "",
                c.get("source_sentence", ""), c.get("source_clause", ""),
                "yes" if c.get("source_verified", True) else "NO",
            ])

    review_path = out / "guidance_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["建议ID", "建议正文", "机器处置", "处置中文", "是否需要重审",
                    "优先级", "独立试验数", "试验ID", "全部成员记录", "档案材料",
                    "触发代码", "主要原因"])
        for r in report["guidance_reviews"]:
            t = r["trace"]
            w.writerow([
                r["guidance_id"], r["guidance_text"], r["suggested_action"],
                r["suggested_action_zh"], "是" if r["needs_review"] else "否",
                r.get("review_priority") or "", t["independent_study_count"],
                ";".join(t["trial_ids"]), ";".join(t["member_record_ids"]),
                ";".join(t.get("dossier_record_ids", [])),
                ";".join(x["code"] for x in r["reasons"]),
                " / ".join(x["message_zh"] for x in r["reasons"]),
            ])
    return [claims_path, review_path]


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    s = str(value).replace("|", "\\|").replace("\n", " ")
    return s if s else "—"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 证据链与建议复核报告（case: {report['case_id']}）")
    lines.append("")
    lines.append(f"> {report['disclaimer']}")
    lines.append("")
    st = report["stats"]
    lines.append(
        f"记录 {st['record_count']} 条｜主张 {st['claim_count']} 项｜"
        f"独立试验 {st['trial_count']} 项｜建议 {st['guidance_count']} 条｜"
        f"待重审 {st['re_review_queue_size']} 条｜表述问题 {st['wording_issue_count']} 项"
    )
    lines.append("")

    # 重审队列（放最前，秘书最关心）
    lines.append("## 一、需要重审的建议（优先队列）")
    lines.append("")
    if not report["re_review_queue"]:
        lines.append("当前没有需要重审的建议。")
    else:
        lines.append("| 优先级 | 建议 | 机器处置 | 触发代码 | 关联试验 |")
        lines.append("|---|---|---|---|---|")
        for q in report["re_review_queue"]:
            lines.append(
                f"| {_cell(q['review_priority'])} | {q['guidance_id']} | "
                f"{_ACTION_EMOJI.get(q['suggested_action'], '')} {q['suggested_action']} | "
                f"{_cell(', '.join(q['triggers']))} | {_cell(', '.join(q['trial_ids']))} |"
            )
    lines.append("")

    # 主张证据链总表
    lines.append("## 二、主张级证据链（人群 / 比较 / 结局 / 方向 / 来源原句）")
    lines.append("")
    lines.append("| 主张ID | 来源 | 人群 | 比较条件 | 干预 | 结局（类别） | 方向 | 来源原句 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in report["claims"]:
        outcome = c.get("outcome") or "—"
        oc = c.get("outcome_class", "unknown")
        direction = c.get("direction", "unclear")
        if c.get("evidential_force") == "inconclusive_no_detected_difference":
            direction += "（≠证明无效）"
        if c.get("direction") == "not_assessed":
            direction = "not_assessed（未测量，证据缺口）"
        lines.append(
            "| {cid} | {rid} | {pop} | {comp} | {iv} | {out}（{oc}） | {d} | “{sent}” |".format(
                cid=c["claim_id"], rid=c["record_id"], pop=_cell(c.get("population")),
                comp=_cell(c.get("comparator") if c.get("comparator") else "原文未说明"),
                iv=_cell(c.get("intervention")), out=_cell(outcome), oc=oc,
                d=direction, sent=_cell(c.get("source_sentence")),
            )
        )
    lines.append("")

    # 试验归并
    lines.append("## 三、同一试验归并与独立性")
    lines.append("")
    for t in report["trials"]:
        lines.append(
            f"- **{t['trial_id']}**：主记录 {t['primary_record_id']}，"
            f"全部记录 {', '.join(t['member_record_ids'])}；"
            f"独立研究计数 = {t['independent_study_count']}"
            + ("；含后续/随访分析，不得作为第二项独立研究。" if t["has_follow_up_analysis"] else "。")
        )
    lines.append("")
    if report["relations"]:
        lines.append("识别到的记录关系：")
        lines.append("")
        for r in report["relations"]:
            ev = f"；依据原句“{_cell(r['evidence_sentence'])}”" if r.get("evidence_sentence") else "；依据结构化字段/类型"
            lines.append(f"- {r['source_id']} —{r['relation_type']}→ {r['target_id']}（{r['confidence']}{ev}）")
        lines.append("")

    # 来源状态
    lines.append("## 四、来源状态（更正 / 撤回传播）")
    lines.append("")
    changed = [r for r in report["records"] if r["status"] != "active"]
    if not changed:
        lines.append("所有来源均为 active。")
    else:
        for r in changed:
            lines.append(f"- **{r['id']}**：`{r['status']}`（kind={r['kind_normalized']}）")
    if report["status_events"]:
        lines.append("")
        lines.append("状态事件：")
        for e in report["status_events"]:
            lines.append(f"- {e['record_id']}: {e['from_status']} → **{e['to_status']}**（由 {e['caused_by']} / {e['relation_type']} 触发）")
    lines.append("")

    # 每条建议的回查详情
    lines.append("## 五、已发布建议回查详情")
    lines.append("")
    for rev in report["guidance_reviews"]:
        icon = _ACTION_EMOJI.get(rev["suggested_action"], "")
        lines.append(f"### {icon} {rev['guidance_id']}：{rev['suggested_action_zh']}")
        lines.append("")
        lines.append(f"> {rev['guidance_text']}")
        lines.append("")
        tr = rev["trace"]
        lines.append(
            f"- 匹配方式：`{tr['match_method']}`；种子引用：{_cell(', '.join(tr['seed_record_ids']))}；"
            f"独立试验数：**{tr['independent_study_count']}**；"
            f"试验簇成员：{_cell(', '.join(tr['member_record_ids']))}"
        )
        if tr.get("dossier_record_ids"):
            lines.append(f"- 更正/补充/批注档案：{', '.join(tr['dossier_record_ids'])}")
        lines.append("")
        lines.append("| 检查 | 结果 | 说明 |")
        lines.append("|---|---|---|")
        for c in rev["checks"]:
            lines.append(f"| {c['title_zh']} (`{c['code']}`) | {_CHECK_ICON.get(c['status'], '')} {c['status']} | {_cell(c['detail_zh'])} |")
        lines.append("")
        if rev.get("suggested_revision"):
            lines.append("**缩小/改写草稿（供起草人参考，不是决定）：**")
            lines.append("")
            lines.append(rev["suggested_revision"]["draft_zh"])
            lines.append("")

    # 表述审计
    lines.append("## 六、表述审计（“未发现差异”不得写成“无效”）")
    lines.append("")
    if not report["wording_issues"]:
        lines.append("未发现把不确定结果改写成无效应结论的表述问题。")
    else:
        for i, iss in enumerate(report["wording_issues"], 1):
            lines.append(f"{i}. [{iss['severity']}] `{iss['code']}`（来源 {iss.get('source_id')}）")
            lines.append(f"   - 原句：“{iss['source_sentence']}”")
            lines.append(f"   - {iss['message_zh']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 七、复现方式")
    lines.append("")
    lines.append("```sh")
    lines.append("python3 -m evidence_workbench data/case.json --outdir reports")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
