"""记录间关系抽取、同一试验归并，以及更正/撤回状态计算。

核心规则：
- 后续分析/随访分析通过“同一试验”等显式表述或结构化字段归并到同一试验簇，
  统计“独立研究数”时只算一个试验；
- 更正/撤回通知形成 ``corrects`` / ``retracts`` 边，状态沿边传播到被指向的论文；
- 所有关系都保留识别依据（kind / 关键词 / 原句），可复核。
"""

from __future__ import annotations

import re
from typing import Any

from .models import SourceStatus
from .nlp import ID_RE, compact

KIND_ALIASES = {
    "abstract": "abstract",
    "摘要": "abstract",
    "paper": "paper",
    "article": "paper",
    "full-text": "paper",
    "fulltext": "paper",
    "methods": "methods",
    "method": "methods",
    "methods-supplement": "methods",
    "supplement": "methods",
    "supplementary": "methods",
    "方法补充": "methods",
    "补充材料": "methods",
    "meeting-note": "meeting_note",
    "meeting_note": "meeting_note",
    "meeting": "meeting_note",
    "annotation": "annotation",
    "annotations": "annotation",
    "会议批注": "meeting_note",
    "批注": "annotation",
    "note": "annotation",
    "notes": "annotation",
    "follow-up-paper": "follow_up",
    "follow-up": "follow_up",
    "followup": "follow_up",
    "后续分析": "follow_up",
    "correction": "correction",
    "erratum": "correction",
    "更正": "correction",
    "retraction": "retraction",
    "retraction-notice": "retraction",
    "retracted": "retracted_paper",
    "retracted-paper": "retracted_paper",
    "retracted_paper": "retracted_paper",
    "withdrawn": "retracted_paper",
    "撤回": "retraction",
    "撤稿": "retraction",
    "corrected": "corrected_paper",
    "corrected-paper": "corrected_paper",
    "corrected_paper": "corrected_paper",
    "expression-of-concern": "concern",
    "concern": "concern",
}

# relation -> 触发词（在目标 ID 附近或全文出现即作为文本证据）
RELATION_LEXICON: list[tuple[str, list[str]]] = [
    ("retracts", ["撤回", "撤销", "撤稿", "retract", "retracted"]),
    ("corrects", ["更正", "勘误", "纠正", "有误", "标签错误", "数据错误", "correction", "erratum", "incorrect"]),
    ("expression_of_concern", ["关注声明", "关切声明", "expression of concern"]),
    ("follow_up_of", [
        "同一试验", "同一项试验", "同一研究", "同一项研究", "同一队列",
        "随访分析", "后续分析", "延长分析", "二次分析", "事后分析",
        "follow-up analysis", "follow up analysis", "same trial", "same study",
        "post hoc", "post-hoc", "secondary analysis", "extension study",
    ]),
    ("supplements", ["方法补充", "补充方法", "补充材料", "supplement", "supplementary method", "methods supplement"]),
    ("annotates", ["会议批注", "批注", "annotation", "meeting note", "reviewer note"]),
    ("cites", ["引用", "参见", "见", "根据", "cited", "see also", "refer"]),
]

KIND_DEFAULT_RELATION = {
    "correction": "corrects",
    "retraction": "retracts",
    "concern": "expression_of_concern",
    "follow_up": "follow_up_of",
    "methods": "supplements",
    "meeting_note": "annotates",
    "annotation": "annotates",
}

# 哪些边表示“同一个试验/研究家族”，用于独立性归并
TRIAL_BOND_TYPES = {"same_trial_as", "follow_up_of"}
# 哪些边属于“关于某论文的档案材料”
DOSSIER_TYPES = {"supplements", "annotates", "corrects", "retracts", "expression_of_concern", "cites"}


def normalize_kind(kind: str | None) -> str:
    if not kind:
        return "unknown"
    key = compact(str(kind)).lower().replace("_", "-")
    if key in KIND_ALIASES:
        return KIND_ALIASES[key]
    raw = str(kind).strip().lower().replace("_", "-")
    return KIND_ALIASES.get(raw, raw)


def find_id_mentions(
    text: str,
    self_id: str | None = None,
    known_ids: set[str] | list[str] | None = None,
) -> list[str]:
    """找出文本中提及的记录编号。

    1) 通用“字母-数字”编号正则（AB-7、G-3 …），也能发现悬空引用；
    2) 对资料中实际存在的 id 做字面边界匹配，兼容 R1、G3 这类无连字符编号。
    """
    ids: list[str] = []

    def add(rid: str) -> None:
        if self_id and rid == self_id:
            return
        if rid not in ids:
            ids.append(rid)

    for m in ID_RE.finditer(text or ""):
        add(m.group(0))

    for kid in known_ids or []:
        if not kid or kid == self_id:
            continue
        pattern = r"(?<![A-Za-z0-9_-])" + re.escape(str(kid)) + r"(?![A-Za-z0-9-])"
        if re.search(pattern, text or ""):
            add(str(kid))
    return ids


def _detect_relation_type(text: str, kind_norm: str) -> tuple[str | None, str | None]:
    low = text.lower()
    for rel, words in RELATION_LEXICON:
        for w in words:
            if w.lower() in low:
                return rel, w
    if kind_norm in KIND_DEFAULT_RELATION:
        return KIND_DEFAULT_RELATION[kind_norm], f"kind={kind_norm}"
    return None, None


def _explicit_targets(record: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for key in ("targets", "target", "about", "corrects", "retracts", "same_trial_as",
                "follow_up_of", "paper_id", "study_id", "related_records", "related", "refs"):
        val = record.get(key)
        if isinstance(val, str):
            targets.append(val)
        elif isinstance(val, (list, tuple)):
            targets.extend(v for v in val if isinstance(v, str))
    # 去重保序
    seen: set[str] = set()
    out = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_relations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从结构化字段与正文表述中抽取记录间关系。"""
    relations: list[dict[str, Any]] = []
    known_ids = {r["id"] for r in records}

    def add(source: str, target: str, rel_type: str, basis: str, evidence: str | None, confidence: str) -> None:
        if source == target:
            return
        for ex in relations:
            if ex["source_id"] == source and ex["target_id"] == target and ex["relation_type"] == rel_type:
                if confidence == "high" and ex["confidence"] != "high":
                    ex.update(basis=basis, evidence_sentence=evidence, confidence=confidence)
                return
        relations.append(
            {
                "source_id": source,
                "target_id": target,
                "relation_type": rel_type,
                "basis": basis,  # explicit_field | kind | lexical
                "evidence_sentence": evidence,
                "confidence": confidence,
                "target_known": target in known_ids,
            }
        )

    for record in records:
        rid = record["id"]
        text = record.get("text", "")
        kind_norm = normalize_kind(record.get("kind"))

        # 1) 显式结构化字段（最高优先级）
        explicit = _explicit_targets(record)
        field_rel = None
        for key, rel in (("corrects", "corrects"), ("retracts", "retracts"),
                         ("same_trial_as", "same_trial_as"), ("follow_up_of", "follow_up_of")):
            if record.get(key):
                field_rel = rel
        for target in explicit:
            rel_type = field_rel or KIND_DEFAULT_RELATION.get(kind_norm) or "cites"
            add(rid, target, rel_type, "explicit_field", None, "high")

        # 显式 trial 归属也编码成 same_trial_as 边（指向 trial 主记录由 group 阶段处理）
        if record.get("trial_id") and record["trial_id"] != rid:
            add(rid, record["trial_id"], "same_trial_as", "explicit_field", None, "high")

        # 2) 正文 ID 提及 + 词法/类型判定
        rel_type, trigger = _detect_relation_type(text, kind_norm)
        for target in find_id_mentions(text, rid, known_ids):
            if rel_type is None:
                guessed = "cites"
                add(rid, target, guessed, "lexical", text.strip()[:200], "low")
                continue
            confidence = "high" if trigger and trigger.startswith("kind=") is False and kind_norm in KIND_DEFAULT_RELATION else "medium"
            if record.get("kind") and normalize_kind(record.get("kind")) in KIND_DEFAULT_RELATION:
                confidence = "high"
            add(rid, target, rel_type, "text_or_kind", text.strip(), confidence)

        # 3) 仅 kind 已说明是通知但正文没写目标 ID：无法定位，留给校验警告
    return relations


class _DSU:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 稳定地让较小 id 当代表
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def group_trials(records: list[dict[str, Any]], relations: list[dict[str, Any]]) -> dict[str, Any]:
    """把 same_trial_as / follow_up_of 边归并为试验簇。

    返回 ``{"trials": [...], "record_trial": {rid: trial_id}}``。
    若存在 follow_up_of 指向的主论文，则用主论文 id 作为 trial_id（稳定且可解释）。
    更正/撤回通知、方法补充、会议批注属于“档案材料”，单独存在时不计为独立试验。
    """
    non_study_kinds = {"correction", "retraction", "concern", "methods", "meeting_note", "annotation"}
    ids = [r["id"] for r in records]
    dsu = _DSU(ids)
    explicit_trial: dict[str, str] = {}

    for r in records:
        if r.get("trial_id"):
            explicit_trial[r["id"]] = str(r["trial_id"])

    bond_edges = [e for e in relations if e["relation_type"] in TRIAL_BOND_TYPES and e["target_known"]]
    for e in bond_edges:
        dsu.union(e["source_id"], e["target_id"])

    clusters: dict[str, list[str]] = {}
    for rid in ids:
        clusters.setdefault(dsu.find(rid), []).append(rid)

    kind_by_id = {r["id"]: normalize_kind(r.get("kind")) for r in records}
    trials: list[dict[str, Any]] = []
    record_trial: dict[str, str] = {}
    for _, members in sorted(clusters.items()):
        members = sorted(members)
        study_members = [m for m in members if kind_by_id.get(m, "unknown") not in non_study_kinds]
        # 纯通知/补充/批注簇不是一项研究，不生成 trial
        if not study_members:
            for m in members:
                record_trial[m] = members[0]
            continue
        # 主记录：follow_up_of / same_trial_as 的被指向方；否则最小 id
        targets = [e["target_id"] for e in bond_edges if e["source_id"] in members]
        primary = None
        for t in targets:
            if t in members and kind_by_id.get(t) not in non_study_kinds:
                primary = t
                break
        if primary is None:
            primary = study_members[0]
        # 显式 trial_id 优先作为簇 id
        ext = [explicit_trial[m] for m in members if m in explicit_trial]
        trial_id = ext[0] if ext else primary
        trials.append(
            {
                "trial_id": trial_id,
                "primary_record_id": primary,
                "member_record_ids": members,
                "independent_study_count": 1,  # 一个簇 = 一项独立试验
                "has_follow_up_analysis": any(
                    normalize_kind(next((r.get("kind") for r in records if r["id"] == m), None)) == "follow_up"
                    for m in members
                ),
            }
        )
        for m in members:
            record_trial[m] = trial_id

    # 显式 trial_id 可能指向不存在的主记录：仍保留映射
    for rid, tid in explicit_trial.items():
        record_trial.setdefault(rid, tid)
        if not any(t["trial_id"] == tid for t in trials):
            trials.append(
                {"trial_id": tid, "primary_record_id": tid, "member_record_ids": [rid],
                 "independent_study_count": 1, "has_follow_up_analysis": False}
            )
    trials.sort(key=lambda t: t["trial_id"])
    return {"trials": trials, "record_trial": record_trial}


def compute_statuses(
    records: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """计算每条记录的当前证据状态与状态事件。

    优先级：显式 override > 记录自带 status > 撤回边 > 更正边 > 关注边 > active。
    """
    overrides = overrides or {}
    statuses: dict[str, str] = {}
    events: list[dict[str, Any]] = []

    own_kind_status = {
        "retracted_paper": SourceStatus.RETRACTED.value,
        "corrected_paper": SourceStatus.CORRECTED.value,
        "concerned_paper": SourceStatus.CONCERN.value,
    }
    valid_status = {s.value for s in SourceStatus}
    for r in records:
        rid = r["id"]
        base = r.get("status")
        if base in valid_status:
            statuses[rid] = base
        else:
            kind_norm = r.get("kind_normalized") or normalize_kind(r.get("kind"))
            statuses[rid] = own_kind_status.get(kind_norm, SourceStatus.ACTIVE.value)

    rank = {
        SourceStatus.ACTIVE.value: 0,
        SourceStatus.CONCERN.value: 1,
        SourceStatus.CORRECTED.value: 2,
        SourceStatus.RETRACTED.value: 3,
    }
    rel_to_status = {
        "retracts": SourceStatus.RETRACTED.value,
        "corrects": SourceStatus.CORRECTED.value,
        "expression_of_concern": SourceStatus.CONCERN.value,
    }
    for e in relations:
        new_status = rel_to_status.get(e["relation_type"])
        if not new_status or not e["target_known"]:
            continue
        tid = e["target_id"]
        if rank[new_status] > rank[statuses.get(tid, SourceStatus.ACTIVE.value)]:
            old = statuses.get(tid, SourceStatus.ACTIVE.value)
            statuses[tid] = new_status
            events.append(
                {
                    "record_id": tid,
                    "from_status": old,
                    "to_status": new_status,
                    "caused_by": e["source_id"],
                    "relation_type": e["relation_type"],
                    "evidence_sentence": e.get("evidence_sentence"),
                }
            )

    for rid, st in overrides.items():
        if st not in rank:
            raise ValueError(f"未知状态 {st!r}，可选：{sorted(rank)}")
        old = statuses.get(rid, SourceStatus.ACTIVE.value)
        statuses[rid] = st
        events.append(
            {"record_id": rid, "from_status": old, "to_status": st,
             "caused_by": "status_override", "relation_type": "manual_update", "evidence_sentence": None}
        )

    return {"statuses": statuses, "events": events}


def build_dependency_graph(relations: list[dict[str, Any]]) -> dict[str, set[str]]:
    """target -> 指向它的 source 集合（反向依赖）。"""
    rev: dict[str, set[str]] = {}
    for e in relations:
        rev.setdefault(e["target_id"], set()).add(e["source_id"])
    return rev
