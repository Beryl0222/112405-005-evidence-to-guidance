"""已发布建议的证据回查与复核决策。

每条建议输出：
- ``trace``：建议 → 支持记录 → 试验簇 → 更正/撤回通知的完整路径；
- ``checks``：人群、比较、结局直接性、方向、来源状态、独立性等逐项检查；
- ``suggested_action``：maintain / narrow / reopen（机器建议，终判权在起草人）；
- ``reasons`` 与 ``suggested_revision``：可直接进复核备忘录的中文说明。
"""

from __future__ import annotations

from typing import Any

from .models import CheckStatus, Decision, Direction, SourceStatus
from .nlp import compact, extract_dose_and_duration, extract_population_qualifiers, parse_guidance_text

# 触发“重新讨论”的检查代码
REOPEN_CODES = {
    "no_traceable_evidence",
    "dangling_reference",
    "source_retracted",
    "source_corrected",
    "source_concern",
    "outcome_not_studied",
    "surrogate_only_for_patient_outcome",
    "inconclusive_result_cannot_support_positive",
    "mixed_or_inconsistent_evidence",
    "unfavorable_result",
    "intervention_mismatch",
}
# 触发“缩小适用范围”的检查代码（在没有 reopen 事项时）
NARROW_CODES = {"population_scope_narrower", "dose_scope_narrower"}

_STATUS_LABEL = {
    SourceStatus.ACTIVE.value: "现行有效",
    SourceStatus.CORRECTED.value: "已被更正",
    SourceStatus.RETRACTED.value: "已被撤回",
    SourceStatus.CONCERN.value: "存在关注声明",
}


def _check(code: str, status: CheckStatus, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "status": status.value, "title_zh": title, "detail_zh": detail, **extra}


def _explicit_refs(g: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("evidence_refs", "support_refs", "supports", "refs", "sources", "evidence", "support"):
        val = g.get(key)
        if isinstance(val, str):
            refs.append(val)
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, str):
                    refs.append(v)
                elif isinstance(v, dict) and v.get("id"):
                    refs.append(str(v["id"]))
    seen: set[str] = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _intervention_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    ca, cb = compact(a), compact(b)
    return ca == cb or ca in cb or cb in ca


_INTERVENTION_SUFFIXES = [
    "干预措施", "治疗方案", "方案", "疗法", "干预", "制剂", "药物", "疫苗", "补充剂",
    "凝胶", "乳膏", "软膏", "片剂", "胶囊", "口服液", "注射液", "喷雾", "贴剂", "颗粒",
]


def intervention_name_variants(name: str | None) -> set[str]:
    """生成干预名的匹配变体：全称 + 去掉类型后缀的核心名（健身操方案→健身操）。"""
    if not name:
        return set()
    c = compact(name)
    variants = {c}
    for suf in _INTERVENTION_SUFFIXES:
        if c.endswith(suf):
            core = c[: -len(suf)]
            if len(core) >= 2:  # 避免“甲”这类单字核心误匹配
                variants.add(core)
    return {v for v in variants if v}


def intervention_mentioned_in(intervention_name: str | None, text: str) -> bool:
    ct = compact(text)
    return any(v in ct for v in intervention_name_variants(intervention_name))


NON_STUDY_KINDS = {"correction", "retraction", "concern", "methods", "meeting_note", "annotation", "guidance"}


def resolve_support(
    guidance: dict[str, Any],
    parsed: dict[str, Any],
    records: list[dict[str, Any]],
    claims_by_record: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], str, list[dict[str, Any]]]:
    """返回 (支持记录 id, 匹配方式, 说明路径)。

    方法补充/通知/批注类记录不会仅因提及干预名而被当成独立研究种子。
    """
    paths: list[dict[str, Any]] = []
    explicit = _explicit_refs(guidance)
    known = {r["id"] for r in records}
    explicit_known = [r for r in explicit if r in known]
    for r in explicit:
        if r not in known:
            paths.append({"from": guidance["id"], "to": r, "via": "显式引用，但资料中找不到该记录", "missing": True})
    if explicit_known:
        for r in explicit_known:
            paths.append({"from": guidance["id"], "to": r, "via": "建议字段显式指定的证据引用"})
        return explicit_known, "explicit_refs", paths

    # 推断：共享干预名（允许建议使用核心简称，如“健身操”对“健身操方案”）
    giv = parsed.get("intervention") or guidance.get("intervention")
    gtext = guidance.get("text", "")
    kind_by_id = {r["id"]: r.get("kind_normalized", "unknown") for r in records}
    matched: list[str] = []
    for r in records:
        rid = r["id"]
        if kind_by_id.get(rid) in NON_STUDY_KINDS:
            continue
        hit_iv = None
        for c in claims_by_record.get(rid, []):
            civ = c.get("intervention")
            if _intervention_match(giv, civ) or intervention_mentioned_in(civ, gtext):
                hit_iv = civ
                break
        if hit_iv is not None:
            if rid not in matched:
                matched.append(rid)
                paths.append(
                    {"from": guidance["id"], "to": rid,
                     "via": f"建议与主张对应同一干预“{hit_iv}”（允许核心简称匹配）"}
                )
            continue
        # 记录正文直接出现完整干预名也可
        if giv and compact(giv) in compact(r.get("text", "")) and rid not in matched:
            matched.append(rid)
            paths.append({"from": guidance["id"], "to": rid, "via": f"记录正文提及干预“{giv}”（无结构化主张）"})
    return matched, "intervention_text_match", paths


def _trial_family(
    seed_ids: list[str],
    trials: list[dict[str, Any]],
    record_trial: dict[str, str],
    relations: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """把种子记录解析为真实试验簇。

    返回 ``(trial_ids, member_record_ids, dossier_seed_ids)``。
    非研究类种子（方法补充/通知）沿出边找到其服务的试验，不计入独立研究数。
    """
    real_trial_ids = {t["trial_id"] for t in trials}
    tids: list[str] = []
    dossier_seeds: list[str] = []

    def add_tid(tid: str | None) -> None:
        if tid and tid in real_trial_ids and tid not in tids:
            tids.append(tid)

    for s in seed_ids:
        tid = record_trial.get(s)
        if tid in real_trial_ids:
            add_tid(tid)
            continue
        # 档案类显式种子：沿 supplements/corrects/... 出边回溯到试验
        resolved = False
        for e in relations or []:
            if e["source_id"] != s:
                continue
            target_tid = record_trial.get(e["target_id"])
            if target_tid in real_trial_ids:
                add_tid(target_tid)
                dossier_seeds.append(s)
                resolved = True
                break
        if not resolved:
            dossier_seeds.append(s)

    members: list[str] = []
    for t in trials:
        if t["trial_id"] in tids:
            for m in t["member_record_ids"]:
                if m not in members:
                    members.append(m)
    # 极端情况下没有试验表但有研究种子，保留种子以保证可追溯
    for s in seed_ids:
        if s not in members and s not in dossier_seeds:
            members.append(s)
    return sorted(tids), sorted(members), sorted(set(dossier_seeds))


def _dossier_for(member_ids: list[str], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """找出指向试验成员的通知/补充/批注记录。"""
    mset = set(member_ids)
    out: list[dict[str, Any]] = []
    for e in relations:
        if e["target_id"] in mset and e["source_id"] not in mset:
            out.append(e)
    return out


def compare_population(
    g_pop: dict[str, Any],
    evidence_quals: list[dict[str, Any]],
    evidence_pop_texts: list[str],
) -> dict[str, Any]:
    """比较建议人群与证据人群，返回错配描述。"""
    ev_min_age = next((q.get("min_age") for q in evidence_quals if q.get("type") == "min_age"), None)
    ev_range = next((q for q in evidence_quals if q.get("type") == "age_range"), None)
    if ev_range:
        ev_min_age = ev_range.get("min_age")
        ev_max_age = ev_range.get("max_age")
    else:
        ev_max_age = next((q.get("max_age") for q in evidence_quals if q.get("type") == "max_age"), None)
    extra_conditions = list(dict.fromkeys(
        q.get("condition") for q in evidence_quals if q.get("type") == "eligibility_condition" and q.get("condition")
    ))

    g_quals = g_pop.get("population_qualifiers", [])
    g_min_age = next((q.get("min_age") for q in g_quals if q.get("type") == "min_age"), None)
    g_range = next((q for q in g_quals if q.get("type") == "age_range"), None)
    if g_range:
        g_min_age = g_range.get("min_age")
        g_max_age = g_range.get("max_age")
    else:
        g_max_age = next((q.get("max_age") for q in g_quals if q.get("type") == "max_age"), None)

    gaps: list[str] = []
    if ev_range and not g_quals:
        gaps.append(f"证据年龄限定 {ev_range.get('raw')}，建议未写明年限")
    elif ev_min_age is not None:
        if g_min_age is None:
            gaps.append(f"证据要求 {ev_min_age} 岁及以上，建议仅使用宽泛人群表述“{g_pop.get('population')}”")
        elif g_min_age < ev_min_age:
            gaps.append(f"建议下限 {g_min_age} 岁宽于证据下限 {ev_min_age} 岁，{g_min_age}–{ev_min_age - 1} 岁缺乏证据")
    if ev_max_age is not None:
        if g_max_age is None and g_range is None and not any(q.get("type") == "max_age" for q in g_quals):
            gaps.append(f"证据上限为 {ev_max_age} 岁，建议未写明该上限，超过 {ev_max_age} 岁属于外推")
        elif g_max_age is not None and g_max_age > ev_max_age:
            gaps.append(f"建议上限 {g_max_age} 岁宽于证据上限 {ev_max_age} 岁")
    if extra_conditions:
        gaps.append("证据还要求额外准入条件：" + "、".join(extra_conditions) + "，建议未体现")

    return {
        "aligned": not gaps,
        "gaps": gaps,
        "evidence_populations": evidence_pop_texts,
        "evidence_min_age": ev_min_age,
        "evidence_max_age": ev_max_age,
        "evidence_age_range": f"{ev_range['min_age']}-{ev_range['max_age']}岁" if ev_range else None,
        "extra_conditions": extra_conditions,
    }


def review_guidance(
    guidance: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    claims_by_record: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    record_trial: dict[str, str],
    statuses: dict[str, str],
) -> dict[str, Any]:
    gid = guidance["id"]
    gtext = guidance.get("text", "")
    parsed = parse_guidance_text(gtext)
    # 允许结构化字段覆盖/补充 NLP
    if guidance.get("population"):
        parsed["population"] = guidance["population"]
    if guidance.get("intervention"):
        parsed["intervention"] = guidance["intervention"]
    if guidance.get("outcome"):
        parsed["outcome"] = guidance["outcome"]

    checks: list[dict[str, Any]] = []
    trace_paths: list[dict[str, Any]] = []

    seed_ids, match_method, paths = resolve_support(guidance, parsed, records, claims_by_record)
    trace_paths.extend(paths)

    known_ids = {r["id"] for r in records}
    missing_refs = [p["to"] for p in paths if p.get("missing")]
    if missing_refs:
        checks.append(_check(
            "dangling_reference", CheckStatus.BLOCK, "存在找不到来源记录的引用",
            f"以下编号不在资料中：{', '.join(missing_refs)}",
        ))

    if not seed_ids:
        checks.append(_check(
            "no_traceable_evidence", CheckStatus.BLOCK, "未能回查到任何支持研究",
            "既没有显式证据引用，也没有按干预名匹配到主张；该建议无法建立可复核证据链，应重新讨论。",
        ))
        return _build_result(
            guidance, parsed, Decision.REOPEN, checks,
            trace={"match_method": match_method, "paths": trace_paths,
                   "seed_record_ids": [], "trial_ids": [], "member_record_ids": [],
                   "independent_study_count": 0},
            supporting_studies=[], narrowed_revision=None,
        )

    trial_ids, member_ids, dossier_seed_ids = _trial_family(seed_ids, trials, record_trial, relations)
    # 关系路径：follow-up / 通知
    for e in relations:
        if e["source_id"] in member_ids and e["target_id"] in member_ids:
            trace_paths.append({"from": e["source_id"], "to": e["target_id"],
                                "via": f"{e['relation_type']}：{_brief(e.get('evidence_sentence'))}",
                                "relation_type": e["relation_type"]})
    dossier_edges = _dossier_for(member_ids, relations)
    dossier_ids = {e["source_id"] for e in dossier_edges} | set(dossier_seed_ids)
    for e in dossier_edges:
        trace_paths.append({"from": e["source_id"], "to": e["target_id"],
                            "via": f"{e['relation_type']}：{_brief(e.get('evidence_sentence'))}",
                            "relation_type": e["relation_type"]})
    for s in dossier_seed_ids:
        if s not in {p.get("from") for p in trace_paths}:
            trace_paths.append({"from": gid, "to": s, "via": "建议显式引用的档案材料"})

    # 汇总试验簇内全部主张与限定
    member_claims: list[dict[str, Any]] = []
    for mid in member_ids:
        member_claims.extend(claims_by_record.get(mid, []))
    result_claims = [c for c in member_claims if c.get("claim_type") != "outcome_not_measured"]
    gap_claims = [c for c in member_claims if c.get("claim_type") == "outcome_not_measured"]

    # -- 检查 1：来源状态（更正/撤回传播） ------------------------------
    bad_status = []
    for mid in member_ids:
        st = statuses.get(mid, SourceStatus.ACTIVE.value)
        if st != SourceStatus.ACTIVE.value:
            bad_status.append((mid, st))
    for e in dossier_edges:
        st = statuses.get(e["target_id"], SourceStatus.ACTIVE.value)
        if e["relation_type"] in {"corrects", "retracts", "expression_of_concern"}:
            # 通知类边已经把状态写到 target；这里补充通知来源本身
            pass
    if any(st == SourceStatus.RETRACTED.value for _, st in bad_status):
        ids_txt = ", ".join(f"{i}（{_STATUS_LABEL[s]}）" for i, s in bad_status)
        checks.append(_check(
            "source_retracted", CheckStatus.BLOCK, "支持研究已被撤回",
            f"{ids_txt}。撤回来源不能继续支撑现行建议，应立即从公共建议中撤下或暂停并重新讨论。",
            impacted_records=bad_status,
        ))
    elif any(st == SourceStatus.CORRECTED.value for _, st in bad_status):
        ids_txt = ", ".join(f"{i}（{_STATUS_LABEL[s]}）" for i, s in bad_status)
        checks.append(_check(
            "source_corrected", CheckStatus.WARN, "支持研究已被更正",
            f"{ids_txt}。更正可能改变效应量或组别解释，在学科审阅者核对更正内容前，该建议必须标记重审。",
            impacted_records=bad_status,
        ))
    elif any(st == SourceStatus.CONCERN.value for _, st in bad_status):
        checks.append(_check(
            "source_concern", CheckStatus.WARN, "支持研究存在关注声明",
            f"{bad_status}；在问题解决前应重新讨论建议表述。", impacted_records=bad_status,
        ))
    else:
        checks.append(_check("source_status_ok", CheckStatus.PASS, "来源状态正常",
                             "支持记录当前均为 active（未见更正/撤回/关注声明）。"))

    # -- 检查 2：独立性（同一试验的后续分析不重复计数） -----------------
    n_trials = len(trial_ids)
    follow_ups = [m for m in member_ids
                  if any(e["source_id"] == m and e["relation_type"] == "follow_up_of" for e in relations)]
    detail = f"回查到 {len(member_ids)} 条记录，归并为 {n_trials} 项独立试验（{', '.join(trial_ids)}）。"
    if follow_ups:
        detail += f" 其中 {', '.join(follow_ups)} 是同一试验的后续/随访分析，不能算作第二项独立研究。"
    checks.append(_check(
        "independent_study_count", CheckStatus.INFO if n_trials >= 1 else CheckStatus.WARN,
        "独立研究计数", detail,
        record_count=len(member_ids), independent_trial_count=n_trials,
        trial_ids=trial_ids, follow_up_records=follow_ups,
    ))

    # -- 检查 3：干预一致性 ---------------------------------------------
    giv = parsed.get("intervention")
    iv_evidence = sorted({c.get("intervention") for c in member_claims if c.get("intervention")})
    iv_ok = any(
        _intervention_match(giv, x) or intervention_mentioned_in(x, gtext)
        for x in iv_evidence
    ) if iv_evidence else bool(giv)
    if iv_evidence and not iv_ok:
        checks.append(_check(
            "intervention_mismatch", CheckStatus.BLOCK, "干预措施不一致",
            f"建议表述“{giv}”与证据主张中的干预 {iv_evidence} 无法对应；不能用这些研究支持该建议。",
        ))
    else:
        shown = giv or "（建议使用简称，已按证据干预核心名匹配）"
        checks.append(_check("intervention_match", CheckStatus.PASS, "干预措施一致",
                             f"建议干预“{shown}”与证据干预 {iv_evidence or '（结构化字段）'} 可对应。"))

    # -- 检查 4：人群适用范围 -------------------------------------------
    ev_quals: list[dict[str, Any]] = []
    ev_pops: list[str] = []
    ev_doses: list[str] = []
    for c in member_claims:
        ev_quals.extend(c.get("population_qualifiers", []))
        for q in c.get("effect_qualifiers", []):
            if q.get("type") in {"dose", "dose_descriptor", "frequency"} and q.get("value"):
                ev_doses.append(str(q["value"]))
        if c.get("population") and c["population"] not in ev_pops:
            ev_pops.append(c["population"])
    # 方法补充/批注等档案记录中的限定也并入（保留可复核来源）
    dossier_sources: list[str] = []
    for e in dossier_edges:
        if e["relation_type"] in {"supplements", "annotates"}:
            rec = next((r for r in records if r["id"] == e["source_id"]), None)
            if not rec:
                continue
            dossier_sources.append(rec["id"])
            d_quals = extract_dose_and_duration(rec.get("text", ""))
            ev_doses.extend(str(q["value"]) for q in d_quals if q["type"] in {"dose", "dose_descriptor", "frequency"})
            _, p_quals = extract_population_qualifiers(rec.get("text", ""))
            ev_quals.extend(p_quals)
    # 成员研究记录全文中的剂量限定（覆盖“主张句与剂量句不在同一句”的情况）
    for r in records:
        if r["id"] in member_ids:
            full_dq = extract_dose_and_duration(r.get("text", ""))
            ev_doses.extend(str(q["value"]) for q in full_dq if q["type"] in {"dose", "dose_descriptor", "frequency"})
    # 记录级结构化 qualifiers 也并入
    for r in records:
        if r["id"] in member_ids and r.get("qualifiers"):
            ev_quals.extend(r["qualifiers"])
    ev_doses = sorted(set(ev_doses))
    pop_cmp = compare_population(parsed, ev_quals, ev_pops)
    if pop_cmp["aligned"]:
        checks.append(_check("population_applicability", CheckStatus.PASS, "人群范围可对应",
                             "建议人群与证据纳入人群未发现额外限定冲突。", comparison=pop_cmp))
    else:
        checks.append(_check(
            "population_scope_narrower", CheckStatus.WARN, "证据人群比建议表述更窄",
            "；".join(pop_cmp["gaps"]) + "。若直接外推到全部宽泛人群属于过度概括，至少应缩小表述。",
            comparison=pop_cmp,
        ))

    # -- 检查 4b：剂量/方案限定 -----------------------------------------
    g_doses = [str(q.get("value")) for q in extract_dose_and_duration(gtext)
               if q["type"] in {"dose", "dose_descriptor", "frequency"}]
    if ev_doses and not any(any(gd in ed or ed in gd for gd in g_doses) for ed in ev_doses):
        via = f"（限定来自 {', '.join(dossier_sources)}）" if dossier_sources else ""
        checks.append(_check(
            "dose_scope_narrower", CheckStatus.WARN, "证据存在特定剂量/方案限定",
            f"证据剂量限定为 {ev_doses}{via}，建议未写明对应剂量；公共建议必须带剂量与使用条件，不能泛化为任意用法。",
            evidence_doses=ev_doses, guidance_doses=g_doses,
        ))
    elif ev_doses:
        checks.append(_check("dose_documented", CheckStatus.PASS, "剂量限定已保留",
                             f"证据与建议剂量可对应：{ev_doses}。"))

    # -- 检查 5：比较条件是否保留 ---------------------------------------
    comps = [(c.get("comparator"), c.get("comparator_status")) for c in result_claims]
    comps_reported = [c for c, st in comps if st and st != "not_specified" and c]
    if comps_reported:
        checks.append(_check("comparator_documented", CheckStatus.PASS, "比较条件已保留",
                             "证据中的比较条件：" + "；".join(sorted(set(comps_reported)))))
    elif comps:
        checks.append(_check("comparator_not_reported", CheckStatus.INFO, "比较条件原文未具体说明",
                             "已如实标记为 not_specified；引用时不得替原文补造“安慰剂/常规治疗”等对照。"))

    # -- 检查 6：结局直接性（替代指标 vs 市民结局） ----------------------
    g_concept = parsed.get("outcome_concept")
    g_class = parsed.get("outcome_class")
    same_concept = [c for c in result_claims if g_concept and c.get("outcome_concept") == g_concept]
    gap_same = [c for c in gap_claims if g_concept and c.get("outcome_concept") == g_concept]
    surrogate_claims = [c for c in result_claims if c.get("outcome_class") == "surrogate"]

    if gap_same:
        srcs = ", ".join(sorted({c["record_id"] for c in gap_same}))
        checks.append(_check(
            "outcome_not_studied", CheckStatus.BLOCK, "建议结局未被支持研究测量",
            f"建议目标结局“{parsed.get('outcome')}”在 {srcs} 中被明确声明未测量/未报告；"
            "没有直接证据支持该公共建议，不能用其他指标的下降替代。",
            claim_ids=[c["claim_id"] for c in gap_same],
        ))
    direct_support_ok = False
    if same_concept:
        dirs = {c["direction"] for c in same_concept}
        has_fav = Direction.FAVORABLE.value in dirs
        has_null = Direction.NO_SIG_DIFF.value in dirs
        has_unfav = Direction.UNFAVORABLE.value in dirs
        if has_fav and not has_null and not has_unfav:
            direct_support_ok = True
            checks.append(_check(
                "outcome_direct_evidence", CheckStatus.PASS, "存在同一结局的直接证据",
                f"找到 {len(same_concept)} 条方向为有利的同结局主张。",
                claim_ids=[c["claim_id"] for c in same_concept],
            ))
        elif has_fav and (has_null or has_unfav):
            checks.append(_check(
                "mixed_or_inconsistent_evidence", CheckStatus.BLOCK,
                "同一结局的研究结论不一致",
                "既有有利结果，也有“未发现差异”或不利结果；证据不一致时应重新讨论，"
                "且“未发现差异”不得写成证明无效。",
                claim_ids=[c["claim_id"] for c in same_concept],
            ))
        elif has_null:
            checks.append(_check(
                "inconclusive_result_cannot_support_positive", CheckStatus.BLOCK,
                "同结局研究仅报告“未发现差异”",
                "未检出差异属于不确定证据，不能支持“降低风险”的阳性建议；"
                "同时也不得写成“证明无效”，应等待新证据并重新讨论。",
                claim_ids=[c["claim_id"] for c in same_concept],
            ))
        elif has_unfav:
            checks.append(_check(
                "unfavorable_result", CheckStatus.BLOCK, "同结局证据方向与建议相反",
                "证据呈不利方向，现行建议应重新讨论。",
                claim_ids=[c["claim_id"] for c in same_concept],
            ))
    if not direct_support_ok and g_class == "patient_important" and surrogate_claims:
        checks.append(_check(
            "surrogate_only_for_patient_outcome", CheckStatus.BLOCK,
            "只有替代指标证据，不能直接支持市民关心的结局",
            f"建议结局“{parsed.get('outcome')}”属于硬结局，而支持主张仅测量 "
            + "、".join(sorted({str(c.get('outcome')) for c in surrogate_claims if c.get('outcome')}))
            + " 等替代指标；替代指标改善不等于住院/死亡等结局改善，应重新讨论或改写建议。",
            claim_ids=[c["claim_id"] for c in surrogate_claims],
        ))
    elif not same_concept and not gap_same and not surrogate_claims:
        checks.append(_check(
            "outcome_unverified", CheckStatus.WARN, "未找到与建议结局对应的主张",
            f"建议结局“{parsed.get('outcome')}”未能与抽取主张对应；需学科审阅者人工确认证据是否相关。",
        ))

    # 替代指标主张本身的方向（信息透明）
    for c in surrogate_claims:
        checks.append(_check(
            "surrogate_direction_info", CheckStatus.INFO, "替代指标主张（不可外推为硬结局）",
            f"{c['claim_id']}：{c.get('outcome')} → {c['direction']}。{c.get('direction_note') or ''}",
            claim_id=c["claim_id"],
        ))

    # -- 汇总决策 -------------------------------------------------------
    codes = {c["code"] for c in checks}
    if codes & REOPEN_CODES:
        decision = Decision.REOPEN
    elif codes & NARROW_CODES:
        decision = Decision.NARROW
    else:
        decision = Decision.MAINTAIN

    narrowed = _build_narrowed_revision(guidance["text"], parsed, pop_cmp, surrogate_claims, gap_same, ev_doses)
    reasons = [
        {"code": c["code"], "severity": c["status"], "message_zh": c["detail_zh"]}
        for c in checks if c["status"] in {CheckStatus.BLOCK.value, CheckStatus.WARN.value}
    ]

    supporting_studies = []
    for t in trials:
        if t["trial_id"] in trial_ids:
            supporting_studies.append({
                "trial_id": t["trial_id"],
                "primary_record_id": t["primary_record_id"],
                "record_ids": t["member_record_ids"],
                "independent_study_count": 1,
                "status": _worst_status([statuses.get(m, SourceStatus.ACTIVE.value) for m in t["member_record_ids"]]),
                "claim_ids": [c["claim_id"] for m in t["member_record_ids"] for c in claims_by_record.get(m, [])],
            })

    trace = {
        "match_method": match_method,
        "seed_record_ids": seed_ids,
        "trial_ids": trial_ids,
        "member_record_ids": member_ids,
        "dossier_record_ids": sorted(dossier_ids),
        "independent_study_count": n_trials,
        "paths": trace_paths,
    }
    return _build_result(guidance, parsed, decision, checks, trace, supporting_studies, narrowed, reasons)


def _build_result(guidance, parsed, decision, checks, trace, supporting_studies, narrowed_revision, reasons=None):
    needs_review = decision != Decision.MAINTAIN
    priority = None
    if decision == Decision.REOPEN:
        if any(c["code"] == "source_retracted" for c in checks):
            priority = "critical"
        else:
            priority = "high"
    elif decision == Decision.NARROW:
        priority = "medium"
    return {
        "guidance_id": guidance["id"],
        "guidance_text": guidance.get("text", ""),
        "parsed": parsed,
        "suggested_action": decision.value,
        "suggested_action_zh": {
            Decision.MAINTAIN.value: "维持现行建议",
            Decision.NARROW.value: "缩小适用范围/改写表述",
            Decision.REOPEN.value: "重新讨论（必须重审）",
        }[decision.value],
        "needs_review": needs_review,
        "review_priority": priority,
        "reasons": reasons if reasons is not None else [
            {"code": c["code"], "severity": c["status"], "message_zh": c["detail_zh"]}
            for c in checks if c["status"] in {"block", "warn"}
        ],
        "checks": checks,
        "trace": trace,
        "supporting_studies": supporting_studies,
        "suggested_revision": narrowed_revision,
    }


def _worst_status(status_list: list[str]) -> str:
    rank = {s.value: i for i, s in enumerate(
        [SourceStatus.ACTIVE, SourceStatus.CONCERN, SourceStatus.CORRECTED, SourceStatus.RETRACTED])}
    if not status_list:
        return SourceStatus.ACTIVE.value
    return max(status_list, key=lambda s: rank.get(s, 0))


def _brief(sentence: str | None, n: int = 60) -> str:
    if not sentence:
        return "结构化关系"
    s = compact(sentence)
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_narrowed_revision(text, parsed, pop_cmp, surrogate_claims, gap_same, ev_doses=None) -> dict[str, Any] | None:
    """给出可直接改写的缩小版表述（机器草稿，需起草人定稿）。"""
    if not pop_cmp.get("gaps") and not surrogate_claims and not gap_same and not ev_doses:
        return None

    pop_raw = pop_cmp.get("evidence_populations")[0] if pop_cmp.get("evidence_populations") else None
    if pop_raw:
        pop_phrase = pop_raw
    elif pop_cmp.get("evidence_age_range"):
        pop_phrase = pop_cmp["evidence_age_range"] + "人群"
    elif pop_cmp.get("evidence_min_age") is not None:
        pop_phrase = f"{pop_cmp['evidence_min_age']} 岁及以上人群"
    else:
        pop_phrase = "证据所述人群"
    conds = pop_cmp.get("extra_conditions") or []
    if conds and not any(c in pop_phrase for c in conds):
        pop_phrase += "（还须满足：" + "、".join(conds) + "）"

    iv = parsed.get("intervention") or "该干预"
    sentences = [f"适用人群应限定为：{pop_phrase}。"]
    if ev_doses:
        sentences.append(f"剂量/频次应限定为：{'、'.join(ev_doses)}。")
    sur_outcomes = sorted({str(c.get("outcome")) for c in surrogate_claims if c.get("outcome")})
    if sur_outcomes:
        sentences.append(
            f"{iv}在原始研究中仅观察到{'、'.join(sur_outcomes)}的有利变化，"
            "这些属于替代指标，不能外推为住院、死亡等市民关心的临床结局。"
        )
    if gap_same:
        sentences.append(f"原始研究明确未测量{parsed.get('outcome')}，不得声称其可降低{parsed.get('outcome')}。")
    sentences.append("是否保留原公共建议，需学科审阅并取得直接结局证据后重新讨论。")
    draft = "【重审草稿，非现行建议】" + "".join(sentences)
    return {
        "narrowed_population": pop_phrase,
        "evidence_doses": ev_doses or [],
        "surrogate_outcomes": sur_outcomes,
        "unmeasured_target_outcome": bool(gap_same),
        "draft_zh": draft,
    }
