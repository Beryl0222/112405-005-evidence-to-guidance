"""基于规则的中文/英文医学文本 NLP 抽取（无第三方依赖）。

设计目标不是做完备句法分析，而是把合成演练文本中
“人群 / 比较条件 / 干预 / 结局 / 方向 / 来源原句”稳定、可复核地抽出来。

所有结果都附带 ``source_sentence``（记录原文的精确子串），
规则未覆盖的信息明确标记为“原文未说明”，绝不臆造。
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from .models import Direction, EvidentialForce, OutcomeClass

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

#: 稳定编号，如 AB-7、G-3、PM-2024-01
ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{0,12}(?:-[A-Za-z0-9]+)+")

_SENTENCE_ENDS = "。！？!?\n"
_CLAUSE_SPLIT = "；;"


def compact(text: str) -> str:
    """去除全部空白，用于中文匹配与规范化字段。"""
    return re.sub(r"\s+", "", text or "")


def spaced(text: str) -> str:
    """折叠连续空白为单个空格并 strip，用于保留可读原文。"""
    return re.sub(r"\s+", " ", text or "").strip()


def iter_sentences(text: str) -> Iterator[tuple[str, int, int]]:
    """按句末标点切句，返回 (原样子串, start, end)。"""
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDS:
            if i + 1 > start and text[start : i + 1].strip():
                yield text[start : i + 1], start, i + 1
            start = i + 1
    if start < len(text) and text[start:].strip():
        yield text[start:], start, len(text)


def iter_clauses(text: str) -> list[dict[str, Any]]:
    """把记录切成命题单元。

    以句末标点定位“原句”，再在句内按分号切分命题分句。
    返回的 ``sentence`` 与 ``clause`` 都是 ``text`` 的精确子串。
    """
    units: list[dict[str, Any]] = []
    for sentence, ss, se in iter_sentences(text):
        # 在句子内部按 ；; 切分，保留偏移
        c_start = ss
        for m in re.finditer(r"[；;]", sentence):
            c_end = ss + m.start() + 1
            clause = text[c_start:c_end]
            if clause.strip():
                units.append(
                    {
                        "clause": clause,
                        "clause_start": c_start,
                        "clause_end": c_end,
                        "sentence": sentence,
                        "sentence_start": ss,
                        "sentence_end": se,
                    }
                )
            c_start = c_end
        tail = text[c_start:se]
        if tail.strip():
            units.append(
                {
                    "clause": tail,
                    "clause_start": c_start,
                    "clause_end": se,
                    "sentence": sentence,
                    "sentence_start": ss,
                    "sentence_end": se,
                }
            )
    return units


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------

# 未发现差异 / 证据不足 —— 只能推出 INCONCLUSIVE
NULL_RESULT_PATTERNS = [
    r"未发现[^，,。；;]{0,15}?(?:差异|区别|不同|获益|改善|效果|疗效|作用|关联)",
    r"未(?:能|曾)?(?:观察到|见|发现|检出|显示|证实|证明)[^，,。；;]{0,12}?(?:差异|区别|统计学意义|显著|改善|获益|效果|疗效|有效|作用)",
    r"无(?:显著|明显|统计学意义)?差异",
    r"差异无统计学意义",
    r"未见(?:明显|显著)?(?:改善|获益|疗效|效果|差异)",
    r"没有(?:发现|观察到|显示|证实)[^，,。；;]{0,12}?(?:差异|效果|疗效|改善|获益)",
    r"并不?(?:显著)?优于",
    r"证据不足",
    r"(?:尚|暂)(?:不能|无法|不足以)(?:得出|证实|证明|确认|判定)",
    r"无充分证据",
    r"结果不确定",
    r"no\s+significant\s+difference",
    r"did\s+not\s+differ",
    r"no\s+difference\s+(?:was\s+)?(?:found|observed|detected)",
    r"not\s+statistically\s+significant",
    r"insufficient\s+evidence",
    r"no\s+(?:clear\s+)?evidence\s+of",
    r"inconclusive",
]
NULL_RESULT_RE = re.compile("|".join(f"(?:{p})" for p in NULL_RESULT_PATTERNS), re.IGNORECASE)

# 强“无效应”结论（需要等效性/非劣等专门设计支撑）
NO_EFFECT_PROOF_RE = re.compile(
    r"(?:证明|证实|确证|确认|表明|显示)[^，,。；;]{0,8}?"
    r"(?:无效|没有效果|无效果|无疗效|没有疗效|无作用|不起作用)"
    r"|(?:等效性检验|equivalence trial)[^。；;]{0,20}?(?:无效|无临床意义)",
)

# 明确的“未测量某结局”（注意：不含“观察”，因为“未观察到差异”属于 null 结果）
UNMEASURED_RE = re.compile(
    r"(?:本研究|该研究|研究|文中|论文)?"
    r"(?:未|没有|并未|未曾|未能)"
    r"(?:测量|测算|评估|评价|评定|报告|报道|检测|检验|收集|统计|涉及|纳入分析)"
    r"(?:过)?"
    r"(?:任何|该|此)?"
    r"(?P<out>[^，,。；;]{1,25})"
)
UNMEASURED_EN_RE = re.compile(
    r"did\s+not\s+(?:measure|assess|evaluate|report|collect)\s+(?P<out>[^.,;]{1,40})",
    re.IGNORECASE,
)

# 干预 / 方案
INTERVENTION_RE = re.compile(
    r"(?P<iv>[一-龥A-Za-z0-9]{1,14}?"
    r"(?:方案|疗法|干预措施?|制剂|药物|疫苗|补充剂|饮食|运动计划|训练计划|"
    r"凝胶|乳膏|软膏|片剂|胶囊|口服液|注射液|喷雾(?:剂)?|贴剂|颗粒|糖浆|滴剂))"
)
INTERVENTION_EN_RE = re.compile(
    r"\b(?P<iv>[A-Z][A-Za-z0-9\-]{1,20})\b(?=\s+(?:treatment|regimen|arm|group|intervention))",
)

# 比较条件
COMPARATOR_PATTERNS = [
    r"(?P<a>[^，,。；;]{1,18}?)(?:与|和|同|跟)(?P<b>[^，,。；;]{1,18}?)(?:进行)?(?:随机)?(?:对照|比较|对比|相比)",
    r"(?:相较于|相对于|与……相比|对比)(?P<b2>[^，,。；;]{1,20})",
]
COMPARATOR_RES = [re.compile(p) for p in COMPARATOR_PATTERNS]
COMPARATOR_NOUN_RE = re.compile(
    r"(?P<c>安慰剂(?:组)?|空白对照(?:组)?|常规治疗(?:组)?|标准治疗(?:组)?|"
    r"阳性对照药?|对照(?:组)?|假手术(?:组)?|usual care|placebo(?: group)?|"
    r"standard[- ]of[- ]care(?: group)?|control group)",
    re.IGNORECASE,
)

# 人群
POP_IN_RE = re.compile(
    r"在(?P<pop>[^。；;]{2,45}?(?:受试者|研究对象|参与者|患者|病人|人群|居民|老人|老年人|儿童|婴儿|孕妇|志愿者|成人))中"
)
POP_NA_RE = re.compile(r"纳入(?P<pop>[^，,。；;]{2,45})")
POP_NOUN_RE = re.compile(
    r"(?P<pop>[^，,。；;]{2,30}?(?:人群|受试者|研究对象|参与者|患者|病人|居民|老年人|老人|儿童|婴儿|孕妇|志愿者|成人))"
)
POP_LEAD_STRIP_RE = re.compile(
    r"^(?:建议|推荐|应当|应该|应|可考虑|可以考虑|可以|可|针对|对于|对|适用于|所有|广大|令|让|使|该|此)+"
)
POP_EN_RE = re.compile(
    r"\b(?:in|among)\s+(?P<pop>[^.,;]{2,60}?\b(?:adults?|patients?|subjects?|participants?|children|elderly|older adults|women|men)\b)",
    re.IGNORECASE,
)

BROAD_POP_WORDS = ["高龄", "老年", "老人", "中老年", "成年", "成人", "普通人群", "一般人群", "全人群", "各年龄", "儿童", "青少年", "孕妇", "女性", "男性"]

# 年龄
AGE_MIN_RE = re.compile(r"(?P<n>\d{1,3})\s*岁\s*(?:及)?(?:以上|以上者|或更大)")
AGE_MAX_RE = re.compile(r"(?P<n>\d{1,3})\s*岁\s*(?:及)?(?:以下|以内|或更小)")
AGE_RANGE_RE = re.compile(r"(?P<lo>\d{1,3})\s*岁?\s*[~\-—–―至到]\s*(?P<hi>\d{1,3})\s*岁")
AGE_EN_RE = re.compile(r"(?:aged?\s*)?(?P<lo>\d{2,3})\s*(?:[-–to]{1,4})\s*(?P<hi>\d{2,3})\s*years?", re.IGNORECASE)

# 剂量
DOSE_RE = re.compile(
    r"(?P<dose>(?:每日|每天|每周|隔日)?\s*\d+(?:\.\d+)?\s*"
    r"(?:mg|MG|毫克|g|克|ml|mL|毫升|μg|ug|微克|IU|单位|片|粒|滴)"
    r"(?:\s*/\s*(?:日|天|kg|公斤|周))?)"
)
DOSE_WORD_RE = re.compile(r"(?P<dose>低剂量|高剂量|中等剂量|常规剂量|标准剂量|减半剂量)")
CONCENTRATION_RE = re.compile(r"(?P<dose>\d+(?:\.\d+)?%)")
FREQUENCY_RE = re.compile(
    r"(?P<freq>(?:每日|每天|一日|1日|每周)\s*\d+\s*次|每日一次|每天一次|一日一次|每周一次|\bbid\b|\btid\b|\bqd\b)",
    re.IGNORECASE,
)

# 随访时长
FOLLOWUP_RE = re.compile(r"(?P<w>\d{1,3})\s*(?:周|个月|月|年)(?:的?)?(?:随访|跟进)?")

# 结果方向：使/让 X 下降……
RESULT_CAUSATIVE_RE = re.compile(
    r"[使让令](?P<out>[^，,。；;]{2,28}?)"
    r"(?:平均|显著|明显|大幅|轻度|有所)?(?P<verb>下降|降低|减少|减轻|升高|上升|增高|增加|增多|改善|好转|提高|提升)"
)
RESULT_MAKE_RE = re.compile(
    r"(?P<out>[^，,。；;]{2,28}?)"
    r"(?:平均|显著|明显|大幅|轻度|有所)?(?P<verb>下降|降低|减少|减轻|升高|上升|增高|增加|增多|改善|好转|提高|提升)"
)
# “降低住院风险”这类动宾前置：结局一直捕获到分句末/标点
RESULT_VERB_OUT_RE = re.compile(
    r"(?P<verb>降低|减少|减轻|缩短|升高|提高|增加|加重)"
    r"(?:了|过)?(?P<out>[^，,。；;]{1,25}?)"
    r"(?=[，,。；;]|$)"
)
# 裸“有效/疗效”
EFFICACY_RE = re.compile(r"(?P<iv>[一-龥A-Za-z0-9]{1,14}?(?:方案|疗法|干预|制剂|药物|疫苗))?[^。；;]{0,12}?(?:证明|证实|表明|显示|提示)?[^。；;]{0,6}?有效(?:性|率)?")

# 英文结果句：reduced LDL cholesterol by 12% ...
RESULT_EN_RE = re.compile(
    r"\b(?P<verb>reduced|decreased|lowered|improved|increased|raised|worsened)\s+"
    r"(?P<out>[A-Za-z][A-Za-z0-9\- ]{1,40}?)"
    r"(?=\s+by\b|\s+compared\b|\s+versus\b|\svs\.?\b|[,.;]|$)",
    re.IGNORECASE,
)
_EN_VERB_MAP = {
    "reduced": "降低", "decreased": "下降", "lowered": "降低",
    "improved": "改善", "increased": "升高", "raised": "升高", "worsened": "加重",
}

# 结局概念词表（顺序即优先级）
OUTCOME_CONCEPTS: list[tuple[str, str, list[str]]] = [
    ("hospitalization", OutcomeClass.PATIENT.value,
     ["住院率", "住院风险", "住院次数", "再住院", "入院率", "住院", "hospitalization", "hospital admission", "inpatient"]),
    ("mortality", OutcomeClass.PATIENT.value,
     ["全因死亡率", "死亡率", "死亡风险", "全因死亡", "病死率", "死亡", "mortality", "death", "all-cause"]),
    ("cvd_event", OutcomeClass.PATIENT.value,
     ["心血管事件", "主要不良心血管事件", "心梗", "心肌梗死", "脑卒中", "中风", "卒中", "发病", "急性事件",
      "cardiovascular event", "myocardial infarction", "stroke", "mace"]),
    ("fall", OutcomeClass.PATIENT.value,
     ["跌倒发生率", "跌倒风险", "跌倒次数", "跌倒", "fall rate", "risk of falls", "falls"]),
    ("safety", OutcomeClass.SAFETY.value,
     ["不良反应", "副作用", "不良事件", "严重不良", "安全", "adverse event", "side effect", "safety"]),
    ("quality_of_life", OutcomeClass.SYMPTOM.value,
     ["生活质量", "生命质量", "活动能力", "功能", "quality of life", "physical function"]),
    ("symptom", OutcomeClass.SYMPTOM.value,
     ["症状", "疼痛", "疲劳", "失眠", "焦虑", "抑郁", "symptom", "pain", "fatigue"]),
    ("cognition", OutcomeClass.SYMPTOM.value,
     ["认知功能", "认知评分", "认知能力", "认知障碍", "认知", "cognitive function", "cognition", "cognitive score"]),
    ("skin_lesion", OutcomeClass.SYMPTOM.value,
     ["皮损数量", "皮损计数", "皮损", "痤疮", "皮疹", "湿疹", "lesion count", "skin lesion", "acne", "rash"]),
    ("surrogate_generic", OutcomeClass.SURROGATE.value,
     ["替代指标", "替代终点", "生物标志物", "标志物", "指标", "水平", "bp", "biomarker", "surrogate endpoint", "surrogate marker"]),
    ("blood_pressure", OutcomeClass.SURROGATE.value, ["血压", "收缩压", "舒张压", "blood pressure", "systolic"]),
    ("lipid", OutcomeClass.SURROGATE.value, ["血脂", "胆固醇", "低密度脂蛋白", "甘油三酯", "ldl", "hdl", "cholesterol"]),
    ("glucose", OutcomeClass.SURROGATE.value, ["血糖", "糖化血红蛋白", "hba1c", "glucose"]),
]

# 对“有害方向”的结局，下降=有利
HARMFUL_OUTCOME_WORDS = [
    "风险", "率", "死亡", "住院", "发病", "事件", "负担", "不良", "副作用", "疼痛",
    "偏高", "异常", "炎症", "指标", "水平", "评分",
]
BENEFICIAL_UP_WORDS = ["生活质量", "评分", "功能", "生存率", "有效率", "改善率"]

DOWN_VERBS = {"下降", "降低", "减少", "减轻", "缩短"}
UP_BAD_VERBS = {"升高", "上升", "增高", "增加", "增多", "加重"}
GOOD_VERBS = {"改善", "好转", "提高", "提升"}


def classify_outcome(outcome: str | None) -> tuple[str | None, str]:
    """返回 (标准概念, OutcomeClass 值)。"""
    if not outcome:
        return None, OutcomeClass.UNKNOWN.value
    norm = compact(outcome).lower()
    for concept, cls, words in OUTCOME_CONCEPTS:
        for w in words:
            if compact(w).lower() in norm:
                return concept, cls
    return None, OutcomeClass.UNKNOWN.value


_DIRECTION_NOTE_IMPROVE = "“{verb}”指向改善，按有利方向解读"


def _direction_for(verb: str, outcome: str | None, clause: str) -> tuple[str, str, str | None]:
    """根据动词方向与结局“好/坏取向”判断临床方向。"""
    norm_out = compact(outcome or "")
    if verb in GOOD_VERBS:
        return Direction.FAVORABLE.value, EvidentialForce.SUPPORTS_BENEFIT.value, _DIRECTION_NOTE_IMPROVE.format(verb=verb)
    harmful = any(w in norm_out for w in HARMFUL_OUTCOME_WORDS)
    beneficial_up = any(w in norm_out for w in BENEFICIAL_UP_WORDS)
    if verb in DOWN_VERBS:
        if harmful or "替代指标" in norm_out or "指标" in norm_out:
            return Direction.FAVORABLE.value, EvidentialForce.SUPPORTS_BENEFIT.value, "有害/偏高指标下降按有利方向解读"
        return Direction.FAVORABLE.value, EvidentialForce.SUPPORTS_BENEFIT.value, "下降类结果按有利方向解读，需结合结局临床含义复核"
    if verb in UP_BAD_VERBS:
        if beneficial_up:
            return Direction.FAVORABLE.value, EvidentialForce.SUPPORTS_BENEFIT.value, "有益指标上升"
        if harmful:
            return Direction.UNFAVORABLE.value, EvidentialForce.SUPPORTS_HARM.value, "有害结局上升"
        return Direction.UNCLEAR.value, EvidentialForce.UNKNOWN.value, "上升方向的临床含义需学科审阅者确认"
    return Direction.UNCLEAR.value, EvidentialForce.UNKNOWN.value, None


def extract_population_qualifiers(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """从文本中抽取人群短语与年龄/基础状况等限定。"""
    population: str | None = None
    m = POP_IN_RE.search(text)
    if m:
        population = compact(m.group("pop"))
    else:
        m = POP_NA_RE.search(text)
        if m:
            population = compact(m.group("pop"))
        else:
            m_en = POP_EN_RE.search(text)
            if m_en:
                population = spaced(m_en.group("pop"))
            else:
                m_noun = POP_NOUN_RE.search(text)
                if m_noun:
                    population = POP_LEAD_STRIP_RE.sub("", compact(m_noun.group("pop"))) or None
    if population is None:
        for w in BROAD_POP_WORDS:
            if w in compact(text):
                population = w + "人群" if not w.endswith("人群") and "人群" not in w else w
                break

    qualifiers: list[dict[str, Any]] = []
    for m_age in AGE_RANGE_RE.finditer(text):
        qualifiers.append(
            {"type": "age_range", "min_age": int(m_age.group("lo")), "max_age": int(m_age.group("hi")),
             "raw": compact(m_age.group(0))}
        )
    m_min = AGE_MIN_RE.search(text)
    if m_min and not any(q["type"] == "age_range" for q in qualifiers):
        qualifiers.append({"type": "min_age", "min_age": int(m_min.group("n")), "raw": compact(m_min.group(0))})
    m_max = AGE_MAX_RE.search(text)
    if m_max and not any(q["type"] == "age_range" for q in qualifiers):
        qualifiers.append({"type": "max_age", "max_age": int(m_max.group("n")), "raw": compact(m_max.group(0))})

    # 人群短语内部 “且/并/同时” 引导的额外准入条件
    if population:
        for m_extra in re.finditer(r"(?:且|并且|并|同时|伴|合并)(?P<c>[^，,。；;]{2,30})", population):
            cond = compact(m_extra.group("c"))
            # 去掉条件末尾被一起吞入的中心词
            cond = re.sub(r"(?:的)?(?:受试者|研究对象|参与者|志愿者|患者|病人|人群|居民|老人|儿童)$", "", cond)
            if cond and not any(q.get("condition") == cond for q in qualifiers):
                qualifiers.append({"type": "eligibility_condition", "condition": cond, "raw": cond})
    return population, qualifiers


def extract_dose_and_duration(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in DOSE_RE.finditer(text):
        out.append({"type": "dose", "value": compact(m.group("dose")), "raw": spaced(m.group("dose"))})
    for m in CONCENTRATION_RE.finditer(text):
        out.append({"type": "dose", "value": compact(m.group("dose")), "raw": m.group("dose")})
    for m in DOSE_WORD_RE.finditer(text):
        out.append({"type": "dose_descriptor", "value": m.group("dose"), "raw": m.group("dose")})
    for m in FREQUENCY_RE.finditer(text):
        out.append({"type": "frequency", "value": compact(m.group("freq")), "raw": spaced(m.group("freq"))})
    for m in FOLLOWUP_RE.finditer(text):
        unit = m.group(0)[len(m.group("w")):].strip()
        out.append({"type": "follow_up", "value": int(m.group("w")), "unit": unit, "raw": compact(m.group(0))})
    # 去重
    seen: set[tuple[str, str]] = set()
    uniq = []
    for q in out:
        key = (q["type"], str(q.get("value", "")))
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq


def extract_comparator(text: str) -> tuple[str | None, str]:
    for rx in COMPARATOR_RES:
        m = rx.search(text)
        if m:
            groups = m.groupdict()
            comp = groups.get("b") or groups.get("b2")
            if comp:
                comp = compact(comp)
                # “人群与对照组不变”这类句子会把“对照组”吞进 b
                return comp, "comparison_phrase"
    m = COMPARATOR_NOUN_RE.search(text)
    if m:
        return compact(m.group("c")), "control_named"
    if re.search(r"对照|compare|versus|vs\.", text, re.IGNORECASE):
        return "对照组（原文未具体说明）", "control_named"
    return None, "not_specified"


# 紧邻“方案/疗法”等后缀时，用于截断左侧情态/人群词语
_INTERVENTION_LEAD_STOP_RE = re.compile(
    r".*(?:人群|受试者|患者|对象|居民|老人|儿童|我们|研究|"
    r"可考虑|可以考虑|建议|推荐|应当|应该|应|可|拟|接受|使用|采用|给予|服用|摄入|注射|启动|加用|联用|"
    r"说明|指出|认为|提出|声称|称|写道|提到|提及|发现|显示|表明)"
)
_INTERVENTION_CODE_RE = re.compile(r"[甲乙丙丁戊己庚辛壬癸A-Za-z0-9]{1,3}\s*(?:号)?方案")


def extract_intervention(text: str) -> str | None:
    # 优先识别“甲方案 / B 方案”这类代号式干预，避免把“高龄人群可考虑”吞入
    m_code = _INTERVENTION_CODE_RE.search(text or "")
    if m_code:
        return compact(m_code.group(0))
    m = INTERVENTION_RE.search(text or "")
    if m:
        name = compact(m.group("iv"))
        m_stop = _INTERVENTION_LEAD_STOP_RE.match(name)
        if m_stop and m_stop.end() < len(name):
            name = name[m_stop.end() :]
        return name or None
    m = INTERVENTION_EN_RE.search(text or "")
    if m:
        return m.group("iv")
    return None


def _base_claim(unit: dict[str, Any], record_text: str) -> dict[str, Any]:
    clause_raw = unit["clause"]
    sentence_raw = unit["sentence"]
    population, pop_q = extract_population_qualifiers(clause_raw)
    if population is None:
        population, pop_q = extract_population_qualifiers(record_text)
    intervention = extract_intervention(clause_raw) or extract_intervention(record_text)
    comparator, comp_status = extract_comparator(clause_raw)
    if comp_status == "not_specified":
        comparator, comp_status = extract_comparator(record_text)
    qualifiers = pop_q + extract_dose_and_duration(clause_raw)
    return {
        "source_sentence": sentence_raw.strip(),
        "source_clause": clause_raw.strip(),
        "population": population,
        "population_qualifiers": pop_q,
        "intervention": intervention,
        "comparator": comparator,
        "comparator_status": comp_status,
        "effect_qualifiers": [q for q in qualifiers if q["type"] in {"dose", "dose_descriptor", "frequency", "follow_up"}],
    }


def extract_claims_from_text(text: str) -> list[dict[str, Any]]:
    """从一条来源记录正文中抽取全部主张（不含 record_id/claim_id）。"""
    claims: list[dict[str, Any]] = []
    units = iter_clauses(text)
    if not units and text.strip():
        units = [{"clause": text, "clause_start": 0, "clause_end": len(text),
                  "sentence": text, "sentence_start": 0, "sentence_end": len(text)}]

    for idx, unit in enumerate(units):
        clause = unit["clause"]
        base = _base_claim(unit, text)
        claim: dict[str, Any] = dict(base)
        claim["claim_type"] = "result"
        claim["outcome"] = None
        claim["outcome_concept"] = None
        claim["outcome_class"] = OutcomeClass.UNKNOWN.value
        claim["direction"] = Direction.UNCLEAR.value
        claim["evidential_force"] = EvidentialForce.UNKNOWN.value
        claim["direction_note"] = None
        claim["confidence"] = "medium"
        claim["extraction"] = {"clause_index": idx}

        # 1) 明确“未测量某结局” -----------------------------------------
        m_un = UNMEASURED_RE.search(clause) or UNMEASURED_EN_RE.search(clause)
        if m_un:
            outcome = compact(m_un.group("out"))
            outcome = re.sub(r"^(?:了|过|的)?", "", outcome)
            concept, cls = classify_outcome(outcome)
            claim.update(
                claim_type="outcome_not_measured",
                outcome=outcome or None,
                outcome_concept=concept,
                outcome_class=cls,
                direction=Direction.NOT_ASSESSED.value,
                evidential_force=EvidentialForce.OUTCOME_NOT_STUDIED.value,
                direction_note="原文明确说明该结局未被测量/报告，属于证据缺口而非阴性结果",
                confidence="high",
            )
            claims.append(claim)
            continue

        # 2) “未发现差异” / 证据不足 ------------------------------------
        m_null = NULL_RESULT_RE.search(clause)
        if m_null and not NO_EFFECT_PROOF_RE.search(clause):
            outcome = None
            m_caus = RESULT_CAUSATIVE_RE.search(clause)
            m_ro = m_caus or RESULT_MAKE_RE.search(clause)
            if m_ro:
                outcome = _clean_outcome_phrase(compact(m_ro.group("out")), claim.get("intervention"))
            hit = _find_known_outcome(clause)
            if hit:
                concept, cls, canonical = hit
                # 若抽取出的短语过长或含 null/范围上下文，用规范结局词
                if not outcome or len(outcome) > len(canonical) + 4 or re.search(r"未发现|未|两组|剂量|组中", outcome):
                    outcome = canonical
            else:
                concept, cls = classify_outcome(outcome)
            claim.update(
                outcome=outcome,
                outcome_concept=concept,
                outcome_class=cls,
                direction=Direction.NO_SIG_DIFF.value,
                evidential_force=EvidentialForce.INCONCLUSIVE.value,
                direction_note="原文为未检出差异/证据不足：不能支持阳性功效，也不能改写为“证明无效”",
                confidence="high",
                extraction={**claim["extraction"], "null_phrase": m_null.group(0)},
            )
            claims.append(claim)
            continue

        # 3) 明确声称“无效”（需强设计支撑，先如实记录方向） -------------
        m_noeff = NO_EFFECT_PROOF_RE.search(clause)
        if m_noeff:
            hit = _find_known_outcome(clause)
            concept, cls, outcome = (hit[0], hit[1], hit[2]) if hit else (None, OutcomeClass.UNKNOWN.value, None)
            claim.update(
                outcome=outcome,
                outcome_concept=concept,
                outcome_class=cls,
                direction=Direction.NO_EFFECT.value,
                evidential_force=EvidentialForce.DEMONSTRATES_NO_EFFECT.value,
                direction_note="原文作出无效应结论；需核查是否有等效性/足够把握度设计支撑",
                confidence="medium",
            )
            claims.append(claim)
            continue

        # 4) 方向性结果（先“降低住院风险”式动宾，再“使指标下降”式） ----
        m_vo = RESULT_VERB_OUT_RE.search(clause)
        if m_vo and _vo_is_result(clause, m_vo):
            outcome = compact(m_vo.group("out"))
            verb = m_vo.group("verb")
            negated = _is_negated_result(clause, m_vo)
            concept, cls = classify_outcome(outcome)
            if negated:
                claim.update(
                    outcome=outcome, outcome_concept=concept, outcome_class=cls,
                    direction=Direction.NO_SIG_DIFF.value,
                    evidential_force=EvidentialForce.INCONCLUSIVE.value,
                    direction_note=f"原文为否定式结果（未/没有…{verb}），按未检出效应处理，不能写成“证明无效”",
                    confidence="medium",
                    extraction={**claim["extraction"], "verb": verb, "negated": True},
                )
            else:
                direction, force, note = _direction_for(verb, outcome, clause)
                claim.update(
                    outcome=outcome, outcome_concept=concept, outcome_class=cls,
                    direction=direction, evidential_force=force,
                    direction_note=note.format(verb=verb) if note and "{" in note else note,
                    confidence="medium",
                    extraction={**claim["extraction"], "verb": verb},
                )
            claims.append(claim)
            continue

        # 5) “使替代指标平均下降” / “指标下降”式 -------------------------
        m_make = RESULT_CAUSATIVE_RE.search(clause)
        if not (m_make and _looks_like_result(m_make)):
            m_generic = RESULT_MAKE_RE.search(clause)
            # 通用模式可能从干预名开始吞，要求结局短语中不含“方案/疗法”等后缀
            if m_generic and _looks_like_result(m_generic) and not re.search(r"方案|疗法|药物|制剂", m_generic.group("out") or ""):
                m_make = m_generic
            else:
                m_make = None
        if m_make:
            outcome = _clean_outcome_phrase(compact(m_make.group("out")), claim.get("intervention"))
            verb = m_make.group("verb")
            negated = _is_negated_result(clause, m_make)
            concept, cls = classify_outcome(outcome)
            if negated:
                claim.update(
                    outcome=outcome, outcome_concept=concept, outcome_class=cls,
                    direction=Direction.NO_SIG_DIFF.value,
                    evidential_force=EvidentialForce.INCONCLUSIVE.value,
                    direction_note=f"原文为否定式结果（未/没有…{verb}），按未检出效应处理，不能写成“证明无效”",
                    confidence="medium",
                    extraction={**claim["extraction"], "verb": verb, "negated": True},
                )
            else:
                direction, force, note = _direction_for(verb, outcome, clause)
                claim.update(
                    outcome=outcome,
                    outcome_concept=concept,
                    outcome_class=cls,
                    direction=direction,
                    evidential_force=force,
                    direction_note=note.format(verb=verb) if note and "{" in note else note,
                    confidence="high" if cls != OutcomeClass.UNKNOWN.value else "medium",
                    extraction={**claim["extraction"], "verb": verb},
                )
            claims.append(claim)
            continue

        # 5b) 英文方向性结果 ---------------------------------------------
        m_en = RESULT_EN_RE.search(clause)
        if m_en:
            outcome = spaced(m_en.group("out")).strip()
            zh_verb = _EN_VERB_MAP.get(m_en.group("verb").lower(), "改善")
            direction, force, note = _direction_for(zh_verb, outcome, clause)
            concept, cls = classify_outcome(outcome)
            claim.update(
                outcome=outcome, outcome_concept=concept, outcome_class=cls,
                direction=direction, evidential_force=force,
                direction_note=f"English result verb '{m_en.group('verb')}' mapped to {zh_verb}",
                confidence="medium",
                extraction={**claim["extraction"], "verb": m_en.group("verb")},
            )
            claims.append(claim)
            continue

        # 6) 裸“有效” ---------------------------------------------------
        m_eff = EFFICACY_RE.search(clause)
        eff_prefix = clause[: m_eff.start()] if m_eff else ""
        negated_efficacy = bool(re.search(r"未|没有|并无|并不|难以|无法|未能|尚不能|证据不足|缺乏", eff_prefix[-18:]))
        if m_eff and not negated_efficacy and not re.search(r"未|无|不|缺乏", clause[: max(0, m_eff.start()) + 1]):
            iv = m_eff.group("iv") if m_eff.groupdict().get("iv") else claim["intervention"]
            claim.update(
                claim_type="efficacy_general",
                intervention=compact(iv) if iv else claim["intervention"],
                outcome=None,
                direction=Direction.FAVORABLE.value,
                evidential_force=EvidentialForce.SUPPORTS_BENEFIT.value,
                direction_note="仅见概括性“有效”表述，未给出具体结局，需回查正文限定",
                confidence="medium",
            )
            claims.append(claim)
            continue

        # 其余分句多为关系声明/背景，不产生主张
    return claims


def _find_known_outcome(text: str) -> tuple[str, str, str] | None:
    norm = compact(text).lower()
    best: tuple[int, str, str, str] | None = None
    for concept, cls, words in OUTCOME_CONCEPTS:
        for w in words:
            cw = compact(w).lower()
            idx = norm.find(cw)
            if idx >= 0:
                # 取最长匹配，避免“指标”抢先于“替代指标”
                if best is None or len(cw) > best[0]:
                    best = (len(cw), concept, cls, w)
    if best:
        _, concept, cls, w = best
        return concept, cls, w
    return None


def _looks_like_result(m: re.Match[str]) -> bool:
    out = m.group("out") or ""
    # 过滤明显不是结局的短语
    if len(out) < 2:
        return False
    if out.endswith(("时", "后", "中", "里")):
        return False
    return True


def _clean_outcome_phrase(outcome: str | None, intervention: str | None = None) -> str | None:
    """去掉结局短语左侧混入的干预名/使令词。"""
    if not outcome:
        return outcome
    out = compact(outcome)
    # 截断到最后一个使令词之后
    m_caus = list(re.finditer(r"[使让令]", out))
    if m_caus:
        out = out[m_caus[-1].end() :]
    if intervention:
        iv = compact(intervention)
        if iv and iv in out:
            out = out.split(iv, 1)[1] if out.startswith(iv) else out.replace(iv, "", 1)
    # 仅清理残留的使令词；“治疗组/试验组”等前缀保留也不影响概念归类
    out = re.sub(r"^[使让令]+", "", out)
    return out or None


def _vo_is_result(clause: str, m: re.Match[str]) -> bool:
    out = compact(m.group("out") or "")
    if len(out) < 2:
        return False
    if classify_outcome(out)[1] != OutcomeClass.UNKNOWN.value:
        return True
    if any(k in out for k in ["风险", "率", "水平", "指标", "评分", "次数", "数量", "负担", "事件", "时间", "程度"]):
        return True
    return False


_NEGATION_BEFORE_VERB_RE = re.compile(r"(?:未|没有|并没|并未|并无|未能|无法|鲜少|很少)\s*$")


def _is_negated_result(clause: str, m: re.Match[str]) -> bool:
    """检查结果动词前紧邻位置是否有否定词（如“未降低”“并未下降”）。"""
    verb_start = m.start("verb") if "verb" in m.groupdict() else m.start()
    window = clause[max(0, verb_start - 6):verb_start]
    return bool(_NEGATION_BEFORE_VERB_RE.search(window))


# ---------------------------------------------------------------------------
# 建议文本解析（复用同一套词表）
# ---------------------------------------------------------------------------

def parse_guidance_text(text: str) -> dict[str, Any]:
    """解析已发布建议：人群、干预、目标结局、期望方向、建议强度。"""
    population, pop_q = extract_population_qualifiers(text)
    intervention = extract_intervention(text)

    outcome = None
    concept = cls = None
    desired = None

    m_vo = re.search(r"(?P<verb>降低|减少|减轻|预防|避免|缩短)(?:了)?(?P<out>[^，,。；;]{1,25})", text)
    if m_vo:
        outcome = compact(m_vo.group("out"))
        desired = "decrease"
        concept, cls = classify_outcome(outcome)
    else:
        hit = _find_known_outcome(text)
        if hit:
            concept, cls, outcome = hit
            if any(w in text for w in ["降低", "减少", "减轻", "预防"]):
                desired = "decrease"
            elif any(w in text for w in ["提高", "改善"]):
                desired = "increase_or_improve"

    strength = "statement"
    for word, code in [("可考虑", "weak_consider"), ("建议", "recommend"), ("推荐", "strong"),
                       ("应当", "strong"), ("应", "recommend"), ("可以", "permissive")]:
        if word in text:
            strength = code
            break

    return {
        "population": population,
        "population_qualifiers": pop_q,
        "intervention": intervention,
        "outcome": outcome,
        "outcome_concept": concept,
        "outcome_class": cls,
        "desired_direction": desired,
        "strength": strength,
    }
