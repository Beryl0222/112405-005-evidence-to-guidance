"""表述审计：守住“未发现差异 ≠ 证明无效”等证据表述边界。"""

from __future__ import annotations

import re
from typing import Any

from .nlp import NULL_RESULT_RE, iter_sentences, compact

# 把不确定结果改写成“无效”的肯定/因果连接
_DISTORT_CONNECTOR_RE = re.compile(
    r"(?:因此|所以|故|因而|可见|这?说明|表明|证明|足以说明|意味着?|即|就是|等于|等同于|"
    r"可以认定|可判定|可以判定|据此认为|认为|结论是)?"
    r"[^。；;]{0,12}?"
    r"(?P<noeffect>无效|没有效果|无效果|没有疗效|无疗效|没有作用|无作用|不起作用|未见效)"
)

# 警示/否定框架：这些词管辖时，句子是在提醒“不能误写”，而非真的误写
_SHIELD_RE = re.compile(r"(?:不能|不应|不得|不可|无法|未能|并不|并非|不等于|不代表|不是|避免|防止|切勿|不要)")

# 直接把“未发现差异”与“无效”用等号/结论连接的强模式
_DIRECT_EQUATION_RE = re.compile(
    r"未(?:发现|见|能证实)?[^。；;]{0,12}?(?:差异|统计学意义|效果|疗效|改善)[^。；;]{0,10}?"
    r"(?:就是|即|等于|说明|表明|证明|因此|所以|故)[^。；;]{0,8}?(?:无效|没有效果|无效果)"
)

_EQUIVALENCE_DESIGN_RE = re.compile(r"等效性检验|非劣效设计|足够把握度|equivalence|non-inferiority", re.IGNORECASE)


def _shielded_locally(sentence: str, noeffect_start: int) -> bool:
    """判断“无效”结论是否处于“不能/不等于/并非……”的警示管辖范围内。

    只在 noeffect 所在的逗号/分号分句内向前找否定/警示词，
    避免前一个转折分句里的“不能说绝对”错误保护后一个肯定式误写。
    """
    clause_start = 0
    for m in re.finditer(r"[，,；;：:]", sentence[:noeffect_start]):
        clause_start = m.end()
    local_prefix = sentence[clause_start:noeffect_start]
    return bool(_SHIELD_RE.search(local_prefix))


def audit_wording(text: str, *, source_id: str | None = None) -> list[dict[str, Any]]:
    """审计一段文本中的证据表述问题。

    返回 issue 列表；每个 issue 带 ``source_sentence`` 原句，便于复核。
    规则刻意保守：只有“未发现差异/证据不足”与肯定式“无效”结论在同句形成
    因果/等同关系，且没有“不能/不应/不等于”等警示框架时才判为违规。
    """
    issues: list[dict[str, Any]] = []
    if not text:
        return issues

    for sentence, start, end in iter_sentences(text):
        clause_text = sentence.strip()
        has_null = bool(NULL_RESULT_RE.search(sentence))
        m_no = _DISTORT_CONNECTOR_RE.search(sentence)
        if has_null and m_no:
            no_start = m_no.start("noeffect")
            shielded = _shielded_locally(sentence, no_start)
            if not shielded:
                issues.append(
                    {
                        "code": "null_result_as_proof_of_no_effect",
                        "severity": "high",
                        "source_id": source_id,
                        "source_sentence": clause_text,
                        "message_zh": (
                            "句子把“未发现差异/证据不足”与“无效”结论直接连接。"
                            "未检出差异只说明现有证据不足以证明有效（inconclusive），"
                            "不能改写为“证明无效”；无效应结论需要等效性检验或足够把握度的证据。"
                        ),
                        "matched_phrase": m_no.group("noeffect"),
                    }
                )
                continue

        # 没有 null 前提，但出现裸“证明/表明……无效”且缺少等效性设计说明
        if not has_null and m_no:
            pre = sentence[: m_no.start("noeffect")]
            equiv_m = _EQUIVALENCE_DESIGN_RE.search(sentence)
            equiv_present = False
            if equiv_m:
                before = sentence[max(0, equiv_m.start() - 4): equiv_m.start()]
                equiv_present = not re.search(r"未|没|无|缺乏|缺少", before)
            if re.search(r"证明|证实|确证|确认|表明|显示|认为|声称|称", pre) and not equiv_present:
                issues.append(
                    {
                        "code": "no_effect_claim_needs_equivalence_evidence",
                        "severity": "medium",
                        "source_id": source_id,
                        "source_sentence": clause_text,
                        "message_zh": (
                            "句子作出“无效/无效果”的断言，但未说明等效性检验、足够把握度等设计支撑；"
                            "请学科审阅者核查后再使用该方向标签。"
                        ),
                        "matched_phrase": m_no.group("noeffect"),
                    }
                )
    return issues
