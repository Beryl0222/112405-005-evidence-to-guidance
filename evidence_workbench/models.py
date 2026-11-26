"""数据模型与稳定枚举。

所有结构化结果都可以直接 ``json.dumps(report, ensure_ascii=False, default=json_default)``。
枚举同时继承 ``str``，因此 ``Direction.FAVORABLE == "favorable"``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Direction(str, Enum):
    """主张的效应方向。

    关键区分：``NO_SIG_DIFF`` 表示“未检出差异/证据不足”，
    它既不支持阳性建议，也不能被改写为“证明无效”。
    """

    FAVORABLE = "favorable"                       # 有利方向（如有害指标下降）
    UNFAVORABLE = "unfavorable"                   # 不利/有害方向
    NO_SIG_DIFF = "no_significant_difference"     # 未发现差异（不确定，证据不足）
    NO_EFFECT = "no_effect_demonstrated"          # 有等效性等强证据证明无临床意义效应
    MIXED = "mixed"
    NOT_ASSESSED = "not_assessed"                 # 该结局未被研究测量
    UNCLEAR = "unclear"


class EvidentialForce(str, Enum):
    """方向背后的证据含义，防止把“未发现差异”当成“无效”。"""

    SUPPORTS_BENEFIT = "supports_benefit_claim"
    SUPPORTS_HARM = "supports_harm_claim"
    INCONCLUSIVE = "inconclusive_no_detected_difference"
    DEMONSTRATES_NO_EFFECT = "demonstrates_no_clinically_meaningful_effect"
    OUTCOME_NOT_STUDIED = "outcome_not_studied"
    UNKNOWN = "unknown"


class OutcomeClass(str, Enum):
    PATIENT = "patient_important"     # 市民关心的硬结局：住院、死亡、心血管事件
    SURROGATE = "surrogate"           # 替代指标/生物标志物
    SAFETY = "safety"                 # 不良反应
    SYMPTOM = "symptom_or_function"   # 症状/功能/生活质量
    UNKNOWN = "unknown"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    CORRECTED = "corrected"
    RETRACTED = "retracted"
    CONCERN = "expression_of_concern"


class Decision(str, Enum):
    MAINTAIN = "maintain"   # 维持现行建议
    NARROW = "narrow"       # 缩小适用范围/改写表述
    REOPEN = "reopen"       # 重新讨论（重审）


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    INFO = "info"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / Enum 转成 JSON 友好的普通结构。"""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    return obj


@dataclass
class Claim:
    """一项可复核主张。

    ``source_sentence`` 必须是来源记录 ``text`` 的原样子串，
    ``source_clause`` 是更短的命题分句（同样是原样子串）。
    """

    claim_id: str
    record_id: str
    claim_type: str  # result | outcome_not_measured | efficacy_general
    source_sentence: str
    source_clause: str
    population: str | None = None
    population_qualifiers: list[dict[str, Any]] = field(default_factory=list)
    intervention: str | None = None
    comparator: str | None = None
    comparator_status: str = "not_specified"  # reported | control_named | not_specified
    outcome: str | None = None
    outcome_concept: str | None = None
    outcome_class: str = OutcomeClass.UNKNOWN.value
    direction: str = Direction.UNCLEAR.value
    evidential_force: str = EvidentialForce.UNKNOWN.value
    direction_note: str | None = None
    effect_qualifiers: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"  # high | medium | low
    extraction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))
