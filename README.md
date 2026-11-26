# 科研证据更新 → 公共建议适用性复核工作台

城市健康顾问组每季度把新研究纳入公共建议时，需要一条**可复核的证据链**：

- 摘要里的“有效”可能只适用于特定年龄、剂量；
- 有些研究只报告替代指标，不能直接支持住院、死亡等市民关心的结局；
- 同一试验的后续/随访分析不是第二项独立研究；
- 论文被更正或撤回后，仍在引用它的建议必须被找出来重审；
- “未发现差异 / 差异无统计学意义”是**证据不足（inconclusive）**，不能改写成“证明无效”。

本仓库提供一个**零第三方依赖、可反复运行、结果确定**的 Python NLP 能力，
从论文摘要、方法补充、会议批注等中文（兼基础英文）文本中抽取主张，
保留每项主张的**研究人群、比较条件、干预、结局、方向和来源原句**，
并支持从已发布建议一键回查支持研究、输出重审队列。

> 所有 `data/`、`examples/` 中的文献与人物均为虚构合成资料，
> 仅用于流程联调，**不承载任何真实医疗建议**。

---

## 1. 快速开始

环境要求 Python 3.11+，无需安装任何依赖。

```sh
# 处理仓库自带 data/case.json，结果写入 reports/
python3 main.py
# 或等价地
python3 -m evidence_workbench data/case.json --outdir reports

# 运行全部测试（资料稳定性 1 个 + 工作台规则 32 个，共 33 个）
python3 -m unittest discover -s tests -v
```

输出目录包含：

| 文件 | 用途 |
|---|---|
| `evidence_report.md` | 给学科审阅者/起草人的中文复核报告（主张链表、试验归并、逐项检查、改写草稿） |
| `evidence_report.json` | 完整机器可读证据链（稳定 schema 1.0） |
| `claims.json` / `claims.csv` | 主张级证据链明细，CSV 带中文表头、`utf-8-sig`，Excel 可直接打开 |
| `guidance_review.csv` | 每条建议的处置、触发原因、支持试验与追踪编号 |
| `re_review_queue.json` | 需要重审的建议队列（按 critical/high/medium 排序） |

加 `--check` 时，只要存在 `needs_review=true` 的建议，CLI 以退出码 `1` 结束，
可接入季度自动化巡检：

```sh
python3 -m evidence_workbench data/case.json --check   # G-3 需重审 → 退出码 1
```

---

## 2. 样例 case 的结论（可复核）

`data/case.json` 中：

- **AB-7**：在 *60 岁以上且基线指标偏高* 的受试者中，甲方案使**替代指标**下降；
  且明确写明“**本研究未测量住院率**”。
- **AB-8**：AB-7 **同一试验**的 24 周随访分析 → 归并为 1 项独立试验，不重复计数。
- **AB-9**：针对 AB-7 的**更正**通知 → AB-7 状态变为 `corrected`，并沿引用链影响 G-3。
- **G-3**：“高龄人群可考虑甲方案降低住院风险。”

工作台对 **G-3 给出 `reopen`（重新讨论/必须重审，优先级 high）**，触发：

1. `source_corrected`：支持研究已被更正（AB-9 → AB-7）；
2. `outcome_not_studied`：建议结局“住院风险/住院率”在 AB-7 中明确未测量；
3. `surrogate_only_for_patient_outcome`：只有替代指标，不能支持硬结局；
4. `population_scope_narrower`：证据限“60 岁以上 + 基线指标偏高”，建议泛化为“高龄人群”。

同时给出缩小/改写草稿（见 `reports/evidence_report.md` 第五节），
但**机器只提建议，维持 / 缩小 / 重新讨论的终判权在建议起草人**。

---

## 3. 季度更新与“证据状态变化影响哪些建议”

### 3.1 模拟状态变更并重跑

```sh
# 假设 AB-7 进一步被撤回
python3 -m evidence_workbench data/case.json --status AB-7=retracted
```

CLI 会同时计算基线与变更后结果，打印受影响建议、新进入重审名单与处置变化；
撤回会把 G-3 的优先级升到 `critical`，检查项变为 `source_retracted`。

Python API：

```python
import json
from evidence_workbench import process_case, review_after_change, impacted_guidance_ids

case = json.load(open("data/case.json", encoding="utf-8"))

report = process_case(case)                       # 确定性报告
impacted_guidance_ids(report, ["AB-9"])           # ['G-3']：更正通知变化也能反查到 G-3

out = review_after_change(case, {"AB-7": "retracted"})
out["newly_needs_review"]      # 原本 maintain 的建议会出现在这里
out["impacted_guidance_ids"]   # ['G-3']
out["updated_report"]          # 重跑后的完整报告
```

### 3.2 两期完整资料对比

`examples/quarter_q1.json`（建议维持）与 `examples/quarter_q2.json`
（新增同一试验随访、一项“未发现显著差异”的研究和一份更正）：

```sh
python3 -m evidence_workbench.compare examples/quarter_q1.json examples/quarter_q2.json
# => {"newly_needs_review": ["GQ-1"], "decision_changes": [maintain -> reopen]}
```

`examples/quarter_review_demo.json` 则在一个 case 内演示全部四种处置：
G-10 `maintain`、G-11 `narrow`（年龄+剂量更窄）、
G-12 `reopen`（唯一支持是“未发现差异”）、G-13 `reopen/critical`（支持研究被撤回）。

```sh
python3 main.py examples/quarter_review_demo.json --outdir reports/demo-quarter
```

---

## 4. NLP 抽取与证据语义

### 4.1 主张对象（`claims[]`）的关键字段

```jsonc
{
  "claim_id": "AB-7#C1",          // 稳定编号：记录号 + 主张序号
  "record_id": "AB-7",
  "claim_type": "result",         // result | outcome_not_measured | efficacy_general
  "population": "60岁以上且基线指标偏高的受试者",
  "population_qualifiers": [      // 结构化限定：年龄边界、额外准入条件
    {"type": "min_age", "min_age": 60, "raw": "60岁以上"},
    {"type": "eligibility_condition", "condition": "基线指标偏高"}
  ],
  "intervention": "甲方案",
  "comparator": null,             // 抽不到时为 null，绝不臆造
  "comparator_status": "not_specified", // reported | control_named | comparison_phrase | not_specified
  "outcome": "替代指标",
  "outcome_concept": "surrogate_generic",
  "outcome_class": "surrogate",   // patient_important | surrogate | safety | symptom_or_function | unknown
  "direction": "favorable",
  "evidential_force": "supports_benefit_claim",
  "source_sentence": "……",        // 必定是记录 text 的原样子串（source_verified=true）
  "source_clause": "……"            // 更短的命题分句，同样是原样子串
}
```

方向枚举（`evidence_workbench/models.py`）刻意区分：

| `direction` | 含义 | 能支持阳性建议吗 |
|---|---|---|
| `favorable` / `unfavorable` | 观察到有利/不利方向 | 视结局直接性与人群匹配而定 |
| `no_significant_difference` | 未发现差异、差异无统计学意义、证据不足 | **不能**；证据含义为 `inconclusive_*`，**也不等于无效证明** |
| `no_effect_demonstrated` | 有等效性等强设计支撑的无效应结论 | 仅在此强条件下使用 |
| `not_assessed` | 该结局未被测量（证据缺口） | 不能 |
| `unclear` / `mixed` | 需学科审阅者判断 | 不能直接使用 |

### 4.2 表述审计

`audit_wording(text)` 会抓出“把未发现差异写成证明无效”的句子，
返回带原句的 issue（代码 `null_result_as_proof_of_no_effect`）。
“**不能把未发现差异等同于无效**”这类警示句受局部否定框架保护，不会被误报。
管线会自动审计所有建议正文与来源文本（含会议批注），汇总到 `wording_issues`。

### 4.3 记录关系与独立性

- 从结构化字段（`follow_up_of` / `corrects` / `retracts` / `same_trial_as` /
  `trial_id` / `evidence_refs` 等）和正文编号提及中识别关系，关系保留依据原句与置信度；
- `same_trial_as` / `follow_up_of` 边经并查集归并为试验簇，一个簇 = 一项独立研究；
- `correction` / `retraction` / `expression-of-concern` / `methods-supplement` /
  `meeting-note` / `annotation` 是档案材料，不单独计为独立研究，
  但通过 `corrects` / `retracts` / `supplements` / `annotates` 边进入追踪链；
- 编号兼容 `AB-7` 与 `R1` 两种形式（后者按资料中的已知 id 做边界匹配）。

### 4.4 建议复核决策

每条建议输出逐项 `checks`（pass / info / warn / block）与机器建议动作：

- **reopen（重新讨论）**：撤回/更正/关注声明、建议结局未测量、只有替代指标、
  唯一支持是“未发现差异”、方向相反、找不到可追溯证据等；
- **narrow（缩小适用范围）**：证据年龄/准入条件/剂量比建议表述更窄，且无 reopen 事项；
- **maintain（维持）**：干预、人群、结局直接性、方向、来源状态均通过。

`trace.paths` 给出 `建议 → 研究 → 同一试验分析 / 更正撤回通知` 的完整边，
`supporting_studies` 给出试验级汇总。

---

## 5. 输入格式

最小输入即仓库现有的 `data/case.json`：

```json
{
  "case_id": "evidence-7",
  "records": [
    {"id": "AB-7", "kind": "abstract", "text": "……"},
    {"id": "AB-8", "kind": "follow-up-paper", "text": "……"},
    {"id": "AB-9", "kind": "correction", "text": "……"}
  ],
  "guidance": [
    {"id": "G-3", "text": "……", "evidence_refs": ["AB-7"]}
  ]
}
```

- `kind` 支持 `abstract/paper/methods(-supplement)/meeting-note/annotation/
  follow-up(-paper)/correction/retraction/expression-of-concern/guidance` 及其中文别名；
- 顶层键也兼容 `sources/documents` 与 `recommendations/advice`；
- 纯文本 NLP 不完整时，可用结构化字段补强且同样进入证据链：
  - 记录：`trial_id`、`same_trial_as`、`follow_up_of`、`corrects`、`retracts`、
    `status`、`qualifiers[]`、`claims[]`；
  - 建议：`evidence_refs`（或 `support_refs/refs/sources`）、`population`、
    `intervention`、`outcome`。
- 显式 `claims[].source_sentence` 会被校验是否为记录原文子串
  （结果中的 `source_verified=false` 即异常）。

---

## 6. Python API 速览

```python
from evidence_workbench import (
    process_case,            # dict -> report dict（纯函数、确定性）
    process_path,            # path -> report，并可写 JSON/MD/CSV
    review_after_change,     # case + 状态变更 -> 影响面 + 前后 diff
    impacted_guidance_ids,   # report + 变化记录 id -> 受影响建议 id
    diff_reports,            # 两期 report -> newly_needs_review 等
    audit_wording,           # 文本 -> 表述问题列表
    extract_claims_from_text,
)

report = process_path("data/case.json", outdir="reports")
```

确定性保证：`json.dumps(process_case(case), sort_keys=True)` 连续两次结果相同；
所有列表按稳定 id/优先级排序，报告不含运行时间戳，便于季度之间做文本 diff。

---

## 7. 目录结构

```
evidence_workbench/
  models.py      # 枚举与 Claim 数据模型
  nlp.py         # 中文规则 NLP：分句、PICO/限定/方向抽取、结局词表
  audit.py       # “未发现差异 ≠ 无效”表述审计
  relations.py   # 编号关系、同一试验归并、更正/撤回状态传播
  guidance.py    # 建议解析、回查、逐项检查与 maintain/narrow/reopen 决策
  change.py      # 状态变更影响分析与两期报告 diff
  pipeline.py    # 端到端纯函数编排
  report.py      # Markdown / CSV 导出
  __main__.py    # python -m evidence_workbench CLI
  compare.py     # python -m evidence_workbench.compare OLD NEW
main.py          # 仓库根便捷入口
data/case.json   # 题给稳定资料（测试依赖其编号与非空文本，勿删改 id）
examples/        # 额外虚构演练案例（季度对比、四类处置）
tests/           # 资料稳定性测试 + 32 个工作台规则测试
reports/         # 最近一次对 data/case.json 的产物
```

## 8. 设计边界（诚实声明）

- 这是**规则 NLP**，优势是每条结论可解释、可复核、零依赖；它不是完备句法解析器，
  未覆盖的表达会被显式标为 `unknown/not_specified/outcome_unverified` 而不是被臆造。
- 机器给出的 `suggested_action` 是**分流建议**：帮助确定先看哪条建议，
  不能替代学科审阅者对研究设计与效应量的判断，也不能替代起草人的公共决策。
