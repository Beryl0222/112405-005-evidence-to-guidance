"""evidence_workbench —— 可反复处理资料的证据链 NLP 工作台。

公开 API：
    process_case(case_dict, status_overrides=None) -> report dict
    process_path(path, outdir=None, status_overrides=None) -> report dict
    review_after_change(case, changes) -> impact dict
    diff_reports(old, new) -> diff dict
    audit_wording(text) -> issues
    extract_claims_from_text(text) -> claims
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import audit_wording
from .change import diff_reports, impacted_guidance_ids, review_after_change
from .nlp import extract_claims_from_text, parse_guidance_text
from .pipeline import DISCLAIMER, process_case
from .report import render_markdown, write_csv_exports

__all__ = [
    "process_case",
    "process_path",
    "review_after_change",
    "diff_reports",
    "impacted_guidance_ids",
    "audit_wording",
    "extract_claims_from_text",
    "parse_guidance_text",
    "render_markdown",
    "DISCLAIMER",
]

__version__ = "1.0.0"


def process_path(
    path: str | Path,
    *,
    outdir: str | Path | None = None,
    status_overrides: dict[str, str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """读取 case JSON、运行管线，并可选写出 JSON/Markdown/队列/主张文件。"""
    p = Path(path)
    case = json.loads(p.read_text(encoding="utf-8-sig"))
    report = process_case(case, status_overrides=status_overrides)
    if write and outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "evidence_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "evidence_report.md").write_text(render_markdown(report), encoding="utf-8")
        (out / "re_review_queue.json").write_text(
            json.dumps(report["re_review_queue"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "claims.json").write_text(
            json.dumps(report["claims"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv_exports(report, out)
    return report
