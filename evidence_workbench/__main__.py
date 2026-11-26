"""命令行入口。

用法：
    python3 -m evidence_workbench [case.json] [--outdir reports]
                                 [--status AB-7=retracted ...] [--check]

不带参数时处理仓库自带的 data/case.json，并把结果写入 reports/。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .change import review_after_change
from .pipeline import process_case
from .report import render_markdown, write_csv_exports


def _default_case_path() -> Path:
    cwd_candidate = Path.cwd() / "data" / "case.json"
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "case.json"


def _parse_status(items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--status 参数格式应为 ID=status，收到：{item}")
        rid, st = item.split("=", 1)
        overrides[rid.strip()] = st.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evidence_workbench",
        description="合成医学文本的证据链抽取与建议重审工作台（零第三方依赖）",
    )
    parser.add_argument("case", nargs="?", help="case JSON 路径；默认 data/case.json")
    parser.add_argument("--outdir", default="reports", help="报告输出目录（默认 reports）")
    parser.add_argument(
        "--status", action="append", default=[],
        help="模拟证据状态变更，可多次使用，如 --status AB-7=retracted",
    )
    parser.add_argument("--no-write", action="store_true", help="只运行并打印摘要，不写文件")
    parser.add_argument("--json", dest="json_stdout", action="store_true",
                        help="向 stdout 输出完整报告 JSON（不打印人类摘要）")
    parser.add_argument("--check", action="store_true", help="若存在需要重审的建议则以退出码 1 结束")
    args = parser.parse_args(argv)

    case_path = Path(args.case) if args.case else _default_case_path()
    case = json.loads(case_path.read_text(encoding="utf-8-sig"))
    overrides = _parse_status(args.status)

    impact = None
    if overrides:
        impact = review_after_change(case, overrides)
        report = impact["updated_report"]
    else:
        report = process_case(case)

    outdir = Path(args.outdir)
    if not args.no_write:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "evidence_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (outdir / "evidence_report.md").write_text(render_markdown(report), encoding="utf-8")
        (outdir / "re_review_queue.json").write_text(
            json.dumps(report["re_review_queue"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (outdir / "claims.json").write_text(
            json.dumps(report["claims"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv_exports(report, outdir)

    if args.json_stdout:
        print(json.dumps(report, ensure_ascii=False))
        return 1 if args.check and report["stats"]["re_review_queue_size"] else 0

    if impact is not None:
        print("== 证据状态变更影响 ==")
        print("变更记录：", ", ".join(impact["changed_record_ids"]))
        print("受影响建议：", ", ".join(impact["impacted_guidance_ids"]) or "（无）")
        print("新进入重审：", ", ".join(impact["newly_needs_review"]) or "（无）")
        print("处置变化：", json.dumps(impact["decision_changes"], ensure_ascii=False))
        print()

    print(f"case={report['case_id']}  {report['disclaimer']}")
    print(f"记录 {report['stats']['record_count']} / 主张 {report['stats']['claim_count']} / "
          f"独立试验 {report['stats']['trial_count']} / 建议 {report['stats']['guidance_count']}")
    for rev in report["guidance_reviews"]:
        print(f"  - {rev['guidance_id']}: {rev['suggested_action']} ({rev['suggested_action_zh']})")
        for r in rev["reasons"]:
            print(f"      [{r['severity']}] {r['code']}")
    print(f"重审队列 {report['stats']['re_review_queue_size']} 条；"
          f"表述问题 {report['stats']['wording_issue_count']} 项")
    if not args.no_write:
        print(f"报告已写入 {outdir}/evidence_report.{{json,md}}")

    if args.check and report["stats"]["re_review_queue_size"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
