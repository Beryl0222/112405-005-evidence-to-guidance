"""两期资料对比命令行。

用法：
    python3 -m evidence_workbench.compare examples/quarter_q1.json examples/quarter_q2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .change import diff_reports
from .pipeline import process_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比两期证据资料，输出需要新重审的建议")
    parser.add_argument("old_case", help="上一季度 case JSON")
    parser.add_argument("new_case", help="本季度 case JSON")
    args = parser.parse_args(argv)

    old = process_case(json.loads(Path(args.old_case).read_text(encoding="utf-8-sig")))
    new = process_case(json.loads(Path(args.new_case).read_text(encoding="utf-8-sig")))
    print(json.dumps(diff_reports(old, new), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
