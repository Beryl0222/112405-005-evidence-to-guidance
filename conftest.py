"""让 pytest 从仓库根目录导入 evidence_workbench（unittest 下无副作用）。"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
