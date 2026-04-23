from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zhishu_dualarm_lab.demos.run_dualarm_with_openpi import main


if __name__ == "__main__":
    main()
