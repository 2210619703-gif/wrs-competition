"""仿真目录约定：脚本统一从这里取路径。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TB_DIR = ROOT.parent.parent
DIR_DATA = TB_DIR / "dataset_learn"
DIR_CAPTURE = TB_DIR / "dataset_gen"
DIR_WEIGHTS = ROOT / "权重"
DIR_CLASSES = ROOT / "类别配置"
DIR_DECISION = ROOT / "决策微调数据"
DIR_DOCS = ROOT / "文档"
