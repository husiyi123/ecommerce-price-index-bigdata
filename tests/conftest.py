"""
pytest 全局配置 — conftest.py
----------------------------
- 将项目根目录和 src 子包加入 sys.path
- 注册 integration 标记
- 为覆盖率报告预加载 src 模块
"""
import os
import sys

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
