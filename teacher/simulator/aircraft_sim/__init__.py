"""核心仿真器接口。

本交付副本只包含场景环境及其运行所需的数据结构，不导入训练器、策略网络、
实验脚本或结果文件。
"""

from .detect_config import DetectSimConfig
from .detect_env import DetectEnv
from .detect_models import DetectAction, DetectRole
from .models import StepResult

__all__ = ["DetectAction", "DetectEnv", "DetectRole", "DetectSimConfig", "StepResult"]
