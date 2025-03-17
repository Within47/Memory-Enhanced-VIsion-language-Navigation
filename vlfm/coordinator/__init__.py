"""
VLFM决策协调器模块

提供MM-Nav和UA-Explore优化协调功能，增强两个创新点的协同效应
"""

from vlfm.coordinator.decision_coordinator import DecisionCoordinator
import logging

__all__ = ["DecisionCoordinator"]

# logger = logging.getLogger(__name__)
#
# class DecisionCoordinator:
#     def __init__(self, config=None):
#         self.config = config or type('Config', (), {})
#
#         # 使用与ITMPolicyV2相同的属性名
#         self.use_mm_nav = getattr(self.config, 'use_mm_nav', False)
#         self.use_ua_explore = getattr(self.config, 'use_ua_explore', False)
#         # ...
#
#         logger.info("VLFM增强功能配置:")
#         logger.info(f"- MM-Nav: {self.use_mm_nav}")
#         # ...
#
#         self._instance_id = id(self)
#         logger.info(f"决策协调器实例ID: {self._instance_id}")