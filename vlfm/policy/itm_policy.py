# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import logging
import numpy as np
from collections import deque
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import cv2
from torch import Tensor
import traceback
from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.utils.metrics_collector import PerformanceMetrics
import time

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.memory import SpatialMemory
from vlfm.uncertainty import UncertaintyEstimator, AdaptiveExplorer
from vlfm.coordinator import DecisionCoordinator


# 确保全局logger定义存在
logger = logging.getLogger(__name__)

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

        # 添加目标验证所需属性 (新增)
        self._confidence_history = deque(maxlen=5)
        self._target_found = False
        self._exploration_coverage = 0.0
        self._last_stagnation_recovery = 0

        # 记录最近位置用于停滞检测
        self._position_history = deque(maxlen=10)

        # 确保步数计数器初始化
        self._step_counter = 0  # 这是原有的变量，不是_step_count

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        """增强的探索逻辑，多阶段策略"""
        # 确保全局logger可用
        global logger  # 可以添加这行确保使用全局logger

        frontiers = self._observations_cache["frontier_sensor"]

        # 安全调用探索阶段判断
        try:
            exploration_phase = self._get_exploration_phase()
        except AttributeError:
            # 如果方法不存在或出错，使用默认值
            exploration_phase = "standard"
        except Exception as e:
            logger.error(f"获取探索阶段失败: {str(e)}")
            exploration_phase = "standard"

        # 检查是否有前沿点
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            logger.warning("没有找到前沿点，探索停止")
            return self._stop_action

        # 获取最佳前沿点
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)

        # 探索停滞检测与恢复 - 新增
        if self._detect_exploration_stagnation():
            logger.warning(f"探索停滞检测: 当前阶段={exploration_phase}, 步数={self._step_count}")
            # 停滞恢复策略
            if hasattr(self, '_last_stagnation_recovery') and self._step_count - self._last_stagnation_recovery < 10:
                # 短时间内再次停滞，采用更激进策略
                logger.info("连续停滞，启用随机探索")
                # 随机选择未探索区域
                unexplored_frontiers = self._find_unexplored_frontiers(frontiers)
                if len(unexplored_frontiers) > 0:
                    random_idx = np.random.randint(0, len(unexplored_frontiers))
                    best_frontier = unexplored_frontiers[random_idx]
                    logger.info(f"随机选择前沿点: {best_frontier}")

            self._last_stagnation_recovery = self._step_count

        # 增强的探索终止条件 - 修改
        # 原逻辑: 仅基于单一置信度阈值
        # 新逻辑: 多因素判断，包括目标验证、时间一致性和探索充分性
        if best_value > 0.85:  # 高置信度
            # 1. 确保最低探索步数
            min_exploration_steps = 15  # 至少进行15步探索
            if self._step_count < min_exploration_steps:
                logger.info(f"探索步数不足({self._step_count}/{min_exploration_steps})，继续探索")
                return self._pointnav(best_frontier)

            # 2. 目标验证
            if self._verify_target_presence(observations):
                logger.info(f"目标验证成功，置信度={best_value:.2f}，切换到目标导航")
                # 标记找到目标
                self._target_found = True
                # 可以切换到目标导航或停止
                return self._pointnav(best_frontier, stop=True)
            else:
                logger.warning("高置信度但目标验证失败，继续探索")

        # 针对不同探索阶段的策略调整 - 新增
        if exploration_phase == "initial":
            # 初始探索阶段：广泛覆盖
            logger.debug("初始探索阶段：优先覆盖范围")
            # 可以在此调整探索参数
        elif exploration_phase == "focused":
            # 聚焦探索阶段：关注高价值区域
            logger.debug("聚焦探索阶段：关注高价值区域")
            # 可以在此调整探索参数

        # 标准探索行为
        return self._pointnav(best_frontier)

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = min(1.0, sorted_values[best_frontier_idx])

        # 修改点: 替换简单的置信度阈值检查，增加多维度验证
        initial_value = best_value

        # 多维度置信度验证机制 - 新增
        if initial_value > 0.7:  # 高置信度阈值
            # 1. 目标检测验证
            target_detection_valid = False
            if 'object_detections' in observations:
                detections = observations['object_detections']
                if hasattr(detections, 'labels') and len(detections.labels) > 0:
                    for i, (label, score) in enumerate(zip(detections.labels, detections.scores)):
                        # 匹配目标与当前任务目标
                        if self._is_target_object(label) and score > 0.4:
                            bbox = detections.boxes[i] if hasattr(detections, 'boxes') else None
                            # 验证检测是否合理（大小、位置）
                            if self._is_detection_reasonable(bbox, observations):
                                target_detection_valid = True
                                break

            # 2. 时间一致性验证 - 防止瞬时误判
            temporal_valid = False
            if not hasattr(self, '_confidence_history') or not self._confidence_history:
                recent_high_confidence = 0
            else:
                history_list = list(self._confidence_history)
                recent_items = history_list[-min(3, len(history_list)):]
                recent_high_confidence = sum(1 for v in recent_items if v > 0.7)

            temporal_valid = recent_high_confidence >= 2  # 至少2/3帧高置信度

            # 记录当前置信度
            self._confidence_history.append(initial_value)

            # 3. 综合判断
            if target_detection_valid and temporal_valid:
                # 验证通过，保持高置信度
                best_value = initial_value
                logger.info(f"高置信度验证通过: {best_value:.2f}, 目标检测有效")
            else:
                # 验证失败，降低置信度
                reduced_value = max(0.6, initial_value * 0.7)  # 降低30%但不低于0.6
                logger.warning(f"高置信度验证失败: {initial_value:.2f} -> {reduced_value:.2f}")
                best_value = reduced_value
        else:
            best_value = initial_value

        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """增强版策略信息接口，用于可视化"""
        # 先获取基础策略信息
        policy_info = super()._get_policy_info(detections)

        # 添加MM-Nav相关信息（如有）
        if self.use_mm_nav and self._memory is not None:
            # 添加记忆点位置用于可视化
            memory_positions = self._memory.positions if hasattr(self._memory, 'positions') else []
            if memory_positions:
                policy_info['memory_positions'] = np.array(memory_positions)

            # 添加循环检测信息
            if self.enable_cycle_detection and hasattr(self._memory, 'detect_cycle'):
                cycle_detected, cycle_info = self._memory.detect_cycle()
                if cycle_detected:
                    policy_info['cycle_detected'] = True
                    if cycle_info and 'segment' in cycle_info:
                        policy_info['cycle_segment'] = cycle_info['segment']

        # 添加UA-Explore相关信息（如有）
        if self.use_ua_explore and self._uncertainty_estimator is not None:
            # 添加不确定性分布信息
            if self._best_frontier is not None and self._frontiers is not None:
                # 从上次排序中获取不确定性值
                if hasattr(self, '_last_uncertainty_info'):
                    policy_info['uncertainty_map'] = self._last_uncertainty_info.get('frontier_uncertainties', {})
                    policy_info['uncertainty_threshold'] = self._last_uncertainty_info.get('threshold', 0.5)

            # 添加探索策略信息
            if self.enable_adaptive_explorer and self._adaptive_explorer is not None:
                if hasattr(self._adaptive_explorer, 'strategy_history') and self._adaptive_explorer.strategy_history:
                    latest_strategy = self._adaptive_explorer.strategy_history[-1]
                    policy_info['exploration_strategy'] = latest_strategy.get('primary_strategy', 'balanced')
                    policy_info['exploration_boost'] = latest_strategy.get('boost_exploration', False)


            if hasattr(self, '_last_planner_state'):
                planner_state = self._last_planner_state
                current_subgoal = planner_state.get('current_subgoal')
                navigation_hint = planner_state.get('navigation_hint')

                if current_subgoal:
                    policy_info['current_subgoal'] = current_subgoal.get('description', '')
                    policy_info['subgoal_criteria'] = current_subgoal.get('criteria', '')

                if navigation_hint:
                    policy_info['navigation_hint'] = navigation_hint

        return policy_info

    def _update_habitat_visualizations(self, policy_info, observations):
        """更新Habitat可视化信息，确保与habitat_visualizer.py无缝集成"""
        # 创建附加可视化信息
        additional_info = []

        # 添加MM-Nav相关信息
        if hasattr(self, 'use_mm_nav') and self.use_mm_nav and hasattr(self, '_memory') and self._memory is not None:
            memory_size = len(getattr(self._memory, 'positions', []))
            additional_info.append(f"Memory: {memory_size} pts")

            # 添加循环检测信息
            cycle_count = 0
            if hasattr(self, '_coordinator') and hasattr(self._coordinator, 'cycle_detector'):
                cycle_count = self._coordinator.cycle_detector.cycle_count
                if cycle_count > 0:
                    additional_info.append(f"Cycles detected: {cycle_count}")

        # 添加UA-Explore信息
        if hasattr(self, 'use_ua_explore') and self.use_ua_explore and hasattr(self, '_uncertainty_estimator'):
            if hasattr(self._uncertainty_estimator, 'threshold'):
                threshold = self._uncertainty_estimator.threshold
                additional_info.append(f"Uncertainty threshold: {threshold:.2f}")

            # 添加探索策略信息
            if hasattr(self, '_adaptive_explorer') and hasattr(self._adaptive_explorer, 'strategy_history'):
                if self._adaptive_explorer.strategy_history:
                    strategy = self._adaptive_explorer.strategy_history[-1].get('primary_strategy', '')
                    if strategy:
                        additional_info.append(f"Strategy: {strategy}")



        # 添加性能信息
        if hasattr(self, '_metrics') and self._metrics:
            success_rate = self._metrics.success_rate
            spl = self._metrics.spl
            efficiency = self._metrics.trajectory_efficiency
            additional_info.append(f"SR: {success_rate:.2f} SPL: {spl:.2f} Eff: {efficiency:.2f}")

        # 更新policy_info
        policy_info['additional_text'] = additional_info

        # 创建记忆可视化图层
        if hasattr(self, '_memory') and self._memory is not None and hasattr(self._memory, 'positions'):
            memory_positions = self._memory.positions
            if memory_positions and len(memory_positions) > 0:
                # 检查是否可以从观察中获取地图尺寸信息
                if ('map_data' in observations and
                    'map_shape' in observations['map_data'] and
                    'world_to_map' in observations['map_data']):

                    # 获取地图信息
                    map_shape = observations['map_data']['map_shape']
                    world_to_map = observations['map_data']['world_to_map']

                    # 创建记忆轨迹可视化
                    memory_vis = np.zeros((map_shape[0], map_shape[1], 3), dtype=np.uint8)

                    # 转换记忆位置到地图坐标
                    map_positions = []
                    for pos in memory_positions:
                        # 确保位置格式正确
                        if len(pos) >= 2:
                            # 应用坐标转换
                            map_x, map_y = int(pos[0] * world_to_map[0, 0] + world_to_map[0, 2]), int(pos[1] * world_to_map[1, 1] + world_to_map[1, 2])
                            if 0 <= map_x < map_shape[1] and 0 <= map_y < map_shape[0]:
                                map_positions.append((map_x, map_y))

                    # 绘制记忆轨迹
                    if len(map_positions) > 1:
                        for i in range(1, len(map_positions)):
                            cv2.line(memory_vis, map_positions[i-1], map_positions[i], (0, 255, 255), 1)

                    # 将记忆可视化添加到策略信息中
                    policy_info['memory_visualization'] = memory_vis

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(self, observations: "TensorDict", frontiers: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """基于多模块集成对前沿进行排序"""
        # 直接使用value_map进行排序
        sorted_frontiers, values = self._value_map.sort_waypoints(frontiers, 0.5)

        # 添加调试信息 - 新增
        logger.info(f"原始前沿值: 最小={min(values):.2f}, 最大={max(values):.2f}, 均值={np.mean(values):.2f}")



        if self._coordinator is None:
            logger.warning("增强功能已启用但coordinator未初始化")
            return sorted_frontiers, values

        # 记录原始值用于比较 - 新增
        original_values = np.array(values.copy())

        try:
            # 确保values是numpy数组
            values_np = np.array(values)

            # 调用决策协调器
            result = self._coordinator.integrate_decisions(
                base_values=values_np,
                memory=self._memory if self.use_mm_nav else None,
                uncertainty_estimator=self._uncertainty_estimator if self.use_ua_explore else None,
                adaptive_explorer=self._adaptive_explorer if self.enable_adaptive_explorer else None,
                frontiers=sorted_frontiers,
                observation=observations
            )

            # 处理各种可能的返回类型
            if isinstance(result, dict):
                enhanced_values = result.get('enhanced_values', values)
            elif isinstance(result, np.ndarray):
                enhanced_values = result
            else:
                enhanced_values = np.array(values)
                logger.warning(f"决策整合返回了意外的类型: {type(result)}")

            # 计算原始值与增强值的差异 - 新增
            value_diff = np.abs(enhanced_values - original_values)
            avg_diff = np.mean(value_diff)
            max_diff = np.max(value_diff)

            # 输出调试信息 - 新增
            logger.info(f"增强功能影响: 平均差异={avg_diff:.4f}, 最大差异={max_diff:.4f}")
            logger.info(f"增强后前沿值: 最小={np.min(enhanced_values):.2f}, 最大={np.max(enhanced_values):.2f}, 均值={np.mean(enhanced_values):.2f}")

            # 如果差异太小，发出警告 - 新增
            if avg_diff < 0.01:
                logger.warning("增强功能几乎没有影响前沿值排序，检查模块配置和权重设置")

            # 重新排序
            indices = np.argsort(-enhanced_values)
            sorted_frontiers = sorted_frontiers[indices]
            values = enhanced_values[indices].tolist()

        except Exception as e:
            logger.error(f"决策整合失败: {str(e)}")

            logger.error(traceback.format_exc())
            return sorted_frontiers, values

        return sorted_frontiers, values

    def _is_target_object(self, label):
        """检查标签是否与目标对象匹配"""
        if not hasattr(self, '_target_object'):
            return False

        # 简单的字符串匹配
        target = self._target_object.lower()
        label = label.lower()

        # 直接匹配
        if target == label:
            return True

        # 部分匹配
        if target in label or label in target:
            return True

        # 同义词匹配 (可以扩展)
        synonyms = {
            'couch': ['sofa', 'loveseat'],
            'tv': ['television', 'monitor', 'screen'],
            'refrigerator': ['fridge'],
            # 可以添加更多同义词
        }

        if target in synonyms:
            for synonym in synonyms[target]:
                if synonym in label:
                    return True

        return False


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 从环境变量读取配置
        self.use_mm_nav = os.environ.get('VLFM_USE_MEMORY', 'false').lower() == 'true'
        self.use_ua_explore = os.environ.get('VLFM_USE_UNCERTAINTY', 'false').lower() == 'true'
        self.enable_memory_repulsion = os.environ.get('VLFM_MEMORY_REPULSION', 'false').lower() == 'true'
        self.enable_cycle_detection = os.environ.get('VLFM_CYCLE_DETECTION', 'false').lower() == 'true'
        self.enable_adaptive_explorer = os.environ.get('VLFM_ADAPTIVE_EXPLORER', 'false').lower() == 'true'

        # 初始化核心模块
        self._initialize_core_modules()

        # 初始化决策协调器
        self._initialize_decision_coordinator()

        # 打印配置信息
        logger.info(f"ITMPolicyV2初始化 - 实际配置: MM-Nav={self.use_mm_nav}, UA-Explore={self.use_ua_explore}")

        # 添加状态管理
        self._high_confidence_detected = False
        self._confidence_threshold = 0.8  # 可配置的置信度阈值
        self._consecutive_high_confidence = 0  # 连续高置信度计数

    def _initialize_core_modules(self):
        """初始化VLFM核心模块，包含错误处理"""
        # 1. 空间记忆模块 (MM-Nav)
        if self.use_mm_nav:
            try:
                memory_config = type('MemoryConfig', (), {
                    'max_memory_size': 1000,
                    'min_distance': 0.2,
                    'repulsion_radius': 2.0 if self.enable_memory_repulsion else 0.0,
                    'repulsion_strength': 1.0,
                    'decay_factor': 0.96,
                    'cycle_detection_window': 10,
                    'cycle_similarity_threshold': 0.75
                })
                self._memory = SpatialMemory(memory_config)
                logger.info("初始化MM-Nav空间记忆模块")
            except Exception as e:
                logger.error(f"MM-Nav初始化失败: {str(e)}")
                self._memory = None
                self.use_mm_nav = False  # 禁用功能
        else:
            self._memory = None

        # 2. 不确定性估计模块 (UA-Nav)
        if self.use_ua_explore:
            uncertainty_config = type('UncertaintyConfig', (), {
                'base_threshold': 0.5,
                'threshold_decay': 0.98,
                'min_threshold': 0.1,
                'max_threshold': 0.8,
                'monte_carlo_samples': 10,
                'max_steps': int(os.environ.get('VLFM_MAX_STEPS', '250'))
            })
            self._uncertainty_estimator = UncertaintyEstimator(uncertainty_config)
            logger.info("初始化UA-Nav不确定性估计模块")
        else:
            self._uncertainty_estimator = None

        # 3. 自适应探索器
        if self.enable_adaptive_explorer:
            explorer_config = type('ExplorerConfig', (), {
                'max_steps': int(os.environ.get('VLFM_MAX_STEPS', '250')),
                'stagnation_threshold': 30,
                'efficiency_factor': 0.5
            })
            self._adaptive_explorer = AdaptiveExplorer(explorer_config)
            logger.info("初始化自适应探索器")
        else:
            self._adaptive_explorer = None



    def _initialize_decision_coordinator(self):
        """初始化决策协调器"""
        if self.use_mm_nav or self.use_ua_explore:
            coordinator_config = type('Config', (), {
                'memory_weight': 0.2,
                'uncertainty_weight': 0.8,
                'semantic_weight': 0,
                'cycle_threshold': 8,
                'log_enabled': True,
                # 传递配置值
                'use_mm_nav': self.use_mm_nav,
                'use_ua_explore': self.use_ua_explore,
                'enable_memory_repulsion': self.enable_memory_repulsion,
                'enable_cycle_detection': self.enable_cycle_detection,
                'enable_adaptive_explorer': self.enable_adaptive_explorer,
            })
            self._coordinator = DecisionCoordinator(coordinator_config)
            logger.info("初始化决策协调器")
        else:
            self._coordinator = None

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        # 状态检查 - 新增
        if self._step_counter % 10 == 0:  # 每10步输出一次状态
            logger.info(f"功能状态检查 [步骤 {self._step_counter}]:")
            logger.info(f"- MM-Nav: {self.use_mm_nav} (记忆模块: {'已初始化' if self._memory is not None else '未初始化'})")
            logger.info(f"- UA-Nav: {self.use_ua_explore} (不确定性估计器: {'已初始化' if self._uncertainty_estimator is not None else '未初始化'})")
            logger.info(f"- 决策协调器: {'已初始化' if self._coordinator is not None else '未初始化'}")

            # 如果启用UA-Nav，提供额外信息
            if self.use_ua_explore and self._uncertainty_estimator is not None:
                logger.info(f"  - 不确定性历史: {len(self._uncertainty_estimator.uncertainty_history)} 条记录")
                logger.info(f"  - 访问计数记录: {len(self._uncertainty_estimator.visit_counts)} 个位置")

        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        # 直接使用value_map进行排序
        sorted_frontiers, values = self._value_map.sort_waypoints(frontiers, 0.5)

        logger.debug(f"原始前沿值: {values[:5]}...")  # 只显示前5个值



        if self._coordinator is None:
            logger.warning("增强功能已启用但coordinator未初始化")
            return sorted_frontiers, values

        try:
            result = self._coordinator.integrate_decisions(
                base_values=values,
                memory=self._memory if self.use_mm_nav else None,
                uncertainty_estimator=self._uncertainty_estimator if self.use_ua_explore else None,
                adaptive_explorer=self._adaptive_explorer if self.enable_adaptive_explorer else None,
                frontiers=sorted_frontiers,
                observation=observations
            )

            if isinstance(result, dict):
                enhanced_values = result.get('enhanced_values', values)
                logger.debug(f"增强后的前沿值: {enhanced_values[:5]}...")
            elif isinstance(result, np.ndarray):
                enhanced_values = result
                logger.debug(f"增强后的前沿值: {enhanced_values[:5]}...")
            else:
                enhanced_values = np.array(values)
                logger.warning("决策整合返回了意外的类型")

            # 重新排序
            indices = np.argsort(-enhanced_values)
            sorted_frontiers = sorted_frontiers[indices]
            values = enhanced_values[indices].tolist()

        except Exception as e:
            logger.error(f"决策整合失败: {str(e)}")
            return sorted_frontiers, values

        return sorted_frontiers, values

    def _reset(self):
        """重置策略状态"""
        super()._reset()

        # 重置frontier相关属性
        self._best_frontier = None
        self._frontiers = None

        # 重置增强模块
        if hasattr(self, '_memory') and self._memory is not None:
            self._memory.reset()
            logger.info("重置记忆模块")

        if hasattr(self, '_uncertainty_estimator') and self._uncertainty_estimator is not None:
            self._uncertainty_estimator.reset()
            logger.info("重置不确定性估计器")

        if hasattr(self, '_adaptive_explorer') and self._adaptive_explorer is not None:
            self._adaptive_explorer.reset()
            logger.info("重置自适应探索器")

        # 重置语义规划器
        if hasattr(self, '_semantic_planner') and self._semantic_planner is not None:
            self._semantic_planner.reset()
            logger.info("重置语义规划器")

        # 重置决策协调器
        if hasattr(self, '_coordinator') and self._coordinator is not None:
            self._coordinator.reset()

        # 重置性能指标
        if hasattr(self, '_metrics') and self._metrics is not None:
            # 当策略实例重用时保存旧指标
            if hasattr(self, '_archive_metrics') and self._metrics.episode_count > 0:
                if not isinstance(self._archive_metrics, list):
                    self._archive_metrics = []
                self._archive_metrics.append(self._metrics.get_summary())

            # 初始化新的指标收集器
            self._metrics = PerformanceMetrics()

    def _extract_position(self, observations):
        """从观察中提取当前位置"""
        if 'gps' in observations:
            return observations['gps'].cpu().numpy()[0]
        return np.zeros(2)  # 默认位置



    def _extract_position(self, observations):
        """从观察中提取当前位置"""
        if 'gps' in observations:
            return observations['gps'].cpu().numpy()[0]
        return np.zeros(2)  # 默认位置

    def _extract_features(self, observations):
        """提取观察的关键特征"""
        features = {}

        # 提取位置
        if 'gps' in observations:
            features['position'] = observations['gps'].cpu().numpy()[0]

        # 提取朝向
        if 'compass' in observations:
            compass = observations['compass'].cpu().numpy()[0]
            features['heading'] = compass

        # 提取当前步数
        features['step_count'] = self._step_count

        # 提取物体检测结果
        if 'object_detections' in observations:
            detections = observations['object_detections']
            if hasattr(detections, 'labels'):
                features['objects'] = detections.labels

        return features

    def _check_state_transition(self, best_value):
        """检查是否需要状态转换"""
        if best_value > self._confidence_threshold:
            self._consecutive_high_confidence += 1
            # 连续3次高置信度才触发转换，避免噪声
            if self._consecutive_high_confidence >= 3:
                return True
        else:
            self._consecutive_high_confidence = 0
        return False

    def _get_exploration_phase(self):
        """确定当前探索阶段"""
        # 确保全局logger可用
        global logger

        # 首先确保有探索覆盖率跟踪
        if not hasattr(self, '_exploration_coverage'):
            # 初始化探索覆盖率跟踪
            self._exploration_coverage = 0.0
            logger.debug("初始化探索覆盖率跟踪")

        # 使用正确的步数计数器
        if self._step_counter < 10:
            return "initial"  # 初始探索
        elif self._exploration_coverage > 0.4 or self._step_counter > 30:
            return "focused"  # 聚焦探索
        else:
            return "standard"  # 标准探索

    def _verify_target_presence(self, observations):
        """多因素目标验证"""
        try:
            # 检查目标检测
            if 'object_detections' in observations:
                detections = observations['object_detections']
                if hasattr(detections, 'labels') and len(detections.labels) > 0:
                    for i, (label, score) in enumerate(zip(detections.labels, detections.scores)):
                        # 确保_is_target_object方法存在，否则使用简单匹配
                        if hasattr(self, '_is_target_object'):
                            is_target = self._is_target_object(label)
                        else:
                            # 后备方案：简单字符串匹配
                            is_target = self._target_object.lower() in label.lower()

                        if is_target and score > 0.65:  # 原来可能是0.8或更高
                            return True

            # 未检测到目标
            return False
        except Exception as e:
            logger.error(f"目标验证失败: {str(e)}")
            return False  # 出错时保守返回

    def _detect_exploration_stagnation(self, observations=None):
        """探索停滞检测"""
        # 确保全局logger可用
        global logger

        try:
            # 确保位置历史存在
            if not hasattr(self, '_position_history'):
                self._position_history = deque(maxlen=10)

            # 安全获取当前位置
            if observations is None:
                # 如果没有提供observations，使用最近位置
                if len(self._position_history) > 0:
                    current_pos = self._position_history[-1]
                else:
                    return False  # 无法判断停滞
            else:
                current_pos = self._extract_position(observations)

            self._position_history.append(current_pos)

            # 检测位置变化不大
            if len(self._position_history) > 5:
                recent_positions = list(self._position_history)[-5:]
                distances = []
                for i in range(len(recent_positions)-1):
                    dist = np.linalg.norm(recent_positions[i+1] - recent_positions[i])
                    distances.append(dist)

                avg_movement = np.mean(distances)
                # 平均移动距离很小表示停滞
                if avg_movement < 0.3:
                    logger.debug(f"检测到探索停滞：平均移动距离={avg_movement:.2f}")
                    return True

            return False
        except Exception as e:
            logger.error(f"探索停滞检测失败：{str(e)}")
            return False


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]

    def _sort_frontiers_by_value(self, observations: "TensorDict", frontiers: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        增强型前沿排序 - 与性能监控集成

        Args:
            observations: 观察数据
            frontiers: 前沿点

        Returns:
            sorted_frontiers: 排序后的前沿点
            sorted_values: 对应的价值
        """
        # 测量性能开始时间
        start_time = time.time()

        # 获取基础值
        base_values = self._base_get_frontier_values(observations, frontiers)

        # 应用增强模块
        enhanced_values = base_values

        # 应用决策集成
        if hasattr(self, "_coordinator") and self._coordinator is not None:
            # 传递观察、前沿和模块给协调器
            enhanced_values = self._coordinator.integrate_decisions(
                base_values=base_values,
                memory=self._memory if hasattr(self, "_memory") else None,
                uncertainty_estimator=self._uncertainty_estimator if hasattr(self, "_uncertainty_estimator") else None,
                semantic_planner=self._semantic_planner if hasattr(self, "_semantic_planner") else None,
                adaptive_explorer=self._adaptive_explorer if hasattr(self, "_adaptive_explorer") else None,
                frontiers=frontiers,
                observation=observations
            )

        # 更新不确定性估计器
        if hasattr(self, "_uncertainty_estimator") and self._uncertainty_estimator is not None:
            self._uncertainty_estimator.update_from_frontier_values(frontiers, enhanced_values)

        # 记录性能指标
        computation_time = time.time() - start_time
        if hasattr(self, "_metrics") and self._metrics is not None:
            self._metrics.update_ua_explore_metrics(decision_time=computation_time)

        # 排序前沿
        indices = np.argsort(enhanced_values)[::-1]

        # 返回排序结果
        return frontiers[indices], [enhanced_values[i] for i in indices]

    def act(self, observations: Dict, rnn_hidden_states: Any, prev_actions: Any, masks: Tensor, deterministic: bool = False) -> Any:
        """
        增强型行为函数 - 与指标监控集成

        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步动作
            masks: 掩码
            deterministic: 是否确定性

        Returns:
            action: 选择的动作
        """
        # 测量性能开始时间
        start_time = time.time()

        # 提取当前位置
        position = self._extract_position(observations)

        # 更新记忆
        if hasattr(self, "_memory") and self._memory is not None:
            self._memory.add_position(position)

        # 执行标准行为选择
        action, value = super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

        # 更新决策协调器
        if hasattr(self, "_coordinator") and self._coordinator is not None:
            self._coordinator.update_action(action.item())

        # 记录性能指标
        total_time = time.time() - start_time
        if hasattr(self, "_metrics") and self._metrics is not None:
            # 记录动作选择时间
            self._metrics.update_ua_explore_metrics(decision_time=total_time)

        return action, value

    def _reset(self) -> None:
        """
        重置函数 - 确保所有增强模块都正确重置
        """
        # 调用基类重置
        super()._reset()

        # 重置frontier相关属性
        self._best_frontier = None
        self._frontiers = None

        # 重置记忆模块
        if hasattr(self, '_memory') and self._memory is not None:
            self._memory.reset()

        # 重置不确定性估计器
        if hasattr(self, '_uncertainty_estimator') and self._uncertainty_estimator is not None:
            self._uncertainty_estimator.reset()

        # 重置自适应探索器
        if hasattr(self, '_adaptive_explorer') and self._adaptive_explorer is not None:
            self._adaptive_explorer.reset()

        # 重置语义规划器
        if hasattr(self, '_semantic_planner') and self._semantic_planner is not None:
            self._semantic_planner.reset()

        # 重置决策协调器
        if hasattr(self, '_coordinator') and self._coordinator is not None:
            self._coordinator.reset()

        # 重置性能指标
        if hasattr(self, '_metrics') and self._metrics is not None:
            # 当策略实例重用时保存旧指标
            if hasattr(self, '_archive_metrics') and self._metrics.episode_count > 0:
                if not isinstance(self._archive_metrics, list):
                    self._archive_metrics = []
                self._archive_metrics.append(self._metrics.get_summary())

            # 初始化新的指标收集器
            self._metrics = PerformanceMetrics()
