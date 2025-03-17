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

        self._confidence_history = deque(maxlen=5)
        self._target_found = False
        self._exploration_coverage = 0.0
        self._last_stagnation_recovery = 0

        self._position_history = deque(maxlen=10)

        self._step_counter = 0

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        global logger # make sure global logger

        frontiers = self._observations_cache["frontier_sensor"]
        try:
            exploration_phase = self._get_exploration_phase()
        except AttributeError:
            exploration_phase = "standard"
        except Exception as e:
            logger.error(f"exploration_phase stage failure: {str(e)}")
            exploration_phase = "standard"
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            logger.warning("Didnot find any frontier, stop exploring!")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)

        # detect stagnation and restoring
        if self._detect_exploration_stagnation():
            logger.warning(f"stagnation detecting: stage={exploration_phase}, step={self._step_count}")
            # restoring
            if hasattr(self, '_last_stagnation_recovery') and self._step_count - self._last_stagnation_recovery < 10:
                # stagnate in a short time, adopt a more severe strategy
                logger.info("stagnate in a short time, randomly explore")
                unexplored_frontiers = self._find_unexplored_frontiers(frontiers)
                if len(unexplored_frontiers) > 0:
                    random_idx = np.random.randint(0, len(unexplored_frontiers))
                    best_frontier = unexplored_frontiers[random_idx]
                    logger.info(f"randomly choose frontier point: {best_frontier}")

            self._last_stagnation_recovery = self._step_count


        if best_value > 0.85:
            # 1. make sure the exploration steps are enough
            min_exploration_steps = 15  # at least 15 steps
            if self._step_count < min_exploration_steps:
                logger.info(f"exploration steps are not enough({self._step_count}/{min_exploration_steps}), keep explore")
                return self._pointnav(best_frontier)

            # 2. target verification
            if self._verify_target_presence(observations):
                logger.info(f"target verified, credibility={best_value:.2f}, switch to objNav")
                self._target_found = True
                return self._pointnav(best_frontier, stop=True)
            else:
                logger.warning("High credibility but verified wrongly, keep explore")

        # different stage exploration
        if exploration_phase == "initial":
            # widely search, guangdu youxian
            logger.debug("widely search initially")

        elif exploration_phase == "focused":
            # focus on  high value area, shendu youxian
            logger.debug("focus on  high value area, shendu youxian")
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

        initial_value = best_value

        if initial_value > 0.7:
            # 1. verification of target
            target_detection_valid = False
            if 'object_detections' in observations:
                detections = observations['object_detections']
                if hasattr(detections, 'labels') and len(detections.labels) > 0:
                    for i, (label, score) in enumerate(zip(detections.labels, detections.scores)):
                        if self._is_target_object(label) and score > 0.4:
                            bbox = detections.boxes[i] if hasattr(detections, 'boxes') else None
                            # verification of size and location
                            if self._is_detection_reasonable(bbox, observations):
                                target_detection_valid = True
                                break

            # 2. make it more smoothly, avoiding shunshi wucha
            temporal_valid = False
            if not hasattr(self, '_confidence_history') or not self._confidence_history:
                recent_high_confidence = 0
            else:
                history_list = list(self._confidence_history)
                recent_items = history_list[-min(3, len(history_list)):]
                recent_high_confidence = sum(1 for v in recent_items if v > 0.7)

            temporal_valid = recent_high_confidence >= 2  # 2 slides high credibility

            self._confidence_history.append(initial_value)

            if target_detection_valid and temporal_valid:
                best_value = initial_value
                logger.info(f"Pass with high credibility {best_value:.2f}, obj detect is available")
            else:
                reduced_value = max(0.6, initial_value * 0.7)  # cut 30% but higher than 0.5
                logger.warning(f"Fail with high credibility: {initial_value:.2f} -> {reduced_value:.2f}")
                best_value = reduced_value
        else:
            best_value = initial_value

        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if self.use_mm_nav and self._memory is not None:
            memory_positions = self._memory.positions if hasattr(self._memory, 'positions') else []
            if memory_positions:
                policy_info['memory_positions'] = np.array(memory_positions)

            if self.enable_cycle_detection and hasattr(self._memory, 'detect_cycle'):
                cycle_detected, cycle_info = self._memory.detect_cycle()
                if cycle_detected:
                    policy_info['cycle_detected'] = True
                    if cycle_info and 'segment' in cycle_info:
                        policy_info['cycle_segment'] = cycle_info['segment']


        return policy_info

    def _update_habitat_visualizations(self, policy_info, observations):
        additional_info = []

        if hasattr(self, 'use_mm_nav') and self.use_mm_nav and hasattr(self, '_memory') and self._memory is not None:
            memory_size = len(getattr(self._memory, 'positions', []))
            additional_info.append(f"Memory: {memory_size} pts")

            cycle_count = 0
            if hasattr(self, '_coordinator') and hasattr(self._coordinator, 'cycle_detector'):
                cycle_count = self._coordinator.cycle_detector.cycle_count
                if cycle_count > 0:
                    additional_info.append(f"Cycles detected: {cycle_count}")


        if hasattr(self, '_metrics') and self._metrics:
            success_rate = self._metrics.success_rate
            spl = self._metrics.spl
            efficiency = self._metrics.trajectory_efficiency
            additional_info.append(f"SR: {success_rate:.2f} SPL: {spl:.2f} Eff: {efficiency:.2f}")

        policy_info['additional_text'] = additional_info

        if hasattr(self, '_memory') and self._memory is not None and hasattr(self._memory, 'positions'):
            memory_positions = self._memory.positions
            if memory_positions and len(memory_positions) > 0:
                if ('map_data' in observations and
                    'map_shape' in observations['map_data'] and
                    'world_to_map' in observations['map_data']):

                    map_shape = observations['map_data']['map_shape']
                    world_to_map = observations['map_data']['world_to_map']

                    memory_vis = np.zeros((map_shape[0], map_shape[1], 3), dtype=np.uint8)

                    map_positions = []
                    for pos in memory_positions:
                        if len(pos) >= 2:
                            map_x, map_y = int(pos[0] * world_to_map[0, 0] + world_to_map[0, 2]), int(pos[1] * world_to_map[1, 1] + world_to_map[1, 2])
                            if 0 <= map_x < map_shape[1] and 0 <= map_y < map_shape[0]:
                                map_positions.append((map_x, map_y))

                    if len(map_positions) > 1:
                        for i in range(1, len(map_positions)):
                            cv2.line(memory_vis, map_positions[i-1], map_positions[i], (0, 255, 255), 1)

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
        # Use value_map directly for sorting
        sorted_frontiers, values = self._value_map.sort_waypoints(frontiers, 0.5)


        if not self.use_mm_nav:
            return sorted_frontiers, values

        if self._coordinator is None:
            logger.warning("Enhancements are enabled but the coordinator is not initialized")
            return sorted_frontiers, values

        original_values = np.array(values.copy())

        try:
            values_np = np.array(values)
            result = self._coordinator.integrate_decisions(
                base_values=values_np,
                memory=self._memory if self.use_mm_nav else None,
                frontiers=sorted_frontiers,
                observation=observations
            )

            if isinstance(result, dict):
                enhanced_values = result.get('enhanced_values', values)
            elif isinstance(result, np.ndarray):
                enhanced_values = result
            else:
                enhanced_values = np.array(values)
                logger.warning(f"decision returned unexpected value: {type(result)}")

            # Calculate the difference between the original value and the enhanced value
            value_diff = np.abs(enhanced_values - original_values)
            avg_diff = np.mean(value_diff)
            max_diff = np.max(value_diff)


            if avg_diff < 0.01:
                logger.warning("The enhancement has little impact on the frontier value sorting, check the module configuration and weight settings")

            indices = np.argsort(-enhanced_values)
            sorted_frontiers = sorted_frontiers[indices]
            values = enhanced_values[indices].tolist()

        except Exception as e:
            logger.error(f"Decision integration failed: {str(e)}")

            logger.error(traceback.format_exc())
            return sorted_frontiers, values

        return sorted_frontiers, values

    def _is_target_object(self, label):
        """Check if the label matches the target object"""
        if not hasattr(self, '_target_object'):
            return False

        target = self._target_object.lower()
        label = label.lower()

        if target == label:
            return True

        if target in label or label in target:
            return True

        # Synonym matching
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

        self.enable_memory_repulsion = os.environ.get('VLFM_MEMORY_REPULSION', 'false').lower() == 'true'
        self.enable_cycle_detection = os.environ.get('VLFM_CYCLE_DETECTION', 'false').lower() == 'true'


        self._initialize_core_modules()


        self._initialize_decision_coordinator()




        self._high_confidence_detected = False
        self._confidence_threshold = 0.8
        self._consecutive_high_confidence = 0

    def _initialize_core_modules(self):
        # MM-Nav
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
                # logger.info("初始化MM-Nav空间记忆模块")
            except Exception as e:
                logger.error(f"MM-Nav initialization failed: {str(e)}")
                self._memory = None
                self.use_mm_nav = False
        else:
            self._memory = None



    def _initialize_decision_coordinator(self):
        """Initialize the decision coordinator"""
        if self.use_mm_nav:
            coordinator_config = type('Config', (), {
                'memory_weight': 1,
                # 'uncertainty_weight': 0.3,
                # 'semantic_weight': 0.3,
                'cycle_threshold': 8,
                'log_enabled': True,

                'use_mm_nav': self.use_mm_nav,
                # 'use_ua_explore': self.use_ua_explore,
                # 'use_llm_gsp': self.use_llm_gsp,
                'enable_memory_repulsion': self.enable_memory_repulsion,
                'enable_cycle_detection': self.enable_cycle_detection,
            })
            self._coordinator = DecisionCoordinator(coordinator_config)
            # logger.info("初始化决策协调器")
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

        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, values = self._value_map.sort_waypoints(frontiers, 0.5)

        # logger.debug(f"original frontiers: {values[:5]}...")  # 只显示前5个值

        if not self.use_mm_nav:
            # logger.debug("Enhancements are not enabled, original sorting is used")
            return sorted_frontiers, values

        if self._coordinator is None:
            # logger.warning("Enhancements are enabled but the coordinator is not initialized")
            return sorted_frontiers, values

        try:
            result = self._coordinator.integrate_decisions(
                base_values=values,
                memory=self._memory if self.use_mm_nav else None,
                frontiers=sorted_frontiers,
                observation=observations
            )

            if isinstance(result, dict):
                enhanced_values = result.get('enhanced_values', values)
                # logger.debug(f"Enhanced frontier value: {enhanced_values[:5]}...")
            elif isinstance(result, np.ndarray):
                enhanced_values = result
                # logger.debug(f"Enhanced frontier value: {enhanced_values[:5]}...")
            else:
                enhanced_values = np.array(values)
                logger.warning("Decision integration returned an unexpected type")

            # 重新排序
            indices = np.argsort(-enhanced_values)
            sorted_frontiers = sorted_frontiers[indices]
            values = enhanced_values[indices].tolist()

        except Exception as e:
            logger.error(f"Decision integration failed: {str(e)}")
            return sorted_frontiers, values

        return sorted_frontiers, values

    def _reset(self):
        super()._reset()
        self._best_frontier = None
        self._frontiers = None


        if hasattr(self, '_memory') and self._memory is not None:
            self._memory.reset()




        if hasattr(self, '_coordinator') and self._coordinator is not None:
            self._coordinator.reset()

        if hasattr(self, '_metrics') and self._metrics is not None:
            if hasattr(self, '_archive_metrics') and self._metrics.episode_count > 0:
                if not isinstance(self._archive_metrics, list):
                    self._archive_metrics = []
                self._archive_metrics.append(self._metrics.get_summary())
            self._metrics = PerformanceMetrics()

    def _extract_position(self, observations):
        if 'gps' in observations:
            return observations['gps'].cpu().numpy()[0]
        return np.zeros(2)

    def _update_value_map(self):
        super()._update_value_map()


    def _extract_position(self, observations):
        if 'gps' in observations:
            return observations['gps'].cpu().numpy()[0]
        return np.zeros(2)  # 默认位置

    def _extract_features(self, observations):
        """Extract key features of observations"""
        features = {}

        if 'gps' in observations:
            features['position'] = observations['gps'].cpu().numpy()[0]

        if 'compass' in observations:
            compass = observations['compass'].cpu().numpy()[0]
            features['heading'] = compass

        features['step_count'] = self._step_count

        if 'object_detections' in observations:
            detections = observations['object_detections']
            if hasattr(detections, 'labels'):
                features['objects'] = detections.labels

        return features

    def _check_state_transition(self, best_value):
        """Check if state transition is required"""
        if best_value > self._confidence_threshold:
            self._consecutive_high_confidence += 1
            # The conversion is triggered only after 3 consecutive high confidences to avoid noise
            if self._consecutive_high_confidence >= 3:
                return True
        else:
            self._consecutive_high_confidence = 0
        return False

    def _get_exploration_phase(self):
        """Determine the current exploration stage"""
        global logger

        # Make sure to have exploration coverage tracking
        if not hasattr(self, '_exploration_coverage'):
            self._exploration_coverage = 0.0
            # logger.debug("初始化探索覆盖率跟踪")

        # Use the correct step counter
        if self._step_counter < 10:
            return "initial"
        elif self._exploration_coverage > 0.4 or self._step_counter > 30:
            return "focused"
        else:
            return "standard"

    def _verify_target_presence(self, observations):
        try:

            # Check target detection
            if 'object_detections' in observations:
                detections = observations['object_detections']
                if hasattr(detections, 'labels') and len(detections.labels) > 0:
                    for i, (label, score) in enumerate(zip(detections.labels, detections.scores)):
                        if hasattr(self, '_is_target_object'):
                            is_target = self._is_target_object(label)
                        else:
                            is_target = self._target_object.lower() in label.lower()

                        if is_target and score > 0.65:  # ori: 0.8
                            return True

            # No target detected
            return False
        except Exception as e:
            logger.error(f"Target verification failed: {str(e)}")
            return False

    def _detect_exploration_stagnation(self, observations=None):
        global logger

        try:
            if not hasattr(self, '_position_history'):
                self._position_history = deque(maxlen=10)

            if observations is None:
                # If no observations are provided, the most recent location is used
                if len(self._position_history) > 0:
                    current_pos = self._position_history[-1]
                else:
                    return False
            else:
                current_pos = self._extract_position(observations)

            self._position_history.append(current_pos)

            if len(self._position_history) > 5:
                recent_positions = list(self._position_history)[-5:]
                distances = []
                for i in range(len(recent_positions)-1):
                    dist = np.linalg.norm(recent_positions[i+1] - recent_positions[i])
                    distances.append(dist)

                avg_movement = np.mean(distances)
                # 平均移动距离很小表示停滞
                if avg_movement < 0.3:
                    logger.debug(f"Exploration stagnation detected: average distance moved={avg_movement:.2f}")
                    return True

            return False
        except Exception as e:
            logger.error(f"Exploration Stagnation Detection Failure：{str(e)}")
            return False


