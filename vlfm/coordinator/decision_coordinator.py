"""
决策协调器和循环检测器模块

优化MM-Nav和UA-Explore两大创新点的协同效果，提高SR和SPL指标
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import numpy as np
import torch
import logging
import time
from collections import deque


logger = logging.getLogger("vlfm")

class CycleDetector:
    """
    Cycle detection utility for visual navigation
    Identifies repetitive patterns in navigation trajectories
    """
    
    def __init__(self, threshold: int = 8, intervention_levels: int = 3):
        """
        Initialize cycle detector
        
        Args:
            threshold: Number of steps to check for cycles
            intervention_levels: Number of intervention levels
        """
        self.threshold = threshold
        self.intervention_levels = intervention_levels
        self.forbidden_actions = set()
        self.cycle_count = 0
        self.intervention_level = 0
        self.position_similarity_threshold = 0.75
        self.action_repeat_threshold = 5
        
    def detect_position_cycle(self, position_history: List[np.ndarray]) -> bool:
        """
        Detect cycles in position history
        
        Args:
            position_history: List of agent positions
            
        Returns:
            bool: Whether a cycle is detected
        """
        if len(position_history) < self.threshold * 2:
            return False
            
        # Compare recent positions with earlier positions
        recent_positions = position_history[-self.threshold:]
        earlier_positions = position_history[-2*self.threshold:-self.threshold]
        
        # Calculate similarity between recent and earlier position sequences
        similarity_score = self._calculate_trajectory_similarity(
            recent_positions, earlier_positions)
            
        return similarity_score > self.position_similarity_threshold
        
    def _calculate_trajectory_similarity(self, 
                                       trajectory1: List[np.ndarray], 
                                       trajectory2: List[np.ndarray]) -> float:
        """
        Calculate similarity between two trajectories
        
        Args:
            trajectory1: First trajectory (list of positions)
            trajectory2: Second trajectory (list of positions)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if len(trajectory1) != len(trajectory2):
            return 0.0
            
        total_distance = 0.0
        max_possible_distance = 0.0
        
        for pos1, pos2 in zip(trajectory1, trajectory2):
            distance = np.linalg.norm(pos1 - pos2)
            # Convert distance to similarity (smaller distance = higher similarity)
            max_possible_distance += 5.0  # Assume max possible distance is 5m
            total_distance += min(distance, 5.0)
            
        if max_possible_distance == 0:
            return 0.0
            
        # Convert to similarity score (1 - normalized distance)
        similarity = 1.0 - (total_distance / max_possible_distance)
        return similarity
        
    def detect_action_cycle(self, action_history: List[int]) -> bool:
        """
        Detect repetitive action patterns
        
        Args:
            action_history: List of recent actions
            
        Returns:
            bool: Whether an action cycle is detected
        """
        if len(action_history) < self.action_repeat_threshold:
            return False
            
        # Check for repeating pattern of actions
        recent_actions = action_history[-self.action_repeat_threshold:]
        
        # Check for simple oscillation (e.g., left-right-left-right)
        if len(recent_actions) >= 4:
            # Look for alternating patterns
            pattern_detected = True
            for i in range(len(recent_actions) - 2):
                if recent_actions[i] != recent_actions[i + 2]:
                    pattern_detected = False
                    break
                    
            if pattern_detected:
                return True
                
        # Check for repeating action
        action_counts = {}
        for action in recent_actions:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
            
        # If any action is repeated more than 70% of the time
        for action, count in action_counts.items():
            if count > self.action_repeat_threshold * 0.7:
                return True
                
        return False
        
    def detect_and_intervene(self, 
                           action_history: List[int],
                           position_history: List[np.ndarray],
                           frontiers: np.ndarray,
                           values: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect cycles and determine intervention strategy
        
        Args:
            action_history: Recent action history
            position_history: Recent position history
            frontiers: Available frontier points
            values: Values assigned to frontiers
            
        Returns:
            Tuple[bool, Dict]: Whether intervention is needed and intervention details
        """
        # Check for position-based cycles
        position_cycle = self.detect_position_cycle(position_history)
        
        # Check for action-based cycles
        action_cycle = self.detect_action_cycle(action_history)
        
        # Determine if intervention is needed
        intervention_needed = position_cycle or action_cycle
        
        if not intervention_needed:
            return False, {}
            
        # Increment cycle count and determine intervention level
        self.cycle_count += 1
        
        # Calculate intervention level based on consecutive cycle detections
        if self.cycle_count > 3:
            self.intervention_level = min(self.intervention_level + 1, 
                                         self.intervention_levels - 1)
        
        # Create intervention information
        intervention_info = {
            'position_cycle': position_cycle,
            'action_cycle': action_cycle,
            'intervention_level': self.intervention_level,
            'cycle_count': self.cycle_count
        }
        
        # Apply intervention based on level
        if self.intervention_level == 0:
            # Level 0: Just provide information, no action modification
            pass
        elif self.intervention_level == 1:
            # Level 1: Temporarily forbid the most frequent recent action
            if len(action_history) > 0:
                recent_actions = action_history[-min(10, len(action_history)):]
                action_counts = {}
                for action in recent_actions:
                    if action not in action_counts:
                        action_counts[action] = 0
                    action_counts[action] += 1
                
                if action_counts:
                    most_frequent = max(action_counts.items(), key=lambda x: x[1])[0]
                    self.forbidden_actions = {most_frequent}
                    intervention_info['forbidden_actions'] = self.forbidden_actions
        elif self.intervention_level >= 2:
            # Level 2+: More aggressive intervention - modify frontier values
            if len(frontiers) > 0 and len(position_history) > 0:
                current_pos = position_history[-1]
                
                # Calculate distances to frontiers
                distances = np.array([np.linalg.norm(current_pos - frontier) 
                                     for frontier in frontiers])
                
                # Get furthest frontier indices
                furthest_indices = np.argsort(distances)[-min(3, len(distances)):]
                
                # Boost values for furthest frontiers to break out of local area
                boost_factor = 1.5 + (self.intervention_level - 2) * 0.25
                
                # Create modified values
                modified_values = values.copy()
                for idx in furthest_indices:
                    modified_values[idx] *= boost_factor
                
                intervention_info['modified_values'] = modified_values
        
        return True, intervention_info
    
    def get_forbidden_actions(self) -> Set[int]:
        """Get the set of currently forbidden actions"""
        return self.forbidden_actions
    
    def reset(self):
        """Reset the cycle detector state"""
        self.forbidden_actions = set()
        self.cycle_count = 0
        self.intervention_level = 0


class DecisionCoordinator:
    """
    Coordinates decision-making among multiple navigation modules
    Integrates memory, uncertainty and semantic planning contributions
    """
    
    def __init__(self, config=None):
        """
        Initialize decision coordinator
        
        Args:
            config: Configuration parameters
        """
        self.config = config or type('Config', (), {})
        
        # 使用与ITMPolicyV2相同的属性名
        self.use_mm_nav = getattr(self.config, 'use_mm_nav', False)
        self.use_ua_explore = getattr(self.config, 'use_ua_explore', False)
        self.enable_memory_repulsion = getattr(self.config, 'enable_memory_repulsion', False)
        self.enable_cycle_detection = getattr(self.config, 'enable_cycle_detection', False)
        self.enable_adaptive_explorer = getattr(self.config, 'enable_adaptive_explorer', False)
        

        
        # Load configuration
        self.memory_weight = getattr(self.config, 'memory_weight', 0.2)
        self.uncertainty_weight = getattr(self.config, 'uncertainty_weight', 0.8)
        self.semantic_weight = getattr(self.config, 'semantic_weight', 0)
        self.cycle_threshold = getattr(self.config, 'cycle_threshold', 8)
        
        # Initialize state tracking
        self.step_counter = 0
        self.action_history = []
        self.position_history = []
        self.success_found = False
        
        # Initialize cycle detector
        self.cycle_detector = CycleDetector(threshold=self.cycle_threshold)
        
        # Performance tracking
        self.stats = {
            'memory_usage': 0,
            'uncertainty_usage': 0,
            'semantic_usage': 0,
            'cycle_interventions': 0,
            'total_decision_time': 0,
            'decisions': 0
        }
        
        # Logging
        self.log_enabled = True
        logger.info("Decision Coordinator initialized")
        
        # Add missing decision_history attribute initialization
        self.decision_history = []
        
        # Add conflict history record
        self.conflict_history = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            'memory_influence': 0.0,
            'uncertainty_influence': 0.0,
            'conflicts_detected': 0,
            'interventions_applied': 0
        }
        
        # Add missing weights_adjustment_history attribute initialization
        self.weights_adjustment_history = []
        
        # Add missing error counts and max_errors
        self.error_counts = {
            'memory_errors': 0,
            'uncertainty_errors': 0,
            'integration_errors': 0
        }
        self.max_errors = 5  # 每个模块最大允许错误次数
    
    
    
    def integrate_decisions(self, base_values, **kwargs):
        """
        整合各模块决策，确保所有模块的贡献都被考虑
        修复核心：使用加权组合代替顺序覆盖
        """
        try:
            # 安全获取参数
            memory = kwargs.get('memory')
            uncertainty_estimator = kwargs.get('uncertainty_estimator')
            frontiers = kwargs.get('frontiers')
            observation = kwargs.get('observation')
            adaptive_explorer = kwargs.get('adaptive_explorer')
            
            if frontiers is None or len(frontiers) == 0:
                logger.warning("没有前沿点可供评估")
                return base_values
            
            # 确保是numpy数组
            if not isinstance(base_values, np.ndarray):
                base_values = np.array(base_values)
                
            # 记录每个模块的贡献和活动状态
            contributions = {
                'base': base_values.copy(),
                'memory': None,
                'uncertainty': None,
            }
            
            active_modules = []
            module_weights = {}
            
            # 1. 计算空间记忆贡献
            if self.use_mm_nav and memory is not None:
                try:
                    memory_values = self._apply_memory_contribution(memory, frontiers, base_values.copy())
                    contributions['memory'] = memory_values
                    active_modules.append('memory')
                    module_weights['memory'] = self.memory_weight
                    
                    # 记录差异
                    memory_diff = np.mean(np.abs(memory_values - base_values))
                    logger.info(f"空间记忆贡献: 平均差异={memory_diff:.4f}")
                except Exception as e:
                    logger.error(f"记忆贡献计算失败: {str(e)}")
            
            # 2. 计算不确定性贡献
            if self.use_ua_explore and uncertainty_estimator is not None:
                try:
                    uncertainty_values = self._apply_uncertainty_contribution(
                        uncertainty_estimator, frontiers, base_values.copy(), observation, adaptive_explorer)
                    contributions['uncertainty'] = uncertainty_values
                    active_modules.append('uncertainty')
                    module_weights['uncertainty'] = self.uncertainty_weight
                    
                    # 记录差异
                    uncertainty_diff = np.mean(np.abs(uncertainty_values - base_values))
                    logger.info(f"不确定性贡献: 平均差异={uncertainty_diff:.4f}")
                except Exception as e:
                    logger.error(f"不确定性贡献计算失败: {str(e)}")
            

            
            # 核心修复：加权组合所有模块的贡献
            if active_modules:
                # 初始化为基础值的一部分权重
                base_weight = 0.15  # 给基础值保留20%的权重
                enhanced_values = base_values.copy() * base_weight
                
                # 计算剩余权重
                remaining_weight = 1.0 - base_weight
                total_module_weight = sum(module_weights[m] for m in active_modules)
                
                # 如果总模块权重不为0，进行归一化
                if total_module_weight > 0:
                    for module in active_modules:
                        # 归一化权重
                        normalized_weight = module_weights[module] / total_module_weight * remaining_weight
                        
                        # 加权组合
                        enhanced_values += contributions[module] * normalized_weight
                        
                        # 记录实际应用的权重
                        logger.info(f"{module}贡献应用: 权重={normalized_weight:.3f}")
                else:
                    # 如果没有有效的模块权重，使用平均权重
                    for module in active_modules:
                        weight = remaining_weight / len(active_modules)
                        enhanced_values += contributions[module] * weight
                        logger.info(f"{module}贡献应用: 权重={weight:.3f}")
            
            # 确保最终值限制在[0,1]范围
            enhanced_values = np.clip(enhanced_values, 0.0, 1.0)
            
            # 记录决策过程
            decision_record = {
                'step': self.step_counter,
                'active_modules': active_modules,
                'module_weights': module_weights,
                'enhanced_values': enhanced_values
            }
            
            # 更新决策历史
            if hasattr(self, 'decision_history'):
                self.decision_history.append(decision_record)
            self.step_counter += 1
            
            # 在这里添加详细的整合日志
            active_count = len(active_modules)
            if active_count > 0:
                logger.info(f"已整合 {active_count} 个模块贡献:")
                for module in active_modules:
                    module_diff = np.mean(np.abs(contributions[module] - base_values)) if contributions[module] is not None else 0
                    module_weight = module_weights.get(module, 0)
                    logger.info(f"  - {module}: 差异={module_diff:.4f}, 权重={module_weight:.2f}")
                
                # 计算最终整合效果
                final_diff = np.mean(np.abs(enhanced_values - base_values))
                logger.info(f"最终整合效果: 与基础值平均差异={final_diff:.4f}")
            
            return enhanced_values
            
        except Exception as e:
            logger.error(f"决策整合失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return base_values
    
    def _apply_memory_contribution(self, memory, frontiers, values):
        """增强型记忆贡献计算"""
        try:
            if not hasattr(memory, 'compute_repulsion'):
                return values
            
            # 获取记忆排斥力
            repulsion_values = memory.compute_repulsion(frontiers)
            
            # 确保排斥值在[0,1]范围内
            repulsion_values = np.clip(repulsion_values, 0.0, 1.0)
            
            # 记忆参数
            memory_weight = self.memory_weight
            memory_sharpness = 1.5  # 控制排斥效应的锐度
            
            # 区域熟悉度调整
            if hasattr(memory, 'familiarity_score'):
                familiarity = memory.familiarity_score()
                # 熟悉区域降低记忆权重，减少排斥效应
                if familiarity > 0.7:
                    memory_weight *= 0.7
                # 记录到性能指标
                if hasattr(self, '_metrics'):
                    self._metrics.update_mm_nav_metrics(
                        memory_size=len(memory.positions) if hasattr(memory, 'positions') else None,
                        repulsion_factor=memory_weight
                    )
            
            # 应用排斥效应 - 改进版
            if len(repulsion_values) > 0:
                # 标准化排斥值
                min_val = min(repulsion_values)
                max_val = max(repulsion_values)
                
                if max_val > min_val:
                    # 应用非线性变换，增强对比度
                    normalized_repulsion = []
                    for v in repulsion_values:
                        # 归一化后的基础排斥值
                        base_repulsion = (v - min_val) / (max_val - min_val)
                        # 应用非线性变换
                        enhanced_repulsion = np.power(base_repulsion, memory_sharpness)
                        normalized_repulsion.append(enhanced_repulsion)
                else:
                    normalized_repulsion = [0.5 for _ in repulsion_values]
                
                # 计算加权和
                alpha = memory_weight
                for i in range(len(values)):
                    # 排斥值越高，对应的frontier价值越低
                    memory_factor = 1.0 - alpha * normalized_repulsion[i]
                    values[i] *= memory_factor
            
            # 确保最终值在[0,1]范围内
            values = np.clip(values, 0.0, 1.0)
            
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            return values
        except Exception as e:
            logger.error(f"记忆贡献计算失败: {str(e)}")
            return values

    def _apply_uncertainty_contribution(self, uncertainty_estimator, frontiers, values, observation, adaptive_explorer):
        """动态平衡的不确定性贡献"""
        try:
            # 获取不确定性信息
            uncertainty_info = uncertainty_estimator.estimate(frontiers, observation)
            uncertainties = uncertainty_info.get('frontier_uncertainties', [])

            if not uncertainties or len(uncertainties) == 0:
                return values

            # 确保长度一致
            uncertainties = np.array(uncertainties[:len(values)])
            
            # 基础权重设置
            base_uncertainty_weight = 0.4

            # 1. 根据任务进度动态调整权重
            progress = observation.get('progress', self.step_counter / 500)
            progress_factor = min(1.0, progress * 1.5)
            progress_adjusted_weight = base_uncertainty_weight * (1.0 - progress_factor * 0.6)

            # 2. 检查是否有高置信度目标检测
            detection_factor = 1.0
            if 'object_detections' in observation:
                detections = observation.get('object_detections')
                if hasattr(detections, 'scores') and len(detections.scores) > 0:
                    max_score = max(detections.scores) if len(detections.scores) > 0 else 0
                    if max_score > 0.6:
                        detection_factor = 0.3
                    elif max_score > 0.4:
                        detection_factor = 0.5

            # 最终权重计算
            uncertainty_weight = progress_adjusted_weight * detection_factor

            # 设置默认策略 - 关键修复点：确保strategy变量总是有值
            strategy = 'balanced'

            # 策略确定
            if progress < 0.25:
                strategy = 'explore'
            elif progress > 0.5 or detection_factor < 0.5:
                strategy = 'exploit'
            # 不需要else语句，因为我们已经设置了默认值

            # 记录调试信息
            logger.info(f"UA-Nav参数: 权重={uncertainty_weight:.3f}, 策略={strategy}, 进度={progress:.2f}")

            # 创建增强值
            enhanced_values = np.array(values.copy())

            # 应用不确定性贡献 (根据策略)
            if strategy == 'explore':
                # 探索模式：高不确定性提升价值
                enhanced_values = enhanced_values * (1 - uncertainty_weight) + uncertainties * uncertainty_weight
            elif strategy == 'exploit':
                # 利用模式：低不确定性提升价值
                enhanced_values = enhanced_values * (1 - uncertainty_weight) + (1 - uncertainties) * uncertainty_weight
            else:  # balanced
                # 平衡模式：值高的区域降低不确定性，值低的区域增加不确定性
                for i in range(len(enhanced_values)):
                    if enhanced_values[i] > 0.6:  # 高价值区域
                        enhanced_values[i] = enhanced_values[i] * (1 - uncertainty_weight * 0.5) + (
                                    1 - uncertainties[i]) * uncertainty_weight * 0.5
                    else:  # 低价值区域
                        enhanced_values[i] = enhanced_values[i] * (1 - uncertainty_weight) + uncertainties[
                            i] * uncertainty_weight

            # 记录贡献差异
            mean_diff = np.mean(np.abs(enhanced_values - values))
            logger.info(f"不确定性贡献: 平均差异={mean_diff:.4f}")

            # 确保返回的是numpy数组
            if not isinstance(enhanced_values, np.ndarray):
                enhanced_values = np.array(enhanced_values)
            return enhanced_values
        except Exception as e:
            logger.error(f"不确定性贡献计算失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # 添加详细错误信息
            
            # 确保返回的是numpy数组
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            return values
    
    def _compute_uncertainty_gradients(self, frontiers, uncertainties):
        """计算不确定性梯度场"""
        gradients = np.zeros(len(frontiers))
        try:
            if len(frontiers) <= 1:
                return gradients
            
            # 计算每个点的局部梯度
            for i, frontier in enumerate(frontiers):
                # 找出近邻点
                distances = np.array([np.linalg.norm(frontier - f) for f in frontiers])
                # 排除自身
                distances[i] = np.inf
                # 获取最近的3个点
                nearest_indices = np.argsort(distances)[:3]
                if len(nearest_indices) == 0:
                    continue
                    
                # 计算梯度：不确定性变化/距离变化
                neighbor_uncertainties = uncertainties[nearest_indices]
                neighbor_distances = distances[nearest_indices]
                
                # 防止除零
                neighbor_distances = np.maximum(neighbor_distances, 0.1)
                
                # 计算加权梯度
                uncertainty_diffs = neighbor_uncertainties - uncertainties[i]
                weighted_diffs = uncertainty_diffs / neighbor_distances
                gradients[i] = np.mean(weighted_diffs)
                
            return gradients
        except Exception as e:
            logger.error(f"计算不确定性梯度失败: {str(e)}")
            return np.zeros(len(frontiers))
        

    
    def update_action(self, action: int):
        """
        Update action history
        
        Args:
            action: Latest action taken
        """
        self.action_history.append(action)
        self.step_counter += 1
        
        # Keep a reasonable history size
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    

    def reset(self):
        """Reset coordinator state"""
        self.step_counter = 0
        self.action_history = []
        self.position_history = []
        self.success_found = False
        
        # Reset cycle detector
        self.cycle_detector.reset()
        
        # Reset statistics
        self.stats = {
            'memory_usage': 0,
            'uncertainty_usage': 0,
            'semantic_usage': 0,
            'cycle_interventions': 0,
            'total_decision_time': 0,
            'decisions': 0
        }
        
        logger.info("Decision Coordinator reset")
    

