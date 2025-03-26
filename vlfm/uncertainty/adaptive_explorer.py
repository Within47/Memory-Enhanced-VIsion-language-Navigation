"""
自适应探索框架

提供动态步数调整、探索价值评估和智能决策功能
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import torch
import logging
import math
import time
from collections import deque

logger = logging.getLogger("vlfm")

class AdaptiveExplorer:
    """
    自适应探索策略优化器
    基于多指标综合分析，动态调整探索与利用的平衡
    """
    
    def __init__(self, config=None):
        """初始化自适应探索框架
        
        Args:
            config: 配置参数
        """
        # 基础参数配置
        self.base_max_steps = getattr(config, 'base_max_steps', 300)
        self.exploration_threshold = getattr(config, 'exploration_threshold', 0.15)
        self.stagnation_threshold = getattr(config, 'stagnation_threshold', 30)
        self.boost_factor = getattr(config, 'boost_factor', 1.5)
        
        # 高级参数
        self.progress_smoothing = getattr(config, 'progress_smoothing', 0.8)
        self.exploration_decay = getattr(config, 'exploration_decay', 0.995)
        self.exploitation_boost = getattr(config, 'exploitation_boost', 1.2)
        self.stagnation_patience = getattr(config, 'stagnation_patience', 3)
        
        # 状态变量
        self.step_count = 0
        self.progress_history = []
        self.smoothed_progress = 0.0
        self.stagnation_count = 0
        self.consecutive_stagnations = 0
        self.exploration_rate = 1.0
        self.strategy_history = []
        
        # 任务状态评估
        self.coverage_estimate = 0.0
        self.efficiency_score = 0.5
        
        # 运行时指标
        self.metrics = {
            'exploration_vs_exploitation': [],
            'progress_rate': [],
            'strategy_switches': 0
        }
        
        # 日志
        self.log_enabled = getattr(config, 'log_enabled', False)
        if self.log_enabled:
            logger.info(f"自适应探索器初始化: 基础步数={self.base_max_steps}, 停滞阈值={self.stagnation_threshold}")
        
        # 状态跟踪
        self.step_counter = 0
        self.last_exploration_value = 0.5
        self.exploration_history = deque(maxlen=20)
        self.explored_area_history = deque(maxlen=10)
        self.detected_objects = set()
        self.value_changes = deque(maxlen=20)
        self.last_decision_time = time.time()
        
        # 环境参数估计
        self.environment_scale = 1.0  # 默认环境规模
        self.estimated_distance_to_goal = float('inf')  # 估计到目标的距离
        self.explored_ratio = 0.0  # 已探索区域比例
        
        # 动态调整参数
        self.dynamic_max_steps = self.base_max_steps
        self.current_strategy = "balanced"  # balanced, explore, exploit
        
        # 目标监测
        self.target_detection_score = 0.0  # 目标检测得分
        self.target_detection_history = deque(maxlen=5)  # 最近几次目标检测得分
        
        # 探索效率评估
        self.exploration_efficiency = 1.0
        self.consecutive_low_efficiency = 0
        
        logger.warning(f"[AdaptiveExplorer] 初始化完成，基础最大步数={self.base_max_steps}，"
                     f"探索效率阈值={self.exploration_threshold}")
        print(f"[AdaptiveExplorer] 初始化完成，基础最大步数={self.base_max_steps}")
        
    def update(self, observation, uncertainty_info):
        """
        更新探索策略
        
        Args:
            observation: 当前观察
            uncertainty_info: 不确定性信息
            
        Returns:
            dict: 更新后的状态信息
        """
        try:
            # 获取当前策略
            strategy = self._get_current_strategy(observation, uncertainty_info)
            
            # 计算策略参数
            strategy_params = self._compute_strategy_parameters(strategy, observation)
            
            # 更新状态
            state = {
                'strategy': strategy,
                'exploration_factor': strategy_params['exploration_factor'],
                'uncertainty_threshold': strategy_params['uncertainty_threshold'],
                'weights': {
                    'memory': strategy_params['memory_weight'],
                    'uncertainty': strategy_params['uncertainty_weight'],
                    'semantic': strategy_params['semantic_weight']
                }
            }
            
            # 记录策略变化
            if not hasattr(self, 'last_strategy') or self.last_strategy != strategy:
                state['strategy_changed'] = True
                self.last_strategy = strategy
            
            return state
            
        except Exception as e:
            logger.error(f"更新探索策略失败: {str(e)}")
            # 返回默认状态
            return {
                'strategy': 'balanced',
                'exploration_factor': 1.0,
                'uncertainty_threshold': 0.5,
                'weights': {
                    'memory': 0.3,
                    'uncertainty': 0.4,
                    'semantic': 0.3
                }
            }
    
    def _update_environment_estimate(self, observation: Dict[str, Any]) -> None:
        """更新环境参数估计
        
        Args:
            observation: 观察数据
        """
        # 更新环境规模估计
        if 'map_scale' in observation:
            self.environment_scale = observation['map_scale']
        elif 'explored_map' in observation and observation['explored_map'] is not None:
            # 基于地图尺寸估计环境规模
            explored_map = observation['explored_map']
            map_size = max(explored_map.shape) if hasattr(explored_map, 'shape') else 100
            self.environment_scale = max(1.0, map_size / 100.0)
        
        # 更新已探索区域比例
        if 'explored_ratio' in observation:
            self.explored_ratio = observation['explored_ratio']
            self.explored_area_history.append(self.explored_ratio)
        elif 'explored_map' in observation and observation['explored_map'] is not None:
            # 计算已探索区域比例
            explored_map = observation['explored_map']
            if hasattr(explored_map, 'sum') and hasattr(explored_map, 'size'):
                explored = explored_map.sum()
                total = explored_map.size
                if total > 0:
                    self.explored_ratio = float(explored) / total
                    self.explored_area_history.append(self.explored_ratio)
        
        # 更新到目标的估计距离
        if 'estimated_distance_to_goal' in observation:
            self.estimated_distance_to_goal = observation['estimated_distance_to_goal']
        elif 'robot_xy' in observation and 'target_xy' in observation:
            # 如果有目标位置，直接计算距离
            robot_xy = observation['robot_xy']
            target_xy = observation['target_xy']
            self.estimated_distance_to_goal = np.linalg.norm(robot_xy - target_xy)
        else:
            # 根据环境规模和已探索比例估计
            # 假设目标在未探索区域，距离与未探索区域大小成正比
            unexplored_ratio = max(0.1, 1.0 - self.explored_ratio)
            self.estimated_distance_to_goal = self.environment_scale * 10.0 * unexplored_ratio
    
    def _update_exploration_efficiency(self, observation: Dict[str, Any]) -> None:
        """更新探索效率评估
        
        Args:
            observation: 观察数据
        """
        # 如果历史记录不足，保持当前效率
        if len(self.explored_area_history) < 2:
            return
        
        # 计算最近几步的探索面积增量
        recent_explored = list(self.explored_area_history)
        
        # 计算探索面积变化率
        area_change_rate = (recent_explored[-1] - recent_explored[0]) / len(recent_explored)
        
        # 如果变化率接近0，说明探索效率低
        if abs(area_change_rate) < self.exploration_threshold / 10:
            self.consecutive_low_efficiency += 1
            # 探索效率随着连续低效率次数下降
            efficiency_decay = max(0.1, 1.0 - 0.2 * self.consecutive_low_efficiency)
            self.exploration_efficiency *= efficiency_decay
        else:
            # 探索效率恢复
            self.consecutive_low_efficiency = max(0, self.consecutive_low_efficiency - 1)
            # 根据变化率计算效率
            normalized_change = min(1.0, area_change_rate * 50.0)  # 标准化变化率
            recovery_rate = 0.2 + 0.8 * normalized_change  # 恢复率与变化率成正比
            self.exploration_efficiency = min(1.0, self.exploration_efficiency + recovery_rate)
        
        # 确保效率在有效范围内
        self.exploration_efficiency = max(0.1, min(1.0, self.exploration_efficiency))
    
    def _evaluate_exploration_value(self, observation: Dict[str, Any]) -> float:
        """评估继续探索的价值
        
        Args:
            observation: 观察数据
            
        Returns:
            探索价值 (0-1)
        """
        # 基础探索价值
        base_value = 0.5
        
        # 1. 考虑已探索区域比例
        if self.explored_ratio > 0.9:
            # 已探索90%以上，价值急剧下降
            area_factor = 0.1
        elif self.explored_ratio > 0.7:
            # 已探索70%以上，价值下降
            area_factor = 0.5 - (self.explored_ratio - 0.7) * 2.0
        else:
            # 探索比例不高，保持较高价值
            area_factor = 0.5 + (0.7 - max(0.2, self.explored_ratio)) * 0.5
        
        # 2. 考虑探索效率
        if self.exploration_efficiency < self.exploration_threshold:
            # 探索效率低，价值显著下降
            efficiency_factor = 0.1 + self.exploration_efficiency * 0.5
        else:
            # 探索效率高，保持较高价值
            efficiency_factor = 0.5 + self.exploration_efficiency * 0.5
        
        # 3. 考虑目标检测评分
        detection_factor = 0.5
        if self.target_detection_history:
            avg_detection = sum(self.target_detection_history) / len(self.target_detection_history)
            max_detection = max(self.target_detection_history)
            
            if max_detection > 0.7:
                # 已有高置信度检测，探索价值下降
                detection_factor = 0.2
            elif avg_detection > 0.4:
                # 有中等置信度检测，适度降低探索价值
                detection_factor = 0.4
            else:
                # 没有好的检测，保持较高探索价值
                detection_factor = 0.6
        
        # 4. 考虑步数相对于动态最大步数的比例
        step_ratio = self.step_counter / max(1, self.dynamic_max_steps)
        if step_ratio > 0.8:
            # 接近最大步数，价值下降
            step_factor = 0.1 + 0.3 * (1.0 - step_ratio)
        else:
            # 步数不多，保持价值
            step_factor = 0.5 + 0.3 * (0.8 - min(0.8, step_ratio))
        
        # 加权组合各因素（可以调整权重）
        weights = {
            'area': 0.3,
            'efficiency': 0.35,
            'detection': 0.25,
            'step': 0.1
        }
        
        final_value = (
            weights['area'] * area_factor +
            weights['efficiency'] * efficiency_factor +
            weights['detection'] * detection_factor +
            weights['step'] * step_factor
        )
        
        return max(0.1, min(1.0, final_value))
    
    def _compute_max_steps(self, observation: Dict[str, Any]) -> int:
        """动态计算最大步数
        
        Args:
            observation: 观察数据
            
        Returns:
            动态最大步数
        """
        # 基础步数
        base_steps = self.base_max_steps
        
        # 根据环境规模调整
        scale_factor = max(1.0, min(2.0, self.environment_scale))
        scaled_steps = base_steps * scale_factor
        
        # 根据目标检测历史调整
        detection_adjustment = 1.0
        if self.target_detection_history:
            max_detection = max(self.target_detection_history)
            if max_detection > 0.8:
                # 高置信度检测，减少步数
                detection_adjustment = 0.8
            elif max_detection > 0.5:
                # 中等置信度检测，略微减少步数
                detection_adjustment = 0.9
            elif max_detection < 0.2 and len(self.target_detection_history) > 3:
                # 持续低置信度检测，增加步数
                detection_adjustment = 1.2
        
        # 根据探索效率调整
        efficiency_adjustment = 1.0
        if self.exploration_efficiency < self.exploration_threshold:
            # 效率低，减少步数避免浪费
            efficiency_adjustment = 0.7 + 0.3 * (self.exploration_efficiency / self.exploration_threshold)
        elif self.exploration_efficiency > 0.7:
            # 效率高，适当增加步数
            efficiency_adjustment = 1.1
        
        # 综合调整
        adjusted_steps = scaled_steps * detection_adjustment * efficiency_adjustment
        
        # 如果已经找到目标（高置信度），可以给额外步数完成导航
        if self.target_detection_score > 0.8:
            navigation_bonus = min(50, int(base_steps * 0.2))
            adjusted_steps += navigation_bonus
            
        # 确保步数在合理范围内
        min_steps = int(base_steps * 0.6)  # 至少原始步数的60%
        max_steps = int(base_steps * 2.2)  # 最多原始步数的220%
        
        return max(min_steps, min(max_steps, int(adjusted_steps)))
    
    def _determine_exploration_strategy(self, progress, stagnation, stagnation_severity, uncertainty_info):
        """
        多因素自适应探索策略
        结合任务进度、停滞检测和不确定性分析动态调整探索/利用平衡
        """
        # 初始化策略
        strategy = {
            'primary_strategy': 'balanced',
            'boost_exploration': False,
            'exploration_factor': 1.0,
            'exploitation_factor': 1.0,
            'uncertainty_threshold': uncertainty_info.get('threshold', 0.5),
            'focus_area': 'global'
        }
        
        # 1. 任务进度驱动的基础策略
        if progress < 0.3:  # 任务早期
            strategy['primary_strategy'] = 'explore'
            strategy['exploration_factor'] = 1.3
            strategy['exploitation_factor'] = 0.7
            strategy['focus_area'] = 'global'
        elif progress < 0.7:  # 任务中期
            strategy['primary_strategy'] = 'balanced'
            strategy['exploration_factor'] = 1.0
            strategy['exploitation_factor'] = 1.0
            strategy['focus_area'] = 'local'
        else:  # 任务后期
            strategy['primary_strategy'] = 'exploit'
            strategy['exploration_factor'] = 0.7
            strategy['exploitation_factor'] = 1.3
            strategy['focus_area'] = 'targeted'
        
        # 2. 停滞检测触发的策略调整
        if stagnation:
            # 激活探索增强，突破停滞状态
            strategy['boost_exploration'] = True
            
            # 根据停滞严重程度动态调整增强强度
            boost_multiplier = 1.0 + 0.5 * stagnation_severity
            strategy['exploration_factor'] *= boost_multiplier
            
            # 连续多次停滞时采取更激进策略
            if self.consecutive_stagnations > self.stagnation_patience:
                strategy['exploration_factor'] *= 1.5
                strategy['focus_area'] = 'global'  # 切换到全局探索
                
                if self.log_enabled:
                    logger.info(f"Activating aggressive exploration due to persistent stagnation ({self.consecutive_stagnations} times)")
        
        # 3. 不确定性信息驱动的调整
        avg_uncertainty = uncertainty_info.get('average_uncertainty', 0.5)
        
        # 分析不确定性趋势
        uncertainty_trend = self._analyze_uncertainty_trend(uncertainty_info)
        
        # 根据趋势调整策略
        if uncertainty_trend < -0.01 and progress > 0.3:
            # 不确定性稳定下降，表明区域已被充分探索
            # 提前转向利用阶段
            if strategy['primary_strategy'] != 'exploit':
                strategy['primary_strategy'] = 'exploit'
                strategy['exploitation_factor'] *= 1.2
                
                if self.log_enabled:
                    logger.info(f"Switching to exploitation early due to decreasing uncertainty")
        
        elif uncertainty_trend > 0.01:
            # 不确定性上升，可能发现了新的复杂区域
            # 临时增强探索
            strategy['exploration_factor'] *= 1.1
        
        # 4. 效率评估驱动的平衡
        if hasattr(self, 'efficiency_score'):
            if self.efficiency_score < 0.3 and progress > 0.2:
                # 探索效率低，采取更保守策略避免资源浪费
                strategy['exploration_factor'] *= 0.9
                strategy['focus_area'] = 'local'
            elif self.efficiency_score > 0.7:
                # 当前策略效率高，加强当前策略
                if strategy['primary_strategy'] == 'explore':
                    strategy['exploration_factor'] *= 1.1
                elif strategy['primary_strategy'] == 'exploit':
                    strategy['exploitation_factor'] *= 1.1
        
        # 记录当前策略
        self.strategy_history.append(strategy)
        if len(self.strategy_history) > 20:
            self.strategy_history = self.strategy_history[-20:]
        
        # 记录探索/利用比例
        exploration_ratio = strategy['exploration_factor'] / (strategy['exploration_factor'] + strategy['exploitation_factor'])
        self.metrics['exploration_vs_exploitation'].append(exploration_ratio)
        
        return strategy
    
    def _estimate_progress(self, observation, memory):
        """
        多维度任务进度估计
        综合考虑步数、覆盖率和目标接近度
        """
        # 基于步数的进度估计
        step_progress = min(1.0, self.step_count / self.base_max_steps)
        
        # 基于内存覆盖的进度估计
        memory_progress = self._estimate_coverage(memory)
        
        # 基于目标接近度的进度估计（如果有目标信息）
        goal_progress = 0.0
        if 'distance_to_goal' in observation:
            dist = observation['distance_to_goal']
            if isinstance(dist, torch.Tensor):
                dist = dist.item()
            # 距离越近，进度越高
            goal_progress = max(0.0, 1.0 - dist / 15.0)  # 假设15m为最远距离
        
        # 组合多维度进度，权重视具体任务而定
        if 'distance_to_goal' in observation:
            # 如果有目标距离信息，给予较高权重
            combined_progress = 0.4 * step_progress + 0.3 * memory_progress + 0.3 * goal_progress
        else:
            # 否则主要依赖步数和内存覆盖
            combined_progress = 0.6 * step_progress + 0.4 * memory_progress
        
        # 记录进度变化率
        if len(self.progress_history) > 0:
            progress_rate = combined_progress - self.progress_history[-1]
            self.metrics['progress_rate'].append(progress_rate)
        
        return combined_progress
    
    def _estimate_coverage(self, memory):
        """估计环境覆盖率"""
        if memory is None:
            return 0.0
            
        # 如果内存模块提供统计信息，使用内存大小估计覆盖率
        if hasattr(memory, 'get_memory_stats'):
            stats = memory.get_memory_stats()
            memory_size = stats.get('memory_size', 0)
            # 根据内存大小估计覆盖率，假设100个点为充分覆盖
            coverage = min(1.0, memory_size / 100)
            
            # 如果有访问统计，也考虑多次访问的情况
            if 'unique_locations' in stats and memory_size > 0:
                unique_ratio = stats['unique_locations'] / memory_size
                # 低唯一比例表示更多重复访问，可能意味着较低的覆盖率
                return coverage * (0.5 + 0.5 * unique_ratio)
            
            return coverage
        
        # 如果没有内存统计，使用基于步数的简单估计
        return min(0.8, self.step_count / (self.base_max_steps * 1.2))
    
    def _detect_stagnation(self):
        """
        高级停滞检测算法
        检测导航中多种类型的停滞模式
        """
        if len(self.progress_history) < self.stagnation_threshold:
            return False
        
        # 1. 进度停滞检测 - 观察最近进度变化
        recent = self.progress_history[-self.stagnation_threshold:]
        progress_change = recent[-1] - recent[0]
        
        # 基本停滞标准 - 进度增长低于阈值
        is_stagnating = progress_change < self.exploration_threshold
        
        # 2. 振荡检测 - 识别在相似区域反复徘徊的模式
        if len(recent) >= 10:  # 需要足够的历史数据
            # 计算进度的一阶差分
            deltas = np.diff(recent)
            # 计算符号变化次数 (正负交替次数)
            sign_changes = np.sum(np.diff(np.signbit(deltas)) != 0)
            # 高频率的符号变化表示振荡
            is_oscillating = sign_changes >= len(deltas) * 0.4  # 40%以上的变化
            
            # 振荡也是一种停滞形式
            is_stagnating = is_stagnating or is_oscillating
        
        return is_stagnating
    
    def _assess_stagnation_severity(self):
        """评估停滞的严重程度"""
        if len(self.progress_history) < self.stagnation_threshold:
            return 0.0
            
        recent = self.progress_history[-self.stagnation_threshold:]
        
        # 1. 进度变化的绝对量 - 越小表示停滞越严重
        progress_change = abs(recent[-1] - recent[0])
        change_severity = 1.0 - min(1.0, progress_change / self.exploration_threshold)
        
        # 2. 持续时间 - 连续停滞越久越严重
        duration_severity = min(1.0, self.consecutive_stagnations / 5)
        
        # 3. 效率下降 - 如果效率评分低，停滞可能更严重
        efficiency_severity = 1.0 - self.efficiency_score
        
        # 综合评分
        combined_severity = 0.4 * change_severity + 0.3 * duration_severity + 0.3 * efficiency_severity
        
        return combined_severity
    
    def _evaluate_exploration_efficiency(self):
        """
        评估探索效率
        基于进度变化率、内存增长和策略稳定性
        """
        efficiency = 0.5  # 默认中等效率
        
        # 1. 进度变化率 - 高变化率表示高效率
        if len(self.metrics['progress_rate']) > 5:
            recent_rates = self.metrics['progress_rate'][-5:]
            avg_rate = sum(recent_rates) / len(recent_rates)
            # 正向调整效率评分
            rate_factor = min(1.0, avg_rate / 0.01)  # 假设0.01为参考增长率
            efficiency += 0.3 * rate_factor
        
        # 2. 探索平衡性 - 过于频繁的策略切换可能表示低效率
        if len(self.strategy_history) > 5:
            strategy_types = [s['primary_strategy'] for s in self.strategy_history[-5:]]
            # 计算不同策略的数量
            unique_strategies = len(set(strategy_types))
            # 太多切换意味着不稳定，可能效率低
            stability_factor = 1.0 - (unique_strategies - 1) / 4
            efficiency += 0.2 * stability_factor
        
        # 3. 步数效率 - 如果进度与步数比例良好
        if self.step_count > 0 and len(self.progress_history) > 0:
            progress = self.progress_history[-1]
            step_efficiency = progress / (self.step_count / 200)  # 假设200步达到进度1.0为参考
            efficiency += 0.2 * min(1.0, step_efficiency)
        
        # 确保结果在[0,1]范围
        return max(0.0, min(1.0, efficiency))
    
    def _analyze_uncertainty_trend(self, uncertainty_info):
        """分析不确定性趋势"""
        if 'uncertainty_history' in uncertainty_info and len(uncertainty_info['uncertainty_history']) > 5:
            history = uncertainty_info['uncertainty_history']
            # 计算线性趋势
            x = np.arange(len(history))
            y = np.array(history)
            # 简化的线性回归
            slope = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (np.mean(x*x) - np.mean(x)**2)
            return slope
        return 0.0
    
    def get_adaptive_weights(self):
        """
        获取适应性权重
        根据当前探索阶段调整不同模块的权重
        """
        # 基于当前状态计算最优权重
        if not self.progress_history:
            return {'memory_weight': 0.4, 'uncertainty_weight': 0.6}
        
        progress = self.progress_history[-1]
        
        # 根据进度调整权重 - 随进度增加记忆权重
        memory_base = 0.35 + 0.3 * progress
        uncertainty_base = 0.65 - 0.3 * progress
        
        # 考虑停滞状态
        if self.consecutive_stagnations > 0:
            # 停滞时增加不确定性权重，鼓励探索
            stagnation_factor = min(0.2, 0.05 * self.consecutive_stagnations)
            memory_base -= stagnation_factor
            uncertainty_base += stagnation_factor
        
        # 确保权重和为1
        total = memory_base + uncertainty_base
        memory_weight = memory_base / total
        uncertainty_weight = uncertainty_base / total
        
        return {
            'memory_weight': memory_weight,
            'uncertainty_weight': uncertainty_weight
        }
    
    def reset(self):
        """重置探索器状态"""
        self.step_count = 0
        self.progress_history = []
        self.smoothed_progress = 0.0
        self.stagnation_count = 0
        self.consecutive_stagnations = 0
        self.exploration_rate = 1.0
        self.strategy_history = []
        self.coverage_estimate = 0.0
        self.efficiency_score = 0.5
        
        # 重置指标
        self.metrics = {
            'exploration_vs_exploitation': [],
            'progress_rate': [],
            'strategy_switches': 0
        }
        
        if self.log_enabled:
            logger.info("自适应探索器已重置")
        
        self.step_counter = 0
        self.last_exploration_value = 0.5
        self.exploration_history.clear()
        self.explored_area_history.clear()
        self.detected_objects.clear()
        self.value_changes.clear()
        self.target_detection_history.clear()
        
        self.environment_scale = 1.0
        self.estimated_distance_to_goal = float('inf')
        self.explored_ratio = 0.0
        
        self.dynamic_max_steps = self.base_max_steps
        self.current_strategy = "balanced"
        
        self.target_detection_score = 0.0
        
        self.exploration_efficiency = 1.0
        self.consecutive_low_efficiency = 0
        
        self.smoothed_progress = 0.0
        self.stagnation_count = 0
        self.consecutive_stagnations = 0
        
        logger.warning("[AdaptiveExplorer] 已重置")
        print("[AdaptiveExplorer] 已重置")
    
    def _compute_strategy_parameters(self, strategy, observation):
        """
        计算探索策略的参数
        
        Args:
            strategy: 当前探索策略
            observation: 当前观察
            
        Returns:
            dict: 策略参数
        """
        params = {
            'exploration_factor': 1.0,
            'uncertainty_threshold': 0.5,
            'memory_weight': 0.3,
            'uncertainty_weight': 0.4,
            'semantic_weight': 0.3
        }
        
        try:
            # 根据策略类型调整参数
            if strategy == 'explore':
                params['exploration_factor'] = 1.5
                params['uncertainty_threshold'] = 0.7
                params['uncertainty_weight'] = 0.6
                params['memory_weight'] = 0.2
                params['semantic_weight'] = 0.2
            elif strategy == 'exploit':
                params['exploration_factor'] = 0.5
                params['uncertainty_threshold'] = 0.3
                params['uncertainty_weight'] = 0.2
                params['memory_weight'] = 0.3
                params['semantic_weight'] = 0.5
            elif strategy == 'balanced':
                # 使用默认值
                pass
            
            # 根据观察调整参数
            if 'progress' in observation:
                progress = observation['progress']
                if progress > 0.8:  # 任务后期
                    params['exploration_factor'] *= 0.7  # 减少探索
                elif progress < 0.2:  # 任务早期
                    params['exploration_factor'] *= 1.3  # 增加探索
            
            # 检查是否有目标物体检测
            if 'object_detections' in observation:
                if len(observation['object_detections']) > 0:
                    # 有目标物体时增加利用权重
                    params['exploration_factor'] *= 0.8
                    params['semantic_weight'] *= 1.2
            
            return params
            
        except Exception as e:
            logger.error(f"计算策略参数失败: {str(e)}")
            return params  # 返回默认参数

    def _get_current_strategy(self, observation, uncertainty_info):
        """
        获取当前探索策略

        Args:
            observation: 当前观察
            uncertainty_info: 不确定性信息

        Returns:
            str: 当前策略 ('explore', 'exploit', 'balanced')
        """
        # 首先设置默认策略
        strategy = 'balanced'

        try:
            # 1. 获取当前进度
            progress = observation.get('progress', self.step_counter / max(1, self.base_max_steps))

            # 2. 决策逻辑
            if progress < 0.3:
                # 任务早期优先探索
                strategy = 'explore'
            elif progress > 0.7:
                # 任务后期优先利用
                strategy = 'exploit'
            # 否则保持默认的balanced策略

            # 更新当前策略属性
            self.current_strategy = strategy

        except Exception as e:
            logger.error(f"获取当前策略失败: {str(e)}")
            # 出错时使用默认策略
            self.current_strategy = strategy

        return strategy