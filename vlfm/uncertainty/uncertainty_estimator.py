import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from dataclasses import dataclass
import time
import cv2

from collections import deque

try:
    from vlfm.utils.tensor_utils import process_tensor, ensure_numpy, batch_process_observation
except ImportError:
    # 如果导入失败，提供内联定义
    def process_tensor(tensor, device='cpu', dtype=None):
        """简化的张量处理函数"""
        if tensor is None:
            return None

        if torch.is_tensor(tensor):
            try:
                return tensor.detach().cpu()
            except:
                return tensor
        return tensor

    def ensure_numpy(tensor):
        """简化的numpy转换函数"""
        if tensor is None:
            return None

        if torch.is_tensor(tensor):
            try:
                return tensor.detach().cpu().numpy()
            except:
                return np.array([])
        return np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor

    def batch_process_observation(observation, device='cpu'):
        """简化的观察处理函数"""
        return observation

logger = logging.getLogger(__name__)

class UncertaintyEstimator:
    """
    基于贝叶斯推理的不确定性估计器
    区分认知不确定性(epistemic)和环境不确定性(aleatoric)
    """

    def __init__(self, config=None):
        """初始化不确定性评估器"""
        # 配置参数
        self.base_threshold = getattr(config, 'base_threshold', 0.5)
        self.threshold_decay = getattr(config, 'threshold_decay', 0.995)
        self.min_threshold = getattr(config, 'min_threshold', 0.2)
        self.max_threshold = getattr(config, 'max_threshold', 0.8)
        self.monte_carlo_samples = getattr(config, 'monte_carlo_samples', 10)

        # 运行状态
        self.threshold = self.base_threshold
        self.prior_variance = 1.0  # 先验方差
        self.observation_history = deque(maxlen=100)
        self.uncertainty_history = deque(maxlen=20)
        self.visit_counts = {}  # 位置访问计数
        self.value_history = {}  # 位置值历史

        # 进度跟踪
        self.steps = 0
        self.max_steps = getattr(config, 'max_steps', 500)
        self.progress_factor = 0.0

        # 记录最近估计
        self.last_estimation = {}

        # 添加位置相关属性和历史记录
        self.position_history = deque(maxlen=20)  # 确保该属性存在
        self.feature_history = {}  # 添加特征历史
        self.detection_history = {}  # 添加检测历史

    def reset(self):
        """重置不确定性评估器状态"""
        self.threshold = self.base_threshold
        self.observation_history.clear()
        self.uncertainty_history.clear()
        self.visit_counts = {}
        self.value_history = {}
        self.steps = 0
        self.progress_factor = 0.0
        self.last_estimation = {}
        if hasattr(self, 'value_variance_history'):
            self.value_variance_history.clear()

    def _estimate_epistemic_uncertainty(self, position, visit_counts):
        """改进的贝叶斯不确定性估计，增加区分度"""
        # 获取访问计数
        pos_key = self._position_to_key(position)
        visits = visit_counts.get(pos_key, 0)

        # 基础不确定性计算
        if visits == 0:
            # 未访问区域最高不确定性
            uncertainty = 1.0
        else:
            # 使用非线性衰减曲线
            decay_rate = 0.5  # 增加衰减速率
            base_uncertainty = np.exp(-decay_rate * (visits ** 0.6))

            # 添加随机扰动，增加多样性
            random_factor = np.random.normal(1.0, 0.05)  # 5%随机变化
            uncertainty = np.clip(base_uncertainty * random_factor, 0.1, 1.0)

        # 考虑空间局部性
        if hasattr(self, '_visited_positions') and len(self._visited_positions) > 0:
            # 计算与最近访问点的距离
            recent_positions = self._visited_positions[-10:]  # 最近10个访问点
            distances = [np.linalg.norm(position - p) for p in recent_positions]
            min_dist = min(distances) if distances else 99.0

            # 距离因子：远离已访问区域的点有更高不确定性
            distance_factor = min(1.0, min_dist / 3.0)
            uncertainty = 0.7 * uncertainty + 0.3 * distance_factor

        return np.clip(uncertainty, 0.15, 1.0)  # 确保合理范围

    def estimate(self, frontiers, observation):
        """
        评估前沿的不确定性
        区分认知不确定性(可通过探索减少)和偶然不确定性(内在随机性)
        """
        start_time = time.time()

        # 确保frontiers是numpy数组
        frontiers = ensure_numpy(frontiers)

        # 更新任务进度
        self.steps += 1
        self.progress_factor = min(1.0, self.steps / self.max_steps)

        try:
            # 更新观察历史
            self._update_observation_history(observation)

            # 计算每个前沿的不确定性
            frontier_uncertainties = []
            epistemic_uncertainties = []  # 认知不确定性
            aleatoric_uncertainties = []  # 偶然不确定性

            for frontier in frontiers:
                # 计算认知不确定性(基于位置访问频次)
                epistemic = self._estimate_epistemic_uncertainty(frontier, self.visit_counts)

                # 计算偶然不确定性(基于观察变化)
                aleatoric = 0.5  # 默认中等不确定性

                # 将frontier转为离散格点以支持访问计数
                position_key = self._position_to_key(frontier)
                value_history = self.value_history.get(position_key, [])

                if len(value_history) > 1:
                    # 如果有值历史，使用其方差作为偶然不确定性估计
                    aleatoric = min(1.0, np.std(value_history) / 0.5)

                # 综合两种不确定性
                total_uncertainty = self._combine_uncertainties(epistemic, aleatoric, self.progress_factor)

                frontier_uncertainties.append(total_uncertainty)
                epistemic_uncertainties.append(epistemic)
                aleatoric_uncertainties.append(aleatoric)

            # 自适应调整阈值
            self._update_threshold(frontier_uncertainties)

            # 添加详细日志 - 新增
            if frontier_uncertainties:
                logger.info(f"不确定性估计: 最小={min(frontier_uncertainties):.2f}, 最大={max(frontier_uncertainties):.2f}, 均值={np.mean(frontier_uncertainties):.2f}")
                # 监控访问计数分布
                visit_counts_values = list(self.visit_counts.values())
                if visit_counts_values:
                    logger.info(f"位置访问计数: 最小={min(visit_counts_values)}, 最大={max(visit_counts_values)}, 均值={np.mean(visit_counts_values):.1f}")

            result = {
                'frontier_uncertainties': frontier_uncertainties,
                'epistemic_uncertainties': epistemic_uncertainties,
                'aleatoric_uncertainties': aleatoric_uncertainties,
                'threshold': self.threshold,
                'task_progress': self.progress_factor,
                'computation_time': time.time() - start_time
            }

            self.last_estimation = result

            if frontier_uncertainties:
                self.uncertainty_history.append(np.mean(frontier_uncertainties))

            return result

        except Exception as e:
            logger.error(f"不确定性估计失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 返回默认值
            default_uncertainties = [0.5] * len(frontiers)
            return {
                'frontier_uncertainties': default_uncertainties,
                'epistemic_uncertainties': default_uncertainties,
                'aleatoric_uncertainties': default_uncertainties,
                'threshold': self.threshold,
                'task_progress': self.progress_factor
            }

    def _update_observation_history(self, observation):
        """
        更新观察历史
        修改点:
        1. 使用统一张量处理工具
        2. 增强错误处理
        3. 增加位置状态跟踪
        """
        try:
            # 提取当前位置
            if 'gps' in observation:
                # 使用统一张量处理 - 修改点1
                current_pos = ensure_numpy(observation['gps'])

                if len(current_pos.shape) > 1:
                    current_pos = current_pos[0]  # 取第一个维度

                # 离散化位置
                grid_pos = tuple(np.round(current_pos / 0.5).astype(int))

                # 更新访问计数
                if grid_pos not in self.visit_counts:
                    self.visit_counts[grid_pos] = 0
                self.visit_counts[grid_pos] += 1

                # 位置状态跟踪 - 新增
                if not hasattr(self, 'position_history'):
                    self.position_history = deque(maxlen=20)
                self.position_history.append(current_pos)

                # 如果有前沿值信息，更新值历史
                if hasattr(self, 'last_frontier_values') and self.last_frontier_values:
                    if grid_pos not in self.value_history:
                        self.value_history[grid_pos] = []
                    # 仅保留最近的历史
                    if len(self.value_history[grid_pos]) >= 5:
                        self.value_history[grid_pos] = self.value_history[grid_pos][-4:]
                    self.value_history[grid_pos].append(np.mean(self.last_frontier_values))

            # 存储观察，确保所有张量都转换为numpy - 修改点2
            # 使用工具函数批量处理
            processed_observation = {}
            for key, value in observation.items():
                # 跳过可能导致存储问题的大张量
                if key in ['rgb', 'depth', 'semantic']:
                    continue

                processed_observation[key] = ensure_numpy(value)

            self.observation_history.append(processed_observation)

        except Exception as e:
            logger.error(f"观察历史更新失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _update_threshold(self, uncertainties):
        """动态调整不确定性阈值"""
        if not uncertainties:
            return

        # 计算当前不确定性分布统计
        current_mean = np.mean(uncertainties)

        # 计算熵增率 - 评估不确定性分布变化
        entropy_change = 0.0
        if len(self.uncertainty_history) > 1:
            # 计算当前和历史不确定性的熵增
            hist_uncertainties = list(self.uncertainty_history)
            hist_mean = np.mean(hist_uncertainties)

            # 如果均值存在显著变化，调整阈值
            mean_change = current_mean - hist_mean
            if abs(mean_change) > 0.1:
                # 不确定性增加 -> 提高阈值
                if mean_change > 0:
                    self.threshold = min(self.max_threshold,
                                       self.threshold + 0.05)
                # 不确定性减少 -> 降低阈值
                else:
                    self.threshold = max(self.min_threshold,
                                       self.threshold - 0.03)
            else:
                # 应用正常衰减
                self.threshold = max(self.min_threshold,
                                   self.threshold * self.threshold_decay)

        # 任务进度感知的阈值调整
        if self.progress_factor > 0.8:
            # 任务后期降低阈值，鼓励利用
            self.threshold = max(self.min_threshold, self.threshold * 0.95)

    def update_from_frontier_values(self, frontiers, values):
        """改进的值历史更新策略"""
        self.last_frontier_values = values

        # 提取当前位置
        if not hasattr(self, '_current_position') or self._current_position is None:
            return

        grid_pos = tuple(np.round(self._current_position / 0.5).astype(int))

        # 更新值历史，使用指数滑动平均
        if grid_pos not in self.value_history:
            self.value_history[grid_pos] = []

        if values:
            # 使用最大值和平均值的加权组合
            value_avg = np.mean(values)
            value_max = np.max(values)
            combined_value = 0.7 * value_max + 0.3 * value_avg

            # 平滑更新
            if self.value_history[grid_pos]:
                alpha = 0.7  # 平滑因子
                smoothed_value = alpha * combined_value + (1 - alpha) * self.value_history[grid_pos][-1]
            else:
                smoothed_value = combined_value

            # 仅保留最近的历史
            if len(self.value_history[grid_pos]) >= 5:
                self.value_history[grid_pos] = self.value_history[grid_pos][-4:]

            self.value_history[grid_pos].append(smoothed_value)

            # 分析前沿值分布特征
            value_variance = np.var(values) if len(values) > 1 else 0.0

            # 存储这一指标
            if not hasattr(self, 'value_variance_history'):
                self.value_variance_history = deque(maxlen=10)
            self.value_variance_history.append(value_variance)

            # 基于分布特征动态调整阈值
            if len(self.value_variance_history) > 5:
                avg_variance = np.mean(self.value_variance_history)
                if avg_variance < 0.01:  # 极低方差 - 可能困在局部最优
                    # 提高阈值促进探索
                    self.threshold = min(self.max_threshold, self.threshold * 1.1)
                elif avg_variance > 0.2:  # 极高方差 - 可能过度探索
                    # 降低阈值，促进局部利用
                    self.threshold = max(self.min_threshold, self.threshold * 0.95)

    def get_progress(self):
        """获取任务进度"""
        return self.progress_factor

    def get_uncertainty_stats(self):
        """获取不确定性统计信息"""
        stats = {
            'current_threshold': self.threshold,
            'mean_uncertainty': np.mean(list(self.uncertainty_history)) if self.uncertainty_history else 0.0,
            'progress': self.progress_factor,
            'visited_locations': len(self.visit_counts)
        }
        return stats

    def visualize(self, map_size: Tuple[int, int] = (500, 500)) -> np.ndarray:
        """生成不确定性可视化图像

        Args:
            map_size: 可视化图像大小

        Returns:
            可视化图像
        """
        # 创建画布
        canvas = np.ones((map_size[0], map_size[1], 3), dtype=np.uint8) * 255

        if len(self.uncertainty_history) == 0:
            return canvas

        # 获取不确定性历史
        uncertainties = self.uncertainty_history
        normalized_uncertainties = np.array(uncertainties) / max(1e-5, max(uncertainties))

        # 计算不确定性热力图
        for i, uncertainty in enumerate(normalized_uncertainties):
            if i >= len(normalized_uncertainties):
                break

            # 颜色表示不确定性（蓝->绿->红，低->高）
            color = self._get_uncertainty_color(uncertainty)

            # 点大小表示不确定性
            radius = int(3 + uncertainty * 10)

            # 绘制点
            x = int(i * (map_size[1] / len(normalized_uncertainties)))
            if 0 <= x < map_size[1]:
                cv2.circle(canvas, (x, int(map_size[0] / 2)), radius, color, -1)

            # 连接相邻点
            if i > 0:
                prev_x = int((i - 1) * (map_size[1] / len(normalized_uncertainties)))
                if 0 <= prev_x < map_size[1]:
                    cv2.line(canvas, (prev_x, int(map_size[0] / 2)), (x, int(map_size[0] / 2)), (200, 200, 200), 1)

        # 绘制信息面板
        panel_height = 120
        panel = np.ones((panel_height, map_size[1], 3), dtype=np.uint8) * 240

        # 绘制不确定性历史曲线
        if len(self.uncertainty_history) > 1:
            steps = np.arange(len(self.uncertainty_history))
            uncertainties = np.array(self.uncertainty_history)

            # 归一化
            max_step = max(1, max(steps))
            steps = steps / max_step * (map_size[1] - 20)

            # 绘制曲线
            curve_height = 70
            for i in range(1, len(steps)):
                # 不确定性曲线
                y1 = int(panel_height - 20 - uncertainties[i-1] * curve_height)
                y2 = int(panel_height - 20 - uncertainties[i] * curve_height)
                x1 = int(10 + steps[i-1])
                x2 = int(10 + steps[i])
                if 0 <= x1 < map_size[1] and 0 <= x2 < map_size[1]:
                    cv2.line(panel, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 添加文字说明
        cv2.putText(panel, "UA-Explore Uncertainty Map", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(panel, f"Steps: {self.steps} | Threshold: {self.threshold:.3f}",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(panel, f"Avg Uncertainty: {np.mean(uncertainties):.3f}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 添加图例
        cv2.putText(panel, "Red: Uncertainty", (map_size[1] - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 合并画布和面板
        result = np.vstack([canvas, panel])

        return result

    def _get_uncertainty_color(self, uncertainty: float) -> Tuple[int, int, int]:
        """获取不确定性对应的颜色

        从蓝色（低不确定性）到红色（高不确定性）

        Args:
            uncertainty: 不确定性值 [0,1]

        Returns:
            BGR颜色元组
        """
        blue = int(255 * (1 - uncertainty))
        green = int(255 * (1 - abs(uncertainty - 0.5) * 2))
        red = int(255 * uncertainty)
        return (blue, green, red)

    def _combine_uncertainties(self, epistemic, aleatoric, progress):
        """组合认知不确定性和偶然不确定性"""
        if progress < 0.3:
            # 早期阶段:80%认知+20%偶然
            return 0.8 * epistemic + 0.2 * aleatoric
        elif progress < 0.7:
            # 中期阶段:平衡两种不确定性
            return 0.5 * epistemic + 0.5 * aleatoric
        else:
            # 后期阶段:30%认知+70%偶然
            return 0.3 * epistemic + 0.7 * aleatoric

    def _position_to_key(self, position):
        """
        将位置转换为字典键

        Args:
            position: numpy数组形式的位置坐标

        Returns:
            tuple: 离散化后的位置坐标元组
        """
        try:
            # 确保position是numpy数组
            if torch.is_tensor(position):
                position = position.cpu().numpy()
            position = np.array(position)

            # 使用较大的网格尺寸来减少状态空间
            grid_size = 0.5  # 0.5米的网格大小

            # 离散化坐标
            discrete_position = np.round(position / grid_size).astype(int)

            # 转换为元组作为字典键
            return tuple(discrete_position)
        except Exception as e:
            logger.error(f"位置转换失败: {str(e)}")
            # 返回一个默认值
            return (0, 0)

    def _get_current_strategy(self, observation, uncertainty_info):
        """更灵活的探索-利用策略选择"""
        progress = self.progress_factor

        # 策略1: 基于进度的渐进转换
        if progress < 0.25:
            base_strategy = 'explore'
        elif progress < 0.5:
            base_strategy = 'balanced'
        else:
            base_strategy = 'exploit'

        # 策略2: 检测到高置信度目标时立即切换到利用
        if 'object_detections' in observation:
            detections = observation.get('object_detections')
            if hasattr(detections, 'scores') and len(detections.scores) > 0:
                max_score = max(detections.scores) if len(detections.scores) > 0 else 0
                if max_score > 0.6:
                    return 'exploit'  # 立即切换到利用模式

        # 策略3: 检测到不确定性变化趋势
        if hasattr(self, '_uncertainty_history') and len(self._uncertainty_history) > 5:
            recent_trend = np.mean(self._uncertainty_history[-3:]) - np.mean(self._uncertainty_history[-6:-3])
            if recent_trend < -0.1 and base_strategy == 'balanced':
                # 不确定性显著下降，提前切换到利用
                return 'exploit'

        return base_strategy

    def _apply_uncertainty_contribution(self, uncertainty_estimator, frontiers, values, observation, adaptive_explorer):
        """增强的不确定性贡献计算"""
        try:
            # 获取不确定性信息
            uncertainty_info = uncertainty_estimator.estimate(frontiers, observation)
            uncertainties = uncertainty_info.get('frontier_uncertainties', [])

            if not uncertainties or len(uncertainties) == 0:
                logger.warning("未获取到有效的不确定性估计")
                return values

            # 确保长度一致
            uncertainties = np.array(uncertainties[:len(values)])

            # 获取探索策略
            strategy = 'balanced'  # 默认平衡策略
            uncertainty_weight = 0.4  # 默认权重

            if adaptive_explorer is not None:
                try:
                    strategy_info = adaptive_explorer.update(observation, uncertainty_info)
                    strategy = strategy_info.get('strategy', 'balanced')
                    uncertainty_weight = strategy_info.get('weights', {}).get('uncertainty', 0.4)
                except Exception as e:
                    logger.error(f"获取探索策略失败: {str(e)}")

            # 创建增强值数组
            enhanced_values = np.array(values.copy())

            # 添加详细日志，帮助调试
            logger.info(f"不确定性贡献: 策略={strategy}, 权重={uncertainty_weight:.2f}")
            logger.info(f"原始值范围: {np.min(values):.2f}-{np.max(values):.2f}")
            logger.info(f"不确定性范围: {np.min(uncertainties):.2f}-{np.max(uncertainties):.2f}")

            # 根据策略应用不确定性贡献 - 关键修复点
            if strategy == 'explore':
                # 探索模式：高不确定性区域更有价值
                for i in range(len(enhanced_values)):
                    if i >= len(uncertainties):
                        continue
                    # 探索模式下，不确定性是正向贡献
                    enhanced_values[i] = enhanced_values[i] * (1 - uncertainty_weight) + \
                                       uncertainties[i] * uncertainty_weight

            elif strategy == 'exploit':
                # 利用模式：低不确定性区域更有价值
                for i in range(len(enhanced_values)):
                    if i >= len(uncertainties):
                        continue
                    # 利用模式下，确定性(1-不确定性)是正向贡献
                    enhanced_values[i] = enhanced_values[i] * (1 - uncertainty_weight) + \
                                       (1.0 - uncertainties[i]) * uncertainty_weight
            else:  # 'balanced'
                # 平衡模式：综合考虑当前值和不确定性
                for i in range(len(enhanced_values)):
                    if i >= len(uncertainties):
                        continue
                    # 根据当前值高低决定不确定性影响方向
                    if enhanced_values[i] > 0.6:  # 高价值区域，更确定更好
                        contribution = (1.0 - uncertainties[i])
                    else:  # 低价值区域，更不确定更好(探索潜力)
                        contribution = uncertainties[i]
                    enhanced_values[i] = enhanced_values[i] * (1 - uncertainty_weight) + \
                                       contribution * uncertainty_weight

            # 确保值范围有效
            if np.max(enhanced_values) > 0:
                # 归一化，保持相对关系
                enhanced_values = enhanced_values / np.max(enhanced_values)

            # 记录最终结果
            logger.info(f"增强后值范围: {np.min(enhanced_values):.2f}-{np.max(enhanced_values):.2f}")

            # 返回修改后的值 - 关键修复点
            return enhanced_values

        except Exception as e:
            logger.error(f"不确定性贡献计算失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return values

    def _process_tensor(self, tensor, device='cpu'):
        """统一张量处理函数"""
        if tensor is None:
            return None

        if torch.is_tensor(tensor):
            return tensor.to(device).detach()
        elif isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, device=device)
        else:
            try:
                return torch.tensor(tensor, device=device)
            except:
                return tensor