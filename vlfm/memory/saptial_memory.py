import numpy as np
import torch
import math
from typing import Dict, List, Tuple, Optional, Any
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger("vlfm")

class SpatialMemory:

    def __init__(self, config=None):
        self.min_distance = getattr(config, 'min_distance', 0.2)
        self.max_memory_size = getattr(config, 'max_memory_size', 1000)
        self.repulsion_radius = getattr(config, 'repulsion_radius', 2.0)
        self.repulsion_strength = getattr(config, 'repulsion_strength', 1.0)
        self.decay_factor = getattr(config, 'decay_factor', 0.98)
        
        self.positions = []  # 位置历史
        self.timestamps = []  # 时间戳
        self.importance = []  # 重要性权重
        self.last_position = None
        self.current_step = 0
        
        # KD tree search
        self.kdtree = None
        self.kdtree_valid = False
        

        self.cycle_detection_window = getattr(config, 'cycle_detection_window', 10)
        self.cycle_similarity_threshold = getattr(config, 'cycle_similarity_threshold', 0.75)
    
    def add_position(self, position):
        """Adding a new location to memory"""
        if self.last_position is not None:
            distance = np.linalg.norm(position - self.last_position)
            if distance < self.min_distance:
                self.current_step += 1
                return False
        
        self.positions.append(position.copy())
        self.timestamps.append(self.current_step)
        self.importance.append(1.0)
        self.last_position = position.copy()
        self.current_step += 1

        # Memory capacity management
        if len(self.positions) > self.max_memory_size:
            # Filter the memories to keep based on importance
            indices = np.argsort(self.importance)
            # Keep the most important 70% and the most recent 30%
            keep_important = indices[-int(self.max_memory_size * 0.7):]
            recent_end = min(len(indices), int(self.max_memory_size * 0.3))
            keep_recent = indices[:recent_end]
            keep_indices = np.concatenate([keep_important, keep_recent])
            keep_indices = np.unique(keep_indices)
            

            self.positions = [self.positions[i] for i in keep_indices]
            self.timestamps = [self.timestamps[i] for i in keep_indices]
            self.importance = [self.importance[i] for i in keep_indices]
        
        # Mark the KD tree needs to be rebuilt
        self.kdtree_valid = False
        
        return True
    
    def update_memory(self):
        if self.positions:
            if not hasattr(self, 'visit_counts'):
                self.visit_counts = {}
            
            for i in range(len(self.importance)):
                # Calculate time decay
                time_factor = np.exp(-(self.current_step - self.timestamps[i]) / 500.0)
                
                # Calculating location importance: Frequency of access - frequently accessed areas are kept in memory
                pos_tuple = tuple(np.round(self.positions[i], 2))
                if pos_tuple in self.visit_counts:
                    visit_factor = min(1.0, self.visit_counts[pos_tuple] / 5.0)
                    importance_decay = self.decay_factor * (0.8 + 0.2 * visit_factor)
                else:
                    importance_decay = self.decay_factor

                self.importance[i] *= importance_decay
                self.importance[i] *= time_factor

            self.kdtree_valid = False
    
    def rebuild_kdtree(self):
        """重建KD树索引"""
        if self.positions:
            from scipy.spatial import cKDTree
            self.kdtree = cKDTree(np.array(self.positions))
            self.kdtree_valid = True
    
    def get_nearest_neighbors(self, position, k=5):
        if not self.positions:
            return [], []
            
        if not self.kdtree_valid:
            self.rebuild_kdtree()
            
        if self.kdtree is not None:
            distances, indices = self.kdtree.query(position, k=min(k, len(self.positions)))
            return distances, indices
        else:
            distances = [np.linalg.norm(position - p) for p in self.positions]
            indices = np.argsort(distances)[:k]
            return [distances[i] for i in indices], indices
    
    def compute_repulsion(self, frontiers):
        """Computing the effect of memory repulsion on the frontier"""
        if len(self.positions) == 0 or len(frontiers) == 0:
            return np.zeros(len(frontiers))

        frontiers_array = np.array(frontiers)
        positions_array = np.array(self.positions)
        importances = np.array(self.importance)

        # Calculate the distance matrix from all frontiers to all memory locations
        distances = np.sqrt(np.sum((frontiers_array[:, np.newaxis, :] - 
                                   positions_array[np.newaxis, :, :]) ** 2, axis=2))

        # Apply repulsive force calculation: Use Gaussian repulsion field
        repulsion = np.zeros(len(frontiers))
        for i in range(len(frontiers)):
            gaussian_repulsion = importances * np.exp(-distances[i]**2 / (2 * self.repulsion_radius**2))
            repulsion[i] = np.sum(gaussian_repulsion)

        if np.max(repulsion) > 0:
            repulsion = repulsion / np.max(repulsion) * self.repulsion_strength
        
        return repulsion
    
    def detect_cycle(self, threshold=None):
        if threshold is None:
            threshold = self.cycle_detection_window
            
        if len(self.positions) < threshold * 2:
            return False, {}
        
        if not hasattr(self, 'cycle_history'):
            self.cycle_history = []
        
        recent_trajectory = self.positions[-threshold:]
        max_similarity = 0.0
        best_segment = None
        
        for i in range(len(self.positions) - threshold * 2):
            earlier_segment = self.positions[i:i+threshold]
            similarity = self._calculate_trajectory_similarity(recent_trajectory, earlier_segment)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_segment = (i, i+threshold)
        
        base_cycle_detected = max_similarity > self.cycle_similarity_threshold
        
        cycle_detected = base_cycle_detected
        if base_cycle_detected:
            self.cycle_history.append(self.current_step)
            if len(self.cycle_history) > 5:
                self.cycle_history.pop(0)
            
            if len(self.cycle_history) >= 3:
                intervals = np.diff(self.cycle_history)
                interval_variance = np.var(intervals)

                # Low variance indicates a stable cycle
                cycle_confidence = 1.0 / (1.0 + interval_variance/100.0)
                
                # Update loop detection results
                cycle_detected = base_cycle_detected and (cycle_confidence > 0.5)
        
        # If a cycle is detected, update the significance of the trackpoints (existing code remains unchanged)
        if cycle_detected and best_segment:
            # Increased the importance of loop waypoints so they are more heavily weighted in repulsion calculations
            for i in range(best_segment[0], best_segment[1]):
                if i < len(self.importance):
                    self.importance[i] = min(2.0, self.importance[i] * 1.5)

            return True, {
                'similarity': max_similarity,
                'segment': best_segment,
                'recent_start': len(self.positions) - threshold,
                'confidence': cycle_confidence if 'cycle_confidence' in locals() else 1.0  # 添加置信度
            }
        
        return False, {}
    
    def _calculate_trajectory_similarity(self, trajectory1, trajectory2):
        """Calculate trajectory similarity"""
        if len(trajectory1) != len(trajectory2):
            return 0.0
        
        total_distances = 0.0
        
        for i in range(len(trajectory1)):
            distance = np.linalg.norm(trajectory1[i] - trajectory2[i])
            total_distances += distance

        avg_distance = total_distances / len(trajectory1)
        max_expected_distance = 5.0  # expected max distance
        
        similarity = 1.0 - min(1.0, avg_distance / max_expected_distance)
        return similarity
    
    def familiarity_score(self):
        """Calculate familiarity with the current area"""
        if not self.positions or self.last_position is None:
            return 0.0

        if not self.kdtree_valid:
            self.rebuild_kdtree()
        
        # Calculate the memory density near the current position
        if self.kdtree is not None:
            indices = self.kdtree.query_ball_point(self.last_position, 3.0)  # point in 3m
            
            # Calculate the average importance of these points
            if indices:
                density_factor = min(1.0, len(indices) / 20.0)  # A maximum of 20 points is considered complete familiarity
                
                # Calculating weighted importance
                total_importance = sum(self.importance[i] for i in indices)
                avg_importance = total_importance / len(indices)
                
                familiarity = 0.7 * density_factor + 0.3 * avg_importance
                return min(1.0, familiarity)
        
        return 0.0
    
    def get_memory_stats(self):
        stats = {
            'memory_size': len(self.positions),
            'unique_locations': len(self.positions),
            'avg_importance': np.mean(self.importance) if self.importance else 0.0,
            'current_step': self.current_step
        }
        return stats
    
    def reset(self):
        self.positions = []
        self.timestamps = []
        self.importance = []
        self.last_position = None
        self.current_step = 0
        self.kdtree = None
        self.kdtree_valid = False
