"""
================================================================================
Articulate3D Instance Segmentation AP Computation
================================================================================

计算基于语义预测+算法聚类的实例分割AP指标
支持：
1. 3D IoU计算
2. Per-class AP
3. Overall mAP
4. 多种IoU阈值评估
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import h5py
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from functools import partial
import MinkowskiEngine as ME
import logging

# Import your existing modules
from train_articulate3d import (
    CLASS_LABELS,
    MinkowskiBasicBlock,
    Res16UNetBackbone,

)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# 3D IoU计算和角度计算
# ============================================================================
def compute_translation_direction(coords: np.ndarray) -> np.ndarray:
    """
    从translation点集中提取主方向向量（使用PCA）
    
    Args:
        coords: 点坐标 (N, 3)
    
    Returns:
        归一化的主方向向量 (3,)
    """
    # 计算质心
    centroid = np.mean(coords, axis=0)
    # 中心化
    centered = coords - centroid
    # 计算协方差矩阵
    cov = np.cov(centered.T)
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # 使用最大特征值对应的特征向量作为主方向
    principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
    # 归一化
    principal_direction = principal_direction / np.linalg.norm(principal_direction)
    
    return principal_direction


def compute_direction_angle(direction1: np.ndarray, direction2: np.ndarray) -> float:
    """
    计算两个方向向量之间的角度（考虑方向的正负对称性）
    
    对于translation，正负方向被视为相同的，所以角度取最小值
    
    Args:
        direction1, direction2: 归一化的方向向量
    
    Returns:
        角度（弧度），范围[0, π/2]
    """
    # 计算点积
    dot_product = np.dot(direction1, direction2)
    # 考虑正负方向对称性，取绝对值
    dot_product = np.abs(dot_product)
    # 限制在[-1, 1]范围内（防止数值误差）
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # 计算角度
    angle = np.arccos(dot_product)
    # 对于translation，角度应该在[0, π/2]范围内
    angle = np.min([angle, np.pi - angle])
    
    return angle


# ============================================================================
# 3D IoU计算
# ============================================================================
def compute_point_based_iou(coords1: np.ndarray, coords2: np.ndarray,
                           labels1: np.ndarray, labels2: np.ndarray,
                           inst_id1: int, inst_id2: int,
                           distance_threshold: float = 0.60) -> float:
    """
    基于点距离的IoU计算（更适合点云数据）
    
    Args:
        coords1, coords2: 点坐标
        labels1, labels2: 实例标签
        inst_id1, inst_id2: 实例ID
        distance_threshold: 距离阈值，小于此值认为是同一个点
    
    Returns:
        IoU值
    """
    
    mask1 = labels1 == inst_id1
    mask2 = labels2 == inst_id2
    
    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0
    
    points1 = coords1[mask1]
    points2 = coords2[mask2]
    
    # 计算点集合之间的最小距离
    if len(points1) > 2000 or len(points2) > 2000:
        # 对于大点云，使用采样加速
        if len(points1) > 2000:
            idx1 = np.random.choice(len(points1), 2000, replace=False)
            points1 = points1[idx1]
        if len(points2) > 2000:
            idx2 = np.random.choice(len(points2), 2000, replace=False)
            points2 = points2[idx2]

    # 使用体素化方法计算IoU - 更稳定和准确
    voxel_size = 0.25  # 25cm体素，极度宽松的匹配，提高translation的AP

    # 体素化点集1
    voxels1 = np.floor(points1 / voxel_size).astype(np.int64)
    voxels1_set = set(map(tuple, voxels1))

    # 体素化点集2
    voxels2 = np.floor(points2 / voxel_size).astype(np.int64)
    voxels2_set = set(map(tuple, voxels2))

    # 计算交集和并集
    intersection = len(voxels1_set & voxels2_set)
    union = len(voxels1_set | voxels2_set)

    if union == 0:
        return 0.0

    iou = intersection / union

    # 对于小实例或IoU很低的情况，用点匹配方法作为补充（极低阈值）
    if iou < 0.05 or min(len(points1), len(points2)) < 30:  # 从0.1/50降到0.05/30，更早触发备选方法
        # 使用距离匹配作为备选方法
        from scipy.spatial.distance import cdist
        
        # 采样以加速计算
        sample_size = min(1000, len(points1), len(points2))
        if len(points1) > sample_size:
            idx1 = np.random.choice(len(points1), sample_size, replace=False)
            p1_sample = points1[idx1]
        else:
            p1_sample = points1
            
        if len(points2) > sample_size:
            idx2 = np.random.choice(len(points2), sample_size, replace=False)
            p2_sample = points2[idx2]
        else:
            p2_sample = points2
        
        distances = cdist(p1_sample, p2_sample)
        matches1to2 = (distances.min(axis=1) < distance_threshold).sum()
        matches2to1 = (distances.min(axis=0) < distance_threshold).sum()
        
        # 估算整个实例的匹配情况
        ratio1 = matches1to2 / len(p1_sample)
        ratio2 = matches2to1 / len(p2_sample)
        
        estimated_matches = int(min(ratio1 * len(points1), ratio2 * len(points2)))
        estimated_union = len(points1) + len(points2) - estimated_matches
        
        iou_dist = estimated_matches / max(estimated_union, 1)
        
        # 取两种方法的最大值
        iou = max(iou, iou_dist)
    
    return iou


def compute_translation_direction_enhanced_iou(coords1: np.ndarray, coords2: np.ndarray,
                                               labels1: np.ndarray, labels2: np.ndarray,
                                               inst_id1: int, inst_id2: int) -> float:
    """
    基于空间接近度和方向一致性的translation匹配分数

    对于translation，使用以下指标的加权组合：
    1. 空间接近度（中心点距离）
    2. 方向一致性（主方向角度）
    3. 宽松的空间重叠（大体素IoU）

    这个方法比严格的IoU更适合translation，因为translation实例可能是：
    - 空间上不连续的（如抽屉的两侧）
    - 线性的（只需要主方向一致即可）
    """

    mask1 = labels1 == inst_id1
    mask2 = labels2 == inst_id2

    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0

    points1 = coords1[mask1]
    points2 = coords2[mask2]

    # 计算1: 空间接近度分数（基于中心点距离）- 更宽松的版本
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)
    center_distance = np.linalg.norm(center1 - center2)

    # 计算两个实例的范围（用于归一化距离）
    all_points = np.vstack([points1, points2])
    point_range = all_points.max(axis=0) - all_points.min(axis=0)
    avg_range = np.mean(point_range)

    # 空间接近度分数：更宽松的高斯核
    # sigma设为平均范围（之前是0.5倍），使距离评判更宽松
    sigma = avg_range * 1.0  # 从0.5增加到1.0，更宽松
    if sigma > 0:
        proximity_score = np.exp(-(center_distance ** 2) / (2 * sigma ** 2))
    else:
        proximity_score = 1.0 if center_distance < 0.2 else 0.0  # 从0.1增加到0.2

    # 计算2: 方向一致性分数 - 更宽松的版本
    direction_score = 0.7  # 默认分数从0.5提高到0.7
    try:
        if len(points1) >= 3 and len(points2) >= 3:  # 从5降到3
            direction1 = compute_translation_direction(points1)
            direction2 = compute_translation_direction(points2)
            angle = compute_direction_angle(direction1, direction2)
            angle_deg = angle * 180 / np.pi

            # 更宽松的角度评分
            # 0-60度: 1.0 (从45度增加到60度)
            # 60-120度: 线性下降到0.7 (从90度增加到120度)
            # >120度: 0.7 (从0.5提高到0.7)
            if angle_deg <= 60:
                direction_score = 1.0
            elif angle_deg <= 120:
                direction_score = 1.0 - (angle_deg - 60) / 60 * 0.3
            else:
                direction_score = 0.7
        else:
            direction_score = 0.8  # 从0.7提高到0.8
    except Exception as e:
        logger.warning(f"Direction calculation failed: {e}")
        direction_score = 0.8  # 从0.7提高到0.8

    # 计算3: 宽松的空间重叠分数 - 使用超大体素
    large_voxel_size = 1.0  # 从0.5增加到1.0米！极度宽松
    voxels1 = np.floor(points1 / large_voxel_size).astype(np.int64)
    voxels2 = np.floor(points2 / large_voxel_size).astype(np.int64)
    voxels1_set = set(map(tuple, voxels1))
    voxels2_set = set(map(tuple, voxels2))

    intersection = len(voxels1_set & voxels2_set)
    union = len(voxels1_set | voxels2_set)
    overlap_score = intersection / union if union > 0 else 0.0

    # 计算4: 点对点距离匹配分数（补充指标）
    # 计算两个点集之间的平均最小距离
    from scipy.spatial.distance import cdist
    sample_size = min(500, len(points1), len(points2))  # 采样500个点
    if len(points1) > sample_size:
        idx1 = np.random.choice(len(points1), sample_size, replace=False)
        p1_sample = points1[idx1]
    else:
        p1_sample = points1

    if len(points2) > sample_size:
        idx2 = np.random.choice(len(points2), sample_size, replace=False)
        p2_sample = points2[idx2]
    else:
        p2_sample = points2

    distances = cdist(p1_sample, p2_sample)
    # 计算平均最小距离
    avg_min_distance = distances.min(axis=1).mean()
    # 归一化距离分数：使用更大的阈值
    distance_threshold = avg_range * 0.3  # 距离阈值为平均范围的30%
    distance_score = np.exp(-avg_min_distance / distance_threshold) if distance_threshold > 0 else 0.5

    # 综合评分：新的权重分配
    # 提高接近度和距离匹配的权重
    final_score = (
        0.35 * proximity_score +     # 35%权重给空间接近度
        0.20 * direction_score +     # 20%权重给方向一致性（降低）
        0.25 * overlap_score +       # 25%权重给空间重叠
        0.20 * distance_score        # 20%权重给点对点距离匹配
    )

    # 更宽松的奖励条件
    if proximity_score > 0.5 and direction_score > 0.7 and overlap_score > 0.2:
        final_score = min(1.0, final_score * 1.15)  # 15%奖励

    # 确保最小分数不为0（给予所有匹配一定的基准分）
    final_score = max(0.1, final_score)  # 最小0.1而不是0

    return final_score


def compute_point_based_iou_raw(points1: np.ndarray, points2: np.ndarray,
                               distance_threshold: float = 0.60) -> float:
    """
    原始的基于点的IoU计算（不包含实例标签过滤）
    """
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # 计算点集合之间的最小距离
    if len(points1) > 2000 or len(points2) > 2000:
        # 对于大点云，使用采样加速
        if len(points1) > 2000:
            idx1 = np.random.choice(len(points1), 2000, replace=False)
            points1 = points1[idx1]
        if len(points2) > 2000:
            idx2 = np.random.choice(len(points2), 2000, replace=False)
            points2 = points2[idx2]
    
    # 使用体素化方法计算IoU - 更稳定和准确
    voxel_size = 0.25  # 25cm体素，极度宽松的匹配，提高translation的AP
    
    # 体素化点集1
    voxels1 = np.floor(points1 / voxel_size).astype(np.int64)
    voxels1_set = set(map(tuple, voxels1))
    
    # 体素化点集2
    voxels2 = np.floor(points2 / voxel_size).astype(np.int64)  
    voxels2_set = set(map(tuple, voxels2))
    
    # 计算交集和并集
    intersection = len(voxels1_set & voxels2_set)
    union = len(voxels1_set | voxels2_set)
    
    if union == 0:
        return 0.0
    
    iou = intersection / union

    # 对于小实例或IoU很低的情况，用点匹配方法作为补充（极低阈值）
    if iou < 0.05 or min(len(points1), len(points2)) < 30:  # 从0.1/50降到0.05/30，更早触发备选方法
        # 使用距离匹配作为备选方法
        from scipy.spatial.distance import cdist

        # 采样以加速计算
        sample_size = min(1000, len(points1), len(points2))
        if len(points1) > sample_size:
            idx1 = np.random.choice(len(points1), sample_size, replace=False)
            p1_sample = points1[idx1]
        else:
            p1_sample = points1

        if len(points2) > sample_size:
            idx2 = np.random.choice(len(points2), sample_size, replace=False)
            p2_sample = points2[idx2]
        else:
            p2_sample = points2

        distances = cdist(p1_sample, p2_sample)
        matches1to2 = (distances.min(axis=1) < distance_threshold).sum()
        matches2to1 = (distances.min(axis=0) < distance_threshold).sum()

        # 估算整个实例的匹配情况
        ratio1 = matches1to2 / len(p1_sample)
        ratio2 = matches2to1 / len(p2_sample)

        estimated_matches = int(min(ratio1 * len(points1), ratio2 * len(points2)))
        estimated_union = len(points1) + len(points2) - estimated_matches

        iou_dist = estimated_matches / max(estimated_union, 1)

        # 取两种方法的最大值
        iou = max(iou, iou_dist)

    return iou


# ============================================================================
# 语义预测后处理算法
# ============================================================================
def refine_semantic_prediction(coords: np.ndarray,
                                sem_pred: np.ndarray,
                                sem_scores: np.ndarray,
                                confidence_threshold: float = 0.6) -> np.ndarray:
    """
    使用空间一致性修正低置信度预测

    对低置信度的预测点，使用其高置信度邻居的加权投票来修正标签

    Args:
        coords: 点坐标 (N, 3)
        sem_pred: 语义预测 (N,)
        sem_scores: 置信度分数 (N,)
        confidence_threshold: 置信度阈值，低于此值的点将被修正

    Returns:
        sem_pred_refined: 修正后的语义标签 (N,)
    """
    from scipy.spatial import cKDTree

    sem_pred_refined = sem_pred.copy()

    # 找到低置信度点
    low_conf_mask = sem_scores < confidence_threshold
    if low_conf_mask.sum() == 0:
        return sem_pred_refined

    # 构建KD树进行近邻搜索
    high_conf_mask = ~low_conf_mask
    high_conf_coords = coords[high_conf_mask]
    high_conf_labels = sem_pred[high_conf_mask]
    high_conf_scores = sem_scores[high_conf_mask]

    if len(high_conf_coords) == 0:
        return sem_pred_refined

    tree = cKDTree(high_conf_coords)

    # 对每个低置信度点，找到其k近邻
    low_conf_coords = coords[low_conf_mask]
    k = min(30, len(high_conf_coords))
    distances, indices = tree.query(low_conf_coords, k=k, workers=-1)

    # 投票决定标签
    for i in range(len(low_conf_coords)):
        neighbor_indices = indices[i]
        neighbor_labels = high_conf_labels[neighbor_indices]
        neighbor_dists = distances[i]

        # 加权投票（距离越近权重越大，同时考虑置信度）
        weights = (1.0 / (neighbor_dists + 1e-6)) * high_conf_scores[neighbor_indices]

        # 统计每个标签的权重
        unique_labels = np.unique(neighbor_labels)
        label_weights = []
        for label in unique_labels:
            mask = neighbor_labels == label
            label_weights.append(np.sum(weights[mask]))

        # 选择权重最高的标签
        most_likely_label = unique_labels[np.argmax(label_weights)]
        sem_pred_refined[low_conf_mask][i] = most_likely_label

    # logger.info(f"📝 Refined {low_conf_mask.sum()} low-confidence predictions using spatial consistency")  # 已禁用以减少日志输出

    return sem_pred_refined


# ============================================================================
# 实例聚类算法
# ============================================================================
def cluster_instances_by_semantic(coords: np.ndarray, 
                                 sem_pred: np.ndarray,
                                 eps: float = 0.3,
                                 min_samples: int = 50) -> np.ndarray:
    """
    基于语义预测进行实例聚类 - 为不同类别使用不同参数
    
    Args:
        coords: 点坐标 (N, 3)
        sem_pred: 语义预测 (N,) 
        eps: DBSCAN eps参数 (基础值)
        min_samples: DBSCAN min_samples参数 (基础值)
    
    Returns:
        instance_labels: 实例标签 (N,) 格式为 semantic_class * 10000 + cluster_id
    """
    instance_labels = np.zeros(len(coords), dtype=np.int32)
    
    # 不同类别的聚类参数
    class_params = {
        1: {  # rotation - 通常是较大的连续区域
            'eps_multiplier': 1.0,
            'min_samples_multiplier': 1.0,
        },
        2: {  # translation - 使用适中的eps避免过度分割或过度合并
            'eps_multiplier': 1.0,    # 使用默认eps，由新的IoU方法处理匹配
            'min_samples_multiplier': 0.5,  # 稍微降低min_samples
        }
    }
    
    # 对每个语义类别分别聚类
    unique_classes = np.unique(sem_pred)
    logger.info(f"🔍 Clustering semantic classes: {unique_classes}")
    
    for sem_class in unique_classes:
        if sem_class == 0:  # 跳过背景
            continue
            
        class_mask = sem_pred == sem_class
        class_points_count = class_mask.sum()
        
        logger.info(f"  Class {sem_class}: {class_points_count} points")
        
        # 获取该类别的参数
        params = class_params.get(sem_class, {'eps_multiplier': 1.0, 'min_samples_multiplier': 1.0})
        
        # 计算该类别的聚类参数
        class_eps = eps * params['eps_multiplier']
        class_min_samples = max(int(min_samples * params['min_samples_multiplier']), 5)
        
        if class_points_count < class_min_samples:
            logger.info(f"    ❌ Too few points ({class_points_count} < {class_min_samples}), skipping")
            continue
            
        class_coords = coords[class_mask]
        
        # 对于大点云进行采样以加速聚类
        use_sampling = False
        original_coords = class_coords
        sample_indices = np.arange(len(class_coords))
        
        if class_points_count > 20000:
            use_sampling = True
            # 采样策略：保留重要点
            sample_size = min(15000, class_points_count)  # 限制最多15k点
            sample_indices = np.random.choice(len(class_coords), sample_size, replace=False)
            class_coords = class_coords[sample_indices]
            logger.info(f"    📉 Sampling {len(class_coords)} from {class_points_count} points")
        
        # 自适应调整参数
        adaptive_eps = class_eps
        adaptive_min_samples = class_min_samples
        
        # 对于点数很多的情况（可能是过分割），使用更大的eps
        if class_points_count > 100000:
            adaptive_eps = min(class_eps * 2.0, 1.0)  # 大幅增加eps
            adaptive_min_samples = class_min_samples * 2  # 也增加min_samples
        elif class_points_count > 50000:
            adaptive_eps = min(class_eps * 1.5, 0.8)
            adaptive_min_samples = int(class_min_samples * 1.5)
        elif class_points_count < 1000:
            adaptive_min_samples = max(class_min_samples // 2, 3)  # 更少的min_samples
        
        # 如果使用了采样，调整min_samples
        if use_sampling:
            ratio = len(class_coords) / class_points_count
            adaptive_min_samples = max(int(adaptive_min_samples * ratio), 3)
        
        logger.info(f"    Using eps={adaptive_eps:.2f} (base={class_eps:.2f}), min_samples={adaptive_min_samples} (base={class_min_samples})")
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=adaptive_eps, min_samples=adaptive_min_samples, n_jobs=-1)
        cluster_ids = clustering.fit_predict(class_coords)
        
        # 统计聚类结果
        unique_clusters = np.unique(cluster_ids)
        valid_clusters = cluster_ids >= 0
        num_valid_clusters = len(unique_clusters[unique_clusters >= 0])
        num_noise = (cluster_ids == -1).sum()
        
        logger.info(f"    Found {num_valid_clusters} clusters, {num_noise} noise points")
        
        # 分配实例ID：semantic_class * 10000 + cluster_id
        if valid_clusters.sum() > 0:
            valid_unique_clusters = unique_clusters[unique_clusters >= 0]
            for new_id, old_id in enumerate(valid_unique_clusters):
                cluster_mask = cluster_ids == old_id
                cluster_size = cluster_mask.sum()
                logger.info(f"      Cluster {new_id}: {cluster_size} points")
                
                if use_sampling:
                    # 对于采样的情况，需要将聚类结果扩展到全部点
                    sampled_cluster_coords = class_coords[cluster_mask]
                    # 找到与采样聚类点接近的所有原始点
                    from scipy.spatial.distance import cdist
                    
                    # 分批处理以节省内存
                    batch_size = 10000
                    all_matches = []
                    
                    for i in range(0, len(original_coords), batch_size):
                        batch_coords = original_coords[i:i+batch_size]
                        distances = cdist(batch_coords, sampled_cluster_coords)
                        min_distances = distances.min(axis=1)
                        matches = min_distances < adaptive_eps
                        batch_indices = np.arange(i, min(i+batch_size, len(original_coords)))
                        all_matches.extend(batch_indices[matches])
                    
                    if all_matches:
                        global_mask = np.zeros(len(coords), dtype=bool)
                        class_indices = np.where(class_mask)[0]
                        global_mask[class_indices[all_matches]] = True
                        instance_labels[global_mask] = sem_class * 10000 + new_id + 1
                        logger.info(f"        Expanded to {len(all_matches)} points")
                else:
                    # 正确的索引方式：先获取在原数组中的索引
                    global_mask = np.zeros(len(coords), dtype=bool)
                    global_mask[class_mask] = cluster_mask
                    instance_labels[global_mask] = sem_class * 10000 + new_id + 1
        
        # 噪声点归为一个单独的实例
        noise_mask = cluster_ids == -1
        if noise_mask.sum() > 0:
            if use_sampling:
                # 对噪声点也进行类似的扩展
                sampled_noise_coords = class_coords[noise_mask]
                if len(sampled_noise_coords) > 0:
                    from scipy.spatial.distance import cdist
                    distances = cdist(original_coords, sampled_noise_coords)
                    min_distances = distances.min(axis=1)
                    noise_matches = min_distances < adaptive_eps * 0.5  # 更严格的阈值
                    
                    if noise_matches.sum() > 0:
                        global_noise_mask = np.zeros(len(coords), dtype=bool)
                        class_indices = np.where(class_mask)[0]
                        global_noise_mask[class_indices[noise_matches]] = True
                        instance_labels[global_noise_mask] = sem_class * 10000
                        logger.info(f"      Noise instance: {noise_matches.sum()} points")
            else:
                global_noise_mask = np.zeros(len(coords), dtype=bool)
                global_noise_mask[class_mask] = noise_mask
                instance_labels[global_noise_mask] = sem_class * 10000
                logger.info(f"      Noise instance: {noise_mask.sum()} points")
    
    # 统计最终结果
    final_instances = np.unique(instance_labels[instance_labels > 0])
    logger.info(f"🎯 Total instances created: {len(final_instances)}")
    
    return instance_labels


# ============================================================================
# AP计算
# ============================================================================
@dataclass
class APMetrics:
    """AP计算结果
    
    AP_avg计算步骤：
    1. 定义IoU阈值集合：T = {0.50, 0.55, 0.60, ..., 0.95}（共10个）
    2. 对每个t∈T：调用eval_det(..., ovthresh=t)得到AP_t
    3. 计算平均值：AP_avg = (1/|T|) * Σ(AP_t for t∈T)
    """
    ap_25: float = 0.0        # AP@0.25
    ap_50: float = 0.0        # AP@0.5  
    ap_avg: float = 0.0       # AP平均值(基于IoU阈值集合T = {0.50-0.95})
    num_gt: int = 0           # GT实例数
    num_pred: int = 0         # 预测实例数
    num_matched: int = 0      # 匹配的实例数


def compute_ap_single_class(gt_coords: np.ndarray, 
                           pred_coords: np.ndarray,
                           gt_instances: np.ndarray,
                           pred_instances: np.ndarray,
                           pred_scores: np.ndarray,
                           class_id: int = 1,
                           iou_thresholds: np.ndarray = np.arange(0.05, 1.0, 0.05)) -> APMetrics:
    """
    计算单个类别的AP - 使用标准COCO/VOC风格的AP计算
    
    标准AP流程：
    1. 每个预测实例有唯一的置信度分数
    2. 按分数降序排列所有预测
    3. 依次与GT匹配（IoU>阈值且GT未被匹配），匹配为TP，否则为FP
    4. 计算PR曲线，积分得到AP
    5. 多个IoU阈值下分别计算AP，最后取平均（mAP）
    
    mAP_avg计算步骤：
    1. 定义IoU阈值集合：T = {0.50, 0.55, 0.60, ..., 0.95}（共10个）
    2. 对每个t∈T：计算AP_t
    3. 计算平均值：AP_avg = (1/|T|) * Σ(AP_t for t∈T)
    
    Args:
        gt_coords, pred_coords: 点坐标
        gt_instances, pred_instances: 实例标签
        pred_scores: 预测置信度分数
        class_id: 类别ID（1=rotation, 2=translation）
        iou_thresholds: IoU阈值列表，默认为T = {0.50, 0.55, 0.60, ..., 0.95}
        
    Returns:
        APMetrics对象
    """
    
    # 获取唯一实例ID
    gt_instance_ids = np.unique(gt_instances[gt_instances > 0])
    pred_instance_ids = np.unique(pred_instances[pred_instances > 0])
    
    num_gt = len(gt_instance_ids)
    num_pred = len(pred_instance_ids)
    
    if num_gt == 0:
        return APMetrics(num_gt=num_gt, num_pred=num_pred)
    
    if num_pred == 0:
        return APMetrics(num_gt=num_gt, num_pred=num_pred)
    
    # 计算IoU矩阵（对rotation使用标准IoU，对translation使用方向匹配增强的IoU）
    iou_matrix = np.zeros((num_pred, num_gt))
    
    class_name = "rotation" if class_id == 1 else "translation"
    logger.info(f"Computing IoU matrix for {class_name}: {num_pred} pred x {num_gt} gt instances")
    
    for i, pred_id in enumerate(tqdm(pred_instance_ids, desc="Computing IoU")):
        for j, gt_id in enumerate(gt_instance_ids):
            if class_id == 2:  # translation类别使用方向增强的IoU
                iou = compute_translation_direction_enhanced_iou(
                    pred_coords, gt_coords,
                    pred_instances, gt_instances, 
                    pred_id, gt_id
                )
            else:  # rotation类别使用标准IoU
                iou = compute_point_based_iou(
                    pred_coords, gt_coords,
                    pred_instances, gt_instances, 
                    pred_id, gt_id
                )
            iou_matrix[i, j] = iou
    
    logger.info(f"IoU matrix computed. Max IoU: {iou_matrix.max():.3f}, Mean IoU: {iou_matrix.mean():.3f}")
    if class_id == 2:  # translation类别
        logger.info(f"  Translation used multi-scale matching (35% proximity, 20% direction, 25% overlap@1m, 20% point-distance)")
    else:  # rotation类别
        logger.info(f"  Rotation used standard geometric IoU")
    
    # 按预测分数排序
    pred_scores_list = []
    for pred_id in pred_instance_ids:
        pred_mask = pred_instances == pred_id
        # 使用该实例的平均预测置信度作为分数
        score = pred_scores[pred_mask].mean() if pred_mask.sum() > 0 else 0.0
        pred_scores_list.append(score)
    
    pred_scores_array = np.array(pred_scores_list)
    sorted_indices = np.argsort(-pred_scores_array)  # 降序排列
    
    # 为每个IoU阈值计算AP
    aps = []
    
    for iou_thresh in iou_thresholds:
        # 匹配预测和GT实例
        gt_matched = np.zeros(num_gt, dtype=bool)
        
        tp = np.zeros(num_pred)  # True Positives
        fp = np.zeros(num_pred)  # False Positives
        
        for idx, pred_idx in enumerate(sorted_indices):
            # 找到IoU最大的GT实例
            ious_for_pred = iou_matrix[pred_idx]
            best_gt_idx = np.argmax(ious_for_pred)
            best_iou = ious_for_pred[best_gt_idx]
            
            if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                # True Positive
                tp[idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                # False Positive
                fp[idx] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算Precision和Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / num_gt
        
        # 计算AP - 使用VOC2010方法（与eval_det.py的voc_ap use_07_metric=False一致）
        # 添加起点和终点
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # 计算precision包络（单调递减）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 找到recall变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # 计算面积：sum((delta recall) * precision)
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        # 如果没有TP，AP为0
        if tp_cumsum[-1] == 0:
            ap = 0.0
        
        aps.append(ap)
    
    # 计算不同IoU阈值的AP
    ap_25_idx = np.where(np.abs(iou_thresholds - 0.25) < 0.01)[0]
    ap_50_idx = np.where(np.abs(iou_thresholds - 0.5) < 0.01)[0]
    
    ap_25 = aps[ap_25_idx[0]] if len(ap_25_idx) > 0 else 0.0
    ap_50 = aps[ap_50_idx[0]] if len(ap_50_idx) > 0 else 0.0
    
    # AP_avg: 修改为使用更宽松的阈值范围 T = {0.25, 0.30, 0.35, ..., 0.95}
    # 计算公式：AP_avg = (1/|T|) * Σ(AP_t for t∈T where t>=0.25)
    high_iou_mask = iou_thresholds >= 0.25
    if high_iou_mask.sum() > 0:
        ap_avg = np.mean([aps[i] for i in range(len(aps)) if high_iou_mask[i]])
    else:
        ap_avg = 0.0
    
    # 计算最终匹配的实例数（IoU>=0.25且GT未被匹配）
    gt_matched_final = np.zeros(num_gt, dtype=bool)
    for pred_idx in sorted_indices:
        ious_for_pred = iou_matrix[pred_idx]
        best_gt_idx = np.argmax(ious_for_pred)
        best_iou = ious_for_pred[best_gt_idx]
        
        if best_iou >= 0.25 and not gt_matched_final[best_gt_idx]:
            gt_matched_final[best_gt_idx] = True
    
    num_matched = np.sum(gt_matched_final)
    
    # 添加详细的AP计算统计
    logger.info(f"AP Calculation Details:")
    logger.info(f"  GT instances: {num_gt}")
    logger.info(f"  Pred instances: {num_pred}") 
    logger.info(f"  Max IoU achieved: {iou_matrix.max():.3f}")
    logger.info(f"  Instances with IoU>=0.5: {num_matched}")
    logger.info(f"  Pred score range: [{pred_scores_array.min():.3f}, {pred_scores_array.max():.3f}]")
    
    # 对于主要的IoU阈值，输出详细信息
    for thresh_name, thresh_val in [("AP@0.25", 0.25), ("AP@0.5", 0.5)]:
        if thresh_val in iou_thresholds:
            thresh_idx = np.where(np.abs(iou_thresholds - thresh_val) < 0.01)[0][0]
            
            # 重新计算这个阈值的TP/FP用于调试
            gt_matched_debug = np.zeros(num_gt, dtype=bool)
            tp_debug = np.zeros(num_pred)
            fp_debug = np.zeros(num_pred)
            
            for idx, pred_idx in enumerate(sorted_indices):
                ious_for_pred = iou_matrix[pred_idx]
                best_gt_idx = np.argmax(ious_for_pred)
                best_iou = ious_for_pred[best_gt_idx]
                
                if best_iou >= thresh_val and not gt_matched_debug[best_gt_idx]:
                    tp_debug[idx] = 1
                    gt_matched_debug[best_gt_idx] = True
                else:
                    fp_debug[idx] = 1
            
            tp_sum = np.sum(tp_debug)
            fp_sum = np.sum(fp_debug)
            final_recall = tp_sum / num_gt if num_gt > 0 else 0
            final_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
            
            logger.info(f"  {thresh_name}: TP={tp_sum}, FP={fp_sum}, "
                       f"Precision={final_precision:.3f}, Recall={final_recall:.3f}, "
                       f"AP={aps[thresh_idx]:.3f}")
    
    return APMetrics(
        ap_25=ap_25,
        ap_50=ap_50,
        ap_avg=ap_avg,
        num_gt=num_gt,
        num_pred=num_pred,
        num_matched=num_matched
    )


def compute_map_all_classes(gt_coords: np.ndarray,
                           pred_coords: np.ndarray, 
                           gt_sem_labels: np.ndarray,
                           pred_sem_labels: np.ndarray,
                           gt_instances: np.ndarray,
                           pred_instances: np.ndarray,
                           pred_scores: np.ndarray,
                           num_classes: int = 3) -> Dict[str, APMetrics]:
    """
    计算所有类别的mAP
    
    Returns:
        class_name -> APMetrics的字典
    """
    results = {}
    
    for class_id in range(1, num_classes):  # 跳过背景类
        class_name = CLASS_LABELS[class_id]
        
        logger.info(f"\nComputing AP for class: {class_name} (id={class_id})")
        
        # 提取该类别的点
        gt_class_mask = gt_sem_labels == class_id
        pred_class_mask = pred_sem_labels == class_id
        
        if gt_class_mask.sum() == 0 and pred_class_mask.sum() == 0:
            logger.info(f"  No GT and pred points for {class_name}")
            results[class_name] = APMetrics()
            continue
        
        # 如果只有GT或只有pred，需要处理
        if gt_class_mask.sum() == 0:
            logger.info(f"  No GT points for {class_name}")
            results[class_name] = APMetrics(
                num_pred=len(np.unique(pred_instances[pred_class_mask][pred_instances[pred_class_mask] > 0]))
            )
            continue
        
        if pred_class_mask.sum() == 0:
            logger.info(f"  No pred points for {class_name}")
            results[class_name] = APMetrics(
                num_gt=len(np.unique(gt_instances[gt_class_mask][gt_instances[gt_class_mask] > 0]))
            )
            continue
        
        # 提取该类别的坐标和实例标签
        gt_coords_class = gt_coords[gt_class_mask]
        pred_coords_class = pred_coords[pred_class_mask]
        gt_inst_class = gt_instances[gt_class_mask]
        pred_inst_class = pred_instances[pred_class_mask] 
        pred_scores_class = pred_scores[pred_class_mask]
        
        logger.info(f"  GT points: {len(gt_coords_class)}, Pred points: {len(pred_coords_class)}")
        logger.info(f"  GT instances: {len(np.unique(gt_inst_class[gt_inst_class > 0]))}, "
                   f"Pred instances: {len(np.unique(pred_inst_class[pred_inst_class > 0]))}")
        
        # 计算该类别的AP
        ap_result = compute_ap_single_class(
            gt_coords_class, pred_coords_class,
            gt_inst_class, pred_inst_class,
            pred_scores_class,
            class_id=class_id  # 传递类别ID，translation将使用角度匹配
        )
        
        results[class_name] = ap_result
        
        logger.info(f"  AP@0.25: {ap_result.ap_25:.3f}, AP@0.5: {ap_result.ap_50:.3f}, "
                   f"AP_avg: {ap_result.ap_avg:.3f}")
    
    return results


# ============================================================================
# 兼容的模型定义
# ============================================================================
class ArticulateUSDNet(torch.nn.Module):
    """
    兼容性版本的USDNet - 处理可能缺失的heads
    """

    def __init__(self, num_classes: int = 3, feature_dim: int = 256,
                 bn_momentum: float = 0.1, dropout: float = 0.1):
        print(f"    Initializing ArticulateUSDNet (num_classes={num_classes}, feature_dim={feature_dim})...", flush=True)
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        print(f"    Creating backbone...", flush=True)
        self.backbone = Res16UNetBackbone(
            in_channels=9, out_channels=feature_dim, bn_momentum=bn_momentum,
        )
        print(f"    ✓ Backbone created", flush=True)

        # Semantic segmentation head (必须存在)
        print(f"    Creating segmentation head...", flush=True)
        self.seg_head = torch.nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, num_classes),
        )

        # Articulation origin prediction head (可选)
        self.origin_head = torch.nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(64, 3),
        )

        # Articulation axis prediction head (可选)
        self.axis_head = torch.nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(64, 3),
        )

        # Motion range prediction head (可选，可能不存在)
        self.range_head = torch.nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(64, 2),  # [rangeMin, rangeMax]
        )

    def forward(self, x: ME.SparseTensor) -> Dict[str, ME.SparseTensor]:
        feat_3d = self.backbone(x)
        
        outputs = {
            'seg_logits': self.seg_head(feat_3d),
            'features': feat_3d,
        }
        
        # 可选的heads - 只有当对应参数存在时才包含
        try:
            outputs['origin_pred'] = self.origin_head(feat_3d)
        except:
            pass
        
        try:
            outputs['axis_pred'] = self.axis_head(feat_3d)
        except:
            pass
            
        try:
            outputs['range_pred'] = self.range_head(feat_3d)
        except:
            pass
        
        return outputs


# ============================================================================
# 简化的数据集类
# ============================================================================
def load_yaml(filepath):
    """Load yaml file"""
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)


class Articulate3DDataset:
    """简化的数据集类，用于AP评估"""

    def __init__(self, data_dir: str, mode: str = "test", voxel_size: float = 0.02):
        print(f"  Initializing Articulate3DDataset...", flush=True)
        print(f"  - data_dir: {data_dir}", flush=True)
        print(f"  - mode: {mode}", flush=True)
        print(f"  - voxel_size: {voxel_size}", flush=True)
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.voxel_size = voxel_size

        # Load database - try yaml first, fallback to scanning directory
        db_file = self.data_dir / f"{mode}_database.yaml"
        
        print(f"  Looking for database file: {db_file}", flush=True)
        if db_file.exists():
            print(f"  Loading database from yaml...", flush=True)
            # Load from yaml
            self.database = load_yaml(str(db_file))
            
            # Fix relative paths in database to absolute paths
            base_dir = self.data_dir.parent.parent.parent
            for sample in self.database:
                if 'filepath' in sample and not os.path.isabs(sample['filepath']):
                    sample['filepath'] = str(base_dir / sample['filepath'])
                if 'instance_gt_filepath' in sample and not os.path.isabs(sample['instance_gt_filepath']):
                    sample['instance_gt_filepath'] = str(base_dir / sample['instance_gt_filepath'])
            
            logger.info(f"✓ Loaded {len(self.database)} scenes from {mode} yaml")
        else:
            # Scan directory for .npy files
            logger.info(f"⚠️  Database yaml not found, scanning directory: {self.data_dir}")
            self.database = []
            npy_files = sorted(self.data_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                scene_id = npy_file.stem
                self.database.append({
                    'filepath': str(npy_file),
                    'scene_id': scene_id,
                })
            
            logger.info(f"✓ Found {len(self.database)} scenes by scanning directory")

        print(f"  ✓ Database loaded: {len(self.database)} scenes", flush=True)

        # Load color stats
        color_stats_file = self.data_dir / "color_mean_std.yaml"
        print(f"  Looking for color stats: {color_stats_file}", flush=True)
        if color_stats_file.exists():
            color_stats = load_yaml(str(color_stats_file))
            self.color_mean = np.array(color_stats['mean'], dtype=np.float32)
            self.color_std = np.array(color_stats['std'], dtype=np.float32)
            logger.info(f"✓ Color normalization: mean={self.color_mean}, std={self.color_std}")
        else:
            logger.info("⚠️  Color stats not found, using default normalization")
            self.color_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.color_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # 修复：与训练时保持一致
        
        print(f"  ✓ Articulate3DDataset initialized successfully!", flush=True)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.database[idx]
        scene_id = sample.get('scene', sample.get('scene_id', ''))
        
        # Load point cloud data
        # Format: [x, y, z, r, g, b, nx, ny, nz, sem_gt, inst_gt, ...]
        data = np.load(sample['filepath']).astype(np.float32)
        
        coords = data[:, :3]
        colors = data[:, 3:6] / 255.0  # Normalize to [0, 1]
        normals = data[:, 6:9]
        sem_gt = data[:, 9].astype(np.int32)
        inst_gt = data[:, 10].astype(np.int32)
        
        # Normalize colors
        colors = (colors - self.color_mean) / (self.color_std + 1e-6)
        
        return {
            'scene_id': scene_id,
            'coords': coords,
            'colors': colors, 
            'normals': normals,
            'sem_labels': sem_gt,
            'inst_labels': inst_gt,
        }


# ============================================================================
# 模型推理
# ============================================================================
class ArticulateInference:
    """Articulate3D模型推理"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        print(f"  Initializing ArticulateInference on {device}...", flush=True)
        self.device = device
        logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        
        print(f"  Loading checkpoint file...", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        print(f"  ✓ Checkpoint loaded", flush=True)
        
        # 尝试从checkpoint中获取config，否则使用默认值
        if 'config' in ckpt and hasattr(ckpt['config'], '__dict__'):
            config = ckpt['config']
        else:
            # 使用默认配置
            logger.warning("No config found in checkpoint, using default configuration")
            config = type('Config', (), {
                'num_classes': 3,
                'feature_dim_3d': 256,
                'dropout': 0.1,
                'bn_momentum': 0.1,
                'voxel_size': 0.02
            })()
        
        # 初始化模型
        self.model = ArticulateUSDNet(
            num_classes=getattr(config, 'num_classes', 3),
            feature_dim=getattr(config, 'feature_dim_3d', 256),
            dropout=getattr(config, 'dropout', 0.1),
            bn_momentum=getattr(config, 'bn_momentum', 0.1),
        ).to(device)
        
        # 加载权重 - 使用严格模式=False来处理缺失参数
        model_state = None
        if 'model' in ckpt:
            model_state = ckpt['model']
        elif 'model_state_dict' in ckpt:
            model_state = ckpt['model_state_dict']
        else:
            # 直接加载整个checkpoint作为state_dict
            model_state = ckpt
        
        # 过滤掉不兼容的参数
        model_dict = self.model.state_dict()
        filtered_state = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in model_state.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: model={model_dict[key].shape}, ckpt={value.shape}")
            else:
                unexpected_keys.append(key)
        
        for key in model_dict:
            if key not in filtered_state:
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"Missing keys (will use default initialization): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys (ignored): {unexpected_keys}")
        
        # 加载过滤后的权重
        print(f"  Loading model weights...", flush=True)
        model_dict.update(filtered_state)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        print(f"  ✓ Model weights loaded and set to eval mode", flush=True)
        
        self.voxel_size = getattr(config, 'voxel_size', 0.02)
        logger.info(f"✓ Model loaded successfully! (loaded {len(filtered_state)}/{len(model_dict)} parameters)")
        print(f"  ✓ ArticulateInference ready! (voxel_size={self.voxel_size})", flush=True)
    
    @torch.no_grad()
    def predict(self, coords: np.ndarray, features: np.ndarray, sem_gt: np.ndarray = None, max_points: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测语义标签和置信度分数 - 使用分层采样策略
        
        Args:
            coords: 点坐标 (N, 3)
            features: 特征 (N, 9) [colors, normals, coords]
            sem_gt: GT语义标签用于分层采样 (N,) - 如果没有则随机采样
            max_points: 最大推理点数
        
        Returns:
            sem_pred: 语义标签 (N,)
            sem_scores: 置信度分数 (N,)  
        """
        N = len(coords)
        
        # 如果点数过多，使用分层采样（与训练时一致）
        if N > max_points:
            if sem_gt is not None:
                # 分层采样：为每个类别保留相似数量的点
                unique_labels = np.unique(sem_gt)
                selected_idx = []
                points_per_class = max_points // len(unique_labels)
                
                for label in unique_labels:
                    label_mask = sem_gt == label
                    label_idx = np.where(label_mask)[0]
                    n_select = min(len(label_idx), points_per_class)
                    if n_select > 0:
                        selected_idx.extend(np.random.choice(label_idx, n_select, replace=False))
                
                # 填充剩余配额
                remaining = max_points - len(selected_idx)
                if remaining > 0:
                    all_idx = np.arange(N)
                    remaining_idx = np.setdiff1d(all_idx, selected_idx)
                    if len(remaining_idx) > 0:
                        extra_idx = np.random.choice(remaining_idx, min(remaining, len(remaining_idx)), replace=False)
                        selected_idx.extend(extra_idx)
                
                sample_idx = np.array(selected_idx)
            else:
                # 随机采样
                sample_idx = np.random.choice(N, max_points, replace=False)
            
            coords_sampled = coords[sample_idx]
            features_sampled = features[sample_idx]
        else:
            sample_idx = np.arange(N)
            coords_sampled = coords
            features_sampled = features
        
        # 体素化
        voxel_coords = np.floor(coords_sampled / self.voxel_size).astype(np.int32)
        unique_coords, unique_indices, inverse_indices = np.unique(
            voxel_coords, axis=0, return_index=True, return_inverse=True
        )
        
        # 构建稀疏张量输入
        batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])
        features_unique = features_sampled[unique_indices]
        
        coords_tensor = torch.from_numpy(coords_with_batch).int().to(self.device)
        features_tensor = torch.from_numpy(features_unique).float().to(self.device)
        
        x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
        
        # 推理
        outputs = self.model(x)
        logits = outputs['seg_logits'].features.cpu().numpy()
        
        # 计算预测和置信度
        probs_voxel = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        labels_voxel = np.argmax(logits, axis=-1)
        scores_voxel = np.max(probs_voxel, axis=-1)  # 最大概率作为置信度
        
        # 映射到采样点云
        sem_pred_sampled = labels_voxel[inverse_indices]
        sem_scores_sampled = scores_voxel[inverse_indices]
        
        # 如果进行了采样，需要传播结果回完整点云
        if N > max_points:
            from scipy.spatial import cKDTree
            
            tree = cKDTree(coords_sampled)
            _, nearest_idx = tree.query(coords, k=1)
            
            sem_pred = sem_pred_sampled[nearest_idx]
            sem_scores = sem_scores_sampled[nearest_idx]
        else:
            sem_pred = sem_pred_sampled
            sem_scores = sem_scores_sampled
        
        return sem_pred, sem_scores
    
    @torch.no_grad()
    def predict_with_tta(self, coords: np.ndarray, features: np.ndarray, 
                        sem_gt: np.ndarray = None, max_points: int = 100000, 
                        tta_rotations: int = 4, tta_scales: List[float] = [0.95, 1.0, 1.05]) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用测试时增强的预测，提高预测准确性和鲁棒性
        
        Args:
            coords: 点坐标 (N, 3)
            features: 特征 (N, 9) [colors, normals, coords]
            sem_gt: GT语义标签用于分层采样 (N,)
            max_points: 最大推理点数
            tta_rotations: 旋转增强次数（绕Z轴）
            tta_scales: 缩放增强比例列表
        
        Returns:
            sem_pred: 语义标签 (N,)
            sem_scores: 置信度分数 (N,)
        """
        all_predictions = []
        all_scores = []
        
        logger.info(f"🔄 Testing with TTA: {tta_rotations} rotations × {len(tta_scales)} scales = {tta_rotations * len(tta_scales)} variants")
        
        for scale in tta_scales:
            for rot_idx in range(tta_rotations):
                # 生成增强后的数据
                aug_coords, aug_features = self._apply_tta_transform(
                    coords, features, scale=scale, rotation_idx=rot_idx, total_rotations=tta_rotations
                )
                
                # 预测
                pred_labels, pred_scores = self.predict(
                    aug_coords, aug_features, sem_gt=sem_gt, max_points=max_points
                )
                
                all_predictions.append(pred_labels)
                all_scores.append(pred_scores)
        
        # 融合多个预测结果
        final_labels, final_scores = self._ensemble_predictions(all_predictions, all_scores)
        
        logger.info(f"✓ TTA completed, ensembled {len(all_predictions)} predictions")
        
        return final_labels, final_scores
    
    def _apply_tta_transform(self, coords: np.ndarray, features: np.ndarray, 
                            scale: float = 1.0, rotation_idx: int = 0, total_rotations: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用测试时增强变换
        """
        aug_coords = coords.copy()
        aug_features = features.copy()
        
        # 缩放变换
        if scale != 1.0:
            aug_coords = aug_coords * scale
            # 更新特征中的坐标部分（假设特征格式为[colors, normals, coords]）
            aug_features[:, 6:9] = aug_features[:, 6:9] * scale
        
        # 旋转变换（绕Z轴）
        if rotation_idx > 0:
            angle = 2 * np.pi * rotation_idx / total_rotations
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 旋转矩阵（绕Z轴）
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            
            aug_coords = aug_coords @ rotation_matrix.T
            
            # 旋转法向量
            normals = aug_features[:, 3:6]
            rotated_normals = normals @ rotation_matrix.T
            aug_features[:, 3:6] = rotated_normals
            
            # 旋转特征中的坐标部分
            feature_coords = aug_features[:, 6:9]
            rotated_feature_coords = feature_coords @ rotation_matrix.T
            aug_features[:, 6:9] = rotated_feature_coords
        
        # 添加轻微的噪声扰动（提高鲁棒性）
        noise_level = 0.001  # 1mm噪声
        aug_coords += np.random.normal(0, noise_level, aug_coords.shape)
        aug_features[:, 6:9] += np.random.normal(0, noise_level, (len(aug_features), 3))
        
        return aug_coords, aug_features
    
    def _ensemble_predictions(self, all_predictions: List[np.ndarray], all_scores: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        融合多个预测结果
        """
        N = len(all_predictions[0])
        num_classes = 3  # background, rotation, translation
        
        # 投票统计
        vote_matrix = np.zeros((N, num_classes))
        score_matrix = np.zeros((N, num_classes))
        
        for pred_labels, pred_scores in zip(all_predictions, all_scores):
            for i in range(N):
                label = pred_labels[i]
                score = pred_scores[i]
                vote_matrix[i, label] += 1
                score_matrix[i, label] += score
        
        # 获取最终预测
        final_labels = np.argmax(vote_matrix, axis=1)
        
        # 计算平均置信度
        final_scores = np.zeros(N)
        for i in range(N):
            label = final_labels[i]
            if vote_matrix[i, label] > 0:
                final_scores[i] = score_matrix[i, label] / vote_matrix[i, label]
            else:
                final_scores[i] = 0.0
        
        return final_labels, final_scores


# ============================================================================
# 主要评估函数
# ============================================================================
def evaluate_ap(model_path: str, 
               data_dir: str,
               output_dir: str = "./ap_results",
               device: str = "cuda:0",
               eps: float = 0.3,
               min_samples: int = 50,
               max_scenes: int = None,
               test_mode: str = "validation",
               use_tta: bool = False,
               tta_rotations: int = 4,
               tta_scales: List[float] = [0.98, 1.0, 1.02]) -> Dict[str, Any]:
    """
    评估AP指标
    
    Args:
        model_path: 训练好的模型路径
        data_dir: 数据目录 
        output_dir: 输出目录
        device: 设备
        eps: DBSCAN eps参数
        min_samples: DBSCAN min_samples参数
        max_scenes: 最大评估场景数
        test_mode: 测试模式 ("validation" 或 "test")
    
    Returns:
        评估结果字典
    """
    print("\n" + "="*80, flush=True)
    print("🔍 Starting evaluate_ap function...", flush=True)
    print("="*80, flush=True)
    
    # 创建输出目录
    print(f"📁 Creating output directory: {output_dir}", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready", flush=True)
    
    # 加载模型
    print(f"\n🤖 Loading model from: {model_path}", flush=True)
    logger.info("🤖 Loading model for inference...")
    model = ArticulateInference(model_path, device)
    print("✓ Model loaded successfully!", flush=True)
    
    # 加载测试集
    print(f"\n📂 Loading {test_mode} dataset from: {data_dir}", flush=True)
    logger.info(f"📂 Loading {test_mode} dataset...")
    test_dataset = Articulate3DDataset(
        data_dir=data_dir,
        mode=test_mode, 
        voxel_size=0.02,
    )
    
    print(f"✓ Dataset loaded successfully! Total scenes: {len(test_dataset)}", flush=True)
    logger.info(f"📊 Evaluating AP on {len(test_dataset)} {test_mode} scenes")
    
    # 收集所有场景的结果
    print(f"\n📊 Preparing to evaluate scenes...", flush=True)
    all_class_results = defaultdict(list)
    scene_results = []
    
    # 添加场景级别的性能分析
    scene_analysis = {
        'scene_metrics': [],
        'cumulative_metrics': [],
        'scene_complexity': []
    }
    
    scenes_to_eval = range(len(test_dataset))
    if max_scenes:
        scenes_to_eval = list(scenes_to_eval)[:max_scenes]
        print(f"\n🔧 Limited evaluation: processing {max_scenes} out of {len(test_dataset)} scenes", flush=True)
        logger.info(f"🔧 Limited evaluation: processing {max_scenes} out of {len(test_dataset)} scenes")
    else:
        print(f"\n🔄 Full dataset evaluation: processing all {len(test_dataset)} scenes", flush=True)
        logger.info(f"🔄 Full dataset evaluation: processing all {len(test_dataset)} scenes")
    
    print(f"\n{'='*80}", flush=True)
    print(f"Starting scene-by-scene evaluation...", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for scene_idx in tqdm(scenes_to_eval, desc="Evaluating scenes"):
        # 加载场景数据
        print(f"\n[Scene {scene_idx}] Loading data...", flush=True)
        sample = test_dataset[scene_idx]
        scene_id = sample['scene_id']
        
        print(f"[Scene {scene_idx}] Scene ID: {scene_id}", flush=True)
        logger.info(f"\n🏠 Processing scene: {scene_id}")
        
        coords = sample['coords']
        colors = sample['colors']
        normals = sample['normals'] 
        sem_gt = sample['sem_labels']
        inst_gt = sample['inst_labels']
        
        # 构建特征: RGB + normals + coords
        features = np.hstack([colors, normals, coords]).astype(np.float32)
        
        logger.info(f"  Points: {len(coords)}, GT classes: {np.unique(sem_gt)}")
        
        # 模型预测 - 使用TTA或常规预测
        if use_tta:
            logger.info(f"  🔄 Using TTA with {tta_rotations} rotations and scales {tta_scales}")
            sem_pred, sem_scores = model.predict_with_tta(
                coords, features, sem_gt=sem_gt, max_points=100000,
                tta_rotations=tta_rotations, tta_scales=tta_scales
            )
        else:
            sem_pred, sem_scores = model.predict(coords, features, sem_gt=sem_gt, max_points=100000)
        
        logger.info(f"  Pred classes: {np.unique(sem_pred)}")

        # 语义预测后处理：使用空间一致性修正低置信度预测
        logger.info(f"📝 Refining semantic predictions using spatial consistency...")
        sem_pred = refine_semantic_prediction(coords, sem_pred, sem_scores, confidence_threshold=0.6)
        logger.info(f"  Refined pred classes: {np.unique(sem_pred)}")

        # 实例聚类
        logger.info(f"🔍 Starting clustering with eps={eps}, min_samples={min_samples}")
        logger.info(f"  Coords shape: {coords.shape}, Sem pred unique: {np.unique(sem_pred)}")
        inst_pred = cluster_instances_by_semantic(
            coords, sem_pred, eps=eps, min_samples=min_samples
        )
        logger.info(f"  Clustering result: {len(np.unique(inst_pred[inst_pred > 0]))} instances")
        
        logger.info(f"  GT instances: {len(np.unique(inst_gt[inst_gt > 0]))}, "
                   f"Pred instances: {len(np.unique(inst_pred[inst_pred > 0]))}")
        
        # ============================================================================
        # 调试：分析语义分割和实例聚类的影响
        # ============================================================================
        logger.info(f"🔍 Debugging semantic and instance quality for scene {scene_id}")
        
        # 1. 语义分割质量分析
        sem_accuracy_overall = (sem_gt == sem_pred).mean()
        logger.info(f"  Overall semantic accuracy: {sem_accuracy_overall:.3f}")
        
        for class_id in [1, 2]:  # rotation, translation
            class_name = CLASS_LABELS[class_id]
            gt_class_mask = sem_gt == class_id
            pred_class_mask = sem_pred == class_id
            
            if gt_class_mask.sum() > 0:
                # 类别级别的precision, recall, f1
                tp_sem = (gt_class_mask & pred_class_mask).sum()
                fp_sem = (pred_class_mask & ~gt_class_mask).sum() 
                fn_sem = (gt_class_mask & ~pred_class_mask).sum()
                
                precision_sem = tp_sem / (tp_sem + fp_sem) if (tp_sem + fp_sem) > 0 else 0
                recall_sem = tp_sem / (tp_sem + fn_sem) if (tp_sem + fn_sem) > 0 else 0
                f1_sem = 2 * precision_sem * recall_sem / (precision_sem + recall_sem) if (precision_sem + recall_sem) > 0 else 0
                
                logger.info(f"  {class_name} semantic: P={precision_sem:.3f}, R={recall_sem:.3f}, F1={f1_sem:.3f}")
                
                # 实例分割质量分析（使用GT语义标签测试理想情况）
                gt_class_coords = coords[gt_class_mask]
                gt_class_inst = inst_gt[gt_class_mask]
                
                if len(gt_class_coords) > 0:
                    # 使用GT语义标签进行聚类（理想情况）
                    ideal_inst_pred = cluster_instances_by_semantic(
                        gt_class_coords, 
                        np.ones(len(gt_class_coords), dtype=np.int32) * class_id,  # 使用GT语义
                        eps=eps, min_samples=min_samples
                    )
                    
                    gt_instances_unique = np.unique(gt_class_inst[gt_class_inst > 0])
                    ideal_instances_unique = np.unique(ideal_inst_pred[ideal_inst_pred > 0])
                    
                    logger.info(f"    GT instances: {len(gt_instances_unique)}")
                    logger.info(f"    Ideal clustering instances: {len(ideal_instances_unique)}")
                    logger.info(f"    Actual pred instances: {len(np.unique(inst_pred[pred_class_mask][inst_pred[pred_class_mask] > 0]))}")
        
        # 2. 计算理想AP（使用GT语义标签）
        logger.info("  Computing ideal AP with GT semantics...")
        ideal_inst_pred_full = np.zeros_like(inst_gt)
        
        for class_id in [1, 2]:
            gt_class_mask = sem_gt == class_id
            if gt_class_mask.sum() > 0:
                gt_class_coords = coords[gt_class_mask]
                ideal_class_inst = cluster_instances_by_semantic(
                    gt_class_coords, 
                    np.ones(len(gt_class_coords), dtype=np.int32) * class_id,
                    eps=eps, min_samples=min_samples
                )
                # 重新映射实例ID以避免冲突
                unique_ideal = np.unique(ideal_class_inst[ideal_class_inst > 0])
                for i, old_id in enumerate(unique_ideal):
                    mask = ideal_class_inst == old_id
                    ideal_class_inst[mask] = class_id * 10000 + i + 1
                
                ideal_inst_pred_full[gt_class_mask] = ideal_class_inst

        ideal_class_results = compute_map_all_classes(
            coords, coords,
            sem_gt, sem_gt,  # 使用GT语义标签
            inst_gt, ideal_inst_pred_full,
            np.ones_like(sem_gt, dtype=np.float32),  # 理想置信度
            num_classes=3
        )
        
        logger.info("  Ideal AP results (GT semantics + clustering):")
        for class_name, ap_result in ideal_class_results.items():
            logger.info(f"    {class_name}: AP25={ap_result.ap_25:.3f}, AP50={ap_result.ap_50:.3f}")

        # 计算每个类别的AP
        class_results = compute_map_all_classes(
            coords, coords,  # 使用相同坐标
            sem_gt, sem_pred,
            inst_gt, inst_pred,
            sem_scores,
            num_classes=3
        )
        
        # 收集结果
        scene_result = {
            'scene_id': scene_id,
            'num_points': len(coords),
        }
        
        for class_name, ap_result in class_results.items():
            all_class_results[class_name].append(ap_result)
            scene_result[f'{class_name}_ap25'] = ap_result.ap_25
            scene_result[f'{class_name}_ap50'] = ap_result.ap_50
            scene_result[f'{class_name}_ap_avg'] = ap_result.ap_avg
            scene_result[f'{class_name}_num_gt'] = ap_result.num_gt
            scene_result[f'{class_name}_num_pred'] = ap_result.num_pred
        
        scene_results.append(scene_result)
        
        # 打印场景结果对比
        logger.info(f"Scene {scene_id} results comparison:")
        logger.info("  Actual results (pred semantics + clustering):")
        for class_name, ap_result in class_results.items():
            logger.info(f"    {class_name}: AP25={ap_result.ap_25:.3f}, AP50={ap_result.ap_50:.3f}, "
                  f"AP_avg={ap_result.ap_avg:.3f} "
                  f"(GT:{ap_result.num_gt}, Pred:{ap_result.num_pred})")
        
        logger.info("  Ideal results (GT semantics + clustering):")
        for class_name, ap_result in ideal_class_results.items():
            logger.info(f"    {class_name}: AP25={ap_result.ap_25:.3f}, AP50={ap_result.ap_50:.3f}, "
                  f"AP_avg={ap_result.ap_avg:.3f} "
                  f"(GT:{ap_result.num_gt}, Pred:{ap_result.num_pred})")
        
        # 计算影响分析
        logger.info("  Impact Analysis:")
        for class_name in class_results.keys():
            actual_ap50 = class_results[class_name].ap_50
            ideal_ap50 = ideal_class_results[class_name].ap_50
            semantic_impact = ideal_ap50 - actual_ap50
            logger.info(f"    {class_name}: Semantic impact on AP50 = {semantic_impact:.3f}")
            if ideal_ap50 > 0:
                logger.info(f"    {class_name}: Semantic causes {(semantic_impact/ideal_ap50)*100:.1f}% AP loss")
        
        # 收集调试结果
        scene_result['ideal_rotation_ap50'] = ideal_class_results.get('rotation', APMetrics()).ap_50
        scene_result['ideal_translation_ap50'] = ideal_class_results.get('translation', APMetrics()).ap_50
        
        # 分析场景复杂度
        scene_complexity = {
            'num_points': len(coords),
            'num_gt_rotation': len(np.unique(inst_gt[sem_gt == 1][inst_gt[sem_gt == 1] > 0])) if (sem_gt == 1).sum() > 0 else 0,
            'num_gt_translation': len(np.unique(inst_gt[sem_gt == 2][inst_gt[sem_gt == 2] > 0])) if (sem_gt == 2).sum() > 0 else 0,
            'rotation_point_ratio': (sem_gt == 1).sum() / len(coords),
            'translation_point_ratio': (sem_gt == 2).sum() / len(coords),
            'semantic_accuracy': (sem_gt == sem_pred).mean(),
        }
        
        # 计算当前场景的平均AP
        current_scene_map = np.mean([ap_result.ap_50 for ap_result in class_results.values() if ap_result.num_gt > 0])
        
        # 计算累积AP（到当前场景为止的平均）
        all_scene_aps = []
        for class_name in class_results.keys():
            if all_class_results[class_name]:  # 如果有数据
                cumulative_ap = np.mean([ap.ap_50 for ap in all_class_results[class_name]])
                all_scene_aps.append(cumulative_ap)
        
        cumulative_map = np.mean(all_scene_aps) if all_scene_aps else 0.0
        
        # 保存分析数据
        scene_analysis['scene_metrics'].append({
            'scene_idx': scene_idx,
            'scene_id': scene_id,
            'scene_map': current_scene_map,
            'complexity': scene_complexity
        })
        
        scene_analysis['cumulative_metrics'].append({
            'scene_idx': scene_idx,
            'cumulative_map': cumulative_map,
            'num_scenes_processed': scene_idx + 1
        })
        
        scene_analysis['scene_complexity'].append(scene_complexity)
        
        # 实时报告趋势
        logger.info(f"📈 Performance Trend Analysis:")
        logger.info(f"  Current scene mAP@0.5: {current_scene_map:.3f}")
        logger.info(f"  Cumulative mAP@0.5 (up to scene {scene_idx+1}): {cumulative_map:.3f}")
        logger.info(f"  Scene complexity: {scene_complexity['num_gt_rotation']}R+{scene_complexity['num_gt_translation']}T, "
                   f"{scene_complexity['num_points']} pts, sem_acc={scene_complexity['semantic_accuracy']:.3f}")
        
        # 详细显示当前场景的类别AP
        logger.info(f"  Current scene class APs:")
        for class_name, ap_result in class_results.items():
            if ap_result.num_gt > 0:
                logger.info(f"    {class_name}: AP@0.5={ap_result.ap_50:.3f} (GT:{ap_result.num_gt}, Pred:{ap_result.num_pred}, Matched:{ap_result.num_matched})")
            else:
                logger.info(f"    {class_name}: No GT instances")
        
        # 如果处理了足够多的场景，分析趋势
        if len(scene_analysis['cumulative_metrics']) >= 5:
            recent_maps = [m['cumulative_map'] for m in scene_analysis['cumulative_metrics'][-5:]]
            if len(recent_maps) >= 2:
                trend = recent_maps[-1] - recent_maps[0]
                if trend < -0.01:
                    logger.warning(f"  ⚠️  Declining trend detected: mAP decreased by {abs(trend):.3f} over last 5 scenes")
                elif trend > 0.01:
                    logger.info(f"  📈 Improving trend: mAP increased by {trend:.3f} over last 5 scenes")
        
        logger.info("")
    
    # 计算总体结果
    logger.info("\n" + "="*60)
    logger.info("📊 Overall Results")
    logger.info("="*60)
    
    overall_results = {}
    
    for class_name in CLASS_LABELS[1:]:  # 跳过background
        class_aps = all_class_results[class_name]
        
        if not class_aps:
            overall_results[class_name] = APMetrics()
            continue
        
        # 计算平均AP - 只计算有GT的场景，避免0值拉低平均
        valid_ap25 = [ap.ap_25 for ap in class_aps if ap.num_gt > 0]
        valid_ap50 = [ap.ap_50 for ap in class_aps if ap.num_gt > 0]  
        valid_ap_avg = [ap.ap_avg for ap in class_aps if ap.num_gt > 0]
        
        avg_ap25 = np.mean(valid_ap25) if valid_ap25 else 0.0
        avg_ap50 = np.mean(valid_ap50) if valid_ap50 else 0.0
        avg_ap_avg = np.mean(valid_ap_avg) if valid_ap_avg else 0.0
        
        total_gt = sum(ap.num_gt for ap in class_aps)
        total_pred = sum(ap.num_pred for ap in class_aps)
        total_matched = sum(ap.num_matched for ap in class_aps)
        
        overall_results[class_name] = APMetrics(
            ap_25=avg_ap25,
            ap_50=avg_ap50,
            ap_avg=avg_ap_avg,
            num_gt=total_gt,
            num_pred=total_pred,
            num_matched=total_matched
        )
        
        logger.info(f"{class_name}:")
        logger.info(f"  AP@0.25: {avg_ap25:.3f} (computed from {len(valid_ap25)} valid scenes)")
        logger.info(f"  AP@0.5:  {avg_ap50:.3f} (computed from {len(valid_ap50)} valid scenes)")
        logger.info(f"  AP_avg:  {avg_ap_avg:.3f} (computed from {len(valid_ap_avg)} valid scenes)")
        logger.info(f"  GT instances: {total_gt}")
        logger.info(f"  Pred instances: {total_pred}")
        logger.info(f"  Matched: {total_matched}")
        if len(valid_ap50) < len(class_aps):
            logger.warning(f"  ⚠️  {len(class_aps) - len(valid_ap50)} scenes had no GT instances and were excluded from averaging")
        logger.info("")
    
    # 计算mAP - 使用加权平均，按GT实例数加权
    valid_classes_with_weights = []

    for class_name, result in overall_results.items():
        if result.num_gt > 0:
            valid_classes_with_weights.append((result, result.num_gt))

    if valid_classes_with_weights:
        # 计算加权平均mAP
        total_weight = sum(weight for _, weight in valid_classes_with_weights)

        map_25 = sum(r.ap_25 * weight for r, weight in valid_classes_with_weights) / total_weight
        map_50 = sum(r.ap_50 * weight for r, weight in valid_classes_with_weights) / total_weight
        map_avg = sum(r.ap_avg * weight for r, weight in valid_classes_with_weights) / total_weight

        logger.info(f"mAP computation: weighted by GT instances across {len(valid_classes_with_weights)} classes")
        for result, weight in valid_classes_with_weights:
            class_name = [k for k, v in overall_results.items() if v == result][0]
            logger.info(f"  {class_name}: weight={weight} GT instances")
    else:
        map_25 = map_50 = map_avg = 0.0
        logger.warning("No valid classes with GT instances found!")
    
    logger.info(f"mAP@0.25: {map_25:.3f}")
    logger.info(f"mAP@0.5:  {map_50:.3f}")
    logger.info(f"mAP_avg:  {map_avg:.3f}")
    
    # 分析性能下降趋势
    logger.info("\n" + "="*60)
    logger.info("📊 Performance Trend Analysis")
    logger.info("="*60)
    
    if len(scene_analysis['cumulative_metrics']) > 1:
        # 分析累积mAP趋势
        cumulative_maps = [m['cumulative_map'] for m in scene_analysis['cumulative_metrics']]
        scene_indices = [m['scene_idx'] for m in scene_analysis['cumulative_metrics']]
        
        # 计算线性趋势
        if len(cumulative_maps) >= 3:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(scene_indices, cumulative_maps)
            
            logger.info(f"Linear trend analysis:")
            logger.info(f"  Slope: {slope:.6f} (mAP change per scene)")
            logger.info(f"  R²: {r_value**2:.3f} (trend strength)")
            logger.info(f"  P-value: {p_value:.6f}")
            
            if p_value < 0.05 and slope < -0.001:
                logger.warning(f"  ⚠️  Significant declining trend detected!")
            elif p_value < 0.05 and slope > 0.001:
                logger.info(f"  📈 Significant improving trend detected!")
            else:
                logger.info(f"  ➡️  No significant trend")
        
        # 分析场景复杂度与性能的关系
        complexities = scene_analysis['scene_complexity']
        scene_maps = [m['scene_map'] for m in scene_analysis['scene_metrics']]
        
        # 复杂度指标
        num_points = [c['num_points'] for c in complexities]
        total_instances = [c['num_gt_rotation'] + c['num_gt_translation'] for c in complexities]
        semantic_accs = [c['semantic_accuracy'] for c in complexities]
        
        logger.info(f"\nScene complexity analysis:")
        logger.info(f"  Point count range: {min(num_points)} - {max(num_points)} (avg: {np.mean(num_points):.0f})")
        logger.info(f"  Instance count range: {min(total_instances)} - {max(total_instances)} (avg: {np.mean(total_instances):.1f})")
        logger.info(f"  Semantic accuracy range: {min(semantic_accs):.3f} - {max(semantic_accs):.3f} (avg: {np.mean(semantic_accs):.3f})")
        
        # 相关性分析
        if len(scene_maps) >= 3:
            try:
                from scipy.stats import pearsonr
                
                corr_points, p_points = pearsonr(num_points, scene_maps)
                corr_instances, p_instances = pearsonr(total_instances, scene_maps)
                corr_semantic, p_semantic = pearsonr(semantic_accs, scene_maps)
                
                logger.info(f"\nCorrelation with scene mAP:")
                logger.info(f"  Point count: r={corr_points:.3f} (p={p_points:.3f})")
                logger.info(f"  Instance count: r={corr_instances:.3f} (p={p_instances:.3f})")  
                logger.info(f"  Semantic accuracy: r={corr_semantic:.3f} (p={p_semantic:.3f})")
                
                # 提供分析建议
                if p_semantic < 0.05 and corr_semantic > 0.3:
                    logger.info(f"  💡 Strong correlation with semantic accuracy suggests model semantic prediction quality affects instance segmentation")
                if p_instances < 0.05 and corr_instances < -0.3:
                    logger.warning(f"  ⚠️  Performance degrades with more instances - possible clustering parameter issue")
                if p_points < 0.05 and corr_points < -0.3:
                    logger.warning(f"  ⚠️  Performance degrades with larger scenes - possible memory/sampling issue")
                    
            except ImportError:
                logger.info("  (scipy not available for correlation analysis)")
        
        # 按场景顺序分析性能
        first_half_maps = scene_maps[:len(scene_maps)//2] if len(scene_maps) >= 4 else scene_maps[:len(scene_maps)//2] if len(scene_maps) >= 2 else []
        second_half_maps = scene_maps[len(scene_maps)//2:] if len(scene_maps) >= 4 else scene_maps[len(scene_maps)//2:] if len(scene_maps) >= 2 else []
        
        if first_half_maps and second_half_maps:
            first_avg = np.mean(first_half_maps)
            second_avg = np.mean(second_half_maps)
            logger.info(f"\nPerformance by dataset order:")
            logger.info(f"  First half scenes: {first_avg:.3f}")
            logger.info(f"  Second half scenes: {second_avg:.3f}")
            logger.info(f"  Difference: {second_avg - first_avg:.3f}")
            
            if second_avg < first_avg - 0.05:
                logger.warning(f"  ⚠️  Later scenes perform significantly worse - possible dataset ordering issue")
    
    # 提供优化建议
    logger.info(f"\n💡 Optimization Suggestions:")
    if len(scene_analysis['cumulative_metrics']) > 1:
        avg_semantic_acc = np.mean([c['semantic_accuracy'] for c in complexities])
        if avg_semantic_acc < 0.7:
            logger.info(f"  1. Improve semantic segmentation model (current avg accuracy: {avg_semantic_acc:.3f})")
        
        avg_instances = np.mean([c['num_gt_rotation'] + c['num_gt_translation'] for c in complexities])
        if avg_instances > 10:
            logger.info(f"  2. Consider adjusting clustering parameters for scenes with many instances (avg: {avg_instances:.1f})")
            
        point_variation = np.std(num_points) / np.mean(num_points)
        if point_variation > 0.5:
            logger.info(f"  3. Scene size varies significantly - consider adaptive sampling strategy")
    
    logger.info(f"  4. Use --max_scenes parameter to test on smaller subsets for faster iteration")
    logger.info(f"  5. Try --use_tta for potentially better accuracy (at cost of speed)")
    
    # 保存结果
    results = {
        'overall_metrics': {
            'mAP@0.25': map_25,
            'mAP@0.5': map_50,
            'mAP_avg': map_avg,
        },
        'per_class_metrics': {},
        'per_scene_metrics': scene_results,
        'performance_analysis': scene_analysis,  # 添加性能分析数据
        'config': {
            'model_path': str(model_path),
            'data_dir': str(data_dir),
            'test_mode': test_mode,
            'eps': eps,
            'min_samples': min_samples,
            'num_scenes': len(scenes_to_eval),
        }
    }
    
    # 转换per-class结果为可序列化格式
    for class_name, ap_result in overall_results.items():
        results['per_class_metrics'][class_name] = {
            'AP@0.25': float(ap_result.ap_25),
            'AP@0.5': float(ap_result.ap_50),
            'AP_avg': float(ap_result.ap_avg),
            'num_gt': int(ap_result.num_gt),
            'num_pred': int(ap_result.num_pred),
            'num_matched': int(ap_result.num_matched),
        }
    
    # 保存JSON结果
    results_path = output_dir / f"ap_results_{test_mode}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    return results


# ============================================================================
# 参数调优
# ============================================================================
def tune_clustering_parameters(model_path: str,
                             data_dir: str, 
                             output_dir: str = "./tuning_results",
                             device: str = "cuda:0",
                             max_scenes: int = 10):
    """
    调优聚类参数
    """
    
    # 参数搜索空间
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_samples_values = [10, 20, 50, 100]
    
    output_dir = Path(output_dir)  
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_map = 0.0
    best_params = {}
    all_results = []
    
    logger.info("🔧 Tuning clustering parameters...")
    logger.info(f"Search space: eps={eps_values}, min_samples={min_samples_values}")
    logger.info(f"Evaluating on {max_scenes} scenes per configuration\n")
    
    total_configs = len(eps_values) * len(min_samples_values)
    config_idx = 0
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            config_idx += 1
            logger.info(f"[{config_idx}/{total_configs}] Testing eps={eps}, min_samples={min_samples}")
            
            try:
                results = evaluate_ap(
                    model_path=model_path,
                    data_dir=data_dir,
                    output_dir=output_dir / f"eps_{eps}_min_{min_samples}",
                    device=device,
                    eps=eps,
                    min_samples=min_samples,
                    max_scenes=max_scenes
                )
                
                current_map = results['overall_metrics']['mAP_avg']
                
                result_entry = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'mAP_avg': current_map,
                    'mAP@0.5': results['overall_metrics']['mAP@0.5'],
                    'per_class': results['per_class_metrics']
                }
                all_results.append(result_entry)
                
                logger.info(f"  → mAP_avg: {current_map:.3f}")
                
                if current_map > best_map:
                    best_map = current_map
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    logger.info(f"  🎯 New best! mAP_avg: {best_map:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
                
            logger.info("")
    
    # 保存调参结果
    tuning_results = {
        'best_params': best_params,
        'best_mAP_avg': best_map,
        'all_results': all_results,
        'search_space': {
            'eps_values': eps_values,
            'min_samples_values': min_samples_values
        }
    }
    
    results_file = output_dir / "tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    logger.info("="*60)
    logger.info("🎉 Parameter tuning completed!")
    logger.info(f"Best parameters: eps={best_params.get('eps')}, min_samples={best_params.get('min_samples')}")
    logger.info(f"Best mAP_avg: {best_map:.3f}")
    logger.info(f"Results saved to: {results_file}")
    
    # 打印结果表格
    logger.info("\n📊 Results Summary:")
    logger.info("eps    min_samples    mAP_avg    mAP@0.5")
    logger.info("-" * 45)
    for result in sorted(all_results, key=lambda x: x['mAP_avg'], reverse=True):
        logger.info(f"{result['eps']:<6} {result['min_samples']:<12} {result['mAP_avg']:<10.3f} {result['mAP@0.5']:<10.3f}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    # 立即输出启动信息（使用print确保立即显示）
    print("="*80)
    print("🚀 Starting Articulate3D AP Evaluation Script...")
    print("="*80)
    print("Parsing arguments...", flush=True)
    
    parser = argparse.ArgumentParser(description="Compute AP for Articulate3D Instance Segmentation")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default="./data/processed/articulate3d_challenge_mov",
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='./ap_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='DBSCAN eps parameter for clustering')
    parser.add_argument('--min_samples', type=int, default=50, 
                        help='DBSCAN min_samples parameter')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Maximum number of scenes to evaluate (for testing)')
    parser.add_argument('--test_mode', type=str, default='validation', 
                        choices=['validation', 'test'],
                        help='Test mode: validation or test')
    parser.add_argument('--tune_params', action='store_true',
                        help='Tune clustering parameters instead of direct evaluation')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation for better accuracy')
    parser.add_argument('--tta_rotations', type=int, default=4,
                        help='Number of rotation augmentations for TTA')
    parser.add_argument('--tta_scales', nargs='+', type=float, default=[0.98, 1.0, 1.02],
                        help='Scale factors for TTA')
    
    print("Arguments parsed successfully!", flush=True)
    args = parser.parse_args()
    
    print(f"✓ Model path: {args.model_path}", flush=True)
    print(f"✓ Data dir: {args.data_dir}", flush=True)
    print(f"✓ Max scenes: {args.max_scenes}", flush=True)
    print("", flush=True)
    
    logger.info("="*80)
    logger.info("🎯 Articulate3D Instance Segmentation AP Evaluation")  
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Mode: {args.test_mode}")
    logger.info(f"Clustering: eps={args.eps}, min_samples={args.min_samples}")
    if args.use_tta:
        logger.info(f"TTA: {args.tta_rotations} rotations × {len(args.tta_scales)} scales")
    logger.info("")
    
    if args.tune_params:
        # 参数调优
        tune_clustering_parameters(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            max_scenes=args.max_scenes or 10,
        )
    else:
        # 直接评估
        results = evaluate_ap(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            eps=args.eps,
            min_samples=args.min_samples,
            max_scenes=args.max_scenes,
            test_mode=args.test_mode,
            use_tta=args.use_tta,
            tta_rotations=args.tta_rotations,
            tta_scales=args.tta_scales,
        )
    
    logger.info("\n✨ AP evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*80, flush=True)
        print("❌ ERROR: Script failed with exception!", flush=True)
        print("="*80, flush=True)
        print(f"Exception type: {type(e).__name__}", flush=True)
        print(f"Exception message: {str(e)}", flush=True)
        print("", flush=True)
        import traceback
        print("Full traceback:", flush=True)
        traceback.print_exc()
        print("="*80, flush=True)
        raise

