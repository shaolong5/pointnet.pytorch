"""
================================================================================
USDNet微调模型可视化工具
================================================================================
功能：
1. 加载微调后的checkpoint
2. 在测试数据上运行推理
3. 可视化预测结果（颜色编码）
4. 计算并显示准确率、mIoU等指标
5. 对比预测vs真实标签
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import zarr
from tqdm import tqdm

# 导入训练脚本中的模型
from train_usdnet_complete import (
    USDNetStudent, 
    SegmentationMetrics,
    GlobalLabelReader,
    collate_fn_sparse
)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not installed. Install with: pip install open3d")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not installed. Install with: pip install plotly")

import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


# ============================================================================
# 配色方案
# ============================================================================
def generate_color_palette(num_classes: int) -> np.ndarray:
    """生成区分度高的颜色方案"""
    np.random.seed(42)
    colors = []
    
    # 预定义一些高对比度的颜色
    base_colors = [
        [255, 0, 0],      # 红
        [0, 255, 0],      # 绿
        [0, 0, 255],      # 蓝
        [255, 255, 0],    # 黄
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 青
        [255, 128, 0],    # 橙
        [128, 0, 255],    # 紫
        [0, 255, 128],    # 青绿
        [255, 0, 128],    # 粉红
    ]
    
    colors.extend(base_colors[:min(num_classes, len(base_colors))])
    
    # 如果类别数更多，随机生成其他颜色
    while len(colors) < num_classes:
        color = np.random.randint(0, 256, 3).tolist()
        colors.append(color)
    
    return np.array(colors[:num_classes], dtype=np.float32) / 255.0


# ============================================================================
# 模型推理
# ============================================================================
class ModelInference:
    """模型推理类"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # 读取配置
        config = ckpt['config']
        self.num_classes = config['num_classes']
        self.voxel_size = config.get('voxel_size', 0.05)
        
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Voxel size: {self.voxel_size}")
        print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  - Best metric: {ckpt.get('best_metric', 'N/A')}")
        
        # 初始化模型
        self.model = USDNetStudent(
            num_classes=self.num_classes,
            feature_dim_3d=config.get('feature_dim_3d', 256),
            feature_dim_2d=config.get('feature_dim_2d', 768),
            dropout=config.get('dropout', 0.1),
        ).to(device)
        
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    @torch.no_grad()
    def predict(self, points: np.ndarray, colors: np.ndarray, 
                normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对点云进行推理
        
        Args:
            points: (N, 3) 点坐标
            colors: (N, 3) RGB颜色 [0, 1]
            normals: (N, 3) 法向量
        
        Returns:
            pred_labels: (N,) 预测标签
            pred_probs: (N, num_classes) 预测概率
        """
        # 体素化
        voxel_coords = np.floor(points / self.voxel_size).astype(np.int32)
        unique_coords, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        
        points_unique = points[unique_indices]
        colors_unique = colors[unique_indices]
        normals_unique = normals[unique_indices]
        
        # 构建特征：RGB + normals + coords (9维)
        features = np.hstack([colors_unique, normals_unique, points_unique]).astype(np.float32)
        
        # 添加batch维度
        batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])
        
        # 转换为tensor
        coords_tensor = torch.from_numpy(coords_with_batch).int().to(self.device)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        
        # 推理
        x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
        seg_logits, _ = self.model(x)
        
        # 获取预测结果
        logits = seg_logits.features.cpu().numpy()  # (M, num_classes)
        pred_probs_voxel = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        pred_labels_voxel = np.argmax(logits, axis=-1)
        
        # 将体素预测映射回原始点云（最近邻）
        from scipy.spatial import cKDTree
        tree = cKDTree(points_unique)
        _, nearest_indices = tree.query(points, k=1)
        
        pred_labels = pred_labels_voxel[nearest_indices]
        pred_probs = pred_probs_voxel[nearest_indices]
        
        return pred_labels, pred_probs


# ============================================================================
# 实例分割工具
# ============================================================================
def segment_instances_by_proximity(
    points: np.ndarray,
    labels: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 50
) -> np.ndarray:
    """
    使用空间聚类将同一语义类别分割成不同实例
    
    Args:
        points: (N, 3) 点坐标
        labels: (N,) 语义标签
        eps: DBSCAN聚类的邻域半径
        min_samples: DBSCAN最小样本数
    
    Returns:
        instance_labels: (N,) 实例标签，格式为 semantic_id * 10000 + instance_id
    """
    instance_labels = np.zeros(len(points), dtype=np.int32)
    
    unique_labels = np.unique(labels[labels >= 0])
    
    for sem_label in unique_labels:
        mask = labels == sem_label
        if mask.sum() < min_samples:
            instance_labels[mask] = sem_label * 10000
            continue
        
        # 对该类别的点进行空间聚类
        class_points = points[mask]
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_ids = clustering.fit_predict(class_points)
        
        # 将聚类结果映射回原始索引
        # 格式: semantic_id * 10000 + instance_id
        instance_ids = np.zeros(len(cluster_ids), dtype=np.int32)
        
        valid_clusters = cluster_ids >= 0
        if valid_clusters.sum() > 0:
            # 重新编号实例（从0开始）
            unique_clusters = np.unique(cluster_ids[valid_clusters])
            for new_id, old_id in enumerate(unique_clusters):
                instance_ids[cluster_ids == old_id] = new_id
        
        # 噪声点（cluster_id == -1）归为instance 0
        instance_ids[cluster_ids == -1] = 0
        
        # 组合语义标签和实例标签
        instance_labels[mask] = sem_label * 10000 + instance_ids
    
    return instance_labels


# ============================================================================
# 可视化
# ============================================================================
def visualize_predictions_matplotlib(
    points: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    num_classes: int
):
    """使用matplotlib生成2D可视化"""
    
    color_palette = generate_color_palette(num_classes)
    
    # 创建图像
    fig = plt.figure(figsize=(20, 8))
    
    # 1. 真实标签（俯视图）
    ax1 = fig.add_subplot(131, projection='3d')
    valid_gt = gt_labels >= 0
    if valid_gt.sum() > 0:
        colors_gt = color_palette[gt_labels[valid_gt]]
        ax1.scatter(points[valid_gt, 0], points[valid_gt, 1], points[valid_gt, 2],
                   c=colors_gt, s=1, alpha=0.6)
    ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=45)
    
    # 2. 预测标签（俯视图）
    ax2 = fig.add_subplot(132, projection='3d')
    colors_pred = color_palette[pred_labels]
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors_pred, s=1, alpha=0.6)
    ax2.set_title('Prediction', fontsize=16, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=45)
    
    # 3. 错误分布（俯视图）
    ax3 = fig.add_subplot(133, projection='3d')
    if valid_gt.sum() > 0:
        correct = (gt_labels == pred_labels) & valid_gt
        incorrect = (gt_labels != pred_labels) & valid_gt
        
        if correct.sum() > 0:
            ax3.scatter(points[correct, 0], points[correct, 1], points[correct, 2],
                       c='green', s=1, alpha=0.3, label='Correct')
        if incorrect.sum() > 0:
            ax3.scatter(points[incorrect, 0], points[incorrect, 1], points[incorrect, 2],
                       c='red', s=2, alpha=0.8, label='Incorrect')
        ax3.legend()
    ax3.set_title('Correct vs Incorrect', fontsize=16, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def visualize_predictions_open3d(
    points: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    num_classes: int,
    window_name: str = "USDNet Prediction"
):
    """使用Open3D进行交互式3D可视化"""
    
    if not HAS_OPEN3D:
        print("⚠️  Open3D not available, skipping interactive visualization")
        return
    
    color_palette = generate_color_palette(num_classes)
    
    # 创建点云对象（预测）
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points)
    colors_pred = color_palette[pred_labels]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
    
    # 创建点云对象（真实）
    pcd_gt = o3d.geometry.PointCloud()
    valid_gt = gt_labels >= 0
    pcd_gt.points = o3d.utility.Vector3dVector(points[valid_gt])
    colors_gt = color_palette[gt_labels[valid_gt]]
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
    
    # 并排显示
    pcd_pred.translate([points[:, 0].max() - points[:, 0].min() + 1, 0, 0])
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_pred],
        window_name=f"{window_name} | Left: GT | Right: Prediction",
        width=1920,
        height=1080,
        left=50,
        top=50,
    )


def visualize_predictions_plotly_html(
    points: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    num_classes: int,
    max_points_per_object: int = 50000,
    use_instance_segmentation: bool = True,
    clustering_eps: float = 0.3,
    min_accuracy_threshold: float = 0.3,
    bbox_padding: float = 0.5
):
    """生成交互式HTML可视化（整个场景+各个部件实例）
    
    对于每个实例的独立HTML，会可视化该实例所在3D包围盒内的所有点，
    目标实例会高亮显示，周围环境以半透明方式显示。
    
    Args:
        bbox_padding: 包围盒扩展距离（米），用于显示物件周围的环境上下文
    """
    
    if not HAS_PLOTLY:
        print("⚠️  Plotly not available, skipping HTML visualization")
        return
    
    print(f"  Generating HTML visualization...")
    
    color_palette = generate_color_palette(num_classes)
    save_path = Path(save_path)
    
    # ========================================================================
    # 使用USDNet的分割能力进行实例分割
    # ========================================================================
    if use_instance_segmentation:
        print(f"  🔍 Performing instance segmentation (DBSCAN eps={clustering_eps})...")
        instance_labels = segment_instances_by_proximity(
            points, pred_labels, eps=clustering_eps, min_samples=50
        )
        
        # 统计实例
        unique_instances = np.unique(instance_labels[instance_labels >= 0])
        instance_stats = []
        
        for inst_label in unique_instances:
            mask = instance_labels == inst_label
            point_count = mask.sum()
            if point_count < 10:
                continue
            
            sem_label = inst_label // 10000
            inst_id = inst_label % 10000
            class_name = class_names[sem_label] if sem_label < len(class_names) else f'Class_{sem_label}'
            
            instance_stats.append({
                'instance_label': int(inst_label),
                'semantic_label': int(sem_label),
                'instance_id': int(inst_id),
                'name': f"{class_name}_{inst_id}" if inst_id > 0 else class_name,
                'count': int(point_count)
            })
        
        # 按点数排序
        instance_stats.sort(key=lambda x: x['count'], reverse=True)
        print(f"  ✓ Found {len(instance_stats)} instances from {len(np.unique(pred_labels[pred_labels >= 0]))} semantic classes")
    else:
        # 不使用实例分割，按语义类别分组
        unique_pred_labels = np.unique(pred_labels[pred_labels >= 0])
        instance_stats = []
        
        for pred_label in unique_pred_labels:
            mask = pred_labels == pred_label
            point_count = mask.sum()
            if point_count < 10:
                continue
            
            class_name = class_names[pred_label] if pred_label < len(class_names) else f'Class_{pred_label}'
            instance_stats.append({
                'instance_label': int(pred_label),
                'semantic_label': int(pred_label),
                'instance_id': 0,
                'name': class_name,
                'count': int(point_count)
            })
        
        instance_stats.sort(key=lambda x: x['count'], reverse=True)
        instance_labels = pred_labels.copy()
    
    # ========================================================================
    # 1. 创建包含整个场景的主HTML文件（带对象列表）
    # ========================================================================
    
    # 创建主HTML（整个场景+对象选择器）
    sample_size = min(len(points), 200000)
    sample_idx = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[sample_idx]
    sample_pred = pred_labels[sample_idx]
    sample_gt = gt_labels[sample_idx] if gt_labels is not None else np.full(sample_size, -1)
    
    # 创建带下拉菜单的图表
    fig_main = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Full Scene - Ground Truth', 'Full Scene - Prediction'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # Ground Truth（整个场景）
    valid_gt = sample_gt >= 0
    if valid_gt.sum() > 0:
        gt_colors = color_palette[sample_gt[valid_gt]]
        fig_main.add_trace(
            go.Scatter3d(
                x=sample_points[valid_gt, 0],
                y=sample_points[valid_gt, 1],
                z=sample_points[valid_gt, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in gt_colors],
                ),
                name='GT - Full Scene',
                hovertemplate='<b>GT</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in sample_gt[valid_gt]],
                visible=True
            ),
            row=1, col=1
        )
    
    # Prediction（整个场景）
    pred_colors = color_palette[sample_pred]
    fig_main.add_trace(
        go.Scatter3d(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            z=sample_points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_colors],
            ),
            name='Pred - Full Scene',
            hovertemplate='<b>Prediction</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in sample_pred],
            visible=True
        ),
        row=1, col=2
    )
    
    # 为每个实例添加trace（初始隐藏）
    trace_idx_start = len(fig_main.data)
    
    for obj_info in instance_stats:  # 显示所有实例
        inst_label = obj_info['instance_label']
        inst_name = obj_info['name']
        sem_label = obj_info['semantic_label']
        
        mask = instance_labels == inst_label
        obj_points = points[mask]
        obj_gt = gt_labels[mask] if gt_labels is not None else np.full(mask.sum(), -1)
        obj_pred = pred_labels[mask]
        
        # 部件使用全部点，不下采样
        
        # GT trace for this instance
        valid_gt_obj = obj_gt >= 0
        if valid_gt_obj.sum() > 0:
            gt_colors_obj = color_palette[obj_gt[valid_gt_obj]]
            fig_main.add_trace(
                go.Scatter3d(
                    x=obj_points[valid_gt_obj, 0],
                    y=obj_points[valid_gt_obj, 1],
                    z=obj_points[valid_gt_obj, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in gt_colors_obj],
                    ),
                    name=f'GT - {inst_name}',
                    hovertemplate=f'<b>GT - {inst_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}',
                    visible=False
                ),
                row=1, col=1
            )
        else:
            # 添加空trace占位
            fig_main.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', visible=False), row=1, col=1)
        
        # Pred trace for this instance
        pred_colors_obj = color_palette[obj_pred]
        fig_main.add_trace(
            go.Scatter3d(
                x=obj_points[:, 0],
                y=obj_points[:, 1],
                z=obj_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_colors_obj],
                ),
                name=f'Pred - {inst_name}',
                hovertemplate=f'<b>Pred - {inst_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}',
                visible=False
            ),
            row=1, col=2
        )
    
    # 创建下拉菜单
    buttons = [
        dict(
            label='Full Scene',
            method='update',
            args=[
                {'visible': [True, True] + [False] * (len(fig_main.data) - 2)},
                {'title.text': f'Full Scene (Sampled: {sample_size:,}/{len(points):,} points)'}
            ]
        )
    ]
    
    # 为每个实例创建按钮
    for i, obj_info in enumerate(instance_stats):
        inst_name = obj_info['name']
        point_count = obj_info['count']
        
        # 计算该实例对应的trace索引
        obj_trace_idx = trace_idx_start + i * 2
        
        visible_list = [False] * len(fig_main.data)
        visible_list[obj_trace_idx] = True      # GT trace
        visible_list[obj_trace_idx + 1] = True  # Pred trace
        
        buttons.append(
            dict(
                label=f'{inst_name} ({point_count:,} pts)',
                method='update',
                args=[
                    {'visible': visible_list},
                    {'title.text': f'Instance: {inst_name} ({point_count:,} points)'}
                ]
            )
        )
    
    # 更新布局
    fig_main.update_layout(
        title=f'Full Scene (Sampled: {sample_size:,}/{len(points):,} points)',
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.01,
                xanchor='left',
                y=1.15,
                yanchor='top',
                bgcolor='lightgray',
                bordercolor='gray',
                borderwidth=2
            )
        ],
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=False,
        height=700,
        margin=dict(t=100)
    )
    
    # 保存主HTML
    main_html = save_path.parent / f"{save_path.stem}_interactive_all.html"
    fig_main.write_html(str(main_html))
    
    print(f"  ✓ Interactive HTML saved: {main_html.name}")
    print(f"    - Full scene + {len(instance_stats)} instances")
    print(f"    - Use dropdown menu to switch between full scene and individual instances")
    
    # ========================================================================
    # 2. 为每个实例创建独立HTML（带指标）
    # ========================================================================
    for obj_info in instance_stats:  # 为所有实例创建独立文件
        inst_label = obj_info['instance_label']
        inst_name = obj_info['name']
        sem_label = obj_info['semantic_label']
        
        # 找到该实例的点
        instance_mask = instance_labels == inst_label
        instance_points = points[instance_mask]
        
        # 计算该实例的3D包围盒
        bbox_min = instance_points.min(axis=0)
        bbox_max = instance_points.max(axis=0)
        
        # 添加padding扩展包围盒，用于显示物件周围的环境上下文
        bbox_min = bbox_min - bbox_padding
        bbox_max = bbox_max + bbox_padding
        
        # 提取包围盒内的所有点（不仅是该实例的点）
        bbox_mask = (
            (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
            (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
            (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
        )
        
        obj_points = points[bbox_mask]
        obj_gt = gt_labels[bbox_mask] if gt_labels is not None else np.full(bbox_mask.sum(), -1)
        obj_pred = pred_labels[bbox_mask]
        
        # 记录哪些点属于目标实例（用于高亮显示）
        obj_instance_mask = instance_labels[bbox_mask] == inst_label
        
        # 部件使用全部点，不下采样
        
        # ========================================================================
        # 计算该实例的评估指标（只针对目标实例的点）
        # ========================================================================
        # 只对目标实例的点计算指标
        inst_gt = obj_gt[obj_instance_mask]
        inst_pred = obj_pred[obj_instance_mask]
        valid_gt_inst = inst_gt >= 0
        
        if valid_gt_inst.sum() > 0:
            # 准确率（只针对目标实例）
            accuracy = (inst_gt[valid_gt_inst] == inst_pred[valid_gt_inst]).sum() / valid_gt_inst.sum()
            
            # 计算IoU（仅针对该语义类别）
            gt_mask = inst_gt[valid_gt_inst] == sem_label
            pred_mask = inst_pred[valid_gt_inst] == sem_label
            
            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            
            # 精确率和召回率
            tp = (gt_mask & pred_mask).sum()
            fp = (~gt_mask & pred_mask).sum()
            fn = (gt_mask & ~pred_mask).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics_text = (
                f"<b>Instance Metrics (Target Object):</b><br>"
                f"Overall Accuracy: {accuracy*100:.2f}%<br>"
                f"IoU (Class {sem_label}): {iou*100:.2f}%<br>"
                f"Precision: {precision*100:.2f}%<br>"
                f"Recall: {recall*100:.2f}%<br>"
                f"F1 Score: {f1_score*100:.2f}%<br>"
                f"Target Instance Points: {obj_instance_mask.sum():,}<br>"
                f"BBox Total Points: {len(obj_gt):,}<br>"
            )
        else:
            accuracy = 0.0
            iou = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            metrics_text = (
                f"<b>Instance Metrics:</b><br>"
                f"No ground truth labels available<br>"
                f"Target Instance Points: {obj_instance_mask.sum():,}<br>"
                f"BBox Total Points: {len(obj_pred):,}<br>"
            )
        
        # 跳过低准确率的实例
        if accuracy < min_accuracy_threshold:
            continue
        
        # 创建独立的双视图
        fig_obj = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Ground Truth (BBox Context)',
                f'Prediction (BBox Context)'
            ),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # GT视图 - 区分目标实例和周围环境
        valid_gt_obj = obj_gt >= 0
        if valid_gt_obj.sum() > 0:
            # 周围环境点（半透明灰色）
            env_mask = valid_gt_obj & ~obj_instance_mask
            if env_mask.sum() > 0:
                env_colors = color_palette[obj_gt[env_mask]]
                # 降低环境点的饱和度（变灰）
                env_colors_gray = env_colors * 0.3 + 0.5  # 使颜色变淡
                fig_obj.add_trace(
                    go.Scatter3d(
                        x=obj_points[env_mask, 0],
                        y=obj_points[env_mask, 1],
                        z=obj_points[env_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in env_colors_gray],
                            opacity=0.3
                        ),
                        name='GT - Environment',
                        hovertemplate='<b>GT (Env)</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                        text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in obj_gt[env_mask]]
                    ),
                    row=1, col=1
                )
            
            # 目标实例点（高亮显示）
            target_gt_mask = valid_gt_obj & obj_instance_mask
            if target_gt_mask.sum() > 0:
                gt_colors_target = color_palette[obj_gt[target_gt_mask]]
                fig_obj.add_trace(
                    go.Scatter3d(
                        x=obj_points[target_gt_mask, 0],
                        y=obj_points[target_gt_mask, 1],
                        z=obj_points[target_gt_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in gt_colors_target],
                        ),
                        name='GT - Target',
                        hovertemplate='<b>GT (Target)</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                        text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in obj_gt[target_gt_mask]]
                    ),
                    row=1, col=1
                )
        
        # Pred视图 - 区分目标实例和周围环境
        # 周围环境点（半透明灰色）
        env_mask_pred = ~obj_instance_mask
        if env_mask_pred.sum() > 0:
            pred_colors_env = color_palette[obj_pred[env_mask_pred]]
            pred_colors_env_gray = pred_colors_env * 0.3 + 0.5
            fig_obj.add_trace(
                go.Scatter3d(
                    x=obj_points[env_mask_pred, 0],
                    y=obj_points[env_mask_pred, 1],
                    z=obj_points[env_mask_pred, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_colors_env_gray],
                        opacity=0.3
                    ),
                    name='Pred - Environment',
                    hovertemplate='<b>Pred (Env)</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                    text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in obj_pred[env_mask_pred]]
                ),
                row=1, col=2
            )
        
        # 目标实例点（高亮显示）
        if obj_instance_mask.sum() > 0:
            pred_colors_target = color_palette[obj_pred[obj_instance_mask]]
            fig_obj.add_trace(
                go.Scatter3d(
                    x=obj_points[obj_instance_mask, 0],
                    y=obj_points[obj_instance_mask, 1],
                    z=obj_points[obj_instance_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_colors_target],
                    ),
                    name='Pred - Target',
                    hovertemplate='<b>Pred (Target)</b><br>Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                    text=[class_names[l] if l < len(class_names) else f'Class_{l}' for l in obj_pred[obj_instance_mask]]
                ),
                row=1, col=2
            )
        
        # 构建标题文本
        title_text = (
            f'<b>{inst_name}</b> (Target: {obj_instance_mask.sum():,} pts | BBox: {len(obj_points):,} pts)<br>'
            f'<span style="font-size:12px">'
            f'Accuracy: {accuracy*100:.1f}% | '
            f'IoU: {iou*100:.1f}% | '
            f'Precision: {precision*100:.1f}% | '
            f'Recall: {recall*100:.1f}% | '
            f'F1: {f1_score*100:.1f}%'
            f'</span>'
        )
        
        fig_obj.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center'
            ),
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            showlegend=False,
            height=600,
            annotations=[
                dict(
                    text=metrics_text,
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=5,
                    font=dict(size=11, family='monospace')
                )
            ]
        )
        
        obj_html = save_path.parent / f"{save_path.stem}_{inst_name.replace(' ', '_')}.html"
        fig_obj.write_html(str(obj_html))
    
    # 统计生成的文件数量
    created_files = sum(1 for obj_info in instance_stats 
                        if obj_info.get('accuracy', 0) >= min_accuracy_threshold or 
                        'accuracy' not in obj_info)
    print(f"  ✓ Created individual HTML files for instances with accuracy >= {min_accuracy_threshold*100:.0f}%")


# ============================================================================
# 评估与统计
# ============================================================================
def evaluate_and_visualize(
    inference: ModelInference,
    zarr_files: List[Path],
    class_names: List[str],
    output_dir: str,
    max_scenes: int = 5,
    interactive: bool = False,
    max_vis_points: int = 500000,
    min_accuracy_threshold: float = 0.3,
    bbox_padding: float = 0.5
):
    """评估模型并生成可视化"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = SegmentationMetrics(
        num_classes=inference.num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    
    print(f"\n{'='*80}")
    print(f"🔍 Evaluating on {min(len(zarr_files), max_scenes)} scenes")
    print(f"{'='*80}\n")
    
    per_class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(inference.num_classes)}
    
    for scene_idx, zarr_path in enumerate(tqdm(zarr_files[:max_scenes], desc="Processing")):
        scene_id = zarr_path.stem.replace('_dino_patch_level', '')
        
        try:
            # 加载数据
            root = zarr.open(str(zarr_path), mode='r')
            
            if 'points' not in root or 'semantic_labels' not in root:
                print(f"⚠️  Skipping {scene_id} (missing data)")
                continue
            
            points = root['points'][:].astype(np.float32)
            gt_labels = root['semantic_labels'][:].astype(np.int32)
            
            # 加载或生成颜色
            if 'colors' in root:
                colors = root['colors'][:].astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = np.full((len(points), 3), 0.5, dtype=np.float32)
            
            # 加载或生成法向量
            if 'normals' in root:
                normals = root['normals'][:].astype(np.float32)
            else:
                normals = np.zeros((len(points), 3), dtype=np.float32)
                normals[:, 2] = 1.0
            
            root.store.close() if hasattr(root.store, 'close') else None
            
            # 限制点数（加速可视化）
            if len(points) > max_vis_points:
                sample_idx = np.random.choice(len(points), max_vis_points, replace=False)
                points_vis = points[sample_idx]
                colors_vis = colors[sample_idx]
                normals_vis = normals[sample_idx]
                gt_labels_vis = gt_labels[sample_idx]
            else:
                points_vis = points
                colors_vis = colors
                normals_vis = normals
                gt_labels_vis = gt_labels
            
            # 推理
            print(f"\n[{scene_idx+1}/{max_scenes}] {scene_id}")
            print(f"  Points: {len(points_vis):,}")
            
            pred_labels, pred_probs = inference.predict(points_vis, colors_vis, normals_vis)
            
            # 更新指标
            metrics.update(pred_labels, gt_labels_vis)
            
            # 统计每个类别
            valid_mask = gt_labels_vis >= 0
            for cls_id in range(inference.num_classes):
                gt_mask = gt_labels_vis == cls_id
                pred_mask = pred_labels == cls_id
                
                tp = ((gt_mask & pred_mask) & valid_mask).sum()
                fp = ((~gt_mask & pred_mask) & valid_mask).sum()
                fn = ((gt_mask & ~pred_mask) & valid_mask).sum()
                
                per_class_stats[cls_id]['tp'] += tp
                per_class_stats[cls_id]['fp'] += fp
                per_class_stats[cls_id]['fn'] += fn
            
            # 计算当前场景指标
            valid_labels = gt_labels_vis[valid_mask]
            valid_preds = pred_labels[valid_mask]
            
            if len(valid_labels) > 0:
                acc = (valid_labels == valid_preds).sum() / len(valid_labels)
                print(f"  Accuracy: {acc*100:.2f}%")
            
            # 生成2D可视化
            save_path = output_dir / f"{scene_id}_visualization.png"
            visualize_predictions_matplotlib(
                points_vis, gt_labels_vis, pred_labels,
                class_names, str(save_path), inference.num_classes
            )
            
            # 生成HTML交互式可视化（按对象分组）
            html_path = output_dir / f"{scene_id}_interactive.html"
            visualize_predictions_plotly_html(
                points_vis, gt_labels_vis, pred_labels,
                class_names, str(html_path), inference.num_classes,
                max_points_per_object=50000,
                use_instance_segmentation=True,  # 启用实例分割
                clustering_eps=0.3,  # 可调整聚类参数
                min_accuracy_threshold=min_accuracy_threshold,
                bbox_padding=bbox_padding
            )
            
            # 可选：交互式3D可视化
            if interactive and HAS_OPEN3D:
                visualize_predictions_open3d(
                    points_vis, gt_labels_vis, pred_labels,
                    class_names, inference.num_classes,
                    window_name=f"Scene {scene_id}"
                )
        
        except Exception as e:
            print(f"❌ Error processing {scene_id}: {e}")
            continue
    
    # ========================================================================
    # 最终报告
    # ========================================================================
    print(f"\n{'='*80}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    print("Overall Metrics:")
    print(f"  Overall Accuracy: {metrics.get_overall_accuracy()*100:.2f}%")
    print(f"  Mean IoU: {metrics.get_mean_iou()*100:.2f}%\n")
    
    print("Per-Class Performance (Top 10 by IoU):")
    print(f"{'Rank':<6} {'Class':<30} {'IoU':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)
    
    class_ious = []
    for cls_id in range(inference.num_classes):
        stats = per_class_stats[cls_id]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            class_ious.append((cls_id, iou, precision, recall))
    
    # 按IoU排序
    class_ious.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (cls_id, iou, precision, recall) in enumerate(class_ious[:10], 1):
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
        print(f"{rank:<6} {class_name:<30} {iou*100:>6.2f}%   {precision*100:>6.2f}%      {recall*100:>6.2f}%")
    
    # 保存详细报告
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("USDNet Model Evaluation Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Checkpoint: {inference.checkpoint_path}\n")
        f.write(f"Scenes evaluated: {min(len(zarr_files), max_scenes)}\n")
        f.write(f"Number of classes: {inference.num_classes}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Overall Accuracy: {metrics.get_overall_accuracy()*100:.2f}%\n")
        f.write(f"  Mean IoU: {metrics.get_mean_iou()*100:.2f}%\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write(f"{'Rank':<6} {'Class':<40} {'IoU':<10} {'Precision':<12} {'Recall':<10} {'Support':<10}\n")
        f.write("-" * 100 + "\n")
        
        for rank, (cls_id, iou, precision, recall) in enumerate(class_ious, 1):
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
            stats = per_class_stats[cls_id]
            support = stats['tp'] + stats['fn']
            f.write(f"{rank:<6} {class_name:<40} {iou*100:>6.2f}%   {precision*100:>6.2f}%      {recall*100:>6.2f}%     {support:<10}\n")
    
    print(f"\n✓ Report saved: {report_path}")
    print(f"✓ Visualizations saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize USDNet finetuned model predictions")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to finetuned checkpoint (e.g., ./c6/finetune/ckpt_best.pt)')
    parser.add_argument('--zarr_root', type=str, required=True,
                       help='Path to zarr data directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations_finetune',
                       help='Directory to save visualizations')
    parser.add_argument('--max_scenes', type=int, default=5,
                       help='Maximum number of scenes to visualize')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive 3D visualization with Open3D')
    parser.add_argument('--test_only', action='store_true',
                       help='Only use scenes not in training (last 20%%)')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None,
                       help='Specific scene IDs to visualize (e.g., 00777c41d4 01ce24e652)')
    parser.add_argument('--scene_list', type=str, default=None,
                       help='Path to text file with scene IDs (one per line)')
    parser.add_argument('--max_points_per_object', type=int, default=50000,
                       help='Maximum points per object in HTML visualization')
    parser.add_argument('--max_vis_points', type=int, default=500000,
                       help='Maximum points for full scene visualization (default: 500000)')
    parser.add_argument('--enable_instance_seg', action='store_true',
                       help='Enable instance segmentation (split objects into separate instances)')
    parser.add_argument('--clustering_eps', type=float, default=0.3,
                       help='DBSCAN epsilon for instance segmentation (default: 0.3m)')
    parser.add_argument('--min_accuracy', type=float, default=0.3,
                       help='Minimum accuracy threshold for individual instance visualization (default: 0.3 = 30%%)')
    parser.add_argument('--bbox_padding', type=float, default=0.5,
                       help='Padding around object bounding box in meters (default: 0.5m)')
    
    args = parser.parse_args()
    
    # 加载全局类别映射
    num_classes, class_names, label_map = GlobalLabelReader.load_global_mapping(args.zarr_root)
    
    if num_classes == 0:
        print("❌ Cannot load global label mapping!")
        return
    
    # 加载模型
    inference = ModelInference(args.checkpoint, device=args.device)
    
    # 加载数据文件
    zarr_root = Path(args.zarr_root)
    all_zarr_files = sorted(zarr_root.glob("*_dino_patch_level.zarr"))
    
    if len(all_zarr_files) == 0:
        print("❌ No zarr files found!")
        return
    
    # 选择测试场景
    if args.scene_ids or args.scene_list:
        # 指定特定场景
        specified_scenes = set()
        
        # 从命令行参数读取
        if args.scene_ids:
            specified_scenes.update(args.scene_ids)
        
        # 从文件读取
        if args.scene_list:
            with open(args.scene_list, 'r') as f:
                for line in f:
                    scene_id = line.strip()
                    if scene_id and not scene_id.startswith('#'):
                        specified_scenes.add(scene_id)
        
        # 过滤zarr文件
        test_files = []
        for zarr_file in all_zarr_files:
            scene_id = zarr_file.stem.replace('_dino_patch_level', '')
            if scene_id in specified_scenes:
                test_files.append(zarr_file)
        
        if len(test_files) == 0:
            print(f"❌ No matching scenes found for: {specified_scenes}")
            print(f"Available scenes (first 10): {[f.stem.replace('_dino_patch_level', '') for f in all_zarr_files[:10]]}")
            return
        
        print(f"📊 Using specified scenes: {len(test_files)}")
        for f in test_files:
            print(f"  - {f.stem.replace('_dino_patch_level', '')}")
    
    elif args.test_only and len(all_zarr_files) >= 5:
        split_idx = int(len(all_zarr_files) * 0.8)
        test_files = all_zarr_files[split_idx:]
        print(f"📊 Using test set: {len(test_files)} scenes (last 20%)")
    else:
        test_files = all_zarr_files
        print(f"📊 Using all scenes: {len(test_files)}")
    
    # 评估并可视化
    evaluate_and_visualize(
        inference=inference,
        zarr_files=test_files,
        class_names=class_names,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
        interactive=args.interactive,
        max_vis_points=args.max_vis_points,
        min_accuracy_threshold=args.min_accuracy,
        bbox_padding=args.bbox_padding
    )
    
    print("\n✨ Visualization complete!")


if __name__ == "__main__":
    main()
