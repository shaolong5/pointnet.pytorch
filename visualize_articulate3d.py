"""
================================================================================
Articulate3D Comprehensive Visualization Tool
================================================================================

Features:
1. Semantic Segmentation Visualization (USDNet)
2. Movable Part Segmentation Visualization (Articulate3D Movable)
3. Interactable Part Segmentation Visualization (Articulate3D Interactable)
4. Motion Information Visualization (articulation origin, axis vectors)
5. Full Scene HTML Visualization + Individual Part HTML Visualization
6. Per-part metrics, semantics, and motion information annotation

Supported Data Formats:
- Zarr format (original semantic segmentation data)
- NPY format (Articulate3D challenge data)
"""

import os
import sys
import json
import argparse
import h5py
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

# Visualization libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not installed. Install with: pip install plotly")

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    print("⚠️  Zarr not installed. Install with: pip install zarr")


# ============================================================================
# Zarr Data Loading (for original semantic segmentation data)
# ============================================================================
class ZarrDataLoader:
    """Zarr format data loader (for USDNet semantic segmentation data)"""
    
    def __init__(self, zarr_root: str):
        self.zarr_root = Path(zarr_root)
        
        # Load global label mapping
        mapping_file = self.zarr_root / "global_label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            self.num_classes = mapping_data.get('num_classes', 0)
            self.class_names = mapping_data.get('class_names', [])
            self.label_map = mapping_data.get('mapping', {})
        else:
            self.num_classes = 0
            self.class_names = []
            self.label_map = {}
        
        # Find all zarr files
        self.zarr_files = sorted(self.zarr_root.glob("*_dino_patch_level.zarr"))
        print(f"✓ Found {len(self.zarr_files)} zarr files, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.zarr_files)
    
    def get_scene_ids(self) -> List[str]:
        return [f.stem.replace('_dino_patch_level', '') for f in self.zarr_files]
    
    def load_scene(self, idx: int) -> Dict[str, Any]:
        """Load zarr scene data"""
        zarr_path = self.zarr_files[idx]
        scene_id = zarr_path.stem.replace('_dino_patch_level', '')
        
        root = zarr.open(str(zarr_path), mode='r')
        
        # Load point cloud data
        points = root['points'][:].astype(np.float32)
        
        # Load colors
        if 'colors' in root:
            colors = root['colors'][:].astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.full((len(points), 3), 0.5, dtype=np.float32)
        
        # Load normals
        if 'normals' in root:
            normals = root['normals'][:].astype(np.float32)
        else:
            normals = np.zeros((len(points), 3), dtype=np.float32)
            normals[:, 2] = 1.0
        
        # Load semantic labels
        if 'semantic_labels' in root:
            semantic_labels = root['semantic_labels'][:].astype(np.int32)
        else:
            semantic_labels = np.full(len(points), -1, dtype=np.int32)
        
        # Load DINO features (if available)
        dino_features = None
        if 'dino_features' in root:
            dino_features = root['dino_features'][:].astype(np.float32)
        
        return {
            'scene_id': scene_id,
            'coords': points,
            'colors': colors,
            'normals': normals,
            'semantic_labels': semantic_labels,
            'dino_features': dino_features,
        }


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    output_dir: str = "./visualizations_articulate3d"
    device: str = "cuda:0"
    voxel_size: float = 0.02
    max_points_vis: int = 100000
    clustering_eps: float = 0.3
    clustering_min_samples: int = 50
    show_motion_arrows: bool = True
    arrow_scale: float = 0.5
    min_accuracy: float = 0.0  # Instance accuracy threshold
    bbox_padding: float = 0.5  # Bounding box padding (meters)
    json_dir: Optional[str] = None  # Articulate3D original JSON file directory


# ============================================================================
# Articulate3D JSON Info Loading
# ============================================================================
def load_articulate3d_json_info(json_dir: str, scene_id: str) -> Dict[str, Any]:
    """
    Load complete info from Articulate3D original JSON files

    
    Args:
        json_dir: Directory containing JSON files
        scene_id: Scene ID (e.g., '0a7cc12c0e')
        
    Returns:
        {
            'parts_info': {pid: {'label': str, 'group': str}},  # Interactable type info
            'articulation_ranges': {pid: {'rangeMin': float, 'rangeMax': float, 'type': str}}
        }
    """
    json_path = Path(json_dir) / f"{scene_id}_artic.json"
    
    result = {
        'parts_info': {},
        'articulation_ranges': {}
    }
    
    if not json_path.exists():
        print(f"  ⚠️ JSON file not found: {json_path}")
        return result
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parse parts info (contains group field for interactable type)
        for part in data['data']['parts']:
            pid = part['pid']
            result['parts_info'][pid] = {
                'label': part.get('label', 'unknown'),
                'group': part.get('group', 'others')  # "doors/windows", "handles movable", "switches", etc.
            }
        
        # Parse articulations info (contains motion range)
        for artic in data['data']['articulations']:
            pid = artic['pid']
            result['articulation_ranges'][pid] = {
                'rangeMin': artic.get('rangeMin', 0),
                'rangeMax': artic.get('rangeMax', 0),
                'type': artic.get('type', 'unknown')  # 'rotation' or 'translation'
            }
        
        print(f"  ✓ Loaded JSON info: {len(result['parts_info'])} parts, {len(result['articulation_ranges'])} articulations")
        return result
        
    except Exception as e:
        print(f"  ⚠️ Failed to load JSON {json_path}: {e}")
        return result


# ============================================================================
# Color Schemes
# ============================================================================
def generate_color_palette(num_classes: int, seed: int = 42) -> np.ndarray:
    """Generate high-contrast color palette"""
    np.random.seed(seed)
    
    # Predefined high-contrast colors
    base_colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 255, 128],    # Teal
        [255, 0, 128],    # Pink
        [128, 255, 0],    # Lime
        [0, 128, 255],    # Sky Blue
    ]
    
    colors = base_colors[:min(num_classes, len(base_colors))]
    
    while len(colors) < num_classes:
        color = np.random.randint(50, 256, 3).tolist()
        colors.append(color)
    
    return np.array(colors[:num_classes], dtype=np.float32) / 255.0


# Articulate3D specific colors
MOVABLE_COLORS = {
    0: [0.5, 0.5, 0.5],     # background - Gray
    1: [1.0, 0.0, 0.0],     # rotation - Red
    2: [0.0, 0.0, 1.0],     # translation - Blue
}

INTERACTABLE_COLORS = {
    0: [0.5, 0.5, 0.5],     # non-interactable - Gray
    1: [0.0, 1.0, 0.0],     # interactable - Green
}

MOVABLE_CLASS_NAMES = ['background', 'rotation', 'translation']
INTERACTABLE_CLASS_NAMES = ['non-interactable', 'interactable']


# ============================================================================
# Majority Vote Classifier (Semantic Segmentation)
# ============================================================================
class MajorityVoteClassifier:
    """
    Majority vote based semantic segmentation classifier
    
    For each instance/part, count all point semantic labels and select the most frequent class.
    """
    
    @staticmethod
    def classify_instance(semantic_labels: np.ndarray) -> Tuple[int, Dict[int, int]]:
        """
        Classify a single instance by majority vote
        
        Args:
            semantic_labels: (N,) semantic labels for all points in this instance
            
        Returns:
            majority_class: the class selected by majority vote
            vote_counts: vote count for each class {class_id: count}
        """
        counter = Counter(semantic_labels)
        majority_class = counter.most_common(1)[0][0]
        return int(majority_class), dict(counter)
    
    @staticmethod
    def classify_all_instances(
        semantic_labels: np.ndarray, 
        instance_labels: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """
        Classify all instances by majority vote
        
        Args:
            semantic_labels: (N,) semantic labels for all points
            instance_labels: (N,) instance labels for all points
            
        Returns:
            results: {instance_id: {'majority_class': int, 'votes': dict, 'confidence': float}}
        """
        results = {}
        unique_instances = np.unique(instance_labels)
        
        for inst_id in unique_instances:
            mask = instance_labels == inst_id
            inst_sem_labels = semantic_labels[mask]
            
            majority_class, votes = MajorityVoteClassifier.classify_instance(inst_sem_labels)
            
            total_points = len(inst_sem_labels)
            confidence = votes[majority_class] / total_points if total_points > 0 else 0.0
            
            results[int(inst_id)] = {
                'majority_class': majority_class,
                'votes': votes,
                'total_points': total_points,
                'confidence': confidence,  # Majority class ratio
            }
        
        return results
    
    @staticmethod
    def propagate_instance_labels(
        instance_labels: np.ndarray,
        instance_classifications: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Propagate instance-level classification to all points
        
        Args:
            instance_labels: (N,) Instance labels for all points
            instance_classifications: Return result from classify_all_instances
            
        Returns:
            propagated_labels: (N,) Classification label for each point (based on majority vote of its instance)
        """
        propagated = np.zeros_like(instance_labels)
        
        for inst_id, info in instance_classifications.items():
            mask = instance_labels == inst_id
            propagated[mask] = info['majority_class']
        
        return propagated
    
    @staticmethod
    def get_scene_classification(semantic_labels: np.ndarray) -> Dict[str, Any]:
        """
        Perform majority vote classification for the entire scene
        
        Args:
            semantic_labels: (N,) Semantic labels for all points in the scene
            
        Returns:
            {'majority_class': int, 'votes': dict, 'confidence': float}
        """
        majority_class, votes = MajorityVoteClassifier.classify_instance(semantic_labels)
        total_points = len(semantic_labels)
        confidence = votes[majority_class] / total_points if total_points > 0 else 0.0
        
        return {
            'majority_class': majority_class,
            'votes': votes,
            'total_points': total_points,
            'confidence': confidence,
        }
    
    @staticmethod
    def format_votes_report(
        votes: Dict[int, int], 
        class_names: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Format votes report
        
        Args:
            votes: {class_id: count}
            class_names: List of class names
            top_k: Show top k classes
            
        Returns:
            Formatted string report
        """
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(votes.values())
        
        lines = []
        for i, (cls_id, count) in enumerate(sorted_votes[:top_k]):
            pct = count / total * 100 if total > 0 else 0
            if class_names and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = f"class_{cls_id}"
            lines.append(f"  {i+1}. {name} ({cls_id}): {count:,} pts ({pct:.1f}%)")
        
        if len(sorted_votes) > top_k:
            lines.append(f"  ... and {len(sorted_votes) - top_k} more classes")
        
        return "\n".join(lines)


# ============================================================================
# Model Import
# ============================================================================
# Import Articulate3D model
try:
    from train_articulate3d import (
        ArticulateUSDNet,
        ArticulateMetrics,
        load_h5_articulation,
        load_yaml,
        CLASS_LABELS as ARTICULATE_CLASS_LABELS,
    )
    HAS_ARTICULATE_MODEL = True
except ImportError:
    HAS_ARTICULATE_MODEL = False
    print("⚠️  Cannot import ArticulateUSDNet from train_articulate3d.py")

try:
    from train_interactable3d import (
        InteractableUSDNet,
        InteractableMetrics,
    )
    HAS_INTERACTABLE_MODEL = True
except ImportError:
    HAS_INTERACTABLE_MODEL = False
    print("⚠️  Cannot import InteractableUSDNet from train_interactable3d.py")


# ============================================================================
# Model Inference Classes
# ============================================================================
class ArticulateModelInference:
    """Articulate3D Movable Parts Model Inference"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        if not HAS_ARTICULATE_MODEL:
            raise RuntimeError("ArticulateUSDNet not available")
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"📦 Loading Articulate3D Movable checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Read configuration
        config = ckpt.get('config', {})
        self.num_classes = config.get('num_classes', 3)
        self.voxel_size = config.get('voxel_size', 0.02)
        self.feature_dim = config.get('feature_dim_3d', 256)
        
        print(f"  - Classes: {self.num_classes} ({MOVABLE_CLASS_NAMES})")
        print(f"  - Voxel size: {self.voxel_size}")
        print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
        
        # Initialize model
        self.model = ArticulateUSDNet(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            dropout=config.get('dropout', 0.1),
        ).to(device)
        
        # Load weights (use strict=False for backward compatibility with old checkpoints without range_head)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        
        # Check if range_head weights are missing
        has_range_head = any('range_head' in k for k in state_dict.keys())
        if not has_range_head:
            print("  ⚠️ Checkpoint missing range_head weights (old format), loading with strict=False")
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)} (range_head will use random init)")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {unexpected}")
        
        self.model.eval()
        print("✓ Articulate3D Movable model loaded!")
    
    @torch.no_grad()
    def predict(self, coords: np.ndarray, features: np.ndarray, 
                sem_gt: np.ndarray = None, max_points: int = 100000) -> Dict[str, np.ndarray]:
        """
        Inference - use the same stratified sampling strategy as training
        
        Args:
            coords: (N, 3) Point coordinates
            features: (N, 9) Features [colors, normals, coords]
            sem_gt: (N,) Semantic GT labels for stratified sampling (if provided)
            max_points: Maximum number of sampled points (100,000 during training)
        
        Returns:
            dict with:
                - seg_labels: (N,) Semantic labels
                - seg_probs: (N, 3) Segmentation probabilities
                - origin_pred: (N, 3) Articulation origin prediction
                - axis_pred: (N, 3) Articulation axis prediction (unit vector)
        """
        N = len(coords)
        
        # If point count exceeds max_points, use stratified or random sampling
        if N > max_points:
            if sem_gt is not None:
                # Stratified sampling (consistent with training) - preserve more rare class points
                unique_labels = np.unique(sem_gt)
                selected_idx = []
                points_per_class = max_points // len(unique_labels)
                
                for label in unique_labels:
                    label_mask = sem_gt == label
                    label_idx = np.where(label_mask)[0]
                    n_select = min(len(label_idx), points_per_class)
                    selected_idx.extend(np.random.choice(label_idx, n_select, replace=False))
                
                # Fill remaining quota
                remaining = max_points - len(selected_idx)
                if remaining > 0:
                    all_idx = np.arange(N)
                    remaining_idx = np.setdiff1d(all_idx, selected_idx)
                    if len(remaining_idx) > 0:
                        extra_idx = np.random.choice(remaining_idx, min(remaining, len(remaining_idx)), replace=False)
                        selected_idx.extend(extra_idx)
                
                sample_idx = np.array(selected_idx)
            else:
                # Random sampling when no GT available
                sample_idx = np.random.choice(N, max_points, replace=False)
            
            coords_sampled = coords[sample_idx]
            features_sampled = features[sample_idx]
        else:
            sample_idx = np.arange(N)
            coords_sampled = coords
            features_sampled = features
        
        # Voxelization
        voxel_coords = np.floor(coords_sampled / self.voxel_size).astype(np.int32)
        unique_coords, unique_indices, inverse_indices = np.unique(
            voxel_coords, axis=0, return_index=True, return_inverse=True
        )
        
        # Add batch dimension
        batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])
        
        # Get unique point features
        features_unique = features_sampled[unique_indices]
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coords_with_batch).int().to(self.device)
        features_tensor = torch.from_numpy(features_unique).float().to(self.device)
        
        # Inference
        x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
        outputs = self.model(x)
        
        # Get predictions
        seg_logits = outputs['seg_logits'].features.cpu().numpy()
        origin_pred = outputs['origin_pred'].features.cpu().numpy()
        axis_pred = outputs['axis_pred'].features.cpu().numpy()
        
        # Get range predictions if available
        range_pred = None
        if 'range_pred' in outputs:
            range_pred = outputs['range_pred'].features.cpu().numpy()  # (N, 2) [rangeMin, rangeMax]
        
        # Normalize axis vectors
        axis_norms = np.linalg.norm(axis_pred, axis=1, keepdims=True)
        axis_pred = axis_pred / np.clip(axis_norms, 1e-6, None)
        
        # Map back to sampled point cloud
        seg_probs_voxel = F.softmax(torch.from_numpy(seg_logits), dim=-1).numpy()
        seg_labels_voxel = np.argmax(seg_logits, axis=-1)
        
        seg_labels_sampled = seg_labels_voxel[inverse_indices]
        seg_probs_sampled = seg_probs_voxel[inverse_indices]
        origin_pred_sampled = origin_pred[inverse_indices]
        axis_pred_sampled = axis_pred[inverse_indices]
        range_pred_sampled = range_pred[inverse_indices] if range_pred is not None else None
        
        # If downsampled, propagate results to full point cloud
        if N > max_points:
            from scipy.spatial import cKDTree
            
            tree = cKDTree(coords_sampled)
            _, nearest_idx = tree.query(coords, k=1)
            
            seg_labels = seg_labels_sampled[nearest_idx]
            seg_probs = seg_probs_sampled[nearest_idx]
            origin_pred_full = origin_pred_sampled[nearest_idx]
            axis_pred_full = axis_pred_sampled[nearest_idx]
            range_pred_full = range_pred_sampled[nearest_idx] if range_pred_sampled is not None else None
        else:
            seg_labels = seg_labels_sampled
            seg_probs = seg_probs_sampled
            origin_pred_full = origin_pred_sampled
            axis_pred_full = axis_pred_sampled
            range_pred_full = range_pred_sampled
        
        return {
            'seg_labels': seg_labels,
            'seg_probs': seg_probs,
            'origin_pred': origin_pred_full,
            'axis_pred': axis_pred_full,
            'range_pred': range_pred_full,
        }


class InteractableModelInference:
    """Articulate3D Interactable Parts Model Inference"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', num_classes: int = 2):
        if not HAS_INTERACTABLE_MODEL:
            raise RuntimeError("InteractableUSDNet not available")
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"📦 Loading Interactable checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Read configuration
        config = ckpt.get('config', {})
        self.num_classes = config.get('num_classes', num_classes)
        self.voxel_size = config.get('voxel_size', 0.02)
        self.feature_dim = config.get('feature_dim_3d', 256)
        
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Voxel size: {self.voxel_size}")
        print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
        
        # Initialize model
        self.model = InteractableUSDNet(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            dropout=config.get('dropout', 0.1),
        ).to(device)
        
        # Load weights
        if 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])
        elif 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        
        self.model.eval()
        print("✓ Interactable model loaded!")
    
    @torch.no_grad()
    def predict(self, coords: np.ndarray, features: np.ndarray, 
                sem_gt: np.ndarray = None, max_points: int = 100000) -> Dict[str, np.ndarray]:
        """Inference - use downsampling strategy"""
        N = len(coords)
        
        # If point count exceeds max_points, use sampling
        if N > max_points:
            if sem_gt is not None:
                # Stratified sampling
                unique_labels = np.unique(sem_gt)
                selected_idx = []
                points_per_class = max_points // len(unique_labels)
                
                for label in unique_labels:
                    label_mask = sem_gt == label
                    label_idx = np.where(label_mask)[0]
                    n_select = min(len(label_idx), points_per_class)
                    selected_idx.extend(np.random.choice(label_idx, n_select, replace=False))
                
                remaining = max_points - len(selected_idx)
                if remaining > 0:
                    all_idx = np.arange(N)
                    remaining_idx = np.setdiff1d(all_idx, selected_idx)
                    if len(remaining_idx) > 0:
                        extra_idx = np.random.choice(remaining_idx, min(remaining, len(remaining_idx)), replace=False)
                        selected_idx.extend(extra_idx)
                
                sample_idx = np.array(selected_idx)
            else:
                sample_idx = np.random.choice(N, max_points, replace=False)
            
            coords_sampled = coords[sample_idx]
            features_sampled = features[sample_idx]
        else:
            sample_idx = np.arange(N)
            coords_sampled = coords
            features_sampled = features
        
        # Voxelization
        voxel_coords = np.floor(coords_sampled / self.voxel_size).astype(np.int32)
        unique_coords, unique_indices, inverse_indices = np.unique(
            voxel_coords, axis=0, return_index=True, return_inverse=True
        )
        
        # Add batch dimension
        batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])
        
        features_unique = features_sampled[unique_indices]
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coords_with_batch).int().to(self.device)
        features_tensor = torch.from_numpy(features_unique).float().to(self.device)
        
        # Inference
        x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
        seg_logits = self.model(x)
        
        # Get prediction results
        logits = seg_logits.features.cpu().numpy()
        probs_voxel = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        labels_voxel = np.argmax(logits, axis=-1)
        
        # Map back to sampled point cloud
        labels_sampled = labels_voxel[inverse_indices]
        probs_sampled = probs_voxel[inverse_indices]
        
        # If downsampled, propagate results to full point cloud
        if N > max_points:
            from scipy.spatial import cKDTree
            
            tree = cKDTree(coords_sampled)
            _, nearest_idx = tree.query(coords, k=1)
            
            labels = labels_sampled[nearest_idx]
            probs = probs_sampled[nearest_idx]
        else:
            labels = labels_sampled
            probs = probs_sampled
        
        return {
            'seg_labels': labels,
            'seg_probs': probs,
        }


# ============================================================================
# Data Loading
# ============================================================================
class Articulate3DDataLoader:
    """Articulate3D Data Loader"""
    
    def __init__(self, data_dir: str, mode: str = "validation"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        # Load database
        db_file = self.data_dir / f"{mode}_database.yaml"
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")
        
        with open(db_file, 'r') as f:
            self.database = yaml.load(f, Loader=yaml.CLoader)
        
        # Fix paths
        base_dir = self.data_dir.parent.parent.parent
        for sample in self.database:
            for key in ['filepath', 'articulation_gt_file', 'expand_dict_file', 'instance_gt_filepath']:
                if key in sample and sample[key] and not os.path.isabs(sample[key]):
                    sample[key] = str(base_dir / sample[key])
        
        # Load color statistics
        color_stats_file = self.data_dir / "color_mean_std.yaml"
        if color_stats_file.exists():
            with open(color_stats_file, 'r') as f:
                color_stats = yaml.load(f, Loader=yaml.CLoader)
            self.color_mean = np.array(color_stats['mean'], dtype=np.float32)
            self.color_std = np.array(color_stats['std'], dtype=np.float32)
        else:
            self.color_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.color_std = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        
        print(f"✓ Loaded {len(self.database)} scenes from {mode} split")
    
    def __len__(self):
        return len(self.database)
    
    def load_scene(self, idx: int) -> Dict[str, Any]:
        """Load scene data"""
        sample = self.database[idx]
        scene_id = sample['scene']
        
        # Load point cloud data
        # Format: [x, y, z, r, g, b, nx, ny, nz, sem_gt, inst_gt, segments, inter_gt]
        data = np.load(sample['filepath']).astype(np.float32)
        
        coords = data[:, :3]
        colors = data[:, 3:6] / 255.0
        normals = data[:, 6:9]
        sem_gt = data[:, 9].astype(np.int32)  # Movable part labels
        inst_gt = data[:, 10].astype(np.int32)
        inter_gt = data[:, 12].astype(np.int32) if data.shape[1] > 12 else np.zeros(len(coords), dtype=np.int32)
        
        # Normalize colors
        colors_norm = (colors - self.color_mean) / (self.color_std + 1e-6)
        
        # Load articulation parameters
        articulations = {}
        if 'articulation_gt_file' in sample and sample['articulation_gt_file']:
            arti_file = sample['articulation_gt_file']
            if os.path.exists(arti_file):
                articulations = load_h5_articulation(arti_file)
        
        return {
            'scene_id': scene_id,
            'coords': coords,
            'colors': colors,  # Original colors [0,1]
            'colors_norm': colors_norm,  # Normalized colors
            'normals': normals,
            'sem_gt': sem_gt,  # Movable part GT (background=0, rotation=1, translation=2)
            'inst_gt': inst_gt,  # Instance GT
            'inter_gt': inter_gt,  # Interactable part GT
            'articulations': articulations,  # Articulation parameters
        }
    
    def get_scene_ids(self) -> List[str]:
        """Get all scene IDs"""
        return [s['scene'] for s in self.database]


# ============================================================================
# Instance Segmentation Utilities
# ============================================================================
def segment_instances_by_proximity(
    points: np.ndarray,
    labels: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 50
) -> np.ndarray:
    """Use DBSCAN clustering to segment same semantic class into different instances"""
    instance_labels = np.zeros(len(points), dtype=np.int32)
    
    unique_labels = np.unique(labels[labels >= 0])
    
    for sem_label in unique_labels:
        mask = labels == sem_label
        if mask.sum() < min_samples:
            instance_labels[mask] = sem_label * 10000
            continue
        
        class_points = points[mask]
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_ids = clustering.fit_predict(class_points)
        
        instance_ids = np.zeros(len(cluster_ids), dtype=np.int32)
        valid_clusters = cluster_ids >= 0
        if valid_clusters.sum() > 0:
            unique_clusters = np.unique(cluster_ids[valid_clusters])
            for new_id, old_id in enumerate(unique_clusters):
                instance_ids[cluster_ids == old_id] = new_id
        
        instance_ids[cluster_ids == -1] = 0
        instance_labels[mask] = sem_label * 10000 + instance_ids
    
    return instance_labels


# ============================================================================
# Motion Information Visualization
# ============================================================================
def create_rotation_arc_trace(
    origin: np.ndarray,
    axis: np.ndarray,
    radius: float = 0.3,
    angle_range: float = 90,  # degrees
    color: str = 'red',
    name: str = "",
    is_prediction: bool = False,
) -> List:
    """
    Create rotation motion range arc visualization
    
    Args:
        origin: Rotation center
        axis: Rotation axis (unit vector)
        radius: Arc radius
        angle_range: Rotation angle range (degrees)
        color: Color
        name: Name
        is_prediction: Whether it is a prediction
    """
    traces = []
    
    # Calculate two orthogonal vectors perpendicular to rotation axis
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    
    # Find a vector not parallel to axis
    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(axis, np.array([0, 1, 0]))
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-6)
    perp2 = np.cross(axis, perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-6)
    
    # Generate points on the arc
    angles = np.linspace(-angle_range/2, angle_range/2, 30) * np.pi / 180
    arc_points = []
    for angle in angles:
        point = origin + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        arc_points.append(point)
    arc_points = np.array(arc_points)
    
    line_dash = 'solid' if not is_prediction else 'dash'
    line_width = 4 if not is_prediction else 3
    
    # Arc line
    traces.append(go.Scatter3d(
        x=arc_points[:, 0],
        y=arc_points[:, 1],
        z=arc_points[:, 2],
        mode='lines',
        line=dict(color=color, width=line_width, dash=line_dash),
        name=f'{name} rotation arc',
        showlegend=False,
        hovertemplate=f'<b>Rotation Range: ±{angle_range/2:.0f}°</b><extra></extra>',
    ))
    
    # Add small arrows at arc endpoints to indicate rotation direction
    if len(arc_points) > 2:
        # Arrow direction at endpoint
        end_dir = arc_points[-1] - arc_points[-2]
        end_dir = end_dir / (np.linalg.norm(end_dir) + 1e-6)
        traces.append(go.Cone(
            x=[arc_points[-1, 0]],
            y=[arc_points[-1, 1]],
            z=[arc_points[-1, 2]],
            u=[end_dir[0] * 0.05],
            v=[end_dir[1] * 0.05],
            w=[end_dir[2] * 0.05],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=0.03,
            showlegend=False,
        ))
    
    return traces


def create_translation_range_trace(
    origin: np.ndarray,
    axis: np.ndarray,
    range_length: float = 0.5,
    color: str = 'blue',
    name: str = "",
    is_prediction: bool = False,
) -> List:
    """
    Create translation motion range line visualization
    
    Args:
        origin: Translation start point
        axis: Translation direction (unit vector)
        range_length: Translation range
        color: Color
        name: Name
        is_prediction: Whether it is a prediction
    """
    traces = []
    
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    
    # Translation range endpoints
    start_point = origin - axis * range_length / 2
    end_point = origin + axis * range_length / 2
    
    line_dash = 'solid' if not is_prediction else 'dash'
    line_width = 5 if not is_prediction else 4
    
    # Main line segment
    traces.append(go.Scatter3d(
        x=[start_point[0], end_point[0]],
        y=[start_point[1], end_point[1]],
        z=[start_point[2], end_point[2]],
        mode='lines',
        line=dict(color=color, width=line_width, dash=line_dash),
        name=f'{name} translation range',
        showlegend=False,
        hovertemplate=f'<b>Translation Range: {range_length:.2f}m</b><extra></extra>',
    ))
    
    # Add endpoint markers at both ends
    traces.append(go.Scatter3d(
        x=[start_point[0], end_point[0]],
        y=[start_point[1], end_point[1]],
        z=[start_point[2], end_point[2]],
        mode='markers',
        marker=dict(size=5, color=color, symbol='x'),
        showlegend=False,
    ))
    
    # Add direction arrow
    traces.append(go.Cone(
        x=[end_point[0]],
        y=[end_point[1]],
        z=[end_point[2]],
        u=[axis[0] * 0.05],
        v=[axis[1] * 0.05],
        w=[axis[2] * 0.05],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=0.04,
        showlegend=False,
    ))
    
    return traces


def create_motion_arrow_trace(
    origin: np.ndarray,
    axis: np.ndarray,
    motion_type: str,
    length: float = 0.5,
    name: str = "",
    is_prediction: bool = False,
    instance_id: int = 0,
    show_label: bool = True,
    show_motion_range: bool = True,
) -> List:
    """
    Create motion arrow Plotly trace
    
    Args:
        origin: Articulation origin (3,)
        axis: Articulation axis vector (3,) - unit vector
        motion_type: 'rotation' or 'translation'
        length: Arrow length
        name: Name prefix
        is_prediction: Whether it is a prediction (use dashed line)
        instance_id: Instance ID
        show_label: Whether to show text label
    
    Returns:
        List of plotly traces
    """
    traces = []
    
    # Color scheme: GT uses solid colors, Pred uses light colors
    if motion_type == 'rotation':
        color = 'red' if not is_prediction else 'salmon'
        type_label = 'ROT'
    elif motion_type == 'translation':
        color = 'blue' if not is_prediction else 'lightblue'
        type_label = 'TRANS'
    else:
        color = 'gray'
        type_label = 'BG'
    
    line_dash = 'solid' if not is_prediction else 'dash'
    label_prefix = 'GT' if not is_prediction else 'Pred'
    
    # Arrow line segment (from origin to endpoint)
    end_point = origin + axis * length
    traces.append(go.Scatter3d(
        x=[origin[0], end_point[0]],
        y=[origin[1], end_point[1]],
        z=[origin[2], end_point[2]],
        mode='lines',
        line=dict(color=color, width=6 if not is_prediction else 4, dash=line_dash),
        name=f'{name} axis',
        showlegend=False,
        hovertemplate=(
            f'<b>{label_prefix} Axis (Instance {instance_id})</b><br>'
            f'Type: {motion_type}<br>'
            f'Direction: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]<br>'
            '<extra></extra>'
        ),
    ))
    
    # Arrow head (cone) - showing direction
    cone_scale = 0.08 if not is_prediction else 0.06
    traces.append(go.Cone(
        x=[end_point[0]],
        y=[end_point[1]],
        z=[end_point[2]],
        u=[axis[0] * cone_scale],
        v=[axis[1] * cone_scale],
        w=[axis[2] * cone_scale],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=0.06,
        name=f'{name} direction',
        showlegend=False,
        hovertemplate=(
            f'<b>{label_prefix} Direction (Instance {instance_id})</b><br>'
            f'Type: {motion_type}<br>'
            f'Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]<br>'
            '<extra></extra>'
        ),
    ))
    
    # Origin marker
    traces.append(go.Scatter3d(
        x=[origin[0]],
        y=[origin[1]],
        z=[origin[2]],
        mode='markers+text' if show_label else 'markers',
        marker=dict(
            size=10 if not is_prediction else 8,
            color=color,
            symbol='diamond',
            line=dict(color='black', width=1) if not is_prediction else dict(color='gray', width=1),
        ),
        text=[f'{label_prefix}_{instance_id}'] if show_label else None,
        textposition='top center',
        textfont=dict(size=10, color=color),
        name=f'{name} origin',
        showlegend=False,
        hovertemplate=(
            f'<b>{label_prefix} Origin (Instance {instance_id})</b><br>'
            f'Type: {motion_type}<br>'
            f'Position: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]<br>'
            '<extra></extra>'
        ),
    ))
    
    # Add motion range visualization
    if show_motion_range and motion_type != 'background':
        if motion_type == 'rotation':
            # Add rotation arc
            range_traces = create_rotation_arc_trace(
                origin, axis, radius=length * 0.6, angle_range=90,
                color=color, name=name, is_prediction=is_prediction
            )
            traces.extend(range_traces)
        elif motion_type == 'translation':
            # Add translation range line
            range_traces = create_translation_range_trace(
                origin, axis, range_length=length * 1.2,
                color=color, name=name, is_prediction=is_prediction
            )
            traces.extend(range_traces)
    
    return traces


def create_motion_info_text(
    origin_gt: Optional[np.ndarray],
    axis_gt: Optional[np.ndarray],
    origin_pred: np.ndarray,
    axis_pred: np.ndarray,
    motion_type: str,
    instance_id: int,
) -> str:
    """Create motion info text description"""
    text = f"<b>Instance {instance_id} ({motion_type})</b><br>"
    
    if origin_gt is not None:
        text += f"<b>GT Origin:</b> [{origin_gt[0]:.3f}, {origin_gt[1]:.3f}, {origin_gt[2]:.3f}]<br>"
        text += f"<b>GT Axis:</b> [{axis_gt[0]:.3f}, {axis_gt[1]:.3f}, {axis_gt[2]:.3f}]<br>"
    
    text += f"<b>Pred Origin:</b> [{origin_pred[0]:.3f}, {origin_pred[1]:.3f}, {origin_pred[2]:.3f}]<br>"
    text += f"<b>Pred Axis:</b> [{axis_pred[0]:.3f}, {axis_pred[1]:.3f}, {axis_pred[2]:.3f}]<br>"
    
    if origin_gt is not None:
        origin_error = np.linalg.norm(origin_pred - origin_gt)
        axis_gt_norm = axis_gt / (np.linalg.norm(axis_gt) + 1e-6)
        cos_angle = np.clip(np.abs(np.dot(axis_pred, axis_gt_norm)), 0, 1)
        axis_error_deg = np.arccos(cos_angle) * 180 / np.pi
        text += f"<b>Origin Error:</b> {origin_error:.4f}m<br>"
        text += f"<b>Axis Error:</b> {axis_error_deg:.2f}°<br>"
    
    return text


def compute_instance_motion_params(
    coords: np.ndarray,
    inst_gt: np.ndarray,
    origin_pred: np.ndarray,
    axis_pred: np.ndarray,
    sem_pred: np.ndarray,
    articulations: Dict[int, Dict],
    json_info: Optional[Dict[str, Any]] = None,
    range_pred: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Compute motion parameters for each instance
    
    Args:
        coords: Point cloud coordinates
        inst_gt: Instance labels
        origin_pred: Origin predictions
        axis_pred: Axis predictions
        sem_pred: Semantic predictions
        articulations: Articulation parameters
        json_info: Extra info from JSON (parts_info and articulation_ranges)
        range_pred: Range predictions (N, 2) [rangeMin, rangeMax]
    
    Returns:
        List of dicts with instance info including:
        - instance_id
        - motion_type
        - origin_gt/pred
        - axis_gt/pred
        - metrics
        - range_min/range_max (GT from JSON)
        - range_min_pred/range_max_pred (from model)
        - interactable_type (from JSON)
    """
    instance_motion_info = []
    
    # Extract range and type info from json_info
    parts_info = json_info.get('parts_info', {}) if json_info else {}
    articulation_ranges = json_info.get('articulation_ranges', {}) if json_info else {}
    
    unique_instances = np.unique(inst_gt)
    
    for inst_id in unique_instances:
        if inst_id <= 0:  # Skip background
            continue
        
        mask = inst_gt == inst_id
        point_count = mask.sum()
        
        if point_count < 10:
            continue
        
        # Get predictions for this instance
        inst_sem_pred = sem_pred[mask]
        inst_origin_pred = origin_pred[mask]
        inst_axis_pred = axis_pred[mask]
        inst_range_pred = range_pred[mask] if range_pred is not None else None
        
        # Use majority vote for motion type
        sem_vote = Counter(inst_sem_pred).most_common(1)[0][0]
        motion_type = MOVABLE_CLASS_NAMES[sem_vote] if sem_vote < len(MOVABLE_CLASS_NAMES) else 'unknown'
        
        # Compute mean predictions
        # First compute mean origin, then snap to nearest movable point
        mean_origin_pred_raw = inst_origin_pred.mean(axis=0)
        inst_coords = coords[mask]
        # Find nearest movable point to the mean origin
        distances = np.linalg.norm(inst_coords - mean_origin_pred_raw, axis=1)
        nearest_idx = np.argmin(distances)
        mean_origin_pred = inst_coords[nearest_idx]
        mean_axis_pred = inst_axis_pred.mean(axis=0)
        mean_axis_pred = mean_axis_pred / (np.linalg.norm(mean_axis_pred) + 1e-6)
        
        # Compute mean range predictions
        mean_range_pred = None
        if inst_range_pred is not None:
            mean_range_pred = inst_range_pred.mean(axis=0)  # (2,) [rangeMin, rangeMax]
        
        # Get GT articulation parameters
        gt_info = articulations.get(inst_id, None)
        
        info = {
            'instance_id': int(inst_id),
            'point_count': int(point_count),
            'motion_type_pred': motion_type,
            'motion_type_gt': MOVABLE_CLASS_NAMES[gt_info['sem_id']] if gt_info else 'unknown',
            'origin_pred': mean_origin_pred,
            'axis_pred': mean_axis_pred,
            'origin_gt': gt_info['origin'] if gt_info else None,
            'axis_gt': gt_info['axis'] if gt_info else None,
        }
        
        # GT motion range from JSON
        range_info = articulation_ranges.get(inst_id, None)
        if range_info:
            info['range_min'] = float(range_info['rangeMin'])
            info['range_max'] = float(range_info['rangeMax'])
            info['range_type'] = range_info['type']
        else:
            info['range_min'] = None
            info['range_max'] = None
            info['range_type'] = None
        
        # Predicted motion range
        if mean_range_pred is not None:
            info['range_min_pred'] = float(mean_range_pred[0])
            info['range_max_pred'] = float(mean_range_pred[1])
        else:
            info['range_min_pred'] = None
            info['range_max_pred'] = None
        
        # Interactable type from JSON
        part_info = parts_info.get(inst_id, None)
        if part_info:
            info['interactable_type'] = part_info['group']
            info['part_label'] = part_info['label']
        else:
            info['interactable_type'] = 'unknown'
            info['part_label'] = 'unknown'
        
        # Compute motion parameter errors
        if gt_info and sem_vote > 0:  # Non-background
            # Origin error (Euclidean distance)
            origin_error = np.linalg.norm(mean_origin_pred - gt_info['origin'])
            info['origin_error'] = float(origin_error)
            
            # Axis error (angle)
            gt_axis = gt_info['axis']
            gt_axis = gt_axis / (np.linalg.norm(gt_axis) + 1e-6)
            cos_angle = np.clip(np.abs(np.dot(mean_axis_pred, gt_axis)), 0, 1)
            axis_error_deg = np.arccos(cos_angle) * 180 / np.pi
            info['axis_error_deg'] = float(axis_error_deg)
        
        instance_motion_info.append(info)
    
    return instance_motion_info


# ============================================================================
# HTML Visualization
# ============================================================================
def visualize_articulate3d_scene_html(
    scene_data: Dict[str, Any],
    movable_pred: Optional[Dict[str, np.ndarray]] = None,
    interactable_pred: Optional[Dict[str, np.ndarray]] = None,
    semantic_pred: Optional[np.ndarray] = None,
    semantic_class_names: Optional[List[str]] = None,
    output_dir: str = "./vis_articulate3d",
    config: VisualizationConfig = None,
    json_dir: Optional[str] = None,
):
    """
    Generate complete HTML visualization for Articulate3D scene
    
    Args:
        scene_data: Scene data (from Articulate3DDataLoader.load_scene)
        movable_pred: Movable part prediction results (from ArticulateModelInference.predict)
        interactable_pred: Interactable part prediction results (from InteractableModelInference.predict)
        semantic_pred: Semantic segmentation prediction results (2564 classes)
        semantic_class_names: Semantic class name list
        output_dir: Output directory
        config: Visualization configuration
        json_dir: Articulate3D original JSON file directory for motion range and interactable type
    """
    if not HAS_PLOTLY:
        print("⚠️  Plotly not available, skipping HTML visualization")
        return
    
    if config is None:
        config = VisualizationConfig()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_id = scene_data['scene_id']
    coords = scene_data['coords']
    colors = scene_data['colors']
    sem_gt = scene_data['sem_gt']  # Movable part GT (3 classes)
    inst_gt = scene_data['inst_gt']
    inter_gt = scene_data['inter_gt']
    articulations = scene_data['articulations']
    
    num_points = len(coords)
    has_semantic = semantic_pred is not None and semantic_class_names is not None
    
    print(f"\n🎨 Visualizing scene: {scene_id} ({num_points:,} points)")
    if has_semantic:
        print(f"   ✓ Semantic segmentation prediction: {len(semantic_class_names)} classes")
    
    # ========================================================================
    # 准备预测结果
    # ========================================================================
    mov_seg_pred = movable_pred['seg_labels'] if movable_pred else np.zeros(num_points, dtype=np.int32)
    origin_pred = movable_pred['origin_pred'] if movable_pred else np.zeros((num_points, 3), dtype=np.float32)
    axis_pred = movable_pred['axis_pred'] if movable_pred else np.zeros((num_points, 3), dtype=np.float32)
    
    inter_seg_pred = interactable_pred['seg_labels'] if interactable_pred else np.zeros(num_points, dtype=np.int32)
    inter_binary_gt = (inter_gt > 0).astype(np.int32)
    
    # ========================================================================
    # 计算评估指标
    # ========================================================================
    # 可动部件分割指标
    mov_metrics = {}
    valid_mov = sem_gt >= 0
    if valid_mov.sum() > 0:
        mov_acc = (sem_gt[valid_mov] == mov_seg_pred[valid_mov]).sum() / valid_mov.sum()
        mov_metrics['accuracy'] = mov_acc
        
        # Per-class IoU
        for c in range(3):
            gt_c = sem_gt == c
            pred_c = mov_seg_pred == c
            intersection = (gt_c & pred_c & valid_mov).sum()
            union = ((gt_c | pred_c) & valid_mov).sum()
            mov_metrics[f'iou_class_{c}'] = intersection / union if union > 0 else 0.0
    
    # 可交互部件分割指标
    inter_metrics = {}
    valid_inter = inter_binary_gt >= 0
    if valid_inter.sum() > 0:
        inter_acc = (inter_binary_gt[valid_inter] == inter_seg_pred[valid_inter]).sum() / valid_inter.sum()
        inter_metrics['accuracy'] = inter_acc
        
        # Binary IoU
        gt_inter = inter_binary_gt == 1
        pred_inter = inter_seg_pred == 1
        intersection = (gt_inter & pred_inter & valid_inter).sum()
        union = ((gt_inter | pred_inter) & valid_inter).sum()
        inter_metrics['iou_interactable'] = intersection / union if union > 0 else 0.0
    
    # Load JSON extra info (motion range and interactable type)
    json_info = None
    if json_dir:
        json_info = load_articulate3d_json_info(json_dir, scene_id)
    
    # Get range predictions from movable_pred if available
    range_pred = movable_pred.get('range_pred', None) if movable_pred else None
    
    # Compute motion parameters
    motion_info = compute_instance_motion_params(
        coords, inst_gt, origin_pred, axis_pred, mov_seg_pred, articulations, json_info, range_pred
    )
    
    # ========================================================================
    # 下采样用于可视化
    # ========================================================================
    if num_points > config.max_points_vis:
        sample_idx = np.random.choice(num_points, config.max_points_vis, replace=False)
    else:
        sample_idx = np.arange(num_points)
    
    vis_coords = coords[sample_idx]
    vis_colors = colors[sample_idx]
    vis_sem_gt = sem_gt[sample_idx]  # 可动部件GT (3类)
    vis_mov_pred = mov_seg_pred[sample_idx]
    vis_inter_gt = inter_binary_gt[sample_idx]
    vis_inter_pred = inter_seg_pred[sample_idx]
    vis_inst_gt = inst_gt[sample_idx]
    vis_semantic_pred = semantic_pred[sample_idx] if has_semantic else None
    
    # 生成语义分割颜色调色板
    if has_semantic:
        num_semantic_classes = len(semantic_class_names)
        semantic_palette = generate_color_palette(num_semantic_classes)
    
    # ========================================================================
    # 创建主HTML（2x2布局）
    # ========================================================================
    # 2x2布局：
    # Row 1 Col 1: 原本的全部运动信息 (GT inter + mov)
    # Row 1 Col 2: 预测的全部运动信息 (Pred inter + mov)
    # Row 2 Col 1: 原本的语义分割 (GT)
    # Row 2 Col 2: 预测的语义分割 (Pred)
    fig_main = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'GT运动信息 (Movable + Interactable)',
            '预测运动信息 (Movable + Interactable)',
            'shape' if has_semantic else 'Shape',
            '语义分割 Pred' if has_semantic else 'Background',
        ),
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    
    # ========================================================================
    # 1行1列: GT运动信息 (Interactable + Movable 合并可视化)
    # ========================================================================
    # 颜色编码: 红色=仅Interactable, 蓝色=仅Movable, 紫色=Both, 灰色=Background
    gt_motion_colors = np.zeros((len(vis_sem_gt), 3), dtype=np.float32)
    gt_motion_text = []
    for i in range(len(vis_sem_gt)):
        is_inter = vis_inter_gt[i] == 1
        is_mov = vis_sem_gt[i] > 0  # movable: rotation(1) or translation(2)
        
        if is_inter and is_mov:
            gt_motion_colors[i] = [1.0, 0.0, 1.0]  # Magenta (both)
            gt_motion_text.append('Both Inter+Mov')
        elif is_inter:
            gt_motion_colors[i] = [1.0, 0.0, 0.0]  # Red (inter only)
            gt_motion_text.append('Interactable')
        elif is_mov:
            gt_motion_colors[i] = [0.0, 0.0, 1.0]  # Blue (mov only)
            gt_motion_text.append(f'Movable-{MOVABLE_CLASS_NAMES[vis_sem_gt[i]]}')
        else:
            gt_motion_colors[i] = [0.5, 0.5, 0.5]  # Gray (background)
            gt_motion_text.append('Background')
    
    fig_main.add_trace(
        go.Scatter3d(
            x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in gt_motion_colors],
            ),
            name='GT Motion',
            text=gt_motion_text,
            hovertemplate=(
                '<b>GT Motion:</b> %{text}<br>'
                'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
        ),
        row=1, col=1
    )
    
    # ========================================================================
    # 1行2列: 预测运动信息 (Interactable + Movable 合并可视化)
    # ========================================================================
    # 颜色编码: 红色=仅Interactable, 蓝色=仅Movable, 紫色=Both, 灰色=Background
    pred_motion_colors = np.zeros((len(vis_mov_pred), 3), dtype=np.float32)
    pred_motion_text = []
    for i in range(len(vis_mov_pred)):
        is_inter = vis_inter_pred[i] == 1
        is_mov = vis_mov_pred[i] > 0  # movable: rotation(1) or translation(2)
        
        if is_inter and is_mov:
            pred_motion_colors[i] = [1.0, 0.0, 1.0]  # Magenta (both)
            pred_motion_text.append('Both Inter+Mov')
        elif is_inter:
            pred_motion_colors[i] = [1.0, 0.0, 0.0]  # Red (inter only)
            pred_motion_text.append('Interactable')
        elif is_mov:
            pred_motion_colors[i] = [0.0, 0.0, 1.0]  # Blue (mov only)
            pred_motion_text.append(f'Movable-{MOVABLE_CLASS_NAMES[vis_mov_pred[i]]}')
        else:
            pred_motion_colors[i] = [0.5, 0.5, 0.5]  # Gray (background)
            pred_motion_text.append('Background')
    
    fig_main.add_trace(
        go.Scatter3d(
            x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_motion_colors],
            ),
            name='Pred Motion',
            text=pred_motion_text,
            hovertemplate=(
                '<b>Pred Motion:</b> %{text}<br>'
                'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
        ),
        row=1, col=2
    )
    
    # 添加运动箭头到运动视图
    if config.show_motion_arrows:
        for info in motion_info:
            motion_type = info['motion_type_gt']
            inst_id = info['instance_id']
            
            # 在 GT运动 视图添加 GT 运动箭头
            if info.get('origin_gt') is not None and info.get('axis_gt') is not None:
                gt_arrows = create_motion_arrow_trace(
                    info['origin_gt'], info['axis_gt'], motion_type,
                    length=config.arrow_scale, name=f'GT_{inst_id}',
                    is_prediction=False, instance_id=inst_id, show_label=True,
                )
                for trace in gt_arrows:
                    fig_main.add_trace(trace, row=1, col=1)
            
            # 在 预测运动 视图添加 Pred 运动箭头
            if info['motion_type_pred'] != 'background':
                pred_arrows = create_motion_arrow_trace(
                    info['origin_pred'], info['axis_pred'], info['motion_type_pred'],
                    length=config.arrow_scale, name=f'Pred_{inst_id}',
                    is_prediction=True, instance_id=inst_id, show_label=True,
                )
                for trace in pred_arrows:
                    fig_main.add_trace(trace, row=1, col=2)
    
    # ========================================================================
    # 2行1列: 语义分割 GT (如果有)
    # ========================================================================
    if has_semantic:
        # 使用原始颜色作为GT（如果没有GT语义标签，使用RGB颜色）
        fig_main.add_trace(
            go.Scatter3d(
                x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in vis_colors],
                ),
                name='Semantic GT (RGB)',
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            ),
            row=2, col=1
        )
        
        # ========================================================================
        # 2行2列: 语义分割预测（使用实例级别的统一颜色）
        # ========================================================================
        # 使用实例分割和多数投票来为每个部件分配统一的颜色
        from collections import Counter

        # 创建每个点的颜色数组（基于实例的多数投票）
        sem_pred_colors_instance = np.zeros((len(vis_coords), 3), dtype=np.float32)

        # 获取所有唯一的实例ID（忽略背景，ID=0）
        unique_instances = np.unique(vis_inst_gt[vis_inst_gt > 0])

        # 为每个实例计算多数语义类别并分配统一颜色
        for inst_id in unique_instances:
            mask = vis_inst_gt == inst_id
            inst_semantic_labels = vis_semantic_pred[mask]

            # 使用多数投票确定该实例的主要类别
            if len(inst_semantic_labels) > 0:
                counter = Counter(inst_semantic_labels)
                majority_class = counter.most_common(1)[0][0]

                # 获取该类别的颜色
                color_idx = min(majority_class, num_semantic_classes - 1)
                sem_pred_colors_instance[mask] = semantic_palette[color_idx]

        # 背景点（实例ID=0）保持原有点级颜色
        background_mask = vis_inst_gt == 0
        if background_mask.sum() > 0:
            for i in np.where(background_mask)[0]:
                color_idx = min(vis_semantic_pred[i], num_semantic_classes - 1)
                sem_pred_colors_instance[i] = semantic_palette[color_idx]

        fig_main.add_trace(
            go.Scatter3d(
                x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sem_pred_colors_instance],
                ),
                name='Semantic Pred (Instance-level)',
                hovertemplate='Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                text=[semantic_class_names[l] if 0 <= l < len(semantic_class_names) else f'Unknown_{l}' for l in vis_semantic_pred],
            ),
            row=2, col=2
        )
    else:
        # 如果没有语义信息，在第2行显示RGB颜色作为占位
        fig_main.add_trace(
            go.Scatter3d(
                x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in vis_colors],
                ),
                name='Shape',
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            ),
            row=2, col=1
        )
    
    # 构建标题文本（包含所有指标和统计信息）
    # 统计运动类型
    num_inter_gt = np.sum(vis_inter_gt == 1)
    num_mov_gt = np.sum(vis_sem_gt > 0)
    num_both_gt = np.sum((vis_inter_gt == 1) & (vis_sem_gt > 0))
    num_inter_pred = np.sum(vis_inter_pred == 1)
    num_mov_pred = np.sum(vis_mov_pred > 0)
    num_both_pred = np.sum((vis_inter_pred == 1) & (vis_mov_pred > 0))
    
    num_rotation_gt = sum(1 for info in motion_info if info['motion_type_gt'] == 'rotation')
    num_translation_gt = sum(1 for info in motion_info if info['motion_type_gt'] == 'translation')
    num_rotation_pred = sum(1 for info in motion_info if info['motion_type_pred'] == 'rotation')
    num_translation_pred = sum(1 for info in motion_info if info['motion_type_pred'] == 'translation')
    
    # 计算运动参数误差统计
    motion_errors = [info for info in motion_info if 'origin_error' in info and info['motion_type_pred'] != 'background']
    if motion_errors:
        avg_origin_err = np.mean([info['origin_error'] for info in motion_errors])
        avg_axis_err = np.mean([info['axis_error_deg'] for info in motion_errors])
        motion_error_str = f' | 运动参数误差: Origin={avg_origin_err:.3f}m, Axis={avg_axis_err:.1f}°'
    else:
        motion_error_str = ''
    
    # 语义分割信息
    semantic_info_str = ""
    if has_semantic:
        # 统计最常见的语义类别
        semantic_counts = Counter(vis_semantic_pred)
        top_classes = semantic_counts.most_common(5)
        top_str = ', '.join([f'{semantic_class_names[c] if c < len(semantic_class_names) else c}' for c, _ in top_classes])
        semantic_info_str = f'<br><span style="font-size:11px; color:#0066cc">🏷️ 语义分割: {num_semantic_classes}类 | Top5: {top_str}</span>'
    
    title_text = (
        f'<b>Articulate3D场景可视化: {scene_id}</b> (点数: {num_points:,})<br>'
        f'<span style="font-size:12px">'
        f'<b>GT运动统计:</b> Inter={num_inter_gt}, Mov={num_mov_gt}, Both={num_both_gt} | '
        f'<b>Pred运动统计:</b> Inter={num_inter_pred}, Mov={num_mov_pred}, Both={num_both_pred}<br>'
        f'<b>运动实例:</b> GT={len(motion_info)} (🔄旋转:{num_rotation_gt}, ↔️平移:{num_translation_gt}) | '
        f'Pred (🔄旋转:{num_rotation_pred}, ↔️平移:{num_translation_pred}){motion_error_str}<br>'
        f'<b>Movable分割:</b> Acc={mov_metrics.get("accuracy", 0)*100:.1f}% | '
        f'IoU: bg={mov_metrics.get("iou_class_0", 0)*100:.1f}%, rot={mov_metrics.get("iou_class_1", 0)*100:.1f}%, trans={mov_metrics.get("iou_class_2", 0)*100:.1f}% | '
        f'<b>Interactable分割:</b> Acc={inter_metrics.get("accuracy", 0)*100:.1f}%, IoU={inter_metrics.get("iou_interactable", 0)*100:.1f}%'
        f'</span>{semantic_info_str}<br>'
        f'<span style="font-size:10px; color:#666">'
        f'🎨 颜色编码: 🔴红=Interactable, 🔵蓝=Movable, 🟣紫=Both, ⚪灰=Background | '
        f'箭头表示运动轴, 实线=GT, 虚线=预测'
        f'</span>'
    )
    
    # 更新布局
    scene_config = dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    
    layout_dict = dict(
        title=dict(
            text=title_text,
            x=0.5, xanchor='center',
            font=dict(size=13)
        ),
        scene=scene_config,
        scene2=scene_config,
        scene3=scene_config,
        scene4=scene_config,
        showlegend=False,
        width=1400,
        height=1100,  # 2x2布局固定高度
        margin=dict(l=20, r=20, t=150, b=20),
    )
    
    fig_main.update_layout(**layout_dict)
    
    # 保存主HTML（使用自定义模板支持滚动）
    main_html_path = output_dir / f"{scene_id}_overview.html"
    
    # 生成带滚动支持的HTML
    html_content = fig_main.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={'scrollZoom': True, 'displayModeBar': True}
    )
    
    # 添加CSS样式支持水平滚动
    custom_css = '''
    <style>
    body {
        overflow-x: auto;
        overflow-y: auto;
        min-width: 1400px;
    }
    .plotly-graph-div {
        min-width: 1400px;
    }
    </style>
    '''
    html_content = html_content.replace('</head>', custom_css + '</head>')
    
    with open(main_html_path, 'w') as f:
        f.write(html_content)
    print(f"  ✓ Main visualization: {main_html_path.name}")
    
    # ========================================================================
    # 为每个可动实例创建独立HTML
    # ========================================================================
    print(f"  📦 Creating individual instance visualizations...")
    print(f"     准确率阈值: {config.min_accuracy*100:.0f}% | 包围盒padding: {config.bbox_padding}m")
    
    skipped_count = 0
    for info in motion_info:
        inst_id = info['instance_id']
        mask = inst_gt == inst_id
        
        if mask.sum() < 10:
            continue
        
        # 计算实例指标（在筛选前计算）
        inst_sem_gt_all = sem_gt[mask]
        inst_mov_pred_all = mov_seg_pred[mask]
        inst_inter_gt_all = inter_binary_gt[mask]
        inst_inter_pred_all = inter_seg_pred[mask]
        
        inst_mov_acc = (inst_sem_gt_all == inst_mov_pred_all).mean()
        inst_inter_acc = (inst_inter_gt_all == inst_inter_pred_all).mean()
        
        # 准确率筛选：综合movable和interactable准确率
        avg_accuracy = (inst_mov_acc + inst_inter_acc) / 2
        if avg_accuracy < config.min_accuracy:
            skipped_count += 1
            continue
        
        # ================================================================
        # 提取3D包围盒内的所有点（包含周围环境）
        # ================================================================
        inst_coords_only = coords[mask]
        
        # 计算实例的3D包围盒
        bbox_min = inst_coords_only.min(axis=0) - config.bbox_padding
        bbox_max = inst_coords_only.max(axis=0) + config.bbox_padding
        
        # 找到包围盒内的所有点
        bbox_mask = (
            (coords[:, 0] >= bbox_min[0]) & (coords[:, 0] <= bbox_max[0]) &
            (coords[:, 1] >= bbox_min[1]) & (coords[:, 1] <= bbox_max[1]) &
            (coords[:, 2] >= bbox_min[2]) & (coords[:, 2] <= bbox_max[2])
        )
        
        # 标记哪些点属于目标实例
        is_target = mask[bbox_mask]
        
        bbox_coords = coords[bbox_mask]
        bbox_colors = colors[bbox_mask]
        bbox_sem_gt = sem_gt[bbox_mask]
        bbox_mov_pred = mov_seg_pred[bbox_mask]
        bbox_inter_gt = inter_binary_gt[bbox_mask]
        bbox_inter_pred = inter_seg_pred[bbox_mask]
        bbox_semantic_pred = semantic_pred[bbox_mask] if has_semantic else None
        
        # 限制点数（保持目标实例和环境的比例）
        max_bbox_points = 50000
        if len(bbox_coords) > max_bbox_points:
            # 分层采样：确保目标实例的点被充分采样
            target_idx = np.where(is_target)[0]
            env_idx = np.where(~is_target)[0]
            
            # 目标实例最多采样一半
            n_target = min(len(target_idx), max_bbox_points // 2)
            n_env = max_bbox_points - n_target
            
            if len(target_idx) > n_target:
                target_sample = np.random.choice(target_idx, n_target, replace=False)
            else:
                target_sample = target_idx
            
            if len(env_idx) > n_env:
                env_sample = np.random.choice(env_idx, n_env, replace=False)
            else:
                env_sample = env_idx
            
            idx = np.concatenate([target_sample, env_sample])
            
            bbox_coords = bbox_coords[idx]
            bbox_colors = bbox_colors[idx]
            bbox_sem_gt = bbox_sem_gt[idx]
            bbox_mov_pred = bbox_mov_pred[idx]
            bbox_inter_gt = bbox_inter_gt[idx]
            bbox_inter_pred = bbox_inter_pred[idx]
            is_target = is_target[idx]
            if has_semantic:
                bbox_semantic_pred = bbox_semantic_pred[idx]
        
        # Get semantic info
        semantic_info_str = ""
        if has_semantic and bbox_semantic_pred is not None:
            # Only count target instance semantics
            target_semantic = bbox_semantic_pred[is_target]
            sem_counts = Counter(target_semantic)
            top_sems = sem_counts.most_common(3)
            sem_names = [f'{semantic_class_names[s] if s < len(semantic_class_names) else s}({c})' for s, c in top_sems]
            semantic_info_str = f' | 🏷️ Semantic: {", ".join(sem_names)}'
        
        # ================================================================
        # 创建实例HTML (2x2布局，不再将GT和Pred放在同一窗口)
        # 左上: GT运动信息 (inter + mov)
        # 右上: Pred运动信息 (inter + mov)
        # Layout: 2x2 grid
        # Top-left: GT Motion (Movable + Interactable)
        # Top-right: Pred Motion (Movable + Interactable)
        # Bottom-left: Shape
        # Bottom-right: Semantic Pred
        # ================================================================
        # 获取 part_label_short
        part_label = info.get('part_label', 'unknown')
        part_label_short = part_label.split('.')[0] if part_label and part_label != 'unknown' else ''

        # 构建语义类别显示文本
        semantic_subtitle = 'Shape'
        if has_semantic and bbox_semantic_pred is not None:
            if part_label_short:
                semantic_subtitle = f'Semantic Pred: {part_label_short}'
            else:
                semantic_subtitle = 'Semantic Pred'

        fig_inst = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'GT Motion (Movable + Interactable)',
                'Pred Motion (Movable + Interactable)',
                'Shape',
                semantic_subtitle,
            ),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        )
        
        # ================================================================
        # 计算颜色：目标实例高亮，环境淡化
        # ================================================================
        # GT运动颜色
        gt_motion_colors = np.zeros((len(bbox_sem_gt), 3), dtype=np.float32)
        for i in range(len(bbox_sem_gt)):
            is_inter = bbox_inter_gt[i] == 1
            is_mov = bbox_sem_gt[i] > 0
            
            if is_inter and is_mov:
                gt_motion_colors[i] = [1.0, 0.0, 1.0]  # Magenta (both)
            elif is_inter:
                gt_motion_colors[i] = [1.0, 0.0, 0.0]  # Red (inter only)
            elif is_mov:
                gt_motion_colors[i] = [0.0, 0.0, 1.0]  # Blue (mov only)
            else:
                gt_motion_colors[i] = [0.5, 0.5, 0.5]  # Gray (background)
        
        # Pred运动颜色
        pred_motion_colors = np.zeros((len(bbox_mov_pred), 3), dtype=np.float32)
        for i in range(len(bbox_mov_pred)):
            is_inter = bbox_inter_pred[i] == 1
            is_mov = bbox_mov_pred[i] > 0
            
            if is_inter and is_mov:
                pred_motion_colors[i] = [1.0, 0.0, 1.0]  # Magenta (both)
            elif is_inter:
                pred_motion_colors[i] = [1.0, 0.0, 0.0]  # Red (inter only)
            elif is_mov:
                pred_motion_colors[i] = [0.0, 0.0, 1.0]  # Blue (mov only)
            else:
                pred_motion_colors[i] = [0.5, 0.5, 0.5]  # Gray (background)
        
        # Point size settings
        env_opacity = 0.3
        target_size = 4  # Larger for target instance
        env_size = 2     # Larger for environment
        semantic_point_size = 3  # Solid larger points for semantic views
        
        point_sizes = np.where(is_target, target_size, env_size)
        # All points same size for semantic views (bottom row)
        semantic_sizes = np.full(len(bbox_coords), semantic_point_size)
        
        # Environment color fading for motion views
        gt_motion_colors_display = gt_motion_colors.copy()
        gt_motion_colors_display[~is_target] = gt_motion_colors_display[~is_target] * env_opacity + 0.7 * (1 - env_opacity)
        
        pred_motion_colors_display = pred_motion_colors.copy()
        pred_motion_colors_display[~is_target] = pred_motion_colors_display[~is_target] * env_opacity + 0.7 * (1 - env_opacity)
        
        # ================================================================
        # Row 1 Col 1: GT Motion
        # ================================================================
        fig_inst.add_trace(
            go.Scatter3d(
                x=bbox_coords[:, 0], y=bbox_coords[:, 1], z=bbox_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=point_sizes.tolist(),
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in gt_motion_colors_display],
                ),
                name='GT Motion',
            ),
            row=1, col=1
        )
        
        # ================================================================
        # 1行2列: Pred运动信息
        # ================================================================
        fig_inst.add_trace(
            go.Scatter3d(
                x=bbox_coords[:, 0], y=bbox_coords[:, 1], z=bbox_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=point_sizes.tolist(),
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_motion_colors_display],
                ),
                name='Pred Motion',
            ),
            row=1, col=2
        )
        
        # 添加运动箭头
        motion_type = info['motion_type_pred']
        motion_type_gt = info['motion_type_gt']
        if config.show_motion_arrows:
            # GT箭头（如果有）
            if info.get('origin_gt') is not None and info.get('axis_gt') is not None:
                arrow_traces = create_motion_arrow_trace(
                    info['origin_gt'], info['axis_gt'], motion_type_gt,
                    length=config.arrow_scale, name='GT',
                    is_prediction=False, instance_id=inst_id, show_label=True,
                )
                for trace in arrow_traces:
                    fig_inst.add_trace(trace, row=1, col=1)
            
            # Pred箭头
            if motion_type != 'background':
                arrow_traces = create_motion_arrow_trace(
                    info['origin_pred'], info['axis_pred'], motion_type,
                    length=config.arrow_scale, name='Pred',
                    is_prediction=True, instance_id=inst_id, show_label=True,
                )
                for trace in arrow_traces:
                    fig_inst.add_trace(trace, row=1, col=2)
        
        # ================================================================
        # Row 2 Col 1: Shape (solid points for all)
        # ================================================================
        # Use larger solid points for all
        
        fig_inst.add_trace(
            go.Scatter3d(
                x=bbox_coords[:, 0], y=bbox_coords[:, 1], z=bbox_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=semantic_sizes.tolist(),
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in bbox_colors],
                ),
                name='RGB',
            ),
            row=2, col=1
        )
        
        # ================================================================
        # Row 2 Col 2: Semantic Segmentation Prediction (只对目标实例上统一颜色)
        # ================================================================
        if has_semantic and bbox_semantic_pred is not None:
            # 获取 part_label_short
            part_label = info.get('part_label', 'unknown')
            part_label_short = part_label.split('.')[0] if part_label and part_label != 'unknown' else 'Unknown'

            # 随机选择一个颜色：黄色、绿色、蓝色或紫色
            color_options = [
                [1.0, 1.0, 0.0],    # 黄色
                [0.0, 1.0, 0.0],    # 绿色
                [0.0, 0.0, 1.0],    # 蓝色
                [0.5, 0.0, 0.5],    # 紫色
            ]
            import random
            uniform_color = random.choice(color_options)

            # 创建颜色数组：目标实例用统一颜色，其他用RGB颜色
            sem_colors = np.zeros((len(bbox_coords), 3), dtype=np.float32)
            for i in range(len(bbox_coords)):
                if is_target[i]:
                    sem_colors[i] = uniform_color
                else:
                    sem_colors[i] = bbox_colors[i]

            # 所有点的类别都显示为 part_label_short
            text_labels = [part_label_short] * len(bbox_coords)

            fig_inst.add_trace(
                go.Scatter3d(
                    x=bbox_coords[:, 0], y=bbox_coords[:, 1], z=bbox_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=semantic_sizes.tolist(),
                        color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sem_colors],
                    ),
                    name='Semantic Pred',
                    hovertemplate='Class: %{text}',
                    text=text_labels,
                ),
                row=2, col=2
            )
        else:
            # No semantic info, show RGB
            fig_inst.add_trace(
                go.Scatter3d(
                    x=bbox_coords[:, 0], y=bbox_coords[:, 1], z=bbox_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=semantic_sizes.tolist(),
                        color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in bbox_colors],
                    ),
                    name='RGB',
                ),
                row=2, col=2
            )
        
        # Build instance title (with all metrics)
        motion_params_str = ""
        motion_range_str = ""
        interactable_type_str = ""
        
        # Get interactable type info
        interactable_type = info.get('interactable_type', 'unknown')
        part_label = info.get('part_label', 'unknown')
        if interactable_type and interactable_type != 'unknown' and interactable_type != 'others':
            type_display_map = {
                'doors/windows': '🚪 Door/Window',
                'handles movable': '🔧 Handle',
                'switches': '🔘 Switch',
                'controls': '🎛️ Control',
                'lids': '📦 Lid',
                'drawers': '🗄️ Drawer',
                'others': '❓ Other'
            }
            type_display = type_display_map.get(interactable_type, f'📍 {interactable_type}')
            interactable_type_str = type_display
        else:
            interactable_type_str = '❓ Unknown'
        
        if motion_type != 'background':
            origin_err = info.get('origin_error', 0)
            axis_err = info.get('axis_error_deg', 0)
            motion_params_str = f'Origin Err: {origin_err:.3f}m | Axis Err: {axis_err:.1f}°'
            
            # GT motion range from JSON
            range_min = info.get('range_min')
            range_max = info.get('range_max')
            # Pred motion range from model
            range_min_pred = info.get('range_min_pred')
            range_max_pred = info.get('range_max_pred')
            
            if motion_type == 'rotation':
                if range_min is not None and range_max is not None:
                    range_min_deg = np.rad2deg(range_min)
                    range_max_deg = np.rad2deg(range_max)
                    range_total_deg = range_max_deg - range_min_deg
                    motion_range_str = f'🔄 GT Range: [{range_min_deg:.1f}°, {range_max_deg:.1f}°] (Total: {range_total_deg:.1f}°)'
                else:
                    motion_range_str = f'🔄 GT Range: (No data)'
                
                if range_min_pred is not None and range_max_pred is not None:
                    range_min_pred_deg = np.rad2deg(range_min_pred)
                    range_max_pred_deg = np.rad2deg(range_max_pred)
                    range_total_pred_deg = range_max_pred_deg - range_min_pred_deg
                    motion_range_str += f' | Pred: [{range_min_pred_deg:.1f}°, {range_max_pred_deg:.1f}°] (Total: {range_total_pred_deg:.1f}°)'
            elif motion_type == 'translation':
                if range_min is not None and range_max is not None:
                    range_total = range_max - range_min
                    motion_range_str = f'↔️ GT Range: [{range_min:.3f}m, {range_max:.3f}m] (Total: {range_total:.3f}m)'
                else:
                    motion_range_str = f'↔️ GT Range: (No data)'
                
                if range_min_pred is not None and range_max_pred is not None:
                    range_total_pred = range_max_pred - range_min_pred
                    motion_range_str += f' | Pred: [{range_min_pred:.3f}m, {range_max_pred:.3f}m] (Total: {range_total_pred:.3f}m)'
        
        # Build hierarchical title with GT and Pred range at top
        # Line 1: Instance info + Interactable type + Part label
        # Line 2: Motion range (GT and Pred)
        # Line 3: Accuracy metrics
        # Line 4: Detailed parameters
        
        part_label = info.get('part_label', 'unknown')
        part_label_short = part_label.split('.')[0] if part_label else 'unknown'
        
        inst_title = (
            f'<b style="font-size:16px">Instance {inst_id}: {motion_type.upper()}</b> '
            f'<span style="font-size:14px; background-color:#e0f7fa; padding:2px 8px; border-radius:4px">{interactable_type_str}</span> '
        )
        
        # Show part semantic label
        if part_label and part_label != 'unknown':
            inst_title += (
                f'<span style="font-size:13px; color:#2e7d32; font-weight:bold"> 📍 {part_label_short}</span> '
                f'<span style="font-size:11px; color:#666">({part_label})</span>'
            )
        
        inst_title += f'<span style="font-size:12px"> ({info["point_count"]:,} pts)</span><br>'
        
        # Show GT motion range at top
        if motion_type != 'background' and motion_range_str:
            inst_title += (
                f'<span style="font-size:14px; color:#1565c0; font-weight:bold">{motion_range_str}</span><br>'
            )

        
        inst_title += (
            f'<span style="font-size:11px">'
            f'GT: {info["motion_type_gt"]} | Pred: {info["motion_type_pred"]} | '
            f'Mov Acc: {inst_mov_acc*100:.1f}% | Inter Acc: {inst_inter_acc*100:.1f}%'
        )
        
        if motion_params_str:
            inst_title += f' | {motion_params_str}'
        
        if semantic_info_str:
            inst_title += semantic_info_str
            
        inst_title += '</span>'
        
        # 如果有运动参数，添加详细信息
        motion_detail = ""
        if motion_type != 'background' and info.get('origin_gt') is not None:
            motion_detail = (
                f'<br><span style="font-size:10px">'
                f'Origin GT: [{info["origin_gt"][0]:.2f}, {info["origin_gt"][1]:.2f}, {info["origin_gt"][2]:.2f}] | '
                f'Pred: [{info["origin_pred"][0]:.2f}, {info["origin_pred"][1]:.2f}, {info["origin_pred"][2]:.2f}] | '
                f'Axis GT: [{info["axis_gt"][0]:.2f}, {info["axis_gt"][1]:.2f}, {info["axis_gt"][2]:.2f}] | '
                f'Pred: [{info["axis_pred"][0]:.2f}, {info["axis_pred"][1]:.2f}, {info["axis_pred"][2]:.2f}]'
                f'</span>'
            )
        
        # Update layout (2x2 grid)
        layout_dict = dict(
            title=dict(
                text=inst_title + motion_detail,
                x=0.5, xanchor='center',
                font=dict(size=12)
            ),
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data'),
            scene3=dict(aspectmode='data'),
            scene4=dict(aspectmode='data'),
            showlegend=False,
            width=1400,
            height=1100,
            margin=dict(l=20, r=20, t=200, b=20),  # Increased top margin
        )
        
        fig_inst.update_layout(**layout_dict)
        
        inst_html_path = output_dir / f"{scene_id}_instance_{inst_id}_{motion_type}.html"
        
        # Generate HTML with scroll support
        inst_html_content = fig_inst.to_html(
            full_html=True,
            include_plotlyjs=True,
            config={'scrollZoom': True, 'displayModeBar': True}
        )
        
        custom_css = '''
        <style>
        body {
            overflow-x: auto;
            overflow-y: auto;
            min-width: 1200px;
        }
        </style>
        '''
        inst_html_content = inst_html_content.replace('</head>', custom_css + '</head>')
        
        with open(inst_html_path, 'w') as f:
            f.write(inst_html_content)
    
    created_count = len(motion_info) - skipped_count
    print(f"  ✓ Created {created_count} instance visualizations (skipped {skipped_count} below {config.min_accuracy*100:.0f}% accuracy)")
    
    # ========================================================================
    # 保存评估报告JSON
    # ========================================================================
    report = {
        'scene_id': scene_id,
        'num_points': num_points,
        'movable_metrics': mov_metrics,
        'interactable_metrics': inter_metrics,
        'instances': motion_info,
    }
    
    report_path = output_dir / f"{scene_id}_report.json"
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    report = convert_to_serializable(report)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Report saved: {report_path.name}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Articulate3D Visualization Tool")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                        default="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_mov",
                        help='Path to Articulate3D processed data (movable)')
    parser.add_argument('--inter_data_dir', type=str,
                        default="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_inter",
                        help='Path to Articulate3D processed data (interactable)')
    parser.add_argument('--mode', type=str, default='validation',
                        choices=['train', 'validation', 'test'],
                        help='Data split to visualize')
    
    # 模型参数
    parser.add_argument('--movable_checkpoint', type=str, default=None,
                        help='Path to movable part segmentation checkpoint')
    parser.add_argument('--interactable_checkpoint', type=str, default=None,
                        help='Path to interactable part segmentation checkpoint')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./visualizations_articulate3d',
                        help='Output directory for visualizations')
    parser.add_argument('--max_scenes', type=int, default=5,
                        help='Maximum number of scenes to visualize')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None,
                        help='Specific scene IDs to visualize')
    
    # 可视化参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for inference')
    parser.add_argument('--max_points_vis', type=int, default=500000,
                        help='Maximum points for visualization')
    parser.add_argument('--arrow_scale', type=float, default=0.5,
                        help='Scale for motion arrows')
    parser.add_argument('--no_motion_arrows', action='store_true',
                        help='Disable motion arrows in visualization')
    parser.add_argument('--min_accuracy', type=float, default=0.0,
                        help='Minimum accuracy threshold for instance visualization (0.0-1.0). Instances below this will be skipped.')
    parser.add_argument('--bbox_padding', type=float, default=0.5,
                        help='Padding around bounding box when extracting instance context (meters)')
    
    # Zarr数据参数（用于原始语义分割可视化）
    parser.add_argument('--zarr_root', type=str, default=None,
                        help='Path to zarr data directory (for USDNet semantic segmentation)')
    parser.add_argument('--semantic_checkpoint', type=str, 
                        default='/mnt/nfs/home/st185933/DINO-Infused-USDNet/checkpointss/finetune/ckpt_best.pt',
                        help='Path to USDNet semantic segmentation checkpoint')
    # 语义标签映射（用于在Articulate3D场景中显示语义类别名称）
    parser.add_argument('--label_mapping', type=str, 
                        default='/mnt/nfs/home/st185933/new_data_elements/global_label_mapping.json',
                        help='Path to global_label_mapping.json for semantic class names')
    
    # Articulate3D原始JSON标注目录（用于获取运动范围和可交互类型）
    parser.add_argument('--json_dir', type=str,
                        default='/mnt/nfs/home/st185933/Articulate3D',
                        help='Path to Articulate3D original JSON annotation directory (for range and interactable type)')
    
    # 注意: zarr数据没有运动GT，所以需要单独指定运动模型来预测
    # 可以复用 --movable_checkpoint 和 --interactable_checkpoint
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎨 Articulate3D & USDNet Visualization Tool")
    print("="*80)
    
    # 配置
    config = VisualizationConfig(
        output_dir=args.output_dir,
        device=args.device,
        voxel_size=args.voxel_size,
        max_points_vis=args.max_points_vis,
        arrow_scale=args.arrow_scale,
        show_motion_arrows=not args.no_motion_arrows,
        min_accuracy=args.min_accuracy,
        bbox_padding=args.bbox_padding,
        json_dir=args.json_dir,
    )
    
    # ========================================================================
    # 模式1: Zarr数据（原始语义分割）
    # ========================================================================
    if args.zarr_root:
        print(f"\n📂 [Mode: Zarr Semantic Segmentation + Motion Prediction]")
        print(f"   Loading zarr data from: {args.zarr_root}")
        print(f"   ⚠️  注意: Zarr数据只有语义分割GT (2564类)，没有运动GT")
        print(f"   ⚠️  所有运动信息都是模型预测的")
        
        zarr_loader = ZarrDataLoader(args.zarr_root)
        
        if zarr_loader.num_classes == 0:
            print("❌ No global label mapping found!")
            return
        
        # 加载USDNet语义模型（如果有）
        semantic_inference = None
        if args.semantic_checkpoint and os.path.exists(args.semantic_checkpoint):
            try:
                from DINO_Infused_USDNet.train_usdnet_complete import USDNetStudent
                
                print(f"📦 Loading USDNet semantic checkpoint: {args.semantic_checkpoint}")
                ckpt = torch.load(args.semantic_checkpoint, map_location=args.device)
                ckpt_config = ckpt.get('config', {})
                
                semantic_inference = USDNetStudent(
                    num_classes=ckpt_config.get('num_classes', zarr_loader.num_classes),
                    feature_dim_3d=ckpt_config.get('feature_dim_3d', 256),
                    feature_dim_2d=ckpt_config.get('feature_dim_2d', 768),
                    dropout=ckpt_config.get('dropout', 0.1),
                ).to(args.device)
                
                semantic_inference.load_state_dict(ckpt['model'])
                semantic_inference.eval()
                print("✓ USDNet semantic model loaded!")
            except Exception as e:
                print(f"⚠️  Failed to load semantic checkpoint: {e}")
        
        # 加载运动预测模型（用于在zarr数据上预测运动）
        mov_inference = None
        inter_inference = None
        
        if args.movable_checkpoint and os.path.exists(args.movable_checkpoint):
            mov_inference = ArticulateModelInference(args.movable_checkpoint, device=args.device)
            print("  ✓ Movable model loaded for motion prediction on zarr data")
        
        if args.interactable_checkpoint and os.path.exists(args.interactable_checkpoint):
            inter_inference = InteractableModelInference(args.interactable_checkpoint, device=args.device)
            print("  ✓ Interactable model loaded for motion prediction on zarr data")
        
        # 选择场景
        all_scene_ids = zarr_loader.get_scene_ids()
        
        if args.scene_ids:
            selected_ids = [s for s in args.scene_ids if s in all_scene_ids]
        else:
            selected_ids = all_scene_ids[:args.max_scenes]
        
        print(f"\n🎯 Visualizing {len(selected_ids)} zarr scenes")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for scene_id in tqdm(selected_ids, desc="Processing zarr scenes"):
            scene_idx = all_scene_ids.index(scene_id)
            scene_data = zarr_loader.load_scene(scene_idx)
            
            # 可视化zarr场景（语义分割 + 运动预测）
            visualize_zarr_scene_html(
                scene_data=scene_data,
                class_names=zarr_loader.class_names,
                num_classes=zarr_loader.num_classes,
                semantic_model=semantic_inference,
                movable_model=mov_inference,
                interactable_model=inter_inference,
                output_dir=str(output_dir),
                config=config,
                device=args.device,
            )
        
        print(f"\n✨ Zarr visualization complete! Output: {output_dir}")
        return
    
    # ========================================================================
    # 模式2: Articulate3D数据
    # ========================================================================
    print(f"\n📂 [Mode: Articulate3D Motion Segmentation]")
    print(f"   Loading data from: {args.data_dir}")
    data_loader = Articulate3DDataLoader(args.data_dir, mode=args.mode)
    
    # 加载语义标签映射（用于显示语义类别名称）
    semantic_class_names = []
    num_semantic_classes = 0
    if args.label_mapping and os.path.exists(args.label_mapping):
        with open(args.label_mapping, 'r') as f:
            label_data = json.load(f)
        semantic_class_names = label_data.get('class_names', [])
        num_semantic_classes = label_data.get('num_classes', 0)
        print(f"   ✓ Loaded semantic label mapping: {num_semantic_classes} classes")
    
    # 加载语义分割模型（可选，用于在Articulate3D数据上预测语义）
    semantic_inference = None
    if args.semantic_checkpoint:
        if not os.path.exists(args.semantic_checkpoint):
            print(f"   ⚠️ Semantic checkpoint file not found: {args.semantic_checkpoint}")
        else:
            try:
                from train_usdnet_complete import USDNetStudent

                print(f"📦 Loading USDNet semantic checkpoint: {args.semantic_checkpoint}")
                ckpt = torch.load(args.semantic_checkpoint, map_location=args.device)
                ckpt_config = ckpt.get('config', {})

                semantic_inference = USDNetStudent(
                    num_classes=ckpt_config.get('num_classes', num_semantic_classes),
                    feature_dim_3d=ckpt_config.get('feature_dim_3d', 256),
                    feature_dim_2d=ckpt_config.get('feature_dim_2d', 768),
                    dropout=ckpt_config.get('dropout', 0.1),
                ).to(args.device)

                semantic_inference.load_state_dict(ckpt['model'])
                semantic_inference.eval()
                print(f"   ✓ USDNet semantic model loaded! ({ckpt_config.get('num_classes', '?')} classes)")
            except Exception as e:
                print(f"   ⚠️ Failed to load semantic checkpoint: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("   ⚠️ No semantic checkpoint provided, semantic segmentation will not be used")
    
    # 加载运动模型
    mov_inference = None
    inter_inference = None
    
    if args.movable_checkpoint and os.path.exists(args.movable_checkpoint):
        mov_inference = ArticulateModelInference(args.movable_checkpoint, device=args.device)
    else:
        print("⚠️  No movable checkpoint provided, using GT labels only")
    
    if args.interactable_checkpoint and os.path.exists(args.interactable_checkpoint):
        inter_inference = InteractableModelInference(args.interactable_checkpoint, device=args.device)
    else:
        print("⚠️  No interactable checkpoint provided, using GT labels only")
    
    # 选择场景
    all_scene_ids = data_loader.get_scene_ids()
    
    if args.scene_ids:
        selected_ids = [s for s in args.scene_ids if s in all_scene_ids]
        if not selected_ids:
            print(f"❌ No matching scenes found. Available: {all_scene_ids[:10]}...")
            return
    else:
        selected_ids = all_scene_ids[:args.max_scenes]
    
    print(f"\n🎯 Visualizing {len(selected_ids)} scenes: {selected_ids}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个场景
    for idx, scene_id in enumerate(tqdm(selected_ids, desc="Processing scenes")):
        # 找到场景索引
        scene_idx = all_scene_ids.index(scene_id)
        
        # 加载场景数据
        scene_data = data_loader.load_scene(scene_idx)
        
        # 准备特征
        coords = scene_data['coords']
        colors_norm = scene_data['colors_norm']
        normals = scene_data['normals']
        features = np.hstack([colors_norm, normals, coords]).astype(np.float32)
        
        # 推理
        mov_pred = None
        inter_pred = None
        semantic_pred = None
        
        # 获取可动部件GT标签（用于分层采样）
        sem_gt = scene_data.get('sem_gt', None)
        
        if mov_inference:
            # 使用GT标签进行分层采样，确保与训练时的数据分布一致
            mov_pred = mov_inference.predict(coords, features, sem_gt=sem_gt, max_points=100000)
        
        if inter_inference:
            # 获取可交互GT标签（用于分层采样）
            inter_gt = scene_data.get('inter_gt', None)
            inter_pred = inter_inference.predict(coords, features, sem_gt=inter_gt, max_points=100000)
        
        # 语义分割推理（如果有模型）
        if semantic_inference is not None:
            with torch.no_grad():
                voxel_coords = np.floor(coords / config.voxel_size).astype(np.int32)
                unique_coords, unique_indices, inverse_indices = np.unique(
                    voxel_coords, axis=0, return_index=True, return_inverse=True
                )
                
                features_unique = features[unique_indices]
                batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
                coords_with_batch = np.hstack([batch_indices, unique_coords])
                
                coords_tensor = torch.from_numpy(coords_with_batch).int().to(args.device)
                features_tensor = torch.from_numpy(features_unique).float().to(args.device)
                
                x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
                seg_logits, _ = semantic_inference(x)
                
                logits = seg_logits.features.cpu().numpy()
                labels_voxel = np.argmax(logits, axis=-1)
                semantic_pred = labels_voxel[inverse_indices]
        
        # 可视化
        visualize_articulate3d_scene_html(
            scene_data=scene_data,
            movable_pred=mov_pred,
            interactable_pred=inter_pred,
            semantic_pred=semantic_pred,
            semantic_class_names=semantic_class_names,
            output_dir=str(output_dir),
            config=config,
            json_dir=config.json_dir,
        )
    
    print(f"\n✨ Visualization complete! Output: {output_dir}")
    print(f"   Open the HTML files in a browser to view the results.")


# ============================================================================
# Zarr场景可视化（语义分割 + 运动预测）
# ============================================================================
def visualize_zarr_scene_html(
    scene_data: Dict[str, Any],
    class_names: List[str],
    num_classes: int,
    semantic_model: Optional[torch.nn.Module] = None,
    movable_model: Optional['ArticulateModelInference'] = None,
    interactable_model: Optional['InteractableModelInference'] = None,
    output_dir: str = "./vis_zarr",
    config: VisualizationConfig = None,
    device: str = 'cuda:0',
):
    """
    生成Zarr场景的HTML可视化
    
    注意: Zarr数据只有语义分割GT (2564类)，没有运动GT
    所有运动信息（可动部件、可交互部件、铰接参数）都是预测的
    
    Args:
        scene_data: 场景数据
        class_names: 语义类别名称列表 (2564个)
        num_classes: 类别数量
        semantic_model: 语义分割模型 (USDNet)
        movable_model: 可动部件预测模型 (ArticulateUSDNet)
        interactable_model: 可交互部件预测模型 (InteractableUSDNet)
        output_dir: 输出目录
        config: 配置
        device: 设备
    """
    if not HAS_PLOTLY:
        print("⚠️  Plotly not available")
        return
    
    if config is None:
        config = VisualizationConfig()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_id = scene_data['scene_id']
    coords = scene_data['coords']
    colors = scene_data['colors']
    normals = scene_data['normals']
    semantic_gt = scene_data['semantic_labels']  # 语义分割GT (2564类)
    
    num_points = len(coords)
    print(f"\n🎨 Visualizing zarr scene: {scene_id} ({num_points:,} points)")
    print(f"   📊 语义分割: GT (2564类) | 运动预测: 模型预测 (无GT)")
    
    # 准备特征
    colors_norm = (colors - 0.5) / 0.25
    features = np.hstack([colors_norm, normals, coords]).astype(np.float32)
    
    # ========================================================================
    # 1. 语义分割预测（如果有模型）
    # ========================================================================
    semantic_pred = None
    if semantic_model is not None:
        with torch.no_grad():
            voxel_coords = np.floor(coords / config.voxel_size).astype(np.int32)
            unique_coords, unique_indices, inverse_indices = np.unique(
                voxel_coords, axis=0, return_index=True, return_inverse=True
            )
            
            features_unique = features[unique_indices]
            batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
            coords_with_batch = np.hstack([batch_indices, unique_coords])
            
            coords_tensor = torch.from_numpy(coords_with_batch).int().to(device)
            features_tensor = torch.from_numpy(features_unique).float().to(device)
            
            x = ME.SparseTensor(features=features_tensor, coordinates=coords_tensor)
            seg_logits, _ = semantic_model(x)
            
            logits = seg_logits.features.cpu().numpy()
            labels_voxel = np.argmax(logits, axis=-1)
            semantic_pred = labels_voxel[inverse_indices]
    
    # ========================================================================
    # 2. 运动预测（可动部件 + 可交互部件 + 铰接参数）
    # ========================================================================
    mov_pred = None
    inter_pred = None
    
    if movable_model is not None:
        mov_pred = movable_model.predict(coords, features)
        print(f"   ✓ Movable prediction: {(mov_pred['seg_labels'] > 0).sum():,} motion points")
    
    if interactable_model is not None:
        inter_pred = interactable_model.predict(coords, features)
        print(f"   ✓ Interactable prediction: {(inter_pred['seg_labels'] > 0).sum():,} interactable points")
    
    # 生成颜色
    color_palette = generate_color_palette(num_classes)
    
    # 下采样
    if num_points > config.max_points_vis:
        sample_idx = np.random.choice(num_points, config.max_points_vis, replace=False)
    else:
        sample_idx = np.arange(num_points)
    
    vis_coords = coords[sample_idx]
    vis_colors = colors[sample_idx]
    vis_semantic_gt = semantic_gt[sample_idx]
    vis_semantic_pred = semantic_pred[sample_idx] if semantic_pred is not None else vis_semantic_gt
    
    vis_mov_pred = mov_pred['seg_labels'][sample_idx] if mov_pred is not None else np.zeros(len(sample_idx), dtype=np.int32)
    vis_inter_pred = inter_pred['seg_labels'][sample_idx] if inter_pred is not None else np.zeros(len(sample_idx), dtype=np.int32)
    
    # 计算语义分割指标
    valid_mask = vis_semantic_gt >= 0
    semantic_accuracy = 0.0
    if valid_mask.sum() > 0 and semantic_pred is not None:
        semantic_accuracy = (vis_semantic_gt[valid_mask] == vis_semantic_pred[valid_mask]).sum() / valid_mask.sum()
    
    # ========================================================================
    # 创建总览HTML (2x2布局)
    # ========================================================================
    # 2x2布局（与articulate3d场景一致）：
    # Row 1 Col 1: 原本的运动信息 (无GT，显示RGB)
    # Row 1 Col 2: 预测的运动信息 (Movable + Interactable)
    # Row 2 Col 1: 语义分割 GT
    # Row 2 Col 2: 语义分割 Pred
    
    has_motion = movable_model is not None or interactable_model is not None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'RGB颜色 (无运动GT)',
            '预测运动信息 (Movable + Interactable)',
            'shape',
            '语义分割 Pred' if semantic_model else 'shape',
        ),
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    
    # ========================================================================
    # 1行1列: RGB颜色 (Zarr场景无运动GT)
    # ========================================================================
    fig.add_trace(
        go.Scatter3d(
            x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in vis_colors]
            ),
            name='RGB',
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
        ),
        row=1, col=1
    )
    
    # ========================================================================
    # 1行2列: 预测运动信息 (Interactable + Movable 合并可视化)
    # ========================================================================
    if has_motion:
        # 颜色编码: 红色=仅Interactable, 蓝色=仅Movable, 紫色=Both, 灰色=Background
        pred_motion_colors = np.zeros((len(vis_mov_pred), 3), dtype=np.float32)
        pred_motion_text = []
        for i in range(len(vis_mov_pred)):
            is_inter = vis_inter_pred[i] == 1
            is_mov = vis_mov_pred[i] > 0  # movable: rotation(1) or translation(2)
            
            if is_inter and is_mov:
                pred_motion_colors[i] = [1.0, 0.0, 1.0]  # Magenta (both)
                pred_motion_text.append('Both Inter+Mov')
            elif is_inter:
                pred_motion_colors[i] = [1.0, 0.0, 0.0]  # Red (inter only)
                pred_motion_text.append('Interactable')
            elif is_mov:
                pred_motion_colors[i] = [0.0, 0.0, 1.0]  # Blue (mov only)
                pred_motion_text.append(f'Movable-{MOVABLE_CLASS_NAMES[vis_mov_pred[i]]}')
            else:
                pred_motion_colors[i] = [0.5, 0.5, 0.5]  # Gray (background)
                pred_motion_text.append('Background')
        
        fig.add_trace(
            go.Scatter3d(
                x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in pred_motion_colors],
                ),
                name='Pred Motion',
                text=pred_motion_text,
                hovertemplate=(
                    '<b>Pred Motion:</b> %{text}<br>'
                    'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ),
            ),
            row=1, col=2
        )
        
        # 添加运动箭头
        if mov_pred is not None and config.show_motion_arrows:
            mov_labels = mov_pred['seg_labels']
            origin_pred = mov_pred['origin_pred']
            axis_pred = mov_pred['axis_pred']
            
            # 实例分割（基于预测的可动标签）
            instance_labels = segment_instances_by_proximity(
                coords, mov_labels, eps=config.clustering_eps, min_samples=config.clustering_min_samples
            )
            
            unique_instances = np.unique(instance_labels[instance_labels > 0])
            motion_info_list = []
            
            for inst_id in unique_instances[:20]:  # 限制最多显示20个实例
                mask = instance_labels == inst_id
                if mask.sum() < 50:
                    continue
                
                sem_label = inst_id // 10000
                if sem_label == 0:  # 跳过背景
                    continue
                
                # Compute mean origin, then snap to nearest movable point
                inst_origin_raw = origin_pred[mask].mean(axis=0)
                inst_coords_mask = coords[mask]
                distances = np.linalg.norm(inst_coords_mask - inst_origin_raw, axis=1)
                nearest_idx = np.argmin(distances)
                inst_origin = inst_coords_mask[nearest_idx]
                inst_axis = axis_pred[mask].mean(axis=0)
                inst_axis = inst_axis / (np.linalg.norm(inst_axis) + 1e-6)
                
                motion_type = MOVABLE_CLASS_NAMES[sem_label] if sem_label < len(MOVABLE_CLASS_NAMES) else 'unknown'
                
                motion_info_list.append({
                    'instance_id': int(inst_id % 10000),
                    'motion_type': motion_type,
                    'origin': inst_origin,
                    'axis': inst_axis,
                    'point_count': int(mask.sum()),
                })
                
                # 添加预测运动箭头
                arrows = create_motion_arrow_trace(
                    inst_origin, inst_axis, motion_type,
                    length=config.arrow_scale, name=f'Pred_{inst_id % 10000}',
                    is_prediction=True, instance_id=int(inst_id % 10000), show_label=True,
                )
                for trace in arrows:
                    fig.add_trace(trace, row=1, col=2)
    else:
        # 如果没有运动模型，显示RGB
        fig.add_trace(
            go.Scatter3d(
                x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in vis_colors]
                ),
                name='RGB',
            ),
            row=1, col=2
        )
    
    # ========================================================================
    # 2行1列: 语义分割 GT
    # ========================================================================
    sem_gt_colors = color_palette[np.clip(vis_semantic_gt, 0, num_classes-1)]
    fig.add_trace(
        go.Scatter3d(
            x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sem_gt_colors]
            ),
            name='Semantic GT',
            hovertemplate='Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            text=[class_names[l] if 0 <= l < len(class_names) else f'Unknown_{l}' for l in vis_semantic_gt],
        ),
        row=2, col=1
    )
    
    # ========================================================================
    # 2行2列: 语义分割预测（使用实例级别的统一颜色）
    # ========================================================================
    # 使用实例分割和多数投票来为每个部件分配统一的颜色
    from collections import Counter

    # 创建每个点的颜色数组（基于实例的多数投票）
    sem_pred_colors_instance = np.zeros((len(vis_coords), 3), dtype=np.float32)

    # 对语义分割进行实例分割
    instance_labels = segment_instances_by_proximity(
        vis_coords, vis_semantic_pred, eps=config.clustering_eps, min_samples=config.clustering_min_samples
    )

    # 获取所有唯一的实例ID（忽略背景，ID=0）
    unique_instances = np.unique(instance_labels[instance_labels > 0])

    # 为每个实例计算多数语义类别并分配统一颜色
    for inst_id in unique_instances:
        mask = instance_labels == inst_id
        inst_semantic_labels = vis_semantic_pred[mask]

        # 使用多数投票确定该实例的主要类别
        if len(inst_semantic_labels) > 0:
            counter = Counter(inst_semantic_labels)
            majority_class = counter.most_common(1)[0][0]

            # 获取该类别的颜色
            color_idx = min(majority_class, num_classes - 1)
            sem_pred_colors_instance[mask] = color_palette[color_idx]

    # 背景点（实例ID=0）保持原有点级颜色
    background_mask = instance_labels == 0
    if background_mask.sum() > 0:
        for i in np.where(background_mask)[0]:
            color_idx = min(vis_semantic_pred[i], num_classes - 1)
            sem_pred_colors_instance[i] = color_palette[color_idx]

    fig.add_trace(
        go.Scatter3d(
            x=vis_coords[:, 0], y=vis_coords[:, 1], z=vis_coords[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sem_pred_colors_instance]
            ),
            name='Semantic Pred (Instance-level)',
            hovertemplate='Class: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            text=[class_names[l] if 0 <= l < len(class_names) else f'Unknown_{l}' for l in vis_semantic_pred],
        ),
        row=2, col=2
    )
    
    # 构建标题（包含所有统计信息）
    motion_info_text = ""
    if has_motion and mov_pred is not None:
        # 统计预测的inter和mov
        num_inter_pred = (inter_pred['seg_labels'] == 1).sum() if inter_pred else 0
        num_mov_pred = (mov_pred['seg_labels'] > 0).sum()
        num_both_pred = 0
        if inter_pred:
            num_both_pred = ((inter_pred['seg_labels'] == 1) & (mov_pred['seg_labels'] > 0)).sum()
        
        num_rotation = (mov_pred['seg_labels'] == 1).sum()
        num_translation = (mov_pred['seg_labels'] == 2).sum()
        
        # 统计预测的实例数
        if 'motion_info_list' in locals():
            num_instances = len(motion_info_list)
            num_rot_inst = sum(1 for info in motion_info_list if info['motion_type'] == 'rotation')
            num_trans_inst = sum(1 for info in motion_info_list if info['motion_type'] == 'translation')
        else:
            num_instances = 0
            num_rot_inst = 0
            num_trans_inst = 0
        
        motion_info_text = (
            f'<br><span style="font-size:11px; color:#666">'
            f'<b>预测运动统计:</b> Inter={num_inter_pred:,}, Mov={num_mov_pred:,}, Both={num_both_pred:,} | '
            f'实例={num_instances} (🔄旋转:{num_rot_inst}, ↔️平移:{num_trans_inst})<br>'
            f'⚠️ 运动信息均为模型预测（无GT） | 🎨 颜色: 🔴红=Inter, 🔵蓝=Mov, 🟣紫=Both, ⚪灰=BG'
            f'</span>'
        )
    
    # 语义分割统计
    semantic_stats = ""
    if semantic_pred is not None:
        semantic_counts = Counter(vis_semantic_pred)
        top_classes = semantic_counts.most_common(5)
        top_str = ', '.join([f'{class_names[c] if c < len(class_names) else c}' for c, _ in top_classes])
        semantic_stats = f' | 语义Acc: {semantic_accuracy*100:.1f}% | Top5: {top_str}'
    
    title_text = (
        f'<b>Zarr场景可视化: {scene_id}</b> (点数: {num_points:,}, 类别数: {num_classes})'
        f'{semantic_stats}'
        f'{motion_info_text}'
    )
    
    scene_config = dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    
    layout_dict = dict(
        title=dict(
            text=title_text,
            x=0.5, xanchor='center',
            font=dict(size=13)
        ),
        scene=scene_config,
        scene2=scene_config,
        scene3=scene_config,
        scene4=scene_config,
        showlegend=False,
        height=1100,  # 2x2布局固定高度
        width=1400,
        margin=dict(l=20, r=20, t=150, b=20),
    )
    
    fig.update_layout(**layout_dict)
    
    html_path = output_dir / f"{scene_id}_overview.html"
    
    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={'scrollZoom': True, 'displayModeBar': True}
    )
    custom_css = '''
    <style>
    body { overflow-x: auto; overflow-y: auto; min-width: 1200px; }
    </style>
    '''
    html_content = html_content.replace('</head>', custom_css + '</head>')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"  ✓ Saved: {html_path.name}")
    
    # ========================================================================
    # 为预测的运动实例创建详细可视化 (如果有运动模型)
    # ========================================================================
    if mov_pred is not None and config.show_motion_arrows:
        print(f"  📦 Creating motion instance visualizations...")
        
        mov_labels = mov_pred['seg_labels']
        origin_pred_all = mov_pred['origin_pred']
        axis_pred_all = mov_pred['axis_pred']
        inter_labels = inter_pred['seg_labels'] if inter_pred is not None else np.zeros(num_points, dtype=np.int32)
        
        # 实例分割
        instance_labels = segment_instances_by_proximity(
            coords, mov_labels, eps=config.clustering_eps, min_samples=config.clustering_min_samples
        )
        
        unique_instances = np.unique(instance_labels[instance_labels > 0])
        
        instance_count = 0
        for inst_label in unique_instances[:30]:  # 限制实例数量
            mask = instance_labels == inst_label
            if mask.sum() < 100:
                continue
            
            sem_label = inst_label // 10000
            if sem_label == 0:  # 跳过背景
                continue
                
            inst_id = inst_label % 10000
            motion_type = MOVABLE_CLASS_NAMES[sem_label] if sem_label < len(MOVABLE_CLASS_NAMES) else 'unknown'
            
            inst_coords = coords[mask]
            inst_colors = colors[mask]
            inst_semantic = semantic_gt[mask]
            inst_mov = mov_labels[mask]
            inst_inter = inter_labels[mask]
            # Compute mean origin, then snap to nearest movable point
            inst_origin_raw = origin_pred_all[mask].mean(axis=0)
            distances = np.linalg.norm(inst_coords - inst_origin_raw, axis=1)
            nearest_idx = np.argmin(distances)
            inst_origin = inst_coords[nearest_idx]
            inst_axis = axis_pred_all[mask].mean(axis=0)
            inst_axis = inst_axis / (np.linalg.norm(inst_axis) + 1e-6)
            
            # 下采样
            if len(inst_coords) > 20000:
                idx = np.random.choice(len(inst_coords), 20000, replace=False)
                inst_coords = inst_coords[idx]
                inst_colors = inst_colors[idx]
                inst_semantic = inst_semantic[idx]
                inst_mov = inst_mov[idx]
                inst_inter = inst_inter[idx]
            
            # 创建实例HTML (3视图: RGB, 语义GT, 运动预测)
            fig_inst = make_subplots(
                rows=1, cols=3,
                subplot_titles=('RGB', '语义分割 GT', '运动/交互 Pred'),
                specs=[[{'type': 'scatter3d'}]*3],
            )
            
            # RGB
            fig_inst.add_trace(
                go.Scatter3d(
                    x=inst_coords[:, 0], y=inst_coords[:, 1], z=inst_coords[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in inst_colors]),
                ),
                row=1, col=1
            )
            
            # 语义GT
            sem_colors = color_palette[np.clip(inst_semantic, 0, num_classes-1)]
            fig_inst.add_trace(
                go.Scatter3d(
                    x=inst_coords[:, 0], y=inst_coords[:, 1], z=inst_coords[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in sem_colors]),
                    hovertemplate='Semantic: %{text}',
                    text=[class_names[l] if 0 <= l < len(class_names) else f'Unknown_{l}' for l in inst_semantic],
                ),
                row=1, col=2
            )
            
            # 运动预测
            mov_colors = np.array([MOVABLE_COLORS.get(l, [0.5, 0.5, 0.5]) for l in inst_mov])
            fig_inst.add_trace(
                go.Scatter3d(
                    x=inst_coords[:, 0], y=inst_coords[:, 1], z=inst_coords[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in mov_colors]),
                ),
                row=1, col=3
            )
            
            # 添加运动箭头
            arrows = create_motion_arrow_trace(
                inst_origin, inst_axis, motion_type,
                length=config.arrow_scale, name=f'Motion',
                is_prediction=True, instance_id=inst_id, show_label=True,
            )
            for trace in arrows:
                fig_inst.add_trace(trace, row=1, col=3)
            
            # 运动范围信息
            if motion_type == 'rotation':
                range_str = f'🔄 旋转范围: ±45°'
            elif motion_type == 'translation':
                range_str = f'↔️ 平移范围: {config.arrow_scale * 1.2:.2f}m'
            else:
                range_str = ''
            
            # 获取最常见的语义类别
            semantic_counts = Counter(inst_semantic[inst_semantic >= 0])
            if semantic_counts:
                top_sem = semantic_counts.most_common(3)
                sem_str = ', '.join([f'{class_names[s] if s < len(class_names) else s}({c})' for s, c in top_sem])
            else:
                sem_str = 'N/A'
            
            interactable_ratio = (inst_inter == 1).mean() * 100 if len(inst_inter) > 0 else 0
            
            fig_inst.update_layout(
                title=dict(
                    text=(
                        f'<b>Instance {inst_id}: {motion_type}</b> ({mask.sum():,} pts)<br>'
                        f'<span style="font-size:11px">'
                        f'语义: {sem_str} | 可交互比例: {interactable_ratio:.1f}% | {range_str}<br>'
                        f'Origin: [{inst_origin[0]:.2f}, {inst_origin[1]:.2f}, {inst_origin[2]:.2f}] | '
                        f'Axis: [{inst_axis[0]:.2f}, {inst_axis[1]:.2f}, {inst_axis[2]:.2f}]<br>'
                        f'⚠️ 运动信息为模型预测（无GT）'
                        f'</span>'
                    ),
                    x=0.5, xanchor='center',
                ),
                scene=dict(aspectmode='data'),
                scene2=dict(aspectmode='data'),
                scene3=dict(aspectmode='data'),
                showlegend=False,
                height=600,
                width=1200,
                margin=dict(l=20, r=20, t=120, b=20),
            )
            
            inst_html_path = output_dir / f"{scene_id}_motion_{inst_id}_{motion_type}.html"
            
            html_content = fig_inst.to_html(
                full_html=True,
                include_plotlyjs=True,
                config={'scrollZoom': True, 'displayModeBar': True}
            )
            custom_css = '''
            <style>
            body { overflow-x: auto; overflow-y: auto; min-width: 1200px; }
            </style>
            '''
            html_content = html_content.replace('</head>', custom_css + '</head>')
            with open(inst_html_path, 'w') as f:
                f.write(html_content)
            
            instance_count += 1
        
        print(f"  ✓ Created {instance_count} motion instance visualizations")


if __name__ == "__main__":
    main()
