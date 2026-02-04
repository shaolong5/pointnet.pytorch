"""
================================================================================
Articulate3D Training Script
================================================================================

Train USDNet for:
1. Movable Part Segmentation (semantic: background=0, rotation=1, translation=2)
2. Articulation Parameter Prediction (origin, axis)

Based on the Articulate3D challenge dataset format.
"""
import os
import sys
import logging
import json
import pickle
import yaml
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import MinkowskiEngine as ME
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TrainingConfig:
    """Training configuration for Articulate3D"""

    # data
    data_dir: str = "./data/processed/articulate3d_challenge_mov"
    json_dir: str = "/mnt/nfs/home/st185933/Articulate3D"  # Original JSON annotations
    train_mode: str = "train"  # "train" or "train_validation"
    voxel_size: float = 0.02
    num_workers: int = 4
    batch_size: int = 1
    pin_memory: bool = True
    max_points_per_sample: int = 100000

    # model
    num_classes: int = 3  # background=0, rotation=1, translation=2
    feature_dim_3d: int = 256
    dropout: float = 0.1
    bn_momentum: float = 0.1

    # articulation prediction
    predict_articulation: bool = True
    weight_seg: float = 1.0
    weight_origin: float = 1.0
    weight_axis: float = 1.0
    weight_range: float = 2.0  # Increased weight since range loss is normalized by π

    # optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    decay_rate: float = 0.95
    decay_step: int = 20
    gradient_clip: float = 1.0

    # Warmup
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6

    # others
    use_mixed_precision: bool = True
    device: str = "cuda:0"
    save_dir: str = "./checkpoints/articulate3d"
    seed: int = 42
    
    # coarse-to-fine training
    use_coarse_to_fine: bool = True
    c2f_alpha: float = 100.0
    c2f_decay: float = 0.4
    c2f_rad: float = 0.1


# ============================================================================
# Articulate3D Data loading
# ============================================================================
CLASS_LABELS = ['background', 'rotation', 'translation']
CLASS_IDS = [0, 1, 2]


def load_yaml(filepath):
    """Load yaml file"""
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)


def load_h5_articulation(filepath):
    """Load articulation parameters from h5 file"""
    articulations = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            inst_id = int(key)
            articulations[inst_id] = {
                'sem_id': int(f[key]['sem_id'][()]),
                'axis': np.array(f[key]['axis'][()], dtype=np.float32),
                'origin': np.array(f[key]['origin'][()], dtype=np.float32),
                'inter_mask': np.array(f[key]['inter_mask'][()], dtype=bool),
            }
    return articulations


def load_motion_range_from_json(json_dir: str, scene_id: str, json_file: str = None) -> Dict[int, Dict[str, float]]:
    """Load motion range from original JSON annotation files
    
    Args:
        json_dir: Directory containing JSON files (optional if json_file provided)
        scene_id: Scene identifier (e.g., '0a7cc12c0e')
        json_file: Direct path to JSON file (optional, overrides json_dir)
    
    Returns:
        Dictionary mapping pid to {'rangeMin': float, 'rangeMax': float, 'type': str}
    """
    if json_file:
        json_path = Path(json_file)
    elif json_dir:
        json_path = Path(json_dir) / f"{scene_id}_artic.json"
    else:
        return {}
    
    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_path}, returning empty ranges")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        range_dict = {}
        for artic in data['data']['articulations']:
            pid = artic['pid']
            range_dict[pid] = {
                'rangeMin': artic['rangeMin'],
                'rangeMax': artic['rangeMax'],
                'type': artic['type']  # 'rotation' or 'translation'
            }
        
        return range_dict
    
    except Exception as e:
        logger.error(f"Failed to load JSON {json_path}: {e}")
        return {}


class Articulate3DDataset(Dataset):
    """Dataset for Articulate3D movable part segmentation and articulation prediction"""

    def __init__(
        self,
        data_dir: str,
        json_dir: str = None,
        mode: str = "train",
        voxel_size: float = 0.02,
        max_points: int = 100000,
        augment: bool = True,
        use_coarse_to_fine: bool = False,
        c2f_alpha: float = 100.0,
        c2f_decay: float = 0.4,
        epoch: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.json_dir = Path(json_dir) if json_dir else self.data_dir
        self.mode = mode
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.augment = augment and mode == "train"
        self.use_coarse_to_fine = use_coarse_to_fine
        self.c2f_alpha = c2f_alpha
        self.c2f_decay = c2f_decay
        self.epoch = epoch

        # Load database - try yaml first, fallback to scanning directory
        if mode == "train_validation":
            db_file = self.data_dir / "train_validation_database.yaml"
        else:
            db_file = self.data_dir / f"{mode}_database.yaml"
        
        if db_file.exists():
            # Load from yaml
            self.database = load_yaml(str(db_file))
            
            # Fix relative paths in database to absolute paths
            base_dir = self.data_dir.parent.parent.parent
            for sample in self.database:
                if 'filepath' in sample and not os.path.isabs(sample['filepath']):
                    sample['filepath'] = str(base_dir / sample['filepath'])
                if 'articulation_gt_file' in sample and not os.path.isabs(sample['articulation_gt_file']):
                    sample['articulation_gt_file'] = str(base_dir / sample['articulation_gt_file'])
                if 'expand_dict_file' in sample and not os.path.isabs(sample['expand_dict_file']):
                    sample['expand_dict_file'] = str(base_dir / sample['expand_dict_file'])
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
                json_file = self.json_dir / f"{scene_id}_artic.json"
                
                if json_file.exists():
                    self.database.append({
                        'filepath': str(npy_file),
                        'scene_id': scene_id,
                        'json_file': str(json_file)
                    })
            
            logger.info(f"✓ Found {len(self.database)} scenes by scanning directory")
        
        if len(self.database) == 0:
            raise ValueError(f"No data found in {self.data_dir}")

        # Load or compute color mean/std
        color_stats_file = self.data_dir / "color_mean_std.yaml"
        if color_stats_file.exists():
            color_stats = load_yaml(str(color_stats_file))
            self.color_mean = np.array(color_stats['mean'], dtype=np.float32)
            self.color_std = np.array(color_stats['std'], dtype=np.float32)
        else:
            logger.info("⚠️  Color stats not found, using default normalization")
            self.color_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.color_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def __len__(self):
        return len(self.database)
    
    def set_epoch(self, epoch: int):
        """Update epoch for coarse-to-fine training"""
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.database[idx]
        scene_id = sample.get('scene', sample.get('scene_id', ''))
        
        # Load point cloud data
        # Format: [x, y, z, r, g, b, nx, ny, nz, sem_gt, inst_gt, segments, inter_gt]
        data = np.load(sample['filepath']).astype(np.float32)
        
        coords = data[:, :3]
        colors = data[:, 3:6] / 255.0  # Normalize to [0, 1]
        normals = data[:, 6:9]
        sem_gt = data[:, 9].astype(np.int32)
        inst_gt = data[:, 10].astype(np.int32)
        
        # Load articulation parameters
        articulations = {}
        if 'articulation_gt_file' in sample and os.path.exists(sample['articulation_gt_file']):
            articulations = load_h5_articulation(sample['articulation_gt_file'])
        
        # Load motion range from JSON
        motion_ranges = {}
        json_file = sample.get('json_file', None)
        if json_file and os.path.exists(json_file):
            # Load directly from specified json file
            motion_ranges = load_motion_range_from_json(None, scene_id, json_file)
        elif self.json_dir:
            # Try to find json file in json_dir
            motion_ranges = load_motion_range_from_json(self.json_dir, scene_id)
        
        # Coarse-to-fine: expand instance labels over epochs
        if self.use_coarse_to_fine and self.mode == "train" and 'expand_dict_file' in sample:
            expand_file = sample['expand_dict_file']
            if os.path.exists(expand_file):
                with open(expand_file, 'rb') as f:
                    expand_dict = pickle.load(f)
                
                # Compute expansion ratio based on epoch
                ratio = 1 - np.exp(-self.epoch / self.c2f_alpha)
                ratio = min(ratio, self.c2f_decay)
                
                expand_idx = expand_dict.get('expand_idx_records', np.array([]))
                expand_inst = expand_dict.get('expand_inst_records', np.array([]))
                expand_sem = expand_dict.get('expand_sem_records', np.array([]))
                
                if len(expand_idx) > 0:
                    num_expand = int(len(expand_idx) * ratio)
                    if num_expand > 0:
                        sample_idx = np.random.choice(len(expand_idx), num_expand, replace=False)
                        sem_gt[expand_idx[sample_idx]] = expand_sem[sample_idx]
                        inst_gt[expand_idx[sample_idx]] = expand_inst[sample_idx]
        
        # Normalize colors
        colors = (colors - self.color_mean) / (self.color_std + 1e-6)
        
        # Data augmentation
        if self.augment:
            # Random rotation around Z axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation_z = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            coords = coords @ rotation_z.T
            normals = normals @ rotation_z.T
            
            # Also rotate articulation parameters
            for inst_id in articulations:
                articulations[inst_id]['origin'] = articulations[inst_id]['origin'] @ rotation_z.T
                articulations[inst_id]['axis'] = articulations[inst_id]['axis'] @ rotation_z.T
            
            # Random scale
            scale = np.random.uniform(0.95, 1.05)
            coords = coords * scale
            for inst_id in articulations:
                articulations[inst_id]['origin'] = articulations[inst_id]['origin'] * scale
            
            # Random jitter
            if np.random.random() < 0.5:
                jitter = np.random.normal(0, 0.01, coords.shape).astype(np.float32)
                coords = coords + jitter
            
            # Random flip
            if np.random.random() < 0.5:
                coords[:, 0] = -coords[:, 0]
                normals[:, 0] = -normals[:, 0]
                for inst_id in articulations:
                    articulations[inst_id]['origin'][0] = -articulations[inst_id]['origin'][0]
                    articulations[inst_id]['axis'][0] = -articulations[inst_id]['axis'][0]
        
        # Subsample if too many points
        if len(coords) > self.max_points:
            # Stratified sampling to preserve rare classes
            unique_labels = np.unique(sem_gt)
            selected_idx = []
            points_per_class = self.max_points // len(unique_labels)
            
            for label in unique_labels:
                label_mask = sem_gt == label
                label_idx = np.where(label_mask)[0]
                n_select = min(len(label_idx), points_per_class)
                selected_idx.extend(np.random.choice(label_idx, n_select, replace=False))
            
            # Fill remaining quota
            remaining = self.max_points - len(selected_idx)
            if remaining > 0:
                all_idx = np.arange(len(coords))
                remaining_idx = np.setdiff1d(all_idx, selected_idx)
                if len(remaining_idx) > 0:
                    extra_idx = np.random.choice(remaining_idx, min(remaining, len(remaining_idx)), replace=False)
                    selected_idx.extend(extra_idx)
            
            selected_idx = np.array(selected_idx)
            coords = coords[selected_idx]
            colors = colors[selected_idx]
            normals = normals[selected_idx]
            sem_gt = sem_gt[selected_idx]
            inst_gt = inst_gt[selected_idx]
        
        # Build per-point articulation targets
        # For each point, we store the articulation parameters of its instance
        origin_targets = np.zeros((len(coords), 3), dtype=np.float32)
        axis_targets = np.zeros((len(coords), 3), dtype=np.float32)
        range_targets = np.zeros((len(coords), 2), dtype=np.float32)  # [rangeMin, rangeMax]
        articulation_mask = np.zeros(len(coords), dtype=np.float32)
        
        for inst_id, arti_params in articulations.items():
            inst_mask = inst_gt == inst_id
            if inst_mask.sum() > 0:
                origin_targets[inst_mask] = arti_params['origin']
                axis_targets[inst_mask] = arti_params['axis']
                articulation_mask[inst_mask] = 1.0
                
                # Load motion range from JSON if available
                if inst_id in motion_ranges:
                    range_targets[inst_mask, 0] = motion_ranges[inst_id]['rangeMin']
                    range_targets[inst_mask, 1] = motion_ranges[inst_id]['rangeMax']
        
        # Normalize axis to unit vector
        axis_norms = np.linalg.norm(axis_targets, axis=1, keepdims=True)
        axis_targets = axis_targets / np.clip(axis_norms, 1e-6, None)
        
        return {
            'scene_id': scene_id,
            'coords': torch.from_numpy(coords).float(),
            'colors': torch.from_numpy(colors).float(),
            'normals': torch.from_numpy(normals).float(),
            'sem_labels': torch.from_numpy(sem_gt).long(),
            'inst_labels': torch.from_numpy(inst_gt).long(),
            'origin_targets': torch.from_numpy(origin_targets).float(),
            'axis_targets': torch.from_numpy(axis_targets).float(),
            'range_targets': torch.from_numpy(range_targets).float(),
            'articulation_mask': torch.from_numpy(articulation_mask).float(),
        }


def collate_fn_articulate3d(batch: List[Dict], voxel_size: float = 0.02) -> Dict[str, Any]:
    """Collate function for sparse convolution"""
    all_coords = []
    all_features = []
    all_sem_labels = []
    all_inst_labels = []
    all_origin_targets = []
    all_axis_targets = []
    all_range_targets = []
    all_arti_masks = []

    for batch_idx, data in enumerate(batch):
        coords = data['coords'].numpy()
        colors = data['colors'].numpy()
        normals = data['normals'].numpy()
        sem_labels = data['sem_labels'].numpy()
        inst_labels = data['inst_labels'].numpy()
        origin_targets = data['origin_targets'].numpy()
        axis_targets = data['axis_targets'].numpy()
        range_targets = data['range_targets'].numpy()
        arti_mask = data['articulation_mask'].numpy()

        # Voxelize coordinates
        voxel_coords = np.floor(coords / voxel_size).astype(np.int32)
        
        # Deduplicate voxel coordinates
        unique_coords, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        
        coords_unique = coords[unique_indices]
        colors_unique = colors[unique_indices]
        normals_unique = normals[unique_indices]
        sem_labels_unique = sem_labels[unique_indices]
        inst_labels_unique = inst_labels[unique_indices]
        origin_unique = origin_targets[unique_indices]
        axis_unique = axis_targets[unique_indices]
        range_unique = range_targets[unique_indices]
        arti_mask_unique = arti_mask[unique_indices]

        # Add batch index
        batch_indices = np.full((len(unique_indices), 1), batch_idx, dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])

        # Features: RGB + normals + coordinates (9 dims)
        combined_features = np.hstack([colors_unique, normals_unique, coords_unique])

        all_coords.append(coords_with_batch)
        all_features.append(combined_features)
        all_sem_labels.append(sem_labels_unique)
        all_inst_labels.append(inst_labels_unique)
        all_origin_targets.append(origin_unique)
        all_axis_targets.append(axis_unique)
        all_range_targets.append(range_unique)
        all_arti_masks.append(arti_mask_unique)

    return {
        'coords': torch.from_numpy(np.vstack(all_coords)).int(),
        'features': torch.from_numpy(np.vstack(all_features)).float(),
        'sem_labels': torch.from_numpy(np.concatenate(all_sem_labels)).long(),
        'inst_labels': torch.from_numpy(np.concatenate(all_inst_labels)).long(),
        'origin_targets': torch.from_numpy(np.vstack(all_origin_targets)).float(),
        'axis_targets': torch.from_numpy(np.vstack(all_axis_targets)).float(),
        'range_targets': torch.from_numpy(np.vstack(all_range_targets)).float(),
        'articulation_mask': torch.from_numpy(np.concatenate(all_arti_masks)).float(),
    }


# ============================================================================
# Model - USDNet with Articulation Prediction
# ============================================================================
class MinkowskiBasicBlock(nn.Module):
    """BasicBlock for Res16UNet"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1, bn_momentum: float = 0.1):
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, dilation=dilation, dimension=3,
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, dilation=dilation, dimension=3,
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, dimension=3,
                ),
                ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum),
            )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Res16UNetBackbone(nn.Module):
    """Res16UNet backbone"""
    BLOCK = MinkowskiBasicBlock
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels: int = 9, out_channels: int = 256, bn_momentum: float = 0.1):
        super().__init__()

        self.inplanes = self.INIT_DIM
        self.relu = ME.MinkowskiReLU(inplace=True)

        # Initial conv
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=self.INIT_DIM,
            kernel_size=3, stride=1, dimension=3,
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)

        # Encoder
        self.conv1p1s2 = ME.MinkowskiConvolution(
            in_channels=self.INIT_DIM, out_channels=self.INIT_DIM,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)
        self.block1 = self._make_layer(self.INIT_DIM, self.PLANES[0], self.LAYERS[0], bn_momentum)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block2 = self._make_layer(self.inplanes, self.PLANES[1], self.LAYERS[1], bn_momentum)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(self.inplanes, self.PLANES[2], self.LAYERS[2], bn_momentum)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(self.inplanes, self.PLANES[3], self.LAYERS[3], bn_momentum)

        # Decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[4],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.inplanes, self.PLANES[4], self.LAYERS[4], bn_momentum)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[5],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.inplanes, self.PLANES[5], self.LAYERS[5], bn_momentum)

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[6],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.inplanes, self.PLANES[6], self.LAYERS[6], bn_momentum)

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[7],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.inplanes, self.PLANES[7], self.LAYERS[7], bn_momentum)

        # Final projection
        self.final = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=out_channels,
            kernel_size=1, stride=1, dimension=3,
        )

    def _make_layer(self, in_channels, out_channels, blocks, bn_momentum):
        layers = []
        layers.append(MinkowskiBasicBlock(in_channels, out_channels, stride=1, bn_momentum=bn_momentum))
        self.inplanes = out_channels * self.BLOCK.expansion
        for _ in range(1, blocks):
            layers.append(MinkowskiBasicBlock(self.inplanes, out_channels, stride=1, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # Initial features
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Encoder
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # Decoder with skip connections
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        feat_3d = self.final(out)
        return feat_3d


class ArticulateUSDNet(nn.Module):
    """
    USDNet for Articulate3D challenge
    
    Outputs:
    - Semantic segmentation (background=0, rotation=1, translation=2)
    - Articulation origin (3D point)
    - Articulation axis (3D unit vector)
    """

    def __init__(self, num_classes: int = 3, feature_dim: int = 256,
                 bn_momentum: float = 0.1, dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.backbone = Res16UNetBackbone(
            in_channels=9, out_channels=feature_dim, bn_momentum=bn_momentum,
        )

        # Semantic segmentation head
        self.seg_head = nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, num_classes),
        )

        # Articulation origin prediction head
        self.origin_head = nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(64, 3),
        )

        # Articulation axis prediction head
        self.axis_head = nn.Sequential(
            ME.MinkowskiLinear(feature_dim, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(64, 3),
        )

        # Motion range prediction head [rangeMin, rangeMax]
        self.range_head = nn.Sequential(
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
        
        seg_logits = self.seg_head(feat_3d)
        origin_pred = self.origin_head(feat_3d)
        axis_pred = self.axis_head(feat_3d)
        range_pred = self.range_head(feat_3d)
        
        return {
            'seg_logits': seg_logits,
            'origin_pred': origin_pred,
            'axis_pred': axis_pred,
            'range_pred': range_pred,
            'features': feat_3d,
        }


# ============================================================================
# Loss functions
# ============================================================================
class ArticulateLoss(nn.Module):
    """Combined loss for segmentation and articulation prediction"""

    def __init__(self, num_classes: int = 3, 
                 weight_seg: float = 1.0,
                 weight_origin: float = 1.0,
                 weight_axis: float = 1.0,
                 weight_range: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()

        self.weight_seg = weight_seg
        self.weight_origin = weight_origin
        self.weight_axis = weight_axis
        self.weight_range = weight_range

        # Ignore background (class 0) for segmentation
        self.seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs: Dict[str, ME.SparseTensor],
                sem_labels: torch.Tensor,
                origin_targets: torch.Tensor,
                axis_targets: torch.Tensor,
                range_targets: torch.Tensor,
                articulation_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        seg_logits = outputs['seg_logits'].features
        origin_pred = outputs['origin_pred'].features
        axis_pred = outputs['axis_pred'].features
        range_pred = outputs['range_pred'].features

        # Segmentation loss
        loss_seg = self.seg_loss_fn(seg_logits, sem_labels)

        # Articulation losses (only for movable parts)
        arti_mask = articulation_mask > 0.5
        
        if arti_mask.sum() > 0:
            # Origin loss: L1 distance
            loss_origin = F.l1_loss(
                origin_pred[arti_mask], 
                origin_targets[arti_mask]
            )
            
            # Axis loss: cosine similarity + L2
            axis_pred_norm = F.normalize(axis_pred[arti_mask], dim=1)
            axis_target_norm = F.normalize(axis_targets[arti_mask], dim=1)
            
            # Axis can point in either direction, so we take the max of both
            cos_sim_pos = (axis_pred_norm * axis_target_norm).sum(dim=1)
            cos_sim_neg = (axis_pred_norm * (-axis_target_norm)).sum(dim=1)
            cos_sim = torch.max(cos_sim_pos, cos_sim_neg)
            loss_axis = (1 - cos_sim).mean()
            
            # Range loss: normalized by typical range values
            # Rotation: typical range is ~π rad, normalize by π
            # Translation: typical range is ~1m, keep as is
            # We use a unified normalization factor for simplicity
            range_min_pred = range_pred[arti_mask, 0]
            range_max_pred = range_pred[arti_mask, 1]
            range_min_target = range_targets[arti_mask, 0]
            range_max_target = range_targets[arti_mask, 1]
            
            # Normalize range values by π (since most are rotations in radians)
            # This makes the loss scale similar to other losses
            RANGE_NORM = 3.14159  # π
            loss_range_min = F.l1_loss(range_min_pred / RANGE_NORM, range_min_target / RANGE_NORM)
            loss_range_max = F.l1_loss(range_max_pred / RANGE_NORM, range_max_target / RANGE_NORM)
            
            # Constraint: max >= min (in original scale)
            constraint_loss = F.relu(range_min_pred - range_max_pred).mean() / RANGE_NORM
            
            loss_range = loss_range_min + loss_range_max + 0.1 * constraint_loss
        else:
            loss_origin = torch.tensor(0.0, device=seg_logits.device)
            loss_axis = torch.tensor(0.0, device=seg_logits.device)
            loss_range = torch.tensor(0.0, device=seg_logits.device)

        # Total loss
        loss_total = (
            self.weight_seg * loss_seg + 
            self.weight_origin * loss_origin + 
            self.weight_axis * loss_axis +
            self.weight_range * loss_range
        )

        return {
            'loss_total': loss_total,
            'loss_seg': loss_seg.detach(),
            'loss_origin': loss_origin.detach() if isinstance(loss_origin, torch.Tensor) else loss_origin,
            'loss_axis': loss_axis.detach() if isinstance(loss_axis, torch.Tensor) else loss_axis,
            'loss_range': loss_range.detach() if isinstance(loss_range, torch.Tensor) else loss_range,
        }


# ============================================================================
# Metrics
# ============================================================================
class ArticulateMetrics:
    """Metrics for Articulate3D evaluation"""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.origin_errors = []
        self.axis_errors = []
        self.range_errors = []

    def update(self, seg_pred: np.ndarray, seg_target: np.ndarray,
               origin_pred: np.ndarray, origin_target: np.ndarray,
               axis_pred: np.ndarray, axis_target: np.ndarray,
               range_pred: np.ndarray, range_target: np.ndarray,
               articulation_mask: np.ndarray):
        
        # Segmentation metrics
        for p, t in zip(seg_pred, seg_target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[p, t] += 1
        
        # Articulation metrics (only for movable parts)
        arti_mask = articulation_mask > 0.5
        if arti_mask.sum() > 0:
            # Origin error
            origin_err = np.linalg.norm(origin_pred[arti_mask] - origin_target[arti_mask], axis=1)
            self.origin_errors.extend(origin_err.tolist())
            
            # Axis error (angle)
            axis_pred_norm = axis_pred[arti_mask] / (np.linalg.norm(axis_pred[arti_mask], axis=1, keepdims=True) + 1e-6)
            axis_target_norm = axis_target[arti_mask] / (np.linalg.norm(axis_target[arti_mask], axis=1, keepdims=True) + 1e-6)
            
            cos_sim_pos = (axis_pred_norm * axis_target_norm).sum(axis=1)
            cos_sim_neg = (axis_pred_norm * (-axis_target_norm)).sum(axis=1)
            cos_sim = np.maximum(cos_sim_pos, cos_sim_neg)
            cos_sim = np.clip(cos_sim, -1, 1)
            angles = np.arccos(cos_sim) * 180 / np.pi  # degrees
            self.axis_errors.extend(angles.tolist())
            
            # Range error (L1)
            range_err_min = np.abs(range_pred[arti_mask, 0] - range_target[arti_mask, 0])
            range_err_max = np.abs(range_pred[arti_mask, 1] - range_target[arti_mask, 1])
            range_err = (range_err_min + range_err_max) / 2
            self.range_errors.extend(range_err.tolist())

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        
        # Segmentation mIoU
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            if tp + fp + fn > 0:
                ious.append(tp / (tp + fp + fn))
        
        metrics['mIoU'] = np.mean(ious) if ious else 0.0
        metrics['accuracy'] = np.diag(self.confusion_matrix).sum() / max(self.confusion_matrix.sum(), 1)
        
        # Per-class IoU
        for i, name in enumerate(CLASS_LABELS):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            metrics[f'IoU_{name}'] = tp / max(tp + fp + fn, 1)
        
        # Articulation metrics
        if self.origin_errors:
            metrics['origin_error_mean'] = np.mean(self.origin_errors)
            metrics['origin_error_median'] = np.median(self.origin_errors)
        
        if self.axis_errors:
            metrics['axis_error_mean'] = np.mean(self.axis_errors)
            metrics['axis_error_median'] = np.median(self.axis_errors)
        
        if self.range_errors:
            metrics['range_error_mean'] = np.mean(self.range_errors)
            metrics['range_error_median'] = np.median(self.range_errors)
            metrics['range_error'] = np.mean(self.range_errors)  # Alias for convenience
        
        return metrics


# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, config, device='cuda:0'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.train_metrics = ArticulateMetrics(config.num_classes)
        self.val_metrics = ArticulateMetrics(config.num_classes)

        self.use_mixed_precision = config.use_mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.best_metric = 0.0

    def get_lr(self, epoch):
        if epoch < self.config.warmup_epochs:
            return self.config.warmup_lr + (self.config.learning_rate - self.config.warmup_lr) * epoch / self.config.warmup_epochs
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.train_metrics.reset()

        current_lr = self.get_lr(epoch)
        self.set_lr(current_lr)

        loss_meter = defaultdict(list)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train] LR={current_lr:.2e}", leave=False)

        for batch in pbar:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            sem_labels = batch['sem_labels'].to(self.device)
            origin_targets = batch['origin_targets'].to(self.device)
            axis_targets = batch['axis_targets'].to(self.device)
            range_targets = batch['range_targets'].to(self.device)
            arti_mask = batch['articulation_mask'].to(self.device)

            # Create sparse tensor
            x = ME.SparseTensor(
                features=features,
                coordinates=coords,
                device=self.device,
            )

            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(x)
                    losses = self.loss_fn(outputs, sem_labels, origin_targets, axis_targets, range_targets, arti_mask)

                self.scaler.scale(losses['loss_total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                losses = self.loss_fn(outputs, sem_labels, origin_targets, axis_targets, range_targets, arti_mask)
                losses['loss_total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                seg_pred = outputs['seg_logits'].features.argmax(dim=1).cpu().numpy()
                seg_target = sem_labels.cpu().numpy()
                origin_pred = outputs['origin_pred'].features.cpu().numpy()
                axis_pred = outputs['axis_pred'].features.cpu().numpy()
                range_pred = outputs['range_pred'].features.cpu().numpy()
                
                self.train_metrics.update(
                    seg_pred, seg_target,
                    origin_pred, origin_targets.cpu().numpy(),
                    axis_pred, axis_targets.cpu().numpy(),
                    range_pred, range_targets.cpu().numpy(),
                    arti_mask.cpu().numpy()
                )

            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    loss_meter[key].append(val.item())

            self.global_step += 1
            pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in loss_meter.items()})

        if self.scheduler and epoch >= self.config.warmup_epochs:
            self.scheduler.step()

        avg_losses = {key: np.mean(vals) for key, vals in loss_meter.items()}
        metrics = self.train_metrics.get_metrics()

        return avg_losses, metrics

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        self.val_metrics.reset()

        loss_meter = defaultdict(list)
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

        for batch in pbar:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            sem_labels = batch['sem_labels'].to(self.device)
            origin_targets = batch['origin_targets'].to(self.device)
            axis_targets = batch['axis_targets'].to(self.device)
            range_targets = batch['range_targets'].to(self.device)
            arti_mask = batch['articulation_mask'].to(self.device)

            x = ME.SparseTensor(
                features=features,
                coordinates=coords,
                device=self.device,
            )

            outputs = self.model(x)
            losses = self.loss_fn(outputs, sem_labels, origin_targets, axis_targets, range_targets, arti_mask)

            seg_pred = outputs['seg_logits'].features.argmax(dim=1).cpu().numpy()
            seg_target = sem_labels.cpu().numpy()
            origin_pred = outputs['origin_pred'].features.cpu().numpy()
            axis_pred = outputs['axis_pred'].features.cpu().numpy()
            range_pred = outputs['range_pred'].features.cpu().numpy()

            self.val_metrics.update(
                seg_pred, seg_target,
                origin_pred, origin_targets.cpu().numpy(),
                axis_pred, axis_targets.cpu().numpy(),
                range_pred, range_targets.cpu().numpy(),
                arti_mask.cpu().numpy()
            )

            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    loss_meter[key].append(val.item())

        avg_losses = {key: np.mean(vals) for key, vals in loss_meter.items()}
        metrics = self.val_metrics.get_metrics()

        return avg_losses, metrics

    def save_checkpoint(self, epoch, is_best=False, save_latest=True):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_metric': self.best_metric,
        }

        if self.scheduler:
            ckpt['scheduler'] = self.scheduler.state_dict()

        # Save epoch checkpoint
        ckpt_path = Path(self.config.save_dir) / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save(ckpt, ckpt_path)
        logger.info(f"💾 Saved checkpoint: {ckpt_path}")

        # Save latest checkpoint (for easy resume)
        if save_latest:
            latest_path = Path(self.config.save_dir) / "latest_checkpoint.pt"
            torch.save(ckpt, latest_path)
            
            # Also save a resume info file
            resume_info = {
                'latest_checkpoint': str(ckpt_path),
                'epoch': epoch,
                'best_metric': self.best_metric,
            }
            info_path = Path(self.config.save_dir) / "resume_info.json"
            with open(info_path, 'w') as f:
                json.dump(resume_info, f, indent=2)

        # Save best model
        if is_best:
            best_path = Path(self.config.save_dir) / "best_model.pt"
            torch.save(ckpt, best_path)
            logger.info(f"🎯 Saved best model: {best_path} (mIoU: {self.best_metric*100:.2f}%)")

    def train(self, train_loader, val_loader, start_epoch=0):
        logger.info("=" * 80)
        logger.info("🚀 Starting Articulate3D training")
        logger.info("=" * 80)

        for epoch in range(start_epoch, self.config.max_epochs):
            # Update dataset epoch for coarse-to-fine
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            train_losses, train_metrics = self.train_epoch(train_loader, epoch)
            
            logger.info(f"\n📊 Epoch {epoch+1}/{self.config.max_epochs}")
            logger.info(f"  [Train] Loss: {train_losses['loss_total']:.4f} | "
                       f"Seg: {train_losses['loss_seg']:.4f} | "
                       f"Origin: {train_losses.get('loss_origin', 0):.4f} | "
                       f"Axis: {train_losses.get('loss_axis', 0):.4f} | "
                       f"Range: {train_losses.get('loss_range', 0):.4f}")
            logger.info(f"  [Train] mIoU: {train_metrics['mIoU']*100:.2f}% | "
                       f"Acc: {train_metrics['accuracy']*100:.2f}%")
            
            if train_metrics.get('origin_error_mean'):
                logger.info(f"  [Train] Origin Error: {train_metrics['origin_error_mean']:.4f}m | "
                           f"Axis Error: {train_metrics.get('axis_error_mean', 0):.2f}° | "
                           f"Range Error: {train_metrics.get('range_error_mean', 0):.4f}")

            # Validation
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_losses, val_metrics = self.validate(val_loader, epoch)
                
                logger.info(f"  [Val] Loss: {val_losses['loss_total']:.4f} | "
                           f"Seg: {val_losses.get('loss_seg', 0):.4f} | "
                           f"Range: {val_losses.get('loss_range', 0):.4f}")
                logger.info(f"  [Val] mIoU: {val_metrics['mIoU']*100:.2f}% | "
                           f"Acc: {val_metrics['accuracy']*100:.2f}%")
                
                for name in CLASS_LABELS:
                    logger.info(f"    IoU_{name}: {val_metrics[f'IoU_{name}']*100:.2f}%")
                
                if val_metrics.get('origin_error_mean'):
                    logger.info(f"  [Val] Origin Error: {val_metrics['origin_error_mean']:.4f}m | "
                               f"Axis Error: {val_metrics.get('axis_error_mean', 0):.2f}° | "
                               f"Range Error: {val_metrics.get('range_error_mean', 0):.4f}")

                # Save best model
                current_metric = val_metrics['mIoU']
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    logger.info(f"  🎯 New best mIoU: {current_metric*100:.2f}%")

                self.save_checkpoint(epoch, is_best=is_best)
            
            # Save checkpoint every 10 epochs (for resume)
            elif (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        logger.info("\n🎉 Training complete!")
        logger.info(f"Best mIoU: {self.best_metric*100:.2f}%")


# ============================================================================
# Main
# ============================================================================
def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in save_dir"""
    save_path = Path(save_dir)
    
    # First try latest_checkpoint.pt
    latest_ckpt = save_path / "latest_checkpoint.pt"
    if latest_ckpt.exists():
        logger.info(f"📂 Found latest checkpoint: {latest_ckpt}")
        return str(latest_ckpt)
    
    # Otherwise find the highest epoch checkpoint
    ckpt_files = list(save_path.glob("ckpt_epoch_*.pt"))
    if ckpt_files:
        # Extract epoch numbers and find max
        epochs = []
        for f in ckpt_files:
            try:
                epoch_num = int(f.stem.split('_')[-1])
                epochs.append((epoch_num, f))
            except:
                continue
        
        if epochs:
            latest = max(epochs, key=lambda x: x[0])
            logger.info(f"📂 Found latest checkpoint: {latest[1]} (epoch {latest[0]})")
            return str(latest[1])
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Articulate3D Training")
    
    parser.add_argument('--data_dir', type=str, 
                        default="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_mov",
                        help='Path to processed Articulate3D data')
    parser.add_argument('--json_dir', type=str,
                        default="/mnt/nfs/home/st185933/Articulate3D",
                        help='Path to original JSON annotations for motion range')
    parser.add_argument('--train_mode', type=str, default='train', choices=['train', 'train_validation'],
                        help='train: use train set only, train_validation: use train+val for final submission')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_dir', type=str, default="./checkpoints/articulate3d")
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path or "auto" to find latest)')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, 
                        help='Pretrained semantic segmentation checkpoint to load')
    parser.add_argument('--use_coarse_to_fine', action='store_true', help='Enable coarse-to-fine training')

    args = parser.parse_args()

    setup_seed(42)

    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.json_dir = args.json_dir
    config.train_mode = args.train_mode
    config.save_dir = args.save_dir
    config.device = args.device
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.voxel_size = args.voxel_size
    config.num_workers = args.num_workers
    config.use_coarse_to_fine = args.use_coarse_to_fine

    logger.info("=" * 80)
    logger.info("🎓 Articulate3D Training - Movable Part Segmentation & Articulation")
    logger.info("=" * 80)
    logger.info(f"  Data dir: {config.data_dir}")
    logger.info(f"  Train mode: {config.train_mode}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Voxel size: {config.voxel_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Coarse-to-fine: {config.use_coarse_to_fine}")

    # Create datasets
    train_dataset = Articulate3DDataset(
        data_dir=config.data_dir,
        json_dir=config.json_dir,
        mode=config.train_mode,
        voxel_size=config.voxel_size,
        max_points=config.max_points_per_sample,
        augment=True,
        use_coarse_to_fine=config.use_coarse_to_fine,
        c2f_alpha=config.c2f_alpha,
        c2f_decay=config.c2f_decay,
    )

    val_dataset = Articulate3DDataset(
        data_dir=config.data_dir,
        json_dir=config.json_dir,
        mode="validation",
        voxel_size=config.voxel_size,
        max_points=config.max_points_per_sample,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_articulate3d, voxel_size=config.voxel_size),
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_articulate3d, voxel_size=config.voxel_size),
        pin_memory=config.pin_memory,
    )

    logger.info(f"\n✓ Train samples: {len(train_dataset)}")
    logger.info(f"✓ Val samples: {len(val_dataset)}")

    # Create model
    logger.info("\n🧰 Initializing ArticulateUSDNet model...")
    model = ArticulateUSDNet(
        num_classes=config.num_classes,
        feature_dim=config.feature_dim_3d,
        bn_momentum=config.bn_momentum,
        dropout=config.dropout,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Parameter count: {total_params:,}")

    # Handle resume/pretrain checkpoint loading
    start_epoch = 0
    resume_path = args.resume
    
    # Auto-find latest checkpoint if resume="auto"
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.save_dir)
        if resume_path:
            logger.info(f"🔍 Auto-resume enabled, found checkpoint")
        else:
            logger.info(f"🔍 Auto-resume enabled, but no checkpoint found. Starting from scratch.")
            resume_path = None
    
    if resume_path and os.path.exists(resume_path):
        logger.info(f"📦 Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f"✓ Resumed from epoch {start_epoch}")
        logger.info(f"✓ Best mIoU so far: {ckpt.get('best_metric', 0.0)*100:.2f}%")
    elif args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        logger.info(f"📦 Loading pretrained weights: {args.pretrain_checkpoint}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=config.device)
        
        # Load backbone weights
        pretrained_dict = ckpt.get('model', ckpt)
        model_dict = model.state_dict()
        
        # Filter out head layers, only load backbone
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and 'backbone' in k:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info(f"✓ Loaded {len(filtered_dict)}/{len(model_dict)} parameters from pretrained model")

    # Compute class weights (inverse frequency)
    logger.info("\nComputing class weights...")
    class_counts = np.zeros(config.num_classes)
    for sample in tqdm(train_dataset.database[:min(50, len(train_dataset.database))], desc="Counting classes"):
        data = np.load(sample['filepath']).astype(np.float32)
        sem_gt = data[:, 9].astype(np.int32)
        for c in range(config.num_classes):
            class_counts[c] += (sem_gt == c).sum()
    
    if class_counts.sum() > 0:
        class_weights = class_counts.sum() / (config.num_classes * class_counts + 1)
        class_weights = torch.from_numpy(class_weights).float().to(config.device)
        logger.info(f"✓ Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    # Loss function
    loss_fn = ArticulateLoss(
        num_classes=config.num_classes,
        weight_seg=config.weight_seg,
        weight_origin=config.weight_origin,
        weight_axis=config.weight_axis,
        weight_range=config.weight_range,
        class_weights=class_weights,
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.decay_step, 
        gamma=config.decay_rate
    )

    # Restore optimizer state if resuming
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=config.device)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info("✓ Restored optimizer state")
        if 'scheduler' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info("✓ Restored scheduler state")

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=config.device,
    )

    # Restore trainer state if resuming
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=config.device)
        trainer.global_step = ckpt.get('global_step', 0)
        trainer.best_metric = ckpt.get('best_metric', 0.0)
        logger.info(f"✓ Restored trainer state (global_step={trainer.global_step}, best_metric={trainer.best_metric*100:.2f}%)")

    # Start training
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    logger.info("\n✨ Done!")


if __name__ == "__main__":
    main()
