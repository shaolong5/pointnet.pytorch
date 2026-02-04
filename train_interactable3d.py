"""
================================================================================
Articulate3D Interactable Part Segmentation Training Script
================================================================================

Train USDNet for Interactable Part Segmentation
Based on movable part trained model as initialization.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import yaml
import h5py
import numpy as np
from functools import partial

import torch
import torch.nn as nn
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
    """Training configuration for Interactable Part Segmentation"""

    # data
    data_dir: str = "./data/processed/articulate3d_challenge_inter"
    train_mode: str = "train"  # "train" or "train_validation"
    voxel_size: float = 0.02
    num_workers: int = 4
    batch_size: int = 1
    pin_memory: bool = True
    max_points_per_sample: int = 100000

    # model
    num_classes: Optional[int] = None  # set dynamically from label mapping
    feature_dim_3d: int = 256
    dropout: float = 0.1
    bn_momentum: float = 0.1

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
    save_dir: str = "./checkpoints/articulate3d_inter"
    seed: int = 42
    
    # coarse-to-fine training
    use_coarse_to_fine: bool = True
    c2f_alpha: float = 100.0
    c2f_decay: float = 0.4
    c2f_rad: float = 0.1


# ============================================================================
# Data loading
# ============================================================================
def load_yaml(filepath):
    """Load yaml file"""
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)


class InteractableDataset(Dataset):
    """Dataset for Articulate3D interactable part segmentation"""

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        voxel_size: float = 0.02,
        max_points: int = 100000,
        augment: bool = True,
        use_coarse_to_fine: bool = False,
        c2f_alpha: float = 100.0,
        c2f_decay: float = 0.4,
        epoch: int = 0,
        label_map: Optional[Dict[int, int]] = None,
        binary: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.augment = augment and mode == "train"
        self.use_coarse_to_fine = use_coarse_to_fine
        self.c2f_alpha = c2f_alpha
        self.c2f_decay = c2f_decay
        self.epoch = epoch

        # Load database
        if mode == "train_validation":
            db_file = self.data_dir / "train_validation_database.yaml"
        else:
            db_file = self.data_dir / f"{mode}_database.yaml"
        
        self.database = load_yaml(str(db_file))
        
        # Fix relative paths in database to absolute paths
        base_dir = self.data_dir.parent.parent.parent  # Go up to USDNet root
        for sample in self.database:
            if 'filepath' in sample and not os.path.isabs(sample['filepath']):
                sample['filepath'] = str(base_dir / sample['filepath'])
            if 'expand_dict_file' in sample and not os.path.isabs(sample['expand_dict_file']):
                sample['expand_dict_file'] = str(base_dir / sample['expand_dict_file'])
        
        logger.info(f"✓ Loaded {len(self.database)} scenes from {mode} split")

        # Load color mean/std
        color_stats = load_yaml(str(self.data_dir / "color_mean_std.yaml"))
        self.color_mean = np.array(color_stats['mean'], dtype=np.float32)
        self.color_std = np.array(color_stats['std'], dtype=np.float32)

        # Label mapping
        self.label_map = label_map
        self.binary = binary

    def __len__(self):
        return len(self.database)
    
    def set_epoch(self, epoch: int):
        """Update epoch for coarse-to-fine training"""
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.database[idx]
        scene_id = sample['scene']
        
        # Load point cloud data
        # Format: [x, y, z, r, g, b, nx, ny, nz, sem_gt, inst_gt, segments, inter_gt]
        # Column 12 (inter_gt) contains interactable part labels
        data = np.load(sample['filepath']).astype(np.float32)
        
        coords = data[:, :3]
        colors = data[:, 3:6] / 255.0  # Normalize to [0, 1]
        normals = data[:, 6:9]
        inter_gt = data[:, 12].astype(np.int32)  # Interactable part labels (raw)
        inst_gt = data[:, 10].astype(np.int32)  # Instance labels for coarse-to-fine
        
        # Coarse-to-fine: expand instance labels over epochs
        if self.use_coarse_to_fine and self.mode == "train" and 'expand_dict_file' in sample:
            expand_dict_file = sample['expand_dict_file']
            if os.path.exists(expand_dict_file):
                try:
                    with open(expand_dict_file, 'rb') as f:
                        import pickle
                        expand_dict = pickle.load(f)
                    
                    # Compute expansion radius based on epoch
                    rad = self.c2f_alpha * (self.c2f_decay ** self.epoch)
                    
                    # Find instances that need expansion
                    unique_instances = np.unique(inst_gt)
                    for inst_id in unique_instances:
                        if inst_id <= 0:
                            continue
                        if str(int(inst_id)) in expand_dict:
                            neighbor_indices = expand_dict[str(int(inst_id))]
                            neighbor_indices = np.array(neighbor_indices, dtype=np.int32)
                            neighbor_indices = neighbor_indices[neighbor_indices < len(coords)]
                            
                            # Expand based on distance
                            inst_mask = (inst_gt == inst_id)
                            inst_coords = coords[inst_mask]
                            if len(inst_coords) > 0:
                                neighbor_coords = coords[neighbor_indices]
                                # Compute distance from neighbors to nearest instance point
                                from scipy.spatial import cKDTree
                                tree = cKDTree(inst_coords)
                                dists, _ = tree.query(neighbor_coords, k=1)
                                expand_mask = dists < rad
                                expand_indices = neighbor_indices[expand_mask]
                                inst_gt[expand_indices] = inst_id
                                inter_gt[expand_indices] = inter_gt[inst_mask][0]
                except Exception as e:
                    logger.warning(f"Failed to load expand_dict for {scene_id}: {e}")
        
        # Normalize colors
        colors = (colors - self.color_mean) / (self.color_std + 1e-6)
        
        # Data augmentation
        if self.augment:
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            coords = coords @ rot_matrix.T
            normals = normals @ rot_matrix.T
            
            # Random scale
            scale = np.random.uniform(0.95, 1.05)
            coords = coords * scale
            
            # Random flip
            if np.random.rand() > 0.5:
                coords[:, 0] = -coords[:, 0]
                normals[:, 0] = -normals[:, 0]
            
            # Random color jitter
            colors = colors + np.random.randn(3) * 0.1
            colors = np.clip(colors, -1, 1)
        
        # Subsample if too many points
        if len(coords) > self.max_points:
            choice = np.random.choice(len(coords), self.max_points, replace=False)
            coords = coords[choice]
            colors = colors[choice]
            normals = normals[choice]
            inter_gt = inter_gt[choice]
            inst_gt = inst_gt[choice]
        
        # Map labels
        if self.binary:
            # Binary mapping: background=0, interactable(any non-zero)=1
            inter_mapped = (inter_gt > 0).astype(np.int32)
        else:
            # Multiclass mapping: raw -> contiguous; unknown -> -1
            inter_mapped = np.full_like(inter_gt, -1, dtype=np.int32)
            if self.label_map is not None:
                unique_raw = np.unique(inter_gt)
                for raw in unique_raw:
                    if int(raw) in self.label_map:
                        inter_mapped[inter_gt == raw] = self.label_map[int(raw)]
        
        return {
            'scene_id': scene_id,
            'coords': torch.from_numpy(coords).float(),
            'colors': torch.from_numpy(colors).float(),
            'normals': torch.from_numpy(normals).float(),
            'inter_labels': torch.from_numpy(inter_mapped).long(),
            'inst_labels': torch.from_numpy(inst_gt).long(),
        }


def collate_fn_interactable(batch: List[Dict], voxel_size: float = 0.02) -> Dict[str, Any]:
    """Collate function for sparse convolution"""
    all_coords = []
    all_features = []
    all_inter_labels = []
    all_inst_labels = []

    for batch_idx, data in enumerate(batch):
        coords = data['coords'].numpy()
        colors = data['colors'].numpy()
        normals = data['normals'].numpy()
        inter_labels = data['inter_labels'].numpy()
        inst_labels = data['inst_labels'].numpy()

        # Voxelize coordinates
        voxel_coords = np.floor(coords / voxel_size).astype(np.int32)
        
        # Deduplicate voxel coordinates
        unique_coords, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        
        coords_unique = coords[unique_indices]
        colors_unique = colors[unique_indices]
        normals_unique = normals[unique_indices]
        inter_labels_unique = inter_labels[unique_indices]
        inst_labels_unique = inst_labels[unique_indices]

        # Add batch index
        batch_indices = np.full((len(unique_indices), 1), batch_idx, dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])

        # Features: RGB + normals + coordinates (9 dims)
        combined_features = np.hstack([colors_unique, normals_unique, coords_unique])

        all_coords.append(coords_with_batch)
        all_features.append(combined_features)
        all_inter_labels.append(inter_labels_unique)
        all_inst_labels.append(inst_labels_unique)

    return {
        'coords': torch.from_numpy(np.vstack(all_coords)).int(),
        'features': torch.from_numpy(np.vstack(all_features)).float(),
        'inter_labels': torch.from_numpy(np.concatenate(all_inter_labels)).long(),
        'inst_labels': torch.from_numpy(np.concatenate(all_inst_labels)).long(),
    }


# ============================================================================
# Model - Simplified USDNet for Interactable Part Segmentation
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
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=stride, dimension=3),
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
        
        out = ME.MinkowskiReLU(inplace=True)(out + identity)
        return out


class Res16UNetBackbone(nn.Module):
    """Res16UNet backbone"""
    BLOCK = MinkowskiBasicBlock
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels: int = 9, out_channels: int = 256, bn_momentum: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.INIT_DIM, kernel_size=5, dimension=3
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)
        
        # Encoder
        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.INIT_DIM, self.PLANES[0], kernel_size=2, stride=2, dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.PLANES[0], momentum=bn_momentum)
        self.block1 = self._make_layer(self.PLANES[0], self.PLANES[0], self.LAYERS[0], bn_momentum)
        
        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.PLANES[0], self.PLANES[1], kernel_size=2, stride=2, dimension=3
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.PLANES[1], momentum=bn_momentum)
        self.block2 = self._make_layer(self.PLANES[1], self.PLANES[1], self.LAYERS[1], bn_momentum)
        
        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.PLANES[1], self.PLANES[2], kernel_size=2, stride=2, dimension=3
        )
        self.bn3 = ME.MinkowskiBatchNorm(self.PLANES[2], momentum=bn_momentum)
        self.block3 = self._make_layer(self.PLANES[2], self.PLANES[2], self.LAYERS[2], bn_momentum)
        
        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.PLANES[2], self.PLANES[3], kernel_size=2, stride=2, dimension=3
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.PLANES[3], momentum=bn_momentum)
        self.block4 = self._make_layer(self.PLANES[3], self.PLANES[3], self.LAYERS[3], bn_momentum)
        
        # Decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.PLANES[3], self.PLANES[4], kernel_size=2, stride=2, dimension=3
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)
        self.block5 = self._make_layer(self.PLANES[4] + self.PLANES[2], self.PLANES[4], self.LAYERS[4], bn_momentum)
        
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.PLANES[4], self.PLANES[5], kernel_size=2, stride=2, dimension=3
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)
        self.block6 = self._make_layer(self.PLANES[5] + self.PLANES[1], self.PLANES[5], self.LAYERS[5], bn_momentum)
        
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.PLANES[5], self.PLANES[6], kernel_size=2, stride=2, dimension=3
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)
        self.block7 = self._make_layer(self.PLANES[6] + self.PLANES[0], self.PLANES[6], self.LAYERS[6], bn_momentum)
        
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.PLANES[6], self.PLANES[7], kernel_size=2, stride=2, dimension=3
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)
        self.block8 = self._make_layer(self.PLANES[7] + self.INIT_DIM, self.PLANES[7], self.LAYERS[7], bn_momentum)
        
        # Final layer
        self.final = ME.MinkowskiConvolution(
            self.PLANES[7], out_channels, kernel_size=1, dimension=3
        )
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def _make_layer(self, in_channels, out_channels, blocks, bn_momentum):
        layers = []
        for _ in range(blocks):
            layers.append(self.BLOCK(in_channels, out_channels, bn_momentum=bn_momentum))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # Encoder
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
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
        
        # Decoder
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
        
        out = self.final(out)
        return out


class InteractableUSDNet(nn.Module):
    """USDNet for Interactable Part Segmentation"""

    def __init__(self, num_classes: int = 3, feature_dim: int = 256,
                 bn_momentum: float = 0.1, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = Res16UNetBackbone(
            in_channels=9,  # RGB + normals + coords
            out_channels=feature_dim,
            bn_momentum=bn_momentum,
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            ME.MinkowskiConvolution(feature_dim, feature_dim // 2, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(feature_dim // 2, momentum=bn_momentum),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiConvolution(feature_dim // 2, num_classes, kernel_size=1, dimension=3),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        features = self.backbone(x)
        seg_logits = self.seg_head(features)
        return seg_logits


# ============================================================================
# Loss and Metrics
# ============================================================================
class InteractableLoss(nn.Module):
    """Loss for interactable part segmentation"""

    def __init__(self, num_classes: int = 3, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, seg_logits: ME.SparseTensor, seg_labels: torch.Tensor) -> torch.Tensor:
        logits = seg_logits.F
        loss = self.ce_loss(logits, seg_labels)
        return loss


class InteractableMetrics:
    """Metrics for interactable part segmentation"""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_tp = np.zeros(self.num_classes)
        self.total_fp = np.zeros(self.num_classes)
        self.total_fn = np.zeros(self.num_classes)

    def update(self, pred: np.ndarray, target: np.ndarray):
        # Ignore invalid labels (-1)
        valid = target >= 0
        pred = pred[valid]
        target = target[valid]
        for c in range(self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            self.total_tp[c] += np.sum(pred_c & target_c)
            self.total_fp[c] += np.sum(pred_c & ~target_c)
            self.total_fn[c] += np.sum(~pred_c & target_c)

    def get_metrics(self) -> Dict[str, float]:
        # Per-class IoU
        iou = self.total_tp / (self.total_tp + self.total_fp + self.total_fn + 1e-8)
        
        # Mean IoU
        miou = np.mean(iou)
        
        # Accuracy
        acc = np.sum(self.total_tp) / (np.sum(self.total_tp + self.total_fp) + 1e-8)
        
        return {
            'mIoU': miou,
            'Accuracy': acc,
            'IoU_per_class': iou,
        }


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
        
        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        self.best_metric = 0.0
        self.global_step = 0

    def get_lr(self, epoch):
        """Get learning rate with warmup"""
        if epoch < self.config.warmup_epochs:
            return self.config.warmup_lr + (self.config.learning_rate - self.config.warmup_lr) * epoch / self.config.warmup_epochs
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        
        # Warmup learning rate
        if epoch < self.config.warmup_epochs:
            lr = self.get_lr(epoch)
            self.set_lr(lr)
        
        metrics = InteractableMetrics(num_classes=self.config.num_classes)
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        for batch_idx, batch in enumerate(pbar):
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            inter_labels = batch['inter_labels'].to(self.device)
            
            # Create sparse tensor
            sparse_input = ME.SparseTensor(features=features, coordinates=coords)
            
            # Forward
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.use_mixed_precision):
                seg_logits = self.model(sparse_input)
                loss = self.loss_fn(seg_logits, inter_labels)
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            with torch.no_grad():
                seg_pred = seg_logits.F.argmax(dim=1).cpu().numpy()
                seg_target = inter_labels.cpu().numpy()
                metrics.update(seg_pred, seg_target)
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mIoU': f"{current_metrics['mIoU']*100:.2f}%",
                    'Acc': f"{current_metrics['Accuracy']*100:.2f}%",
                })
        
        avg_loss = total_loss / len(train_loader)
        final_metrics = metrics.get_metrics()
        
        return avg_loss, final_metrics

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        
        metrics = InteractableMetrics(num_classes=self.config.num_classes)
        total_loss = 0.0
        
        for batch in tqdm(val_loader, desc="Validation"):
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            inter_labels = batch['inter_labels'].to(self.device)
            
            sparse_input = ME.SparseTensor(features=features, coordinates=coords)
            
            with autocast(enabled=self.config.use_mixed_precision):
                seg_logits = self.model(sparse_input)
                loss = self.loss_fn(seg_logits, inter_labels)
            
            seg_pred = seg_logits.F.argmax(dim=1).cpu().numpy()
            seg_target = inter_labels.cpu().numpy()
            metrics.update(seg_pred, seg_target)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        final_metrics = metrics.get_metrics()
        
        return avg_loss, final_metrics

    def save_checkpoint(self, epoch, is_best=False, save_latest=True):
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': asdict(self.config),
        }
        
        # Save epoch checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(self.config.save_dir, f'ckpt_epoch_{epoch+1}.pt')
            torch.save(checkpoint, save_path)
            logger.info(f"💾 Saved checkpoint: {save_path}")
        
        # Save latest checkpoint
        if save_latest:
            latest_path = os.path.join(self.config.save_dir, 'latest_checkpoint.pt')
            torch.save(checkpoint, latest_path)
            
            # Save resume info
            resume_info = {
                'epoch': epoch,
                'best_metric': self.best_metric,
                'global_step': self.global_step,
            }
            import json
            with open(os.path.join(self.config.save_dir, 'resume_info.json'), 'w') as f:
                json.dump(resume_info, f, indent=2)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 New best model! mIoU: {self.best_metric*100:.2f}%")

    def train(self, train_loader, val_loader, start_epoch=0):
        logger.info("\n" + "=" * 80)
        logger.info("🚀 Starting training...")
        logger.info("=" * 80)
        
        for epoch in range(start_epoch, self.config.max_epochs):
            # Update dataset epoch for coarse-to-fine
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            logger.info(f"\n📊 Epoch {epoch+1}/{self.config.max_epochs} - Train:")
            logger.info(f"  Loss: {train_loss:.4f}")
            logger.info(f"  mIoU: {train_metrics['mIoU']*100:.2f}%")
            logger.info(f"  Accuracy: {train_metrics['Accuracy']*100:.2f}%")
            
            # Validate
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.max_epochs:
                val_loss, val_metrics = self.validate(val_loader)
                
                logger.info(f"\n📊 Epoch {epoch+1}/{self.config.max_epochs} - Val:")
                logger.info(f"  Loss: {val_loss:.4f}")
                logger.info(f"  mIoU: {val_metrics['mIoU']*100:.2f}%")
                logger.info(f"  Accuracy: {val_metrics['Accuracy']*100:.2f}%")
                
                # Save checkpoint
                is_best = val_metrics['mIoU'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['mIoU']
                
                self.save_checkpoint(epoch, is_best=is_best)
                
                logger.info(f"\n🎯 Best mIoU: {self.best_metric*100:.2f}%")
            else:
                # Save latest checkpoint even without validation
                self.save_checkpoint(epoch, is_best=False)
            
            # Step scheduler
            if self.scheduler and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 Training complete! Best mIoU: {self.best_metric*100:.2f}%")
        logger.info("=" * 80)


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
        logger.info(f"Found latest_checkpoint.pt")
        return str(latest_ckpt)
    
    # Otherwise find the highest epoch checkpoint
    ckpt_files = list(save_path.glob("ckpt_epoch_*.pt"))
    if ckpt_files:
        # Extract epoch numbers
        epochs = []
        for f in ckpt_files:
            try:
                epoch = int(f.stem.split('_')[-1])
                epochs.append((epoch, f))
            except:
                continue
        if epochs:
            latest = max(epochs, key=lambda x: x[0])
            logger.info(f"Found latest epoch checkpoint: {latest[1].name}")
            return str(latest[1])
    
    return None


def build_label_mapping(data_dir: str) -> Dict[int, int]:
    """Build raw->contiguous label mapping for interactable labels from train+val splits."""
    base = Path(data_dir)
    raw_labels = set()
    for split in ["train_database.yaml", "validation_database.yaml"]:
        dbf = base / split
        if not dbf.exists():
            continue
        database = load_yaml(str(dbf))
        # Fix relative paths to absolute
        repo_root = base.parent.parent.parent
        for sample in database:
            fp = sample.get('filepath')
            if fp and not os.path.isabs(fp):
                fp = str(repo_root / fp)
            if fp and os.path.exists(fp):
                try:
                    data = np.load(fp)
                    if data.shape[1] >= 13:
                        inter_gt = data[:, 12].astype(np.int32)
                        raw_labels.update(map(int, np.unique(inter_gt)))
                except Exception:
                    continue
    labels_sorted = sorted(list(raw_labels))
    label_map = {raw: idx for idx, raw in enumerate(labels_sorted)}
    logger.info(f"🔎 Built label mapping for inter: {label_map}")
    return label_map


def main():
    parser = argparse.ArgumentParser(description="Articulate3D Interactable Part Segmentation Training")
    
    parser.add_argument('--data_dir', type=str, 
                        default="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_inter",
                        help='Path to processed interactable part data')
    parser.add_argument('--train_mode', type=str, default='train', choices=['train', 'train_validation'],
                        help='train: use train set only, train_validation: use train+val for final submission')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_dir', type=str, default="./checkpoints/articulate3d_inter")
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path or "auto" to find latest)')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, 
                        help='Pretrained movable part checkpoint to load (recommended)')
    parser.add_argument('--use_coarse_to_fine', action='store_true', help='Enable coarse-to-fine training')
    parser.add_argument('--binary', action='store_true', help='Use binary labels for interactable (default: true)')
    parser.add_argument('--multiclass', action='store_true', help='Use multiclass labels built from dataset (overrides --binary)')

    args = parser.parse_args()

    setup_seed(42)

    config = TrainingConfig()
    config.data_dir = args.data_dir
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
    logger.info("🎓 Articulate3D Training - Interactable Part Segmentation")
    logger.info("=" * 80)
    logger.info(f"  Data dir: {config.data_dir}")
    logger.info(f"  Train mode: {config.train_mode}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Voxel size: {config.voxel_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Coarse-to-fine: {config.use_coarse_to_fine}")

    # Decide label mode
    use_multiclass = bool(args.multiclass)
    use_binary = not use_multiclass  # default to binary
    if use_binary:
        label_map = None
        config.num_classes = 2
        logger.info("Using binary interactable labels: background=0, interactable=1")
    else:
        label_map = build_label_mapping(config.data_dir)
        config.num_classes = len(label_map)

    # Create datasets
    train_dataset = InteractableDataset(
        data_dir=config.data_dir,
        mode=config.train_mode,
        voxel_size=config.voxel_size,
        max_points=config.max_points_per_sample,
        augment=True,
        use_coarse_to_fine=config.use_coarse_to_fine,
        c2f_alpha=config.c2f_alpha,
        c2f_decay=config.c2f_decay,
        label_map=label_map,
        binary=use_binary,
    )

    val_dataset = InteractableDataset(
        data_dir=config.data_dir,
        mode="validation",
        voxel_size=config.voxel_size,
        max_points=config.max_points_per_sample,
        augment=False,
        label_map=label_map,
        binary=use_binary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_interactable, voxel_size=config.voxel_size),
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_interactable, voxel_size=config.voxel_size),
        pin_memory=config.pin_memory,
    )

    logger.info(f"\n✓ Train samples: {len(train_dataset)}")
    logger.info(f"✓ Val samples: {len(val_dataset)}")

    # Create model
    logger.info("\n🧰 Initializing InteractableUSDNet model...")
    model = InteractableUSDNet(
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
        logger.info(f"📦 Loading pretrained movable part weights: {args.pretrain_checkpoint}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=config.device)
        
        # Load backbone weights from movable part model
        pretrained_dict = ckpt.get('model', ckpt)
        model_dict = model.state_dict()
        
        # Filter: only load backbone and seg_head (compatible parts)
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            # Load backbone completely
            if 'backbone' in k and k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
            # Try to load seg_head first layers (feature extraction part)
            elif 'seg_head.0' in k or 'seg_head.1' in k or 'seg_head.2' in k:
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info(f"✓ Loaded {len(filtered_dict)}/{len(model_dict)} parameters from movable part model")
        logger.info(f"  (Backbone + partial seg_head initialized from movable part training)")

    # Compute class weights on mapped labels
    logger.info("\nComputing class weights...")
    class_counts = np.zeros(config.num_classes)
    for sample in tqdm(train_dataset.database[:min(50, len(train_dataset.database))], desc="Counting classes"):
        data = np.load(sample['filepath']).astype(np.float32)
        inter_gt = data[:, 12].astype(np.int32)
        if use_binary:
            mapped = (inter_gt > 0).astype(np.int32)
        else:
            mapped = np.full_like(inter_gt, -1, dtype=np.int32)
            for raw, idx in label_map.items():
                mapped[inter_gt == raw] = idx
        for c in range(config.num_classes):
            class_counts[c] += (mapped == c).sum()
    
    if class_counts.sum() > 0:
        class_weights = class_counts.sum() / (config.num_classes * class_counts + 1)
        class_weights = torch.from_numpy(class_weights).float().to(config.device)
        logger.info(f"✓ Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    # Loss function
    loss_fn = InteractableLoss(
        num_classes=config.num_classes,
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
