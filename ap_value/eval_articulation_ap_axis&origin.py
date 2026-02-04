#!/usr/bin/env python3
"""
================================================================================
Articulation AP50 Evaluation Script
================================================================================

Evaluate trained ArticulateUSDNet model for:
- Origin AP50 (threshold: distance <= 0.1m)
- Axis AP50 (threshold: angle <= 30 degrees)

Usage:
    python eval_articulation_ap.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /path/to/data \
        --split validation \
        --origin_threshold 0.1 \
        --axis_threshold 30.0 \
        --output submission.json
"""

import os
import sys
import json
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_articulate3d import (
    Articulate3DDataset,
    ArticulateUSDNet,
    collate_fn_articulate3d,
    load_h5_articulation,
    load_yaml,
)


# ============================================================================
# Articulation AP Evaluation Functions
# ============================================================================

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall."""
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # Correct AP calculation
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculate area under PR curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def aggregate_point_predictions_to_instance(
    coords: np.ndarray,
    inst_labels: np.ndarray,
    origin_pred: np.ndarray,
    axis_pred: np.ndarray,
    seg_pred: Optional[np.ndarray] = None,
    min_points: int = 10,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Aggregate point-level predictions to instance-level.

    Args:
        coords: (N, 3) point coordinates
        inst_labels: (N,) instance labels (0 = background)
        origin_pred: (N, 3) origin predictions
        axis_pred: (N, 3) axis predictions
        seg_pred: (N,) semantic predictions (optional, for filtering)
        min_points: minimum points for valid instance

    Returns:
        {inst_id: {'origin': [x,y,z], 'axis': [x,y,z], 'center': [x,y,z]}}
    """
    predictions = {}

    # Get unique instance IDs (ignore background)
    unique_insts = np.unique(inst_labels[inst_labels > 0])

    for inst_id in unique_insts:
        mask = inst_labels == inst_id

        # Filter by semantic class if provided (only movable parts)
        if seg_pred is not None:
            mask = mask & (seg_pred > 0)

        if mask.sum() < min_points:
            continue

        # Aggregate origin: average of all point predictions
        origin = origin_pred[mask].mean(axis=0)

        # Aggregate axis: average then normalize
        axis = axis_pred[mask].mean(axis=0)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm

        # Compute instance center (for reference)
        center = coords[mask].mean(axis=0)

        predictions[inst_id] = {
            'origin': origin,
            'axis': axis,
            'center': center,
            'num_points': mask.sum(),
        }

    return predictions


def compute_origin_distance(pred_origin: np.ndarray, gt_origin: np.ndarray) -> float:
    """Compute L2 distance between predicted and ground truth origin."""
    return np.linalg.norm(pred_origin - gt_origin)


def compute_axis_angle(pred_axis: np.ndarray, gt_axis: np.ndarray) -> float:
    """
    Compute angle (in degrees) between predicted and ground truth axis.
    Handles axis symmetry (direction can be flipped).
    """
    # Normalize
    pred_norm = pred_axis / (np.linalg.norm(pred_axis) + 1e-6)
    gt_norm = gt_axis / (np.linalg.norm(gt_axis) + 1e-6)

    # Handle symmetry: max of dot product and negative dot product
    cos_sim = max(np.dot(pred_norm, gt_norm), np.dot(pred_norm, -gt_norm))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Convert to degrees
    angle = np.arccos(cos_sim) * 180.0 / np.pi
    return angle


def eval_ap_origin(
    pred_all: Dict[str, List[Tuple[int, np.ndarray, float]]],
    gt_all: Dict[str, Dict[int, np.ndarray]],
    threshold: float = 1.5,
    use_07_metric: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for origin prediction.

    Args:
        pred_all: {scene_id: [(inst_id, origin, score), ...]}
        gt_all: {scene_id: {inst_id: origin}}
        threshold: distance threshold for correct prediction
        use_07_metric: whether to use VOC07 11-point metric

    Returns:
        (ap, recall, precision)
    """
    # Construct GT objects
    class_recs = {}
    npos = 0

    for scene_id in gt_all.keys():
        gt_instances = gt_all[scene_id]
        det = [False] * len(gt_instances)
        npos += len(gt_instances)

        # Convert to array format
        inst_ids = list(gt_instances.keys())
        origins = np.array([gt_instances[i] for i in inst_ids])

        class_recs[scene_id] = {
            'inst_ids': inst_ids,
            'origins': origins,
            'det': det,
        }

    # Pad empty scenes
    for scene_id in pred_all.keys():
        if scene_id not in class_recs:
            class_recs[scene_id] = {
                'inst_ids': [],
                'origins': np.array([]).reshape(0, 3),
                'det': [],
            }

    if npos == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    # Construct predictions
    scene_ids = []
    confidence = []
    origins_pred = []
    inst_ids_pred = []

    for scene_id in pred_all.keys():
        for inst_id, origin, score in pred_all[scene_id]:
            scene_ids.append(scene_id)
            confidence.append(score)
            origins_pred.append(origin)
            inst_ids_pred.append(inst_id)

    confidence = np.array(confidence)
    origins_pred = np.array(origins_pred)  # (N, 3)

    # Sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidence = confidence[sorted_ind]
    origins_pred = origins_pred[sorted_ind]
    scene_ids = [scene_ids[i] for i in sorted_ind]
    inst_ids_pred = [inst_ids_pred[i] for i in sorted_ind]

    # Go through detections and mark TPs and FPs
    nd = len(scene_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        scene_id = scene_ids[d]
        pred_origin = origins_pred[d]
        pred_inst_id = inst_ids_pred[d]

        R = class_recs[scene_id]
        gt_origins = R['origins']
        gt_inst_ids = R['inst_ids']

        if len(gt_origins) > 0:
            # Find best matching GT instance
            distances = np.linalg.norm(gt_origins - pred_origin, axis=1)
            jmax = np.argmin(distances)
            ovmax = distances[jmax]

            # Check if correct
            if ovmax <= threshold:
                # Check if already detected
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # Compute precision-recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return ap, rec, prec


def eval_ap_axis(
    pred_all: Dict[str, List[Tuple[int, np.ndarray, float]]],
    gt_all: Dict[str, Dict[int, np.ndarray]],
    threshold: float = 120.0,
    use_07_metric: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for axis prediction.

    Args:
        pred_all: {scene_id: [(inst_id, axis, score), ...]}
        gt_all: {scene_id: {inst_id: axis}}
        threshold: angle threshold (in degrees) for correct prediction
        use_07_metric: whether to use VOC07 11-point metric

    Returns:
        (ap, recall, precision)
    """
    # Construct GT objects
    class_recs = {}
    npos = 0

    for scene_id in gt_all.keys():
        gt_instances = gt_all[scene_id]
        det = [False] * len(gt_instances)
        npos += len(gt_instances)

        # Convert to array format
        inst_ids = list(gt_instances.keys())
        axes = np.array([gt_instances[i] for i in inst_ids])

        class_recs[scene_id] = {
            'inst_ids': inst_ids,
            'axes': axes,
            'det': det,
        }

    # Pad empty scenes
    for scene_id in pred_all.keys():
        if scene_id not in class_recs:
            class_recs[scene_id] = {
                'inst_ids': [],
                'axes': np.array([]).reshape(0, 3),
                'det': [],
            }

    if npos == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    # Construct predictions
    scene_ids = []
    confidence = []
    axes_pred = []
    inst_ids_pred = []

    for scene_id in pred_all.keys():
        for inst_id, axis, score in pred_all[scene_id]:
            scene_ids.append(scene_id)
            confidence.append(score)
            axes_pred.append(axis)
            inst_ids_pred.append(inst_id)

    confidence = np.array(confidence)
    axes_pred = np.array(axes_pred)  # (N, 3)

    # Sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidence = confidence[sorted_ind]
    axes_pred = axes_pred[sorted_ind]
    scene_ids = [scene_ids[i] for i in sorted_ind]
    inst_ids_pred = [inst_ids_pred[i] for i in sorted_ind]

    # Go through detections and mark TPs and FPs
    nd = len(scene_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        scene_id = scene_ids[d]
        pred_axis = axes_pred[d]

        R = class_recs[scene_id]
        gt_axes = R['axes']
        gt_inst_ids = R['inst_ids']

        if len(gt_axes) > 0:
            # Find best matching GT instance (by angle)
            angles = []
            for gt_axis in gt_axes:
                angle = compute_axis_angle(pred_axis, gt_axis)
                angles.append(angle)
            angles = np.array(angles)

            jmax = np.argmin(angles)
            ovmax = angles[jmax]

            # Check if correct
            if ovmax <= threshold:
                # Check if already detected
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # Compute precision-recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return ap, rec, prec


def eval_ap_combined(
    pred_all: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, float]]],
    gt_all: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    origin_threshold: float = 1.5,
    axis_threshold: float = 120.0,
    use_07_metric: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for combined origin+axis prediction.
    Both origin AND axis must be correct for a TP.

    Args:
        pred_all: {scene_id: [(inst_id, origin, axis, score), ...]}
        gt_all: {scene_id: {inst_id: (origin, axis)}}
        origin_threshold: distance threshold (meters)
        axis_threshold: angle threshold (degrees)
        use_07_metric: whether to use VOC07 11-point metric

    Returns:
        (ap, recall, precision)
    """
    # Construct GT objects
    class_recs = {}
    npos = 0

    for scene_id in gt_all.keys():
        gt_instances = gt_all[scene_id]
        det = [False] * len(gt_instances)
        npos += len(gt_instances)

        # Convert to array format
        inst_ids = list(gt_instances.keys())
        origins = np.array([gt_instances[i][0] for i in inst_ids])
        axes = np.array([gt_instances[i][1] for i in inst_ids])

        class_recs[scene_id] = {
            'inst_ids': inst_ids,
            'origins': origins,
            'axes': axes,
            'det': det,
        }

    # Pad empty scenes
    for scene_id in pred_all.keys():
        if scene_id not in class_recs:
            class_recs[scene_id] = {
                'inst_ids': [],
                'origins': np.array([]).reshape(0, 3),
                'axes': np.array([]).reshape(0, 3),
                'det': [],
            }

    if npos == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    # Construct predictions
    scene_ids = []
    confidence = []
    origins_pred = []
    axes_pred = []
    inst_ids_pred = []

    for scene_id in pred_all.keys():
        for inst_id, origin, axis, score in pred_all[scene_id]:
            scene_ids.append(scene_id)
            confidence.append(score)
            origins_pred.append(origin)
            axes_pred.append(axis)
            inst_ids_pred.append(inst_id)

    confidence = np.array(confidence)
    origins_pred = np.array(origins_pred)  # (N, 3)
    axes_pred = np.array(axes_pred)  # (N, 3)

    # Sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidence = confidence[sorted_ind]
    origins_pred = origins_pred[sorted_ind]
    axes_pred = axes_pred[sorted_ind]
    scene_ids = [scene_ids[i] for i in sorted_ind]
    inst_ids_pred = [inst_ids_pred[i] for i in sorted_ind]

    # Go through detections and mark TPs and FPs
    nd = len(scene_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        scene_id = scene_ids[d]
        pred_origin = origins_pred[d]
        pred_axis = axes_pred[d]

        R = class_recs[scene_id]
        gt_origins = R['origins']
        gt_axes = R['axes']
        gt_inst_ids = R['inst_ids']

        if len(gt_origins) > 0:
            # Find best matching GT instance
            # Check both origin and axis
            best_j = -1
            best_score = -1

            for j in range(len(gt_origins)):
                gt_origin = gt_origins[j]
                gt_axis = gt_axes[j]

                # Check origin
                origin_dist = compute_origin_distance(pred_origin, gt_origin)
                origin_ok = origin_dist <= origin_threshold

                # Check axis
                axis_angle = compute_axis_angle(pred_axis, gt_axis)
                axis_ok = axis_angle <= axis_threshold

                # Both must be correct
                if origin_ok and axis_ok:
                    # Use origin distance as primary score (lower is better)
                    # Invert to match the "max" logic
                    score = -origin_dist
                    if score > best_score:
                        best_score = score
                        best_j = j

            if best_j >= 0:
                # Found a matching GT
                if not R['det'][best_j]:
                    tp[d] = 1.0
                    R['det'][best_j] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # Compute precision-recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return ap, rec, prec


# ============================================================================
# Main Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate_articulation_ap(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    origin_threshold: float = 1.5,
    axis_threshold: float = 120.0,
    use_07_metric: bool = False,
    min_points: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate articulation AP50 for origin and axis.

    Returns:
        {
            'ap_origin': float,
            'ap_axis': float,
            'ap_combined': float,
            'predictions': {...},  # For saving to JSON
            'metrics': {...},
        }
    """
    model.eval()

    # Store all predictions and ground truth
    pred_origin_all = {}  # {scene_id: [(inst_id, origin, score), ...]}
    pred_axis_all = {}
    pred_combined_all = {}  # {scene_id: [(inst_id, origin, axis, score), ...]}
    gt_origin_all = {}    # {scene_id: {inst_id: origin}}
    gt_axis_all = {}
    gt_combined_all = {}  # {scene_id: {inst_id: (origin, axis)}}

    predictions_for_json = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        coords = batch['coords'].to(device)
        features = batch['features'].to(device)
        sem_labels = batch['sem_labels']
        inst_labels = batch['inst_labels']
        scene_ids = batch.get('scene_id', [])

        # Get ground truth articulation parameters if available
        origin_targets = batch.get('origin_targets')
        axis_targets = batch.get('axis_targets')

        # Create sparse tensor
        x = ME.SparseTensor(
            features=features,
            coordinates=coords,
            device=device,
        )

        # Forward pass
        outputs = model(x)

        # Get predictions
        seg_pred = outputs['seg_logits'].features.argmax(dim=1).cpu().numpy()
        origin_pred = outputs['origin_pred'].features.cpu().numpy()
        axis_pred = outputs['axis_pred'].features.cpu().numpy()

        # Get coordinates in original scale
        coords_np = coords[:, 1:].cpu().numpy()  # Remove batch index

        # Process each sample in batch
        batch_size = len(np.unique(coords[:, 0].cpu().numpy()))

        for batch_idx in range(batch_size):
            # Get mask for this batch item
            batch_mask = coords[:, 0].cpu().numpy() == batch_idx

            # Extract data for this sample
            sample_coords = coords_np[batch_mask]
            sample_seg_pred = seg_pred[batch_mask]
            sample_inst_pred = inst_labels[batch_mask].numpy() if torch.is_tensor(inst_labels) else inst_labels[batch_mask]
            sample_origin_pred = origin_pred[batch_mask]
            sample_axis_pred = axis_pred[batch_mask]

            # Get scene ID
            if isinstance(scene_ids, list) and len(scene_ids) > batch_idx:
                scene_id = scene_ids[batch_idx]
            else:
                scene_id = f"scene_{batch_idx}"

            # Aggregate to instance level
            inst_predictions = aggregate_point_predictions_to_instance(
                coords=sample_coords,
                inst_labels=sample_inst_pred,
                origin_pred=sample_origin_pred,
                axis_pred=sample_axis_pred,
                seg_pred=sample_seg_pred,
                min_points=min_points,
            )

            # Store predictions with confidence score
            # Use segmentation confidence as score (max softmax prob)
            seg_logits = outputs['seg_logits'].features[batch_mask].cpu()
            seg_probs = torch.softmax(seg_logits, dim=1).numpy()

            for inst_id, pred_data in inst_predictions.items():
                # Get points for this instance
                inst_mask = sample_inst_pred == inst_id

                # Use average semantic confidence as score
                if inst_mask.sum() > 0:
                    inst_probs = seg_probs[inst_mask]
                    # Confidence for predicted class
                    score = float(np.mean(np.max(inst_probs, axis=1)))
                else:
                    score = 0.5

                # Store origin prediction
                if scene_id not in pred_origin_all:
                    pred_origin_all[scene_id] = []
                pred_origin_all[scene_id].append((inst_id, pred_data['origin'], score))

                # Store axis prediction
                if scene_id not in pred_axis_all:
                    pred_axis_all[scene_id] = []
                pred_axis_all[scene_id].append((inst_id, pred_data['axis'], score))

                # Store combined prediction (origin + axis)
                if scene_id not in pred_combined_all:
                    pred_combined_all[scene_id] = []
                pred_combined_all[scene_id].append((inst_id, pred_data['origin'], pred_data['axis'], score))

                # Store for JSON output
                predictions_for_json.append({
                    'scene_id': scene_id,
                    'pid': int(inst_id),
                    'origin': pred_data['origin'].tolist(),
                    'axis': pred_data['axis'].tolist(),
                })

            # Debug: print first scene predictions vs GT
            if debug and len(pred_origin_all) == 1:
                print(f"\n{'='*60}")
                print(f"DEBUG - Scene: {scene_id}")
                print(f"{'='*60}")
                print(f"Predictions ({len(inst_predictions)} instances):")
                for inst_id, pred in inst_predictions.items():
                    print(f"  Inst {inst_id}: origin={pred['origin']}, axis={pred['axis']}")

            # Load GT articulation parameters if available
            if origin_targets is not None and axis_targets is not None:
                # origin_targets and axis_targets are already concatenated by collate_fn
                # We need to extract the data for this specific batch item
                all_origin_gt = origin_targets.numpy() if torch.is_tensor(origin_targets) else origin_targets
                all_axis_gt = axis_targets.numpy() if torch.is_tensor(axis_targets) else axis_targets

                # Extract GT for this batch item using batch_mask
                sample_origin_gt = all_origin_gt[batch_mask]
                sample_axis_gt = all_axis_gt[batch_mask]

                # Get unique instances in GT
                unique_insts = np.unique(sample_inst_pred[sample_inst_pred > 0])

                gt_origin_all[scene_id] = {}
                gt_axis_all[scene_id] = {}
                gt_combined_all[scene_id] = {}

                for inst_id in unique_insts:
                    inst_mask = sample_inst_pred == inst_id
                    if inst_mask.sum() < min_points:
                        continue

                    # Use GT values (constant across instance) - take first point's value
                    gt_origin_all[scene_id][inst_id] = sample_origin_gt[inst_mask][0]
                    gt_axis_all[scene_id][inst_id] = sample_axis_gt[inst_mask][0]
                    gt_combined_all[scene_id][inst_id] = (
                        sample_origin_gt[inst_mask][0],
                        sample_axis_gt[inst_mask][0]
                    )

                # Debug: print GT for first scene
                if debug and len(gt_origin_all) == 1:
                    print(f"\nGround Truth ({len(gt_origin_all[scene_id])} instances):")
                    for inst_id in gt_origin_all[scene_id].keys():
                        origin_gt = gt_origin_all[scene_id][inst_id]
                        axis_gt = gt_axis_all[scene_id][inst_id]
                        print(f"  Inst {inst_id}: origin={origin_gt}, axis={axis_gt}")

                    # Print comparison
                    print(f"\nComparison (distance threshold={origin_threshold}m, angle threshold={axis_threshold}°):")
                    for inst_id in list(set(list(inst_predictions.keys()) + list(gt_origin_all[scene_id].keys()))):
                        if inst_id in inst_predictions and inst_id in gt_origin_all[scene_id]:
                            pred_origin = inst_predictions[inst_id]['origin']
                            pred_axis = inst_predictions[inst_id]['axis']
                            gt_origin = gt_origin_all[scene_id][inst_id]
                            gt_axis = gt_axis_all[scene_id][inst_id]

                            origin_dist = compute_origin_distance(pred_origin, gt_origin)
                            axis_angle = compute_axis_angle(pred_axis, gt_axis)

                            origin_ok = "✓" if origin_dist <= origin_threshold else "✗"
                            axis_ok = "✓" if axis_angle <= axis_threshold else "✗"

                            print(f"  Inst {inst_id}:")
                            print(f"    Origin: pred={pred_origin}, gt={gt_origin}, dist={origin_dist:.4f}m {origin_ok}")
                            print(f"    Axis:   pred={pred_axis}, gt={gt_axis}, angle={axis_angle:.2f}° {axis_ok}")
                        elif inst_id in inst_predictions:
                            print(f"  Inst {inst_id}: Only in prediction (not in GT)")
                        elif inst_id in gt_origin_all[scene_id]:
                            print(f"  Inst {inst_id}: Only in GT (not predicted)")
                    print(f"{'='*60}\n")

    # Compute AP for origin
    ap_origin, rec_origin, prec_origin = eval_ap_origin(
        pred_origin_all, gt_origin_all, origin_threshold, use_07_metric
    )

    # Compute AP for axis
    ap_axis, rec_axis, prec_axis = eval_ap_axis(
        pred_axis_all, gt_axis_all, axis_threshold, use_07_metric
    )

    # Compute AP for combined (origin AND axis must both be correct)
    ap_combined, rec_combined, prec_combined = eval_ap_combined(
        pred_combined_all, gt_combined_all, origin_threshold, axis_threshold, use_07_metric
    )

    # Additional error metrics
    origin_errors = []
    axis_errors = []

    for scene_id in gt_origin_all.keys():
        gt_origins = gt_origin_all[scene_id]
        if scene_id in pred_origin_all:
            for inst_id, origin_pred, _ in pred_origin_all[scene_id]:
                if inst_id in gt_origins:
                    error = compute_origin_distance(origin_pred, gt_origins[inst_id])
                    origin_errors.append(error)

    for scene_id in gt_axis_all.keys():
        gt_axes = gt_axis_all[scene_id]
        if scene_id in pred_axis_all:
            for inst_id, axis_pred, _ in pred_axis_all[scene_id]:
                if inst_id in gt_axes:
                    error = compute_axis_angle(axis_pred, gt_axes[inst_id])
                    axis_errors.append(error)

    return {
        'ap_origin': ap_origin,
        'ap_axis': ap_axis,
        'ap_combined': ap_combined,
        'recall_origin': rec_origin,
        'precision_origin': prec_origin,
        'recall_axis': rec_axis,
        'precision_axis': prec_axis,
        'recall_combined': rec_combined,
        'precision_combined': prec_combined,
        'predictions': predictions_for_json,
        'metrics': {
            'origin_error_mean': np.mean(origin_errors) if origin_errors else 0.0,
            'origin_error_median': np.median(origin_errors) if origin_errors else 0.0,
            'axis_error_mean': np.mean(axis_errors) if axis_errors else 0.0,
            'axis_error_median': np.median(axis_errors) if axis_errors else 0.0,
        },
    }


def save_predictions_to_json(results: Dict[str, Any], output_path: str):
    """Save AP50 results to JSON file."""
    output = {
        "origin AP50": float(f"{results['ap_origin']*100:.2f}"),
        "axis AP50": float(f"{results['ap_axis']*100:.2f}"),
        "origin+axis AP50": float(f"{results['ap_combined']*100:.2f}"),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Articulation AP50 Evaluation")

    parser.add_argument('--model_path', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--json_dir', type=str, default=None,
                        help='Path to JSON annotations for motion range')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'train_validation'],
                        help='Data split to evaluate')
    parser.add_argument('--num_scenes', type=int, default=None,
                        help='Number of scenes to evaluate (default: all scenes)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for sparse convolution')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--origin_threshold', type=float, default=1.5,
                        help='Origin distance threshold for AP (meters)')
    parser.add_argument('--axis_threshold', type=float, default=120.0,
                        help='Axis angle threshold for AP (degrees)')
    parser.add_argument('--min_points', type=int, default=10,
                        help='Minimum points for valid instance')
    parser.add_argument('--use_07_metric', action='store_true',
                        help='Use VOC07 11-point metric')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information (predictions vs GT)')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output JSON file for predictions')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print("Articulation AP50 Evaluation")
    print("=" * 80)
    print(f"  Model: {args.model_path}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Split: {args.split}")
    if args.num_scenes:
        print(f"  Num scenes: {args.num_scenes}")
    print(f"  Output: {args.output}")
    print("=" * 80)
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.model_path, map_location=args.device)

    # Create model
    print("Creating model...")
    model = ArticulateUSDNet(
        num_classes=3,
        feature_dim=256,
        bn_momentum=0.1,
        dropout=0.1,
    ).to(args.device)

    # Load model weights (allow partial loading for checkpoints without range_head)
    if 'model' in ckpt:
        model_dict = model.state_dict()
        pretrained_dict = ckpt['model']

        # Filter out keys that don't match in size
        matched_dict = {}
        missing_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    print(f"  ⚠ Skipping {k}: shape mismatch ({model_dict[k].shape} vs {v.shape})")
            else:
                missing_keys.append(k)

        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        if missing_keys:
            print(f"  ⚠ Missing keys (not loaded): {missing_keys}")

        print(f"✓ Loaded from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        model_dict = model.state_dict()
        pretrained_dict = ckpt

        # Filter out keys that don't match
        matched_dict = {}
        missing_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_dict[k] = v
            else:
                missing_keys.append(k)

        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        if missing_keys:
            print(f"  ⚠ Missing keys (not loaded): {missing_keys}")

        print("✓ Loaded model weights")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    full_dataset = Articulate3DDataset(
        data_dir=args.data_dir,
        json_dir=args.json_dir,
        mode=args.split,
        voxel_size=args.voxel_size,
        max_points=100000,
        augment=False,
    )

    # Limit number of scenes if specified
    if args.num_scenes is not None and args.num_scenes < len(full_dataset):
        # Create a subset by limiting the database
        dataset = full_dataset
        dataset.database = dataset.database[:args.num_scenes]
        print(f"✓ Limited to {len(dataset.database)} scenes (out of {len(full_dataset)} total)")
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate_fn_articulate3d(x, args.voxel_size),
        pin_memory=True,
    )

    print(f"✓ Loaded {len(dataset)} scenes")
    print()

    # Run evaluation
    print("Running evaluation...")
    results = evaluate_articulation_ap(
        model=model,
        dataloader=dataloader,
        device=args.device,
        origin_threshold=args.origin_threshold,
        axis_threshold=args.axis_threshold,
        use_07_metric=args.use_07_metric,
        min_points=args.min_points,
        debug=args.debug,
    )

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"origin AP50:     {results['ap_origin']*100:.2f}%")
    print(f"axis AP50:       {results['ap_axis']*100:.2f}%")
    print(f"origin+axis AP50: {results['ap_combined']*100:.2f}%")
    print("=" * 80)

    # Save predictions
    if args.output:
        save_predictions_to_json(results, args.output)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
