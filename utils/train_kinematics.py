#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Articulate3D kinematics training (robust + USDNet-style) - Enhanced version
- Added data diagnostics to help debug missing positive samples
- Better error reporting and statistics
"""
from __future__ import annotations
import os, sys, math, time, random, glob
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------------------
# 数据集导入
# ----------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
POINTNET_DIR = PROJECT_ROOT / "pointnet"
if POINTNET_DIR.exists():
    sys.path.insert(0, str(POINTNET_DIR))
else:
    alt = Path("/home/shaolongshi/data/pointnet.pytorch/pointnet")
    if alt.exists():
        sys.path.insert(0, str(alt))

try:
    from dataset_articulate3d import Articulate3DDataset as BaseDataset
except Exception as e:
    print(f"[ERROR] import dataset_articulate3d failed: {e}")
    sys.exit(1)

# ----------------------------
# 小工具
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d: return d[k]
    return default

# ----------------------------
# 数据诊断工具
# ----------------------------
def diagnose_dataset(dataset: BaseDataset, num_samples: int = 10):
    """诊断数据集，检查是否有正样本"""
    print("\n" + "="*60)
    print("DATASET DIAGNOSIS")
    print("="*60)
    
    stats = {
        'total_scenes': len(dataset),
        'scenes_with_joints': 0,
        'scenes_with_interactive': 0,
        'scenes_with_big_parts': 0,
        'total_joint_points': 0,
        'total_interactive_points': 0,
        'joint_types': {},
        'interactive_classes': {},
        'big_classes': {}
    }
    
    # 采样检查
    num_to_check = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_check, replace=False)
    
    print(f"Checking {num_to_check} random samples...")
    
    for idx in indices:
        try:
            sample = dataset[idx]
            tgt = sample['targets']
            
            # 检查关节
            jt = first_present(tgt, ['joint_type', 'point_joint_types'])
            if jt is not None:
                n_joints = (jt > 0).sum().item()
                if n_joints > 0:
                    stats['scenes_with_joints'] += 1
                    stats['total_joint_points'] += n_joints
                    for jt_val in jt[jt > 0].unique().tolist():
                        stats['joint_types'][jt_val] = stats['joint_types'].get(jt_val, 0) + 1
            
            # 检查交互部件
            inter = first_present(tgt, ['interactive_labels', 'small_head_class'])
            if inter is not None:
                n_inter = (inter > 0).sum().item()
                if n_inter > 0:
                    stats['scenes_with_interactive'] += 1
                    stats['total_interactive_points'] += n_inter
                    for cls in inter[inter > 0].unique().tolist():
                        stats['interactive_classes'][cls] = stats['interactive_classes'].get(cls, 0) + 1
            
            # 检查大部件
            big = first_present(tgt, ['big_head_class', 'large_part_labels'])
            if big is not None:
                n_big = (big >= 0).sum().item()
                if n_big > 0:
                    stats['scenes_with_big_parts'] += 1
                    for cls in big[big >= 0].unique().tolist():
                        stats['big_classes'][cls] = stats['big_classes'].get(cls, 0) + 1
            
            print(f"  Scene {sample['scene_id']}: joints={n_joints}, interactive={n_inter}, big_parts={n_big}")
            
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
    
    # 打印统计
    print("\n" + "-"*40)
    print("STATISTICS:")
    print(f"  Scenes with joints: {stats['scenes_with_joints']}/{num_to_check}")
    print(f"  Scenes with interactive: {stats['scenes_with_interactive']}/{num_to_check}")
    print(f"  Scenes with big parts: {stats['scenes_with_big_parts']}/{num_to_check}")
    print(f"  Total joint points: {stats['total_joint_points']}")
    print(f"  Total interactive points: {stats['total_interactive_points']}")
    
    if stats['joint_types']:
        print(f"  Joint types found: {stats['joint_types']}")
    if stats['interactive_classes']:
        print(f"  Interactive classes: {stats['interactive_classes']}")
    if stats['big_classes']:
        print(f"  Big part classes: {stats['big_classes']}")
    
    if stats['scenes_with_joints'] == 0 and stats['scenes_with_interactive'] == 0:
        print("\n[WARNING] No positive samples found! Check your JSON files and parsing logic.")
    
    print("="*60 + "\n")
    
    return stats

# ----------------------------
# Positive-aware 包装数据集
# ----------------------------
class PositiveAwareDataset(torch.utils.data.Dataset):
    """
    包装底层 Articulate3D 数据集；若一次采样全背景，则重采样至多 max_resample 次。
    正样本定义：任一点 jt>0 或 interactive_label>0。
    """
    def __init__(self, base: BaseDataset, max_resample: int = 3, want_joint_or_interactive: bool = True):
        self.base = base
        self.max_resample = max_resample
        self.want = want_joint_or_interactive

    def __len__(self): return len(self.base)
    @property
    def scene_ids(self): return getattr(self.base, "scene_ids", None)

    def __getitem__(self, idx: int):
        tries = 0
        best_sample = None
        best_score = -1
        
        while tries < self.max_resample:
            # 使用随机索引而非固定索引，增加多样性
            actual_idx = random.randint(0, len(self.base) - 1) if tries > 0 else idx
            sample = self.base[actual_idx]
            tgt = sample['targets']
            jt = first_present(tgt, ['joint_type','point_joint_types'])
            inter = first_present(tgt, ['interactive_labels','small_head_class'])
            
            # 计算正样本分数
            score = 0
            if jt is not None and (jt > 0).any(): 
                score += (jt > 0).sum().item()
            if inter is not None and (inter > 0).any(): 
                score += (inter > 0).sum().item()
            
            # 保存最好的样本
            if score > best_score:
                best_sample = sample
                best_score = score
            
            tries += 1
            
            # 如果找到足够好的样本，提前退出
            if score > 100 or (not self.want):  # 至少100个正样本点
                sample['meta'] = {'resample_tries': tries, 'positive_score': score}
                return sample
        
        # 返回最好的样本
        if best_sample is not None:
            best_sample['meta'] = {'resample_tries': self.max_resample, 'positive_score': best_score}
            return best_sample
        else:
            sample['meta'] = {'resample_tries': self.max_resample, 'positive_score': 0}
            return sample

# ----------------------------
# Focal Loss（多类）
# ----------------------------
class FocalLoss(nn.Module):
    """ 多类 focal loss，抗极端类不均衡（Lin et al., 2017） """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [*, C], target: [*]
        p = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

# ----------------------------
# 模型（PointNet风格 + 多头）
# ----------------------------
class HierarchicalArticulateKinematicModel(nn.Module):
    def __init__(self, num_large_parts: int, num_interactive_classes: int, hidden_dim=512, in_ch=3, use_dropout=True):
        super().__init__()
        self.num_large_parts = num_large_parts
        self.num_interactive_classes = num_interactive_classes
        dr = 0.3 if use_dropout else 0.0

        self.conv1 = nn.Conv1d(in_ch, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn1=nn.BatchNorm1d(64); self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(256); self.bn4=nn.BatchNorm1d(512); self.bn5=nn.BatchNorm1d(1024)

        self.fusion = nn.Sequential(
            nn.Linear(256+1024, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1 if use_dropout else 0.0),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        self.large_part_seg = nn.Sequential(
            nn.Linear(hidden_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_large_parts+1)  # 0=背景
        )

        self.joint_type_head = nn.Sequential(
            nn.Linear(hidden_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dr*0.5),
            nn.Linear(128,3)  # 0=fixed,1=rev,2=pri
        )
        self.joint_axis_head = nn.Sequential(
            nn.Linear(hidden_dim,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,3)
        )
        self.joint_origin_head = nn.Sequential(
            nn.Linear(hidden_dim,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,3)
        )
        self.joint_range_head = nn.Sequential(
            nn.Linear(hidden_dim,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,2)
        )

        self.interactive_seg = nn.Sequential(
            nn.Linear(hidden_dim + (num_large_parts+1), 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dr*0.5),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_interactive_classes)
        )
        self.interactive_association = nn.Sequential(
            nn.Linear(hidden_dim + (num_large_parts+1), 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_large_parts+1)
        )
        self.hierarchy_predictor = nn.Sequential(
            nn.Linear(1024,512), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(512,256), nn.ReLU(),
            nn.Linear(256, (num_large_parts)*(num_interactive_classes))
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,1), nn.Sigmoid()
        )

        # NEW: interactable 小件的 center-shift 回归头（点 -> 所属部件中心的向量）
        self.center_shift_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Dropout(dr*0.5),
            nn.Linear(hidden_dim//2, 3)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,(nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.constant_(m.bias,0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        B,N,_ = xyz.shape
        x = xyz.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        local = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(local)))
        x = F.relu(self.bn5(self.conv5(x)))
        g = torch.max(x, dim=2)[0]                 # [B,1024]
        local = local.transpose(1,2)               # [B,N,256]
        gexp = g.unsqueeze(1).expand(-1,N,-1)      # [B,N,1024]
        feat = torch.cat([local, gexp], dim=-1)    # [B,N,1280]
        flat = self.fusion(feat.reshape(B*N, -1))  # [B*N,H]
        H = flat.reshape(B,N,-1)

        large = self.large_part_seg(flat).reshape(B,N,-1)
        jt = self.joint_type_head(flat).reshape(B,N,-1)
        axis_raw = self.joint_axis_head(flat).reshape(B,N,-1)
        axis = F.normalize(axis_raw, dim=-1, eps=1e-8)
        origin = self.joint_origin_head(flat).reshape(B,N,-1)
        range_raw = self.joint_range_head(flat).reshape(B,N,-1)
        center = range_raw[...,0:1]; half = F.softplus(range_raw[...,1:2])
        rng = torch.cat([center-half, center+half], dim=-1)

        large_prob = F.softmax(large.detach(), dim=-1)
        inter_in = torch.cat([H, large_prob], dim=-1)
        inter_flat = inter_in.reshape(B*N, -1)
        inter_logits = self.interactive_seg(inter_flat).reshape(B,N,-1)
        inter_assoc  = self.interactive_association(inter_flat).reshape(B,N,-1)

        hier = self.hierarchy_predictor(g).view(B, self.num_large_parts, self.num_interactive_classes)
        hier = torch.sigmoid(hier)

        conf = self.confidence_head(flat).reshape(B,N).squeeze(-1)

        # NEW: center-shift 预测
        center_shift = self.center_shift_head(flat).reshape(B,N,3)

        clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
        return {
            "large_part_logits": clean(large),
            "jt_logits": clean(jt),
            "axis": clean(axis),
            "origin": clean(origin),
            "range": clean(rng),
            "interactive_logits": clean(inter_logits),
            "interactive_assoc": clean(inter_assoc),
            "hierarchy_matrix": clean(hier),
            "confidence": clean(conf),
            "center_shift": clean(center_shift),
        }

# ----------------------------
# 损失与评估
# ----------------------------
@dataclass
class LossWeights:
    jt: float=1.0; inter: float=1.5; part: float=1.0
    axis: float=2.0; origin: float=1.5; rng: float=1.0
    center: float=1.0   # NEW: center-shift 辅助损失

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, exclude_bg: bool=False, eps: float=1e-6) -> torch.Tensor:
    """Multi-class soft Dice loss; optionally exclude channel 0 (background)."""
    # logits [B,N,C], target [B,N]
    B,N,C = logits.shape
    prob = F.softmax(logits, dim=-1)
    one = F.one_hot(target.clamp_min(0), C).float()
    if exclude_bg and C>1:
        prob = prob[...,1:]; one = one[...,1:]
    # flatten over B*N
    prob_f = prob.reshape(-1, prob.shape[-1])
    one_f  = one.reshape(-1,  one.shape[-1])
    inter = (prob_f * one_f).sum(dim=0)
    denom = prob_f.sum(dim=0) + one_f.sum(dim=0)
    dice = 1.0 - (2*inter + eps) / (denom + eps)
    return dice.mean()

def compute_losses(pred, batch, lw: LossWeights, seg_only=False, use_focal=True):
    tgt = batch['targets']
    jt_t   = first_present(tgt, ['joint_type','point_joint_types'])
    part_t = first_present(tgt, ['big_head_class','large_part_labels'])
    axis_t = first_present(tgt, ['joint_axis','point_joint_axes'])
    org_t  = first_present(tgt, ['joint_origin','point_joint_origins'])
    rng_t  = first_present(tgt, ['joint_range','point_joint_ranges'])
    inter_t= first_present(tgt, ['interactive_labels','small_head_class'])

    device = pred['jt_logits'].device
    mask_int = (jt_t > 0) if jt_t is not None else torch.zeros_like(pred['jt_logits'][...,0]).bool()

    # 1) 关节类型
    if use_focal:
        counts = torch.bincount(jt_t.reshape(-1), minlength=3).float()
        w = 1.0/(counts+100.0); w[0]=0.1; w = w/w.sum()*3
        L_jt = FocalLoss(alpha=w.to(device), gamma=2.0)(pred['jt_logits'].reshape(-1,3), jt_t.reshape(-1))
    else:
        w = torch.tensor([0.1,1.0,1.0], device=device)
        L_jt = F.cross_entropy(pred['jt_logits'].permute(0,2,1), jt_t, weight=w, label_smoothing=0.05)

    # 2) 大头分割（将 -1 移到通道0） + Dice
    part_labels = part_t.clone()
    part_labels[part_t >= 0] += 1
    part_labels[part_t < 0] = 0
    part_logits = pred['large_part_logits']
    L_part_ce   = F.cross_entropy(part_logits.permute(0,2,1), part_labels)
    L_part_dice = dice_loss_from_logits(part_logits, part_labels, exclude_bg=False)
    L_part = 1.0 * L_part_ce + 1.0 * L_part_dice

    # 3) 交互小头（多类, 0=background） + Dice
    C = pred['interactive_logits'].shape[-1]
    inter_counts = torch.bincount(inter_t.reshape(-1), minlength=C).float()
    inter_w = torch.ones(C, device=device)
    for i in range(C):
        if inter_counts[i] > 0:
            inter_w[i] = 0.01 if i==0 else min(10.0, 1000.0/(inter_counts[i]+10.0))
    inter_w = inter_w / inter_w.sum() * C
    if use_focal:
        L_inter_ce = FocalLoss(alpha=inter_w, gamma=3.0)(pred['interactive_logits'].reshape(-1,C), inter_t.reshape(-1))
    else:
        L_inter_ce = F.cross_entropy(pred['interactive_logits'].permute(0,2,1), inter_t, weight=inter_w)
    L_inter_dice = dice_loss_from_logits(pred['interactive_logits'], inter_t, exclude_bg=True)
    # CE 比重更大，Dice 作为稳定器
    L_inter = 5.0 * L_inter_ce + 2.0 * L_inter_dice

    # 3.1) NEW: center-shift 辅助（仅对 inter>0 的点）
    cs = pred.get('center_shift')
    if cs is not None and inter_t is not None:
        pts = batch['points']                  # [B,N,3]（已与归一化对齐）
        pids = batch.get('point_part_ids')     # [B,N]
        mask = (inter_t > 0)
        if pids is None:
            # fallback：批级单中心
            centers = []
            for b in range(pts.shape[0]):
                m = mask[b]
                if m.any():
                    c = pts[b][m].mean(dim=0, keepdim=True)
                    centers.append(c.expand_as(pts[b]))
                else:
                    centers.append(torch.zeros_like(pts[b]))
            centers = torch.stack(centers, dim=0)
        else:
            centers = torch.zeros_like(pts)
            B,N = pids.shape
            for b in range(B):
                ids = torch.unique(pids[b][mask[b]])
                for pid in ids.tolist():
                    if pid < 0: continue
                    m = (pids[b]==pid)
                    if m.any():
                        centers[b][m] = pts[b][m].mean(dim=0, keepdim=True)
        target_vec = (centers - pts)
        L_center = (target_vec[mask] - cs[mask]).abs().mean() if mask.any() else torch.tensor(0.0, device=device)
    else:
        L_center = torch.tensor(0.0, device=device)

    # 4) 回归（USDNet-style；translation：只优化 axis；rotation：axis + origin 到真轴距离）
    if not seg_only and mask_int.any():
        is_rot  = (jt_t == 1)
        # is_tran = (jt_t == 2)  # 如需对 translation origin 极小权重，可单独加
        # axis cos（两种类型都优化）
        cos_all = (pred['axis'] * axis_t).sum(dim=-1).clamp(-1,1)
        L_axis = (1.0 - cos_all[mask_int]).mean()
        # origin 对 rotation： || a* × (o_pred - o_gt) ||
        if is_rot.any():
            a = F.normalize(axis_t[is_rot], dim=-1, eps=1e-8)
            o = pred['origin'][is_rot]
            ogt = org_t[is_rot]
            v = torch.cross(a, (o - ogt), dim=-1)
            L_org = v.norm(dim=-1).mean()
        else:
            L_org = torch.tensor(0.0, device=device)
        # range：两类都用 L1
        L_rng  = F.l1_loss(pred['range'][mask_int],  rng_t[mask_int])
    else:
        L_axis=L_org=L_rng=torch.tensor(0.0, device=device)

    total = lw.jt*L_jt + lw.inter*L_inter + lw.part*L_part + lw.center*L_center
    if not seg_only: total += lw.axis*L_axis + lw.origin*L_org + lw.rng*L_rng

    logs = {k: float(v.detach().item()) for k,v in dict(
        L_total=total, L_jt=L_jt, 
        L_inter=L_inter, L_inter_ce=L_inter_ce, L_inter_dice=L_inter_dice,
        L_part=L_part,   L_part_ce=L_part_ce,   L_part_dice=L_part_dice,
        L_axis=L_axis, L_org=L_org, L_rng=L_rng, L_center=L_center
    ).items()}
    return total, logs

@torch.no_grad()
def eval_batch(pred, batch) -> Dict[str,float]:
    tgt = batch['targets']
    jt_t   = first_present(tgt, ['joint_type','point_joint_types'])
    inter_t= first_present(tgt, ['interactive_labels','small_head_class'])
    part_t = first_present(tgt, ['big_head_class','large_part_labels'])
    axis_t = first_present(tgt, ['joint_axis','point_joint_axes'])

    jt_pred = pred['jt_logits'].argmax(dim=-1)
    inter_pred = pred['interactive_logits'].argmax(dim=-1)
    part_pred = pred['large_part_logits'].argmax(dim=-1)

    part_labels = part_t.clone()
    part_labels[part_t >= 0] += 1
    part_labels[part_t < 0] = 0

    mask_part = (part_labels >= 0)
    mask_int = (jt_t > 0)

    jt_acc = (jt_pred == jt_t).float().mean().item()
    inter_acc = (inter_pred == inter_t).float().mean().item()
    part_acc = ((part_pred == part_labels) & mask_part).float().sum().item() / (mask_part.float().sum().item()+1e-8)

    # 每类交互准确率
    cls_acc = {}
    C = pred['interactive_logits'].shape[-1]
    for c in range(C):
        m = (inter_t == c)
        if m.any():
            cls_acc[f"acc_inter_cls{c}"] = (((inter_pred==c)&m).float().sum() / m.float().sum()).item()

    jt_pos_ratio = (jt_t > 0).float().mean().item()
    inter_pos_ratio = (inter_t > 0).float().mean().item()

    if mask_int.any():
        axis_cos = ((pred['axis'] * axis_t).sum(dim=-1)).clamp(-1,1)[mask_int].mean().item()
    else:
        axis_cos = 0.0

    out = dict(acc_jt=jt_acc, acc_inter=inter_acc, acc_part=part_acc,
               axis_cos=axis_cos, jt_pos_ratio=jt_pos_ratio, inter_pos_ratio=inter_pos_ratio)
    out.update(cls_acc)
    return out

# ----------------------------
# 数据集构建 & 辅助工具
# ----------------------------
@torch.no_grad()
def list_all_scene_ids_from_json(artic_root: str):
    parts = {Path(p).name.replace("_parts.json","")
             for p in glob.glob(os.path.join(artic_root, "*_parts.json"))}
    artic = {Path(p).name.replace("_artic.json","")
             for p in glob.glob(os.path.join(artic_root, "*_artic.json"))}
    return sorted(parts & artic)

@torch.no_grad()
def make_dataset(args, split: str, scene_ids: List[str]):
    # 兼容 dataset 的不同签名：尝试传 labeled_face_ratio/mesh_glob，缺就忽略
    def _build():
        try:
            return BaseDataset(
                artic_root=args.artic_root, scan_root=args.scan_root, n_points=args.points,
                split=split, augment=(split=='train'), normalize=True,
                mesh_glob=args.mesh_glob, range_epsilon=0.01
            )
        except TypeError:
            try:
                return BaseDataset(
                    artic_root=args.artic_root, scan_root=args.scan_root, n_points=args.points,
                    split=split, augment=(split=='train'), normalize=True,
                    mesh_glob=args.mesh_glob
                )
            except TypeError:
                return BaseDataset(
                    artic_root=args.artic_root, scan_root=args.scan_root, n_points=args.points,
                    split=split, augment=(split=='train'), normalize=True
                )
    ds = _build()
    ds.scene_ids = scene_ids
    return ds

@torch.no_grad()
def split_scenes(args):
    # 用 JSON 列出所有成对 id（不检查 mesh）
    all_ids = list_all_scene_ids_from_json(args.artic_root)
    if not all_ids:
        raise SystemExit("No paired *_parts.json & *_artic.json under artic_root")

    # probe：用 mesh_glob 验证哪些 scene 真有网格
    try:
        probe = BaseDataset(
            artic_root=args.artic_root, scan_root=args.scan_root,
            n_points=16, split='all', augment=False, normalize=True,
            mesh_glob=args.mesh_glob
        )
    except TypeError:
        probe = BaseDataset(
            artic_root=args.artic_root, scan_root=args.scan_root,
            n_points=16, split='all', augment=False, normalize=True
        )

    available = [sid for sid in all_ids if sid in getattr(probe, "scene_ids", [])]
    if len(available) == 0:
        raise SystemExit("No scenes with meshes found under scan_root (after probe)")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(available)); rng.shuffle(idx)
    if args.overfit_one:
        sid = args.overfit_scene if (args.overfit_scene and args.overfit_scene in available) else available[idx[0]]
        print(f"[Split] overfit_one on scene: {sid}")
        return [sid], [sid]
    ratio = args.force_split if args.force_split > 0 else 0.8
    n_train = int(len(available) * ratio)
    if len(available) > 1:
        n_train = max(1, min(len(available)-1, n_train))
    else:
        n_train = 1
    train_ids = [available[i] for i in idx[:n_train]]
    val_ids   = [available[i] for i in idx[n_train:]]
    print(f"[Split] train/val = {len(train_ids)}/{len(val_ids)}")
    return train_ids, val_ids

# ----------------------------
# 训练循环
# ----------------------------
def move_to(obj, device):
    if torch.is_tensor(obj): return obj.to(device, non_blocking=True)
    if isinstance(obj, dict): return {k: move_to(v, device) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return type(obj)(move_to(v, device) for v in obj)
    return obj

def train_one_epoch(model, loader, optimizer, scaler, device, lw, seg_only, accum_steps, max_norm, epoch_idx=1):
    model.train()
    logs_accum = {}; n_batches=0
    for bidx, batch in enumerate(loader):
        batch = move_to(batch, device)
        if bidx % accum_steps == 0: optimizer.zero_grad()

        # AMP
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(batch['points'])
                loss, logs = compute_losses(out, batch, lw=lw, seg_only=seg_only, use_focal=True)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (bidx+1) % accum_steps == 0:
                if max_norm>0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer); scaler.update()
        else:
            out = model(batch['points'])
            loss, logs = compute_losses(out, batch, lw=lw, seg_only=seg_only, use_focal=True)
            (loss/accum_steps).backward()
            if (bidx+1) % accum_steps == 0:
                if max_norm>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        metrics = eval_batch(out, batch)
        for k,v in {**logs, **metrics}.items():
            logs_accum[k] = logs_accum.get(k,0.0) + float(v)
        n_batches += 1

        # 首个 batch 打印直方图
        if epoch_idx==1 and bidx==0:
            t = batch['targets']
            inter = first_present(t, ['interactive_labels','small_head_class'])
            C = out['interactive_logits'].shape[-1]
            hist = torch.bincount(inter.reshape(-1), minlength=C).cpu().tolist()
            jt = first_present(t, ['joint_type','point_joint_types'])
            jt_ratio = (jt>0).float().mean().item()
            meta = batch.get('meta', {})
            tries = meta.get('resample_tries', '?') if isinstance(meta, dict) else '?'
            score = meta.get('positive_score', '?') if isinstance(meta, dict) else '?'
            print(f"[DEBUG] inter_hist={hist} | jt_pos={jt_ratio:.3f} | tries={tries} score={score}")

        if bidx % 10 == 0:
            print(f"  Batch [{bidx}/{len(loader)}]  L={logs['L_total']:.4f} | JT_pos={metrics['jt_pos_ratio']:.3f}  Inter_pos={metrics['inter_pos_ratio']:.3f}")

    for k in logs_accum: logs_accum[k] /= max(n_batches,1)
    return logs_accum

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    logs_accum = {}; n_batches=0
    for batch in loader:
        batch = move_to(batch, device)
        out = model(batch['points'])
        metrics = eval_batch(out, batch)
        for k,v in metrics.items():
            logs_accum[k] = logs_accum.get(k,0.0) + float(v)
        n_batches += 1
    for k in logs_accum: logs_accum[k] /= max(n_batches,1)
    return logs_accum

def save_ckpt(path, model, optimizer, scaler, epoch, best_metric, label_maps: Dict[str,Any]|None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'best_metric': best_metric,
        'model_config': {
            'num_large_parts': model.num_large_parts,
            'num_interactive_classes': model.num_interactive_classes,
            'hidden_dim': 512
        },
        'label_mappings': label_maps
    }, path)
    print(f"[SAVE] Checkpoint -> {path}")

def train_loop(model, train_loader, val_loader, optimizer, scaler, device,
               epochs, eval_interval, out_dir, seg_only, accum_steps, max_norm, label_maps):
    best = None
    for ep in range(1, epochs+1):
        print(f"\n[Epoch {ep}/{epochs}]")
        logs = train_one_epoch(model, train_loader, optimizer, scaler, device,
                               lw=LossWeights(), seg_only=seg_only,
                               accum_steps=accum_steps, max_norm=max_norm, epoch_idx=ep)
        if math.isnan(logs['L_total']):
            print(f"[E{ep:03d}] NaN loss encountered, stop."); break

        metric=None
        if val_loader and (ep%eval_interval==0 or ep==epochs):
            valm = validate_one_epoch(model, val_loader, device)
            metric = valm.get('acc_inter', 0.0)
            print(f"[E{ep:03d}] Train: L={logs['L_total']:.3f} JT={logs['acc_jt']:.3f} Part={logs['acc_part']:.3f} Inter={logs['acc_inter']:.3f} "
                  f"| Pos JT={logs['jt_pos_ratio']:.3f} Inter={logs['inter_pos_ratio']:.3f}")
            print(f"[E{ep:03d}] Val  : JT={valm['acc_jt']:.3f} Part={valm['acc_part']:.3f} Inter={valm['acc_inter']:.3f} Axis={valm['axis_cos']:.3f}")

            inter_cls = []
            for k,v in valm.items():
                if k.startswith('acc_inter_cls'):
                    cid = int(k.replace('acc_inter_cls',''))
                    name = label_maps.get('interactive_id_to_label', {}).get(cid, str(cid))
                    inter_cls.append(f"{name}:{v:.2f}")
            print(f"[E{ep:03d}] Inter-class Acc: {', '.join(inter_cls) if inter_cls else 'none'}")
        else:
            print(f"[E{ep:03d}] Train: L={logs['L_total']:.3f} JT={logs['acc_jt']:.3f} Part={logs['acc_part']:.3f} Inter={logs['acc_inter']:.3f}")

        if logs['jt_pos_ratio']<0.01 and logs['inter_pos_ratio']<0.01:
            print("[WARN] Positive sample ratio extremely low, check sampling.")

        if metric is not None and (best is None or metric>best):
            best = metric
            save_ckpt(os.path.join(out_dir,'best.pth'), model, optimizer, scaler, ep, best, label_maps)
            print(f"[E{ep:03d}] Saved best (acc_inter={metric:.4f})")

        if ep%10==0:
            save_ckpt(os.path.join(out_dir,f'checkpoint_{ep}.pth'), model, optimizer, scaler, ep, best, label_maps)

    save_ckpt(os.path.join(out_dir,'final.pth'), model, optimizer, scaler, epochs, best, label_maps)
    print(f"\nTraining complete. Best val acc_inter = {best:.4f}" if best is not None else "Training complete.")

# ----------------------------
# MAIN
# ----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser("Train Articulate3D kinematics (robust + USDNet-style)")
    p.add_argument("--artic_root", required=True)
    p.add_argument("--scan_root",  required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--mesh_glob",  default="scans/mesh_aligned_0.05.ply",
                   help="ScanNet++ 对齐网格常见文件名/通配符")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=4)
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--workers",type=int, default=4)
    p.add_argument("--force_split", type=float, default=0.0)
    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--overfit_scene", type=str, default="")
    p.add_argument("--seg_only", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp",  action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--find_positive", action="store_true", help="只做正样本场景扫描，不训练")
    p.add_argument("--diagnose", action="store_true", help="运行数据诊断")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Articulate3D Kinematic Training  (robust + USDNet-style)")
    print("="*60)
    print(f"  Articulate3D  : {args.artic_root}")
    print(f"  ScanNet++     : {args.scan_root}")
    print(f"  Mesh glob     : {args.mesh_glob}")
    print(f"  Output        : {args.out_dir}")
    print(f"  Epochs={args.epochs}  Batch={args.batch}  Points={args.points}  "
          f"AMP={args.amp}  GradAccum={args.grad_accum}  SegOnly={args.seg_only}")
    print("="*60 + "\n")

    # 数据诊断模式
    if args.diagnose:
        print("Running dataset diagnosis...")
        test_ds = BaseDataset(
            artic_root=args.artic_root, scan_root=args.scan_root,
            n_points=args.points, split='all', augment=False, normalize=True,
            mesh_glob=args.mesh_glob
        )
        stats = diagnose_dataset(test_ds, num_samples=min(20, len(test_ds)))
        
        if stats['scenes_with_joints'] == 0 and stats['scenes_with_interactive'] == 0:
            print("\n[ERROR] No positive samples found in dataset!")
            print("Please check:")
            print("1. Your JSON files contain 'articulations' data")
            print("2. The articulations have non-zero ranges")
            print("3. Part labels are being correctly parsed")
            sys.exit(1)
        return

    if args.find_positive:
        # 粗扫：每个 scene 试采样几次，找出正样本命中率高的（便于 overfit 调试）
        from itertools import islice
        all_ids = list_all_scene_ids_from_json(args.artic_root)
        stats = []
        for sid in all_ids:
            try:
                ds = make_dataset(args, 'all', [sid])
                got = 0
                for _ in range(6):
                    s = ds[0]
                    t = s['targets']
                    jt = first_present(t, ['joint_type','point_joint_types'])
                    inter = first_present(t, ['interactive_labels','small_head_class'])
                    if (jt is not None and (jt>0).any()) or (inter is not None and (inter>0).any()):
                        got += 1
                stats.append((sid, got/6.0))
            except:
                stats.append((sid, 0.0))
        stats.sort(key=lambda x: x[1], reverse=True)
        print("[find_positive] Top scenes:")
        for sid, r in islice(stats, 20):
            print(f"  {sid}: {r:.2f}")
        return

    # 划分
    train_ids, val_ids = split_scenes(args)

    # 构建数据集（train 包装为 PositiveAware）
    base_train = make_dataset(args, 'train', train_ids)
    
    # 先诊断基础训练集
    print("\nDiagnosing training dataset...")
    train_stats = diagnose_dataset(base_train, num_samples=min(10, len(base_train)))
    
    if train_stats['scenes_with_joints'] == 0 and train_stats['scenes_with_interactive'] == 0:
        print("\n[ERROR] No positive samples in training set!")
        print("Cannot proceed with training. Please check your data.")
        sys.exit(1)
    
    train_ds = PositiveAwareDataset(base_train, max_resample=5, want_joint_or_interactive=True)
    val_ds   = make_dataset(args, 'val',   val_ids)

    # 类别映射（尽力从底层拿，不强制）
    label_maps = dict(
        large_label_to_id = getattr(base_train, 'large_label_to_id', {}),
        large_id_to_label = getattr(base_train, 'large_id_to_label', {}),
        num_large_classes = getattr(base_train, 'num_large_classes', 5),
        interactive_label_to_id = getattr(base_train, 'interactive_label_to_id', {}),
        interactive_id_to_label = getattr(base_train, 'interactive_id_to_label', {}),
        num_interactive_classes = getattr(base_train, 'num_interactive_classes', 7),
    )
    num_large = int(label_maps['num_large_classes'])
    num_inter = int(label_maps['num_interactive_classes'])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalArticulateKinematicModel(num_large_parts=num_large,
                                                 num_interactive_classes=num_inter,
                                                 hidden_dim=512, in_ch=3).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type=="cuda") else None

    print(f"[Model] Params: {count_params(model):,}")
    print(f"[Model] Large parts(out) = {num_large+1} (含背景0)")
    print(f"[Model] Interactive classes(out) = {num_inter}")
    print(f"[Device] {device}")
    print("[INFO] Focal + Dice losses enabled (handling class imbalance)")

    train_loop(model, train_loader, val_loader, optim, scaler, device,
               epochs=args.epochs, eval_interval=1, out_dir=args.out_dir,
               seg_only=args.seg_only, accum_steps=args.grad_accum, max_norm=args.max_norm,
               label_maps=label_maps)

if __name__ == "__main__":
    main()