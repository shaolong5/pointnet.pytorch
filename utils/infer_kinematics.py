#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Articulate3D kinematics (compatible with train_kinematics.py).
- Fixed imports and path handling for your environment
"""
from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional clustering
try:
    from sklearn.cluster import DBSCAN
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        out[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    return out

def to_cpu_detach(x):
    return x.detach().cpu() if torch.is_tensor(x) else x

def save_pt(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))

def cluster_instances(points: torch.Tensor,
                      inter_labels: torch.Tensor,
                      jt_logits: torch.Tensor,
                      axis: torch.Tensor,
                      origin: torch.Tensor,
                      rng: torch.Tensor,
                      eps_ratio: float,
                      min_samples: int,
                      min_inst_pts: int) -> List[Dict[str, Any]]:
    """Return simple instance proposals over interactive points via DBSCAN per class."""
    if not _HAS_SK:
        return []
    pts_np = points.cpu().numpy()
    N = pts_np.shape[0]
    inter = inter_labels.cpu().numpy()
    instances: List[Dict[str, Any]] = []
    next_id = 1
    scale = max(pts_np.max() - pts_np.min(), 1e-6)
    
    for cls in sorted(set(int(c) for c in inter.tolist())):
        if cls == 0:  # skip background
            continue
        mask = (inter == cls)
        idxs = mask.nonzero()[0]
        if idxs.size == 0:
            continue
        X = pts_np[idxs]
        eps = float(eps_ratio) * float(scale)
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        except Exception:
            continue
        labels = db.labels_
        for u in sorted(set(labels.tolist())):
            if u < 0:
                continue
            m = (labels == u)
            if m.sum() < min_inst_pts:
                continue
            sel = torch.from_numpy(idxs[m]).long()
            # aggregate predictions within the cluster
            jt = jt_logits[sel].argmax(-1)
            jt_majority = int(torch.mode(jt).values.item())
            ax = F.normalize(axis[sel].mean(0, keepdim=False), dim=0, eps=1e-8)
            og = origin[sel].mean(0, keepdim=False)
            rg = rng[sel].mean(0, keepdim=False)
            instances.append(dict(
                instance_id=next_id,
                class_id=int(cls),
                joint_type=int(jt_majority),
                axis=ax.cpu().tolist(),
                origin=og.cpu().tolist(),
                range=rg.cpu().tolist(),
                num_points=int(sel.numel()),
                point_indices=sel.cpu().tolist()  # Added for visualization
            ))
            next_id += 1
    return instances

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser("Articulate3D inference")
    ap.add_argument("--artic_root", required=True)
    ap.add_argument("--scan_root",  required=True)
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mesh_glob", default="scans/mesh_aligned_0.05.ply")
    ap.add_argument("--amp", action="store_true")
    # clustering
    ap.add_argument("--enable_clustering", action="store_true", default=True)
    ap.add_argument("--eps_ratio", type=float, default=0.02)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--min_instance_points", type=int, default=30)
    args = ap.parse_args()

    # Setup paths - add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import model + dataset from local files
    try:
        from train_kinematics import HierarchicalArticulateKinematicModel
    except Exception as e:
        print(f"[ERROR] import model failed: {e}")
        sys.exit(1)
    try:
        from dataset_articulate3d import Articulate3DDataset
    except Exception as e:
        print(f"[ERROR] import dataset failed: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load checkpoint
    print(f"[Loading] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    state = None
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "model" in ckpt):
        state = ckpt.get("model_state_dict", ckpt.get("model"))
        model_cfg = ckpt.get("model_config", {})
        label_maps = ckpt.get("label_mappings", {})
    else:
        state = ckpt
        model_cfg = {}
        label_maps = {}
    state = strip_module_prefix(state)

    # Derive config
    num_large = int(model_cfg.get("num_large_parts", 5))
    num_inter = int(model_cfg.get("num_interactive_classes", 7))
    hidden_dim = int(model_cfg.get("hidden_dim", 512))

    print(f"[Model] num_large_parts={num_large}, num_interactive_classes={num_inter}, hidden_dim={hidden_dim}")
    model = HierarchicalArticulateKinematicModel(
        num_large_parts=num_large,
        num_interactive_classes=num_inter,
        hidden_dim=hidden_dim, 
        in_ch=3,
        use_dropout=False
    ).to(device)
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   
        print(f"[WARN] missing keys: {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys: {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")
    model.eval()

    # Build dataset/loader - use same parameters as training
    print(f"[Dataset] Loading from {args.artic_root} and {args.scan_root}")
    ds = Articulate3DDataset(
        artic_root=args.artic_root, 
        scan_root=args.scan_root,
        n_points=args.points, 
        split="all", 
        augment=False, 
        normalize=True,
        mesh_glob=args.mesh_glob
    )
    print(f"[Dataset] Found {len(ds)} scenes")
    
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    out_dir = Path(args.out_dir)
    preds_dir = out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Save label mappings for viz
    if label_maps:
        with open(out_dir / "label_mappings.json", "w", encoding="utf-8") as f:
            json.dump({k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                       for k, v in label_maps.items()}, f, ensure_ascii=False, indent=2)
        print(f"[Save] Label mappings -> {out_dir / 'label_mappings.json'}")

    dtype = torch.float16 if (args.amp and device.type == "cuda") else torch.float32
    print(f"[Infer] scenes={len(ds)} | AMP={args.amp} | clustering={args.enable_clustering and _HAS_SK}")
    
    with torch.no_grad():
        for i, batch in enumerate(dl):
            pts = batch["points"].to(device, dtype=dtype)
            scene_ids = batch["scene_id"]
            
            # Handle both single string and list of strings
            if isinstance(scene_ids, str):
                scene_ids = [scene_ids]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=args.amp):
                pred = model(pts)

            B = pts.shape[0]
            for b in range(B):
                sid = scene_ids[b]
                pb = {k: to_cpu_detach(v[b]) if torch.is_tensor(v) else v for k, v in pred.items()}
                sample_pts = to_cpu_detach(pts[b]).float()
                
                # Get predictions
                inter_labels = pb["interactive_logits"].argmax(-1)
                
                # Create instance labels array
                instance_labels = torch.full_like(inter_labels, -1)
                instances = []
                
                if args.enable_clustering and _HAS_SK:
                    instances = cluster_instances(
                        sample_pts, inter_labels,
                        pb["jt_logits"], pb["axis"], pb["origin"], pb["range"],
                        eps_ratio=args.eps_ratio, 
                        min_samples=args.min_samples,
                        min_inst_pts=args.min_instance_points
                    )
                    
                    # Create instance labels from clustering results
                    for inst in instances:
                        if 'point_indices' in inst:
                            instance_labels[inst['point_indices']] = inst['instance_id']
                
                # Package instance results for compatibility with visualizer
                instance_results = {
                    'instances': instances,
                    'instance_labels': instance_labels.cpu().numpy()
                }
                
                save_pt(preds_dir / f"{sid}.pt", {
                    "scene_id": sid,
                    "points": sample_pts,
                    "pred": pb,
                    "instance_results": instance_results,
                })
                print(f"[{i*args.batch + b + 1}/{len(ds)}] Saved {sid} -> {preds_dir / f'{sid}.pt'} ({len(instances)} instances)")

    print(f"\n[Complete] Results saved to {out_dir}")
    print(f"Run visualization with: python visualize_results.py --results_dir {out_dir} --show")

if __name__ == "__main__":
    main()