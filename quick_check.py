#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick checks for your Articulate3D kinematics project.

Modes:
  - ckpt:     print checkpoint metadata
  - datalen:  print dataset sizes (available scenes)
  - collapse: run a few batches and estimate if logits collapsed

Design goals:
  • Works with your dataset file living at arbitrary paths
    (env ARTIC_DS or common repo locations).
  • Tries to load your actual training model first (TinyPointNet from
    utils/train_kinematics*.py); falls back to Hierarchical* model if present.
  • Prints missing/unexpected keys so you can tell if the ckpt truly matches
    the instantiated model (very important before judging collapse!).

Examples
--------
# 1) dataset sizes only
python -u quick_check.py datalen \
  --artic_root /home/shaolongshi/data/Articulate3D \
  --scan_root  /home/shaolongshi/data/ScanNet++

# 2) inspect checkpoint file
python -u quick_check.py ckpt --ckpt /home/shaolongshi/data/kin_ckpts/best.pth

# 3) collapse check (recommended)
python -u quick_check.py collapse \
  --ckpt      /home/shaolongshi/data/kin_ckpts/best.pth \
  --artic_root /home/shaolongshi/data/Articulate3D \
  --scan_root  /home/shaolongshi/data/ScanNet++ \
  --batch 8 --batches 5 --points 2048 --device cuda

If your dataset file is not in a standard place, set:
  ARTIC_DS=/absolute/path/to/dataset_articulate3d.py
"""
from __future__ import annotations
import argparse, os, sys, math, json, pathlib, importlib.util
from typing import Tuple, Dict, Any


DATASET_FILE = "/home/shaolongshi/data/pointnet.pytorch/pointnet/dataset_articulate3d.py"
# Optional heavy deps imported lazily where needed

# -----------------------------
# Helpers
# -----------------------------
THIS = pathlib.Path(__file__).resolve()
REPO = THIS.parent  # assuming quick_check.py lives in repo root

def expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def import_module_from_file(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{mod_name}' from '{file_path}'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -----------------------------
# Dataset import (robust)
# -----------------------------

def load_dataset_module() -> Any:
    env_file = os.environ.get("ARTIC_DS", "")
    candidates = []
    if env_file:
        candidates.append(pathlib.Path(expand(env_file)))
    # common repo locations
    candidates += [
        REPO / "dataset_articulate3d.py",
        REPO / "pointnet" / "dataset_articulate3d.py",
        REPO / "utils" / "dataset_articulate3d.py",
        pathlib.Path("/home/shaolongshi/data/pointnet.pytorch/pointnet/dataset_articulate3d.py"),
    ]
    for p in candidates:
        if p.exists():
            return import_module_from_file("dataset_articulate3d", str(p))
    raise ModuleNotFoundError("dataset_articulate3d.py not found. Set ARTIC_DS=/abs/path/to/dataset_articulate3d.py")


# -----------------------------
# Model import (prefers your TinyPointNet)
# -----------------------------

def load_model_class() -> Tuple[Any, str]:
    """Return (ModelClass, flavor). Tries TinyPointNet from your training script,
    then falls back to HierarchicalArticulateKinematicModel if available.
    """
    # Try utils/train_kinematics*.py candidates
    tk_candidates = [
        REPO / "utils" / "train_kinematics.py",
        REPO / "utils" / "train_kinematics_explicit.py",
    ]
    for p in tk_candidates:
        if p.exists():
            try:
                mod = import_module_from_file("train_kinematics_mod", str(p))
                if hasattr(mod, "TinyPointNet"):
                    return getattr(mod, "TinyPointNet"), "tiny"
            except Exception:
                pass
    # Fallback: kinetic_model (hierarchical)
    # Try pointnet.kinetic_model then local kinetic_model
    try:
        sys.path.insert(0, str(REPO))
        from pointnet.kinetic_model import HierarchicalArticulateKinematicModel as M  # type: ignore
        return M, "hier"
    except Exception:
        try:
            from kinetic_model import HierarchicalArticulateKinematicModel as M  # type: ignore
            return M, "hier"
        except Exception as e:
            raise ImportError("No usable model class found (TinyPointNet / Hierarchical*).") from e


# -----------------------------
# Checkpoint IO
# -----------------------------

def load_ckpt(path: str, weights_only: bool = False) -> Dict[str, Any]:
    import torch
    return torch.load(expand(path), map_location="cpu", weights_only=weights_only)


def print_ckpt_info(ckpt: Dict[str, Any]) -> None:
    def get_sched_epoch(sched):
        if sched is None: return None
        if isinstance(sched, dict): return sched.get('last_epoch', None)
        return getattr(sched, 'last_epoch', None)
    epoch = ckpt.get('epoch')
    optim = ckpt.get('optimizer', ckpt.get('optim', {}))
    sched = ckpt.get('sched', None)
    best  = ckpt.get('best', ckpt.get('best_metric'))
    opt_states = len(optim.get('state', {})) if isinstance(optim, dict) else None
    print(f"epoch: {epoch}")
    print(f"optimizer states: {opt_states}")
    print(f"sched.last_epoch: {get_sched_epoch(sched)}")
    print(f"best/best_metric: {best}")
    has_model = isinstance(ckpt.get('model', None), dict)
    print(f"contains model state_dict: {has_model}")


# -----------------------------
# Collapse utilities
# -----------------------------

def pick_logits(out) -> Tuple[Any, str]:
    import torch
    if isinstance(out, dict):
        for k in ['large_part_seg','interactive_seg','seg','seg_logits','part_logits']:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v, k
    if isinstance(out, (list, tuple)):
        for t in out:
            if hasattr(t, 'ndim') and getattr(t, 'ndim', 0) >= 3:
                return t, 'tuple_item'
    if hasattr(out, 'ndim'):
        return out, 'tensor'
    return None, 'none'


def class_dim_of(logits, hint_classes: int) -> int:
    # Guess which dim is class dim
    if logits.shape[-1] in (hint_classes, 2, 3, 11): return -1
    if logits.shape[1]   in (hint_classes, 2, 3, 11): return 1
    # Fallback: last dim if small
    return -1 if logits.shape[-1] <= 64 else 1


# -----------------------------
# Modes
# -----------------------------

def do_datalen(args):
    Dmod = load_dataset_module()
    D = getattr(Dmod, 'Articulate3DDatasetFull', getattr(Dmod, 'Articulate3DDataset'))
    for split in ['train','val']:
        ds = D(artic_root=expand(args.artic_root), scan_root=expand(args.scan_root),
               split=split, n_points=args.points, augment=False, normalize=True)
        print(f"{split:5s} len = {len(ds)}")


def do_collapse(args):
    import torch, numpy as np
    # 1) dataset to get K (num_large_classes)
    Dmod = load_dataset_module()
    D = getattr(Dmod, 'Articulate3DDatasetFull', getattr(Dmod, 'Articulate3DDataset'))
    ds_for_K = D(artic_root=expand(args.artic_root), scan_root=expand(args.scan_root),
                 split='train', n_points=args.points, augment=False, normalize=True,
                 num_large_classes=args.num_classes)
    K = max(1, len(getattr(ds_for_K, 'large_label_to_id', {}) ) or args.num_classes)

    # 2) model
    M, flavor = load_model_class()
    if flavor == 'tiny':
        model = M(num_large_classes=K)
    else:  # hierarchical fallback: try common ctor signatures
        tried = [
            dict(num_classes=args.num_classes, inter_classes=args.inter_classes),
            dict(num_seg_classes=args.num_classes, inter_classes=args.inter_classes),
            {},
        ]
        model = None
        for kw in tried:
            try:
                model = M(**kw)
                break
            except TypeError:
                continue
        if model is None:
            raise RuntimeError("Failed to construct fallback model; please adjust ctor in quick_check.py")

    # 3) load ckpt
    ckpt = load_ckpt(args.ckpt, weights_only=False)
    state = ckpt.get('model', ckpt)
    res = model.load_state_dict(state, strict=False)
    missing = getattr(res, 'missing_keys', [])
    unexpected = getattr(res, 'unexpected_keys', [])
    print(f"[CKPT] missing: {len(missing)} unexpected: {len(unexpected)}")
    for k in missing[:10]:
        print("  missing:", k)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device).eval()

    # 4) dataloader
    ds = D(artic_root=expand(args.artic_root), scan_root=expand(args.scan_root),
           split='val', n_points=args.points, augment=False, normalize=True,
           num_large_classes=K)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    total = 0
    counts = None
    ents = []
    import math as _math
    for i, b in enumerate(dl):
        if i >= args.batches: break
        x = b['points'].to(device).float()
        with torch.no_grad():
            out = model(x)
        logits, key = pick_logits(out)
        if logits is None:
            raise RuntimeError("No classification logits found (expect one of: large_part_seg / interactive_seg / seg_*)")
        dimc = class_dim_of(logits, K)
        probs = torch.softmax(logits, dim=dimc)
        preds = probs.argmax(dim=dimc)
        if dimc == 1:  # (B,C,N)->(B,N)
            preds = preds.transpose(1, -1)
        # entropy per point
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=dimc).mean().item()
        ents.append(ent)
        # counts
        C = probs.shape[dimc]
        cur = torch.bincount(preds.flatten().cpu(), minlength=C)
        counts = cur if counts is None else (counts + cur)
        total += preds.numel()
        uniq = torch.nonzero(cur).flatten().tolist()
        print(f"Batch {i}  logits='{key}'  unique classes: {uniq}")

    if not counts is None and total > 0:
        dom_ratio = counts.max().item() / total
        avg_ent = float(sum(ents)/len(ents)) if ents else 0.0
        lnC = float(np.log(len(counts)))
        print("-"*50)
        print(f"dominant ratio = {dom_ratio*100:.2f}%")
        print(f"avg entropy    = {avg_ent:.4f}   ln(C) = {lnC:.4f}")
        if dom_ratio >= args.collapse_ratio or abs(avg_ent - lnC) < args.entropy_tolerance:
            print("=> COLLAPSE LIKELY（主导类≥阈值 或 平均熵≈ln(C)）")
        else:
            print("=> Not collapsed")
    else:
        print("No predictions gathered.")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Quick checker: ckpt-info / collapse / datalen")
    ap.add_argument('mode', choices=['ckpt','collapse','datalen'])
    ap.add_argument('--ckpt', help='Path to checkpoint (.pth)')
    ap.add_argument('--artic_root', help='Articulate3D root')
    ap.add_argument('--scan_root', help='ScanNet++ root')
    ap.add_argument('--num_classes', type=int, default=11)
    ap.add_argument('--inter_classes', type=int, default=2)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--batches', type=int, default=5)
    ap.add_argument('--points', type=int, default=2048)
    ap.add_argument('--device', default=None)
    # thresholds for collapse judgement
    ap.add_argument('--collapse_ratio', type=float, default=0.90, help='dominant class ratio threshold')
    ap.add_argument('--entropy_tolerance', type=float, default=0.05, help='|avg_ent - ln(C)| threshold')
    ap.add_argument('--weights_only', action='store_true', help='use torch.load(..., weights_only=True) for untrusted ckpts')
    args = ap.parse_args()

    if args.mode == 'ckpt':
        if not args.ckpt: raise SystemExit("--ckpt is required for ckpt mode")
        ckpt = load_ckpt(args.ckpt, weights_only=args.weights_only)
        print_ckpt_info(ckpt)
        return
    if args.mode == 'datalen':
        if not (args.artic_root and args.scan_root): raise SystemExit("--artic_root/--scan_root required")
        do_datalen(args); return
    if args.mode == 'collapse':
        if not (args.ckpt and args.artic_root and args.scan_root): raise SystemExit("--ckpt/--artic_root/--scan_root required")
        do_collapse(args); return

if __name__ == '__main__':
    main()
