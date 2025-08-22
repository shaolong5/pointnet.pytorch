#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, json, time, argparse, inspect
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

# =============== 数据集导入（Full 优先，缺省退 Basic） ===============
try:
    from pointnet.dataset_articulate3d import Articulate3DDatasetFull as Articulate3DDataset
    DATASET_NAME = "Articulate3DDatasetFull"
except Exception:
    from pointnet.dataset_articulate3d import Articulate3DDataset
    DATASET_NAME = "Articulate3DDataset"

import pointnet.kinetic_model as km

# =============== Mesh 查找规则（和 compare 脚本一致并多一层） ===============
CANDIDATE_MESH_NAMES = [
    "mesh_aligned_0.05.ply",
    "mesh_aligned.ply",
    "mesh.ply",
    "mesh_aligned_0.05.obj",
    "mesh.obj",
]

def list_scene_ids(artic_root: str):
    parts = sorted(glob.glob(os.path.join(artic_root, "*_parts.json")))
    ids = []
    for p in parts:
        sid = os.path.basename(p).replace("_parts.json", "")
        ajson = os.path.join(artic_root, f"{sid}_artic.json")
        if os.path.exists(ajson):
            ids.append(sid)
    return ids

def robust_find_mesh(scan_root: str, sid: str):
    # 1) <scan_root>/data/<sid>/scans/**
    base0 = os.path.join(scan_root, "data", sid, "scans")
    for name in CANDIDATE_MESH_NAMES:
        hits = glob.glob(os.path.join(base0, "**", name), recursive=True)
        if hits: return hits[0]
    # 2) <scan_root>/<sid>/**
    base1 = os.path.join(scan_root, sid)
    for name in CANDIDATE_MESH_NAMES:
        hits = glob.glob(os.path.join(base1, "**", name), recursive=True)
        if hits: return hits[0]
    # 3) 兜底：全库递归搜（慢，但稳），用 scene_id 过滤
    for name in CANDIDATE_MESH_NAMES:
        for h in glob.glob(os.path.join(scan_root, "**", name), recursive=True):
            if sid in h:
                return h
    return None

# =============== “配对体检” ===============
def precheck(artic_root: str, scan_root: str, sample_n: int = 8) -> int:
    parts = sorted(glob.glob(os.path.join(artic_root, "*_parts.json")))
    total = len(parts)
    if total == 0:
        print(f"[ERR] 没找到 *_parts.json 于 {artic_root}")
        return 1

    ok = 0
    missing_artic, missing_mesh = [], []
    examples = []

    for p in parts:
        sid = os.path.basename(p).replace("_parts.json", "")
        ajson = os.path.join(artic_root, f"{sid}_artic.json")
        if not os.path.exists(ajson):
            missing_artic.append(sid); continue

        mesh = robust_find_mesh(scan_root, sid)
        if not mesh:
            missing_mesh.append(sid); continue

        # 轻量读取
        try:
            with open(p, "r") as f: parts_json = json.load(f)
            with open(ajson, "r") as f: artic_json = json.load(f)
        except Exception as e:
            print(f"[WARN] {sid} JSON 解析失败: {e}")
            continue

        # PLY 头部统计（可选）
        v_cnt = f_cnt = None
        if mesh.lower().endswith(".ply"):
            try:
                with open(mesh, "rb") as f:
                    for _ in range(200):
                        line = f.readline()
                        if not line: break
                        s = line.decode("latin1", errors="ignore").strip()
                        if s == "end_header": break
                        m = re.match(r"element\s+vertex\s+(\d+)", s)
                        if m: v_cnt = int(m.group(1))
                        m = re.match(r"element\s+face\s+(\d+)", s)
                        if m: f_cnt = int(m.group(1))
            except Exception as e:
                print(f"[WARN] {sid} 读取 PLY header 失败: {e}")

        ok += 1
        if len(examples) < sample_n:
            examples.append((sid, mesh, v_cnt, f_cnt,
                             list(parts_json)[:5] if isinstance(parts_json, dict) else [],
                             list(artic_json)[:5] if isinstance(artic_json, dict) else []))

    print("\n=== 配对体检汇总 ===")
    print(f"Articulate3D 标注场景数: {total}")
    print(f"配对成功（有 *_artic.json 且找到网格）: {ok}")
    print(f"缺少 *_artic.json: {len(missing_artic)}")
    print(f"缺少网格(mesh): {len(missing_mesh)}")

    if missing_artic:
        print(f"\n缺少 *_artic.json 的 scene_id（前 20 个）：")
        print(", ".join(missing_artic[:20]))
    if missing_mesh:
        print(f"\n在 {scan_root} 未找到网格的 scene_id（前 20 个）：")
        print(", ".join(missing_mesh[:20]))

    print("\n=== 随机样例（最多 8 个）===")
    for sid, mesh, v_cnt, f_cnt, p_keys, a_keys in examples:
        print(f"- {sid}")
        print(f"  mesh: {mesh}")
        if v_cnt is not None:
            print(f"  PLY 顶点数: {v_cnt}, 面数: {f_cnt}")
        print(f"  parts.json 顶层键: {p_keys}")
        print(f"  artic.json 顶层键: {a_keys}")

    return 0

# =============== DataLoader 容错：遇到错误直接跳过该样本 ===============
class SkipErrorDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except Exception as e:
            # 例如：No mesh found / _read_artic_json 解析失败等
            print(f"[skip] {e}")
            return None

def drop_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:  # 整个 batch 都被过滤
        return None
    return default_collate(batch)

# =============== 杂项工具 ===============
def seed_all(seed=0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj

def build_model():
    Model = getattr(km, "HierarchicalArticulateKinematicModel", None) \
         or getattr(km, "PointNetKinematics", None)
    assert Model is not None, "在 kinetic_model.py 里找不到模型类"
    model = Model()
    Loss = getattr(km, "HierarchicalKinematicLoss", None)
    if Loss is None:
        raise RuntimeError("找不到 HierarchicalKinematicLoss，请在 kinetic_model.py 中实现/导入")
    return model, Loss()

def construct_dataset(DS, args, split):
    """根据数据集 __init__ 的真实签名拼参数（避免 unexpected kw）"""
    sig = inspect.signature(DS.__init__)
    params = set(sig.parameters.keys())

    artic_root = os.path.expanduser(args.artic_root)
    scan_root  = os.path.expanduser(args.scan_root)
    kwargs = {}

    if "split" in params:      kwargs["split"] = split
    if "augment" in params:    kwargs["augment"] = (split == "train")
    if "normalize" in params:  kwargs["normalize"] = True

    # 点数参数别名
    if   "points"     in params: kwargs["points"]     = args.points
    elif "num_points" in params: kwargs["num_points"] = args.points
    elif "npoints"    in params: kwargs["npoints"]    = args.points
    elif "n_points"   in params: kwargs["n_points"]   = args.points

    # 标注根目录
    if   "artic_root" in params:   kwargs["artic_root"] = artic_root
    elif "dataset_dir" in params:  kwargs["dataset_dir"] = artic_root
    elif "root" in params:         kwargs["root"]        = artic_root

    # 扫描/网格目录
    if   "scan_root" in params: kwargs["scan_root"] = scan_root
    elif "mesh_dir"  in params: kwargs["mesh_dir"]  = scan_root
    elif "mesh_root" in params: kwargs["mesh_root"] = scan_root

    print(f"[dataset] using {DS.__name__} signature: {sig}")
    print(f"[dataset] kwargs -> {kwargs}")
    return DS(**kwargs)

@torch.no_grad()
def evaluate(model, loader, device, max_batches=10):
    model.eval()
    tot = 0.0; n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        if batch is None:    continue
        x = batch["points"].to(device)
        pred = model(x)
        tgt  = to_device(batch.get("targets", {}), device)
        loss_fn = getattr(model, "loss_fn", None)
        if loss_fn is None:
            from pointnet.kinetic_model import HierarchicalKinematicLoss
            loss_fn = HierarchicalKinematicLoss()
        losses = loss_fn(pred, tgt)
        tot += float(losses["total"]); n += 1
    model.train()
    return tot / max(1, n)

def main(a):
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 只做配对体检 ----
    if a.precheck:
        raise SystemExit(precheck(os.path.expanduser(a.artic_root),
                                  os.path.expanduser(a.scan_root),
                                  a.check_sample))

    # ---- 构造数据集 / DataLoader（带容错跳过）----
    DS = Articulate3DDataset
    train_ds = construct_dataset(DS, a, split="train")
    val_ds   = construct_dataset(DS, a, split="val")

    train_loader = DataLoader(
        SkipErrorDataset(train_ds), batch_size=a.batch, shuffle=True,
        num_workers=a.workers, pin_memory=True, drop_last=False,
        collate_fn=drop_none_collate, persistent_workers=False
    )
    val_loader = DataLoader(
        SkipErrorDataset(val_ds), batch_size=a.batch, shuffle=False,
        num_workers=a.workers, pin_memory=True, drop_last=False,
        collate_fn=drop_none_collate, persistent_workers=False
    )

    # ---- 模型 / 损失 / 优化器 / 调度 ----
    model, loss_fn = build_model()
    model.loss_fn = loss_fn
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=a.epochs)

    # AMP 新 API
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    os.makedirs(a.out_dir, exist_ok=True)
    best = float("inf"); start_epoch = 0

    # ---- 断点续训 ----
    if a.resume and os.path.isfile(a.resume):
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best", best)
        print(f"[resume] loaded {a.resume} @epoch {start_epoch-1}")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0 = time.time(); running = 0.0; trained_steps = 0

        for step, batch in enumerate(train_loader, 1):
            if batch is None:  # 整个 batch 都被跳过
                continue
            x   = batch["points"].to(device)
            tgt = to_device(batch.get("targets", {}), device)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=a.amp):
                pred   = model(x)
                losses = loss_fn(pred, tgt)
                loss   = losses["total"]

            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(optim)
            scaler.update()
            trained_steps += 1

            running += float(loss)
            if step % a.log_every == 0:
                lr = optim.param_groups[0]['lr']
                avg = running / a.log_every
                print(f"[{epoch+1:03d}/{a.epochs:03d}] step {step:04d}/{len(train_loader):04d} "
                      f"loss={avg:.4f} lr={lr:.2e}")
                running = 0.0

        # 若这一轮一个 batch 都没训练到，直接评估/保存，以免 ZeroDivision
        if trained_steps == 0:
            print(f"[warn] epoch {epoch+1}: 本轮没有可用 batch（样本都被跳过或找不到 mesh）")

        # PyTorch >=1.1 建议：optimizer.step() 之后再 scheduler.step()
        scheduler.step()

        val_loss = evaluate(model, val_loader, device, max_batches=a.val_batches)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}  (time {dt:.1f}s)")

        # 保存 ckpt
        ckpt_path = os.path.join(a.out_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": scheduler.state_dict(),
            "best": best,
        }, ckpt_path)

        if val_loss < best:
            best = val_loss
            best_path = os.path.join(a.out_dir, "best.pth")
            try:
                import shutil; shutil.copy2(ckpt_path, best_path)
            except Exception:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": scheduler.state_dict(),
                    "best": best,
                }, best_path)
            print(f"[best] -> {best_path} (val_loss {best:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artic_root", required=True, help="Articulate3D 根目录（含 *_parts.json 与 *_artic.json）")
    ap.add_argument("--scan_root",  required=True, help="ScanNet++ 根目录（含 data/<scene>/scans/...mesh）")
    ap.add_argument("--out_dir",    default="./kin_ckpts")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--points", type=int, default=2048)   # 注意：数据集参数名是 n_points，这里做了别名映射
    ap.add_argument("--workers",type=int, default=4)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--wd",     type=float, default=1e-4)
    ap.add_argument("--clip",   type=float, default=1.0, help="gradient clipping max-norm，0 关闭")
    ap.add_argument("--amp",    action="store_true", help="启用 AMP 混合精度（推荐）")
    ap.add_argument("--resume", type=str, default="", help="ckpt 路径，断点续训")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--val_batches", type=int, default=10, help="验证最多评估多少个 batch（加速）")
    ap.add_argument("--precheck", action="store_true", help="只做配对体检，不训练")
    ap.add_argument("--check_sample", type=int, default=8, help="体检时最多打印的样例数")
    main(ap.parse_args())
