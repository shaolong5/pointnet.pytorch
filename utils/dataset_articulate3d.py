"""
Articulate3D Dataset with Enhanced Parsing
==========================================
Fixed version that properly extracts articulation data from JSONs.
"""
from __future__ import annotations
import os, glob, json, re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

# ============================= 类别定义 =============================
BIG_CLASSES = {"none":0, "door":1, "window":2, "drawer":3, "lid":4, "cabinet":5}
SMALL_CLASSES = {"none":0, "handle":1, "switch":2, "control":3, "knob":4, "button":5, "hook":6}
AUX_ROLES = {"none":0, "base":1, "frame":2, "hinge":3}

_DEFAULT_MESH_GLOB = "scans/mesh_aligned_*.ply"

CANON_FIX = {
    "cabnet.": "cabinet.",
    "hande base": "handle base",
    "hinger": "hinge",
    "doorframe": "door frame",
    "box lid.": "lid.",
}

# ============================= 工具函数 =============================
def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 支持多种JSON格式
    if isinstance(raw, dict):
        if "data" in raw:
            return raw["data"]
        elif "parts" in raw or "articulations" in raw:
            return raw
    return raw

def canon_label(s: str) -> str:
    s = str(s or "").strip()
    low = s.lower()
    for bad, good in CANON_FIX.items():
        if low.startswith(bad):
            return good + s[len(bad):]
    return s

# ============================= JSON 解析 (增强版) =============================
@dataclass
class Articulation:
    type: int
    axis: np.ndarray
    origin: np.ndarray
    range: np.ndarray
    base: List[str]

_JT_MAP_STR = {
    "fixed":0, "none":0, "static":0, "0":0,
    "rotation":1, "revolute":1, "hinge":1, "rot":1, "1":1, "rotating":1,
    "translation":2, "prismatic":2, "slide":2, "lin":2, "2":2, "sliding":2,
}

def map_joint_type(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, str):
        return _JT_MAP_STR.get(val.lower(), 0)
    try:
        t = int(val)
        return max(0, min(2, t))  # Clamp to [0,2]
    except:
        return 0

def read_parts_json(parts_path: str) -> Dict[str, Dict[str, Any]]:
    """Enhanced parsing for parts JSON"""
    try:
        data = load_json(parts_path)
    except Exception as e:
        print(f"[Warning] Failed to load {parts_path}: {e}")
        return {}
    
    parts = {}
    
    # Try to find parts in various locations
    parts_list = None
    if isinstance(data, dict):
        parts_list = data.get("parts", data.get("part", data.get("segments", [])))
    elif isinstance(data, list):
        parts_list = data
    
    if not parts_list:
        # Try to extract from other structures
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    if isinstance(data[key][0], dict) and any(k in data[key][0] for k in ["pid", "partId", "id"]):
                        parts_list = data[key]
                        break
    
    if parts_list:
        for p in parts_list:
            if not isinstance(p, dict):
                continue
            
            # Try multiple key variations for part ID
            pid = p.get("pid") or p.get("partId") or p.get("id") or p.get("part_id") or p.get("segment_id")
            if pid is None:
                continue
            pid = str(pid)
            
            # Get label and group with fallbacks
            label = p.get("label") or p.get("name") or p.get("category") or ""
            group = p.get("group") or p.get("type") or p.get("class") or ""
            
            # Get triangle indices with many fallbacks
            tri = (p.get("triIndices") or p.get("tri_indices") or 
                   p.get("face_indices") or p.get("faces") or 
                   p.get("triangles") or p.get("tris") or 
                   p.get("face_ids") or p.get("faceIndices") or [])
            
            parts[pid] = {
                "label": str(label),
                "group": str(group),
                "tri": tri
            }
    
    # Additional label extraction if available
    if isinstance(data, dict) and "labels" in data:
        for pid, entry in parts.items():
            if not entry.get("label"):
                entry["label"] = str(data["labels"].get(pid, ""))
    
    return parts

def read_artic_json(artic_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Articulation]]:
    """Enhanced parsing for articulation JSON"""
    try:
        data = load_json(artic_path)
    except Exception as e:
        print(f"[Warning] Failed to load {artic_path}: {e}")
        return {}, {}
    
    parts_meta = {}
    arts = {}
    
    # Parse parts metadata
    if isinstance(data, dict) and "parts" in data:
        for p in data["parts"]:
            if not isinstance(p, dict):
                continue
            pid = p.get("pid") or p.get("partId") or p.get("id") or p.get("part_id")
            if pid is None:
                continue
            pid = str(pid)
            parts_meta[pid] = {
                "label": str(p.get("label") or p.get("name") or ""),
                "group": str(p.get("group") or p.get("type") or "")
            }
    
    # Parse articulations - try multiple locations
    artic_list = None
    if isinstance(data, dict):
        artic_list = (data.get("articulations") or 
                      data.get("articulation") or 
                      data.get("joints") or 
                      data.get("joint") or [])
    
    if artic_list:
        for a in artic_list:
            if not isinstance(a, dict):
                continue
            
            pid = a.get("pid") or a.get("partId") or a.get("id") or a.get("part_id")
            if pid is None:
                continue
            pid = str(pid)
            
            # Parse joint type
            joint_type = a.get("type") or a.get("jointType") or a.get("joint_type")
            t = map_joint_type(joint_type)
            
            # Parse axis with fallbacks
            axis_raw = (a.get("axis") or a.get("jointAxis") or 
                       a.get("joint_axis") or a.get("direction") or [0,0,1])
            axis = np.asarray(axis_raw, dtype=np.float32).reshape(3)
            
            # Parse origin
            origin_raw = (a.get("origin") or a.get("jointOrigin") or 
                         a.get("joint_origin") or a.get("pivot") or 
                         a.get("center") or [0,0,0])
            origin = np.asarray(origin_raw, dtype=np.float32).reshape(3)
            
            # Parse range
            rmin = float(a.get("rangeMin") or a.get("min") or a.get("range_min") or 
                        a.get("limit_min") or a.get("lower_limit", 0.0))
            rmax = float(a.get("rangeMax") or a.get("max") or a.get("range_max") or 
                        a.get("limit_max") or a.get("upper_limit", 0.0))
            
            # Handle range as array
            if "range" in a and isinstance(a["range"], (list, tuple)) and len(a["range"]) >= 2:
                rmin, rmax = float(a["range"][0]), float(a["range"][1])
            
            # Parse base/parent
            base_raw = (a.get("base") or a.get("parent") or 
                       a.get("parent_id") or a.get("base_part") or [])
            if isinstance(base_raw, (int, float)):
                base = [str(int(base_raw))]
            elif isinstance(base_raw, str):
                base = [base_raw]
            else:
                base = [str(x) for x in (base_raw or [])]
            
            arts[pid] = Articulation(
                type=t, 
                axis=axis, 
                origin=origin, 
                range=np.asarray([rmin, rmax], np.float32), 
                base=base
            )
    
    # Debug output for first few files
    if len(arts) > 0 and np.random.random() < 0.1:  # 10% chance to debug
        print(f"[Debug] Found {len(arts)} articulations in {os.path.basename(artic_path)}")
        for pid, art in list(arts.items())[:2]:
            print(f"  Part {pid}: type={art.type}, range={art.range}")
    
    return parts_meta, arts

# ============================= 类别判定 (更宽松) =============================
def big_head_from(label: str, group: str) -> int:
    """Enhanced big head detection with more patterns"""
    l = canon_label(label).strip().lower()
    g = (group or "").strip().lower()
    
    # Check group-based patterns first
    if "door" in g or "window" in g:
        if "door" in l and "frame" not in l:
            return BIG_CLASSES["door"]
        if "window" in l and "frame" not in l:
            return BIG_CLASSES["window"]
    
    # Direct label patterns
    if any(x in l for x in ["door.", "door ", "door_"]) and "frame" not in l:
        return BIG_CLASSES["door"]
    if any(x in l for x in ["window.", "window ", "window_"]) and "frame" not in l:
        return BIG_CLASSES["window"]
    if any(x in l for x in ["drawer.", "drawer ", "drawer_"]):
        return BIG_CLASSES["drawer"]
    if any(x in l for x in ["lid.", "lid ", "lid_", "cover.", "cover "]):
        return BIG_CLASSES["lid"]
    if any(x in l for x in ["cabinet.", "cabinet ", "cupboard", "wardrobe"]):
        return BIG_CLASSES["cabinet"]
    
    # Fallback patterns
    if "cabinet" in g and "drawer" in l:
        return BIG_CLASSES["drawer"]
    if g == "lids" or "lid" in g:
        return BIG_CLASSES["lid"]
    
    return BIG_CLASSES["none"]

def small_head_from(label: str, group: str) -> int:
    """Enhanced small head detection"""
    l = canon_label(label).strip().lower()
    g = (group or "").strip().lower()
    
    # Direct patterns
    if any(x in l for x in ["handle.", "handle ", "handle_", "doorhandle", "door_handle"]):
        return SMALL_CLASSES["handle"]
    if any(x in l for x in ["switch.", "switch ", "light_switch", "lightswitch"]):
        return SMALL_CLASSES["switch"]
    if any(x in l for x in ["control.", "control ", "controller"]):
        return SMALL_CLASSES["control"]
    if any(x in l for x in ["knob.", "knob ", "doorknob", "door_knob"]):
        return SMALL_CLASSES["knob"]
    if any(x in l for x in ["button.", "button ", "push_button"]):
        return SMALL_CLASSES["button"]
    if any(x in l for x in ["hook.", "hook ", "coat_hook"]):
        return SMALL_CLASSES["hook"]
    
    # Group-based
    if "switch" in g and "light" in l:
        return SMALL_CLASSES["switch"]
    if "handle" in g:
        return SMALL_CLASSES["handle"]
    
    return SMALL_CLASSES["none"]

def aux_role_from(label: str, group: str) -> int:
    """Auxiliary role detection"""
    l = canon_label(label).strip().lower()
    g = (group or "").strip().lower()
    
    if any(x in l for x in [" base", "_base", "base_", "baseplate"]):
        return AUX_ROLES["base"]
    if "frame" in l and ("door" in g or "window" in g):
        return AUX_ROLES["frame"]
    if any(x in l for x in ["hinge", "pivot", "joint"]):
        return AUX_ROLES["hinge"]
    
    return AUX_ROLES["none"]

# ============================= Mesh 查找 =============================
def find_mesh(scan_root: str, sid: str, mesh_glob: Optional[str]) -> Optional[str]:
    scan_root = _expand(scan_root)
    
    if mesh_glob:
        pat = os.path.join(scan_root, sid, mesh_glob)
        cands = glob.glob(pat, recursive=True)
        if cands:
            filtered = [c for c in cands if not ('semantic' in c or 'mask' in c)]
            if filtered:
                filtered.sort(key=lambda x: (len(x), x))
                return filtered[0]
    
    patterns = [
        "scans/mesh_aligned_0.05.ply",
        "scans/mesh_aligned_0.01.ply",
        "scans/mesh_aligned_*.ply",
        "scans/mesh.ply",
        "mesh_aligned_*.ply",
        "**/mesh*.ply",
    ]
    
    for pattern in patterns:
        pat = os.path.join(scan_root, sid, pattern)
        cands = glob.glob(pat, recursive=True)
        if cands:
            filtered = [c for c in cands 
                       if not any(x in os.path.basename(c) 
                                 for x in ['semantic', 'mask', '_labels'])]
            if filtered:
                def get_resolution(path):
                    match = re.search(r'mesh_aligned_(\d+\.?\d*)', path)
                    if match:
                        return float(match.group(1))
                    return 0.05
                
                filtered.sort(key=lambda x: (abs(get_resolution(x) - 0.05), len(x), x))
                return filtered[0]
    
    return None

def build_face_to_part(mesh: trimesh.Trimesh, parts_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str,int], Dict[int,str]]:
    n_faces = int(getattr(mesh, "faces", np.empty((0,3))).shape[0])
    f2p = np.full((n_faces,), -1, dtype=np.int64)
    pid2i = {}
    i2pid = {}
    
    for i, pid in enumerate(sorted(parts_dict.keys())):
        pid2i[pid] = i
        i2pid[i] = pid
    
    def _extract_indices(entry):
        for k in ["tri", "triIndices", "tri_indices", "face_indices", "faces", "triangles"]:
            if k in entry and entry[k] is not None:
                try:
                    arr = np.asarray(entry[k], dtype=np.int64).reshape(-1)
                    return arr
                except:
                    pass
        return None
    
    for pid, entry in parts_dict.items():
        arr = _extract_indices(entry)
        if arr is None:
            continue
        arr = arr[(arr >= 0) & (arr < n_faces)]
        if len(arr) > 0:
            f2p[arr] = pid2i[pid]
    
    return f2p, pid2i, i2pid

# ============================= 主数据集类 =============================
class Articulate3DDataset(Dataset):
    def __init__(self,
                 artic_root: str,
                 scan_root: str,
                 split: str = "all",
                 n_points: int = 2048,
                 augment: bool = False,
                 normalize: bool = True,
                 mesh_glob: Optional[str] = None,
                 range_epsilon: float = 0.01,
                 labeled_face_ratio: float = 0.0):  # 添加这个参数以兼容
        super().__init__()
        
        # 标签映射
        self.large_label_to_id = {"door":1, "window":2, "drawer":3, "lid":4, "cabinet":5}
        self.large_id_to_label = {0:"background", 1:"door", 2:"window", 3:"drawer", 4:"lid", 5:"cabinet"}
        self.num_large_classes = 5
        
        self.interactive_label_to_id = {"background":0, "handle":1, "switch":2, "control":3, "knob":4, "button":5, "hook":6}
        self.interactive_id_to_label = {0:"background", 1:"handle", 2:"switch", 3:"control", 4:"knob", 5:"button", 6:"hook"}
        self.num_interactive_classes = 7
        
        self.artic_root = _expand(artic_root)
        self.scan_root = _expand(scan_root)
        self.split = split
        self.n_points = int(n_points)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.mesh_glob = mesh_glob
        self.range_eps = float(range_epsilon)
        
        # 收集场景
        self.scene_ids = []
        self.mesh_paths = {}
        
        json_pairs = []
        for p in sorted(glob.glob(os.path.join(self.artic_root, "*_parts.json"))):
            sid = os.path.basename(p)[:-11]
            if os.path.exists(os.path.join(self.artic_root, f"{sid}_artic.json")):
                json_pairs.append(sid)
        
        print(f"[Dataset] Found {len(json_pairs)} scenes with paired JSONs")
        
        for sid in json_pairs:
            mesh_path = find_mesh(self.scan_root, sid, self.mesh_glob)
            if mesh_path is not None:
                self.scene_ids.append(sid)
                self.mesh_paths[sid] = mesh_path
        
        if not self.scene_ids:
            print(f"[ERROR] No usable scenes found!")
            print(f"  - artic_root: {self.artic_root}")
            print(f"  - scan_root: {self.scan_root}")
            raise RuntimeError("No usable scenes found.")
        
        print(f"[Articulate3DDataset] Found {len(self.scene_ids)} usable scenes")
    
    def __len__(self) -> int:
        return len(self.scene_ids)
    
    @staticmethod
    def _rot_z(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=np.float32)
    
    def _maybe_augment(self, pts, axes, origins):
        if self.split != "train" or not self.augment:
            return pts, axes, origins
        R = torch.from_numpy(self._rot_z(float(np.random.uniform(0, 2*np.pi)))).float()
        pts = pts @ R.T
        axes = torch.nn.functional.normalize(axes @ R.T, dim=-1, eps=1e-8)
        origins = origins @ R.T
        pts = pts + torch.randn_like(pts) * 0.01
        s = float(np.random.uniform(0.8, 1.2))
        pts = pts * s
        origins = origins * s
        return pts, axes, origins
    
    def _maybe_normalize(self, pts, origins):
        if not self.normalize:
            return pts, origins, {}
        ctr = pts.mean(0)
        pts = pts - ctr
        sc = pts.abs().max().clamp(min=1e-6)
        pts = pts / sc
        origins = (origins - ctr) / sc
        return pts, origins, {"centroid": ctr, "scale": sc}
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sid = self.scene_ids[index]  # 不要循环索引，让PositiveAwareDataset处理重采样
        
        # 文件路径
        mesh_path = self.mesh_paths.get(sid)
        parts_path = os.path.join(self.artic_root, f"{sid}_parts.json")
        artic_path = os.path.join(self.artic_root, f"{sid}_artic.json")
        
        if mesh_path is None:
            raise FileNotFoundError(f"mesh not found for {sid}")
        
        # 解析JSON
        parts_face = read_parts_json(parts_path)
        parts_meta, arts = read_artic_json(artic_path)
        
        # 合并元数据
        for pid, meta in parts_meta.items():
            if pid not in parts_face:
                parts_face[pid] = {"label": meta.get("label",""), "group": meta.get("group",""), "tri": []}
            else:
                if meta.get("label"):
                    parts_face[pid]["label"] = meta["label"]
                if meta.get("group"):
                    parts_face[pid]["group"] = meta["group"]
        
        # 加载mesh
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "dump"):
            mesh = mesh.dump().sum()
        
        f2p, pid2i, i2pid = build_face_to_part(mesh, parts_face)
        n_faces = f2p.shape[0]
        
        if n_faces == 0:
            raise RuntimeError(f"no faces in mesh: {mesh_path}")
        
        # 采样点
        pts_np, face_idx = trimesh.sample.sample_surface(mesh, self.n_points)
        points = torch.from_numpy(pts_np.astype(np.float32))
        pids_int = torch.from_numpy(f2p[np.asarray(face_idx, dtype=np.int64)]).long()
        N = points.shape[0]
        
        # 初始化监督
        joint_type = torch.zeros((N,), dtype=torch.long)
        joint_axis = torch.zeros((N,3), dtype=torch.float32)
        joint_origin = torch.zeros((N,3), dtype=torch.float32)
        joint_range = torch.zeros((N,2), dtype=torch.float32)
        interactive_labels = torch.zeros((N,), dtype=torch.long)  # 使用 interactive 类别而非 binary
        big_head_class = torch.zeros((N,), dtype=torch.long)
        
        # 逐part计算
        def climb_host(pid0: str) -> Tuple[int, Optional[str]]:
            seen = set()
            cur = pid0
            for _ in range(12):
                if cur in seen:
                    break
                seen.add(cur)
                info_c = parts_face.get(cur, {})
                b = big_head_from(info_c.get("label",""), info_c.get("group",""))
                if b != BIG_CLASSES["none"]:
                    return b, cur
                a_c = arts.get(cur)
                if a_c and a_c.base:
                    cur = a_c.base[0]
                    continue
                break
            return BIG_CLASSES["none"], None
        
        # 统计
        has_joint = False
        has_interactive = False
        
        for i_pid in torch.unique(pids_int).tolist():
            if i_pid < 0:
                continue
            pid = i2pid.get(int(i_pid))
            if not pid:
                continue
            
            info = parts_face.get(pid, {})
            label = info.get("label", "")
            group = info.get("group", "")
            
            big = big_head_from(label, group)
            small = small_head_from(label, group)
            
            # 获取关节信息
            a = arts.get(pid)
            jt = 0
            ax = np.zeros(3, np.float32)
            og = np.zeros(3, np.float32)
            rg = np.zeros(2, np.float32)
            
            if a is not None:
                jt = int(a.type)
                ax = a.axis.astype(np.float32)
                og = a.origin.astype(np.float32)
                rg = a.range.astype(np.float32)
                
                # 检查是否真的可交互
                if jt > 0 and abs(float(rg[1] - rg[0])) > self.range_eps:
                    has_joint = True
                    # 如果是可动关节，优先设置为对应的小头类别
                    if small == SMALL_CLASSES["none"]:
                        # 如果没有明确的小头类别，根据大头类别推断
                        if big == BIG_CLASSES["door"] or big == BIG_CLASSES["drawer"]:
                            small = SMALL_CLASSES["handle"]
                        elif big == BIG_CLASSES["lid"]:
                            small = SMALL_CLASSES["knob"]
            
            # 填充点级监督
            m = (pids_int == int(i_pid))
            if m.any():
                if jt > 0:
                    joint_type[m] = jt
                    joint_axis[m] = torch.from_numpy(ax).expand(m.sum(), -1)
                    joint_origin[m] = torch.from_numpy(og).expand(m.sum(), -1)
                    joint_range[m] = torch.from_numpy(rg).expand(m.sum(), -1)
                
                if big > 0:
                    big_head_class[m] = big - 1  # 转换为0-based (去掉none)
                else:
                    big_head_class[m] = -1  # 背景
                
                if small > 0:
                    interactive_labels[m] = small
                    has_interactive = True
        
        # 归一化轴
        joint_axis = torch.nn.functional.normalize(joint_axis, dim=-1, eps=1e-8)
        
        # 数据增强和归一化
        points, joint_axis, joint_origin = self._maybe_augment(points, joint_axis, joint_origin)
        points, joint_origin, norm_info = self._maybe_normalize(points, joint_origin)
        
        # 准备输出 - 使用兼容的键名
        targets = {
            "joint_type": joint_type,           # 保持原名称
            "point_joint_types": joint_type,    # 添加别名
            "joint_axis": joint_axis,           
            "point_joint_axes": joint_axis,     
            "joint_origin": joint_origin,       
            "point_joint_origins": joint_origin,
            "joint_range": joint_range,         
            "point_joint_ranges": joint_range,  
            "interactive_labels": interactive_labels,  # 使用类别而非binary
            "small_head_class": interactive_labels,    # 别名
            "big_head_class": big_head_class,
            "large_part_labels": big_head_class,        # 别名
        }
        
        # Debug输出
        if self.split == "train" and index < 3:
            n_joint = int((joint_type > 0).sum())
            n_inter = int((interactive_labels > 0).sum())
            n_big = int((big_head_class >= 0).sum())
            print(f"[Sample {index}] scene={sid} N={N} | joint={n_joint} inter={n_inter} big={n_big}")
            if has_joint or has_interactive:
                print(f"  -> Found articulated parts! joint_types: {torch.unique(joint_type).tolist()}")
        
        return {
            "scene_id": sid,
            "points": points,
            "point_part_ids": pids_int,
            "targets": targets,
            "mesh_path": mesh_path,
            "norm": norm_info,
        }

# 兼容别名
Articulate3DDeterministic = Articulate3DDataset
dataset_articulate3d = Articulate3DDataset