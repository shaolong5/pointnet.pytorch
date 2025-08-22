#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_text.py — Single-canvas Plotly viewer
切换三个视图：interactive_seg / large_part / joint_type
- 每个类别独立 trace → 离散图例
- 交互实例聚合 → 轴线 + 原点 + 右侧实例表
- 一键样式常量，便于整体调风格
"""

import os, glob, json, argparse
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

__VERSION__ = "vstyle-1.5"

# ===== Aesthetics（整体风格一处改） =====
SCENE_BG   = "#f3f7ff"   # 左侧3D背景
PAPER_BG   = "#f6f8fc"   # 页面背景
LEGEND_FONT = 18
TABLE_FONT  = 16
CLASS_TEXT  = True       # 3D 中心是否标出类别名
CLASS_SIZE  = 22
SHOW_SCENE_ID = False  # 关掉场景ID标题


# ---------------- helpers ----------------
def npy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def to_labels(a):
    a = npy(a)
    return a.argmax(1).astype(np.int64) if a.ndim == 2 else a.astype(np.int64)

def load_names_json(path):
    if not path or not os.path.exists(path): return None, None
    try:
        js = json.load(open(path, "r", encoding="utf-8"))
        lp = js.get("large_part", None)
        it = js.get("interactive", None)
        return (list(map(str, lp)) if lp else None,
                list(map(str, it)) if it else None)
    except Exception:
        return None, None

def get_names_from_pred(pred, key, K):
    k = f"{key}_names"
    if isinstance(pred, dict) and k in pred:
        try:
            return list(map(str, pred[k]))
        except Exception:
            pass
    return [f"{key}_{i}" for i in range(int(K))]

def get_joint_type_labels(pred, N):
    jt = None
    if isinstance(pred, dict):
        for k in ["point_joint_types","joint_type","joint_types","jt_logits","joint_type_logits"]:
            if k in pred:
                jt = to_labels(pred[k]); break
    if jt is None or jt.shape[0] != N:
        jt = np.zeros(N, np.int64)
    return jt, {0:"fixed", 1:"revolute", 2:"prismatic"}

def get_axes(pred):
    for k in ["point_joint_axes","joint_axes","axes","axis"]:
        if isinstance(pred, dict) and k in pred: return npy(pred[k])
    return None

def get_origins(pred):
    for k in ["point_joint_origins","joint_origins","origins","origin"]:
        if isinstance(pred, dict) and k in pred: return npy(pred[k])
    return None

def get_ranges(pred):
    for k in ["point_joint_ranges","joint_ranges","ranges","range"]:
        if isinstance(pred, dict) and k in pred: return npy(pred[k])
    return None

def get_mode_labels(pred, pts, mode):
    N = len(pts)
    if mode == "interactive_seg":
        for k in ["interactive_seg","interactive_labels","interactive"]:
            if isinstance(pred, dict) and k in pred:
                lab = to_labels(pred[k]);  return lab if lab.shape[0]==N else np.zeros(N,np.int64)
        return np.zeros(N, np.int64)
    if mode == "large_part":
        for k in ["large_part_seg","part_seg","large_part_labels","parts"]:
            if isinstance(pred, dict) and k in pred:
                lab = to_labels(pred[k]);  return lab if lab.shape[0]==N else np.zeros(N,np.int64)
        return np.zeros(N, np.int64)
    if mode == "joint_type":
        jt,_ = get_joint_type_labels(pred, N);  return jt
    return np.zeros(N, np.int64)

def get_assoc_ids(pred, interactive_mask=None):
    if not isinstance(pred, dict): return None
    for k in ["interactive_association","association","assoc_ids"]:
        if k in pred:
            a = npy(pred[k])
            ids = a.argmax(1).astype(np.int64) if a.ndim==2 else a.astype(np.int64)
            if interactive_mask is not None:
                ids = ids.copy();  ids[~interactive_mask] = -1
            return ids
    return None

def nanmean_unit(v):
    v = np.asarray(v, float);  v = v if v.ndim==2 else v[None,:]
    v = v[np.all(np.isfinite(v),1)]
    if v.size==0: return None
    m = np.nanmean(v,0); n = np.linalg.norm(m)+1e-12
    return None if n<1e-9 else (m/n)

def nanmean_vec(v):
    v = np.asarray(v, float);  v = v if v.ndim==2 else v[None,:]
    v = v[np.all(np.isfinite(v),1)]
    if v.size==0: return None
    return np.nanmean(v,0)

def axis_to_name(a):
    a = np.asarray(a, float); i = int(np.argmax(np.abs(a)))
    return ["x","y","z"][i] + ("+" if a[i]>=0 else "-")

def aggregate_instances(pts, pred, assoc_ids, min_points=20):
    N = len(pts)
    if assoc_ids is None or assoc_ids.shape[0]!=N:
        inter = get_mode_labels(pred, pts, "interactive_seg")
        assoc_ids = np.where(inter>0, 0, -1)

    jt, jtmap = get_joint_type_labels(pred, N)
    axes, origins, ranges = get_axes(pred), get_origins(pred), get_ranges(pred)
    out = []
    for gid in [i for i in np.unique(assoc_ids) if i>=0]:
        m = (assoc_ids==gid)
        if m.sum() < min_points: continue
        v = m.copy()
        if axes    is not None and axes.shape[0]==N:    v &= np.all(np.isfinite(axes),1)
        if origins is not None and origins.shape[0]==N: v &= np.all(np.isfinite(origins),1)
        if ranges  is not None and ranges.shape[0]==N:  v &= (np.all(np.isfinite(ranges),1) if ranges.ndim==2 else np.isfinite(ranges))

        jt_id = int(np.bincount(jt[v], minlength=3).argmax()) if v.any() else None
        axis  = nanmean_unit(axes[v])    if axes    is not None and v.any() else None
        origin= nanmean_vec (origins[v]) if origins is not None and v.any() else None
        rng   = None
        if ranges is not None and v.any():
            if ranges.ndim==2 and ranges.shape[1]>=2:
                r = np.nanmean(ranges[v],0)[:2];  rng=[float(r[0]), float(r[1])]
            else:
                rng = [float(np.nanmean(ranges[v]))]

        out.append(dict(
            id=int(gid), count=int(m.sum()),
            joint_type=jtmap.get(jt_id,"?"),
            axis=None if axis is None else axis.tolist(),
            axis_name="" if axis is None else axis_to_name(axis),
            origin=None if origin is None else origin.tolist(),
            range=rng
        ))
    return out

def fmt_vec3(v): return "" if v is None else "["+", ".join(f"{x:.3f}" for x in v[:3])+"]"
def fmt_range(r, units="deg"):
    if r is None: return ""
    if len(r)==1:
        val=r[0]; return f"{np.degrees(val):.1f}°" if units=="deg" else f"{val:.1f}"
    a,b=r[:2];  return f"[{np.degrees(a):.1f}, {np.degrees(b):.1f}]°" if units=="deg" else f"[{a:.1f}, {b:.1f}]"

def add_axes(fig, pts, inst, row=1, col=1):
    pts = np.asarray(pts, float)
    span = float(np.linalg.norm(pts.max(0)-pts.min(0)))
    scale = 0.3*(span+1e-9)
    idx = []
    for it in inst:
        if it["axis"] is None or it["origin"] is None: continue
        a = np.asarray(it["axis"], float);  o = np.asarray(it["origin"], float)
        p1, p2 = o - a*scale*0.5, o + a*scale*0.5
        # line
        fig.add_trace(go.Scatter3d(x=[p1[0],p2[0]],y=[p1[1],p2[1]],z=[p1[2],p2[2]],
                                   mode="lines", line=dict(width=5, color="#ef4444"),
                                   name=f"axis:{it['id']}", hoverinfo="skip", showlegend=False),
                      row=row, col=col)
        idx.append(len(fig.data)-1)
        # origin + label
        fig.add_trace(go.Scatter3d(x=[o[0]],y=[o[1]],z=[o[2]],
                                   mode="markers+text",
                                   text=[f"id {it['id']}"], textposition="top center",
                                   marker=dict(size=7, color="#10b981"),
                                   name=f"origin:{it['id']}", hoverinfo="skip", showlegend=False),
                      row=row, col=col)
        idx.append(len(fig.data)-1)
        # small axis text near middle
        mid = (p1+p2)/2.0
        fig.add_trace(go.Scatter3d(x=[mid[0]],y=[mid[1]],z=[mid[2]],
                                   mode="text", text=[f"axis {it['axis_name']}"],
                                   textfont=dict(size=12, color="#475569"),
                                   hoverinfo="skip", showlegend=False),
                      row=row, col=col)
        idx.append(len(fig.data)-1)
    return idx

def add_class_traces(fig, pts, labels, names, mode, row=1, col=1,
                     hover=False, visible=True, class_text=CLASS_TEXT, class_size=CLASS_SIZE):

    idx = []
    labels = np.asarray(labels, np.int64)
    for u in [int(u) for u in np.unique(labels)]:
        m = (labels==u)
        if not m.any(): continue
        nm = names[u] if u < len(names) else f"{mode}_{u}"
        fig.add_trace(go.Scatter3d(
            x=pts[m,0], y=pts[m,1], z=pts[m,2],
            mode="markers",
            marker=dict(size=3, opacity=0.95),
            name=nm, showlegend=True,
            text=([nm]*int(m.sum()) if hover else None),
            hoverinfo=("text" if hover else "skip"),
            visible=visible
        ), row=row, col=col)
        idx.append(len(fig.data)-1)
        if class_text:
            c = pts[m].mean(0)
            fig.add_trace(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode="text", text=[nm],
                textfont=dict(size=class_size, color="#111827"),
                name=f"label:{nm}", showlegend=False, visible=visible
            ), row=row, col=col)
            idx.append(len(fig.data)-1)
    return idx

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--color_by", choices=["interactive_seg","large_part","joint_type"], default="interactive_seg")
    ap.add_argument("--show_point_hover", action="store_true")
    ap.add_argument("--units", choices=["deg","rad"], default="deg")
    ap.add_argument("--min_points", type=int, default=20)
    ap.add_argument("--names_json", default="")
    ap.add_argument("--large_part_names", default="")
    ap.add_argument("--interactive_names", default="")
    args = ap.parse_args()

    print(f"[view_text {__VERSION__}] color_by={args.color_by}, out_dir={args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.pt")))
    if not files:
        print(f"[ERR] No .pt in {args.pred_dir}");  return

    json_lp, json_it = load_names_json(args.names_json)

    for f in files:
        d = torch.load(f, map_location="cpu")
        pred = d.get("pred", d if isinstance(d,dict) else {})
        pts  = npy(d["points"]) if "points" in d else None
        if pts is None:
            print("[WARN] no points in", f);  continue
        N = len(pts);  scene = Path(f).stem

        labs_it  = get_mode_labels(pred, pts, "interactive_seg")
        labs_lp  = get_mode_labels(pred, pts, "large_part")
        labs_jt  = get_mode_labels(pred, pts, "joint_type")

        K_it = int(labs_it.max())+1 if labs_it.size else 1
        K_lp = int(labs_lp.max())+1 if labs_lp.size else 1
        names_it = (args.interactive_names.split(",") if args.interactive_names else None) or json_it or get_names_from_pred(pred,"interactive",K_it)
        names_lp = (args.large_part_names.split(",")  if args.large_part_names  else None) or json_lp or get_names_from_pred(pred,"large_part",K_lp)
        _, jt_map = get_joint_type_labels(pred, N)
        names_jt = [jt_map.get(i, f"jt_{i}") for i in range(int(labs_jt.max())+1)]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"scene"}, {"type":"domain"}]],
            column_widths=[0.68, 0.32],
            subplot_titles=("", "Instances")  # 左侧留空标题，全局再写 scene
        )

        # 三组点云（离散图例）
        idx_it = add_class_traces(fig, pts, labs_it, names_it, "interactive",
                                  row=1,col=1, hover=args.show_point_hover, visible=(args.color_by=="interactive_seg"))
        idx_lp = add_class_traces(fig, pts, labs_lp, names_lp, "large_part",
                                  row=1,col=1, hover=args.show_point_hover, visible=(args.color_by=="large_part"))
        idx_jt = add_class_traces(fig, pts, labs_jt, names_jt, "joint_type",
                                  row=1,col=1, hover=args.show_point_hover, visible=(args.color_by=="joint_type"))

        # 实例聚合 + 轴线
        assoc_ids = get_assoc_ids(pred, interactive_mask=(labs_it>0))
        inst      = aggregate_instances(pts, pred, assoc_ids, min_points=args.min_points)
        idx_axes  = add_axes(fig, pts, inst, row=1, col=1)

        # 右侧表格（大字号 + 合理列宽 + 不遮挡）
        if not inst:
            cells = [["-"],["-"],[""],[""],[""],["0"]]
        else:
            cells = [
                [it["id"] for it in inst],
                [it["joint_type"] for it in inst],
                [f"{fmt_vec3(it['axis'])} ({it['axis_name']})" for it in inst],
                [fmt_vec3(it["origin"]) for it in inst],
                [fmt_range(it["range"], units=args.units) for it in inst],
                [it["count"] for it in inst],
            ]
        fig.add_trace(go.Table(
            columnwidth=[0.7, 1.1, 1.8, 1.4, 1.6, 0.8],
            header=dict(values=["id","type","axis(dir)","origin","range","#pts"],
                        align="left", height=34,
                        font=dict(size=TABLE_FONT+1, color="#1f2937"),
                        fill_color="#eef2ff", line_color="#c7d2fe"),
            cells=dict(values=cells, align="left", height=30,
                       font=dict(size=TABLE_FONT),
                       fill_color="white", line_color="#eef2ff")),
            row=1, col=2
        )
        idx_table = [len(fig.data)-1]   # <-- 记录表格 trace，给切换按钮用

        # 3D 场景 & 全局样式
        fig.update_scenes(row=1,col=1, aspectmode="data",
                          xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                          bgcolor=SCENE_BG)

        legend_cfg = dict(orientation="h", y=1.10, yanchor="bottom",
                          x=0.5, xanchor="center", font=dict(size=LEGEND_FONT),
                          bgcolor="rgba(0,0,0,0)", borderwidth=0)
        showlegend_it = (np.unique(labs_it).size > 1)
        showlegend_lp = (np.unique(labs_lp).size > 1)
        showlegend_jt = (np.unique(labs_jt).size > 1)
        init_showlegend = {"interactive_seg":showlegend_it,
                           "large_part":showlegend_lp,
                           "joint_type":showlegend_jt}[args.color_by]

        # 顶部切换按钮（注意：样式放在 updatemenus，而不是每个 button）
        total = len(fig.data)
        mask_it = [False]*total; mask_lp = [False]*total; mask_jt = [False]*total
        for i in idx_it: mask_it[i]=True
        for i in idx_lp: mask_lp[i]=True
        for i in idx_jt: mask_jt[i]=True
        for i in idx_axes + idx_table:
            mask_it[i]=mask_lp[i]=mask_jt[i]=True

        buttons = [
            dict(label="interactive_seg", method="update",
                 args=[{"visible":mask_it}, {"legend":legend_cfg, "showlegend":showlegend_it}]),
            dict(label="large_part",     method="update",
                 args=[{"visible":mask_lp}, {"legend":legend_cfg, "showlegend":showlegend_lp}]),
            dict(label="joint_type",     method="update",
                 args=[{"visible":mask_jt}, {"legend":legend_cfg, "showlegend":showlegend_jt}]),
        ]
        fig.update_layout(updatemenus=[
            dict(type="buttons", direction="left", x=0.02, y=(1.18 if SHOW_SCENE_ID else 1.06),
                 buttons=buttons, bgcolor="#ffffff",
                 bordercolor="#e5e7eb", borderwidth=1,
                 pad=dict(l=6,r=6,t=6,b=6))
        ])

        fig.update_layout(
            title=(scene if SHOW_SCENE_ID else None), title_font_size=18,
            margin=dict(l=10,r=26,t=(80 if SHOW_SCENE_ID else 48),b=10),
            legend=legend_cfg, showlegend=init_showlegend,
            paper_bgcolor=PAPER_BG
        )

        out = os.path.join(args.out_dir, f"{scene}.html")
        pio.write_html(fig, file=out, auto_open=False, include_plotlyjs=True)
        print("saved:", out)

if __name__ == "__main__":
    main()
