#!/usr/bin/env python3
"""
visualize_results.py - Complete visualization with articulation parameters and semantic labels
Based on view_text.py structure but enhanced for better display
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

__VERSION__ = "v2.0-complete"

# ===== Style Constants =====
SCENE_BG = "#f3f7ff"      # 3D scene background
PAPER_BG = "#f6f8fc"      # Page background  
LEGEND_FONT = 16
TABLE_FONT = 14
TITLE_FONT = 20
AXIS_WIDTH = 8            # Articulation axis line width
ORIGIN_SIZE = 12          # Origin sphere size
SHOW_CLASS_LABELS = True  # Show class names in 3D
CLASS_LABEL_SIZE = 18

# Semantic color schemes
LARGE_PART_COLORS = {
    0: '#E0E0E0',  # background
    1: '#FF6B6B',  # door (red)
    2: '#4ECDC4',  # window (teal)
    3: '#45B7D1',  # drawer (blue)
    4: '#96CEB4',  # lid (green)
    5: '#DDA15E',  # cabinet (brown)
}

INTERACTIVE_COLORS = {
    0: '#F0F0F0',  # background
    1: '#FF6B35',  # handle (orange)
    2: '#F7931E',  # switch (yellow-orange)  
    3: '#FDC935',  # control (yellow)
    4: '#8B4789',  # knob (purple)
    5: '#E91E63',  # button (pink)
    6: '#00BCD4',  # hook (cyan)
}

JOINT_TYPE_COLORS = {
    0: '#808080',  # fixed (gray)
    1: '#FF0000',  # revolute (red)
    2: '#0000FF',  # prismatic (blue)
}

# Default semantic names
DEFAULT_LARGE_NAMES = ['background', 'door', 'window', 'drawer', 'lid', 'cabinet']
DEFAULT_INTERACTIVE_NAMES = ['background', 'handle', 'switch', 'control', 'knob', 'button', 'hook']
DEFAULT_JOINT_NAMES = ['fixed', 'revolute', 'prismatic']


# ===== Helper Functions =====
def to_numpy(x):
    """Convert tensor to numpy"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_labels(x):
    """Convert logits to labels"""
    x = to_numpy(x)
    if x.ndim == 2:  # logits
        return x.argmax(axis=1).astype(np.int64)
    return x.astype(np.int64)


def load_label_mappings(results_dir):
    """Load label mappings from JSON"""
    path = Path(results_dir) / "label_mappings.json"
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def get_semantic_names(label_mappings, key):
    """Extract semantic names from label mappings"""
    if not label_mappings:
        return None
    
    if key == 'large':
        mapping = label_mappings.get('large_id_to_label', {})
    elif key == 'interactive':
        mapping = label_mappings.get('interactive_id_to_label', {})
    else:
        return None
    
    if not mapping:
        return None
    
    # Convert to list
    max_id = max(int(k) for k in mapping.keys())
    names = ['unknown'] * (max_id + 1)
    for k, v in mapping.items():
        names[int(k)] = v
    return names


def extract_predictions(pred_dict, points):
    """Extract all predictions from the dict"""
    N = len(points)
    results = {}
    
    # Large part segmentation
    for key in ['large_part_logits', 'large_part_seg', 'part_seg']:
        if key in pred_dict:
            results['large_part'] = to_labels(pred_dict[key])
            break
    if 'large_part' not in results:
        results['large_part'] = np.zeros(N, dtype=np.int64)
    
    # Interactive segmentation
    for key in ['interactive_logits', 'interactive_seg', 'interactive_labels']:
        if key in pred_dict:
            results['interactive'] = to_labels(pred_dict[key])
            break
    if 'interactive' not in results:
        results['interactive'] = np.zeros(N, dtype=np.int64)
    
    # Joint types
    for key in ['jt_logits', 'joint_type', 'point_joint_types']:
        if key in pred_dict:
            results['joint_type'] = to_labels(pred_dict[key])
            break
    if 'joint_type' not in results:
        results['joint_type'] = np.zeros(N, dtype=np.int64)
    
    # Articulation parameters
    for key in ['axis', 'joint_axis', 'point_joint_axes']:
        if key in pred_dict:
            results['axis'] = to_numpy(pred_dict[key])
            break
    
    for key in ['origin', 'joint_origin', 'point_joint_origins']:
        if key in pred_dict:
            results['origin'] = to_numpy(pred_dict[key])
            break
    
    for key in ['range', 'joint_range', 'point_joint_ranges']:
        if key in pred_dict:
            results['range'] = to_numpy(pred_dict[key])
            break
    
    return results


def aggregate_instances(points, predictions, min_points=20):
    """Aggregate points into instances based on articulation parameters"""
    N = len(points)
    interactive = predictions.get('interactive', np.zeros(N))
    joint_type = predictions.get('joint_type', np.zeros(N))
    axis = predictions.get('axis')
    origin = predictions.get('origin')
    ranges = predictions.get('range')
    
    # Find interactive points
    inter_mask = interactive > 0
    if not inter_mask.any():
        return []
    
    # Simple clustering by interactive class
    instances = []
    for cls_id in np.unique(interactive[inter_mask]):
        if cls_id == 0:
            continue
        
        mask = (interactive == cls_id)
        if mask.sum() < min_points:
            continue
        
        # Get majority joint type
        jt_vals = joint_type[mask]
        jt_id = np.bincount(jt_vals).argmax() if len(jt_vals) > 0 else 0
        
        # Average axis (normalized)
        inst_axis = None
        if axis is not None and axis.shape[0] == N:
            valid_axes = axis[mask]
            valid_axes = valid_axes[np.all(np.isfinite(valid_axes), axis=1)]
            if len(valid_axes) > 0:
                mean_axis = np.mean(valid_axes, axis=0)
                norm = np.linalg.norm(mean_axis)
                if norm > 1e-6:
                    inst_axis = mean_axis / norm
        
        # Average origin
        inst_origin = None
        if origin is not None and origin.shape[0] == N:
            valid_origins = origin[mask]
            valid_origins = valid_origins[np.all(np.isfinite(valid_origins), axis=1)]
            if len(valid_origins) > 0:
                inst_origin = np.mean(valid_origins, axis=0)
        
        # Average range
        inst_range = None
        if ranges is not None and ranges.shape[0] == N:
            if ranges.ndim == 2 and ranges.shape[1] >= 2:
                valid_ranges = ranges[mask]
                valid_ranges = valid_ranges[np.all(np.isfinite(valid_ranges), axis=1)]
                if len(valid_ranges) > 0:
                    inst_range = np.mean(valid_ranges, axis=0)[:2]
            else:
                valid_ranges = ranges[mask]
                valid_ranges = valid_ranges[np.isfinite(valid_ranges)]
                if len(valid_ranges) > 0:
                    inst_range = [np.mean(valid_ranges)]
        
        instances.append({
            'class_id': int(cls_id),
            'joint_type': int(jt_id),
            'axis': inst_axis.tolist() if inst_axis is not None else None,
            'origin': inst_origin.tolist() if inst_origin is not None else None,
            'range': inst_range.tolist() if inst_range is not None else None,
            'num_points': int(mask.sum())
        })
    
    return instances


def get_axis_name(axis):
    """Convert axis vector to readable name"""
    if axis is None:
        return "?"
    axis = np.asarray(axis)
    abs_axis = np.abs(axis)
    max_idx = np.argmax(abs_axis)
    direction = "+" if axis[max_idx] > 0 else "-"
    return ['X', 'Y', 'Z'][max_idx] + direction


def format_range(range_vals, units='deg'):
    """Format range values for display"""
    if range_vals is None:
        return "N/A"
    
    if len(range_vals) == 1:
        val = range_vals[0]
        if units == 'deg':
            return f"{np.degrees(val):.1f}°"
        return f"{val:.3f}"
    
    min_val, max_val = range_vals[:2]
    if units == 'deg':
        return f"[{np.degrees(min_val):.1f}°, {np.degrees(max_val):.1f}°]"
    return f"[{min_val:.3f}, {max_val:.3f}]"


def add_articulation_axes(fig, points, instances, scene_scale):
    """Add articulation axes and origins to the figure"""
    for idx, inst in enumerate(instances):
        if inst['joint_type'] == 0:  # Skip fixed joints
            continue
        
        axis = inst.get('axis')
        origin = inst.get('origin')
        
        if axis is None or origin is None:
            continue
        
        axis = np.array(axis)
        origin = np.array(origin)
        
        # Draw axis line
        axis_length = scene_scale * 0.15
        start = origin - axis * axis_length / 2
        end = origin + axis * axis_length / 2
        
        color = JOINT_TYPE_COLORS[inst['joint_type']]
        
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=AXIS_WIDTH),
            name=f"Axis {idx}",
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        # Draw origin sphere
        fig.add_trace(go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode='markers+text',
            marker=dict(size=ORIGIN_SIZE, color='yellow', 
                       line=dict(color='black', width=2)),
            text=[f"Inst {idx}"],
            textposition='top center',
            name=f"Origin {idx}",
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)


def create_visualization(data, label_mappings=None, mode='interactive', units='deg'):
    """Create complete visualization with articulation info"""
    
    points = to_numpy(data['points'])
    pred = data.get('pred', {})
    scene_id = data.get('scene_id', 'unknown')
    
    # Extract predictions
    predictions = extract_predictions(pred, points)
    
    # Get semantic names
    large_names = get_semantic_names(label_mappings, 'large') or DEFAULT_LARGE_NAMES
    interactive_names = get_semantic_names(label_mappings, 'interactive') or DEFAULT_INTERACTIVE_NAMES
    joint_names = DEFAULT_JOINT_NAMES
    
    # Aggregate instances
    instances = aggregate_instances(points, predictions)
    
    # Create figure with 3D view and table
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'table'}]],
        column_widths=[0.65, 0.35],
        subplot_titles=('3D View', 'Articulation Parameters')
    )
    
    # Determine which labels to show based on mode
    if mode == 'interactive':
        labels = predictions['interactive']
        names = interactive_names
        colors_map = INTERACTIVE_COLORS
    elif mode == 'large_part':
        labels = predictions['large_part']
        names = large_names
        colors_map = LARGE_PART_COLORS
    else:  # joint_type
        labels = predictions['joint_type']
        names = joint_names
        colors_map = JOINT_TYPE_COLORS
    
    # Add point cloud colored by selected mode
    for label_id in np.unique(labels):
        mask = labels == label_id
        label_name = names[label_id] if label_id < len(names) else f"Class_{label_id}"
        color = colors_map.get(label_id, '#808080')
        
        fig.add_trace(go.Scatter3d(
            x=points[mask, 0],
            y=points[mask, 1],
            z=points[mask, 2],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.8),
            name=label_name,
            text=[label_name] * mask.sum(),
            hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Add class label text in 3D
        if SHOW_CLASS_LABELS and mask.sum() > 10:
            center = points[mask].mean(axis=0)
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='text',
                text=[label_name],
                textfont=dict(size=CLASS_LABEL_SIZE, color='black'),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
    
    # Add articulation axes
    scene_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    add_articulation_axes(fig, points, instances, scene_scale)
    
    # Create table data
    if instances:
        table_data = {
            'ID': [],
            'Class': [],
            'Joint': [],
            'Axis': [],
            'Origin': [],
            'Range': [],
            'Points': []
        }
        
        for idx, inst in enumerate(instances):
            table_data['ID'].append(idx)
            table_data['Class'].append(interactive_names[inst['class_id']] 
                                      if inst['class_id'] < len(interactive_names) 
                                      else f"Class_{inst['class_id']}")
            table_data['Joint'].append(joint_names[inst['joint_type']])
            table_data['Axis'].append(f"{get_axis_name(inst['axis'])}")
            
            # Format origin
            if inst['origin'] is not None:
                origin_str = f"[{inst['origin'][0]:.2f}, {inst['origin'][1]:.2f}, {inst['origin'][2]:.2f}]"
            else:
                origin_str = "N/A"
            table_data['Origin'].append(origin_str)
            
            table_data['Range'].append(format_range(inst['range'], units))
            table_data['Points'].append(inst['num_points'])
    else:
        table_data = {
            'ID': ['-'],
            'Class': ['No instances'],
            'Joint': ['-'],
            'Axis': ['-'],
            'Origin': ['-'],
            'Range': ['-'],
            'Points': ['0']
        }
    
    # Add table
    fig.add_trace(go.Table(
        header=dict(
            values=list(table_data.keys()),
            fill_color='#eef2ff',
            align='left',
            font=dict(size=TABLE_FONT + 1, color='#1f2937'),
            height=35
        ),
        cells=dict(
            values=list(table_data.values()),
            fill_color='white',
            align='left',
            font=dict(size=TABLE_FONT),
            height=30,
            line_color='#e5e7eb'
        )
    ), row=1, col=2)
    
    # Update 3D scene
    fig.update_scenes(
        aspectmode='data',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor=SCENE_BG
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Scene: {scene_id}</b> | Mode: {mode}",
            font=dict(size=TITLE_FONT)
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=0.65,
            font=dict(size=LEGEND_FONT)
        ),
        height=800,
        paper_bgcolor=PAPER_BG,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Articulate3D Complete Visualization")
    parser.add_argument("--results_dir", required=True, help="Directory with inference results")
    parser.add_argument("--out_dir", required=True, help="Output directory for HTML files")
    parser.add_argument("--color_by", choices=["interactive", "large_part", "joint_type"], 
                       default="interactive", help="Coloring mode")
    parser.add_argument("--units", choices=["deg", "rad"], default="deg", 
                       help="Units for range display")
    parser.add_argument("--min_points", type=int, default=20,
                       help="Minimum points for instance")
    parser.add_argument("--batch", action="store_true", help="Process all scenes")
    parser.add_argument("--scene_id", help="Process specific scene")
    
    args = parser.parse_args()
    
    print(f"[visualize_results {__VERSION__}] Starting...")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label mappings
    label_mappings = load_label_mappings(args.results_dir)
    if label_mappings:
        print("✅ Loaded label mappings")
    else:
        print("⚠️  No label mappings found, using defaults")
    
    # Find prediction files
    pred_dir = Path(args.results_dir) / "preds"
    if not pred_dir.exists():
        # Try alternative location
        pred_dir = Path(args.results_dir)
    
    pred_files = sorted(pred_dir.glob("*.pt"))
    if not pred_files:
        print(f"❌ No .pt files found in {pred_dir}")
        return
    
    # Filter by scene_id if specified
    if args.scene_id and not args.batch:
        pred_files = [f for f in pred_files if f.stem == args.scene_id]
        if not pred_files:
            print(f"❌ Scene {args.scene_id} not found")
            return
    
    print(f"📊 Processing {len(pred_files)} scenes...")
    
    # Process each file
    for i, pred_file in enumerate(pred_files, 1):
        try:
            print(f"[{i}/{len(pred_files)}] Processing {pred_file.stem}...", end=" ")
            
            # Load data
            data = torch.load(pred_file, map_location='cpu')
            
            # Create visualization
            fig = create_visualization(data, label_mappings, 
                                     mode=args.color_by, units=args.units)
            
            # Save HTML
            output_path = out_dir / f"{pred_file.stem}.html"
            pio.write_html(fig, file=str(output_path), 
                          auto_open=False, include_plotlyjs='cdn')
            
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Create index page
    create_index_html(out_dir)
    
    print(f"\n✅ Complete! Results saved to {out_dir}")
    print(f"   Open {out_dir}/index.html in a browser to view all scenes")


def create_index_html(out_dir):
    """Create an index page for all visualizations"""
    html_files = sorted(out_dir.glob("*.html"))
    html_files = [f for f in html_files if f.name != 'index.html']
    
    if not html_files:
        return
    
    index_content = """<!DOCTYPE html>
<html>
<head>
    <title>Articulate3D Visualization Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #1a202c;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        .subtitle {
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.2rem;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f7fafc;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4a5568;
        }
        .stat-label {
            color: #a0aec0;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .filter-container {
            margin-bottom: 25px;
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
        }
        .search-box {
            padding: 10px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            width: 300px;
            transition: border-color 0.2s;
        }
        .search-box:focus {
            outline: none;
            border-color: #667eea;
        }
        .view-mode {
            display: flex;
            gap: 10px;
        }
        .mode-btn {
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .mode-btn:hover {
            border-color: #667eea;
            background: #f7fafc;
        }
        .mode-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .scene-card {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .scene-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: left 0.3s;
        }
        .scene-card:hover::before {
            left: 0;
        }
        .scene-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        .scene-icon {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        .scene-name {
            font-weight: 600;
            color: #2d3748;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }
        .scene-info {
            color: #a0aec0;
            font-size: 0.9rem;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ Articulate3D Visualization</h1>
        <div class="subtitle">Complete Results with Articulation Parameters</div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">""" + str(len(html_files)) + """</div>
                <div class="stat-label">Total Scenes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">View Modes</div>
            </div>
        </div>
        
        <div class="filter-container">
            <input type="text" class="search-box" id="searchInput" 
                   placeholder="🔍 Search scenes..." onkeyup="filterScenes()">
        </div>
        
        <div class="grid" id="sceneGrid">
"""
    
    for html_file in html_files:
        scene_id = html_file.stem
        index_content += f"""
            <a href="{html_file.name}" class="scene-card" data-scene="{scene_id.lower()}">
                <div class="scene-icon">📊</div>
                <div class="scene-name">{scene_id}</div>
                <div class="scene-info">Click to view details</div>
            </a>
"""
    
    index_content += """
        </div>
    </div>
    
    <script>
        function filterScenes() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const cards = document.querySelectorAll('.scene-card');
            
            cards.forEach(card => {
                const sceneName = card.dataset.scene;
                if (sceneName.includes(filter)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
    
    with open(out_dir / 'index.html', 'w') as f:
        f.write(index_content)


if __name__ == "__main__":
    main()