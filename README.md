# Articulate3D Motion Prediction & Semantic Segmentation

A comprehensive framework for 3D scene understanding with articulation prediction, combining DINO-based semantic segmentation with motion parameter estimation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Pipeline](#training-pipeline)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a multi-stage training and inference system for:

1. **Semantic Segmentation** - Fine-grained object classification (2564 classes) using DINO-infused USDNet
2. **Movable Part Segmentation** - Identifying parts that can move (rotation/translation)
3. **Articulation Parameter Prediction** - Predicting motion axes, origins, and ranges
4. **Interactable Part Segmentation** - Identifying parts that humans can interact with

### Architecture

- **Backbone**: Res16UNet (MinkowskiEngine sparse convolution)
- **Feature Extraction**: DINOv3 ViT-B/16 for visual features
- **Motion Prediction**: Multi-head architecture (segmentation + origin + axis + range)

## ✨ Features

- ✅ Two-stage training: DINO distillation + semantic segmentation fine-tuning
- ✅ Articulation-aware architecture with motion range prediction
- ✅ Class-balanced loss with frequency-based weighting
- ✅ Coarse-to-fine sampling strategy for handling class imbalance
- ✅ Mixed-precision training with gradient accumulation
- ✅ Comprehensive 3D visualization with motion arrows and semantic labels
- ✅ HTML-based interactive visualization for individual instances

## 🔧 Environment Setup

### Prerequisites

- CUDA 12.6+
- Python 3.10+
- conda/mamba

### Installation

```bash
# Create conda environment
conda create -n usdnet python=3.10
conda activate usdnet

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install MinkowskiEngine
pip install MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl

# Install other dependencies
pip install numpy scipy scikit-learn tqdm pyyaml h5py zarr numcodecs
pip install open3d plotly matplotlib pillow plyfile

# Install DINOv2 (for feature extraction)
pip install torch torchvision
```

### Load CUDA Module (on SLURM cluster)

```bash
module load CUDA/12.6.0
export LD_LIBRARY_PATH=/path/to/.conda/envs/usdnet/lib:$LD_LIBRARY_PATH
```

## 📁 Data Preparation

### 1. Process ScanNet++ Data (Optional - for semantic segmentation)

```bash
cd DINO-Infused-USDNet

# Extract DINO features and prepare Zarr dataset
python batch_ditr_paper_strict_zarr.py \
    --data_root /path/to/scannetpp \
    --output_root ../new_data_elements \
    --device cuda:0 \
    --num_workers 8
```

### 2. Articulate3D Challenge Data

The dataset should be organized as:

```
motion/USDNet/data/processed/
├── articulate3d_challenge_mov/    # Movable parts
│   ├── train/
│   │   ├── {scene_id}_mov.npy
│   │   └── {scene_id}_articulation.h5
│   ├── validation/
│   ├── train_database.yaml
│   └── validation_database.yaml
├── articulate3d_challenge_inter/  # Interactable parts
│   ├── train/
│   │   └── {scene_id}_inter.npy
│   └── ...
```

### 3. JSON Annotations (for motion ranges)

```
Articulate3D/
├── {scene_id}_artic.json  # Contains motion ranges and part labels
└── ...
```

## 🚀 Training Pipeline

### Stage 1: Semantic Segmentation Training (Required)

This stage has **two sub-stages**: DINO pretraining and semantic finetuning.

#### Stage 1.1: DINO Distillation Pre-training

```bash
cd DINO-Infused-USDNet

# Run pretraining (DINO feature distillation only)
python train_usdnet_complete.py \
    --stage pretrain \
    --zarr_root ../new_data_elements \
    --batch_size 8 \
    --max_epochs 100 \
    --learning_rate 1e-3 \
    --save_dir ./checkpointss \
    --device cuda:0
```

**Pretrain Checkpoint:**
```
/mnt/nfs/home/st185933/DINO-Infused-USDNet/checkpointss/pretrain/ckpt_best.pt
```

#### Stage 1.2: Semantic Segmentation Fine-tuning

```bash
# Run finetuning (loads pretrain checkpoint, trains segmentation head)
python train_usdnet_complete.py \
    --stage finetune \
    --zarr_root ../new_data_elements \
    --pretrain_checkpoint ./checkpointss/pretrain/ckpt_best.pt \
    --batch_size 8 \
    --max_epochs 100 \
    --learning_rate 1e-3 \
    --save_dir ./checkpointss \
    --device cuda:0

# Or submit SLURM job
sbatch finetune_job.sh
```

**Finetune Checkpoint:**
```
/mnt/nfs/home/st185933/DINO-Infused-USDNet/checkpointss/finetune/ckpt_best.pt
```

**Key Files:**
- `train_usdnet_complete.py` - Main training script (supports both pretrain and finetune stages)
- `finetune_job.sh` - SLURM job script

**Training Mode Differences:**
| Stage | `--stage` | Segmentation Loss | Distillation Loss | Output |
|-------|-----------|-------------------|-------------------|--------|
| Pretrain | `pretrain` | ❌ Disabled | ✅ Enabled | DINO-aligned backbone |
| Finetune | `finetune` | ✅ Enabled | ✅ (0.5 weight) | Full semantic model |

### Stage 2: Movable Part Segmentation + Motion Prediction

```bash
# Submit training job
sbatch train_articulate3d.sh

# Or run directly
python train_articulate3d.py \
    --data_dir /path/to/articulate3d_challenge_mov \
    --json_dir /path/to/Articulate3D \
    --batch_size 4 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --voxel_size 0.02 \
    --save_dir ./checkpoints/articulate3d_mov \
    --use_coarse_to_fine
```

**Training Features:**
- Predicts: segmentation (3 classes) + origin (3D) + axis (3D unit vector) + range (min/max)
- Loss components: segmentation + origin L1 + axis cosine + range L1 (normalized by π)
- Class weights: balanced for background/rotation/translation
- Coarse-to-fine: gradually focuses on hard examples

**Key Hyperparameters:**
- `weight_seg: 1.0` - Segmentation loss weight
- `weight_origin: 1.0` - Origin prediction loss weight
- `weight_axis: 1.0` - Axis prediction loss weight
- `weight_range: 2.0` - Range prediction loss weight (increased due to normalization)

### Stage 3: Interactable Part Segmentation

```bash
# Submit training job
sbatch train_interactable3d.sh

# Or run directly
python train_interactable3d.py \
    --data_dir /path/to/articulate3d_challenge_inter \
    --pretrain_checkpoint ./checkpoints/articulate3d_mov/best_model.pt \
    --batch_size 4 \
    --max_epochs 300 \
    --save_dir ./checkpoints/articulate3d_inter \
    --binary
```

**Training Features:**
- Binary classification: interactable (1) vs non-interactable (0)
- Initializes from movable part model for transfer learning
- Class-weighted loss to handle imbalance

## 📊 Visualization

### Comprehensive Scene Visualization

```bash
bash test_visualize_articulate3d.sh articulate
```

Or run manually:

```bash
python visualize_articulate3d.py \
    --data_dir /path/to/articulate3d_challenge_mov \
    --inter_data_dir /path/to/articulate3d_challenge_inter \
    --json_dir /path/to/Articulate3D \
    --mode validation \
    --movable_checkpoint ./checkpoints/articulate3d_range_finetune/ckpt_epoch_049.pt \
    --interactable_checkpoint ./checkpoints/articulate3d_inter/best_model.pt \
    --output_dir ./visualizations_articulate3d \
    --max_scenes 5 \
    --device cuda:0 \
    --voxel_size 0.02
```

## only segmentation and classification visualizations
python visualize_finetuned_model.py \
  --checkpoint ./checkpoints/ckpt_best.pt \
  --zarr_root /data/scannet_zarr \
  --output_dir ./results_viz

### Visualization Outputs

**1. Scene Overview HTML** (`{scene_id}_overview.html`)
- 2×2 grid layout:
  - Top-left: GT motion (movable + interactable)
  - Top-right: Predicted motion
  - Bottom-left: RGB colors
  - Bottom-right: Semantic segmentation predictions
- Displays overall metrics: accuracy, IoU, motion statistics

**2. Instance-specific HTML** (`{scene_id}_instance_{inst_id}_{motion_type}.html`)
- Per-instance visualization with 2×2 layout
- **Title Information:**
  - Instance ID, motion type, interactable type
  - Part semantic label (e.g., "handle", "door")
  - **GT Range**: e.g., `🔄 GT Range: [-90.0°, 0.0°] (Total: 90.0°)`
  - **Pred Range**: e.g., `Pred: [-85.0°, -5.0°] (Total: 80.0°)`
  - Accuracy metrics and parameter errors
- Motion arrows showing GT vs predicted axes
- Larger solid points for semantic views

**3. JSON Report** (`{scene_id}_report.json`)
- Detailed metrics per instance
- Motion parameters (origin, axis, range)
- Interactable type and part labels

### Visualization Features

- ✅ Motion arrows with rotation arcs and translation vectors
- ✅ Color-coded visualization:
  - 🔴 Red = Interactable only
  - 🔵 Blue = Movable only
  - 🟣 Purple = Both interactable & movable
  - ⚪ Gray = Background
- ✅ Semantic labels with confidence scores
- ✅ Bbox extraction with padding for context
- ✅ Interactive 3D viewer (scroll, zoom, rotate)
- ✅ All text in English for consistency

## 📂 Project Structure

```
.
├── train_articulate3d.py          # Movable part training
├── train_articulate3d.sh          # SLURM job script
├── train_interactable3d.py        # Interactable part training
├── train_interactable3d.sh        # SLURM job script
├── visualize_articulate3d.py      # Comprehensive visualization tool
├── test_visualize_articulate3d.sh # Visualization launcher
├── diagnose_movable.py            # Diagnostic tool for data issues
├── diagnose_range.py              # Range prediction diagnostics
│
├── checkpoints/
│   ├── articulate3d_mov/          # Movable part checkpoints
│   ├── articulate3d_range_finetune/ # Range-aware checkpoints
│   └── articulate3d_inter/        # Interactable part checkpoints
│
├── DINO-Infused-USDNet/
│   ├── train_usdnet_complete.py   # Semantic segmentation training
│   ├── batch_ditr_paper_strict_zarr.py # Data processing
│   └── visualize_finetuned_model.py    # Semantic visualization
│
├── motion/USDNet/data/processed/  # Processed datasets
├── Articulate3D/                  # Original JSON annotations
├── new_data_elements/             # Zarr datasets
└── visualizations_articulate3d/   # Visualization outputs
```

## 📈 Results

### Motion Parameter Prediction

| Parameter | Metric | Value |
|-----------|--------|-------|
| Origin | Mean Error | ~0.05m |
| Axis | Mean Error | ~15° |
| Range | Loss | ~0.23 (normalized) |

**Note:** Translation class is challenging due to:
- Class imbalance (fewer samples)
- Smaller part sizes (e.g., drawer slides)
- Requires more training or data augmentation

### Key Insights

1. **Range Loss Normalization**: Dividing by π makes rotation ranges comparable to other losses
2. **Semantic Labels**: Part labels (e.g., "handle") help interpret predictions
3. **JSON Integration**: GT motion ranges from original annotations improve training
4. **Visualization**: Interactive HTML files enable detailed analysis

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Library Not Found**
```bash
export LD_LIBRARY_PATH=/path/to/.conda/envs/usdnet/lib:$LD_LIBRARY_PATH
module load CUDA/12.6.0
```

**2. Range Predictions Not Available**
- Ensure using checkpoint with `range_head` (e.g., `ckpt_epoch_049.pt`)
- Old checkpoints without range prediction will use default values

**3. Missing JSON Annotations**
- Visualization will work without JSON but won't show GT ranges
- Check `--json_dir` path is correct

**4. Out of Memory**
- Reduce `--batch_size`
- Reduce `--max_points_vis` for visualization
- Use gradient accumulation: `--accumulate_grad_batches 4`

**5. Poor Translation Class Performance**
- Use `--use_coarse_to_fine` flag
- Increase `weight_seg` for translation class
- Collect more translation examples

### Performance Tips

1. **Speed up training**: Set `export OMP_NUM_THREADS=4-8`
2. **Reduce memory**: Use voxel downsampling with larger `voxel_size`
3. **Better accuracy**: Use class weights and coarse-to-fine sampling
4. **Visualization**: Set `--min_accuracy 0.3` to skip low-quality instances

## 🎓 Citation

If you use this code, please cite:

```bibtex
@misc{articulate3d-motion-prediction,
  title={Articulate3D Motion Prediction with DINO-Infused USDNet},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourrepo}}
}
```

## 📝 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- **MinkowskiEngine**: Sparse convolution framework
- **DINOv2**: Self-supervised vision transformer
- **Articulate3D Challenge**: Dataset and evaluation
- **ScanNet++**: 3D scene understanding benchmark

---

**Last Updated**: December 31, 2025

For questions or issues, please open an issue on GitHub or contact the maintainer.
