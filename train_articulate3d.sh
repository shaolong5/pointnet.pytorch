#!/bin/bash
#SBATCH --job-name=articulate3d
#SBATCH --output=articulate3d_%j.out
#SBATCH --error=articulate3d_%j.err
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ============================================================================
# Articulate3D Training Script
# Movable Part Segmentation + Articulation Prediction
# ============================================================================

echo "=========================================="
echo "Starting Articulate3D Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
module load CUDA/12.6.0
# Environment setup
cd /mnt/nfs/home/st185933

# Activate conda environment (adjust to your environment name)
# source ~/.bashrc
# conda activate usdnet

# Robust conda activation
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate usdnet
else
  source ~/.bashrc
  conda activate usdnet
fi

# Speed up MinkowskiEngine
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0

# Data and checkpoint paths
DATA_DIR="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_mov"
SAVE_DIR="./checkpoints/articulate3d_mov"
PRETRAIN_CKPT="/mnt/nfs/home/st185933/DINO-Infused-USDNet/checkpointss/finetune/ckpt_best.pt"

# Training parameters
BATCH_SIZE=1
EPOCHS=300
LR=0.0001
VOXEL_SIZE=0.02
NUM_WORKERS=4

echo ""
echo "Configuration:"
echo "  Data dir: $DATA_DIR"
echo "  Save dir: $SAVE_DIR"
echo "  Pretrain ckpt: $PRETRAIN_CKPT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Voxel size: $VOXEL_SIZE"
echo ""

# Run training with resume
python train_articulate3d.py \
    --data_dir "$DATA_DIR" \
    --train_mode train \
    --batch_size $BATCH_SIZE \
    --max_epochs $EPOCHS \
    --learning_rate $LR \
    --voxel_size $VOXEL_SIZE \
    --num_workers $NUM_WORKERS \
    --save_dir "$SAVE_DIR" \
    --resume auto \
    --use_coarse_to_fine

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
