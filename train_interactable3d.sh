#!/bin/bash
#SBATCH --job-name=inter3d
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/nfs/home/st185933/interactable3d_%j.out
#SBATCH --error=/mnt/nfs/home/st185933/interactable3d_%j.err
module load CUDA/12.6.0
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment (robust)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate usdnet
else
  source ~/.bashrc
  conda activate usdnet
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1

# Navigate to working directory
cd /mnt/nfs/home/st185933

# Display system info
echo ""
echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Configuration
DATA_DIR="/mnt/nfs/home/st185933/motion/USDNet/data/processed/articulate3d_challenge_inter"
SAVE_DIR="/mnt/nfs/home/st185933/checkpoints/articulate3d_inter"
PRETRAIN_CKPT="/mnt/nfs/home/st185933/checkpoints/articulate3d_mov/best_model.pt"

# Run training
echo "=========================================="
echo "Starting Interactable Part Training"
echo "=========================================="
echo "Data: $DATA_DIR"
echo "Save: $SAVE_DIR"
echo "Pretrain: $PRETRAIN_CKPT"
echo ""

python train_interactable3d.py \
  --data_dir $DATA_DIR \
  --train_mode train \
  --batch_size 1 \
  --max_epochs 300 \
  --learning_rate 1e-4 \
  --voxel_size 0.02 \
  --num_workers 4 \
  --save_dir $SAVE_DIR \
  --pretrain_checkpoint $PRETRAIN_CKPT \
  --binary \
  --use_coarse_to_fine \
  --device cuda:0

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
