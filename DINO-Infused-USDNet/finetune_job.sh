#!/bin/bash
#SBATCH --job-name=usdnet_finetune
#SBATCH --partition=slowlane
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aisa-gpuA01
#SBATCH --time=12:00:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

cd /mnt/nfs/home/st185933/DINO-Infused-USDNet

# Load CUDA module
module load CUDA/12.6.0

source activate usdnet

python train_usdnet_complete.py \
    --stage finetune \
    --zarr_root ../new_data_elements \
    --resume ./checkpoints/finetune/ckpt_epoch_081.pt \
    --save_dir ./checkpoints \
    --batch_size 8 \
    --num_workers 2 \
    --max_epochs 100 \
    --learning_rate 1e-3 \
    --device cuda:0
