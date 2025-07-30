#!/bin/bash
#SBATCH --job-name=train_yolo
#SBATCH --output=../results/logs/yolo_%j.log
#SBATCH --error=../results/logs/yolo_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# === Load environment modules ===
module load python/3.11.7
module load cuda/12.3.1
module load cudnn/9.2.0.82-12

# === Safety: fail on error ===
set -e

# === Activate virtualenv ===
source /work/FAC/FBM/DEE/mrobinso/moult/michele/MoultGPT/mistral_env/bin/activate

# === Diagnostics ===
echo "🔍 Host: $(hostname)"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo "💾 Pip path: $(which pip)"
echo "🔧 CUDA:"
nvidia-smi

# === Run YOLO training ===
python train_yolo_model.py
