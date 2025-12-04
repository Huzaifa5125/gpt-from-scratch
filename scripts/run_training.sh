#!/bin/bash
# ============================================
# SHELL SCRIPT TO RUN GPT TRAINING
# ============================================

# Exit on error
set -e

echo "============================================"
echo "🚀 GPT TRAINING ON WIKITEXT-103"
echo "============================================"
echo ""

# Configuration
NUM_GPUS=4                    # Number of GPUs to use
MASTER_PORT=29500             # Port for distributed training

# Create directories
mkdir -p checkpoints logs data_cache

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "============================================"
echo "📥 PREPARING DATA..."
echo "============================================"

# Prepare data first (single process to avoid duplicate downloads)
python3 -c "from dataset import prepare_data; prepare_data()"

echo ""
echo "============================================"
echo "🏋️ STARTING MULTI-GPU TRAINING (${NUM_GPUS} GPUs)"
echo "============================================"
echo ""

# Run distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_ddp.py

echo ""
echo "============================================"
echo "✅ TRAINING COMPLETE!"
echo "============================================"
echo ""

# Run evaluation
echo "📊 Running evaluation..."
python3 evaluate.py --checkpoint ./checkpoints/best_model.pt

echo ""
echo "============================================"
echo "🤖 TESTING GENERATION..."
echo "============================================"
echo ""

# Test generation
python3 generate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --prompt "The history of artificial intelligence" \
    --max_tokens 100

echo ""
echo "✅ All done!  Check ./checkpoints for saved models."