# ============================================
# UTILITY FUNCTIONS
# ============================================

import torch
import math
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any


def setup_logging(log_dir: str, rank: int = 0):
    """Setup logging configuration"""
    if rank != 0:
        logging.basicConfig(level=logging.WARNING)
        return logging.getLogger(__name__)
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math. cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    config: Dict[str, Any],
    output_dir: str,
    is_best: bool = False,
    rank: int = 0
):
    """Save model checkpoint"""
    if rank != 0:  # Only save from main process
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model state dict (handle DDP wrapper)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'model': vars(config['model']),
            'training': vars(config['training']),
        }
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"🏆 Best model saved: {best_path}")
    
    # Save latest pointer
    latest_path = os. path.join(output_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scaler=None, device='cuda'):
    """Load model checkpoint"""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model_state = checkpoint['model_state_dict']
    
    # Handle DDP wrapper
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint. get('epoch', 0),
        'global_step': checkpoint. get('global_step', 0),
        'train_loss': checkpoint.get('train_loss', float('inf')),
        'val_loss': checkpoint.get('val_loss', float('inf')),
    }


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss"""
    return math.exp(min(loss, 20))  # Cap to avoid overflow


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        info.append(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB (reserved: {reserved:.1f}GB)")
    
    return " | ".join(info)


def print_model_summary(model, config):
    """Print model summary"""
    total_params, trainable_params = model.count_parameters()
    
    print("\n" + "=" * 60)
    print("📊 MODEL SUMMARY")
    print("=" * 60)
    print(f"Architecture: GPT")
    print(f"Vocabulary size: {config. vocab_size:,}")
    print(f"Context length: {config.context_length}")
    print(f"Embedding dim: {config.emb_dim}")
    print(f"Attention heads: {config.n_heads}")
    print(f"Layers: {config.n_layers}")
    print(f"Dropout: {config.drop_rate}")
    print("-" * 60)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (FP32): ~{total_params * 4 / 1e9:.2f} GB")
    print("=" * 60 + "\n")