# ============================================
# CONFIGURATION FILE
# ============================================
# Optimized for DGX Station with 4x V100-32GB

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    """Model Architecture Configuration"""
    # Model architecture
    vocab_size: int = 50257          # GPT-2 tokenizer vocabulary
    context_length: int = 512        # Increased for V100 (more memory)
    emb_dim: int = 768             # Embedding dimension (GPT-2 small)
    n_heads: int = 12                # Attention heads
    n_layers: int = 12              # Transformer blocks (GPT-2 small)
    
    # Regularization
    drop_rate: float = 0.15          # Dropout rate
    qkv_bias: bool = False           # QKV projection bias


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Paths
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Training hyperparameters
    batch_size: int = 16             # Per-GPU batch size
    gradient_accumulation_steps: int = 4  # Effective batch = 16 * 4 * 4 GPUs = 256
    learning_rate: float = 3e-4
    min_lr: float = 3e-5             # Minimum LR (10% of max)
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    
    # Data
    stride: int = 256                # Overlap between sequences
    num_workers: int = 4             # DataLoader workers per GPU
    
    # Checkpointing
    save_every_n_steps: int = 5000   # Save checkpoint every N steps
    eval_every_n_steps: int = 5000   # Evaluate every N steps
    log_every_n_steps: int = 1000     # Log every N steps
    
    # Mixed precision
    use_amp: bool = True             # Use Automatic Mixed Precision (faster on V100)
    
    # Multi-GPU
    use_ddp: bool = True             # Use DistributedDataParallel
    
    # Resume training
    resume_from: Optional[str] = None  # Path to checkpoint to resume from


@dataclass
class GenerationConfig:
    """Text Generation Configuration"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


def get_config():
    """Return all configurations"""
    return {
        "model": GPTConfig(),
        "training": TrainingConfig(),
        "generation": GenerationConfig(),
    }


# Create output directories
def setup_directories(config: TrainingConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📁 Log directory: {config.log_dir}")