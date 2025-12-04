# 🤖 GPT-2 From Scratch (PyTorch)

A clean, educational, and interpretable implementation of **GPT-2** built from scratch using PyTorch. This project demonstrates the complete pipeline of training a Large Language Model (LLM) — from data tokenization and transformer architecture to distributed training on multi-GPU setups.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Trained-success.svg)
![Perplexity](https://img.shields.io/badge/Perplexity-19.66-brightgreen.svg)

## 🎯 Project Goals

1.  **Demystify LLMs:** Readable code for the Transformer architecture without hidden abstractions.
2.  **Performance:** Match official GPT-2 Small performance (Perplexity ~18-20).
3.  **Scalability:** Support **Distributed Data Parallel (DDP)** for multi-GPU training.

## 📊 Training Results

We successfully trained a **GPT-2 Small (124M)** model on the **WikiText-103** dataset using 4x NVIDIA V100 GPUs.

| Metric | Our Result | Official GPT-2 Small |
|--------|------------|----------------------|
| **Parameters** | **124M** | 124M |
| **Dataset** | WikiText-103 | WebText |
| **Validation Loss** | **2.96** | ~2.9 |
| **Test Perplexity** | **19.66** | ~18-20 |
| **Training Time** | ~20 hours | N/A |

---

## 🏗️ Architecture

This model follows the standard **Decoder-only Transformer** architecture:
*   **Embeddings:** Learned Token + Positional Embeddings.
*   **Layers:** 12 Decoder Blocks.
*   **Attention:** Multi-Head Causal Self-Attention (12 heads).
*   **Activation:** GELU.
*   **Context Window:** 512 tokens (Adjustable).

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/gpt-from-scratch.git
cd gpt-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model

You can train on a single GPU or scale up to multiple GPUs (DDP).

#### 🟢 Option A: Multi-GPU (Recommended for Speed)
Uses **DistributedDataParallel (DDP)**. This splits the batch across GPUs and syncs gradients.

```bash
# Run on 4 GPUs (Adjust nproc_per_node to match your GPU count)
torchrun --nproc_per_node=4 src/train_ddp.py
```

#### 🔵 Option B: Single GPU
Good for debugging or if you only have one GPU available.

```bash
# Run on GPU 0
python src/train.py --gpu 0
```

### 3. Generating Text

Once trained, generate text using your saved checkpoint. We use a **repetition penalty** to prevent the model from looping text.

```bash
# Interactive Mode (Chat-like interface)
python src/generate.py --interactive --repetition_penalty 1.3 --temperature 0.7

# Single Prompt Command
python src/generate.py \
    --prompt "The history of artificial intelligence" \
    --max_tokens 150 \
    --repetition_penalty 1.3
```

---

## ⚙️ Configuration Guide (Adjusting for Your GPU)

You can modify `src/config.py` to fit the memory (VRAM) of your specific hardware.

### 1. High-End (A100 / V100 / 3090 / 4090 - 24GB+ VRAM)
*Goal: Full GPT-2 Small (124M)*
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024   # Full context
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
```
*Batch Size in `train_ddp.py`: 16 or 32*

### 2. Mid-Range (T4 / 3060 / 2080Ti - 12-16GB VRAM)
*Goal: GPT-2 Small (Compressed Context)*
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 512    # ⬇️ Reduced context
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
```
*Batch Size in `train_ddp.py`: 8 or 12*

### 3. Low-End (Consumer GPU / Colab Free - 6-8GB VRAM)
*Goal: Baby GPT (~35M Params)*
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256    # ⬇️ Short context
    emb_dim: int = 384           # ⬇️ Smaller dimension
    n_heads: int = 6             # ⬇️ Fewer heads
    n_layers: int = 6            # ⬇️ Shallower
```
*Batch Size in `train_ddp.py`: 32*

---

## 📁 Repository Structure

```
gpt-from-scratch/
├── src/
│   ├── config.py        # Hyperparameters
│   ├── model.py         # GPT Architecture (PyTorch)
│   ├── dataset.py       # Data loader for WikiText-103
│   ├── train_ddp.py     # Multi-GPU training script
│   ├── train.py         # Single-GPU training script
│   ├── generate.py      # Inference/Generation script
│   └── utils.py         # Logging and checkpointing utilities
├── scripts/
│   └── run_training.sh  # Bash helper script
├── checkpoints/         # Saved models
└── logs/                # Training metrics
```

## 🙏 Acknowledgments
*   **OpenAI** for the GPT-2 paper.
*   **Salesforce** for the WikiText-103 dataset.
*   **Andrej Karpathy** for inspiration via nanoGPT.