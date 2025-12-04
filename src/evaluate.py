#!/usr/bin/env python3
# ============================================
# EVALUATION SCRIPT
# ============================================

import argparse
import json
import torch
import tiktoken
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import GPTConfig, TrainingConfig
from model import GPTModel
from dataset import prepare_data, WikiTextDataset
from utils import calculate_perplexity, AverageMeter


def evaluate_on_dataset(model, dataloader, device, use_amp=True):
    """Evaluate model on a dataset"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch. no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            
            with autocast(enabled=use_amp):
                logits, loss = model(x, y)
            
            loss_meter.update(loss.item())
    
    return loss_meter.avg


def plot_training_curves(metrics_path: str, output_path: str = None):
    """Plot training curves from saved metrics"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity curves
    train_ppls = [calculate_perplexity(l) for l in train_losses]
    val_ppls = [calculate_perplexity(l) for l in val_losses]
    
    axes[1]. plot(epochs, train_ppls, 'b-o', label='Train Perplexity', linewidth=2)
    axes[1].plot(epochs, val_ppls, 'r-o', label='Validation Perplexity', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training vs Validation Perplexity')
    axes[1]. legend()
    axes[1]. grid(True, alpha=0.3)
    
    plt. tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {output_path}")
    else:
        plt.show()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GPT model')
    parser. add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--metrics', type=str, default='./checkpoints/training_metrics.json',
                        help='Path to training metrics JSON')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--plot_only', action='store_true', help='Only plot training curves')
    args = parser.parse_args()
    
    # Plot training curves
    if args.plot_only:
        metrics = plot_training_curves(args.metrics, './checkpoints/training_curves.png')
        print(f"\n📊 Final Results:")
        print(f"   Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"   Best validation perplexity: {metrics['best_val_ppl']:.2f}")
        return
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📂 Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config_dict = checkpoint['config']['model']
    model_config = GPTConfig(**model_config_dict)
    
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded")
    
    # Load data
    print("\n📥 Loading data...")
    train_tokens, val_tokens, test_tokens, tokenizer = prepare_data()
    
    # Create test dataloader
    test_dataset = WikiTextDataset(
        tokens=test_tokens,
        context_length=model_config.context_length,
        stride=model_config.context_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Test sequences: {len(test_dataset):,}")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss = evaluate_on_dataset(model, test_loader, device)
    test_ppl = calculate_perplexity(test_loss)
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    print("=" * 60)
    
    # Plot training curves if metrics file exists
    try:
        plot_training_curves(args.metrics, './checkpoints/training_curves.png')
    except FileNotFoundError:
        print("⚠️ Metrics file not found, skipping plot")
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_perplexity': test_ppl,
        'checkpoint': args.checkpoint,
    }
    
    with open('./checkpoints/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to ./checkpoints/evaluation_results.json")


if __name__ == '__main__':
    main()