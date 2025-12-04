#!/usr/bin/env python3
# ============================================
# SINGLE GPU TRAINING SCRIPT
# ============================================
# Run with: python train.py

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import get_config, setup_directories
from model import GPTModel
from dataset import prepare_data, create_dataloaders
from utils import (
    setup_logging, get_lr, save_checkpoint, load_checkpoint,
    calculate_perplexity, AverageMeter, get_gpu_memory_info, print_model_summary
)


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, global_step, config, logger):
    """Train for one epoch"""
    model.train()
    
    train_config = config['training']
    total_steps = len(train_loader) * train_config.max_epochs
    
    loss_meter = AverageMeter()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(progress):
        x, y = x. to(device), y.to(device)
        
        # Update learning rate
        lr = get_lr(
            global_step,
            train_config.warmup_steps,
            total_steps,
            train_config.learning_rate,
            train_config.min_lr
        )
        for param_group in optimizer. param_groups:
            param_group['lr'] = lr
        
        # Forward pass with AMP
        with autocast(enabled=train_config.use_amp):
            logits, loss = model(x, y)
            loss = loss / train_config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
            scaler. unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        
        loss_meter.update(loss.item() * train_config.gradient_accumulation_steps)
        
        progress.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'ppl': f'{calculate_perplexity(loss_meter.avg):.2f}',
            'lr': f'{lr:.2e}',
            'step': global_step
        })
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(model, val_loader, device, config):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    for x, y in tqdm(val_loader, desc="Validating"):
        x, y = x.to(device), y.to(device)
        
        with autocast(enabled=config['training'].use_amp):
            logits, loss = model(x, y)
        
        loss_meter.update(loss.item())
    
    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Train GPT on WikiText-103 (Single GPU)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)
    
    # Get configuration
    config = get_config()
    model_config = config['model']
    train_config = config['training']
    
    # Setup
    setup_directories(train_config)
    logger = setup_logging(train_config.log_dir)
    
    logger.info("=" * 60)
    logger.info("🚀 GPT TRAINING ON WIKITEXT-103 (Single GPU)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    
    # Prepare data
    train_tokens, val_tokens, test_tokens, tokenizer = prepare_data()
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(train_tokens, val_tokens, config)
    
    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val batches: {len(val_loader):,}")
    
    # Create model
    model = GPTModel(model_config).to(device)
    print_model_summary(model, model_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scaler = GradScaler(enabled=train_config.use_amp)
    
    # Resume if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scaler, device)
        start_epoch = checkpoint_info['epoch'] + 1
        global_step = checkpoint_info['global_step']
        best_val_loss = checkpoint_info['val_loss']
    
    # Training loop
    for epoch in range(start_epoch, train_config.max_epochs):
        epoch_start = time.time()
        
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, global_step, config, logger
        )
        
        val_loss = validate(model, val_loader, device, config)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Time={epoch_time:.1f}s")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model, optimizer, None, scaler,
            epoch, global_step, train_loss, val_loss,
            config, train_config.output_dir, is_best
        )
    
    logger. info(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()