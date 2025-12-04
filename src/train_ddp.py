#!/usr/bin/env python3
# ============================================
# MULTI-GPU TRAINING SCRIPT (DDP)
# ============================================
# Run with: torchrun --nproc_per_node=4 train_ddp.py

import os
import sys
import time
import math
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import GPTConfig, TrainingConfig, get_config, setup_directories
from model import GPTModel
from dataset import prepare_data, create_dataloaders, get_tokenizer
from utils import (
    setup_logging, get_lr, save_checkpoint, load_checkpoint,
    calculate_perplexity, AverageMeter, get_gpu_memory_info, print_model_summary
)


def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def cleanup_ddp():
    """Cleanup distributed training"""
    dist. destroy_process_group()


def train_epoch(
    model, train_loader, optimizer, scaler, device, 
    epoch, global_step, config, logger, rank
):
    """Train for one epoch"""
    model.train()
    
    train_config = config['training']
    total_steps = len(train_loader) * train_config.max_epochs
    
    loss_meter = AverageMeter()
    
    # Progress bar only on rank 0
    if rank == 0:
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    else:
        progress = train_loader
    
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
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with AMP
        with autocast(enabled=train_config.use_amp):
            logits, loss = model(x, y)
            loss = loss / train_config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler. step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        
        # Update metrics
        loss_meter.update(loss.item() * train_config.gradient_accumulation_steps)
        
        # Update progress bar
        if rank == 0:
            progress.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'ppl': f'{calculate_perplexity(loss_meter.avg):.2f}',
                'lr': f'{lr:.2e}',
                'step': global_step
            })
        
        # Logging
        if rank == 0 and global_step % train_config.log_every_n_steps == 0:
            logger.info(
                f"Step {global_step} | Loss: {loss_meter. avg:.4f} | "
                f"PPL: {calculate_perplexity(loss_meter.avg):.2f} | LR: {lr:.2e}"
            )
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(model, val_loader, device, config, rank):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    if rank == 0:
        progress = tqdm(val_loader, desc="Validating")
    else:
        progress = val_loader
    
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        
        with autocast(enabled=config['training']. use_amp):
            logits, loss = model(x, y)
        
        loss_meter.update(loss.item())
    
    # Synchronize validation loss across GPUs
    loss_tensor = torch.tensor([loss_meter.avg], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GPT on WikiText-103')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp()
    
    # Get configuration
    config = get_config()
    model_config = config['model']
    train_config = config['training']
    
    # Setup directories and logging
    if rank == 0:
        setup_directories(train_config)
    logger = setup_logging(train_config.log_dir, rank)
    
    # Print configuration
    if rank == 0:
        logger.info("=" * 60)
        logger. info("🚀 GPT TRAINING ON WIKITEXT-103")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Device: {device}")
        logger. info(f"Effective batch size: {train_config. batch_size * train_config. gradient_accumulation_steps * world_size}")
    
    # Prepare data (only on rank 0, then share)
    if rank == 0:
        logger.info("\n📥 Preparing data...")
    train_tokens, val_tokens, test_tokens, tokenizer = prepare_data()
    
    # Wait for all processes
    dist.barrier()
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(
        train_tokens, val_tokens, config, rank, world_size
    )
    
    if rank == 0:
        logger.info(f"Train batches: {len(train_loader):,}")
        logger.info(f"Val batches: {len(val_loader):,}")
    
    # Create model
    model = GPTModel(model_config). to(device)
    
    if rank == 0:
        print_model_summary(model, model_config)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=train_config.use_amp)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume or train_config.resume_from:
        checkpoint_path = args.resume or train_config.resume_from
        checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, scaler, device)
        start_epoch = checkpoint_info['epoch'] + 1
        global_step = checkpoint_info['global_step']
        best_val_loss = checkpoint_info['val_loss']
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("🏋️ STARTING TRAINING")
        logger.info("=" * 60)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, train_config.max_epochs):
        # Set epoch for sampler (important for shuffling in DDP)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, global_step, config, logger, rank
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device, config, rank)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        if rank == 0:
            train_ppl = calculate_perplexity(train_loss)
            val_ppl = calculate_perplexity(val_loss)
            
            logger.info("\n" + "=" * 60)
            logger.info(f"📊 EPOCH {epoch+1}/{train_config.max_epochs} SUMMARY")
            logger.info("=" * 60)
            logger.info(f"⏱️  Time: {epoch_time:.1f}s")
            logger. info(f"📉 Train Loss: {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
            logger.info(f"📈 Val Loss:   {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
            logger.info(f"📊 Gap: {val_loss - train_loss:.4f}")
            logger.info(f"💾 {get_gpu_memory_info()}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info("🏆 New best validation loss!")
            
            save_checkpoint(
                model, optimizer, None, scaler,
                epoch, global_step, train_loss, val_loss,
                config, train_config. output_dir, is_best, rank
            )
        
        # Synchronize
        dist.barrier()
    
    # Training complete
    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")
        logger.info(f"Model saved to: {train_config. output_dir}")
        
        # Save final metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_val_ppl': calculate_perplexity(best_val_loss),
        }
        
        import json
        with open(os.path.join(train_config.output_dir, 'training_metrics.json'), 'w') as f:
            json. dump(metrics, f, indent=2)
    
    cleanup_ddp()


if __name__ == '__main__':
    main()