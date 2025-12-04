# ============================================
# DATASET AND DATALOADER
# ============================================

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle


class WikiTextDataset(Dataset):
    """WikiText Dataset for Language Modeling"""
    
    def __init__(self, tokens, context_length, stride=None):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride if stride else context_length
        self.n_sequences = max(0, (len(tokens) - context_length) // self.stride)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def get_tokenizer():
    """Get GPT-2 BPE tokenizer"""
    return tiktoken.get_encoding("gpt2")


def tokenize_dataset(dataset_split, tokenizer, desc="Tokenizing"):
    """Tokenize a dataset split"""
    all_tokens = []
    for text in tqdm(dataset_split['text'], desc=desc):
        if text. strip():
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            all_tokens.extend(tokens)
    return all_tokens


def prepare_data(cache_dir="./data_cache"):
    """
    Load and prepare WikiText-103 dataset
    Caches tokenized data for faster subsequent loads
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache = os.path.join(cache_dir, "train_tokens. pkl")
    val_cache = os.path.join(cache_dir, "val_tokens.pkl")
    test_cache = os.path.join(cache_dir, "test_tokens.pkl")
    
    tokenizer = get_tokenizer()
    
    # Check if cached data exists
    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("📂 Loading cached tokenized data...")
        with open(train_cache, 'rb') as f:
            train_tokens = pickle.load(f)
        with open(val_cache, 'rb') as f:
            val_tokens = pickle.load(f)
        with open(test_cache, 'rb') as f:
            test_tokens = pickle.load(f)
    else:
        print("📥 Loading WikiText-103 dataset...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        
        print(f"📊 Dataset sizes:")
        print(f"   Train: {len(dataset['train']):,} samples")
        print(f"   Validation: {len(dataset['validation']):,} samples")
        print(f"   Test: {len(dataset['test']):,} samples")
        
        print("\n🔄 Tokenizing datasets...")
        train_tokens = tokenize_dataset(dataset['train'], tokenizer, "Tokenizing train")
        val_tokens = tokenize_dataset(dataset['validation'], tokenizer, "Tokenizing validation")
        test_tokens = tokenize_dataset(dataset['test'], tokenizer, "Tokenizing test")
        
        # Cache for future use
        print("💾 Caching tokenized data...")
        with open(train_cache, 'wb') as f:
            pickle.dump(train_tokens, f)
        with open(val_cache, 'wb') as f:
            pickle.dump(val_tokens, f)
        with open(test_cache, 'wb') as f:
            pickle.dump(test_tokens, f)
    
    print(f"\n📊 Token counts:")
    print(f"   Train: {len(train_tokens):,} tokens")
    print(f"   Validation: {len(val_tokens):,} tokens")
    print(f"   Test: {len(test_tokens):,} tokens")
    
    return train_tokens, val_tokens, test_tokens, tokenizer


def create_dataloaders(train_tokens, val_tokens, config, rank=0, world_size=1):
    """
    Create DataLoaders for training and validation
    Supports both single-GPU and multi-GPU (DDP) training
    """
    model_config = config["model"]
    train_config = config["training"]
    
    # Create datasets
    train_dataset = WikiTextDataset(
        tokens=train_tokens,
        context_length=model_config.context_length,
        stride=train_config.stride
    )
    
    val_dataset = WikiTextDataset(
        tokens=val_tokens,
        context_length=model_config.context_length,
        stride=model_config.context_length  # No overlap for validation
    )
    
    # Create samplers for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_config. num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler