#!/usr/bin/env python3
# ============================================
# IMPROVED TEXT GENERATION SCRIPT
# ============================================

import argparse
import torch
import tiktoken

from config import GPTConfig
from model import GPTModel


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    print(f"📂 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config_dict = checkpoint['config']['model']
    model_config = GPTConfig(**model_config_dict)
    
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    
    return model, model_config


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,  # NEW: Penalize repetition
    device: str = 'cuda'
):
    """Generate text with repetition penalty"""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device). unsqueeze(0)
    
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Crop to context length
        idx_cond = tokens if tokens.size(1) <= model.config. context_length else tokens[:, -model.config.context_length:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Last token logits
        
        # Apply repetition penalty
        if len(generated_tokens) > 0:
            for prev_token in set(generated_tokens[-50:]):  # Check last 50 tokens
                logits[0, prev_token] /= repetition_penalty
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Stop at end of text token or newline patterns
        if next_token.item() == tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]:
            break
    
    return tokenizer.decode(tokens[0]. tolist())


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained GPT model')
    parser. add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--prompt', type=str, default='The history of artificial intelligence')
    parser.add_argument('--max_tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.7)  # Lower = more focused
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser. add_argument('--repetition_penalty', type=float, default=1.3)  # NEW
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    model, config = load_model(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("🤖 INTERACTIVE TEXT GENERATION")
        print("=" * 60)
        print("Commands:")
        print("  'quit' - Exit")
        print("  'temp 0.5' - Change temperature")
        print("  'rep 1.5' - Change repetition penalty")
        print("=" * 60)
        
        temp = args.temperature
        rep_penalty = args.repetition_penalty
        
        while True:
            prompt = input("\n📝 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            elif prompt.startswith('temp '):
                temp = float(prompt. split()[1])
                print(f"🌡️ Temperature set to {temp}")
                continue
            elif prompt.startswith('rep '):
                rep_penalty = float(prompt.split()[1])
                print(f"🔄 Repetition penalty set to {rep_penalty}")
                continue
            
            if not prompt:
                continue
            
            print("\n🤖 Generating...")
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args. max_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=rep_penalty,
                device=device
            )
            
            print("\n" + "-" * 50)
            print(generated)
            print("-" * 50)
    else:
        print(f"\n📝 Prompt: {args.prompt}")
        print(f"🌡️ Temperature: {args.temperature}")
        print(f"🔄 Repetition Penalty: {args.repetition_penalty}")
        print("-" * 50)
        
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        print(generated)


if __name__ == '__main__':
    main()