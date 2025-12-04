# ============================================
# GPT MODEL ARCHITECTURE
# ============================================

import torch
import torch.nn as nn
import math


class LayerNorm(nn.Module):
    """Layer Normalization with learnable parameters"""
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch. Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention"""
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.emb_dim = config.emb_dim
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config. emb_dim, 3 * config.emb_dim, bias=config.qkv_bias)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)
        
        self.attn_dropout = nn.Dropout(config. drop_rate)
        self. resid_dropout = nn.Dropout(config.drop_rate)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1). bool()
        )

    def forward(self, x: torch.Tensor) -> torch. Tensor:
        B, T, C = x.shape
        
        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.emb_dim, dim=2)
        
        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self. head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous(). view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        
        return out


class FeedForward(nn.Module):
    """Feed Forward Network with GELU activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, 4 * config.emb_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.emb_dim, config. emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor) -> torch. Tensor:
        x = self. c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LayerNorm"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self. ln2 = LayerNorm(config.emb_dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """Complete GPT Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn. Embedding(config.vocab_size, config.emb_dim),      # Token embeddings
            'wpe': nn.Embedding(config.context_length, config.emb_dim),  # Position embeddings
            'drop': nn.Dropout(config.drop_rate),
            'blocks': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            'ln_f': LayerNorm(config. emb_dim),
        })
        
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self. transformer['wte'].weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn. endswith('c_proj.weight'):
                torch.nn.init. normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch. nn.init.normal_(module. weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init. zeros_(module.bias)
        elif isinstance(module, nn. Embedding):
            torch.nn. init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch. Tensor, targets: torch.Tensor = None):
        B, T = idx. shape
        assert T <= self.config.context_length, f"Sequence length {T} exceeds context length {self.config.context_length}"
        
        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer['blocks']:
            x = block(x)
        
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits. size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx