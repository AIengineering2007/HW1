# hw1-build: caspian-7
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    # Experiment switch (Part E). "none" = the exact Part B architecture.
    # "no_position_embedding" = A1, "post_ln" = A2. Any other value is ignored.
    ablation: str = "none"
    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")

        if self.n_head <= 0:
            raise ValueError("n_head must be positive")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")



class CausalSelfAttentionK(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=False
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.attn_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.block_size, config.block_size)
        )

        self.register_buffer(
            "mask",
            mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value

            past_len = past_k.size(-2)

            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.mask[:, :, past_len:past_len + T, :past_len + T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ v

        output = output.transpose(1, 2).contiguous().view(B, T, C)

        output = self.c_proj(output)

        present = (k, v)

        if use_cache:
            return output, present

        return output, None



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=False
        )

        self.fc2 = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)

        x = F.gelu(x, approximate="tanh")

        x = self.fc2(x)

        x = self.dropout(x)

        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionK(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.post_ln = config.ablation == "post_ln"

    def forward(self, x, past_key_value=None, use_cache=False):
        if self.post_ln:
            # A2 (Post-LN): x = LN(x + Attn(x)); x = LN(x + MLP(x))
            attn_output, present = self.attn(
                x,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            x = self.ln1(x + attn_output)
            x = self.ln2(x + self.mlp(x))
        else:
            # Part B (Pre-LN): x = x + Attn(LN1(x)); x = x + MLP(LN2(x))
            attn_output, present = self.attn(
                self.ln1(x),
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            x = x + attn_output
            x = x + self.mlp(self.ln2(x))

        if use_cache:
            return x, present

        return x



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # A1 removes the learned position embedding entirely
        self.use_pos = config.ablation != "no_position_embedding"

        if self.use_pos:
            self.wpe = nn.Embedding(
                config.block_size,
                config.n_embd
            )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)

        self.lm_head.weight = self.wte.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    
    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        B, T = idx.shape

        past_len = 0

        if past_kv is not None:
            past_len = past_kv[0][0].size(-2)

        if past_len + T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(
            past_len,
            past_len + T,
            dtype=torch.long,
            device=idx.device
        )

        x = self.wte(idx)

        if self.use_pos:
            x = x + self.wpe(positions)

        x = self.drop(x)

        presents = []

        for i, block in enumerate(self.blocks):
            layer_past = None

            if past_kv is not None:
                layer_past = past_kv[i]

            if use_cache:
                x, present = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True
                )

                presents.append(present)

            else:
                x = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=False
                )

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        if use_cache:
            return logits, loss, presents

        return logits, loss


    def num_params(self):
        return sum(
            p.numel()
            for p in {id(p): p for p in self.parameters()}.values()
            if p.requires_grad
        )
