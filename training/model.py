"""
RHDT: a Decision Transformer whose attention layers perform Hopfield retrieval.

Each attention layer first refines its queries with the modern Hopfield update
(Ramsauer et al., 2021)

    q <- softmax(beta * q K^T) K,        beta = beta_scale / sqrt(head_dim)

`num_iterations` times, then attends to the values with the refined queries.
beta_scale is 1 unless a config sets it (-b<X>, see configs.sh).
The keys K are the stored patterns; the key-orthogonality regularizer in
train_rhdt.py reads them from `last_K`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldAttention(nn.Module):
    """Causal multi-head attention with iterative Hopfield retrieval."""

    def __init__(self, embed_dim, num_heads=4, num_iterations=1, dropout=0.1, beta_scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_iterations = num_iterations
        self.beta = beta_scale / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.last_K = None  # keys of the last forward pass, [B, H, T, head_dim]

    def _attention_weights(self, q, K, mask):
        scores = self.beta * torch.matmul(q, K.transpose(-2, -1))
        scores = scores.masked_fill(mask == 0, -1e9)
        # Rows with every position masked would give NaN; zero them instead
        return torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Keep the graph while training so the regularizer can backpropagate
        self.last_K = K if self.training else K.detach()

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            mask = mask * attention_mask.unsqueeze(1).unsqueeze(2)

        # Hopfield retrieval: move the queries toward the stored patterns (keys)
        q = Q
        for _ in range(self.num_iterations):
            q = torch.matmul(self._attention_weights(q, K, mask), K)

        weights = self.dropout(self._attention_weights(q, K, mask))
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class HopfieldBlock(nn.Module):
    """Pre-norm transformer block with Hopfield attention."""

    def __init__(self, embed_dim, num_heads=4, num_iterations=1, dropout=0.1, beta_scale=1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = HopfieldAttention(embed_dim, num_heads, num_iterations, dropout, beta_scale)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class RHDT(nn.Module):
    """Robust Hopfield Decision Transformer.

    Input sequence per timestep: (return-to-go, state, action) tokens.
    The action at timestep t is predicted from the state token of t.
    """

    def __init__(self, state_dim, action_dim, embed_dim=128, num_layers=3, num_heads=4,
                 max_ep_len=1000, seq_len=20, num_iterations=1, dropout=0.1,
                 embedding_dropout=0.05, device='cuda', beta_scale=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.max_ep_len = max_ep_len
        self.device = device

        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)

        self.embed_ln = nn.LayerNorm(embed_dim)
        self.emb_drop = nn.Dropout(embedding_dropout)

        self.blocks = nn.ModuleList([
            HopfieldBlock(embed_dim, num_heads, num_iterations, dropout, beta_scale)
            for _ in range(num_layers)
        ])

        self.out_ln = nn.LayerNorm(embed_dim)
        self.action_head = nn.Sequential(nn.Linear(embed_dim, action_dim), nn.Tanh())

        self.to(device)

    def get_K_matrices(self):
        """Keys of every layer from the last forward pass."""
        return [block.attn.last_K for block in self.blocks if block.attn.last_K is not None]

    def forward(self, states, actions, rtg, timesteps, attn_mask=None):
        B, T = states.shape[:2]

        if attn_mask is None:
            attn_mask = torch.ones(B, T, device=self.device)

        s_emb = self.embed_state(states) + self.embed_timestep(timesteps)
        a_emb = self.embed_action(actions) + self.embed_timestep(timesteps)
        r_emb = self.embed_return(rtg) + self.embed_timestep(timesteps)

        # Interleave as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        x = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(B, 3 * T, self.embed_dim)
        x = self.emb_drop(self.embed_ln(x))

        mask = attn_mask.unsqueeze(2).expand(B, T, 3).reshape(B, 3 * T)

        for block in self.blocks:
            x = block(x, mask)

        x = self.out_ln(x).reshape(B, T, 3, self.embed_dim)
        return self.action_head(x[:, :, 1])  # predict a_t from the s_t token

    def get_action(self, states, actions, rtg, timesteps):
        """Action for the last timestep of a (possibly shorter) context."""
        states = states.view(1, -1, self.state_dim)
        actions = actions.view(1, -1, self.action_dim)
        rtg = rtg.view(1, -1, 1)
        timesteps = timesteps.view(1, -1)

        T = states.shape[1]
        if T > self.seq_len:
            states = states[:, -self.seq_len:]
            actions = actions[:, -self.seq_len:]
            rtg = rtg[:, -self.seq_len:]
            timesteps = timesteps[:, -self.seq_len:]
            T = self.seq_len

        # Left-pad short contexts to seq_len and mask the padding
        if T < self.seq_len:
            pad = self.seq_len - T
            states = torch.cat([torch.zeros(1, pad, self.state_dim, device=self.device), states], 1)
            actions = torch.cat([torch.ones(1, pad, self.action_dim, device=self.device) * -10, actions], 1)
            rtg = torch.cat([torch.zeros(1, pad, 1, device=self.device), rtg], 1)
            timesteps = torch.cat([torch.zeros(1, pad, dtype=torch.long, device=self.device), timesteps], 1)
            attn_mask = torch.cat([torch.zeros(1, pad, device=self.device),
                                   torch.ones(1, T, device=self.device)], 1)
        else:
            attn_mask = None

        return self.forward(states, actions, rtg, timesteps, attn_mask)[0, -1]
