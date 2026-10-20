"""
Robust Decision Transformer (RDT) baseline: train one model.

Our implementation of RDT (Xu et al., ICLR 2025, https://arxiv.org/abs/2407.04285): a
GPT-2 Decision Transformer with embedding dropout, a Gaussian-weighted action loss (each
sample weighted by exp(-beta_a x its squared error)) and a reward-prediction head, without
RDT's iterative data correction. Each task has its own beta_a, beta_r, reward-loss weight
and embedding dropout (see train_rdt.sh).

    python train_rdt.py --task hopper-medium-v2 --seed 42 --wmse-action-coef 0.1 \
        --wmse-reward-coef 0 --reward-loss-coef 0.1 --embedding-dropout 0.05 \
        --save-path models_rdt_tuned/seed_42/rdt_hopper-medium-v2.pt

train_rdt.sh runs this for every (seed, task); eval_rdt.py evaluates the checkpoints.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model

# Datasets, environments and seeding are shared with RHDT (../training)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'training'))
from data import TASKS, load_task  # noqa: E402
from train_rhdt import set_seed  # noqa: E402


def reset_seed_for_model(seed):
    """Reseed every generator right before the model is created, for a deterministic init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== MODEL ====================

class RobustDecisionTransformer(nn.Module):
    """Decision Transformer on a GPT-2 backbone with embedding dropout and a reward head."""

    def __init__(self, state_dim, action_dim, embed_dim=128, num_layers=3, num_heads=1,
                 max_ep_len=1000, seq_len=20, dropout=0.1, embedding_dropout=0.1,
                 action_tanh=True, predict_reward=True, device='cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.max_ep_len = max_ep_len
        self.predict_reward = predict_reward
        self.device = device

        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)

        self.embed_ln = nn.LayerNorm(embed_dim)
        self.out_ln = nn.LayerNorm(embed_dim)

        if embedding_dropout > 0:
            self.emb_drop = nn.Dropout(embedding_dropout)
        else:
            self.emb_drop = None

        config = GPT2Config(
            vocab_size=1, n_embd=embed_dim, n_layer=num_layers, n_head=num_heads,
            n_inner=4*embed_dim, activation_function='relu',
            resid_pdrop=dropout, attn_pdrop=dropout, n_positions=1024,
        )
        self.transformer = GPT2Model(config)

        self.predict_dropout = nn.Dropout(dropout)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )

        if predict_reward:
            self.reward_head = nn.Linear(embed_dim, 1)

        self.to(device)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_len, self.embed_dim)

        stacked_inputs = self.embed_ln(stacked_inputs)

        if self.emb_drop is not None:
            stacked_inputs = self.emb_drop(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_len)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs.last_hidden_state

        x = self.out_ln(x)
        x = self.predict_dropout(x)

        x = x.reshape(batch_size, seq_len, 3, self.embed_dim)

        out_states = x[:, :, 1, :]
        out_actions = x[:, :, 2, :]

        action_preds = self.action_head(out_states)

        if self.predict_reward:
            reward_preds = self.reward_head(out_actions)
        else:
            reward_preds = None

        return action_preds, reward_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if states.shape[1] > self.seq_len:
            states = states[:, -self.seq_len:]
            actions = actions[:, -self.seq_len:]
            returns_to_go = returns_to_go[:, -self.seq_len:]
            timesteps = timesteps[:, -self.seq_len:]

        seq_len = states.shape[1]
        if seq_len < self.seq_len:
            pad_len = self.seq_len - seq_len

            states = torch.cat([
                torch.zeros((1, pad_len, self.state_dim), device=states.device),
                states
            ], dim=1)

            actions = torch.cat([
                torch.ones((1, pad_len, self.action_dim), device=actions.device) * -10.,
                actions
            ], dim=1)

            returns_to_go = torch.cat([
                torch.zeros((1, pad_len, 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1)

            timesteps = torch.cat([
                torch.zeros((1, pad_len), device=timesteps.device, dtype=torch.long),
                timesteps
            ], dim=1)

            attention_mask = torch.cat([
                torch.zeros((1, pad_len), device=states.device),
                torch.ones((1, seq_len), device=states.device)
            ], dim=1)
        else:
            attention_mask = None

        action_preds, _ = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask
        )

        return action_preds[0, -1]


# ==================== TRAINING ====================

def weighted_mse_loss(predicted, target, mask, coef=1.0):
    """MSE in which each (sample, timestep) is weighted by exp(-coef x its mean squared error)."""
    with torch.no_grad():
        diff = torch.square(predicted.detach() - target.detach()).mean(-1, keepdim=True)
        weight = torch.exp(-coef * diff)

    loss = F.mse_loss(predicted, target.detach(), reduction="none")
    loss = loss * weight
    return (loss * mask.unsqueeze(-1)).mean()


class RDTTrainer:
    """AdamW with linear warm-up and gradient clipping; action loss plus weighted reward loss."""

    def __init__(self, rdt, learning_rate=1e-4, weight_decay=1e-4, warmup_steps=10000,
                 use_weighted_loss=True, wmse_action_coef=1.0, wmse_reward_coef=1.0,
                 reward_loss_coef=1.0):
        self.rdt = rdt
        self.use_weighted_loss = use_weighted_loss
        self.wmse_action_coef = wmse_action_coef
        self.wmse_reward_coef = wmse_reward_coef
        self.reward_loss_coef = reward_loss_coef

        self.optimizer = AdamW(rdt.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    def train_step(self, batch):
        self.rdt.train()
        actions = batch['actions']
        rewards = batch['rewards']
        mask = batch['attention_mask']

        action_preds, reward_preds = self.rdt(batch['states'], actions, batch['returns_to_go'],
                                              batch['timesteps'], mask)

        # beta = 0 turns a weighted MSE into the plain MSE over unpadded timesteps
        if self.use_weighted_loss and self.wmse_action_coef > 0:
            action_loss = weighted_mse_loss(action_preds, actions, mask, self.wmse_action_coef)
        else:
            action_loss = torch.sum((action_preds - actions) ** 2 * mask.unsqueeze(-1)) / torch.sum(mask.unsqueeze(-1))
        total_loss = action_loss

        reward_loss = torch.tensor(0.0, device=self.rdt.device)
        if self.rdt.predict_reward:
            if self.use_weighted_loss and self.wmse_reward_coef > 0:
                reward_loss = weighted_mse_loss(reward_preds, rewards, mask, self.wmse_reward_coef)
            else:
                reward_loss = (torch.sum((reward_preds - rewards) ** 2 * mask.unsqueeze(-1))
                               / torch.sum(mask.unsqueeze(-1)))
            total_loss = total_loss + self.reward_loss_coef * reward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rdt.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'reward_loss': reward_loss.item() if self.rdt.predict_reward else 0.0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }


# ==================== DATA ====================

def create_dataset_with_rewards(dataset, seq_len=20, max_ep_len=1000, scale=1000.0):
    """Split the dataset into trajectories with normalized states and scaled rewards and returns-to-go."""
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']

    obs_mean = np.mean(observations, axis=0)
    obs_std = np.std(observations, axis=0) + 1e-6

    observations = (observations - obs_mean) / obs_std

    N = len(observations)

    if np.sum(terminals) == 0:
        terminals = np.copy(terminals)
        for i in range(max_ep_len - 1, N, max_ep_len):
            terminals[i] = 1.0

    returns_to_go = np.zeros_like(rewards)
    prev_return = 0
    for i in reversed(range(N)):
        returns_to_go[i] = rewards[i] + prev_return * (1 - terminals[i])
        prev_return = returns_to_go[i]

    returns_to_go = returns_to_go / scale
    scaled_rewards = rewards / scale

    episode_starts = [0]
    for i in range(1, N):
        if terminals[i - 1]:
            episode_starts.append(i)
    episode_starts.append(N)

    trajectories = []
    for i in range(len(episode_starts) - 1):
        start = episode_starts[i]
        end = episode_starts[i + 1]

        trajectories.append({
            'observations': observations[start:end],
            'actions': actions[start:end],
            'rewards': scaled_rewards[start:end],
            'returns_to_go': returns_to_go[start:end],
            'timesteps': np.arange(end - start),
            'length': end - start
        })

    traj_lens = np.array([traj['length'] for traj in trajectories])
    p_sample = traj_lens / traj_lens.sum()

    return {
        'trajectories': trajectories,
        'p_sample': p_sample,
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'scale': scale,
        'target_return': np.max(returns_to_go),
    }


def get_batch_with_rewards(dataset_info, batch_size, seq_len, max_ep_len, device, rng):
    """Batch of subsequences with their rewards, left-padded and masked."""
    trajectories = dataset_info['trajectories']
    batch_inds = rng.choice(len(trajectories), size=batch_size, replace=True, p=dataset_info['p_sample'])

    states, actions, returns, rewards, timesteps, masks = [], [], [], [], [], []
    for idx in batch_inds:
        traj = trajectories[idx]
        traj_len = traj['length']
        si = rng.randint(0, traj_len)

        s = traj['observations'][si:si + seq_len]
        a = traj['actions'][si:si + seq_len]
        rtg = traj['returns_to_go'][si:si + seq_len]
        r = traj['rewards'][si:si + seq_len]
        ts = traj['timesteps'][si:si + seq_len]
        ts[ts >= max_ep_len] = max_ep_len - 1

        tlen = len(s)
        if tlen < seq_len:
            s = np.concatenate([np.zeros((seq_len - tlen, s.shape[1])), s])
            a = np.concatenate([np.ones((seq_len - tlen, a.shape[1])) * -10., a])
            rtg = np.concatenate([np.zeros(seq_len - tlen), rtg])
            r = np.concatenate([np.zeros(seq_len - tlen), r])
            ts = np.concatenate([np.zeros(seq_len - tlen, dtype=int), ts])
            mask = np.concatenate([np.zeros(seq_len - tlen), np.ones(tlen)])
        else:
            mask = np.ones(seq_len)

        states.append(s)
        actions.append(a)
        returns.append(rtg)
        rewards.append(r)
        timesteps.append(ts)
        masks.append(mask)

    return {
        'states': torch.FloatTensor(np.array(states)).to(device),
        'actions': torch.FloatTensor(np.array(actions)).to(device),
        'returns_to_go': torch.FloatTensor(np.array(returns)).unsqueeze(-1).to(device),
        'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(device),
        'timesteps': torch.LongTensor(np.array(timesteps)).to(device),
        'attention_mask': torch.FloatTensor(np.array(masks)).to(device),
    }


# ==================== MAIN ====================

def parse_args():
    p = argparse.ArgumentParser(description='Train one RDT model.')
    p.add_argument('--task', default='halfcheetah-medium-v2', choices=TASKS, metavar='TASK',
                   help=f"D4RL task, one of {', '.join(TASKS)} (default: %(default)s)")
    p.add_argument('--seed', type=int, default=42, help='training seed (default: %(default)s)')
    p.add_argument('--device', default='cuda', help='torch device (default: %(default)s)')
    p.add_argument('--offline-steps', type=int, default=100000, help='training steps (default: %(default)s)')
    p.add_argument('--wmse-action-coef', type=float, default=0.1,
                   help='beta_a of the Gaussian-weighted action loss; 0 = plain MSE (default: %(default)s)')
    p.add_argument('--wmse-reward-coef', type=float, default=0.1,
                   help='beta_r of the Gaussian-weighted reward loss; 0 = plain MSE (default: %(default)s)')
    p.add_argument('--reward-loss-coef', type=float, default=0.1,
                   help='weight of the reward-prediction loss; 0 = reward head unused (default: %(default)s)')
    p.add_argument('--embedding-dropout', type=float, default=0.1,
                   help='dropout on the token embeddings (default: %(default)s)')
    p.add_argument('--save-path', required=True,
                   help='checkpoint (.pt) to write; its metadata goes to the same path with .json')
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    set_seed(args.seed)

    print("=" * 80)
    print(f"RDT  task={args.task}  seed={args.seed}  steps={args.offline_steps}")
    print(f"beta_a={args.wmse_action_coef}  beta_r={args.wmse_reward_coef}  "
          f"reward loss weight={args.reward_loss_coef}  embedding dropout={args.embedding_dropout}")
    print("=" * 80)

    dataset, env = load_task(args.task)
    env.close()  # training only; eval_rdt.py makes its own environment
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    model_path = args.save_path
    json_path = os.path.splitext(model_path)[0] + '.json'
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    metadata = {
        'task': args.task,
        'model_type': 'rdt',
        'offline_steps': args.offline_steps,
        'seed': args.seed,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'embed_dim': 128,
        'num_layers': 3,
        'num_heads': 1,
        'seq_len': 20,
        'max_ep_len': 1000,
        'dropout': 0.1,
        'embedding_dropout': args.embedding_dropout,
        'use_weighted_loss': True,
        'wmse_action_coef': args.wmse_action_coef,
        'wmse_reward_coef': args.wmse_reward_coef,
        'reward_loss_coef': args.reward_loss_coef,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 10000,
        'batch_size': 64,
        'timestamp': datetime.now().isoformat(),
    }

    dataset_info = create_dataset_with_rewards(dataset, seq_len=metadata['seq_len'],
                                               max_ep_len=metadata['max_ep_len'], scale=1000.0)

    reset_seed_for_model(args.seed)
    rdt = RobustDecisionTransformer(
        state_dim=state_dim, action_dim=action_dim, embed_dim=metadata['embed_dim'],
        num_layers=metadata['num_layers'], num_heads=metadata['num_heads'],
        max_ep_len=metadata['max_ep_len'], seq_len=metadata['seq_len'], dropout=metadata['dropout'],
        embedding_dropout=metadata['embedding_dropout'], action_tanh=True, predict_reward=True,
        device=args.device)
    train_rng = np.random.RandomState(args.seed)
    trainer = RDTTrainer(
        rdt, learning_rate=metadata['learning_rate'], weight_decay=metadata['weight_decay'],
        warmup_steps=metadata['warmup_steps'], use_weighted_loss=metadata['use_weighted_loss'],
        wmse_action_coef=metadata['wmse_action_coef'], wmse_reward_coef=metadata['wmse_reward_coef'],
        reward_loss_coef=metadata['reward_loss_coef'])

    pbar = tqdm(range(args.offline_steps), desc="Training RDT")
    for step in pbar:
        batch = get_batch_with_rewards(dataset_info, metadata['batch_size'], metadata['seq_len'],
                                       metadata['max_ep_len'], args.device, train_rng)
        losses = trainer.train_step(batch)
        if (step + 1) % 100 == 0:
            pbar.set_postfix({'loss': f"{losses['loss']:.4f}", 'act': f"{losses['action_loss']:.4f}"})

    metadata.update({
        'obs_mean': dataset_info['obs_mean'].tolist(),
        'obs_std': dataset_info['obs_std'].tolist(),
        'target_return': float(dataset_info['target_return']),
        'scale': float(dataset_info['scale']),
    })

    # Temp file + rename, so an interrupted save never looks like a finished run
    torch.save({'model_state_dict': rdt.state_dict(), 'metadata': metadata}, model_path + '.tmp')
    os.replace(model_path + '.tmp', model_path)
    with open(json_path + '.tmp', 'w') as f:
        json.dump(metadata, f, indent=2)
    os.replace(json_path + '.tmp', json_path)
    print(f"Saved to: {model_path} (metadata: {json_path})")


if __name__ == "__main__":
    main()
