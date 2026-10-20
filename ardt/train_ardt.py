"""
Adversarially Robust Decision Transformer (ARDT) baseline: train one model.

Reimplementation of Tang et al., "Adversarially Robust Decision Transformer",
NeurIPS 2024 (https://arxiv.org/abs/2407.18414):
  1. Warm-up: two value networks regress the dataset's returns-to-go.
  2. Minimax estimation: alternating expectile regression. The lower network
     (expectile alpha) plays the adversary, the upper one (1 - alpha) the
     protagonist's best response.
  3. Every trajectory is relabelled with the upper network's minimax returns-to-go.
  4. A Decision Transformer is trained on the relabelled trajectories.

    python train_ardt.py --task hopper-medium-v2 --seed 42 --alpha 0.8 --minimax-steps 10000 \
        --save-path models_ardt_tuned/seed_42/ardt_hopper-medium-v2.pt

train_ardt.sh runs this for every (seed, task) with each task's settings; eval_ardt.py
evaluates the checkpoints.
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

# Datasets, environments, seeding and rollouts are shared with RHDT (../training)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'training'))
from data import TASKS, load_task  # noqa: E402
from train_rhdt import evaluate, set_seed  # noqa: E402

from ckpt_selection import CheckpointSelector, add_selection_args, make_validator  # noqa: E402


def reset_seed_for_model(seed):
    """Reseed every generator right before a model is created, for a deterministic init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== MINIMAX RETURN ESTIMATION ====================

class ExpectileValueNetwork(nn.Module):
    """Q(s, a) network used for the expectile regression."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, n_layers=3, device='cuda'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, states, actions):
        return self.network(torch.cat([states, actions], dim=-1))


def expectile_loss(pred, target, expectile):
    """Expectile regression loss |tau - 1(u < 0)| * u^2 with u = target - pred."""
    diff = target - pred
    weight = torch.where(diff >= 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


class MinimaxReturnEstimator:
    """Minimax returns-to-go from two alternately trained expectile networks.

    q_lower (expectile alpha) estimates the adversary's worst case, q_upper
    (expectile 1 - alpha) the protagonist's best response to it.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, n_layers=3,
                 alpha=0.1, learning_rate=3e-4, device='cuda'):
        self.alpha = alpha
        self.device = device
        self.q_lower = ExpectileValueNetwork(state_dim, action_dim, hidden_dim, n_layers, device)
        self.q_upper = ExpectileValueNetwork(state_dim, action_dim, hidden_dim, n_layers, device)
        self.optimizer_lower = AdamW(self.q_lower.parameters(), lr=learning_rate)
        self.optimizer_upper = AdamW(self.q_upper.parameters(), lr=learning_rate)

    def warmup_step(self, states, actions, returns_to_go):
        """Regress both networks on the dataset returns-to-go (MSE = expectile 0.5)."""
        self.q_lower.train()
        self.q_upper.train()

        loss_lower = F.mse_loss(self.q_lower(states, actions), returns_to_go)
        loss_upper = F.mse_loss(self.q_upper(states, actions), returns_to_go)

        self.optimizer_lower.zero_grad()
        loss_lower.backward()
        self.optimizer_lower.step()

        self.optimizer_upper.zero_grad()
        loss_upper.backward()
        self.optimizer_upper.step()

        return {'warmup_loss_lower': loss_lower.item(), 'warmup_loss_upper': loss_upper.item()}

    def minimax_step(self, states, actions, rewards, next_states, next_actions, dones, gamma=0.99):
        """One alternating update: each network bootstraps from the other one's next-state values."""
        self.q_lower.train()
        self.q_upper.train()

        with torch.no_grad():
            next_values_upper = self.q_upper(next_states, next_actions)
            target_lower = rewards + gamma * (1 - dones.float()) * next_values_upper
            next_values_lower = self.q_lower(next_states, next_actions)
            target_upper = rewards + gamma * (1 - dones.float()) * next_values_lower

        pred_lower = self.q_lower(states, actions)
        pred_upper = self.q_upper(states, actions)
        loss_lower = expectile_loss(pred_lower, target_lower, self.alpha)      # adversary: pessimistic
        loss_upper = expectile_loss(pred_upper, target_upper, 1 - self.alpha)  # best response

        self.optimizer_lower.zero_grad()
        loss_lower.backward()
        self.optimizer_lower.step()

        self.optimizer_upper.zero_grad()
        loss_upper.backward()
        self.optimizer_upper.step()

        return {'minimax_loss_lower': loss_lower.item(), 'minimax_loss_upper': loss_upper.item()}

    def estimate_minimax_returns(self, states, actions):
        """Minimax returns-to-go for relabelling: the upper network's values."""
        self.q_upper.eval()
        with torch.no_grad():
            return self.q_upper(states, actions)


# ==================== DECISION TRANSFORMER ====================

class DecisionTransformer(nn.Module):
    """Decision Transformer (Chen et al., 2021) on a GPT-2 backbone."""

    def __init__(self, state_dim, action_dim, embed_dim=128, num_layers=3, num_heads=1,
                 seq_len=20, max_ep_len=1000, dropout=0.1, action_tanh=True, device='cuda'):
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

        config = GPT2Config(
            vocab_size=1, n_embd=embed_dim, n_layer=num_layers, n_head=num_heads,
            n_inner=4 * embed_dim, activation_function='relu',
            resid_pdrop=dropout, attn_pdrop=dropout, n_positions=1024,
        )
        self.transformer = GPT2Model(config)

        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
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

        # Interleave as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embed_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_len)

        x = self.transformer(inputs_embeds=stacked_inputs,
                             attention_mask=stacked_attention_mask).last_hidden_state
        x = x.reshape(batch_size, seq_len, 3, self.embed_dim).permute(0, 2, 1, 3)
        return self.predict_action(x[:, 1])  # predict a_t from the s_t token

    def get_action(self, states, actions, returns_to_go, timesteps):
        """Action for the last timestep of a (possibly shorter) context."""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if states.shape[1] > self.seq_len:
            states = states[:, -self.seq_len:]
            actions = actions[:, -self.seq_len:]
            returns_to_go = returns_to_go[:, -self.seq_len:]
            timesteps = timesteps[:, -self.seq_len:]

        # Left-pad short contexts to seq_len and mask the padding
        seq_length = states.shape[1]
        if seq_length < self.seq_len:
            pad = self.seq_len - seq_length
            attention_mask = torch.cat([
                torch.zeros(pad, device=self.device),
                torch.ones(seq_length, device=self.device)
            ]).to(dtype=torch.long).reshape(1, -1)
            states = torch.cat([
                torch.zeros((1, pad, self.state_dim), device=self.device), states
            ], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.ones((1, pad, self.action_dim), device=self.device) * -10., actions
            ], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros((1, pad, 1), device=self.device), returns_to_go
            ], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros((1, pad), device=self.device), timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        return self.forward(states, actions, returns_to_go, timesteps, attention_mask)[0, -1]


class DTTrainer:
    """AdamW with linear warm-up and gradient clipping; masked MSE on actions."""

    def __init__(self, dt, learning_rate=1e-4, weight_decay=1e-4, warmup_steps=10000):
        self.dt = dt
        self.optimizer = AdamW(dt.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    def train_step(self, batch):
        self.dt.train()
        mask = batch['attention_mask']
        action_preds = self.dt(batch['states'], batch['actions'], batch['returns_to_go'],
                               batch['timesteps'], mask)
        mask = mask.unsqueeze(-1)
        loss = torch.sum((action_preds - batch['action_targets']) ** 2 * mask) / torch.sum(mask)

        if torch.isnan(loss) or torch.isinf(loss):  # skip the update
            return {'loss': float('nan'), 'lr': self.optimizer.param_groups[0]['lr']}

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        return {'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']}


# ==================== DATA ====================

def discount_cumsum(x, gamma):
    """Discounted cumulative sums from every step to the end."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def create_dataset_for_ardt(dataset, max_ep_len=1000, discount=1.0, scale=1000.0):
    """Split the dataset into trajectories with scaled rewards and returns-to-go."""
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    N = len(observations)

    obs_mean = np.mean(observations, axis=0)
    obs_std = np.std(observations, axis=0) + 1e-6

    if np.sum(terminals) == 0:  # no episode boundaries: cut every max_ep_len steps
        terminals = np.copy(terminals)
        for i in range(max_ep_len - 1, N, max_ep_len):
            terminals[i] = 1.0

    episode_starts = [0] + [i for i in range(1, N) if terminals[i - 1]] + [N]
    print(f"Found {len(episode_starts) - 1} episodes in the dataset")

    trajectories = []
    for start, end in zip(episode_starts[:-1], episode_starts[1:]):
        trajectories.append({
            'observations': observations[start:end],
            'actions': actions[start:end],
            'rewards': rewards[start:end],
            'returns_to_go': discount_cumsum(rewards[start:end], gamma=discount),
            'length': end - start,
        })

    all_returns = np.concatenate([traj['returns_to_go'] for traj in trajectories])
    for traj in trajectories:
        traj['returns_to_go'] = traj['returns_to_go'] / scale
        traj['rewards'] = traj['rewards'] / scale

    traj_lens = np.array([traj['length'] for traj in trajectories])
    return {
        'trajectories': trajectories,
        'p_sample': traj_lens / traj_lens.sum(),  # sample trajectories in proportion to length
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'target_return': np.max(all_returns / scale),
        'scale': scale,
    }


def get_batch_for_minimax(dataset_info, batch_size, device, rng):
    """Batch of single transitions (s, a, r, s', a', done) and the dataset returns-to-go."""
    trajectories = dataset_info['trajectories']
    obs_mean = dataset_info['obs_mean']
    obs_std = dataset_info['obs_std']
    batch_inds = rng.choice(len(trajectories), size=batch_size, replace=True, p=dataset_info['p_sample'])

    states, actions, rewards, next_states, next_actions, dones, rtg = [], [], [], [], [], [], []
    for idx in batch_inds:
        traj = trajectories[idx]
        traj_len = traj['length']
        si = rng.randint(0, max(1, traj_len - 1))

        s = (traj['observations'][si] - obs_mean) / obs_std
        a = traj['actions'][si]
        if si + 1 < traj_len:
            ns = (traj['observations'][si + 1] - obs_mean) / obs_std
            na = traj['actions'][si + 1]
            done = False
        else:
            ns, na, done = s, a, True

        states.append(s)
        actions.append(a)
        rewards.append(traj['rewards'][si])
        next_states.append(ns)
        next_actions.append(na)
        dones.append(done)
        rtg.append(traj['returns_to_go'][si])

    return {
        'states': torch.FloatTensor(np.array(states)).to(device),
        'actions': torch.FloatTensor(np.array(actions)).to(device),
        'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(device),
        'next_states': torch.FloatTensor(np.array(next_states)).to(device),
        'next_actions': torch.FloatTensor(np.array(next_actions)).to(device),
        'dones': torch.BoolTensor(np.array(dones)).unsqueeze(-1).to(device),
        'returns_to_go': torch.FloatTensor(np.array(rtg)).unsqueeze(-1).to(device),
    }


def relabel_trajectories_with_minimax(dataset_info, estimator, device):
    """Store the estimator's minimax returns-to-go in every trajectory."""
    obs_mean = dataset_info['obs_mean']
    obs_std = dataset_info['obs_std']
    for traj in tqdm(dataset_info['trajectories'], desc="Relabeling"):
        states = torch.FloatTensor((traj['observations'] - obs_mean) / obs_std).to(device)
        actions = torch.FloatTensor(traj['actions']).to(device)
        minimax_rtg = estimator.estimate_minimax_returns(states, actions)
        traj['minimax_returns_to_go'] = minimax_rtg.cpu().numpy().flatten()


def get_batch_for_dt(dataset_info, batch_size, seq_len, max_ep_len, device, rng):
    """Batch of subsequences conditioned on the minimax returns-to-go, left-padded and masked."""
    trajectories = dataset_info['trajectories']
    obs_mean = dataset_info['obs_mean']
    obs_std = dataset_info['obs_std']
    batch_inds = rng.choice(len(trajectories), size=batch_size, replace=True, p=dataset_info['p_sample'])

    s, a, rtg, timesteps, mask = [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[batch_inds[i]]
        si = rng.randint(0, traj['length'])

        traj_s = traj['observations'][si:si + seq_len]
        traj_a = traj['actions'][si:si + seq_len]
        traj_rtg = traj['minimax_returns_to_go'][si:si + seq_len]
        traj_timesteps = np.arange(si, si + len(traj_s))
        traj_timesteps[traj_timesteps >= max_ep_len] = max_ep_len - 1

        tlen = len(traj_s)
        if tlen < seq_len:
            traj_s = np.concatenate([np.zeros((seq_len - tlen, traj_s.shape[1])), traj_s], axis=0)
            traj_a = np.concatenate([np.ones((seq_len - tlen, traj_a.shape[1])) * -10., traj_a], axis=0)
            traj_rtg = np.concatenate([np.zeros(seq_len - tlen), traj_rtg], axis=0)
            traj_timesteps = np.concatenate([np.zeros(seq_len - tlen, dtype=int), traj_timesteps], axis=0)
            traj_mask = np.concatenate([np.zeros(seq_len - tlen), np.ones(tlen)], axis=0)
        else:
            traj_mask = np.ones(seq_len)

        s.append((traj_s - obs_mean) / obs_std)
        a.append(traj_a)
        rtg.append(traj_rtg)
        timesteps.append(traj_timesteps)
        mask.append(traj_mask)

    return {
        'states': torch.FloatTensor(np.array(s)).to(device),
        'actions': torch.FloatTensor(np.array(a)).to(device),
        'returns_to_go': torch.FloatTensor(np.array(rtg)).unsqueeze(-1).to(device),
        'timesteps': torch.LongTensor(np.array(timesteps)).to(device),
        'action_targets': torch.FloatTensor(np.array(a)).to(device),
        'attention_mask': torch.FloatTensor(np.array(mask)).to(device),
    }


# ==================== MAIN ====================

def parse_args():
    p = argparse.ArgumentParser(description='Train one ARDT model.')
    p.add_argument('--task', default='halfcheetah-medium-v2', choices=TASKS, metavar='TASK',
                   help=f"D4RL task, one of {', '.join(TASKS)} (default: %(default)s)")
    p.add_argument('--seed', type=int, default=42, help='training seed (default: %(default)s)')
    p.add_argument('--device', default='cuda', help='torch device (default: %(default)s)')
    p.add_argument('--alpha', type=float, default=0.1,
                   help='expectile level of the adversary; smaller = more conservative (default: %(default)s)')
    p.add_argument('--warmup-steps', type=int, default=5000,
                   help='value-network warm-up steps (default: %(default)s)')
    p.add_argument('--minimax-steps', type=int, default=20000,
                   help='minimax expectile-regression steps (default: %(default)s)')
    p.add_argument('--offline-steps', type=int, default=100000,
                   help='Decision Transformer training steps (default: %(default)s)')
    add_selection_args(p)
    p.add_argument('--save-path', required=True,
                   help='checkpoint (.pt) to write; its metadata goes to the same path with .json')
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    set_seed(args.seed)

    print("=" * 80)
    print(f"ARDT  task={args.task}  seed={args.seed}  alpha={args.alpha}")
    print(f"Steps: warm-up {args.warmup_steps}, minimax {args.minimax_steps}, DT {args.offline_steps}")
    print("=" * 80)

    dataset, env = load_task(args.task)
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    model_path = args.save_path
    json_path = os.path.splitext(model_path)[0] + '.json'
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    metadata = {
        'task': args.task,
        'model_type': 'ardt',
        'alpha': args.alpha,
        'warmup_steps': args.warmup_steps,
        'minimax_steps': args.minimax_steps,
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
        'timestamp': datetime.now().isoformat(),
    }

    dataset_info = create_dataset_for_ardt(dataset, max_ep_len=1000)

    # ---- Phase 1: minimax returns-to-go ----
    reset_seed_for_model(args.seed)
    estimator = MinimaxReturnEstimator(state_dim, action_dim, hidden_dim=256, n_layers=3,
                                       alpha=args.alpha, learning_rate=3e-4, device=args.device)
    train_rng = np.random.RandomState(args.seed)

    pbar = tqdm(range(args.warmup_steps), desc="Warmup")
    for step in pbar:
        batch = get_batch_for_minimax(dataset_info, 256, args.device, train_rng)
        losses = estimator.warmup_step(batch['states'], batch['actions'], batch['returns_to_go'])
        if (step + 1) % 100 == 0:
            pbar.set_postfix({'loss_L': f"{losses['warmup_loss_lower']:.4f}",
                              'loss_U': f"{losses['warmup_loss_upper']:.4f}"})

    pbar = tqdm(range(args.minimax_steps), desc="Minimax")
    for step in pbar:
        batch = get_batch_for_minimax(dataset_info, 256, args.device, train_rng)
        losses = estimator.minimax_step(batch['states'], batch['actions'], batch['rewards'],
                                        batch['next_states'], batch['next_actions'], batch['dones'])
        if (step + 1) % 100 == 0:
            pbar.set_postfix({'loss_L': f"{losses['minimax_loss_lower']:.4f}",
                              'loss_U': f"{losses['minimax_loss_upper']:.4f}"})

    relabel_trajectories_with_minimax(dataset_info, estimator, args.device)

    # ---- Phase 2: Decision Transformer on the relabelled data ----
    reset_seed_for_model(args.seed + 1000)
    dt = DecisionTransformer(state_dim, action_dim, embed_dim=128, num_layers=3, num_heads=1,
                             seq_len=20, max_ep_len=1000, dropout=0.1, action_tanh=True, device=args.device)
    trainer = DTTrainer(dt)
    train_rng = np.random.RandomState(args.seed + 2000)

    # Evaluation and validation are conditioned on the highest minimax return-to-go
    target_return = float(np.max(np.concatenate(
        [traj['minimax_returns_to_go'] for traj in dataset_info['trajectories']])))
    validate = make_validator(evaluate, env, target_return, dataset_info['obs_mean'], dataset_info['obs_std'],
                              dataset_info['scale'], args.task, args.val_episodes, args.val_seed)
    selector = CheckpointSelector(args.offline_steps, args.ckpt_interval, args.ckpt_keep_last, validate,
                                  enabled=(args.select_metric != 'none'), metric=args.select_metric,
                                  n_episodes=args.val_episodes, val_seed=args.val_seed)

    pbar = tqdm(range(args.offline_steps), desc="Training ARDT")
    selector.log = pbar.write
    for step in pbar:
        batch = get_batch_for_dt(dataset_info, 64, 20, 1000, args.device, train_rng)
        losses = trainer.train_step(batch)
        selector.maybe_validate(dt, step + 1)
        if (step + 1) % 100 == 0:
            pbar.set_postfix({'loss': f"{losses['loss']:.4f}"})

    metadata.update({
        'obs_mean': dataset_info['obs_mean'].tolist(),
        'obs_std': dataset_info['obs_std'].tolist(),
        'target_return': target_return,
        'original_target_return': float(dataset_info['target_return']),
        'scale': float(dataset_info['scale']),
    })
    selector.select(dt)
    metadata.update(selector.metadata())

    # Temp file + rename, so an interrupted save never looks like a finished run
    torch.save({'model_state_dict': dt.state_dict(), 'metadata': metadata}, model_path + '.tmp')
    os.replace(model_path + '.tmp', model_path)
    with open(json_path + '.tmp', 'w') as f:
        json.dump(metadata, f, indent=2)
    os.replace(json_path + '.tmp', json_path)
    print(f"Saved to: {model_path} (metadata: {json_path})")

    env.close()


if __name__ == "__main__":
    main()
