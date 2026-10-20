"""
D4RL MuJoCo locomotion data for RHDT.

Offline trajectories come from the Hugging Face copy of D4RL used by Decision
Transformer; the evaluation environments come from D4RL (Gym + mujoco-py).
"""

import os

import gym
import numpy as np
import torch

HF_DATASET = 'edbeeching/decision_transformer_gym_replay'

# Tasks available in HF_DATASET
TASKS = tuple(f'{env}-{split}-v2'
              for env in ('halfcheetah', 'hopper', 'walker2d')
              for split in ('medium', 'medium-replay', 'expert'))

# D4RL reference returns (random policy, expert policy) used for normalization
REF_RETURNS = {
    'halfcheetah': (-280.178377, 12135.0),
    'hopper': (-20.272305, 3234.3),
    'walker2d': (1.629008, 4592.3),
}


def normalized_score(task, ret):
    """D4RL normalized score: 0 = random policy, 100 = expert policy."""
    random_ret, expert_ret = REF_RETURNS[task.split('-')[0]]
    return 100 * (ret - random_ret) / (expert_ret - random_ret)


def make_env(task):
    """Gym environment of a D4RL task."""
    os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')  # skip warnings about unused D4RL suites
    try:
        import d4rl  # noqa: F401  (registers the D4RL environments with Gym)
    except ImportError as e:
        raise ImportError('D4RL is needed for the evaluation environments; '
                          'see environment.yml for the tested versions.') from e
    return gym.make(task)


def load_task(task):
    """Return the offline dataset (dict of flat arrays) and the Gym environment."""
    from datasets import load_dataset

    observations, actions, rewards, terminals = [], [], [], []
    for traj in load_dataset(HF_DATASET, task)['train']:
        observations.append(np.array(traj['observations']))
        actions.append(np.array(traj['actions']))
        rewards.append(np.array(traj['rewards']))
        terminals.append(np.array(traj['dones']))
    dataset = {
        'observations': np.vstack(observations),
        'actions': np.vstack(actions),
        'rewards': np.concatenate(rewards),
        'terminals': np.concatenate(terminals),
    }
    print(f"Loaded {task}: {len(dataset['observations'])} transitions from {HF_DATASET}")
    return dataset, make_env(task)


def build_trajectories(dataset, max_ep_len=1000, scale=1000.0):
    """Split the dataset into trajectories with normalized states and scaled returns-to-go."""
    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']

    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    obs = (obs - obs_mean) / obs_std

    N = len(obs)
    if terminals.sum() == 0:  # no episode boundaries: cut every max_ep_len steps
        terminals = np.zeros_like(terminals)
        for i in range(max_ep_len - 1, N, max_ep_len):
            terminals[i] = 1.0

    rtg = np.zeros_like(rewards)
    running = 0
    for i in reversed(range(N)):
        rtg[i] = rewards[i] + running * (1 - terminals[i])
        running = rtg[i]
    rtg /= scale

    starts = [0] + [i + 1 for i in range(N - 1) if terminals[i]] + [N]
    trajs = []
    for s, e in zip(starts[:-1], starts[1:]):
        trajs.append({
            'obs': obs[s:e], 'actions': actions[s:e],
            'rtg': rtg[s:e], 'timesteps': np.arange(e - s), 'len': e - s,
        })

    lens = np.array([t['len'] for t in trajs])
    return {
        'trajs': trajs,
        'p_sample': lens / lens.sum(),  # sample trajectories in proportion to length
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'scale': scale,
        'target_return': rtg.max(),  # condition evaluation on the best return in the data
    }


def sample_batch(data, batch_size, seq_len, max_ep_len, device, rng):
    """Sample `batch_size` subsequences of length `seq_len`, left-padded and masked."""
    trajs, p = data['trajs'], data['p_sample']
    idxs = rng.choice(len(trajs), batch_size, p=p)

    states, actions, rtgs, timesteps, masks = [], [], [], [], []
    for idx in idxs:
        t = trajs[idx]
        si = rng.randint(0, t['len'])

        s = t['obs'][si:si + seq_len]
        a = t['actions'][si:si + seq_len]
        r = t['rtg'][si:si + seq_len]
        ts = np.clip(t['timesteps'][si:si + seq_len], 0, max_ep_len - 1)

        L = len(s)
        if L < seq_len:
            pad = seq_len - L
            s = np.vstack([np.zeros((pad, s.shape[1])), s])
            a = np.vstack([np.ones((pad, a.shape[1])) * -10, a])
            r = np.concatenate([np.zeros(pad), r])
            ts = np.concatenate([np.zeros(pad, dtype=int), ts])
            mask = np.concatenate([np.zeros(pad), np.ones(L)])
        else:
            mask = np.ones(seq_len)

        states.append(s)
        actions.append(a)
        rtgs.append(r)
        timesteps.append(ts)
        masks.append(mask)

    return {
        'states': torch.FloatTensor(np.array(states)).to(device),
        'actions': torch.FloatTensor(np.array(actions)).to(device),
        'rtg': torch.FloatTensor(np.array(rtgs)).unsqueeze(-1).to(device),
        'timesteps': torch.LongTensor(np.array(timesteps)).to(device),
        'mask': torch.FloatTensor(np.array(masks)).to(device),
    }
