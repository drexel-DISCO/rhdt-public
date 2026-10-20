"""
Train, and optionally evaluate, one RHDT model on one D4RL locomotion task.

    python train_rhdt.py --task hopper-medium-v2 --seed 42 \
        --lambda-gp 0.001 --lambda-sep 0.2 --output-dir runs/hopper --mode train

Training loss: the RDT-style weighted MSE on actions, plus
  * a gradient penalty, lambda_gp * E ||d a_hat / d s||^2, which bounds the
    policy's sensitivity to observation noise, and
  * key orthogonality, lambda_sep * mean((K_n K_n^T - I)^2) over the
    L2-normalized attention keys of every layer, which separates the stored
    Hopfield patterns.

Modes (--mode):
  train  train and save the checkpoint
  eval   evaluate the saved checkpoint; exit code 3 if there is none yet
  all    train if there is no matching checkpoint, then evaluate

Training also writes the loss history next to the checkpoint, rhdt_<task>_losses.json:
each loss term averaged over every 500 steps.

A run whose outputs already match the requested settings is skipped, so an
interrupted sweep can simply be restarted. Use --force to redo it.

train.sh runs this script over seeds, tasks and configs.
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from data import TASKS, build_trajectories, load_task, normalized_score, sample_batch
from model import RHDT

# Fixed hyperparameters (same for every task)
EMBED_DIM = 128
NUM_LAYERS = 3
SEQ_LEN = 20            # context length in timesteps
MAX_EP_LEN = 1000
RTG_SCALE = 1000.0      # returns-to-go are divided by this
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 10000    # linear learning-rate warmup
GRAD_CLIP = 0.25

LOSS_INTERVAL = 500     # the loss history averages each loss term over this many steps
CVAR_ALPHA = 0.1        # CVaR-10: mean of the worst 10% of episodes
# Result keys written by one evaluation; replaced together when a json is re-evaluated
EVAL_KEY_PREFIXES = ('score_', 'return_', 'std_', 'median_', 'cvar10_', 'episode_scores_')
EVAL_KEYS = ('degradation', 'cvar10_degradation', 'eval_episodes', 'corruptions', 'cvar_alpha')

EXIT_NO_CHECKPOINT = 3  # --mode eval and there is no matching trained model yet


def set_seed(seed):
    """Seed every random number generator and make PyTorch deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


# ==================== TRAINING ====================

def k_orthogonality_loss(K):
    """Mean squared deviation of the keys' cosine-similarity matrix from the identity.

    K: [B, H, T, D] keys. Pushes the keys of different tokens toward orthogonality.
    """
    T = K.shape[2]
    K_norm = F.normalize(K, dim=-1)
    gram = torch.matmul(K_norm, K_norm.transpose(-2, -1))  # [B, H, T, T]
    identity = torch.eye(T, device=K.device).unsqueeze(0).unsqueeze(0)
    return ((gram - identity) ** 2).mean()


def train_step(model, optimizer, batch, lambda_gp, lambda_sep):
    model.train()

    states = batch['states'].requires_grad_(True)  # for the gradient penalty
    actions = batch['actions']
    mask = batch['mask']

    action_preds = model(states, actions, batch['rtg'], batch['timesteps'], mask)

    # Weighted MSE: down-weight samples the model fits poorly (likely corrupted)
    diff = (action_preds.detach() - actions).pow(2).mean(-1, keepdim=True)
    weight = torch.exp(-diff)
    action_loss = (weight * (action_preds - actions).pow(2) * mask.unsqueeze(-1)).mean()

    total_loss = action_loss
    gp_loss = torch.tensor(0.0, device=states.device)
    sep_loss = torch.tensor(0.0, device=states.device)

    if lambda_gp > 0:
        grads = torch.autograd.grad(action_preds.sum(), states, create_graph=True)[0]
        gp_loss = (grads ** 2).sum(-1).mean()
        total_loss = total_loss + lambda_gp * gp_loss

    if lambda_sep > 0:
        K_matrices = model.get_K_matrices()
        sep_loss = sum(k_orthogonality_loss(K) for K in K_matrices) / len(K_matrices)
        total_loss = total_loss + lambda_sep * sep_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        'loss': total_loss.item(),
        'action_loss': action_loss.item(),
        'gp_loss': gp_loss.item(),
        'sep_loss': sep_loss.item(),
    }


def train(model, data, args):
    """Train the model. Returns the loss history: the step at the end of every
    LOSS_INTERVAL steps (and the last step) and each loss term averaged over those steps."""
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / WARMUP_STEPS, 1))
    rng = np.random.RandomState(args.seed)
    lambda_sep = args.lambda_sep if args.method != 'none' else 0.0

    # train_step returns the losses as Python floats, so recording them does not
    # change the training
    history = {'step': []}
    sums, n = {}, 0

    # Frequent redraws only make sense on a terminal; in log files they pile up
    pbar = tqdm(range(args.steps), desc="Training", mininterval=0.1 if sys.stderr.isatty() else 60)
    for step in pbar:
        batch = sample_batch(data, BATCH_SIZE, SEQ_LEN, MAX_EP_LEN, args.device, rng)
        losses = train_step(model, optimizer, batch, args.lambda_gp, lambda_sep)
        scheduler.step()

        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + value
        n += 1
        if (step + 1) % LOSS_INTERVAL == 0 or step + 1 == args.steps:
            history['step'].append(step + 1)
            for key, total in sums.items():
                history.setdefault(key, []).append(total / n)
            sums, n = {}, 0

        if (step + 1) % 500 == 0:
            pbar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                gp=f"{losses['gp_loss']:.4f}",
                sep=f"{losses['sep_loss']:.4f}",
            )
    return history


# ==================== EVALUATION ====================

def corrupt_observation(obs, magnitude, obs_std, rng):
    """Gaussian observation noise with std = magnitude x the dataset std of each dimension."""
    return obs + rng.randn(*obs.shape) * obs_std * magnitude


def evaluate(model, env, target_return, obs_mean, obs_std, scale, n_episodes, max_ep_len, corruption, seed):
    """Roll out the policy; every observation is corrupted when corruption > 0.

    Returns the mean and std of the episode returns, and all returns in episode order.
    """
    model.eval()
    returns = []

    obs_mean_t = torch.FloatTensor(obs_mean).to(model.device)
    obs_std_t = torch.FloatTensor(obs_std).to(model.device)

    for ep in range(n_episodes):
        ep_seed = seed + ep * 1000
        rng = np.random.RandomState(ep_seed)
        torch.manual_seed(ep_seed)

        obs = env.reset(seed=ep_seed)
        if isinstance(obs, tuple):  # Gym >= 0.26 returns (obs, info)
            obs = obs[0]

        if corruption > 0:
            obs = corrupt_observation(obs, corruption, obs_std, rng)

        states = torch.FloatTensor(obs).to(model.device).view(1, model.state_dim)
        actions = torch.zeros(0, model.action_dim, device=model.device)
        rtg = torch.tensor([[target_return]], device=model.device, dtype=torch.float32)
        timesteps = torch.tensor([[0]], device=model.device, dtype=torch.long)

        total_reward = 0
        for t in range(max_ep_len):
            actions = torch.cat([actions, torch.zeros(1, model.action_dim, device=model.device)])

            with torch.no_grad():
                action = model.get_action((states - obs_mean_t) / obs_std_t, actions, rtg, timesteps)
            actions[-1] = action

            result = env.step(action.cpu().numpy())
            if len(result) == 4:
                obs, reward, done, _ = result
            else:  # Gym >= 0.26
                obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            total_reward += reward

            if done:
                break

            if corruption > 0:
                obs = corrupt_observation(obs, corruption, obs_std, rng)

            states = torch.cat([states, torch.FloatTensor(obs).to(model.device).view(1, model.state_dim)])
            rtg = torch.cat([rtg, (rtg[0, -1] - float(reward) / scale).view(1, 1)], 1)
            timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], device=model.device, dtype=torch.long)], 1)

        returns.append(total_reward)

    return np.mean(returns), np.std(returns), np.array(returns)


def cvar(values, alpha=CVAR_ALPHA):
    """CVaR-alpha: mean of the worst alpha-fraction of values (at least one value)."""
    v = np.sort(np.asarray(values, dtype=float))
    k = max(1, math.ceil(alpha * len(v) - 1e-9))  # -1e-9 guards float error, e.g. 0.1 * 30
    return float(v[:k].mean())


def episode_stats(ep_scores):
    """Summary of per-episode normalized scores, keyed as in the json (<name>_<corruption>)."""
    s = np.asarray(ep_scores, dtype=float)
    return {
        'std': float(np.std(s)),
        'median': float(np.median(s)),
        'cvar10': cvar(s, CVAR_ALPHA),
    }


# ==================== RESULT FILES ====================

def write_json(path, obj):
    """Write via a temp file so an interrupted write never leaves a partial file."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def backfill_episode_stats(result_path):
    """Add the summary stats missing from a results json, computed from its stored
    per-episode scores (no re-evaluation needed)."""
    if not os.path.exists(result_path):
        return
    try:
        with open(result_path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    added = []
    for key in list(saved):
        if not key.startswith('episode_scores_'):
            continue
        level = key[len('episode_scores_'):]
        for name, value in episode_stats(saved[key]).items():
            if f'{name}_{level}' not in saved:
                saved[f'{name}_{level}'] = value
                added.append(f'{name}_{level}')
    if added:
        write_json(result_path, saved)
        print(f"Added {', '.join(added)} to {result_path} from its stored episode scores.")


def setting_mismatches(saved, args):
    """Run-defining settings where a saved run differs from the requested one."""
    requested = {
        'task': args.task,
        'seed': args.seed,
        'method': args.method,
        'lambda_gp': args.lambda_gp,
        'lambda_sep': args.lambda_sep,
        'num_heads': args.num_heads,
        'num_iterations': args.num_iterations,
        'beta_scale': args.beta_scale,
        'steps': args.steps,
    }
    # A saved file without num_iterations counts as 1 iteration, without beta_scale as 1
    defaults = {'num_iterations': 1, 'beta_scale': 1.0}
    problems = []
    for key, want in requested.items():
        have = saved.get(key, defaults.get(key))
        if have != want:
            problems.append(f"{key} saved={have!r} requested={want!r}")
    return problems


def checkpoint_matches(args, model_path):
    """True if model_path holds a finished checkpoint trained with exactly these settings."""
    if not os.path.exists(model_path):
        return False
    try:
        saved = torch.load(model_path, map_location='cpu', weights_only=False)['metadata']
    except Exception as e:
        print(f"Found {model_path} but could not read it ({e}).")
        return False
    problems = setting_mismatches(saved, args)
    if problems:
        print(f"Found {model_path} but it was trained with different settings ({'; '.join(problems)}).")
        return False
    return True


def run_already_complete(args, model_path, result_path):
    """True if this exact run already finished and can be skipped.

    train:      the checkpoint exists, loads and matches the settings.
    eval / all: the results json matches the settings, has every statistic for
                every requested corruption level, and used at least as many
                episodes as requested (a fuller evaluation also satisfies a smaller one).
    """
    if args.mode == 'train':
        return checkpoint_matches(args, model_path)

    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Found {result_path} but could not read it ({e}).")
        return False

    problems = setting_mismatches(saved, args)
    corruptions = [float(c) for c in args.corruptions.split(',')]
    required = ('score', 'median', 'cvar10', 'episode_scores')
    missing = [c for c in corruptions if any(f'{name}_{c}' not in saved for name in required)]
    if missing:
        problems.append(f"no score/median/CVaR-10/episode scores for corruption(s) {missing}")
    saved_eps = saved.get('eval_episodes')
    if saved_eps is None or saved_eps < args.eval_episodes:
        problems.append(f"eval_episodes saved={saved_eps} requested={args.eval_episodes}")

    if problems:
        print(f"Found {result_path} but it is not complete for this run ({'; '.join(problems)}).")
        return False
    return True


# ==================== MAIN ====================

def parse_args():
    p = argparse.ArgumentParser(description='Train, and optionally evaluate, one RHDT model.')
    p.add_argument('--task', default='halfcheetah-medium-v2', choices=TASKS, metavar='TASK',
                   help=f"D4RL task, one of {', '.join(TASKS)} (default: %(default)s)")
    p.add_argument('--seed', type=int, default=42, help='training seed (default: %(default)s)')
    p.add_argument('--device', default='cuda', help='torch device (default: %(default)s)')
    p.add_argument('--steps', type=int, default=100000, help='training steps (default: %(default)s)')
    p.add_argument('--num-heads', type=int, default=4, help='attention heads (default: %(default)s)')
    p.add_argument('--num-iterations', type=int, default=1,
                   help='Hopfield updates per attention layer; 0 = standard attention (default: %(default)s)')
    p.add_argument('--beta-scale', type=float, default=1.0,
                   help='Hopfield beta as a multiple of 1/sqrt(head size) (default: %(default)s)')
    p.add_argument('--lambda-gp', type=float, default=0.0025, help='gradient-penalty weight (default: %(default)s)')
    p.add_argument('--method', default='k_ortho', choices=['k_ortho', 'none'],
                   help='key regularizer; none disables it (default: %(default)s)')
    p.add_argument('--lambda-sep', type=float, default=0.1, help='key-orthogonality weight (default: %(default)s)')
    p.add_argument('--mode', default='all', choices=['all', 'train', 'eval'],
                   help='train: train and save the model; eval: evaluate the saved model; all: both '
                        '(default: %(default)s)')
    p.add_argument('--eval-episodes', type=int, default=1,
                   help='evaluation episodes per noise level (default: %(default)s)')
    p.add_argument('--corruptions', default='0.0',
                   help='comma-separated evaluation noise levels, in units of the dataset std (default: %(default)s)')
    p.add_argument('--output-dir', default=None,
                   help='directory for the checkpoint, loss history and results '
                        '(default: models_rhdt/<exp_name>/seed_<seed>)')
    p.add_argument('--force', action='store_true', help='redo the run even if it is already complete')
    args = p.parse_args()
    if args.num_iterations < 0:
        p.error('--num-iterations must be >= 0')
    if args.beta_scale <= 0:
        p.error('--beta-scale must be > 0')
    return args


def experiment_name(args):
    if args.method == 'none':
        name = f"h{args.num_heads}_gp{args.lambda_gp}"
    else:
        name = f"h{args.num_heads}_gp{args.lambda_gp}_{args.method}{args.lambda_sep}"
    name += f"_it{args.num_iterations}"
    return name + (f"_b{args.beta_scale:g}" if args.beta_scale != 1 else '')


def main():
    # Deterministic cuBLAS: must be set before the first CUDA operation (train.sh exports it too)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    set_seed(args.seed)

    exp_name = experiment_name(args)
    out_dir = args.output_dir or os.path.join('models_rhdt', exp_name, f'seed_{args.seed}')
    model_path = os.path.join(out_dir, f"rhdt_{args.task}.pt")
    result_path = os.path.join(out_dir, f"rhdt_{args.task}.json")
    history_path = os.path.join(out_dir, f"rhdt_{args.task}_losses.json")

    print("=" * 60)
    print(f"RHDT  task={args.task}  seed={args.seed}  mode={args.mode}")
    print(f"Experiment: {exp_name}")
    print(f"Gradient penalty: {args.lambda_gp}   Key regularizer: {args.method} {args.lambda_sep}")
    print(f"Heads: {args.num_heads}   Hopfield iterations: {args.num_iterations}"
          + (f"   beta: {args.beta_scale:g} x 1/sqrt(head size)" if args.beta_scale != 1 else ''))
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    # Skip finished runs (checked before the slow data load)
    if not args.force and args.mode != 'train':
        backfill_episode_stats(result_path)
    if not args.force and run_already_complete(args, model_path, result_path):
        done_file = model_path if args.mode == 'train' else result_path
        print(f"Already complete, skipping: {done_file}  (use --force to rerun)")
        return

    # A matching checkpoint is evaluated instead of retrained (e.g. to add corruption
    # levels or episodes). --force retrains, except in eval mode, where it re-evaluates.
    reuse_checkpoint = (args.mode != 'train'
                        and (not args.force or args.mode == 'eval')
                        and checkpoint_matches(args, model_path))
    if reuse_checkpoint:
        print(f"Checkpoint exists, skipping training and evaluating it: {model_path}")
    elif args.mode == 'eval':
        if not os.path.exists(model_path):
            print(f"No trained model yet: {model_path}  (skipping; train it first)")
        raise SystemExit(EXIT_NO_CHECKPOINT)

    dataset, env = load_task(args.task)
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    data = build_trajectories(dataset, max_ep_len=MAX_EP_LEN, scale=RTG_SCALE)

    model = RHDT(state_dim, action_dim, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
                 num_heads=args.num_heads, max_ep_len=MAX_EP_LEN, seq_len=SEQ_LEN,
                 num_iterations=args.num_iterations, device=args.device, beta_scale=args.beta_scale)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(out_dir, exist_ok=True)

    if not reuse_checkpoint:
        history = train(model, data, args)

        metadata = {
            'task': args.task,
            'exp_name': exp_name,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'embed_dim': EMBED_DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': args.num_heads,
            'seq_len': SEQ_LEN,
            'max_ep_len': MAX_EP_LEN,
            'num_iterations': args.num_iterations,
            'method': args.method,
            'lambda_gp': args.lambda_gp,
            'lambda_sep': args.lambda_sep,
            'steps': args.steps,
            'seed': args.seed,
            'obs_mean': data['obs_mean'].tolist(),
            'obs_std': data['obs_std'].tolist(),
            'target_return': float(data['target_return']),
            'scale': float(data['scale']),
        }
        if args.beta_scale != 1:  # recorded only when not the default
            metadata['beta_scale'] = args.beta_scale

        # Loss history first, so every saved checkpoint has one next to it
        loss_history = {
            'task': args.task,
            'exp_name': exp_name,
            'seed': args.seed,
            'method': args.method,
            'lambda_gp': args.lambda_gp,
            'lambda_sep': args.lambda_sep,
            'num_heads': args.num_heads,
            'num_iterations': args.num_iterations,
            'steps': args.steps,
        }
        if args.beta_scale != 1:
            loss_history['beta_scale'] = args.beta_scale
        loss_history['interval'] = LOSS_INTERVAL
        loss_history['history'] = history
        write_json(history_path, loss_history)

        # Temp file + rename, so an interrupted save never looks like a finished run
        tmp_path = model_path + '.tmp'
        torch.save({'model_state_dict': model.state_dict(), 'metadata': metadata}, tmp_path)
        os.replace(tmp_path, model_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Loss history saved to: {history_path}")

        if args.mode == 'train':
            env.close()
            return
    else:
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
        metadata = checkpoint['metadata']
        model.load_state_dict(checkpoint['model_state_dict'])
        # Evaluate with the statistics the model was trained with
        data['obs_mean'] = np.array(metadata['obs_mean'])
        data['obs_std'] = np.array(metadata['obs_std'])
        data['target_return'] = metadata['target_return']
        data['scale'] = metadata['scale']
        print(f"Loaded model from: {model_path}")

    # ==================== EVALUATION ====================
    corruptions = [float(c) for c in args.corruptions.split(',')]
    results = {
        'exp_name': exp_name,
        'task': args.task,
        'method': args.method,
        'lambda_gp': args.lambda_gp,
        'lambda_sep': args.lambda_sep,
        'num_heads': args.num_heads,
        'num_iterations': args.num_iterations,
        'steps': args.steps,
        'seed': args.seed,
        'eval_episodes': args.eval_episodes,
        'corruptions': args.corruptions,
    }
    if args.beta_scale != 1:
        results['beta_scale'] = args.beta_scale

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for corruption in corruptions:
        set_seed(args.seed)
        ret, std, ep_returns = evaluate(
            model, env, data['target_return'], data['obs_mean'], data['obs_std'], data['scale'],
            args.eval_episodes, MAX_EP_LEN, corruption, args.seed + int(corruption * 100000),
        )
        score = normalized_score(args.task, ret)
        ep_scores = [float(normalized_score(args.task, r)) for r in ep_returns]  # in episode order
        stats = episode_stats(ep_scores)
        print(f"Corruption {corruption}: mean {score:.1f} ± {stats['std']:.1f}  median {stats['median']:.1f}  "
              f"CVaR-10 {stats['cvar10']:.1f}  (return {ret:.1f} ± {std:.1f}, {len(ep_scores)} episodes)")
        results[f'score_{corruption}'] = score  # mean normalized score
        for name, value in stats.items():       # std_, median_, cvar10_
            results[f'{name}_{corruption}'] = value
        results[f'return_{corruption}'] = ret
        results[f'episode_scores_{corruption}'] = ep_scores
    results['cvar_alpha'] = CVAR_ALPHA

    if 'score_0.0' in results and 'score_0.3' in results:
        results['degradation'] = results['score_0.0'] - results['score_0.3']
        results['cvar10_degradation'] = results['cvar10_0.0'] - results['cvar10_0.3']
        print(f"Degradation: {results['degradation']:.1f}  (CVaR-10: {results['cvar10_degradation']:.1f})")

    # Re-evaluating a checkpoint: keep the json's other fields, replace its old evaluation
    if reuse_checkpoint and os.path.exists(result_path):
        try:
            with open(result_path) as f:
                existing = json.load(f)
            existing = {k: v for k, v in existing.items()
                        if not k.startswith(EVAL_KEY_PREFIXES) and k not in EVAL_KEYS}
            existing.update(results)
            results = existing
        except (OSError, json.JSONDecodeError):
            pass  # unreadable old json: just write the new results

    write_json(result_path, results)
    print(f"\nSaved: {result_path}")

    env.close()


if __name__ == "__main__":
    main()
