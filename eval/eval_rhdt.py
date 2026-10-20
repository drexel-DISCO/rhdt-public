"""
Evaluate one trained RHDT checkpoint under observation noise.

For every noise level, rolls the policy out --eval-episodes times with
evaluation seed --seed and writes <output-dir>/<checkpoint name>_corruptions.json
with the return and D4RL-normalized score of every episode. eval.sh runs this
once per (checkpoint, evaluation seed); results.py pools the results.

    python eval_rhdt.py --model-path ../training/models_rhdt_sweep/iter1/sepheavy/seed_42/rhdt_hopper-medium-v2.pt \
        --output-dir results --seed 11 --eval-episodes 50 --corruptions 0.0,0.1,0.2,0.3

With --resume, noise levels already complete in an existing result file are kept
and only the missing ones are rolled out. Every level reseeds from (seed, noise
level), so a level evaluated alone gives the same result as in a full run.
"""

import argparse
import os
import sys

import numpy as np
import torch

# The model, environments and rollouts are shared with training/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'training'))
from data import make_env, normalized_score  # noqa: E402
from model import RHDT  # noqa: E402
from train_rhdt import evaluate, set_seed  # noqa: E402

from results import assemble, atomic_write_json, ckey, load_json, reusable_levels  # noqa: E402


def evaluate_checkpoint(model_path, corruptions, episodes, iterations, device, seed, existing=None):
    """Result dict for one checkpoint and evaluation seed; levels complete in `existing` are reused."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    task = metadata['task']
    if iterations is None:
        iterations = metadata.get('num_iterations', 1)

    header = {
        'model_type': 'rhdt',
        'task': task,
        'model_path': model_path,
        'exp_name': metadata.get('exp_name', ''),
        'num_heads': metadata.get('num_heads', 1),
        'lambda_gradient_penalty': metadata.get('lambda_gp', 0.0),
        'seed': seed,
        'eval_episodes': episodes,
        'iterations': iterations,
        'corruption_levels': corruptions,
    }

    blocks = reusable_levels(existing, episodes, seed=seed, iterations=iterations, task=task)
    todo = [c for c in corruptions if c not in blocks]
    if blocks:
        print(f"Reusing complete noise levels: {sorted(blocks)}")
    if not todo:
        return assemble(header, blocks)

    env = make_env(task)
    set_seed(seed)
    model = RHDT(metadata['state_dim'], metadata['action_dim'],
                 embed_dim=metadata.get('embed_dim', 128), num_layers=metadata.get('num_layers', 3),
                 num_heads=metadata.get('num_heads', 4), max_ep_len=metadata.get('max_ep_len', 1000),
                 seq_len=metadata.get('seq_len', 20), num_iterations=iterations, device=device,
                 beta_scale=metadata.get('beta_scale', 1.0))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {model_path}: {task}, {metadata.get('exp_name', '')}, "
          f"{iterations} Hopfield iteration(s) at evaluation")

    obs_mean = np.array(metadata['obs_mean'])
    obs_std = np.array(metadata['obs_std'])
    max_ep_len = metadata.get('max_ep_len', 1000)

    print(f"\n{'Noise':<8} {'Score':>8} {'Return':>10} {'Std':>8}")
    for corruption in todo:
        set_seed(seed)
        ret, std, returns = evaluate(
            model, env, metadata['target_return'], obs_mean, obs_std, metadata['scale'],
            episodes, max_ep_len, corruption, seed + int(corruption * 100000),
        )
        score = normalized_score(task, ret)
        print(f"{corruption:<8} {score:>8.1f} {ret:>10.1f} {std:>8.1f}")

        k = ckey(corruption)
        blocks[corruption] = {
            f'return_{k}': float(ret),
            f'return_std_{k}': float(std),
            f'score_{k}': float(score),
            f'raw_returns_{k}': [float(r) for r in returns],
            f'raw_scores_{k}': [float(normalized_score(task, r)) for r in returns],
        }

    env.close()
    # Header + levels (reused and new, sorted) + degradation relative to the clean score
    return assemble(header, blocks)


def main():
    p = argparse.ArgumentParser(description='Evaluate one RHDT checkpoint under observation noise.')
    p.add_argument('--model-path', required=True, help='checkpoint written by training/train_rhdt.py')
    p.add_argument('--output-dir', default='.', help='directory for the result json (default: %(default)s)')
    p.add_argument('--corruptions', default='0.0,0.1,0.2,0.3',
                   help='comma-separated noise levels, in units of the dataset std (default: %(default)s)')
    p.add_argument('--eval-episodes', type=int, default=50, help='episodes per noise level (default: %(default)s)')
    p.add_argument('--iterations', type=int, default=None,
                   help='Hopfield iterations per attention layer (default: as in training)')
    p.add_argument('--seed', type=int, default=11, help='evaluation seed (default: %(default)s)')
    p.add_argument('--device', default='cuda', help='torch device (default: %(default)s)')
    p.add_argument('--resume', action='store_true',
                   help='keep the complete noise levels of an existing result and evaluate only the missing ones')
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    corruptions = [float(c) for c in args.corruptions.split(',')]

    name = os.path.basename(args.model_path).replace('.pt', '')
    result_path = os.path.join(args.output_dir, f'{name}_corruptions.json')
    existing = load_json(result_path) if args.resume else None

    results = evaluate_checkpoint(args.model_path, corruptions, args.eval_episodes, args.iterations,
                                  args.device, args.seed, existing=existing)
    if existing is not None and results == existing:
        print(f"Already complete: {result_path}")
        return
    atomic_write_json(results, result_path)
    print(f"Saved: {result_path}")


if __name__ == '__main__':
    main()
