"""
Evaluate one RDT checkpoint, without noise and under observation noise.

Same protocol and rollout code as the RHDT evaluation (../eval): for every noise
level, --eval-episodes episodes with evaluation seed --seed, written to
<output-dir>/<checkpoint name>_corruptions.json with the return and D4RL-normalized
score of every episode. train_rdt.sh runs this once per (checkpoint, evaluation seed)
at the paper's four noise levels; results.py pools the results.

    python eval_rdt.py --model-path models_rdt_tuned/seed_42/rdt_hopper-medium-v2.pt \
        --output-dir results --seed 11 --eval-episodes 50 --corruptions 0.0,0.1,0.2,0.3

With --resume, noise levels already complete in an existing result file are kept
and only the missing ones are rolled out.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Environments, seeding and rollouts are shared with RHDT (../training)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'training'))
from data import make_env, normalized_score  # noqa: E402
from train_rhdt import evaluate, set_seed  # noqa: E402

from results import assemble, atomic_write_json, ckey, load_json, reusable_levels  # noqa: E402
from train_rdt import RobustDecisionTransformer  # noqa: E402


def evaluate_checkpoint(model_path, corruptions, episodes, device, seed, existing=None):
    """Result dict for one checkpoint and evaluation seed; levels complete in `existing` are reused."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    task = metadata['task']

    header = {
        'model_type': 'rdt',
        'task': task,
        'model_path': model_path,
        'train_with_corruption': metadata.get('train_with_corruption', False),
        'seed': seed,
        'eval_episodes': episodes,
        'corruption_levels': corruptions,
    }

    blocks = reusable_levels(existing, episodes, seed=seed, task=task)
    todo = [c for c in corruptions if c not in blocks]
    if blocks:
        print(f"Reusing complete noise levels: {sorted(blocks)}")
    if not todo:
        return assemble(header, blocks)

    env = make_env(task)
    set_seed(seed)
    model = RobustDecisionTransformer(
        metadata['state_dim'], metadata['action_dim'], embed_dim=metadata.get('embed_dim', 128),
        num_layers=metadata.get('num_layers', 3), num_heads=metadata.get('num_heads', 1),
        max_ep_len=metadata.get('max_ep_len', 1000), seq_len=metadata.get('seq_len', 20),
        dropout=metadata.get('dropout', 0.1), embedding_dropout=metadata.get('embedding_dropout', 0.1),
        device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {model_path}: {task}")

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
    p = argparse.ArgumentParser(description='Evaluate one RDT checkpoint.')
    p.add_argument('--model-path', required=True, help='checkpoint written by train_rdt.py')
    p.add_argument('--output-dir', default='.', help='directory for the result json (default: %(default)s)')
    p.add_argument('--corruptions', default='0.0,0.1,0.2,0.3',
                   help='comma-separated noise levels, in units of the dataset std (default: %(default)s)')
    p.add_argument('--eval-episodes', type=int, default=50, help='episodes per noise level (default: %(default)s)')
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

    results = evaluate_checkpoint(args.model_path, corruptions, args.eval_episodes, args.device,
                                  args.seed, existing=existing)
    if existing is not None and results == existing:
        print(f"Already complete: {result_path}")
        return
    atomic_write_json(results, result_path)
    print(f"Saved: {result_path}")


if __name__ == '__main__':
    main()
