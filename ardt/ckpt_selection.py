"""
Checkpoint selection for the ARDT baseline.

During the last `keep_last * interval` training steps, a candidate checkpoint is
taken every `interval` steps and scored with a short validation rollout without
noise, under its own seed (999 by default, disjoint from the evaluation seeds). The
candidate with the best validation score is saved. Validation saves and restores
every random number generator, so it does not change training. With
--select-metric none the final weights are saved.

    add_selection_args(parser)
    validate = make_validator(evaluate, env, target_return, obs_mean, obs_std, scale,
                              task, args.val_episodes, args.val_seed)
    selector = CheckpointSelector(total_steps, args.ckpt_interval, args.ckpt_keep_last, validate)
    for step in ...:
        ...train step...
        selector.maybe_validate(model, step + 1)
    selector.select(model)                   # loads the best candidate's weights
    metadata.update(selector.metadata())
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'training'))
from data import normalized_score  # noqa: E402


def add_selection_args(parser):
    parser.add_argument('--select-metric', default='clean', choices=['clean', 'none'],
                        help="'clean' = keep the candidate with the best validation score without noise, "
                             "'none' = keep the final weights (default: %(default)s)")
    parser.add_argument('--ckpt-interval', type=int, default=10000,
                        help='steps between candidate checkpoints (default: %(default)s)')
    parser.add_argument('--ckpt-keep-last', type=int, default=5,
                        help='number of final candidates to choose from (default: %(default)s)')
    parser.add_argument('--val-episodes', type=int, default=10,
                        help='validation episodes per candidate (default: %(default)s)')
    parser.add_argument('--val-seed', type=int, default=999,
                        help='validation seed, must differ from the evaluation seeds (default: %(default)s)')
    return parser


class preserve_rng:
    """Save and restore the python, numpy and torch (CPU and CUDA) random states."""

    def __enter__(self):
        self.py = random.getstate()
        self.np = np.random.get_state()
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def __exit__(self, *exc):
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


def make_validator(evaluate_fn, env, target_return, obs_mean, obs_std, scale, task,
                   n_episodes, val_seed):
    """validate(model) -> normalized mean return of the validation episodes, without noise.

    evaluate_fn(model, env, target_return, obs_mean, obs_std, scale, n_episodes, max_ep_len,
    noise level, seed) returns the mean return first, like training/train_rhdt.evaluate.
    """
    def validate(model):
        with preserve_rng():
            ret = evaluate_fn(model, env, target_return, obs_mean, obs_std, scale,
                              n_episodes, 1000, 0.0, val_seed)[0]
        model.train()
        return float(normalized_score(task, ret))
    return validate


class CheckpointSelector:
    def __init__(self, total_steps, interval, keep_last, validate_fn, enabled=True, log=print,
                 metric='clean', n_episodes=10, val_seed=999):
        self.total_steps = total_steps
        self.interval = interval
        self.first = total_steps - keep_last * interval
        self.validate_fn = validate_fn
        self.enabled = enabled
        self.log = log
        self.metric, self.n_episodes, self.val_seed = metric, n_episodes, val_seed
        self.candidates = []  # (step, validation score, state_dict on the CPU)
        self.selected_step = total_steps

    def maybe_validate(self, model, done_steps):
        if not self.enabled or done_steps % self.interval != 0 or done_steps <= self.first:
            return
        val = self.validate_fn(model)
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.candidates.append((done_steps, val, sd))
        self.log(f"  [ckpt @ {done_steps}] validation ({self.metric}) score = {val:.2f}")

    def select(self, model):
        """Load the best candidate's weights into `model`; returns (selected step, {step: score})."""
        if not self.candidates:
            return self.selected_step, {}
        best_step, best_val, best_sd = max(self.candidates, key=lambda c: c[1])
        model.load_state_dict(best_sd)
        self.selected_step = best_step
        print(f"\nCheckpoint selection ({self.metric}, {self.n_episodes} validation episodes, seed {self.val_seed}):")
        for st, v, _ in self.candidates:
            print(f"  step {st:>7}: {v:6.2f}{'  <-- selected' if st == best_step else ''}")
        return best_step, self.val_scores()

    def val_scores(self):
        return {int(st): float(v) for st, v, _ in self.candidates}

    def metadata(self):
        return {'select_metric': self.metric if self.enabled else 'none',
                'selected_step': int(self.selected_step),
                'val_scores': self.val_scores(),
                'val_episodes': self.n_episodes,
                'val_seed': self.val_seed}
