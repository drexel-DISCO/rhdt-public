#!/usr/bin/env python
"""
Training curves of RHDT from the loss histories that ../training/train_rhdt.py writes next
to each checkpoint (rhdt_<task>_losses.json: each loss term averaged over every 500 steps).

For every task and config given with --run, this averages each loss term over the training
seeds (mean and sample standard deviation across seeds) and writes

    <output-dir>/dynamics_<task>_<config>.json   the curves of every seed, their mean and std
    <output-dir>/dynamics.pdf and .png           one row per task and config, one column per
                                                 loss term (needs matplotlib)

It also compares the weights of every retrained model with the paper's model of the same
run in --paper-root: where they are identical, the curves are exactly those of the paper's
model. dynamics.sh runs this; directly:

    python dynamics_plot.py --models-root models_dynamics --output-dir results_dynamics \\
        --paper-root ../training/models_rhdt_sweep --seeds 42 48213 \\
        --run halfcheetah-medium-v2 gpheavy --run hopper-medium-replay-v2 quarter
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch

# Loss terms of train_rhdt.py: loss = action_loss + lambda_gp * gp_loss + lambda_sep * sep_loss
TERMS = {
    'loss': 'Total loss',
    'action_loss': 'Action loss',
    'gp_loss': 'Gradient penalty',
    'sep_loss': 'Key orthogonality',
}
SETTINGS = ('method', 'lambda_gp', 'lambda_sep', 'num_heads', 'num_iterations', 'beta_scale', 'steps')
ENVS = {'halfcheetah': 'HalfCheetah', 'hopper': 'Hopper', 'walker2d': 'Walker2d'}


def task_label(task):
    """halfcheetah-medium-replay-v2 -> HalfCheetah Medium-Replay"""
    env, dataset = task.rsplit('-', 1)[0].split('-', 1)
    return f'{ENVS.get(env, env)} {dataset.title()}'


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_atomic(text, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', prefix='.tmp_')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def compare_weights(path, paper_path):
    """'identical' or 'different' weights, or why the two checkpoints could not be compared."""
    if not os.path.exists(paper_path):
        return 'no paper model'
    try:
        a, b = (torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'] for p in (path, paper_path))
    except Exception as e:
        return f'unreadable ({e})'
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    return 'identical' if same else 'different'


def summarize(a, task, cfg):
    """Curves of one task and config: per seed, and their mean and std over the seeds."""
    runs, notes = {}, []
    for seed in a.seeds:
        run_dir = os.path.join(a.models_root, f'iter{a.iterations}', cfg, f'seed_{seed}')
        model_path = os.path.join(run_dir, f'rhdt_{task}.pt')
        history = load_json(os.path.join(run_dir, f'rhdt_{task}_losses.json'))
        if not os.path.exists(model_path):
            notes.append(f'seed {seed}: not trained yet')
        elif history is None:
            notes.append(f'seed {seed}: its checkpoint has no loss history (e.g. a copy of an older model); '
                         f'delete {model_path} and train again')
        else:
            runs[seed] = history
    if not runs:
        return None, notes

    first = next(iter(runs.values()))
    step = first['history']['step']
    for seed, history in runs.items():
        if history['history']['step'] != step:
            raise SystemExit(f'{task} {cfg}: seed {seed} has other steps than seed {next(iter(runs))}')
    curves = {term: np.array([h['history'][term] for h in runs.values()]) for term in TERMS}
    paper = {}
    for seed in runs:
        rel = os.path.join(f'iter{a.iterations}', cfg, f'seed_{seed}', f'rhdt_{task}.pt')
        paper[str(seed)] = compare_weights(os.path.join(a.models_root, rel), os.path.join(a.paper_root, rel))

    return {
        'task': task,
        'config': cfg,
        'settings': {key: first[key] for key in SETTINGS if key in first},
        'interval': first['interval'],
        'seeds': list(runs),
        'missing_seeds': [s for s in a.seeds if s not in runs],
        'paper_weights': paper,
        'step': step,
        'mean': {term: c.mean(0).tolist() for term, c in curves.items()},
        'std': {term: (c.std(0, ddof=1) if len(c) > 1 else np.zeros(len(step))).tolist()
                for term, c in curves.items()},
        'per_seed': {str(seed): {term: h['history'][term] for term in TERMS} for seed, h in runs.items()},
    }, notes


def print_summary(g, n_seeds, paper_root):
    s = g['settings']
    print(f"\n{g['task']} ({g['config']}: lambda_gp {s['lambda_gp']:g}, lambda_sep {s['lambda_sep']:g}), "
          f"{len(g['seeds'])} of {n_seeds} training seeds")
    print(f"  mean ± std over the seeds of each term, averaged over the {g['interval']} steps up to the step shown")
    step = g['step']
    rows = [0] + [int(np.argmin(np.abs(np.array(step) - f * step[-1]))) for f in (0.25, 0.5, 0.75)] + [len(step) - 1]
    table = [['step'] + list(TERMS.values())]
    for i in dict.fromkeys(rows):
        table.append([str(step[i])] + [f"{g['mean'][t][i]:.4g} ± {g['std'][t][i]:.2g}" if any(g['mean'][t]) else '-'
                                       for t in TERMS])
    widths = [max(len(row[c]) for row in table) for c in range(len(table[0]))]
    for row in table:
        print('  ' + '  '.join(x.ljust(w) for x, w in zip(row, widths)).rstrip())

    by_status = {}
    for seed, status in g['paper_weights'].items():
        by_status.setdefault(status, []).append(seed)
    if by_status.get('identical'):
        print(f"  Weights identical to the paper's models: {len(by_status['identical'])} of {len(g['seeds'])} seeds")
    if by_status.get('different'):
        print(f"  WARNING: the weights of seeds {', '.join(by_status['different'])} differ from the paper's "
              f"models, so their curves are not exactly those of the paper's models (other GPU or software?)")
    for status, seeds in by_status.items():
        if status not in ('identical', 'different'):
            print(f"  Not compared with the paper's models in {paper_root} (seeds {', '.join(seeds)}): {status}")


def plot(groups, path):
    """One row per task and config, one column per loss term: mean over the seeds, ± std shaded."""
    try:
        import matplotlib
    except ImportError:
        print('\nmatplotlib is not installed, so there is no figure (pip install matplotlib)')
        return []
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(groups), len(TERMS), figsize=(3.3 * len(TERMS), 2.5 * len(groups) + 0.4),
                             squeeze=False)
    for row, g in zip(axes, groups):
        step = np.array(g['step']) / 1000
        for ax, term in zip(row, TERMS):
            mean, std = np.array(g['mean'][term]), np.array(g['std'][term])
            if mean.any():
                ax.plot(step, mean, color='C0', lw=1.2)
                ax.fill_between(step, mean - std, mean + std, color='C0', alpha=0.25, lw=0)
                ax.set_yscale('log')
            else:  # weight 0: train_rhdt.py does not compute the term
                ax.text(0.5, 0.5, r'not used ($\lambda = 0$)', ha='center', va='center', color='0.4',
                        transform=ax.transAxes)
                ax.set_yticks([])
            ax.set_xlim(0, step[-1])
            ax.grid(alpha=0.3)
        s = g['settings']
        row[0].set_ylabel(f"{task_label(g['task'])}\n"
                          rf"$\lambda_{{gp}}$ = {s['lambda_gp']:g}, $\lambda_{{sep}}$ = {s['lambda_sep']:g}")
    for ax, term in zip(axes[0], TERMS):
        ax.set_title(TERMS[term], fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('training step (thousands)')
    fig.tight_layout()
    written = []
    for ext, kwargs in (('.pdf', {}), ('.png', {'dpi': 150})):
        fig.savefig(path + ext, **kwargs)
        written.append(path + ext)
    plt.close(fig)
    return written


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models-root', required=True, help='retrained models with their loss histories (train.sh -o)')
    p.add_argument('--output-dir', required=True, help='where the curves and the figure are written')
    p.add_argument('--paper-root', default=os.path.join(here, '..', 'training', 'models_rhdt_sweep'),
                   help="the paper's models, to compare the weights with (default: ../training/models_rhdt_sweep)")
    p.add_argument('--iterations', type=int, default=1, help='Hopfield iterations (default: %(default)s)')
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='training seeds')
    p.add_argument('--run', nargs=2, action='append', required=True, metavar=('TASK', 'CONFIG'),
                   help='a task and its config name')
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    groups, incomplete = [], False
    for task, cfg in dict.fromkeys(map(tuple, a.run)):
        g, notes = summarize(a, task, cfg)
        if g is None:
            print(f'\n{task} ({cfg}): no loss histories in {a.models_root}')
        else:
            path = os.path.join(a.output_dir, f'dynamics_{task}_{cfg}.json')
            write_atomic(json.dumps(g, indent=2) + '\n', path)
            print_summary(g, len(a.seeds), a.paper_root)
            print(f'  Written: {path}')
            groups.append(g)
        for note in notes:
            print(f'  Missing: {note}')
        incomplete |= bool(notes)

    if groups:
        written = plot(groups, os.path.join(a.output_dir, 'dynamics'))
        if written:
            print(f"\nFigure: {' and '.join(written)}")
    if incomplete:
        print('\nSome runs have no loss history yet; the curves above average over the seeds that have one.')


if __name__ == '__main__':
    main()
