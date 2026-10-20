#!/usr/bin/env python
"""
Ablation table from the summaries that ../eval/eval.sh writes.

For every training seed, eval.sh pools the D4RL-normalized scores of all episodes of all
evaluation seeds (e.g. 10 x 50 = 500 per noise level) into a mean, median, IQM and CVaR-10.
This script averages each of these over the training seeds (mean and sample standard
deviation across seeds) and writes, per task:

    <results-root>/ablation_<task>.json   the settings of every variant, the numbers of
                                          every training seed and the averages
    <results-root>/ablation_<task>.md     one table per statistic

and prints the table of mean scores. A training seed counts only when its pool is
complete (all evaluation seeds, all episodes). ablation.sh runs this; directly:

    python ablation_table.py --models-root models_ablation --results-root results_ablation \\
        --seeds 42 48213 --eval-seeds 11,13 --episodes 50 --corruptions 0.0,0.3 \\
        --run halfcheetah-medium-v2 full gpheavy 1 --run halfcheetah-medium-v2 non non 1
"""

import argparse
import json
import os
import statistics
import tempfile

LABELS = {
    'full': 'Full RHDT',
    'gp_only': 'GP only',
    'sep_only': 'Orthogonality only',
    'non': 'No penalties',
    'no_hopfield': 'No Hopfield update (0 iterations)',
    'one_head': 'One head',
}
STATS = {'mean': 'Mean', 'median': 'Median', 'iqm': 'IQM', 'cvar10': 'CVaR-10'}
SETTINGS = (('lambda_gp', 'λ_gp'), ('lambda_sep', 'λ_sep'), ('num_iterations', 'Iterations'),
            ('num_heads', 'Heads'))


def ckey(c):
    """Noise level -> key in the summaries (as in ../eval/results.py)."""
    return str(float(c))


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


def read_config(path):
    """The key=value lines of a config.txt that train.sh writes next to each model."""
    try:
        with open(path) as f:
            return dict(line.strip().split('=', 1) for line in f if '=' in line)
    except OSError:
        return {}


def variant_scores(a, task, variant, cfg, iterations):
    """Settings, per-seed statistics and their averages over training seeds for one variant."""
    summary = load_json(os.path.join(a.results_root, 'summary', f'{task}_{cfg}_iter{iterations}.json')) or {}
    n_episodes = len(a.eval_seeds) * a.episodes
    settings = {}
    for seed in a.seeds:
        settings = read_config(os.path.join(a.models_root, f'iter{iterations}', cfg, f'seed_{seed}', 'config.txt'))
        if settings:
            break

    per_seed, averages = {}, {}
    for c in a.corruptions:
        k = ckey(c)
        values = {s: [] for s in STATS}
        for seed in a.seeds:
            entry = (summary.get(str(seed)) or {}).get(k)
            if (not isinstance(entry, dict) or entry.get('eval_seeds') != a.eval_seeds
                    or entry.get('n_episodes') != n_episodes or any(s not in entry for s in STATS)):
                continue
            per_seed.setdefault(str(seed), {})[k] = {s: entry[s] for s in STATS}
            for s in STATS:
                values[s].append(entry[s])
        averages[k] = {s: {'mean': statistics.mean(v),
                           'std': statistics.stdev(v) if len(v) > 1 else 0.0,
                           'n_seeds': len(v)} if v else None
                       for s, v in values.items()}
    return {
        'variant': variant,
        'label': LABELS.get(variant, variant),
        'config': cfg,
        'iterations': iterations,
        'settings': {key: settings.get(key) for key, _ in SETTINGS},
        'scores': averages,
        'per_seed': per_seed,
    }


def cell(avg, n_total):
    if avg is None:
        return '-'
    text = f"{avg['mean']:.1f} ± {avg['std']:.1f}"
    return text if avg['n_seeds'] == n_total else f"{text} ({avg['n_seeds']}/{n_total})"


def table_rows(rows, levels, stat, n_total):
    """Header and one row per variant: label, settings, one cell per noise level."""
    header = ['Variant'] + [name for _, name in SETTINGS] + [f'Noise {ckey(c)}' for c in levels]
    body = [[r['label']] + [r['settings'][key] or '?' for key, _ in SETTINGS]
            + [cell(r['scores'][ckey(c)][stat], n_total) for c in levels] for r in rows]
    return header, body


def print_table(header, body):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    for line in [header] + body:
        print('  ' + '  '.join(str(x).ljust(w) for x, w in zip(line, widths)).rstrip())


def markdown(task, rows, a):
    n = len(a.seeds)
    out = [f'# Ablation: {task}', '',
           f'D4RL-normalized score: mean ± standard deviation over {n} training seeds '
           f'({", ".join(map(str, a.seeds))}). Each training seed pools {len(a.eval_seeds)} evaluation '
           f'seeds × {a.episodes} episodes per noise level; a cell marked (k/{n}) has only k complete seeds.',
           '']
    for stat, name in STATS.items():
        header, body = table_rows(rows, a.corruptions, stat, n)
        out += [f'## {name}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        out += ['| ' + ' | '.join(str(x) for x in line) + ' |' for line in body]
        out.append('')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models-root', required=True, help='trained models (train.sh -o)')
    p.add_argument('--results-root', required=True, help='results (eval.sh -o); the tables are written here')
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='training seeds')
    p.add_argument('--eval-seeds', required=True, help='evaluation seeds, comma-separated')
    p.add_argument('--episodes', type=int, required=True, help='episodes per noise level and evaluation seed')
    p.add_argument('--corruptions', required=True, help='noise levels, comma-separated')
    p.add_argument('--run', nargs=4, action='append', required=True,
                   metavar=('TASK', 'VARIANT', 'CONFIG', 'ITERATIONS'),
                   help='one variant of one task: its config name and Hopfield iterations')
    a = p.parse_args()
    a.eval_seeds = [int(s) for s in a.eval_seeds.split(',') if s.strip()]
    a.corruptions = [float(c) for c in a.corruptions.split(',') if c.strip()]

    tasks = {}
    for task, variant, cfg, iterations in a.run:
        tasks.setdefault(task, []).append(variant_scores(a, task, variant, cfg, int(iterations)))

    os.makedirs(a.results_root, exist_ok=True)
    n = len(a.seeds)
    for task, rows in tasks.items():
        result = {'task': task, 'training_seeds': a.seeds, 'eval_seeds': a.eval_seeds,
                  'episodes': a.episodes, 'noise_levels': a.corruptions, 'variants': rows}
        json_path = os.path.join(a.results_root, f'ablation_{task}.json')
        md_path = os.path.join(a.results_root, f'ablation_{task}.md')
        write_atomic(json.dumps(result, indent=2) + '\n', json_path)
        write_atomic(markdown(task, rows, a), md_path)

        print(f'\n{task}: mean D4RL-normalized score ± std over {n} training seeds '
              f'({len(a.eval_seeds)} evaluation seeds x {a.episodes} episodes each)')
        print_table(*table_rows(rows, a.corruptions, 'mean', n))
        incomplete = [r['label'] for r in rows
                      if any(v['mean'] is None or v['mean']['n_seeds'] < n for v in r['scores'].values())]
        if incomplete:
            print(f'  Incomplete (not every training seed is trained and evaluated): {", ".join(incomplete)}')
        print(f'  Written: {json_path}\n           {md_path} (also median, IQM, CVaR-10)')


if __name__ == '__main__':
    main()
