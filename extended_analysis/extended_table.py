#!/usr/bin/env python
"""
Score tables of the extended analysis, from the summaries that ../eval/eval.sh writes.

Each point of a sweep is one config at one number of Hopfield iterations, read from the
source that holds its models and results (the paper's runs, the ablation, or this folder).
For every training seed, eval.sh pools the D4RL-normalized scores of all episodes of all
evaluation seeds into a mean, median, IQM and CVaR-10. This script averages each of them
over the training seeds (mean and sample standard deviation across seeds) and writes, per
sweep and task:

    <output-dir>/<sweep>_<task>.json   the settings and source of every point, the numbers
                                       of every training seed and the averages
    <output-dir>/<sweep>_<task>.md     one table per statistic

and prints the table of mean scores. A training seed counts only when its pool is complete
(all evaluation seeds, all episodes). extended_analysis.sh runs this; directly:

    python extended_table.py --output-dir results_extended/tables \\
        --source main ../training/models_rhdt_sweep ../eval/results_rhdt \\
        --source extended models_extended results_extended \\
        --seeds 42 48213 --eval-seeds 11,13 --episodes 50 --corruptions 0.0,0.3 \\
        --point beta halfcheetah-medium-v2 1 gpheavy 1 main \\
        --point beta halfcheetah-medium-v2 2 gpheavy-b2 1 extended
"""

import argparse
import json
import os
import statistics
import tempfile

# sweep -> (title, label of a point given its value)
SWEEPS = {
    'beta': ('Hopfield β (× 1/√head size)', lambda v: f'β ×{v}'),
    'gp': ('Gradient-penalty weight (× tuned λ_GP, λ_ortho tuned)', lambda v: f'λ_GP ×{v}'),
    'ortho': ('Key-orthogonality weight (× tuned λ_ortho, λ_GP tuned)', lambda v: f'λ_ortho ×{v}'),
    'nhop': ('Hopfield iterations', lambda v: f'N_hop = {v}'),
}
STATS = {'mean': 'Mean', 'median': 'Median', 'iqm': 'IQM', 'cvar10': 'CVaR-10'}
SETTINGS = (('lambda_gp', 'λ_gp'), ('lambda_sep', 'λ_ortho'), ('num_iterations', 'Iterations'),
            ('beta_scale', 'β'))
EVAL_SH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'eval', 'eval.sh'))


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


def point_scores(a, sweep, task, value, cfg, iterations, source):
    """Settings, per-seed statistics and their averages over training seeds for one point."""
    models_root, results_root = a.sources[source]
    summary = load_json(os.path.join(results_root, 'summary', f'{task}_{cfg}_iter{iterations}.json')) or {}
    n_episodes = len(a.eval_seeds) * a.episodes
    settings = {}
    for seed in a.seeds:
        settings = read_config(os.path.join(models_root, f'iter{iterations}', cfg, f'seed_{seed}', 'config.txt'))
        if settings:
            break
    if settings:
        settings.setdefault('beta_scale', '1')  # config.txt records beta only when it is not 1

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
        'value': value,
        'label': SWEEPS[sweep][1](value),
        'config': cfg,
        'iterations': iterations,
        'source': source,
        'models_root': models_root,
        'results_root': results_root,
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
    """Header and one row per point: label, settings, source, one cell per noise level."""
    header = ['Point'] + [name for _, name in SETTINGS] + ['Source'] + [f'Noise {ckey(c)}' for c in levels]
    body = [[r['label']] + [r['settings'][key] or '?' for key, _ in SETTINGS] + [r['source']]
            + [cell(r['scores'][ckey(c)][stat], n_total) for c in levels] for r in rows]
    return header, body


def print_table(header, body):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    for line in [header] + body:
        print('  ' + '  '.join(str(x).ljust(w) for x, w in zip(line, widths)).rstrip())


def markdown(sweep, task, rows, a):
    n = len(a.seeds)
    out = [f'# {SWEEPS[sweep][0]}: {task}', '',
           f'D4RL-normalized score: mean ± standard deviation over {n} training seeds '
           f'({", ".join(map(str, a.seeds))}). Each training seed pools {len(a.eval_seeds)} evaluation '
           f'seeds × {a.episodes} episodes per noise level; a cell marked (k/{n}) has only k complete seeds. '
           f'Source: where the point\'s models and results are.',
           '']
    for stat, name in STATS.items():
        header, body = table_rows(rows, a.corruptions, stat, n)
        out += [f'## {name}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        out += ['| ' + ' | '.join(str(x) for x in line) + ' |' for line in body]
        out.append('')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True, help='where the tables are written')
    p.add_argument('--source', nargs=3, action='append', required=True,
                   metavar=('NAME', 'MODELS_ROOT', 'RESULTS_ROOT'),
                   help='a source of points: trained models (train.sh -o) and their results (eval.sh -o)')
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='training seeds')
    p.add_argument('--eval-seeds', required=True, help='evaluation seeds, comma-separated')
    p.add_argument('--episodes', type=int, required=True, help='episodes per noise level and evaluation seed')
    p.add_argument('--corruptions', required=True, help='noise levels, comma-separated')
    p.add_argument('--point', nargs=6, action='append', required=True,
                   metavar=('SWEEP', 'TASK', 'VALUE', 'CONFIG', 'ITERATIONS', 'SOURCE'),
                   help='one point of a sweep: its config, Hopfield iterations and source')
    a = p.parse_args()
    a.sources = {name: (models, results) for name, models, results in a.source}
    a.eval_seeds = [int(s) for s in a.eval_seeds.split(',') if s.strip()]
    a.corruptions = [float(c) for c in a.corruptions.split(',') if c.strip()]
    for sweep, _, _, _, _, source in a.point:
        if sweep not in SWEEPS:
            p.error(f'unknown sweep {sweep!r}; sweeps: {", ".join(SWEEPS)}')
        if source not in a.sources:
            p.error(f'unknown source {source!r}; give it with --source')

    tables = {}
    for sweep, task, value, cfg, iterations, source in a.point:
        tables.setdefault((sweep, task), []).append(
            point_scores(a, sweep, task, value, cfg, int(iterations), source))

    os.makedirs(a.output_dir, exist_ok=True)
    n = len(a.seeds)
    for (sweep, task), rows in tables.items():
        result = {'sweep': sweep, 'task': task, 'training_seeds': a.seeds, 'eval_seeds': a.eval_seeds,
                  'episodes': a.episodes, 'noise_levels': a.corruptions, 'points': rows}
        json_path = os.path.join(a.output_dir, f'{sweep}_{task}.json')
        md_path = os.path.join(a.output_dir, f'{sweep}_{task}.md')
        write_atomic(json.dumps(result, indent=2) + '\n', json_path)
        write_atomic(markdown(sweep, task, rows, a), md_path)

        print(f'\n{SWEEPS[sweep][0]}, {task}: mean D4RL-normalized score ± std over {n} training seeds')
        print_table(*table_rows(rows, a.corruptions, 'mean', n))
        for r in rows:
            if all(v['mean'] is not None and v['mean']['n_seeds'] == n for v in r['scores'].values()):
                continue
            print(f"  {r['label']} is not fully evaluated in {r['results_root']}")
            if r['source'] != 'extended':
                print(f"    to evaluate it: bash {EVAL_SH} -s {','.join(map(str, a.seeds))} "
                      f"-c {r['config']}/{task} -i {r['iterations']} -E {a.episodes} "
                      f"-S {','.join(map(str, a.eval_seeds))} -C {','.join(map(str, a.corruptions))} "
                      f"-M {r['models_root']} -o {r['results_root']}")
        print(f'  Written: {json_path}\n           {md_path} (also median, IQM, CVaR-10)')


if __name__ == '__main__':
    main()
