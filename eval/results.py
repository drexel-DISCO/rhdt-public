#!/usr/bin/env python
"""
Result bookkeeping for eval.sh: completeness checks, job planning, pooled summaries.
Torch-free, so it is cheap to call once per eval.sh run.

Per-run results (written by eval_rhdt.py):
  <results_root>/iterations_<N>/seed_<train>/eval_seed_<es>/<cfg>/rhdt_<task>_corruptions.json

Pooled summary (written here), one file per (cfg, task):
  <results_root>/summary/<task>_<cfg>_iter<N>.json
  {
    "<train_seed>": {
      "<corruption>": {"mean", "std", "median", "iqm", "cvar10", "n_episodes", "eval_seeds"}
    }
  }
Statistics are over the normalized episode scores (raw_scores_<c>) pooled across
all eval seeds of that training seed, e.g. 10 eval seeds x 50 episodes = 500.

With --seed-summary-dir, summarize also writes the whole table per training seed:
  <seed_summary_dir>/rhdt_<seed>.json   {"<task>_<cfg>_iter<N>": {corruption: {stats}}}

Completeness is checked at every level and only the missing part is redone:
  job        result JSON complete                     -> skip
  level      raw_*/return*/score_<c> missing or short  -> rerun only that level
  field      header / degradation_<c> missing          -> rewrite from checkpoint, no rollouts
  summary    stat missing                              -> compute only that stat
             pool changed (eval seeds / n_episodes)    -> recompute the entry

    python results.py plan      <common args> --models-root DIR
    python results.py summarize <common args>
"""

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np

# Fields eval_rhdt.py writes. A JSON missing any of them is incomplete.
HEADER_KEYS = ('model_type', 'task', 'model_path', 'exp_name', 'num_heads',
               'lambda_gradient_penalty', 'seed', 'eval_episodes', 'iterations',
               'corruption_levels')
LEVEL_FIELDS = ('return', 'return_std', 'score', 'raw_returns', 'raw_scores')
LIST_FIELDS = ('raw_returns', 'raw_scores')

STAT_KEYS = ('mean', 'std', 'median', 'iqm', 'cvar10')
POOL_KEYS = ('n_episodes', 'eval_seeds')


def ckey(c):
    """Corruption level -> JSON key suffix. Matches f'{float}' in the eval script."""
    return str(float(c))


def cvar10(scores):
    """Mean of the worst 10% of episodes (at least one)."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = max(1, int(np.ceil(0.10 * len(s))))
    return float(s[:k].mean())


def iqm(scores):
    """Interquartile mean: mean of the middle 50% after dropping the lowest and highest
    25% of episodes. Same as scipy.stats.trim_mean(x, 0.25), as used by rliable."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = int(0.25 * len(s))
    return float(s[k:len(s) - k].mean())


STATS = {
    'mean': lambda x: float(np.mean(x)),
    'std': lambda x: float(np.std(x)),        # ddof=0, same as return_std_<c>
    'median': lambda x: float(np.median(x)),
    'iqm': iqm,
    'cvar10': cvar10,
}


# ---------------------------------------------------------------------------
# I/O

def load_json(path):
    """dict, or None if the file is absent, unreadable or not a JSON object."""
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def atomic_write_json(obj, path):
    """Write via temp file + rename so a killed job never leaves a half-written JSON."""
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@contextmanager
def locked(path):
    """Exclusive lock for read-modify-write of `path`, e.g. two runs on different GPUs
    finishing at the same time. The lock file exists only while the lock is held: the
    holder deletes it before releasing, and a process that locked a file which was
    deleted meanwhile (path now gone or a different file) retries on the current one."""
    import fcntl
    d, base = os.path.split(path)
    os.makedirs(d or '.', exist_ok=True)
    lock_path = os.path.join(d, f'.{base}.lock')
    while True:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            st, cur = os.fstat(fd), os.stat(lock_path)
            if (st.st_dev, st.st_ino) == (cur.st_dev, cur.st_ino):
                break
        except FileNotFoundError:
            pass
        os.close(fd)
    try:
        yield
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
        os.close(fd)  # releases the lock


# ---------------------------------------------------------------------------
# Per-run result JSON checks

def same_protocol(res, **expected):
    """Existing results are only reusable if produced under the same protocol.
    A field that is absent can't contradict; it gets filled on the next write."""
    for k, v in expected.items():
        if v is not None and k in res and res[k] != v:
            return False, f'{k} {res[k]}!={v}'
    return True, ''


def level_complete(res, c, n_episodes):
    k = ckey(c)
    for f in LEVEL_FIELDS:
        v = res.get(f'{f}_{k}')
        if f in LIST_FIELDS:
            if not isinstance(v, list) or len(v) != n_episodes:
                return False
        elif not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
    return True


def stored_levels(res):
    out = []
    for key in res:
        if key.startswith('raw_scores_'):
            try:
                out.append(float(key[len('raw_scores_'):]))
            except ValueError:
                pass
    return sorted(out)


def level_block(res, c):
    k = ckey(c)
    return {f'{f}_{k}': res[f'{f}_{k}'] for f in LEVEL_FIELDS}


def reusable_levels(existing, n_episodes, **expected):
    """{level: block} of complete levels in an existing result that can be kept."""
    if existing is None:
        return {}
    ok, why = same_protocol(existing, eval_episodes=n_episodes, **expected)
    if not ok:
        print(f"  existing result not reusable ({why}); re-evaluating all levels")
        return {}
    return {c: level_block(existing, c) for c in stored_levels(existing)
            if level_complete(existing, c, n_episodes)}


def assemble(header, blocks):
    """Header + per-level blocks (sorted) + degradation derived from the clean score."""
    levels = sorted(blocks)
    out = dict(header)
    out['corruption_levels'] = levels
    for c in levels:
        out.update(blocks[c])
    if 0.0 in blocks:
        clean = blocks[0.0][f'score_{ckey(0.0)}']
        for c in levels:
            if c > 0:
                out[f'degradation_{ckey(c)}'] = clean - blocks[c][f'score_{ckey(c)}']
    return out


def check_result(res, corruptions, n_episodes, **expected):
    """-> (levels_to_run, missing_fields, note). Both lists empty means complete."""
    if res is None:
        return list(corruptions), [], 'new'
    ok, why = same_protocol(res, eval_episodes=n_episodes, **expected)
    if not ok:
        return list(corruptions), [], f'redo ({why})'
    levels = [c for c in corruptions if not level_complete(res, c, n_episodes)]
    fields = [k for k in HEADER_KEYS if k not in res]
    done = [c for c in stored_levels(res) if level_complete(res, c, n_episodes)]
    if 0.0 in done:
        fields += [f'degradation_{ckey(c)}' for c in done
                   if c > 0 and f'degradation_{ckey(c)}' not in res]
    parts = []
    if levels:
        parts.append('levels ' + ','.join(ckey(c) for c in levels))
    if fields:
        parts.append('fields ' + ','.join(fields))
    return levels, fields, '; '.join(parts)


# ---------------------------------------------------------------------------
# Paths (single source of truth for eval.sh's layout)

def ckpt_path(models_root, cfg, seed, task):
    return os.path.join(models_root, cfg, f'seed_{seed}', f'rhdt_{task}.pt')


def result_dir(results_root, iterations, seed, eval_seed, cfg):
    return os.path.join(results_root, f'iterations_{iterations}', f'seed_{seed}',
                        f'eval_seed_{eval_seed}', cfg)


def result_file(results_root, iterations, seed, eval_seed, cfg, task):
    return os.path.join(result_dir(results_root, iterations, seed, eval_seed, cfg),
                        f'rhdt_{task}_corruptions.json')


def summary_file(results_root, iterations, cfg, task):
    return os.path.join(results_root, 'summary', f'{task}_{cfg}_iter{iterations}.json')


# ---------------------------------------------------------------------------
# Commands

def cmd_plan(a):
    """Print one TSV line per job that still has work to do:
    seed  cfg  task  eval_seed  ckpt  result_dir  note"""
    total = done = no_ckpt = 0
    warned = set()
    for seed in a.seeds:
        for cfg, task in a.runs:
            ckpt = ckpt_path(a.models_root, cfg, seed, task)
            for es in a.eval_seeds:
                total += 1
                res = load_json(result_file(a.results_root, a.iterations, seed, es, cfg, task))
                levels, fields, note = check_result(
                    res, a.corruptions, a.episodes,
                    seed=es, iterations=a.iterations, task=task)
                if not levels and not fields:
                    done += 1
                    continue
                if not os.path.isfile(ckpt):
                    no_ckpt += 1
                    if ckpt not in warned:
                        warned.add(ckpt)
                        print(f"WARNING: missing checkpoint {ckpt}", file=sys.stderr)
                    continue
                rdir = result_dir(a.results_root, a.iterations, seed, es, cfg)
                print('\t'.join(map(str, (seed, cfg, task, es, ckpt, rdir, note))))
    print(f"plan: {total} jobs | {done} complete | {total - done - no_ckpt} to run"
          f" | {no_ckpt} blocked by missing checkpoint", file=sys.stderr)


def _sorted_summary(summ):
    def num(s):
        try:
            return (0, float(s), s)
        except ValueError:
            return (1, 0.0, s)
    return {s: ({c: summ[s][c] for c in sorted(summ[s], key=num)}
                if isinstance(summ[s], dict) else summ[s])
            for s in sorted(summ, key=num)}


def cmd_summarize(a):
    n_expected = len(a.eval_seeds) * a.episodes
    reused = computed = incomplete = 0
    for cfg, task in a.runs:
        path = summary_file(a.results_root, a.iterations, cfg, task)
        rows = []
        with locked(path):
            summ = load_json(path)
            if summ is None:
                if os.path.exists(path):
                    print(f"WARNING: unreadable {path}; rebuilding it", file=sys.stderr)
                summ = {}
            changed = False
            for seed in a.seeds:
                cache = {}
                for c in a.corruptions:
                    per_seed = summ.get(str(seed))
                    entry = per_seed.get(ckey(c)) if isinstance(per_seed, dict) else None
                    entry = entry if isinstance(entry, dict) else {}
                    pool_ok = (entry.get('eval_seeds') == list(a.eval_seeds)
                               and entry.get('n_episodes') == n_expected)
                    todo = [k for k in STAT_KEYS if k not in entry] if pool_ok else list(STAT_KEYS)
                    if not todo:
                        reused += 1
                        rows.append((seed, c, entry, 'reused'))
                        continue

                    pool, missing = [], []
                    for es in a.eval_seeds:
                        if es not in cache:
                            cache[es] = load_json(result_file(
                                a.results_root, a.iterations, seed, es, cfg, task))
                        res = cache[es]
                        if (res is None
                                or not same_protocol(res, seed=es, eval_episodes=a.episodes,
                                                     iterations=a.iterations, task=task)[0]
                                or not level_complete(res, c, a.episodes)):
                            missing.append(es)
                            continue
                        pool.extend(res[f'raw_scores_{ckey(c)}'])
                    if missing:
                        incomplete += 1
                        print(f"  [incomplete] {cfg}/{task} seed={seed} corruption={ckey(c)}:"
                              f" no complete result for eval seeds {missing}", file=sys.stderr)
                        continue

                    new = dict(entry) if pool_ok else {}
                    for k in todo:
                        new[k] = STATS[k](pool)
                    new['n_episodes'] = len(pool)
                    new['eval_seeds'] = list(a.eval_seeds)
                    summ.setdefault(str(seed), {})[ckey(c)] = {k: new[k] for k in STAT_KEYS + POOL_KEYS}
                    changed = True
                    computed += 1
                    rows.append((seed, c, new, 'computed ' + ','.join(todo)))
            if changed:
                atomic_write_json(_sorted_summary(summ), path)

        if rows:
            print(f"\n{cfg}/{task} -> {path}")
            print(f"  {'seed':<8} {'corr':<6} {'mean':>7} {'std':>7} {'median':>7} {'iqm':>7} {'cvar10':>7}  n")
            for seed, c, e, how in rows:
                print(f"  {seed:<8} {ckey(c):<6} {e['mean']:>7.1f} {e['std']:>7.1f}"
                      f" {e['median']:>7.1f} {e['iqm']:>7.1f} {e['cvar10']:>7.1f}  {e['n_episodes']}  ({how})")
    print(f"\nsummary entries: {computed} computed | {reused} already complete"
          f" | {incomplete} skipped (incomplete pool)")
    if a.seed_summary_dir:
        write_seed_summaries(a)


def write_seed_summaries(a):
    """One file per training seed with the whole table printed above:
    <seed_summary_dir>/rhdt_<seed>.json
      {"<task>_<cfg>_iter<N>": {corruption: {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
    Keys are the per-task summary file names, so runs with other cfgs or iterations
    in the same file don't collide. Entries are merged per corruption level; runs and
    levels not in this invocation are kept."""
    levels = [ckey(c) for c in a.corruptions]
    for seed in a.seeds:
        path = os.path.join(a.seed_summary_dir, f'rhdt_{seed}.json')
        with locked(path):
            out = load_json(path) or {}
            for cfg, task in a.runs:
                per_seed = (load_json(summary_file(a.results_root, a.iterations, cfg, task)) or {}).get(str(seed))
                if not isinstance(per_seed, dict):
                    continue
                got = {c: per_seed[c] for c in levels if c in per_seed}
                if got:
                    key = f'{task}_{cfg}_iter{a.iterations}'
                    merged = {**out.get(key, {}), **got}
                    out[key] = {c: merged[c] for c in sorted(merged, key=float)}
            if out:
                atomic_write_json(out, path)
                print(f"seed summary: {path} ({len(out)} runs)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['plan', 'summarize'])
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='training seeds')
    p.add_argument('--runs', nargs='+', required=True, help='cfg/task pairs, e.g. default/hopper-medium-v2')
    p.add_argument('--eval-seeds', required=True, help='comma-separated')
    p.add_argument('--corruptions', required=True, help='comma-separated')
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--iterations', type=int, required=True)
    p.add_argument('--results-root', default='results_rhdt')
    p.add_argument('--models-root', default=None, help='required for plan')
    p.add_argument('--seed-summary-dir', default=None,
                   help='summarize: also write <dir>/rhdt_<seed>.json with the whole table per seed')
    a = p.parse_args()

    a.runs = [tuple(r.split('/', 1)) for r in a.runs]
    if any(len(r) != 2 for r in a.runs):
        p.error('--runs entries must look like cfg/task')
    a.eval_seeds = [int(s) for s in a.eval_seeds.split(',') if s.strip()]
    a.corruptions = [float(c) for c in a.corruptions.split(',') if c.strip()]

    if a.command == 'plan':
        if not a.models_root:
            p.error('plan needs --models-root')
        cmd_plan(a)
    else:
        cmd_summarize(a)


if __name__ == '__main__':
    main()
