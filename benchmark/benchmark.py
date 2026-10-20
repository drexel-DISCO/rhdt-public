#!/usr/bin/env python
"""
Compute cost of RHDT and the DT, RDT and ARDT baselines on one task.

Every method runs with its own code and the settings of its paper runs for the task:
RHDT's config from ../training/configs.sh, the baselines' rows of task_settings in
../dt/train_dt.sh, ../rdt/train_rdt.sh and ../ardt/train_ardt.sh. Batch 64, context 20.

  Parameters  of the policy that acts (ARDT: also its two value networks)
  Inference   one episode of --episode-steps steps in the task's environment with
              gradients off, timing every get_action call with the GPU synchronized
              before and after it. Untimed calls on an empty context come first (below).
              The peak memory includes the loaded model.
  Training    the full training step: forward pass, every loss term, backward pass,
              gradient clipping, optimizer and learning-rate step. --timed-steps steps
              are timed one by one with the GPU synchronized before and after each, so
              batch sampling stays outside the timer. Peak memory is
              torch.cuda.max_memory_allocated() over the timed steps.

Every timed measurement starts after an untimed warm-up of at least --warmup-steps
training steps (--inference-warmup get_action calls) and at least --warmup-seconds, so
GPU and CPU clocks are steady whatever ran before. Every method is measured in the same
order: inference, then training.

ARDT trains two value networks before its Decision Transformer. Its Decision Transformer
step is timed like DT's, on returns-to-go relabelled by an untimed pass first. After it,
the value networks' warm-up and minimax steps (batch 256) are timed the same way, and
the relabelling pass once, as extra stages. Weights are freshly initialized, since timing
and memory do not depend on their values. As in training, PyTorch runs in deterministic
mode.

Each method runs in its own process, so memory statistics never mix. Run it on an
otherwise idle GPU, in the environment from ../training/environment.yml:

    python benchmark.py --device cuda:0                       # all four methods
    python benchmark.py --methods rhdt,ardt --device cuda:0   # some of them
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic cuBLAS, as in training

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
for _sub in ('training', 'dt', 'rdt', 'ardt'):
    sys.path.insert(0, os.path.join(ROOT, _sub))

METHODS = ('rhdt', 'dt', 'rdt', 'ardt')
NAMES = {'rhdt': 'RHDT', 'dt': 'DT', 'rdt': 'RDT', 'ardt': 'ARDT'}
BATCH, CONTEXT, MAX_EP_LEN = 64, 20, 1000


# ==================== SETTINGS OF THE PAPER RUNS ====================

def task_row(launcher, task):
    """The values in a task's row of task_settings in a baseline's train_<x>.sh."""
    with open(os.path.join(ROOT, launcher)) as f:
        for line in f:
            m = re.match(r'\s*"(\S+)\s+([^"]*)"', line)
            if m and m.group(1) == task:
                return m.group(2).split()
    raise SystemExit(f'{launcher}: no task_settings row for {task}')


def shell_variable(path, name):
    with open(os.path.join(ROOT, path)) as f:
        m = re.search(rf'^{name}=(\S+)', f.read(), re.M)
    if not m:
        raise SystemExit(f'{path}: {name} not found')
    return m.group(1)


def rhdt_config(task):
    """(config, lambda_gp, lambda_sep, heads) of the task's paper config in configs.sh."""
    script = 'source "$1"; c=$(tuned_config "$2") || exit 1; echo "$c $(config_lambdas "$c") $(config_heads "$c")"'
    out = subprocess.run(['bash', '-c', script, 'bash', os.path.join(ROOT, 'training', 'configs.sh'), task],
                         capture_output=True, text=True, check=True).stdout.split()
    return out[0], float(out[1]), float(out[2]), int(out[3])


# ==================== METHODS ====================
# Each setup returns: settings, the paper's training steps, a function building a fresh
# policy, the evaluation statistics, and a function building (sample batch, train step).

def setup_rhdt(task, dataset, device, seed):
    import train_rhdt as T
    from data import build_trajectories, sample_batch
    from model import RHDT

    cfg, gp, sep, heads = rhdt_config(task)
    iterations = int(shell_variable('training/train.sh', 'ITERATIONS'))
    steps = int(shell_variable('training/train.sh', 'STEPS'))
    data = build_trajectories(dataset, max_ep_len=T.MAX_EP_LEN, scale=T.RTG_SCALE)
    sd, ad = dataset['observations'].shape[1], dataset['actions'].shape[1]

    def policy():
        return RHDT(sd, ad, embed_dim=T.EMBED_DIM, num_layers=T.NUM_LAYERS, num_heads=heads,
                    max_ep_len=T.MAX_EP_LEN, seq_len=T.SEQ_LEN, num_iterations=iterations, device=device)

    def training():
        model = policy()
        optimizer = AdamW(model.parameters(), lr=T.LEARNING_RATE, weight_decay=T.WEIGHT_DECAY)
        scheduler = LambdaLR(optimizer, lambda s: min((s + 1) / T.WARMUP_STEPS, 1))
        rng = np.random.RandomState(seed)

        def step(batch):
            T.train_step(model, optimizer, batch, gp, sep)
            scheduler.step()
        return lambda: sample_batch(data, T.BATCH_SIZE, T.SEQ_LEN, T.MAX_EP_LEN, device, rng), step

    settings = {'config': cfg, 'lambda_gp': gp, 'lambda_sep': sep, 'heads': heads,
                'hopfield_iterations': iterations}
    return {'settings': settings, 'run_steps': steps, 'policy': policy, 'stats': data, 'training': training}


def setup_dt(task, dataset, device, seed):
    import train_dt as D

    steps = int(task_row('dt/train_dt.sh', task)[0])
    info = D.create_sequences_from_dataset(dataset, max_ep_len=MAX_EP_LEN, discount=1.0)
    sd, ad = dataset['observations'].shape[1], dataset['actions'].shape[1]

    def policy():
        return D.DecisionTransformer(sd, ad, embed_dim=128, num_layers=3, num_heads=1, seq_len=CONTEXT,
                                     max_ep_len=MAX_EP_LEN, dropout=0.1, action_tanh=True, device=device)

    def training():
        D.reset_seed_for_model(seed)
        trainer = D.DTTrainer(policy())
        rng = np.random.RandomState(seed)
        return lambda: D.get_batch(info, BATCH, CONTEXT, MAX_EP_LEN, device, rng), trainer.train_step

    return {'settings': {}, 'run_steps': steps, 'policy': policy, 'stats': info, 'training': training}


def setup_rdt(task, dataset, device, seed):
    import train_rdt as R

    a, r, w, e = (float(x) for x in task_row('rdt/train_rdt.sh', task))
    steps = int(shell_variable('rdt/train_rdt.sh', 'OFFLINE_STEPS'))
    info = R.create_dataset_with_rewards(dataset, seq_len=CONTEXT, max_ep_len=MAX_EP_LEN, scale=1000.0)
    sd, ad = dataset['observations'].shape[1], dataset['actions'].shape[1]

    def policy():
        return R.RobustDecisionTransformer(sd, ad, embed_dim=128, num_layers=3, num_heads=1,
                                           max_ep_len=MAX_EP_LEN, seq_len=CONTEXT, dropout=0.1,
                                           embedding_dropout=e, action_tanh=True, predict_reward=True,
                                           device=device)

    def training():
        R.reset_seed_for_model(seed)
        trainer = R.RDTTrainer(policy(), learning_rate=1e-4, weight_decay=1e-4, warmup_steps=10000,
                               use_weighted_loss=True, wmse_action_coef=a, wmse_reward_coef=r,
                               reward_loss_coef=w)
        rng = np.random.RandomState(seed)
        return lambda: R.get_batch_with_rewards(info, BATCH, CONTEXT, MAX_EP_LEN, device, rng), trainer.train_step

    settings = {'beta_a': a, 'beta_r': r, 'reward_loss_weight': w, 'embedding_dropout': e}
    return {'settings': settings, 'run_steps': steps, 'policy': policy, 'stats': info, 'training': training}


def setup_ardt(task, dataset, device, seed):
    import train_ardt as A

    alpha, steps, warmup, minimax = task_row('ardt/train_ardt.sh', task)
    alpha, steps, warmup, minimax = float(alpha), int(steps), int(warmup), int(minimax)
    info = A.create_dataset_for_ardt(dataset, max_ep_len=MAX_EP_LEN)
    sd, ad = dataset['observations'].shape[1], dataset['actions'].shape[1]

    def policy():
        return A.DecisionTransformer(sd, ad, embed_dim=128, num_layers=3, num_heads=1, seq_len=CONTEXT,
                                     max_ep_len=MAX_EP_LEN, dropout=0.1, action_tanh=True, device=device)

    def estimator():
        A.reset_seed_for_model(seed)
        return A.MinimaxReturnEstimator(sd, ad, hidden_dim=256, n_layers=3, alpha=alpha,
                                        learning_rate=3e-4, device=device)

    def prepare():
        """Untimed relabelling, so the Decision Transformer's batches have returns-to-go."""
        A.relabel_trajectories_with_minimax(info, estimator(), device)

    def training():
        A.reset_seed_for_model(seed + 1000)
        trainer = A.DTTrainer(policy())
        rng = np.random.RandomState(seed + 2000)
        return lambda: A.get_batch_for_dt(info, BATCH, CONTEXT, MAX_EP_LEN, device, rng), trainer.train_step

    def stages(warmup_args, n_timed, dev):
        """Value-network warm-up and minimax steps, and the relabelling pass."""
        est = estimator()
        rng = np.random.RandomState(seed)

        def sample():
            return A.get_batch_for_minimax(info, 256, device, rng)

        def warmup_step(b):
            est.warmup_step(b['states'], b['actions'], b['returns_to_go'])

        def minimax_step(b):
            est.minimax_step(b['states'], b['actions'], b['rewards'], b['next_states'], b['next_actions'],
                             b['dones'])

        out = {'value_network_parameters': n_params(est.q_lower) + n_params(est.q_upper)}
        for name, step, run_steps in (('value_warmup', warmup_step, warmup), ('minimax', minimax_step, minimax)):
            times, peak, warm = time_steps(sample, step, *warmup_args, n_timed, dev)
            out[name] = {**ms_stats(times), 'peak_mib': mib(peak), 'run_steps': run_steps, 'warmup': warm}
        out['relabel_s'] = timed(lambda: A.relabel_trajectories_with_minimax(info, est, device), dev)[0]
        return out

    from ckpt_selection import add_selection_args
    selection = add_selection_args(argparse.ArgumentParser()).parse_args([])
    settings = {'alpha': alpha, 'warmup_steps': warmup, 'minimax_steps': minimax,
                'validation_episodes': selection.ckpt_keep_last * selection.val_episodes}
    return {'settings': settings, 'run_steps': steps, 'policy': policy, 'stats': info, 'training': training,
            'prepare': prepare, 'stages': stages}


SETUPS = {'rhdt': setup_rhdt, 'dt': setup_dt, 'rdt': setup_rdt, 'ardt': setup_ardt}


# ==================== MEASUREMENT ====================

def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def timed(fn, dev):
    """(seconds, result) of fn(), with the GPU synchronized before and after."""
    sync(dev)
    t0 = time.perf_counter()
    out = fn()
    sync(dev)
    return time.perf_counter() - t0, out


def reset_peak(dev):
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def peak_bytes(dev):
    return torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else None


def mib(n):
    return None if n is None else n / 2 ** 20


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def ms_stats(seconds):
    ms = np.array(seconds) * 1000
    return {'mean_ms': float(ms.mean()), 'std_ms': float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
            'median_ms': float(np.median(ms)), 'p95_ms': float(np.percentile(ms, 95)), 'n': len(ms)}


def warm_up(fn, n_min, seconds, dev):
    """Run fn() untimed at least n_min times and for at least `seconds`; returns (runs, seconds)."""
    sync(dev)
    start = time.perf_counter()
    n = 0
    while n < n_min or time.perf_counter() - start < seconds:
        fn()
        n += 1
    sync(dev)
    return {'runs': n, 'seconds': time.perf_counter() - start}


def time_steps(sample, step, n_warmup, warmup_s, n_timed, dev):
    """Per-step times of n_timed training steps after the warm-up, the peak memory, and the warm-up."""
    warm = warm_up(lambda: step(sample()), n_warmup, warmup_s, dev)
    reset_peak(dev)
    times = []
    for _ in range(n_timed):
        batch = sample()  # outside the timer
        times.append(timed(lambda: step(batch), dev)[0])
    return times, peak_bytes(dev), warm


def time_episode(policy, env, stats, n_steps, n_warmup, warmup_s, dev, seed):
    """Seconds of every get_action call in one episode (the evaluation rollout of ../training),
    the episode's wall-clock seconds, and the warm-up."""
    policy.eval()
    obs_mean = torch.as_tensor(stats['obs_mean'], dtype=torch.float32, device=dev)
    obs_std = torch.as_tensor(stats['obs_std'], dtype=torch.float32, device=dev)
    ad = policy.action_dim
    times = []
    with torch.no_grad():
        empty = (torch.zeros(1, policy.state_dim, device=dev), torch.zeros(1, ad, device=dev),
                 torch.zeros(1, 1, device=dev), torch.zeros(1, 1, dtype=torch.long, device=dev))
        warm = warm_up(lambda: policy.get_action(*empty), n_warmup, warmup_s, dev)
        start = time.perf_counter()
        obs = env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        states = torch.as_tensor(obs, dtype=torch.float32, device=dev).view(1, -1)
        actions = torch.zeros(0, ad, device=dev)
        rtg = torch.tensor([[float(stats['target_return'])]], dtype=torch.float32, device=dev)
        timesteps = torch.zeros(1, 1, dtype=torch.long, device=dev)
        for t in range(n_steps):
            actions = torch.cat([actions, torch.zeros(1, ad, device=dev)])
            inputs = (states - obs_mean) / obs_std
            seconds, action = timed(lambda: policy.get_action(inputs, actions, rtg, timesteps), dev)
            times.append(seconds)
            actions[-1] = action
            result = env.step(action.cpu().numpy())
            obs, reward = result[0], result[1]
            done = result[2] if len(result) == 4 else (result[2] or result[3])
            if done:
                break
            states = torch.cat([states, torch.as_tensor(obs, dtype=torch.float32, device=dev).view(1, -1)])
            rtg = torch.cat([rtg, (rtg[0, -1] - float(reward) / stats['scale']).view(1, 1)], 1)
            timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], dtype=torch.long, device=dev)], 1)
    return times, time.perf_counter() - start, warm


def run_method(method, args):
    from data import load_task
    from train_rhdt import set_seed

    dev = torch.device(args.device)
    set_seed(args.seed)
    dataset, env = load_task(args.task)
    m = SETUPS[method](args.task, dataset, args.device, args.seed)
    result = {'method': NAMES[method], 'task': args.task, 'settings': m['settings'], 'device': args.device,
              'gpu': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'deterministic': torch.are_deterministic_algorithms_enabled()}

    # 1. Inference: a fresh model, gradients off
    reset_peak(dev)
    model = m['policy']()
    result['parameters'] = n_params(model)
    weights = mib(torch.cuda.memory_allocated(dev)) if dev.type == 'cuda' else None
    calls, episode_s, warm = time_episode(model, env, m['stats'], args.episode_steps, args.inference_warmup,
                                          args.warmup_seconds, dev, args.seed)
    result['inference'] = {**ms_stats(calls), 'episode_s': episode_s, 'peak_mib': mib(peak_bytes(dev)),
                           'weights_mib': weights, 'warmup': warm}
    del model
    reset_peak(dev)

    # 2. Training step (ARDT: its Decision Transformer, on relabelled returns-to-go)
    if 'prepare' in m:
        m['prepare']()
    sample, step = m['training']()
    times, peak, warm = time_steps(sample, step, args.warmup_steps, args.warmup_seconds, args.timed_steps, dev)
    result['training'] = {**ms_stats(times), 'peak_mib': mib(peak), 'run_steps': m['run_steps'], 'warmup': warm}
    del sample, step
    reset_peak(dev)

    # 3. ARDT's value-network stages
    if 'stages' in m:
        result['stages'] = m['stages']((args.warmup_steps, args.warmup_seconds), args.timed_steps, dev)
    env.close()
    return result


# ==================== REPORT ====================

def hours(seconds):
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}"


def fmt(x, spec='.1f'):
    return '-' if x is None else format(x, spec)


def print_table(header, rows, left=1):
    """Columns sized to fit; the first `left` columns left-aligned, the rest right-aligned."""
    widths = [max(len(str(x)) for x in col) for col in zip(*header, *rows)]
    for line in header + rows:
        cells = [str(x).ljust(w) if i < left else str(x).rjust(w) for i, (x, w) in enumerate(zip(line, widths))]
        print('  '.join(cells).rstrip())


def report(results, args):
    first = results[0]
    print(f"\n{args.task} on {first['gpu'] or args.device} | PyTorch {first['torch']} (CUDA {first['cuda']}) | "
          f"deterministic algorithms: {'on' if first['deterministic'] else 'off'}")
    print(f"batch {BATCH}, context {CONTEXT} | training: {args.timed_steps} timed steps | "
          f"inference: one {args.episode_steps}-step episode | each after an untimed warm-up of at least "
          f"{args.warmup_steps} steps ({args.inference_warmup} calls) and {args.warmup_seconds:g} s\n")
    header = [['', '', 'Training step', '', '', '', 'get_action', '', '', ''],
              ['Method', 'Parameters', 'mean ms', 'std', 'median', 'peak MiB', 'mean ms', 'median', 'p95', 'peak MiB']]
    rows = []
    for r in results:
        t, i = r['training'], r['inference']
        rows.append([r['method'], f"{r['parameters']:,}",
                     f"{t['mean_ms']:.2f}", f"{t['std_ms']:.2f}", f"{t['median_ms']:.2f}", fmt(t['peak_mib']),
                     f"{i['mean_ms']:.3f}", f"{i['median_ms']:.3f}", f"{i['p95_ms']:.3f}", fmt(i['peak_mib'])])
    print_table(header, rows)
    for r in results:
        if r['inference']['n'] < args.episode_steps:
            print(f"  {r['method']}: the episode ended after {r['inference']['n']} steps")

    for r in results:
        if 'stages' not in r:
            continue
        s = r['stages']
        print(f"\n{r['method']} extra stages: two value networks, {s['value_network_parameters']:,} parameters, "
              f"batch 256")
        for name, label in (('value_warmup', 'value warm-up step'), ('minimax', 'minimax step')):
            x = s[name]
            print(f"  {label:<19}{x['mean_ms']:8.2f} ± {x['std_ms']:.2f} ms  peak {fmt(x['peak_mib'])} MiB"
                  f"  ({x['run_steps']:,} steps per run)")
        print(f"  {'relabelling pass':<19}{s['relabel_s']:8.2f} s  (once per run)")

    by_name = {r['method']: r['training'] for r in results}
    if 'DT' in by_name and 'ARDT' in by_name:
        d, a = by_name['DT']['mean_ms'], by_name['ARDT']['mean_ms']
        print(f"\nCheck: ARDT's Decision Transformer step is DT's network, batch and loss: "
              f"{a:.2f} ms vs DT's {d:.2f} ms ({100 * (a - d) / d:+.1f}%)")

    print("\nTraining time per run, from the mean step times (batch sampling and evaluation excluded):")
    for r in results:
        t = r['training']
        total = t['run_steps'] * t['mean_ms'] / 1000
        line = f"{t['run_steps']:,} steps"
        if 'stages' in r:
            s = r['stages']
            total += sum(s[k]['run_steps'] * s[k]['mean_ms'] / 1000 for k in ('value_warmup', 'minimax'))
            total += s['relabel_s']
            line = (f"{s['value_warmup']['run_steps']:,} warm-up + {s['minimax']['run_steps']:,} minimax steps, "
                    f"relabelling, {t['run_steps']:,} DT steps")
            extra = r['settings']['validation_episodes'] * r['inference']['episode_s']
            line += (f"; checkpoint validation adds {r['settings']['validation_episodes']} episodes, "
                     f"about {hours(extra)}")
        print(f"  {r['method']:<6}{hours(total):>6} h:mm  ({line})")


# ==================== MAIN ====================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', default=','.join(METHODS),
                   help=f"comma-separated, from {', '.join(METHODS)} (default: all)")
    p.add_argument('--task', default='halfcheetah-medium-v2', help='D4RL task (default: %(default)s)')
    p.add_argument('--device', default='cuda:0', help='torch device (default: %(default)s)')
    p.add_argument('--seed', type=int, default=42, help='seed (default: %(default)s)')
    p.add_argument('--warmup-steps', type=int, default=100,
                   help='untimed training steps before timing, at least (default: %(default)s)')
    p.add_argument('--warmup-seconds', type=float, default=10.0,
                   help='untimed warm-up before every timed measurement, at least (default: %(default)s)')
    p.add_argument('--timed-steps', type=int, default=1000, help='timed training steps (default: %(default)s)')
    p.add_argument('--episode-steps', type=int, default=1000, help='steps of the timed episode (default: %(default)s)')
    p.add_argument('--inference-warmup', type=int, default=10,
                   help='untimed get_action calls before the episode, at least (default: %(default)s)')
    p.add_argument('--out', default='benchmark_results.json', help='results file (default: %(default)s)')
    p.add_argument('--child', help=argparse.SUPPRESS)  # internal: run one method, write its result here
    a = p.parse_args()
    a.methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    unknown = sorted(set(a.methods) - set(METHODS))
    if unknown or not a.methods:
        p.error(f"unknown method(s) {', '.join(unknown)}; choose from {', '.join(METHODS)}")
    if min(a.warmup_steps, a.inference_warmup, a.warmup_seconds, a.timed_steps - 1, a.episode_steps - 1) < 0:
        p.error('--timed-steps and --episode-steps need at least 1, the warm-ups at least 0')
    return a


def main():
    args = parse_args()
    if args.child:
        with open(args.child, 'w') as f:
            json.dump(run_method(args.methods[0], args), f)
        return

    results = []
    for method in args.methods:
        print(f"==== {NAMES[method]} ====", flush=True)
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        cmd = [sys.executable, os.path.abspath(__file__), '--methods', method, '--child', path,
               '--task', args.task, '--device', args.device, '--seed', str(args.seed),
               '--warmup-steps', str(args.warmup_steps), '--warmup-seconds', str(args.warmup_seconds),
               '--timed-steps', str(args.timed_steps),
               '--episode-steps', str(args.episode_steps), '--inference-warmup', str(args.inference_warmup)]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            with open(path) as f:
                results.append(json.load(f))
        else:
            print(f"{NAMES[method]} failed (exit code {rc}); see its output above", file=sys.stderr)
        os.remove(path)
    if not results:
        sys.exit(1)
    report(results, args)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll numbers: {args.out}")
    if len(results) < len(args.methods):
        sys.exit(1)


if __name__ == '__main__':
    main()
