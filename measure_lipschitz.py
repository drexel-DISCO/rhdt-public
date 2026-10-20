"""Measure local Lipschitz constants of trained RHDT policies.

For each checkpoint, the script samples the same training segments (fixed data seed),
computes the full Jacobian of all predicted actions in a segment with respect to each
state, and reports per-state statistics at clean and at noise-corrupted states:

  lip   = ||J_t||_2       local Lipschitz constant of the policy at s_t (spectral norm)
  fro2  = ||J_t||_F^2     what the randomized (Hutchinson) gradient penalty targets
  pen   = ||J_t^T 1||^2   what the training gradient penalty uses

J_t has one row per predicted action dimension at every valid step of the segment and one
column per state dimension of s_t. Padded steps are excluded as inputs and as outputs.
Noise is added in normalized state coordinates as sigma * N(0, 1), which equals noise with
standard deviation sigma times the dataset standard deviation in raw coordinates, as in
evaluation. Every model sees the same segments and the same noise.

Place this file in eval/ (next to eval_rhdt.py). Example, HalfCheetah Medium ablation:

  python measure_lipschitz.py --task halfcheetah-medium-v2 \
      --seeds 907561,48213,2024887,739104,26451,42 --device cuda:0 \
      --model full=../training/models_rhdt_sweep/iter1/gpheavy/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --model ortho_only=../training/models_rhdt_sweep/iter1/gpheavy_sep/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --model gp_only=../training/models_rhdt_sweep/iter1/gpheavy_gp/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --model none=../training/models_rhdt_sweep/iter1/non/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --out lipschitz_halfcheetah-medium.json

Run once with --check first: it compares the autograd Jacobian with finite differences.
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [os.path.join(_here, os.pardir, 'training'), os.path.join(_here, 'training')]
_training = next((d for d in _candidates if os.path.isfile(os.path.join(d, 'model.py'))), None)
if _training is None:
    sys.exit("Cannot find the RHDT training/ folder (model.py, data.py, train_rhdt.py). "
             "Put this script in eval/ or in the repository root, next to training/.")
sys.path.insert(0, _training)
from data import build_trajectories, load_task, sample_batch  # noqa: E402
from model import RHDT  # noqa: E402
from train_rhdt import MAX_EP_LEN, RTG_SCALE, SEQ_LEN  # noqa: E402


def load_model(path, device):
    """Build the model exactly as eval_rhdt.py does, in eval mode (dropout off)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    m = checkpoint['metadata']
    model = RHDT(m['state_dim'], m['action_dim'],
                 embed_dim=m.get('embed_dim', 128), num_layers=m.get('num_layers', 3),
                 num_heads=m.get('num_heads', 4), max_ep_len=m.get('max_ep_len', 1000),
                 seq_len=m.get('seq_len', 20), num_iterations=m.get('num_iterations', 1), device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, m


def predict(model, batch, states):
    return model(states, batch['actions'], batch['rtg'], batch['timesteps'], batch['mask'])  # B, C, A


def jacobians(model, batch, states):
    """Per-state Jacobians, shape (B, C, C*A, S): row block = all actions, columns = s_t."""
    states = states.detach().clone().requires_grad_(True)
    preds = predict(model, batch, states)
    B, C, A = preds.shape
    rows = []
    for tp in range(C):
        for j in range(A):
            g, = torch.autograd.grad(preds[:, tp, j].sum(), states, retain_graph=True)
            rows.append(g)                                    # d a_{tp,j} / d s_t, shape B, C, S
    J = torch.stack(rows, dim=1).view(B, C, A, C, -1)         # B, tp, j, t, S
    J = J * batch['mask'][:, :, None, None, None]             # drop padded outputs
    return J.permute(0, 3, 1, 2, 4).reshape(B, C, C * A, -1)  # B, t, (tp, j), S


def stats_for(J, mask):
    valid = mask.bool()
    lip = torch.linalg.matrix_norm(J, ord=2)[valid]
    fro2 = (J ** 2).sum(dim=(-1, -2))[valid]
    pen = (J.sum(dim=-2) ** 2).sum(dim=-1)[valid]
    return lip, fro2, pen


def noisy(states, mask, sigma, seed):
    if sigma == 0:
        return states
    g = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(states.shape, generator=g).to(states.device)
    return states + sigma * noise * mask[..., None]


def check(model, batch, eps=1e-3):
    """Compare J_t u with a central finite difference for a few random (b, t, u)."""
    J = jacobians(model, batch, batch['states'])
    rng = np.random.RandomState(0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(5):
            valid = torch.nonzero(batch['mask']).cpu().numpy()
            b, t = valid[rng.randint(len(valid))]
            u = torch.randn(batch['states'].shape[-1], device=batch['states'].device)
            u = u / u.norm()
            plus, minus = batch['states'].clone(), batch['states'].clone()
            plus[b, t] += eps * u
            minus[b, t] -= eps * u
            fd = ((predict(model, batch, plus) - predict(model, batch, minus))[b] / (2 * eps))
            fd = (fd * batch['mask'][b][:, None]).reshape(-1)
            ad = J[b, t] @ u
            rel = ((fd - ad).norm() / (ad.norm() + 1e-12)).item()
            worst = max(worst, rel)
    print(f"Jacobian check: largest relative difference to finite differences = {worst:.2e} "
          f"({'OK' if worst < 1e-2 else 'CHECK FAILED'})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--model', action='append', required=True,
                   help='LABEL=PATH, where PATH may contain {seed}; repeat for each model')
    p.add_argument('--seeds', required=True, help='comma-separated training seeds')
    p.add_argument('--sigmas', default='0.0,0.3', help='noise levels (multiples of the dataset std)')
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--data-seed', type=int, default=0, help='seed for the sampled segments and noise')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=None, help='write all per-seed results to this JSON file')
    p.add_argument('--check', action='store_true', help='only run the finite-difference check on the first model')
    args = p.parse_args()

    models = []
    for item in args.model:
        label, _, path = item.partition('=')
        if not path:
            p.error(f'--model needs LABEL=PATH, got {item!r}')
        models.append((label, path))
    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    sigmas = [float(x) for x in args.sigmas.split(',')]

    dataset, _ = load_task(args.task)
    data = build_trajectories(dataset, max_ep_len=MAX_EP_LEN, scale=RTG_SCALE)
    rng = np.random.RandomState(args.data_seed)
    batches = [sample_batch(data, args.batch_size, SEQ_LEN, MAX_EP_LEN, args.device, rng) for _ in range(args.batches)]

    if args.check:
        model, _ = load_model(models[0][1].format(seed=seeds[0]), args.device)
        check(model, batches[0])
        return

    results = {}
    for label, template in models:
        for seed in seeds:
            path = template.format(seed=seed)
            model, meta = load_model(path, args.device)
            for sigma in sigmas:
                lips, fros, pens = [], [], []
                for i, batch in enumerate(batches):
                    states = noisy(batch['states'], batch['mask'], sigma, args.data_seed * 1000 + i)
                    lip, fro2, pen = stats_for(jacobians(model, batch, states), batch['mask'])
                    lips.append(lip.cpu()); fros.append(fro2.cpu()); pens.append(pen.cpu())
                lip, fro2, pen = torch.cat(lips), torch.cat(fros), torch.cat(pens)
                results.setdefault(label, {}).setdefault(str(sigma), {})[seed] = {
                    'lip_mean': lip.mean().item(), 'lip_median': lip.median().item(),
                    'fro2_mean': fro2.mean().item(), 'pen_mean': pen.mean().item(),
                    'n_states': int(lip.numel()), 'checkpoint': path,
                    'lambda_gp': meta.get('lambda_gp'), 'lambda_sep': meta.get('lambda_sep'),
                    'num_heads': meta.get('num_heads'), 'num_iterations': meta.get('num_iterations', 1),
                }
            print(f"done: {label} seed {seed}")

    print(f"\n{args.task}: mean over states per seed, then mean ± std across {len(seeds)} seeds")
    print(f"{'model':<14}{'sigma':>6}{'||J||_2 mean':>20}{'||J||_2 median':>20}{'||J||_F^2':>20}{'||J^T 1||^2':>20}")
    for label, by_sigma in results.items():
        for sigma, per_seed in by_sigma.items():
            row = f"{label:<14}{float(sigma):>6.1f}"
            for key in ('lip_mean', 'lip_median', 'fro2_mean', 'pen_mean'):
                vals = [v[key] for v in per_seed.values()]
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"{statistics.mean(vals):>13.3f} ± {sd:<4.2f}"
            print(row)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'task': args.task, 'seeds': seeds, 'data_seed': args.data_seed,
                       'batches': args.batches, 'batch_size': args.batch_size, 'results': results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
