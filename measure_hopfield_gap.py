"""Does the Hopfield update pull corrupted queries toward clean ones?

For each checkpoint, the script runs the same training segments (fixed data seed) twice,
once with clean states and once with noise-corrupted states, and records in every attention
layer, for every state token and head:

  q0   the query before the Hopfield update
  q1   the query after the update(s)
  r1   the readout softmax(beta q1 K^T) V that the layer actually uses
  r0   the readout softmax(beta q0 K^T) V the layer would use without the update
       (same keys and values; a counterfactual within the same trained model)

Noise enters at the input, so the corrupted pass changes the keys and values as well as the
queries. The script therefore also updates both queries against the SAME keys:

  q1_kc  clean and corrupted queries both updated with the clean keys
         (intact memory, as in the attractor analysis and Theorem 1)
  q1_kn  clean and corrupted queries both updated with the corrupted keys
         (the memory the model actually has at test time)

It reports clean-vs-corrupted gaps, relative to the size of the clean vector because the
update also rescales the queries:

  rel_q0, rel_q1   mean ||x_noisy - x_clean|| / mean ||x_clean|| for q0 and q1
  q_ratio          rel_q1 / rel_q0         (< 1: the update shrinks the relative gap; keys differ)
  kc_ratio         gap after updating both queries with the clean keys / rel_q0
  kn_ratio         gap after updating both queries with the corrupted keys / rel_q0
                   (< 1: with the keys held fixed, the update shrinks the query perturbation)
  q_shrink         fraction of queries whose own relative gap shrinks
  cos_q0, cos_q1   mean cosine similarity of clean and corrupted queries
  rel_r0, rel_r1   as above for the readouts without and with the update
  r_ratio          rel_r1 / rel_r0         (< 1: the update reduces what the layer passes on)

Noise is sigma * N(0, 1) in normalized state coordinates, i.e. sigma times the dataset
standard deviation, as in evaluation and in measure_lipschitz.py (same segments and noise for
the same --data-seed). Place this file in eval/ (or the repository root). Example:

  python measure_hopfield_gap.py --task halfcheetah-medium-v2 \
      --seeds 907561,48213,2024887,739104,26451,42 --sigmas 0.1,0.2,0.3 --device cuda:0 \
      --model full=/path/to/models/iter1/gpheavy/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --model none=/path/to/models/iter1/non/seed_{seed}/rhdt_halfcheetah-medium-v2.pt \
      --out hopfield_gap_halfcheetah-medium.json

Run once with --check first: it confirms that recording leaves the model's outputs unchanged.
"""
import argparse
import json
import os
import statistics
import sys
import types

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


def recording_forward(self, x, attention_mask=None):
    """Same computation as HopfieldAttention.forward, plus a record of q0, q1, r0 and r1."""
    B, T, C = x.shape
    Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    self.last_K = K if self.training else K.detach()

    mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        mask = mask * attention_mask.unsqueeze(1).unsqueeze(2)

    q = Q
    for _ in range(self.num_iterations):
        q = torch.matmul(self._attention_weights(q, K, mask), K)

    weights = self.dropout(self._attention_weights(q, K, mask))
    readout = torch.matmul(weights, V)                                  # B, H, T, head_dim
    readout_without = torch.matmul(self._attention_weights(Q, K, mask), V)
    self._record = {'q0': Q.detach(), 'q1': q.detach(), 'r1': readout.detach(), 'r0': readout_without.detach(),
                    'K': K.detach(), 'mask': mask}

    out = readout.transpose(1, 2).contiguous().view(B, T, C)
    return self.out_proj(out)


def attach_recorders(model):
    for block in model.blocks:
        block.attn.forward = types.MethodType(recording_forward, block.attn)


def hopfield_update(attn, q, K, mask):
    """The layer's Hopfield update of queries q against keys K (same iterations and mask handling)."""
    with torch.no_grad():
        for _ in range(attn.num_iterations):
            q = torch.matmul(attn._attention_weights(q, K, mask), K)
    return q


def predict(model, batch, states):
    return model(states, batch['actions'], batch['rtg'], batch['timesteps'], batch['mask'])


def noisy(states, mask, sigma, seed):
    if sigma == 0:
        return states
    g = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(states.shape, generator=g).to(states.device)
    return states + sigma * noise * mask[..., None]


def records(model, batch, states):
    with torch.no_grad():
        predict(model, batch, states)
    return [{k: v.clone() for k, v in block.attn._record.items()} for block in model.blocks]


def state_tokens(x, mask):
    """Rows for the state tokens (positions 1, 4, 7, ...) of valid timesteps: shape (n, head_dim)."""
    s = x[:, :, 1::3, :]                                                # B, H, C, head_dim
    valid = mask.bool()[:, None, :].expand(-1, s.shape[1], -1)          # B, H, C
    return s[valid]


class GapAccumulator:
    def __init__(self):
        self.sums = {}

    def add(self, name, clean, corrupted):
        diff = (corrupted - clean).norm(dim=-1)
        size = clean.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(clean, corrupted, dim=-1)
        d = self.sums.setdefault(name, {'diff': 0.0, 'size': 0.0, 'cos': 0.0, 'n': 0, 'rel': []})
        d['diff'] += diff.sum().item()
        d['size'] += size.sum().item()
        d['cos'] += cos.sum().item()
        d['n'] += diff.numel()
        d['rel'].append((diff / size.clamp_min(1e-12)).cpu())

    def rel(self, name):
        d = self.sums[name]
        return d['diff'] / d['size']

    def cos(self, name):
        d = self.sums[name]
        return d['cos'] / d['n']

    def per_vector_rel(self, name):
        return torch.cat(self.sums[name]['rel'])


def measure(model, batches, sigma, data_seed):
    layers = [GapAccumulator() for _ in model.blocks]
    for i, batch in enumerate(batches):
        clean = records(model, batch, batch['states'])
        corrupted = records(model, batch, noisy(batch['states'], batch['mask'], sigma, data_seed * 1000 + i))
        for block, acc, rc, rn in zip(model.blocks, layers, clean, corrupted):
            for name in ('q0', 'q1', 'r0', 'r1'):
                acc.add(name, state_tokens(rc[name], batch['mask']), state_tokens(rn[name], batch['mask']))
            mask = rc['mask']  # the attention mask is the same in both passes
            # both queries updated against the clean keys: rc['q1'] is the clean query's own update
            corrupted_kc = hopfield_update(block.attn, rn['q0'], rc['K'], mask)
            acc.add('q1_kc', state_tokens(rc['q1'], batch['mask']), state_tokens(corrupted_kc, batch['mask']))
            # both queries updated against the corrupted keys: rn['q1'] is the corrupted query's own update
            clean_kn = hopfield_update(block.attn, rc['q0'], rn['K'], mask)
            acc.add('q1_kn', state_tokens(clean_kn, batch['mask']), state_tokens(rn['q1'], batch['mask']))
    out = []
    for acc in layers:
        q_shrink = (acc.per_vector_rel('q1') < acc.per_vector_rel('q0')).float().mean().item()
        out.append({
            'rel_q0': acc.rel('q0'), 'rel_q1': acc.rel('q1'), 'q_ratio': acc.rel('q1') / acc.rel('q0'),
            'q_shrink': q_shrink, 'cos_q0': acc.cos('q0'), 'cos_q1': acc.cos('q1'),
            'rel_r0': acc.rel('r0'), 'rel_r1': acc.rel('r1'), 'r_ratio': acc.rel('r1') / acc.rel('r0'),
            'kc_ratio': acc.rel('q1_kc') / acc.rel('q0'), 'kn_ratio': acc.rel('q1_kn') / acc.rel('q0'),
            'kc_shrink': (acc.per_vector_rel('q1_kc') < acc.per_vector_rel('q0')).float().mean().item(),
            'kn_shrink': (acc.per_vector_rel('q1_kn') < acc.per_vector_rel('q0')).float().mean().item(),
            'cos_q1_kc': acc.cos('q1_kc'), 'cos_q1_kn': acc.cos('q1_kn'),
            'n_vectors': acc.sums['q0']['n'],
        })
    return out


def check(model, batch):
    with torch.no_grad():
        before = predict(model, batch, batch['states'])
        attach_recorders(model)
        after = predict(model, batch, batch['states'])
    diff = (before - after).abs().max().item()
    print(f"Recording check: largest output difference with and without recording = {diff:.2e} "
          f"({'OK' if diff < 1e-5 else 'CHECK FAILED'})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--model', action='append', required=True,
                   help='LABEL=PATH, where PATH may contain {seed}; repeat for each model')
    p.add_argument('--seeds', required=True, help='comma-separated training seeds')
    p.add_argument('--sigmas', default='0.1,0.2,0.3', help='noise levels (multiples of the dataset std)')
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--data-seed', type=int, default=0, help='seed for the sampled segments and noise')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=None, help='write all per-seed results to this JSON file')
    p.add_argument('--check', action='store_true', help='only check that recording leaves outputs unchanged')
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
            attach_recorders(model)
            for sigma in sigmas:
                results.setdefault(label, {}).setdefault(str(sigma), {})[seed] = {
                    'layers': measure(model, batches, sigma, args.data_seed), 'checkpoint': path,
                    'num_iterations': meta.get('num_iterations', 1), 'num_heads': meta.get('num_heads'),
                    'lambda_gp': meta.get('lambda_gp'), 'lambda_sep': meta.get('lambda_sep'),
                }
            print(f"done: {label} seed {seed}")

    keys = ('rel_q0', 'q_ratio', 'kc_ratio', 'kn_ratio', 'kc_shrink', 'kn_shrink',
            'cos_q0', 'cos_q1', 'cos_q1_kc', 'cos_q1_kn', 'r_ratio')
    print(f"\n{args.task}: per layer, mean ± std across {len(seeds)} seeds")
    for label, by_sigma in results.items():
        for sigma, per_seed in by_sigma.items():
            n_layers = len(next(iter(per_seed.values()))['layers'])
            for layer in range(n_layers):
                row = f"{label:<10} sigma {float(sigma):.1f} layer {layer + 1}:"
                for key in keys:
                    vals = [v['layers'][layer][key] for v in per_seed.values()]
                    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    row += f"  {key} {statistics.mean(vals):.3f}±{sd:.3f}"
                print(row)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'task': args.task, 'seeds': seeds, 'data_seed': args.data_seed,
                       'batches': args.batches, 'batch_size': args.batch_size, 'results': results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
