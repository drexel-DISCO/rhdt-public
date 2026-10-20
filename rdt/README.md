# RDT baseline

Our implementation of the [Robust Decision Transformer](https://arxiv.org/abs/2407.04285)
(Xu et al., ICLR 2025): a GPT-2 Decision Transformer with embedding dropout, a
Gaussian-weighted action loss and a reward-prediction head. It is trained on the same six
D4RL tasks as RHDT and evaluated with the RHDT protocol ([`../eval`](../eval)) at the same
noise levels. It does not include RDT's iterative data correction, which repairs corrupted
actions and rewards in the training data; here the training data is clean and only the
observations at test time are noisy.

| File | Purpose |
|---|---|
| `train_rdt.sh` | Trains the missing models, evaluates them and writes the summaries, in parallel jobs |
| `train_rdt.py` | Trains one model (one task and training seed) |
| `eval_rdt.py` | Evaluates one checkpoint with one evaluation seed |
| `results.py` | Plans the training and evaluation jobs and pools the results into summaries |

The datasets, environments, seeding, evaluation rollouts and score normalization are the
ones in `../training` (`data.py`, `train_rhdt.py`), and the training seeds come from
`../training/configs.sh`. Use the environment from `../training/environment.yml`, which
includes `transformers` for the GPT-2 backbone.

## Usage

```bash
bash train_rdt.sh -s 48213 -d cuda:0 -j 8          # one training seed, 8 jobs at a time
bash train_rdt.sh -d cuda:0,cuda:1 -j 16           # all paper seeds on two GPUs
bash train_rdt.sh -s 48213 -t hopper-medium-v2     # one task
bash train_rdt.sh -s 48213 -f hopper-medium-v2     # retrain and re-evaluate one task
bash train_rdt.sh -s 48213 -n                      # list what would run
```

| Option | Meaning | Default |
|---|---|---|
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated (jobs are spread over them in turn) | `cuda:0` |
| `-j JOBS` | number of jobs at a time | `8` |
| `-t TASKS` | tasks, comma-separated | all six |
| `-f TASKS` | retrain these tasks even if their checkpoint exists, then re-evaluate them | |
| `-E EPISODES` | episodes per noise level and evaluation seed | `50` |
| `-C LEVELS` | noise levels | `0.0,0.1,0.2,0.3` |
| `-S SEEDS` | evaluation seeds | `11,13,17,19,23,29,31,37,41,43` |
| `-M DIR` | checkpoint root | `models_rdt_tuned` |
| `-o DIR` | results root | `results_rdt_tuned` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list what would be trained and evaluated | |

`train_rdt.sh` works in three steps, and each skips what is already done, so an interrupted
run continues where it stopped when you rerun the same command:

1. **Train** one model per (training seed, task). Existing checkpoints are kept, with a
   warning if they were trained with settings other than the current ones.
2. **Evaluate** every checkpoint with every evaluation seed. Complete results are kept, and
   a result missing some noise levels gets only those rolled out.
3. **Summarize** the scores per training seed, as in `../eval`.

`-f TASK` retrains that task (only the forced tasks run unless `-t` is also given). Its old
results and summary entries are deleted first, so they are regenerated from the new
checkpoint; if retraining fails, the old checkpoint is kept and evaluated again.

In a terminal, the progress bar of every running training is shown live; otherwise a status
report is printed every 60 seconds. Every job writes its full output to a log (below). The
script exits with status 1 if a training or an evaluation did not finish.

## Settings

Every task trains for 100,000 steps (`OFFLINE_STEPS`) with its own row of `task_settings`
at the top of `train_rdt.sh`; the values are also stored in each checkpoint:

| Task | β_a (action loss) | β_r (reward loss) | Reward-loss weight | Embedding dropout |
|---|---|---|---|---|
| halfcheetah-medium-v2 | 0.1 | 0 | 0 | 0.05 |
| halfcheetah-medium-replay-v2 | 2.0 | 0 | 0.1 | 0.5 |
| hopper-medium-v2 | 0.1 | 0 | 0.1 | 0.05 |
| hopper-medium-replay-v2 | 1.0 | 0 | 0 | 0.35 |
| walker2d-medium-v2 | 0.3 | 0 | 0.1 | 0.35 |
| walker2d-medium-replay-v2 | 0.3 | 0 | 0.1 | 0.05 |

The loss is the action loss plus the reward-loss weight times the reward loss. With β > 0, a
loss is Gaussian-weighted: the squared error of every timestep is weighted by
exp(−β × that squared error, averaged over the action dimensions), with no gradient through
the weight, and averaged over all entries of the batch (padding counts as zero). With β = 0
it is the plain squared error, summed over the action dimensions and averaged over the
unpadded timesteps. With a reward-loss weight of 0, the reward head does not affect training.

Everything else is fixed in `train_rdt.py`:

| Setting | Value |
|---|---|
| Backbone | GPT-2 (`transformers`): 3 layers, 1 head, embedding 128, MLP 512 with ReLU |
| Dropout | 0.1 (GPT-2's embedding, attention and residual dropout, and before the prediction heads), plus the task's embedding dropout on the token embeddings after their LayerNorm |
| Context | 20 timesteps of (return-to-go, state, action) |
| Heads | Action from the state token (tanh); reward from the action token |
| Rewards, returns-to-go | Divided by 1,000 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4, 10,000 linear warm-up steps, gradient clipping 0.25 |
| Batch size | 64 |
| Saved model | The weights after the last step |
| Conditioning | The highest return-to-go in the dataset |

## Outputs

```
models_rdt_tuned/seed_<seed>/
    rdt_<task>.pt           # weights and metadata: settings, observation statistics,
    rdt_<task>.json         #   target return (the .json has the metadata)
    rdt_<task>_train.log
results_rdt_tuned/
    seed_<seed>/eval_seed_<s>/rdt_<task>_corruptions.json   # same fields as ../eval
    seed_<seed>/eval_seed_<s>/rdt_<task>.log
    summary/<task>.json
        {"<seed>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
    seed_summary/rdt_<seed>.json
        {"<task>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
```

The statistics are computed over the normalized scores of all episodes of all evaluation
seeds of one training seed; see [`../eval/README.md`](../eval/README.md) for the protocol
and the definitions.

The two Python scripts can also be run on their own. `train_rdt.py`'s defaults for the four
settings above (0.1 each) are not the paper's for any task, so pass the task's row:

```bash
python train_rdt.py --task hopper-medium-v2 --seed 42 --wmse-action-coef 0.1 \
    --wmse-reward-coef 0 --reward-loss-coef 0.1 --embedding-dropout 0.05 \
    --save-path models_rdt_tuned/seed_42/rdt_hopper-medium-v2.pt
python eval_rdt.py --model-path models_rdt_tuned/seed_42/rdt_hopper-medium-v2.pt \
    --output-dir results_rdt_tuned/seed_42/eval_seed_11 --seed 11
```
