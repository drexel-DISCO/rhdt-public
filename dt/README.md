# DT baseline

Our implementation of the [Decision Transformer](https://arxiv.org/abs/2106.01345)
(Chen et al., NeurIPS 2021) with no robustness method, trained on the same six D4RL tasks
as RHDT and evaluated with the RHDT protocol ([`../eval`](../eval)) without noise.

| File | Purpose |
|---|---|
| `train_dt.sh` | Trains the missing models, evaluates them and writes the summaries, in parallel jobs |
| `train_dt.py` | Trains one model (one task and training seed) |
| `eval_dt.py` | Evaluates one checkpoint with one evaluation seed |
| `results.py` | Plans the training and evaluation jobs and pools the results into summaries |

The datasets, environments, seeding, evaluation rollouts and score normalization are the
ones in `../training` (`data.py`, `train_rhdt.py`), and the training seeds come from
`../training/configs.sh`. Use the environment from `../training/environment.yml`, which
includes `transformers` for the GPT-2 backbone.

## Usage

```bash
bash train_dt.sh -s 48213 -d cuda:0 -j 8           # one training seed, 8 jobs at a time
bash train_dt.sh -d cuda:0,cuda:1 -j 16            # all paper seeds on two GPUs
bash train_dt.sh -s 48213 -t hopper-medium-v2      # one task
bash train_dt.sh -s 48213 -f hopper-medium-v2      # retrain and re-evaluate one task
bash train_dt.sh -s 48213 -n                       # list what would run
```

| Option | Meaning | Default |
|---|---|---|
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated (jobs are spread over them in turn) | `cuda:0` |
| `-j JOBS` | number of jobs at a time | `8` |
| `-t TASKS` | tasks, comma-separated | all six |
| `-f TASKS` | retrain these tasks even if their checkpoint exists, then re-evaluate them | |
| `-E EPISODES` | episodes per noise level and evaluation seed | `50` |
| `-C LEVELS` | noise levels | `0.0` (no noise) |
| `-S SEEDS` | evaluation seeds | `11,13,17,19,23,29,31,37,41,43` |
| `-M DIR` | checkpoint root | `models_dt_tuned` |
| `-o DIR` | results root | `results_dt_tuned` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list what would be trained and evaluated | |

`train_dt.sh` works in three steps, and each skips what is already done, so an interrupted
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

Every task trains for 100,000 steps (`task_settings` at the top of `train_dt.sh`; the value is
also stored in each checkpoint). Everything else is fixed in `train_dt.py`:

| Setting | Value |
|---|---|
| Backbone | GPT-2 (`transformers`): 3 layers, 1 head, embedding 128, MLP 512 with ReLU |
| Dropout | 0.1 (embedding, attention, residual) |
| Context | 20 timesteps of (return-to-go, state, action) |
| Loss | Squared action error, averaged over unpadded timesteps |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4, 10,000 linear warm-up steps, gradient clipping 0.25 |
| Batch size | 64 |
| Saved model | The weights after the last step |
| Conditioning | The highest return-to-go in the dataset |

## Outputs

```
models_dt_tuned/seed_<seed>/
    dt_<task>.pt            # weights and metadata: settings, observation statistics,
    dt_<task>.json          #   target return (the .json has the metadata)
    dt_<task>_train.log
results_dt_tuned/
    seed_<seed>/eval_seed_<s>/dt_<task>_corruptions.json   # same fields as ../eval
    seed_<seed>/eval_seed_<s>/dt_<task>.log
    summary/<task>.json
        {"<seed>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
    seed_summary/dt_<seed>.json
        {"<task>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
```

The statistics are computed over the normalized scores of all episodes of all evaluation
seeds of one training seed; see [`../eval/README.md`](../eval/README.md) for the protocol
and the definitions.

The two Python scripts can also be run on their own:

```bash
python train_dt.py --task hopper-medium-v2 --seed 42 \
    --save-path models_dt_tuned/seed_42/dt_hopper-medium-v2.pt
python eval_dt.py --model-path models_dt_tuned/seed_42/dt_hopper-medium-v2.pt \
    --output-dir results_dt_tuned/seed_42/eval_seed_11 --seed 11
```
