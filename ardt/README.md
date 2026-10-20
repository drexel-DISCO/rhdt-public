# ARDT baseline

Our implementation of the
[Adversarially Robust Decision Transformer](https://arxiv.org/abs/2407.18414) (Tang et al.,
NeurIPS 2024), trained on the same six D4RL tasks as RHDT and evaluated with the same
protocol ([`../eval`](../eval)).

ARDT trains a Decision Transformer on returns-to-go relabelled with minimax values:

1. **Warm-up**: two value networks Q(s, a) regress the dataset's returns-to-go.
2. **Minimax estimation**: alternating expectile regression, one network with expectile α
   (the adversary) and the other with expectile 1 − α (the protagonist's response), each
   bootstrapping from the other's value at the next state.
3. **Relabelling**: every trajectory gets the second network's values as returns-to-go.
4. **Decision Transformer** (GPT-2 backbone: 3 layers, 1 head, embedding size 128, context
   length 20) trained on the relabelled trajectories.

| File | Purpose |
|---|---|
| `train_ardt.sh` | Trains the missing models, evaluates them and writes the summaries, in parallel jobs |
| `train_ardt.py` | Trains one model (one task and training seed) |
| `ckpt_selection.py` | Chooses the checkpoint to save by a validation rollout |
| `eval_ardt.py` | Evaluates one checkpoint with one evaluation seed |
| `results.py` | Plans the training and evaluation jobs and pools the results into summaries |

The datasets, environments, seeding, evaluation rollouts and score normalization are the
ones in `../training` (`data.py`, `train_rhdt.py`), and the training seeds come from
`../training/configs.sh`. Use the environment from `../training/environment.yml`, which
includes `transformers` for the GPT-2 backbone.

## Usage

```bash
bash train_ardt.sh -s 48213 -d cuda:0 -j 8           # one training seed, 8 jobs at a time
bash train_ardt.sh -d cuda:0,cuda:1 -j 16            # all paper seeds on two GPUs
bash train_ardt.sh -s 48213 -t hopper-medium-v2      # one task
bash train_ardt.sh -s 48213 -f hopper-medium-v2      # retrain and re-evaluate one task
bash train_ardt.sh -s 48213 -n                       # list what would run
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
| `-M DIR` | checkpoint root | `models_ardt_tuned` |
| `-o DIR` | results root | `results_ardt_tuned` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list what would be trained and evaluated | |

`train_ardt.sh` works in three steps, and each skips what is already done, so an interrupted
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

Each task is trained with its own row of `task_settings` at the top of `train_ardt.sh`; the
values are also stored in each checkpoint.

| Task | α | Minimax steps |
|---|---|---|
| halfcheetah-medium-v2 | 0.1 | 20,000 |
| halfcheetah-medium-replay-v2 | 0.8 | 10,000 |
| hopper-medium-v2 | 0.8 | 10,000 |
| hopper-medium-replay-v2 | 1.0 | 20,000 |
| walker2d-medium-v2 | 0.8 | 10,000 |
| walker2d-medium-replay-v2 | 1.0 | 10,000 |

Every task uses 5,000 warm-up steps and 100,000 Decision Transformer steps. The value
networks have 3 hidden layers of 256 units and are trained with AdamW (learning rate 3e-4,
batch size 256, γ = 0.99). The Decision Transformer is trained with AdamW (learning rate
1e-4, weight decay 1e-4, 10,000 linear warm-up steps, gradient clipping 0.25, batch size 64)
and, at evaluation, is conditioned on the highest relabelled return-to-go in the dataset.

**Checkpoint selection.** The saved model is the best of the last five checkpoints (every
10,000 steps, i.e. at 60k to 100k steps) by the mean normalized score of a 10-episode
validation rollout without noise. Validation uses seed 999, which no evaluation seed
uses, and saves and restores every random number generator, so it does not change
training. Each checkpoint records `selected_step` and `val_scores`. To save the final
weights instead, set `SELECTION_ARGS=(--select-metric none)` in `train_ardt.sh`
(`train_ardt.py -h` lists the other selection options).

## Outputs

```
models_ardt_tuned/seed_<seed>/
    ardt_<task>.pt          # weights and metadata: settings, observation statistics,
    ardt_<task>.json        #   target return, checkpoint selection (the .json has the metadata)
    ardt_<task>_train.log
results_ardt_tuned/
    seed_<seed>/eval_seed_<s>/ardt_<task>_corruptions.json   # same fields as ../eval
    seed_<seed>/eval_seed_<s>/ardt_<task>.log
    summary/<task>.json
        {"<seed>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
    seed_summary/ardt_<seed>.json
        {"<task>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
```

The statistics are computed over the normalized scores of all episodes of all evaluation
seeds of one training seed; see [`../eval/README.md`](../eval/README.md) for the protocol
and the definitions.

The two Python scripts can also be run on their own:

```bash
python train_ardt.py --task hopper-medium-v2 --seed 42 --alpha 0.8 --minimax-steps 10000 \
    --save-path models_ardt_tuned/seed_42/ardt_hopper-medium-v2.pt
python eval_ardt.py --model-path models_ardt_tuned/seed_42/ardt_hopper-medium-v2.pt \
    --output-dir results_ardt_tuned/seed_42/eval_seed_11 --seed 11
```
