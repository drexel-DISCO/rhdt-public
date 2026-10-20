# Evaluation

Evaluates the models trained in [`../training`](../training) under observation noise and
summarizes the scores per training seed. One *job* evaluates one checkpoint (training
seed, config, task) with one evaluation seed: 50 episodes at each noise level. By default
every checkpoint gets 10 evaluation seeds, so each noise level is scored over 500 episodes.

| File | Purpose |
|---|---|
| `eval.sh` | Plans the missing jobs, runs them in parallel, checks them and writes the summaries |
| `eval_rhdt.py` | Evaluates one checkpoint with one evaluation seed |
| `results.py` | Finds the missing jobs (`plan`) and pools the results into summaries (`summarize`) |

The model, environments and rollouts are the ones in `../training` (`model.py`, `data.py`,
`train_rhdt.py`), and the seeds, tasks and configs come from `../training/configs.sh`, so use
the same environment as for training (`../training/environment.yml`).

## Usage

```bash
bash eval.sh -s 48213 -d cuda:0 -j 8              # one training seed, 8 jobs at a time
bash eval.sh -d cuda:0,cuda:1 -j 16               # all paper seeds on two GPUs
bash eval.sh -s 48213 -c half/walker2d-medium-replay-v2   # one more task/config pair
bash eval.sh -s 48213 -n                          # list the pending jobs without running them
```

| Option | Meaning | Default |
|---|---|---|
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated (jobs are spread over them in turn) | `cuda:0` |
| `-j JOBS` | number of jobs at a time | `4` |
| `-t TASKS` | tasks, comma-separated, or `all` | `all` |
| `-c CONFIGS` | configs applied to every task; `tuned` = each task's paper config; `CONFIG/TASK` adds one pair | `tuned,non` |
| `-i N` | Hopfield iterations (`0` = standard attention); evaluates `<models>/iter<N>` with N iterations | `1` |
| `-E EPISODES` | episodes per noise level and evaluation seed | `50` |
| `-C LEVELS` | noise levels | `0.0,0.1,0.2,0.3` |
| `-S SEEDS` | evaluation seeds | `11,13,17,19,23,29,31,37,41,43` |
| `-M DIR` | trained models (the output root of `train.sh`) | `../training/models_rhdt_sweep` |
| `-o DIR` | results root | `results_rhdt` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list the pending jobs only | |

Nothing finished is redone: complete results are skipped, a result missing some noise
levels gets only those rolled out, and summary statistics are only computed where missing.
Each job reseeds itself from its evaluation seed and noise level, so parallel runs, reruns
and partial reruns all give the same numbers. Every job writes a log next to its result;
`eval.sh` lists the jobs that did not finish and exits with 1 if there are any.

## Protocol

Every observation the policy receives, including the first, gets Gaussian noise with
standard deviation equal to the noise level times the dataset's per-dimension standard
deviation. Episode `k` of evaluation seed `s` at noise level `c` uses seed
`s + 100000·c + 1000·k` for both the environment reset and the noise. Scores are D4RL-normalized
(0 = random policy, 100 = expert).

## Outputs

```
results_rhdt/
  iterations_<N>/seed_<seed>/eval_seed_<s>/<config>/
      rhdt_<task>_corruptions.json   # per noise level c: return_<c>, return_std_<c>, score_<c>,
                                     # raw_returns_<c>, raw_scores_<c> (every episode), degradation_<c>
      rhdt_<task>.log
  summary/<task>_<config>_iter<N>.json
      {"<seed>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
  seed_summary/rhdt_<seed>.json
      {"<task>_<config>_iter<N>": {"<c>": {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
```

The summary statistics are computed over the normalized scores of all episodes of all
evaluation seeds of one training seed: the mean, standard deviation, median, interquartile
mean (IQM, the mean of the middle 50%) and CVaR-10 (the mean of the worst 10%).
