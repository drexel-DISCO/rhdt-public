# Ablations

Trains and evaluates six variants of RHDT on the tasks you choose, with the paper's
protocol, and writes an ablation table. Every variant is trained with all six training
seeds, and every model is evaluated with the 10 evaluation seeds (50 episodes each per
noise level), as in [`../eval`](../eval).

| Variant | `-v` name | Change from the paper's model | Config |
|---|---|---|---|
| Full RHDT | `full` | none: the task's tuned λ_gp and λ_sep | `tuned` |
| GP only | `gp_only` | λ_sep = 0 (the tuned λ_gp is kept) | `tuned-gp` |
| Orthogonality only | `sep_only` | λ_gp = 0 (the tuned λ_sep is kept) | `tuned-sep` |
| No penalties | `non` | λ_gp = λ_sep = 0 | `non` |
| No Hopfield update | `no_hopfield` | 0 Hopfield iterations: standard attention | `tuned`, `-i 0` |
| One head | `one_head` | 1 attention head instead of 4 | `tuned-h1` |

"tuned" is the task's paper config (for halfcheetah-medium-v2, `gpheavy`: λ_gp = 0.005,
λ_sep = 0.05; see [`../training/README.md`](../training/README.md#paper-configuration)).
Everything else is the paper's setup: 1 Hopfield iteration, 4 heads, 100,000 steps.

| File | Purpose |
|---|---|
| `ablation.sh` | Trains the variants, evaluates them and writes the table |
| `ablation_table.py` | Builds the table from the evaluation summaries |

`ablation.sh` runs [`../training/train.sh`](../training) and [`../eval/eval.sh`](../eval)
with this folder's own output directories, so the paper's models and results are never
touched. Use the environment from `../training/environment.yml`.

## Usage

```bash
bash ablation.sh -t halfcheetah-medium-v2 -d cuda:0 -j 6
bash ablation.sh -t hopper-medium-v2,walker2d-medium-v2 -d cuda:0,cuda:1 -j 12
bash ablation.sh -t halfcheetah-medium-v2 -v no_hopfield,one_head   # only some variants
bash ablation.sh -t halfcheetah-medium-v2 -m train -j 6              # train now ...
bash ablation.sh -t halfcheetah-medium-v2 -m eval -j 12              # ... evaluate later
bash ablation.sh -t halfcheetah-medium-v2 -n                          # list the runs and jobs
```

| Option | Meaning | Default |
|---|---|---|
| `-t TASKS` | tasks, comma-separated, or `all` | required |
| `-v VARIANTS` | variants, comma-separated (names above) | all six |
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated | `cuda:0` |
| `-j JOBS` | training runs, and then evaluation jobs, at a time | `6` |
| `-m MODE` | `train`, `eval` (evaluate and write the table) or `all` | `all` |
| `-E EPISODES` | episodes per noise level and evaluation seed | `50` |
| `-C LEVELS` | noise levels | `0.0,0.1,0.2,0.3` |
| `-S SEEDS` | evaluation seeds | `11,13,17,19,23,29,31,37,41,43` |
| `-M DIR` | trained models | `ablation/models_ablation` |
| `-o DIR` | results and tables | `ablation/results_ablation` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list the training runs and evaluation jobs only | |

For one task this is 36 training runs (6 variants x 6 seeds) and 360 evaluation jobs.
Training comes first (the five variants with one Hopfield iteration together, then the
variant without), then evaluation, then the table. Finished training runs and evaluation
jobs are skipped, so after `Ctrl-C` or a crash, rerun the same command to continue. The
exit code is 1 if a training run or an evaluation job did not finish (see the output for
their logs) and 130 after an interruption.

Full RHDT and No penalties have the same settings as the paper's runs in
`../training/models_rhdt_sweep/iter1/`, so they give the same models. To reuse those
instead of retraining them, copy their config folders before running, e.g. for
halfcheetah-medium-v2:

```bash
mkdir -p models_ablation/iter1
cp -r ../training/models_rhdt_sweep/iter1/gpheavy ../training/models_rhdt_sweep/iter1/non models_ablation/iter1/
```

## Outputs

```
models_ablation/iter<N>/<config>/seed_<seed>/rhdt_<task>.pt     (and config.txt, as in ../training)
results_ablation/
    iterations_<N>/...                       per-run results, as in ../eval
    summary/<task>_<config>_iter<N>.json     per training seed: mean, std, median, IQM, CVaR-10
    ablation_<task>.json                     the table's numbers, every variant's settings,
                                             and the statistics of every training seed
    ablation_<task>.md                       the tables: mean, median, IQM and CVaR-10
```

In the table, each training seed's score pools all episodes of all its evaluation seeds
(500 per noise level); each cell is the mean ± standard deviation of that score over the
training seeds. A cell marked `(k/6)` includes only the k training seeds whose
models are trained and fully evaluated.

The configs can also be used directly, e.g. `bash ../training/train.sh -c tuned-gp` trains
the GP-only variant of every task; `bash ../training/train.sh -h` lists them.
