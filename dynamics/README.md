# Training dynamics

Curves of every loss term during the training of the paper's models.
[`../training/train_rhdt.py`](../training) averages each loss term over every 500 training
steps and saves this loss history next to the checkpoint (`rhdt_<task>_losses.json`).
Recording it does not change the training: the losses are pulled to the CPU every step for
the progress bar anyway.

The paper's models were trained before the history was recorded, so `dynamics.sh` trains
them again, into this folder. Training is deterministic, so on the same hardware and
software the retrained models are the paper's models, and the curves describe exactly the
models evaluated in the paper. `dynamics_plot.py` checks this: it compares the weights of
every retrained model with the paper's model.

| File | Purpose |
|---|---|
| `dynamics.sh` | Retrains the paper's models of the chosen tasks, then plots the curves |
| `dynamics_plot.py` | Averages the loss histories over the training seeds, plots them, and compares the weights with the paper's models |

`dynamics.sh` runs [`../training/train.sh`](../training) with the paper's settings (each
task's tuned config, 1 Hopfield iteration, 100,000 steps, all six training seeds) and this
folder's own output directory, so the paper's models in `../training/models_rhdt_sweep` are
never touched. Use the environment from `../training/environment.yml`; the figure also needs
matplotlib (`pip install matplotlib`).

## Usage

```bash
bash dynamics.sh -d cuda:0 -j 6                # halfcheetah-medium-v2 and hopper-medium-replay-v2: 12 runs
bash dynamics.sh -d cuda:0,cuda:1 -j 12        # 12 runs at a time on two GPUs
bash dynamics.sh -t all -d cuda:0 -j 6         # all six tasks: 36 runs
bash dynamics.sh -m plot                       # plot the runs finished so far
bash dynamics.sh -n                            # list the runs
```

| Option | Meaning | Default |
|---|---|---|
| `-t TASKS` | tasks, comma-separated, or `all` | `halfcheetah-medium-v2,hopper-medium-replay-v2` |
| `-c CONFIGS` | configs, as in `../training/train.sh`; `tuned` = each task's paper config | `tuned` |
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated | `cuda:0` |
| `-j JOBS` | training runs at a time | `6` |
| `-m MODE` | `train`, `plot` (only plot the runs finished so far) or `all` | `all` |
| `-M DIR` | retrained models and their loss histories | `dynamics/models_dynamics` |
| `-o DIR` | curves and figure | `dynamics/results_dynamics` |
| `-P DIR` | the paper's models, to compare the weights with | `training/models_rhdt_sweep` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list the training runs only | |

The default tasks are those of the ablations ([`../ablation`](../ablation)). Finished runs
are skipped, so after `Ctrl-C` or a crash, rerun the same command to continue. The exit
code is 1 if a training run failed (see its log) and 130 after an interruption. Do not copy
the paper's checkpoints into `models_dynamics`: they have no loss history, and a run whose
checkpoint exists is not trained again (`dynamics_plot.py` reports such runs).

Every model that `../training/train.sh` trains from now on gets its loss history right
away, without a rerun.

## Outputs

```
models_dynamics/iter1/<config>/seed_<seed>/
    rhdt_<task>.pt                    the retrained model, as in ../training
    rhdt_<task>_losses.json           its loss history
    config.txt
models_dynamics/logs/                 training logs (with -j > 1)
results_dynamics/
    dynamics_<task>_<config>.json     the curves of every seed, their mean and standard
                                      deviation over the seeds, and the weight check
    dynamics.pdf, dynamics.png        the figure
```

A loss history holds the run's settings and the curves:

```
{"task": "halfcheetah-medium-v2", "seed": 42, "method": "k_ortho", "lambda_gp": 0.005, "lambda_sep": 0.05,
 ..., "steps": 100000, "interval": 500,
 "history": {"step": [500, 1000, ..., 100000], "loss": [...], "action_loss": [...], "gp_loss": [...], "sep_loss": [...]}}
```

Each entry averages the 500 steps up to and including `step`. `loss` is the training
objective, `action_loss + lambda_gp * gp_loss + lambda_sep * sep_loss`: `action_loss` is the
weighted MSE on actions, `gp_loss` the gradient penalty E‖∂â/∂s‖², and `sep_loss` the key
orthogonality mean((K̂K̂ᵀ − I)²), the last two before weighting. A term with weight 0 is not
computed and is recorded as 0.

The figure has one row per task and config and one column per loss term: the mean over the
training seeds with ± one standard deviation shaded, on a log scale.

For every seed, `dynamics_plot.py` prints whether the retrained weights are identical to
those of the paper's model in `-P`. If they differ (e.g. another GPU, driver or PyTorch
version than the paper's runs), the curves come from models trained the same way, but not
exactly from the paper's models.
