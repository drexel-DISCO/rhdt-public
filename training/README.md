# Training

Trains RHDT on six D4RL MuJoCo locomotion tasks. One *run* trains one model for one
(task, seed, regularization config). `train.sh` launches any set of runs, one at a time
or several in parallel, and skips runs that have already finished, so an interrupted
sweep can simply be restarted with the same command.

| File | Purpose |
|---|---|
| `train.sh` | Launches runs over seeds × tasks × configs, sequentially or in parallel |
| `train_rhdt.py` | Trains (and optionally evaluates) a single model |
| `model.py` | RHDT: a Decision Transformer whose attention performs Hopfield retrieval |
| `data.py` | Offline datasets, evaluation environments, D4RL score normalization |
| `configs.sh` | The paper's seeds, tasks and regularization configs (also used by `../eval`) |
| `environment.yml` | Software versions used for the paper |

The trained models are evaluated with [`../eval`](../eval).

## Setup

```bash
conda env create -f environment.yml
conda activate rhdt

# mujoco-py needs the MuJoCo 2.1.0 binaries
mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

The offline datasets are downloaded from the Hugging Face Hub
([`edbeeching/decision_transformer_gym_replay`](https://huggingface.co/datasets/edbeeching/decision_transformer_gym_replay))
on first use.

## Usage

```bash
bash train.sh                                    # the paper's runs: 6 seeds x 6 tasks x 2 configs = 72 runs
bash train.sh -s 48213 -d cuda:0 -j 6            # one seed, 6 runs at a time
bash train.sh -d cuda:0,cuda:1 -j 12             # 12 runs at a time, 6 per GPU
bash train.sh -s 42 -t hopper-medium-v2 -c gponly,seponly   # other configs
bash train.sh -s 42 -c half/walker2d-medium-replay-v2         # a single task/config pair
bash train.sh -n                                 # list the runs without starting them
```

| Option | Meaning | Default |
|---|---|---|
| `-s SEEDS` | training seeds, comma-separated | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated (parallel runs are spread evenly) | `cuda:0` |
| `-j JOBS` | number of runs at a time | `1` |
| `-m MODE` | `train`: train and save checkpoints; `eval`: evaluate saved checkpoints; `all`: both | `train` |
| `-t TASKS` | tasks, comma-separated, or `all` | `all` |
| `-c CONFIGS` | regularization configs (below), applied to every task; `tuned` = each task's paper config; `CONFIG/TASK` adds one pair | `tuned,non` |
| `-i N` | Hopfield iterations per attention layer; `0` = standard attention | `1` |
| `-E EPISODES` | evaluation episodes per noise level | `300` |
| `-C LEVELS` | evaluation noise levels | `0.0,0.1,0.2,0.3` |
| `-o DIR` | output root | `training/models_rhdt_sweep` |
| `-e ENV` | conda environment to activate first | |
| `-f` | redo runs that are already complete | |
| `-n` | dry run: print the runs and commands only | |

Paths given on the command line are relative to where you run `train.sh`; the default
output root is always `training/models_rhdt_sweep`, which is where `../eval` looks.
`-m eval` and `-m all` score each model right after training (300 episodes per noise level,
one seed); the full evaluation protocol with 10 evaluation seeds is in [`../eval`](../eval).

With `-j 1` the output of each run is shown in the terminal. With `-j` > 1 it goes to
`<out>/logs/<date-time>/<config>_seed<seed>_<task>.log`, the terminal shows one line
per run as it starts and finishes, and each run gets an equal share of the CPU cores
(`OMP_NUM_THREADS`, unless you set it yourself). `Ctrl-C` stops all runs; rerun the same
command to continue.
The exit code is 0 if every run succeeded, 1 if any failed, and 3 if `-m eval` found
models that are not trained yet.

## Paper configuration

The loss is the RDT-style weighted MSE on actions plus two regularizers: a gradient
penalty λ_gp · E‖∂â/∂s‖² and key orthogonality λ_sep · mean((K̂K̂ᵀ − I)²) over the
L2-normalized attention keys of every layer.
Each task uses its own (λ_gp, λ_sep), the `tuned` config. Every task is also trained
without either regularizer (`non`).

| Task | Config | λ_gp | λ_sep |
|---|---|---|---|
| halfcheetah-medium-v2 | `gpheavy` | 0.005 | 0.05 |
| halfcheetah-medium-replay-v2 | `gpdoxy` | 0.020 | 0.025 |
| hopper-medium-v2 | `sepheavy` | 0.001 | 0.2 |
| hopper-medium-replay-v2 | `quarter` | 0.0005 | 0.025 |
| walker2d-medium-v2 | `sepheavy` | 0.001 | 0.2 |
| walker2d-medium-replay-v2 | `quarter` | 0.0005 | 0.025 |
| all tasks | `non` | 0 | 0 |

The other named configs (`default`, `half`, `double`, `gponly`, `seponly`) are listed by
`bash train.sh -h`. For ablations, a config name can be changed with a suffix:
`<config>-gp` keeps only its λ_gp (λ_sep = 0), `<config>-sep` only its λ_sep (λ_gp = 0),
`<config>-gpx<M>` and `<config>-sepx<M>` multiply λ_gp or λ_sep by M (e.g. `-gpx0.25`),
`<config>-h<N>` uses N attention heads, and `<config>-b<X>` sets the Hopfield β to X times
the default 1/√(head size), e.g. `-b1.5`. Suffixes combine in this order
(`gpheavy-gp-h1-b2`, `gpheavy-gpx4-b2`). `tuned-gp`, `tuned-sep`, `tuned-gpx4`, `tuned-h1`,
`tuned-b1.5` and so on apply the change to each task's tuned config, e.g.
`bash train.sh -c tuned-gp`. Each derived config trains into its own folder, and a
checkpoint records its β, which evaluation uses. [`../ablation`](../ablation) runs the
paper's ablations with them, and [`../extended_analysis`](../extended_analysis) the sweeps of
β, the penalty weights and the Hopfield iterations.
All other settings are the same for every run:

| Setting | Value |
|---|---|
| Training steps | 100,000 |
| Batch size | 64 sequences of 20 timesteps (60 tokens) |
| Model | embedding 128, 3 layers, 4 heads, 1 Hopfield iteration |
| Dropout | 0.1 (attention, MLP), 0.05 (embeddings) |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4, 10K-step linear warmup, gradient-norm clip 0.25 |
| Returns-to-go | scaled by 1/1000; evaluation is conditioned on the highest return-to-go in the dataset |

## Outputs

```
models_rhdt_sweep/iter<N>/<config>/seed_<seed>/
    config.txt               # the config this folder holds
    rhdt_<task>.pt           # model weights + metadata (settings, observation mean/std, target return)
    rhdt_<task>_losses.json  # loss history: each loss term averaged over every 500 steps
    rhdt_<task>.json         # evaluation results (-m eval / all)
```

A folder whose `config.txt` records a different config is never written to; `train.sh`
stops with an error instead. The results JSON holds, for each noise level `<c>`, the mean
D4RL-normalized score (`score_<c>`), its `std_<c>`, `median_<c>`, `cvar10_<c>` (mean of
the worst 10% of episodes), the mean raw return (`return_<c>`), and every episode's score
in order (`episode_scores_<c>`). Evaluation adds Gaussian noise to every observation, with
standard deviation equal to the noise level times the dataset's per-dimension standard deviation.

The loss history holds, at every 500th training step, the mean over the last 500 steps of
the total loss and of each of its terms (action loss, gradient penalty, key orthogonality);
[`../dynamics`](../dynamics) plots it.

## Training a single model

`train.sh` calls `train_rhdt.py` once per run, for example:

```bash
python train_rhdt.py --task hopper-medium-v2 --seed 42 --device cuda:0 \
    --lambda-gp 0.001 --lambda-sep 0.2 --mode train \
    --output-dir models_rhdt_sweep/iter1/sepheavy/seed_42
```

`python train_rhdt.py -h` lists all options.

## Reproducibility

Runs are deterministic: cuDNN is in deterministic mode, `torch.use_deterministic_algorithms(True)`
is on and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, so retraining with the same seed on the same
hardware and software gives an identical checkpoint. On a GPU, running in parallel gives
the same checkpoints as running one at a time, since every run is a separate process. On a
CPU (`-d cpu`), results also depend on the number of CPU threads, which `-j` reduces.
