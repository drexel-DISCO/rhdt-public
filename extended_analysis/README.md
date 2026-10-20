# Extended analysis

Sweeps around the paper's model on the ablation tasks (by default halfcheetah-medium-v2 and
hopper-medium-replay-v2), plus two measurements of the trained models. Every point of a
sweep is trained with all six training seeds and evaluated with the paper's protocol
(10 evaluation seeds × 50 episodes at noise levels 0.0, 0.1, 0.2 and 0.3), as in
[`../eval`](../eval).

| Sweep | Values (default) | Change from the paper's model | Config |
|---|---|---|---|
| `beta` | 0.5, 1, 1.5, 2, 5 | Hopfield β, × 1/√(head size) | `tuned-b<X>` |
| `gp` | 0, 0.25, 1, 4 | λ_GP × its tuned value, λ_ortho tuned | `tuned-gpx<M>` (0: `tuned-sep`) |
| `ortho` | 0, 0.25, 1, 4 | λ_ortho × its tuned value, λ_GP tuned | `tuned-sepx<M>` (0: `tuned-gp`) |
| `nhop` | 0, 1, 2, 3 | Hopfield iterations | `tuned`, `-i N` |

"tuned" is the task's paper config (for halfcheetah-medium-v2, `gpheavy`: λ_GP = 0.005,
λ_ortho = 0.05; see [`../training/README.md`](../training/README.md#paper-configuration)).
The value 1 is the paper's model in every sweep, and λ = 0 is the ablation's GP-only or
orthogonality-only model, so each penalty curve has four points over a 16× range. For
smoother curves add 0.5 and 2 (`-X 0,0.25,0.5,1,2,4`).

| Measurement | Over | Script |
|---|---|---|
| Query-gap ratio (does the Hopfield update pull corrupted queries toward clean ones?) and key norms ‖K‖ | β sweep | `measure_hopfield_gap.py` |
| Local Lipschitz constant ‖J‖₂ of the policy | λ_GP sweep | `measure_lipschitz.py` |

| File | Purpose |
|---|---|
| `extended_analysis.sh` | Trains and evaluates the points that do not exist yet, writes the tables, runs the measurements |
| `extended_table.py` | Builds the score tables from the evaluation summaries |
| `measure_hopfield_gap.py` | Clean-vs-corrupted gaps of the queries and readouts, before and after the Hopfield update, and key and query norms |
| `measure_lipschitz.py` | ‖J‖₂, ‖J‖_F² and the gradient-penalty term at clean and corrupted states |

## Points that already exist

A point whose models exist for every training seed in the paper's runs
(`../training/models_rhdt_sweep`, results in `../eval/results_rhdt`) or in the ablation
(`../ablation/models_ablation`, results in `../ablation/results_ablation`) is read from
there, not trained again. With the defaults these are β 1 and 1.5, λ ×1 and 1 and 2 Hopfield
iterations from the paper's runs, and λ_GP ×0, λ_ortho ×0 and 0 Hopfield iterations from the
ablation. Every other point is trained and evaluated in this folder, and nothing outside it
is written. If a point there is trained but not evaluated, the table marks it and prints the
`eval.sh` command that evaluates it where it is.

With all of these in place, each task needs 8 new points (β 0.5, 2 and 5; λ_GP ×0.25 and ×4;
λ_ortho ×0.25 and ×4; 3 Hopfield iterations), i.e. 48 training runs and 480 evaluation jobs;
96 and 960 for both tasks. `-X 0,0.25,0.5,1,2,4` adds 24 runs per task. Run
`bash extended_analysis.sh -n` first: it lists every point, where it is read from or that
it is trained here, and the runs and jobs that would start.

## Usage

```bash
bash extended_analysis.sh -d cuda:0 -j 6
bash extended_analysis.sh -d cuda:0,cuda:1 -j 12 -X 0,0.25,0.5,1,2,4   # smoother penalty curves
bash extended_analysis.sh -v beta,nhop -d cuda:0 -j 6                   # some sweeps
bash extended_analysis.sh -m train -d cuda:0 -j 6                       # train now ...
bash extended_analysis.sh -m eval -d cuda:0 -j 12                       # ... evaluate later
bash extended_analysis.sh -m measure -d cuda:0                          # only the measurements
bash extended_analysis.sh -n                                            # list the points, runs and jobs
```

| Option | Meaning | Default |
|---|---|---|
| `-t TASKS` | tasks, comma-separated, or `all` | `halfcheetah-medium-v2,hopper-medium-replay-v2` |
| `-v SWEEPS` | sweeps: `beta`, `gp`, `ortho`, `nhop` | all four |
| `-B VALUES` | β sweep, multiples of 1/√(head size) | `0.5,1,1.5,2,5` |
| `-X VALUES` | λ_GP and λ_ortho sweeps, multiples of the tuned weight | `0,0.25,1,4` |
| `-H VALUES` | Hopfield iterations | `0,1,2,3` |
| `-s SEEDS` | training seeds | `907561,48213,2024887,739104,26451,42` |
| `-d DEVICES` | device, or several comma-separated; the measurements use the first | `cuda:0` |
| `-j JOBS` | training runs, and then evaluation jobs, at a time | `6` |
| `-m MODE` | `train`, `eval` (evaluate and write the tables), `measure` or `all` | `all` |
| `-E EPISODES` | episodes per noise level and evaluation seed | `50` |
| `-C LEVELS` | noise levels | `0.0,0.1,0.2,0.3` |
| `-S SEEDS` | evaluation seeds | `11,13,17,19,23,29,31,37,41,43` |
| `-M DIR` | models trained here | `extended_analysis/models_extended` |
| `-o DIR` | results, tables and measurements | `extended_analysis/results_extended` |
| `-P DIR,DIR` | the paper's models and results | `training/models_rhdt_sweep,eval/results_rhdt` |
| `-A DIR,DIR` | the ablation's models and results | `ablation/models_ablation,ablation/results_ablation` |
| `-e ENV` | conda environment to activate first | |
| `-n` | dry run: list the points, training runs and evaluation jobs only | |

Training comes first (points with the same number of Hopfield iterations together), then
evaluation, the tables and the measurements. Finished training runs and evaluation jobs are
skipped, so after `Ctrl-C` or a crash, rerun the same command to continue. The exit code is
1 if a step did not finish (see its output) and 130 after an interruption.

## Outputs

```
models_extended/iter<N>/<config>/seed_<seed>/rhdt_<task>.pt     (and config.txt, as in ../training)
results_extended/
    iterations_<N>/..., summary/...           per-run results and summaries, as in ../eval
    tables/<sweep>_<task>.json                every point's settings, source, the numbers of every
                                              training seed and their averages
    tables/<sweep>_<task>.md                  tables of the mean, median, IQM and CVaR-10
    measure/hopfield_gap_<task>.json, .log    query gaps and norms, per β (labels b0.5, b1, ...)
    measure/lipschitz_<task>.json, .log       ‖J‖₂ etc., per λ_GP multiple (labels gp0, gp0.25, ...)
```

In a table, each training seed's score pools all episodes of all its evaluation seeds; each
cell is the mean ± standard deviation of that score over the training seeds, and a cell
marked `(k/6)` includes only the k training seeds that are trained and fully evaluated.
The source column says where the point was read from: `main` (the paper's runs),
`ablation`, or `extended` (this folder).

Both measurements use the same 8 batches of 64 training segments (data seed 0) for every
model, and the same observation noise, σ times the dataset's standard deviation as in the
evaluation:

- `measure_hopfield_gap.py` runs every segment clean and corrupted (σ = 0.1, 0.2, 0.3) and
  reports, per attention layer, the relative clean-vs-corrupted gap of the state queries
  before and after the Hopfield update and their ratio (`q_ratio`; `kc_ratio` and `kn_ratio`
  update both queries against the same clean or corrupted keys), the same for the readouts
  (`r_ratio`), and, from the clean pass, the mean key norm `k_norm` over all valid tokens,
  the largest key norm per segment and head `k_max`, and the query norms `q0_norm` and
  `q1_norm` before and after the update.
- `measure_lipschitz.py` computes, at every valid state of every segment, the Jacobian of all
  predicted actions with respect to that state and reports its spectral norm ‖J‖₂ (the local
  Lipschitz constant), ‖J‖_F² and ‖Jᵀ1‖² (the training gradient penalty), at σ = 0 and 0.3.

Each script first runs a check (the recorded forward pass equals the model's; the autograd
Jacobian equals finite differences) and stops if it fails. Both print a table of the mean ±
standard deviation over the training seeds, and can be run on any checkpoints, e.g.

```bash
python measure_lipschitz.py --task hopper-medium-replay-v2 --seeds 42,48213 --device cuda:0 \
    --model gp1=../training/models_rhdt_sweep/iter1/quarter/seed_{seed}/rhdt_hopper-medium-replay-v2.pt \
    --model gp4=models_extended/iter1/quarter-gpx4/seed_{seed}/rhdt_hopper-medium-replay-v2.pt
```

The configs can also be used directly, e.g. `bash ../training/train.sh -c tuned-gpx4` trains
λ_GP × 4 for every task; `bash ../training/train.sh -h` lists the config suffixes.
