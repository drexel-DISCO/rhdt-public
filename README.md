# Robust Hopfield Decision Transformer (RHDT)

Official code for **Robust Hopfield Decision Transformer**, NeurIPS 2026.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RHDT is a Decision Transformer whose attention layers perform Hopfield retrieval: each
query is first moved toward the stored patterns (the keys) with the modern Hopfield update,
then attends to the values. Two regularizers make the policy robust to corrupted
observations:

- a **gradient penalty**, λ_gp · E‖∂â/∂s‖², which bounds how strongly the predicted action
  reacts to changes in the observation, and
- **key orthogonality**, λ_sep · mean((K̂K̂ᵀ − I)²) over the L2-normalized attention keys,
  which keeps the stored patterns apart.

The models are trained on six D4RL MuJoCo locomotion datasets and evaluated with Gaussian
noise added to every observation the policy receives.

## Repository structure

```
training/           train RHDT: train.sh, train_rhdt.py, model.py, data.py, configs.sh, environment.yml
eval/               evaluate trained models under observation noise: eval.sh, eval_rhdt.py, results.py
dt/                 Decision Transformer baseline: train_dt.sh, train_dt.py, eval_dt.py
rdt/                Robust Decision Transformer baseline: train_rdt.sh, train_rdt.py, eval_rdt.py
ardt/               Adversarially Robust Decision Transformer baseline: train_ardt.sh, train_ardt.py, eval_ardt.py
ablation/           ablations of RHDT (penalties, Hopfield update, heads): ablation.sh
extended_analysis/  sweeps of β, the penalty weights and the Hopfield iterations, with measurements: extended_analysis.sh
dynamics/           loss curves of the paper's models during training: dynamics.sh, dynamics_plot.py
benchmark/          compute cost of RHDT and the baselines: benchmark.py
```

## Setup

```bash
conda env create -f training/environment.yml
conda activate rhdt
```

`mujoco-py` also needs the MuJoCo 2.1.0 binaries; see [`training/README.md`](training/README.md#setup).
The offline datasets are downloaded from the Hugging Face Hub on first use.

## Reproducing the paper

```bash
# 1. Train: 6 seeds x 6 tasks x {each task's tuned regularization, no regularization} = 72 models
cd training
bash train.sh -d cuda:0 -j 6

# 2. Evaluate: per model, 10 evaluation seeds x 50 episodes at noise levels 0.0, 0.1, 0.2, 0.3
cd ../eval
bash eval.sh -d cuda:0 -j 8
```

Both steps skip work that is already done, so an interrupted run continues where it
stopped when you rerun the same command. The pooled scores (mean, median, IQM, CVaR-10)
of every training seed end up in `eval/results_rhdt/summary/` and
`eval/results_rhdt/seed_summary/`. See [`training/README.md`](training/README.md) and
[`eval/README.md`](eval/README.md) for all options, the hyperparameters of each task, and
the output formats.

## Ablations

[`ablation/ablation.sh`](ablation) trains and evaluates six variants for the tasks you
choose: Full RHDT, gradient penalty only, key orthogonality only, no penalties, no
Hopfield update, and one attention head. Each variant gets all six training seeds and
the full evaluation protocol, and the script writes a table per task:

```bash
cd ablation
bash ablation.sh -t halfcheetah-medium-v2 -d cuda:0 -j 6
```

See [`ablation/README.md`](ablation/README.md) for the variants and options.

## Extended analysis

[`extended_analysis/extended_analysis.sh`](extended_analysis) sweeps the Hopfield β (0.5× to
5×), each penalty weight (0× to 4× its tuned value, the other one fixed) and the number of
Hopfield iterations (0 to 3) on the ablation tasks, with all six training seeds and the full
evaluation protocol. It also measures the query-gap ratio and the key norms over β, and the
policy's local Lipschitz constant ‖J‖₂ over λ_GP. Points that already exist in the paper's
runs or the ablation are read from there; the others are trained in the folder itself:

```bash
cd extended_analysis
bash extended_analysis.sh -d cuda:0 -j 6
```

See [`extended_analysis/README.md`](extended_analysis/README.md) for the sweeps, options and outputs.

## Training dynamics

Training saves each loss term averaged over every 500 steps next to the checkpoint.
[`dynamics/dynamics.sh`](dynamics) trains the paper's models of the ablation tasks again (six
seeds each) into a separate folder and plots these curves. Training is deterministic, so
the retrained models are the paper's models; the script checks this by comparing the
weights with those in `training/models_rhdt_sweep`:

```bash
cd dynamics
bash dynamics.sh -d cuda:0 -j 6
```

See [`dynamics/README.md`](dynamics/README.md) for the options and the output format.

## Compute cost

[`benchmark/benchmark.py`](benchmark/benchmark.py) measures, for RHDT, DT, RDT and ARDT on one
task, the time and peak GPU memory of a full training step (1,000 timed steps after 100
warm-up steps, batch 64, context 20), the time of every `get_action` call in a 1,000-step
episode with its peak memory, and the parameter counts. Each method uses its own code and
the settings of its paper runs. Run it on an otherwise idle GPU:

```bash
python benchmark/benchmark.py --device cuda:0    # halfcheetah-medium-v2, all four methods
```

## Baselines

`dt/`, `rdt/` and `ardt/` contain our implementations of the
[Decision Transformer](https://arxiv.org/abs/2106.01345) (Chen et al., NeurIPS 2021), the
[Robust Decision Transformer](https://arxiv.org/abs/2407.04285) (Xu et al., ICLR 2025) and the
[Adversarially Robust Decision Transformer](https://arxiv.org/abs/2407.18414) (Tang et al.,
NeurIPS 2024). They use the same environment; `training/environment.yml` includes
`transformers` for their GPT-2 backbone.

The baselines run the same way as RHDT: each task is trained with its own settings, then
evaluated with the protocol of `eval/` (DT without noise only) into the same kind of
summaries (`<baseline>/results_<baseline>_tuned/summary/` and `.../seed_summary/`):

```bash
bash dt/train_dt.sh -d cuda:0 -j 8
bash rdt/train_rdt.sh -d cuda:0 -j 8
bash ardt/train_ardt.sh -d cuda:0 -j 8
```

See [`dt/README.md`](dt/README.md), [`rdt/README.md`](rdt/README.md) and
[`ardt/README.md`](ardt/README.md) for the settings and all options.

## Citation

If you use this code, please cite our NeurIPS 2026 paper (BibTeX to be added).

## License

MIT; see [LICENSE](LICENSE).

## Acknowledgments

Built on [Decision Transformer](https://github.com/kzl/decision-transformer) and the
[D4RL](https://github.com/Farama-Foundation/D4RL) benchmark; the attention follows
[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217).
