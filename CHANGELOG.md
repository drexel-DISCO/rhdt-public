# Changelog

All notable changes to the RHDT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `extended_analysis/`: `extended_analysis.sh` sweeps the Hopfield β, each penalty weight and
  the number of Hopfield iterations on the chosen tasks (by default the ablation tasks). Points
  that exist in the paper's runs or the ablation are read from there; the others are trained
  and evaluated in the folder. `extended_table.py` writes a score table per sweep, and
  `measure_hopfield_gap.py` and `measure_lipschitz.py` measure the query-gap ratio and key
  norms over β and the policy's local Lipschitz constant over λ_GP.
- Derived configs `<config>-gpx<M>` and `<config>-sepx<M>`: λ_gp or λ_sep multiplied by M.
- `dynamics/`: `dynamics.sh` trains the paper's models of the chosen tasks (by default the
  ablation tasks) again into a separate folder, and `dynamics_plot.py` plots every loss term
  over training, averaged over the training seeds, and checks that the retrained weights are
  identical to the paper's models.
- `train_rhdt.py` saves a loss history next to each checkpoint (`rhdt_<task>_losses.json`):
  each loss term averaged over every 500 steps. The training itself is unchanged.
- `benchmark/benchmark.py`: training-step time and peak memory, `get_action` time and peak
  memory, and parameter counts of RHDT, DT, RDT and ARDT on one task, each with its own code
  and the settings of its paper runs.
- `dt/`: the DT baseline as run for the paper. `train_dt.sh` trains every task for 100,000
  steps, evaluates without noise with the protocol of `eval/`, and summarizes, in parallel
  jobs.
- `rdt/`: the RDT baseline as run for the paper. `train_rdt.sh` trains every task with its own
  Gaussian-weighting, reward-loss and embedding-dropout settings, evaluates at the four noise
  levels with the protocol of `eval/`, and summarizes, in parallel jobs.
- `ablation/`: `ablation.sh` trains and evaluates six variants of RHDT (Full RHDT, GP only,
  orthogonality only, no penalties, no Hopfield update, one head) on the chosen tasks, with
  separate outputs, and writes an ablation table per task.
- Derived configs for `train.sh` and `eval.sh`: `<config>-gp`, `<config>-sep`,
  `<config>-h<N>` and `<config>-b<X>` (only the gradient penalty, only key orthogonality,
  N attention heads, the Hopfield β X times 1/√(head size)), combined in this order, and
  `tuned-gp` etc. for each task's tuned config. `train_rhdt.py --beta-scale` sets β;
  checkpoints with another β than the default record it, and `eval_rhdt.py` uses it.
- `-i 0` in `train.sh` and `eval.sh` (`--num-iterations 0` in `train_rhdt.py`): standard
  attention without a Hopfield update, for ablations.

### Changed
- `ardt/run_ardt.sh` is now `ardt/train_ardt.sh`, like the launchers of `dt/` and `rdt/`.

### Removed
- The earlier DT scripts (`dt_train_and_test.py`, `eval_dt_corruptions.py`, `train_dt.sh`,
  `run.sh` and their copies of `benchmark_dataloader.py`, `dt_train.py` and `dt_utils.py`),
  replaced by the new `dt/`.
- The earlier RDT scripts (`rdt_train_and_test.py`, `eval_rdt_corruptions.py`, `train_dt.sh`
  and their copies of `benchmark_dataloader.py`, `dt_train.py` and `dt_utils.py`), replaced
  by the new `rdt/`.
- The unused `mixed` checkpoint-selection metric of `ardt/` (`--select-metric mixed`, validation
  with and without noise); checkpoints are selected by validation without noise.

## [v2.0.0] - 2026-10-04

NeurIPS 2026 camera-ready release.

### Added
- `training/`: training with the camera-ready settings (one Hopfield iteration, per-task
  gradient-penalty and key-orthogonality weights, six training seeds) and `train.sh` to run
  any set of seeds, tasks and configs one at a time or in parallel.
- `eval/`: evaluation under observation noise with 10 evaluation seeds x 50 episodes per
  noise level, resumable jobs, and pooled summaries (mean, median, IQM, CVaR-10).
- `ardt/`: the ARDT baseline as run for the paper: per-task settings, checkpoint selection by a
  validation rollout, and `run_ardt.sh` to train, evaluate with the protocol of `eval/` and
  summarize, in parallel jobs.
- `transformers` 4.44.2 in `training/environment.yml`, for the GPT-2 backbone of the baselines.

### Changed
- License: MIT (the LICENSE file said CC0 while the README said MIT).
- README rewritten for the NeurIPS release.

### Removed
- The root-level ICML scripts (`models.py`, `train_rhdt.py`, `eval_rhdt.py`, `utils.py`,
  `benchmark_dataloader.py`, `train_all.sh`, `requirements.txt`), replaced by `training/`
  and `eval/`.
- The earlier ARDT scripts (`ardt_train_and_test.py`, `eval_ardt_corruptions.py`,
  `train_ardt.sh`, `run.sh` and their copies of `benchmark_dataloader.py`, `dt_train.py` and
  `dt_utils.py`), replaced by the new `ardt/`.

### Fixed
- Evaluation results recorded `lambda_gradient_penalty` as 0.0 for every model; it now holds
  the model's gradient-penalty weight.

## [v1.1.0] - 2026-01-30

### Changed
- **Migrated from Gym to Gymnasium** for modern API compatibility
  - Updated `env.reset()` to handle `(obs, info)` tuple return
  - Updated `env.step()` to handle 5-value return: `(obs, reward, terminated, truncated, info)`
  - Added backward compatibility fallback for old gym API
  - Applies to: `train_rhdt.py`, `eval_rhdt.py`
  
- **Modernized data loading pipeline**
  - Auto-selects best data source: Minari → HuggingFace → D4RL
  - Updated `BenchmarkDataLoader` with intelligent source selection
  - Prioritizes actively maintained datasets over legacy D4RL
  - Better error messages when datasets unavailable
  
- **Enhanced environment compatibility**
  - Full NumPy 2.0 support
  - Compatible with Python 3.8-3.12
  - Removed deprecated gym warnings
  - Modern dependency versions

### Fixed
- Gym deprecation warnings eliminated
- NumPy 2.0 compatibility issues resolved
- More robust dataset loading with fallback options
- Better error handling for missing dependencies

### Documentation
- Updated README.md with Gymnasium information
- Added API compatibility notes in code comments
- Clarified data source priority and auto-selection

## [v1.0.0] - 2026-01-29

### Added
- Initial release for ICML 2026 submission
- RHDT implementation with Hopfield attention
- Multi-seed evaluation framework with pooled statistics
- Pattern separation regularization (key orthogonality)
- Gradient penalty for Lipschitz constraint
- Comprehensive documentation
- Training automation scripts
- RDT baseline support in `rdt/` subdirectory

### Features
- Modular architecture with clean model separation
- Multi-seed evaluation with nested JSON output
- Pooled distribution statistics (mean, median, CVaR, percentiles)
- Task-specific hyperparameter configurations
- Observation corruption testing framework
- LaTeX-ready output formatting

---

## Version Numbering

- **Major version (X.0.0)**: Breaking API changes
- **Minor version (1.X.0)**: New features, backward compatible
- **Patch version (1.0.X)**: Bug fixes, backward compatible
