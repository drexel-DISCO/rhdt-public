#!/usr/bin/env bash
# =============================================================================
# Training dynamics of RHDT. For every task given with -t: retrain the paper's model
# with every training seed into this folder, with each loss term averaged over every
# 500 training steps, and plot the curves.
#
#   bash dynamics.sh -d cuda:0 -j 6                  # the ablation tasks (below)
#   bash dynamics.sh -t all -d cuda:0,cuda:1 -j 12   # all six tasks
#   bash dynamics.sh -m plot                         # plot the runs finished so far
#   bash dynamics.sh -n                              # list the runs, start nothing
#
# The default tasks are those of the ablations, halfcheetah-medium-v2 and
# hopper-medium-replay-v2: 12 runs. Training is ../training/train.sh with the paper's
# settings (each task's tuned config, 1 Hopfield iteration, 100k steps). It is
# deterministic, so on the same hardware and software the retrained models are the
# paper's models and the curves are theirs; dynamics_plot.py checks this against
# the paper's models in ../training/models_rhdt_sweep, which are never written to.
#   models_dynamics/iter1/<config>/seed_<seed>/rhdt_<task>.pt and rhdt_<task>_losses.json
#   results_dynamics/dynamics_<task>_<config>.json   the curves of every seed, mean and std
#   results_dynamics/dynamics.pdf and .png           the figure
# Finished training runs are skipped, so rerun the same command to continue after an
# interruption.
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "dynamics.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SH=$SCRIPT_DIR/../training/train.sh
# shellcheck source=SCRIPTDIR/../training/configs.sh
source "$SCRIPT_DIR/../training/configs.sh"
PYTHON=${PYTHON:-python}

ITERATIONS=1  # Hopfield iterations of the paper's models

# ---- Command line --------------------------------------------------------------
TASKS=halfcheetah-medium-v2,hopper-medium-replay-v2
CONFIGS=tuned
SEEDS=$PAPER_SEEDS
DEVICES=cuda:0
JOBS=6
MODE=all
MODELS_ROOT=$SCRIPT_DIR/models_dynamics
RESULTS_ROOT=$SCRIPT_DIR/results_dynamics
PAPER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)/training/models_rhdt_sweep
CONDA_ENV=
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash dynamics.sh [options]

  -t TASKS       tasks, comma-separated, or all     (default: $TASKS)
  -c CONFIGS     configs, as in ../training/train.sh; tuned = each task's paper
                 config                              (default: $CONFIGS)
  -s SEEDS       training seeds, comma-separated    (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated (default: $DEVICES)
  -j JOBS        training runs at a time            (default: $JOBS)
  -m MODE        train: only train; plot: only plot the runs finished so far;
                 all: both                          (default: $MODE)
  -M DIR         retrained models and loss histories (default: dynamics/models_dynamics)
  -o DIR         curves and figure                  (default: dynamics/results_dynamics)
  -P DIR         the paper's models, to check the retrained ones against
                                                    (default: training/models_rhdt_sweep)
  -e ENV         conda environment to activate first
  -n             dry run: list the training runs, start nothing
  -h             show this help
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

same_dir() {  # <dir> <dir>: true if both exist and are the same directory
    local a b
    a=$(cd "$1" 2> /dev/null && pwd -P) && b=$(cd "$2" 2> /dev/null && pwd -P) && [[ $a == "$b" ]]
}

while getopts ":t:c:s:d:j:m:M:o:P:e:nh" opt; do
    case $opt in
        t) TASKS=$OPTARG ;;
        c) CONFIGS=$OPTARG ;;
        s) SEEDS=$OPTARG ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        m) MODE=$OPTARG ;;
        M) MODELS_ROOT=$(abs_path "$OPTARG") ;;
        o) RESULTS_ROOT=$(abs_path "$OPTARG") ;;
        P) PAPER_ROOT=$(abs_path "$OPTARG") ;;
        e) CONDA_ENV=$OPTARG ;;
        n) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        :) die "option -$OPTARG needs a value (see -h)" ;;
        *) die "unknown option -$OPTARG (see -h)" ;;
    esac
done
shift $((OPTIND - 1))
(( $# == 0 )) || die "unexpected argument '$1'; lists are comma-separated, e.g. -s 42,48213"

expand_pairs "$TASKS" "$CONFIGS" || exit 1
parse_seeds "$SEEDS" || exit 1
parse_list "$DEVICES" DEVICE_LIST
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
case $MODE in train|plot|all) ;; *) die "-m must be train, plot or all, got '$MODE'" ;; esac
same_dir "$MODELS_ROOT" "$PAPER_ROOT" && die "-M is the folder of the paper's models; retrain into another folder"
TASKS=$(IFS=,; echo "${TASK_LIST[*]}")
CONFIGS=$(IFS=,; echo "${CONFIG_LIST[*]}")
SEEDS=$(IFS=,; echo "${SEED_LIST[*]}")
DEVICES=$(IFS=,; echo "${DEVICE_LIST[*]}")

if [[ -n $CONDA_ENV ]]; then
    command -v conda > /dev/null 2>&1 || die "conda not found, but -e $CONDA_ENV was given"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || die "could not load conda's shell functions"
    conda activate "$CONDA_ENV" || die "could not activate conda environment '$CONDA_ENV'"
fi
command -v "$PYTHON" > /dev/null 2>&1 || die "'$PYTHON' not found; activate the environment or use -e ENV"

echo "RHDT training dynamics: ${#SEED_LIST[@]} training seed(s) x ${#PAIRS[@]} task/config pair(s) | devices ${DEVICE_LIST[*]} | jobs $JOBS"
for pair in "${PAIRS[@]}"; do
    cfg=${pair%%/*}
    read -r gp sep <<< "$(config_lambdas "$cfg")"
    printf '  %-30s %-10s lambda_gp=%-7s lambda_sep=%s\n' "${pair#*/}" "$cfg" "$gp" "$sep"
done
echo "  models and loss histories: $MODELS_ROOT"
echo "  curves and figure: $RESULTS_ROOT"
echo "  paper's models, for the check: $PAPER_ROOT"

FAILED=()
CHILD=

# Ctrl-C or kill: stop the running step (it stops its own jobs), then stop here.
# The steps run in the background so that a signal reaches this script at once.
# shellcheck disable=SC2329  # invoked by the trap below
on_signal() {
    if [[ -n $CHILD ]]; then
        kill -TERM "$CHILD" 2> /dev/null
        wait "$CHILD" 2> /dev/null
    fi
    echo "Interrupted. Rerun the same command to continue." >&2
    exit 130
}
trap on_signal INT TERM

run_step() {  # <description> <command...>
    local desc=$1 rc
    shift
    echo
    echo "==== $desc"
    "$@" &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=
    if (( rc != 0 )); then
        FAILED+=("$desc (exit code $rc)")
    fi
}

# ---- 1. Train ------------------------------------------------------------------
if [[ $MODE != plot ]]; then
    DRY=()
    if (( DRY_RUN )); then
        DRY=(-n)
    fi
    run_step "Training" bash "$TRAIN_SH" -m train -s "$SEEDS" -t "$TASKS" -c "$CONFIGS" -i "$ITERATIONS" \
        -d "$DEVICES" -j "$JOBS" -o "$MODELS_ROOT" "${DRY[@]}"
fi

# ---- 2. Plot -------------------------------------------------------------------
if [[ $MODE != train ]] && (( ! DRY_RUN )); then
    plot_args=(--models-root "$MODELS_ROOT" --output-dir "$RESULTS_ROOT" --paper-root "$PAPER_ROOT"
               --iterations "$ITERATIONS" --seeds "${SEED_LIST[@]}")
    for pair in "${PAIRS[@]}"; do
        plot_args+=(--run "${pair#*/}" "${pair%%/*}")
    done
    run_step "Curves" "$PYTHON" "$SCRIPT_DIR/dynamics_plot.py" "${plot_args[@]}"
fi

if (( ${#FAILED[@]} )); then
    echo >&2
    echo "Some steps did not finish; see their output above:" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
exit 0
