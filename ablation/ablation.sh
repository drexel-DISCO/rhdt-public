#!/usr/bin/env bash
# =============================================================================
# RHDT ablations. For every task given with -t: train six variants of the paper's
# model with every training seed, evaluate every model with every evaluation seed,
# and write a table.
#
#   bash ablation.sh -t halfcheetah-medium-v2 -d cuda:0 -j 6
#   bash ablation.sh -t hopper-medium-v2,walker2d-medium-v2 -d cuda:0,cuda:1 -j 12
#   bash ablation.sh -t halfcheetah-medium-v2 -v no_hopfield,one_head   # some variants
#   bash ablation.sh -t halfcheetah-medium-v2 -n                         # show the runs
#
# Variants ("tuned" is the task's paper config, e.g. gpheavy for halfcheetah-medium-v2):
#   full          Full RHDT                                            config tuned
#   gp_only       the tuned lambda_gp only, lambda_sep = 0             config tuned-gp
#   sep_only      the tuned lambda_sep only, lambda_gp = 0             config tuned-sep
#   non           no penalties                                         config non
#   no_hopfield   Full RHDT without the Hopfield update (0 iterations) config tuned, -i 0
#   one_head      Full RHDT with one attention head instead of four    config tuned-h1
# Everything else is as in the paper (1 Hopfield iteration, 4 heads, 100k steps).
#
# Training and evaluation are ../training/train.sh and ../eval/eval.sh with their own
# output folders, so the paper's models and results are not touched:
#   models_ablation/iter<N>/<config>/seed_<seed>/rhdt_<task>.pt
#   results_ablation/               per-run results and summaries, as in ../eval
#   results_ablation/ablation_<task>.json and .md    the ablation table
# Finished training runs and evaluation jobs are skipped, so rerun the same command
# to continue after an interruption.
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "ablation.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SH=$SCRIPT_DIR/../training/train.sh
EVAL_SH=$SCRIPT_DIR/../eval/eval.sh
# shellcheck source=SCRIPTDIR/../training/configs.sh
source "$SCRIPT_DIR/../training/configs.sh"
PYTHON=${PYTHON:-python}

# Variant -> "<config for train.sh -c> <Hopfield iterations>"
ALL_VARIANTS="full gp_only sep_only non no_hopfield one_head"
#ALL_VARIANTS="gp_only sep_only non no_hopfield one_head"
variant_spec() {  # <variant>
    case $1 in
        full)        echo "tuned 1" ;;
        gp_only)     echo "tuned-gp 1" ;;
        sep_only)    echo "tuned-sep 1" ;;
        non)         echo "non 1" ;;
        no_hopfield) echo "tuned 0" ;;
        one_head)    echo "tuned-h1 1" ;;
        *)           return 1 ;;
    esac
}

# ---- Command line --------------------------------------------------------------
TASKS=
VARIANTS=${ALL_VARIANTS// /,}
SEEDS=$PAPER_SEEDS
DEVICES=cuda:0
JOBS=6
MODE=all
EPISODES=50
CORRUPTIONS=0.0,0.1,0.2,0.3
#CORRUPTIONS=0.0,0.3
EVAL_SEEDS=11,13,17,19,23,29,31,37,41,43
MODELS_ROOT=$SCRIPT_DIR/models_ablation
RESULTS_ROOT=$SCRIPT_DIR/results_ablation
CONDA_ENV=
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash ablation.sh -t TASKS [options]

  -t TASKS       tasks, comma-separated, or all (required)
  -v VARIANTS    variants, comma-separated        (default: all six)
                 full, gp_only, sep_only, non, no_hopfield, one_head
  -s SEEDS       training seeds, comma-separated  (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated (default: $DEVICES)
  -j JOBS        training runs, and then evaluation jobs, at a time (default: $JOBS)
  -m MODE        train: only train; eval: only evaluate and write the table;
                 all: both                        (default: $MODE)
  -E EPISODES    episodes per noise level and evaluation seed (default: $EPISODES)
  -C LEVELS      noise levels                     (default: $CORRUPTIONS)
  -S SEEDS       evaluation seeds                 (default: $EVAL_SEEDS)
  -M DIR         trained models                   (default: ablation/models_ablation)
  -o DIR         results and tables               (default: ablation/results_ablation)
  -e ENV         conda environment to activate first
  -n             dry run: list the training runs and evaluation jobs, start nothing
  -h             show this help

Variants ("tuned" is the task's paper config, listed by bash ../training/train.sh -h):
  full           Full RHDT                                    config tuned
  gp_only        the tuned lambda_gp only, lambda_sep = 0     config tuned-gp
  sep_only       the tuned lambda_sep only, lambda_gp = 0     config tuned-sep
  non            no penalties                                 config non
  no_hopfield    Full RHDT with 0 Hopfield iterations         config tuned, -i 0
  one_head       Full RHDT with 1 attention head              config tuned-h1
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while getopts ":t:v:s:d:j:m:E:C:S:M:o:e:nh" opt; do
    case $opt in
        t) TASKS=$OPTARG ;;
        v) VARIANTS=$OPTARG ;;
        s) SEEDS=$OPTARG ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        m) MODE=$OPTARG ;;
        E) EPISODES=$OPTARG ;;
        C) CORRUPTIONS=$OPTARG ;;
        S) EVAL_SEEDS=$OPTARG ;;
        M) MODELS_ROOT=$(abs_path "$OPTARG") ;;
        o) RESULTS_ROOT=$(abs_path "$OPTARG") ;;
        e) CONDA_ENV=$OPTARG ;;
        n) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        :) die "option -$OPTARG needs a value (see -h)" ;;
        *) die "unknown option -$OPTARG (see -h)" ;;
    esac
done
shift $((OPTIND - 1))
(( $# == 0 )) || die "unexpected argument '$1'; lists are comma-separated, e.g. -s 42,48213"

[[ -n $TASKS ]] || die "give the task(s) with -t, e.g. -t halfcheetah-medium-v2 (see -h)"
[[ $TASKS == all ]] && TASKS=$PAPER_TASKS
parse_list "$TASKS" TASK_LIST
for task in "${TASK_LIST[@]}"; do
    check_task "$task" || exit 1
done
parse_list "$VARIANTS" VARIANT_LIST
(( ${#VARIANT_LIST[@]} )) || die "no variants given"
for v in "${VARIANT_LIST[@]}"; do
    variant_spec "$v" > /dev/null || die "unknown variant '$v'; variants: ${ALL_VARIANTS// /, }"
done
parse_seeds "$SEEDS" || exit 1
parse_list "$DEVICES" DEVICE_LIST
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
case $MODE in train|eval|all) ;; *) die "-m must be train, eval or all, got '$MODE'" ;; esac
[[ $EPISODES =~ ^[1-9][0-9]*$ ]] || die "-E needs a positive integer, got '$EPISODES'"
parse_list "$CORRUPTIONS" LEVEL_LIST
(( ${#LEVEL_LIST[@]} )) || die "no noise levels given"
for level in "${LEVEL_LIST[@]}"; do
    [[ $level =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || die "noise level '$level' is not a number"
done
parse_list "$EVAL_SEEDS" EVAL_SEED_LIST
(( ${#EVAL_SEED_LIST[@]} )) || die "no evaluation seeds given"
for es in "${EVAL_SEED_LIST[@]}"; do
    [[ $es =~ ^[0-9]+$ ]] || die "evaluation seed '$es' is not a non-negative integer"
done
TASKS=$(IFS=,; echo "${TASK_LIST[*]}")
SEEDS=$(IFS=,; echo "${SEED_LIST[*]}")
DEVICES=$(IFS=,; echo "${DEVICE_LIST[*]}")
CORRUPTIONS=$(IFS=,; echo "${LEVEL_LIST[*]}")
EVAL_SEEDS=$(IFS=,; echo "${EVAL_SEED_LIST[*]}")

if [[ -n $CONDA_ENV ]]; then
    command -v conda > /dev/null 2>&1 || die "conda not found, but -e $CONDA_ENV was given"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || die "could not load conda's shell functions"
    conda activate "$CONDA_ENV" || die "could not activate conda environment '$CONDA_ENV'"
fi
command -v "$PYTHON" > /dev/null 2>&1 || die "'$PYTHON' not found; activate the environment or use -e ENV"

# train.sh and eval.sh take one Hopfield iteration count per call: group the variants'
# configs by it (iterations -> "config,config,...")
declare -A GROUP=()
for v in "${VARIANT_LIST[@]}"; do
    read -r spec iters <<< "$(variant_spec "$v")"
    GROUP[$iters]+="${GROUP[$iters]:+,}$spec"
done
mapfile -t ITER_LIST < <(printf '%s\n' "${!GROUP[@]}" | sort -rn)

echo "RHDT ablation: tasks ${TASK_LIST[*]} | training seeds ${SEED_LIST[*]} | devices ${DEVICE_LIST[*]} | jobs $JOBS"
echo "  evaluation: ${#EVAL_SEED_LIST[@]} evaluation seeds x $EPISODES episodes at noise levels $CORRUPTIONS"
echo "  models: $MODELS_ROOT"
echo "  results: $RESULTS_ROOT"
for task in "${TASK_LIST[@]}"; do
    echo "  $task (tuned config: $(tuned_config "$task")):"
    for v in "${VARIANT_LIST[@]}"; do
        read -r spec iters <<< "$(variant_spec "$v")"
        cfg=$(resolve_config "$spec" "$task")
        read -r gp sep <<< "$(config_lambdas "$cfg")"
        printf '    %-12s %-14s lambda_gp=%-7s lambda_sep=%-6s iterations=%s heads=%s\n' \
            "$v" "$cfg" "$gp" "$sep" "$iters" "$(config_heads "$cfg")"
    done
done

DRY=()
if (( DRY_RUN )); then
    DRY=(-n)
fi
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
    echo "Ablation interrupted. Rerun the same command to continue." >&2
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
if [[ $MODE != eval ]]; then
    for iters in "${ITER_LIST[@]}"; do
        run_step "Training ${GROUP[$iters]} with $iters Hopfield iteration(s)" \
            bash "$TRAIN_SH" -s "$SEEDS" -t "$TASKS" -c "${GROUP[$iters]}" -i "$iters" \
                -d "$DEVICES" -j "$JOBS" -o "$MODELS_ROOT" "${DRY[@]}"
    done
fi

# ---- 2. Evaluate, then tabulate ------------------------------------------------------
# shellcheck disable=SC2329  # run by run_step
evaluate() {  # <iterations>
    if (( DRY_RUN )); then
        # Before training, every model is missing; the plan line counts them
        bash "$EVAL_SH" -s "$SEEDS" -t "$TASKS" -c "${GROUP[$1]}" -i "$1" -d "$DEVICES" -j "$JOBS" \
            -E "$EPISODES" -C "$CORRUPTIONS" -S "$EVAL_SEEDS" -M "$MODELS_ROOT" -o "$RESULTS_ROOT" -n 2>&1 |
            grep -v '^WARNING: missing checkpoint'
        return 0
    fi
    bash "$EVAL_SH" -s "$SEEDS" -t "$TASKS" -c "${GROUP[$1]}" -i "$1" -d "$DEVICES" -j "$JOBS" \
        -E "$EPISODES" -C "$CORRUPTIONS" -S "$EVAL_SEEDS" -M "$MODELS_ROOT" -o "$RESULTS_ROOT"
}

if [[ $MODE != train ]]; then
    for iters in "${ITER_LIST[@]}"; do
        run_step "Evaluating ${GROUP[$iters]} with $iters Hopfield iteration(s)" evaluate "$iters"
    done
    if (( ! DRY_RUN )); then
        table_args=(--models-root "$MODELS_ROOT" --results-root "$RESULTS_ROOT"
                    --seeds "${SEED_LIST[@]}" --eval-seeds "$EVAL_SEEDS"
                    --episodes "$EPISODES" --corruptions "$CORRUPTIONS")
        for task in "${TASK_LIST[@]}"; do
            for v in "${VARIANT_LIST[@]}"; do
                read -r spec iters <<< "$(variant_spec "$v")"
                table_args+=(--run "$task" "$v" "$(resolve_config "$spec" "$task")" "$iters")
            done
        done
        run_step "Ablation table" "$PYTHON" "$SCRIPT_DIR/ablation_table.py" "${table_args[@]}"
    fi
fi

if (( ${#FAILED[@]} )); then
    echo >&2
    echo "Some steps did not finish; see their output above:" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
exit 0
