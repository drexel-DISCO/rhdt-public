#!/usr/bin/env bash
# =============================================================================
# Evaluate trained RHDT models under observation noise.
#
# One job = one checkpoint (training seed, config, task) x one evaluation seed:
# a run of eval_rhdt.py that rolls the policy out at every noise level. With the
# defaults, each checkpoint gets 10 evaluation seeds x 50 episodes = 500 episodes
# per noise level. Jobs run in parallel (-j); each seeds itself, so the results
# are the same as running them one by one.
#
#   bash eval.sh -s 48213 -d cuda:0 -j 8
#   bash eval.sh -d cuda:0,cuda:1 -j 16                    # all paper seeds, two GPUs
#   bash eval.sh -s 48213 -c half/walker2d-medium-replay-v2  # one more config
#   bash eval.sh -s 48213 -n                               # list pending jobs only
#
# Nothing finished is redone:
#   result json complete              -> job skipped
#   some noise levels missing         -> only those levels are rolled out
#   header/degradation fields missing -> rewritten from the checkpoint, no rollouts
#   summary statistic missing         -> only that statistic is computed
#
# Outputs, under results_rhdt/:
#   iterations_<N>/seed_<seed>/eval_seed_<es>/<config>/rhdt_<task>_corruptions.json (and .log)
#   summary/<task>_<config>_iter<N>.json
#       {seed: {noise level: {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
#   seed_summary/rhdt_<seed>.json
#       {"<task>_<config>_iter<N>": {noise level: {...}}}
# Statistics are over the normalized scores of all episodes of all evaluation seeds.
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "eval.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/../training/configs.sh
source "$SCRIPT_DIR/../training/configs.sh"
PYTHON=${PYTHON:-python}

# ---- Command line --------------------------------------------------------------
SEEDS=$PAPER_SEEDS
DEVICES=cuda:0
JOBS=4
TASKS=all
CONFIGS=tuned,non
ITERATIONS=1
EPISODES=50
CORRUPTIONS=0.0,0.1,0.2,0.3
EVAL_SEEDS=11,13,17,19,23,29,31,37,41,43
MODELS_ROOT=../training/models_rhdt_sweep  # relative to this directory
RESULTS_ROOT=results_rhdt                  # relative to this directory
CONDA_ENV=
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash eval.sh [options]

  -s SEEDS       training seeds, comma-separated    (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated; jobs are spread over them
                 in turn, e.g. cuda:0,cuda:1          (default: $DEVICES)
  -j JOBS        number of jobs at a time             (default: $JOBS)
  -t TASKS       tasks, comma-separated, or all       (default: $TASKS)
  -c CONFIGS     configs, comma-separated, applied to every task; tuned = each
                 task's paper config; CONFIG/TASK adds one pair (default: $CONFIGS)
  -i N           Hopfield iterations (0 = standard attention): evaluates
                 <models>/iter<N> with N iterations
                                                      (default: $ITERATIONS)
  -E EPISODES    episodes per noise level and evaluation seed (default: $EPISODES)
  -C LEVELS      noise levels                         (default: $CORRUPTIONS)
  -S SEEDS       evaluation seeds                     (default: $EVAL_SEEDS)
  -M DIR         trained models, i.e. train.sh's output root
                                                      (default: training/models_rhdt_sweep)
  -o DIR         results root                         (default: eval/$RESULTS_ROOT)
  -e ENV         conda environment to activate first
  -n             dry run: list the pending jobs, start nothing
  -h             show this help

EOF
    print_configs
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while getopts ":s:d:j:t:c:i:E:C:S:M:o:e:nh" opt; do
    case $opt in
        s) SEEDS=$OPTARG ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        t) TASKS=$OPTARG ;;
        c) CONFIGS=$OPTARG ;;
        i) ITERATIONS=$OPTARG ;;
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

parse_seeds "$SEEDS" || exit 1
expand_pairs "$TASKS" "$CONFIGS" || exit 1
parse_list "$DEVICES" DEVICE_LIST
parse_list "$CORRUPTIONS" LEVEL_LIST
parse_list "$EVAL_SEEDS" EVAL_SEED_LIST
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
[[ $ITERATIONS =~ ^[0-9]+$ ]] || die "-i needs a non-negative integer, got '$ITERATIONS'"
ITERATIONS=$((10#$ITERATIONS))  # 00 -> 0, so the models are read from iter0
[[ $EPISODES =~ ^[1-9][0-9]*$ ]] || die "-E needs a positive integer, got '$EPISODES'"
(( ${#LEVEL_LIST[@]} )) || die "no noise levels given"
for level in "${LEVEL_LIST[@]}"; do
    [[ $level =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || die "noise level '$level' is not a number"
done
(( ${#EVAL_SEED_LIST[@]} )) || die "no evaluation seeds given"
for es in "${EVAL_SEED_LIST[@]}"; do
    [[ $es =~ ^[0-9]+$ ]] || die "evaluation seed '$es' is not a non-negative integer"
done
CORRUPTIONS=$(IFS=,; echo "${LEVEL_LIST[*]}")
EVAL_SEEDS=$(IFS=,; echo "${EVAL_SEED_LIST[*]}")

# Default paths are relative to this directory (paths given above are absolute now)
cd "$SCRIPT_DIR" || die "cannot enter $SCRIPT_DIR"

# ---- Environment (also needed to plan) ---------------------------------------------
if [[ -n $CONDA_ENV ]]; then
    command -v conda > /dev/null 2>&1 || die "conda not found, but -e $CONDA_ENV was given"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || die "could not load conda's shell functions"
    conda activate "$CONDA_ENV" || die "could not activate conda environment '$CONDA_ENV'"
fi
command -v "$PYTHON" > /dev/null 2>&1 || die "'$PYTHON' not found; activate the environment or use -e ENV"
# One thread per job: rollouts use batch size 1, and many jobs each starting a
# thread per core would oversubscribe the CPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

COMMON=(--seeds "${SEED_LIST[@]}" --runs "${PAIRS[@]}" --eval-seeds "$EVAL_SEEDS"
        --corruptions "$CORRUPTIONS" --episodes "$EPISODES" --iterations "$ITERATIONS"
        --results-root "$RESULTS_ROOT")
MODELS=$MODELS_ROOT/iter$ITERATIONS

PLAN_FILE=$(mktemp) || die "could not create a temporary file"
# shellcheck disable=SC2329  # invoked by the trap below
cleanup() {
    local running pids
    running=$(jobs -pr)
    read -r -a pids <<< "${running//$'\n'/ }"
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2> /dev/null
    fi
    rm -f "$PLAN_FILE"
}
trap cleanup EXIT
trap 'echo "Interrupted: stopping the running jobs. Rerun the same command to continue." >&2; exit 130' INT TERM

# ---- Plan: the jobs that still have something missing -------------------------------
echo "RHDT evaluation: ${#SEED_LIST[@]} training seed(s) x ${#PAIRS[@]} task/config pair(s): ${PAIRS[*]}"
echo "  ${#EVAL_SEED_LIST[@]} evaluation seeds x $EPISODES episodes at noise levels $CORRUPTIONS," \
     "$ITERATIONS Hopfield iteration(s)"
echo "  models: $MODELS   results: $RESULTS_ROOT   devices: ${DEVICE_LIST[*]}   jobs: $JOBS"
"$PYTHON" results.py plan "${COMMON[@]}" --models-root "$MODELS" > "$PLAN_FILE" || die "planning failed"
mapfile -t PLAN < "$PLAN_FILE"

if (( DRY_RUN )); then
    printf '%s\n' "${PLAN[@]}" |
        awk -F'\t' 'NF {printf "seed=%s %s/%s eval_seed=%s  [%s]\n", $1, $2, $3, $4, $7}'
    exit 0
fi

# ---- Run: at most -j jobs at a time --------------------------------------------------
n=${#PLAN[@]}
i=0
for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r seed cfg task es ckpt rdir note <<< "$line"
    device=${DEVICE_LIST[$((i % ${#DEVICE_LIST[@]}))]}
    i=$((i + 1))
    mkdir -p "$rdir" || die "could not create $rdir"
    echo "[$i/$n] seed=$seed $cfg/$task eval_seed=$es on $device  [$note]"
    "$PYTHON" eval_rhdt.py \
        --model-path "$ckpt" \
        --output-dir "$rdir" \
        --corruptions "$CORRUPTIONS" \
        --eval-episodes "$EPISODES" \
        --iterations "$ITERATIONS" \
        --device "$device" \
        --seed "$es" \
        --resume \
        > "$rdir/rhdt_${task}.log" 2>&1 &
    while (( $(jobs -pr | wc -l) >= JOBS )); do
        wait -n
    done
done
wait

# ---- Check: anything still incomplete failed (see its log) ----------------------------
INCOMPLETE=0
if (( n > 0 )); then
    "$PYTHON" results.py plan "${COMMON[@]}" --models-root "$MODELS" > "$PLAN_FILE" 2> /dev/null
    if [[ -s $PLAN_FILE ]]; then
        INCOMPLETE=1
        echo "WARNING: $(wc -l < "$PLAN_FILE") job(s) did not finish; see their logs:" >&2
        awk -F'\t' '{printf "  %s/rhdt_%s.log  [%s]\n", $6, $3, $7}' "$PLAN_FILE" >&2
    fi
fi

# ---- Summaries: pooled over evaluation seeds, per training seed and noise level -------
"$PYTHON" results.py summarize "${COMMON[@]}" --seed-summary-dir "$RESULTS_ROOT/seed_summary" ||
    die "summarizing failed"
exit "$INCOMPLETE"
