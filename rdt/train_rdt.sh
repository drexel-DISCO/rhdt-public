#!/usr/bin/env bash
# =============================================================================
# RDT baseline (Robust Decision Transformer): train, then evaluate with the same
# protocol as RHDT (../eval), in parallel.
#
#   bash train_rdt.sh -s 48213 -d cuda:0 -j 8
#   bash train_rdt.sh -d cuda:0,cuda:1 -j 16            # all paper seeds, two GPUs
#   bash train_rdt.sh -s 48213 -t hopper-medium-v2       # only this task
#   bash train_rdt.sh -s 48213 -f hopper-medium-v2       # retrain + re-evaluate this task
#   bash train_rdt.sh -s 48213 -n                        # show what would run
#
# 1. train    one job per (seed, task); skipped if <models>/seed_<s>/rdt_<task>.pt exists
# 2. evaluate one job per (seed, task, evaluation seed): 10 evaluation seeds x 50 episodes
#             per noise level; complete results are skipped, missing levels rolled out
# 3. summary  <results>/summary/<task>.json
#                 {seed: {noise level: {mean, std, median, iqm, cvar10, n_episodes, eval_seeds}}}
#             <results>/seed_summary/rdt_<seed>.json
#                 {"<task>": {noise level: {...}}}
#             statistics over the normalized scores of all episodes of all evaluation seeds
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "train_rdt.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/../training/configs.sh
source "$SCRIPT_DIR/../training/configs.sh"
PYTHON=${PYTHON:-python}

# ---- RDT training settings -----------------------------------------------------
# Every task is trained for OFFLINE_STEPS steps with its own row below; the values are
# stored in the checkpoint. The model and optimizer settings are fixed in train_rdt.py
# (see README.md).
OFFLINE_STEPS=100000
#   WMSE_ACTION  beta_a of the Gaussian-weighted action loss (0 = plain MSE)
#   WMSE_REWARD  beta_r of the Gaussian-weighted reward loss (0 = plain MSE)
#   REWARD_LOSS  weight of the reward-prediction loss (0 = no reward prediction)
#   EMB_DROPOUT  dropout on the token embeddings
#                                  WMSE_ACTION WMSE_REWARD REWARD_LOSS EMB_DROPOUT
task_settings=(
  "halfcheetah-medium-v2           0.1         0.0         0.0         0.05"
  "halfcheetah-medium-replay-v2    2.0         0.0         0.1         0.50"
  "hopper-medium-v2                0.1         0.0         0.1         0.05"
  "hopper-medium-replay-v2         1.0         0.0         0.0         0.35"
  "walker2d-medium-v2              0.3         0.0         0.1         0.35"
  "walker2d-medium-replay-v2       0.3         0.0         0.1         0.05"
)
# Changing a row does not retrain existing checkpoints (they are skipped, with a
# warning if their saved settings differ); use -f TASK to retrain that model.

# In a terminal, the progress bar of every running training is shown live. When the
# output goes to a file, a status report is printed every PROGRESS_EVERY seconds
# instead (0 = off). The full output of every run is in its log.
PROGRESS_EVERY=60

# ---- Command line --------------------------------------------------------------
SEEDS=
DEVICES=cuda:0
JOBS=8
TASKS=
RETRAIN=
EPISODES=50
CORRUPTIONS=0.0,0.1,0.2,0.3
EVAL_SEEDS=11,13,17,19,23,29,31,37,41,43
MODELS_ROOT=models_rdt_tuned    # relative to this directory
RESULTS_ROOT=results_rdt_tuned  # relative to this directory
CONDA_ENV=
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash train_rdt.sh [options]

  -s SEEDS       training seeds, comma-separated (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated; jobs are spread over them
                 in turn, e.g. cuda:0,cuda:1 (default: $DEVICES)
  -j JOBS        number of jobs at a time (default: $JOBS)
  -t TASKS       tasks to run, comma-separated (default: all six; with -f and no -t,
                 only the forced ones)
  -f TASKS       retrain these tasks even if their checkpoint exists; their old
                 results and summary entries are deleted first and regenerated
  -E EPISODES    episodes per noise level and evaluation seed (default: $EPISODES)
  -C LEVELS      noise levels (default: $CORRUPTIONS)
  -S SEEDS       evaluation seeds (default: $EVAL_SEEDS)
  -M DIR         checkpoint root (default: rdt/$MODELS_ROOT)
  -o DIR         results root (default: rdt/$RESULTS_ROOT)
  -e ENV         conda environment to activate first
  -n             dry run: list what would be trained and evaluated
  -h             show this help

-s, -t and -f may be repeated. Each task's settings are in task_settings at the top
of this script.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while getopts ":s:d:j:t:f:E:C:S:M:o:e:nh" opt; do
    case $opt in
        s) SEEDS+=" $OPTARG" ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        t) TASKS+=" $OPTARG" ;;
        f) RETRAIN+=" $OPTARG" ;;
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

parse_seeds "${SEEDS:-$PAPER_SEEDS}" || exit 1
parse_list "$DEVICES" DEVICE_LIST
parse_list "$TASKS" SELECTED
parse_list "$RETRAIN" FORCED
parse_list "$CORRUPTIONS" LEVEL_LIST
parse_list "$EVAL_SEEDS" EVAL_SEED_LIST
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
[[ $EPISODES =~ ^[1-9][0-9]*$ ]] || die "-E needs a positive integer, got '$EPISODES'"
for t in "${SELECTED[@]}" "${FORCED[@]}"; do
    check_task "$t" || exit 1
done
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

# Tasks this run touches: -t (plus any forced ones), else only the forced ones, else all
if (( ${#SELECTED[@]} )); then
    pick=" ${SELECTED[*]} ${FORCED[*]} "
elif (( ${#FORCED[@]} )); then
    pick=" ${FORCED[*]} "
else
    pick=" $PAPER_TASKS "
fi
tasks=()
for t in $PAPER_TASKS; do
    [[ $pick == *" $t "* ]] && tasks+=("$t")
done

# settings_for TASK -> "WMSE_ACTION WMSE_REWARD REWARD_LOSS EMB_DROPOUT"
settings_for() {
    local row t a r l e
    for row in "${task_settings[@]}"; do
        read -r t a r l e <<< "$row"
        if [[ $t == "$1" ]]; then
            echo "$a $r $l $e"
            return 0
        fi
    done
    return 1
}
[[ $OFFLINE_STEPS =~ ^[1-9][0-9]*$ ]] || die "OFFLINE_STEPS needs a positive integer, got '$OFFLINE_STEPS'"
num='^([0-9]+\.?[0-9]*|\.[0-9]+)$'
for row in "${task_settings[@]}"; do
    read -r t a r l e extra <<< "$row"
    check_task "$t" || die "task_settings: unknown task '$t'"
    [[ $a =~ $num && $r =~ $num && $l =~ $num && $e =~ $num && -z $extra ]] ||
        die "task_settings: the row of $t needs 4 numbers: '$row'"
done
for t in "${tasks[@]}"; do
    settings_for "$t" > /dev/null || die "task_settings: no row for $t"
done

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
# One thread per job: many jobs each starting a thread per core would oversubscribe
# the CPU (training batches and rollouts are CPU-bound)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# What an existing checkpoint's saved settings should be (else a warning), per task
expect=("offline_steps=$OFFLINE_STEPS")
for t in "${tasks[@]}"; do
    read -r a r l e <<< "$(settings_for "$t")"
    expect+=("$t:wmse_action_coef=$a" "$t:wmse_reward_coef=$r" "$t:reward_loss_coef=$l" "$t:embedding_dropout=$e")
done
train_args=(--seeds "${SEED_LIST[@]}" --tasks "${tasks[@]}" --models-root "$MODELS_ROOT")
eval_args=("${train_args[@]}" --results-root "$RESULTS_ROOT" --eval-seeds "$EVAL_SEEDS"
           --corruptions "$CORRUPTIONS" --episodes "$EPISODES")
force_args=()
if (( ${#FORCED[@]} )); then
    force_args=(--force "${FORCED[@]}")
fi
seed_summary_dir=$RESULTS_ROOT/seed_summary

plan_file=$(mktemp) || die "could not create a temporary file"
# shellcheck disable=SC2329  # invoked by the trap below
cleanup() {
    local running pids
    running=$(jobs -pr)
    read -r -a pids <<< "${running//$'\n'/ }"
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2> /dev/null
    fi
    rm -f "$plan_file"
}
trap cleanup EXIT
trap 'echo "Interrupted: stopping the running jobs. Rerun the same command to continue." >&2; exit 130' INT TERM

# ---- Progress display ------------------------------------------------------------
declare -A train_logs=()  # running trainings: pid -> "label<TAB>log", in launch order
train_pids=()
last_progress=$SECONDS
live=0
[[ -t 1 ]] && live=1      # stdout is a terminal: live progress bars
dash_lines=0              # lines of the live display currently on screen

# Latest progress bar of a training log (or its last line before any bar appears)
bar_for() {
    local text line
    text=$(tail -c 8192 "$1" 2> /dev/null | tr '\r' '\n')
    line=$(grep -E '^[^|]*: *[0-9]+%\|' <<< "$text" | tail -n 1)
    [[ -n $line ]] || line=$(grep -v '^[[:space:]]*$' <<< "$text" | tail -n 1)
    printf '%s' "${line:-starting}"
}

# Trainings that ended since the last call: one permanent line each
report_ended() {
    local running pid label log
    running=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${train_pids[@]}"; do
        [[ -n ${train_logs[$pid]:-} && $running != *" $pid "* ]] || continue
        IFS=$'\t' read -r label log <<< "${train_logs[$pid]}"
        clear_dash
        if grep -q "Saved to:" "$log" 2> /dev/null; then
            echo "[done] $label: checkpoint saved"
        else
            echo "[FAILED] $label: see $log"
        fi
        unset "train_logs[$pid]"
    done
}

# Live display: one line per running training, redrawn in place at the bottom
clear_dash() {
    (( live && dash_lines > 0 )) || return 0
    printf '\033[%dA\033[J' "$dash_lines"
    dash_lines=0
}
draw_dash() {
    local cols pid label log line n=0
    cols=$(tput cols 2> /dev/null || echo 120)
    if (( dash_lines > 0 )); then
        printf '\033[%dA' "$dash_lines"
    fi
    for pid in "${train_pids[@]}"; do
        [[ -n ${train_logs[$pid]:-} ]] || continue
        IFS=$'\t' read -r label log <<< "${train_logs[$pid]}"
        line="$label | $(bar_for "$log")"
        printf '%s\033[K\n' "${line:0:$((cols - 1))}"
        n=$((n + 1))
    done
    printf '\033[J'
    dash_lines=$n
}

# Periodic report for non-terminal output: phase, %, steps, time left, loss per training
show_progress() {
    local pid label log items=()
    report_ended
    for pid in "${train_pids[@]}"; do
        [[ -n ${train_logs[$pid]:-} ]] || continue
        IFS=$'\t' read -r label log <<< "${train_logs[$pid]}"
        items+=("$label" "$log")
    done
    (( ${#items[@]} )) || return 0
    echo "[progress $(date +%H:%M:%S)] $(( ${#items[@]} / 2 )) training"
    "$PYTHON" results.py progress "${items[@]}"
}

# Block until at most $1 jobs are running, showing the training progress meanwhile
wait_until() {
    while (( $(jobs -pr | wc -l) > $1 )); do
        if (( live && ${#train_logs[@]} > 0 )); then
            report_ended
            draw_dash
            sleep 1
        else
            sleep 2
            if (( PROGRESS_EVERY > 0 && ${#train_logs[@]} > 0 && SECONDS - last_progress >= PROGRESS_EVERY )); then
                show_progress
                last_progress=$SECONDS
            fi
        fi
    done
}

# Run the lines of $plan_file through -j parallel slots; $1 = function that launches one
run_pool() {
    local launch=$1 line n i=0
    local -a plan
    mapfile -t plan < "$plan_file"
    n=${#plan[@]}
    for line in "${plan[@]}"; do
        i=$((i + 1))
        clear_dash
        "$launch" "$i" "$n" "$line" "${DEVICE_LIST[$(((i - 1) % ${#DEVICE_LIST[@]}))]}"
        wait_until $((JOBS - 1))
    done
    wait_until 0
    wait
}

# shellcheck disable=SC2329  # launched by run_pool
launch_train() {  # <i> <n> <plan line> <device>
    local seed task ckpt note a r l e
    IFS=$'\t' read -r seed task ckpt note <<< "$3"
    read -r a r l e <<< "$(settings_for "$task")"
    mkdir -p "$(dirname "$ckpt")"
    echo "[train $1/$2] seed=$seed $task on $4  [$note]  wmse_a=$a wmse_r=$r reward_loss=$l emb_dropout=$e"
    "$PYTHON" train_rdt.py --task "$task" --device "$4" --seed "$seed" \
        --offline-steps "$OFFLINE_STEPS" --wmse-action-coef "$a" --wmse-reward-coef "$r" \
        --reward-loss-coef "$l" --embedding-dropout "$e" --save-path "$ckpt" \
        > "${ckpt%.pt}_train.log" 2>&1 &
    train_logs[$!]="seed=$seed $task"$'\t'"${ckpt%.pt}_train.log"
    train_pids+=("$!")
}

# shellcheck disable=SC2329  # launched by run_pool
launch_eval() {  # <i> <n> <plan line> <device>
    local seed task es ckpt rdir note
    IFS=$'\t' read -r seed task es ckpt rdir note <<< "$3"
    mkdir -p "$rdir"
    echo "[eval $1/$2] seed=$seed $task eval_seed=$es on $4  [$note]"
    "$PYTHON" eval_rdt.py \
        --model-path "$ckpt" \
        --output-dir "$rdir" \
        --corruptions "$CORRUPTIONS" \
        --eval-episodes "$EPISODES" \
        --device "$4" \
        --seed "$es" \
        --resume \
        > "$rdir/rdt_${task}.log" 2>&1 &
}

echo "RDT: training seeds ${SEED_LIST[*]} | tasks ${tasks[*]} | devices ${DEVICE_LIST[*]} | jobs $JOBS${FORCED[*]:+ | retrain ${FORCED[*]}}"
echo "  evaluation: ${#EVAL_SEED_LIST[@]} evaluation seeds x $EPISODES episodes at noise levels $CORRUPTIONS"
echo "  settings ($OFFLINE_STEPS steps; WMSE_ACTION WMSE_REWARD REWARD_LOSS EMB_DROPOUT):"
for t in "${tasks[@]}"; do
    printf '    %-30s %s\n' "$t" "$(settings_for "$t")"
done

# ---- 1. Train (existing checkpoints are skipped) ----------------------------------
"$PYTHON" results.py plan-train "${train_args[@]}" --expect "${expect[@]}" "${force_args[@]}" > "$plan_file" ||
    die "planning failed"

if (( DRY_RUN )); then
    awk -F'\t' 'NF {printf "train  seed=%s %s  [%s]\n", $1, $2, $4}' "$plan_file"
    "$PYTHON" results.py plan "${eval_args[@]}" --no-ckpt-check "${force_args[@]}" |
        awk -F'\t' 'NF {printf "eval   seed=%s %s eval_seed=%s  [%s]\n", $1, $2, $3, $6}'
    exit 0
fi

TRAIN_FAILED=0
if [[ -s $plan_file ]]; then
    # Forced models: delete their old results first, so they can only be regenerated
    # from whatever checkpoint is on disk (the new one, or the old one if training fails)
    if (( ${#FORCED[@]} )); then
        "$PYTHON" results.py invalidate "${eval_args[@]}" "${force_args[@]}" --seed-summary-dir "$seed_summary_dir"
    fi
    stamp=$(mktemp)
    train_plan=$(mktemp)
    cp "$plan_file" "$train_plan"
    if (( live )); then
        echo "Live progress below; full output of a run: $MODELS_ROOT/seed_<seed>/rdt_<task>_train.log"
    elif (( PROGRESS_EVERY > 0 )); then
        echo "Progress every ${PROGRESS_EVERY}s; full output of a run: $MODELS_ROOT/seed_<seed>/rdt_<task>_train.log"
    fi
    run_pool launch_train
    report_ended
    clear_dash
    train_logs=()
    train_pids=()
    # A training succeeded if its checkpoint was (re)written during this run; the old
    # .pt is only replaced once the new one is completely saved, so failures keep it
    while IFS=$'\t' read -r seed task ckpt note; do
        if [[ ! $ckpt -nt $stamp ]]; then
            if (( TRAIN_FAILED++ == 0 )); then
                echo "WARNING: training failed (see the log):" >&2
            fi
            if [[ $note == forced ]]; then
                echo "  ${ckpt%.pt}_train.log  (kept the old checkpoint; its evaluations are redone from it)" >&2
            else
                echo "  ${ckpt%.pt}_train.log  (no checkpoint; its evaluations are skipped)" >&2
            fi
        fi
    done < "$train_plan"
    rm -f "$stamp" "$train_plan"
fi

# ---- 2. Evaluate (only what is missing) -------------------------------------------
INCOMPLETE=0
"$PYTHON" results.py plan "${eval_args[@]}" > "$plan_file" || die "planning failed"
if [[ -s $plan_file ]]; then
    run_pool launch_eval
    "$PYTHON" results.py plan "${eval_args[@]}" > "$plan_file" 2> /dev/null
    if [[ -s $plan_file ]]; then
        INCOMPLETE=1
        echo "WARNING: $(wc -l < "$plan_file") evaluation job(s) did not finish; see their logs:" >&2
        awk -F'\t' '{printf "  %s/rdt_%s.log  [%s]\n", $5, $2, $6}' "$plan_file" >&2
    fi
fi

# ---- 3. Summaries: pooled over evaluation seeds, per training seed and noise level ----
"$PYTHON" results.py summarize "${eval_args[@]}" --seed-summary-dir "$seed_summary_dir" ||
    die "summarizing failed"
if (( TRAIN_FAILED || INCOMPLETE )); then
    exit 1
fi
exit 0
