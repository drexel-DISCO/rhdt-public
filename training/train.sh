#!/usr/bin/env bash
# =============================================================================
# Train RHDT on the D4RL locomotion tasks.
#
# Every (seed, task, config) combination is one independent run of
# train_rhdt.py. Runs execute one after another, or several at a time with -j.
# Finished runs are skipped, so an interrupted sweep can simply be restarted.
#
#   bash train.sh                                    # paper setup (see -h)
#   bash train.sh -s 48213 -d cuda:0                 # one seed
#   bash train.sh -s 42,48213 -d cuda:0,cuda:1 -j 8  # 8 runs at a time on two GPUs
#   bash train.sh -m eval -j 4                       # evaluate the trained models
#   bash train.sh -n                                 # list the runs, start nothing
#
# Outputs: <out>/iter<N>/<config>/seed_<seed>/rhdt_<task>.{pt,json}, with a
# config.txt per folder; a folder that holds a different config is refused.
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "train.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/configs.sh
source "$SCRIPT_DIR/configs.sh"
PYTHON=${PYTHON:-python}

# Settings shared by every run (the attention heads, NUM_HEADS, are in configs.sh)
STEPS=100000
METHOD=k_ortho

# ---- Command line --------------------------------------------------------------
SEEDS=$PAPER_SEEDS
DEVICES=cuda:0
JOBS=1
MODE=train
TASKS=all
CONFIGS=tuned,non
ITERATIONS=1
EVAL_EPISODES=300
CORRUPTIONS=0.0,0.1,0.2,0.3
OUT_ROOT=models_rhdt_sweep  # relative to this directory
CONDA_ENV=
FORCE=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash train.sh [options]

  -s SEEDS       training seeds, comma-separated    (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated; parallel runs are spread
                 over them evenly, e.g. cuda:0,cuda:1 (default: $DEVICES)
  -j JOBS        number of runs at a time             (default: $JOBS)
  -m MODE        train: train and save checkpoints    (default: $MODE)
                 eval:  evaluate the saved checkpoints
                 all:   train if needed, then evaluate
  -t TASKS       tasks, comma-separated, or all       (default: $TASKS)
  -c CONFIGS     configs, comma-separated, applied to every task; tuned = each
                 task's paper config; CONFIG/TASK adds one pair (default: $CONFIGS)
  -i N           Hopfield iterations per attention layer; 0 = standard attention
                                                      (default: $ITERATIONS)
  -E EPISODES    evaluation episodes per noise level  (default: $EVAL_EPISODES)
  -C LEVELS      evaluation noise levels              (default: $CORRUPTIONS)
  -o DIR         output root                          (default: training/$OUT_ROOT)
  -e ENV         conda environment to activate first
  -f             redo runs that are already complete
  -n             dry run: print the runs and commands, start nothing
  -h             show this help

EOF
    print_configs
    echo
    echo "With -j > 1 each run's output goes to <out>/logs/<date-time>/<config>_seed<seed>_<task>.log"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while getopts ":s:d:j:m:t:c:i:E:C:o:e:fnh" opt; do
    case $opt in
        s) SEEDS=$OPTARG ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        m) MODE=$OPTARG ;;
        t) TASKS=$OPTARG ;;
        c) CONFIGS=$OPTARG ;;
        i) ITERATIONS=$OPTARG ;;
        E) EVAL_EPISODES=$OPTARG ;;
        C) CORRUPTIONS=$OPTARG ;;
        o) OUT_ROOT=$(abs_path "$OPTARG") ;;
        e) CONDA_ENV=$OPTARG ;;
        f) FORCE=1 ;;
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
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
[[ $ITERATIONS =~ ^[0-9]+$ ]] || die "-i needs a non-negative integer, got '$ITERATIONS'"
ITERATIONS=$((10#$ITERATIONS))  # 00 -> 0, so the output folder is iter0
[[ $EVAL_EPISODES =~ ^[1-9][0-9]*$ ]] || die "-E needs a positive integer, got '$EVAL_EPISODES'"
case $MODE in train|eval|all) ;; *) die "-m must be train, eval or all, got '$MODE'" ;; esac
(( ${#LEVEL_LIST[@]} )) || die "no noise levels given"
for level in "${LEVEL_LIST[@]}"; do
    [[ $level =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || die "noise level '$level' is not a number"
done
CORRUPTIONS=$(IFS=,; echo "${LEVEL_LIST[*]}")

# Default paths are relative to this directory (paths given above are absolute now)
cd "$SCRIPT_DIR" || die "cannot enter $SCRIPT_DIR"

# ---- Runs --------------------------------------------------------------------
RUN_SEED=()
RUN_TASK=()
RUN_CFG=()
for seed in "${SEED_LIST[@]}"; do
    for pair in "${PAIRS[@]}"; do
        RUN_SEED+=("$seed")
        RUN_CFG+=("${pair%%/*}")
        RUN_TASK+=("${pair#*/}")
    done
done
N=${#RUN_SEED[@]}

run_dir() {  # <config> <seed>
    echo "$OUT_ROOT/iter$ITERATIONS/$1/seed_$2"
}

run_label() {  # <run index>
    echo "seed=${RUN_SEED[$1]} ${RUN_TASK[$1]} ${RUN_CFG[$1]}"
}

build_cmd() {  # <run index> <device>  -> CMD
    local gp sep beta
    read -r gp sep <<< "$(config_lambdas "${RUN_CFG[$1]}")"
    CMD=("$PYTHON" train_rhdt.py
         --task "${RUN_TASK[$1]}" --seed "${RUN_SEED[$1]}" --device "$2"
         --steps "$STEPS" --num-heads "$(config_heads "${RUN_CFG[$1]}")" --num-iterations "$ITERATIONS"
         --method "$METHOD" --lambda-gp "$gp" --lambda-sep "$sep"
         --mode "$MODE" --eval-episodes "$EVAL_EPISODES" --corruptions "$CORRUPTIONS"
         --output-dir "$(run_dir "${RUN_CFG[$1]}" "${RUN_SEED[$1]}")")
    beta=$(config_beta "${RUN_CFG[$1]}")
    if [[ $beta != 1 ]]; then
        CMD+=(--beta-scale "$beta")
    fi
    if (( FORCE )); then
        CMD+=(--force)
    fi
}

# One config.txt per output folder records what the folder holds. Refuse to mix
# results of different configs under the same name.
check_config_files() {
    local i dir gp sep beta beta_line expected conflict=0
    local -A checked=()
    for ((i = 0; i < N; i++)); do
        dir=$(run_dir "${RUN_CFG[i]}" "${RUN_SEED[i]}")
        [[ -n ${checked[$dir]:-} ]] && continue
        checked[$dir]=1
        read -r gp sep <<< "$(config_lambdas "${RUN_CFG[i]}")"
        beta=$(config_beta "${RUN_CFG[i]}")
        beta_line=
        if [[ $beta != 1 ]]; then  # recorded only when not the default
            beta_line=$'\n'"beta_scale=$beta"
        fi
        expected="cfg_name=${RUN_CFG[i]}
lambda_gp=$gp
lambda_sep=$sep
num_iterations=$ITERATIONS
num_heads=$(config_heads "${RUN_CFG[i]}")$beta_line
method=$METHOD
steps=$STEPS
seed=${RUN_SEED[i]}"
        if [[ -f $dir/config.txt ]]; then
            if [[ "$(< "$dir/config.txt")" != "$expected" ]]; then
                echo "Error: $dir already holds results of a different config:" >&2
                sed 's/^/    /' "$dir/config.txt" >&2
                conflict=1
            fi
        elif (( ! DRY_RUN )); then
            mkdir -p "$dir" || return 1
            printf '%s\n' "$expected" > "$dir/config.txt.tmp.$$" || return 1
            mv "$dir/config.txt.tmp.$$" "$dir/config.txt" || return 1
        fi
    done
    if (( conflict )); then
        echo "Use another config name or output root (-o), or move the old folder." >&2
        return 1
    fi
}

fmt_duration() {  # <seconds>
    printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60))
}

# ---- Plan ----------------------------------------------------------------------
LOG_DIR=
if (( JOBS > 1 )); then
    LOG_DIR=$OUT_ROOT/logs/$(date +%Y%m%d-%H%M%S)
fi
echo "RHDT training: $N runs = ${#SEED_LIST[@]} seed(s) x ${#PAIRS[@]} task/config pair(s)"
echo "  mode=$MODE  devices=${DEVICE_LIST[*]}  jobs=$JOBS  hopfield_iterations=$ITERATIONS  steps=$STEPS"
echo "  output: $OUT_ROOT/iter$ITERATIONS/<config>/seed_<seed>/"
if [[ $MODE != train ]]; then
    echo "  evaluation: $EVAL_EPISODES episodes at noise levels $CORRUPTIONS"
fi
if [[ -n $LOG_DIR ]]; then
    echo "  logs: $LOG_DIR/"
fi

check_config_files || exit 1

if (( DRY_RUN )); then
    echo
    for ((i = 0; i < N; i++)); do
        build_cmd "$i" "${DEVICE_LIST[$((i % JOBS % ${#DEVICE_LIST[@]}))]}"
        printf '[%*d/%d] %s\n    ' ${#N} $((i + 1)) "$N" "$(run_label "$i")"
        printf '%q ' "${CMD[@]}"
        echo
    done
    if (( JOBS > 1 )); then
        echo "(with -j > 1, devices are assigned to free slots at run time)"
    fi
    exit 0
fi

# ---- Environment -----------------------------------------------------------------
if [[ -n $CONDA_ENV ]]; then
    command -v conda > /dev/null 2>&1 || die "conda not found, but -e $CONDA_ENV was given"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || die "could not load conda's shell functions"
    conda activate "$CONDA_ENV" || die "could not activate conda environment '$CONDA_ENV'"
fi
command -v "$PYTHON" > /dev/null 2>&1 || die "'$PYTHON' not found; activate the environment or use -e ENV"
# Deterministic cuBLAS; must be set before CUDA initializes
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
# Parallel runs share the CPU cores: without a cap, every run starts one PyTorch
# thread per core and the CPU is oversubscribed. Set OMP_NUM_THREADS to override.
if (( JOBS > 1 )) && [[ -z ${OMP_NUM_THREADS:-} ]]; then
    cores=$(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 1)
    OMP_NUM_THREADS=$((cores / JOBS > 1 ? cores / JOBS : 1))
    export OMP_NUM_THREADS
    echo "  CPU threads per run: $OMP_NUM_THREADS ($cores cores / $JOBS runs; set OMP_NUM_THREADS to change)"
fi

# ---- Run -------------------------------------------------------------------------
N_OK=0
N_DONE=0
FAILED=()
NOT_TRAINED=()
RUN_LOG=()
RUN_START=()
SLOT_RUN=()  # parallel slot -> run index, -1 = free
SLOT_PID=()  # parallel slot -> pid of the subshell running it
FG_PID=      # with -j 1: pid of the current run
SECONDS=0

report() {  # <run index> <exit code>
    local i=$1 rc=$2 took="" note="" count
    N_DONE=$((N_DONE + 1))
    count=$(printf '%*d/%d' ${#N} "$N_DONE" "$N")
    if [[ -n ${RUN_START[i]:-} ]]; then
        took=" in $(fmt_duration $((SECONDS - RUN_START[i])))"
    fi
    case $rc in
        0)
            N_OK=$((N_OK + 1))
            if [[ -n ${RUN_LOG[i]:-} ]] && grep -q 'Already complete' "${RUN_LOG[i]}"; then
                note=" (already complete)"
            fi
            echo "[done   $count] $(run_label "$i")$took$note" ;;
        3)
            NOT_TRAINED+=("$(run_label "$i")")
            echo "[skip   $count] $(run_label "$i"): no trained model yet" ;;
        *)
            FAILED+=("$(run_label "$i") (exit $rc${RUN_LOG[i]:+, log: ${RUN_LOG[i]}})")
            echo "[FAILED $count] $(run_label "$i") (exit $rc)${RUN_LOG[i]:+, see ${RUN_LOG[i]}}" ;;
    esac
}

# shellcheck disable=SC2329  # invoked by the trap below
stop_runs() {
    trap - INT TERM
    echo >&2
    echo "Interrupted: stopping the runs in progress..." >&2
    local pid
    if [[ -n $FG_PID ]]; then
        kill -TERM "$FG_PID" 2> /dev/null
    fi
    for pid in "${SLOT_PID[@]}"; do
        if [[ -n $pid ]]; then
            pkill -TERM -P "$pid" 2> /dev/null
        fi
    done
    wait
    echo "Stopped. Rerun the same command to continue; finished runs are skipped." >&2
    exit 130
}
trap stop_runs INT TERM

if (( JOBS == 1 )); then
    if (( ${#DEVICE_LIST[@]} > 1 )); then
        echo "  (-j 1 runs one at a time on ${DEVICE_LIST[0]})"
    fi
    for ((i = 0; i < N; i++)); do
        build_cmd "$i" "${DEVICE_LIST[0]}"
        echo
        echo "############ [$((i + 1))/$N] $(run_label "$i") on ${DEVICE_LIST[0]} ############"
        RUN_START[i]=$SECONDS
        # In the background + wait, so Ctrl-C or kill stops it right away
        "${CMD[@]}" &
        FG_PID=$!
        wait "$FG_PID"
        rc=$?
        FG_PID=
        report "$i" "$rc"
    done
else
    STATUS_DIR=$(mktemp -d) || die "could not create a temporary directory"
    trap 'rm -rf "$STATUS_DIR"' EXIT
    mkdir -p "$LOG_DIR" || die "could not create $LOG_DIR"
    for ((k = 0; k < JOBS; k++)); do
        SLOT_RUN[k]=-1
        SLOT_PID[k]=
    done

    # Report finished runs and free their slots. With "lost", also free slots whose
    # runner died without reporting (no child processes are left at that point).
    collect() {  # [lost]
        local k i
        for ((k = 0; k < JOBS; k++)); do
            i=${SLOT_RUN[k]}
            (( i >= 0 )) || continue
            if [[ -f $STATUS_DIR/$i ]]; then
                report "$i" "$(< "$STATUS_DIR/$i")"
            elif [[ ${1:-} == lost ]]; then
                report "$i" 255
            else
                continue
            fi
            SLOT_RUN[k]=-1
            SLOT_PID[k]=
        done
    }

    busy_slots() {
        local k n=0
        for ((k = 0; k < JOBS; k++)); do
            if (( SLOT_RUN[k] >= 0 )); then
                n=$((n + 1))
            fi
        done
        echo "$n"
    }

    next=0
    while (( next < N || $(busy_slots) > 0 )); do
        collect
        for ((k = 0; k < JOBS && next < N; k++)); do
            (( SLOT_RUN[k] < 0 )) || continue
            # Slot k always uses the same device, so the devices share the load evenly
            device=${DEVICE_LIST[$((k % ${#DEVICE_LIST[@]}))]}
            build_cmd "$next" "$device"
            RUN_LOG[next]=$LOG_DIR/${RUN_CFG[next]}_seed${RUN_SEED[next]}_${RUN_TASK[next]}.log
            RUN_START[next]=$SECONDS
            (
                "${CMD[@]}" > "${RUN_LOG[next]}" 2>&1
                echo $? > "$STATUS_DIR/$next.tmp"
                mv "$STATUS_DIR/$next.tmp" "$STATUS_DIR/$next"
            ) &
            SLOT_RUN[k]=$next
            SLOT_PID[k]=$!
            echo "[start] $(run_label "$next") on $device"
            next=$((next + 1))
        done
        if (( $(busy_slots) > 0 )); then
            wait -n  # until any run finishes
            if (( $? == 127 )); then  # no runner left at all
                collect lost
            fi
        fi
    done
fi

# ---- Summary ---------------------------------------------------------------------
echo
echo "================================================================"
summary="$N runs in $(fmt_duration $SECONDS): $N_OK ok"
if (( ${#NOT_TRAINED[@]} )); then
    summary+=", ${#NOT_TRAINED[@]} not trained yet"
fi
if (( ${#FAILED[@]} )); then
    summary+=", ${#FAILED[@]} failed"
fi
echo "$summary"
if (( ${#NOT_TRAINED[@]} )); then
    echo "Not trained yet (train them first, then evaluate again):"
    printf '  %s\n' "${NOT_TRAINED[@]}"
fi
if (( ${#FAILED[@]} )); then
    echo "Failed:"
    printf '  %s\n' "${FAILED[@]}"
fi
echo "================================================================"

if (( ${#FAILED[@]} )); then
    exit 1
elif (( ${#NOT_TRAINED[@]} )); then
    exit 3
fi
exit 0
