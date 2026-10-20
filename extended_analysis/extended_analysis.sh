#!/usr/bin/env bash
# =============================================================================
# Extended analysis of RHDT: sweeps of the Hopfield beta, of each penalty weight and of
# the number of Hopfield iterations, each evaluated with the paper's protocol, and
# measurements of the trained models.
#
#   bash extended_analysis.sh -d cuda:0 -j 6
#   bash extended_analysis.sh -d cuda:0,cuda:1 -j 12 -X 0,0.25,0.5,1,2,4   # smoother penalty curves
#   bash extended_analysis.sh -v beta -m measure                             # only measure the beta sweep
#   bash extended_analysis.sh -n                                             # list what would run
#
# Sweeps, on every task given with -t ("tuned" is the task's paper config):
#   beta    Hopfield beta, x 1/sqrt(head size), 1 iteration    config tuned-b<X>   -B 0.5,1,1.5,2,5
#   gp      lambda_gp x its tuned value, lambda_sep tuned      config tuned-gpx<M> -X 0,0.25,1,4
#   ortho   lambda_sep x its tuned value, lambda_gp tuned      config tuned-sepx<M>
#   nhop    Hopfield iterations                                config tuned, -i N  -H 0,1,2,3
# (gp 0 is the ablation's orthogonality-only model, tuned-sep; ortho 0 its GP-only model,
# tuned-gp; the value 1 is the paper's model.)
#
# A point whose models exist for every training seed in the paper's runs
# (../training/models_rhdt_sweep, results in ../eval/results_rhdt) or in the ablation
# (../ablation/models_ablation, results in ../ablation/results_ablation) is read from there,
# not trained again. With the defaults these are beta 1 and 1.5, gp and ortho 1, nhop 1 and 2
# (paper's runs) and gp 0, ortho 0 and nhop 0 (ablation). Every other point is trained and
# evaluated here; nothing outside this folder is written:
#   models_extended/iter<N>/<config>/seed_<seed>/rhdt_<task>.pt
#   results_extended/                          per-run results and summaries, as in ../eval
#   results_extended/tables/<sweep>_<task>.json and .md      the score of every point
#   results_extended/measure/hopfield_gap_<task>.json         query-gap ratio and key norms vs beta
#   results_extended/measure/lipschitz_<task>.json            ||J||_2 vs lambda_gp
# Finished training runs and evaluation jobs are skipped, so rerun the same command to
# continue after an interruption.
# =============================================================================

if [ -z "${BASH_VERSINFO[0]:-}" ] || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 403 )); then
    echo "extended_analysis.sh needs bash 4.3 or newer" >&2
    exit 1
fi
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TRAIN_SH=$REPO_DIR/training/train.sh
EVAL_SH=$REPO_DIR/eval/eval.sh
# shellcheck source=SCRIPTDIR/../training/configs.sh
source "$SCRIPT_DIR/../training/configs.sh"
PYTHON=${PYTHON:-python}

ALL_SWEEPS="beta gp ortho nhop"

# Config (before resolving "tuned") and Hopfield iterations of one point of a sweep
point_spec() {  # <sweep> <value> -> "<config> <iterations>"
    case $1/$2 in
        beta/1 | gp/1 | ortho/1) echo "tuned 1" ;;
        beta/*)  echo "tuned-b$2 1" ;;
        gp/0)    echo "tuned-sep 1" ;;
        gp/*)    echo "tuned-gpx$2 1" ;;
        ortho/0) echo "tuned-gp 1" ;;
        ortho/*) echo "tuned-sepx$2 1" ;;
        nhop/*)  echo "tuned $2" ;;
    esac
}

# ---- Command line --------------------------------------------------------------
TASKS=halfcheetah-medium-v2,hopper-medium-replay-v2
SWEEPS=${ALL_SWEEPS// /,}
BETAS=0.5,1,1.5,2,5
FACTORS=0,0.25,1,4
NHOPS=0,1,2,3
SEEDS=$PAPER_SEEDS
DEVICES=cuda:0
JOBS=6
MODE=all
EPISODES=50
CORRUPTIONS=0.0,0.1,0.2,0.3
EVAL_SEEDS=11,13,17,19,23,29,31,37,41,43
MODELS_ROOT=$SCRIPT_DIR/models_extended
RESULTS_ROOT=$SCRIPT_DIR/results_extended
MAIN_MODELS=$REPO_DIR/training/models_rhdt_sweep
MAIN_RESULTS=$REPO_DIR/eval/results_rhdt
ABL_MODELS=$REPO_DIR/ablation/models_ablation
ABL_RESULTS=$REPO_DIR/ablation/results_ablation
CONDA_ENV=
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash extended_analysis.sh [options]

  -t TASKS       tasks, comma-separated, or all  (default: $TASKS)
  -v SWEEPS      sweeps, comma-separated: beta, gp, ortho, nhop (default: all four)
  -B VALUES      beta sweep: beta as a multiple of 1/sqrt(head size) (default: $BETAS)
  -X VALUES      gp and ortho sweeps: multiples of the tuned weight (default: $FACTORS)
  -H VALUES      nhop sweep: Hopfield iterations     (default: $NHOPS)
  -s SEEDS       training seeds, comma-separated     (default: $PAPER_SEEDS)
  -d DEVICES     device, or several comma-separated  (default: $DEVICES)
  -j JOBS        training runs, and then evaluation jobs, at a time (default: $JOBS)
  -m MODE        train, eval (evaluate and write the tables), measure, or all (default: $MODE)
  -E EPISODES    episodes per noise level and evaluation seed (default: $EPISODES)
  -C LEVELS      noise levels                        (default: $CORRUPTIONS)
  -S SEEDS       evaluation seeds                    (default: $EVAL_SEEDS)
  -M DIR         models trained here                 (default: extended_analysis/models_extended)
  -o DIR         results, tables and measurements    (default: extended_analysis/results_extended)
  -P DIR,DIR     the paper's models and results      (default: training/models_rhdt_sweep,eval/results_rhdt)
  -A DIR,DIR     the ablation's models and results   (default: ablation/models_ablation,ablation/results_ablation)
  -e ENV         conda environment to activate first
  -n             dry run: list the points, training runs and evaluation jobs, start nothing
  -h             show this help

Sweeps ("tuned" is the task's paper config, listed by bash ../training/train.sh -h):
  beta    Hopfield beta, x 1/sqrt(head size)     config tuned-b<X> (1: tuned)
  gp      lambda_gp x its tuned value            config tuned-gpx<M> (0: tuned-sep, 1: tuned)
  ortho   lambda_sep x its tuned value           config tuned-sepx<M> (0: tuned-gp, 1: tuned)
  nhop    Hopfield iterations                    config tuned, -i N
A point whose models exist for every training seed under -P or -A is read from there;
the others are trained and evaluated here.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

pair_option() {  # <option> <DIR,DIR> -> PAIR_A, PAIR_B (absolute)
    [[ $2 == *,* ]] || die "-$1 needs two folders, MODELS,RESULTS"
    PAIR_A=$(abs_path "${2%%,*}")
    PAIR_B=$(abs_path "${2#*,}")
}

while getopts ":t:v:B:X:H:s:d:j:m:E:C:S:M:o:P:A:e:nh" opt; do
    case $opt in
        t) TASKS=$OPTARG ;;
        v) SWEEPS=$OPTARG ;;
        B) BETAS=$OPTARG ;;
        X) FACTORS=$OPTARG ;;
        H) NHOPS=$OPTARG ;;
        s) SEEDS=$OPTARG ;;
        d) DEVICES=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        m) MODE=$OPTARG ;;
        E) EPISODES=$OPTARG ;;
        C) CORRUPTIONS=$OPTARG ;;
        S) EVAL_SEEDS=$OPTARG ;;
        M) MODELS_ROOT=$(abs_path "$OPTARG") ;;
        o) RESULTS_ROOT=$(abs_path "$OPTARG") ;;
        P) pair_option P "$OPTARG"; MAIN_MODELS=$PAIR_A; MAIN_RESULTS=$PAIR_B ;;
        A) pair_option A "$OPTARG"; ABL_MODELS=$PAIR_A; ABL_RESULTS=$PAIR_B ;;
        e) CONDA_ENV=$OPTARG ;;
        n) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        :) die "option -$OPTARG needs a value (see -h)" ;;
        *) die "unknown option -$OPTARG (see -h)" ;;
    esac
done
shift $((OPTIND - 1))
(( $# == 0 )) || die "unexpected argument '$1'; lists are comma-separated, e.g. -s 42,48213"

# A list of numbers -> the named array, normalized (2.0 -> 2, so config names match), sorted
parse_values() {  # <option> <text> <array name> <integer|positive|nonnegative>
    local -n _out=$3
    local v
    local -a items=()
    parse_list "$2" items
    (( ${#items[@]} )) || die "-$1 needs at least one value"
    _out=()
    for v in "${items[@]}"; do
        case $4 in
            integer) [[ $v =~ ^[0-9]+$ ]] || die "-$1: '$v' is not a non-negative integer" ;;
            *)       [[ $v =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || die "-$1: '$v' is not a number" ;;
        esac
        v=$(LC_ALL=C awk -v x="$v" 'BEGIN { printf "%g\n", x }')
        if [[ $4 == positive && $v == 0 ]]; then
            die "-$1: values must be > 0"
        fi
        _out+=("$v")
    done
    mapfile -t _out < <(printf '%s\n' "${_out[@]}" | sort -g -u)
}

[[ $TASKS == all ]] && TASKS=$PAPER_TASKS
parse_list "$TASKS" TASK_LIST
(( ${#TASK_LIST[@]} )) || die "no tasks given"
for task in "${TASK_LIST[@]}"; do
    check_task "$task" || exit 1
done
SWEEP_LIST=()
parse_list "$SWEEPS" SWEEP_LIST
(( ${#SWEEP_LIST[@]} )) || die "no sweeps given"
for sweep in "${SWEEP_LIST[@]}"; do
    [[ " $ALL_SWEEPS " == *" $sweep "* ]] || die "unknown sweep '$sweep'; sweeps: ${ALL_SWEEPS// /, }"
done
parse_values B "$BETAS" BETA_LIST positive
parse_values X "$FACTORS" FACTOR_LIST nonnegative
parse_values H "$NHOPS" NHOP_LIST integer
parse_seeds "$SEEDS" || exit 1
parse_list "$DEVICES" DEVICE_LIST
(( ${#DEVICE_LIST[@]} )) || die "no device given"
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "-j needs a positive integer, got '$JOBS'"
case $MODE in train|eval|measure|all) ;; *) die "-m must be train, eval, measure or all, got '$MODE'" ;; esac
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

# ---- Points --------------------------------------------------------------------
has_all_seeds() {  # <models root> <iterations> <config> <task>
    local seed
    for seed in "${SEED_LIST[@]}"; do
        [[ -f $1/iter$2/$3/seed_$seed/rhdt_$4.pt ]] || return 1
    done
}

point_source() {  # <iterations> <config> <task> -> main, ablation or extended
    if has_all_seeds "$MAIN_MODELS" "$@"; then
        echo main
    elif has_all_seeds "$ABL_MODELS" "$@"; then
        echo ablation
    else
        echo extended
    fi
}

# shellcheck disable=SC2329  # used by measure
source_models() {  # <source> -> its models root
    case $1 in
        main)     echo "$MAIN_MODELS" ;;
        ablation) echo "$ABL_MODELS" ;;
        *)        echo "$MODELS_ROOT" ;;
    esac
}

sweep_values() {  # <sweep> -> its values, one per line
    case $1 in
        beta)     printf '%s\n' "${BETA_LIST[@]}" ;;
        gp|ortho) printf '%s\n' "${FACTOR_LIST[@]}" ;;
        nhop)     printf '%s\n' "${NHOP_LIST[@]}" ;;
    esac
}

POINTS=()            # "sweep task value config iterations source"
declare -A TRAIN=()  # iterations -> "config/task,..." trained here
declare -A QUEUED=()
for task in "${TASK_LIST[@]}"; do
    for sweep in "${SWEEP_LIST[@]}"; do
        while read -r value; do
            read -r spec iters <<< "$(point_spec "$sweep" "$value")"
            cfg=$(resolve_config "$spec" "$task")
            config_lambdas "$cfg" > /dev/null || die "invalid config $cfg"
            src=$(point_source "$iters" "$cfg" "$task")
            POINTS+=("$sweep $task $value $cfg $iters $src")
            if [[ $src == extended && -z ${QUEUED[$iters/$cfg/$task]:-} ]]; then
                QUEUED[$iters/$cfg/$task]=1
                TRAIN[$iters]+="${TRAIN[$iters]:+,}$cfg/$task"
            fi
        done < <(sweep_values "$sweep")
    done
done
mapfile -t TRAIN_ITERS < <(printf '%s\n' "${!TRAIN[@]}" | sort -n)

echo "RHDT extended analysis: tasks ${TASK_LIST[*]} | training seeds ${SEED_LIST[*]} | devices ${DEVICE_LIST[*]} | jobs $JOBS"
echo "  evaluation: ${#EVAL_SEED_LIST[@]} evaluation seeds x $EPISODES episodes at noise levels $CORRUPTIONS"
echo "  paper's runs: $MAIN_MODELS, $MAIN_RESULTS"
echo "  ablation: $ABL_MODELS, $ABL_RESULTS"
echo "  trained here: $MODELS_ROOT, $RESULTS_ROOT"
last=
for p in "${POINTS[@]}"; do
    read -r sweep task value cfg iters src <<< "$p"
    if [[ $sweep/$task != "$last" ]]; then
        echo "  $task, $sweep:"
        last=$sweep/$task
    fi
    read -r gp sep <<< "$(config_lambdas "$cfg")"
    case $src in
        main)     where="paper's runs" ;;
        ablation) where="ablation" ;;
        *)        where="trained here" ;;
    esac
    printf '    %-6s %-20s iterations=%s lambda_gp=%-9s lambda_sep=%-8s beta=%-4s %s\n' \
        "$value" "$cfg" "$iters" "$gp" "$sep" "$(config_beta "$cfg")" "$where"
done
n_train=0
pairs=()
for iters in "${TRAIN_ITERS[@]}"; do
    parse_list "${TRAIN[$iters]}" pairs
    n_train=$((n_train + ${#pairs[@]} * ${#SEED_LIST[@]}))
done
echo "  $n_train training runs here (${#SEED_LIST[@]} seeds per point; finished ones are skipped)"

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
    echo "Extended analysis interrupted. Rerun the same command to continue." >&2
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
if [[ $MODE == train || $MODE == all ]]; then
    for iters in "${TRAIN_ITERS[@]}"; do
        run_step "Training with $iters Hopfield iteration(s): ${TRAIN[$iters]}" \
            bash "$TRAIN_SH" -m train -s "$SEEDS" -c "${TRAIN[$iters]}" -i "$iters" \
                -d "$DEVICES" -j "$JOBS" -o "$MODELS_ROOT" "${DRY[@]}"
    done
fi

# ---- 2. Evaluate, then tabulate ------------------------------------------------------
# shellcheck disable=SC2329  # run by run_step
evaluate() {  # <iterations>
    if (( DRY_RUN )); then
        # Before training, every model is missing; the plan line counts them
        bash "$EVAL_SH" -s "$SEEDS" -c "${TRAIN[$1]}" -i "$1" -d "$DEVICES" -j "$JOBS" -E "$EPISODES" \
            -C "$CORRUPTIONS" -S "$EVAL_SEEDS" -M "$MODELS_ROOT" -o "$RESULTS_ROOT" -n 2>&1 |
            grep -v '^WARNING: missing checkpoint'
        return 0
    fi
    bash "$EVAL_SH" -s "$SEEDS" -c "${TRAIN[$1]}" -i "$1" -d "$DEVICES" -j "$JOBS" -E "$EPISODES" \
        -C "$CORRUPTIONS" -S "$EVAL_SEEDS" -M "$MODELS_ROOT" -o "$RESULTS_ROOT"
}

if [[ $MODE == eval || $MODE == all ]]; then
    for iters in "${TRAIN_ITERS[@]}"; do
        run_step "Evaluating with $iters Hopfield iteration(s): ${TRAIN[$iters]}" evaluate "$iters"
    done
    if (( ! DRY_RUN )); then
        table_args=(--output-dir "$RESULTS_ROOT/tables"
                    --source main "$MAIN_MODELS" "$MAIN_RESULTS"
                    --source ablation "$ABL_MODELS" "$ABL_RESULTS"
                    --source extended "$MODELS_ROOT" "$RESULTS_ROOT"
                    --seeds "${SEED_LIST[@]}" --eval-seeds "$EVAL_SEEDS"
                    --episodes "$EPISODES" --corruptions "$CORRUPTIONS")
        for p in "${POINTS[@]}"; do
            read -r -a fields <<< "$p"
            table_args+=(--point "${fields[@]}")
        done
        run_step "Tables" "$PYTHON" "$SCRIPT_DIR/extended_table.py" "${table_args[@]}"
    fi
fi

# ---- 3. Measure ----------------------------------------------------------------------
# Query-gap ratio and key norms over the beta sweep, ||J||_2 over the gp sweep. A point
# is measured once its models exist for every training seed.
# shellcheck disable=SC2329  # run by run_step
measure() {  # <script> <output name> <sweep> <label prefix> <task>
    local script=$1 name=$2 sweep=$3 prefix=$4 task=$5 p s t value cfg iters src root
    local -a models=()
    for p in "${POINTS[@]}"; do
        read -r s t value cfg iters src <<< "$p"
        [[ $s == "$sweep" && $t == "$task" ]] || continue
        root=$(source_models "$src")
        if has_all_seeds "$root" "$iters" "$cfg" "$task"; then
            models+=(--model "$prefix$value=$root/iter$iters/$cfg/seed_{seed}/rhdt_$task.pt")
        else
            echo "  skipped: $sweep $value ($cfg) is not trained for every seed yet"
        fi
    done
    if (( ! ${#models[@]} )); then
        echo "  nothing to measure"
        return 0
    fi
    local cmd=("$PYTHON" "$SCRIPT_DIR/$script" --task "$task" --seeds "$SEEDS" --device "${DEVICE_LIST[0]}"
               "${models[@]}")
    if (( DRY_RUN )); then
        printf '  %q' "${cmd[@]}" --out "$RESULTS_ROOT/measure/${name}_$task.json"
        echo
        return 0
    fi
    mkdir -p "$RESULTS_ROOT/measure" || return 1
    "${cmd[@]}" --check || return 1
    "${cmd[@]}" --out "$RESULTS_ROOT/measure/${name}_$task.json" 2>&1 | tee "$RESULTS_ROOT/measure/${name}_$task.log"
}

if [[ $MODE == measure || $MODE == all ]]; then
    for task in "${TASK_LIST[@]}"; do
        if [[ " ${SWEEP_LIST[*]} " == *" beta "* ]]; then
            run_step "Query-gap ratio and key norms vs beta, $task" \
                measure measure_hopfield_gap.py hopfield_gap beta b "$task"
        fi
        if [[ " ${SWEEP_LIST[*]} " == *" gp "* ]]; then
            run_step "||J||_2 vs lambda_gp, $task" measure measure_lipschitz.py lipschitz gp gp "$task"
        fi
    done
fi

if (( ${#FAILED[@]} )); then
    echo >&2
    echo "Some steps did not finish; see their output above:" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
exit 0
