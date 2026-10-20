# shellcheck shell=bash disable=SC2034  # variables are used by the scripts sourcing this
# Paper setup shared by training/train.sh and eval/eval.sh (both source this file):
# seeds, tasks, regularization configs, and how -s/-t/-c select runs.

PAPER_SEEDS="907561,48213,2024887,739104,26451,42"
PAPER_TASKS="halfcheetah-medium-v2 halfcheetah-medium-replay-v2 hopper-medium-v2 hopper-medium-replay-v2 walker2d-medium-v2 walker2d-medium-replay-v2"

# Attention heads of every model, unless its config name ends in -h<N> (below)
NUM_HEADS=4

# Regularization configs: "<lambda_gp> <lambda_sep>". The name is also the output
# folder, so keep names and values unchanged once results exist.
CONFIG_NAMES="non default half quarter double gpheavy gpdoxy sepheavy gponly seponly"
#
# Derived configs, used by the ablations (../ablation) and ../extended_analysis:
#   <config>-gp       the config's lambda_gp only (lambda_sep = 0)
#   <config>-sep      the config's lambda_sep only (lambda_gp = 0)
#   <config>-gpx<M>   the config with lambda_gp multiplied by M, e.g. -gpx0.25
#   <config>-sepx<M>  the config with lambda_sep multiplied by M, e.g. -sepx4
#   <config>-h<N>     the config with N attention heads instead of NUM_HEADS
#   <config>-b<X>     the config with the Hopfield beta X times 1/sqrt(head size), e.g. -b1.5
# They combine in this order (gpheavy-gp-h1-b2), and tuned-gp, tuned-sep, tuned-gpx4,
# tuned-h1, tuned-b1.5 ... stand for each task's tuned config with that change.
# A config name is <base>[-gp|-sep][-gpx<M>][-sepx<M>][-h<N>][-b<X>], in this order. Sets
# CONFIG_BASE, CONFIG_ONLY (gp, sep or empty), CONFIG_GPX and CONFIG_SEPX (empty if not
# given), CONFIG_HEADS and CONFIG_BETA; fails if malformed.
_config_parts() {  # <config>
    local num='([0-9]+(\.[0-9]+)?)'
    [[ $1 =~ ^([a-z]+)(-gp|-sep)?(-gpx$num)?(-sepx$num)?(-h([1-9][0-9]*))?(-b$num)?$ ]] || return 1
    CONFIG_BASE=${BASH_REMATCH[1]}
    CONFIG_ONLY=${BASH_REMATCH[2]#-}
    CONFIG_GPX=${BASH_REMATCH[4]}
    CONFIG_SEPX=${BASH_REMATCH[7]}
    CONFIG_HEADS=${BASH_REMATCH[10]:-$NUM_HEADS}
    CONFIG_BETA=${BASH_REMATCH[12]:-1}
    # beta and the multipliers > 0
    [[ $CONFIG_BETA == *[1-9]* ]] || return 1
    [[ -z $CONFIG_GPX || $CONFIG_GPX == *[1-9]* ]] || return 1
    [[ -z $CONFIG_SEPX || $CONFIG_SEPX == *[1-9]* ]]
}

_times() {  # <number> <factor> -> their product, e.g. 0.005 0.25 -> 0.00125
    LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { printf "%.10g\n", a * b }'
}

config_lambdas() {  # <config> -> "<lambda_gp> <lambda_sep>"; fails for an unknown config
    local gp sep
    _config_parts "$1" || return 1
    case $CONFIG_BASE in
        non)      gp=0 sep=0 ;;
        default)  gp=0.0025 sep=0.1 ;;
        half)     gp=0.001 sep=0.05 ;;
        quarter)  gp=0.0005 sep=0.025 ;;
        double)   gp=0.005 sep=0.2 ;;
        gpheavy)  gp=0.005 sep=0.05 ;;
        gpdoxy)   gp=0.020 sep=0.025 ;;
        sepheavy) gp=0.001 sep=0.2 ;;
        gponly)   gp=0.0025 sep=0.0 ;;
        seponly)  gp=0.0 sep=0.1 ;;
        *)        return 1 ;;
    esac
    case $CONFIG_ONLY in
        gp)  sep=0 ;;
        sep) gp=0 ;;
    esac
    if [[ -n $CONFIG_GPX ]]; then
        gp=$(_times "$gp" "$CONFIG_GPX")
    fi
    if [[ -n $CONFIG_SEPX ]]; then
        sep=$(_times "$sep" "$CONFIG_SEPX")
    fi
    echo "$gp $sep"
}

config_heads() {  # <config> -> attention heads
    if _config_parts "$1"; then
        echo "$CONFIG_HEADS"
    else
        echo "$NUM_HEADS"
    fi
}

config_beta() {  # <config> -> Hopfield beta as a multiple of 1/sqrt(head size)
    if _config_parts "$1"; then
        echo "$CONFIG_BETA"
    else
        echo 1
    fi
}

# Config used for each task in the paper ("-c tuned")
tuned_config() {
    case $1 in
        halfcheetah-medium-v2)        echo gpheavy ;;
        halfcheetah-medium-replay-v2) echo gpdoxy ;;
        hopper-medium-v2)             echo sepheavy ;;
        hopper-medium-replay-v2)      echo quarter ;;
        walker2d-medium-v2)           echo sepheavy ;;
        walker2d-medium-replay-v2)    echo quarter ;;
        *)                            return 1 ;;
    esac
}

# Print the configs and the tuned config of each task (for -h)
print_configs() {
    local name gp sep task
    echo "Configs (lambda_gp = gradient penalty, lambda_sep = key orthogonality):"
    printf '  %-10s %-10s %s\n' config lambda_gp lambda_sep
    for name in $CONFIG_NAMES; do
        read -r gp sep <<< "$(config_lambdas "$name")"
        printf '  %-10s %-10s %s\n' "$name" "$gp" "$sep"
    done
    echo
    echo "Tuned config of each task (-c tuned):"
    for task in $PAPER_TASKS; do
        printf '  %-30s %s\n' "$task" "$(tuned_config "$task")"
    done
    echo
    echo "Derived configs (tuned-gp etc. apply the change to each task's tuned config):"
    echo "  <config>-gp       the config's lambda_gp only (lambda_sep = 0)"
    echo "  <config>-sep      the config's lambda_sep only (lambda_gp = 0)"
    echo "  <config>-gpx<M>   lambda_gp multiplied by M, e.g. -gpx0.25"
    echo "  <config>-sepx<M>  lambda_sep multiplied by M, e.g. -sepx4"
    echo "  <config>-h<N>     N attention heads instead of $NUM_HEADS"
    echo "  <config>-b<X>     Hopfield beta X times the default 1/sqrt(head size), e.g. -b1.5"
}

# Split a list separated by commas and/or spaces into the named array, dropping repeats
parse_list() {  # <text> <array name>
    local -n _list=$2
    local -A _seen=()
    local _item _items
    read -r -a _items <<< "${1//,/ }"
    _list=()
    for _item in "${_items[@]}"; do
        [[ -n ${_seen[$_item]:-} ]] && continue
        _seen[$_item]=1
        _list+=("$_item")
    done
}

# -s SEEDS -> SEED_LIST, validated; 042 becomes 42 so folder names match the seed
parse_seeds() {  # <text>
    local i
    parse_list "$1" SEED_LIST
    if (( ! ${#SEED_LIST[@]} )); then
        echo "Error: no seeds given" >&2
        return 1
    fi
    for i in "${!SEED_LIST[@]}"; do
        if [[ ! ${SEED_LIST[i]} =~ ^[0-9]+$ ]]; then
            echo "Error: seed '${SEED_LIST[i]}' is not a non-negative integer" >&2
            return 1
        fi
        SEED_LIST[i]=$((10#${SEED_LIST[i]}))
    done
    parse_list "${SEED_LIST[*]}" SEED_LIST  # repeats after normalizing, e.g. 42,042
}

check_task() {  # <task>
    [[ " $PAPER_TASKS " == *" $1 "* ]] && return 0
    echo "Error: unknown task '$1'; valid tasks: ${PAPER_TASKS// /, }" >&2
    return 1
}

# -t TASKS and -c CONFIGS -> PAIRS ("<config>/<task>"), in order, without repeats.
# A config applies to every selected task; "tuned" stands for each task's paper
# config ("tuned-gp" etc. a derived config of it); "<config>/<task>" adds just that pair.
expand_pairs() {  # <tasks> <configs>
    local tasks=$1 spec task
    local -A seen=()
    [[ $tasks == all ]] && tasks=$PAPER_TASKS
    parse_list "$tasks" TASK_LIST
    parse_list "$2" CONFIG_LIST
    if (( ! ${#TASK_LIST[@]} || ! ${#CONFIG_LIST[@]} )); then
        echo "Error: no tasks or no configs given" >&2
        return 1
    fi
    for task in "${TASK_LIST[@]}"; do
        check_task "$task" || return 1
    done
    PAIRS=()
    for spec in "${CONFIG_LIST[@]}"; do
        if [[ $spec == */* ]]; then
            check_task "${spec#*/}" || return 1
            _add_pair "${spec%%/*}" "${spec#*/}" || return 1
        else
            for task in "${TASK_LIST[@]}"; do
                _add_pair "$spec" "$task" || return 1
            done
        fi
    done
}

# Config name of a -c entry for one task: "tuned" and "tuned-..." become the task's
# tuned config (with the derived suffix); other names stay as they are
resolve_config() {  # <config, "tuned" or "tuned-..."> <task>
    if [[ $1 == tuned || $1 == tuned-* ]]; then
        echo "$(tuned_config "$2")${1#tuned}"
    else
        echo "$1"
    fi
}

_add_pair() {  # <config, "tuned" or "tuned-..."> <task>; uses expand_pairs' "seen"
    local cfg
    cfg=$(resolve_config "$1" "$2")
    if ! config_lambdas "$cfg" > /dev/null; then
        echo "Error: unknown config '$1' (see -h)" >&2
        return 1
    fi
    [[ -n ${seen[$cfg/$2]:-} ]] && return 0
    seen[$cfg/$2]=1
    PAIRS+=("$cfg/$2")
}

# Absolute version of a path given relative to the caller's directory
abs_path() {  # <path>
    if [[ $1 == /* ]]; then echo "$1"; else echo "$PWD/$1"; fi
}
